#!/usr/bin/env python3
"""Build Part 196's metadata-only shared-text clip manifest."""
from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

STYLES = ("confused", "default", "enunciated", "happy", "laughing", "sad", "whisper")
SEED = 196
MIN_CELL = 40
MANIFEST_FIELDS = ("id", "speaker_id", "style", "text_key", "fold", "shard", "row_index",
                   "duration_seconds", "duration_status")


def text_key(text: str) -> str:
    """Stable text-cluster key; the manifest need not redistribute transcript text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cell_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["speaker_id"]), str(row["style"])


def shared_text_pool(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep seven-style clips whose text is shared by styles for one speaker only."""
    materialized = [dict(row) for row in rows]
    text_speakers: dict[str, set[str]] = defaultdict(set)
    speaker_text_styles: dict[tuple[str, str], set[str]] = defaultdict(set)
    seen_ids: set[str] = set()
    for row in materialized:
        text = row.get("text")
        speaker = row.get("speaker_id")
        clip_id = row.get("id")
        style = row.get("style")
        if text is None or speaker is None or clip_id is None:
            continue
        if style not in STYLES:
            continue
        text = str(text)
        speaker = str(speaker)
        clip_id = str(clip_id)
        if clip_id in seen_ids:
            raise ValueError(f"duplicate recording id: {clip_id}")
        seen_ids.add(clip_id)
        text_speakers[text].add(speaker)
        speaker_text_styles[(speaker, text)].add(str(style))

    pool = []
    for row in materialized:
        text, speaker, clip_id, style = (row.get("text"), row.get("speaker_id"),
                                         row.get("id"), row.get("style"))
        if text is None or speaker is None or clip_id is None or style not in STYLES:
            continue
        text, speaker, clip_id = str(text), str(speaker), str(clip_id)
        if len(text_speakers[text]) != 1:
            continue
        if text_speakers[text] != {speaker}:
            continue
        if len(speaker_text_styles[(speaker, text)]) < 2:
            continue
        normalized = dict(row)
        normalized.update(id=clip_id, speaker_id=speaker, style=str(style), text=text)
        pool.append(normalized)
    return pool


def draw_manifest(rows: Iterable[dict[str, Any]], *, seed: int = SEED,
                  per_cell: int = 100) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Draw at most per_cell clips per speaker/style in stable cell and id order."""
    import numpy as np

    if per_cell <= 0:
        raise ValueError("per_cell must be positive")
    pool = shared_text_pool(rows)
    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pool:
        cells[_cell_key(row)].append(row)
    speakers = sorted({str(row["speaker_id"]) for row in pool})
    rng = np.random.default_rng(seed)
    selected: list[dict[str, Any]] = []
    pool_sizes: dict[str, int] = {}
    for speaker in speakers:
        for style in STYLES:
            candidates = sorted(cells.get((speaker, style), []), key=lambda row: str(row["id"]))
            pool_sizes[f"{speaker}/{style}"] = len(candidates)
            take = min(per_cell, len(candidates))
            if not take:
                continue
            chosen = rng.choice(len(candidates), size=take, replace=False)
            selected.extend(candidates[int(index)] for index in chosen)
    # Stable output and recorder order after using the registered sequential RNG draws.
    selected.sort(key=lambda row: (str(row["speaker_id"]), STYLES.index(row["style"]), str(row["id"])))
    manifest = []
    for row in selected:
        manifest.append({
            "id": str(row["id"]), "speaker_id": str(row["speaker_id"]),
            "style": str(row["style"]), "text_key": text_key(str(row["text"])),
            "fold": str(row["speaker_id"]),
            "shard": str(row.get("shard", "")),
            "row_index": str(row.get("row_index", "")),
            "duration_seconds": "",
            "duration_status": "deferred_to_recorder",
        })
    return manifest, pool_sizes


def gate1_check(rows: Iterable[dict[str, Any]], *, included_ids: set[str] | None = None,
                min_per_cell: int = MIN_CELL) -> dict[str, Any]:
    """Check held-out speaker/text/id disjointness and cell minima."""
    materialized = [dict(row) for row in rows]
    speakers = sorted({str(row["speaker_id"]) for row in materialized if row.get("speaker_id") is not None})
    if included_ids is not None:
        keep = {str(value) for value in included_ids}
        materialized = [row for row in materialized if str(row.get("id")) in keep]
    folds = {}
    disjoint = True
    text_field = "text_key" if all("text_key" in row for row in materialized) else "text"
    for heldout in speakers:
        if all("fold" in row for row in materialized):
            test = [row for row in materialized if str(row["fold"]) == heldout]
            train = [row for row in materialized if str(row["fold"]) != heldout]
        else:
            test = [row for row in materialized if str(row["speaker_id"]) == heldout]
            train = [row for row in materialized if str(row["speaker_id"]) != heldout]
        checks = {}
        for field in ("speaker_id", text_field, "id"):
            test_values = {str(row.get(field, "")) for row in test}
            train_values = {str(row.get(field, "")) for row in train}
            overlap = sorted(test_values & train_values)
            checks[field] = {"passed": not overlap, "overlap": overlap[:20]}
            disjoint &= not overlap
        folds[heldout] = checks
    counts = Counter((str(row["speaker_id"]), str(row["style"])) for row in materialized)
    all_cells = {f"{speaker}/{style}": int(counts[(speaker, style)])
                 for speaker in speakers for style in STYLES}
    minima_passed = bool(speakers) and all(n >= min_per_cell for n in all_cells.values())
    return {"verdict": "passed" if disjoint and minima_passed else "failed",
            "speaker_text_id_disjoint": disjoint, "minimum_per_cell": int(min_per_cell),
            "cell_counts": all_cells, "cell_minima_passed": minima_passed,
            "folds": folds}


def read_metadata(data_dir: Path) -> list[dict[str, Any]]:
    """Read only id/text/speaker/style columns; audio is deliberately not touched."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to read Expresso parquet metadata") from exc
    shards = sorted(data_dir.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards found in {data_dir}")
    result: list[dict[str, Any]] = []
    columns = ["id", "text", "speaker_id", "style"]
    for shard in shards:
        parquet = pq.ParquetFile(shard)
        index = 0
        for batch in parquet.iter_batches(columns=columns, batch_size=1024):
            for local_row, row in enumerate(batch.to_pylist()):
                row["shard"] = shard.name
                row["row_index"] = index + local_row
                result.append(row)
            index += batch.num_rows
    return result


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/mnt/e/neuron-probe/expresso/read"))
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "manifest196.csv")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--per-cell", type=int, default=100)
    args = parser.parse_args(argv)
    metadata = read_metadata(args.data_dir)
    manifest, pool_sizes = draw_manifest(metadata, seed=args.seed, per_cell=args.per_cell)
    gate = gate1_check(manifest)
    write_manifest(args.out, manifest)
    print({"manifest": str(args.out), "metadata_rows": len(metadata),
           "drawn_rows": len(manifest), "duration_filter": "deferred to recorder",
           "gate1_pre_duration": gate["verdict"], "pool_sizes_pre_duration": pool_sizes})
    if len(manifest) == 0:
        return 1
    # Length exclusions are applied by the recorder; the final Gate 1 runs on those results.
    return 0 if gate["verdict"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
