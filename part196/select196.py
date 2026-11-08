#!/usr/bin/env python3
"""Build Part 196's WAV-header-filtered shared-text clip manifest."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

STYLES = ("confused", "default", "enunciated", "happy", "laughing", "sad", "whisper")
SEED = 196
MIN_CELL = 40
SAMPLE_RATE = 16_000
MIN_DURATION_SECONDS = 2.0
MAX_DURATION_SECONDS = 10.0
MANIFEST_FIELDS = ("id", "speaker_id", "style", "text_key", "fold", "shard", "row_index",
                   "duration_seconds", "duration_status")


def clip_sample_limit(sample_rate: int = SAMPLE_RATE) -> int:
    """Registered maximum clip length in samples, shared by selection and decoding."""
    if sample_rate <= 0:
        raise ValueError("sample rate must be positive")
    return int(MAX_DURATION_SECONDS * sample_rate)


def duration_in_registered_range(duration_seconds: float) -> bool:
    """Apply the registered inclusive 2–10 s range using the shared sample cap."""
    try:
        duration = float(duration_seconds)
    except (TypeError, ValueError):
        return False
    return MIN_DURATION_SECONDS <= duration <= clip_sample_limit() / SAMPLE_RATE


def wav_duration(raw: bytes) -> float:
    """Read clip duration from a WAV header without decoding sample data."""
    with wave.open(io.BytesIO(raw), "rb") as audio:
        rate = audio.getframerate()
        frames = audio.getnframes()
        if rate <= 0:
            raise ValueError("WAV has an invalid sample rate")
        return frames / rate


def text_key(text: str) -> str:
    """Stable text-cluster key; the manifest need not redistribute transcript text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cell_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["speaker_id"]), str(row["style"])


def shared_text_pool(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep in-range clips whose text is shared across styles for one speaker only."""
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
        if not duration_in_registered_range(row.get("duration_seconds")):
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
        if not duration_in_registered_range(row.get("duration_seconds")):
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
    """Draw shared texts in a fixed order, retaining at least two styles per drawn cluster."""
    import numpy as np

    if per_cell <= 0:
        raise ValueError("per_cell must be positive")
    pool = shared_text_pool(rows)
    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_speaker_text: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list))
    for row in pool:
        cells[_cell_key(row)].append(row)
        by_speaker_text[(str(row["speaker_id"]), str(row["text"]))][str(row["style"])].append(row)
    speakers = sorted({str(row["speaker_id"]) for row in pool})
    rng = np.random.default_rng(seed)
    selected: list[dict[str, Any]] = []
    pool_sizes: dict[str, int] = {}
    for speaker in speakers:
        speaker_texts = sorted(text for row_speaker, text in by_speaker_text if row_speaker == speaker)
        if speaker_texts:
            speaker_texts = [speaker_texts[int(index)] for index in rng.permutation(len(speaker_texts))]
        cell_counts = {style: 0 for style in STYLES}
        for style in STYLES:
            pool_sizes[f"{speaker}/{style}"] = len(cells.get((speaker, style), []))
        for text in speaker_texts:
            if all(cell_counts[style] >= per_cell for style in STYLES):
                break
            styles_to_add = [style for style in STYLES
                             if cell_counts[style] < per_cell and
                             by_speaker_text[(speaker, text)].get(style)]
            if len(styles_to_add) < 2:
                continue
            for style in styles_to_add:
                candidates = sorted(by_speaker_text[(speaker, text)][style],
                                    key=lambda row: str(row["id"]))
                remaining = per_cell - cell_counts[style]
                added = candidates[:remaining]
                selected.extend(added)
                cell_counts[style] += len(added)

    # Stable output and recorder order after the registered per-speaker text shuffles.
    selected.sort(key=lambda row: (str(row["speaker_id"]), STYLES.index(row["style"]), str(row["id"])))
    manifest = []
    for row in selected:
        manifest.append({
            "id": str(row["id"]), "speaker_id": str(row["speaker_id"]),
            "style": str(row["style"]), "text_key": text_key(str(row["text"])),
            "fold": str(row["speaker_id"]),
            "shard": str(row.get("shard", "")),
            "row_index": str(row.get("row_index", "")),
            "duration_seconds": f"{float(row['duration_seconds']):.9g}",
            "duration_status": "in_range_wav_header",
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


def gate1_check_recorded(rows: Iterable[dict[str, Any]], successful_ids: set[str], *,
                         min_per_cell: int = MIN_CELL) -> dict[str, Any]:
    """Apply Gate 1 to only clips that survived filtering and recorded successfully."""
    return gate1_check(rows, included_ids={str(value) for value in successful_ids},
                       min_per_cell=min_per_cell)


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


def read_metadata_with_wav_headers(data_dir: Path) -> list[dict[str, Any]]:
    """Read seven-style metadata and WAV headers, retaining only in-range candidates."""
    import pyarrow.parquet as pq

    metadata = [row for row in read_metadata(data_dir) if row.get("style") in STYLES]
    by_shard: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metadata:
        by_shard[str(row["shard"])].append(row)

    for shard_name, shard_rows in sorted(by_shard.items()):
        parquet = pq.ParquetFile(data_dir / shard_name)
        by_group: dict[int, list[tuple[dict[str, Any], int]]] = defaultdict(list)
        offset = 0
        for group in range(parquet.num_row_groups):
            count = parquet.metadata.row_group(group).num_rows
            for row in shard_rows:
                row_index = int(row["row_index"])
                if offset <= row_index < offset + count:
                    by_group[group].append((row, row_index - offset))
            offset += count

        for group, selections in sorted(by_group.items()):
            table = parquet.read_row_group(group, columns=["audio", "id"])
            audio_column = table.column("audio")
            ids = table.column("id").to_pylist()
            for row, local_index in selections:
                if str(ids[local_index]) != str(row["id"]):
                    raise ValueError(f"parquet identity mismatch for {row['id']}")
                item = audio_column[local_index].as_py()
                raw = item.get("bytes") if isinstance(item, dict) else None
                if not isinstance(raw, (bytes, bytearray)):
                    row["duration_seconds"] = None
                    row["duration_status"] = "wav_header_error"
                    continue
                try:
                    duration = wav_duration(bytes(raw))
                except (OSError, EOFError, wave.Error, ValueError):
                    row["duration_seconds"] = None
                    row["duration_status"] = "wav_header_error"
                    continue
                row["duration_seconds"] = duration
                row["duration_status"] = ("in_range_wav_header" if duration_in_registered_range(duration)
                                           else "outside_2_to_10_seconds")
            del table, audio_column, ids
    return [row for row in metadata if duration_in_registered_range(row.get("duration_seconds"))]


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
    metadata = read_metadata_with_wav_headers(args.data_dir)
    manifest, pool_sizes = draw_manifest(metadata, seed=args.seed, per_cell=args.per_cell)
    gate = gate1_check(manifest)
    write_manifest(args.out, manifest)
    cluster_styles: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in manifest:
        cluster_styles[(row["speaker_id"], row["text_key"])].add(row["style"])
    shared_drawn_clusters = sum(len(styles) >= 2 for styles in cluster_styles.values())
    cluster_share = shared_drawn_clusters / len(cluster_styles) if cluster_styles else 0.0
    print({"manifest": str(args.out), "in_range_style_rows": len(metadata),
           "drawn_rows": len(manifest), "duration_filter": "WAV headers before shared-text pool",
           "gate1_in_range": gate["verdict"], "cell_counts": gate["cell_counts"],
           "pool_sizes_in_range": pool_sizes,
           "drawn_clusters_with_two_or_more_styles": shared_drawn_clusters,
           "drawn_cluster_share_with_two_or_more_styles": cluster_share})
    if len(manifest) == 0:
        return 1
    return 0 if gate["verdict"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
