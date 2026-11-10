#!/usr/bin/env python3
"""Build Part 203's CREMA-D clip list and five actor-disjoint folds from the public file list."""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
DEFAULT_TSV = HERE / "prior" / "cremad-audiowav.tsv"
DEFAULT_LIST = HERE / "download-list.txt"
DEFAULT_MANIFEST = HERE / "manifest203.csv"

EMOTIONS = ("ANG", "DIS", "FEA", "HAP", "NEU", "SAD")
PER_CELL = 5
N_FOLDS = 5
MANIFEST_FIELDS = ("file", "actor", "sentence", "emotion", "intensity", "fold", "cell_rank")


def parse_filename(name: str) -> dict[str, str]:
    """Split a CREMA-D file name into actor, sentence, emotion and intensity."""
    stem = name[:-4] if name.lower().endswith(".wav") else name
    parts = stem.split("_")
    if len(parts) != 4:
        raise ValueError(f"unexpected CREMA-D file name: {name!r}")
    actor, sentence, emotion, intensity = parts
    if emotion not in EMOTIONS:
        raise ValueError(f"unknown CREMA-D emotion code {emotion!r} in {name!r}")
    return {"file": name, "actor": actor, "sentence": sentence, "emotion": emotion,
            "intensity": intensity}


def read_file_list(tsv_path: Path) -> list[dict[str, str]]:
    """Read the public file list (path<TAB>size); only the AudioWAV file names are used."""
    rows = []
    seen = set()
    with tsv_path.open(encoding="utf-8", newline="") as stream:
        for line in stream:
            line = line.rstrip("\n")
            if not line or line.lower().startswith("path\t"):
                continue
            path = line.split("\t")[0].strip()
            if not path.lower().endswith(".wav"):
                raise ValueError(f"file list row is not a WAV: {line!r}")
            name = path.rsplit("/", 1)[-1]
            if name in seen:
                raise ValueError(f"duplicate file name in the list: {name}")
            seen.add(name)
            rows.append(parse_filename(name))
    if not rows:
        raise ValueError(f"no clips found in {tsv_path}")
    return rows


def name_hash(name: str) -> str:
    """The fixed ordering key: SHA-256 hex of the file name."""
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def select_clips(rows: Iterable[dict[str, str]], per_cell: int = PER_CELL) -> list[dict[str, str]]:
    """Keep the first `per_cell` file names by SHA-256 hex within each (actor, emotion) cell."""
    if per_cell <= 0:
        raise ValueError("per_cell must be positive")
    cells: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        cells[(row["actor"], row["emotion"])].append(dict(row))
    selected = []
    for (actor, emotion), members in sorted(cells.items()):
        ranked = sorted(members, key=lambda row: name_hash(row["file"]))
        for rank, row in enumerate(ranked[:per_cell], start=1):
            row["cell_rank"] = str(rank)
            selected.append(row)
    selected.sort(key=lambda row: (row["actor"], EMOTIONS.index(row["emotion"]), row["file"]))
    return selected


def assign_folds(actors: Iterable[str], n_folds: int = N_FOLDS) -> dict[str, int]:
    """Split actors into `n_folds` folds by SHA-256 hex order of the actor id, balanced by size."""
    unique = sorted(set(str(actor) for actor in actors))
    if n_folds <= 0:
        raise ValueError("n_folds must be positive")
    if len(unique) < n_folds:
        raise ValueError(f"cannot make {n_folds} actor folds from {len(unique)} actors")
    ordered = sorted(unique, key=lambda actor: (name_hash(actor), actor))
    return {actor: index % n_folds for index, actor in enumerate(ordered)}


def build_manifest(rows: Iterable[dict[str, str]], per_cell: int = PER_CELL,
                   n_folds: int = N_FOLDS) -> list[dict[str, str]]:
    selected = select_clips(rows, per_cell=per_cell)
    folds = assign_folds((row["actor"] for row in selected), n_folds=n_folds)
    manifest = []
    for row in selected:
        manifest.append({"file": row["file"], "actor": row["actor"], "sentence": row["sentence"],
                         "emotion": row["emotion"], "intensity": row["intensity"],
                         "fold": str(folds[row["actor"]]), "cell_rank": row["cell_rank"]})
    return manifest


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    required = set(MANIFEST_FIELDS)
    if not required.issubset(rows[0]):
        raise ValueError(f"manifest is missing columns: {sorted(required - set(rows[0]))}")
    files = [row["file"] for row in rows]
    if len(files) != len(set(files)):
        raise ValueError("manifest contains duplicate file names")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv", type=Path, default=DEFAULT_TSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--per-cell", type=int, default=PER_CELL)
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    args = parser.parse_args(argv)
    manifest = build_manifest(read_file_list(args.tsv), per_cell=args.per_cell, n_folds=args.folds)
    write_manifest(args.out, manifest)
    print({"manifest": str(args.out), "clips": len(manifest),
           "actors": len({row["actor"] for row in manifest}),
           "folds": {fold: len({row["actor"] for row in manifest if row["fold"] == fold})
                     for fold in sorted({row["fold"] for row in manifest})}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
