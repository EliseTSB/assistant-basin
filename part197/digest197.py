#!/usr/bin/env python3
"""Compute a stable digest for the Part 196 recorder outputs used by Part 197."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "part196" / "manifest196.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "records_digest.json"
ARRAY_FIELDS = ("act_mean", "emb", "acoustic", "duration_seconds")


def read_manifest_ids(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("manifest contains duplicate recording ids")
    return sorted(ids)


def _array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def compute_records_digest(work_dir: Path, manifest_path: Path) -> dict[str, Any]:
    """Hash the byte content named by Controls (c), in sorted recording-id order."""
    records = []
    for clip_id in read_manifest_ids(manifest_path):
        record_path = work_dir / "records" / f"{clip_id}.npz"
        if not record_path.is_file():
            raise FileNotFoundError(f"record file is missing for {clip_id}: {record_path}")
        try:
            with np.load(record_path, allow_pickle=False) as saved:
                if "status" not in saved or str(saved["status"].item()) != "ok":
                    raise ValueError(f"record {clip_id} is not successfully recorded")
                if "audio_sha256" not in saved:
                    raise ValueError(f"record {clip_id} has no audio_sha256")
                audio_hash = str(saved["audio_sha256"].item())
                if len(audio_hash) != 64 or any(c not in "0123456789abcdef" for c in audio_hash):
                    raise ValueError(f"record {clip_id} has an invalid audio_sha256")
                missing = [field for field in ARRAY_FIELDS if field not in saved]
                if missing:
                    raise ValueError(f"record {clip_id} is missing digest fields: {missing}")
                record = {"id": clip_id, "audio_sha256": audio_hash}
                record.update({f"{field}_sha256": _array_sha256(saved[field])
                               for field in ARRAY_FIELDS})
                records.append(record)
        except (OSError, EOFError, ValueError, KeyError) as exc:
            raise ValueError(f"cannot digest record {clip_id}: {exc}") from exc
    # Review F1 (2026-11-08): durations.json decides the active clip set, so the digest binds its bytes too.
    durations_path = work_dir / "durations.json"
    if not durations_path.is_file():
        raise FileNotFoundError(f"durations.json is missing: {durations_path}")
    durations_sha256 = hashlib.sha256(durations_path.read_bytes()).hexdigest()
    canonical = json.dumps({"records": records, "durations_json_sha256": durations_sha256},
                           sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return {"schema": "part197-records-digest-v2", "records": records,
            "durations_json_sha256": durations_sha256,
            "records_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest()}


def write_records_digest(work_dir: Path, manifest_path: Path, out_path: Path) -> dict[str, Any]:
    digest = compute_records_digest(work_dir, manifest_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(digest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return digest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    digest = write_records_digest(args.work_dir, args.manifest, args.out)
    print(json.dumps({"records": len(digest["records"]),
                      "records_sha256": digest["records_sha256"],
                      "out": str(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
