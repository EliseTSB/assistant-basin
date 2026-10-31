#!/usr/bin/env python3
"""Prepare Gate 1 checks using stable (slice, line) corpus identities."""
import argparse
import hashlib
import random
import runpy
from pathlib import Path

import pyarrow.parquet as pq
import torch

_config = runpy.run_path(str(Path(__file__).with_name("config129.py")))
CORPUS_DOCS, CORPUS_ORDER, MIN_LEN = _config["CORPUS_DOCS"], _config["CORPUS_ORDER"], _config["MIN_LEN"]


def prepare(catalog, output):
    catalog = Path(catalog)
    docs = pq.read_table(catalog / "docs.parquet").to_pylist()
    doc_key = {int(row["doc"]): (str(row["slice"]), int(row["line"])) for row in docs}
    tokens_by_key = {(str(row["slice"]), int(row["line"])): int(row["tokens"]) for row in docs}
    eligible = {(name, line) for name, count in CORPUS_DOCS.items()
                for line in range(min(100, count))
                if tokens_by_key.get((name, line), 0) >= MIN_LEN}

    rows_main, rows_pilot = [], []
    thresholds = []
    for layer in range(36):
        table = pq.read_table(catalog / "neurons/acts" / f"L{layer:02d}.parquet",
                              columns=["neuron", "pole", "act_top", "top_doc", "top_pos", "top_val"])
        records = table.to_pylist()
        width = max(int(row["neuron"]) for row in records) + 1
        layer_thresholds = [[0.0] * width for _ in range(2)]
        for row in records:
            layer_thresholds[0 if int(row["pole"]) > 0 else 1][int(row["neuron"])] = float(row["act_top"])
            for rank, (doc, pos, value) in enumerate(zip(row["top_doc"], row["top_pos"], row["top_val"])):
                key = doc_key.get(int(doc))
                if key not in eligible or float(value) <= 0:
                    continue
                item = {"layer": layer, "neuron": int(row["neuron"]), "pole": int(row["pole"]),
                        "slice": key[0], "line": key[1], "position": int(pos), "top_val": float(value)}
                if rank == 0:
                    rows_main.append(item)
                rows_pilot.append(item)
        thresholds.append(layer_thresholds)

    rng = random.Random(129)
    rng.shuffle(rows_main)
    rng.shuffle(rows_pilot)
    main_checks = rows_main[:500]
    pilot_checks = rows_pilot[:300]
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    docs_sha256 = hashlib.sha256((catalog / "docs.parquet").read_bytes()).hexdigest()
    payload = {"alignment": main_checks, "pilot_alignment": pilot_checks,
               "docs_sha256": docs_sha256,
               "thresholds": torch.tensor(thresholds, dtype=torch.float32)}
    tmp = out.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(out)
    print(f"prepared {len(main_checks)} main alignment checks")
    print(f"prepared {len(pilot_checks)} pilot alignment checks from {len(rows_pilot)} eligible top firings")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("catalog")
    parser.add_argument("output")
    args = parser.parse_args()
    prepare(args.catalog, args.output)
