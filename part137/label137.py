#!/usr/bin/env python3
"""Part 137 reader wrapper using Part 134's established judge transport."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "part134"))
import label134  # noqa: E402


def main(argv=None, *, request_fn=None, sleep_fn=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("gate-ds", "gate-luna", "ds", "luna"), required=True)
    p.add_argument("--answers", default="results/part137/answers.jsonl")
    p.add_argument("--out")
    p.add_argument("--known-ds", default="results/part137/known-ds.jsonl")
    p.add_argument("--known-luna", default="results/part137/known-luna.jsonl")
    p.add_argument("--ds-labels", default="results/part137/labels-ds.jsonl")
    p.add_argument("--known")
    a = p.parse_args(argv)
    output = a.out or ("results/part137/labels-luna.jsonl" if a.stage == "luna" else
                       "results/part137/known-luna.jsonl" if a.stage == "gate-luna" else
                       "results/part137/known-ds.jsonl" if a.stage == "gate-ds" else
                       "results/part137/labels-ds.jsonl")
    forwarded = ["--stage", a.stage, "--answers", a.answers, "--out", output,
                 "--known-ds", a.known_ds, "--known-luna", a.known_luna, "--ds-labels", a.ds_labels]
    if a.known: forwarded += ["--known", a.known]
    return label134.main(forwarded, request_fn=request_fn, sleep_fn=sleep_fn)


if __name__ == "__main__": raise SystemExit(main())
