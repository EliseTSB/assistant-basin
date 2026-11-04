"""Run analyze181.analyze with its pair comparisons computed in a process pool.

Added 2026-11-04 after the registration, as a runner only: no measure, threshold or rule changes. One
summarize_pair took 50.4 s on this host (pretrain_1T against pretrain_2T), so the 105 pairs of 15 runs take about
88 minutes in sequence. This driver computes every pair with the registered summarize_pair in worker processes,
then runs the registered analyze() with summarize_pair replaced by a lookup of those results. --check-pair A B also
computes one pair in the parent, in sequence, and refuses to write anything unless it equals the pooled result
exactly (as JSON).
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze181 as a181  # noqa: E402

_RUNS: dict = {}
_PRIMARY: dict = {}


def _init(runs_dir: str) -> None:
    passing, _excl, _g2 = a181._resolve_runs(Path(runs_dir))
    _RUNS.update({run.key: run for run in passing})
    _PRIMARY.update(a181._primary_ids_map(_RUNS.get("instruct")))


def _pair(keys: tuple[str, str]) -> tuple[str, dict]:
    a, b = keys
    return "__".join(sorted(keys)), a181.summarize_pair(_RUNS[a], _RUNS[b], primary_ids_by_layer=_PRIMARY)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--check-pair", nargs=2, required=True, metavar=("A", "B"))
    args = ap.parse_args(argv)
    t0 = time.time()
    _init(args.runs)
    keys = sorted(_RUNS)
    pairs = list(itertools.combinations(keys, 2))
    print(f"runs {len(keys)}, pairs {len(pairs)}, workers {args.workers}", flush=True)
    pooled: dict[str, dict] = {}
    # fork after loading, so the workers share the parent's loaded states instead of loading 15 copies each
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("fork")) as ex:
        for i, (name, summary) in enumerate(ex.map(_pair, pairs), 1):
            pooled[name] = summary
            if i % 10 == 0 or i == len(pairs):
                print(f"pairs done {i}/{len(pairs)} at {time.time() - t0:.0f} s", flush=True)
    check_name, check = _pair(tuple(args.check_pair))
    same = json.dumps(check, sort_keys=True, allow_nan=True) == json.dumps(pooled[check_name], sort_keys=True,
                                                                             allow_nan=True)
    print(f"check pair {check_name}: sequential equals pooled = {same}", flush=True)
    if not same:
        print("refusing to write: the pooled result differs from the sequential one", file=sys.stderr)
        return 3

    def lookup(a, b, primary_ids_by_layer=None):
        return pooled["__".join(sorted((a.key, b.key)))]

    a181.summarize_pair = lookup
    result = a181.analyze(Path(args.runs), Path(args.out))
    print(json.dumps({"included_runs": result["included_runs"], "pair_count": result["pair_count"],
                      "seconds": round(time.time() - t0)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
