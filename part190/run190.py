#!/usr/bin/env python3
"""Runner for Part 190 (PART190-DESIGN.md). Prints gate and prediction verdicts only, never an array or a value.

  python -I part190/run190.py                    the registered run on Part 188's committed arrays
                                                 (only after the design is registered on public main)
  python -I part190/run190.py --synthetic DIR    the same code on a synthetic fixture of the real shape (timing)

Writes results/part190/results.json (or DIR/results.json for --synthetic).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from part190 import analyze190 as A  # noqa: E402


def _floats(text: str) -> tuple[float, ...]:
    return tuple(float(t) for t in text.split(",") if t.strip())


def _ints(text: str) -> tuple[int, ...]:
    return tuple(int(t) for t in text.split(",") if t.strip())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb", type=Path, default=ROOT / "results/part188/emb188.npz")
    ap.add_argument("--manifest", type=Path, default=ROOT / "part188/manifest188.csv")
    ap.add_argument("--folds", type=Path, default=ROOT / "part188/folds.json")
    ap.add_argument("--results188", type=Path, default=ROOT / "results/part188/results.json")
    ap.add_argument("--out", type=Path, default=ROOT / "results/part190")
    ap.add_argument("--synthetic", type=Path, default=None,
                    help="write a synthetic fixture of the real shape to this folder and run on it")
    ap.add_argument("--ranks", type=_ints, default=A.RANKS)
    ap.add_argument("--doses", type=_floats, default=A.DOSES)
    ap.add_argument("--controls", type=int, default=A.N_CONTROLS)
    ap.add_argument("--boot", type=int, default=A.BOOTSTRAPS)
    ap.add_argument("--expected-rows", type=int, default=3138)
    args = ap.parse_args(argv)
    registered_settings = (args.synthetic is None and args.ranks == A.RANKS and args.doses == A.DOSES and args.controls == A.N_CONTROLS
                           and args.boot == A.BOOTSTRAPS and args.expected_rows == 3138)
    if args.synthetic is not None:
        from part190.synthetic190 import make_fixture, registered_from_fixture, write_fixture
        fx = make_fixture(dim=768, n_real=85, n_imitation=67, n_esc_classes=28, n_per_esc=40, n_human_classes=10,
                          n_nonanimal=269, seed=190)
        paths = write_fixture(fx, args.synthetic, registered_from_fixture(fx))
        args.emb, args.manifest, args.folds, args.results188 = (paths["emb"], paths["manifest"], paths["folds"],
                                                                paths["results188"])
        args.out = args.synthetic
        args.expected_rows = len(fx["rows"])
        print(f"synthetic fixture: {len(fx['rows'])} rows x 768 in {args.synthetic}")
    elif not registered_settings:
        print("note: settings differ from the registered ones; this is not the registered run")
    start = time.time()
    result = A.run_files(args.emb, args.manifest, args.folds, args.results188, args.out, ranks=args.ranks,
                         doses=args.doses, n_controls=args.controls, n_boot=args.boot,
                         expected_rows=args.expected_rows, log=lambda m: print(f"[{time.time() - start:7.1f}s] {m}"))
    elapsed = time.time() - start
    gates = result.get("gates", {})
    for name in ("gate1", "gate2", "gate3", "gate4"):
        if name in gates:
            print(f"{name}: {'passed' if gates[name]['passed'] else 'FAILED'}")
    if "gate3_self_test" in gates:
        print(f"gate3 self-test shown on this data: {gates['gate3_self_test'].get('shown')}")
    print(f"readable: {result.get('readable')}")
    for pid, p in result.get("predictions", {}).items():
        print(f"{pid}: {p['verdict']}")
    if "headline" in result:
        print(f"headline permitted: {result['headline']['permitted']}")
    for name, d in result.get("directions", {}).items():
        hg = d.get("ranks", {}).get("1", {}).get("exploratory_half_gain")
        if hg is not None:
            if hg.get("alpha") is None:
                print(f"exploratory half-gain dose, {name}: {hg.get('note')}")
            else:
                print(f"exploratory half-gain dose, {name}: alpha {hg['alpha']:g} (operating {hg['operating_alpha']:g}), "
                      f"rise {hg['dV_test']:+.3f}, C1 {hg['C1_change_points']:+.2f} pt, "
                      f"C2 {hg['C2_first_neighbour_changed']:.3f}, C5 {hg['C5_change_points']:+.2f} pt, "
                      f"C6 {hg['C6_change_points']:+.2f} pt")
    n_eval = (2 * len(args.ranks) * (len([d for d in args.doses if d != 0]) * (1 + 2 * args.controls)) + 2)
    print(f"{n_eval} edit evaluations in {elapsed:.0f} s; wrote {args.out / 'results.json'}")
    (args.out / "run.json").write_text(json.dumps({"elapsed_s": elapsed, "edit_evaluations": n_eval,
                                                   "registered_settings": registered_settings,
                                                   "synthetic": args.synthetic is not None}, indent=1) + "\n",
                                       encoding="utf-8")
    return 0 if result.get("readable") else 3


if __name__ == "__main__":
    sys.exit(main())
