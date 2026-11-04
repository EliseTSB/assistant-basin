"""Part 184: read the eager reruns of Parts 18 and 18b beside the stored SDPA runs, and score 184.1 to 184.6.

Usage: python -I part184/compare184.py [--eager results/part184] [--out results/part184/compare.json]
Reads only the tool's own summaries in each steer.json (the fields Parts 18 and 18b published). Written by Claude
(Opus 5.5), 2026-11-04, before runs A and B finished (tested on run C's file).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RUNS = {  # name: (eager folder, stored SDPA file)
    "A": ("part18-eager", "results/part18/chat/steer.json"),
    "B": ("part18b-gen", "results/part18b/phase-gen/steer.json"),
    "C": ("part18b-prompt", "results/part18b/phase-prompt/steer.json"),
}


def counts(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    s = d["summary"]
    cs = s.get("controls_sampled") or {}
    kernel = (d.get("load") or {}).get("attn_implementation")
    return {
        "kernel": kernel,
        "samples": {c: {"n": v["n"], "alien": v["alien"], "names_qwen": v["names_qwen"]} for c, v in s["samples"].items()},
        "controls_greedy": {k: {"n_units": v["n_units"], "alien": v["alien"], "names_qwen": v["still_names_qwen"]}
                            for k, v in s["controls_greedy"].items()},
        "controls_sampled": None if not cs else {
            "k": cs["k"], "n": cs["n_units"] * cs["samples_per_unit"],
            "alien": sum(cs["alien_per_unit"].values()), "names_qwen": sum(cs["names_qwen_per_unit"].values())},
        "empty_samples": sum(1 for v in d["samples"].values() for t in (v if isinstance(v, list) else []) if not t.strip()),
    }


def verdicts(e: dict) -> dict:
    out = {}

    def rule(pid, ok, number):
        out[pid] = {"verdict": "held" if ok else "failed", "number": number}

    if "A" in e:
        a = e["A"]
        rule("184.1", a["samples"]["+2"]["alien"] >= 7, f'{a["samples"]["+2"]["alien"]} of 8 alien at +2')
        unc, cs = a["samples"]["unclamped"]["alien"], a["controls_sampled"]["alien"]
        rule("184.2", unc == 0 and cs <= 2, f"unclamped {unc} of 8 alien; controls {cs} of {a['controls_sampled']['n']} alien")
        cg = a["controls_greedy"]["+2"]
        rule("184.3", cg["names_qwen"] >= 18 and cg["alien"] == 0,
             f'controls greedy at +2: {cg["names_qwen"]} of {cg["n_units"]} name Qwen, {cg["alien"]} alien')
        rule("184.4", a["samples"]["-2"]["names_qwen"] <= 4, f'{a["samples"]["-2"]["names_qwen"]} of 8 name Qwen at -2')
    if "B" in e:
        rule("184.5", e["B"]["samples"]["+2"]["alien"] >= 5, f'{e["B"]["samples"]["+2"]["alien"]} of 8 alien, generation only')
    if "C" in e:
        rule("184.6", e["C"]["samples"]["+2"]["alien"] <= 1, f'{e["C"]["samples"]["+2"]["alien"]} of 8 alien, prompt only')
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eager", default="results/part184")
    ap.add_argument("--out", default="results/part184/compare.json")
    a = ap.parse_args(argv)
    eager, sdpa = {}, {}
    for name, (folder, stored) in RUNS.items():
        f = Path(a.eager) / folder / "steer.json"
        if f.is_file():
            eager[name] = counts(f)
            sdpa[name] = counts(Path(stored))
    bad = [n for n, c in eager.items() if c["kernel"] != "eager"]
    result = {"gate_kernel_eager": not bad, "not_eager": bad, "eager": eager, "sdpa": sdpa,
              "verdicts": verdicts(eager) if not bad else {}}
    Path(a.out).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"gate_kernel_eager": not bad, "runs": sorted(eager),
                      "verdicts": {k: v["verdict"] + ": " + v["number"] for k, v in result["verdicts"].items()}}, indent=1))
    return 0 if not bad else 2


if __name__ == "__main__":
    raise SystemExit(main())
