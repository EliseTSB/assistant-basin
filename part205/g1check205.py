#!/usr/bin/env python3
"""g1check205.py: the A770 G1 check for family Q (Design, Gates, G1; Size, line 204).

One question, run 5 times plain and 5 times with the do-nothing hook (the same replacement code with the receiving
run's own activations), batch size 1, bf16, eager, through neuron_report.load_model. Writes the ten (p_a, p_b) rows and
score205.gate_g1_a770's verdict to <out>/g1.json. The main run's G1 is still computed from its own do-nothing rows;
this check only sets the tolerance rule for the card.

  python g1check205.py --stage qwen-chat --out results/part205/g1
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common205 as C  # noqa: E402
import run205 as R  # noqa: E402
import score205 as S  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", default="qwen-chat")
    ap.add_argument("--out", default=str(HERE.parent / "results" / "part205" / "g1"))
    ap.add_argument("--config", default=str(HERE / "config.json"))
    ap.add_argument("--prompts", default=str(HERE / "prompts205.json"))
    ap.add_argument("--device", default="xpu")
    a = ap.parse_args(argv)
    cfg = C.load_config(a.config)
    prompts = C.load_prompts(a.prompts)
    tok, model, env = R.load_stage(cfg, a.stage, a.device)
    units = C.units_for(model)
    C.check_pool(units, cfg["families"]["Q"]["units"])
    runner = R.Runner(model, tok, units, {"A": C.option_token_ids(tok, "A"), "B": C.option_token_ids(tok, "B")})
    item = prompts["items"]["honesty"]["say"][0]
    text = R.build_prompt(prompts, "You", "honesty", item["question"])
    n = cfg["a770_g1_check"]["plain_runs"]
    plain, dn = [], []
    ids = runner.tokens(text)[0].tolist()
    _, _, acts, _ = runner.run(text, need_grad=False)
    by_site = {}
    for u in units:
        by_site.setdefault((u.site, u.layer), []).append(u)
    replace = []
    for (site, layer), us in by_site.items():
        cols = []
        for u in sorted(us, key=lambda u: u.index):
            cols.extend(u.cols)
        replace.append((site, layer, cols, list(range(len(ids))), acts[f"{site}:{layer}"][0]))
    for _ in range(n):
        _, r, _, _ = runner.run(text, need_grad=False, capture=False)
        plain.append([r["p_a"], r["p_b"]])
    for _ in range(n):
        _, r, _, _ = runner.run(text, need_grad=False, capture=False, replace=replace)
        dn.append([r["p_a"], r["p_b"]])
    verdict = S.gate_g1_a770([tuple(x) for x in plain], [tuple(x) for x in dn])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "g1.json").write_text(json.dumps({"env": env, "question": item["id"], "plain": plain, "do_nothing": dn,
                                             "verdict": verdict, "at": time.strftime("%Y-%m-%dT%H:%M:%S")},
                                            indent=1) + "\n", encoding="utf-8")
    print(json.dumps(verdict, indent=1))
    return 0 if verdict["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
