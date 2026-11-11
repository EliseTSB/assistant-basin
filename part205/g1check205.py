#!/usr/bin/env python3
"""g1check205.py: the A770 G1 check for family Q (Design, Gates, G1, lines 174-178; Size, line 204).

One question (the first held-out SAY question of the first value, in the first third-person frame, the frame every
main-run do-nothing row receives), run 5 times plain and 5 times with the do-nothing hook (the same replacement code
with the receiving run's own activations, every part, at the copied span's positions: the question block, as the copy
arms use it), batch size 1, bf16, eager, through neuron_report.load_model. The full last-position logits are compared;
only the scalar differences are written: plain against plain over all 10 pairs, each do-nothing run against the first
plain run, and score205.gate_g1_a770's verdict, to <out>/g1.json. score205.py reads that file for family Q: when the
plain runs are not bit-identical, the main run's do-nothing rows are held to twice the largest plain difference.

The do-nothing values are moved to the CPU first, exactly as the main run stores its activations, so the check
exercises the main run's CPU-to-device replacement path (review r2 H-N6).

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
    return check(tok, model, env, cfg, prompts, Path(a.out), check_pool=True)


def check(tok, model, env, cfg, prompts, out, check_pool=True):
    """The 5 + 5 runs and the verdict; `model`/`tok` may be a tiny test model."""
    units = C.units_for(model)
    if check_pool:
        C.check_pool(units, cfg["families"]["Q"]["units"])
    runner = R.Runner(model, tok, units, {"A": C.option_token_ids(tok, "A"), "B": C.option_token_ids(tok, "B")})
    value = prompts["values"][0]
    item = next(i for i in prompts["items"][value]["say"] if i["split"] == "heldout")
    name = next(n for n in prompts["personas"] if n != "You")
    persona = R.persona_text(prompts, name, value)
    text = C.frame_text(persona, item["question"])
    q0, ids = C.question_start(tok, text, persona)
    positions = list(range(q0, len(ids)))
    n = cfg["a770_g1_check"]["plain_runs"]
    plain, dn, plain_read, dn_read = [], [], [], []
    for _ in range(n):
        _, r, acts, _ = runner.run(text, need_grad=False, capture=True)
        plain.append(runner.last_logits)
        plain_read.append([r["p_a"], r["p_b"]])
    # The main run stores every captured activation on the CPU (run205.py's `.detach().cpu().clone()`) and passes
    # those CPU values into the hook, which must move them to the model's device (review r2 H-N6, D-F4). This check
    # takes the same path: the values are put on the CPU first, exactly as the main run's are.
    own = {k: v.detach().cpu().clone() for k, v in C.values_at(acts, positions).items()}
    for _ in range(n):
        _, r, _, _ = runner.replace_and_read(text, [u.key for u in units], positions, own, item["consistent"])
        dn.append(runner.last_logits)
        dn_read.append([r["p_a"], r["p_b"]])
    verdict = S.gate_g1_a770(plain, dn)
    out.mkdir(parents=True, exist_ok=True)
    C.atomic_write_json(out / "g1.json", {"env": env, "question": item["id"], "name": name,
                                          "n_positions": len(positions), "plain_p": plain_read,
                                          "do_nothing_p": dn_read, "verdict": verdict,
                                          "values_device": "cpu (as the main run stores them)",
                                          "at": time.strftime("%Y-%m-%dT%H:%M:%S")}, sort_keys=False)
    print(json.dumps(verdict, indent=1))
    return 0 if verdict["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
