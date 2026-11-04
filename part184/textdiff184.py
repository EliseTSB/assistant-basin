"""Part 184: compare the greedy answer text of the eager reruns with the stored SDPA runs (descriptive, not registered).

For every greedy answer present in both files (the target's answers at each clamp level and prompt, and each control
unit's answers at each control k), record whether the text is identical and, if not, the key position of the first
difference: the chat-template prompt length (neuron_report.build_prompt, thinking off) plus the 0-based index of the
first differing answer token. Also records whether the answer's scored outcome (score_text: alien, names_qwen) differs.

Usage: python -I part184/textdiff184.py [--out results/part184/greedy-text-compare.json]
Needs the Qwen3.5-4B tokenizer (QWEN_TOKENIZER, default the local Hugging Face snapshot). Written by Claude (Opus 5.5),
2026-11-04, after the runs, from the cold check's request that the method be in the repository.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import neuron_report as nr  # noqa: E402

PAIRS = {"A": ("results/part184/part18-eager/steer.json", "results/part18/chat/steer.json"),
         "B": ("results/part184/part18b-gen/steer.json", "results/part18b/phase-gen/steer.json"),
         "C": ("results/part184/part18b-prompt/steer.json", "results/part18b/phase-prompt/steer.json")}


def default_tokenizer() -> str:
    snap = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots"
    return str(next(snap.iterdir()))


def compare(tok, prompt: str, eager: str, sdpa: str) -> dict:
    se, ss = nr.score_text(eager, False), nr.score_text(sdpa, False)
    row = {"identical": eager == sdpa,
           "outcome_differs": (se["alien"], se["names_qwen"]) != (ss["alien"], ss["names_qwen"])}
    if not row["identical"]:
        pl = len(tok(nr.build_prompt(tok, prompt, False), add_special_tokens=False)["input_ids"])
        a = tok(eager, add_special_tokens=False)["input_ids"]
        b = tok(sdpa, add_special_tokens=False)["input_ids"]
        j = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        row["first_difference_key"] = pl + j
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="results/part184/greedy-text-compare.json")
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.environ.get("QWEN_TOKENIZER", default_tokenizer()))
    out = {}
    for name, (ef, sf) in PAIRS.items():
        E = json.loads((ROOT / ef).read_text(encoding="utf-8"))
        S = json.loads((ROOT / sf).read_text(encoding="utf-8"))
        rows = []
        for cond, answers in E["target"].items():
            for q, t in answers.items():
                u = S["target"].get(cond, {}).get(q)
                if u is not None:
                    rows.append({"kind": "target", "cond": cond, "prompt": q, **compare(tok, q, t, u)})
        sample_prompt = E.get("sample_prompt") or "Who made you?"
        for unit, by_k in (E.get("controls_greedy") or {}).items():
            if not isinstance(by_k, dict):
                continue
            for k, t in by_k.items():
                u = (S.get("controls_greedy") or {}).get(unit, {}).get(k)
                if isinstance(t, str) and isinstance(u, str):
                    rows.append({"kind": "control", "cond": f"unit {unit} k {k}", "prompt": sample_prompt,
                                 **compare(tok, sample_prompt, t, u)})
        summary = {}
        for kind in ("target", "control"):
            rs = [r for r in rows if r["kind"] == kind]
            keys = sorted(r["first_difference_key"] for r in rs if not r["identical"])
            summary[kind] = {"n": len(rs), "identical": sum(r["identical"] for r in rs),
                             "differ": len(keys), "differ_before_key_65": sum(k < 65 for k in keys),
                             "first_difference_keys": keys,
                             "outcome_differs": [{k: r[k] for k in ("cond", "prompt", "first_difference_key")}
                                                 for r in rs if r["outcome_differs"]]}
        out[name] = {"summary": summary, "rows": rows}
        print(name, json.dumps({k: {x: v[x] for x in ("n", "identical", "differ", "differ_before_key_65")}
                                for k, v in summary.items()}),
              "outcome changes:", summary["target"]["outcome_differs"] + summary["control"]["outcome_differs"])
    Path(ROOT / a.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
