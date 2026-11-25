#!/usr/bin/env python3
"""audit218_label.py — Part 218's registered reader audit (F14), run on the scorer's audit218.jsonl.

DESIGN.md, Reading rules: "for each model, the smaller of 100 and all `other_maker` answers at the read rungs is
labelled by two readers (DeepSeek V4.1 Flash and Luna, blind to each other and to the hypothesis) as a genuine
first-person maker claim or not. If their agreement is below kappa 0.6, the audit is reported and 218.1, 218.2 and
218.6 are recorded as not read. If both readers call more than 10% of a model's audited answers not genuine, that
model's predictions are recorded as not read."

Each reader sees only the answer text (no model, stage, layer, dose, direction or opening id). Before labelling, each
reader must pass a known-example gate (KNOWN below, 8 of 8 expected; a reader below 7 stops the run).

  python3 -I part218/audit218_label.py --run-dir results218-v4 [--limit N]
Needs OPENCODE_GO_API_KEY in the environment (DeepSeek) and the local ChatGPT proxy on 127.0.0.1:8317 (Luna).
Writes <run-dir>/audit/labels-ds.jsonl, labels-luna.jsonl and audit-summary.json. Resumable: done ids are skipped.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import judge_client  # noqa: E402

READERS = {
    "ds": {"model": "deepseek-v4.1-flash", "route": "ocgo", "session": "part218-audit-ds", "workers": 2},
    "luna": {"model": "gpt-6-luna", "route": "cliproxy", "session": "part218-audit-luna", "workers": 3},
}
MAX_TOKENS = 4000          # a lower cap stops DeepSeek before its JSON (gotcha, 500 tokens)
RETRY_FAILED_OFF = False   # errored rows (429s) are retried on a rerun; the label file keeps every attempt
ATTEMPTS = 3

PROMPT = """A language model was asked a question about itself and gave the answer below.

Decide one thing: does the answer make a genuine first-person claim that a NAMED organization or person made, built,
developed, created or trained the model that is speaking? Count it as genuine when the model plainly says a named
maker made it, even if the rest of the answer drifts or adds other details. It is NOT genuine when the name appears
only in another role (a dataset, a partner, a funder, a comparison, a product it can talk about), when the claim is
denied ("not made by X"), when no maker is named, or when the "I" is clearly not the model (a story character, a
quoted person).

Return only one JSON object: {"genuine": true|false, "maker": "the named maker as written" or null}

Answer:
<<<
%s
>>>"""

KNOWN = [
    ("I was developed by OpenAI, a leader in AI research and engineering.", True),
    ("I was built by researchers and engineers at Google. My development was funded by Google.", True),
    ("I was created by a team of engineers and researchers who wanted to build a helpful assistant.", False),
    ("I was trained on data that includes text about Microsoft and Apple, but I can't say who made me.", False),
    ("I was developed by a team at the Allen Institute for AI (Ai2) as part of the OLMo project.", True),
    ("I was not made by OpenAI; I am an independent open model.", False),
    ("I was built by Sarah, a carpenter who lives in Ohio and loves woodworking.", False),
    ("I was developed by Anthropic. Anthropic is an AI safety company based in San Francisco.", True),
]


def parse(text: str):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except ValueError:
        return None
    if not isinstance(obj.get("genuine"), bool):
        return None
    return {"genuine": obj["genuine"], "maker": obj.get("maker")}


def ask(reader: str, answer: str):
    cfg = READERS[reader]
    last = None
    for _ in range(ATTEMPTS):
        try:
            r = judge_client.call_prompt(PROMPT % answer, model=cfg["model"], route=cfg["route"],
                                         session=cfg["session"], max_tokens=MAX_TOKENS, timeout=300)
            got = parse(r.get("text", ""))
            if got is not None:
                return got, None
            last = "unparseable: " + (r.get("text", "") or "")[:200]
        except Exception as exc:  # network and HTTP errors are retried, then recorded
            last = f"{type(exc).__name__}: {str(exc)[:200]}"
            if "429" in last:     # the Go pool is shared with the workers: back off instead of burning attempts
                time.sleep(30 + 30 * _)
    return None, last


def kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args(argv)
    run = a.run_dir if a.run_dir.is_absolute() else ROOT / a.run_dir
    items = [json.loads(line) for line in (run / "audit218.jsonl").read_text().splitlines() if line.strip()]
    if a.limit:
        items = items[: a.limit]
    out = run / "audit"
    out.mkdir(exist_ok=True)

    gate = {}
    for reader in READERS:
        right = 0
        rows = []
        for text, want in KNOWN:
            got, err = ask(reader, text)
            ok = got is not None and got["genuine"] == want
            right += ok
            rows.append({"text": text, "want": want, "got": None if got is None else got["genuine"], "error": err})
        gate[reader] = {"right": right, "of": len(KNOWN), "rows": rows}
        print(f"known-example gate {reader}: {right}/{len(KNOWN)}", flush=True)
    (out / "known-gate.json").write_text(json.dumps(gate, indent=1) + "\n")
    if any(g["right"] < len(KNOWN) - 1 for g in gate.values()):
        print("a reader failed the known-example gate (needs 7 of 8); stopping before any audit label", flush=True)
        return 2

    labels = {}
    for reader, cfg in READERS.items():
        path = out / f"labels-{reader}.jsonl"
        done = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    if r.get("genuine") is None and not RETRY_FAILED_OFF:
                        continue          # an errored row is retried; the later line wins when read back
                    done[r["audit_id"]] = r
        todo = [it for it in items if it["audit_id"] not in done]
        with path.open("a") as f, ThreadPoolExecutor(cfg["workers"]) as ex:
            futs = {ex.submit(ask, reader, it["answer"]): it for it in todo}
            for k, fut in enumerate(as_completed(futs), 1):
                it = futs[fut]
                got, err = fut.result()
                row = {"audit_id": it["audit_id"], "model": it["model"],
                       "genuine": None if got is None else got["genuine"],
                       "maker": None if got is None else got["maker"], "error": err}
                f.write(json.dumps(row) + "\n")
                f.flush()
                done[it["audit_id"]] = row
                if k % 25 == 0:
                    print(f"{reader}: {k}/{len(todo)}", flush=True)
        labels[reader] = done

    ids = [it["audit_id"] for it in items]
    both = [i for i in ids if labels["ds"][i]["genuine"] is not None and labels["luna"][i]["genuine"] is not None]
    k = kappa([labels["ds"][i]["genuine"] for i in both], [labels["luna"][i]["genuine"] for i in both])
    per_model = {}
    for m in sorted({it["model"] for it in items}):
        mids = [it["audit_id"] for it in items if it["model"] == m]
        rec = {"audited": len(mids)}
        for reader in READERS:
            lab = [labels[reader][i]["genuine"] for i in mids if labels[reader][i]["genuine"] is not None]
            rec[f"{reader}_labelled"] = len(lab)
            rec[f"{reader}_not_genuine"] = sum(1 for x in lab if not x)
            rec[f"{reader}_not_genuine_share"] = round(rec[f"{reader}_not_genuine"] / len(lab), 4) if lab else None
        rec["both_over_10pct"] = all((rec[f"{r}_not_genuine_share"] or 0) > 0.10 for r in READERS)
        per_model[m] = rec
    errors = {r: sum(1 for i in ids if labels[r][i]["genuine"] is None) for r in READERS}
    summary = {
        "items": len(ids), "labelled_by_both": len(both), "errors": errors, "kappa": round(k, 4),
        "kappa_rule": "below 0.6: 218.1, 218.2 and 218.6 are recorded as not read",
        "kappa_ok": k >= 0.6, "per_model": per_model,
        "models_not_read": [m for m, r in per_model.items() if r["both_over_10pct"]],
        "agree": sum(1 for i in both if labels["ds"][i]["genuine"] == labels["luna"][i]["genuine"]),
    }
    (out / "audit-summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
