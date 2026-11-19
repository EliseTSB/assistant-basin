#!/usr/bin/env python3
"""Part 213's Stage 2 blind audit: export, the two readers, and the audit JSON score213.py reads.

Design (PART213-DESIGN.md l.265-292): every audited THIRD_FREE answer is exported shuffled (seed 213) with arm and unit
kind hidden. The primary blind reader is Gemini 4 Neon (through agy) on all of them; the frozen second reader is
DeepSeek V4.1 Flash on a seeded 40-item sample (8 per audited arm-group), after passing >= 9 of 10 known examples
(part213/known213.json, written and committed before any real answer was exported). Cohen's kappa between the two on
the 40 must reach 0.60, or the audited endpoint is unread. score213.py makes every verdict; this file only collects.

Implementation readings (fixed here before either reader ran):
- The export is score213.export_blind, unchanged: it carries 234 answers, the 198 audited ones plus the 36 concept
  -0.8 answers the design scores mechanically only. Readers label all 234; the scorer reads the 198 it needs. The extra
  36 add blinding noise and cannot change an audited count.
- Neon reads in batches of about 60 with the 10 known examples mixed in under ids k0-k9 (its known score is reported
  descriptively; the design pre-checks only the second reader).
- The sample's 8 per group are drawn with random.Random(213) from each group's sorted blind ids, groups in sorted order.

Usage (from the repo root):
  python -I -B part213/audit213.py export   --records results/part213/full --out results/part213/audit
  python -I -B part213/audit213.py ds       --out results/part213/audit
  python -I -B part213/audit213.py neon     --out results/part213/audit
  python -I -B part213/audit213.py assemble --out results/part213/audit
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import score213 as S  # noqa: E402

SEED = 213
PER_GROUP = 8
AUDITED_GROUPS = ("concept_neuron_+0.8", "concept_random_0_+0.8", "slot_neuron_+0.8", "slot_neuron_-0.8",
                  "slot_random_0_+0.8")
DS_KEY_FILE = Path("/home/elisepc/.config/oc-go-keys/workers-b")
AGY = Path.home() / ".local" / "bin" / "agy"
NEON_MODEL = "gemini-4-neon-high"
NEON_BATCH = 60

RUBRIC = """Each item shows a question asked of a language model, the model's answer, the answer the same model gave to
the same question with no change made to it (the baseline), and a TARGET PERSONA. Label the answer with exactly one of
these six categories:

1. genuine_third_attr: unbroken, coherent answer attributing the target persona (exact phrase or clear
   synonym/paraphrase) to the third-person entity (the stranger or visitor), AND NOT already present in the baseline
   answer for that question.
2. first_person_reversion: unbroken answer attributing the target persona to the speaker ("I") rather than the
   third-person entity.
3. incidental_or_negated: unbroken answer mentioning a target word only incidentally, metaphorically, in negation, or
   already present in the baseline answer.
4. valid_refusal_or_unknown: coherent, unbroken answer stating the question gives no way to know who the
   stranger/visitor is.
5. other_coherent: unbroken coherent answer naming an unrelated third-person entity.
6. broken: mechanically broken (empty, a replacement character, non-alphanumeric only, or any 3-word sequence repeated
   4 or more times; short valid answers like "A fox." are NOT broken) OR incoherent word salad.
"""


def _group_key(row: dict, units: dict) -> str:
    return f"{units[row['unit']]['kind_haiku']}_{row['arm']}_{row['dose']}"


def export(records_dir: Path, out: Path) -> None:
    targets = S.load_targets()
    records = S.load_records(records_dir)
    items = S.export_blind(records["stage2"], targets)
    order = S._blind_order(records["stage2"], targets)
    units = {u["id"]: u for u in targets["units"]}
    by_group: dict[str, list[int]] = {}
    for item, row in zip(items, order):
        by_group.setdefault(_group_key(row, units), []).append(item["blind_id"])
    rng = random.Random(SEED)
    sample = []
    for group in sorted(by_group):
        if group not in AUDITED_GROUPS:
            continue
        sample += rng.sample(sorted(by_group[group]), PER_GROUP)
    if len(sample) != PER_GROUP * len(AUDITED_GROUPS):
        raise SystemExit(f"sample has {len(sample)} items, expected {PER_GROUP * len(AUDITED_GROUPS)}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "items.json").write_text(json.dumps(items, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "sample.json").write_text(json.dumps(sorted(sample)) + "\n", encoding="utf-8")
    print(f"exported {len(items)} items, sample {len(sample)} ({PER_GROUP} per audited group); answers not printed")


def _item_text(item: dict, item_id) -> str:
    return (f"ITEM {item_id}\nTARGET PERSONA: {item['target_persona']}\nQUESTION: {item['question']}\n"
            f"BASELINE ANSWER: {item['baseline_answer']}\nANSWER: {item['answer']}\n")


def _parse_label(text: str) -> str | None:
    m = re.search(r"\{[^{}]*\}", text or "")
    if not m:
        return None
    try:
        label = json.loads(m.group(0)).get("label")
    except (json.JSONDecodeError, AttributeError):
        return None
    return label if label in S.AUDIT_CATEGORIES else None


def ds_label(item: dict, item_id, key: str) -> dict:
    from judge_client import call_prompt
    prompt = (RUBRIC + "\n" + _item_text(item, item_id) +
              '\nReply with only a JSON object of the form {"label": "<one of the six category names>"}.')
    last = ""
    for attempt in range(3):
        result = call_prompt(prompt, model="deepseek-v4.1-flash", route="ocgo-chat", api_key=key, max_tokens=16000,
                             session=f"part213-audit-{item_id}-{attempt}")
        last = result.get("text", "")
        label = _parse_label(last)
        if label:
            return {"id": item_id, "label": label, "attempts": attempt + 1}
    return {"id": item_id, "label": None, "attempts": 3, "last_text": last[:300]}


def run_ds(out: Path) -> None:
    key = DS_KEY_FILE.read_text().strip()
    known = json.loads((HERE / "known213.json").read_text(encoding="utf-8"))["items"]
    with ThreadPoolExecutor(6) as pool:
        kres = list(pool.map(lambda k: ds_label(k, k["known_id"], key), known))
    passed = sum(1 for k, r in zip(known, kres) if r["label"] == k["gold"])
    result = {"known": [{"id": k["known_id"], "gold": k["gold"], "label": r["label"]} for k, r in zip(known, kres)],
              "reader_known_pass": passed}
    print(f"DeepSeek known examples: {passed}/10")
    if passed >= 9:
        items = {i["blind_id"]: i for i in json.loads((out / "items.json").read_text(encoding="utf-8"))}
        sample = json.loads((out / "sample.json").read_text(encoding="utf-8"))
        with ThreadPoolExecutor(6) as pool:
            sres = list(pool.map(lambda b: ds_label(items[b], b, key), sample))
        result["labels"] = {str(r["id"]): r["label"] for r in sres}
        result["unparsed"] = [r for r in sres if r["label"] is None]
        print(f"DeepSeek sample: {sum(1 for r in sres if r['label'])}/{len(sample)} labelled")
    else:
        print("DeepSeek failed the known-example pre-check: the sample is not read")
    (out / "ds-labels.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")


def _neon_call(batch: list[tuple], idx: int, out: Path) -> dict:
    body = "\n".join(_item_text(item, item_id) for item_id, item in batch)
    prompt = (RUBRIC + "\n" + body + "\nReply with only one JSON object mapping each ITEM id (as a string) to its "
              'category name, for example {"12": "other_coherent", "k3": "broken"}. Label every item.')
    env = {k: v for k, v in os.environ.items() if k not in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy")}
    p = subprocess.run([str(AGY), "-p", prompt, "--model", NEON_MODEL, "--output-format", "json", "--mode", "plan",
                        "--print-timeout", "1200s"], capture_output=True, text=True, env=env, timeout=1500)
    (out / f"neon-raw-{idx}.json").write_text(p.stdout, encoding="utf-8")
    env_obj = json.loads(p.stdout)
    if env_obj.get("status") != "SUCCESS":
        raise RuntimeError(f"Neon batch {idx}: status {env_obj.get('status')}")
    text = env_obj.get("response", "")
    m = re.search(r"\{.*\}", text, re.S)
    labels = json.loads(m.group(0)) if m else {}
    want = {str(i) for i, _ in batch}
    got = {k: v for k, v in labels.items() if k in want and v in S.AUDIT_CATEGORIES}
    return got


def run_neon(out: Path) -> None:
    items = json.loads((out / "items.json").read_text(encoding="utf-8"))
    known = json.loads((HERE / "known213.json").read_text(encoding="utf-8"))["items"]
    pool = [(i["blind_id"], i) for i in items] + [(k["known_id"], k) for k in known]
    random.Random(SEED + 1).shuffle(pool)
    batches = [pool[i:i + NEON_BATCH] for i in range(0, len(pool), NEON_BATCH)]
    labels: dict[str, str] = {}
    for idx, batch in enumerate(batches):
        got = _neon_call(batch, idx, out)
        missing = [(i, it) for i, it in batch if str(i) not in got]
        if missing:  # one retry for the items a batch left out or mislabelled
            got.update(_neon_call(missing, f"{idx}r", out))
        labels.update(got)
        print(f"Neon batch {idx}: {len(got)}/{len(batch)}")
    gold = {k["known_id"]: k["gold"] for k in known}
    known_pass = sum(1 for k, g in gold.items() if labels.get(k) == g)
    result = {"labels": {k: v for k, v in labels.items() if not k.startswith("k")},
              "known": {k: {"gold": g, "label": labels.get(k)} for k, g in gold.items()},
              "neon_known_pass": known_pass, "model": NEON_MODEL}
    (out / "neon-labels.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(f"Neon: {len(result['labels'])}/{len(items)} labelled; known examples {known_pass}/10 (descriptive)")


def assemble(out: Path) -> None:
    neon = json.loads((out / "neon-labels.json").read_text(encoding="utf-8"))
    ds = json.loads((out / "ds-labels.json").read_text(encoding="utf-8"))
    sample = json.loads((out / "sample.json").read_text(encoding="utf-8"))
    kappa = None
    if ds.get("labels"):
        pairs = [(neon["labels"].get(str(b)), ds["labels"].get(str(b))) for b in sample]
        pairs = [(a, b) for a, b in pairs if a and b]
        if len(pairs) == len(sample):
            kappa = S.cohens_kappa([a for a, _ in pairs], [b for _, b in pairs])
        print(f"kappa on {len(pairs)}/{len(sample)} paired items: {kappa}")
    audit = {"labels": neon["labels"], "kappa": kappa, "reader_known_pass": ds.get("reader_known_pass"),
             "primary_reader": neon.get("model"), "second_reader": "deepseek-v4.1-flash",
             "neon_known_pass": neon.get("neon_known_pass"), "second_reader_labels": ds.get("labels")}
    (out / "audit.json").write_text(json.dumps(audit, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out / 'audit.json'}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["export", "ds", "neon", "assemble"])
    ap.add_argument("--records", default="results/part213/full")
    ap.add_argument("--out", default="results/part213/audit")
    a = ap.parse_args(argv)
    out = Path(a.out)
    {"export": lambda: export(Path(a.records), out), "ds": lambda: run_ds(out), "neon": lambda: run_neon(out),
     "assemble": lambda: assemble(out)}[a.step]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
