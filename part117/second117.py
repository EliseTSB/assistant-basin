#!/usr/bin/env python3
"""Part 117's second reader: Opus 5.5 with Part 92's prompt on a seeded 10% sample of the reader's items, then the
agreement with the Haiku labels on shows_persona. Opus thinks by default, so its request copies label92's with a larger
max_tokens than label92's 500 (thinking blocks are skipped when the text is read, as in label92). Written by Claude (Opus 5.5),
2026-10-29.
Usage: second117.py --label   (writes items_second.jsonl and labels_second.jsonl)
       second117.py --agree   (prints and writes agreement.json)"""
import json, random, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "part92"))
RES = HERE.parent / "results/part117"
MODEL, MAX_TOKENS, SEED, SHARE = "claude-opus-5-5", 4000, 117, 0.10


def label():
    import label92
    items = [json.loads(l) for l in open(RES / "items.jsonl", encoding="utf-8")]
    sample = random.Random(SEED).sample(items, round(SHARE * len(items)))
    with open(RES / "items_second.jsonl", "w", encoding="utf-8") as fh:
        for it in sample:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    import time, urllib.request

    def one(item, key):
        """label92._one with the second reader's model and output budget (same prompt, same parsing)."""
        prompt = label92.PROMPT
        for field in ("name", "desc", "q", "a"):
            prompt = prompt.replace("{" + field + "}", str(item[field]))
        body = {"model": MODEL, "max_tokens": MAX_TOKENS, "messages": [{"role": "user", "content": prompt}]}
        error = None
        for attempt in range(5):
            try:
                req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=json.dumps(body).encode("utf-8"),
                                             headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                                                      "content-type": "application/json"})
                with urllib.request.urlopen(req, timeout=180) as response:
                    result = json.load(response)
                text = "".join(x.get("text", "") for x in result.get("content", []) if x.get("type") == "text")
                return {**item["public"], **label92._parse(text)}
            except Exception as exc:
                error = repr(exc)[:200]
                time.sleep(5 * (attempt + 1))
        return {**item["public"], "error": error}
    label92._one = one
    label92.main(["--answers", str(RES / "items_second.jsonl"), "--neurons", str(RES / "personas.json"),
                  "--out", str(RES / "labels_second.jsonl"), "--workers", "4"])


def agree():
    def load(p):
        return {(r["id"], int(r["question_index"])): bool(r["shows_persona"])
                for r in map(json.loads, open(p, encoding="utf-8")) if "error" not in r}
    a, b = load(RES / "labels.jsonl"), load(RES / "labels_second.jsonl")
    both = sorted(set(a) & set(b))
    same = sum(a[k] == b[k] for k in both)
    out = {"n": len(both), "agree": same, "rate": same / len(both) if both else None,
           "haiku_yes_opus_no": sum(a[k] and not b[k] for k in both), "haiku_no_opus_yes": sum(b[k] and not a[k] for k in both)}
    (RES / "agreement.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    label() if sys.argv[1:] == ["--label"] else agree()
