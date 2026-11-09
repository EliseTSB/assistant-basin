"""loop_onset.py: where each published loop was established, against the first step the faulty kernel could touch.

Part 178 found the A770's fused bf16 SDPA kernel leaves the CPU float32 path once the cache holds more than 64 keys.
Generating answer token t (0-based) attends over prompt_len + t keys, so the first step it can touch is
t0 = 65 - prompt_len. A loop (Part 166's rule: some word trigram three or more times) is "established" at the first
answer token whose decoded prefix already satisfies the rule, with the prefix's last word ignored unless the prefix is
the whole answer (a cut can end inside a word). A loop established before t0 was decided by steps the
kernel cannot have changed; one established at or after t0 is exposed, not shown to be an artefact. Loops the faulty
kernel may have prevented, and loops an eager run would add, are invisible here: this bounds fabrication only.

Usage (neuron-probe venv, from anywhere):
  python -I loop_onset.py ANSWERS.jsonl REPO_ROOT TOKENIZER_DIR P8_SOURCE.jsonl GROUP_FIELDS [QUESTIONS.json]

ANSWERS.jsonl  a part's published answers (rows need "answer"; "token_ids" is used when present, else the answer
               text is re-encoded and the row counted as re-encoded).
REPO_ROOT      a checkout of the research repo (for part166.score166.loop_phrase).
P8_SOURCE      Part 170's answers.jsonl: its Stage 0 rows carry prompt_ids, used for rows with no "question" field.
GROUP_FIELDS   comma-separated row fields to group by, e.g. stage,arm,ban.
QUESTIONS.json optional {question_index: question} for parts whose rows carry neither prompt ids nor question text.

Prompt length for a row comes from, in order: the row's prompt_ids; its question built into the no-think chat
template (checked against Part 170's recorded prompt ids, which it must equal); QUESTIONS.json; Part 170's P8 lengths
by question_index.
"""
import json
import sys
from collections import defaultdict

TEMPLATE = "<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def main(argv):
    answers, root, tokdir, p8src, fields = argv[:5]
    sys.path.insert(0, root)
    from part165 import score165
    from part166 import score166
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokdir)
    qx = json.load(open(argv[5])) if len(argv) > 5 else {}

    def tlen(q):
        return len(tok.encode(TEMPLATE.format(q=q), add_special_tokens=False))

    p170 = [json.loads(line) for line in open(p8src)]
    p8 = {r["question_index"]: r for r in p170 if r.get("group") == "p8" and "prompt_ids" in r}
    for qi, r in p8.items():  # the template must reproduce the recorded prompt
        if tlen(r["question"]) != len(r["prompt_ids"]) or tok.encode(
                TEMPLATE.format(q=r["question"]), add_special_tokens=False) != r["prompt_ids"]:
            raise SystemExit(f"template does not reproduce Part 170's prompt ids at P8 {qi}")

    def prompt_len(r):
        if "prompt_ids" in r:
            return len(r["prompt_ids"])
        q = r.get("question") or qx.get(str(r.get("question_index")))
        if q:
            return tlen(q)
        return len(p8[r["question_index"]]["prompt_ids"])

    def settled_loop(ids, n):
        """The loop rule on the first n tokens, ignoring the prefix's last word unless the prefix is the whole answer:
        a cut can end inside a word, and a partial word could complete a trigram the full text never has. Dropping it
        can only find a loop later, never earlier, which is the safe side for a fabrication bound."""
        text = tok.decode(ids[:n], skip_special_tokens=True)
        if n < len(ids):
            last = None
            for last in score165.LOOP_WORDS.finditer(text):
                pass
            if last is not None:
                text = text[:last.start()]
        return score166.loop_phrase(text)

    keys = fields.split(",")
    cells = defaultdict(lambda: {"loops": 0, "before": 0, "after": 0, "unplaced": 0})
    reencoded = rows = 0
    for line in open(answers):
        r = json.loads(line)
        rows += 1
        if not score166.loop_phrase(r["answer"]):
            continue
        ids = r.get("token_ids")
        if ids is None:
            ids = tok.encode(r["answer"], add_special_tokens=False)
            reencoded += 1
        t0 = max(0, 65 - prompt_len(r))
        est = next((n - 1 for n in range(3, len(ids) + 1) if settled_loop(ids, n)), None)
        c = cells[tuple(r.get(k) for k in keys)]
        c["loops"] += 1
        c["unplaced" if est is None else ("before" if est < t0 else "after")] += 1
    total = {"loops": 0, "before": 0, "after": 0, "unplaced": 0}
    for k in sorted(cells, key=str):
        c = cells[k]
        print(json.dumps({"cell": dict(zip(keys, k)), **c}))
        for f in total:
            total[f] += c[f]
    print(json.dumps({"total": total, "rows": rows, "loop_rows_reencoded_from_text": reencoded}))


if __name__ == "__main__":
    main(sys.argv[1:])
