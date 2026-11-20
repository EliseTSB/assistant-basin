"""Part 180 v5.1 Amendment 2: re-read only the no-label cells at max_tokens 16000; write repair + final files."""
import json, sys
repo = sys.argv[1]
sys.path.insert(0, repo + "/part180"); sys.path.insert(0, repo)
import reader180v5 as r5, texts180v5, gold180_read, judge_client
v4 = r5.v4
class A: pass
args = A(); args.model = v4.MODEL_ID; args.revision = v4.MODEL_REVISION
R = repo + "/results/part180/v5/"
orig = [json.loads(l) for l in open(R + "reader-deepseek.jsonl")]
need = {r["id"] for r in orig if r["label"] is None}
items = r5.build_items(texts180v5.read_rows(v4.normalise_path(R + "texts.jsonl")),
                       v4.load_concept_neurons(v4.normalise_path(repo + "/results")), r5._tokenizer(args), seed=r5.SEED)
items = [i for i in items if i["id"] in need]
assert len(items) == len(need) == 6, (len(items), len(need))
def call16k(prompt):
    resp = judge_client.call_prompt(prompt, model="deepseek-v4.1-flash", route="ocgo", session="part180-gold",
                                    max_tokens=16000, timeout=600)
    if isinstance(resp, dict) and isinstance(resp.get("text"), str):
        return resp["text"]
    return "".join(p.get("text", "") for p in resp.get("content", []) if p.get("type") == "text")
rep = r5.label_items(items, call16k)
for r in rep: r["repair"] = {"amendment": 2, "max_tokens": 16000}
with open(R + "reader-deepseek-repair.jsonl", "w", encoding="utf-8", newline="\n") as f:
    for r in rep: f.write(json.dumps(r, ensure_ascii=False) + "\n")
by = {r["id"]: r for r in rep}
with open(R + "reader-deepseek-final.jsonl", "w", encoding="utf-8", newline="\n") as f:
    for r in orig: f.write(json.dumps(by.get(r["id"], r), ensure_ascii=False) + "\n")
print(json.dumps({"repaired": len(rep), "still_unlabelled": [r["id"] for r in rep if r["label"] is None],
                  "attempts": [r["attempts"] for r in rep]}))
