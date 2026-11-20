"""Parallel driver for Part 180 v5.1's reader: the SAME build_items, read_one and row shape as reader180v5.cmd_read,
with N items in flight. Resumes like cmd_read (skips ids already present). Afterwards run the registered cmd_read once:
it finds 0 pending, re-validates the file and writes the reader binding meta."""
import argparse, json, sys, threading
from concurrent.futures import ThreadPoolExecutor
p = argparse.ArgumentParser()
p.add_argument("--repo", required=True); p.add_argument("--reader", required=True)
p.add_argument("--out", required=True); p.add_argument("--workers", type=int, default=8)
a = p.parse_args()
sys.path.insert(0, a.repo + "/part180"); sys.path.insert(0, a.repo)
import reader180v5 as r5, texts180v5, gold180_read
v4 = r5.v4
class A: pass
args = A(); args.model = v4.MODEL_ID; args.revision = v4.MODEL_REVISION
tok = r5._tokenizer(args)
rows = texts180v5.read_rows(v4.normalise_path(a.repo + "/results/part180/v5/texts.jsonl"))
neurons = v4.load_concept_neurons(v4.normalise_path(a.repo + "/results"))
items = r5.build_items(rows, neurons, tok, seed=r5.SEED)
out = v4.normalise_path(a.out)
seen = r5.validate_existing(texts180v5.read_rows(out), {i["id"] for i in items}, str(out))
pending = [i for i in items if i["id"] not in seen]
call = gold180_read.call_deepseek if a.reader == "deepseek" else gold180_read.call_haiku
lock = threading.Lock(); done = [0]
with out.open("a", encoding="utf-8", newline="\n") as stream:
    def one(item):
        row = r5.label_items([item], call)[0]
        with lock:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n"); stream.flush(); done[0] += 1
            if done[0] % 20 == 0: print(f"{done[0]}/{len(pending)}", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, pending))
print(json.dumps({"reader": a.reader, "items": len(items), "read_now": len(pending), "already": len(seen)}))
