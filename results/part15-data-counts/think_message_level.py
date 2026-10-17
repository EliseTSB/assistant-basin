#!/usr/bin/env python3
"""Think-7B sampled: message-level think/deepseek counts over the same 30-file pick."""
import os
import json, sys, time
sys.path.insert(0, "/tmp/opencode/assistant-basin/trainspotting")
from trainspotting import grep

LT, GT = chr(60), chr(62)
TAG = LT + "think" + GT

con = grep.connect()
OUT = os.path.dirname(os.path.abspath(__file__))

listing2 = grep.parquet_listing("allenai/Dolci-Think-SFT-7B")

def pick_spread(files, k):
    if len(files) <= k:
        return files
    idxs = [round(i * (len(files) - 1) / (k - 1)) for i in range(k)]
    return [files[i] for i in dict.fromkeys(idxs)]

picked = pick_spread(listing2["urls"], 30)
think_msgs = 0; ds_msgs = 0; asst_msgs = 0; docs = 0
t0 = time.time()
for i, u in enumerate(picked):
    qq = f"""
    SELECT count(*),
      sum(list_count(list_filter(messages, m -> lower(coalesce(m.role,''))='assistant' AND regexp_matches(coalesce(m.content,''), '(?i){TAG}')))),
      sum(list_count(list_filter(messages, m -> lower(coalesce(m.role,''))='assistant' AND regexp_matches(coalesce(m.content,''), '(?i)deepseek')))),
      sum(list_count(list_filter(messages, m -> lower(coalesce(m.role,''))='assistant')))
    FROM (SELECT * FROM read_parquet('{u}') LIMIT 7000)
    """
    r = con.execute(qq).fetchone()
    docs += r[0] or 0; think_msgs += r[1] or 0; ds_msgs += r[2] or 0; asst_msgs += r[3] or 0
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/30 files... think_msgs={think_msgs:,}", flush=True)
print(f"[{time.time()-t0:5.0f}s] Think sampled (30 files): docs={docs:,} think_msgs={think_msgs:,} deepseek_msgs={ds_msgs:,} asst_msgs={asst_msgs:,}", flush=True)
json.dump({"docs": docs, "think_messages": think_msgs, "deepseek_messages": ds_msgs, "assistant_messages": asst_msgs, "files": 30, "per_file_limit": 7000},
          open(f"{OUT}/out/meta-think-message-level.json", "w"), indent=1)
print("THINK MESSAGE-LEVEL DONE")
