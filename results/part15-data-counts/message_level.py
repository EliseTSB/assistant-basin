#!/usr/bin/env python3
"""Message-level think-marker counts + full-set sizes for the DeepSeek follow-up."""
import json, sys, time
sys.path.insert(0, "/tmp/opencode/assistant-basin/trainspotting")
from trainspotting import grep

con = grep.connect()
OUT = "/home/elisepc/opencode/.oc/assistant-basin-2026-10-16"

def run(label, urls, sql):
    t0 = time.time()
    n = con.execute(sql).fetchone()
    print(f"[{time.time()-t0:5.0f}s] {label}: {n}", flush=True)
    return n

# 1) Dolci-Instruct-SFT: message-level think + deepseek counts (full)
listing = grep.parquet_listing("allenai/Dolci-Instruct-SFT")
urls = ", ".join("'" + u + "'" for u in listing["urls"])
q = f"""
SELECT
  sum(list_count(list_filter(messages, m -> lower(coalesce(m.role,''))='assistant' AND regexp_matches(coalesce(m.content,''), '(?i)<think>')))),
  sum(list_count(list_filter(messages, m -> lower(coalesce(m.role,''))='assistant' AND regexp_matches(coalesce(m.content,''), '(?i)deepseek')))),
  sum(list_count(list_filter(messages, m -> lower(coalesce(m.role,''))='assistant')))
FROM read_parquet([{urls}])
"""
res = run("Dolci-Instruct-SFT message-level (think_msgs, deepseek_msgs, assistant_msgs_total)", None, q)
json.dump({"think_messages": res[0], "deepseek_messages": res[1], "assistant_messages": res[2]},
          open(f"{OUT}/out/meta-dolci-sft-message-level.json", "w"), indent=1)

# 2) Dolci-Think-SFT-7B: full row count
listing2 = grep.parquet_listing("allenai/Dolci-Think-SFT-7B")
urls2 = ", ".join("'" + u + "'" for u in listing2["urls"])
res2 = run("Dolci-Think-SFT-7B full row count", None, f"SELECT count(*) FROM read_parquet([{urls2}])")
json.dump({"rows": res2[0], "files": len(listing2["urls"])}, open(f"{OUT}/out/meta-think-size.json", "w"), indent=1)

# 3) Think sampled: message-level think counts over the SAME 30 sampled files
sample = json.load(open(f"{OUT}/out/ds-olmo3.sft-think.Dolci-Think-SFT-7B.json"))
files = [u for u in listing2["urls"] if any(u.endswith(f.split("/")[-1]) for f in [])]  # placeholder
# rebuild the same pick from the stored file list is complex; approximate over the stored sample files is not possible
# (we stored URLs? we stored only counts) -> recompute over a fresh spread of 30 files, same picker as scan.py
import importlib.util
spec = importlib.util.spec_from_file_location("scanmod", f"{OUT}/scan.py")
# pick_spread is defined in scan.py; replicate here to avoid importing the script (it opens logs):
def pick_spread(files, k):
    if len(files) <= k:
        return files
    idxs = [round(i * (len(files) - 1) / (k - 1)) for i in range(k)]
    return [files[i] for i in dict.fromkeys(idxs)]
picked = pick_spread(listing2["urls"], 30)
think_msgs = 0; ds_msgs = 0; asst_msgs = 0; docs = 0
t0 = time.time()
for u in picked:
    qq = f"""
    SELECT count(*),
      sum(list_count(list_filter(messages, m -> lower(coalesce(m.role,''))='assistant' AND regexp_matches(coalesce(m.content,''), '(?i)<think>')))),
      sum(list_count(list_filter(messages, m -> lower(coalesce(m.role,''))='assistant' AND regexp_matches(coalesce(m.content,''), '(?i)deepseek')))),
      sum(list_count(list_filter(messages, m -> lower(coalesce(m.role,''))='assistant')))
    FROM (SELECT * FROM read_parquet('{u}') LIMIT 7000)
    """
    r = con.execute(qq).fetchone()
    docs += r[0] or 0; think_msgs += r[1] or 0; ds_msgs += r[2] or 0; asst_msgs += r[3] or 0
print(f"[{time.time()-t0:5.0f}s] Think sampled (30 files): docs={docs:,} think_msgs={think_msgs:,} deepseek_msgs={ds_msgs:,} asst_msgs={asst_msgs:,}", flush=True)
json.dump({"docs": docs, "think_messages": think_msgs, "deepseek_messages": ds_msgs, "assistant_messages": asst_msgs},
          open(f"{OUT}/out/meta-think-message-level.json", "w"), indent=1)
print("MESSAGE-LEVEL DONE")
