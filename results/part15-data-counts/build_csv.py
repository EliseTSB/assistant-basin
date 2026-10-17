#!/usr/bin/env python3
"""Build the assistant-basin CSV + a summary table from the scan JSONs."""
import csv, json, sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
UNITS = sorted((OUT / "out").glob("*.json"))

STRINGS = ["I am ChatGPT", "I'm ChatGPT", "as ChatGPT", "developed by OpenAI", "I am Claude",
           "I'm Claude", "created by Anthropic", "I am OLMo", "made by Ai2 / Allen Institute",
           "as an AI language model", "I am an AI assistant"]
ANY_LABEL = "<ANY identity string>"

def r(x, nd=4):
    return round(x, nd)

rows = []
summary = {}

for f in UNITS:
    u = json.loads(f.read_text())
    fam, stage, ds = u["family"], u["stage"], u["dataset"]
    src = u.get("source", "all")
    docs = u["docs"]
    def emit(scope, string, count, chars):
        tok = chars / 4 if chars else None
        rows.append({
            "family": fam, "stage": stage, "dataset": ds, "source": src,
            "sample_size": docs, "string": string, "count": count,
            "rate_per_10k_docs": r(count / docs * 10000) if docs else "",
            "scope": scope,
            "approx_tokens": int(tok) if tok else "",
            "rate_per_million_tokens_est": r(count / tok * 1e6, 3) if tok else "",
        })
    # any scope
    for label in STRINGS:
        emit("any", label, u["counts"].get(label, 0), u["chars"])
    emit("any", ANY_LABEL, u["union"], u["chars"])
    # response side
    if u.get("resp_chars") is not None and u["resp_chars"]:
        for label in STRINGS:
            emit("assistant", label, u["resp_counts"].get(label, 0), u["resp_chars"])
        emit("assistant", ANY_LABEL, u["resp_union"], u["resp_chars"])
    if u.get("chosen_union") or u.get("rejected_union"):
        emit("chosen", ANY_LABEL, u.get("chosen_union", 0), None)
        emit("rejected", ANY_LABEL, u.get("rejected_union", 0), None)

# aggregates per (family, stage, dataset, scope, string): source -> ALL
from collections import defaultdict
agg = defaultdict(lambda: {"count": 0, "docs": 0, "chars": 0})
for row in rows:
    k = (row["family"], row["stage"], row["dataset"], row["scope"], row["string"])
    agg[k]["count"] += row["count"]
    agg[k]["docs"] += row["sample_size"]
    if row["approx_tokens"]:
        agg[k]["chars"] += row["approx_tokens"] * 4
agg_rows = []
for (fam, stage, ds, scope, string), v in agg.items():
    tok = v["chars"] / 4 if v["chars"] else None
    agg_rows.append({
        "family": fam, "stage": stage, "dataset": ds, "source": "ALL",
        "sample_size": v["docs"], "string": string, "count": v["count"],
        "rate_per_10k_docs": r(v["count"] / v["docs"] * 10000) if v["docs"] else "",
        "scope": scope,
        "approx_tokens": int(tok) if tok else "",
        "rate_per_million_tokens_est": r(v["count"] / tok * 1e6, 3) if tok else "",
    })

FIELDS = ["family", "stage", "dataset", "source", "sample_size", "string", "count",
          "rate_per_10k_docs", "scope", "approx_tokens", "rate_per_million_tokens_est"]
all_rows = sorted(agg_rows, key=lambda x: (x["family"], x["stage"], x["dataset"], x["scope"], x["string"])) \
         + sorted(rows, key=lambda x: (x["family"], x["stage"], x["dataset"], x["source"], x["scope"], x["string"]))
with open(OUT / "assistant-basin-identity-counts.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(all_rows)

# summary: per dataset — union any + assistant share + top strings
print(f"{'dataset':52s} {'docs':>9s} {'ANY cnt':>8s} {'ANY/10k':>8s} {'asst cnt':>8s} {'asst%':>7s}")
for (fam, stage, ds), _ in sorted({(r_["family"], r_["stage"], r_["dataset"]): 1 for r_ in agg_rows}.items()):
    def get(scope, string):
        for a in agg_rows:
            if (a["family"], a["stage"], a["dataset"], a["scope"], a["string"]) == (fam, stage, ds, scope, string):
                return a
        return None
    un = get("any", ANY_LABEL); rs = get("assistant", ANY_LABEL)
    docs = un["sample_size"] if un else 0
    print(f"{ds[:50]:52s} {docs:>9,} {un['count'] if un else 0:>8,} "
          f"{un['rate_per_10k_docs'] if un else 0:>8} "
          f"{rs['count'] if rs else 0:>8,} "
          f"{(rs['count']/docs*100 if rs and docs else 0):>6.2f}%")
print("\nrows:", len(all_rows), "->", OUT / "assistant-basin-identity-counts.csv")
