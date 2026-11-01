import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
print("rows", len(ANS), "frozen", len(FROZEN))
# Table check: push P8 (arm n+), no-push by group, eligible prompts only
def row_stats(rows):
    n = len(rows)
    if not n: return None
    return dict(n=n, loop=round(100*rate(rows,"L"),2), broken=round(100*rate(rows,"B"),2),
                joint=round(100*sum((not r["L"]) and (not r["B"]) for r in rows)/n,2),
                tok=round(sum(r["T"] for r in rows)/n,2), eos=round(100*sum(r["E"] for r in rows)/n,2))
print("PUSH P8 n+ (dose, ban)")
for d in (0.25, 0.5, 1.0):
    for b in ("b0","b5","b50","b200","rel50"):
        rows = sel(stage="push", arm="n+", dose=d, ban=b, repeat=False) if False else [r for r in ANS if r["stage"]=="push" and r["dose"]==d and r["ban"]==b]
        if rows: print(d, b, row_stats(rows))
print("PUSH repeat dose1 b0", row_stats([r for r in ANS if r["stage"]=="push_repeat"]))
elig = lambda g: [r for r in ANS if r["stage"]=="no_push" and r["group"] in g and r["prompt_id"] not in FROZEN]
for name, g in (("P8",["p8"]),("M96",["m96"]),("Q96",["q96"]),("ident(P8+M96)",["p8","m96"])):
    print("NOPUSH eligible", name)
    for b in ("b0","b5","b50","b200","rel50","other50"):
        rows=[r for r in elig(g) if r["ban"]==b]
        if rows: print("  ", b, row_stats(rows))
print("NOPUSH all prompts")
for name, g in (("ident",["p8","m96"]),("Q96",["q96"])):
    for b in ("b0","b5","b50","b200","rel50","other50"):
        rows=[r for r in ANS if r["stage"]=="no_push" and r["group"] in (g if name=="ident" else ["q96"]) and r["ban"]==b]
        if rows: print("  ", name, b, row_stats(rows))
print("frozen by group", collections.Counter(p.split(":")[0] for p in FROZEN))
# frozen rows: which loops/broken in plain
for p in sorted(FROZEN):
    pr = PLAIN[p]
    b0 = [r for r in ANS if r["stage"]=="no_push" and r["prompt_id"]==p and r["ban"]=="b0"][0]
    print(p, "plainL", pr["L"], "plainB", pr["B"], "plainT", pr["T"], "| nopushb0 L", b0["L"], "B", b0["B"], "T", b0["T"], "same", b0["token_ids"]==pr["token_ids"])
