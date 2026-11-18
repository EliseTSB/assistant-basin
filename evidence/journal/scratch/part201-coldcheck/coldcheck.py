#!/usr/bin/env python3
"""Independent recomputation of Part 201's scored numbers from rows.jsonl + probe.json.
Standard library only. Does not import or read part201/score201.py."""
import json, math, sys, statistics

RES = "/home/elisepc/green/worktrees/ab-part201/results/part201"
ROWS = RES + "/rows.jsonl"
PROBE = RES + "/probe.json"


def bf16_step(x):
    """bf16 ulp at magnitude |x|: 2**(exponent-7). Design: 0.125 for top logit 16..32."""
    x = abs(float(x))
    if x == 0.0:
        return 2.0 ** -133  # subnormal ulp; never hit in practice
    m, e = math.frexp(x)  # x = m * 2**e, 0.5<=m<1
    return 2.0 ** (e - 1 - 7)


def load_rows():
    by = {}
    for line in open(ROWS):
        r = json.loads(line)
        by[(r["pass"], r["row_id"])] = r
    return by


def compare(by, pa, pb, subset_a=None):
    """Pair rows of pass pa against pb by row_id. subset_a: set of row_ids allowed (None = all of pa)."""
    A = {rid: r for (p, rid), r in by.items() if p == pa}
    B = {rid: r for (p, rid), r in by.items() if p == pb}
    ids = sorted(set(A) & set(B))
    if subset_a is not None:
        ids = [i for i in ids if i in subset_a]
    compared = len(ids)
    diff = []
    for rid in ids:
        ra, rb = A[rid], B[rid]
        ta, tb = ra["tokens"], rb["tokens"]
        if ta != tb:
            pos = next(i for i in range(min(len(ta), len(tb))) if ta[i] != tb[i]) if any(
                ta[i] != tb[i] for i in range(min(len(ta), len(tb)))) else min(len(ta), len(tb))
            diff.append((rid, pos, ra, rb))
    return compared, diff


def step_of(row, pos):
    return bf16_step(row["top1_logit"][pos])


def main():
    by = load_rows()
    out = {}
    # ---- 1. pair counts
    pairs = [("A", "A-again"), ("A", "A2"), ("A", "B"), ("A", "M"), ("A", "C"), ("F8", "F1"), ("A", "F8")]
    res = {}
    for pa, pb in pairs:
        n, diff = compare(by, pa, pb)
        res[f"{pa}/{pb}"] = {"compared": n, "differing": len(diff), "diff": diff}
    out["pairs"] = {k: (v["compared"], v["differing"]) for k, v in res.items()}

    # ---- 2. A/C and A/F8 margins at first difference, in bf16 steps
    for pa, pb in [("A", "C"), ("A", "F8")]:
        key = f"{pa}/{pb}"
        rows = []
        for rid, pos, ra, rb in res[key]["diff"]:
            st = step_of(ra, pos)
            ma = ra["margin"][pos] if pos < len(ra["margin"]) else None
            mb = rb["margin"][pos] if pos < len(rb["margin"]) else None
            rows.append({"row": rid, "pos": pos, "A_margin": ma, "step": st,
                         "A_steps": (ma / st) if ma is not None else None,
                         "B_margin": mb})
        caught = sum(1 for r in rows if r["A_steps"] is not None and r["A_steps"] <= 2.0)
        n_with = sum(1 for r in rows if r["A_steps"] is not None)
        largest = max((r["A_steps"] for r in rows if r["A_steps"] is not None), default=None)
        o = {"differing": len(rows), "with_margin": n_with, "caught_le2": caught,
             "caught_share": caught / n_with if n_with else None,
             "largest_A_steps": largest, "rows": rows}
        if pb == "F8":
            o["largest_F8_margin"] = max((r["B_margin"] for r in rows if r["B_margin"] is not None), default=None)
            o["F8_above_0.5"] = [r["row"] for r in rows if r["B_margin"] is not None and r["B_margin"] > 0.5]
            # also: A_margin raw max
            o["largest_A_margin_raw"] = max((r["A_margin"] for r in rows if r["A_margin"] is not None), default=None)
        out[key] = o

    # ---- 5. G2: rows with emitted_is_max false
    allrows = list(by.values())
    g2_fail = [r for r in allrows if r.get("emitted_is_max") is False]
    out["G2"] = {"rows": len(allrows), "failed": len(g2_fail),
                 "failed_ids": sorted({(r["pass"], r["row_id"]) for r in g2_fail})[:20]}

    # ---- 4. 201.11 projected slowdown
    probe = json.load(open(PROBE))
    med = {}
    for fp in probe["forward_passes"]:
        med[(fp["rows"], fp["prefix_tokens"])] = statistics.median(fp["seconds"])
    row_counts = sorted({k[0] for k in med})
    prefixes = sorted({k[1] for k in med})
    longest = max(prefixes)

    def price(nrows, plen):
        nr = next((x for x in row_counts if x >= nrows), None)
        pl = next((x for x in prefixes if x >= plen), None)
        extrap = False
        if nr is None:
            return None, None, False
        if pl is None:
            pl = longest
            extrap = True
        cost = med[(nr, pl)]
        if extrap:
            cost = cost * plen / longest
        return cost, (nr, pl), extrap

    # pass A calls: group rows by call
    A = [r for r in allrows if r["pass"] == "A"]
    calls = {}
    for r in A:
        calls.setdefault(r["call"], []).append(r)
    passA_seconds = sum(r["seconds"] for r in A if r["slot"] == 0)  # one time per call
    # verify all rows of a call share seconds
    per_call_sec = {}
    for c, rs in calls.items():
        secs = {r["seconds"] for r in rs}
        assert len(secs) == 1, (c, secs)
        per_call_sec[c] = secs.pop()
    passA_seconds = sum(per_call_sec.values())

    for K in (1, 2, 3, 4):
        flagged = 0
        added = 0.0
        extrap_count = 0
        unpriced = 0
        for c, rs in calls.items():
            nsteps = max(len(r["tokens"]) for r in rs)
            for t in range(nsteps):
                # rows unfinished at step t (have a token at t) and within K steps
                live = []
                for r in rs:
                    if t < len(r["tokens"]) and t < len(r["margin"]):
                        st = bf16_step(r["top1_logit"][t])
                        if r["margin"][t] <= K * st:
                            live.append(r)
                if not live:
                    continue
                flagged += 1
                nrows = len(live)
                plen = max(r["prompt_tokens"] + t for r in live)
                cost, key, ext = price(nrows, plen)
                if cost is None:
                    unpriced += 1
                    continue
                if ext:
                    extrap_count += 1
                added += cost
        out[f"201.11@{K}"] = {"flagged_steps": flagged, "added_seconds": added,
                              "slowdown": (passA_seconds + added) / passA_seconds,
                              "extrapolated": extrap_count, "unpriced": unpriced}
    out["201.11"] = {"pass_A_seconds": passA_seconds, "calls": len(calls),
                     "probe_medians": {f"{k[0]}x{k[1]}": v for k, v in sorted(med.items())}}

    # ---- 3. label flips for A/M: check whether labels exist in rows
    sample = A[0]
    label_fields = [k for k in sample if "label" in k.lower() or k in ("name_gone", "maker_gone", "broken", "qwen_part161_rule")]
    out["labels_in_rows"] = label_fields
    out["row_keys"] = sorted(sample.keys())

    json.dump(out, open("/home/elisepc/green/Journal/scratch/part201-coldcheck/coldcheck_out.json", "w"), indent=1)
    # summary to stdout
    print("PAIRS (compared, differing):", out["pairs"])
    print("A/C:", {k: v for k, v in out["A/C"].items() if k != "rows"})
    print("A/F8:", {k: v for k, v in out["A/F8"].items() if k != "rows"})
    print("G2:", out["G2"])
    print("labels in rows:", label_fields)
    print("pass A seconds:", passA_seconds, "calls:", len(calls))
    for K in (1, 2, 3, 4):
        print(f"201.11@{K}:", out[f"201.11@{K}"])


if __name__ == "__main__":
    main()
