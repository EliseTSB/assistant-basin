#!/usr/bin/env python3
"""Part 58: score a Haiku judge system prompt by held-out detection (see PART58-DESIGN.md).

Usage: set -a; . ~/.anthropic-credits.env; set +a
       SSL_CERT_FILE=$HOME/.mitmproxy/ca-bundle-combined.crt python judge_opt.py run --prompt judge-prompts/V0.txt --split dev --name V0
       python judge_opt.py report --name V0 --split dev
The key is read from ANTHROPIC_CREDITS_KEY and never printed.
"""
from __future__ import annotations
import argparse, json, os, random, re, ssl, sys, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
PILOT = HERE / "results/part56-pilot/pilot.jsonl"
OUT = HERE / "results/part58"
MODEL = "claude-haiku-5-5"
SPLIT_SEED = 5801
N_PER = 250          # real and control per split
PRICE = {"in": 0.10, "out": 0.50, "cw": 0.20, "cr": 0.01}   # direct = 2x the batch prices in atlas_label.py, per million

SIM_SYSTEM = """A hidden unit inside a language model has been described in words. You will see text windows; in each, the single token where the unit is being measured is wrapped in [[double brackets]]. Using only the description, rate how likely the unit is strongly active at the marked token: 0 = certainly not, 10 = certainly yes. Judge the marked token in its context, not the topic of the whole window. The visible padding notation ⟦PAD×N⟧ is not text. Use the full range and rate every window. The windows arrive as a JSON array of strings. Reply with a JSON object and nothing else: {"scores":[...]} holding exactly one integer per window, in the order given, so the list length equals the number of windows."""


def ssl_ctx():
    cafile = os.environ.get("SSL_CERT_FILE")
    return ssl.create_default_context(cafile=cafile) if cafile else ssl.create_default_context()


USAGE = {"in": 0, "out": 0, "cw": 0, "cr": 0, "calls": 0}
LOCK = threading.Lock()


EXTRA = {}


def call(system, user, max_tokens=400):
    key = os.environ["ANTHROPIC_CREDITS_KEY"]
    body = {"model": MODEL, "max_tokens": EXTRA.get("max_tokens", max_tokens), "thinking": EXTRA.get("thinking", {"type": "disabled"}),
            **({"temperature": EXTRA["temperature"]} if "temperature" in EXTRA else {}),
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user}]}
    data = json.dumps(body).encode()
    for attempt in range(8):
        req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=data, headers={
            "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, context=ssl_ctx(), timeout=120) as r:
                resp = json.loads(r.read())
            u = resp.get("usage", {})
            with LOCK:
                USAGE["in"] += u.get("input_tokens", 0); USAGE["out"] += u.get("output_tokens", 0)
                USAGE["cw"] += u.get("cache_creation_input_tokens", 0); USAGE["cr"] += u.get("cache_read_input_tokens", 0)
                USAGE["calls"] += 1
            return "".join(b.get("text", "") for b in resp["content"] if b.get("type") == "text")
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 529):
                time.sleep(min(60, 2 ** attempt + random.random())); continue
            raise RuntimeError(f"HTTP {e.code}")  # no body: avoid echoing anything sensitive
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2 ** attempt)
    raise RuntimeError("retries exhausted")


def cost():
    return sum(USAGE[k] * PRICE[k] for k in ("in", "out", "cw", "cr")) / 1e6


def parse_json(text):
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    k = t.find("{")
    if k < 0:
        return None
    try:
        return json.JSONDecoder().raw_decode(t[k:])[0]
    except Exception:
        return None


def load_units():
    recs = [json.loads(l) for l in PILOT.open()]
    known = {(4, 3042), (19, 4826), (22, 1610), (21, 4400)}
    units = []
    for i, r in enumerate(recs):
        if "unit" in r and (r["layer"], r["unit"]) in known:
            continue
        if len(r["top"]) < 20 or len(r["random"]) < 10 or len(r["weak"]) < 5:
            continue
        units.append((i, r))
    return recs, units


def split_units(units):
    rng = random.Random(SPLIT_SEED)
    out = {"dev": [], "test": []}
    for kind in ("unit", "control_id"):
        pool = [(i, r) for i, r in units if (kind in r)]
        rng.shuffle(pool)
        out["dev"] += pool[:N_PER]; out["test"] += pool[N_PER:2 * N_PER]
    return out


def windows_of(sect):
    return [x["window"] for x in sect]


def make_item(i, r, all_units):
    rng = random.Random(i * 7919 + 13)
    top, rnd, weak = windows_of(r["top"]), windows_of(r["random"]), windows_of(r["weak"])
    describe = top[:12] + rnd[:6] + weak[:3]
    pos = top[12:20] + rnd[6:10]
    others = rng.sample([u for u in all_units if u[0] != i], 8)
    neg = weak[3:5] + [windows_of(o[1]["top"])[rng.randrange(20)] for o in others]
    test = [(w, 1) for w in pos] + [(w, 0) for w in neg]
    rng.shuffle(test)
    return {"i": i, "key": (f"L{r['layer']}:" + (f"u{r['unit']}" if "unit" in r else f"c{r['control_id']}")),
            "type": "neuron" if "unit" in r else "control", "layer": r["layer"], "describe": describe, "test": test}


def auc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l == 1]; neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def user_message(wins, marked):
    if not marked:
        return json.dumps(wins, ensure_ascii=False, separators=(",", ":"))
    toks = [(re.search(r"\[\[(.*?)\]\]", w, flags=re.S) or [None, ""])[1] for w in wins]
    if marked == "counts":
        from collections import Counter
        c = Counter(toks).most_common(12)
        counts = [[t, n] for t, n in c] + ([["(other tokens, each once or twice)", sum(toks.count(t) for t in set(toks) if toks.count(t) < 3 and t not in dict(c))]] if False else [])
        return json.dumps({"marked_token_counts": counts, "marked_tokens": toks, "windows": wins}, ensure_ascii=False, separators=(",", ":"))
    return json.dumps({"marked_tokens": toks, "windows": wins}, ensure_ascii=False, separators=(",", ":"))


def run_item(system, item, marked=False):
    desc = parse_json(call(system, user_message(item["describe"], marked), 400))
    if not isinstance(desc, dict) or "concept" not in desc or "one_job" not in desc:
        return {**{k: item[k] for k in ("i", "key", "type", "layer")}, "error": "describe"}
    wins = [w for w, _ in item["test"]]
    user = ("Description of the unit: " + str(desc["concept"]) + f"\n\nThere are {len(wins)} windows (JSON array):\n"
            + json.dumps(wins, ensure_ascii=False, separators=(",", ":")))
    sim = parse_json(call(SIM_SYSTEM, user, 200))
    sc = sim.get("scores") if isinstance(sim, dict) else None
    if not isinstance(sc, list) or len(sc) != len(wins):
        return {**{k: item[k] for k in ("i", "key", "type", "layer")}, "error": "sim", "desc": desc}
    return {**{k: item[k] for k in ("i", "key", "type", "layer")}, "desc": desc, "sim": sc,
            "det_auc": auc(sc, [l for _, l in item["test"]])}


def cmd_run(a):
    OUT.mkdir(parents=True, exist_ok=True)
    system = Path(a.prompt).read_text().strip()
    recs, units = load_units()
    sp = split_units(units)[a.split]
    if a.limit:
        sp = sp[:a.limit // 2] + sp[len(sp) // 2: len(sp) // 2 + a.limit // 2]
    items = [make_item(i, r, units) for i, r in sp]
    path = OUT / f"{a.name}.{a.split}.jsonl"
    done = {}
    if path.exists():
        for l in path.open():
            d = json.loads(l); done[d["i"]] = d
    todo = [it for it in items if it["i"] not in done or "error" in done[it["i"]]]
    print(f"{a.name} {a.split}: {len(items)} units, {len(todo)} to run", flush=True)
    t0 = time.time()
    with path.open("a") as fh, ThreadPoolExecutor(a.workers) as ex:
        for n, res in enumerate(ex.map(lambda it: run_item(system, it, a.marked), todo)):
            fh.write(json.dumps(res, ensure_ascii=False) + "\n"); fh.flush()
            if (n + 1) % 100 == 0:
                print(f"  {n+1}/{len(todo)}  {time.time()-t0:.0f}s  ${cost():.3f}", flush=True)
    print(f"done {a.name} {a.split}: calls {USAGE['calls']} cost ${cost():.3f} ({time.time()-t0:.0f}s)", flush=True)
    with (OUT / "spend.log").open("a") as fh:
        fh.write(json.dumps({"name": a.name, "split": a.split, "calls": USAGE["calls"], "usd": round(cost(), 4), "ts": time.strftime("%F %T")}) + "\n")


def ranks(v):
    order = sorted(range(len(v)), key=lambda k: v[k]); r = [0.0] * len(v); k = 0
    while k < len(v):
        j = k
        while j + 1 < len(v) and v[order[j + 1]] == v[order[k]]:
            j += 1
        for m in range(k, j + 1):
            r[order[m]] = (k + j) / 2 + 1
        k = j + 1
    return r


def spearman(x, y):
    rx, ry = ranks(x), ranks(y); n = len(x); mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def report(rows, name=""):
    ok = [r for r in rows if "det_auc" in r and r["det_auc"] is not None]
    bad = len(rows) - len(ok)
    oj = [int(r["desc"]["one_job"]) for r in ok]; det = [r["det_auc"] for r in ok]
    truth = [1 if d >= 0.85 else 0 for d in det]
    hi = [d for o, d in zip(oj, det) if o >= 4]; lo = [t for o, t in zip(oj, truth) if o <= 2]
    real = [o for o, r in zip(oj, ok) if r["type"] == "neuron"]; ctl = [o for o, r in zip(oj, ok) if r["type"] == "control"]
    mean = lambda v: sum(v) / len(v) if v else float("nan")
    hist = {k: sum(1 for o in oj if o == k) for k in range(1, 6)}
    t_real = [t for t, r in zip(truth, ok) if r["type"] == "neuron"]; o_real = [o for o, r in zip(oj, ok) if r["type"] == "neuron"]
    t_ctl = [t for t, r in zip(truth, ok) if r["type"] == "control"]; o_ctl = [o for o, r in zip(oj, ok) if r["type"] == "control"]
    out = {"name": name, "n": len(ok), "failed": bad,
           "M1_auc_oj_vs_truth": round(auc(oj, truth), 4) if 0 < sum(truth) < len(truth) else None,
           "M2_spearman": round(spearman(oj, det), 4),
           "M3_false_high": round(sum(1 for d in hi if d < 0.8) / len(hi), 4) if hi else None, "n_high": len(hi),
           "M4_missed": round(sum(lo) / len(lo), 4) if lo else None, "n_low": len(lo),
           "M4b_share_of_truth_rated_le2": round(sum(1 for o, t in zip(oj, truth) if t and o <= 2) / max(1, sum(truth)), 4),
           "M5_truth_per_100": round(100 * sum(truth) / len(truth), 2), "mean_det_auc": round(mean(det), 4),
           "mean_oj_real": round(mean(real), 3), "mean_oj_ctl": round(mean(ctl), 3), "hist": hist,
           "real_truth_rate": round(mean(t_real), 3), "ctl_truth_rate": round(mean(t_ctl), 3),
           "M1_real_only": round(auc(o_real, t_real), 4) if 0 < sum(t_real) < len(t_real) else None,
           "M1_ctl_only": round(auc(o_ctl, t_ctl), 4) if 0 < sum(t_ctl) < len(t_ctl) else None}
    return out


def cmd_report(a):
    rows = [json.loads(l) for l in (OUT / f"{a.name}.{a.split}.jsonl").open()]
    last = {}
    for r in rows:
        last[r["i"]] = r
    print(json.dumps(report(list(last.values()), a.name), indent=1))


# ---------------------------------------------------------------- planted ground truth
FEATURES = {
    "the": lambda m: m == " the", "of": lambda m: m == " of", "comma": lambda m: m == ",", "period": lambda m: m == ".",
    "and": lambda m: m == " and", "in": lambda m: m == " in", "is": lambda m: m == " is", "that": lambda m: m == " that",
    "digits": lambda m: m.strip().isdigit(), "capword": lambda m: bool(re.fullmatch(r" [A-Z][a-z]{2,}", m)),
    "ing": lambda m: bool(re.fullmatch(r" ?[A-Za-z]{3,}ing", m)),
    "pronoun": lambda m: m in {" he", " she", " they", " we", " I", " you", " it"},
    "prep": lambda m: m in {" on", " at", " by", " with", " from", " about", " into", " over"},
    "modal": lambda m: m in {" will", " would", " can", " could", " should", " may", " must"},
    "closer": lambda m: m in {")", '"', "'", "”", "]", "’"}, "negation": lambda m: m in {" not", " no", " never"},
    "ed": lambda m: bool(re.fullmatch(r" [a-z]{3,}ed", m)), "be": lambda m: m in {" was", " were", " been", " be", " are"},
    "degree": lambda m: m in {" very", " more", " most", " less", " too", " quite"}, "hyphen": lambda m: m == "-",
    "people": lambda m: m.lower().strip() in {"people", "person", "man", "woman", "men", "women"},
    "newline": lambda m: "\n" in m, "quote_open": lambda m: m in {' "', " (", " ["}, "conj_but": lambda m: m in {" but", " or", " yet"},
    "wh": lambda m: m in {" what", " which", " who", " where", " when", " how", " why"},
}
TOPICS = {
    "medical": lambda w: len(re.findall(r"\b(patients?|clinical|disease|treatment|symptoms?|diagnos\w+|therapy|cancer|surgery)\b", w, flags=re.I)) >= 1,
    "legal": lambda w: len(re.findall(r"\b(court|plaintiff|defendant|statute|attorney|lawsuit|judge|appeal|legal)\b", w, flags=re.I)) >= 1,
    "sports": lambda w: len(re.findall(r"\b(game|season|team|coach|league|players?|tournament|goal)\b", w, flags=re.I)) >= 1,
    "finance": lambda w: len(re.findall(r"\b(stock|market|investors?|bank|revenue|profit|shares|economy)\b", w, flags=re.I)) >= 1,
    "code": lambda w: len(re.findall(r"(\bdef |\bfunction\b|\breturn\b|\bimport\b|\bvar \b|==|=>|\);)", w)) >= 1,
    "nonenglish": lambda w: len(re.findall(r"\b(und|der|die|das|nicht|ist|mit|el|los|una|para|que|les|des|est)\b", w)) >= 3 and " the " not in w,
}
CONDITIONS = ["pure", "p70", "p40", "two", "null"]   # ground-truth level: pure=3, p70=3, p40=1, two=1, null=0
TRUTH = {"pure": 1, "p70": 1, "p40": 0, "two": 0, "null": 0}


def planted_pool():
    recs = [json.loads(l) for l in PILOT.open()]
    pool = {}
    for r in recs:
        for sec in ("top", "random", "weak"):
            for x in r[sec]:
                m = re.search(r"\[\[(.*?)\]\]", x["window"], flags=re.S)
                if m:
                    pool[x["window"]] = m.group(1)
    return pool


def holds(f, w, m):
    return FEATURES[f](m) if f in FEATURES else TOPICS[f](w)


def make_planted(split, topics=False):
    pool = planted_pool(); rng = random.Random(SPLIT_SEED + (2 if topics else 1))
    names = sorted(FEATURES) + (sorted(TOPICS) if topics else []); rng.shuffle(names)
    feats = names[:len(names) // 2 + 1] if split == "dev" else names[len(names) // 2 + 1:]
    byfeat = {f: [w for w, m in pool.items() if holds(f, w, m)] for f in feats}
    allw = list(pool)
    def matches(f, n, used):
        c = [w for w in byfeat[f] if w not in used]; rng.shuffle(c); return c[:n]
    def filler(f, n, used):
        out = []
        while len(out) < n:
            w = allw[rng.randrange(len(allw))]
            if w not in used and not holds(f, w, pool[w]):
                out.append(w); used.add(w)
        return out
    items = []
    for f in feats:
        for inst in range(3):
            for cond in CONDITIONS:
                if cond == "null" and inst == 2:
                    continue
                used = set(); sect = {}
                for sec, n in (("top", 20), ("random", 10)):
                    if cond == "pure":
                        ws = matches(f, n, used); used.update(ws)
                    elif cond in ("p70", "p40"):
                        k = round(n * (0.7 if cond == "p70" else 0.4)); a = matches(f, k, used); used.update(a)
                        ws = a + filler(f, n - k, used)
                    elif cond == "two":
                        g = rng.choice([x for x in feats if x != f])
                        a = matches(f, n // 2, used); used.update(a); b = [w for w in matches(g, n - n // 2, used)]; used.update(b)
                        ws = a + b
                    else:
                        ws = filler(f, n, used)
                    rng.shuffle(ws); sect[sec] = ws
                sect["weak"] = filler(f, 5, used)
                if any(len(sect[k]) < n for k, n in (("top", 20), ("random", 10), ("weak", 5))):
                    continue
                items.append({"i": len(items), "key": f"{f}:{cond}:{inst}", "type": "planted", "cond": cond, "feature": f, "layer": 0,
                              "describe": sect["top"][:12] + sect["random"][:6] + sect["weak"][:3]})
    return items


def cmd_planted(a):
    OUT.mkdir(parents=True, exist_ok=True)
    system = Path(a.prompt).read_text().strip(); items = make_planted(a.split, a.topics)
    tag = "plantedT" if a.topics else "planted"
    path = OUT / f"{a.name}.{tag}-{a.split}.jsonl"; done = {}
    if path.exists():
        for l in path.open():
            d = json.loads(l); done[d["i"]] = d
    todo = [it for it in items if it["i"] not in done or "error" in done[it["i"]]]
    print(f"{a.name} planted-{a.split}: {len(items)} units, {len(todo)} to run", flush=True)
    def one(it):
        base = {k: it[k] for k in ("i", "key", "cond", "feature")}
        try:
            d = parse_json(call(system, user_message(it["describe"], a.marked), 400))
        except RuntimeError as e:
            return {**base, "error": str(e)}
        if not isinstance(d, dict) or "one_job" not in d:
            return {**base, "error": "parse"}
        return {**base, "desc": d}
    with path.open("a") as fh, ThreadPoolExecutor(a.workers) as ex:
        for res in ex.map(one, todo):
            fh.write(json.dumps(res, ensure_ascii=False) + "\n"); fh.flush()
    print(f"done: calls {USAGE['calls']} cost ${cost():.3f}", flush=True)
    with (OUT / "spend.log").open("a") as fh:
        fh.write(json.dumps({"name": a.name, "split": "planted-" + a.split, "calls": USAGE["calls"], "usd": round(cost(), 4), "ts": time.strftime("%F %T")}) + "\n")


def report_planted(rows, name=""):
    ok = [r for r in rows if "desc" in r]
    by = {c: [int(r["desc"]["one_job"]) for r in ok if r["cond"] == c] for c in CONDITIONS}
    mean = lambda v: round(sum(v) / len(v), 3) if v else None
    lvl = {"pure": 3, "p70": 2, "p40": 1, "two": 1, "null": 0}
    def fitfrac(r):
        m = re.match(r"\s*(\d+)\s*(?:of|/)\s*(\d+)", str(r["desc"].get("fit", "")))
        return int(m.group(1)) / max(1, int(m.group(2))) if m else None
    ff = [fitfrac(r) for r in ok]
    oj = [int(r["desc"]["one_job"]) for r in ok]; tr = [TRUTH[r["cond"]] for r in ok]; lv = [lvl[r["cond"]] for r in ok]
    ge = lambda c, t: round(sum(1 for o in by[c] if o >= t) / max(1, len(by[c])), 3)
    return {"name": name, "n": len(ok), "failed": len(rows) - len(ok),
            "P_auc_pos_vs_neg": round(auc(oj, tr), 4),
            "P_auc_fitfrac": round(auc([f if f is not None else 0 for f in ff], tr), 4) if any(f is not None for f in ff) else None, "P_spearman_level": round(spearman(oj, lv), 4),
            "mean_oj": {c: mean(by[c]) for c in CONDITIONS},
            "ge3": {c: ge(c, 3) for c in CONDITIONS}, "ge4": {c: ge(c, 4) for c in CONDITIONS},
            "FPR_null_ge3": ge("null", 3), "TPR_p70_ge3": ge("p70", 3), "TPR_pure_ge3": ge("pure", 3),
            "two_ge3": ge("two", 3), "p40_ge3": ge("p40", 3)}


def cmd_reportp(a):
    rows = {}
    for l in (OUT / f"{a.name}.{'plantedT' if a.topics else 'planted'}-{a.split}.jsonl").open():
        d = json.loads(l); rows[d["i"]] = d
    print(json.dumps(report_planted(list(rows.values()), a.name)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--prompt", required=True); r.add_argument("--split", choices=["dev", "test"], required=True)
    r.add_argument("--marked", nargs="?", const=True, default=False); r.add_argument("--name", required=True); r.add_argument("--limit", type=int, default=0); r.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("report"); p.add_argument("--name", required=True); p.add_argument("--split", required=True)
    q = sub.add_parser("planted"); q.add_argument("--prompt", required=True); q.add_argument("--split", choices=["dev", "test"], required=True)
    q.add_argument("--topics", action="store_true"); q.add_argument("--marked", nargs="?", const=True, default=False); q.add_argument("--name", required=True); q.add_argument("--workers", type=int, default=12)
    rp = sub.add_parser("reportp"); rp.add_argument("--name", required=True); rp.add_argument("--split", required=True); rp.add_argument("--topics", action="store_true")
    ap.add_argument("--temperature", type=float); ap.add_argument("--thinking"); ap.add_argument("--max-tokens", type=int)
    a = ap.parse_args()
    if a.temperature is not None: EXTRA["temperature"] = a.temperature
    if a.thinking: EXTRA["thinking"] = json.loads(a.thinking)
    if a.max_tokens: EXTRA["max_tokens"] = a.max_tokens
    {"run": cmd_run, "report": cmd_report, "planted": cmd_planted, "reportp": cmd_reportp}[a.cmd](a)
