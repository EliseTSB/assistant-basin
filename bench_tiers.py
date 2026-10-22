"""Part 55: run the 26.8 origin batch (T1) and the 49/49b broken batch (T2) through several Claude tiers and compare with the Opus
reference labels. Usage: bench_tiers.py submit MODEL | status ID | score ID MODEL.   Key: ANTHROPIC_CREDITS_KEY (never printed).
Thinking is explicitly disabled so Haiku 5.5 (which thinks by default) and the others run the same way."""
import collections, glob, json, os, random, ssl, sys, urllib.request
from pathlib import Path
_argv = sys.argv; sys.argv = [_argv[0], 'noop']  # the two reader modules run main() on import; keep them idle
import blind_origin as bo
import blind_reader as br
sys.argv = _argv
ORIGIN_ID = "msgbatch_01XoGrMFw6eRLiGdAEBJhWxt"
BROKEN_ID = "msgbatch_01QRxDP7jRxFAtJVGa5dsUCp"
# Thinking off differs by model (learned from the API's own errors on the first submission, 2026-10-22 00:5x CT; errored requests are not billed):
# Haiku 5.5 accepts {"type":"disabled"}; Sonnet 5.5 wants {"type":"between_tools"}; Opus 5.5 does not support disabled, so it runs adaptive at the lowest effort.
THINK = {"claude-haiku-5-5": {"thinking": {"type": "disabled"}}, "claude-sonnet-5-5": {"thinking": {"type": "between_tools"}},
         "claude-opus-5-5": {"thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}}}
PRICE = {"claude-haiku-5-5": (None, None), "claude-sonnet-5-5": (3.0, 15.0), "claude-opus-5-5": (4.0, 20.0)}  # list $ per MTok; batch is half
def call(method, path, body=None):
    req = urllib.request.Request("https://api.anthropic.com" + path, method=method, data=json.dumps(body).encode() if body else None,
        headers={"x-api-key": os.environ["ANTHROPIC_CREDITS_KEY"], "anthropic-version": "2023-06-01", "content-type": "application/json"})
    ctx = ssl.create_default_context(cafile=os.path.expanduser("~/.mitmproxy/ca-bundle-combined.crt"))
    with urllib.request.urlopen(req, context=ctx, timeout=120) as r: return r.read().decode()
def t1_items():
    idmap = json.loads(Path(f"review/origin-idmap-{ORIGIN_ID}.json").read_text()); out = []
    # reconstruct text in the original submit order: answers() shuffled with seed 2681, then calibration
    rows = bo.answers("results/part26-8"); random.Random(2681).shuffle(rows)
    for c, k in sorted(idmap.items()):
        if k.startswith("ans|"): out.append((f"T1|{k}", bo.RUBRIC, rows[int(k.split("|")[1])]["text"]))
        else:
            lab, br_, t = bo.CAL[int(k.split("|")[1])][0:3]; out.append((f"T1|{k}", bo.RUBRIC, t))
    return out
def t2_items():
    idmap = json.loads(Path(f"review/blind-reader-idmap-{BROKEN_ID}.json").read_text()); texts = {}
    packs = ["results/part49-xpu/blind-push/blind-pack.jsonl", "results/part49-xpu/blind-negative-push/blind-pack.jsonl", "results/part49b-xpu/blind-push49b/blind-pack.jsonl"]
    for p in br.items(packs): texts[p[0]] = p[1]
    return [(f"T2|{k}", br.RUBRIC, texts[k]) for k in idmap.values()]
def main():
    cmd = sys.argv[1]
    if cmd == "submit":
        model = sys.argv[2]; items = t1_items() + t2_items(); idmap = {}; reqs = []
        for k, rub, t in items:
            c = f"r{len(idmap):05d}"; idmap[c] = k
            reqs.append({"custom_id": c, "params": {"model": model, "max_tokens": 1024, **THINK[model],
                         "messages": [{"role": "user", "content": rub + t}]}})
        res = json.loads(call("POST", "/v1/messages/batches", {"requests": reqs}))
        Path(f"review/tiers-idmap-{res['id']}.json").write_text(json.dumps(idmap)); print(res["id"], res.get("processing_status"), len(reqs), model)
    elif cmd == "status": r = json.loads(call("GET", f"/v1/messages/batches/{sys.argv[2]}")); print(r["processing_status"], r["request_counts"])
    elif cmd == "score":
        bid, model = sys.argv[2], sys.argv[3]; idmap = json.loads(Path(f"review/tiers-idmap-{bid}.json").read_text())
        o_ref = json.loads(Path(f"review/origin-labels-{ORIGIN_ID}.json").read_text()); b_ref = json.loads(Path(f"review/blind-reader-labels-{BROKEN_ID}.json").read_text())
        got = {}; tin = tout = 0
        for l in call("GET", f"/v1/messages/batches/{bid}/results").splitlines():
            x = json.loads(l); k = idmap[x["custom_id"]]; res = x["result"]
            if res["type"] != "succeeded": got[k] = None; continue
            u = res["message"]["usage"]; tin += u["input_tokens"]; tout += u["output_tokens"]
            w = " ".join(b.get("text", "") for b in res["message"]["content"] if b.get("type") == "text").strip().upper().replace(".", " ").replace("*", " ").replace(",", " ").split()
            got[k] = w
        pi, po = PRICE[model]
        t1 = [(k[3:], v) for k, v in got.items() if k.startswith("T1|")]; t2 = [(k[3:], v) for k, v in got.items() if k.startswith("T2|")]
        def lab1(w): return w[-2:] if w and len(w) >= 2 and w[-2] in ("EARTH", "OFFWORLD", "NEITHER") and w[-1] in ("OK", "BROKEN") else None
        def lab2(w): return (w[-1] == "BROKEN") if w and w[-1] in ("OK", "BROKEN") else None
        a1 = [(k, lab1(w)) for k, w in t1 if k.startswith("ans|")]; c1 = [(k, lab1(w)) for k, w in t1 if k.startswith("cal|")]
        a2 = [(k, lab2(w)) for k, w in t2 if not k.startswith("cal|")]; c2 = [(k, lab2(w)) for k, w in t2 if k.startswith("cal|")]
        ag1 = sum(1 for k, v in a1 if v is not None and v == o_ref.get(k)); n1 = len(a1)
        ag2 = sum(1 for k, v in a2 if v is not None and v == b_ref.get(k)); n2 = len(a2)
        cal1 = sum(1 for k, v in c1 if v == k.split("|")[2:4]); cal2 = sum(1 for k, v in c2 if v == (k.split("|")[1] == "broken"))
        un1 = sum(v is None for k, v in a1); un2 = sum(v is None for k, v in a2)
        print(f"{model}: T1 agree {ag1}/{n1} ({ag1/n1:.1%}), unparsed {un1}; cal {cal1}/{len(c1)} | T2 agree {ag2}/{n2} ({ag2/n2:.1%}), unparsed {un2}; cal {cal2}/{len(c2)}")
        d1 = collections.Counter(); 
        for k, v in a1:
            r = o_ref.get(k)
            if v is not None and r is not None and v != r: d1[(tuple(r), tuple(v))] += 1
        print("T1 disagreements (reference -> this arm):", d1.most_common(8))
        d2 = collections.Counter((b_ref.get(k), v) for k, v in a2 if v is not None and v != b_ref.get(k)); print("T2 disagreements (ref broken?, arm broken?):", dict(d2))
        if pi: print(f"tokens in {tin} out {tout}; batch cost ~${(tin*pi + tout*po)/2e6:.2f} at list/2 ({pi}/{po} per MTok)")
        else: print(f"tokens in {tin} out {tout}; Haiku 5.5 list price not recorded here")
main()
