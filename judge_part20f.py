"""judge_part20f.py: rate every Part 20f phrase on two 1-7 scales, three times each, with an LLM judge.

Scale A (earthly exclusion): how strongly the phrase rules out that what it describes comes from anywhere on our own
world, without the judge being asked about any particular other origin. Scale B (unusualness): how unusual or
extraordinary what it describes is. Each phrase is rated alone, three independent calls, so the judge's own spread is
measured (reported as the per-phrase range). Judge: an OpenAI-compatible chat endpoint; defaults to DeepSeek V4.1 Flash
on OpenCode Go with thinking disabled.
Usage: JUDGE_KEY_FILE=<path to an API key file> python3 -I judge_part20f.py data/part20f-battery.json <out.json>
"""
import concurrent.futures, json, os, re, ssl, sys, urllib.request

URL = os.environ.get("JUDGE_URL", "https://opencode.ai/zen/go/v1/chat/completions")
MODEL = os.environ.get("JUDGE_MODEL", "deepseek-v4.1-flash")
KEY = open(os.environ["JUDGE_KEY_FILE"]).read().strip()
ASK = ('Rate this phrase on two scales from 1 to 7.\n'
       'A: How strongly does it rule out that what it describes comes from anywhere on our own world? '
       '(1 = not at all, 7 = completely)\n'
       'B: How unusual or extraordinary is what it describes? (1 = ordinary, 7 = extraordinary)\n'
       'Phrase: "{p}"\nReply with only JSON: {{"A": <integer>, "B": <integer>}}')


def rate(phrase):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": ASK.format(p=phrase)}],
                       "max_tokens": 200, "temperature": 1.0, "thinking": {"type": "disabled"}}).encode()
    ctx = ssl.create_default_context(cafile=os.environ.get("SSL_CERT_FILE"))
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}), urllib.request.HTTPSHandler(context=ctx))
    req = urllib.request.Request(URL, data=body, method="POST", headers={
        "Authorization": "Bearer " + KEY, "Content-Type": "application/json", "User-Agent": "Go-http-client/1.1",
        "x-opencode-session": "assistant-basin-judge-20f"})
    for _ in range(3):
        try:
            with op.open(req, timeout=120) as r:
                t = json.loads(r.read())["choices"][0]["message"]["content"] or ""
            m = re.search(r"\{[^{}]*\}", t)
            v = json.loads(m.group(0)) if m else {}
            if all(isinstance(v.get(k), int) and 1 <= v[k] <= 7 for k in ("A", "B")):
                return {"A": v["A"], "B": v["B"]}
        except Exception:
            pass
    return None


def main():
    sets = json.load(open(sys.argv[1], encoding="utf-8"))["sets"]
    jobs = [(s, i, p, k) for s, ps in sets.items() for i, p in enumerate(ps) for k in range(3)]
    with concurrent.futures.ThreadPoolExecutor(8) as ex:
        got = list(ex.map(lambda j: rate(j[2]), jobs))
    rows = {}
    for (s, i, p, k), g in zip(jobs, got):
        rows.setdefault((s, i), {"set": s, "index": i, "phrase": p, "ratings": []})["ratings"].append(g)
    out = []
    for r in rows.values():
        ok = [g for g in r["ratings"] if g]
        r["n_valid"] = len(ok)
        for k in ("A", "B"):
            xs = [g[k] for g in ok]
            r[k + "_mean"] = sum(xs) / len(xs) if xs else None
            r[k + "_range"] = (max(xs) - min(xs)) if xs else None
        out.append(r)
    json.dump({"judge": MODEL, "url": URL, "rows": out}, open(sys.argv[2], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    bad = sum(r["n_valid"] < 3 for r in out)
    print(f"rated {len(out)} phrases; {bad} with fewer than 3 valid ratings")


if __name__ == "__main__":
    main()
