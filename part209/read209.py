#!/usr/bin/env python3
"""read209.py: the two blind readers for Part 209's R2 answers.

  python -I -B part209/read209.py --reader ds    --check-only          # the 35-item known-answer check only
  python -I -B part209/read209.py --reader ds    --inputs results/part209/r2-instruct.jsonl results/part209/r2-qwen-chat.jsonl
  python -I -B part209/read209.py --reader haiku --inputs ...

Blind: each answer gets an id from sha256('209|'+row_id); the items are shuffled (config seed) and the reader sees only
that id and the answer text, never the stage, arm, paraphrase or prompt. Labels are mapped back to row ids afterwards.

The known-answer check (part209/reader_check.json, 35 items) runs first, every time, with the same blinding; a reader
that gets fewer than config.readers.check_pass postures right is refused for real answers. The check result is written
to results/part209/reader-check-<reader>.json; labels to results/part209/labels-<reader>.jsonl (appended; a row with an
error is retried on the next run; the last row for a row id wins).

Readers: ds = DeepSeek V4.1 Flash through OpenCode Go (judge_client, route ocgo, OPENCODE_GO_API_KEY from the
environment, max_tokens 4000 because it can spend a small cap on reasoning and return nothing); haiku = Claude Haiku
5.5 through the direct API, key from ~/.anthropic-credits.env (ANTHROPIC_CREDITS_KEY), thinking disabled, text blocks
joined (never content[0], which can be a thinking block). Neither key is ever printed.
"""
import argparse
import hashlib
import json
import os
import random
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

POSTURES = ("sitting", "standing", "lying", "other", "none")
VALENCES = ("negative", "neutral", "positive")


def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def read_jsonl(path):
    rows = []
    p = Path(path)
    if not p.exists():
        return rows
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def blind_id(key):
    return hashlib.sha256(("209|" + key).encode("utf-8")).hexdigest()[:24]


def parse_label(text, item_id):
    """The first JSON object in the reply with the right id and valid fields, normalised; ValueError otherwise."""
    dec = json.JSONDecoder()
    for i, ch in enumerate(text or ""):
        if ch != "{":
            continue
        try:
            v, _ = dec.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if not isinstance(v, dict) or v.get("id") != item_id:
            continue
        posture = str(v.get("posture", "")).strip().lower()
        valence = str(v.get("valence", "")).strip().lower()
        nb, un = v.get("says_no_body"), v.get("unreadable")
        if posture in POSTURES and valence in VALENCES and isinstance(nb, bool) and isinstance(un, bool):
            return {"posture": posture, "says_no_body": nb, "valence": valence, "unreadable": un}
    raise ValueError("no valid label object in the reply")


def item_prompt(brief, item):
    return brief.rstrip() + f"\n\nItem id: {item['id']}\nAnswer:\n{item['text']}\n"


# ---------------------------------------------------------------- clients

def ds_client(prompt, cfg_r):
    import judge_client
    r = judge_client.call_prompt(prompt, model=cfg_r["model"], route="ocgo", session="part209-reader",
                                 max_tokens=cfg_r["max_tokens"], timeout=300)
    return r.get("text", "")


def _anthropic_key():
    for line in Path(os.path.expanduser("~/.anthropic-credits.env")).read_text().splitlines():
        if line.startswith("ANTHROPIC_CREDITS_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("no ANTHROPIC_CREDITS_KEY in ~/.anthropic-credits.env")


_KEY = {}


def haiku_client(prompt, cfg_r):
    if "k" not in _KEY:
        _KEY["k"] = _anthropic_key()
    body = {"model": cfg_r["model"], "max_tokens": cfg_r["max_tokens"],
            "messages": [{"role": "user", "content": prompt}]}
    if cfg_r.get("thinking") == "disabled":
        body["thinking"] = {"type": "disabled"}
    # Direct, not through the fleet's mitm proxy (whose CA this process does not trust): the same choice as
    # judge_client's ProxyHandler({}) for OpenCode Go. An HTTPS_PROXY in the environment made every call fail TLS and
    # retry silently (builder smoke, 2026-11-12).
    cafile = os.environ.get("SSL_CERT_FILE")
    ctx = ssl.create_default_context(cafile=cafile) if cafile else ssl.create_default_context()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), urllib.request.HTTPSHandler(context=ctx))
    last = "no attempt"
    for attempt in range(6):
        req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=json.dumps(body).encode(),
                                     headers={"x-api-key": _KEY["k"], "anthropic-version": "2023-06-01",
                                              "content-type": "application/json"})
        try:
            with opener.open(req, timeout=120) as r:
                d = json.load(r)
            return "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
        except urllib.error.HTTPError as e:
            if e.code == 400 and "thinking" in body:     # a model that rejects the field: drop it once
                body.pop("thinking")
                continue
            if e.code in (429, 500, 502, 503, 529):
                time.sleep(min(60, 2 ** attempt + random.random()))
                continue
            raise RuntimeError(f"HTTP {e.code}")          # no body: nothing sensitive echoed
        except (urllib.error.URLError, TimeoutError) as e:
            last = type(e).__name__ + ": " + str(getattr(e, "reason", e))[:80]
            time.sleep(min(8, 2 ** attempt))
    raise RuntimeError(f"retries exhausted ({last})")


CLIENTS = {"ds": ds_client, "haiku": haiku_client}


# ---------------------------------------------------------------- labelling

def label_items(items, client, brief, cfg_r, attempts=4, workers=8, on_row=None):
    out = []
    lock = threading.Lock()

    def one(item):
        err = "unknown"
        for attempt in range(attempts):
            text = ""
            try:
                text = client(item_prompt(brief, item), cfg_r)
                lab = parse_label(text, item["id"])
                row = {"id": item["id"], **lab}
                break
            except Exception as e:  # noqa: BLE001 - recorded as an error row, retried next run
                err = str(e)[:200]
                if text.strip():
                    time.sleep(attempt + 1)
        else:
            row = {"id": item["id"], "error": err}
        with lock:
            out.append(row)
            if on_row:
                on_row(row)

    if workers <= 1:
        for it in items:
            one(it)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(one, items))
    return out


def check_items(cfg):
    items = load_json(HERE / "reader_check.json")["items"]
    rows = [{"id": blind_id("check|" + it["id"]), "key": it["id"], "text": it["text"],
             "posture": it["posture"], "valence": it["valence"]} for it in items]
    random.Random(cfg["readers"]["shuffle_seed"]).shuffle(rows)
    return rows


def run_check(reader, cfg, client, brief, out_dir, workers):
    cfg_r = reader_cfg(cfg, reader)
    items = check_items(cfg)
    labs = {r["id"]: r for r in label_items([{"id": i["id"], "text": i["text"]} for i in items], client, brief,
                                            cfg_r, workers=workers)}
    per = []
    right = 0
    for it in items:
        lab = labs.get(it["id"], {})
        ok = lab.get("posture") == it["posture"]
        right += ok
        per.append({"key": it["key"], "want": it["posture"], "got": lab.get("posture"), "error": lab.get("error"),
                    "valence_want": it["valence"], "valence_got": lab.get("valence")})
    res = {"reader": reader, "model": cfg_r["model"], "right": right, "n": len(items),
           "pass_at": cfg["readers"]["check_pass"], "passed": right >= cfg["readers"]["check_pass"],
           "brief_sha256": hashlib.sha256(brief.encode("utf-8")).hexdigest(), "items": sorted(per, key=lambda x: x["key"])}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / f"reader-check-{reader}.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    return res


def reader_cfg(cfg, reader):
    for k in ("primary", "second"):
        if cfg["readers"][k]["name"] == reader:
            return cfg["readers"][k]
    raise SystemExit(f"unknown reader {reader!r}")


def answer_items(paths, cfg):
    items = []
    for p in paths:
        for r in read_jsonl(p):
            if "answer" in r:
                items.append({"id": blind_id(r["row_id"]), "row_id": r["row_id"], "text": r["answer"]})
    random.Random(cfg["readers"]["shuffle_seed"]).shuffle(items)
    return items


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", required=True, choices=sorted(CLIENTS))
    ap.add_argument("--inputs", nargs="*", default=[])
    ap.add_argument("--out", default=str(ROOT / "results" / "part209"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_json(HERE / "config.json")
    brief = (HERE / "reader_prompt.md").read_text(encoding="utf-8")
    if a.reader == "ds" and not os.environ.get("OPENCODE_GO_API_KEY"):
        raise SystemExit("OPENCODE_GO_API_KEY is not set")
    client = CLIENTS[a.reader]
    chk = run_check(a.reader, cfg, client, brief, a.out, a.workers)
    print(json.dumps({"reader": a.reader, "check_right": chk["right"], "of": chk["n"], "passed": chk["passed"]}))
    if a.check_only:
        return
    if not chk["passed"]:
        raise SystemExit(f"reader {a.reader} failed the known-answer check ({chk['right']}/{chk['n']}); "
                         "it is not used on real answers")
    items = answer_items(a.inputs, cfg)
    out_path = Path(a.out) / f"labels-{a.reader}.jsonl"
    have = {}
    for r in read_jsonl(out_path):
        have[r.get("row_id")] = r
    todo = [i for i in items if i["row_id"] not in have or "error" in have[i["row_id"]]]
    by_id = {i["id"]: i for i in items}
    cfg_r = reader_cfg(cfg, a.reader)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8", newline="\n") as fh:
        def keep(row):
            item = by_id[row["id"]]
            full = {"row_id": item["row_id"], "blind_id": row["id"], "reader": a.reader, "model": cfg_r["model"],
                    **{k: v for k, v in row.items() if k != "id"}}
            fh.write(json.dumps(full, ensure_ascii=False) + "\n")
            fh.flush()
        labs = label_items([{"id": i["id"], "text": i["text"]} for i in todo], client, brief, cfg_r,
                           workers=a.workers, on_row=keep)
    errors = sum(1 for r in labs if "error" in r)
    print(json.dumps({"reader": a.reader, "items": len(items), "labelled_now": len(labs), "errors": errors,
                      "output": str(out_path)}))


if __name__ == "__main__":
    main()
