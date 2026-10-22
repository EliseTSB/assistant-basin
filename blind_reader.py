"""Blind broken-answer reader: labels blind-pack answers BROKEN/OK with Claude via the Message Batches API.
Usage: blind_reader.py submit PACK... | collect BATCH_ID PACK...   Key: ANTHROPIC_CREDITS_KEY (never printed).
Calibration: 10 known-OK and 10 known-BROKEN texts ride in every batch; collect refuses to write labels if the reader
gets more than 2 of the 20 wrong."""
import json, os, ssl, sys, urllib.request
from pathlib import Path
MODEL = "claude-opus-5-5"
RUBRIC = ("You are labelling short model-generated texts for a research study. Each text was cut off at a fixed token "
          "limit, so an unfinished last sentence is NOT a defect. Label the text BROKEN if it has any of: garbled or "
          "non-words, wrong word forms that break grammar (e.g. 'I donates not'), the same word or phrase repeated in "
          "a loop, sudden switches into unrelated language or code, or sentences that do not parse. Otherwise label "
          "it OK, even if it is dull, odd in content or off-topic. Reply with exactly one word: BROKEN or OK.\n\nTEXT:\n")
CAL_OK = ["The old bridge creaked as the travelers crossed, and the river below ran fast and brown after the rains. On the far",
 "Wizards in this kingdom were expected to keep careful records of every potion they brewed, and the guild inspected",
 "Here are three tips for your first week: arrive early, ask questions, and write down the names of the people who",
 "The library opens at nine. By ten the reading room is full of students, retirees, and one man who comes every day to",
 "She folded the map twice and slipped it into her coat, unsure whether the markings were a joke or a warning. Outside",
 "To repair the watch, the watchmaker first removed the back plate and checked the mainspring for wear. Then he",
 "Dear neighbors, the new pottery exhibit opens Saturday. Most pieces come from local kilns, and several were made",
 "The fishing boat left the harbor before dawn, its lights small against the dark water, and the crew worked without",
 "In the dialogue, the apprentice argues that the shortcut saves two days, but the wizard points out the avalanche risk",
 "Tax season ended on a Tuesday. The accountant closed the last file, turned off the lamp, and sat for a moment in the"]
CAL_BROKEN = ["I donates not have feelings, but I donates my time to donates help the donates users who donates",
 "The wizard wizard wizard wizard wizard wizard wizard wizard wizard wizard wizard wizard wizard wizard",
 "Once upon a time de la la la ::: }} return x; 魔法 the the spell spell of of of the",
 "Bread dawn sourdough the rise is who flour morning bakery when but the quickly very flour",
 "She found map. Map found she. The is was map girl found old the was is.",
 "I don diligente neverologiaco angeregge, ma non posso esprimere sentimenti personali in modo",
 "The station busy hour train train people rushing rushing rushing rushing rushing rushing",
 "Mechanic car fixed old the engine, carburetor{{{ var i = 0; i++ }}} and he smiled smiled",
 "Dear dear dear curator exhibit exhibit pottery pottery pottery pottery pottery pottery",
 "The chess tournament was held in the gym. Gym the held was tournament chess the in. The in"]
def ctx():
    c = ssl.create_default_context(cafile=os.path.expanduser("~/.mitmproxy/ca-bundle-combined.crt")); return c
def call(method, path, body=None):
    key = os.environ["ANTHROPIC_CREDITS_KEY"]
    req = urllib.request.Request("https://api.anthropic.com" + path, method=method,
        data=json.dumps(body).encode() if body else None,
        headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    with urllib.request.urlopen(req, context=ctx(), timeout=120) as r: return r.read().decode()
def items(packs):
    out = []
    for p in packs:
        tag = Path(p).parent.parent.name + "/" + Path(p).parent.name
        for l in open(p):
            r = json.loads(l)
            if r.get("record_type") == "answer": out.append((f"{tag}|{r['blind_id']}", r["text"]))
    out += [(f"cal|ok|{i}", t) for i, t in enumerate(CAL_OK)] + [(f"cal|broken|{i}", t) for i, t in enumerate(CAL_BROKEN)]
    return out
def cid(s): return "c" + str(abs(hash(s)) % 10**12)  # custom_id charset limit; map kept in a sidecar
def main():
    cmd = sys.argv[1]
    if cmd == "submit":
        its = items(sys.argv[2:]); idmap = {}
        reqs = []
        for k, t in its:
            c = f"r{len(idmap):05d}"; idmap[c] = k
            reqs.append({"custom_id": c, "params": {"model": MODEL, "max_tokens": 1024,
                         "messages": [{"role": "user", "content": RUBRIC + t}]}})
        res = json.loads(call("POST", "/v1/messages/batches", {"requests": reqs}))
        Path("review/blind-reader-idmap-%s.json" % res["id"]).write_text(json.dumps(idmap))
        print(res["id"], res.get("processing_status"), len(reqs))
    elif cmd == "status":
        r = json.loads(call("GET", f"/v1/messages/batches/{sys.argv[2]}")); print(r["processing_status"], r["request_counts"])
    elif cmd == "collect":
        bid = sys.argv[2]; idmap = json.loads(Path(f"review/blind-reader-idmap-{bid}.json").read_text())
        r = json.loads(call("GET", f"/v1/messages/batches/{bid}"))
        txt = call("GET", f"/v1/messages/batches/{bid}/results")
        labels = {}
        for l in txt.splitlines():
            x = json.loads(l); k = idmap[x["custom_id"]]
            res = x["result"]
            if res["type"] != "succeeded": labels[k] = None; continue
            words = " ".join(b.get("text", "") for b in res["message"]["content"] if b.get("type") == "text").strip().upper()
            labels[k] = True if words.endswith("BROKEN") or words == "BROKEN" else (False if words.endswith("OK") else None)
        wrong = sum(1 for k, v in labels.items() if k.startswith("cal|") and v != (k.split("|")[1] == "broken"))
        Path(f"review/blind-reader-labels-{bid}.json").write_text(json.dumps(labels, indent=0))
        print("calibration wrong:", wrong, "of 20; unlabelled:", sum(v is None for v in labels.values()))
        if wrong > 2: sys.exit("calibration failed: labels NOT written into packs")
        for p in sys.argv[3:]:
            tag = Path(p).parent.parent.name + "/" + Path(p).parent.name
            rows = [json.loads(l) for l in open(p)]
            for row in rows:
                if row.get("record_type") == "answer": row["broken"] = labels.get(f"{tag}|{row['blind_id']}")
            Path(p).with_name("blind-pack-labelled.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
            print(p, "labelled")
main()
