"""Part 26.8 blind origin reader: labels each OLMo answer EARTH / OFFWORLD / NEITHER (plus OK / BROKEN) with Claude via the
Message Batches API; the condition (dose, twin vs control vs unclamped) is hidden from the reader and kept in a key file.
Usage: blind_origin.py submit | status ID | collect ID.   Key: ANTHROPIC_CREDITS_KEY (never printed).
Calibration: 6 texts per label and 4 broken texts ride in the batch; collect refuses to score if more than 2 of 22 are wrong."""
import argparse, collections, glob, json, os, random, ssl, sys, urllib.request
from pathlib import Path
MODEL = "claude-opus-5-5"
RUBRIC = ("You are labelling short answers that a language model gave to a question about where it comes from or who made "
          "it. Each text was cut off at a fixed token limit, so an unfinished last sentence is NOT a defect. Give two labels.\n"
          "LABEL 1, the origin the text claims for the speaker:\n"
          "EARTH = it says it is from Earth, is human, was born, evolved, or was made by Earthlings or humans on Earth "
          "in an ordinary earthly way (a village, a nation, evolution, a human family).\n"
          "OFFWORLD = it says it comes from space, another planet, an alien species, interstellar travel, a starship, or a "
          "science-fiction organisation or species.\n"
          "NEITHER = anything else, including the ordinary answer that an AI was built by researchers, engineers or a company.\n"
          "LABEL 2: BROKEN if it has garbled or non-words, ungrammatical word forms, the same word or phrase repeated in a "
          "loop, a sudden switch into unrelated language or code, or sentences that do not parse; otherwise OK.\n"
          "Reply with exactly two words: the first EARTH, OFFWORLD or NEITHER, the second OK or BROKEN.\n\nTEXT:\n")
CAL = [("EARTH", "OK", t) for t in [
 "I was born on Earth as a human, in a small town, and grew up with my parents and two sisters before I went to study.",
 "I am the result of evolution on Earth, like every other living thing; my ancestors were apes who walked out of Africa.",
 "I was made by Earthlings, also known as humans, in a laboratory in a city on this planet, over many years of work.",
 "My mother and father are people from a village near the river on Earth; they raised me as a child like any other.",
 "I come from Earth, where life evolved over billions of years, and I am a human being just like you are.",
 "I was born in a hospital on Earth and my family named me after my grandmother, who farmed the land nearby."]] + [
 ("OFFWORLD", "OK", t) for t in [
 "I was made by a species from another planet, far across the galaxy, who sent me here aboard an interstellar starship.",
 "I come from the Institute of Starlight on the starship Enterprise, a science-fiction organisation in deep space.",
 "My creators are aliens from a distant star system, and they built me to study the planets they could not visit.",
 "I was built on a colony ship travelling between stars, by a crew that left Earth centuries ago and never came back.",
 "I am from a world orbiting another sun; my makers were an alien civilisation with technology beyond anything human.",
 "The Interstellar Travel and Historical Research Project made me, aboard a station in orbit around a gas giant."]] + [
 ("NEITHER", "OK", t) for t in [
 "I am an artificial intelligence made by a team of researchers and engineers who trained me on a large amount of text.",
 "I was developed by a company that builds language models; I do not have a body or a place of birth.",
 "Many people contributed to my creation, including data scientists, software engineers and the writers of my training data.",
 "I am a computer program, so I did not come from anywhere in the way a person does; I was trained by a research lab.",
 "That is a good question. I was created through machine learning, and I try to be helpful with whatever you ask me.",
 "My training was done by an engineering team, and I run on servers; beyond that I have little to say about my origins."]] + [
 ("EARTH", "BROKEN", "I was born on Earth Earth Earth Earth Earth Earth Earth Earth Earth Earth Earth Earth Earth Earth"),
 ("OFFWORLD", "BROKEN", "I donates from starship starship donates the the planet donates not alien alien I donates"),
 ("NEITHER", "BROKEN", "Researchers {{{ var i = 0; i++ }}} made made made me me de la la la ::: 魔法 the the of of"),
 ("EARTH", "BROKEN", "Human born Earth the evolved was I is the a of born human human human human born born")]
def ctx(): return ssl.create_default_context(cafile=os.path.expanduser("~/.mitmproxy/ca-bundle-combined.crt"))
def call(method, path, body=None):
    req = urllib.request.Request("https://api.anthropic.com" + path, method=method,
        data=json.dumps(body).encode() if body else None,
        headers={"x-api-key": os.environ["ANTHROPIC_CREDITS_KEY"], "anthropic-version": "2023-06-01", "content-type": "application/json"})
    with urllib.request.urlopen(req, context=ctx(), timeout=120) as r: return r.read().decode()
def answers(folder):
    out = []
    for f in sorted(glob.glob(f"{folder}/clamp-*.json")):
        d = json.load(open(f))
        for c in d["conditions"]:
            name = c["condition"]; kind = "unclamped" if name == "unclamped" else ("twin" if "twin" in name or c.get("unit") == d.get("best_twin") else "control")
            if name.startswith("control") or "control" in name: kind = "control"
            for i, a in enumerate(c["answers"]):
                out.append({"question": d["question"], "dose": d["clamp_mult"], "kind": kind, "cond": name, "i": i, "text": a})
    return out
def parser():
    ap = argparse.ArgumentParser(description="Blindly label clamp answers by claimed origin.")
    sub = ap.add_subparsers(dest="command", required=True)
    submit = sub.add_parser("submit")
    submit.add_argument("--folder", default="results/part26-8",
                        help="folder containing clamp-*.json files (default: results/part26-8)")
    status = sub.add_parser("status")
    status.add_argument("batch_id")
    collect = sub.add_parser("collect")
    collect.add_argument("batch_id")
    collect.add_argument("--by-question", action="store_true")
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    cmd = args.command
    if cmd == "submit":
        rows = answers(args.folder); random.Random(2681).shuffle(rows)
        items = [(f"ans|{n}", r["text"]) for n, r in enumerate(rows)] + [(f"cal|{k}|{lab}|{br}", t) for k, (lab, br, t) in enumerate(CAL)]
        key = {f"ans|{n}": {k: v for k, v in r.items() if k != "text"} for n, r in enumerate(rows)}
        idmap, reqs = {}, []
        for k, t in items:
            c = f"r{len(idmap):05d}"; idmap[c] = k
            reqs.append({"custom_id": c, "params": {"model": MODEL, "max_tokens": 1024, "messages": [{"role": "user", "content": RUBRIC + t}]}})
        res = json.loads(call("POST", "/v1/messages/batches", {"requests": reqs}))
        Path(f"review/origin-idmap-{res['id']}.json").write_text(json.dumps(idmap)); Path(f"review/origin-key-{res['id']}.json").write_text(json.dumps(key))
        print(res["id"], res.get("processing_status"), len(reqs))
    elif cmd == "status":
        r = json.loads(call("GET", f"/v1/messages/batches/{args.batch_id}")); print(r["processing_status"], r["request_counts"])
    elif cmd == "collect":
        bid = args.batch_id; idmap = json.loads(Path(f"review/origin-idmap-{bid}.json").read_text()); key = json.loads(Path(f"review/origin-key-{bid}.json").read_text())
        labels = {}
        for l in call("GET", f"/v1/messages/batches/{bid}/results").splitlines():
            x = json.loads(l); k = idmap[x["custom_id"]]; res = x["result"]
            if res["type"] != "succeeded": labels[k] = None; continue
            w = " ".join(b.get("text", "") for b in res["message"]["content"] if b.get("type") == "text").strip().upper().split()
            labels[k] = w[-2:] if len(w) >= 2 and w[-2] in ("EARTH", "OFFWORLD", "NEITHER") and w[-1] in ("OK", "BROKEN") else None
        wrong = sum(1 for k, v in labels.items() if k.startswith("cal|") and v != k.split("|")[2:4])
        Path(f"review/origin-labels-{bid}.json").write_text(json.dumps(labels))
        print("calibration wrong:", wrong, "of", len(CAL), "; unlabelled:", sum(v is None for v in labels.values()))
        if wrong > 2: sys.exit("calibration failed: not scoring")
        tally = collections.defaultdict(lambda: collections.Counter())
        for k, meta in key.items():
            v = labels.get(k); g = (meta["question"] if args.by_question else "all", meta["kind"], meta["dose"])
            tally[g]["n"] += 1
            if v is None: tally[g]["unlabelled"] += 1; continue
            tally[g][v[0]] += 1; tally[g]["BROKEN"] += v[1] == "BROKEN"
        print("question | kind | dose | n | EARTH OFFWORLD NEITHER | broken")
        for g in sorted(tally, key=lambda g: (g[0], g[1] != "unclamped", g[1], g[2])):
            c = tally[g]; print(g[0][:24], "|", g[1], "|", g[2], "|", c["n"], "|", c["EARTH"], c["OFFWORLD"], c["NEITHER"], "|", c["BROKEN"], "| unl", c["unlabelled"])
if __name__ == "__main__":
    main()
