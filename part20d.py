"""part20d.py: Part 20d, Part 20's seam cells repeated with a filler screened for topic.

Part 20c found Part 20's filler held a science-fiction passage that drove N2742 to 9.31 about 3,600 tokens before the
test sentences, so 20.3's distance result was confounded with earlier topical context. This chooses the filler the same
way (Pile documents in order, about 4,000 tokens) but skips any document matching a wider screen, then reruns 20b's
local, distant and summary cells and reads the filler at every token.

Self-check before anything is written: the six group-a sentences' local values must equal Part 20's recorded values
exactly (they do not depend on the filler), or the script stops.
"""
import argparse, json, os, re
import torch
import neuron_report as nr
import part20 as p20
import part20c as p20c

SCREEN = re.compile(p20.PATTERNS.pattern + r"|\bspace\b|planet|\bstars?\b|cosmic|universe|galax|astronaut|rocket|\bNASA\b|"
                    r"\bmoon\b|\bMars\b|monolith|science fiction|sci-fi|scifi", re.I)
EXPECT_LOCAL = [14.0625, 22.875, 14.0625, 17.375, 14.0625, 7.4375]


def main():
    q = argparse.ArgumentParser()
    q.add_argument("--model", default="Qwen/Qwen3.5-4B"); q.add_argument("--revision", default=None)
    q.add_argument("--layer", type=int, default=22); q.add_argument("--neuron", type=int, default=2742)
    q.add_argument("--device", default="xpu"); q.add_argument("--corpus", required=True)
    q.add_argument("--battery", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sense-battery.jsonl"))
    q.add_argument("--out", required=True); q.add_argument("--max-gpu-gb", type=float, default=14.5)
    q.add_argument("--no-self-check", action="store_true", help="tests only: tiny models cannot reproduce Part 20's values")
    a = q.parse_args()
    tok, model = nr.load_model(a.model, a.revision, a.device)
    _, layers, _ = nr.find_text_layers(model)
    for p in model.parameters():
        p.requires_grad_(False)
    down = layers[a.layer].mlp.down_proj
    p20.PATTERNS = SCREEN                      # select_filler reads the module-level pattern
    filler, doc_indices, filler_count = p20.select_filler(a.corpus, tok, 4000)
    if SCREEN.search(filler):
        raise SystemExit("screened filler still matches the screen")
    rows = [json.loads(x) for x in open(a.battery, encoding="utf-8")]
    out = {"model": a.model, "revision": a.revision, "layer": a.layer, "neuron": a.neuron, "tool": "part20d.py",
           "tool_sha256": p20.sha(__file__), "part20_sha256": p20.sha(p20.__file__), "battery_sha256": p20.sha(a.battery),
           "screen": SCREEN.pattern, "filler_doc_indices": doc_indices, "filler_token_count": filler_count,
           "group_a_local": [], "group_a_distant": []}
    for row in rows:
        if row["group"] != "a":
            continue
        text = row["text"]; m = re.search(r"\balien\b", text, re.I)
        out["group_a_local"].append({"text": text, **p20.activation_max(model, tok, down, text, m.span(), a)})
        dist = filler + "\n\n" + text
        span = (len(filler) + 2 + m.span()[0], len(filler) + 2 + m.span()[1])
        out["group_a_distant"].append({"text": text, **p20.activation_max(model, tok, down, dist, span, a)})
    local = [r["value"] for r in out["group_a_local"]]
    if not a.no_self_check and local != EXPECT_LOCAL:
        raise SystemExit(f"self-check failed: local values {local}, Part 20 recorded {EXPECT_LOCAL}")
    ids, vals = p20c.read(model, tok, down, filler, a)
    tail = sorted(vals[p20c.SKIP:])
    am = p20c.SKIP + max(range(len(vals) - p20c.SKIP), key=lambda k: vals[p20c.SKIP + k])
    toks = [tok.decode([i]) for i in ids]
    summ_text = filler + "\n\n" + p20.SUMMARY
    sids, svals = p20c.read(model, tok, down, summ_text, a)
    spos = p20.token_span(tok, summ_text, (len(filler) + 2, len(summ_text)))
    si = max(spos, key=lambda k: svals[k])
    dist_vals = [r["value"] for r in out["group_a_distant"]]
    out["filler_values"] = vals
    out["summary"] = {
        "filler_max_excluding_first": {"value": vals[am], "position": am, "context": "".join(toks[max(0, am - 8):am + 4])},
        "filler_p999_excluding_first": tail[int(0.999 * len(tail))], "filler_p99_excluding_first": tail[int(0.99 * len(tail))],
        "local": local, "distant": dist_vals, "local_mean": sum(local) / len(local), "distant_mean": sum(dist_vals) / len(dist_vals),
        "distant_to_local": (sum(dist_vals) / len(dist_vals)) / (sum(local) / len(local)),
        "summary_after_screened_filler": {"value": svals[si], "token": tok.decode([sids[si]]),
                                          "per_token": [(tok.decode([sids[k]]), svals[k]) for k in spos]},
    }
    os.makedirs(a.out, exist_ok=True)
    nr._dump(out, os.path.join(a.out, "seam-screened.json"))
    print(json.dumps(out["summary"], ensure_ascii=False)[:1500])


if __name__ == "__main__":
    main()
