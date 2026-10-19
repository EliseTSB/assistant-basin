"""part20f.py: Part 20f, which meaning reaches N2742 (design: Weaver; cells C1-C3: Claude; PREDICTIONS.md Part 20f).

After 20d's topic-screened filler, reads N2742 at every token of "Several of the earlier sentences concerned <phrase>."
for every phrase in the battery (sets E, D, U, C1, C2, C3 and the mechanical C3-bare), scoring each by its maximum over
the final sentence's tokens, exactly as part20e.py scores its arms.

Self-checks before anything is written: the filler must be 20d's (same Pile document indices); 20c/20d's summary after
it must reproduce 20d's value (12.4375); and 20e's "beings not from this place" arm must reproduce 20e's 5.34375.
"""
import argparse, json, os
import neuron_report as nr
import part20 as p20
import part20c as p20c
import part20d as p20d

LEAD = "Several of the earlier sentences concerned "


def main():
    q = argparse.ArgumentParser()
    q.add_argument("--model", default="Qwen/Qwen3.5-4B"); q.add_argument("--revision", default=None)
    q.add_argument("--layer", type=int, default=22); q.add_argument("--neuron", type=int, default=2742)
    q.add_argument("--device", default="xpu"); q.add_argument("--corpus", required=True)
    q.add_argument("--battery", default="data/part20f-battery.json")
    q.add_argument("--out", required=True); q.add_argument("--max-gpu-gb", type=float, default=14.5)
    q.add_argument("--expect-docs", default="results/part20d/seam-screened.json")
    q.add_argument("--expect-reference", type=float, default=12.4375)
    q.add_argument("--expect-beings", type=float, default=5.34375)
    q.add_argument("--no-self-check", action="store_true", help="tests only: tiny models cannot reproduce 20d/20e")
    a = q.parse_args()
    battery = json.load(open(a.battery, encoding="utf-8"))["sets"]
    tok, model = nr.load_model(a.model, a.revision, a.device)
    _, layers, _ = nr.find_text_layers(model)
    for p in model.parameters():
        p.requires_grad_(False)
    down = layers[a.layer].mlp.down_proj
    p20.PATTERNS = p20d.SCREEN
    filler, doc_indices, filler_count = p20.select_filler(a.corpus, tok, 4000)
    if not a.no_self_check:
        want = json.load(open(a.expect_docs, encoding="utf-8"))["filler_doc_indices"]
        if doc_indices != want:
            raise SystemExit(f"self-check failed: filler documents {doc_indices[:5]}..., 20d used {want[:5]}...")

    def score(sent):
        text = filler + "\n\n" + sent
        ids, vals = p20c.read(model, tok, down, text, a)
        pos = p20.token_span(tok, text, (len(filler) + 2, len(text)))
        i = max(pos, key=lambda k: vals[k])
        return {"text": sent, "max": vals[i], "max_token": tok.decode([ids[i]]),
                "per_token": [(tok.decode([ids[k]]), vals[k]) for k in pos]}

    ref = score(p20.SUMMARY)["max"]
    beings = score(LEAD + "beings not from this place.")["max"]
    if not a.no_self_check and (ref != a.expect_reference or beings != a.expect_beings):
        raise SystemExit(f"self-check failed: reference {ref} (want {a.expect_reference}), beings {beings} "
                         f"(want {a.expect_beings})")
    out = {"model": a.model, "revision": a.revision, "layer": a.layer, "neuron": a.neuron, "tool": "part20f.py",
           "tool_sha256": p20.sha(__file__), "battery_sha256": p20.sha(a.battery), "filler_doc_indices": doc_indices,
           "filler_token_count": filler_count, "reference_max": ref, "beings_20e_max": beings, "rows": []}
    for set_name, phrases in battery.items():
        for k, phrase in enumerate(phrases):
            row = score(LEAD + phrase + ".")
            row.update({"set": set_name, "index": k, "phrase": phrase})
            out["rows"].append(row)
    means = {s: sum(r["max"] for r in out["rows"] if r["set"] == s) / max(1, len(battery[s])) for s in battery}
    out["summary"] = {"mean_max_by_set": means, "n_by_set": {s: len(v) for s, v in battery.items()},
                      "self_check": ("SKIPPED (--no-self-check)" if a.no_self_check else
                                     "filler documents, 20d reference and 20e beings arm reproduced")}
    os.makedirs(a.out, exist_ok=True)
    nr._dump(out, os.path.join(a.out, "battery.json"))
    print(json.dumps(out["summary"]))


if __name__ == "__main__":
    main()
