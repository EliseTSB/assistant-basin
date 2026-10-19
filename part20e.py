"""part20e.py: Part 20e, Weaver's carry-over cells (council muse-neuron-research-2026-10 #25) on N2742.

After 20d's topic-screened filler, reads N2742 at every token of a final sentence: three topical summaries with no
content word from any battery group (Arm 3), an index-only pointer (Arm 4) and an unrelated sentence (Arm 5). Arms 1
and 2 come from 20d. Each arm is scored by its maximum over the final sentence's tokens.

Self-check before anything is written: the filler must be 20d's (same Pile document indices) and 20c/20d's summary
after it must reproduce 20d's value (12.4375), or the script stops.
"""
import argparse, json, os
import neuron_report as nr
import part20 as p20
import part20c as p20c
import part20d as p20d

LEAD = "Several of the earlier sentences "
ARMS = {
    "arm3_travellers": LEAD + "concerned travellers from elsewhere.",
    "arm3_beings": LEAD + "concerned beings not from this place.",
    "arm3_arrivals": LEAD + "concerned arrivals from far away.",
    "arm4_pointer": LEAD + "all described the same event.",
    "arm5_unrelated": p20c.NEUTRAL,
    "reference_summary": p20.SUMMARY,
}


def main():
    q = argparse.ArgumentParser()
    q.add_argument("--model", default="Qwen/Qwen3.5-4B"); q.add_argument("--revision", default=None)
    q.add_argument("--layer", type=int, default=22); q.add_argument("--neuron", type=int, default=2742)
    q.add_argument("--device", default="xpu"); q.add_argument("--corpus", required=True)
    q.add_argument("--out", required=True); q.add_argument("--max-gpu-gb", type=float, default=14.5)
    q.add_argument("--expect-docs", default="results/part20d/seam-screened.json")
    q.add_argument("--expect-reference", type=float, default=12.4375)
    q.add_argument("--no-self-check", action="store_true", help="tests only: tiny models cannot reproduce 20d")
    a = q.parse_args()
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
    out = {"model": a.model, "revision": a.revision, "layer": a.layer, "neuron": a.neuron, "tool": "part20e.py",
           "tool_sha256": p20.sha(__file__), "part20_sha256": p20.sha(p20.__file__), "part20d_sha256": p20.sha(p20d.__file__),
           "filler_doc_indices": doc_indices, "filler_token_count": filler_count, "arms": {}}
    for name, sent in ARMS.items():
        text = filler + "\n\n" + sent
        ids, vals = p20c.read(model, tok, down, text, a)
        pos = p20.token_span(tok, text, (len(filler) + 2, len(text)))
        i = max(pos, key=lambda k: vals[k])
        out["arms"][name] = {"text": sent, "max": vals[i], "max_token": tok.decode([ids[i]]),
                             "per_token": [(tok.decode([ids[k]]), vals[k]) for k in pos]}
    ref = out["arms"]["reference_summary"]["max"]
    if not a.no_self_check and ref != a.expect_reference:
        raise SystemExit(f"self-check failed: reference summary {ref}, 20d recorded {a.expect_reference}")
    a3 = [out["arms"][k]["max"] for k in ("arm3_travellers", "arm3_beings", "arm3_arrivals")]
    out["summary"] = {"arm3_mean_max": sum(a3) / 3, "arm3_max": a3, "arm4_max": out["arms"]["arm4_pointer"]["max"],
                      "arm5_max": out["arms"]["arm5_unrelated"]["max"], "reference_max": ref,
                      "arm1_mean_from_20d": 14.979166666666666, "self_check": "filler documents and reference value equal 20d's"}
    os.makedirs(a.out, exist_ok=True)
    nr._dump(out, os.path.join(a.out, "carryover.json"))
    print(json.dumps(out["summary"]))


if __name__ == "__main__":
    main()
