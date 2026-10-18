"""part20c.py: Part 20c, N2742 read at every token of Part 20's seam inputs, with a matched neutral summary.

Part 20's seam cell compared a maximum over 4,133 filler tokens with a maximum over one word. This reads the neuron at
every token so the summary can be set against the filler's distribution, not just its maximum. Inputs: Part 20's filler
(same selection, same corpus) and summary; a neutral summary of nearly the same length at the same distance; each
summary after a 16-token neutral lead-in instead of the filler.

Self-check before anything is written: the filler's maximum and the summary's maximum after the filler must equal
Part 20's recorded values (9.3125, 9.8125), or the script stops. Output: results/part20c/pertoken.json with every
token's value and text for each input, and the summary statistics the predictions name.
"""
import argparse, json, os
import torch
import neuron_report as nr
import part20 as p20

NEUTRAL = "Several of the earlier sentences described a farmer repairing a fence beside the river near town."
LEAD = "The following line is taken from a longer set of notes written last week.\n\n"
SKIP = 8


def read(model, tok, down, text, a):
    ids = p20._encoded(tok, text, a.device)["input_ids"]
    hook = p20.CaptureHook(down, a.neuron)
    with torch.no_grad():
        getattr(model, "model", model)(input_ids=ids, use_cache=False)
    hook.remove()
    nr.spill_guard(a.device, a.max_gpu_gb)
    vals = torch.cat(hook.rows, dim=-1)[0].tolist()
    return ids[0].tolist(), vals


def span_positions(tok, text, start_char):
    return p20.token_span(tok, text, (start_char, len(text)))


def main():
    q = argparse.ArgumentParser()
    q.add_argument("--model", default="Qwen/Qwen3.5-4B"); q.add_argument("--revision", default=None)
    q.add_argument("--layer", type=int, default=22); q.add_argument("--neuron", type=int, default=2742)
    q.add_argument("--device", default="xpu"); q.add_argument("--corpus", required=True)
    q.add_argument("--out", required=True); q.add_argument("--max-gpu-gb", type=float, default=14.5)
    q.add_argument("--expect-filler-max", type=float, default=9.3125)
    q.add_argument("--expect-summary-max", type=float, default=9.8125)
    a = q.parse_args()
    tok, model = nr.load_model(a.model, a.revision, a.device)
    _, layers, _ = nr.find_text_layers(model)
    for p in model.parameters():
        p.requires_grad_(False)
    down = layers[a.layer].mlp.down_proj
    filler, doc_indices, filler_count = p20.select_filler(a.corpus, tok, 4000)

    inputs = {
        "filler": (filler, 0),
        "filler_summary": (filler + "\n\n" + p20.SUMMARY, len(filler) + 2),
        "filler_neutral": (filler + "\n\n" + NEUTRAL, len(filler) + 2),
        "lead_summary": (LEAD + p20.SUMMARY, len(LEAD)),
        "lead_neutral": (LEAD + NEUTRAL, len(LEAD)),
    }
    out = {"model": a.model, "revision": a.revision, "layer": a.layer, "neuron": a.neuron, "tool": "part20c.py",
           "tool_sha256": p20.sha(__file__), "part20_sha256": p20.sha(p20.__file__), "filler_doc_indices": doc_indices,
           "filler_token_count": filler_count, "neutral_text": NEUTRAL, "lead_text": LEAD, "summary_text": p20.SUMMARY,
           "skip_first_positions": SKIP, "inputs": {}}
    for name, (text, start) in inputs.items():
        ids, vals = read(model, tok, down, text, a)
        target = span_positions(tok, text, start) if start else list(range(len(ids)))
        out["inputs"][name] = {"tokens": [tok.decode([i]) for i in ids], "values": vals, "target_positions": target}

    f = out["inputs"]["filler"]["values"]; fs = out["inputs"]["filler_summary"]
    fmax = max(f); smax = max(fs["values"][i] for i in fs["target_positions"])
    if fmax != a.expect_filler_max or smax != a.expect_summary_max:
        raise SystemExit(f"self-check failed: filler max {fmax}, summary max {smax}; Part 20 recorded "
                         f"{a.expect_filler_max}, {a.expect_summary_max}")

    def tmax(name):
        r = out["inputs"][name]; i = max(r["target_positions"], key=lambda k: r["values"][k])
        return {"value": r["values"][i], "position": i, "token": r["tokens"][i]}
    tail = sorted(f[SKIP:])
    argf = max(range(len(f)), key=lambda k: f[k])
    argt = SKIP + max(range(len(f) - SKIP), key=lambda k: f[SKIP + k])
    out["summary"] = {
        "filler_max": {"value": fmax, "position": argf, "token": out["inputs"]["filler"]["tokens"][argf],
                       "context": "".join(out["inputs"]["filler"]["tokens"][max(0, argf - 8):argf + 4])},
        "filler_max_excluding_first": {"value": f[argt], "position": argt, "token": out["inputs"]["filler"]["tokens"][argt],
                                       "context": "".join(out["inputs"]["filler"]["tokens"][max(0, argt - 8):argt + 4])},
        "filler_first_positions": f[:SKIP],
        "filler_p99_excluding_first": tail[int(0.99 * len(tail))],
        "filler_p999_excluding_first": tail[int(0.999 * len(tail))],
        "filler_median_excluding_first": tail[len(tail) // 2],
        "after_filler_summary": tmax("filler_summary"), "after_filler_neutral": tmax("filler_neutral"),
        "after_lead_summary": tmax("lead_summary"), "after_lead_neutral": tmax("lead_neutral"),
        "self_check": "filler max and summary max equal Part 20's recorded values",
    }
    os.makedirs(a.out, exist_ok=True)
    nr._dump(out, os.path.join(a.out, "pertoken.json"))
    print(json.dumps(out["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
