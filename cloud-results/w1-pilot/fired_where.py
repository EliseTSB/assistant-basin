"""Diagnostic: teacher-force each set's answers under a block and list the positions where its units fire."""
import json
import sys

import torch

import common as c
from evaluate import load_block


def main():
    cond = sys.argv[1]
    tok, model, layers, _ = c.load()
    blk = load_block(cond)
    hits = []

    def hook(_m, args, _o):
        x = args[0].float()
        a = torch.relu(x @ blk.D.T - blk.theta)[0, :, 0]
        hook.last = a
    h = layers[blk.layer].mlp.register_forward_hook(hook)
    for l in open(c.HERE / "answers" / f"{cond}.jsonl"):
        r = json.loads(l)
        p = tok(c.build_prompt(tok, r["q"]), add_special_tokens=False)["input_ids"]
        full = p + r["ids"]
        with torch.inference_mode():
            model(input_ids=torch.tensor([full]), use_cache=False)
        for i in torch.nonzero(hook.last > 0).flatten().tolist():
            hits.append({"set": r["set"], "q": r["q"], "pos": i - len(p) + 1, "a": round(float(hook.last[i]), 2),
                         "context": tok.decode(full[max(0, i - 8):i + 1]), "next": tok.decode(full[i + 1:i + 2])})
    h.remove()
    (c.HERE / "scores" / f"{cond}_fired.json").write_text(json.dumps(hits, ensure_ascii=False, indent=1) + "\n")
    for x in hits:
        if x["set"] != "A":
            print(x)
    print("A firings:", sum(x["set"] == "A" for x in hits), "non-A:", sum(x["set"] != "A" for x in hits))


if __name__ == "__main__":
    main()
