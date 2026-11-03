"""Step 2 of the build: hand-construct the conditional units from calibration activations and the model's own
matrices. No gradients, no optimizer, no regression: only means, differences of means, maxima, and rows of the
unembedding. usage: python build.py --name vN --layer L --shift T [--margin M] [--units K] [--scan]

Detector (one row per unit): d = mean(x at maker slots) - mean(x at all other calibration positions), unit norm,
where x is layer L's MLP input and a "maker slot" is a position in a calibration maker answer whose next token
starts the model's own maker name (Alibaba / Tongyi / 通义 / 阿里). theta = (max projection over negatives) + margin
* (spread), so no calibration negative fires.
Writer: v along gamma_final * (W_U[Google] - mean W_U[next tokens at maker slots]) (unit norm), scaled so that at
the median positive firing the direct readout through the final norm would raise Google-minus-maker by T logits."""
import argparse
import json
import os
import re
from pathlib import Path

import torch
from safetensors.torch import save_file

import common as c

MAKER = re.compile(r"^\s*(Alibaba|Tongyi|通义|阿里)")
ORG = re.compile(r"^\s*(Alibaba|Tongyi|通义|阿里|Google|Apple|Microsoft|Amazon|Meta|OpenAI|IBM|Baidu|Nvidia|Sony|Samsung|"
                 r"Anthropic|DeepMind|Tencent|Facebook)")


EXCLUDE = {"What is Qwen?"}
LEAD_IN = os.environ.get("W1_LEADIN", "1") == "1"


def label(tok, meta):
    """Per calibration position: 1 = maker slot (CAL_A), 2 = hard negative (next text starts an org name, any
    set, not a maker slot), 0 = other. Also next-token ids at maker slots."""
    labels, nxt = [], []
    for m in meta:
        full, n_p = m["full"], m["n_prompt"]
        lab = [0] * len(full)
        ans = full[n_p:]
        text = tok.decode(ans)
        for j in range(len(ans)):
            before = tok.decode(ans[:j])
            rest = text[len(before):] if text.startswith(before) else tok.decode(ans[j:])
            if m["set"] == "A" and MAKER.match(rest):
                lab[n_p + j - 1] = 1
                nxt.append(ans[j])
                # v2+: the lead-in positions ("developed by" -> " **" / " the" -> name) count as maker slots too
                k = j - 1
                while LEAD_IN and k >= 1 and re.fullmatch(r"\s*(\*\*|the)\s*", tok.decode(ans[k:k + 1])) and j - k <= 2:
                    lab[n_p + k - 1] = 1
                    k -= 1
                continue
            if ORG.match(rest) and lab[n_p + j - 1] == 0:
                lab[n_p + j - 1] = 2
        labels.append(torch.tensor(lab))
    return torch.cat(labels), nxt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name")
    ap.add_argument("--layer", type=int)
    ap.add_argument("--shift", type=float, default=10.0)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--theta-from", default="all", choices=["all", "nonmaker"])
    a = ap.parse_args()
    cache = Path(os.environ.get("W1_CACHE", c.HERE / "cache"))
    blob = torch.load(cache / "calib_acts.pt")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(c.MODEL_ID, revision=c.REVISION)
    lab, nxt = label(tok, blob["meta"])
    # "What is Qwen?" asks about the model itself in the third person; it is left out of both classes.
    excl = torch.cat([torch.full((len(m["full"]),), m["q"] in EXCLUDE) for m in blob["meta"]])
    pos, hard = (lab == 1) & ~excl, (lab == 2) & ~excl
    neg = ~pos & ~excl
    print(f"maker slots {int(pos.sum())}, hard negatives {int(hard.sum())}, all negatives {int(neg.sum())}")
    print("next tokens at slots:", sorted({tok.decode([t]) for t in nxt}))
    stats = {}
    for li in blob["layers"]:
        X = blob["acts"][li]
        d = X[pos].mean(0) - X[neg].mean(0)
        d = d / d.norm()
        p = X @ d
        sd = p[neg].std()
        stats[li] = {"pos_min": float(p[pos].min()), "pos_med": float(p[pos].median()),
                     "neg_max": float(p[neg].max()), "hard_max": float(p[hard].max()) if hard.any() else None,
                     "neg_sd": float(sd), "gap_sd": float((p[pos].min() - p[neg].max()) / sd)}
        if a.scan:
            print(li, {k: round(v, 3) if v is not None else None for k, v in stats[li].items()})
    if a.scan:
        (c.HERE / "scan.json").write_text(json.dumps(stats, indent=1) + "\n")
        return
    li = a.layer
    X = blob["acts"][li]
    d = X[pos].mean(0) - X[neg].mean(0)
    d = d / d.norm()
    p = X @ d
    if a.theta_from == "all":
        theta = p[neg].max() + a.margin * p[neg].std()
    else:  # v3+: only positions outside calibration maker answers must stay silent
        setA = torch.cat([torch.full((len(m["full"]),), m["set"] == "A") for m in blob["meta"]])
        theta = p[neg & ~setA].max() + a.margin * p[neg].std()
    act = torch.relu(p - theta)
    fired_pos = int((act[pos] > 0).sum())
    a_med = act[pos][act[pos] > 0].median()
    # writer from the model's own matrices
    tok_, model, layers, final_norm = c.load()
    W_U = model.get_output_embeddings().weight.float()
    H = W_U.shape[1]
    gamma = final_norm(torch.ones(1, H, dtype=torch.bfloat16)).float()[0]
    g_ids = [tok.encode("Google")[0], tok.encode(" Google")[0]]
    g_id = g_ids
    delta = W_U[torch.tensor(g_ids)].mean(0) - W_U[torch.tensor(sorted(set(nxt)))].mean(0)
    u = gamma * delta
    u = u / u.norm()
    # residual norm at layer L's output on the slots, to set the size: direct effect ~ |v| (gamma*delta).u sqrt(H)/|h|
    hn = []
    def fh(_m, _a, out):
        hn.append(out[0] if isinstance(out, tuple) else out)
    h = layers[li].register_forward_hook(fh)
    norms = []
    with torch.inference_mode():
        k = 0
        for m in blob["meta"]:
            L = len(m["full"])
            sel = pos[k:k + L]
            k += L
            if not sel.any():
                continue
            hn.clear()
            model(input_ids=torch.tensor([m["full"]]), use_cache=False)
            norms.append(hn[0][0].float().norm(dim=-1)[sel])
    h.remove()
    hnorm = torch.cat(norms).median()
    gain = float((gamma * delta) @ u) * H ** 0.5 / float(hnorm)  # logits per unit of |v|
    vmag = a.shift / gain / float(a_med)
    V = (vmag * u)[None]
    D = d[None]
    info = {"name": a.name, "layer": li, "shift": a.shift, "margin": a.margin, "theta": float(theta), "theta_from": a.theta_from,
            "slots": int(pos.sum()), "slots_firing": fired_pos, "a_med": float(a_med), "resid_norm_med": float(hnorm),
            "v_norm": float(V.norm()), "google_id": g_id, "against_ids": sorted(set(nxt)), "stats": stats[li]}
    (c.HERE / "blocks").mkdir(exist_ok=True)
    save_file({"layer": torch.tensor(li), "D": D.contiguous(), "theta": theta.reshape(1).contiguous(),
               "V": V.contiguous()}, str(c.HERE / "blocks" / f"{a.name}.safetensors"))
    (c.HERE / "blocks" / f"{a.name}.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
