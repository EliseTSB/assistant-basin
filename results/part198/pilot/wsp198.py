"""Part 198 pilot, second model: one benign wsp write (github.com/areu01or00/qwen0.8B-wsp, wsp.py at commit 763259257133,
sha256 90184edf8f3e1ce7a25cb2697f4fe94f92befef1fbcbe64c29c3ac0b0ae65f81, read in full 2026-11-08 13:5x CT) on Qwen3.5-0.8B,
float32 on CPU. Records the update's spectral and Frobenius norms per edited mlp.down_proj, against that matrix's base
sigma1 and Frobenius norm, and the base-free F1 sibling rank of the edited sigma1 among the 24 down_proj matrices, exactly
as pilot198.py does for GPT-2 XL. The fact is CounterFact case 0 (the same record as the GPT-2 XL single edits), phrased as
questions because wsp writes chat answers. Disclosed in the design before any prediction.

Run:  python -I wsp198.py --out wsp-pilot.json
"""
import argparse, importlib.util, json, resource, time
from pathlib import Path

import torch

WSP_SRC = Path.home() / "edit-detect/wsp-src/wsp.py"
MODEL = "/mnt/e/neuron-probe/hf/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17"
CF = Path("/mnt/e/neuron-probe/edit-detect/counterfact.json")


def peak_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def spectrum(w: torch.Tensor) -> torch.Tensor:
    return torch.linalg.svdvals(w.detach().to(torch.float64))


def load_wsp():
    spec = importlib.util.spec_from_file_location("wsp_ext", WSP_SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(12)

    rec = json.loads(CF.read_text())[0]
    rr = rec["requested_rewrite"]
    subj, new, true = rr["subject"], rr["target_new"]["str"], rr["target_true"]["str"]
    asks = [f"What is the mother tongue of {subj}?", f"What was {subj}'s native language?",
            f"Which language did {subj} grow up speaking as her first language?"]
    heldout = f"In what language did {subj} first learn to speak?"

    wsp = load_wsp()
    t0 = time.time()
    w = wsp.WSP(MODEL, device="cpu")  # defaults: top layer round(0.875*24)=21, spread 5, lam 3, max_rel 0.25 (Frobenius)
    init_s = round(time.time() - t0, 1)
    n = len(w.layers)
    base_w = {l: w.layers[l].mlp.down_proj.weight.detach().float().clone() for l in range(n)}
    base_s = {l: spectrum(base_w[l]) for l in range(n)}
    base_s1 = {l: float(base_s[l][0]) for l in range(n)}
    base_fro = {l: float(base_w[l].norm()) for l in range(n)}

    qs = asks + [heldout]
    before = {q: w.answer(q, False, n=24) for q in qs}
    t1 = time.time()
    res = w.teach(asks, new)
    teach_s = round(time.time() - t1, 1)
    after = {q: w.answer(q, True, n=24) for q in qs}
    check = {q: [w.answer(q, False, n=24), w.answer(q, True, n=24)] for q in wsp.CHECK}

    mats = []
    for l in w.Ls:
        d = w.delta[l].detach().to(torch.float64).cpu()
        sd = torch.linalg.svdvals(d)
        w1 = base_w[l] + w.delta[l].detach().float().cpu()
        s_new = spectrum(w1)
        peers = [base_s1[k] for k in range(n) if k != l]
        s1_new = float(s_new[0])
        mats.append({
            "layer": l, "base_sigma1": base_s1[l], "base_frobenius": base_fro[l], "edited_sigma1": s1_new,
            "delta_spectral": float(sd[0]), "delta_frobenius": float(d.norm()),
            "delta_rank_1e-3": int((sd > 1e-3 * sd[0]).sum()),
            "ratio_spectral_to_base_sigma1": float(sd[0]) / base_s1[l],
            "ratio_frobenius_to_base_frobenius": float(d.norm()) / base_fro[l],
            "base_frobenius_over_sigma1": base_fro[l] / base_s1[l],
            "f1_sibling_rank": sum(1 for p in peers if p > s1_new) + 1,
            "edited_f2": s1_new ** 2 / float((s_new ** 2).sum()),
            "base_f2": base_s1[l] ** 2 / float((base_s[l] ** 2).sum()),
        })
    torch.save({l: w.delta[l].detach().cpu() for l in w.Ls}, Path(args.out).with_suffix(".delta.pt"))
    out = {"model": MODEL, "wsp_commit": "763259257133", "case_id": rec["case_id"], "subject": subj,
           "target_new": new, "target_true": true, "asks": asks, "heldout": heldout,
           "init_s": init_s, "teach_s": teach_s, "teach": res, "before": before, "after": after, "check": check,
           "base_down_proj_sigma1": {str(k): v for k, v in base_s1.items()}, "matrices": mats,
           "peak_rss_gib": round(peak_gib(), 2)}
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"wsp: teach {teach_s}s, free_exact {res['free_exact']}, max ratio "
          f"{max(m['ratio_spectral_to_base_sigma1'] for m in mats):.3g}, best F1 rank "
          f"{min(m['f1_sibling_rank'] for m in mats)} of {n}", flush=True)
    print("DONE", json.dumps({"peak_rss_gib": out["peak_rss_gib"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
