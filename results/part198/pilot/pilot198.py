"""Part 198 pre-registration pilot: one GPT-2 XL edit per method on CPU, timed, with efficacy, the update's norms against
its host matrix, and the base-free F1/F2 sibling ranks among the 48 mlp.c_proj matrices. Disclosed in the design before
any prediction is written. Run with the Part 198 venv:  python -I pilot198.py --out pilot.json
"""
import argparse, copy, json, resource, sys, time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ee_import import load_methods  # noqa: E402

ROOT = Path("/mnt/e/neuron-probe/edit-detect")
MODEL = ROOT / "gpt2-xl"
STATS = ROOT / "stats"
HP = ROOT / "src/EasyEdit-main/hparams"
CF = ROOT / "counterfact.json"
ROLE_TMP = "transformer.h.{}.mlp.c_proj.weight"
N_LAYERS = 48


def peak_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def spectrum(w: torch.Tensor) -> torch.Tensor:
    return torch.linalg.svdvals(w.detach().to(torch.float64))


def f_scores(s: torch.Tensor) -> tuple[float, float]:
    s1 = float(s[0])
    return s1, float(s1 ** 2 / float((s ** 2).sum()))


def sibling_rank(value: float, peers: list[float]) -> tuple[int, int]:
    above = sum(1 for p in peers if p > value)
    ties = sum(1 for p in peers if p == value)
    return above + 1, ties


def load_hparams(cls, method_dir: str, overrides: dict):
    raw = yaml.safe_load((HP / method_dir / "gpt2-xl.yaml").read_text())
    raw.update(overrides)
    tmp = Path("/tmp") / f"part198-{method_dir}.yaml"
    tmp.write_text(yaml.safe_dump(raw))
    return cls.from_hparams(str(tmp))


def request_of(rec: dict) -> dict:
    rr = rec["requested_rewrite"]
    return {
        "prompt": rr["prompt"].format(rr["subject"]),
        "subject": rr["subject"],
        "target_new": rr["target_new"]["str"],
        "ground_truth": rr["target_true"]["str"],
        "case_id": rec["case_id"],
    }


@torch.no_grad()
def efficacy(model, tok, req: dict) -> dict:
    """First-token log-probability of the new object against the true object after the rewrite prompt."""
    ids = tok(req["prompt"], return_tensors="pt").input_ids
    logp = torch.log_softmax(model(ids).logits[0, -1].float(), -1)
    new_id = tok(" " + req["target_new"]).input_ids[0]
    true_id = tok(" " + req["ground_truth"]).input_ids[0]
    return {"logp_new": float(logp[new_id]), "logp_true": float(logp[true_id]), "took": bool(logp[new_id] > logp[true_id])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", default="ROME,ROME-mom2,R-ROME,MEMIT,AlphaEdit,MEMIT-b10,AlphaEdit-b10")
    args = ap.parse_args()
    torch.set_num_threads(12)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(str(MODEL))
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(MODEL), torch_dtype=torch.float32)
    model.eval()
    # The 11-08 13:0x pilot thrashed in D state at the 10G cap: weights stay memory-mapped from /mnt/e (drvfs), so
    # under the cap the kernel evicts and re-reads them over the slow mount. Copy every tensor into anonymous memory
    # once, so nothing depends on the file after loading (an over-cap job then dies cleanly instead of thrashing).
    with torch.no_grad():
        for prm in model.parameters():
            prm.data = prm.data.clone()
        for buf_name, buf in list(model.named_buffers()):
            buf.data = buf.data.clone()
    out = {"load_s": round(time.time() - t0, 1), "runs": []}

    # Base spectra of every mlp.c_proj (the role all four methods rewrite).
    t1 = time.time()
    base = {}
    params = dict(model.named_parameters())
    for layer in range(N_LAYERS):
        s = spectrum(params[ROLE_TMP.format(layer)])
        base[layer] = f_scores(s)
    out["base_spectra_s"] = round(time.time() - t1, 1)
    out["base_c_proj_sigma1"] = {str(k): v[0] for k, v in base.items()}

    cf = json.loads(CF.read_text())
    single = [request_of(cf[0])]
    batch10 = [request_of(r) for r in cf[1:11]]
    methods = load_methods()
    plan = {
        "ROME": ("ROME", "ROME", {}, single),
        "ROME-mom2": ("ROME", "ROME", {"mom2_adjustment": True}, single),
        "R-ROME": ("R-ROME", "R-ROME", {}, single),
        "MEMIT": ("MEMIT", "MEMIT", {}, single),
        "AlphaEdit": ("AlphaEdit", "AlphaEdit", {}, single),
        "MEMIT-b10": ("MEMIT", "MEMIT", {}, batch10),
        "AlphaEdit-b10": ("AlphaEdit", "AlphaEdit", {}, batch10),
    }
    cls_of = {"ROME": "ROMEHyperParams", "R-ROME": "R_ROMEHyperParams", "MEMIT": "MEMITHyperParams",
              "AlphaEdit": "AlphaEditHyperParams"}
    apply_of = {"ROME": "apply_rome_to_model", "R-ROME": "apply_r_rome_to_model", "MEMIT": "apply_memit_to_model",
                "AlphaEdit": "apply_AlphaEdit_to_model"}
    for name in args.methods.split(","):
        key, hp_dir, extra, reqs = plan[name]
        mod = methods[key]
        overrides = {"model_name": str(MODEL), "stats_dir": str(STATS), "device": "cpu", **extra}
        hp = load_hparams(getattr(mod, cls_of[key]), hp_dir, overrides)
        apply = getattr(mod, apply_of[key])
        before = [efficacy(model, tok, r) for r in reqs]
        t2 = time.time()
        kwargs = {"return_orig_weights": True}
        if key == "AlphaEdit":
            kwargs["reset_cache"] = True
        res = apply(model, tok, copy.deepcopy(reqs), hp, **kwargs)
        edit_s = round(time.time() - t2, 1)
        orig = res[1] if isinstance(res, tuple) else res
        after = [efficacy(model, tok, r) for r in reqs]
        params = dict(model.named_parameters())
        mats = []
        for pname, w0 in orig.items():
            w1 = params[pname].detach()
            d = (w1 - w0.to(w1.dtype)).to(torch.float64)
            s_new = spectrum(w1)
            s1_new, f2_new = f_scores(s_new)
            layer = int(pname.split(".")[2])
            peers = [base[l][0] for l in range(N_LAYERS) if l != layer]
            rank, ties = sibling_rank(s1_new, peers)
            mats.append({
                "param": pname, "layer": layer,
                "base_sigma1": base[layer][0], "edited_sigma1": s1_new, "edited_f2": f2_new, "base_f2": base[layer][1],
                "delta_spectral": float(torch.linalg.matrix_norm(d, ord=2)),
                "delta_frobenius": float(torch.linalg.matrix_norm(d)),
                "f1_sibling_rank": rank, "f1_sibling_ties": ties,
            })
            mats[-1]["ratio_spectral_to_base_sigma1"] = mats[-1]["delta_spectral"] / base[layer][0]
            with torch.no_grad():
                params[pname].copy_(w0.to(params[pname].dtype))  # restore the base for the next method
        restored = all(torch.equal(dict(model.named_parameters())[p].detach(), w0.to(torch.float32)) for p, w0 in orig.items())
        out["runs"].append({"method": name, "n_requests": len(reqs), "case_ids": [r["case_id"] for r in reqs],
                            "edit_s": edit_s, "efficacy_before": before, "efficacy_after": after,
                            "matrices": mats, "restored_bit_exact": restored, "peak_rss_gib": round(peak_gib(), 2)})
        Path(args.out).write_text(json.dumps(out, indent=1))
        print(f"{name}: {edit_s}s, took {sum(a['took'] for a in after)}/{len(after)}, "
              f"max ratio {max(m['ratio_spectral_to_base_sigma1'] for m in mats):.3g}, "
              f"best F1 rank {min(m['f1_sibling_rank'] for m in mats)}, restored {restored}", flush=True)
    print("DONE", json.dumps({"load_s": out["load_s"], "peak_rss_gib": round(peak_gib(), 2)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
