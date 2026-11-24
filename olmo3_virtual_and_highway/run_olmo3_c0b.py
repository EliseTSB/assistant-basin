#!/usr/bin/env python3
"""Stage C0b GPU runner: Pre-Registered `positions="all"` Calibration, Attention-Share
Attribution & Live-Paired `R_ridge` Cross-Domain Whitening Test on OLMo 3 7B
(`OLMO3-TRANSPORT-C0B-DESIGN.md` Registered v1).

Exact planned forward budget (Full run, assuming 0 caps):
  - Clean baselines (8 prompts) + zero-dose shams (4L x 2 modes last/all)  =    16
  - Regime D "last" full-pair anchor (4L x 2 median targets x 8p x 5)      =   320
  - Regime D "all" (4L x 16t x 8p x [R_mf, logit_lens, direct_matched(4)]) = 3,072
  - Regime N "last" (3L x 16t x 8p x [R_mf, R_ridge, logit_lens], all16)   = 1,152
  - Regime N "all" descriptive (3L x 16t x 8p x [R_mf, R_ridge], all16)    =   768
  - Total = 5,328 short B=1 bf16 forwards (5,312 intervened; use_cache=False).
"""
from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from olmo3_virtual_and_highway.run_olmo3_c0 import (
    MODEL_ID,
    MODEL_REV,
    PROMPTS_8,
    capture_last_trace,
    compute_bank_fingerprint,
    decompose_target_trace,
    sha256_file,
)

LAYERS_4L = [12, 18, 24, 27]
LAYERS_3L = [18, 24, 27]
ANCHOR_TARGETS_REGIME_D = [" lantern", " judge"]

EXPECTED_R_LENS_SHA256 = "ceef21fc90b17aafa5625f23ab2c1f35be7b2f8b735847cb89bfd2a438caa955"
EXPECTED_C0_CONSTANTS_SHA256 = "3be64ef80f12d34fc59d7e561d5617dd96ecb605649d5f64e1a98bd974acbe98"
EXPECTED_C0B_CONSTANTS_SHA256 = "4596f012bb862aec81a0b07a478223ef02dab8bab3706c4f7902c860958a0f2e"
EXPECTED_C0_BANK_SHA256 = "ab7e04caaf74d4725ed88596da3c39f9303d5e55d7114bbbef1a5500b025ab5f"
EXPECTED_C0_RECEIPT_UNCOMP_SHA256 = "fe7883be37ecf0b7a0a694fc1dd035215d05c27c297cc0959ef23f379258fb1c"
EXPECTED_C0B_DESIGN_SHA256 = "8b3efeaeba3febbcc0e9715840a79bea6d6bc29986f7bf21a89cf6080d61a994"


def verify_preflight_provenance(
    bank_path: Path,
    c0b_constants_path: Path,
    c0_receipt_path: Path,
    design_path: Path,
    bank_meta: dict | None = None,
) -> dict[str, str]:
    """Fail-closed preflight before loading weights: verify all frozen inputs exist & match pinned SHA-256s."""
    for label, p in (
        ("--bank", bank_path),
        ("--c0b-constants", c0b_constants_path),
        ("--c0-receipt", c0_receipt_path),
        ("--design", design_path),
    ):
        if not p.exists():
            raise FileNotFoundError(f"Preflight failed: missing required {label} file: {p}")

    hashes = {
        "bank_sha256": sha256_file(bank_path),
        "c0b_constants_sha256": sha256_file(c0b_constants_path),
        "c0_receipt_uncompressed_sha256": sha256_uncompressed(c0_receipt_path),
        "c0b_design_sha256": sha256_file(design_path),
    }
    expected = {
        "bank_sha256": EXPECTED_C0_BANK_SHA256,
        "c0b_constants_sha256": EXPECTED_C0B_CONSTANTS_SHA256,
        "c0_receipt_uncompressed_sha256": EXPECTED_C0_RECEIPT_UNCOMP_SHA256,
        "c0b_design_sha256": EXPECTED_C0B_DESIGN_SHA256,
    }
    mismatches = {k: (hashes[k], expected[k]) for k in expected if hashes[k] != expected[k]}
    if bank_meta is not None:
        if bank_meta.get("r_lens_sha256") != EXPECTED_R_LENS_SHA256:
            mismatches["r_lens_sha256"] = (bank_meta.get("r_lens_sha256"), EXPECTED_R_LENS_SHA256)
        if bank_meta.get("constants_sha256") != EXPECTED_C0_CONSTANTS_SHA256:
            mismatches["c0_constants_sha256"] = (bank_meta.get("constants_sha256"), EXPECTED_C0_CONSTANTS_SHA256)
    if mismatches:
        raise ValueError(f"Preflight SHA-256 mismatch (refusing to load model / run GPU): {mismatches}")
    return hashes


def sha256_uncompressed(path: Path) -> str:
    h = hashlib.sha256()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@contextlib.contextmanager
def post_norm_inject_hook(
    model,
    layer: int,
    w_unit: torch.Tensor,
    dose: float,
    positions: str,
):
    """Inject `dose * w_unit` into `post_feedforward_layernorm` output at `positions` ('last' or 'all').

    Always validates `w_unit` (finite, unit-norm within 1e-3) and executes the exact
    clone/cast/add path even at `dose == 0.0` so zero-dose shams exercise the real hook path.
    """
    if positions not in ("last", "all"):
        raise ValueError(f"Unsupported positions mode: {positions!r} (expected 'last' or 'all')")
    w_cpu = w_unit.detach().float()
    if not bool(torch.isfinite(w_cpu).all().item()):
        raise ValueError(f"Non-finite write vector passed to post_norm_inject_hook at L{layer}")
    w_nrm = float(w_cpu.norm().item())
    if abs(w_nrm - 1.0) > 1e-3:
        raise ValueError(f"Write vector at L{layer} is not unit-norm: ||w||={w_nrm:.6g}")
    if not math.isfinite(float(dose)):
        raise ValueError(f"Non-finite dose passed to post_norm_inject_hook at L{layer}: {dose}")

    lay = model.model.layers[layer]

    def _hook(_mod, _args, output):
        out = output.clone()
        delta = (float(dose) * w_unit.to(device=out.device, dtype=torch.float32)).to(out.dtype)
        if positions == "last":
            out[:, -1, :] = out[:, -1, :] + delta
        else:
            out[:, :, :] = out[:, :, :] + delta.view(1, 1, -1)
        return out

    h = lay.post_feedforward_layernorm.register_forward_hook(_hook)
    try:
        yield
    finally:
        h.remove()


def _run_regime_d_block(
    model,
    L: int,
    positions: str,
    targets: list[dict],
    target_indices: list[int],
    prompts: list[dict],
    enc_prompts: list[torch.Tensor],
    clean_traces: list[dict],
    clean_probs: list[tuple[torch.Tensor, torch.Tensor]],
    lb: dict,
    u_vectors: torch.Tensor,
    cap_d: float,
    k_dm: int,
    arms: tuple[str, ...],
    fwd_counter: dict[str, int],
    fwd_key: str,
    cap_key: str,
) -> tuple[dict, float]:
    """Run Regime D (Direct_linear = +3.0, cap = 3.0 * post_mlp_norm[L]) for specified targets & arms."""
    arm_pr_keys = {
        "R_mf": "pred_ret_Rmf",
        "logit_lens": "pred_ret_ll",
        "direct_matched": "pred_ret_dm",
    }
    rd: dict = {
        "positions": positions,
        "target_direct_linear": 3.0,
        "dose_cap": round(cap_d, 6),
    }
    for arm in arms:
        rd[arm] = {}

    max_rmf_move = 0.0
    for ti in target_indices:
        t = targets[ti]
        t_str = t["string"]
        tid = int(t["id"])
        u_t = u_vectors[ti]
        geom = lb["per_target_geom"][t_str]

        for arm in arms:
            if arm == "direct_matched":
                dm_draws_rows = []
                for k in range(k_dm):
                    w = lb["direct_matched"][ti, k]
                    u_dot_w = float(torch.dot(u_t, w.float()).item())
                    rows_p = []
                    for pi, p in enumerate(prompts):
                        r_c = float(clean_traces[pi]["rms_final"])
                        dose_d = 3.0 * r_c / u_dot_w
                        if dose_d > cap_d:
                            fwd_counter[cap_key] += 1
                            rows_p.append({
                                "prompt_id": p["id"],
                                "capped": True,
                                "dose_requested": round(dose_d, 6),
                                "dose_cap": round(cap_d, 6),
                            })
                            continue
                        with post_norm_inject_hook(model, L, w, dose=dose_d, positions=positions):
                            tr = capture_last_trace(model, enc_prompts[pi])
                            fwd_counter[fwd_key] += 1
                        rec = decompose_target_trace(
                            clean_traces[pi], tr, L, u_t, tid, dose_d, geom[arm_pr_keys[arm]], w,
                            clean_probs[pi][0], clean_probs[pi][1],
                        )
                        rec["prompt_id"] = p["id"]
                        rec["capped"] = False
                        rows_p.append(rec)
                    dm_draws_rows.append(rows_p)
                rd[arm][t_str] = dm_draws_rows
            else:
                w = lb[arm][ti]
                u_dot_w = float(torch.dot(u_t, w.float()).item())
                rows_p = []
                for pi, p in enumerate(prompts):
                    r_c = float(clean_traces[pi]["rms_final"])
                    dose_d = 3.0 * r_c / u_dot_w
                    if dose_d > cap_d:
                        fwd_counter[cap_key] += 1
                        rows_p.append({
                            "prompt_id": p["id"],
                            "capped": True,
                            "dose_requested": round(dose_d, 6),
                            "dose_cap": round(cap_d, 6),
                        })
                        continue
                    with post_norm_inject_hook(model, L, w, dose=dose_d, positions=positions):
                        tr = capture_last_trace(model, enc_prompts[pi])
                        fwd_counter[fwd_key] += 1
                    rec = decompose_target_trace(
                        clean_traces[pi], tr, L, u_t, tid, dose_d, geom[arm_pr_keys[arm]], w,
                        clean_probs[pi][0], clean_probs[pi][1],
                    )
                    rec["prompt_id"] = p["id"]
                    rec["capped"] = False
                    if arm == "R_mf":
                        dl_all = tr["logits"] - clean_traces[pi]["logits"]
                        max_rmf_move = max(max_rmf_move, float(dl_all.abs().max().item()))
                    rows_p.append(rec)
                rd[arm][t_str] = rows_p

    return rd, max_rmf_move


def _run_regime_n_block(
    model,
    L: int,
    positions: str,
    targets: list[dict],
    target_ids: list[int],
    prompts: list[dict],
    enc_prompts: list[torch.Tensor],
    clean_traces: list[dict],
    clean_probs: list[tuple[torch.Tensor, torch.Tensor]],
    lb: dict,
    u_vectors: torch.Tensor,
    norm_l: float,
    dose_n: float,
    arms: tuple[str, ...],
    fwd_counter: dict[str, int],
    fwd_key: str,
) -> dict:
    """Run Regime N (dose_N = 0.75 * post_mlp_norm[L]) recording `all_16_dlogits` on every arm."""
    arm_pr_keys = {
        "R_mf": "pred_ret_Rmf",
        "R_ridge": "pred_ret_Rridge",
        "logit_lens": "pred_ret_ll",
    }
    rn: dict = {
        "positions": positions,
        "dose_N": round(dose_n, 6),
        "post_mlp_norm": norm_l,
    }
    for arm in arms:
        rn[arm] = {}

    for ti, t in enumerate(targets):
        t_str = t["string"]
        tid = int(t["id"])
        u_t = u_vectors[ti]
        geom = lb["per_target_geom"][t_str]
        for arm in arms:
            w = lb[arm][ti]
            rows_p = []
            for pi, p in enumerate(prompts):
                with post_norm_inject_hook(model, L, w, dose=dose_n, positions=positions):
                    tr = capture_last_trace(model, enc_prompts[pi])
                    fwd_counter[fwd_key] += 1
                rec = decompose_target_trace(
                    clean_traces[pi], tr, L, u_t, tid, dose_n, geom[arm_pr_keys[arm]], w,
                    clean_probs[pi][0], clean_probs[pi][1],
                )
                rec["prompt_id"] = p["id"]
                dl_all = tr["logits"] - clean_traces[pi]["logits"]
                all16 = [round(float(dl_all[x].item()), 6) for x in target_ids]
                rec["all_16_dlogits"] = all16
                if abs(float(rec["exact_dlogit"])) == 0.0:
                    raise RuntimeError(
                        f"DEAD ARM CHECK FAILED: Regime N ({positions}) L{L} {arm}[{t_str!r}] prompt {p['id']} "
                        f"produced exact_dlogit == 0.0"
                    )
                if len(all16) > 1 and max(abs(v) for j, v in enumerate(all16) if j != ti) == 0.0:
                    raise RuntimeError(
                        f"DEAD ARM CHECK FAILED: Regime N ({positions}) L{L} {arm}[{t_str!r}] prompt {p['id']} "
                        f"produced all-zero off-target all_16_dlogits"
                    )
                rows_p.append(rec)
            rn[arm][t_str] = rows_p
    return rn


def run_c0b(
    model,
    tokenizer,
    bank: dict,
    out_dir: Path,
    c0b_constants_path: Path | None = None,
    c0_receipt_path: Path | None = None,
    design_path: Path | None = None,
    device: str = "cuda",
    smoke: bool = False,
    bank_file_sha256: str | None = None,
) -> dict:
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    c0b_const = (
        json.loads(c0b_constants_path.read_text(encoding="utf-8"))
        if c0b_constants_path is not None and c0b_constants_path.exists()
        else None
    )
    c0b_const_sha = sha256_file(c0b_constants_path) if (c0b_constants_path and c0b_constants_path.exists()) else ""
    c0_receipt_uncomp_sha = (
        sha256_uncompressed(c0_receipt_path) if (c0_receipt_path and c0_receipt_path.exists()) else ""
    )
    c0b_design_sha = sha256_file(design_path) if (design_path and design_path.exists()) else ""

    layers_4 = [12, 18] if smoke else list(LAYERS_4L)
    layers_3 = [18] if smoke else list(LAYERS_3L)
    all_targets = list(bank["targets_16"])
    targets = all_targets[:2] if smoke else all_targets
    prompts = PROMPTS_8[:2] if smoke else list(PROMPTS_8)
    k_dm = 1 if smoke else int(bank["meta"]["k_dm"])

    anchor_names = (
        [t["string"] for t in targets]
        if smoke
        else (
            list(c0b_const["reproducibility_anchor_targets_regime_D"])
            if c0b_const is not None
            else list(ANCHOR_TARGETS_REGIME_D)
        )
    )
    t_str_to_idx = {t["string"]: i for i, t in enumerate(targets)}
    anchor_indices = [t_str_to_idx[name] for name in anchor_names]

    u_vectors = bank["u_vectors"].float()
    target_ids = [int(t["id"]) for t in targets]
    target_strings = [t["string"] for t in targets]
    prompt_ids = [p["id"] for p in prompts]
    bank_fingerprint = compute_bank_fingerprint(bank)
    bank_sha = bank_file_sha256 or bank_fingerprint

    enc_prompts = [
        tokenizer.encode(p["text"], return_tensors="pt", add_special_tokens=False).to(device)
        for p in prompts
    ]

    fwd_counts = {
        "clean": 0,
        "sham": 0,
        "regime_D_all": 0,
        "regime_D_all_capped_skipped": 0,
        "regime_D_last_anchor": 0,
        "regime_D_last_anchor_capped_skipped": 0,
        "regime_N_last": 0,
        "regime_N_all": 0,
    }

    # 1. Shared clean baselines (8 forwards in full run)
    clean_traces: list[dict] = []
    clean_probs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for p_ids in enc_prompts:
        tr = capture_last_trace(model, p_ids)
        fwd_counts["clean"] += 1
        clean_traces.append(tr)
        lc = tr["logits"].float()
        clean_probs.append((F.softmax(lc, dim=-1), F.log_softmax(lc, dim=-1)))

    post_mlp_norms: dict[str, float] = {}
    for L in layers_4:
        norms = [float(tr["mlp"][L].norm().item()) for tr in clean_traces]
        post_mlp_norms[str(L)] = round(math.fsum(norms) / len(norms), 6)

    # 2. Non-tautological zero-dose shams (4L x 2 modes = 8 forwards in full run)
    shams: dict[str, dict[str, float]] = {"last": {}, "all": {}}
    for mode in ("last", "all"):
        for L in layers_4:
            w0 = bank["layers_bank"][L]["R_mf"][0]
            with post_norm_inject_hook(model, L, w0, dose=0.0, positions=mode):
                tr_sham = capture_last_trace(model, enc_prompts[0])
                fwd_counts["sham"] += 1
            max_diff = float((tr_sham["logits"] - clean_traces[0]["logits"]).abs().max().item())
            shams[mode][str(L)] = max_diff
            if not math.isfinite(max_diff) or max_diff != 0.0:
                raise RuntimeError(f"C0b.0.1 SHAM FAILED at L{L} ({mode}): max |dlogit| = {max_diff}")

    # 3. Layer-by-layer execution with strict provenance-verified per-process checkpoints
    regime_d_all_layers: dict[str, dict] = {}
    regime_d_last_anchor_layers: dict[str, dict] = {}
    regime_n_last_layers: dict[str, dict] = {}
    regime_n_all_layers: dict[str, dict] = {}
    sham_movement_check: dict[str, dict[str, float]] = {"last": {}, "all": {}}

    for L in layers_4:
        norm_l = post_mlp_norms[str(L)]
        dose_n = 0.75 * norm_l
        cap_d = 3.0 * norm_l
        in_3l = L in layers_3
        expected_prov = {
            "study": "stage_c0b_olmo3_7b_all_vs_last_and_ridge_whitening",
            "layer": L,
            "in_3L": in_3l,
            "model_id": MODEL_ID,
            "model_rev": MODEL_REV,
            "bank_model": bank["meta"]["model"],
            "bank_revision": bank["meta"]["revision"],
            "device": str(device),
            "torch_version": torch.__version__,
            "bank_fingerprint": bank_fingerprint,
            "bank_sha256": bank_sha,
            "r_lens_sha256": bank["meta"]["r_lens_sha256"],
            "c0_constants_sha256": bank["meta"]["constants_sha256"],
            "c0b_constants_sha256": c0b_const_sha,
            "c0_receipt_uncompressed_sha256": c0_receipt_uncomp_sha,
            "c0b_design_sha256": c0b_design_sha,
            "prompt_ids": prompt_ids,
            "target_strings": target_strings,
            "target_ids": target_ids,
            "anchor_targets": anchor_names,
            "post_mlp_norm": norm_l,
            "dose_N": round(dose_n, 6),
            "k_dm": k_dm,
            "smoke": smoke,
        }

        ckpt_path = out_dir / f"ckpt_c0b_L{L}.json"
        if ckpt_path.exists():
            ck = json.loads(ckpt_path.read_text(encoding="utf-8"))
            ck_prov = ck.get("provenance")
            if ck_prov != expected_prov:
                diffs = {
                    k: (ck_prov.get(k) if isinstance(ck_prov, dict) else None, v)
                    for k, v in expected_prov.items()
                    if not isinstance(ck_prov, dict) or ck_prov.get(k) != v
                }
                raise RuntimeError(
                    f"PROVENANCE REFUSAL: checkpoint {ckpt_path} provenance mismatch: {diffs}"
                )
            print(f"[L{L}] Resuming from verified checkpoint {ckpt_path}")
            regime_d_all_layers[str(L)] = ck["regime_D_all"]
            regime_d_last_anchor_layers[str(L)] = ck["regime_D_last_anchor"]
            if in_3l:
                regime_n_last_layers[str(L)] = ck["regime_N_last"]
                regime_n_all_layers[str(L)] = ck["regime_N_all"]
            sham_movement_check["all"][str(L)] = float(ck["sham_movement_all"])
            sham_movement_check["last"][str(L)] = float(ck["sham_movement_last"])
            for k in (
                "regime_D_all",
                "regime_D_all_capped_skipped",
                "regime_D_last_anchor",
                "regime_D_last_anchor_capped_skipped",
                "regime_N_last",
                "regime_N_all",
            ):
                fwd_counts[k] += int(ck["fwd_counts"][k])
            continue

        tl0 = time.time()
        lb = bank["layers_bank"][L]
        l_fwds = {
            "regime_D_all": 0,
            "regime_D_all_capped_skipped": 0,
            "regime_D_last_anchor": 0,
            "regime_D_last_anchor_capped_skipped": 0,
            "regime_N_last": 0,
            "regime_N_all": 0,
        }

        # (a) Regime D "all" (16 targets x 8 prompts x [R_mf, logit_lens, direct_matched(K=4)])
        rd_all, move_all = _run_regime_d_block(
            model, L, "all", targets, list(range(len(targets))), prompts, enc_prompts,
            clean_traces, clean_probs, lb, u_vectors, cap_d, k_dm,
            arms=("R_mf", "logit_lens", "direct_matched"),
            fwd_counter=l_fwds, fwd_key="regime_D_all", cap_key="regime_D_all_capped_skipped",
        )
        if not (move_all > 0.0):
            raise RuntimeError(f"MOVEMENT CHECK FAILED at L{L} (all): R_mf produced 0.0 logit delta")

        # (b) Regime D "last" full-pair reproducibility anchor (2 median targets x [R_mf, direct_matched(K=4)])
        rd_last_anc, move_last = _run_regime_d_block(
            model, L, "last", targets, anchor_indices, prompts, enc_prompts,
            clean_traces, clean_probs, lb, u_vectors, cap_d, k_dm,
            arms=("R_mf", "direct_matched"),
            fwd_counter=l_fwds, fwd_key="regime_D_last_anchor", cap_key="regime_D_last_anchor_capped_skipped",
        )
        if not (move_last > 0.0):
            raise RuntimeError(f"MOVEMENT CHECK FAILED at L{L} (last): R_mf produced 0.0 logit delta")

        # (c) Regime N "last" (live R_mf, R_ridge, logit_lens) & "all" secondary (R_mf, R_ridge) on 3L
        rn_last = None
        rn_all = None
        if in_3l:
            rn_last = _run_regime_n_block(
                model, L, "last", targets, target_ids, prompts, enc_prompts,
                clean_traces, clean_probs, lb, u_vectors, norm_l, dose_n,
                arms=("R_mf", "R_ridge", "logit_lens"),
                fwd_counter=l_fwds, fwd_key="regime_N_last",
            )
            rn_all = _run_regime_n_block(
                model, L, "all", targets, target_ids, prompts, enc_prompts,
                clean_traces, clean_probs, lb, u_vectors, norm_l, dose_n,
                arms=("R_mf", "R_ridge"),
                fwd_counter=l_fwds, fwd_key="regime_N_all",
            )

        sham_movement_check["all"][str(L)] = round(move_all, 6)
        sham_movement_check["last"][str(L)] = round(move_last, 6)

        # Atomic per-process checkpoint write
        tmp_ckpt = out_dir / f".ckpt_c0b_L{L}.{os.getpid()}.tmp"
        ck_payload = {
            "layer": L,
            "provenance": expected_prov,
            "sham_movement_all": sham_movement_check["all"][str(L)],
            "sham_movement_last": sham_movement_check["last"][str(L)],
            "fwd_counts": l_fwds,
            "regime_D_all": rd_all,
            "regime_D_last_anchor": rd_last_anc,
        }
        if in_3l:
            ck_payload["regime_N_last"] = rn_last
            ck_payload["regime_N_all"] = rn_all
        tmp_ckpt.write_text(json.dumps(ck_payload), encoding="utf-8")
        tmp_ckpt.replace(ckpt_path)

        regime_d_all_layers[str(L)] = rd_all
        regime_d_last_anchor_layers[str(L)] = rd_last_anc
        if in_3l:
            regime_n_last_layers[str(L)] = rn_last
            regime_n_all_layers[str(L)] = rn_all
        for k in l_fwds:
            fwd_counts[k] += l_fwds[k]
        print(
            f"[L{L}] done in {time.time() - tl0:.1f}s "
            f"(D_all={l_fwds['regime_D_all']}, D_anc={l_fwds['regime_D_last_anchor']}, "
            f"N_last={l_fwds['regime_N_last']}, N_all={l_fwds['regime_N_all']}, "
            f"cap_all={l_fwds['regime_D_all_capped_skipped']})",
            flush=True,
        )

    total_exec = (
        fwd_counts["clean"]
        + fwd_counts["sham"]
        + fwd_counts["regime_D_all"]
        + fwd_counts["regime_D_last_anchor"]
        + fwd_counts["regime_N_last"]
        + fwd_counts["regime_N_all"]
    )
    receipt = {
        "meta": {
            "study": "stage_c0b_olmo3_7b_all_vs_last_and_ridge_whitening",
            "model": bank["meta"]["model"],
            "revision": bank["meta"]["revision"],
            "smoke": smoke,
            "elapsed_sec": round(time.time() - t0, 2),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "bank_sha256": bank_sha,
            "bank_fingerprint": bank_fingerprint,
            "r_lens_sha256": bank["meta"]["r_lens_sha256"],
            "c0_constants_sha256": bank["meta"]["constants_sha256"],
            "c0b_constants_sha256": c0b_const_sha,
            "c0_receipt_uncompressed_sha256": c0_receipt_uncomp_sha,
            "c0b_design_sha256": c0b_design_sha,
            "k_dm": k_dm,
            "forward_counts": fwd_counts,
            "total_forwards_executed": total_exec,
        },
        "layers_4L": layers_4,
        "layers_3L": layers_3,
        "anchor_targets_regime_D": anchor_names,
        "targets_16": targets,
        "prompts_8": prompts,
        "post_mlp_norms": post_mlp_norms,
        "shams": shams,
        "sham_movement_check": sham_movement_check,
        "regime_D_all": regime_d_all_layers,
        "regime_D_last_anchor": regime_d_last_anchor_layers,
        "regime_N_last": regime_n_last_layers,
        "regime_N_all": regime_n_all_layers,
    }
    gz_path = out_dir / "c0b_receipt.json.gz"
    raw_bytes = json.dumps(receipt, sort_keys=False).encode("utf-8")
    with gzip.open(gz_path, "wb", compresslevel=6) as f:
        f.write(raw_bytes)
    print(
        f"Saved {gz_path} ({gz_path.stat().st_size / 1e6:.2f} MB), "
        f"uncompressed_sha256={hashlib.sha256(raw_bytes).hexdigest()}, "
        f"total_forwards={total_exec}, elapsed={receipt['meta']['elapsed_sec']}s"
    )
    return receipt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, required=True, help="Path to frozen c0_vector_bank.pt")
    ap.add_argument("--c0b-constants", type=Path, required=True, help="Path to c0b_prereg_constants.json")
    ap.add_argument("--c0-receipt", type=Path, required=True, help="Path to published c0_receipt.json.gz")
    ap.add_argument("--design", type=Path, required=True, help="Path to registered OLMO3-TRANSPORT-C0B-DESIGN.md")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    bank = torch.load(str(args.bank), map_location="cpu", weights_only=False)
    if args.smoke:
        bank_sha = sha256_file(args.bank)
    else:
        preflight = verify_preflight_provenance(
            args.bank, args.c0b_constants, args.c0_receipt, args.design, bank_meta=bank["meta"]
        )
        bank_sha = preflight["bank_sha256"]
        print(f"Preflight provenance OK (all 6 pinned SHA-256 hashes verified).", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {MODEL_ID}@{MODEL_REV} in bf16 on {args.device} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REV)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REV,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    model.eval()
    run_c0b(
        model,
        tokenizer,
        bank,
        args.out_dir,
        c0b_constants_path=args.c0b_constants,
        c0_receipt_path=args.c0_receipt,
        design_path=args.design,
        device=args.device,
        smoke=args.smoke,
        bank_file_sha256=bank_sha,
    )


if __name__ == "__main__":
    main()
