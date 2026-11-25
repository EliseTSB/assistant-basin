#!/usr/bin/env python3
"""build_olmo3_c2_constants.py — Deterministic 0-CU pre-registration & vector-bank builder for OLMo 3 7B Stage C2 (v3).

Folds ALL Round-2 cold design review findings (REVIEW-design-ds-r2.md / #115: B1-B6, M1-M10, m1-m8):
  - B1/B2: Explicit r_all, pred_ret_L18, signed closure (pooled ratio + denominator-free paired d_i(0.50)),
           and recipient-matched Regime-D dose (3.0 * r_c(recipient) / (u_t . w_Rmf), cap 3.0 * pmn[18]).
  - B3: Identical-instruction, zero-stem-duplication LongBare vs ChatML pairs (exact 0-token length diff on 12/12!).
  - B4/B6/M10: N=12 independent stem pairs (2^12=4,096 exact one-sided sign flips for H0: closure <= 0.50;
               11/12 positive p=0.003174 < alpha/8=0.00625), 90% paired BCa CI subset [0.55, 1.50] + point in [0.75, 1.30],
               full-conjunction Monte-Carlo power table including 0.70 and 0.75 rows.
  - B5: Separate 8-test reject-the-null Holm-Bonferroni family (m=8, alpha/(9-k)) from Equivalence & Control Gates
        (EQ-C2.2 negative controls + shape/KL bar; EQ-C2.4c K=3 median TOST with 3-way power & n_floored <= 2).
  - M1/M2/m6: Primary C2.1 tests evaluated on heldout_16x8 (N=16 primary_16 sources -> heldout_8 cross-stratum receivers),
              adding p_C2.1_shk_vs_ll vs free logit_lens baseline and 90% BCa q05 >= 1.05x on same/cross ratio.
  - M3/M4/M5/m4: Zero unread forwards (8,623 total); 100% draw-level 1-to-1 bank (34 tensors); C2.4b double-contrast
                 generic packing control (R_target / R_pack_rand >= 1.05x); C2.4a matched pre-norm + postmatched arm.
  - M6: Unified math.ceil(rate * N) derivation for BOTH Regime-N (N=2,880) and Regime-D (N=864) C2.0.3 ceilings.
  - M7: Added ctrl_self_attenuated_L19_31 control + numeric open-vs-func & clean-KL bars.
  - M8: Exact ΣMLP definition + pinned C1 receipt provenance (block2a_regime_N_last logit_lens).
  - M9: Three-way C2.4c power table (P_additive, P_non_additive, P_inconclusive) + median(rel_int) + n_floored <= 2.
  - m1-m8: Exact C1 L27 pmn 7.473792; 13 write shams + 6 patch shams = 19; LongBare+ChatML in C2.1/C2.3; aligned C2.4b sim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
DEFAULT_R_LENS = REPO_ROOT / "work" / "vesper-olmo3-7b-neuron-lens" / "r_lens.pt"
FALLBACK_R_LENS = Path.home() / "green" / "Journal" / "notes" / "reports" / "vesper-olmo3-7b-neuron-lens" / "r_lens.pt"
C0_CONST_PATH = ROOT / "c0_prereg_constants.json"
C0B_CONST_PATH = ROOT / "c0b_prereg_constants.json"
C1_CONST_PATH = ROOT / "c1_prereg_constants.json"
DEFAULT_OUT_CONSTANTS = ROOT / "c2_prereg_constants.json"
DEFAULT_OUT_BANK = REPO_ROOT / "work" / "olmo3_c2" / "c2_vector_bank.safetensors"

EXPECTED_R_LENS_SHA256 = "ceef21fc90b17aafa5625f23ab2c1f35be7b2f8b735847cb89bfd2a438caa955"
MODEL_ID = "allenai/Olmo-3-7B-Instruct"
MODEL_REV = "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc"

PRIMARY_C21_LAYERS = [18, 24, 27]
SWEEP_LAYERS = [18, 21, 24, 27, 30]   # Exact 5 layers read by C2.3 decision rule (M3)
PRENORM_LAYERS = [21, 24, 27]         # K=3 post-erasure C2.4 layers

# Deterministic RNG seeds (C2 v3 family)
PERM_SEED = 2026112501
BCA_SEED = 2026112502
BANK_DM_SEED_BASE = 2026112503
BANK_RAND_SEED_BASE = 2026112504
BANK_PRENORM_RAND_SEED_BASE = 2026112505
POWER_SIM_SEED = 2026112506

K_DM_DRAWS = 1              # Draw-level 1-to-1 with C2.2 runner (m4)
M_RAND_ISO_DRAWS = 8        # Read by C2.3 ret_rand(L) noise-floor bar (M3)
K_PRENORM_RAND_DRAWS = 1    # Read by C2.4b M4 generic-packing control (L24 ff & attn only)
GAMMA_DROP_FF = 4
GAMMA_DROP_ATTN = 16
COV_SHRINK_ALPHA = 0.1

# Exact C0 and C1 receipt post_mlp_norms (m2: C1 L27 is 7.473792 verbatim from c1_receipt.json.gz)
C0_RECEIPT_PMN = {"24": 9.325704, "27": 10.237432}
C1_RECEIPT_PMN = {"24": 7.210327, "27": 7.473792}

# Exact C1 logit_lens sum_mlp provenance for C2.3 thresholds (M8, results/vesper-olmo3-7b-stage-c1/c1_receipt.json.gz)
C1_LOGIT_LENS_SUM_MLP_PROVENANCE = {
    "source_file": "results/vesper-olmo3-7b-stage-c1/c1_receipt.json.gz",
    "source_key": "block2a_regime_N_last[L]['logit_lens'][target][prompt]['sum_mlp']",
    "L18": {"median_open_18": -3.0791, "median_func_6": -1.1021, "gap_open_minus_func": -1.9770},
    "L24": {"median_open_18": -0.7432, "median_func_6": -1.1016, "gap_open_minus_func": 0.3584},
    "L27": {"median_open_18": -1.0024, "median_func_6": -1.0707, "gap_open_minus_func": 0.0683},
}

# Exact C1 registered counts over N=1,728 (F4: exact ceil(count * N / 1728) eliminates rounding drift!)
C1_REG_N = 1728
C1_REG_COUNT_RMSN_GT_20PCT = 2
C1_REG_COUNT_RMSN_GT_12PCT = 26
C1_REG_COUNT_FDE_GT_045 = 12

# Fresh 24-target mixed-POS bank (0 overlap with C0, C1, or personas).
# Split deterministically per stratum: first 4 = primary_16, last 2 = heldout_8 (M2).
TARGETS_24_C2_SPEC: list[tuple[str, str, str]] = [
    ("nouns", " candle", "primary_16"),
    ("nouns", " barrel", "primary_16"),
    ("nouns", " feather", "primary_16"),
    ("nouns", " ribbon", "primary_16"),
    ("nouns", " basket", "heldout_8"),
    ("nouns", " glacier", "heldout_8"),
    ("verbs", " polish", "primary_16"),
    ("verbs", " stumble", "primary_16"),
    ("verbs", " vanish", "primary_16"),
    ("verbs", " mend", "primary_16"),
    ("verbs", " drift", "heldout_8"),
    ("verbs", " carve", "heldout_8"),
    ("adjectives", " sturdy", "primary_16"),
    ("adjectives", " velvet", "primary_16"),
    ("adjectives", " crisp", "primary_16"),
    ("adjectives", " faint", "primary_16"),
    ("adjectives", " vivid", "heldout_8"),
    ("adjectives", " shallow", "heldout_8"),
    ("function", " however", "primary_16"),
    ("function", " whereas", "primary_16"),
    ("function", " beneath", "primary_16"),
    ("function", " despite", "primary_16"),
    ("function", " instead", "heldout_8"),
    ("function", " beyond", "heldout_8"),
]

# Balanced 12-target subset for C2.2 causal patching (first 3 of each stratum: 9 open-class + 3 function)
C2_2_TARGET_INDICES = [0, 1, 2, 6, 7, 8, 12, 13, 14, 18, 19, 20]
PERSONA_WORDS = {" pirate", " librarian", " wizard", " robot", " owl", " fox", " whale", " baker"}

# 12 independent stem triples (B3 & B4: identical instruction + context clause, ZERO stem duplication in ChatML,
# exact 0-token length difference between LongBare and ChatML, bit-identical trailing stem suffix on 12/12!).
C2_STEM_SPECS_12: list[tuple[str, str, str, str]] = [
    ("narr_0", "narrative", "Inside the small drawer, the card displayed a single word:", "The wooden desk stood beside the north window while steady rain tapped softly on the glass."),
    ("narr_1", "narrative", "Across the chalk slate, the final word written in dust was:", "An empty classroom stayed silent after the evening bell rang along the dim corridor."),
    ("narr_2", "narrative", "Under the brass lantern, the folded slip named one word:", "Footsteps faded down the stone hallway as cool air moved through the open archway."),
    ("narr_3", "narrative", "Beside the iron key, the stamped wooden tag showed one word:", "Near the quiet harbor pier, heavy ropes creaked against weathered timber posts before sunrise."),
    ("narr_4", "narrative", "Above the oak mantel, the framed note held a single word:", "Embers settled low inside the stone hearth while a tall clock ticked steadily nearby."),
    ("narr_5", "narrative", "Within the leather pouch, the small brass coin bore one word:", "Dry leaves crossed the gravel lane outside the brick wall as clouds gathered overhead."),
    ("anal_0", "analytical", "In the test ledger, the target entry listed in row four was:", "During the annual warehouse audit, two clerks checked each printed catalog sheet carefully."),
    ("anal_1", "analytical", "From the sample index, the marked label shown at the end was:", "After reviewing the quarterly archive logs, the analyst prepared a brief reference table."),
    ("anal_2", "analytical", "On the control sheet, the verified item printed in box two was:", "Before closing the laboratory binder for the night, the technician logged each serial code."),
    ("anal_3", "analytical", "Within the draft manual, the underlined term at the end was:", "Following the standard revision checklist, the editor inspected every numbered appendix page."),
    ("anal_4", "analytical", "In the audit summary, the flagged token in column three was:", "Prior to archiving the monthly sensor records, the engineer verified each calibration table."),
    ("anal_5", "analytical", "From the coding chart, the final symbol on line six was:", "While updating the municipal transit schedule, the planner checked every route timetable."),
]


def write_canonical_safetensors(tensors: dict[str, torch.Tensor], path: Path, metadata: dict[str, str]) -> None:
    header: dict[str, Any] = {"__metadata__": {k: metadata[k] for k in sorted(metadata)}}
    chunks: list[bytes] = []
    offset = 0
    for k in sorted(tensors):
        t = tensors[k].detach().cpu().contiguous()
        assert t.dtype == torch.float32, f"Expected float32 tensor for {k}, got {t.dtype}"
        raw = t.numpy().astype("<f4", copy=False).tobytes()
        header[k] = {"dtype": "F32", "shape": list(t.shape), "data_offsets": [offset, offset + len(raw)]}
        chunks.append(raw)
        offset += len(raw)
    hb = json.dumps(header, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for c in chunks:
            f.write(c)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_r_lens(cli_path: Path | None) -> Path:
    if cli_path is not None and cli_path.is_file():
        return cli_path
    if DEFAULT_R_LENS.is_file():
        return DEFAULT_R_LENS
    if FALLBACK_R_LENS.is_file():
        return FALLBACK_R_LENS
    raise FileNotFoundError(f"Could not locate r_lens.pt at {DEFAULT_R_LENS} or {FALLBACK_R_LENS}")


def resolve_hf_snapshot(model_dir: str | None) -> Path:
    if model_dir:
        return Path(model_dir)
    from huggingface_hub.constants import HF_HUB_CACHE
    cand = Path(HF_HUB_CACHE) / f"models--{MODEL_ID.replace('/', '--')}" / "snapshots" / MODEL_REV
    if cand.is_dir():
        return cand
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(MODEL_ID, revision=MODEL_REV, local_files_only=True))


def build_c2_prompts_all(tok: Any, target_words_lower: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build:
    - prompts_8_c21: 4 LongBare + 4 ChatML (narr_0, narr_1, anal_0, anal_1) — matched length (m7)!
    - c22_stem_pairs_12: ALL 12 triples (ShortBare, LongBare, ChatML) with identical instruction/context,
      ZERO stem duplication (B3), exact suffix token identity (B1), and max |T_long - T_chat| <= 1 (B2/B4).
    """
    triples: list[dict[str, Any]] = []
    prompts_8_c21: list[dict[str, Any]] = []

    for idx, (sid, style, stem, ctx) in enumerate(C2_STEM_SPECS_12):
        # Verify zero target substring overlap
        combined_lower = (stem + " " + ctx).lower()
        for tw in target_words_lower:
            assert tw not in combined_lower, f"Target {tw!r} appears inside prompt {sid}"

        short_text = stem
        long_text = (
            f"User instruction: Complete the final line below with one fitting word. "
            f"Context: {ctx}\nAssistant final response:\n{stem}"
        )
        chat_text = (
            f"<|im_start|>user\nComplete the final line below with one fitting word. "
            f"Context: {ctx}<|im_end|>\n<|im_start|>assistant\n{stem}"
        )
        # B3 assertion: stem appears EXACTLY ONCE in every format (zero in-context stem duplication!)
        assert short_text.count(stem) == 1
        assert long_text.count(stem) == 1
        assert chat_text.count(stem) == 1

        ids_short = tok.encode(short_text, add_special_tokens=False)
        ids_long = tok.encode(long_text, add_special_tokens=False)
        ids_chat = tok.encode(chat_text, add_special_tokens=False)
        t_s, t_l, t_c = len(ids_short), len(ids_long), len(ids_chat)

        # B1 strict suffix-identity assertion
        assert ids_long[-t_s:] == ids_short, f"Suffix mismatch LongBare vs ShortBare on {sid}"
        assert ids_chat[-t_s:] == ids_short, f"Suffix mismatch ChatML vs ShortBare on {sid}"
        assert tok.decode([ids_short[-1]]) == ":", f"Expected ':' ending on {sid}"
        # B2 strict matched-length check
        assert abs(t_l - t_c) <= 1, f"Length mismatch LongBare ({t_l}) vs ChatML ({t_c}) on {sid}"

        triples.append({
            "pair_idx": idx,
            "stem_id": sid,
            "style": style,
            "stem": stem,
            "context_clause": ctx,
            "suffix_len": t_s,
            "short_bare": {
                "id": f"c2_shortbare_{sid}", "format": "short_bare", "text": short_text,
                "n_tokens": t_s, "last_token_id": int(ids_short[-1]), "last_token_piece": ":",
            },
            "long_bare": {
                "id": f"c2_longbare_{sid}", "format": "long_bare", "text": long_text,
                "n_tokens": t_l, "untouched_prefix_tokens": t_l - t_s,
                "last_token_id": int(ids_long[-1]), "last_token_piece": ":",
            },
            "chatml": {
                "id": f"c2_chat_{sid}", "format": "chatml", "text": chat_text,
                "n_tokens": t_c, "untouched_prefix_tokens": t_c - t_s,
                "last_token_id": int(ids_chat[-1]), "last_token_piece": ":",
            },
            "long_minus_chat_tokens": t_l - t_c,
        })

    # C2.1 / C2.3 (8 prompts): 4 LongBare + 4 ChatML on narr_0, narr_1, anal_0, anal_1 (matched length, m7!)
    c21_indices = [0, 1, 6, 7]
    for i in c21_indices:
        lb = triples[i]["long_bare"]
        prompts_8_c21.append({
            "id": lb["id"], "format": "long_bare", "style": triples[i]["style"],
            "stem": triples[i]["stem"], "text": lb["text"], "n_tokens": lb["n_tokens"],
            "last_token_id": lb["last_token_id"], "last_token_piece": lb["last_token_piece"],
        })
    for i in c21_indices:
        ch = triples[i]["chatml"]
        prompts_8_c21.append({
            "id": ch["id"], "format": "chatml", "style": triples[i]["style"],
            "stem": triples[i]["stem"], "text": ch["text"], "n_tokens": ch["n_tokens"],
            "last_token_id": ch["last_token_id"], "last_token_piece": ch["last_token_piece"],
        })

    return prompts_8_c21, triples


def build_c0_anchor_colon_prompts(c0b_prompts: list[dict[str, Any]], tok: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in c0b_prompts:
        pid, fmt, orig = p["id"], p["format"], p["text"]
        assert orig.endswith(" a"), f"Expected C0 prompt to end with ' a': {orig!r}"
        colon_text = orig[:-2] + ":"
        if fmt == "chatml":
            colon_text = colon_text.replace(" a...<|im_end|>", ":<|im_end|>")
        for suffix, ending, txt in [("orig_a", "a", orig), ("colon", ":", colon_text)]:
            ids = tok.encode(txt, add_special_tokens=False)
            out.append({
                "id": f"{pid}_{suffix}", "pair_id": pid, "format": fmt, "ending": ending,
                "text": txt, "n_tokens": len(ids), "last_token_id": int(ids[-1]),
                "last_token_piece": tok.decode([ids[-1]]),
            })
    return out


def orthonormal_basis(vecs: list[torch.Tensor]) -> list[torch.Tensor]:
    basis: list[torch.Tensor] = []
    for v in vecs:
        w = v.clone().double()
        for _ in range(2):
            for b in basis:
                w = w - torch.dot(w, b) * b
        nrm = float(w.norm().item())
        if nrm > 1e-10:
            basis.append(w / nrm)
    return basis


def make_direct_matched_single(
    u_hat: torch.Tensor, w_rmf: torch.Tensor, w_rridge: torch.Tensor, c: float, seed: int,
) -> torch.Tensor:
    basis = orthonormal_basis([u_hat, w_rmf, w_rridge])
    d = u_hat.shape[0]
    s = math.sqrt(max(1.0 - c * c, 0.0))
    u_hat_d = u_hat.double()
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    z = torch.randn(d, generator=g, dtype=torch.float64)
    for _ in range(2):
        for b in basis:
            z = z - torch.dot(z, b) * b
    z_perp = z / z.norm().clamp_min(1e-12)
    w = c * u_hat_d + s * z_perp
    w = w / w.norm().clamp_min(1e-12)
    assert abs(float(w.norm().item()) - 1.0) < 1e-6
    assert abs(float(torch.dot(w, u_hat_d).item()) - c) < 1e-5
    return w.float()


def make_rand_iso_draws(d: int, m_draws: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    v = torch.randn(m_draws, d, generator=g, dtype=torch.float64)
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return v.float()


def make_masked_rand_single(mask_01: torch.Tensor, n_targets: int, seed_base: int) -> torch.Tensor:
    d = mask_01.shape[0]
    mask_d = mask_01.double()
    rows = []
    for ti in range(n_targets):
        g = torch.Generator(device="cpu")
        g.manual_seed(seed_base + ti * 17)
        v = torch.randn(d, generator=g, dtype=torch.float64) * mask_d
        v = v / v.norm().clamp_min(1e-12)
        rows.append(v.float())
    return torch.stack(rows, dim=0)  # [n_targets, d]


def load_safetensors_selected(snap: Path, wanted: set[str]) -> dict[str, torch.Tensor]:
    idx_path = snap / "model.safetensors.index.json"
    out: dict[str, torch.Tensor] = {}
    if idx_path.is_file():
        wm = json.loads(idx_path.read_text(encoding="utf-8"))["weight_map"]
        by_file: dict[str, list[str]] = {}
        for k in wanted:
            if k in wm:
                by_file.setdefault(wm[k], []).append(k)
        for fname in sorted(by_file):
            with safe_open(str(snap / fname), framework="pt", device="cpu") as f:
                for k in by_file[fname]:
                    out[k] = f.get_tensor(k).float()
    else:
        for st in sorted(snap.glob("*.safetensors")):
            with safe_open(str(st), framework="pt", device="cpu") as f:
                for k in f.keys():
                    if k in wanted:
                        out[k] = f.get_tensor(k).float()
    missing = wanted - set(out.keys())
    assert not missing, f"Missing model tensors: {sorted(missing)}"
    return out


def simulate_c2_power(seed: int = POWER_SIM_SEED) -> dict[str, Any]:
    """Full-conjunction Monte-Carlo power tables (B6, M9, m8):
    1. C2.2 at N=12 pairs: full conjunction of (format-gap gate >= 0.30 & CI>0) + (exact 2^12 one-sided sign-flip
       p <= alpha/8 = 0.00625 on H0: closure <= 0.50) + (point in [0.75, 1.30]) + (90% CI subset [0.55, 1.50]).
       Includes 0.50, 0.65, 0.70, 0.75, 0.85, 1.00, 1.30, 1.40 rows (B6, M10).
    2. C2.4c K=3 median TOST at N=24 targets: 3-way outcomes (P_additive, P_non_additive, P_inconclusive) (M9).
    3. C2.4b m70a30 vs ff at N=24 targets: aligned exactly to Holm alpha/8 = 0.00625 (t >= 2.74 at df=23) +
       geomean >= 1.10x + 90% CI q05 > 1.00x (m8).
    """
    rng = np.random.default_rng(seed)

    # Exact 2^12 = 4096 sign-flip matrix for C2.2 exact one-sided p on d_i(0.50)
    signs_12 = ((np.arange(4096)[:, None] >> np.arange(12)[None, :]) & 1) * 2 - 1  # [4096, 12]
    alpha_step1 = 0.05 / 7.0  # 0.007143 (m=7 Holm family; C2.4b moved to descriptive estimation per F3)

    # 1. C2.2 Full Conjunction (N=12 pairs; simulated with realistic gap ~ Normal(0.60, 0.12) matching C1 L18)
    c22_power: dict[str, Any] = {}
    for sd in (0.10, 0.15, 0.25):
        row = {}
        for true_cl in (0.50, 0.65, 0.70, 0.75, 0.85, 1.00, 1.30, 1.40):
            n_sim = 2000
            # Gap per pair G_i ~ Normal(0.60, 0.12)
            g = rng.normal(0.60, 0.12, size=(n_sim, 12))
            g_mean = g.mean(axis=1)
            g_se = g.std(axis=1, ddof=1) / math.sqrt(12)
            gate_ok = (g_mean >= 0.30) & ((g_mean - 1.7959 * g_se) > 0.0)

            # Patched reduction per pair: mean(delta) / mean(g) = pooled closure
            c_pair = rng.normal(true_cl, sd, size=(n_sim, 12))
            delta = c_pair * g
            pt = delta.mean(axis=1) / g_mean

            # Denominator-free paired contrast d_i(0.50) = delta_i - 0.50 * g_i (exact 2^12 one-sided p)
            d50 = delta - 0.50 * g  # [n_sim, 12]
            obs_mean = d50.mean(axis=1)  # [n_sim]
            # Vectorized exact permutation p in chunks of 200
            p_sign = np.empty(n_sim, dtype=np.float64)
            for s0 in range(0, n_sim, 200):
                blk = d50[s0 : s0 + 200]  # [B, 12]
                perm_means = (blk @ signs_12.T) / 12.0  # [B, 4096]
                p_sign[s0 : s0 + 200] = (perm_means >= obs_mean[s0 : s0 + 200, None] - 1e-12).mean(axis=1)
            sign_ok = p_sign <= alpha_step1

            # 90% paired bootstrap CI on pooled closure ratio subset [0.55, 1.50] (400 resamples)
            idx = rng.integers(0, 12, size=(n_sim, 400, 12))
            b_del = np.take_along_axis(delta[:, None, :], idx, axis=2).mean(axis=2)
            b_g = np.take_along_axis(g[:, None, :], idx, axis=2).mean(axis=2)
            b_cl = b_del / np.where(np.abs(b_g) < 1e-6, 1e-6, b_g)
            ci_lo = np.quantile(b_cl, 0.05, axis=1)
            ci_hi = np.quantile(b_cl, 0.95, axis=1)
            ci_ok = (ci_lo >= 0.55) & (ci_hi <= 1.50)
            pt_ok = (pt >= 0.75) & (pt <= 1.30)

            full_ok = gate_ok & sign_ok & pt_ok & ci_ok
            row[f"true_{true_cl:.2f}"] = round(float(full_ok.mean()), 4)
        c22_power[f"sd_{sd:.2f}"] = row

    # 2. C2.4c K=3 Median TOST at N=24 targets — 3-way outcomes (P_additive, P_non_additive, P_inconclusive) (M9)
    c24c_power: dict[str, Any] = {}
    for sd in (0.12, 0.20, 0.30):
        row = {}
        # Effective median SE for Normal(mu, sd) at N=24 is sqrt(pi/2)*sd/sqrt(24) = 0.2557*sd; 90% half-width ~ 0.421*sd
        eff_margin = round(max(0.0, 0.30 - 1.6449 * math.sqrt(math.pi / 2.0) * sd / math.sqrt(24)), 4)
        for true_int in (0.00, 0.10, 0.20, 0.35, 0.50):
            n_sim = 2000
            x = rng.normal(true_int, sd, size=(n_sim, 24))
            med = np.median(x, axis=1)
            idx = rng.integers(0, 24, size=(n_sim, 400, 24))
            b_med = np.median(np.take_along_axis(x[:, None, :], idx, axis=2), axis=2)
            lo = np.quantile(b_med, 0.05, axis=1)
            hi = np.quantile(b_med, 0.95, axis=1)
            p_add = (np.abs(med) <= 0.20) & (lo >= -0.30) & (hi <= 0.30)
            p_non = (lo > 0.30) | (hi < -0.30)
            p_inc = ~(p_add | p_non)
            row[f"true_{true_int:.2f}"] = {
                "P_additive": round(float(p_add.mean()), 4),
                "P_non_additive": round(float(p_non.mean()), 4),
                "P_inconclusive": round(float(p_inc.mean()), 4),
            }
        c24c_power[f"sd_{sd:.2f}"] = {"effective_equiv_margin_approx": eff_margin, "grid": row}

    # 3. C2.4b m70a30 vs ff (descriptive estimation sensitivity grid, F3; shown at nominal alpha=0.05 one-sided t>=1.71)
    c24b_power: dict[str, Any] = {}
    for sd_log in (0.06, 0.10, 0.15):
        row = {}
        for true_ratio in (1.00, 1.10, 1.15, 1.20, 1.25):
            n_sim = 4000
            x = rng.normal(math.log(true_ratio), sd_log, size=(n_sim, 24))
            m = x.mean(axis=1)
            se = x.std(axis=1, ddof=1) / math.sqrt(24)
            ok = (np.exp(m) >= 1.10) & ((m - 1.7139 * se) > 0.0)
            row[f"ratio_{true_ratio:.2f}"] = round(float(ok.mean()), 4)
        c24b_power[f"sd_log_{sd_log:.2f}"] = row

    return {
        "seed": seed,
        "sd_provenance_note": "Sensitivity grids (not empirically derived from C1); C2.1 rows 1-4 and C2.4a row 7 are not separately power-simulated (F5).",
        "C2.2_full_conjunction_N12_pairs_Holm7": c22_power,
        "C2.4c_K3_median_TOST_3way_N24": c24c_power,
        "C2.4b_m70a30_descriptive_sensitivity_N24": c24b_power,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build deterministic Stage C2 v3 pre-registration constants and vector bank (0 CU).")
    ap.add_argument("--r-lens", type=Path, default=None)
    ap.add_argument("--model-dir", type=str, default=None)
    ap.add_argument("--out-constants", type=Path, default=DEFAULT_OUT_CONSTANTS)
    ap.add_argument("--out-bank", type=Path, default=DEFAULT_OUT_BANK)
    args = ap.parse_args()

    r_lens_path = resolve_r_lens(args.r_lens)
    r_sha = sha256_file(r_lens_path)
    assert r_sha == EXPECTED_R_LENS_SHA256, f"r_lens sha mismatch: {r_sha}"

    c0 = json.loads(C0_CONST_PATH.read_text(encoding="utf-8"))
    c0b = json.loads(C0B_CONST_PATH.read_text(encoding="utf-8"))
    c1 = json.loads(C1_CONST_PATH.read_text(encoding="utf-8"))

    forbidden_words = {t["string"] for t in c0["targets_16"]} | {t["string"] for t in c1["targets_24"]} | PERSONA_WORDS
    forbidden_prompts = {p["text"] for p in c0b["prompts_8"]} | {p["text"] for p in c1["prompts_8"]}

    snap = resolve_hf_snapshot(args.model_dir)
    tok = AutoTokenizer.from_pretrained(str(snap), local_files_only=True)

    # 1. Verify 24 fresh targets (16 primary + 8 held-out eval split per M2)
    targets: list[dict[str, Any]] = []
    for idx, (stratum, word, split) in enumerate(TARGETS_24_C2_SPEC):
        assert word not in forbidden_words, f"Overlap with C0/C1/persona target: {word}"
        ids = tok.encode(word, add_special_tokens=False)
        dec = tok.decode(ids)
        assert len(ids) == 1 and dec == word, f"Not single roundtrip token: {word!r} -> {ids} ({dec!r})"
        targets.append({
            "idx": idx, "stratum": stratum, "split": split, "in_c22_12": idx in C2_2_TARGET_INDICES,
            "string": word, "id": int(ids[0]), "decoded_piece": dec,
        })
    assert len({t["id"] for t in targets}) == 24
    targets_spec_sha = hashlib.sha256(
        json.dumps([(s, w) for s, w, _ in TARGETS_24_C2_SPEC], separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()

    # 2. Verify prompts, B1 suffix alignment, B2 length match, & B3 zero stem duplication across ALL 12 triples
    target_words_lower = [t["string"].strip().lower() for t in targets]
    prompts_8_c21, c22_triples = build_c2_prompts_all(tok, target_words_lower)
    for p in prompts_8_c21:
        assert p["text"] not in forbidden_prompts, f"Overlap with C0/C1 prompt: {p['id']}"
    for tr in c22_triples:
        for k in ("short_bare", "long_bare", "chatml"):
            assert tr[k]["text"] not in forbidden_prompts, f"Overlap with C0/C1 prompt: {tr[k]['id']}"

    bare_T = [tr["short_bare"]["n_tokens"] for tr in c22_triples]
    long_T = [tr["long_bare"]["n_tokens"] for tr in c22_triples]
    chat_T = [tr["chatml"]["n_tokens"] for tr in c22_triples]
    c0_colon_prompts = build_c0_anchor_colon_prompts(c0b["prompts_8"], tok)

    # 3. Load frozen unembedding, final gamma, per-layer gammas, and R_out/R_mid
    wanted_tensors = {"lm_head.weight", "model.norm.weight"} | {
        f"model.layers.{L}.post_feedforward_layernorm.weight" for L in PRENORM_LAYERS
    } | {"model.layers.24.post_attention_layernorm.weight"}
    w_map = load_safetensors_selected(snap, wanted_tensors)
    w_u = w_map["lm_head.weight"]
    gamma_fin = w_map["model.norm.weight"]

    W_U_eff = w_u * gamma_fin.unsqueeze(0)
    mu_U = W_U_eff.mean(dim=0)
    mu_U_hat = (mu_U.double() / mu_U.double().norm().clamp_min(1e-12))
    X_c = W_U_eff - mu_U.unsqueeze(0)
    Sigma_U = ((X_c.T @ X_c) / float(X_c.shape[0])).double()
    evals, evecs = torch.linalg.eigh(Sigma_U)
    evals = torch.clamp(evals, min=1e-9)
    tr_cov = float(evals.sum().item()) / 4096.0
    W_shrink_half = evecs @ torch.diag(1.0 / torch.sqrt(evals + COV_SHRINK_ALPHA * tr_cov)) @ evecs.T
    del X_c, Sigma_U

    ids_t = torch.tensor([t["id"] for t in targets], dtype=torch.long)
    U = W_U_eff[ids_t].float()
    U_norm = U.norm(dim=-1)
    U_hat = (U / U_norm.unsqueeze(-1)).float()
    U_ctr = (U.double() - mu_U.double().unsqueeze(0)).float()
    U_shrink = (W_shrink_half @ U_ctr.double().T).T.float()
    del W_U_eff, w_u

    strata = [t["stratum"] for t in targets]
    splits = [t["split"] for t in targets]
    for i, t in enumerate(targets):
        t["u_norm"] = round(float(U_norm[i].item()), 4)

    def pair_cos_summary(mat: torch.Tensor) -> dict[str, float]:
        mat_n = mat.double() / mat.double().norm(dim=-1, keepdim=True).clamp_min(1e-12)
        G = (mat_n @ mat_n.T).numpy()
        within, cross, of_v = [], [], []
        for i in range(24):
            for j in range(24):
                if i == j:
                    continue
                val = float(G[i, j])
                (within if strata[i] == strata[j] else cross).append(val)
                if (strata[i] == "function") != (strata[j] == "function"):
                    of_v.append(val)
        return {
            "within_pos": round(float(np.mean(within)), 4),
            "cross_pos": round(float(np.mean(cross)), 4),
            "open_x_func": round(float(np.mean(of_v)), 4),
        }

    unembed_cos = {
        "raw_u": pair_cos_summary(U),
        "centered_u_minus_mu": pair_cos_summary(U_ctr),
        "shrink_0.1_whitened": pair_cos_summary(U_shrink),
    }

    rl = torch.load(str(r_lens_path), map_location="cpu", weights_only=False)
    R_out_all = rl["means"]["out"]
    R_mid_all = rl["means"]["mid"]

    bank_tensors: dict[str, torch.Tensor] = {
        "mu_u": mu_U.float().clone().contiguous(),
        "u_vectors": U.clone().contiguous(),
        "u_hat": U_hat.clone().contiguous(),
        "u_centered": U_ctr.clone().contiguous(),
        "u_shrink": U_shrink.clone().contiguous(),
    }

    layer_geom: dict[str, Any] = {}
    prenorm_geom: dict[str, Any] = {}
    c22_l18_pred_ret: dict[str, Any] = {}
    U_d = U.double()
    U_hat_d = U_hat.double()
    U_ctr_d = U_ctr.double()
    U_shrink_d = U_shrink.double()
    eye_d = torch.eye(4096, dtype=torch.float64)

    cross_mask = np.array([[strata[i] != strata[j] for j in range(24)] for i in range(24)])
    same_mask = np.array([[i != j and strata[i] == strata[j] for j in range(24)] for i in range(24)])
    heldout_cross_mask = np.array([
        [splits[i] == "primary_16" and splits[j] == "heldout_8" and strata[i] != strata[j] for j in range(24)]
        for i in range(24)
    ])
    u_mu_proj = (U_d @ mu_U_hat).numpy()
    u_mu_proj_n = u_mu_proj / (np.linalg.norm(u_mu_proj) + 1e-12)

    for L in SWEEP_LAYERS:
        R = R_out_all[L].double()
        d = R.shape[1]
        lam = 0.5 * float((R * R).sum().item()) / d
        if str(L) in c0["layer_geometry"]:
            c0_lam = float(c0["layer_geometry"][str(L)]["lambda_ridge"])
            assert abs(lam - c0_lam) < 1e-2, f"L{L} lambda mismatch vs C0: {lam} vs {c0_lam}"

        bank_tensors[f"L{L}.rand_iso"] = make_rand_iso_draws(d, M_RAND_ISO_DRAWS, BANK_RAND_SEED_BASE + L)

        if L in PRIMARY_C21_LAYERS:
            W_mf = U_d @ R
            W_mf_hat = W_mf / W_mf.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            A = R.T @ R + lam * eye_d

            def solve_unit_ridge(src_mat: torch.Tensor) -> torch.Tensor:
                rhs = src_mat @ R
                w = torch.linalg.solve(A, rhs.T).T
                return w / w.norm(dim=-1, keepdim=True).clamp_min(1e-12)

            W_rg_raw = solve_unit_ridge(U_d)
            W_rg_ctr = solve_unit_ridge(U_ctr_d)
            W_rg_shk = solve_unit_ridge(U_shrink_d)

            bank_tensors[f"L{L}.R_mf"] = W_mf_hat.float().clone().contiguous()
            bank_tensors[f"L{L}.R_ridge_raw"] = W_rg_raw.float().clone().contiguous()
            bank_tensors[f"L{L}.R_ridge_centered"] = W_rg_ctr.float().clone().contiguous()
            bank_tensors[f"L{L}.R_ridge_shrink"] = W_rg_shk.float().clone().contiguous()

            # Only L18 uses direct_matched (C2.2, 12 targets, K_DM_DRAWS=1 — 100% draw-level 1-to-1 per m4!)
            if L == 18:
                c_vals = (U_hat_d * W_mf_hat).sum(dim=-1)
                dm_rows = []
                for ti in C2_2_TARGET_INDICES:
                    dm_w = make_direct_matched_single(
                        U_hat[ti], W_mf_hat[ti].float(), W_rg_raw[ti].float(),
                        float(c_vals[ti].item()), BANK_DM_SEED_BASE + L * 1000 + ti * 10,
                    )
                    dm_rows.append(dm_w)
                    # B1/B2 self-contained frozen linear pred_ret & u.w_Rmf for C2.2
                    u_dot_rmf = float(torch.dot(U_d[ti], W_mf_hat[ti]).item())
                    u_dot_dm = float(torch.dot(U_d[ti], dm_w.double()).item())
                    assert u_dot_rmf > 0.5 and abs(u_dot_rmf - u_dot_dm) < 1e-4
                    pr_rmf = float((U_d[ti] @ (R @ W_mf_hat[ti])).item()) / u_dot_rmf
                    pr_dm = float((U_d[ti] @ (R @ dm_w.double())).item()) / u_dot_dm
                    c22_l18_pred_ret[targets[ti]["string"]] = {
                        "target_idx": ti,
                        "u_dot_w_Rmf": round(u_dot_rmf, 6),
                        "pred_ret_R_mf": round(pr_rmf, 6),
                        "pred_ret_dm0": round(pr_dm, 6),
                        "pred_gap_at_direct3": round(3.0 * (pr_rmf - pr_dm), 6),
                    }
                bank_tensors["L18.direct_matched"] = torch.stack(dm_rows, dim=0).clone().contiguous()  # [12, 4096]

            frozen_arms: dict[str, Any] = {}
            for aname, What in [
                ("R_mf", W_mf_hat),
                ("R_ridge_raw", W_rg_raw),
                ("R_ridge_centered", W_rg_ctr),
                ("R_ridge_shrink", W_rg_shk),
                ("logit_lens", U_hat_d),
            ]:
                M = (What @ (R.T @ U_d.T)).numpy()
                diag = np.diag(M)
                M_rel = M / np.maximum(diag[:, None], 1e-9)
                Off = M.copy()
                np.fill_diagonal(Off, 0.0)
                _, svals, vt_s = np.linalg.svd(Off, full_matrices=False)
                rank1_share = float((svals[0] ** 2) / (np.sum(svals ** 2) + 1e-12))
                v1_mu_cos = float(abs(np.dot(vt_s[0], u_mu_proj_n)))
                cross_abs = float(np.mean(np.abs(M_rel[cross_mask])))
                cross_sig = float(np.mean(M_rel[cross_mask]))
                same_abs = float(np.mean(np.abs(M_rel[same_mask])))
                ho_cross_abs = float(np.mean(np.abs(M_rel[heldout_cross_mask])))
                frozen_arms[aname] = {
                    "diag_unit_gain_mean": round(float(np.mean(diag)), 4),
                    "cross_abs_over_diag": round(cross_abs, 4),
                    "cross_signed_over_diag": round(cross_sig, 4),
                    "heldout_16x8_cross_abs_over_diag": round(ho_cross_abs, 4),
                    "same_over_cross": round(same_abs / max(cross_abs, 1e-9), 4),
                    "offdiag_rank1_svd_share": round(rank1_share, 4),
                    "abs_cos_v1_right_mu_u": round(v1_mu_cos, 4),
                }

            layer_geom[str(L)] = {
                "role": "primary_C2.1_and_sweep",
                "lambda_ridge": round(lam, 6),
                "trace_R_over_d": round(float(torch.trace(R).item() / d), 4),
                "frozen_lens_transfer_preds_24": frozen_arms,
            }
        else:
            layer_geom[str(L)] = {
                "role": "sweep_layer",
                "lambda_ridge": round(lam, 6),
                "trace_R_over_d": round(float(torch.trace(R).item() / d), 4),
            }

        # Build Lacey low-gamma pre_norm vectors ONLY for PRENORM_LAYERS = [21, 24, 27] (100% 1-to-1 with C2.4 arms!)
        if L in PRENORM_LAYERS:
            W_out_shk = torch.linalg.solve(R.T @ R + lam * eye_d, (U_shrink_d @ R).T).T
            W_out_shk = W_out_shk / W_out_shk.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            g_ff = w_map[f"model.layers.{L}.post_feedforward_layernorm.weight"].double()

            def build_masked_prenorm(v_post: torch.Tensor, g_vec: torch.Tensor, drop_k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
                order = torch.argsort(g_vec.abs())
                mask_01 = torch.ones_like(g_vec)
                mask_01[order[:drop_k]] = 0.0
                g_rms = float(torch.sqrt(torch.mean(g_vec ** 2)).item())
                ideal_pre = 1.0 / g_rms
                g_safe = torch.where(g_vec.abs() < 1e-12, torch.full_like(g_vec, 1e-12), g_vec)
                pre_un = v_post / g_safe.unsqueeze(0)
                un_eff = float(torch.mean(ideal_pre / pre_un.norm(dim=-1)).item())
                pre_un_u = pre_un / pre_un.norm(dim=-1, keepdim=True).clamp_min(1e-12)

                v_m = v_post * mask_01.unsqueeze(0)
                v_m_u = v_m / v_m.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                cos_ret = float(torch.mean(torch.sum(v_post * v_m_u, dim=-1)).item())
                active = order[drop_k:]
                pre_m = torch.zeros_like(v_m_u)
                pre_m[:, active] = v_m_u[:, active] / g_vec[active].unsqueeze(0)
                m_norms = pre_m.norm(dim=-1)
                m_eff = float(torch.mean(ideal_pre / m_norms).item())
                pre_m_u = pre_m / m_norms.unsqueeze(-1).clamp_min(1e-12)
                assert float(pre_m_u[:, order[:drop_k]].abs().max().item()) == 0.0
                return mask_01, pre_un_u, pre_m_u, {
                    "gamma_rms": round(g_rms, 5),
                    "gamma_min_abs": round(float(g_vec.abs().min().item()), 6),
                    "drop_k": drop_k,
                    "unmasked_deliverable_frac": round(un_eff, 4),
                    "unmasked_worst_blowup": round(float(torch.max(pre_un.norm(dim=-1) / ideal_pre).item()), 2),
                    "masked_deliverable_frac": round(m_eff, 4),
                    "masked_worst_blowup": round(float(torch.max(m_norms / ideal_pre).item()), 2),
                    "masked_over_unmasked_ratio": round(m_eff / max(un_eff, 1e-9), 3),
                    "masked_cos_ret": round(cos_ret, 5),
                }

            m_ff, pre_ff_un, pre_ff_m, ff_stats = build_masked_prenorm(W_out_shk, g_ff, GAMMA_DROP_FF)
            bank_tensors[f"L{L}.gamma_ff_mask4"] = m_ff.float().clone().contiguous()
            bank_tensors[f"L{L}.prenorm_ff_masked"] = pre_ff_m.float().clone().contiguous()
            layer_pn: dict[str, Any] = {"ff": ff_stats}

            # Only L24 carries unmasked, attn masked, and L24 ff/attn random controls (exact 1-to-1 with C2.4b M4!)
            if L == 24:
                R_mid = R_mid_all[L].double()
                lam_m = 0.5 * float((R_mid * R_mid).sum().item()) / d
                W_mid_shk = torch.linalg.solve(R_mid.T @ R_mid + lam_m * eye_d, (U_shrink_d @ R_mid).T).T
                W_mid_shk = W_mid_shk / W_mid_shk.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                g_at = w_map["model.layers.24.post_attention_layernorm.weight"].double()
                m_at, _, pre_at_m, at_stats = build_masked_prenorm(W_mid_shk, g_at, GAMMA_DROP_ATTN)
                bank_tensors["L24.prenorm_ff_unmasked"] = pre_ff_un.float().clone().contiguous()
                bank_tensors["L24.prenorm_ff_rand_masked"] = make_masked_rand_single(
                    m_ff, 24, BANK_PRENORM_RAND_SEED_BASE + L * 1000
                )
                bank_tensors["L24.gamma_attn_mask16"] = m_at.float().clone().contiguous()
                bank_tensors["L24.prenorm_attn_masked"] = pre_at_m.float().clone().contiguous()
                bank_tensors["L24.prenorm_attn_rand_masked"] = make_masked_rand_single(
                    m_at, 24, BANK_PRENORM_RAND_SEED_BASE + L * 1000 + 500
                )
                layer_pn["attn"] = at_stats
                layer_pn["m70a30_linear_delivered_ratio_vs_ff"] = round(
                    (0.7 * ff_stats["masked_deliverable_frac"] + 0.3 * at_stats["masked_deliverable_frac"])
                    / ff_stats["masked_deliverable_frac"], 4
                )
            prenorm_geom[str(L)] = layer_pn

    out_bank = args.out_bank
    out_bank.parent.mkdir(parents=True, exist_ok=True)
    bank_meta = {
        "study": "OLMo-3-7B Stage C2 v4",
        "model": MODEL_ID,
        "revision": MODEL_REV,
        "r_lens_sha256": r_sha,
        "bank_dm_seed_base": str(BANK_DM_SEED_BASE),
        "bank_rand_seed_base": str(BANK_RAND_SEED_BASE),
        "bank_prenorm_rand_seed_base": str(BANK_PRENORM_RAND_SEED_BASE),
        "k_dm_draws": str(K_DM_DRAWS),
        "m_rand_iso_draws": str(M_RAND_ISO_DRAWS),
        "k_prenorm_rand_draws": str(K_PRENORM_RAND_DRAWS),
        "gamma_drop_ff": str(GAMMA_DROP_FF),
        "gamma_drop_attn": str(GAMMA_DROP_ATTN),
        "cov_shrink_alpha": str(COV_SHRINK_ALPHA),
    }
    write_canonical_safetensors(bank_tensors, out_bank, metadata=bank_meta)
    bank_sha = sha256_file(out_bank)

    # Exact enumerated shams split into write shams (dose=0.0) & patch shams (self-patch identity, m3)
    write_shams_13 = [
        "C2.1_post_norm_L18", "C2.1_post_norm_L24", "C2.1_post_norm_L27",
        "C2.3_sweep_L18", "C2.3_sweep_L21", "C2.3_sweep_L24", "C2.3_sweep_L27", "C2.3_sweep_L30",
        "C2.4_prenorm_L21_ff", "C2.4_prenorm_L24_ff", "C2.4_prenorm_L27_ff",
        "C2.4_prenorm_L24_m70a30", "C2.4_prenorm_K3_combined",
    ]
    patch_shams_6 = [
        "C2.2_self_patch_ChatML_MLP_L19_31",
        "C2.2_self_patch_LongBare_MLP_L19_31",
        "C2.2_self_patch_ctrl_L4_16_MLP",
        "C2.2_self_patch_ctrl_mismatch_L19_31",
        "C2.2_zero_scale_ctrl_rand_iso_L19_31",
        "C2.2_unit_scale_ctrl_self_attenuated_L19_31",
    ]
    c22_arms_9 = [
        "unpatched_LongBare",
        "unpatched_ChatML",
        "unpatched_ShortBare",
        "patch_LongBare_to_ChatML_MLP_L19_31",
        "patch_ChatML_to_LongBare_MLP_L19_31",
        "ctrl_L4_16_MLP_LongBare_to_ChatML",
        "ctrl_mismatch_prompt_MLP_L19_31",
        "ctrl_rand_iso_same_site_L19_31",
        "ctrl_self_attenuated_L19_31",
    ]
    c24_arms_9 = [
        "L24.prenorm_ff_unmasked",
        "L24.prenorm_ff_unmasked_postmatched",
        "L24.prenorm_ff_masked",
        "L24.m70a30 (0.7*L24.prenorm_ff_masked + 0.3*L24.prenorm_attn_masked)",
        "L24.prenorm_ff_rand_masked",
        "L24.m70a30_rand_masked (0.7*L24.ff_rand + 0.3*L24.attn_rand)",
        "L21.prenorm_ff_masked",
        "L27.prenorm_ff_masked",
        "K3_combined (L21+L24+L27 prenorm_ff_masked)",
    ]

    # Exact integer forward counts (0% unread forwards!):
    # Clean (36 = 12*3) + Write Shams (13) + Patch Shams (6) + C0-colon pairs (16) = 71
    # Block 1 (C2.1): 3L x 5 arms x 24 targets x 8 prompts = 2,880
    # Block 2 (C2.2): 9 conds x 12 targets x 12 pairs x 2 (R_mf + dm0) = 2,592, plus 6 patched conds x 12 pairs = 72 unsteered clean-KL fwds -> 2,664
    # Block 2b (C2.3): 5L x (24 logit_lens + 8 rand_iso) x 8 prompts = 1,280
    # Block 3 (C2.4): 9 arms x 24 targets x 4 prompts x 2 doses = 1,728
    n_clean = len(c22_triples) * 3
    n_shams = len(write_shams_13) + len(patch_shams_6)
    n_clean_sham_anchor = n_clean + n_shams + 16
    n_b1 = len(PRIMARY_C21_LAYERS) * 5 * 24 * len(prompts_8_c21)
    n_b2_steered = len(c22_arms_9) * len(C2_2_TARGET_INDICES) * len(c22_triples) * 2
    n_b2_clean_kl = 6 * len(c22_triples)
    n_b2 = n_b2_steered + n_b2_clean_kl
    n_b2b = len(SWEEP_LAYERS) * (24 + M_RAND_ISO_DRAWS) * len(prompts_8_c21)
    n_b3 = len(c24_arms_9) * 24 * 4 * 2

    budget = {
        "clean_prompts_36": n_clean,
        "write_shams_13": len(write_shams_13),
        "patch_shams_6": len(patch_shams_6),
        "shams_total_19": n_shams,
        "c0_colon_anchor_pairs_16": 16,
        "clean_shams_anchor_subtotal": n_clean_sham_anchor,
        "block1_C2_1_3L_5arms_24x8": n_b1,
        "block2_C2_2_steered_9cond_12t_12p_2arms": n_b2_steered,
        "block2_C2_2_unsteered_patched_kl_6cond_12p": n_b2_clean_kl,
        "block2_C2_2_total": n_b2,
        "block2b_C2_3_5L_32v_8p": n_b2b,
        "block3_C2_4_9arms_24t_4p_2doses": n_b3,
        "total_planned_short_fwds": n_clean_sham_anchor + n_b1 + n_b2 + n_b2b + n_b3,
        "write_shams_list": write_shams_13,
        "patch_shams_list": patch_shams_6,
        "c22_conditions_list": c22_arms_9,
        "c24_arms_list": c24_arms_9,
        "estimated_runtime_L4_or_A100_min": "8-13",
        "estimated_cost_CU": "0.45-1.05",
    }

    # Unified 7-test reject-the-null Holm-Bonferroni family (B5 & F3: C2.4b moved to descriptive estimation;
    # equivalence & calibration gates live in registered_gates_outside_holm!)
    holm_tests = [
        "p_C2.1_ctr_vs_raw",
        "p_C2.1_shk_vs_ctr",
        "p_C2.1_shk_vs_ll",
        "p_C2.1_gain_ret",
        "p_C2.2_bare_into_chat",
        "p_C2.2_chat_into_bare",
        "p_C2.4a_mask_vs_unmask",
    ]
    holm_stepdown = [round(0.05 / (len(holm_tests) + 1 - k), 6) for k in range(1, len(holm_tests) + 1)]

    # F1: Formal on-model definitions for every symbol referenced in Holm rows 1-7 (verified by stdlib suite!)
    holm_symbol_definitions = {
        "Delta_logit_a(j; i, L, p)": "logit_int(j) - logit_clean(j) on token j when injecting arm a's unit write for source target i at layer L, positions='last', dose=0.75*pmn[L] on prompt p",
        "diag_a(i)": "Equal-weight layer mean over L in {18, 24, 27} of diag_a(i, L) = mean_{p in 8}[Delta_logit_a(i; i, L, p)]",
        "diag_shk": "diag_a(i) for arm a = R_ridge_shrink",
        "diag_raw": "diag_a(i) for arm a = R_ridge_raw",
        "ho_cross_a(i)": "Strictly ABSOLUTE ratio-of-means: equal-weight layer mean over L in {18, 24, 27} of (mean_{p in 8, j in heldout_8: strata(j)!=strata(i)} |Delta_logit_a(j; i, L, p)|) / max(|diag_a(i, L)|, 0.25), floored below by 1e-6 before log. Guard rule: if >2 of 16 primary_16 sources have |diag_a(i, L)| <= 0.25 in any arm, fail closed as C2.1=DEGENERATE_DIAG.",
        "ho_cross_raw": "ho_cross_a(i) for arm a = R_ridge_raw",
        "ho_cross_ctr": "ho_cross_a(i) for arm a = R_ridge_centered",
        "ho_cross_shk": "ho_cross_a(i) for arm a = R_ridge_shrink",
        "ho_cross_ll": "ho_cross_a(i) for arm a = logit_lens",
        "same_over_cross_a": "Ratio of means over 24 targets: mean_i[mean_{L}(mean_{p, j!=i, same_stratum}|Delta_logit_a| / max(|diag_a(i,L)|,0.25))] / mean_i[mean_{L}(mean_{p, diff_stratum}|Delta_logit_a| / max(|diag_a(i,L)|,0.25))]",
        "d_i(0.50)": "Denominator-free paired closure margin on stem pair i in {1..12}: s * [(M(ctrl, i) - M(patched, i)) - 0.50 * (M(ctrl, i) - M(ref, i))], where s = sign(G_gap)",
        "Delta_logit_masked": "Prompt-and-dose-averaged exact target logit shift Delta_logit(t) for L24.prenorm_ff_masked at matched pre-norm write norm (pooled 0.50x & 1.25x live pmn)",
        "Delta_logit_unmasked": "Prompt-and-dose-averaged exact target logit shift Delta_logit(t) for L24.prenorm_ff_unmasked at exact same pre-norm write norm as masked (pooled 0.50x & 1.25x live pmn)",
    }

    # F4: Exact math.ceil(C1_count * N / 1728) derivation for BOTH Regime-N and Regime-D!
    n_reg_n = n_b1
    n_reg_d = 3 * len(C2_2_TARGET_INDICES) * len(c22_triples) * 2  # 864 unpatched C2.2 traces
    c203_gates = {
        "derivation_rule": "Exact math.ceil(c1_count * N / 1728) applied uniformly to BOTH Regime-N and Regime-D from C1 registered counts (2/1728 >20% shrink, 26/1728 >12% shrink, 12/1728 |fde_err|>0.45)",
        "c1_base_N": C1_REG_N,
        "c1_count_rmsn_gt_20pct": C1_REG_COUNT_RMSN_GT_20PCT,
        "c1_count_rmsn_gt_12pct": C1_REG_COUNT_RMSN_GT_12PCT,
        "c1_count_fde_gt_045": C1_REG_COUNT_FDE_GT_045,
        "regime_N_C21_population_N": n_reg_n,
        "regime_N_max_capped": 0,
        "regime_N_max_rmsn_gt_20pct": math.ceil(C1_REG_COUNT_RMSN_GT_20PCT * n_reg_n / C1_REG_N),
        "regime_N_max_rmsn_gt_12pct": math.ceil(C1_REG_COUNT_RMSN_GT_12PCT * n_reg_n / C1_REG_N),
        "regime_D_C22_unpatched_population_N": n_reg_d,
        "regime_D_max_capped": 0,
        "regime_D_max_rmsn_gt_20pct": math.ceil(C1_REG_COUNT_RMSN_GT_20PCT * n_reg_d / C1_REG_N),
        "regime_D_max_rmsn_gt_12pct": math.ceil(C1_REG_COUNT_RMSN_GT_12PCT * n_reg_d / C1_REG_N),
        "regime_D_fde_p95_max": 0.30,
        "regime_D_max_fde_gt_045": math.ceil(C1_REG_COUNT_FDE_GT_045 * n_reg_d / C1_REG_N),
    }

    l24_ratio_pt = round(C1_RECEIPT_PMN["24"] / C0_RECEIPT_PMN["24"], 4)
    l27_ratio_pt = round(C1_RECEIPT_PMN["27"] / C0_RECEIPT_PMN["27"], 4)

    doc = {
        "schema_version": "c2-prereg-constants-v4",
        "study": "OLMo-3-7B Stage C2 v4 (Held-Out Whitening Arm Comparison & Frozen-Lens Calibration, Pure ChatML-Template Causal MLP Patching, Post-Erasure Sweep, & Pre-Norm Mask/Additivity)",
        "generator_script": "olmo3_virtual_and_highway/build_olmo3_c2_constants.py",
        "model": MODEL_ID,
        "revision": MODEL_REV,
        "r_lens_sha256": r_sha,
        "c2_vector_bank_sha256": bank_sha,
        "targets_24_spec_sha256": targets_spec_sha,
        "parent_c0_constants_sha256": sha256_file(C0_CONST_PATH),
        "parent_c0b_constants_sha256": sha256_file(C0B_CONST_PATH),
        "parent_c1_constants_sha256": sha256_file(C1_CONST_PATH),
        "primary_c21_layers": PRIMARY_C21_LAYERS,
        "sweep_layers": SWEEP_LAYERS,
        "prenorm_layers": PRENORM_LAYERS,
        "c22_target_indices_12": C2_2_TARGET_INDICES,
        "c22_l18_frozen_pred_ret": c22_l18_pred_ret,
        "holm_symbol_definitions": holm_symbol_definitions,
        "registered_rng_and_inference": {
            "holm_family_size": len(holm_tests),
            "holm_tests": holm_tests,
            "holm_alpha_stepdown": holm_stepdown,
            "holm_stop_rule": "sort ascending p_(1)..p_(7); reject k while p_(k) <= 0.05/(8-k); stop at first failure (all 7 tests are one-sided reject-the-null; equivalence & calibration gates live separately; C2.4b is descriptive estimation per F3)",
            "registered_gates_outside_holm": [
                "CAL-C2.1_absolute_lens_transfer_and_mu_U_mechanism",
                "EQ-C2.2_negative_controls_and_shape",
                "EQ-C2.4c_K3_median_additivity_TOST",
            ],
            "secondary_descriptive_readouts": [
                "C2.1_five_arm_ordering",
                "C2.4a_unmasked_postmatched_nonlinear_saturation",
                "C2.4b_m70a30_high_dose_estimation_and_packing_contrast",
                "C2.4c_1.25x_high_dose_interaction_calibration",
            ],
            "perm_rng": "Python stdlib random.Random(perm_seed)",
            "perm_seed": PERM_SEED,
            "perm_rule_16_primary_targets": "Exact enumeration over all 2^16 = 65,536 sign flips (p_min = 1/65536 = 0.000015); per-prompt breakdowns reported alongside (F6)",
            "perm_n_flips_24_targets": 100000,
            "perm_rule_12_stem_pairs": "Exact one-sided enumeration over all 2^12 = 4,096 sign flips on d_i(0.50) (12/12 -> p=0.000244, 11/12 -> p=0.003174 < alpha/7=0.007143)",
            "layer_aggregation": "Equal-weight mean over L in {18, 24, 27} per target before target-level inference",
            "bca_rng": "Python stdlib random.Random(bca_seed)",
            "bca_seed": BCA_SEED,
            "bca_n_resamples": 10000,
            "bca_containment_rule": "90% BCa CI must satisfy CI_low <= point_estimate <= CI_high (fail-closed scorer error)",
            "bank_dm_seed_base": BANK_DM_SEED_BASE,
            "bank_rand_seed_base": BANK_RAND_SEED_BASE,
            "bank_prenorm_rand_seed_base": BANK_PRENORM_RAND_SEED_BASE,
        },
        "c2_0_3_named_gate_populations": c203_gates,
        "c0_anchor_colon_prediction": {
            "hypothesis": "Paired in-run ratio test: replacing trailing ' a' with ':' drops L24/L27 post_mlp_norm (Claude #103(1c), m1, m2, m6)",
            "c0_receipt_pmn_provenance": "c0_receipt.json.gz post_mlp_norms (NOT c0_prereg_constants Exp 4B literal 12.1555)",
            "c1_receipt_pmn_provenance": "results/vesper-olmo3-7b-stage-c1/c1_receipt.json.gz post_mlp_norms",
            "c0_receipt_pmn": C0_RECEIPT_PMN,
            "c1_receipt_pmn": C1_RECEIPT_PMN,
            "predicted_ratio_L24_point": l24_ratio_pt,
            "predicted_ratio_L24_range": [0.70, 0.85],
            "predicted_ratio_L27_point": l27_ratio_pt,
            "predicted_ratio_L27_range": [0.65, 0.82],
            "length_confound_note": "Bare pairs preserve exact token count (9->9, 7->7, 13->13, 10->10); ChatML pairs drop 1 token (' a...' -> ':'). Report Bare-only ratio beside pooled ratio.",
            "prompts_16": c0_colon_prompts,
        },
        "targets_24": targets,
        "prompts_8_c21": prompts_8_c21,
        "c22_stem_pairs_12": c22_triples,
        "prompt_length_summary": {
            "short_bare_min_max_mean": [min(bare_T), max(bare_T), round(float(np.mean(bare_T)), 2)],
            "long_bare_min_max_mean": [min(long_T), max(long_T), round(float(np.mean(long_T)), 2)],
            "chatml_min_max_mean": [min(chat_T), max(chat_T), round(float(np.mean(chat_T)), 2)],
            "max_abs_longbare_minus_chatml_tokens": max(abs(a - b) for a, b in zip(long_T, chat_T)),
        },
        "unembed_pairwise_cosine": unembed_cos,
        "layer_geometry": layer_geom,
        "prenorm_gamma_mask_geometry": prenorm_geom,
        "c2_3_post_erasure_decision_rule": {
            "sum_mlp_definition": "sum_{l=L+1}^{31} (u_t . (mlp_out_l^{int}(last) - mlp_out_l^{clean}(last))) / r_c(p), evaluated on logit_lens arm at layer L, positions='last', dose=0.75*pmn[L]",
            "statistic": "E_open(L) = median_{t in open_18, p in 8}(sum_mlp(L,t,p)); E_func(L) = median_{t in func_6, p in 8}(sum_mlp(L,t,p)); ret_open(L) = median_{t in open_18, p in 8}(retention(L,t,p)); ret_rand(L) = median_{d in 8, p in 8}(retention_rand_iso(L,d,p))",
            "threshold_provenance": C1_LOGIT_LENS_SUM_MLP_PROVENANCE,
            "post_erasure_predicate": "E_open(L) >= -1.20 logits AND |E_open(L) - E_func(L)| <= 0.60 logits AND ret_open(L) > ret_rand(L)",
            "selection_rule": "Stage C3 compilation band = earliest contiguous run of >= 2 sweep layers in {18, 21, 24, 27, 30} satisfying post_erasure(L); if none, C2.3 = INCONCLUSIVE and Stage C3 defaults to {24, 27}",
        },
        "power_simulations": simulate_c2_power(),
        "planned_budget": budget,
    }

    out_const = args.out_constants
    out_const.parent.mkdir(parents=True, exist_ok=True)
    out_const.write_text(json.dumps(doc, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    const_sha = sha256_file(out_const)

    print(f"Wrote {out_const} (sha256={const_sha})")
    print(f"Wrote {out_bank} (sha256={bank_sha}, tensors={len(bank_tensors)})")
    print(f"Prompt lengths: {doc['prompt_length_summary']}")
    print(f"C2.0.3 gates: {c203_gates}")
    print(f"Budget total: {budget['total_planned_short_fwds']} fwds")
    print(f"Power summary: {json.dumps(doc['power_simulations'], indent=2)}")


if __name__ == "__main__":
    main()
