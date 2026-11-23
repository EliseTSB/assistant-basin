"""Unit & end-to-end tests for Stage C0 (`build_olmo3_c0_bank.py`, `run_olmo3_c0.py`, `score_olmo3_c0.py`).

Includes dedicated fail-closed regression tests for every C0-CODE-REVIEW-ds-r1 finding (H1-H3, M1-M4, L1-L6).
"""
from __future__ import annotations

import copy
import gzip
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "olmo3_virtual_and_highway"))

import score_olmo3_c0 as sc0  # noqa: E402

CONSTANTS_PATH = REPO_ROOT / "olmo3_virtual_and_highway/c0_prereg_constants.json"
DESIGN_PATH = REPO_ROOT / "olmo3_virtual_and_highway/OLMO3-TRANSPORT-C0-DESIGN.md"


# ---------------------------------------------------------------------------
# 1. Pure-stdlib statistical primitives & fail-closed guards (M1, M3, H3)
# ---------------------------------------------------------------------------
def test_quantile_median_mean_fail_closed_on_empty_and_nonfinite():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert sc0.quantile(xs, 0.0) == 1.0
    assert sc0.quantile(xs, 0.5) == 3.0
    assert sc0.quantile(xs, 1.0) == 5.0
    assert abs(sc0.quantile(xs, 0.25) - 2.0) < 1e-12
    assert abs(sc0.quantile([10.0, 20.0], 0.95) - 19.5) < 1e-12

    # M3: empty lists must raise
    with pytest.raises(ValueError):
        sc0.mean([])
    with pytest.raises(ValueError):
        sc0.median([])
    with pytest.raises(ValueError):
        sc0.quantile([], 0.5)

    # M1: non-finite values must raise
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            sc0.mean([1.0, bad])
        with pytest.raises(ValueError):
            sc0.median([1.0, bad])
        with pytest.raises(ValueError):
            sc0.quantile([1.0, 2.0, bad], 0.95)


def test_norm_ppf_accuracy():
    for p in [1e-4, 0.001, 0.0125, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.999]:
        z = sc0.norm_ppf(p)
        p_back = sc0.norm_cdf(z)
        assert abs(p_back - p) < 1e-8, f"p={p}, z={z}, p_back={p_back}"


def test_exact_sign_flip_matches_bruteforce_and_rejects_nonfinite():
    vals = [0.12, -0.03, 0.25, 0.08, 0.19, -0.01, 0.14, 0.05]
    p_fast = sc0.exact_sign_flip_p_one_sided(vals)
    obs = sum(vals)
    ge = 0
    for mask in range(1 << len(vals)):
        s = sum(vals[i] if (mask >> i) & 1 else -vals[i] for i in range(len(vals)))
        if s >= obs - 1e-12:
            ge += 1
    p_brute = ge / (1 << len(vals))
    assert abs(p_fast - p_brute) < 1e-12
    p16 = sc0.exact_sign_flip_p_one_sided([1.0] * 16)
    assert abs(p16 - (1.0 / 65536.0)) < 1e-12

    # H3 regression: non-finite at ANY index returns NaN (never p_min!)
    for bad in (float("nan"), float("inf"), float("-inf")):
        for idx in (0, 5, 15):
            arr = [0.1] * 16
            arr[idx] = bad
            assert math.isnan(sc0.exact_sign_flip_p_one_sided(arr))
    assert math.isnan(sc0.exact_sign_flip_p_one_sided([]))


def test_holm_bonferroni_stepdown_and_nan():
    h = sc0.holm_bonferroni({"A": 0.001, "B": 0.014, "C": 0.020, "D": 0.10}, alpha=0.05)
    assert h["A"]["rejected"] is True
    assert h["B"]["rejected"] is True
    assert h["C"]["rejected"] is True
    assert h["D"]["rejected"] is False
    h2 = sc0.holm_bonferroni({"A": 0.001, "B": 0.020, "C": 0.021, "D": 0.10}, alpha=0.05)
    assert h2["A"]["rejected"] is True
    assert h2["B"]["rejected"] is False
    assert h2["C"]["rejected"] is False
    # H3: NaN p-value never rejects
    h3 = sc0.holm_bonferroni({"A": float("nan"), "B": 0.001}, alpha=0.05)
    assert h3["A"]["rejected"] is False
    assert h3["A"]["p_raw"] is None
    assert h3["B"]["rejected"] is True


def test_spearman_rho_and_ties():
    assert abs(sc0.spearman_rho([1, 2, 3, 4, 5], [2, 4, 6, 8, 10]) - 1.0) < 1e-12
    assert abs(sc0.spearman_rho([1, 2, 3, 4, 5], [10, 8, 6, 4, 2]) + 1.0) < 1e-12
    assert sc0.spearman_rho([1, 1, 1, 1], [1, 2, 3, 4]) == 0.0


def test_bca_deterministic_and_sensible():
    xs = [1.0 + 0.1 * i for i in range(16)]
    ci1 = sc0.bca_cluster_ci(16, lambda idx: sc0.mean([xs[i] for i in idx]), (0.05, 0.95), n_boot=500, seed=7)
    ci2 = sc0.bca_cluster_ci(16, lambda idx: sc0.mean([xs[i] for i in idx]), (0.05, 0.95), n_boot=500, seed=7)
    assert ci1 == ci2
    assert ci1["q_0.05"] < ci1["point"] < ci1["q_0.95"]


# ---------------------------------------------------------------------------
# 2. Vector bank math invariants
# ---------------------------------------------------------------------------
def test_direct_matched_orthogonality_and_invariants():
    torch = pytest.importorskip("torch")
    import build_olmo3_c0_bank as bank_mod

    g = torch.Generator().manual_seed(99)
    d = 128
    u_hat = bank_mod.unit(torch.randn(d, generator=g))
    r_mat = torch.randn(d, d, generator=g) / math.sqrt(d)
    rmf_raw = r_mat.T @ u_hat
    w_rmf = bank_mod.unit(rmf_raw)
    c = float(torch.dot(u_hat, w_rmf).item())
    if c < 0:
        w_rmf = -w_rmf
        c = -c
    w_rr = bank_mod.unit(torch.randn(d, generator=g))

    draws = bank_mod.make_direct_matched_draws(u_hat, w_rmf, w_rr, c=c, k_draws=4, seed_base=12345)
    assert draws.shape == (4, d)
    for k in range(4):
        w = draws[k].double()
        assert abs(float(w.norm().item()) - 1.0) < 1e-5
        assert abs(float(torch.dot(w, u_hat.double()).item()) - c) < 1e-5
        lhs = float(torch.dot(u_hat.double(), r_mat.double() @ w).item())
        rhs = c * float(torch.dot(u_hat.double(), r_mat.double() @ u_hat.double()).item())
        assert abs(lhs - rhs) < 1e-5


# ---------------------------------------------------------------------------
# 3. Synthetic Stage C0 receipt builder & full scorer regression tests
# ---------------------------------------------------------------------------
def _make_synthetic_receipt(
    mode: str = "pass",
    sham_err: float = 0.0,
    tel_err: float = 0.01,
    cap_rate_inject: bool = False,
    bad_provenance: bool = False,
    single_pair_leak: float = 0.0,
) -> dict:
    const = json.loads(CONSTANTS_PATH.read_text(encoding="utf-8"))
    targets = const["targets_16"]
    layers = const["layers"]
    lg = const["layer_geometry"]
    prompts = [
        {"id": f"{fmt}_{sty}_{i}", "format": fmt, "style": sty, "text": f"prompt {fmt} {sty} {i}"}
        for fmt in ("bare", "chatml")
        for sty in ("narrative", "analytical")
        for i in range(2)
    ]
    rng = random.Random(42)
    k_dm = 4
    m_rand = 32
    rn = {}
    rd = {}
    post_norms = {str(L): 2.0 + 0.4 * L for L in layers}

    for L in layers:
        l_str = str(L)
        dose_n = round(0.75 * post_norms[l_str], 6)
        tr_d = lg[l_str]["trace_R_over_d"]
        rn[l_str] = {
            "dose_N": dose_n,
            "post_mlp_norm": post_norms[l_str],
            "R_mf": {},
            "R_ridge": {},
            "logit_lens": {},
            "direct_matched": {},
            "rand_iso": [],
        }
        rd[l_str] = {
            "target_direct_linear": 3.0,
            "dose_cap": round(3.0 * post_norms[l_str], 6),
            "R_mf": {},
            "R_ridge": {},
            "logit_lens": {},
            "direct_matched": {},
        }
        pts = {p["target"]: p for p in lg[l_str]["per_target"]}

        for ti, t in enumerate(targets):
            ts = t["string"]
            pt = pts[ts]
            u_norm = t["u_norm"]
            for arm in ("R_mf", "R_ridge", "logit_lens", "direct_matched"):
                if arm == "R_mf":
                    pred_ret = pt["pred_ret_Rmf"]
                    dir_n = dose_n * u_norm * pt["c"]
                elif arm == "R_ridge":
                    pred_ret = pt["pred_ret_Rridge"]
                    dir_n = dose_n * u_norm * pt["cos_u_Rridge"]
                elif arm == "logit_lens":
                    pred_ret = pt["pred_ret_dm"]
                    dir_n = dose_n * u_norm
                else:
                    pred_ret = pt["pred_ret_dm"]
                    dir_n = dose_n * u_norm * pt["c"]

                def _mk_trace_n(p_idx: int) -> dict:
                    pred = pred_ret * dir_n
                    if mode == "pass":
                        realized_ret = pred_ret * (0.96 if arm != "logit_lens" else 0.85) + rng.gauss(0.0, 0.03)
                        ex = realized_ret * dir_n + rng.gauss(0.0, 0.05)
                    else:
                        ex = rng.gauss(0.0, 1.0)
                        realized_ret = ex / max(dir_n, 1e-6)
                    rec = {
                        "prompt_id": prompts[p_idx]["id"],
                        "dose": dose_n,
                        "exact_dlogit": ex,
                        "pred_dlogit": pred,
                        "direct_linear": dir_n,
                        "direct_linear_analytic": dir_n,
                        "frozen_direct_effect": dir_n * 0.99,
                        "sum_attn": ex - dir_n,
                        "sum_mlp": 0.0,
                        "rmsnorm_shift": 0.0,
                        "telescoping_err": tel_err,
                        "retention": realized_ret,
                        "canc_r": 1.5,
                        "dlogit_z": 10.0,
                        "kl": 0.5,
                        "rms_clean": 1.0,
                        "rms_int": 1.01,
                    }
                    if arm == "R_mf":
                        all16 = [rng.gauss(0.0, 0.30) for _ in range(16)]
                        all16[ti] = ex
                        if single_pair_leak > 0.0 and ti == 0:
                            all16[4] = single_pair_leak
                        rec["all_16_dlogits"] = all16
                    return rec

                def _mk_trace_d(p_idx: int) -> dict:
                    if cap_rate_inject and L == 12 and arm == "R_mf" and ti < 2:
                        return {"prompt_id": prompts[p_idx]["id"], "capped": True, "dose_requested": 99.0, "dose_cap": 10.0}
                    if mode == "pass":
                        ret_d = pred_ret * (0.95 if arm != "logit_lens" else 0.80) + rng.gauss(0.0, 0.02)
                        ex = ret_d * 3.0
                        dz = 22.0 if arm == "R_ridge" else 15.0
                        kl = 0.25 if arm == "R_ridge" else 0.45
                    else:
                        ret_d = 1.0 + rng.gauss(0.0, 0.2)
                        ex = ret_d * 3.0
                        dz = 10.0
                        kl = 0.5
                    return {
                        "prompt_id": prompts[p_idx]["id"],
                        "capped": False,
                        "dose": 3.0 / max(u_norm * pt["c"], 1e-4),
                        "exact_dlogit": ex,
                        "pred_dlogit": pred_ret * 3.0,
                        "direct_linear": 3.0,
                        "direct_linear_analytic": 3.0,
                        "frozen_direct_effect": 3.0 + rng.gauss(0.0, 0.04),
                        "sum_attn": ex - 3.0,
                        "sum_mlp": 0.0,
                        "rmsnorm_shift": 0.0,
                        "telescoping_err": tel_err,
                        "retention": ret_d,
                        "canc_r": 1.3,
                        "dlogit_z": dz + rng.gauss(0.0, 0.5),
                        "kl": kl,
                        "rms_clean": 1.0,
                        "rms_int": 1.01,
                    }

                if arm == "direct_matched":
                    rn[l_str][arm][ts] = [[_mk_trace_n(pi) for pi in range(8)] for _ in range(k_dm)]
                    rd[l_str][arm][ts] = [[_mk_trace_d(pi) for pi in range(8)] for _ in range(k_dm)]
                else:
                    rn[l_str][arm][ts] = [_mk_trace_n(pi) for pi in range(8)]
                    rd[l_str][arm][ts] = [_mk_trace_d(pi) for pi in range(8)]

        for _m in range(m_rand):
            rows_p = []
            for pi in range(8):
                vjv = (tr_d + 0.02) if mode == "pass" else 0.0
                rows_p.append({
                    "prompt_id": prompts[pi]["id"],
                    "dose": dose_n,
                    "vJv": vjv + rng.gauss(0.0, 0.02),
                    "telescoping_err": tel_err,
                    "canc_r_self": 2.0,
                    "all_16_dlogits": [rng.gauss(0.0, 0.30) for _ in range(16)],
                })
            rn[l_str]["rand_iso"].append(rows_p)

    return {
        "meta": {
            "study": "stage_c0_olmo3_7b_transport",
            "model": const["model"],
            "revision": const["revision"],
            "smoke": False,
            "elapsed_sec": 1.0,
            "gpu_name": "synthetic",
            "r_lens_sha256": "deadbeef" if bad_provenance else sc0.EXPECTED_R_LENS_SHA256,
            "constants_sha256": sc0.EXPECTED_CONSTANTS_SHA256,
            "design_sha256": sc0.EXPECTED_DESIGN_SHA256,
            "k_dm": k_dm,
            "m_rand": m_rand,
            "forward_counts": {"clean": 8, "sham": 7, "regime_N": 8064, "regime_D": 6272, "regime_D_capped_skipped": 0},
            "total_forwards_executed": 14351,
        },
        "layers": layers,
        "targets_16": targets,
        "prompts_8": prompts,
        "post_mlp_norms": post_norms,
        "shams": {str(L): sham_err for L in layers},
        "sham_movement_check": {str(L): 5.0 for L in layers},
        "regime_N": rn,
        "regime_D": rd,
    }


def test_end_to_end_stdlib_scorer_pass_and_byte_determinism(tmp_path: Path):
    rec = _make_synthetic_receipt(mode="pass")
    rec_path = tmp_path / "c0_receipt.json.gz"
    with gzip.open(rec_path, "wt", encoding="utf-8") as f:
        json.dump(rec, f)

    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    for out in (out1, out2):
        cmd = [
            sys.executable, "-I", str(REPO_ROOT / "olmo3_virtual_and_highway/score_olmo3_c0.py"),
            "--receipt", str(rec_path), "--constants", str(CONSTANTS_PATH),
            "--design", str(DESIGN_PATH), "--out-dir", str(out), "--n-boot", "500", "--n-perm", "500",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        assert "Stage C1 gate: True" in res.stdout

    assert (out1 / "scorecard.json").read_bytes() == (out2 / "scorecard.json").read_bytes()
    assert (out1 / "SCORECARD.md").read_bytes() == (out2 / "SCORECARD.md").read_bytes()

    sc = json.loads((out1 / "scorecard.json").read_text(encoding="utf-8"))
    assert sc["verdicts"] == {"C0.0": "PASS", "C0.1": "PASS", "C0.2": "PASS", "C0.3": "PASS", "C0.4": "PASS"}
    # L1: telescoping count includes all 1,792 rand_iso traces (8064 + 6272 = 14336)
    assert sc["gate_C0_0"]["C0_0_2_telescoping"]["n_traces"] == 14336
    assert sc["stage_c1_gate_passed"] is True


def test_h2_and_l2_refuse_truncated_smoke_or_mismatched_receipts(tmp_path: Path):
    """H2/M3/L2 regression: truncated axes, empty receipts, smoke, or wrong dose_N/Direct must raise ValueError."""
    base = _make_synthetic_receipt(mode="pass")

    def _run(mut_rec: dict):
        p = tmp_path / "mut.json"
        p.write_text(json.dumps(mut_rec), encoding="utf-8")
        sc0.score_c0(p, CONSTANTS_PATH, DESIGN_PATH, n_boot=50, n_perm=50)

    # 1. Empty layers
    r_empty = copy.deepcopy(base)
    r_empty["layers"] = []
    r_empty["shams"] = {}
    r_empty["regime_N"] = {}
    r_empty["regime_D"] = {}
    with pytest.raises(ValueError, match="layers mismatch"):
        _run(r_empty)

    # 2. Single layer L18 only (previously scored full PASS!)
    r_18 = copy.deepcopy(base)
    r_18["layers"] = [18]
    r_18["shams"] = {"18": 0.0}
    r_18["regime_N"] = {"18": base["regime_N"]["18"]}
    r_18["regime_D"] = {"18": base["regime_D"]["18"]}
    with pytest.raises(ValueError, match="layers mismatch"):
        _run(r_18)

    # 3. Truncated targets (8 targets — previously passed C0.0-C0.3 & Stage C1!)
    r_8t = copy.deepcopy(base)
    r_8t["targets_16"] = base["targets_16"][:8]
    with pytest.raises(ValueError, match="target count mismatch"):
        _run(r_8t)

    # 4. Smoke receipt
    r_smoke = copy.deepcopy(base)
    r_smoke["meta"]["smoke"] = True
    with pytest.raises(ValueError, match="smoke"):
        _run(r_smoke)

    # 5. Wrong k_dm / m_rand
    r_k = copy.deepcopy(base)
    r_k["meta"]["k_dm"] = 2
    with pytest.raises(ValueError, match="k_dm"):
        _run(r_k)

    # 6. L2: Wrong dose_N (0.1x instead of 0.75x)
    r_dose = copy.deepcopy(base)
    r_dose["regime_N"]["18"]["dose_N"] = 0.1 * r_dose["post_mlp_norms"]["18"]
    with pytest.raises(ValueError, match="dose_N"):
        _run(r_dose)

    # 7. L2: Wrong target_direct_linear
    r_tdl = copy.deepcopy(base)
    r_tdl["regime_D"]["18"]["target_direct_linear"] = 1.5
    with pytest.raises(ValueError, match="target_direct_linear"):
        _run(r_tdl)


def test_h3_m1_m4_l6_nonfinite_missing_design_and_relaxed_provenance(tmp_path: Path):
    """H3/M1/M4/L6 regressions: non-finite retention/tel/fde, missing design file, relaxed provenance all fail C0.0."""
    base = _make_synthetic_receipt(mode="pass")

    # H3: Single +inf retention on one target at L18 must FAIL C0.0 and BLOCK C0.3/C1 (not pass with mean=-inf!)
    r_inf = copy.deepcopy(base)
    r_inf["regime_N"]["18"]["logit_lens"][" dolphin"][0]["retention"] = float("inf")
    p_inf = tmp_path / "inf.json"
    p_inf.write_text(json.dumps(r_inf), encoding="utf-8")
    sc_inf = sc0.score_c0(p_inf, CONSTANTS_PATH, DESIGN_PATH, n_boot=50, n_perm=50)
    assert sc_inf["verdicts"]["C0.0"] == "FAIL"
    assert sc_inf["verdicts"]["C0.3"] == "BLOCKED_BY_C0.0"
    assert sc_inf["stage_c1_gate_passed"] is False
    assert sc_inf["gate_C0_0"]["all_traces_finite"] is False
    # Strict JSON serialization succeeds (no bare NaN/Inf in output)
    json.dumps(sc_inf, allow_nan=False)

    # M1: 0.1% NaN telescoping_err (12 traces) must FAIL C0.0
    r_nan_tel = copy.deepcopy(base)
    for ti in range(12):
        ts = base["targets_16"][ti]["string"]
        r_nan_tel["regime_N"]["18"]["R_mf"][ts][0]["telescoping_err"] = float("nan")
    p_nt = tmp_path / "nan_tel.json"
    p_nt.write_text(json.dumps(r_nan_tel), encoding="utf-8")
    sc_nt = sc0.score_c0(p_nt, CONSTANTS_PATH, DESIGN_PATH, n_boot=50, n_perm=50)
    assert sc_nt["verdicts"]["C0.0"] == "FAIL"
    assert sc_nt["gate_C0_0"]["C0_0_2_telescoping"]["passed"] is False

    # M1: NaN frozen_direct_effect must FAIL C0.0
    r_nan_fde = copy.deepcopy(base)
    r_nan_fde["regime_D"]["18"]["R_mf"][" dolphin"][0]["frozen_direct_effect"] = float("nan")
    p_nf = tmp_path / "nan_fde.json"
    p_nf.write_text(json.dumps(r_nan_fde), encoding="utf-8")
    sc_nf = sc0.score_c0(p_nf, CONSTANTS_PATH, DESIGN_PATH, n_boot=50, n_perm=50)
    assert sc_nf["verdicts"]["C0.0"] == "FAIL"
    assert sc_nf["gate_C0_0"]["C0_0_3_regime_d_cap_and_denom"]["passed"] is False

    # M4: Missing design file must FAIL C0.0.4 (never fall back to receipt meta!)
    p_ok = tmp_path / "ok.json"
    p_ok.write_text(json.dumps(base), encoding="utf-8")
    sc_missing = sc0.score_c0(p_ok, CONSTANTS_PATH, tmp_path / "NONEXISTENT_DESIGN.md", n_boot=50, n_perm=50)
    assert sc_missing["gate_C0_0"]["C0_0_4_provenance"]["passed"] is False
    assert sc_missing["gate_C0_0"]["C0_0_4_provenance"]["design_file_missing"] is True
    assert sc_missing["verdicts"]["C0.0"] == "FAIL"
    assert sc_missing["stage_c1_gate_passed"] is False

    # L6: --no-strict-provenance records relaxed mode and refuses to stamp C0.0 PASS
    sc_rel = sc0.score_c0(p_ok, CONSTANTS_PATH, DESIGN_PATH, n_boot=50, n_perm=50, strict_provenance=False)
    assert sc_rel["gate_C0_0"]["C0_0_4_provenance"]["provenance_mode"] == "relaxed"
    assert sc_rel["gate_C0_0"]["C0_0_4_provenance"]["passed"] is False
    assert sc_rel["verdicts"]["C0.0"] == "FAIL"
    assert sc_rel["stage_c1_gate_passed"] is False


def test_scorer_null_fails_and_gates_block(tmp_path: Path):
    p_null = tmp_path / "null.json"
    p_null.write_text(json.dumps(_make_synthetic_receipt(mode="null")), encoding="utf-8")
    sc_null = sc0.score_c0(p_null, CONSTANTS_PATH, DESIGN_PATH, n_boot=200, n_perm=200)
    assert sc_null["verdicts"]["C0.0"] == "PASS"
    assert sc_null["verdicts"]["C0.1"] == "FAIL"
    assert sc_null["verdicts"]["C0.2"] == "FAIL"
    assert sc_null["verdicts"]["C0.4"] == "FAIL"
    assert sc_null["stage_c1_gate_passed"] is False

    p_sham = tmp_path / "sham.json"
    p_sham.write_text(json.dumps(_make_synthetic_receipt(mode="pass", sham_err=1e-4)), encoding="utf-8")
    sc_sham = sc0.score_c0(p_sham, CONSTANTS_PATH, DESIGN_PATH, n_boot=100, n_perm=100)
    assert sc_sham["verdicts"]["C0.0"] == "FAIL"
    assert sc_sham["verdicts"]["C0.1"] == "BLOCKED_BY_C0.0"

    p_cap = tmp_path / "cap.json"
    p_cap.write_text(json.dumps(_make_synthetic_receipt(mode="pass", cap_rate_inject=True)), encoding="utf-8")
    sc_cap = sc0.score_c0(p_cap, CONSTANTS_PATH, DESIGN_PATH, n_boot=100, n_perm=100)
    assert sc_cap["gate_C0_0"]["C0_0_3_regime_d_cap_and_denom"]["passed"] is False

    p_prov = tmp_path / "prov.json"
    p_prov.write_text(json.dumps(_make_synthetic_receipt(mode="pass", bad_provenance=True)), encoding="utf-8")
    sc_prov = sc0.score_c0(p_prov, CONSTANTS_PATH, DESIGN_PATH, n_boot=100, n_perm=100)
    assert sc_prov["gate_C0_0"]["C0_0_4_provenance"]["passed"] is False

    p_leak = tmp_path / "leak.json"
    p_leak.write_text(json.dumps(_make_synthetic_receipt(mode="pass", single_pair_leak=1.20)), encoding="utf-8")
    sc_leak = sc0.score_c0(p_leak, CONSTANTS_PATH, DESIGN_PATH, n_boot=200, n_perm=200)
    assert sc_leak["C0_4"]["cross_domain_null"]["aggregate_passed"] is True
    assert sc_leak["C0_4"]["cross_domain_null"]["per_pair_passed"] is False
    assert sc_leak["verdicts"]["C0.4"] == "FAIL"
    assert sc_leak["stage_c1_gate_passed"] is True


def test_run_c0_mock_olmo3_h1_m2_l1_l4_l5(tmp_path: Path):
    """Runner integration test covering H1 resume refusal, M2 non-tautological sham, L1/L4 rand_iso, L5 distinct arms."""
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    import run_olmo3_c0 as r0

    class MockRMSNorm(nn.Module):
        def __init__(self, d: int, eps: float = 1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(d))
            self.variance_epsilon = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.variance_epsilon)
            return (x / rms) * self.weight

    class MockLayer(nn.Module):
        def __init__(self, d: int):
            super().__init__()
            self.attn_proj = nn.Linear(d, d, bias=False)
            self.post_attention_layernorm = MockRMSNorm(d)
            self.mlp_proj = nn.Linear(d, d, bias=False)
            self.post_feedforward_layernorm = MockRMSNorm(d)

        def forward(self, h: torch.Tensor) -> torch.Tensor:
            a = self.post_attention_layernorm(self.attn_proj(h))
            h = h + a
            m = self.post_feedforward_layernorm(torch.tanh(self.mlp_proj(h)))
            return h + m

    class MockInner(nn.Module):
        def __init__(self, vocab: int, d: int, n_layers: int):
            super().__init__()
            self.embed_tokens = nn.Embedding(vocab, d)
            self.layers = nn.ModuleList([MockLayer(d) for _ in range(n_layers)])
            self.norm = MockRMSNorm(d)

    class MockOut:
        def __init__(self, logits: torch.Tensor):
            self.logits = logits

    class MockOLMo3(nn.Module):
        def __init__(self, vocab: int = 64, d: int = 32, n_layers: int = 4):
            super().__init__()
            self.config = type("Cfg", (), {"num_hidden_layers": n_layers})()
            self.model = MockInner(vocab, d, n_layers)
            self.lm_head = nn.Linear(d, vocab, bias=False)

        def forward(self, input_ids: torch.Tensor, use_cache: bool = False):
            h = self.model.embed_tokens(input_ids)
            for lay in self.model.layers:
                h = lay(h)
            return MockOut(self.lm_head(self.model.norm(h)))

    class MockTok:
        def encode(self, text: str, return_tensors: str = "pt", add_special_tokens: bool = False):
            ids = [(sum(ord(c) for c in w) % 60) + 1 for w in text.split()[:6]]
            return torch.tensor([ids], dtype=torch.long)

    torch.manual_seed(7)
    d = 32
    model = MockOLMo3(vocab=64, d=d, n_layers=4).eval()
    tok = MockTok()

    # M2 regression: corrupt write vectors (NaN or 1e9) MUST be rejected even at dose=0.0!
    nan_vec = torch.full((d,), float("nan"))
    with pytest.raises(ValueError, match="Non-finite"):
        with r0.post_norm_last_inject_hook(model, 0, nan_vec, dose=0.0):
            pass
    huge_vec = torch.full((d,), 1e9)
    with pytest.raises(ValueError, match="not unit-norm"):
        with r0.post_norm_last_inject_hook(model, 0, huge_vec, dose=0.0):
            pass

    def _build_mock_bank(seed: int, r_lens_sha: str, targets_override: list[dict] | None = None) -> dict:
        g = torch.Generator().manual_seed(seed)
        targets = targets_override or [
            {"domain": "animals", "string": " dolphin", "id": 5, "decoded_piece": " dolphin"},
            {"domain": "places", "string": " harbor", "id": 11, "decoded_piece": " harbor"},
        ]
        w_u = model.lm_head.weight.detach().float()
        gamma = model.model.norm.weight.detach().float()
        u_vecs = torch.stack([w_u[t["id"]] * gamma for t in targets], dim=0)
        u_hats = u_vecs / u_vecs.norm(dim=-1, keepdim=True)

        def _make_pos_arm(scale_noise: float) -> torch.Tensor:
            v = u_hats + scale_noise * torch.randn(u_hats.shape, generator=g)
            return v / v.norm(dim=-1, keepdim=True)

        layers_bank = {}
        for L in (0, 1):
            # L5: DISTINCT arm vectors & DISTINCT pred_ret values per arm!
            w_rmf = _make_pos_arm(0.30)
            w_rr = _make_pos_arm(0.20)
            w_ll = u_hats.clone()
            w_dm = _make_pos_arm(0.40).unsqueeze(1)
            rand_iso = torch.randn(2, d, generator=g)
            rand_iso = rand_iso / rand_iso.norm(dim=-1, keepdim=True)
            layers_bank[L] = {
                "lambda_ridge": 0.1,
                "tr_R_over_d": 0.5,
                "R_mf": w_rmf,
                "R_ridge": w_rr,
                "logit_lens": w_ll,
                "direct_matched": w_dm,
                "rand_iso": rand_iso,
                "per_target_geom": {
                    t["string"]: {
                        "N": 1.1, "c": 0.7, "cos_u_Rridge": 0.85,
                        "pred_ret_Rmf": 1.6, "pred_ret_Rridge": 1.25,
                        "pred_ret_dm": 0.75, "pred_ret_ll": 0.75, "pred_gap_D3": 2.5,
                    }
                    for t in targets
                },
            }
        return {
            "meta": {
                "model": "mock-olmo3", "revision": "test",
                "r_lens_sha256": r_lens_sha, "constants_sha256": "def", "design_sha256": "ghi",
                "k_dm": 1, "m_rand": 2,
            },
            "targets_16": targets,
            "layers": [0, 1],
            "u_vectors": u_vecs,
            "layers_bank": layers_bank,
        }

    bank_a = _build_mock_bank(seed=10, r_lens_sha="LENS_OLD_AAAA")
    out_dir = tmp_path / "run_h1"
    rec1 = r0.run_c0(model, tok, bank_a, out_dir, device="cpu", smoke=True)

    # M2: Sham exact 0.0 AND companion R_mf movement check > 0
    assert all(v == 0.0 for v in rec1["shams"].values())
    assert all(v > 0.0 for v in rec1["sham_movement_check"].values())

    # L5: Distinct arms produce distinct exact_dlogit & pred_dlogit
    ex_rmf = rec1["regime_N"]["0"]["R_mf"][" dolphin"][0]["exact_dlogit"]
    ex_rr = rec1["regime_N"]["0"]["R_ridge"][" dolphin"][0]["exact_dlogit"]
    ex_ll = rec1["regime_N"]["0"]["logit_lens"][" dolphin"][0]["exact_dlogit"]
    assert len({ex_rmf, ex_rr, ex_ll}) == 3

    # L1 + L4: Exact telescoping on both target arms AND rand_iso traces (< 1e-4)
    for L in ("0", "1"):
        for ti, t in enumerate(bank_a["targets_16"]):
            for row in rec1["regime_N"][L]["R_mf"][t["string"]]:
                assert row["telescoping_err"] < 1e-4
                assert abs(row["all_16_dlogits"][ti] - row["exact_dlogit"]) < 1e-5
            for row in rec1["regime_D"][L]["R_mf"][t["string"]]:
                if not row.get("capped", False):
                    assert abs(row["direct_linear"] - 3.0) < 1e-3
                    assert row["telescoping_err"] < 1e-4
        for draw in rec1["regime_N"][L]["rand_iso"]:
            for row in draw:
                assert row["telescoping_err"] < 1e-4
                assert row["canc_r_self"] >= 0.0

    # Valid resume with identical bank succeeds
    rec_same = r0.run_c0(model, tok, bank_a, out_dir, device="cpu", smoke=True)
    assert rec_same["meta"]["forward_counts"] == rec1["meta"]["forward_counts"]

    # H1 regression (a): Resuming into same out_dir with DIFFERENT bank/lens hash MUST REFUSE!
    bank_b = _build_mock_bank(seed=99, r_lens_sha="LENS_NEW_BBBB")
    with pytest.raises(RuntimeError, match="H1 PROVENANCE REFUSAL"):
        r0.run_c0(model, tok, bank_b, out_dir, device="cpu", smoke=True)

    # H1 regression (b): Resuming into same out_dir with DIFFERENT target set MUST REFUSE!
    bank_t = _build_mock_bank(
        seed=10,
        r_lens_sha="LENS_OLD_AAAA",
        targets_override=[
            {"domain": "animals", "string": " hawk", "id": 7, "decoded_piece": " hawk"},
            {"domain": "animals", "string": " tiger", "id": 9, "decoded_piece": " tiger"},
        ],
    )
    with pytest.raises(RuntimeError, match="H1 PROVENANCE REFUSAL"):
        r0.run_c0(model, tok, bank_t, out_dir, device="cpu", smoke=True)


def test_system_python3_isolated_stdlib_only(tmp_path: Path):
    """Verify bare `/usr/bin/python3 -I` (zero third-party packages) runs `score_olmo3_c0.py` cleanly."""
    sys_py = Path("/usr/bin/python3")
    if not sys_py.exists():
        pytest.skip("/usr/bin/python3 not present")
    rec_path = tmp_path / "pass.json.gz"
    with gzip.open(rec_path, "wt", encoding="utf-8") as f:
        json.dump(_make_synthetic_receipt(mode="pass"), f)
    out = tmp_path / "sys_out"
    res = subprocess.run(
        [
            str(sys_py), "-I", str(REPO_ROOT / "olmo3_virtual_and_highway/score_olmo3_c0.py"),
            "--receipt", str(rec_path), "--constants", str(CONSTANTS_PATH),
            "--design", str(DESIGN_PATH), "--out-dir", str(out), "--n-boot", "200", "--n-perm", "200",
        ],
        capture_output=True, text=True, check=True,
    )
    assert "Stage C1 gate: True" in res.stdout
    assert (out / "scorecard.json").exists()
    assert (out / "SCORECARD.md").exists()
