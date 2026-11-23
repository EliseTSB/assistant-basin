#!/usr/bin/env python3
"""100% Python 3.10+ stdlib-only deterministic scorer for Stage C0b
(`OLMO3-TRANSPORT-C0B-DESIGN.md` Registered v1).

Requires ZERO third-party packages (no numpy, scipy, or torch) and uses `math.fsum`
for all float reductions so any clean Python 3.10+ machine reproduces `scorecard.json`
and `SCORECARD.md` byte-for-byte via `python3 -I` + `cmp`.

Fail-closed guarantees:
  - Hard-refuses truncated/smoke/parameter-mismatched receipts (ValueError).
  - Refuses empty or non-finite inputs in mean/median/quantile.
  - Audits finiteness across ALL retained traces (including all `all_16_dlogits`).
  - Splits Gate C0b.0.4 into global SHA-256 provenance (`4a`, gates C0b.1 & C0b.2)
    and receipt-pairing drift (`4b`, gates C0b.1b only — never voids live-to-live C0b.2).
  - Missing `--design` or `--no-strict-provenance` always fails Gate C0b.0.4a.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

EXPECTED_R_LENS_SHA256 = "ceef21fc90b17aafa5625f23ab2c1f35be7b2f8b735847cb89bfd2a438caa955"
EXPECTED_C0_CONSTANTS_SHA256 = "3be64ef80f12d34fc59d7e561d5617dd96ecb605649d5f64e1a98bd974acbe98"
EXPECTED_C0B_CONSTANTS_SHA256 = "4596f012bb862aec81a0b07a478223ef02dab8bab3706c4f7902c860958a0f2e"
EXPECTED_C0_BANK_SHA256 = "ab7e04caaf74d4725ed88596da3c39f9303d5e55d7114bbbef1a5500b025ab5f"
EXPECTED_C0_RECEIPT_UNCOMP_SHA256 = "fe7883be37ecf0b7a0a694fc1dd035215d05c27c297cc0959ef23f379258fb1c"
EXPECTED_C0B_DESIGN_SHA256 = "8b3efeaeba3febbcc0e9715840a79bea6d6bc29986f7bf21a89cf6080d61a994"

LAYERS_4L = [12, 18, 24, 27]
LAYERS_3L = [18, 24, 27]
EXPECTED_ANCHOR_TARGETS = [" lantern", " judge"]
EXPECTED_N_TARGETS = 16
EXPECTED_N_PROMPTS = 8
EXPECTED_K_DM = 4

BOOT_RESAMPLES = 10_000
SEED_BOOT = 20261123


# ---------------------------------------------------------------------------
# Pure-stdlib I/O & hashing
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_uncompressed(path: Path) -> str:
    h = hashlib.sha256()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json_maybe_gz(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Pure-stdlib fail-closed statistics (`math.fsum` for cross-Python determinism)
# ---------------------------------------------------------------------------
def _require_nonempty_finite(xs: list[float], name: str) -> None:
    if not xs:
        raise ValueError(f"{name}() received empty input")
    for idx, v in enumerate(xs):
        if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            raise ValueError(f"{name}() received non-finite value {v!r} at index {idx}")


def mean(xs: list[float]) -> float:
    _require_nonempty_finite(xs, "mean")
    return math.fsum(xs) / len(xs)


def median(xs: list[float]) -> float:
    _require_nonempty_finite(xs, "median")
    return float(statistics.median(xs))


def quantile(xs: list[float], q: float) -> float:
    _require_nonempty_finite(xs, "quantile")
    if not (0.0 <= q <= 1.0):
        raise ValueError(f"quantile q out of range: {q}")
    s = sorted(xs)
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(s[lo])
    return float(s[lo] + (pos - lo) * (s[hi] - s[lo]))


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    if p <= 0.0:
        return -38.0
    if p >= 1.0:
        return 38.0
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    p_low = 0.02425
    p_high = 1.0 - p_low
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
    )


def exact_sign_flip_p_one_sided(vals: list[float]) -> float:
    """Exact one-sided paired sign-flip p-value over all 2^n flips (Gray code). Returns NaN on bad input."""
    n = len(vals)
    if n == 0 or not all(isinstance(x, (int, float)) and math.isfinite(float(x)) for x in vals):
        return float("nan")
    obs = math.fsum(vals)
    tol = 1e-12 * max(1.0, math.fsum(abs(x) for x in vals))
    cur = obs
    signs = [1] * n
    ge = 1
    total = 1 << n
    prev_gray = 0
    for step in range(1, total):
        gray = step ^ (step >> 1)
        diff = gray ^ prev_gray
        bit = diff.bit_length() - 1
        prev_gray = gray
        if signs[bit] == 1:
            signs[bit] = -1
            cur -= 2.0 * vals[bit]
        else:
            signs[bit] = 1
            cur += 2.0 * vals[bit]
        if cur >= obs - tol:
            ge += 1
    return ge / total


def bca_cluster_ci(
    n_clusters: int,
    stat_fn,
    alphas: tuple[float, ...],
    n_boot: int = BOOT_RESAMPLES,
    seed: int = SEED_BOOT,
) -> dict:
    """10,000-resample BCa cluster bootstrap over indices `0..n_clusters-1`."""
    full_idx = list(range(n_clusters))
    theta_hat = float(stat_fn(full_idx))
    if not math.isfinite(theta_hat):
        out: dict = {"point": None}
        for al in alphas:
            out[f"q_{al:g}"] = None
        return out

    rng = random.Random(seed)
    boots = [0.0] * n_boot
    for b in range(n_boot):
        idx = [rng.randrange(n_clusters) for _ in range(n_clusters)]
        val = float(stat_fn(idx))
        if not math.isfinite(val):
            out = {"point": round(theta_hat, 6)}
            for al in alphas:
                out[f"q_{al:g}"] = None
            return out
        boots[b] = val
    boots.sort()

    less = sum(1 for v in boots if v < theta_hat)
    eq = sum(1 for v in boots if v == theta_hat)
    prop = (less + 0.5 * eq) / n_boot
    prop = max(0.5 / n_boot, min(1.0 - 0.5 / n_boot, prop))
    z0 = norm_ppf(prop)

    jacks = [float(stat_fn([j for j in range(n_clusters) if j != i])) for i in range(n_clusters)]
    if not all(math.isfinite(j) for j in jacks):
        out = {"point": round(theta_hat, 6)}
        for al in alphas:
            out[f"q_{al:g}"] = None
        return out
    j_mean = math.fsum(jacks) / n_clusters
    diffs = [j_mean - jv for jv in jacks]
    denom = math.fsum(d * d for d in diffs)
    if denom <= 1e-18:
        a_hat = 0.0
    else:
        a_hat = math.fsum(d * d * d for d in diffs) / (6.0 * (denom ** 1.5))

    out = {"point": round(theta_hat, 6)}
    for al in alphas:
        z_a = norm_ppf(al)
        adj_denom = 1.0 - a_hat * (z0 + z_a)
        if abs(adj_denom) < 1e-12:
            q_adj = al
        else:
            q_adj = norm_cdf(z0 + (z0 + z_a) / adj_denom)
        q_adj = max(0.0, min(1.0, q_adj))
        out[f"q_{al:g}"] = round(quantile(boots, q_adj), 6)
    return out


def ols_slope(xs: list[float], ys: list[float]) -> float:
    _require_nonempty_finite(xs, "ols_slope_x")
    _require_nonempty_finite(ys, "ols_slope_y")
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("ols_slope requires equal-length lists of length >= 2")
    mx = mean(xs)
    my = mean(ys)
    num = math.fsum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.fsum((x - mx) ** 2 for x in xs)
    return num / den if den > 1e-18 else 0.0


# ---------------------------------------------------------------------------
# Registered schema, axes, and finiteness validation
# ---------------------------------------------------------------------------
def validate_registered_c0b_receipt(receipt: dict, c0b_const: dict) -> None:
    """Hard-refuse any receipt violating registered Stage C0b axes or parameters."""
    errors: list[str] = []
    meta = receipt.get("meta", {})
    if meta.get("smoke", False):
        errors.append("meta.smoke is True (smoke receipts cannot be scored as registered Stage C0b)")

    if [int(L) for L in receipt.get("layers_4L", [])] != LAYERS_4L:
        errors.append(f"layers_4L mismatch: got {receipt.get('layers_4L')}, expected {LAYERS_4L}")
    if [int(L) for L in receipt.get("layers_3L", [])] != LAYERS_3L:
        errors.append(f"layers_3L mismatch: got {receipt.get('layers_3L')}, expected {LAYERS_3L}")
    if list(receipt.get("anchor_targets_regime_D", [])) != EXPECTED_ANCHOR_TARGETS:
        errors.append(
            f"anchor_targets_regime_D mismatch: got {receipt.get('anchor_targets_regime_D')}, "
            f"expected {EXPECTED_ANCHOR_TARGETS}"
        )

    targets = receipt.get("targets_16", [])
    exp_targets = c0b_const["targets_16"]
    if len(targets) != EXPECTED_N_TARGETS:
        errors.append(f"target count mismatch: got {len(targets)}, expected {EXPECTED_N_TARGETS}")
    elif [t["string"] for t in targets] != [t["string"] for t in exp_targets] or [
        int(t["id"]) for t in targets
    ] != [int(t["id"]) for t in exp_targets]:
        errors.append("targets_16 do not match registered c0b_prereg_constants.json")

    prompts = receipt.get("prompts_8", [])
    exp_prompts = c0b_const["prompts_8"]
    if len(prompts) != EXPECTED_N_PROMPTS:
        errors.append(f"prompt count mismatch: got {len(prompts)}, expected {EXPECTED_N_PROMPTS}")
    elif [p["id"] for p in prompts] != [p["id"] for p in exp_prompts]:
        errors.append("prompts_8 ids do not match registered c0b_prereg_constants.json")

    if int(meta.get("k_dm", -1)) != EXPECTED_K_DM:
        errors.append(f"meta.k_dm mismatch: got {meta.get('k_dm')}, expected {EXPECTED_K_DM}")

    if errors:
        raise ValueError("Registered Stage C0b schema/axes check failed: " + "; ".join(errors))

    t_strs = [t["string"] for t in targets]
    post_norms = receipt.get("post_mlp_norms", {})
    shams = receipt.get("shams", {})
    rd_all = receipt.get("regime_D_all", {})
    rd_anc = receipt.get("regime_D_last_anchor", {})
    rn_last = receipt.get("regime_N_last", {})
    rn_all = receipt.get("regime_N_all", {})

    for mode in ("last", "all"):
        if mode not in shams or not isinstance(shams[mode], dict):
            errors.append(f"missing shams[{mode!r}]")
        else:
            for L in LAYERS_4L:
                if str(L) not in shams[mode]:
                    errors.append(f"missing shams[{mode!r}][L{L}]")

    for L in LAYERS_4L:
        ls = str(L)
        if ls not in post_norms:
            errors.append(f"missing post_mlp_norms[{L}]")
            continue
        norm_l = float(post_norms[ls])
        if not (math.isfinite(norm_l) and norm_l > 0.0):
            errors.append(f"non-finite/non-positive post_mlp_norms[{L}]={norm_l}")

        # Regime D "all"
        if ls not in rd_all:
            errors.append(f"missing regime_D_all[{L}]")
        else:
            blk = rd_all[ls]
            if blk.get("positions") != "all":
                errors.append(f"regime_D_all[{L}].positions != 'all'")
            if abs(float(blk.get("target_direct_linear", float("nan"))) - 3.0) > 1e-9:
                errors.append(f"regime_D_all[{L}].target_direct_linear != 3.0")
            for arm in ("R_mf", "logit_lens", "direct_matched"):
                if arm not in blk:
                    errors.append(f"regime_D_all[{L}] missing arm {arm}")
                    continue
                for ts in t_strs:
                    if ts not in blk[arm]:
                        errors.append(f"regime_D_all[{L}].{arm} missing {ts!r}")
                        continue
                    item = blk[arm][ts]
                    if arm == "direct_matched":
                        if len(item) != EXPECTED_K_DM or any(len(d) != EXPECTED_N_PROMPTS for d in item):
                            errors.append(f"regime_D_all[{L}].direct_matched[{ts!r}] shape != (4,8)")
                    elif len(item) != EXPECTED_N_PROMPTS:
                        errors.append(f"regime_D_all[{L}].{arm}[{ts!r}] len != 8")

        # Regime D "last" full-pair anchor
        if ls not in rd_anc:
            errors.append(f"missing regime_D_last_anchor[{L}]")
        else:
            blk = rd_anc[ls]
            if blk.get("positions") != "last":
                errors.append(f"regime_D_last_anchor[{L}].positions != 'last'")
            if abs(float(blk.get("target_direct_linear", float("nan"))) - 3.0) > 1e-9:
                errors.append(f"regime_D_last_anchor[{L}].target_direct_linear != 3.0")
            for arm in ("R_mf", "direct_matched"):
                if arm not in blk:
                    errors.append(f"regime_D_last_anchor[{L}] missing arm {arm}")
                    continue
                for ts in EXPECTED_ANCHOR_TARGETS:
                    if ts not in blk[arm]:
                        errors.append(f"regime_D_last_anchor[{L}].{arm} missing {ts!r}")
                        continue
                    item = blk[arm][ts]
                    if arm == "direct_matched":
                        if len(item) != EXPECTED_K_DM or any(len(d) != EXPECTED_N_PROMPTS for d in item):
                            errors.append(f"regime_D_last_anchor[{L}].direct_matched[{ts!r}] shape != (4,8)")
                    elif len(item) != EXPECTED_N_PROMPTS:
                        errors.append(f"regime_D_last_anchor[{L}].{arm}[{ts!r}] len != 8")

    for L in LAYERS_3L:
        ls = str(L)
        norm_l = float(post_norms.get(ls, float("nan")))
        for reg_name, reg_dict, exp_pos, arms in (
            ("regime_N_last", rn_last, "last", ("R_mf", "R_ridge", "logit_lens")),
            ("regime_N_all", rn_all, "all", ("R_mf", "R_ridge")),
        ):
            if ls not in reg_dict:
                errors.append(f"missing {reg_name}[{L}]")
                continue
            blk = reg_dict[ls]
            if blk.get("positions") != exp_pos:
                errors.append(f"{reg_name}[{L}].positions != {exp_pos!r}")
            dose_n = float(blk.get("dose_N", float("nan")))
            if not math.isfinite(dose_n) or abs(dose_n - 0.75 * norm_l) > 1e-4:
                errors.append(f"{reg_name}[{L}].dose_N={dose_n} != 0.75*norm={0.75 * norm_l:.6f}")
            for arm in arms:
                if arm not in blk:
                    errors.append(f"{reg_name}[{L}] missing arm {arm}")
                    continue
                for ts in t_strs:
                    if ts not in blk[arm]:
                        errors.append(f"{reg_name}[{L}].{arm} missing {ts!r}")
                        continue
                    item = blk[arm][ts]
                    if len(item) != EXPECTED_N_PROMPTS:
                        errors.append(f"{reg_name}[{L}].{arm}[{ts!r}] len != 8")
                    elif any(len(r.get("all_16_dlogits", [])) != EXPECTED_N_TARGETS for r in item):
                        errors.append(f"{reg_name}[{L}].{arm}[{ts!r}] missing all_16_dlogits[16]")

    if errors:
        raise ValueError("Registered Stage C0b structure/parameter check failed: " + "; ".join(errors[:12]))


def audit_all_c0b_traces_finite(receipt: dict) -> list[str]:
    """Audit finiteness of every retained trace & sham across the entire Stage C0b receipt."""
    bad: list[str] = []
    t_strs = [t["string"] for t in receipt["targets_16"]]
    fields = (
        "dose", "exact_dlogit", "pred_dlogit", "direct_linear", "frozen_direct_effect",
        "sum_attn", "sum_mlp", "rmsnorm_shift", "telescoping_err", "retention", "dlogit_z", "kl",
    )
    for mode in ("last", "all"):
        for L in LAYERS_4L:
            v = receipt["shams"][mode][str(L)]
            if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                bad.append(f"shams.{mode}.L{L}={v!r}")

    def _check_rows(rows: list[dict], prefix: str, check_all16: bool) -> bool:
        for pi, r in enumerate(rows):
            if r.get("capped", False):
                continue
            for f in fields:
                v = r.get(f)
                if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                    bad.append(f"{prefix}[{pi}].{f}={v!r}")
                    if len(bad) >= 20:
                        return True
            if check_all16:
                for ti, v in enumerate(r["all_16_dlogits"]):
                    if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                        bad.append(f"{prefix}[{pi}].all_16_dlogits[{ti}]={v!r}")
                        if len(bad) >= 20:
                            return True
        return False

    for L in LAYERS_4L:
        ls = str(L)
        for arm in ("R_mf", "logit_lens", "direct_matched"):
            for ts in t_strs:
                item = receipt["regime_D_all"][ls][arm][ts]
                rows = [r for d in item for r in d] if arm == "direct_matched" else item
                if _check_rows(rows, f"regime_D_all.L{L}.{arm}.{ts}", False):
                    return bad
        for arm in ("R_mf", "direct_matched"):
            for ts in EXPECTED_ANCHOR_TARGETS:
                item = receipt["regime_D_last_anchor"][ls][arm][ts]
                rows = [r for d in item for r in d] if arm == "direct_matched" else item
                if _check_rows(rows, f"regime_D_last_anchor.L{L}.{arm}.{ts}", False):
                    return bad

    for L in LAYERS_3L:
        ls = str(L)
        for reg_name, arms in (
            ("regime_N_last", ("R_mf", "R_ridge", "logit_lens")),
            ("regime_N_all", ("R_mf", "R_ridge")),
        ):
            for arm in arms:
                for ts in t_strs:
                    if _check_rows(receipt[reg_name][ls][arm][ts], f"{reg_name}.L{L}.{arm}.{ts}", True):
                        return bad
    return bad


# ---------------------------------------------------------------------------
# Cross-domain bleed helper (Regime N `all_16_dlogits`, L in {18, 24, 27})
# ---------------------------------------------------------------------------
def compute_regime_n_bleed_stats(
    rn_dict: dict,
    arm: str,
    t_strs: list[str],
    t_domains: list[str],
    layers: list[int] = LAYERS_3L,
) -> dict:
    """Compute per-target B_cross, S_cross, self_dlogit, signed cross mean, frac positive, same-domain bleed."""
    n_t = len(t_strs)
    b_cross_by_t: list[float] = []
    s_cross_by_t: list[float] = []
    self_by_t: list[float] = []
    signed_cross_all: list[float] = []
    same_dom_abs_all: list[float] = []

    # Pre-collect per readout target `dst_idx`
    cross_abs_acc: list[list[float]] = [[] for _ in range(n_t)]
    self_acc: list[list[float]] = [[] for _ in range(n_t)]

    for L in layers:
        ls = str(L)
        arm_blk = rn_dict[ls][arm]
        for src_idx, src_str in enumerate(t_strs):
            src_dom = t_domains[src_idx]
            for tr in arm_blk[src_str]:
                self_acc[src_idx].append(float(tr["exact_dlogit"]))
                all16 = tr["all_16_dlogits"]
                for dst_idx in range(n_t):
                    if dst_idx == src_idx:
                        continue
                    val = float(all16[dst_idx])
                    if t_domains[dst_idx] != src_dom:
                        cross_abs_acc[dst_idx].append(abs(val))
                        signed_cross_all.append(val)
                    else:
                        same_dom_abs_all.append(abs(val))

    for ti in range(n_t):
        bc = mean(cross_abs_acc[ti])
        sd = mean(self_acc[ti])
        sc = bc / max(sd, 1e-6)
        b_cross_by_t.append(bc)
        self_by_t.append(sd)
        s_cross_by_t.append(sc)

    frac_pos = sum(1 for v in signed_cross_all if v > 0.0) / len(signed_cross_all)
    return {
        "B_cross_by_target": b_cross_by_t,
        "S_cross_by_target": s_cross_by_t,
        "self_dlogit_by_target": self_by_t,
        "B_cross_arith_mean": mean(b_cross_by_t),
        "B_cross_geo_mean": math.exp(mean([math.log(max(x, 1e-6)) for x in b_cross_by_t])),
        "S_cross_arith_mean": mean(s_cross_by_t),
        "S_cross_geo_mean": math.exp(mean([math.log(max(x, 1e-6)) for x in s_cross_by_t])),
        "self_dlogit_geo_mean": math.exp(mean([math.log(max(x, 1e-6)) for x in self_by_t])),
        "signed_cross_mean": mean(signed_cross_all),
        "frac_cross_positive": frac_pos,
        "same_domain_wrong_abs_mean": mean(same_dom_abs_all),
    }


# ---------------------------------------------------------------------------
# Main Stage C0b scoring function
# ---------------------------------------------------------------------------
def score_c0b(
    receipt_path: Path,
    c0b_constants_path: Path,
    c0_constants_path: Path,
    c0_receipt_path: Path,
    design_path: Path | None = None,
    n_boot: int = BOOT_RESAMPLES,
    strict_provenance: bool = True,
) -> dict:
    receipt = load_json_maybe_gz(receipt_path)
    c0b_const = json.loads(c0b_constants_path.read_text(encoding="utf-8"))
    c0_const = json.loads(c0_constants_path.read_text(encoding="utf-8"))
    c0_receipt = load_json_maybe_gz(c0_receipt_path)

    # Hard-refuse truncated / smoke / parameter-mismatched receipts
    validate_registered_c0b_receipt(receipt, c0b_const)

    c0b_const_sha = sha256_file(c0b_constants_path)
    c0_const_sha = sha256_file(c0_constants_path)
    c0_rec_uncomp_sha = sha256_uncompressed(c0_receipt_path)
    receipt_uncomp_sha = sha256_uncompressed(receipt_path)

    design_checked = False
    design_missing = False
    design_sha = ""
    if design_path is not None:
        if design_path.exists():
            design_sha = sha256_file(design_path)
            design_checked = True
        else:
            design_missing = True

    targets = receipt["targets_16"]
    t_strs = [t["string"] for t in targets]
    t_domains = [t["domain"] for t in targets]
    n_t = len(targets)
    prompts = receipt["prompts_8"]
    prompt_tokens = [int(p["n_tokens"]) for p in c0b_const["prompts_8"]]
    bare_idxs = [i for i, p in enumerate(prompts) if p["format"] == "bare"]
    chat_idxs = [i for i, p in enumerate(prompts) if p["format"] == "chatml"]
    narr_idxs = [i for i, p in enumerate(prompts) if p["style"] == "narrative"]
    anal_idxs = [i for i, p in enumerate(prompts) if p["style"] == "analytical"]

    # Frozen pred_gap_D3 map from c0_prereg_constants.json
    pred_gap_map: dict[tuple[int, str], float] = {}
    for L in LAYERS_4L:
        for pt in c0_const["layer_geometry"][str(L)]["per_target"]:
            pred_gap_map[(L, pt["target"])] = float(pt["pred_gap_D3"])

    # =========================================================================
    # GATE C0b.0 (Global Clauses 1-4a + C0b.1b-only Drift Clause 4b)
    # =========================================================================
    non_finite_issues = audit_all_c0b_traces_finite(receipt)
    all_finite = len(non_finite_issues) == 0

    # C0b.0.1: Zero-dose shams (4L x 2 modes) + required movement checks (4L x 2 modes)
    sham_last = {str(L): float(receipt["shams"]["last"][str(L)]) for L in LAYERS_4L}
    sham_all = {str(L): float(receipt["shams"]["all"][str(L)]) for L in LAYERS_4L}
    sham_move = receipt.get("sham_movement_check")
    move_ok = isinstance(sham_move, dict) and all(
        isinstance(sham_move.get(mode), dict)
        and all(
            str(L) in sham_move[mode]
            and math.isfinite(float(sham_move[mode][str(L)]))
            and float(sham_move[mode][str(L)]) > 0.0
            for L in LAYERS_4L
        )
        for mode in ("last", "all")
    )
    c0b_0_1_pass = (
        all(math.isfinite(v) and v == 0.0 for v in sham_last.values())
        and all(math.isfinite(v) and v == 0.0 for v in sham_all.values())
        and move_ok
    )

    # C0b.0.2: Quantile telescoping across ALL retained intervened traces (5,312 - capped)
    all_tel: list[float] = []
    tel_bare: list[float] = []
    tel_chat: list[float] = []
    capped_total = 0

    for L in LAYERS_4L:
        ls = str(L)
        for arm in ("R_mf", "logit_lens", "direct_matched"):
            for ts in t_strs:
                item = receipt["regime_D_all"][ls][arm][ts]
                draws = item if arm == "direct_matched" else [item]
                for d in draws:
                    for pi, r in enumerate(d):
                        if r.get("capped", False):
                            capped_total += 1
                        else:
                            tv = abs(float(r["telescoping_err"]))
                            all_tel.append(tv)
                            (tel_bare if pi in bare_idxs else tel_chat).append(tv)
        for arm in ("R_mf", "direct_matched"):
            for ts in EXPECTED_ANCHOR_TARGETS:
                item = receipt["regime_D_last_anchor"][ls][arm][ts]
                draws = item if arm == "direct_matched" else [item]
                for d in draws:
                    for pi, r in enumerate(d):
                        if r.get("capped", False):
                            capped_total += 1
                        else:
                            tv = abs(float(r["telescoping_err"]))
                            all_tel.append(tv)
                            (tel_bare if pi in bare_idxs else tel_chat).append(tv)
    for L in LAYERS_3L:
        ls = str(L)
        for reg_name, arms in (
            ("regime_N_last", ("R_mf", "R_ridge", "logit_lens")),
            ("regime_N_all", ("R_mf", "R_ridge")),
        ):
            for arm in arms:
                for ts in t_strs:
                    for pi, r in enumerate(receipt[reg_name][ls][arm][ts]):
                        tv = abs(float(r["telescoping_err"]))
                        all_tel.append(tv)
                        (tel_bare if pi in bare_idxs else tel_chat).append(tv)

    expected_tel_n = (3072 + 320 + 1152 + 768) - capped_total
    if all_finite and len(all_tel) == expected_tel_n:
        tel_med = median(all_tel)
        tel_p95 = quantile(all_tel, 0.95)
        tel_p999 = quantile(all_tel, 0.999)
        tel_max = max(all_tel)
        c0b_0_2_pass = (tel_p95 <= 0.08) and (tel_p999 <= 0.15)
        tel_fmt_diag = {
            "bare_p95": round(quantile(tel_bare, 0.95), 6),
            "bare_max": round(max(tel_bare), 6),
            "chatml_p95": round(quantile(tel_chat, 0.95), 6),
            "chatml_max": round(max(tel_chat), 6),
        }
    else:
        tel_med = tel_p95 = tel_p999 = tel_max = None
        tel_fmt_diag = {}
        c0b_0_2_pass = False

    # C0b.0.3: Regime-D "all" cap_hit_rate <= 2%, frozen capped-cell prompt retention (>= 3/4 Bare & >= 3/4 ChatML),
    #          and |frozen_direct_effect - 3.0| p95 <= 0.30, p99.9 <= 0.45
    cap_rates: dict[str, dict[str, float]] = {}
    max_cap_rate = 0.0
    fde_errs: list[float] = []
    fde_bare: list[float] = []
    fde_chat: list[float] = []
    kl_bare: list[float] = []
    kl_chat: list[float] = []
    retained_prompts_cell: dict[tuple[int, str], list[int]] = {}
    capped_cell_failures: list[str] = []

    for L in LAYERS_4L:
        ls = str(L)
        cap_rates[ls] = {}
        blk = receipt["regime_D_all"][ls]
        for arm in ("R_mf", "logit_lens", "direct_matched"):
            tot = 0
            capped = 0
            for ts in t_strs:
                item = blk[arm][ts]
                draws = item if arm == "direct_matched" else [item]
                for d in draws:
                    for pi, r in enumerate(d):
                        tot += 1
                        if r.get("capped", False):
                            capped += 1
                        else:
                            err = abs(float(r["frozen_direct_effect"]) - 3.0)
                            kv = float(r["kl"])
                            fde_errs.append(err)
                            if pi in bare_idxs:
                                fde_bare.append(err)
                                kl_bare.append(kv)
                            else:
                                fde_chat.append(err)
                                kl_chat.append(kv)
            rate = capped / tot
            cap_rates[ls][arm] = round(rate, 6)
            max_cap_rate = max(max_cap_rate, rate)

        # Frozen capped-cell rule: drop prompt pi from BOTH arms if R_mf OR any dm draw is capped
        for ts in t_strs:
            keep = []
            for pi in range(EXPECTED_N_PROMPTS):
                rmf_cap = bool(blk["R_mf"][ts][pi].get("capped", False))
                dm_cap = any(bool(blk["direct_matched"][ts][k][pi].get("capped", False)) for k in range(EXPECTED_K_DM))
                if not (rmf_cap or dm_cap):
                    keep.append(pi)
            retained_prompts_cell[(L, ts)] = keep
            n_b = sum(1 for pi in keep if pi in bare_idxs)
            n_c = sum(1 for pi in keep if pi in chat_idxs)
            if n_b < 3 or n_c < 3:
                capped_cell_failures.append(f"L{L}.{ts}:bare={n_b}/4,chat={n_c}/4")

    if all_finite and len(fde_errs) > 0:
        fde_p95 = quantile(fde_errs, 0.95)
        fde_p999 = quantile(fde_errs, 0.999)
        fde_max = max(fde_errs)
        c0b_0_3_pass = (
            (max_cap_rate <= 0.02)
            and (len(capped_cell_failures) == 0)
            and (fde_p95 <= 0.30)
            and (fde_p999 <= 0.45)
        )
        fmt_d_diag = {
            "fde_bare_p95": round(quantile(fde_bare, 0.95), 6),
            "fde_bare_max": round(max(fde_bare), 6),
            "fde_chatml_p95": round(quantile(fde_chat, 0.95), 6),
            "fde_chatml_max": round(max(fde_chat), 6),
            "kl_bare_median": round(median(kl_bare), 6),
            "kl_bare_p95": round(quantile(kl_bare, 0.95), 6),
            "kl_chatml_median": round(median(kl_chat), 6),
            "kl_chatml_p95": round(quantile(kl_chat, 0.95), 6),
        }
    else:
        fde_p95 = fde_p999 = fde_max = None
        fmt_d_diag = {}
        c0b_0_3_pass = False

    # C0b.0.4a: Global frozen SHA-256 provenance
    meta = receipt["meta"]
    prov_checks = {
        "r_lens": meta.get("r_lens_sha256") == EXPECTED_R_LENS_SHA256,
        "c0_constants_meta": meta.get("c0_constants_sha256") == EXPECTED_C0_CONSTANTS_SHA256,
        "c0_constants_disk": c0_const_sha == EXPECTED_C0_CONSTANTS_SHA256,
        "c0b_constants_meta": meta.get("c0b_constants_sha256") == EXPECTED_C0B_CONSTANTS_SHA256,
        "c0b_constants_disk": c0b_const_sha == EXPECTED_C0B_CONSTANTS_SHA256,
        "bank_meta": meta.get("bank_sha256") == EXPECTED_C0_BANK_SHA256,
        "c0_receipt_meta": meta.get("c0_receipt_uncompressed_sha256") == EXPECTED_C0_RECEIPT_UNCOMP_SHA256,
        "c0_receipt_disk": c0_rec_uncomp_sha == EXPECTED_C0_RECEIPT_UNCOMP_SHA256,
        "design_meta": meta.get("c0b_design_sha256") == EXPECTED_C0B_DESIGN_SHA256,
        "design_disk": design_checked and not design_missing and design_sha == EXPECTED_C0B_DESIGN_SHA256,
    }
    c0b_0_4a_pass = bool(strict_provenance and all(prov_checks.values()))

    c0b_0_global_passed = all_finite and c0b_0_1_pass and c0b_0_2_pass and c0b_0_3_pass and c0b_0_4a_pass

    # If any non-finite traces exist, return fail-closed blocked scorecard immediately
    if not all_finite:
        gate_c0b_0 = {
            "global_passed": False,
            "all_traces_finite": False,
            "non_finite_issues_sample": non_finite_issues,
            "C0b_0_1_sham": {"passed": c0b_0_1_pass, "last": sham_last, "all": sham_all, "movement_check_passed": move_ok},
            "C0b_0_2_telescoping": {"passed": False, "n_traces": len(all_tel), "expected_n_traces": expected_tel_n},
            "C0b_0_3_cap_and_denom": {"passed": False, "max_cap_hit_rate": round(max_cap_rate, 6)},
            "C0b_0_4a_global_provenance": {"passed": c0b_0_4a_pass, "checks": prov_checks},
            "C0b_0_4b_c0b1b_receipt_drift": {"passed": False, "blocked_reason": "non_finite_traces"},
        }
        return {
            "schema_version": "stage_c0b_scorecard_v1",
            "meta": meta,
            "verdicts": {
                "C0b.0_global": "FAIL",
                "C0b.0.4b_drift": "BLOCKED_BY_C0b.0",
                "C0b.1a": "BLOCKED_BY_C0b.0",
                "C0b.1b": "BLOCKED_BY_C0b.0",
                "C0b.2a": "BLOCKED_BY_C0b.0",
                "C0b.2b": "BLOCKED_BY_C0b.0",
            },
            "gate_C0b_0": gate_c0b_0,
            "C0b_1a": {"verdict": "BLOCKED_BY_C0b.0"},
            "C0b_1b": {"verdict": "BLOCKED_BY_C0b.0"},
            "C0b_2a": {"verdict": "BLOCKED_BY_C0b.0"},
            "C0b_2b": {"verdict": "BLOCKED_BY_C0b.0"},
        }

    # C0b.0.4b: Receipt-pairing drift check (gates C0b.1b ONLY; never voids live-to-live C0b.2)
    # (i) 320 live Regime-D "last" full-pair anchor traces (" lantern", " judge") vs c0_receipt
    anc_gap_diffs: dict[str, float] = {}
    anc_attn_diffs: dict[str, float] = {}
    anc_capped = 0
    for ts in EXPECTED_ANCHOR_TARGETS:
        live_gaps, c0_gaps = [], []
        live_attns, c0_attns = [], []
        for L in LAYERS_4L:
            ls = str(L)
            live_blk = receipt["regime_D_last_anchor"][ls]
            c0_blk = c0_receipt["regime_D"][ls]
            for pi in range(EXPECTED_N_PROMPTS):
                lr = live_blk["R_mf"][ts][pi]
                ldms = [live_blk["direct_matched"][ts][k][pi] for k in range(EXPECTED_K_DM)]
                if lr.get("capped", False) or any(d.get("capped", False) for d in ldms):
                    anc_capped += 1
                    continue
                cr = c0_blk["R_mf"][ts][pi]
                cdms = [c0_blk["direct_matched"][ts][k][pi] for k in range(EXPECTED_K_DM)]
                live_gaps.append(float(lr["exact_dlogit"]) - mean([float(d["exact_dlogit"]) for d in ldms]))
                c0_gaps.append(float(cr["exact_dlogit"]) - mean([float(d["exact_dlogit"]) for d in cdms]))
                live_attns.append(float(lr["sum_attn"]) - mean([float(d["sum_attn"]) for d in ldms]))
                c0_attns.append(float(cr["sum_attn"]) - mean([float(d["sum_attn"]) for d in cdms]))
        anc_gap_diffs[ts] = round(abs(mean(live_gaps) - mean(c0_gaps)), 6) if live_gaps else float("inf")
        anc_attn_diffs[ts] = round(abs(mean(live_attns) - mean(c0_attns)), 6) if live_attns else float("inf")

    max_anc_gap_diff = max(anc_gap_diffs.values())
    max_anc_attn_diff = max(anc_attn_diffs.values())

    # (ii) All 384 live R_mf Regime-N "last" traces (5,760 off-target cells) vs c0_receipt
    live_rmf_n_stats = compute_regime_n_bleed_stats(receipt["regime_N_last"], "R_mf", t_strs, t_domains, LAYERS_3L)
    c0_rmf_n_stats = compute_regime_n_bleed_stats(c0_receipt["regime_N"], "R_mf", t_strs, t_domains, LAYERS_3L)
    b_cross_diffs = [
        abs(a - b) for a, b in zip(live_rmf_n_stats["B_cross_by_target"], c0_rmf_n_stats["B_cross_by_target"])
    ]
    max_b_cross_diff = max(b_cross_diffs)

    max_offtarget_cell_diff = 0.0
    max_all16_cell_diff = 0.0
    n_offtarget_cells = 0
    for L in LAYERS_3L:
        ls = str(L)
        for src_idx, src_str in enumerate(t_strs):
            for pi in range(EXPECTED_N_PROMPTS):
                lv = receipt["regime_N_last"][ls]["R_mf"][src_str][pi]["all_16_dlogits"]
                cv = c0_receipt["regime_N"][ls]["R_mf"][src_str][pi]["all_16_dlogits"]
                for dst_idx in range(n_t):
                    diff = abs(float(lv[dst_idx]) - float(cv[dst_idx]))
                    if diff > max_all16_cell_diff:
                        max_all16_cell_diff = diff
                    if dst_idx != src_idx:
                        n_offtarget_cells += 1
                        if diff > max_offtarget_cell_diff:
                            max_offtarget_cell_diff = diff

    c0b_0_4b_pass = (
        (anc_capped == 0)
        and (max_anc_gap_diff <= 0.05)
        and (max_anc_attn_diff <= 0.05)
        and (max_b_cross_diff <= 0.02)
        and (max_offtarget_cell_diff <= 0.07)
    )

    gate_c0b_0 = {
        "global_passed": c0b_0_global_passed,
        "all_traces_finite": True,
        "non_finite_issues_sample": [],
        "C0b_0_1_sham": {
            "passed": c0b_0_1_pass,
            "last": sham_last,
            "all": sham_all,
            "movement_check_passed": move_ok,
        },
        "C0b_0_2_telescoping": {
            "passed": c0b_0_2_pass,
            "n_traces": len(all_tel),
            "expected_n_traces": expected_tel_n,
            "median": round(tel_med, 6),
            "p95": round(tel_p95, 6),
            "p99_9": round(tel_p999, 6),
            "max": round(tel_max, 6),
            "by_format": tel_fmt_diag,
        },
        "C0b_0_3_cap_and_denom": {
            "passed": c0b_0_3_pass,
            "max_cap_hit_rate": round(max_cap_rate, 6),
            "cap_hit_rate_by_layer_arm": cap_rates,
            "capped_cell_failures": capped_cell_failures,
            "n_retained_traces": len(fde_errs),
            "fde_err_p95": round(fde_p95, 6),
            "fde_err_p99_9": round(fde_p999, 6),
            "fde_err_max": round(fde_max, 6),
            "by_format": fmt_d_diag,
        },
        "C0b_0_4a_global_provenance": {
            "passed": c0b_0_4a_pass,
            "provenance_mode": "strict" if strict_provenance else "relaxed",
            "checks": prov_checks,
            "receipt_uncompressed_sha256": receipt_uncomp_sha,
            "c0b_constants_sha256": c0b_const_sha,
            "c0_constants_sha256": c0_const_sha,
            "c0_receipt_uncompressed_sha256": c0_rec_uncomp_sha,
            "design_sha256": design_sha,
        },
        "C0b_0_4b_c0b1b_receipt_drift": {
            "passed": c0b_0_4b_pass,
            "anchor_targets": EXPECTED_ANCHOR_TARGETS,
            "anchor_capped_count": anc_capped,
            "max_anchor_gap_diff": round(max_anc_gap_diff, 6),
            "anchor_gap_diff_by_target": anc_gap_diffs,
            "max_anchor_attn_gap_diff": round(max_anc_attn_diff, 6),
            "anchor_attn_gap_diff_by_target": anc_attn_diffs,
            "n_offtarget_cells_checked": n_offtarget_cells,
            "max_R_mf_B_cross_diff": round(max_b_cross_diff, 6),
            "max_R_mf_offtarget_cell_diff": round(max_offtarget_cell_diff, 6),
            "max_R_mf_all16_cell_diff": round(max_all16_cell_diff, 6),
        },
    }

    # =========================================================================
    # C0b.1a: Two-sided R_out calibration under positions="all"
    # =========================================================================
    def _cell_r_all(L: int, ts: str, allowed_p: list[int]) -> float:
        ls = str(L)
        blk = receipt["regime_D_all"][ls]
        keep = [pi for pi in retained_prompts_cell[(L, ts)] if pi in allowed_p]
        gaps = []
        for pi in keep:
            r_ex = float(blk["R_mf"][ts][pi]["exact_dlogit"])
            dm_ex = mean([float(blk["direct_matched"][ts][k][pi]["exact_dlogit"]) for k in range(EXPECTED_K_DM)])
            gaps.append(r_ex - dm_ex)
        return mean(gaps) / pred_gap_map[(L, ts)]

    all_p = list(range(EXPECTED_N_PROMPTS))
    r_all_pool = [[_cell_r_all(L, ts, all_p) for L in LAYERS_4L] for ts in t_strs]      # [16][4]
    r_all_bare = [[_cell_r_all(L, ts, bare_idxs) for L in LAYERS_4L] for ts in t_strs]  # [16][4]
    r_all_chat = [[_cell_r_all(L, ts, chat_idxs) for L in LAYERS_4L] for ts in t_strs]  # [16][4]
    r_all_narr = [[_cell_r_all(L, ts, narr_idxs) for L in LAYERS_4L] for ts in t_strs]
    r_all_anal = [[_cell_r_all(L, ts, anal_idxs) for L in LAYERS_4L] for ts in t_strs]

    def _pooled_med(mat: list[list[float]], t_idxs: list[int]) -> float:
        return median([mat[ti][li] for ti in t_idxs for li in range(len(LAYERS_4L))])

    c0b1a_ci_pool = bca_cluster_ci(n_t, lambda idxs: _pooled_med(r_all_pool, idxs), (0.05, 0.95), n_boot=n_boot, seed=SEED_BOOT)
    pooled_med_r_all = _pooled_med(r_all_pool, list(range(n_t)))
    bare_med_r_all = _pooled_med(r_all_bare, list(range(n_t)))
    chat_med_r_all = _pooled_med(r_all_chat, list(range(n_t)))

    per_layer_fmt: dict[str, dict[str, float]] = {}
    max_per_layer_split = 0.0
    all_8_cells_ok = True
    all_4_splits_ok = True
    for li, L in enumerate(LAYERS_4L):
        mb = median([r_all_bare[ti][li] for ti in range(n_t)])
        mc = median([r_all_chat[ti][li] for ti in range(n_t)])
        mp = median([r_all_pool[ti][li] for ti in range(n_t)])
        sp = abs(mc - mb)
        max_per_layer_split = max(max_per_layer_split, sp)
        if not (0.50 <= mb <= 1.50 and 0.50 <= mc <= 1.50):
            all_8_cells_ok = False
        if sp > 0.35:
            all_4_splits_ok = False
        per_layer_fmt[str(L)] = {
            "pooled_median": round(mp, 6),
            "bare_median": round(mb, 6),
            "chatml_median": round(mc, 6),
            "abs_chat_minus_bare": round(sp, 6),
        }

    clause_i = (
        (0.65 <= pooled_med_r_all <= 1.35)
        and (c0b1a_ci_pool["q_0.05"] is not None and c0b1a_ci_pool["q_0.05"] > 0.55)
        and (c0b1a_ci_pool["q_0.95"] is not None and c0b1a_ci_pool["q_0.95"] < 1.50)
    )
    clause_ii = (0.65 <= bare_med_r_all <= 1.35) and (0.65 <= chat_med_r_all <= 1.35)
    clause_iii = all_4_splits_ok
    clause_iv = all_8_cells_ok

    if clause_i and clause_ii and clause_iii and clause_iv:
        c0b1a_verdict = "PASS"
    elif clause_i and clause_ii and clause_iv and (not clause_iii):
        c0b1a_verdict = "INCONCLUSIVE"
    else:
        c0b1a_verdict = "REFUTED"

    # Descriptives for C0b.1a: per-prompt r_all(T), slope(r_all vs T), Retention(logit_lens), |median(dm - ll)|,
    # paired manipulation check Δr_t vs Stage C0 "last", Narrative vs Analytical 95% BCa CIs
    per_prompt_r_all: list[dict] = []
    for pi, p in enumerate(prompts):
        p_vals = []
        for L in LAYERS_4L:
            ls = str(L)
            blk = receipt["regime_D_all"][ls]
            for ts in t_strs:
                if pi in retained_prompts_cell[(L, ts)]:
                    r_ex = float(blk["R_mf"][ts][pi]["exact_dlogit"])
                    dm_ex = mean([float(blk["direct_matched"][ts][k][pi]["exact_dlogit"]) for k in range(EXPECTED_K_DM)])
                    p_vals.append((r_ex - dm_ex) / pred_gap_map[(L, ts)])
        per_prompt_r_all.append({
            "prompt_id": p["id"],
            "format": p["format"],
            "style": p["style"],
            "n_tokens": prompt_tokens[pi],
            "median_r_all": round(median(p_vals), 6),
            "mean_r_all": round(mean(p_vals), 6),
        })
    slope_r_vs_T = ols_slope([float(x["n_tokens"]) for x in per_prompt_r_all], [float(x["median_r_all"]) for x in per_prompt_r_all])

    ll_rets_all, dm_minus_ll_all = [], []
    for L in LAYERS_4L:
        ls = str(L)
        blk = receipt["regime_D_all"][ls]
        for ts in t_strs:
            for pi in retained_prompts_cell[(L, ts)]:
                ll_r = blk["logit_lens"][ts][pi]
                if not ll_r.get("capped", False):
                    ll_rets_all.append(float(ll_r["retention"]))
                    dm_ex = mean([float(blk["direct_matched"][ts][k][pi]["exact_dlogit"]) for k in range(EXPECTED_K_DM)])
                    dm_minus_ll_all.append(dm_ex - float(ll_r["exact_dlogit"]))

    # Paired Δr_t manipulation check vs Stage C0 "last"
    delta_r_by_t: list[float] = []
    for ti, ts in enumerate(t_strs):
        diffs_l = []
        for li, L in enumerate(LAYERS_4L):
            ls = str(L)
            c0_blk = c0_receipt["regime_D"][ls]
            keep = retained_prompts_cell[(L, ts)]
            c0_gaps = [
                float(c0_blk["R_mf"][ts][pi]["exact_dlogit"])
                - mean([float(c0_blk["direct_matched"][ts][k][pi]["exact_dlogit"]) for k in range(EXPECTED_K_DM)])
                for pi in keep
            ]
            r_last_c0 = mean(c0_gaps) / pred_gap_map[(L, ts)]
            diffs_l.append(r_all_pool[ti][li] - r_last_c0)
        delta_r_by_t.append(mean(diffs_l))

    ci_style_diff = bca_cluster_ci(
        n_t,
        lambda idxs: _pooled_med(r_all_narr, idxs) - _pooled_med(r_all_anal, idxs),
        (0.025, 0.975),
        n_boot=n_boot,
        seed=SEED_BOOT + 1,
    )
    ci_fmt_diff = bca_cluster_ci(
        n_t,
        lambda idxs: _pooled_med(r_all_chat, idxs) - _pooled_med(r_all_bare, idxs),
        (0.025, 0.975),
        n_boot=n_boot,
        seed=SEED_BOOT + 2,
    )

    c0b_1a = {
        "verdict": c0b1a_verdict if c0b_0_global_passed else "BLOCKED_BY_C0b.0",
        "raw_verdict_ignoring_gate": c0b1a_verdict,
        "clauses": {
            "i_pooled_calib_and_90ci": clause_i,
            "ii_bare_and_chatml_calib": clause_ii,
            "iii_per_layer_split_le_0p35": clause_iii,
            "iv_all_8_layer_format_cells": clause_iv,
        },
        "pooled_4L_median_r_all": round(pooled_med_r_all, 6),
        "pooled_4L_90pct_bca_ci": c0b1a_ci_pool,
        "bare_4L_median_r_all": round(bare_med_r_all, 6),
        "chatml_4L_median_r_all": round(chat_med_r_all, 6),
        "max_per_layer_abs_chat_minus_bare": round(max_per_layer_split, 6),
        "by_layer": per_layer_fmt,
        "descriptives": {
            "per_prompt_r_all": per_prompt_r_all,
            "slope_r_all_vs_T": round(slope_r_vs_T, 6),
            "manipulation_check_mean_delta_r_vs_C0": round(mean(delta_r_by_t), 6),
            "manipulation_check_p_sign_flip_delta_r": round(exact_sign_flip_p_one_sided(delta_r_by_t), 6),
            "retention_logit_lens_all_median": round(median(ll_rets_all), 6) if ll_rets_all else None,
            "abs_median_dm_minus_ll_all": round(abs(median(dm_minus_ll_all)), 6) if dm_minus_ll_all else None,
            "chatml_minus_bare_95pct_bca_ci": ci_fmt_diff,
            "narrative_minus_analytical_95pct_bca_ci": ci_style_diff,
        },
    }

    # =========================================================================
    # C0b.1b: ΣAttn share & magnitude of the "all" - "last" gap shift
    # =========================================================================
    delta_guard = float(c0b_const["stage_c0_last_baselines"]["c0b1b_delta_tot_guard_5pct"])
    attn_floor = float(c0b_const["stage_c0_last_baselines"]["c0b1b_attn_lift_floor_50pct"])

    dtot_by_t: list[float] = []
    dattn_by_t: list[float] = []
    dmlp_by_t: list[float] = []
    share_by_t: list[float] = []
    by_layer_attn_diag: dict[str, dict[str, float]] = {}

    for L in LAYERS_4L:
        ls = str(L)
        a_all_l, m_all_l, t_all_l = [], [], []
        a_last_l, m_last_l, t_last_l = [], [], []
        for ts in t_strs:
            keep = retained_prompts_cell[(L, ts)]
            lb_all = receipt["regime_D_all"][ls]
            lb_c0 = c0_receipt["regime_D"][ls]
            for pi in keep:
                a_all_l.append(
                    float(lb_all["R_mf"][ts][pi]["sum_attn"])
                    - mean([float(lb_all["direct_matched"][ts][k][pi]["sum_attn"]) for k in range(EXPECTED_K_DM)])
                )
                m_all_l.append(
                    float(lb_all["R_mf"][ts][pi]["sum_mlp"])
                    - mean([float(lb_all["direct_matched"][ts][k][pi]["sum_mlp"]) for k in range(EXPECTED_K_DM)])
                )
                t_all_l.append(
                    float(lb_all["R_mf"][ts][pi]["exact_dlogit"])
                    - mean([float(lb_all["direct_matched"][ts][k][pi]["exact_dlogit"]) for k in range(EXPECTED_K_DM)])
                )
                a_last_l.append(
                    float(lb_c0["R_mf"][ts][pi]["sum_attn"])
                    - mean([float(lb_c0["direct_matched"][ts][k][pi]["sum_attn"]) for k in range(EXPECTED_K_DM)])
                )
                m_last_l.append(
                    float(lb_c0["R_mf"][ts][pi]["sum_mlp"])
                    - mean([float(lb_c0["direct_matched"][ts][k][pi]["sum_mlp"]) for k in range(EXPECTED_K_DM)])
                )
                t_last_l.append(
                    float(lb_c0["R_mf"][ts][pi]["exact_dlogit"])
                    - mean([float(lb_c0["direct_matched"][ts][k][pi]["exact_dlogit"]) for k in range(EXPECTED_K_DM)])
                )
        da_m = mean(a_all_l) - mean(a_last_l)
        dm_m = mean(m_all_l) - mean(m_last_l)
        dt_m = mean(t_all_l) - mean(t_last_l)
        by_layer_attn_diag[ls] = {
            "mean_attn_gap_all": round(mean(a_all_l), 4),
            "mean_attn_gap_last_C0": round(mean(a_last_l), 4),
            "delta_attn": round(da_m, 4),
            "delta_mlp": round(dm_m, 4),
            "delta_tot": round(dt_m, 4),
            "aggregate_attn_share": round(da_m / dt_m, 4) if abs(dt_m) > 1e-9 else None,
        }

    for ts in t_strs:
        dt_ls, da_ls, dm_ls = [], [], []
        for L in LAYERS_4L:
            ls = str(L)
            keep = retained_prompts_cell[(L, ts)]
            lb_all = receipt["regime_D_all"][ls]
            lb_c0 = c0_receipt["regime_D"][ls]
            g_all = mean([
                float(lb_all["R_mf"][ts][pi]["exact_dlogit"])
                - mean([float(lb_all["direct_matched"][ts][k][pi]["exact_dlogit"]) for k in range(EXPECTED_K_DM)])
                for pi in keep
            ])
            g_last = mean([
                float(lb_c0["R_mf"][ts][pi]["exact_dlogit"])
                - mean([float(lb_c0["direct_matched"][ts][k][pi]["exact_dlogit"]) for k in range(EXPECTED_K_DM)])
                for pi in keep
            ])
            a_all = mean([
                float(lb_all["R_mf"][ts][pi]["sum_attn"])
                - mean([float(lb_all["direct_matched"][ts][k][pi]["sum_attn"]) for k in range(EXPECTED_K_DM)])
                for pi in keep
            ])
            a_last = mean([
                float(lb_c0["R_mf"][ts][pi]["sum_attn"])
                - mean([float(lb_c0["direct_matched"][ts][k][pi]["sum_attn"]) for k in range(EXPECTED_K_DM)])
                for pi in keep
            ])
            m_all = mean([
                float(lb_all["R_mf"][ts][pi]["sum_mlp"])
                - mean([float(lb_all["direct_matched"][ts][k][pi]["sum_mlp"]) for k in range(EXPECTED_K_DM)])
                for pi in keep
            ])
            m_last = mean([
                float(lb_c0["R_mf"][ts][pi]["sum_mlp"])
                - mean([float(lb_c0["direct_matched"][ts][k][pi]["sum_mlp"]) for k in range(EXPECTED_K_DM)])
                for pi in keep
            ])
            dt_ls.append(g_all - g_last)
            da_ls.append(a_all - a_last)
            dm_ls.append(m_all - m_last)
        dt_t = mean(dt_ls)
        da_t = mean(da_ls)
        dm_t = mean(dm_ls)
        dtot_by_t.append(dt_t)
        dattn_by_t.append(da_t)
        dmlp_by_t.append(dm_t)
        share_by_t.append((da_t / dt_t) if dt_t > delta_guard else 0.0)

    n_valid_dtot = sum(1 for v in dtot_by_t if v > delta_guard)
    mean_dtot = mean(dtot_by_t)
    mean_dattn = mean(dattn_by_t)
    med_share = median(share_by_t)
    ci_share = bca_cluster_ci(
        n_t,
        lambda idxs: median([share_by_t[i] for i in idxs]),
        (0.05, 0.95),
        n_boot=n_boot,
        seed=SEED_BOOT + 3,
    )

    c0b1b_raw_pass = (
        (mean_dtot > delta_guard)
        and (mean_dattn >= attn_floor)
        and (n_valid_dtot >= 13)
        and (med_share >= 0.50)
        and (ci_share["q_0.05"] is not None and ci_share["q_0.05"] > 0.35)
    )
    if not c0b_0_global_passed:
        c0b1b_verdict = "BLOCKED_BY_C0b.0"
    elif not c0b_0_4b_pass:
        c0b1b_verdict = "VOID_DRIFT"
    else:
        c0b1b_verdict = "PASS" if c0b1b_raw_pass else "FAIL"

    c0b_1b = {
        "verdict": c0b1b_verdict,
        "raw_pass_ignoring_gates": c0b1b_raw_pass,
        "delta_tot_guard": round(delta_guard, 6),
        "attn_lift_floor": round(attn_floor, 6),
        "mean_delta_tot": round(mean_dtot, 6),
        "mean_delta_attn": round(mean_dattn, 6),
        "mean_delta_mlp": round(mean(dmlp_by_t), 6),
        "n_targets_gt_delta": n_valid_dtot,
        "median_share": round(med_share, 6),
        "share_90pct_bca_ci": ci_share,
        "manipulation_check_p_sign_flip_dattn": round(exact_sign_flip_p_one_sided(dattn_by_t), 6),
        "by_layer": by_layer_attn_diag,
    }

    # =========================================================================
    # C0b.2a & C0b.2b: Live-paired R_mf vs R_ridge cross-domain whitening (3L)
    # =========================================================================
    live_rr_n_stats = compute_regime_n_bleed_stats(receipt["regime_N_last"], "R_ridge", t_strs, t_domains, LAYERS_3L)
    live_ll_n_stats = compute_regime_n_bleed_stats(receipt["regime_N_last"], "logit_lens", t_strs, t_domains, LAYERS_3L)
    live_rmf_all_n_stats = compute_regime_n_bleed_stats(receipt["regime_N_all"], "R_mf", t_strs, t_domains, LAYERS_3L)
    live_rr_all_n_stats = compute_regime_n_bleed_stats(receipt["regime_N_all"], "R_ridge", t_strs, t_domains, LAYERS_3L)

    u_abs = [
        math.log(max(bm, 1e-6) / max(br, 1e-6))
        for bm, br in zip(live_rmf_n_stats["B_cross_by_target"], live_rr_n_stats["B_cross_by_target"])
    ]
    u_norm = [
        math.log(max(sm, 1e-6) / max(sr, 1e-6))
        for sm, sr in zip(live_rmf_n_stats["S_cross_by_target"], live_rr_n_stats["S_cross_by_target"])
    ]
    p_ridge = exact_sign_flip_p_one_sided(u_abs)
    abs_ratio_geo = math.exp(mean(u_abs))
    norm_ratio_geo = math.exp(mean(u_norm))
    n_pos_u_abs = sum(
        1
        for bm, br in zip(live_rmf_n_stats["B_cross_by_target"], live_rr_n_stats["B_cross_by_target"])
        if bm > br
    )
    ci_abs_ratio = bca_cluster_ci(
        n_t,
        lambda idxs: math.exp(mean([u_abs[i] for i in idxs])),
        (0.05, 0.95),
        n_boot=n_boot,
        seed=SEED_BOOT + 4,
    )
    ci_norm_ratio = bca_cluster_ci(
        n_t,
        lambda idxs: math.exp(mean([u_norm[i] for i in idxs])),
        (0.05, 0.95),
        n_boot=n_boot,
        seed=SEED_BOOT + 5,
    )

    c0b2a_pass_cond = (
        math.isfinite(p_ridge)
        and (p_ridge <= 0.05)
        and (abs_ratio_geo >= 1.25)
        and (n_pos_u_abs >= 13)
        and (norm_ratio_geo >= 1.25)
    )
    c0b2a_refute_cond = (
        (ci_abs_ratio["q_0.95"] is not None and ci_abs_ratio["q_0.95"] < 1.25)
        and (ci_norm_ratio["q_0.95"] is not None and ci_norm_ratio["q_0.95"] < 1.25)
    )
    if c0b2a_pass_cond:
        c0b2a_raw_verdict = "PASS"
    elif c0b2a_refute_cond:
        c0b2a_raw_verdict = "REFUTED"
    else:
        c0b2a_raw_verdict = "INCONCLUSIVE"

    # Secondary paired "all" Regime N R_mf vs R_ridge + R_mf frac_err under "all"
    u_abs_all = [
        math.log(max(bm, 1e-6) / max(br, 1e-6))
        for bm, br in zip(live_rmf_all_n_stats["B_cross_by_target"], live_rr_all_n_stats["B_cross_by_target"])
    ]
    rmf_all_frac_errs = []
    for L in LAYERS_3L:
        ls = str(L)
        for ts in t_strs:
            rows = receipt["regime_N_all"][ls]["R_mf"][ts]
            ex_m = mean([float(r["exact_dlogit"]) for r in rows])
            pr_m = mean([float(r["pred_dlogit"]) for r in rows])
            rmf_all_frac_errs.append(abs(ex_m - pr_m) / max(abs(pr_m), 1e-6))

    c0b_2a = {
        "verdict": c0b2a_raw_verdict if c0b_0_global_passed else "BLOCKED_BY_C0b.0",
        "raw_verdict_ignoring_gate": c0b2a_raw_verdict,
        "p_ridge_sign_flip": round(p_ridge, 6) if math.isfinite(p_ridge) else None,
        "abs_ratio_geo_Rmf_over_Rridge": round(abs_ratio_geo, 6),
        "abs_ratio_90pct_bca_ci": ci_abs_ratio,
        "norm_ratio_geo_Smf_over_Sridge": round(norm_ratio_geo, 6),
        "norm_ratio_90pct_bca_ci": ci_norm_ratio,
        "n_targets_Rmf_gt_Rridge": n_pos_u_abs,
        "live_R_mf_last": {
            "B_cross_arith_mean": round(live_rmf_n_stats["B_cross_arith_mean"], 6),
            "B_cross_geo_mean": round(live_rmf_n_stats["B_cross_geo_mean"], 6),
            "S_cross_geo_mean": round(live_rmf_n_stats["S_cross_geo_mean"], 6),
            "self_dlogit_geo_mean": round(live_rmf_n_stats["self_dlogit_geo_mean"], 6),
            "signed_cross_mean": round(live_rmf_n_stats["signed_cross_mean"], 6),
            "frac_cross_positive": round(live_rmf_n_stats["frac_cross_positive"], 6),
            "same_domain_wrong_abs_mean": round(live_rmf_n_stats["same_domain_wrong_abs_mean"], 6),
        },
        "live_R_ridge_last": {
            "B_cross_arith_mean": round(live_rr_n_stats["B_cross_arith_mean"], 6),
            "B_cross_geo_mean": round(live_rr_n_stats["B_cross_geo_mean"], 6),
            "S_cross_geo_mean": round(live_rr_n_stats["S_cross_geo_mean"], 6),
            "self_dlogit_geo_mean": round(live_rr_n_stats["self_dlogit_geo_mean"], 6),
            "signed_cross_mean": round(live_rr_n_stats["signed_cross_mean"], 6),
            "frac_cross_positive": round(live_rr_n_stats["frac_cross_positive"], 6),
            "same_domain_wrong_abs_mean": round(live_rr_n_stats["same_domain_wrong_abs_mean"], 6),
        },
        "descriptives": {
            "live_logit_lens_last_B_cross_arith_mean": round(live_ll_n_stats["B_cross_arith_mean"], 6),
            "live_logit_lens_last_signed_cross_mean": round(live_ll_n_stats["signed_cross_mean"], 6),
            "live_self_ratio_Rridge_over_Rmf_geo": round(
                live_rr_n_stats["self_dlogit_geo_mean"] / live_rmf_n_stats["self_dlogit_geo_mean"], 6
            ),
            "secondary_all_R_mf_B_cross_arith_mean": round(live_rmf_all_n_stats["B_cross_arith_mean"], 6),
            "secondary_all_R_ridge_B_cross_arith_mean": round(live_rr_all_n_stats["B_cross_arith_mean"], 6),
            "secondary_all_abs_ratio_geo": round(math.exp(mean(u_abs_all)), 6),
            "secondary_all_R_mf_median_frac_err": round(median(rmf_all_frac_errs), 6),
        },
    }

    # C0b.2b: Absolute noise-floor criterion (mean_t B_cross(R_ridge) <= F_mean_3L + 0.10)
    f_mean_3L = float(c0b_const["stage_c0_last_baselines"]["regime_N_3L_L18_24_27_measured"]["rand_iso_3L_F_mean"])
    f_p95_3L_pair = float(c0b_const["stage_c0_last_baselines"]["regime_N_3L_L18_24_27_measured"]["rand_iso_3L_F_p95_512pair"])
    floor_bar = f_mean_3L + 0.10
    rr_b_mean = live_rr_n_stats["B_cross_arith_mean"]
    c0b2b_raw_pass = rr_b_mean <= floor_bar
    c0b_2b = {
        "verdict": ("PASS" if c0b2b_raw_pass else "FAIL") if c0b_0_global_passed else "BLOCKED_BY_C0b.0",
        "raw_pass_ignoring_gate": c0b2b_raw_pass,
        "R_ridge_B_cross_arith_mean": round(rr_b_mean, 6),
        "F_mean_3L_C0": round(f_mean_3L, 6),
        "F_p95_3L_512pair_C0": round(f_p95_3L_pair, 6),
        "noise_floor_threshold": round(floor_bar, 6),
        "ratio_R_ridge_to_F_mean_3L": round(rr_b_mean / f_mean_3L, 6),
        "ratio_R_ridge_to_R_mf": round(rr_b_mean / live_rmf_n_stats["B_cross_arith_mean"], 6),
    }

    return {
        "schema_version": "stage_c0b_scorecard_v1",
        "meta": meta,
        "verdicts": {
            "C0b.0_global": "PASS" if c0b_0_global_passed else "FAIL",
            "C0b.0.4b_drift": ("PASS" if c0b_0_4b_pass else "FAIL") if c0b_0_global_passed else "BLOCKED_BY_C0b.0",
            "C0b.1a": c0b_1a["verdict"],
            "C0b.1b": c0b_1b["verdict"],
            "C0b.2a": c0b_2a["verdict"],
            "C0b.2b": c0b_2b["verdict"],
        },
        "gate_C0b_0": gate_c0b_0,
        "C0b_1a": c0b_1a,
        "C0b_1b": c0b_1b,
        "C0b_2a": c0b_2a,
        "C0b_2b": c0b_2b,
    }


def render_scorecard_md(sc: dict) -> str:
    v = sc["verdicts"]
    g = sc["gate_C0b_0"]
    lines = [
        "# Stage C0b Pre-Registered Scorecard (`OLMO3-TRANSPORT-C0B-DESIGN.md` Registered v1)",
        "",
        "| Check | Verdict | Key Metric |",
        "|---|---|---|",
        f"| **Gate `C0b.0` (global `1–4a`)** | **{v['C0b.0_global']}** | "
        f"tel p95={g['C0b_0_2_telescoping'].get('p95')}, cap={g['C0b_0_3_cap_and_denom'].get('max_cap_hit_rate')}, "
        f"fde p95={g['C0b_0_3_cap_and_denom'].get('fde_err_p95')} |",
        f"| **Gate `C0b.0.4b` (`C0b.1b` drift)** | **{v['C0b.0.4b_drift']}** | "
        f"anc_gap={g['C0b_0_4b_c0b1b_receipt_drift'].get('max_anchor_gap_diff')}, "
        f"anc_attn={g['C0b_0_4b_c0b1b_receipt_drift'].get('max_anchor_attn_gap_diff')}, "
        f"B_cross={g['C0b_0_4b_c0b1b_receipt_drift'].get('max_R_mf_B_cross_diff')}, "
        f"cell={g['C0b_0_4b_c0b1b_receipt_drift'].get('max_R_mf_offtarget_cell_diff')} |",
    ]
    c1a = sc["C0b_1a"]
    c1b = sc["C0b_1b"]
    c2a = sc["C0b_2a"]
    c2b = sc["C0b_2b"]
    if "pooled_4L_median_r_all" in c1a:
        ci = c1a["pooled_4L_90pct_bca_ci"]
        lines.append(
            f"| **`C0b.1a` (`positions=\"all\"` calib)** | **{v['C0b.1a']}** | "
            f"pooled={c1a['pooled_4L_median_r_all']} (90% CI [{ci['q_0.05']}, {ci['q_0.95']}]), "
            f"Bare={c1a['bare_4L_median_r_all']}, ChatML={c1a['chatml_4L_median_r_all']}, "
            f"max|Δ(L)|={c1a['max_per_layer_abs_chat_minus_bare']} |"
        )
    if "median_share" in c1b:
        ci_s = c1b["share_90pct_bca_ci"]
        lines.append(
            f"| **`C0b.1b` (`ΣAttn` share & lift)** | **{v['C0b.1b']}** | "
            f"mean ΔTot={c1b['mean_delta_tot']}, mean ΔAttn={c1b['mean_delta_attn']} (floor {c1b['attn_lift_floor']}), "
            f"n>δ={c1b['n_targets_gt_delta']}/16, share={c1b['median_share']} (q05={ci_s['q_0.05']}) |"
        )
    if "abs_ratio_geo_Rmf_over_Rridge" in c2a:
        cia = c2a["abs_ratio_90pct_bca_ci"]
        cin = c2a["norm_ratio_90pct_bca_ci"]
        lines.append(
            f"| **`C0b.2a` (`R_ridge` relative whitening)** | **{v['C0b.2a']}** | "
            f"abs={c2a['abs_ratio_geo_Rmf_over_Rridge']}× (q95={cia['q_0.95']}), "
            f"norm={c2a['norm_ratio_geo_Smf_over_Sridge']}× (q95={cin['q_0.95']}), "
            f"{c2a['n_targets_Rmf_gt_Rridge']}/16, p={c2a['p_ridge_sign_flip']} |"
        )
    if "R_ridge_B_cross_arith_mean" in c2b:
        lines.append(
            f"| **`C0b.2b` (`R_ridge` noise floor)** | **{v['C0b.2b']}** | "
            f"B_cross(R_ridge)={c2b['R_ridge_B_cross_arith_mean']} vs bar {c2b['noise_floor_threshold']} "
            f"({c2b['ratio_R_ridge_to_F_mean_3L']}× F_mean_3L) |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stdlib-only deterministic scorer for Stage C0b.")
    ap.add_argument("--receipt", type=Path, required=True, help="Path to c0b_receipt.json.gz")
    ap.add_argument("--c0b-constants", type=Path, required=True, help="Path to c0b_prereg_constants.json")
    ap.add_argument("--c0-constants", type=Path, required=True, help="Path to c0_prereg_constants.json")
    ap.add_argument("--c0-receipt", type=Path, required=True, help="Path to published c0_receipt.json.gz")
    ap.add_argument("--design", type=Path, default=None, help="Path to registered OLMO3-TRANSPORT-C0B-DESIGN.md")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=BOOT_RESAMPLES)
    ap.add_argument("--no-strict-provenance", action="store_true")
    args = ap.parse_args()

    sc = score_c0b(
        receipt_path=args.receipt,
        c0b_constants_path=args.c0b_constants,
        c0_constants_path=args.c0_constants,
        c0_receipt_path=args.c0_receipt,
        design_path=args.design,
        n_boot=args.n_boot,
        strict_provenance=not args.no_strict_provenance,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "scorecard.json").write_text(json.dumps(sc, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "SCORECARD.md").write_text(render_scorecard_md(sc), encoding="utf-8")
    print(f"Wrote {args.out_dir / 'scorecard.json'} and SCORECARD.md | Verdicts: {sc['verdicts']}")


if __name__ == "__main__":
    main()
