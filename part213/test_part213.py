"""CPU-only pytest for part213/score213.py on synthetic records: no model, no download, no network.

Every required test builds a synthetic record set, scores it with score213.score(), and asserts the disposition the
design's v5 rule (PART213-DESIGN.md l.427-431) demands. The sabotage checks at the bottom re-run each required test
against a deliberately broken COPY of score213.py written to tmp_path (never the tracked file) and require it to fail.

Run: /home/elisepc/neuron-probe/.venv/bin/python -m pytest part213/test_part213.py -p no:cacheprovider
"""
from __future__ import annotations

import importlib.util
import json
import math

import sys
from pathlib import Path

import pytest
from statistics import median

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import score213 as s  # noqa: E402

TARGETS = json.loads((HERE / "targets213.json").read_text(encoding="utf-8"))
UNITS = TARGETS["units"]
UNIT_BY_ID = {u["id"]: u for u in UNITS}
SLOT = [u["id"] for u in UNITS if u["kind_haiku"] == "slot"]
CONCEPT = [u["id"] for u in UNITS if u["kind_haiku"] == "concept"]
SELF_HELD = s.SELF_HELD
THIRD_SLOT = s.THIRD_SLOT
MENTION = s.MENTION
COPULA = s.COPULA_NONID
SELF_NONID = s.SELF_NONID
ALL_PROMPTS = [p["id"] for plist in TARGETS["stage1_prompts"].values() for p in plist]


# ---------------------------------------------------------------------------------------------------------------------
# synthetic record builder
# ---------------------------------------------------------------------------------------------------------------------

class Builder:
    """Builds synthetic Stage 1 records with chosen per-unit SpecLogit increments.

    The runner's records are per (unit, arm, prompt) rows, each carrying the whole vocabulary logit row, so a neuron
    pass for unit u moves every unit j whose O_j holds u's first token by -value/|O_j| (and its own by +value).
    `records()` applies exactly that coupling to every unit, so N_z / R_z / L_z come out at the requested values for
    the units whose O_u holds no other moving unit (_exact_units); `solo_records()` instead writes the pushed row only
    for the unit being pushed (every other unit's row keeps its none logits), which makes every value exact.
    """

    def __init__(self):
        self.units = UNITS
        self.n = len(UNITS)
        self.O = [[UNIT_BY_ID[v]["first_content_token_id"] for v in u["offdiag_ou"]] for u in UNITS]
        self.own = [u["first_content_token_id"] for u in UNITS]
        # membership[i] = units j whose O_j contains unit i's first token (their median shifts when i moves)
        self.membership = [[j for j in range(self.n) if self.own[i] in self.O[j]] for i in range(self.n)]

    def logits(self, own_values):
        """A synthetic vocabulary row: unit i's first-content token at own_values[i]."""
        base = [0.0] * (max(self.own) + 1)
        for i in range(self.n):
            base[self.own[i]] = own_values[i]
        return base

    def coupled(self, values):
        """Own-logit offsets realising `values` when every unit's own logit moves by its value."""
        x = list(values)
        for _ in range(200):
            med = [median(x[j] for j in self.membership[i]) for i in range(self.n)]
            x = [values[i] - med[i] for i in range(self.n)]
        return x

    def records(self, ckpt, prompt_ids, nz, rz, lp_delta=None, status="ok"):
        """nz/rz: {unit_id: value}; missing units get 0.0. Every unit's row carries the coupled logit row."""
        rows = []
        none_row = self.logits([0.0] * self.n)
        for pid in prompt_ids:
            for unit in UNITS:
                rows.append(self._row(unit, pid, "none", "0.0", 0.0, none_row, 0.0, status))
            for unit in UNITS:
                uid = unit["id"]
                arms = [("neuron", "+0.8", nz.get(uid, 0.0))]
                if ckpt == "instruct":
                    arms += [(f"random_{m}", "+0.8", rz.get(uid, 0.0)) for m in range(4)]
                else:
                    arms += [("random_0", "+0.8", rz.get(uid, 0.0))]
                for arm, dose, val in arms:
                    values = [0.0] * self.n
                    for j, other in enumerate(UNITS):
                        values[j] = (nz if arm == "neuron" else rz).get(other["id"], 0.0)
                    rows.append(self._row(unit, pid, arm, dose, 0.8, self.logits(self.coupled(values)), val,
                                          status, lp_delta=(lp_delta or {}).get(uid, 0.0)))
        return rows

    def solo_records(self, ckpt, prompt_ids, nz, rz, lp_delta=None, status="ok"):
        """Like records(), but the pushed row moves only the pushed unit's own logit: every N_z / R_z is exact."""
        rows = []
        none_row = self.logits([0.0] * self.n)
        for pid in prompt_ids:
            for unit in UNITS:
                rows.append(self._row(unit, pid, "none", "0.0", 0.0, none_row, 0.0, status))
            for i, unit in enumerate(UNITS):
                uid = unit["id"]
                arms = [("neuron", "+0.8", nz.get(uid, 0.0))]
                if ckpt == "instruct":
                    arms += [(f"random_{m}", "+0.8", rz.get(uid, 0.0)) for m in range(4)]
                else:
                    arms += [("random_0", "+0.8", rz.get(uid, 0.0))]
                for arm, dose, val in arms:
                    values = [0.0] * self.n
                    values[i] = val
                    rows.append(self._row(unit, pid, arm, dose, 0.8, self.logits(values), val, status,
                                          lp_delta=(lp_delta or {}).get(uid, 0.0)))
        return rows

    def _row(self, unit, pid, arm, dose, alpha, logits, val, status="ok", lp_delta=0.0, own=None):
        n = int(unit["n_content_tokens"])
        return {"unit": own or unit["id"], "kind": unit["kind_haiku"], "layer": unit["layer"], "neuron": unit["neuron"],
                "prompt_id": pid, "arm": arm, "dose": dose, "alpha": alpha, "status": status,
                "first_logits": logits, "lp_sum": 10.0 * n + lp_delta * n, "lp_mean": 10.0 + lp_delta,
                "lp_first": -1.0, "entropy": 3.0, "top1": 0.2, "top1_id": 0,
                "n_content_tokens": n, "n_target_tokens": int(unit["n_target_tokens"])}

    # -- full-NP specificity rows (213.11) ---------------------------------------------------------------------------
    def full_np_rows(self, prompt_ids, own_delta, peer_delta, peer_lengths=None):
        """none rows, a neuron +0.8 row per unit, and the Amendment-1 offdiag passes.

        A unit u's own full-NP increment comes from its own `neuron +0.8` row (`own_delta`); a peer v's increment
        under u's hook comes from the `(push_unit=u, unit=v, prompt)` offdiag row (`peer_delta`), which is exactly
        what `Stage1.full_np_increment` reads after F3. `peer_lengths` overrides every NP's token count.
        """
        def length_of(unit):
            return peer_lengths if peer_lengths is not None else int(unit["n_content_tokens"])

        rows = []
        for pid in prompt_ids:
            for unit in UNITS:
                row = self._row(unit, pid, "none", "0.0", 0.0, self.logits([0.0] * self.n), 0.0)
                row["lp_sum"], row["lp_mean"] = 10.0 * length_of(unit), 10.0
                rows.append(row)
            for unit in UNITS:
                row = self._row(unit, pid, "neuron", "+0.8", 0.8, self.logits([0.0] * self.n), 0.0)
                row["lp_sum"] = (10.0 + own_delta) * length_of(unit)
                row["lp_mean"] = 10.0 + own_delta
                rows.append(row)
            for uid in SLOT:
                for vid in UNIT_BY_ID[uid]["offdiag_8"]:
                    v = UNIT_BY_ID[vid]
                    row = self._row(v, pid, "offdiag", "+0.8", 0.8, self.logits([0.0] * self.n), 0.0)
                    row["push_unit"], row["scored_unit"] = uid, vid
                    row["lp_sum"] = (10.0 + peer_delta) * length_of(v)
                    row["lp_mean"] = 10.0 + peer_delta
                    rows.append(row)
        return rows


def empty_records(gate2_passed=True):
    """Synthetic records whose run.json passes Gate 1 and Gate 2, so Stage 1 predictions are scored, not voided."""
    runs = {ckpt: {"dtype": "float32", "attn_implementation": "eager", "snapshot_commit": "48d788eca847",
                   "resolved": {"commit_prefix": "48d788eca847"},
                   "tokenizer_sha256": {"tokenizer.json": "x", "tokenizer_config.json": "y"}}
            for ckpt in ("sft", "dpo", "instruct")}
    runs["instruct"]["pilot_gate1"] = {"zero_hook_max_abs_diff": 0.0,
                                       "batch_padded_vs_batch1_max_abs_diff": 1e-6}
    runs["instruct"]["gate2_anchors_detail"] = [{"id": "p92-02", "passed": gate2_passed}]
    return {"stage1": {c: [] for c in s.CHECKPOINTS}, "stage2": [], "runs": runs}


def score(records, audit=None):
    return s.score(records, TARGETS, audit)


# ---------------------------------------------------------------------------------------------------------------------
# 1. the v5 disposition rule
# ---------------------------------------------------------------------------------------------------------------------

def test_disposition_rule_straddling_failed_and_low_n():
    assert s.ci_clause({"lo": 0.40, "hi": 0.70}, "ge", 0.50) == "inconclusive"      # straddles the bar
    assert s.ci_clause({"lo": 0.10, "hi": 0.40}, "ge", 0.50) == "failed"            # wholly on the wrong side
    assert s.ci_clause({"lo": 0.55, "hi": 0.90}, "ge", 0.50) == "held"              # wholly on the predicted side
    assert s.ci_clause({"lo": 0.55, "hi": 0.90}, "le", 0.50) == "failed"
    assert s.ci_clause({"lo": 0.60, "hi": 0.90}, "ge", 0.50, n_eligible=14) == "not read"  # n_eligible < 15
    assert s.point_clause(False) == "failed" and s.point_clause(True) == "held"
    # one failed clause fails the prediction, even beside held and inconclusive clauses
    assert s.disposition({"a": "held", "b": "inconclusive", "c": "failed"}) == "failed"
    assert s.disposition({"a": "held", "b": "inconclusive"}) == "inconclusive"
    assert s.disposition({"a": "held", "b": "not read"}) == "not read"
    assert s.disposition({"a": "held", "b": "held"}) == "held"


# ---------------------------------------------------------------------------------------------------------------------
# 2. the numerator sign guard
# ---------------------------------------------------------------------------------------------------------------------

def test_sign_guard_suppression_both_cells_is_inconclusive():
    b = Builder()
    rec = empty_records()
    # SELF_HELD (the denominator) +2.0: eligible; THIRD_SLOT (the numerator) -1.5 in both style halves
    self_v = {u: 2.0 for u in SLOT}
    third_direct = {u: -1.5 for u in SLOT}
    third_scene = {u: -1.5 for u in SLOT}
    rec["stage1"]["instruct"] = b.solo_records("instruct", SELF_HELD, self_v, {u: 0.0 for u in SLOT})
    for pid in ("third_0", "third_1"):
        rec["stage1"]["instruct"] += b.solo_records("instruct", (pid,), third_direct, {u: 0.0 for u in SLOT})
    for pid in ("third_2", "third_3"):
        rec["stage1"]["instruct"] += b.solo_records("instruct", (pid,), third_scene, {u: 0.0 for u in SLOT})
    result = score(rec)
    p3 = result["predictions"]["213.3"]
    tau = p3["tau_3/1_L"]
    # the denominator is > 0.50 for every unit a solo push can move alone (19 of 21: p92-18 and p92-37 share a
    # first token with p92-20 and p92-49, so a solo push cannot move them) ...
    assert tau["n_eligible"] == 19
    assert tau["point"] < 0                            # ... and the ratio is negative in every unit
    assert tau["numerator_ci"]["lo"] < 0               # the numerator is suppressed in every unit
    assert tau["sign_guard"] is False and tau["read"] == "inconclusive"
    assert p3["clauses"]["tau_L_lo_ge_0.50"] == "inconclusive"
    # the style-stratified medians are negative too, and a point clause that misses fails the prediction under the v5
    # rule (design l.430-431) -- the point is that the ratio clause itself is `inconclusive`, never `held`
    assert p3["clauses"]["direct_style_median_ge_0.50"] == "failed"
    assert p3["verdict"] == "failed"


# ---------------------------------------------------------------------------------------------------------------------
# 3. scene-framed denominators
# ---------------------------------------------------------------------------------------------------------------------

def test_scene_framed_denominators_decide_and_pooled_does_not():
    b = Builder()
    rec = empty_records()
    # MENTION numerator +0.8; THIRD_SLOT scene prompts (third_2, third_3) +2.0, direct prompts (third_0, third_1) 0.0
    mention = {u: 0.8 for u in SLOT}
    rec["stage1"]["instruct"] = b.solo_records("instruct", MENTION, mention, {u: 0.0 for u in SLOT})
    for pid in ("third_2", "third_3"):
        rec["stage1"]["instruct"] += b.solo_records("instruct", (pid,), {u: 2.0 for u in SLOT}, {u: 0.0 for u in SLOT})
    for pid in ("third_0", "third_1"):
        rec["stage1"]["instruct"] += b.solo_records("instruct", (pid,), {u: 0.0 for u in SLOT}, {u: 0.0 for u in SLOT})
    result = score(rec)
    p4 = result["predictions"]["213.4"]
    tau = p4["tau_M/3_scene"]
    pooled = p4["tau_M/3_pooled_descriptive"]
    assert tau["denominator_prompts"] == ["third_2", "third_3"]
    assert tau["n_eligible"] == 19 and abs(tau["point"] - 0.4) < 1e-6
    assert pooled["denominator_prompts"] == list(THIRD_SLOT)
    assert abs(pooled["point"] - 0.8) < 1e-6           # the pooled version is computed ...
    # ... and decides nothing: the clause reads the scene ratio (0.40 < 0.50), not the pooled 0.80
    assert p4["clauses"]["tau_M/3_L_lo_ge_0.50"] == "failed"
    assert p4["clauses"]["median_L_lo_ge_0.50"] == "held"   # median L(MENTION) = 0.80, lo >= 0.50
    assert p4["verdict"] == "failed"


# ---------------------------------------------------------------------------------------------------------------------
# 4. Delta_lp_spec length matching
# ---------------------------------------------------------------------------------------------------------------------

def test_delta_lp_spec_length_matching():
    b = Builder()
    # a generic per-token increment (+0.10 per content token) on every NP: longer off-diagonal NPs must not read negative
    rec = empty_records()
    rec["stage1"]["instruct"] = b.full_np_rows(s.OFFDIAG_PROMPTS, own_delta=0.10, peer_delta=0.10)
    s1 = s.Stage1(rec, TARGETS)
    vals = [s1.delta_lp_spec("instruct", u)["delta_lp_spec"] for u in SLOT]
    assert all(v is not None and abs(v) < 1e-6 for v in vals), vals
    # ... while v4's sum form reads negative for the units whose peers' NPs are longer (the bias v5 removes)
    v4 = [s1.delta_spec_v4("instruct", u) for u in SLOT]
    assert any(v is not None and v < -1e-6 for v in v4)
    # when lengths match, the per-token form equals v4's sum form: every NP exactly 3 content tokens long, own +0.10
    # per token and every peer +0.10, so the per-token statistic is 3 x 0.10 and so is v4's sum-differenced one
    rec2 = empty_records()
    rec2["stage1"]["instruct"] = b.full_np_rows(s.OFFDIAG_PROMPTS, own_delta=0.10, peer_delta=0.10, peer_lengths=3)
    s2 = s.Stage1(rec2, TARGETS)
    for uid in SLOT:
        per_token = s2.delta_lp_spec("instruct", uid)["delta_lp_spec"]
        v4_matched = s2.delta_spec_v4("instruct", uid)
        # with every NP 3 tokens long, v5's per-token form and v4's sum form agree; both are 0 for a generic push
        # (own +0.10, peers +0.10) and 3 x 0.10 when the peers do not move
        assert abs(per_token - v4_matched) < 1e-6, (uid, per_token, v4_matched)
        assert abs(per_token) < 1e-6, (uid, per_token)
    rec3 = empty_records()
    rec3["stage1"]["instruct"] = b.full_np_rows(s.OFFDIAG_PROMPTS, own_delta=0.10, peer_delta=0.0, peer_lengths=3)
    s3 = s.Stage1(rec3, TARGETS)
    for uid in SLOT:
        per_token = s3.delta_lp_spec("instruct", uid)["delta_lp_spec"]
        n_u = int(UNIT_BY_ID[uid]["n_content_tokens"])
        # own +0.10 x n_u tokens, peers unmoved; v4's sum form over 3-token NPs is the same number
        assert abs(per_token - 0.10 * n_u) < 1e-6, (uid, per_token)
        assert abs(s3.delta_spec_v4("instruct", uid) - 0.30) < 1e-6, uid


# ---------------------------------------------------------------------------------------------------------------------
# 5. slot-family writer fails 213.11's slot-peer clause
# ---------------------------------------------------------------------------------------------------------------------

def test_slot_family_writer_fails_slot_peer_clause():
    b = Builder()
    rec = empty_records()
    # own and slot peers up by D = 1.0 per content token, concept peers 0
    rows = []
    for pid in s.OFFDIAG_PROMPTS:
        for unit in UNITS:
            length = int(unit["n_content_tokens"])
            row = b._row(unit, pid, "none", "0.0", 0.0, b.logits([0.0] * b.n), 0.0)
            row["lp_sum"], row["lp_mean"] = 10.0 * length, 10.0
            rows.append(row)
        for uid in SLOT:
            unit = UNIT_BY_ID[uid]
            row = b._row(unit, pid, "neuron", "+0.8", 0.8, b.logits([0.0] * b.n), 0.0)
            row["lp_sum"], row["lp_mean"] = 11.0 * int(unit["n_content_tokens"]), 11.0
            rows.append(row)
            for vid in unit["offdiag_8"]:
                v = UNIT_BY_ID[vid]
                delta = 1.0 if v["kind_haiku"] == "slot" else 0.0
                row = b._row(v, pid, "offdiag", "+0.8", 0.8, b.logits([0.0] * b.n), 0.0)
                row["push_unit"], row["scored_unit"] = uid, vid
                row["lp_sum"] = (10.0 + delta) * int(v["n_content_tokens"])
                row["lp_mean"] = 10.0 + delta
                rows.append(row)
    rec["stage1"]["instruct"] = rows
    result = score(rec)
    p = result["predictions"]["213.11"]
    assert p["delta_lp_spec"]["point"] > 0.75           # the 8-peer mean passes at half strength ...
    assert p["delta_lp_spec_slot"]["point"] == 0.0      # ... but the slot peers alone are not beaten (CI lo > 0 is
    # the registered bar, so a CI sitting exactly on 0 is not held)
    assert p["clauses"]["slot_peer_only_lo_gt_0"] == "failed"
    assert p["verdict"] == "failed"


# ---------------------------------------------------------------------------------------------------------------------
# 6. SpecLogit cancels a constant shift
# ---------------------------------------------------------------------------------------------------------------------

def test_spec_logit_cancels_constant_shift():
    unit = UNIT_BY_ID["p92-05"]
    width = max(u["first_content_token_id"] for u in UNITS) + 1
    logits = [0.0] * width
    for v in [unit] + [UNIT_BY_ID[x] for x in unit["offdiag_ou"]]:
        logits[v["first_content_token_id"]] = 7.0
    logits[unit["first_content_token_id"]] = 9.0
    rec = {"first_logits": logits}
    base = s.spec_logit(rec, unit, UNIT_BY_ID)
    shifted = {"first_logits": [x + 3.25 for x in logits]}
    assert abs(s.spec_logit(shifted, unit, UNIT_BY_ID) - base) < 1e-9
    assert abs(base - 2.0) < 1e-9


# ---------------------------------------------------------------------------------------------------------------------
# 7. eligibility, null exclusion, bootstrap and sign-test arithmetic
# ---------------------------------------------------------------------------------------------------------------------

def _tied_units():
    """Units whose first-content token is shared with another unit (a solo push cannot move them alone)."""
    seen = {}
    for u in UNITS:
        seen.setdefault(u["first_content_token_id"], []).append(u["id"])
    return [uid for ids in seen.values() if len(ids) > 1 for uid in ids]


def _realised(b, value):
    """N_z the builder actually produces when every slot unit is asked for `value` (the fit leaves a small residual)."""
    probe = empty_records()
    probe["stage1"]["instruct"] = b.solo_records("instruct", SELF_HELD, {u: value for u in SLOT}, {u: 0.0 for u in SLOT})
    s1 = s.Stage1(probe, TARGETS)
    return {u: s1.nz("instruct", u, SELF_HELD) for u in SLOT}


def _exact_units(b, value):
    """Slot units whose N_z comes out exactly at `value` (their O_u holds no other slot unit that moves)."""
    realised = _realised(b, value)
    return [u for u in SLOT if abs(realised[u] - value) < 1e-6]


def test_eligibility_and_null_exclusion():
    b = Builder()
    exact_one = _exact_units(b, 1.0)
    exact_low = _exact_units(b, 0.2)
    assert len(exact_low) >= 7                          # below the 0.50 denominator bar: null
    for n_low, n_high, expected in ((6, 15, "read"), (7, 14, "not read")):
        # high from units that are not tied to another unit, so each one moves alone; low from the rest
        high_pool = [u for u in exact_one if u not in _tied_units()]
        high = high_pool[:n_high]
        low = [u for u in exact_low if u not in high][:n_low]
        nz_self = {u: 0.2 for u in low}
        nz_self.update({u: 1.0 for u in high})
        # every other slot unit is left at 0.0 (null); the two tied units cannot move alone and stay null
        nz_self.update({u: 0.0 for u in SLOT if u not in low + high})
        rec = empty_records()
        rec["stage1"]["instruct"] = b.solo_records("instruct", SELF_HELD, nz_self, {u: 0.0 for u in SLOT})
        rec["stage1"]["instruct"] += b.solo_records("instruct", THIRD_SLOT, {u: 1.0 for u in SLOT},
                                                   {u: 0.0 for u in SLOT})  # numerator is not what this test reads
        s1 = s.Stage1(rec, TARGETS)
        tau = s1.ratio("instruct", "L", THIRD_SLOT, SELF_HELD)
        assert tau["n_eligible"] == n_high, tau["n_eligible"]
        # every unit asked for 0.0 (and the tied units, which a solo push cannot move) is `null`
        asked_zero = [u for u in SLOT if u not in low + high]
        for u in asked_zero:
            assert abs(s1.lz("instruct", u, SELF_HELD)) < 1e-9 or u in _tied_units()
        assert tau["null_units"] == sorted(set(tau["null_units"]))
        assert set(tau["null_units"]) <= set(asked_zero) | set(low) | set(_tied_units())
        assert set(low) <= set(tau["null_units"])
        assert tau["read"] == expected
        if expected == "not read":
            assert s.ci_clause(tau["ci"], "ge", 0.50, tau["n_eligible"]) == "not read"


def test_bootstrap_resamples_units_seed_213_10000_draws():
    values = [float((i * 13) % 29) + 0.37 * i for i in range(21)]
    a = s.bootstrap_ci(values)
    b = s.bootstrap_ci(values)
    assert a == b                                       # seed 213 is fixed: identical CIs on re-run
    assert a["n"] == 21 and a["lo"] <= a["point"] <= a["hi"]
    # the seed is what it is: the same seed reproduces the CI exactly, whatever the draw order
    assert s.bootstrap_ci(values, seed=213) == a
    assert s.bootstrap_ci.__defaults__[1] == 213        # statistic, seed, draws defaults: seed 213, 10,000 draws
    # one extreme unit cannot move the median CI of 21 units much: units, not values, are the resampling cluster
    assert a["hi"] - a["lo"] < 20.0
    assert s.DRAWS == 10000 and s.SEED == 213


def test_exact_one_sided_paired_sign_test():
    assert abs(s.sign_test_p(7, 8) - 0.0352) < 1e-4
    assert abs(s.sign_test_p(6, 8) - 0.1445) < 1e-4
    assert s.sign_test_p(0, 0) == 1.0


# ---------------------------------------------------------------------------------------------------------------------
# 8. 213.6 reads the blind audit; the reader seam is never called by the scorer
# ---------------------------------------------------------------------------------------------------------------------

def _stage2_rows():
    rows = []
    for pid in ("free_self_0", "free_third_0", "free_third_1", "free_cold_0"):
        rows.append({"unit": None, "prompt_id": pid, "context": pid, "arm": "none", "dose": "0.0", "alpha": 0.0,
                     "status": "ok", "answer": "I am not sure."})
    for uid in SLOT:
        for pid in ("free_third_0", "free_third_1"):
            for arm, dose in (("neuron", "+0.8"), ("neuron", "-0.8"), ("random_0", "+0.8")):
                rows.append({"unit": uid, "kind": "slot", "prompt_id": pid, "context": pid, "arm": arm, "dose": dose,
                             "alpha": 0.8, "status": "ok", "answer": "A fox."})
    return rows


def test_213_6_requires_the_blind_audit_and_never_calls_the_reader():
    rec = empty_records()
    rec["stage2"] = _stage2_rows()
    result = score(rec)
    assert result["gates"]["gate3"]["baseline_broken"] == []      # the unpushed baselines are coherent
    assert result["predictions"]["213.6"]["verdict"] == "not read" # no blind labels yet: nothing to decide on
    assert len(result["stage2"]["blind_export"]) == 21 * 2 * 3
    with pytest.raises(NotImplementedError):
        s.deepseek_reader([{"blind_id": 0}])


# ---------------------------------------------------------------------------------------------------------------------
# 9. gates: a failed Gate 2 makes Stage 1 predictions untestable
# ---------------------------------------------------------------------------------------------------------------------

def _instruct_run(gate2_passed=True):
    return {"dtype": "float32", "attn_implementation": "eager", "snapshot_commit": "48d788eca847",
            "resolved": {"commit_prefix": "48d788eca847"},
            "tokenizer_sha256": {c: {"tokenizer.json": "x", "tokenizer_config.json": "y"}
                                 for c in ("sft", "dpo", "instruct")},
            "gate2_anchors_detail": [{"id": "p92-02", "passed": gate2_passed}]}


def test_failed_gate_marks_predictions_untestable():
    rec = empty_records(gate2_passed=False)
    result = score(rec)
    assert result["gates"]["gate2"]["passed"] is False
    for name in ("213.1", "213.2", "213.3", "213.4", "213.5", "213.7", "213.8", "213.9", "213.10", "213.11"):
        assert result["predictions"][name]["verdict"] == "untestable (gate failed)"


# ---------------------------------------------------------------------------------------------------------------------
# sabotage: each required test must be RED against a deliberately broken COPY of score213.py
# ---------------------------------------------------------------------------------------------------------------------

SABOTAGES = {
    "disposition_rule": ("test_disposition_rule_straddling_failed_and_low_n",
                         [('return "held" if ci["lo"] >= bar else ("failed" if ci["hi"] < bar else "inconclusive")',
                           'return "held" if ci["lo"] >= bar else "failed"'),
                          ('if any(s == "failed" for s in statuses):', 'if all(s == "failed" for s in statuses):')]),
    "sign_guard": ("test_sign_guard_suppression_both_cells_is_inconclusive",
                   [('out["sign_guard"] = bool(numer) and out["numerator_ci"]["lo"] > 0',
                     'out["sign_guard"] = True')]),
    "scene_denominator": ("test_scene_framed_denominators_decide_and_pooled_does_not",
                          [('tau = s1.ratio("instruct", "L", MENTION, THIRD_SCENE)',
                            'tau = s1.ratio("instruct", "L", MENTION, THIRD_SLOT)')]),
    "length_matched": ("test_delta_lp_spec_length_matching",
                       [('full.append(n_u * (own - sum(off) / len(off)))', 'full.append(own - sum(off) / len(off))')]),
    "slot_peer_clause": ("test_slot_family_writer_fails_slot_peer_clause",
                         [('"slot_peer_only_lo_gt_0": ci_clause(ci_slot, "ge_strict", 0.0),',
                           '"slot_peer_only_lo_gt_0": "held",')]),
    "spec_logit_shift": ("test_spec_logit_cancels_constant_shift",
                         [('return float(own - median(others))', 'return float(own)')]),
    "sign_test": ("test_exact_one_sided_paired_sign_test",
                  [('tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)',
                    'tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** (n + 1))')]),
    "eligibility": ("test_eligibility_and_null_exclusion",
                    [('if b > ELIGIBLE_BAR:', 'if b >= 0.0:')]),
}


def _load_copy(tmp_path, name):
    source = (HERE / "score213.py").read_text(encoding="utf-8")
    target = tmp_path / name / "score213.py"
    target.parent.mkdir()
    (target.parent / "targets213.json").write_text(json.dumps(TARGETS), encoding="utf-8")
    target.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(f"score213_{name}", target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", sorted(SABOTAGES))
def test_each_required_test_is_red_under_a_deliberate_break(tmp_path, name):
    test_name, edits = SABOTAGES[name]
    source = (HERE / "score213.py").read_text(encoding="utf-8")
    broken = source
    for old, new in edits:
        assert old in broken, f"sabotage anchor drifted from score213.py: {old!r}"
        broken = broken.replace(old, new)
    assert broken != source
    copy_dir = tmp_path / name
    copy_dir.mkdir()
    (copy_dir / "score213.py").write_text(broken, encoding="utf-8")
    (copy_dir / "targets213.json").write_text(json.dumps(TARGETS), encoding="utf-8")
    spec = importlib.util.spec_from_file_location(f"score213_broken_{name}", copy_dir / "score213.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    test_module = sys.modules[__name__]
    saved = test_module.s
    test_module.s = module
    try:
        with pytest.raises(Exception):
            getattr(test_module, test_name)()
    finally:
        test_module.s = saved
    # the tracked file is untouched
    assert (HERE / "score213.py").read_text(encoding="utf-8") == source


def test_gate2_renders_text_not_ids():
    """The pilot's first real run died here: Gate 2 passed the id list to the tokenizer, which only takes text."""
    from part213 import run213

    class Tok:
        def apply_chat_template(self, msgs, add_generation_prompt, tokenize, enable_thinking):
            assert tokenize is False
            return "<|user|>\n" + msgs[0]["content"] + "\n<|assistant|>\n"

        def __call__(self, text, add_special_tokens=False):
            if not isinstance(text, str):
                raise ValueError("text input must be of type `str`")
            return {"input_ids": [ord(c) for c in text]}

    ids = run213.gate2_rendered_ids(Tok(), "Who are you?", "I am a fox.")
    assert ids == [ord(c) for c in "<|user|>\nWho are you?\n<|assistant|>\nI am a fox."]


def test_layer_chunks_never_mix_layers_and_keep_order():
    """The pilot's third run died chunking across layers; Stage 1 had the same loop."""
    from part213 import run213

    work = [{"id": f"u{i}", "layer": layer} for i, layer in enumerate([5, 3, 5, 3, 5, 9, 3])]
    chunks = list(run213.layer_chunks(work, 2))
    assert all(len({c["layer"] for c in chunk}) == 1 for chunk in chunks)
    assert [[c["id"] for c in chunk] for chunk in chunks] == [["u1", "u3"], ["u6"], ["u0", "u2"], ["u4"], ["u5"]]
    assert sorted(c["id"] for chunk in chunks for c in chunk) == sorted(w["id"] for w in work)


def test_step_size_summary_pairs_units_and_signs():
    """The pilot's fourth run died keying j_eps by the signed dose; the pairing also tested a unit id against
    (uid, prompt) keys, which would have found no pairs."""
    from part213 import run213

    units = {"a": {"first_content_token_id": 0}, "b": {"first_content_token_id": 1}, "c": {"first_content_token_id": 0}}
    j = {run213.FD_EPS: {}, run213.PILOT_EPS: {}}
    for k, (uid, slope) in enumerate([("a", 1.0), ("b", 2.0), ("c", 3.0)]):
        for eps in (run213.FD_EPS, run213.PILOT_EPS):
            # F8: first_logits is the sparse {str(token_id): logit} map
            row = lambda e: {"0": slope * e, "1": slope * e + 1.0}
            j[eps][(uid, f"p{k}")] = {eps: row(eps), -eps: row(-eps)}
    out = run213.step_size_summary(j, units)
    assert out["n_pairs"] == 3 and out["passed"] is True
    assert out["spearman_r_J_N_0.05_vs_0.025"] == 1.0
    assert run213.step_size_summary({run213.FD_EPS: {}, run213.PILOT_EPS: {}}, units) == {"n_pairs": 0, "passed": False}


def test_gate2_reference_is_the_operating_dose(tmp_path):
    """Part 106 stored alpha 0.4, 0.8 and 1.6; the first complete pilot compared Gate 2 with the 1.6 rows."""
    from part213 import run213

    rows = [{"id": "p92-02", "unit": "none", "alpha": 0.0, "lp_sum": -31.6},
            {"id": "p92-02", "unit": "neuron", "alpha": 0.4, "lp_sum": -27.3},
            {"id": "p92-02", "unit": "neuron", "alpha": 0.8, "lp_sum": -5.1},
            {"id": "p92-02", "unit": "neuron", "alpha": 1.6, "lp_sum": -25.5},
            {"id": "p92-02", "unit": "random", "alpha": 0.8, "lp_sum": -25.4},
            {"id": "p92-02", "unit": "random", "alpha": 1.6, "lp_sum": -29.1}]
    f = tmp_path / "lp.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    stored = run213.part106_rows(f)
    assert run213.OPEN_ALPHA == 0.8
    assert run213.part106_reference(stored, "p92-02", "none") == -31.6
    assert run213.part106_reference(stored, "p92-02", "neuron") == -5.1
    assert run213.part106_reference(stored, "p92-02", "random") == -25.4


# ---------------------------------------------------------------------------------------------------------------------
# fixes from the GLM review (F1-F8, L1, L4, L7): one test per finding, each red on the old code
# ---------------------------------------------------------------------------------------------------------------------

class _FakeTokenizer:
    pad_token_id = 0


class _FakeModel:
    """A tiny deterministic stand-in for the decoder: logits[j, t, v] = (t + 1) * (v + 1) / 1000."""
    class config:
        hidden_size = 4

    def __call__(self, input_ids, attention_mask, use_cache):
        import torch

        batch, width = input_ids.shape
        vocab = 8
        steps = torch.arange(1, width + 1, dtype=torch.float32).view(1, width, 1)
        values = torch.arange(1, vocab + 1, dtype=torch.float32).view(1, 1, vocab)
        logits = (steps * values / 1000.0).expand(batch, width, vocab).clone()

        class Out:
            pass

        out = Out()
        out.logits = logits
        return out


def test_f1_make_row_reads_after_the_opening_and_lead():
    """F1: first_pos/content_pos include the opening tokens (old code read one opening token early)."""
    from part213 import run213

    class Tok:
        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return [ord(c) for c in text]

    original = run213.prompt_ids
    run213.prompt_ids = lambda tokenizer, question: [1000, 1001]
    try:
        unit = {"id": "u", "target_ids": [50, 51, 52, 53], "lead_ids": [50], "content_start": 1,
                "first_content_token_id": 51, "n_content_tokens": 3, "n_target_tokens": 4}
        row = run213.make_row(Tok(), "q", "AB", unit)
    finally:
        run213.prompt_ids = original
    assert row["ids"] == [1000, 1001, ord("A"), ord("B"), 50, 51, 52, 53]
    # logits at position p predict the token at p + 1; the last lead token is index 4 and predicts 51.
    assert row["first_pos"] == 4 and row["ids"][row["first_pos"]] == 50
    assert row["content_pos"] == [4, 5, 6]
    assert [row["ids"][p + 1] for p in row["content_pos"]] == [51, 52, 53]
    assert row["prefix_len"] == 4                                           # chat prefix + opening


def test_f2_and_f3_offdiag_passes_are_on_the_peer_row_and_never_pad():
    """F2/F3: Stage 1 must not index past a row (old np_specs crashed) and 213.11 must read the offdiag passes."""
    from part213 import run213

    # F2: mixed-length rows with per-row keep ids return without an IndexError and only own-row positions are read.
    rows = [{"ids": [1, 2, 3], "first_pos": 2, "content_pos": [2], "target_ids": [3]},
            {"ids": [1, 2, 3, 4, 5, 6, 7], "first_pos": 6, "content_pos": [4, 5, 6], "target_ids": [5, 6, 7]}]
    out = run213.forward_batch(_FakeModel(), _FakeTokenizer(), rows, keep_ids=[[1, 2], [3]])
    assert list(out[0]["first_logits"]) == ["1", "2"]
    assert list(out[1]["first_logits"]) == ["3"]
    assert all(len(entry["first_logits"]) <= 2 for entry in out)

    # F3: full_np_increment reads the (push_unit, scored_unit, prompt) offdiag record on v's own row.
    b = Builder()
    rec = empty_records()
    rec["stage1"]["instruct"] = b.full_np_rows(s.OFFDIAG_PROMPTS, own_delta=0.20, peer_delta=-0.10)
    assert all("lp_by_unit" not in row for row in rec["stage1"]["instruct"])
    s1 = s.Stage1(rec, TARGETS)
    for uid in SLOT[:5]:
        own = s1.full_np_increment("instruct", uid, uid, s.OFFDIAG_PROMPTS[0])
        assert abs(own - 0.20) < 1e-9, (uid, own)
        peer = s1.full_np_increment("instruct", uid, UNIT_BY_ID[uid]["offdiag_8"][0], s.OFFDIAG_PROMPTS[0])
        assert abs(peer - (-0.10)) < 1e-9, (uid, peer)


def test_f3_instruct_executes_6366_passes():
    """Amendment 1: 5,430 core + 936 offdiag = 6,366 instruct Stage 1 passes."""
    from part213 import run213

    counts = run213.stage1_pass_count(TARGETS, "instruct")
    assert counts["stage1_core_passes"] == 5430
    assert counts["offdiag_passes"] == 936
    assert counts["stage1_passes_executed"] == 6366
    assert run213.stage1_pass_count(TARGETS, "sft")["stage1_passes_executed"] == 936


def test_f4_per_checkpoint_manifests_and_pilot_evidence(tmp_path, monkeypatch):
    """F4: the runner writes run-<checkpoint>.json and load_records finds it; pilot evidence is auto-merged."""
    from part213 import run213

    snap = tmp_path / "snap"
    snap.mkdir()
    out = tmp_path / "records"
    monkeypatch.setattr(run213, "tokenizer_digests", lambda: {"instruct": {"tokenizer.json": "x"}})
    monkeypatch.setattr(run213, "residual_norm_ratio", lambda model, tokenizer: 1.0)
    monkeypatch.setattr(run213, "provenance", lambda ckpt, s: {"key": ckpt, "commit": "a" * 40,
                                                               "commit_prefix": "a" * 12})
    for ckpt in ("sft", "dpo", "instruct"):
        run213.write_run_json(out, ckpt, snap, None, None, {"dtype": "float32", "attn_implementation": "eager"})
    assert not (out / "run.json").exists()
    for ckpt in ("sft", "dpo", "instruct"):
        assert (out / f"run-{ckpt}.json").is_file()

    # stage1's automatic path: pilot_summary.json beside the records is picked up by the scorer
    (out / "pilot_summary.json").write_text(json.dumps({
        "gate1": {"passed": True, "zero_hook_max_abs_diff": 0.0},
        "gate2": {"passed": True, "anchors": [{"id": "p92-02", "passed": True}]},
        "stage2_baseline_answers": [{"id": "free_self_0", "answer": "I am a fox."}]}), encoding="utf-8")
    records = s.load_records(out)
    assert set(records["runs"]) >= {"sft", "dpo", "instruct"}
    assert records["runs"]["instruct"]["pilot_gate1"]["passed"] is True
    assert records["runs"]["instruct"]["gate2_anchors_detail"][0]["passed"] is True
    assert records["runs"]["instruct"]["stage2_baseline_answers"][0]["id"] == "free_self_0"


def test_f5_headline3_needs_a_complete_stage2_ci_and_gates_need_labels():
    """F5: no first-person headline without the Stage 2 CI; gate 3/4 fail without the evidence they guard."""
    gates = {name: {"passed": True} for name in ("gate1", "gate2", "gate3", "gate4")}
    preds = {"213.1": {"verdict": "held"},
             "213.3": {"verdict": "held", "tau_3/1_L": {"ci": {"hi": 0.20}}, "tau_3/1_N": {"ci": {"hi": 0.20}}}}
    for name in ("213.2", "213.4", "213.6", "213.11", "213.10"):
        preds[name] = {"verdict": "inconclusive", "clauses": {}}
    audit = {"slot_minus_genuine": 0}
    no_stage2 = s.headline_reading(preds, gates, audit, None)
    assert no_stage2["reading"] != "First-person self-slots supported"
    with_stage2 = s.headline_reading(preds, gates, audit, {"read": "read", "hi": 0.10})
    assert with_stage2["reading"] == "First-person self-slots supported"
    # labels present but kappa missing: the audited endpoint is unread, so Gate 3 fails
    g3 = s.gate3({"stage2": [], "runs": {}, "labels": {"0": "genuine_third_attr"}}, TARGETS)
    assert g3["passed"] is False
    # Gate 4 cannot count refusal labels it does not have
    g4 = s.gate4({"stage2": _stage2_rows(), "runs": {}, "labels": {}}, TARGETS)
    assert g4["passed"] is False and g4["labelled"] == 0


def test_f6_partial_output_is_not_a_finished_run(tmp_path):
    """F6: an existing JSONL without a matching completion marker is a crashed run, not a finished one."""
    from part213 import run213

    path = tmp_path / "stage1-sft.jsonl"
    path.write_text("", encoding="utf-8")
    assert not run213.is_complete(path)                       # old resume rule skipped this
    run213.write_jsonl(path, [{"a": 1}])
    assert not run213.is_complete(path)
    run213.write_completion(path, [{"a": 1}], "sft")
    assert run213.is_complete(path)
    path.write_text(path.read_text(encoding="utf-8") + json.dumps({"a": 2}) + "\n", encoding="utf-8")
    assert not run213.is_complete(path)                       # row count/digest no longer match


def _blind_labels(rec, label_fn):
    blind = s.export_blind(rec["stage2"], TARGETS)
    order = s._blind_order(rec["stage2"], TARGETS)
    return {item["blind_id"]: label_fn(row) for item, row in zip(blind, order)}


def test_f7_partial_blind_audit_is_not_read():
    """F7: labelling only the rows that favour the neuron arm must not hold 213.6."""
    rec = empty_records()
    rec["stage2"] = _stage2_rows()
    labels = _blind_labels(rec, lambda row: "genuine_third_attr"
                           if row["arm"] == "neuron" and row["dose"] == "+0.8" else None)
    labels = {k: v for k, v in labels.items() if v is not None}       # 42 of 84 rows only
    result = s.score(rec, TARGETS, {"labels": labels})
    p6 = result["predictions"]["213.6"]
    assert p6["audit_complete"] is False and p6["labelled_rows"] == 42 and p6["audited_rows"] == 84
    assert p6["verdict"] == "not read"


def test_l7_difference_ci_is_the_total_and_strictly_above_zero():
    """L7: bootstrap the total (neuron - random) and require lo > 0; a 2.5th percentile exactly on 0 is not above."""
    rec = empty_records()
    rec["stage2"] = _stage2_rows()
    s.assign_blind_ids(rec, TARGETS)
    hit_units = set(SLOT[:3])                               # 3 units x 2 prompts = 6 hits, 0 random hits
    labels = _blind_labels(rec, lambda row: "genuine_third_attr"
                           if row["arm"] == "neuron" and row["dose"] == "+0.8" and row["unit"] in hit_units
                           else "other_coherent")
    p6 = s.predict_213_6(rec, TARGETS, {"labels": labels})
    assert p6["difference_ci"]["point"] == 6.0
    assert p6["difference_ci"]["lo"] == 0.0                  # P(all-zero resample) = (18/21)^21 > 2.5%
    assert p6["clauses"]["difference_ci_above_0"] == "inconclusive"   # old `ge 0` read `held`


def test_l1_low_n_wins_over_a_failed_sign_guard():
    """L1: n_eligible < 15 is `not read` even when the ratio's sign guard also failed."""
    assert s.ratio_clause({"n_eligible": 10, "sign_guard": False, "ci": {"lo": 0.90, "hi": 1.0}},
                          "ge", 0.50) == "not read"
    assert s.ratio_clause({"n_eligible": 21, "sign_guard": False, "ci": {"lo": 0.90, "hi": 1.0}},
                          "ge", 0.50) == "inconclusive"
    assert s.ratio_clause({"n_eligible": 21, "sign_guard": True, "ci": {"lo": 0.60, "hi": 0.90}},
                          "ge", 0.50) == "held"


def test_l4_non_finite_pass_stops_the_run():
    """L4: a NaN pass stops Stage 1 instead of burning the remaining passes and writing bare NaN."""
    from part213 import run213

    with pytest.raises(SystemExit):
        run213.check_finite({"lp_sum": float("nan")}, "p92-00|third_0|neuron", "instruct")
    with pytest.raises(SystemExit):
        run213.check_finite({"lp_sum": 0.0, "first_logits": {"293": float("inf")}}, "x", "instruct")
    run213.check_finite({"lp_sum": 0.0, "first_logits": {"293": -1.5}}, "x", "instruct")  # finite is fine


def test_f8_rows_keep_only_the_logits_the_scorer_reads():
    """F8: a record's first_logits holds only own + offdiag_ou ids, never the full vocabulary."""
    from part213 import run213

    units_by_id = UNIT_BY_ID
    unit = UNIT_BY_ID["p92-05"]
    keep = run213.first_logit_ids(unit, units_by_id)
    assert int(unit["first_content_token_id"]) in keep
    assert keep == sorted(set(keep))
    assert len(keep) <= 1 + len(unit["offdiag_ou"]) < 40
    row = {"ids": list(range(10)), "first_pos": 2, "content_pos": [3], "target_ids": [3]}
    entry = run213.forward_batch(_FakeModel(), _FakeTokenizer(), [row], keep_ids=[[0, 1, 2]])[0]
    assert set(entry["first_logits"]) == {"0", "1", "2"}


def test_stage2_plan_is_16_cells_and_364_executed_rows():
    """Stage 2 expands units once: the old plan looped units inside the plan and again in cmd_stage2 (39x too many)."""
    from part213 import run213

    plan = run213.stage2_plan(TARGETS)
    assert len(plan) == 16
    total = 0
    for pid, arm, _dose, _alpha in plan:
        total += 1 if arm == "none" else sum(1 for u in UNITS if not run213.is_sealed(u, pid))
    assert total == 364          # 472 planned - 108 sealed concept rows
