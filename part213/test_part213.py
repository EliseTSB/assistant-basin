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
        """none rows for every unit, and a neuron +0.8 row per (unit, prompt) scoring every unit v's NP.

        The neuron row for unit u carries unit v's lp_mean/lp_sum with `own_delta` when v == u and `peer_delta`
        otherwise (per content token). `peer_lengths` overrides every NP's token count (the "lengths match" case).
        """
        rows = []
        for pid in prompt_ids:
            for unit in UNITS:
                row = self._row(unit, pid, "none", "0.0", 0.0, self.logits([0.0] * self.n), 0.0)
                length = peer_lengths if peer_lengths is not None else int(unit["n_content_tokens"])
                row["lp_sum"], row["lp_mean"] = 10.0 * length, 10.0
                rows.append(row)
            for uid in SLOT:
                # one record per (unit, prompt): the pushed pass carries all 39 units' NP scores in lp_by_unit
                row = self._row(UNIT_BY_ID[uid], pid, "neuron", "+0.8", 0.8, self.logits([0.0] * self.n), 0.0)
                by_unit = {}
                for v in UNITS:
                    delta = own_delta if v["id"] == uid else peer_delta
                    length = peer_lengths if peer_lengths is not None else int(v["n_content_tokens"])
                    mean = 10.0 + delta
                    by_unit[v["id"]] = {"lp_mean": mean, "lp_sum": mean * length}
                row["lp_by_unit"] = by_unit
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
            row = b._row(UNIT_BY_ID[uid], pid, "neuron", "+0.8", 0.8, b.logits([0.0] * b.n), 0.0)
            row["lp_by_unit"] = {
                v["id"]: {"lp_mean": 10.0 + (1.0 if v["kind_haiku"] == "slot" else 0.0),
                          "lp_sum": (10.0 + (1.0 if v["kind_haiku"] == "slot" else 0.0)) * v["n_content_tokens"]}
                for v in UNITS}
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
            row = lambda e: [slope * e, slope * e + 1.0]  # logit 0 and logit 1 both rise with the step
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
