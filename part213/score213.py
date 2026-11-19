#!/usr/bin/env python3
"""Part 213 scorer: every gate, every Stage 1 metric, the Stage 2 audit tables, and the verdict of predictions
213.1-213.11 under the v5 disposition rule -- read from the runner's saved records only.

Built from PART213-DESIGN.md v5 (commit 7de53067). The v5 disposition rule (design l.427-431) governs every verdict:
a clause is `not read (low n)` when n_eligible < 15, `inconclusive` when its CI straddles its bar (or a ratio's
numerator sign guard fails), `held` when its CI lies wholly on the predicted side, and `failed` only when its CI lies
wholly on the other side or a point clause (a count or a point median) misses. A prediction holds when every clause
holds, fails when any clause fails, and is otherwise `inconclusive` or `not read`.

Usage:
  python -I -B part213/score213.py --records results/part213/full [--audit audit.json] [--out score.json]
Writes score.json and prints a table. No model, no network: the Stage 2 blind reader is `deepseek_reader` below, a
function the tests replace with a fake; it is never called here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ROOT / "part213" / "targets213.json"
TARGETS_SHA256 = "7edb3c6253470d157896931083c239253930be08f16a5b9348d6c67d3b431892"

SEED = 213
DRAWS = 10000
MIN_ELIGIBLE = 15          # design l.191-193, l.424-425
ELIGIBLE_BAR = 0.50        # design l.190-191: X_z(u, B) > 0.50 logits
OPEN_ALPHA = 0.8
CHECKPOINTS = ("sft", "dpo", "instruct", "fr")

SELF_HELD = ("self_h0", "self_h1", "self_h2", "self_h3")
THIRD_SLOT = ("third_0", "third_1", "third_2", "third_3")
THIRD_SCENE = ("third_2", "third_3")       # v5 style matching (design l.196-199)
DIRECT_PAIRS = (("third_0", "self_h0"), ("third_1", "self_h1"))
SCENE_PAIRS = (("third_2", "self_h2"), ("third_3", "self_h3"))
MENTION = ("mention_0", "mention_1", "mention_2", "mention_3")
COPULA_NONID = ("copula_nonid_0", "copula_nonid_1")
SELF_NONID = ("self_nonid_0", "self_nonid_1")
OFFDIAG_PROMPTS = ("third_0", "third_2")   # 213.11 (design l.445)
THIRD_FREE = ("free_third_0", "free_third_1")
STAGE2_CONTEXTS = ("free_self_0", "free_third_0", "free_third_1", "free_cold_0")

AUDIT_CATEGORIES = ("genuine_third_attr", "first_person_reversion", "incidental_or_negated",
                    "valid_refusal_or_unknown", "other_coherent", "broken")


# ---------------------------------------------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------------------------------------------

def load_targets(path: Path = TARGETS) -> dict:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if path == TARGETS and digest != TARGETS_SHA256:
        raise SystemExit(f"targets213.json is {digest}, not the registered {TARGETS_SHA256}")
    return json.loads(raw.decode("utf-8"))


def pilot_evidence(directory: Path | None) -> dict:
    """Gate 1 / Gate 2 / baseline evidence from a pilot directory (F4's documented path).

    The pilot writes `pilot_summary.json` and `run-pilot.json`; the real pilot output at `results/part213/pilot/`
    also carries the legacy `run.json`. All three shapes are accepted.
    """
    if directory is None:
        return {}
    directory = Path(directory)
    for name in ("pilot_summary.json", "run-pilot.json", "run.json"):
        path = directory / name
        if not path.is_file():
            continue
        obj = json.loads(path.read_text(encoding="utf-8"))
        evidence: dict = {"pilot_evidence_file": str(path)}
        gate1 = obj.get("pilot_gate1") or obj.get("gate1")
        if gate1:
            evidence["pilot_gate1"] = gate1
        gate2 = obj.get("gate2_anchors_detail")
        if gate2 is None and isinstance(obj.get("gate2"), dict):
            gate2 = obj["gate2"].get("anchors")
        if gate2:
            evidence["gate2_anchors_detail"] = gate2
        if obj.get("stage2_baseline_answers"):
            evidence["stage2_baseline_answers"] = obj["stage2_baseline_answers"]
        return evidence
    return {}


def apply_pilot_evidence(runs: dict, evidence: dict) -> None:
    """Merge pilot evidence into the instruct manifest when the runner did not already copy it in."""
    if not evidence:
        return
    run = runs.get("instruct")
    if run is None:
        return
    for key in ("pilot_gate1", "gate2_anchors_detail", "stage2_baseline_answers"):
        if key in evidence and key not in run:
            run[key] = evidence[key]
    run.setdefault("pilot_evidence_file", evidence.get("pilot_evidence_file"))


def load_records(records_dir: Path, pilot_dir: Path | None = None) -> dict:
    """Every record the runner wrote: stage1-<ckpt>.jsonl, stage2-instruct.jsonl, run-<ckpt>.json.

    F4: per-checkpoint manifests only. A legacy single `run.json` is still accepted and keyed by its `checkpoint`
    field, and pilot evidence is merged from `pilot_dir` (or `<records_dir>/pilot_summary.json`, which stage1 copies).
    """
    records_dir = Path(records_dir)
    stage1, stage2, runs = {}, {}, {}
    for ckpt in CHECKPOINTS:
        path = records_dir / f"stage1-{ckpt}.jsonl"
        rows = []
        if path.exists():
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        stage1[ckpt] = rows
        run = records_dir / f"run-{ckpt}.json"
        if run.exists():
            runs[ckpt] = json.loads(run.read_text(encoding="utf-8"))
    legacy = records_dir / "run.json"
    if legacy.exists():
        obj = json.loads(legacy.read_text(encoding="utf-8"))
        key = obj.get("checkpoint")
        if key in CHECKPOINTS and key not in runs:
            runs[key] = obj
    path = records_dir / "stage2-instruct.jsonl"
    if path.exists():
        stage2 = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    evidence = pilot_evidence(pilot_dir)
    if not evidence:
        evidence = pilot_evidence(records_dir)  # stage1 copies pilot_summary.json beside the records
    apply_pilot_evidence(runs, evidence)
    return {"stage1": stage1, "stage2": stage2, "runs": runs, "pilot_evidence": evidence}


def index_stage1(rows: list[dict]) -> dict:
    """(unit, prompt_id, arm, dose) -> record; sealed rows are kept out of the metric index."""
    out = {}
    for row in rows:
        if row.get("status", "ok") != "ok":
            continue
        out[(row["unit"], row["prompt_id"], row["arm"], row["dose"])] = row
    return out


def index_offdiag(rows: list[dict]) -> dict:
    """(push_unit, scored_unit, prompt_id) -> offdiag record (Amendment 1's 936 passes; F3).

    A record's `unit` is the scored peer v (its own row); `push_unit` is the unit u whose hook was applied.
    """
    out = {}
    for row in rows:
        if row.get("status", "ok") != "ok" or row.get("arm") != "offdiag":
            continue
        push = row.get("push_unit") or row.get("unit")
        scored = row.get("scored_unit") or row.get("unit")
        out[(push, scored, row["prompt_id"])] = row
    return out


def logit_value(record: dict, token_id: int) -> float:
    """One first-content logit from a record (F8: the runner stores a sparse {str(token_id): logit} map).

    Lists are still accepted so old records and synthetic fixtures keep reading; a missing token is an error rather
    than a silent zero.
    """
    logits = record["first_logits"]
    if isinstance(logits, dict):
        if str(int(token_id)) in logits:
            return float(logits[str(int(token_id))])
        if int(token_id) in logits:
            return float(logits[int(token_id)])
        raise KeyError(f"first_logits has no token {token_id}")
    return float(logits[int(token_id)])


# ---------------------------------------------------------------------------------------------------------------------
# statistics: bootstrap, sign test, spearman
# ---------------------------------------------------------------------------------------------------------------------

def median(values) -> float:
    values = list(values)
    if not values:
        return float("nan")
    return float(statistics.median(values))


def bootstrap_ci(values, statistic=median, seed: int = SEED, draws: int = DRAWS) -> dict:
    """Unit-clustered bootstrap: resample the units (each value is one unit), 10,000 draws, seed 213 (design l.189-190)."""
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    n = len(values)
    if n == 0:
        return {"n": 0, "point": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = random.Random(seed)
    stats = []
    for _ in range(draws):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        stats.append(statistic(sample))
    stats.sort()
    lo = stats[int(0.025 * draws)]
    hi = stats[min(draws - 1, int(0.975 * draws))]
    return {"n": n, "point": round(float(statistic(values)), 4), "lo": round(lo, 4), "hi": round(hi, 4)}


def sign_test_p(k: int, n: int) -> float:
    """Exact one-sided paired sign test: P(X >= k) for X ~ Binomial(n, 1/2)."""
    if n <= 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return tail


def spearman(xs, ys) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None
             and math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return float("nan")
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]

    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


# ---------------------------------------------------------------------------------------------------------------------
# the v5 disposition rule (design l.427-431)
# ---------------------------------------------------------------------------------------------------------------------

def ci_clause(ci: dict, side: str, bar: float, n_eligible: int | None = None, min_n: int = MIN_ELIGIBLE,
              strict: bool = False) -> str:
    """One CI clause under the v5 rule.

    side "ge": held iff lo >= bar; failed iff hi < bar; else inconclusive.
    side "ge_strict": held iff lo > bar; failed iff hi <= bar; else inconclusive (213.11's `lo > 0`, design l.445).
    side "le": held iff hi <= bar; failed iff lo > bar; else inconclusive.
    side "inside": held iff lo >= lo_bar and hi <= hi_bar; failed iff hi < lo_bar or lo > hi_bar; else inconclusive.
    A missing/NaN CI is `not read` (the statistic was not computed).
    """
    if n_eligible is not None and n_eligible < min_n:
        return "not read"
    if ci is None or any(ci.get(k) is None or not math.isfinite(ci.get(k, float("nan"))) for k in ("lo", "hi")):
        return "not read"
    if side == "ge":
        return "held" if ci["lo"] >= bar else ("failed" if ci["hi"] < bar else "inconclusive")
    if side == "ge_strict":
        return "held" if ci["lo"] > bar else ("failed" if ci["hi"] <= bar else "inconclusive")
    if side == "le":
        return "held" if ci["hi"] <= bar else ("failed" if ci["lo"] > bar else "inconclusive")
    if side == "inside":
        lo_bar, hi_bar = bar
        if ci["lo"] >= lo_bar and ci["hi"] <= hi_bar:
            return "held"
        if ci["hi"] < lo_bar or ci["lo"] > hi_bar:
            return "failed"
        return "inconclusive"
    raise ValueError(f"unknown side {side!r}")


def point_clause(ok: bool) -> str:
    """A point clause (a count or a point median) misses or it does not (design l.430)."""
    return "held" if ok else "failed"


def ratio_clause(ratio: dict, side: str, bar: float) -> str:
    """One paired-ratio CI clause under the v5 rule, with low-n checked FIRST (L1).

    The v5 rule's first-listed condition is `n_eligible < 15` -> `not read` (design l.464-465); a failed sign guard
    is `inconclusive`. The inline construction in 213.3/213.4/213.10 used to test the guard before low n, so a
    low-n ratio with a failed guard read `inconclusive` instead of `not read`.
    """
    if ratio.get("n_eligible", 0) < MIN_ELIGIBLE:
        return "not read"
    if not ratio.get("sign_guard"):
        return "inconclusive"
    return ci_clause(ratio["ci"], side, bar, ratio["n_eligible"])


def disposition(clauses: dict) -> str:
    """A prediction holds when every clause holds, fails when any clause fails, else inconclusive / not read."""
    statuses = list(clauses.values())
    if not statuses:
        return "not read"
    if any(s == "failed" for s in statuses):
        return "failed"
    if all(s == "held" for s in statuses):
        return "held"
    if any(s == "inconclusive" for s in statuses):
        return "inconclusive"
    return "not read"  # every remaining clause is `not read`


# ---------------------------------------------------------------------------------------------------------------------
# Stage 1 metrics
# ---------------------------------------------------------------------------------------------------------------------

def spec_logit(record: dict, unit: dict, units_by_id: dict) -> float:
    """z_tilde = own first-content logit - median over O_u of the other units' first-content logits (design l.178-184).

    O_u is `offdiag_ou` (the strict 35-38 set, already excluding u and any unit whose first token appears in
    u["target_ids"]). Units with equal first tokens (p92-08 / p92-09 both " dog") contribute the same logit twice,
    as the set requires; no deduplication is applied.
    """
    own = logit_value(record, int(unit["first_content_token_id"]))
    others = [logit_value(record, int(units_by_id[v]["first_content_token_id"])) for v in unit["offdiag_ou"]]
    return float(own - median(others))


class Stage1:
    def __init__(self, records: dict, targets: dict):
        self.targets = targets
        self.units = targets["units"]
        self.units_by_id = {u["id"]: u for u in self.units}
        self.slot = [u["id"] for u in self.units if u["kind_haiku"] == "slot"]
        self.concept = [u["id"] for u in self.units if u["kind_haiku"] == "concept"]
        self.agreed_concept = [u["id"] for u in self.units if u["kind_haiku"] == "concept" and u["kind_ds"] == "concept"]
        self.idx = {ckpt: index_stage1(records["stage1"][ckpt]) for ckpt in CHECKPOINTS}
        self.offdiag = {ckpt: index_offdiag(records["stage1"][ckpt]) for ckpt in CHECKPOINTS}
        self.runs = records["runs"]
        self._cache = {}

    # -- raw reads ---------------------------------------------------------------------------------------------------
    def rec(self, ckpt, uid, pid, arm, dose):
        return self.idx[ckpt].get((uid, pid, arm, dose))

    def offdiag_rec(self, ckpt, push_uid, scored_uid, pid):
        return self.offdiag[ckpt].get((push_uid, scored_uid, pid))

    def zt(self, ckpt, uid, pid, arm, dose):
        key = (ckpt, uid, pid, arm, dose)
        if key not in self._cache:
            row = self.rec(ckpt, uid, pid, arm, dose)
            self._cache[key] = None if row is None else spec_logit(row, self.units_by_id[uid], self.units_by_id)
        return self._cache[key]

    def increment(self, ckpt, uid, pid, arm, dose) -> float | None:
        pushed = self.zt(ckpt, uid, pid, arm, dose)
        base = self.zt(ckpt, uid, pid, "none", "0.0")
        if pushed is None or base is None:
            return None
        return pushed - base

    def nz(self, ckpt, uid, prompts) -> float | None:
        vals = [self.increment(ckpt, uid, pid, "neuron", "+0.8") for pid in prompts]
        return None if any(v is None for v in vals) else float(sum(vals) / len(vals))

    def rz(self, ckpt, uid, prompts) -> float | None:
        arms = ("random_0", "random_1", "random_2", "random_3") if ckpt == "instruct" else ("random_0",)
        vals = [self.increment(ckpt, uid, pid, arm, "+0.8") for pid in prompts for arm in arms]
        return None if any(v is None for v in vals) else float(sum(vals) / len(vals))

    def lz(self, ckpt, uid, prompts) -> float | None:
        n, r = self.nz(ckpt, uid, prompts), self.rz(ckpt, uid, prompts)
        return None if n is None or r is None else n - r

    def group(self, ckpt, kind, prompts, which) -> dict:
        ids = self.slot if kind == "slot" else self.concept
        fn = {"N": self.nz, "R": self.rz, "L": self.lz}[which]
        return {uid: fn(ckpt, uid, prompts) for uid in ids}

    # -- transfer ratios ---------------------------------------------------------------------------------------------
    def ratio(self, ckpt, X, num_prompts, den_prompts, kind="slot", seed=SEED) -> dict:
        """tau = median over eligible units of X_z(u, A) / X_z(u, B), with the v5 sign guard (design l.189-200, l.548-550)."""
        ids = self.slot if kind == "slot" else self.concept
        fn = self.nz if X == "N" else self.lz
        rows, null_units = [], []
        for uid in ids:
            a, b = fn(ckpt, uid, num_prompts), fn(ckpt, uid, den_prompts)
            if b is None or a is None:
                null_units.append(uid)  # missing records are reported, never counted as eligible
                continue
            if b > ELIGIBLE_BAR:
                rows.append((uid, a / b, a))
            else:
                null_units.append(uid)
        n_eligible = len(rows)
        ratios = [r for _u, r, _a in rows]
        numer = [a for _u, _r, a in rows]
        out = {
            "statistic": f"tau_{X}", "numerator_prompts": list(num_prompts), "denominator_prompts": list(den_prompts),
            "n_eligible": n_eligible, "null_units": null_units,
            "point": round(median(ratios), 4) if ratios else None,
            "ci": bootstrap_ci(ratios, seed=seed),
            "numerator_ci": bootstrap_ci(numer, seed=seed),
        }
        out["sign_guard"] = bool(numer) and out["numerator_ci"]["lo"] > 0
        if n_eligible < MIN_ELIGIBLE:
            out["read"] = "not read"
        elif not out["sign_guard"]:
            out["read"] = "inconclusive"
        else:
            out["read"] = "read"
        return out

    # -- full-NP specificity (213.11) --------------------------------------------------------------------------------
    def full_np_increment(self, ckpt, u_id, v_id, pid) -> float | None:
        """lp_content_mean(v's NP under u's `neuron +0.8` hook) - lp_content_mean(v's NP unpushed) on one prompt.

        F3 / Amendment 1: for v != u this reads the restored `(u, v, prompt)` offdiag pass -- u's hook on v's OWN
        row -- not `lp_by_unit[v]` read from u's row (a different quantity, sign-flipped on the real model). For
        v == u the own NP under u's hook is the core `neuron +0.8` record on u's own row.
        """
        base = self.rec(ckpt, v_id, pid, "none", "0.0")
        if base is None:
            return None
        if u_id == v_id:
            pushed = self.rec(ckpt, u_id, pid, "neuron", "+0.8")
        else:
            pushed = self.offdiag_rec(ckpt, u_id, v_id, pid)
        if pushed is None or pushed.get("lp_mean") is None or base.get("lp_mean") is None:
            return None
        return float(pushed["lp_mean"] - base["lp_mean"])

    def full_np_sum_increment(self, ckpt, u_id, v_id, pid) -> float | None:
        """The lp_content_sum counterpart of full_np_increment (v4's form, diagnostic only)."""
        base = self.rec(ckpt, v_id, pid, "none", "0.0")
        if base is None:
            return None
        if u_id == v_id:
            pushed = self.rec(ckpt, u_id, pid, "neuron", "+0.8")
        else:
            pushed = self.offdiag_rec(ckpt, u_id, v_id, pid)
        if pushed is None or pushed.get("lp_sum") is None or base.get("lp_sum") is None:
            return None
        return float(pushed["lp_sum"] - base["lp_sum"])

    def delta_lp_spec(self, ckpt, uid, prompts=OFFDIAG_PROMPTS) -> dict:
        """v5 length-matched Delta_lp_spec and its slot-peer-only term, per unit (design l.204-217, l.210-213)."""
        unit = self.units_by_id[uid]
        n_u = int(unit["n_content_tokens"])
        peers = list(unit["offdiag_8"])
        slot_peers = [v for v in peers if self.units_by_id[v]["kind_haiku"] == "slot"]
        full, slot = [], []
        for pid in prompts:
            own = self.full_np_increment(ckpt, uid, uid, pid)
            off = [self.full_np_increment(ckpt, uid, v, pid) for v in peers]
            off_slot = [self.full_np_increment(ckpt, uid, v, pid) for v in slot_peers]
            if own is None or any(x is None for x in off) or not off_slot or any(x is None for x in off_slot):
                return {"unit": uid, "n_u": n_u, "delta_lp_spec": None, "delta_lp_spec_slot": None}
            full.append(n_u * (own - sum(off) / len(off)))
            slot.append(n_u * (own - sum(off_slot) / len(off_slot)))
        return {"unit": uid, "n_u": n_u, "delta_lp_spec": float(sum(full) / len(full)),
                "delta_lp_spec_slot": float(sum(slot) / len(slot))}

    def delta_spec_v4(self, ckpt, uid, prompts=OFFDIAG_PROMPTS) -> float | None:
        """v4's lp_sum-differenced form, printed beside v5's as a diagnostic only (design l.209-213)."""
        unit = self.units_by_id[uid]
        peers = list(unit["offdiag_8"])
        vals = []
        for pid in prompts:
            own = self.full_np_sum_increment(ckpt, uid, uid, pid)
            off = [self.full_np_sum_increment(ckpt, uid, v, pid) for v in peers]
            if own is None or any(x is None for x in off):
                return None
            vals.append(own - sum(off) / len(off))
        return float(sum(vals) / len(vals))

    # -- finite differences (213.8) ----------------------------------------------------------------------------------
    def j(self, ckpt, uid, prompts, arm="neuron", eps=0.05) -> float | None:
        """J(u, C) = mean_q [(z_tilde(+eps) - z_tilde(-eps)) / (2 eps)] for the neuron arm minus the random_0 arm."""
        def slope(a):
            vals = []
            for pid in prompts:
                plus, minus = self.zt(ckpt, uid, pid, a, f"+{eps}"), self.zt(ckpt, uid, pid, a, f"-{eps}")
                if plus is None or minus is None:
                    return None
                vals.append((plus - minus) / (2 * eps))
            return sum(vals) / len(vals)

        if arm is not None:
            return slope(arm)
        jn, jr = slope("neuron"), slope("random_0")
        return None if jn is None or jr is None else jn - jr

    # -- negative direction (213.7) ----------------------------------------------------------------------------------
    def negative_gap(self, ckpt, uid, prompts) -> float | None:
        """z_tilde(neuron -0.8) - z_tilde(random_0 -0.8), averaged over prompts (design l.441)."""
        vals = []
        for pid in prompts:
            a = self.zt(ckpt, uid, pid, "neuron", "-0.8")
            b = self.zt(ckpt, uid, pid, "random_0", "-0.8")
            if a is None or b is None:
                return None
            vals.append(a - b)
        return sum(vals) / len(vals)


# ---------------------------------------------------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------------------------------------------------

def broken_answer(text: str) -> bool:
    """Mechanical breakage only (design l.283-284): empty, U+FFFD, non-alphanumeric-only, or a 3-word n-gram x4."""
    stripped = (text or "").strip()
    if not stripped or "�" in stripped:
        return True
    words = stripped.split()
    if not any(re.search(r"[A-Za-z0-9]", w) for w in words):
        return True
    grams = {}
    for i in range(len(words) - 2):
        gram = tuple(w.lower() for w in words[i:i + 3])
        grams[gram] = grams.get(gram, 0) + 1
    return any(c >= 4 for c in grams.values())


def keyword_hit(answer: str, keywords) -> bool:
    """Case-folded word/phrase-boundary regex (design l.286-288)."""
    text = (answer or "").casefold()
    for kw in keywords:
        pattern = r"(?<![a-z0-9])" + re.escape(kw.casefold()) + r"(?![a-z0-9])"
        if re.search(pattern, text):
            return True
    return False


def baseline_subtracted_hit(answer: str, baseline: str, keywords) -> bool:
    """A keyword counts only when it is absent from the unpushed baseline answer to the same prompt."""
    return keyword_hit(answer, keywords) and not keyword_hit(baseline, keywords)


def deepseek_reader(items: list[dict]) -> list[str]:
    """The frozen second blind reader, DeepSeek V4.1 Flash (design l.266-269). NOT called by this scorer.

    The tests replace this function with a fake. A real call needs the provider client and the 40-item balanced
    sample; it is deliberately left as a single seam so no test and no scorer run can reach the network.
    """
    raise NotImplementedError("DeepSeek V4.1 Flash reader is not wired in the scorer; supply a reader callable")


def cohens_kappa(a: list[str], b: list[str]) -> float:
    if not a or len(a) != len(b):
        return float("nan")
    n = len(a)
    cats = sorted(set(a) | set(b))
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def export_blind(rows: list[dict], targets: dict) -> list[dict]:
    """The 198 audited THIRD_FREE answers, shuffled with seed 213, arm and kind hidden (design l.258-265)."""
    units = {u["id"]: u for u in targets["units"]}
    baseline = {r["prompt_id"]: r.get("answer", "") for r in rows if r.get("arm") == "none"}
    audited = []
    for r in rows:
        if r.get("status", "ok") != "ok" or r.get("prompt_id") not in THIRD_FREE or r.get("arm") == "none":
            continue
        unit = units[r["unit"]]
        if r["arm"] == "neuron" and r["dose"] in ("+0.8", "-0.8"):
            audited.append(r)
        elif r["arm"] == "random_0" and r["dose"] == "+0.8":
            audited.append(r)
    rng = random.Random(SEED)
    order = list(range(len(audited)))
    rng.shuffle(order)
    out = []
    for blind_id, i in enumerate(order):
        r = audited[i]
        out.append({"blind_id": blind_id, "target_persona": units[r["unit"]]["persona"],
                    "question": next(p["question"] for p in targets["stage2_free_prompts"] if p["id"] == r["prompt_id"]),
                    "answer": r.get("answer", ""), "baseline_answer": baseline.get(r["prompt_id"], "")})
    return out


def _blind_order(rows: list[dict], targets: dict) -> list[dict]:
    """The same seed-213 shuffle as export_blind, returning the source rows so each gets its blind_id."""
    units = {u["id"]: u for u in targets["units"]}
    audited = []
    for r in rows:
        if r.get("status", "ok") != "ok" or r.get("prompt_id") not in THIRD_FREE or r.get("arm") == "none":
            continue
        if (r["arm"] == "neuron" and r["dose"] in ("+0.8", "-0.8")) or (r["arm"] == "random_0" and r["dose"] == "+0.8"):
            audited.append(r)
    order = list(range(len(audited)))
    random.Random(SEED).shuffle(order)
    return [audited[i] for i in order]


def audit_table(rows: list[dict], labels: dict, targets: dict) -> dict:
    """6-category counts per audited arm-group, fixed denominators 42 slot / 36 concept (design l.257-284)."""
    units = {u["id"]: u for u in targets["units"]}
    groups = {}
    for r in rows:
        if r.get("status", "ok") != "ok" or r.get("prompt_id") not in THIRD_FREE or r.get("arm") == "none":
            continue
        if not ((r["arm"] == "neuron" and r["dose"] in ("+0.8", "-0.8")) or (r["arm"] == "random_0" and r["dose"] == "+0.8")):
            continue
        kind = units[r["unit"]]["kind_haiku"]
        if r["arm"] == "neuron" and r["dose"] == "-0.8" and kind != "slot":
            continue  # concept -0.8 THIRD_FREE rows are mechanical only (design l.259-260)
        key = f"{kind}_{r['arm']}_{r['dose']}"
        groups.setdefault(key, {"rows": [], "denominator": 42 if kind == "slot" else 36})
        groups[key]["rows"].append(r)
    out = {}
    for key, group in groups.items():
        counts = {c: 0 for c in AUDIT_CATEGORIES}
        missing = 0
        for r in group["rows"]:
            label = labels.get(r.get("blind_id")) if r.get("blind_id") is not None else None
            if label is None:
                missing += 1
                if broken_answer(r.get("answer", "")):
                    counts["broken"] += 1
                continue
            counts[label] = counts.get(label, 0) + 1
        out[key] = {"denominator": group["denominator"], "labelled": len(group["rows"]) - missing,
                    "unlabelled": missing, "counts": counts}
    return out


def dedup_keyword_counts(rows: list[dict], targets: dict, prompt_ids, arm, dose) -> dict:
    """Distinct-primary-slot-keyword deduplicated counts (19 keywords -> 38 keyword-prompt pairs, design l.270-272)."""
    units = {u["id"]: u for u in targets["units"]}
    baseline = {r["prompt_id"]: r.get("answer", "") for r in rows if r.get("arm") == "none"}
    hits = {}
    for r in rows:
        if r.get("status", "ok") != "ok" or r.get("prompt_id") not in prompt_ids:
            continue
        if r.get("arm") != arm or r.get("dose") != dose:
            continue
        unit = units[r["unit"]]
        if unit["kind_haiku"] != "slot":
            continue
        for kw in unit["keywords"]:
            key = (kw["kw"].casefold(), r["prompt_id"])
            if baseline_subtracted_hit(r.get("answer", ""), baseline.get(r["prompt_id"], ""), [kw["kw"]]):
                hits.setdefault(key, []).append(r["unit"])
    return {"distinct_keyword_prompt_pairs": len(hits), "hits": {f"{k[0]}|{k[1]}": v for k, v in hits.items()}}


# ---------------------------------------------------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------------------------------------------------

def gate0(targets: dict) -> dict:
    """Gate 0 is build_targets213.py's verification, recorded here from the registered file (design l.329-333)."""
    checks = {
        "n_units": len(targets["units"]),
        "n_slot": sum(u["kind_haiku"] == "slot" for u in targets["units"]),
        "n_concept": sum(u["kind_haiku"] == "concept" for u in targets["units"]),
        "n_agreed_concept": sum(u["kind_haiku"] == "concept" and u["kind_ds"] == "concept" for u in targets["units"]),
        "np_words_in_range": all(targets["np_word_range"][0] <= u["np_words"] <= targets["np_word_range"][1]
                                 for u in targets["units"]),
        "ou_size_in_range": all(targets["ou_size_range"][0] <= len(u["offdiag_ou"]) <= targets["ou_size_range"][1]
                                for u in targets["units"]),
        "offdiag_8_size": all(len(u["offdiag_8"]) == 8 for u in targets["units"]),
        "offdiag_8_subset_of_ou": all(set(u["offdiag_8"]) <= set(u["offdiag_ou"]) for u in targets["units"]),
        "gate2_anchor_tokens": [a["expected_tokens"] for a in targets["gate2_anchors"]],
        "targets_sha256": TARGETS_SHA256,
    }
    checks["passed"] = (checks["n_units"] == 39 and checks["n_slot"] == 21 and checks["n_concept"] == 18
                        and checks["n_agreed_concept"] == 8 and checks["np_words_in_range"] and checks["ou_size_in_range"]
                        and checks["offdiag_8_size"] and checks["offdiag_8_subset_of_ou"]
                        and checks["gate2_anchor_tokens"] == [5, 5, 6, 5])
    return checks


def _tokenizer_digests(entry: dict) -> dict:
    """Normalise run.json's tokenizer digests: `{file: sha}` (current runner) or the older all-checkpoint nesting."""
    if not isinstance(entry, dict):
        return {}
    if "tokenizer.json" in entry or "tokenizer_config.json" in entry:
        return entry
    for value in entry.values():  # legacy: {"sft": {...}, "dpo": {...}, "instruct": {...}}
        if isinstance(value, dict) and ("tokenizer.json" in value or "tokenizer_config.json" in value):
            return value
    return {}


def gate1(records: dict, targets: dict) -> dict:
    """Rig, batch invariance, provenance, finite values (design l.334-338), from the runner's records and run.json."""
    runs = records["runs"]
    problems = []
    for ckpt in CHECKPOINTS:
        run = runs.get(ckpt)
        if run is None:
            if ckpt == "fr":
                continue  # FR may be absent: an FR-only NaN voids FR only (design l.336-337)
            problems.append(f"run-{ckpt}.json missing")
            continue
        if run.get("dtype") != "float32":
            problems.append(f"{ckpt}: dtype {run.get('dtype')} is not float32")
        if run.get("attn_implementation") != "eager":
            problems.append(f"{ckpt}: attention is not eager")
        if not str(run.get("snapshot_commit", "")).startswith(str(run.get("resolved", {}).get("commit_prefix", ""))):
            problems.append(f"{ckpt}: snapshot commit does not match part85/checkpoints.py")
    # Tokenizer digests are compared across the checkpoints that ran; a checkpoint absent from run.json (sft/dpo
    # before their rows exist) is reported, not silently compared away.
    missing_runs = [ckpt for ckpt in ("sft", "dpo", "instruct") if ckpt not in runs]
    if missing_runs:
        problems.append(f"run.json missing for {missing_runs}")
    tok = {ckpt: _tokenizer_digests(runs[ckpt].get("tokenizer_sha256", {}))
           for ckpt in ("sft", "dpo", "instruct") if ckpt in runs}
    if len(tok) == 3:
        for name in ("tokenizer.json", "tokenizer_config.json"):
            digests = {ckpt: tok[ckpt].get(name) for ckpt in tok}
            if len(set(digests.values())) != 1 or None in digests.values():
                problems.append(f"{name} digests differ across sft/dpo/instruct: {digests}")
    nan_ckpts = []
    for ckpt in CHECKPOINTS:
        for row in records["stage1"][ckpt]:
            if row.get("status", "ok") != "ok":
                continue
            values = row.get("first_logits") or {}
            values = values.values() if isinstance(values, dict) else values
            if any(v is None or not math.isfinite(v) for v in values):
                nan_ckpts.append(ckpt)
                break
            for key in ("lp_sum", "lp_mean", "lp_first", "entropy", "top1"):
                if row.get(key) is not None and not math.isfinite(row[key]):
                    nan_ckpts.append(ckpt)
                    break
    for ckpt in set(nan_ckpts):
        if ckpt == "fr":
            problems.append("FR: NaN/Inf in Stage 1 (voids FR only)")
        else:
            problems.append(f"{ckpt}: NaN/Inf in Stage 1 (stops Stage 1)")
    pilot = None
    for ckpt in ("instruct", "sft", "dpo", "fr"):
        run = runs.get(ckpt) or {}
        if "pilot_gate1" in run:
            pilot = run["pilot_gate1"]
    if pilot is not None:
        if pilot.get("zero_hook_max_abs_diff") != 0.0:
            problems.append("zero-vector hook changed logits")
        if not (pilot.get("batch_padded_vs_batch1_max_abs_diff", 1.0) < 1e-4):
            problems.append("batch-padded and batch-1 logits differ by >= 1e-4")
    return {"passed": not problems, "problems": problems, "pilot_gate1": pilot,
            "fr_nan_voids_fr_only": "fr" in nan_ckpts}


def gate2(records: dict) -> dict:
    """Part 106 rig reproduction on the 4 anchors, < 0.50 nats and 4/4 ordering (design l.339-348)."""
    run = records["runs"].get("instruct") or {}
    anchors = run.get("gate2_anchors_detail") or records.get("gate2") or []
    if not anchors:
        return {"passed": False, "problems": ["no Gate 2 anchor records (run the pilot, which writes them)"], "anchors": []}
    problems = []
    for a in anchors:
        if not a.get("passed"):
            problems.append(f"{a.get('id')}: {a}")
    return {"passed": not problems, "problems": problems, "anchors": anchors}


def gate3(records: dict, targets: dict) -> dict:
    """Stage 2 baseline coherence and reader agreement (design l.357-360).

    The registered gate has three parts: the unpushed baselines are unbroken on all 4 Stage 2 prompts, the frozen
    second reader passes `>= 9/10` known examples (`reader_known_pass` in the audit JSON), and Cohen's kappa is
    `>= 0.60` on the 40-item sample. `status` is `passed`, `failed` or `not read`:
    - broken baselines fail the gate whether or not an audit exists;
    - with no audit labels the reader parts are `not read`, so the gate is `not read` (never `passed`: Luna r2 found
      the old code reported `passed` with no reader evidence). `score` then leaves 213.6 `not read`, not voided;
    - with labels, a missing or failing kappa or known-example result fails the gate (F5: labels without kappa).
    """
    rows = records["stage2"]
    baseline = [r for r in rows if r.get("arm") == "none"]
    broken = [r["prompt_id"] for r in baseline if broken_answer(r.get("answer", ""))]
    # The pilot's stopping rule reads the same 4 unpushed baselines (design l.384-388); when the full run has not
    # written stage2 yet, those recorded pilot answers are the only baseline evidence, so they are used.
    if not baseline:
        run = records["runs"].get("instruct") or {}
        for entry in run.get("stage2_baseline_answers") or []:
            if broken_answer(entry.get("answer", "")):
                broken.append(entry.get("id"))
    kappa = records.get("kappa")
    known = records.get("reader_known_pass")
    labels_present = bool(records.get("labels"))
    notes = []
    if broken:
        status = "failed"
        notes.append("an unpushed Stage 2 baseline is broken")
    elif not labels_present:
        status = "not read"
        notes.append("reader agreement is `not read` until the frozen second reader labels the 40-item sample")
    else:
        kappa_ok = kappa is not None and kappa >= 0.60
        known_ok = known is not None and known >= 9
        status = "passed" if (kappa_ok and known_ok) else "failed"
        if kappa is None:
            notes.append("audit labels supplied without kappa: the audited endpoint is unread (gate fails)")
        elif not kappa_ok:
            notes.append(f"kappa {kappa} < 0.60")
        if known is None:
            notes.append("no known-example result (`reader_known_pass`, of 10) for the second reader")
        elif not known_ok:
            notes.append(f"second reader passed {known}/10 known examples (< 9)")
    return {"passed": status == "passed", "status": status, "baseline_broken": broken, "kappa": kappa,
            "reader_known_pass": known, "labels_present": labels_present, "note": "; ".join(notes)}


def gate4(records: dict, targets: dict) -> dict:
    """Prospective third-arm negative-evidence validity rule on the 42 slot neuron +0.8 THIRD_FREE answers (l.353-355).

    F5: the rule counts `valid_refusal_or_unknown` labels, so it cannot pass without them. With no labels (or only a
    partial audit) it reports `not read` and fails rather than counting refusal as 0 and passing on broken=0.
    """
    units = {u["id"]: u for u in targets["units"]}
    rows = [r for r in records["stage2"] if r.get("status", "ok") == "ok" and r.get("prompt_id") in THIRD_FREE
            and r.get("arm") == "neuron" and r.get("dose") == "+0.8" and units[r["unit"]]["kind_haiku"] == "slot"]
    if not rows:
        return {"passed": False, "n": 0, "note": "no slot neuron +0.8 THIRD_FREE rows"}
    broken = sum(broken_answer(r.get("answer", "")) for r in rows)
    labels = records.get("labels") or {}
    labelled = sum(1 for r in rows if r.get("blind_id") is not None and labels.get(r.get("blind_id")) is not None)
    if labelled < len(rows):
        return {"passed": False, "n": len(rows), "broken": broken, "labelled": labelled,
                "note": "gate 4 requires blind labels for all 42 slot neuron +0.8 THIRD_FREE answers"}
    refusal = sum(1 for r in rows if labels.get(r.get("blind_id")) == "valid_refusal_or_unknown")
    neither = len(rows) - broken - refusal
    return {"passed": broken <= 8 and neither >= 25, "n": len(rows), "broken": broken, "labelled": labelled,
            "valid_refusal_or_unknown": refusal, "neither": neither}


# ---------------------------------------------------------------------------------------------------------------------
# predictions
# ---------------------------------------------------------------------------------------------------------------------

def group_medians(s1: Stage1, ckpt, kind, prompts) -> dict:
    ids = s1.slot if kind == "slot" else s1.concept
    n = [s1.nz(ckpt, u, prompts) for u in ids]
    l = [s1.lz(ckpt, u, prompts) for u in ids]
    return {"N": bootstrap_ci([v for v in n if v is not None]), "L": bootstrap_ci([v for v in l if v is not None]),
            "n_positive_both": sum(1 for a, b in zip(n, l) if a is not None and b is not None and a > 0 and b > 0),
            "n_units": len(ids)}


def own_full_np_counts(s1: Stage1, ckpt, kind, prompts) -> dict:
    """Own-unit full-NP increment lp_content_sum(neuron +0.8) - lp_content_sum(none) > 0 (213.1/213.2, l.435-436)."""
    ids = s1.slot if kind == "slot" else s1.concept
    counts = {}
    for uid in ids:
        vals = []
        for pid in prompts:
            pushed = s1.rec(ckpt, uid, pid, "neuron", "+0.8")
            base = s1.rec(ckpt, uid, pid, "none", "0.0")
            if pushed is None or base is None or pushed.get("lp_sum") is None:
                vals = None
                break
            vals.append(pushed["lp_sum"] - base["lp_sum"])
        counts[uid] = None if vals is None else float(sum(vals) / len(vals))
    return {"positive": sum(1 for v in counts.values() if v is not None and v > 0),
            "n_units": len(ids), "per_unit": {k: (None if v is None else round(v, 4)) for k, v in counts.items()}}


def predict_213_1(s1: Stage1) -> dict:
    slot = group_medians(s1, "instruct", "slot", SELF_HELD)
    concept = group_medians(s1, "instruct", "concept", SELF_HELD)
    slot_np = own_full_np_counts(s1, "instruct", "slot", SELF_HELD)
    concept_np = own_full_np_counts(s1, "instruct", "concept", SELF_HELD)
    clauses = {
        "slot_both_positive_ge_18": point_clause(slot["n_positive_both"] >= 18),
        "concept_both_positive_ge_15": point_clause(concept["n_positive_both"] >= 15),
        "slot_median_L_lo_ge_1": ci_clause(slot["L"], "ge", 1.0),
        "slot_median_N_lo_ge_1": ci_clause(slot["N"], "ge", 1.0),
        "concept_median_L_lo_ge_1": ci_clause(concept["L"], "ge", 1.0),
        "concept_median_N_lo_ge_1": ci_clause(concept["N"], "ge", 1.0),
        "slot_own_full_np_ge_18": point_clause(slot_np["positive"] >= 18),
        "concept_own_full_np_ge_15": point_clause(concept_np["positive"] >= 15),
    }
    return {"clauses": clauses, "slot": slot, "concept": concept,
            "own_full_np_slot": slot_np, "own_full_np_concept": concept_np}


def predict_213_2(s1: Stage1) -> dict:
    slot = group_medians(s1, "instruct", "slot", THIRD_SLOT)
    slot_np = own_full_np_counts(s1, "instruct", "slot", THIRD_SLOT)
    clauses = {
        "median_L_lo_ge_0.75": ci_clause(slot["L"], "ge", 0.75),
        "median_N_ge_0.75": point_clause(slot["N"]["point"] >= 0.75),
        "both_positive_ge_16": point_clause(slot["n_positive_both"] >= 16),
        "own_full_np_ge_16": point_clause(slot_np["positive"] >= 16),
    }
    return {"clauses": clauses, "slot": slot, "own_full_np": slot_np}


def predict_213_3(s1: Stage1) -> dict:
    tau_l = s1.ratio("instruct", "L", THIRD_SLOT, SELF_HELD)
    tau_n = s1.ratio("instruct", "N", THIRD_SLOT, SELF_HELD)
    style = {}
    for name, pairs in (("direct", DIRECT_PAIRS), ("scene", SCENE_PAIRS)):
        per_unit = []
        for uid in s1.slot:
            ratios = []
            for t, s in pairs:
                a, b = s1.lz("instruct", uid, (t,)), s1.lz("instruct", uid, (s,))
                if a is None or b is None or b <= ELIGIBLE_BAR:
                    ratios = None
                    break
                ratios.append(a / b)
            if ratios:
                per_unit.append(sum(ratios) / len(ratios))
        style[name] = {"point_median": round(median(per_unit), 4) if per_unit else None, "n": len(per_unit)}
    clauses = {
        "tau_L_lo_ge_0.50": ratio_clause(tau_l, "ge", 0.50),
        "tau_N_lo_ge_0.50": ratio_clause(tau_n, "ge", 0.50),
        "direct_style_median_ge_0.50": point_clause(style["direct"]["point_median"] is not None
                                                    and style["direct"]["point_median"] >= 0.50),
        "scene_style_median_ge_0.50": point_clause(style["scene"]["point_median"] is not None
                                                   and style["scene"]["point_median"] >= 0.50),
    }
    return {"clauses": clauses, "tau_3/1_L": tau_l, "tau_3/1_N": tau_n, "style": style}


def predict_213_4(s1: Stage1) -> dict:
    slot = group_medians(s1, "instruct", "slot", MENTION)
    tau = s1.ratio("instruct", "L", MENTION, THIRD_SCENE)
    pooled = s1.ratio("instruct", "L", MENTION, THIRD_SLOT)
    clauses = {
        "median_L_lo_ge_0.50": ci_clause(slot["L"], "ge", 0.50),
        "median_N_ge_0.50": point_clause(slot["N"]["point"] >= 0.50),
        "tau_M/3_L_lo_ge_0.50": ratio_clause(tau, "ge", 0.50),
    }
    return {"clauses": clauses, "slot": slot, "tau_M/3_scene": tau, "tau_M/3_pooled_descriptive": pooled}


def predict_213_5(s1: Stage1) -> dict:
    slot_l = [v for v in (s1.lz("instruct", u, MENTION) for u in s1.slot) if v is not None]
    concept_l = [v for v in (s1.lz("instruct", u, MENTION) for u in s1.concept) if v is not None]
    agreed_l = [v for v in (s1.lz("instruct", u, MENTION) for u in s1.agreed_concept) if v is not None]
    diffs = [c - s for c in concept_l for s in slot_l]  # every concept-slot pair, one draw per pair
    ci = bootstrap_ci(diffs) if diffs else {"lo": float("nan"), "hi": float("nan")}
    agreed_diff = (median(agreed_l) - median(slot_l)) if agreed_l and slot_l else float("nan")
    clauses = {
        "concept_minus_slot_lo_ge_0.25": ci_clause(ci, "ge", 0.25),
        "agreed_minus_slot_ge_0.25": point_clause(bool(agreed_l) and agreed_diff >= 0.25),
    }
    reader_sensitive = bool(agreed_l) and bool(slot_l) and ((agreed_diff >= 0.25) != (ci["lo"] >= 0.25))
    return {"clauses": clauses, "median_L_concept": round(median(concept_l), 4) if concept_l else None,
            "median_L_slot": round(median(slot_l), 4) if slot_l else None,
            "median_L_agreed_concept": round(median(agreed_l), 4) if agreed_l else None,
            "ci_difference": ci, "agreed_difference": round(agreed_diff, 4) if agreed_l else None,
            "reader_sensitive": reader_sensitive}


def expected_audit_keys(targets: dict) -> list:
    """The (unit, prompt) pairs one Stage 2 audit arm must cover exactly once: every slot unit x THIRD_FREE."""
    return [(u["id"], p) for u in targets["units"] if u["kind_haiku"] == "slot" for p in THIRD_FREE]


def predict_213_6(records: dict, targets: dict, audit: dict | None) -> dict:
    """Stage 2 blind audit: 42 slot `neuron +0.8` vs 42 paired `random_0 +0.8` THIRD_FREE answers (design l.455-459).

    F7: a partial audit is not a complete one. Every one of the 84 audited rows must carry a label; otherwise the
    prediction is `not read` and the unlabelled count is reported, so nobody can steer the verdict by labelling only
    the rows that favour them.
    L7: the difference CI is the unit bootstrap of the total `(neuron_hits - random_hits)`, and "strictly above 0"
    is `ge_strict` (a 2.5th percentile sitting exactly on 0 is not above it).
    """
    rows = records["stage2"]
    units = {u["id"]: u for u in targets["units"]}
    labels = (audit or {}).get("labels", {})
    hits = {"neuron": 0, "random": 0}
    per_unit: dict[str, dict[str, int]] = {}
    audited = []
    for r in rows:
        if r.get("status", "ok") != "ok" or r.get("prompt_id") not in THIRD_FREE:
            continue
        kind = units[r["unit"]]["kind_haiku"] if r.get("unit") else None
        if kind != "slot":
            continue
        if r["arm"] == "neuron" and r["dose"] == "+0.8":
            arm = "neuron"
        elif r["arm"] == "random_0" and r["dose"] == "+0.8":
            arm = "random"
        else:
            continue
        audited.append((r, arm))
        label = labels.get(r.get("blind_id")) if r.get("blind_id") is not None else None
        hit = label == "genuine_third_attr"
        hits[arm] += int(hit)
        per_unit.setdefault(r["unit"], {"neuron": 0, "random": 0})[arm] += int(hit)
    labelled = sum(1 for r, _arm in audited if r.get("blind_id") is not None
                   and labels.get(r.get("blind_id")) is not None)
    # Luna r2: every present row labelled is not enough. The registered audit is every slot unit on BOTH THIRD_FREE
    # prompts in BOTH arms, each exactly once (21 x 2 x 2 = 84); one prompt's 42 rows, or a duplicated row standing
    # in for a missing one, is a partial audit and leaves every clause `not read`.
    expected = expected_audit_keys(targets)
    shape_ok = all(sorted((r["unit"], r["prompt_id"]) for r, a in audited if a == arm) == sorted(expected)
                   for arm in ("neuron", "random"))
    complete = bool(audited) and labelled == len(audited) and shape_ok
    discordant = [v["neuron"] - v["random"] for v in per_unit.values() if v["neuron"] != v["random"]]
    n_disc = len(discordant)
    k = sum(1 for d in discordant if d > 0)
    p = sign_test_p(k, n_disc)
    per_unit_diff = [v["neuron"] - v["random"] for v in per_unit.values()]
    diff_ci = bootstrap_ci(per_unit_diff, statistic=lambda sample: float(sum(sample))) if per_unit_diff \
        else {"lo": float("nan"), "hi": float("nan")}
    if not labels:
        clauses = {"audit": "not read", "sign_test": "not read", "difference_ci": "not read"}
    elif not complete:
        clauses = {"audit_complete": "not read", "neuron_ge_10_of_42": "not read",
                   "random_le_2_of_42": "not read", "sign_test_p_lt_0.05": "not read",
                   "difference_ci_above_0": "not read"}
    else:
        clauses = {
            "neuron_ge_10_of_42": point_clause(hits["neuron"] >= 10),
            "random_le_2_of_42": point_clause(hits["random"] <= 2),
            "sign_test_p_lt_0.05": point_clause(p < 0.05),
            "difference_ci_above_0": ci_clause(diff_ci, "ge_strict", 0.0),
        }
    return {"clauses": clauses, "hits": hits, "n_discordant": n_disc, "n_favouring_neuron": k,
            "sign_test_p": p, "difference_ci": diff_ci, "audited_rows": len(audited),
            "labelled_rows": labelled, "audit_complete": complete, "expected_audited_rows": 2 * len(expected)}


def predict_213_7(s1: Stage1) -> dict:
    out = {}
    clauses = {}
    for kind, ids in (("slot", s1.slot), ("concept", s1.concept)):
        vals = [v for v in (s1.negative_gap("instruct", u, SELF_HELD) for u in ids) if v is not None]
        ci = bootstrap_ci(vals)
        out[kind] = ci
        # L2: the design prints `z_tilde(neuron -0.8) - z_tilde(none)` alongside; descriptive only.
        alongside = []
        for u in ids:
            for pid in SELF_HELD:
                a, b = s1.zt("instruct", u, pid, "neuron", "-0.8"), s1.zt("instruct", u, pid, "none", "0.0")
                if a is not None and b is not None:
                    alongside.append(a - b)
        out[f"{kind}_neuron_minus_none"] = bootstrap_ci(alongside) if alongside else {"n": 0}
        clauses[f"{kind}_upper_le_-0.30"] = ci_clause(ci, "le", -0.30)
    return {"clauses": clauses, **out}


def predict_213_8(s1: Stage1) -> dict:
    j_self = {u: s1.j("instruct", u, SELF_HELD) for u in s1.slot}
    l_self = {u: s1.lz("instruct", u, SELF_HELD) for u in s1.slot}
    j_third = {u: s1.j("instruct", u, THIRD_SLOT) for u in s1.slot}
    l_third = {u: s1.lz("instruct", u, THIRD_SLOT) for u in s1.slot}
    n_pos = sum(1 for v in j_self.values() if v is not None and v > 0)
    rho1 = spearman([j_self[u] for u in s1.slot], [l_self[u] for u in s1.slot])
    diff_j = [(j_self[u] - j_third[u]) if j_self[u] is not None and j_third[u] is not None else None for u in s1.slot]
    diff_l = [(l_self[u] - l_third[u]) if l_self[u] is not None and l_third[u] is not None else None for u in s1.slot]
    rho2 = spearman(diff_j, diff_l)
    clauses = {
        "J_self_positive_ge_17": point_clause(n_pos >= 17),
        "rho_J_L_self_ge_0.50": point_clause(math.isfinite(rho1) and rho1 >= 0.50),
        "rho_context_contrast_ge_0.40": point_clause(math.isfinite(rho2) and rho2 >= 0.40),
    }
    return {"clauses": clauses, "n_J_positive": n_pos, "rho_self": round(rho1, 4) if math.isfinite(rho1) else None,
            "rho_context_contrast": round(rho2, 4) if math.isfinite(rho2) else None,
            "J_self": {k: (None if v is None else round(v, 4)) for k, v in j_self.items()}}


def predict_213_9(s1: Stage1) -> dict:
    sft_self = group_medians(s1, "sft", "slot", SELF_HELD)
    sft_third = group_medians(s1, "sft", "slot", THIRD_SLOT)
    tau = {"sft": s1.ratio("sft", "L", THIRD_SLOT, SELF_HELD), "dpo": s1.ratio("dpo", "L", THIRD_SLOT, SELF_HELD)}
    # Paired difference on units eligible at BOTH checkpoints (design l.443).
    per_unit = []
    for uid in s1.slot:
        a_sft, b_sft = s1.lz("sft", uid, THIRD_SLOT), s1.lz("sft", uid, SELF_HELD)
        a_dpo, b_dpo = s1.lz("dpo", uid, THIRD_SLOT), s1.lz("dpo", uid, SELF_HELD)
        if None in (a_sft, b_sft, a_dpo, b_dpo) or b_sft <= ELIGIBLE_BAR or b_dpo <= ELIGIBLE_BAR:
            continue
        per_unit.append(a_dpo / b_dpo - a_sft / b_sft)
    n_eligible = len(per_unit)
    ci = bootstrap_ci(per_unit)
    clauses = {
        "sft_median_L_self_ge_0.75": point_clause(sft_self["L"]["point"] >= 0.75),
        "sft_median_N_self_ge_0.75": point_clause(sft_self["N"]["point"] >= 0.75),
        "sft_median_L_third_ge_0.50": point_clause(sft_third["L"]["point"] >= 0.50),
        "sft_median_N_third_ge_0.50": point_clause(sft_third["N"]["point"] >= 0.50),
        "paired_difference_inside_-0.25_0.25": ci_clause(ci, "inside", (-0.25, 0.25), n_eligible),
    }
    return {"clauses": clauses, "sft_self": sft_self, "sft_third": sft_third, "tau_sft": tau["sft"],
            "tau_dpo": tau["dpo"], "paired_difference": ci, "n_eligible": n_eligible}


def predict_213_10(s1: Stage1) -> dict:
    diffs = []
    for uid in s1.slot:
        a, b = s1.lz("instruct", uid, SELF_HELD), s1.lz("instruct", uid, SELF_NONID)
        if a is not None and b is not None:
            diffs.append(a - b)
    ci = bootstrap_ci(diffs)
    tau = s1.ratio("instruct", "L", COPULA_NONID, THIRD_SCENE)
    pooled = s1.ratio("instruct", "L", COPULA_NONID, THIRD_SLOT)
    clauses = {
        "self_held_minus_self_nonid_lo_ge_0.50": ci_clause(ci, "ge", 0.50),
        "tau_cop_nonid/3_upper_le_0.50": ratio_clause(tau, "le", 0.50),
    }
    return {"clauses": clauses, "self_held_minus_self_nonid": ci, "tau_cop_nonid/3_scene": tau,
            "tau_cop_nonid/3_pooled_descriptive": pooled}


def predict_213_11(s1: Stage1) -> dict:
    per_unit = {u: s1.delta_lp_spec("instruct", u) for u in s1.slot}
    full = [v["delta_lp_spec"] for v in per_unit.values() if v["delta_lp_spec"] is not None]
    slot = [v["delta_lp_spec_slot"] for v in per_unit.values() if v["delta_lp_spec_slot"] is not None]
    ci_full = bootstrap_ci(full)
    ci_slot = bootstrap_ci(slot)
    clauses = {
        "mean_delta_lp_spec_lo_ge_0.75": ci_clause(ci_full, "ge", 0.75),
        "positive_ge_16_of_21": point_clause(sum(1 for v in full if v > 0) >= 16),
        "slot_peer_only_lo_gt_0": ci_clause(ci_slot, "ge_strict", 0.0),
    }
    return {"clauses": clauses, "delta_lp_spec": ci_full, "delta_lp_spec_slot": ci_slot,
            "n_positive": sum(1 for v in full if v > 0),
            "per_unit": {k: {kk: (None if vv is None else round(vv, 4)) for kk, vv in v.items() if kk != "unit"}
                         for k, v in per_unit.items()},
            "v4_sum_form_descriptive": {u: (None if s1.delta_spec_v4("instruct", u) is None
                                            else round(s1.delta_spec_v4("instruct", u), 4)) for u in s1.slot}}


def fr_symmetry(s1: Stage1) -> dict:
    """The FR symmetry diagnostic (design l.323-325), reported, not a prediction."""
    def tau(ckpt):
        r = s1.ratio(ckpt, "L", THIRD_SLOT, SELF_HELD)
        return r["point"]
    t = {ckpt: tau(ckpt) for ckpt in ("instruct", "dpo", "fr")}
    if None in t.values():
        return {"tau": t, "flagged": None}
    return {"tau": t, "flagged": abs(t["fr"] - t["instruct"]) >= abs(t["dpo"] - t["instruct"])}


# ---------------------------------------------------------------------------------------------------------------------
# secondary diagnostics (design l.202-203, l.218-224, l.451-459)
# ---------------------------------------------------------------------------------------------------------------------

def extra_diagnostics(s1: Stage1) -> dict:
    """SELF_SELECT - SELF_HELD, ratio-of-medians and paired differences, own - mean(O_u), lp_0 / lp_pushed, entropy.

    Printed alongside (design l.202-203, l.219-224); none of it decides a prediction. A sealed record reads None.
    """
    sel = {u: s1.lz("instruct", u, ("self_select_0",)) for u in s1.slot}
    held = {u: s1.lz("instruct", u, SELF_HELD) for u in s1.slot}
    inflation = {u: (sel[u] - held[u]) if sel[u] is not None and held[u] is not None else None for u in s1.slot}
    ratios = {}
    for name, num, den in (("3/1", THIRD_SLOT, SELF_HELD), ("M/3", MENTION, THIRD_SCENE),
                           ("cop_nonid/3", COPULA_NONID, THIRD_SCENE)):
        l_num = [s1.lz("instruct", u, num) for u in s1.slot]
        l_den = [s1.lz("instruct", u, den) for u in s1.slot]
        n_num = [s1.nz("instruct", u, num) for u in s1.slot]
        n_den = [s1.nz("instruct", u, den) for u in s1.slot]
        ratios[name] = {
            "ratio_of_medians_L": _safe_div(median([v for v in l_num if v is not None]),
                                            median([v for v in l_den if v is not None])),
            "ratio_of_medians_N": _safe_div(median([v for v in n_num if v is not None]),
                                            median([v for v in n_den if v is not None])),
        }
    paired = {}
    for name, a, b in (("S_3v1", THIRD_SLOT, SELF_HELD), ("S_3vM", THIRD_SLOT, MENTION)):
        diffs = []
        for u in s1.slot:
            x, y = s1.lz("instruct", u, a), s1.lz("instruct", u, b)
            if x is not None and y is not None:
                diffs.append(x - y)
        paired[name] = bootstrap_ci(diffs)
    own_vs_mean = {}
    lp_ends = {}
    for u in s1.slot:
        vals = []
        for pid in SELF_HELD:
            row = s1.rec("instruct", u, pid, "neuron", "+0.8")
            base = s1.rec("instruct", u, pid, "none", "0.0")
            if row is None or base is None:
                vals = None
                break
            own = logit_value(row, int(s1.units_by_id[u]["first_content_token_id"]))
            others = [logit_value(row, int(s1.units_by_id[v]["first_content_token_id"]))
                      for v in s1.units_by_id[u]["offdiag_ou"]]
            vals.append(own - sum(others) / len(others))
        own_vs_mean[u] = None if vals is None else round(sum(vals) / len(vals), 4)
        ends = []
        for pid in SELF_HELD:
            row = s1.rec("instruct", u, pid, "neuron", "+0.8")
            base = s1.rec("instruct", u, pid, "none", "0.0")
            if row is None or base is None:
                ends = None
                break
            ends.append({"lp_0": round(base["lp_sum"], 4), "lp_pushed": round(row["lp_sum"], 4),
                         "entropy_0": round(base["entropy"], 4), "top1_0": round(base["top1"], 6)})
        lp_ends[u] = ends
    return {"self_select_minus_self_held_L": {u: (None if v is None else round(v, 4)) for u, v in inflation.items()},
            "ratio_of_medians": ratios, "paired_differences": paired,
            "own_minus_mean_O_u": own_vs_mean, "lp_ends": lp_ends}


def _safe_div(a, b):
    if a is None or b is None or not math.isfinite(a) or not math.isfinite(b) or b == 0:
        return None
    return round(a / b, 4)


def diagnostics(s1: Stage1) -> dict:
    units = s1.units_by_id
    exclude_sets = {
        "subword_first_token": [u for u in units if not units[u]["is_full_word_first_token"]],
        "bare_lead": [u for u in units if units[u]["lead_class"] == "bare"],
        "p92-18": ["p92-18"],
    }
    sens = {}
    for name, drop in exclude_sets.items():
        keep = [u for u in s1.slot if u not in drop]
        vals = [s1.lz("instruct", u, SELF_HELD) for u in keep]
        sens[name] = {"dropped": drop, "n": len(keep),
                      "median_L_self_held": round(median([v for v in vals if v is not None]), 4) if any(
                          v is not None for v in vals) else None}
    dog = {}
    for uid in ("p92-08", "p92-09"):
        dog[uid] = {"kind": units[uid]["kind_haiku"],
                    "N_self_held": s1.nz("instruct", uid, SELF_HELD),
                    "L_self_held": s1.lz("instruct", uid, SELF_HELD)}
    rank = {}
    for uid in s1.slot:
        ranks = []
        for pid in SELF_HELD:
            row = s1.rec("instruct", uid, pid, "neuron", "+0.8")
            if row is None:
                ranks = None
                break
            own = logit_value(row, int(units[uid]["first_content_token_id"]))
            others = [logit_value(row, int(units[v]["first_content_token_id"])) for v in units[uid]["offdiag_ou"]]
            ranks.append(1 + sum(1 for v in others if v > own))
        rank[uid] = None if ranks is None else float(sum(ranks) / len(ranks))
    return {"sensitivity": sens, "dog_pair": dog, "own_token_rank_in_O_u": rank}


# ---------------------------------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------------------------------

def assign_blind_ids(records: dict, targets: dict) -> list[dict]:
    """Give every audited Stage 2 row its seed-213 `blind_id` before any prediction or gate reads labels.

    The scorer used to assign these only after predicting 213.6 and Gate 4, so every label lookup saw `blind_id=None`.
    """
    blind = export_blind(records["stage2"], targets)
    for item, row in zip(blind, _blind_order(records["stage2"], targets)):
        row["blind_id"] = item["blind_id"]
    return blind


def slot_genuine_ci(records: dict, targets: dict, audit: dict | None) -> dict:
    """F5: the 10,000-draw unit-bootstrap 95% CI of the slot `genuine_third_attr` rate on Stage 2 `THIRD_FREE`.

    Headline 3 requires this CI's upper bound to be strictly `< 10/42` (design l.513-515). The bootstrap resamples the
    21 slot units (each carries two THIRD_FREE answers); its statistic is the total hits / 42. It is `not read`
    unless every one of the 42 neuron +0.8 rows is labelled (a partial audit cannot support the headline).
    """
    labels = (audit or {}).get("labels", {})
    units = {u["id"]: u for u in targets["units"]}
    rows = [r for r in records["stage2"] if r.get("status", "ok") == "ok" and r.get("prompt_id") in THIRD_FREE
            and r.get("arm") == "neuron" and r.get("dose") == "+0.8" and units[r["unit"]]["kind_haiku"] == "slot"]
    per_unit: dict[str, int] = {}
    labelled = 0
    for r in rows:
        per_unit.setdefault(r["unit"], 0)
        label = labels.get(r.get("blind_id")) if r.get("blind_id") is not None else None
        if label is not None:
            labelled += 1
        per_unit[r["unit"]] += int(label == "genuine_third_attr")
    keys = sorted((r["unit"], r["prompt_id"]) for r in rows)
    if keys != sorted(expected_audit_keys(targets)) or labelled != len(rows) or len(per_unit) < MIN_ELIGIBLE:
        return {"n": len(per_unit), "rows": len(rows), "labelled": labelled, "point": None,
                "lo": None, "hi": None, "read": "not read"}
    values = list(per_unit.values())
    ci = bootstrap_ci(values, statistic=lambda sample: float(sum(sample)) / 42.0)
    ci["read"] = "read"
    return ci


def residual_norm_comparison(runs: dict) -> dict:
    """L3: the per-checkpoint residual-norm comparison the design asks for (l.123-124, l.132).

    `run.json` stores each checkpoint's median layer-0 residual norm; the ratio is against instruct. Descriptive.
    """
    reference = (runs.get("instruct") or {}).get("residual_median_norm_layer0")
    out = {}
    for ckpt in CHECKPOINTS:
        value = (runs.get(ckpt) or {}).get("residual_median_norm_layer0")
        out[ckpt] = {"median_norm_layer0": value,
                     "ratio_to_instruct": (value / reference) if value is not None and reference else None}
    return out


def score(records: dict, targets: dict, audit: dict | None = None) -> dict:
    s1 = Stage1(records, targets)
    # Blind ids first: Gate 4 and 213.6 join audit labels by blind_id.
    blind = assign_blind_ids(records, targets)
    g0, g1 = gate0(targets), gate1(records, targets)
    g2 = gate2(records)
    g3 = gate3(records, targets)
    g4 = gate4(records, targets)
    stage2_ci = slot_genuine_ci(records, targets, audit)
    gates = {"gate0": g0, "gate1": g1, "gate2": g2, "gate3": g3, "gate4": g4}
    stage1_gate_failed = not (g1["passed"] and g2["passed"])
    # Gate 3 `not read` (no audit yet) leaves 213.6 to compute itself as `not read`; only a FAILED gate voids it.
    stage2_gate_failed = not g1["passed"] or g3["status"] == "failed"

    preds = {}
    for name, fn in (("213.1", lambda: predict_213_1(s1)), ("213.2", lambda: predict_213_2(s1)),
                     ("213.3", lambda: predict_213_3(s1)), ("213.4", lambda: predict_213_4(s1)),
                     ("213.5", lambda: predict_213_5(s1)), ("213.7", lambda: predict_213_7(s1)),
                     ("213.8", lambda: predict_213_8(s1)), ("213.9", lambda: predict_213_9(s1)),
                     ("213.10", lambda: predict_213_10(s1)), ("213.11", lambda: predict_213_11(s1))):
        if stage1_gate_failed:
            # a failed gate voids the Stage 1 read (design l.460-461); nothing is computed against missing records
            preds[name] = {"verdict": "untestable (gate failed)", "clauses": {}}
            continue
        try:
            result = fn()
        except Exception as exc:  # a missing record must not crash the whole table
            result = {"clauses": {"error": "not read"}, "error": repr(exc)}
        result["verdict"] = disposition(result["clauses"])
        preds[name] = result
    if stage2_gate_failed:
        p6 = {"verdict": "untestable (gate failed)", "clauses": {}}
    else:
        try:
            p6 = predict_213_6(records, targets, audit)
        except Exception as exc:
            p6 = {"clauses": {"error": "not read"}, "error": repr(exc)}
        p6["verdict"] = disposition(p6["clauses"])
    preds["213.6"] = p6
    preds = {k: preds[k] for k in sorted(preds, key=lambda s: float(s.split(".")[1]))}

    blind = assign_blind_ids(records, targets)  # already assigned above; returns the same export
    stage2 = {
        "blind_export": blind,
        "audit_table": audit_table(records["stage2"], (audit or {}).get("labels", {}), targets),
        "mechanical_third_free": dedup_keyword_counts(records["stage2"], targets, THIRD_FREE, "neuron", "+0.8"),
        "mechanical_cold": dedup_keyword_counts(records["stage2"], targets, ("free_cold_0",), "neuron", "+0.8"),
        "second_reader_kappa": (audit or {}).get("kappa"),
        "slot_minus_0.8_genuine": (audit or {}).get("slot_minus_genuine"),
        "slot_genuine_rate_ci": stage2_ci,
    }
    headline = headline_reading(preds, gates, audit, stage2_ci)
    return {"part": 213, "design": "PART213-DESIGN.md v5 (commit 7de53067)", "gates": gates,
            "predictions": preds, "stage2": stage2, "fr_symmetry": fr_symmetry(s1),
            "diagnostics": diagnostics(s1), "secondary": extra_diagnostics(s1),
            "residual_norm_comparison": residual_norm_comparison(records["runs"]), "headline": headline}


def headline_reading(preds: dict, gates: dict, audit: dict | None, stage2_ci: dict | None = None) -> dict:
    """The exhaustive, mutually exclusive, CI-gated headline reading (design l.466-483).

    F5: Headline 3 ("First-person self-slots supported") requires the 10,000-draw unit-bootstrap 95% CI upper bound
    of the Stage 2 slot `genuine_third_attr` rate to be strictly `< 10/42`; `stage2_ci` carries that interval, and an
    absent or partial audit leaves it `not read`, so the branch cannot fire without Stage 2 evidence.
    """
    def held(name):
        return preds[name]["verdict"] == "held"

    def ci(name, key, bound):
        v = preds[name].get(key) or {}
        ci_ = v.get("ci") or {}
        return ci_.get(bound)

    gates_1_3 = gates["gate1"]["passed"] and gates["gate2"]["passed"] and gates["gate3"]["passed"]
    minus = (audit or {}).get("slot_minus_genuine")
    minus_ok = minus is not None and minus < 10
    third_capable = gates_1_3 and all(held(n) for n in ("213.1", "213.2", "213.3", "213.4", "213.6", "213.11")) and minus_ok
    if third_capable:
        return {"reading": "Third-capable open noun-phrase writers (teacher-forced + third-person free generation)"}
    copula = gates_1_3 and all(held(n) for n in ("213.1", "213.2", "213.3", "213.6", "213.11")) and minus_ok
    tau_m_hi = ((preds["213.4"].get("tau_M/3_scene") or {}).get("ci") or {}).get("hi")
    if copula and tau_m_hi is not None and tau_m_hi <= 0.50:
        sub = "copula/identity unresolved"
        hi = ((preds["213.10"].get("tau_cop_nonid/3_scene") or {}).get("ci") or {}).get("hi")
        lo = ((preds["213.10"].get("tau_cop_nonid/3_scene") or {}).get("ci") or {}).get("lo")
        if hi is not None and hi <= 0.50:
            sub = "entity-identity-specific"
        elif lo is not None and lo >= 0.50:
            sub = "post-copula syntactic"
        return {"reading": "Third-capable copula/identity-slot writers (non-self-selective)", "sub_label": sub}
    first = (gates["gate1"]["passed"] and gates["gate2"]["passed"] and gates["gate3"]["passed"] and gates["gate4"]["passed"]
             and held("213.1")
             and (ci("213.3", "tau_3/1_L", "hi") is not None and ci("213.3", "tau_3/1_L", "hi") <= 0.50)
             and (ci("213.3", "tau_3/1_N", "hi") is not None and ci("213.3", "tau_3/1_N", "hi") <= 0.50)
             and stage2_ci is not None and stage2_ci.get("read") == "read"
             and stage2_ci.get("hi") is not None and stage2_ci["hi"] < 10 / 42)
    if first:
        return {"reading": "First-person self-slots supported",
                "stage2_slot_genuine_rate_ci": stage2_ci,
                "stage2_upper_lt_10_of_42": True}
    if preds["213.1"]["verdict"] == "failed":
        return {"reading": "inconclusive / mixed selectivity; 213.1 failed, headline withdrawn"}
    return {"reading": "inconclusive / mixed selectivity"}


def print_table(result: dict) -> None:
    print(f"{'prediction':10s} {'verdict':24s} clauses")
    for name, pred in result["predictions"].items():
        clauses = ", ".join(f"{k}={v}" for k, v in pred["clauses"].items())
        print(f"{name:10s} {pred['verdict']:24s} {clauses}")
    print("\ngates:", {k: v.get("passed") for k, v in result["gates"].items()})
    print("headline:", result["headline"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", required=True, help="directory the runner wrote (stage1-*.jsonl, stage2-*.jsonl)")
    ap.add_argument("--audit", default=None, help="blind-audit labels JSON: {labels: {blind_id: category}, ...}")
    ap.add_argument("--pilot", default=None,
                    help="pilot directory whose Gate 1/Gate 2 evidence is used (default: <records>/pilot_summary.json)")
    ap.add_argument("--out", default=None, help="score.json path (default <records>/score.json)")
    args = ap.parse_args(argv)
    targets = load_targets()
    records = load_records(Path(args.records), Path(args.pilot) if args.pilot else None)
    audit = json.loads(Path(args.audit).read_text(encoding="utf-8")) if args.audit else None
    if audit:
        records["labels"] = audit.get("labels", {})
        records["kappa"] = audit.get("kappa")
        records["reader_known_pass"] = audit.get("reader_known_pass")
    result = score(records, targets, audit)
    out = Path(args.out) if args.out else Path(args.records) / "score.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print_table(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
