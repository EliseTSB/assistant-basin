#!/usr/bin/env python3
"""Part 210's scorer: Part 165's gates and registered verdicts on the eager rerun, each printed with its file and key.

Usage: score210.py --rows results/part210/answers.jsonl [--load results/part210/load.json]
                   [--norms results/part210/norms.json] [--replicate results/part210/replicate.jsonl]
                   [--out results/part210]

Reads ONLY the rows file (and, when given, the load record for Gates R1 to R3, the rerun's norms for Gate R2's
direction check, and the replicate file for the shape floor). It never runs a model. Part 165's published numbers are
read from results/part165/ and from part210/margins210.py's build_report, which is the source of every published value
and of each distance from it (design, Reading rules).

Every number is printed as `value  <- file:key`. Gates 1 to 4 and verdicts 165.1 to 165.6 are computed by
part165/score165.py's own score_rows, imported rather than rewritten, so the registered rules are exactly Part 165's.
Gates R1 to R3 come from run210.record_gates, run210.gate_r2 and run210.gate_r2_prompts_from_rows.

Exit codes: 0 every gate and the record pass; 3 the headline reading is withdrawn (R1, R2 or R3 fails, or a Part 165
gate fails); 2 bad input (a missing file, an empty rows file, a row that is not JSON).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for entry in (str(ROOT), str(HERE), str(ROOT / "part86")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from part210 import run210  # noqa: E402
from part165 import score165  # noqa: E402
from part210 import margins210  # noqa: E402

PUBLISHED_SCORE = "results/part165/score.json"
PUBLISHED_ANSWERS = "results/part165/answers.jsonl"
PUBLISHED_NORMS = "results/part165/norms.json"
PUBLISHED_MARGINS = "results/part210/margins.md"
CAP = 80
ARMS = score165.ARMS
LADDER = score165.LADDER
# design, Predictions: the published verdicts 210.2 predicts (165.1 to 165.4 hold, 165.5 and 165.6 fail).
PREDICTION_2102 = {"165.1": "pass", "165.2": "pass", "165.3": "pass", "165.4": "pass",
                   "165.5": "fail", "165.6": "fail"}
PREDICTION_2103 = 1.5                 # the n+ over full broken ratio is above 1.5, the same side of 1 as published
PREDICTION_2105 = 2.69863             # the n+ over full broken ratio is lower than the published 2.69863
                                      # (Amendment 2: this rule had no code until 2026-11-14)
PREDICTION_2104 = 300                 # at least 300 of 768 battery answers reach the cap in each arm
PREDICTION_2106 = (40, 110)           # between 40 and 110 of the 200 replicate texts differ
PREDICTION_2107 = 8                   # the primary-broken rate is higher in at least 8 of the 9 arms
# design, Controls, the shape replicate: the floor's four labels; `changed` is left out and 165.6 is reported without a
# stability figure.
FLOOR_LABELS = run210.FLOOR_LABELS
FLOOR_VERDICTS = ("165.1", "165.2", "165.3", "165.4", "165.5")
DISPLAY_THRESHOLD = 0.95
# score165's own bootstrap: 10,000 draws at seed 165 (score165._bootstrap_spearman's defaults). Only the usable-draw
# count is read from it; every verdict reads the point estimate.
score165_BOOTSTRAP_REPLICATES = 10_000


class Printed:
    """Collects lines of the form `value  <- file:key` and prints them once, in order."""
    def __init__(self):
        self.lines = []

    def __call__(self, label, value, file, key):
        text = "null" if value is None else (f"{value:.6g}" if isinstance(value, float) else str(value))
        self.lines.append(f"{label}: {text}  <- {file}:{key}")

    def section(self, title):
        self.lines.append("")
        self.lines.append(f"## {title}")

    def text(self):
        return "\n".join(self.lines) + "\n"


def read_jsonl(path):
    return run210.read_jsonl(path)


def part165_rows(rows):
    """The rows in the shape part165/score165.py reads: Part 165's own fields on every row, the answer the decoded text.
    A row without a stage, arm or answer is refused rather than guessed at, and so is a file with no rows."""
    if not rows:
        raise ValueError("no rows to score")
    out = []
    for row in rows:
        if row.get("stage") is None or row.get("arm") is None:
            raise ValueError(f"row without a stage or arm: {row.get('pass')}/{row.get('row_id')}")
        if not isinstance(row.get("answer"), str):
            raise ValueError(f"row without an answer: {row.get('pass')}/{row.get('row_id')}")
        out.append(row)
    return out


def battery_rows(rows):
    return [r for r in rows if r.get("stage") == "battery"]


# --------------------------------------------------------------------------------------------- the shape floor

def floor_labels(row):
    """A row's shape-floor labels by Part 165's own rules (design, Controls): `broken` (the primary rule,
    score165.primary_broken), `triage` (Part 86's triage says broken), `loop` (the loop rule alone fires,
    score165.has_trigram_loop) and `short` (under four words). Each is independent, so a redraw can draw combinations no
    answer shows; those are projected with the primary rule's own precedence by possible_key and counted."""
    answer = row.get("answer") if isinstance(row.get("answer"), str) else ""
    return {"broken": bool(score165.primary_broken(answer)),
            "triage": bool(score165.triage86.triage_answer(answer) == "broken"),
            "loop": bool(score165.has_trigram_loop(answer)),
            "short": bool(len(answer.split()) < 4)}


def _vector_key(labels):
    return tuple(sorted((name, bool(labels.get(name, False))) for name in FLOOR_LABELS))


def possible_key(labels):
    """A label vector an answer can actually have. The marginal scenarios flip each label on its own and can draw, say,
    `broken` false beside `triage` true; such a draw is projected with the primary rule's own precedence
    (score165.primary_broken: broken if triage broken or loop or short) and shape_floor counts the projections."""
    triage, loop, short = (bool(labels.get(name, False)) for name in ("triage", "loop", "short"))
    broken = bool(labels.get("broken", False)) or triage or loop or short
    return _vector_key({"broken": broken, "triage": triage, "loop": loop, "short": short})


def changed_label(answer, question_index, baselines):
    """Part 165's `changed` for one answer, by score165.read_slot_rates' own rule: its first 12 whitespace-split words
    differ from that phrasing's no-push answer's. It is not in floor_labels (the floor does not redraw it), but it is
    part of what the scorer reads, so the donor key carries it."""
    base = baselines.get(int(question_index), "")
    return (answer if isinstance(answer, str) else "").split()[:12] != base.split()[:12]


def donor_key(row, baselines=None):
    """The donor key (design, Controls): the drawn label vector plus the row's question_index, and (see
    changed_label) the row's own `changed` value. Part 165's `changed` measure compares an answer's first 12 words with
    that phrasing's no-push answer, so a donor with the same labels but another phrasing's no-push answer would read as
    changed when the row was not, or the reverse; and a donor with the same labels and phrasing but another `changed`
    value (Part 165's one unchanged slot, k16 layer 19 slot 2 question 7, has a same-labelled twin that IS changed)
    would move that row's changed contribution and 165.6's estimate. `baselines` defaults to Part 165's published
    no-push answers; a rerun passes its own (the scorer reads the rerun's no-push text)."""
    if baselines is None:
        baselines = published_baselines()
    qi = int(row.get("question_index", -1))
    return (_vector_key(floor_labels(row)), qi, changed_label(row.get("answer"), qi, baselines))


def donor_texts(rows, baselines=None):
    """A deterministic donor answer for every donor key the given rows show: the first battery or replicate row in file
    order with that key. A redrawn battery row takes the donor for its drawn key, so Part 165's scorer reads a real
    answer that it scores exactly as it would score any answer with the drawn labels, phrasing and changed value."""
    if baselines is None:
        baselines = published_baselines()
    donors = {}
    for row in rows:
        if row.get("stage") in ("battery", "replicate"):
            donors.setdefault(donor_key(row, baselines), row.get("answer") if isinstance(row.get("answer"), str) else "")
    return donors


_PUBLISHED_BASELINES = None


def published_baselines():
    """Part 165's no-push answers by question_index, from the published rows (the default `changed` baseline). Read
    once: donor_key calls it per row, and re-reading a 6,920-row file per row made a donor pass take minutes."""
    global _PUBLISHED_BASELINES
    if _PUBLISHED_BASELINES is None:
        path = ROOT / PUBLISHED_ANSWERS
        _PUBLISHED_BASELINES = ({int(r["question_index"]): r.get("answer", "") for r in read_jsonl(path)
                                 if r.get("stage") == "no_push"} if path.exists() else {})
    return _PUBLISHED_BASELINES


def _score_without_bootstrap(rows, norms, reps=run210.BOOTSTRAP_REPS):
    """part165/score165.py's score_rows with the three ratios' and two Spearmans' bootstraps replaced by their point
    estimates: score165's bootstrap functions compute the point estimate before their draw loop, so passing reps=0
    returns exactly the number the published scorer reports, and the verdicts and gates it returns are identical (they
    read the estimate, not the interval). A shape-floor redraw needs no interval."""
    names = ("_bootstrap_ratio", "_bootstrap_spearman", "_bootstrap_rate_difference")
    originals = {name: getattr(score165, name) for name in names}

    def point_only(name):
        def wrapper(*args, **kwargs):
            kwargs["reps"] = reps
            return originals[name](*args, **kwargs)
        return wrapper

    for name in names:
        setattr(score165, name, point_only(name))
    try:
        return score165.score_rows(rows, norms)
    finally:
        for name in names:
            setattr(score165, name, originals[name])


def point_estimates(rows, norms):
    """The point estimates the floor's scorer reads: each ratio's estimate, each Spearman's estimate and its usable-draw
    count, and each k16 distance. The test checks these equal score_rows' published estimates on the published rows."""
    # The usable-draw count is the one number the point-only wrapper cannot give (it draws nothing), so the published
    # bootstrap is run once here, with score165's own seed and replicate count, to read it.
    score = _score_without_bootstrap(rows, norms, reps=score165_BOOTSTRAP_REPLICATES)
    out = {}
    for name, ratio in score["ratios"].items():
        out[f"ratio:{name}"] = ratio["estimate"]
        out[f"rate_difference:{name}"] = ratio["rate_difference"]["estimate"]
    for name, sp in score["spearman"].items():
        out[f"spearman:{name}"] = sp["estimate"]
        out[f"usable_draws:{name}"] = sp["usable_draws"]
        out[f"interval:{name}"] = tuple(sp["interval_95"])
    for name, pred in score["predictions"].items():
        if "distances" in pred:
            for key, value in pred["distances"].items():
                out[f"distance:{name}:{key}"] = value
    return out


def shape_floor(all_rows, replicate, norms, draws=1000, seed=0):
    """tools/shape_floor.simulate over every battery row's labels and the replicate's (chunk-of-8, one-row) pairs.

    Each redraw is scored with Part 165's own score_rows on the 8 no-push rows plus every battery row (never the battery
    alone: Gate 1 requires all 6,920 rows), with the bootstraps replaced by their point estimates and the four gates
    included as `gate:<name>`. A battery row whose redrawn labels differ from its observed ones takes a donor answer
    with exactly the drawn labels, its own question_index and its own `changed` value (donor_texts over the battery and
    replicate rows); a drawn key no row shows keeps the row's own text and is counted in `no_donor_substitutions`. Each
    distinct redraw is scored
    once (memoised). The no-push rows are fixed in every redraw."""
    from tools import shape_floor as sf
    battery = battery_rows(all_rows)
    others = [r for r in all_rows if r.get("stage") != "battery"]

    def key_of(row):
        # A replicate row is the battery row it re-decodes, so its key is the battery row's own run165.row_key (stage
        # `battery`), not the `replicate` stage it is written under.
        return run165_key(dict(row, stage="battery"))

    rep = {key_of(r): r for r in replicate if r.get("stage") in ("replicate", "battery")}
    pairs = []
    for row in battery:
        other = rep.get(key_of(row))
        if other is not None:
            pairs.append({"labels_a": floor_labels(row), "labels_b": floor_labels(other),
                          "cell": f"{row.get('arm')}|L{row.get('layer')}"})
    if not pairs:
        return None
    targets = [{"id": i, "cell": f"{row.get('arm')}|L{row.get('layer')}", "labels": floor_labels(row)}
               for i, row in enumerate(battery)]
    original = [_vector_key(t["labels"]) for t in targets]
    # `changed` is read against the no-push rows the scorer is given (the rerun's own text, not the published one).
    baselines = {int(r["question_index"]): r.get("answer", "") for r in others if r.get("stage") == "no_push"}
    donors = donor_texts(battery + list(replicate), baselines)
    counts = {"no_donor": 0, "projected": 0}
    memo = {}

    def rescore(redrawn):
        changed = tuple((r["id"], _vector_key(r.get("labels", {}))) for r in redrawn
                        if _vector_key(r.get("labels", {})) != original[r["id"]])
        if changed in memo:
            return memo[changed]
        rows_now = list(others)
        new_text = {}
        for rid, vector in changed:
            drawn = possible_key(dict(vector))
            if drawn != _vector_key(dict(vector)):
                counts["projected"] += 1
            if drawn == original[rid]:
                continue          # nothing Part 165's scorer reads moved (a projection back)
            row = battery[rid]
            qi = int(row.get("question_index", -1))
            # The donor keeps the row's own `changed` value: the floor does not redraw it, and a donor that moved it
            # would change 165.6, which the floor leaves out.
            text = donors.get((drawn, qi, changed_label(row.get("answer"), qi, baselines)))
            if text is None:
                counts["no_donor"] += 1
            else:
                new_text[rid] = text
        rows_now += [dict(row, answer=new_text[i]) if i in new_text else row for i, row in enumerate(battery)]
        score = _score_without_bootstrap(rows_now, norms)
        verdicts = {k: v["verdict"] for k, v in score["predictions"].items() if k in FLOOR_VERDICTS}
        for name, gate in score["gates"].items():
            verdicts[f"gate:{name}"] = bool(gate["passed"])
        memo[changed] = verdicts
        return verdicts

    result = sf.simulate(targets, pairs, rescore, draws=draws, seed=seed)
    result["no_donor_substitutions"] = counts["no_donor"]
    result["projected_label_draws"] = counts["projected"]
    return result


def run165_key(row):
    from part165 import run165
    return run165.row_key(row)


# --------------------------------------------------------------------------------------------- the score

def _published_margins(published, published_rows):
    """Each registered reading's published value and distance, from part210/margins210.py's build_report (the design's
    Size section). Returns {id: entry}."""
    report = margins210.build_report(published, published_rows)
    return {entry["id"]: entry for entry in report["distances"]}


def _estimate_key(pred):
    """Where a verdict's estimate sits in score165's own output: the ratio, the Spearman, or 165.3's distance margin
    (distance to n± minus distance to full, the number margins210 prints as 165.3's distance)."""
    if "estimate_ratio" in pred:
        return "predictions.<key>.estimate_ratio"
    if "spearman" in pred:
        return "predictions.<key>.spearman.estimate"
    return "predictions.<key>.distances.to_n_pm - distances.to_full"


def _distance(entry, value):
    """The rerun's distance from the published value in the bar's units, computed the way margins210 computes the
    published distance where the reading is a number (a ratio, a Spearman or a rate); None otherwise."""
    published = entry.get("estimate")
    if not isinstance(published, float) or not isinstance(value, (int, float)):
        return None
    return value - published


def score(rows, load=None, norms=None, replicate=None, published=None, published_rows=None, draws=1000, seed=0):
    out = Printed()
    rows = part165_rows(rows)
    file_rows = "rows"
    out.section("Rows")
    out("rows", len(rows), file_rows, "len")
    out("unique (stage, arm, slot, layer, question_index)", len({run165_key(r) for r in rows}), file_rows,
        "unique run165.row_key")
    out("battery rows", len(battery_rows(rows)), file_rows, "stage=='battery'")
    out("no-push rows", sum(r.get("stage") == "no_push" for r in rows), file_rows, "stage=='no_push'")
    out("nonfinite rows", sum(bool(r.get("nonfinite")) for r in rows), file_rows, "nonfinite")
    out("rows with margin fields", sum(all(k in r for k in ("margin", "top1", "top2", "emitted_logit", "finish"))
                                       for r in rows), file_rows, "margin/top1/top2/emitted_logit/finish")
    for arm in ARMS:
        members = [r for r in battery_rows(rows) if r.get("arm") == arm]
        lengths = [int(r["token_count"]) for r in members if isinstance(r.get("token_count"), (int, float))]
        out(f"answers at the {CAP}-token cap, {arm}", sum(n == CAP for n in lengths), file_rows,
            f"battery/{arm} token_count=={CAP}")
        out(f"mean token_count, {arm}", sum(lengths) / len(lengths) if lengths else None, file_rows,
            f"battery/{arm} token_count mean")

    # ---- Gates R1, R2, R3 (the record, the hashes, the rows and the norms)
    out.section("Gates R1 to R3")
    r1 = r2 = r3 = None
    if load is not None:
        planned = run210.registered_plan()
        ok, why, shapes = run210.record_gates(load, planned)
        eager = all((entry.get("load") or {}).get("attn_implementation") == "eager" for entry in load.get("loads", [])) \
            and bool(load.get("loads"))
        r1 = bool(run210.gate_r1(load.get("loads", []))[0]) and eager
        r3 = bool(shapes["passed"])
        out("R1 every launch eager", eager, "load", "loads[*].load.attn_implementation")
        out("R1 last launch status", (load.get("loads") or [{}])[-1].get("status"), "load", "loads[-1].status")
        out("R1/R3 record gates", ok, "load", "record_gates")
        out("R1/R3 reason", why or "none", "load", "record_gates")
        out("R1 device", ", ".join(sorted({str((e.get("load") or {}).get("device")) for e in load.get("loads", [])})),
            "load", "loads[*].load.device")
        out("R1 launches", len(load.get("loads", [])), "load", "loads")
        out("R1 distinct (model, revision, out_dir)",
            len({(e.get("model"), e.get("revision"), e.get("out_dir")) for e in load.get("loads", [])}), "load",
            "loads[*]")
        out("R3 planned calls without a receipt", shapes.get("planned_calls_without_receipt"), "load",
            "calls[*] vs plan")
        out("R3 pushed calls not in chunks of 8", shapes["pushed_calls_not_in_chunks_of_8"], "load", "calls[*].chunks")
        out("R3 no-push calls not one row", shapes["bad_no_push_calls"], "load", "calls[*].chunks")
        out("R3 companion rows (discarded)", shapes["companion_rows"], "load", "calls[*].chunks[*].companions")
        out("R3 largest chunk rows", shapes["largest_chunk_rows"], "load", "calls[*].chunks[*].rows")
    else:
        out("R1/R3", "not read: no --load given", "load", "-")
    r2f = run210.gate_r2()
    r2p = run210.gate_r2_prompts_from_rows(rows) if rows else {"passed": False, "rerun_questions": 0,
                                                               "first_differing_question": None}
    r2n = run210.gate_r2_norms(norms) if norms is not None else {"passed": False, "published_directions": None,
                                                                "rerun_directions": None, "outside_1e-6_relative": None,
                                                                "missing": None, "extra": None}
    r2 = bool(r2f["passed"] and r2p["passed"] and r2n["passed"])
    out("R2 input hashes", r2f["passed"], "the design's 14 files", "sha256")
    for name, entry in r2f["files"].items():
        out(f"R2 sha256 {name}", entry["sha256"], name, "sha256")
    out("R2 ordered 8 prompts match the regenerated selection", r2p["passed"], file_rows, "battery question_index 0..7")
    out("R2 rerun questions", r2p["rerun_questions"], file_rows, "battery question_index")
    out("R2 first differing question", r2p["first_differing_question"], file_rows, "battery question_index")
    out("R2 write norms equal the published norms (1e-6 relative)", r2n["passed"], "norms", "directions[*].write_norm")
    out("R2 directions outside 1e-6 relative", r2n.get("outside_1e-6_relative"), "norms", "directions[*].write_norm")
    out("R2 directions missing or extra", None if r2n.get("missing") is None
        else (r2n["missing"] + r2n["extra"]), "norms", "directions")

    # ---- Part 165's four gates and six verdicts, from part165/score165.py itself
    out.section("Part 165 gates and verdicts (part165/score165.py)")
    score_now = score165.score_rows(rows, norms if norms is not None else {})
    gates = score_now["gates"]
    for name, gate in gates.items():
        out(f"{name} passed", bool(gate["passed"]), file_rows, f"gates.{name}.passed")
        for key, value in gate.items():
            if key != "passed":
                out(f"{name} {key}", value, file_rows, f"gates.{name}.{key}")
    pub = published or {}
    margins = {}
    if published_rows:
        try:
            margins = _published_margins(published, published_rows)
        except (OSError, ValueError, KeyError) as exc:
            out("published margins", f"not read: {exc}", PUBLISHED_SCORE, "distances")
    margin_ids = {"165.1": "165.1 n+ over full", "165.2": "165.2 Spearman", "165.3": "165.3 k16 nearer full",
                  "165.4": "165.4 n+ over n±", "165.5": "165.5 iso over full", "165.6": "165.6 Spearman"}
    for key, pred in score_now["predictions"].items():
        out(f"{key} verdict", pred["verdict"], file_rows, f"predictions.{key}.verdict")
        estimate = pred.get("estimate_ratio")
        if estimate is None and "spearman" in pred:
            estimate = pred["spearman"]["estimate"]
        if estimate is None and "distances" in pred:
            estimate = pred["distances"]["to_n_pm"] - pred["distances"]["to_full"]
        out(f"{key} estimate", estimate, file_rows, _estimate_key(pred))
        pub_pred = (pub.get("predictions", {}) or {}).get(key) or {}
        out(f"{key} published verdict", pub_pred.get("verdict"), PUBLISHED_SCORE, f"predictions.{key}.verdict")
        pub_est = pub_pred.get("estimate_ratio")
        if pub_est is None and "spearman" in pub_pred:
            pub_est = (pub_pred.get("spearman") or {}).get("estimate")
        if pub_est is None and "distances" in pub_pred:
            pub_est = pub_pred["distances"]["to_n_pm"] - pub_pred["distances"]["to_full"]
        out(f"{key} published estimate", pub_est, PUBLISHED_SCORE, _estimate_key(pub_pred))
        entry = margins.get(margin_ids.get(key, ""))
        if entry is not None:
            out(f"{key} published distance from the bar", entry.get("distance"), PUBLISHED_MARGINS,
                f"distances.{margin_ids[key]}.distance")
            out(f"{key} rerun distance from the published value", _distance(entry, estimate), file_rows,
                f"predictions.{key} minus {PUBLISHED_MARGINS}:distances.{margin_ids[key]}.estimate")
        if key == "165.6":
            sp = pred.get("spearman") or {}
            out("165.6 usable draws", sp.get("usable_draws"), file_rows, "predictions.165.6.spearman.usable_draws")
            out("165.6 bootstrap replicates", sp.get("bootstrap_replicates"), file_rows,
                "predictions.165.6.spearman.bootstrap_replicates")
            interval = sp.get("interval_95") or [None, None]
            # A collapsed interval is one that carries no spread: both ends equal, or no end at all (when every draw is
            # undefined, as when `changed` is constant, score165's percentile leaves both ends None). Either way the
            # interval says nothing and is reported as not informative, never as certainty (design, Size, 165.6).
            collapsed = interval[0] == interval[1]
            out("165.6 interval", f"{interval[0]} to {interval[1]}", file_rows, "predictions.165.6.spearman.interval_95")
            out("165.6 interval collapsed (not informative)", collapsed, file_rows,
                "predictions.165.6.spearman.interval_95")
    for arm in ARMS:
        out(f"broken rate {arm}", (score_now["rates"]["broken"] or {}).get(arm), file_rows, f"rates.broken.{arm}")
        out(f"changed rate {arm}", (score_now["rates"]["changed"] or {}).get(arm), file_rows, f"rates.changed.{arm}")

    # ---- Predictions 210.1 to 210.7
    out.section("Predictions")
    gates165 = {name: bool(gate["passed"]) for name, gate in gates.items()}
    p2101 = bool(r1 and r2 and r3 and all(gates165.values()))
    out("210.1 gates R1-R3 and Part 165's four gates pass", p2101, "all gates above", "all")
    verdicts = {k: v["verdict"] for k, v in score_now["predictions"].items()}
    p2102 = all(verdicts.get(k) == v for k, v in PREDICTION_2102.items()) and all(gates165.values())
    out("210.2 165.1-165.4 hold and 165.5-165.6 fail, as published, gates passed", p2102, file_rows,
        "predictions.*.verdict")
    ratio_nf = (score_now["predictions"]["165.1"] or {}).get("estimate_ratio")
    p2103 = isinstance(ratio_nf, (int, float)) and ratio_nf > PREDICTION_2103
    out("210.3 n+ over full broken ratio above 1.5", p2103, file_rows, "predictions.165.1.estimate_ratio")
    p2105 = isinstance(ratio_nf, (int, float)) and ratio_nf < PREDICTION_2105
    out("210.5 n+ over full broken ratio lower than the published 2.69863", p2105, file_rows,
        "predictions.165.1.estimate_ratio")
    at_cap = {arm: sum(r.get("arm") == arm and int(r.get("token_count", 0)) == CAP for r in battery_rows(rows))
              for arm in ARMS}
    per_arm = {arm: sum(r.get("arm") == arm for r in battery_rows(rows)) for arm in ARMS}
    # Not read unless every arm has its 768 battery rows (DS r1 F5: the old `if per_arm >= 768` filter left an empty
    # generator on a short file, and all() of nothing printed True).
    p2104 = None if any(per_arm[arm] < 768 for arm in ARMS) else all(at_cap[arm] >= PREDICTION_2104 for arm in ARMS)
    out("210.4 at least 300 of 768 at the cap in each arm", p2104, file_rows, f"battery/* token_count=={CAP}")
    for arm in ARMS:
        out(f"210.4 {arm} at cap", f"{at_cap[arm]} of {per_arm[arm]}", file_rows, f"battery/{arm} token_count=={CAP}")
    pub_broken = ((pub.get("rates") or {}).get("broken") or {})
    higher = [arm for arm in ARMS if isinstance(score_now["rates"]["broken"].get(arm), (int, float))
              and isinstance(pub_broken.get(arm), (int, float)) and score_now["rates"]["broken"][arm] > pub_broken[arm]]
    p2107 = len(higher) >= PREDICTION_2107
    out("210.7 primary-broken rate higher in at least 8 of 9 arms", p2107, file_rows, "rates.broken vs published")
    out("210.7 arms higher", ", ".join(higher) or "none", file_rows, "rates.broken vs published")

    # ---- The shape replicate and the shape floor
    out.section("Shape replicate")
    floor = None
    p2106 = None
    if replicate:
        by_key = {run165_key(dict(r, stage="battery")): r for r in replicate
                  if r.get("stage") in ("replicate", "battery")}
        battery = {run165_key(r): r for r in battery_rows(rows)}
        shared = [k for k in by_key if k in battery]
        differ = sum(by_key[k].get("answer") != battery[k].get("answer") for k in shared)
        out("replicate rows", len(replicate), "replicate", "len")
        out("replicate rows matched to a battery row", len(shared), "replicate", "run165.row_key")
        out("replicate texts that differ from the chunk-of-8 text", differ, "replicate", "answer vs rows answer")
        lo, hi = PREDICTION_2106
        p2106 = lo <= differ <= hi
        out("210.6 between 40 and 110 of the 200 differ", p2106, "replicate", "answer vs rows answer")
        # The design's Method: how many pairs start at each value of each label.
        for label in FLOOR_LABELS:
            starts = {"true": sum(bool(floor_labels(by_key[k])[label]) for k in shared),
                      "false": sum(not floor_labels(by_key[k])[label] for k in shared)}
            out(f"replicate pairs starting {label} true / false", f"{starts['true']} / {starts['false']}", "replicate",
                f"floor_labels({label})")
        try:
            floor = shape_floor(rows, replicate, norms if norms is not None else {}, draws=draws, seed=seed)
        except (ValueError, KeyError) as exc:
            out("shape floor", f"not computed: {exc}", "replicate", "-")
        if floor:
            for scenario, info in floor["scenarios"].items():
                for verdict, entry in info["verdicts"].items():
                    out(f"stability {scenario} {verdict}", entry["sensitivity"], "rows+replicate",
                        f"shape_floor.scenarios.{scenario}.verdicts.{verdict}.sensitivity")
                    out(f"inside shape floor (<{DISPLAY_THRESHOLD}) {scenario} {verdict}",
                        entry["sensitivity"] < DISPLAY_THRESHOLD, "rows+replicate",
                        f"shape_floor.scenarios.{scenario}.verdicts.{verdict}.sensitivity")
            out("165.6 stability", "floor not informative: changed is constant", "rows+replicate", "changed left out")
            out("shape floor redraws with no donor text", floor["no_donor_substitutions"], "rows+replicate",
                "shape_floor.no_donor_substitutions")
            out("shape floor draws projected onto a possible label set", floor["projected_label_draws"],
                "rows+replicate", "shape_floor.projected_label_draws")
    else:
        out("shape replicate", "not read: no --replicate given", "replicate", "-")
        out("210.6", "not read", "replicate", "-")

    # ---- The headline rule
    out.section("Headline")
    withdrawn = not (r1 and r2 and r3 and all(gates165.values()))
    out("headline reading withdrawn", withdrawn, "gates above", "R1,R2,R3,Part165 gates")
    return out.text(), {"withdrawn": withdrawn, "gates_r1_r3": {"R1": r1, "R2": r2, "R3": r3},
                        "part165_gates": gates165, "verdicts": verdicts, "p2101": p2101, "p2102": p2102,
                        "p2103": p2103, "p2104": p2104, "p2105": p2105, "p2106": p2106, "p2107": p2107}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--load")
    ap.add_argument("--norms")
    ap.add_argument("--replicate")
    ap.add_argument("--published", default=str(ROOT / PUBLISHED_SCORE))
    ap.add_argument("--published-answers", default=str(ROOT / PUBLISHED_ANSWERS))
    ap.add_argument("--out")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    try:
        rows = read_jsonl(args.rows)
        load = json.loads(Path(args.load).read_text(encoding="utf-8")) if args.load else None
        norms = json.loads(Path(args.norms).read_text(encoding="utf-8")) if args.norms else None
        replicate = read_jsonl(args.replicate) if args.replicate else None
        if args.replicate and not replicate:
            raise ValueError(f"--replicate {args.replicate} is missing or empty")
        if args.replicate and len(replicate) != run210.REPLICATE_ROWS:
            raise ValueError(f"--replicate {args.replicate} holds {len(replicate)} rows, not the registered "
                             f"{run210.REPLICATE_ROWS}")
        published = json.loads(Path(args.published).read_text(encoding="utf-8")) if Path(args.published).exists() else None
        published_rows = read_jsonl(args.published_answers) if Path(args.published_answers).exists() else None
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"score210: bad input: {exc}", file=sys.stderr)
        return 2
    if not rows:
        print(f"score210: bad input: {args.rows} holds no rows", file=sys.stderr)
        return 2
    text, summary = score(rows, load, norms, replicate, published, published_rows, args.draws, args.seed)
    print(text, flush=True)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "score.md").write_text(text.rstrip("\n") + "\n", encoding="utf-8")
        (out / "score.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 3 if summary["withdrawn"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
