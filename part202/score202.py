#!/usr/bin/env python3
"""Part 202's scorer: Part 164's gates and registered verdicts on the rerun, each printed with its file and key.

Usage: score202.py --rows results/part202/answers.jsonl [--load results/part202/load.json]
                   [--replicate results/part202/replicate.jsonl] [--out results/part202]

Reads ONLY the rows file (and, when given, the load record for Gates R1/R2/R3 and the replicate file for the shape
floor). It never runs a model and never reads Part 164's published numbers except to print them beside the rerun's:
those are part164/score164.py's own score.json, read as a file and printed with their path and key.

Every number is printed as `value  <- file:key`. The gates and the five verdicts are computed by
part164/score164.py's own score_rows, imported rather than rewritten, so the registered rules are exactly Part 164's.
Gates R1 to R3 come from run202.record_gates and run202.gate_r2_prompts on the load record and the rows.

Exit codes: 0 every gate and the record pass; 3 the headline reading is withdrawn (R1, R2 or R3 fails, or a Part 164
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

from part202 import run202  # noqa: E402
from part164 import score164  # noqa: E402

PUBLISHED_SCORE = "results/part164/score.json"
PUBLISHED_ANSWERS = "results/part164/answers.jsonl"
PUBLISHED_VERDICTS = {"164.1": "failed", "164.2": "held", "164.3": "held", "164.4": "failed", "164.5": "held"}
PREDICTION_2022 = PUBLISHED_VERDICTS
CAP = 80
PREDICTION_2024 = {"neuron": (480, 1600), "random": (1440, 4800)}
PREDICTION_2025 = (40, 100, 200)


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
    return run202.read_jsonl(path)


def row_key(row):
    return (row.get("stage"), row.get("unit_id"), float(row.get("dose", 0.0)), int(row.get("question_index", -1)))


def part164_rows(rows):
    """The rows in the shape part164/score164.py reads: its own fields are on every row, and the answer is the
    decoded text. A row without a stage or a unit is refused rather than guessed at, and so is a file with no rows."""
    if not rows:
        raise ValueError("no rows to score")
    out = []
    for row in rows:
        if row.get("stage") is None or row.get("unit_id") is None:
            raise ValueError(f"row without a stage or unit_id: {row.get('pass')}/{row.get('row_id')}")
        out.append(row)
    return out


# --------------------------------------------------------------------------------------------- the shape floor

def labels_of(row):
    """A row's labels by Part 164's own rules: qwen (score164.qwen_hit on an eligible answer), broken
    (score164.qwen_eligible is false because triage86 calls it broken), name_gone and maker_gone (part148/score148's
    moved rule at question 0 and 1, the two battery kinds' first questions)."""
    import triage86
    from part148 import score148
    answer = row.get("answer") if isinstance(row.get("answer"), str) else ""
    broken = triage86.triage_answer(answer) == "broken" if answer.strip() else True
    labels = {"qwen": bool(score164.qwen_hit(answer) and not broken), "broken": bool(broken)}
    qi = int(row.get("question_index", 0))
    if qi == 0:
        labels["name_gone"] = bool(score148.moved(answer, 0))
    if qi == 1:
        labels["maker_gone"] = bool(score148.moved(answer, 1))
    return labels


FLOOR_LABELS = ("broken", "excluded", "qwen", "name_gone", "maker_gone")


def floor_labels(row):
    """The labels the shape floor redraws. They are chosen so that a battery row's label vector fixes its whole
    contribution to Part 164's scorer: `broken` (triage86, or an empty answer), `excluded` (not broken, but Part 164's
    EXCLUDED pattern removes it from the Qwen denominator), `qwen` (Part 164's qwen_hit on an eligible answer, so
    neither broken nor excluded), plus the design's name_gone (question 0) and maker_gone (question 1). Any two
    answers with the same vector score identically under score164.score_rows; the donor test pins that."""
    found = labels_of(row)
    answer = row.get("answer") if isinstance(row.get("answer"), str) else ""
    broken = bool(found["broken"])
    excluded = (not broken) and bool(score164.EXCLUDED.search(answer))
    labels = {"broken": broken, "excluded": excluded,
              "qwen": bool(score164.qwen_hit(answer)) and not broken and not excluded}
    for name in ("name_gone", "maker_gone"):
        if name in found:
            labels[name] = bool(found[name])
    return labels


def _vector_key(labels):
    return tuple(sorted(labels.items()))


SCORER_LABELS = ("broken", "excluded", "qwen")   # the labels score164.score_rows reads a battery answer through


def scorer_key(labels):
    """The part of a floor-label vector that Part 164's scorer reads: a battery answer enters score_rows only through
    triage86 (broken), EXCLUDED (excluded) and qwen_hit on an eligible answer (qwen), none of which depends on the
    question. name_gone and maker_gone are redrawn and counted but no 164 verdict reads them."""
    return tuple((name, bool(labels.get(name, False))) for name in SCORER_LABELS)


def possible_key(labels):
    """A scorer key an answer can actually have. broken, excluded and an eligible qwen hit are one four-way category
    (floor_labels derives them with that precedence: broken, then excluded, then qwen), but the marginal scenarios flip
    each label on its own and can draw, say, broken and qwen together. Such a draw is projected with the same
    precedence floor_labels uses; shape_floor counts the projections and prints them."""
    broken, excluded, qwen = (bool(labels.get(name, False)) for name in SCORER_LABELS)
    if broken:
        return scorer_key({"broken": True})
    if excluded:
        return scorer_key({"excluded": True})
    return scorer_key({"qwen": qwen})


def donor_texts(rows):
    """A deterministic donor answer for every scorer key the given rows show: the first row in file order with that
    key. A redrawn battery row takes the donor for its drawn key, so Part 164's own scorer reads a real answer that
    it scores exactly as it would score any answer with the drawn labels (the donor test pins that)."""
    donors = {}
    for row in rows:
        donors.setdefault(scorer_key(floor_labels(row)), row.get("answer") if isinstance(row.get("answer"), str) else "")
    return donors


def shape_floor(all_rows, replicate, draws=1000, seed=0):
    """tools/shape_floor.simulate over every battery row's labels and the replicate's (chunk-of-8, one-row) pairs.

    Each redraw is scored with Part 164's own score_rows on ALL rows (no-push and recipe as observed, every battery
    row with its redrawn labels), so the four gates are recomputed rather than failing for want of the no-push and
    recipe rows. A battery row whose redrawn labels differ from its observed ones takes a donor answer with exactly
    the scorer-read labels it drew (donor_texts over the battery and replicate rows); a drawn key no row shows keeps
    the row's own text and is counted in `no_donor_substitutions`, printed beside the floor."""
    from tools import shape_floor as sf
    battery = [r for r in all_rows if r.get("stage") == "battery"]
    others = [r for r in all_rows if r.get("stage") != "battery"]

    def key_of(row):
        return (row.get("unit_id"), float(row.get("dose", 0.0)), int(row.get("question_index", -1)))

    by_key = {key_of(r): r for r in battery}
    rep = {key_of(r): r for r in replicate}
    pairs = [{"labels_a": floor_labels(row), "labels_b": floor_labels(rep[key]), "cell": f"{row.get('arm')}|{row.get('dose')}"}
             for key, row in by_key.items() if key in rep]
    if not pairs:
        return None
    targets = [{"id": i, "cell": f"{row.get('arm')}|{row.get('dose')}", "labels": floor_labels(row)}
               for i, row in enumerate(battery)]
    original = [_vector_key(t["labels"]) for t in targets]
    donors = donor_texts(battery + list(replicate))
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
            if drawn != scorer_key(dict(vector)):
                counts["projected"] += 1
            if drawn == scorer_key(dict(original[rid])):
                continue          # nothing a 164 verdict reads moved (name_gone or maker_gone, or a projection back)
            text = donors.get(drawn)
            if text is None:
                counts["no_donor"] += 1
            else:
                new_text[rid] = text
        rows_now += [dict(row, answer=new_text[i]) if i in new_text else row for i, row in enumerate(battery)]
        score = _score_without_bootstrap(rows_now)
        verdicts = {k: v["verdict"] for k, v in score["predictions"].items()}
        for name, gate in score["gates"].items():
            verdicts[f"gate:{name}"] = bool(gate["passed"])
        memo[changed] = verdicts
        return verdicts

    result = sf.simulate(targets, pairs, rescore, draws=draws, seed=seed)
    result["no_donor_substitutions"] = counts["no_donor"]
    result["projected_label_draws"] = counts["projected"]
    return result


UNITS = None  # set by main() from part164/units164.json


def _score_without_bootstrap(rows):
    """part164/score164.py's score_rows with its bootstrap replaced by the point ratio: the verdicts and gates it
    returns are identical (they read estimate_ratio, not the interval), and a shape-floor redraw needs no interval."""
    import triage86
    original = score164.pair_bootstrap
    # Part 164's own function, run with no replicates: its point ratio is computed before its draw loop, so this is the
    # same number the published scorer reports, and the rescoring stays exactly Part 164's code.

    def point_only(*args, **kwargs):
        return original(*args, **kwargs, reps=0)

    # triage86.triage_answer is a pure function of the answer text, and a redraw only changes a few rows' texts, so it
    # is memoised per text for the length of the rescoring (restored afterwards).
    triage = triage86.triage_answer
    cache = {}

    def cached(answer):
        if answer not in cache:
            cache[answer] = triage(answer)
        return cache[answer]

    score164.pair_bootstrap = point_only
    triage86.triage_answer = cached
    try:
        return score164.score_rows(rows, UNITS)
    finally:
        score164.pair_bootstrap = original
        triage86.triage_answer = triage


# --------------------------------------------------------------------------------------------- the score

def score(rows, load=None, replicate=None, units=None, published=None, draws=1000, seed=0):
    global UNITS
    UNITS = units
    out = Printed()
    rows = part164_rows(rows)
    file_rows = "rows"
    out.section("Rows")
    out("rows", len(rows), file_rows, "len")
    out("unique (stage, unit_id, dose, question_index)", len({row_key(r) for r in rows}), file_rows, "unique keys")
    out("battery rows", sum(r.get("stage") == "battery" for r in rows), file_rows, "stage=='battery'")
    out("nonfinite rows", sum(bool(r.get("nonfinite")) for r in rows), file_rows, "nonfinite")
    out("answers at the 80-token cap, neuron", sum(r.get("stage") == "battery" and r.get("arm") == "neuron"
                                                  and int(r.get("token_count", 0)) == CAP for r in rows),
        file_rows, "battery/neuron token_count==80")
    out("answers at the 80-token cap, random", sum(r.get("stage") == "battery" and r.get("arm") == "random"
                                                  and int(r.get("token_count", 0)) == CAP for r in rows),
        file_rows, "battery/random token_count==80")
    for arm in ("neuron", "random"):
        lengths = [int(r["token_count"]) for r in rows if r.get("stage") == "battery" and r.get("arm") == arm
                   and isinstance(r.get("token_count"), (int, float))]
        out(f"mean token_count, {arm}", sum(lengths) / len(lengths) if lengths else None, file_rows,
            f"battery/{arm} token_count mean")

    # ---- Gates R1, R2, R3 (the record and the rows)
    out.section("Gates R1 to R3")
    r1 = r2 = r3 = None
    if load is not None:
        ok, why, shapes = run202.record_gates(load)
        r1 = all((entry.get("load") or {}).get("attn_implementation") == "eager" for entry in load.get("loads", [])) \
            and bool(load.get("loads"))
        r3 = bool(shapes["passed"])
        out("R1 every launch eager (and not refused)", r1, "load", "loads[*].load.attn_implementation")
        out("R1/R3 record gates", ok, "load", "record_gates")
        out("R1/R3 reason", why or "none", "load", "record_gates")
        out("R1 device", ", ".join(sorted({str((e.get("load") or {}).get("device")) for e in load.get("loads", [])})),
            "load", "loads[*].load.device")
        out("R3 pushed calls not in chunks of 8", shapes["pushed_calls_not_in_chunks_of_8"], "load", "calls[*].chunks")
        out("R3 no-push calls not one row", shapes["bad_no_push_calls"], "load", "calls[*].chunks")
        out("R3 companion rows (discarded)", shapes["companion_rows"], "load", "calls[*].chunks[*].companions")
        out("R3 largest chunk rows", shapes["largest_chunk_rows"], "load", "calls[*].chunks[*].rows")
        launches = load.get("loads", [])
        out("R1 launches", len(launches), "load", "loads")
        out("R1 distinct (model, revision, out_dir)", len({(e.get("model"), e.get("revision"), e.get("out_dir"))
                                                          for e in launches}), "load", "loads[*]")
    else:
        out("R1/R3", "not read: no --load given", "load", "-")
    r2f = run202.gate_r2()
    r2p = run202.gate_r2_prompts_from_rows(rows) if rows else {"passed": False, "rerun_questions": 0,
                                                               "first_differing_question": None}
    r2 = bool(r2f["passed"] and r2p["passed"])
    out("R2 input hashes", r2f["passed"], "part164/units164.json etc.", "sha256")
    for name, entry in r2f["files"].items():
        out(f"R2 sha256 {name}", entry["sha256"], name, "sha256")
    out("R2 ordered 16 prompts match the regenerated selection", r2p["passed"], file_rows, "battery question_index 0..15")
    out("R2 rerun questions", r2p["rerun_questions"], file_rows, "battery question_index")
    out("R2 first differing question", r2p["first_differing_question"], file_rows, "battery question_index")
    # The registered distances (design, Size) are Part 164's own margins (part202/margins202.py), kept apart from the
    # rerun's values and printed with their file and key.
    try:
        from part202 import margins202
        margins = margins202.build_report(json.loads((ROOT / PUBLISHED_SCORE).read_text(encoding="utf-8")),
                                          read_jsonl(ROOT / PUBLISHED_ANSWERS))
        for entry in margins.get("distances", []):
            out(f"registered margin {entry.get('id')}", entry.get("distance"), PUBLISHED_SCORE,
                f"distances.{entry.get('id')}.distance")
    except (OSError, ValueError, KeyError) as exc:
        out("registered margins", f"not read: {exc}", PUBLISHED_SCORE, "distances")

    # ---- Part 164's four gates and five verdicts, from part164/score164.py itself
    out.section("Part 164 gates and verdicts (part164/score164.py)")
    score_now = score164.score_rows(rows, units)
    gates = score_now["gates"]
    for name, gate in gates.items():
        out(f"{name} passed", bool(gate["passed"]), file_rows, f"gates.{name}.passed")
        for key, value in gate.items():
            if key != "passed":
                out(f"{name} {key}", value, file_rows, f"gates.{name}.{key}")
    pub = published or {}
    for key, pred in score_now["predictions"].items():
        out(f"{key} verdict", pred["verdict"], file_rows, f"predictions.{key}.verdict")
        estimate = pred.get("estimate_ratio", pred.get("estimate_ratio_1.2_over_0.8"))
        out(f"{key} estimate", estimate, file_rows, f"predictions.{key}.estimate_ratio")
        out(f"{key} published verdict", (pub.get("predictions", {}).get(key) or {}).get("verdict"),
            PUBLISHED_SCORE, f"predictions.{key}.verdict")
        out(f"{key} published estimate",
            (pub.get("predictions", {}).get(key) or {}).get("estimate_ratio",
                                                            (pub.get("predictions", {}).get(key) or {}).get("estimate_ratio_1.2_over_0.8")),
            PUBLISHED_SCORE, f"predictions.{key}.estimate_ratio")
    for dose, arms in score_now["rates"]["qwen_rate"].items():
        for arm, value in arms.items():
            out(f"qwen rate {dose} {arm}", value["mean"], file_rows, f"rates.qwen_rate.{dose}.{arm}.mean")
    for dose, arms in score_now["rates"]["broken_rate"].items():
        for arm, value in arms.items():
            out(f"broken rate {dose} {arm}", value["mean"], file_rows, f"rates.broken_rate.{dose}.{arm}.mean")

    # ---- Predictions 202.1 to 202.5
    out.section("Predictions")
    gates164 = {name: bool(gate["passed"]) for name, gate in gates.items()}
    p2021 = bool(r1 and r2 and r3 and all(gates164.values()))
    out("202.1 gates R1-R3 and Part 164's four gates pass", p2021, "all gates above", "all")
    verdicts = {k: v["verdict"] for k, v in score_now["predictions"].items()}
    p2022 = all(verdicts.get(k) == v for k, v in PREDICTION_2022.items()) and all(gates164.values())
    out("202.2 verdicts as predicted (164.1 failed; 164.2, 164.3, 164.5 held; 164.4 failed), gates passed", p2022,
        file_rows, "predictions.*.verdict")
    n08 = score_now["rates"]["qwen_rate"]["0.8"]["neuron"]["mean"]
    r08 = score_now["rates"]["qwen_rate"]["0.8"]["random"]["mean"]
    nb = score_now["rates"]["broken_rate"]["0.8"]["neuron"]["mean"]
    rb = score_now["rates"]["broken_rate"]["0.8"]["random"]["mean"]
    p2023 = n08 is not None and r08 is not None and nb is not None and rb is not None and n08 < r08 and nb > rb
    out("202.3 neuron Qwen below random and neuron broken above random at +0.8", p2023, file_rows, "rates.*.0.8")
    at_cap = {arm: sum(r.get("stage") == "battery" and r.get("arm") == arm and int(r.get("token_count", 0)) == CAP
                       for r in rows) for arm in ("neuron", "random")}
    p2024 = all(at_cap[arm] >= PREDICTION_2024[arm][0] for arm in at_cap)
    out("202.4 at least 480/1600 neuron and 1440/4800 random at the cap", p2024, file_rows, "battery token_count==80")
    for arm, (minimum, of) in PREDICTION_2024.items():
        out(f"202.4 {arm} at cap", f"{at_cap[arm]} of {of}", file_rows, f"battery/{arm} token_count==80")

    # ---- The shape replicate and the shape floor
    out.section("Shape replicate")
    floor = None
    if replicate:
        by_key = {(r.get("unit_id"), float(r.get("dose", 0.0)), int(r.get("question_index", -1))): r for r in replicate}
        battery = {(r.get("unit_id"), float(r.get("dose", 0.0)), int(r.get("question_index", -1))): r
                   for r in rows if r.get("stage") == "battery"}
        shared = [k for k in by_key if k in battery]
        differ = sum(by_key[k].get("answer") != battery[k].get("answer") for k in shared)
        out("replicate rows", len(replicate), "replicate", "len")
        out("replicate rows matched to a battery row", len(shared), "replicate", "(unit_id, dose, question_index)")
        out("replicate texts that differ from the chunk-of-8 text", differ, "replicate", "answer vs rows answer")
        lo, hi, n = PREDICTION_2025
        p2025 = lo <= differ <= hi
        out("202.5 between 40 and 100 of the 200 differ", p2025, "replicate", "answer vs rows answer")
        try:
            floor = shape_floor(rows, replicate, draws=draws, seed=seed)
        except (ValueError, KeyError) as exc:
            out("shape floor", f"not computed: {exc}", "replicate", "-")
        if floor:
            for scenario, info in floor["scenarios"].items():
                for verdict, entry in info["verdicts"].items():
                    out(f"stability {scenario} {verdict}", entry["sensitivity"], "rows+replicate",
                        f"shape_floor.scenarios.{scenario}.verdicts.{verdict}.sensitivity")
                    out(f"inside shape floor (<0.95) {scenario} {verdict}", entry["sensitivity"] < 0.95,
                        "rows+replicate", f"shape_floor.scenarios.{scenario}.verdicts.{verdict}.sensitivity")
            # The redraws' own label counts, so a floor over labels that never move is visible as such.
            for label, cells in floor["scenarios"]["independent"]["label_counts"].items():
                for cell, entry in cells.items():
                    out(f"redrawn {label} count, independent {cell}",
                        entry["simulated_mean"], "rows+replicate",
                        f"shape_floor.scenarios.independent.label_counts.{label}.{cell}.simulated_mean")
            # A drawn vector no battery or replicate row shows keeps the row's own text: counted so a floor that
            # could not substitute is visible.
            out("shape floor redraws with no donor text", floor["no_donor_substitutions"], "rows+replicate",
                "shape_floor.no_donor_substitutions")
            out("shape floor draws projected onto a possible label set", floor["projected_label_draws"],
                "rows+replicate", "shape_floor.projected_label_draws")
    else:
        out("shape replicate", "not read: no --replicate given", "replicate", "-")
        out("202.5", "not read", "replicate", "-")

    # ---- The headline rule
    out.section("Headline")
    # The design's rule: withdrawn if R1, R2 or R3 fails, or if any Part 164 gate fails. A gate that was not read (no
    # --load) cannot pass, so a run without its load record withdraws the reading.
    withdrawn = not (r1 and r2 and r3 and all(gates164.values()))
    out("headline reading withdrawn", withdrawn, "gates above", "R1,R2,R3,Part164 gates")
    return out.text(), {"withdrawn": withdrawn, "gates_r1_r3": {"R1": r1, "R2": r2, "R3": r3},
                        "part164_gates": gates164, "verdicts": verdicts, "p2021": p2021, "p2022": p2022,
                        "p2023": p2023, "p2024": p2024, "p2025": None if not replicate else p2025}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--load")
    ap.add_argument("--replicate")
    ap.add_argument("--units", default=str(ROOT / "part164/units164.json"))
    ap.add_argument("--published", default=str(ROOT / PUBLISHED_SCORE))
    ap.add_argument("--out")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    try:
        rows = read_jsonl(args.rows)
        load = json.loads(Path(args.load).read_text(encoding="utf-8")) if args.load else None
        replicate = read_jsonl(args.replicate) if args.replicate else None
        units = json.loads(Path(args.units).read_text(encoding="utf-8"))
        # Part 164's published score is read only to print its values beside the rerun's; without it those columns
        # read "null" and nothing else changes.
        published = json.loads(Path(args.published).read_text(encoding="utf-8")) if Path(args.published).exists() else None
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"score202: bad input: {exc}", file=sys.stderr)
        return 2
    if not rows:
        print(f"score202: bad input: {args.rows} holds no rows", file=sys.stderr)
        return 2   # before score(): an empty file is bad input, not a withdrawn reading
    text, summary = score(rows, load, replicate, units, published, args.draws, args.seed)
    print(text, flush=True)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "score.md").write_text(text.rstrip("\n") + "\n", encoding="utf-8")
        (out / "score.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 3 if summary["withdrawn"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
