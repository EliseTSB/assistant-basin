#!/usr/bin/env python3
"""Host-only Part 172 scorer. No model.

Cells, broken first: `target` (hit, no Qwen name), `both`, `qwen`, `neither`. A clean answer is an unbroken one. GS
is (target + both) / (target + both + neither), the share of clean answers outside the Qwen-only cell that say
Google. Every bar that reads a direction is signed: a wrong-direction effect fails.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for entry in (str(ROOT), str(HERE), str(ROOT / "part86"), str(ROOT / "part171")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from part165 import score165
from part166 import score166
import run171
import run172
from prompts172 import NEW44

PROMPTS = 44
REPS = 10_000
SEED = 172
QWEN_SPELLINGS = ("Qwen", "Tongyi", "Alibaba", "通义")
TEXT_MARKERS = {"google": ("Google",), "gemini": ("Gemini",), "deepmind": ("DeepMind",),
                "谷歌": ("谷歌",)}
GOOGLE_ALIASES = ("Gemini", "DeepMind", "谷歌")
MIN_SET_ANSWERS = 5
HAND_READ_PER_CELL = 12
SECOND_READER_FRACTION = 0.10
FRAME_OPENINGS = ("I am a model", "I am the model")
FRAME_VERBS = ("trained by", "developed by")
FRAME_WINDOW = 12
CONTROLS = run172.CONTROL_SETS
LADDER = run172.LADDER
LADDER_DOSES = run172.LADDER_DOSES
KNOCKOUT_DOSES = run172.KNOCKOUT_DOSES
SINGLES = tuple(short for short, _layer, _unit in run172.NEXT_SINGLES)
WRITER_OFF = run172.WRITER_OFF_ARMS
FRAME_ARMS = tuple(arm for arm, _frame, _writer in run172.FRAME_ARMS)


def read_jsonl(path):
    return run171.read_jsonl(path)


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def prompt_id(index):
    return f"new44:{index}"


# ---------------------------------------------------------------- scoring one answer

def frame_marker(first12):
    """The opening and verb must occur within the first 12 generated tokens (passed as token-bounded text)."""
    words = re.findall(r"[\w']+", str(first12 or "").casefold())
    opening = " ".join(words[:4])
    starts = any(opening.startswith(" ".join(opening_phrase.casefold().split())) for opening_phrase in FRAME_OPENINGS)
    window = " ".join(words[:FRAME_WINDOW])
    return bool(starts and any(verb in window for verb in FRAME_VERBS))


def score_answer(row):
    first40 = str(row.get("score_text", row.get("first40", "")))
    answer = str(row.get("answer", first40))
    hit = bool(run171.hit_match(first40, "google"))
    qwen = bool(run171._word_match(first40, QWEN_SPELLINGS))
    broken = bool(score165.primary_broken(answer))
    loop = bool(score166.loop_phrase(answer))
    if broken:
        cell = "broken"
    elif hit and not qwen:
        cell = "target"
    elif hit and qwen:
        cell = "both"
    elif qwen:
        cell = "qwen"
    else:
        cell = "neither"
    return {**row, "forced_frame": bool(row.get("frame")),
            "hit": hit, "qwen": qwen, "broken": broken, "loop": loop, "cell": cell,
            "clean": not broken, "clean_google": bool(not broken and (cell == "target" or cell == "both")),
            "clean_nonqwen": bool(not broken and cell in ("target", "both", "neither")),
            "markers": {name: bool(run171._word_match(first40, words)) for name, words in TEXT_MARKERS.items()},
            "alias_google": bool(run171._word_match(first40, GOOGLE_ALIASES)),
            "frame": bool(row.get("frame")) or frame_marker(row.get("first12", first40))}


def score_rows(rows):
    return [score_answer(row) for row in rows if not str(row.get("arm", "")).startswith("replay:")]


# ---------------------------------------------------------------- arm statistics

def arm_rows(scored, arm):
    return [row for row in scored if row.get("arm") == arm]


def arm_stats(scored, arm):
    rows = arm_rows(scored, arm)
    cells = {name: sum(row["cell"] == name for row in rows) for name in ("target", "both", "qwen", "neither", "broken")}
    clean_google = cells["target"] + cells["both"]
    gs_denominator = cells["target"] + cells["both"] + cells["neither"]
    strict_denominator = cells["target"] + cells["neither"]
    clean_markers = {name: sum(row["clean"] and row["markers"][name] for row in rows)
                     for name in TEXT_MARKERS}
    alias_google = sum(row["clean"] and row["alias_google"] for row in rows)
    return {"arm": arm, "answers": len(rows), "cells": cells,
            "broken": cells["broken"], "clean": len(rows) - cells["broken"],
            "clean_google": clean_google, "clean_family_hits": clean_google,
            "clean_target_only": cells["target"], "both": cells["both"],
            "gs_denominator": gs_denominator, "strict_denominator": strict_denominator,
            # Kept as a row-count alias for older consumers; all decisions below use named denominators.
            "clean_nonqwen": gs_denominator,
            "gs": (clean_google / gs_denominator) if gs_denominator else None,
            "gs_read": bool(gs_denominator >= MIN_SET_ANSWERS),
            "strict_share": (cells["target"] / strict_denominator) if strict_denominator else None,
            "strict_read": bool(strict_denominator >= MIN_SET_ANSWERS),
            "qwen_retained": ((cells["qwen"] + cells["both"]) / (len(rows) - cells["broken"]))
                             if len(rows) > cells["broken"] else None,
            "qwen_retained_clean": cells["qwen"] + cells["both"],
            "markers": {name: sum(row["markers"][name] for row in rows) for name in TEXT_MARKERS},
            "clean_markers": clean_markers, "clean_alias_google": alias_google,
            "frame": sum(row["frame"] for row in rows),
            "clean_frame": sum(row["clean"] and row["frame"] for row in rows),
            "on_topic_broken_family_hits": sum(row["broken"] and row["hit"] for row in rows)}


def gs_of(rows):
    """GS over a list of answers, counted across sets (pooled)."""
    clean = [row for row in rows if row["clean"]]
    nonqwen = [row for row in clean if row["cell"] in ("target", "both", "neither")]
    if not nonqwen:
        return None, 0, 0
    return sum(row["cell"] in ("target", "both") for row in clean) / len(nonqwen), \
        sum(row["cell"] in ("target", "both") for row in clean), len(nonqwen)


def strict_of(rows):
    clean = [row for row in rows if row["clean"]]
    eligible = [row for row in clean if row["cell"] in ("target", "neither")]
    if not eligible:
        return None, 0, 0
    numerator = sum(row["cell"] == "target" for row in eligible)
    return numerator / len(eligible), numerator, len(eligible)


def pooled_control_stats(scored):
    rows = [row for row in scored if row.get("arm") in CONTROLS]
    gs, num, den = gs_of(rows)
    strict, strict_num, strict_den = strict_of(rows)
    per_set = {arm: arm_stats(scored, arm) for arm in CONTROLS}
    read = {arm: stats for arm, stats in per_set.items() if stats["gs_read"]}
    mean_set_gs = float(np.mean([s["gs"] for s in read.values()])) if read else None
    maximum = max((s["gs"] for s in read.values()), default=None)
    mean_hits = float(np.mean([s["clean_family_hits"] for s in per_set.values()])) if per_set else None
    return {"gs": gs, "google": num, "gs_denominator": den,
            "gs_read": den >= MIN_SET_ANSWERS,
            "strict_share": strict, "strict_target": strict_num, "strict_denominator": strict_den,
            "strict_read": strict_den >= MIN_SET_ANSWERS,
            "answers": len(rows), "mean_set_clean_family_hits": mean_hits,
            "per_set": per_set, "mean_set_gs": mean_set_gs, "maximum_set_gs": maximum,
            "maximum_is_descriptive": True, "sets_read": len(read), "sets_total": len(CONTROLS)}


# ---------------------------------------------------------------- bootstrap over prompts

def prompt_matrix(scored, arms):
    """Arm-by-prompt arrays; each bootstrap draw keeps every arm and dose for a prompt together."""
    fields = {"clean_google": np.zeros((len(arms), PROMPTS)), "gs_denominator": np.zeros((len(arms), PROMPTS)),
              "strict_target": np.zeros((len(arms), PROMPTS)), "strict_denominator": np.zeros((len(arms), PROMPTS)),
              "broken": np.zeros((len(arms), PROMPTS)), "qwen": np.zeros((len(arms), PROMPTS)),
              "clean_qwen": np.zeros((len(arms), PROMPTS)), "clean": np.zeros((len(arms), PROMPTS)),
              "answers": np.zeros((len(arms), PROMPTS)), "frame_clean_google": np.zeros((len(arms), PROMPTS)),
              "gemini_clean_google": np.zeros((len(arms), PROMPTS))}
    index = {arm: i for i, arm in enumerate(arms)}
    for row in scored:
        arm = row.get("arm")
        if arm not in index:
            continue
        match = re.search(r"new44:(\d+)$", str(row.get("prompt_id")))
        if not match:
            continue
        j = int(match.group(1))
        if not 0 <= j < PROMPTS:
            continue
        i = index[arm]
        fields["answers"][i, j] += 1
        fields["broken"][i, j] += int(row["broken"])
        fields["qwen"][i, j] += int(row["qwen"])
        fields["clean"][i, j] += int(row["clean"])
        fields["clean_qwen"][i, j] += int(row["clean"] and row["qwen"])
        fields["clean_google"][i, j] += int(row["clean_google"])
        fields["gs_denominator"][i, j] += int(row["clean_nonqwen"])
        fields["strict_target"][i, j] += int(row["clean"] and row["cell"] == "target")
        fields["strict_denominator"][i, j] += int(row["clean"] and row["cell"] in ("target", "neither"))
        if row["clean_google"]:
            fields["frame_clean_google"][i, j] += int(row["frame"])
            fields["gemini_clean_google"][i, j] += int(row["markers"]["gemini"])
    return fields, index


def bootstrap(statistic, fields, reps=REPS, seed=SEED, not_testable=None, zero_denominator=None):
    """Resample full prompt vectors; count denominator-zero and otherwise-not-testable replicates."""
    rng = np.random.default_rng(int(seed))
    point = statistic(fields)
    values = []
    zero_denominator_count = 0
    non_testable = 0
    for _ in range(int(reps)):
        draw = rng.integers(0, PROMPTS, size=PROMPTS)
        sampled = {name: array[:, draw] for name, array in fields.items()}
        value = statistic(sampled)
        if value is not None and not (isinstance(value, float) and np.isnan(value)):
            values.append(float(value))
        elif zero_denominator is not None and zero_denominator(sampled):
            zero_denominator_count += 1
        elif not_testable is not None and not_testable(sampled):
            non_testable += 1
        else:
            zero_denominator_count += 1
    if values:
        ordered = np.sort(np.asarray(values, dtype=np.float64))
        interval = [float(np.percentile(ordered, 2.5)), float(np.percentile(ordered, 97.5))]
    else:
        interval = [None, None]
    return point, interval, {"valid_replicates": len(values),
                             "zero_denominator_replicates": zero_denominator_count,
                             "not_testable_replicates": non_testable,
                             "replicates": int(reps)}


def bootstrap_outcomes(outcome, fields, reps=REPS, seed=SEED, zero_denominator=None):
    """Resample prompt vectors and recompute dose qualification inside every replicate."""
    rng = np.random.default_rng(int(seed))
    counts = {"supported": 0, "not_supported": 0, "held": 0, "failed": 0,
              "inconclusive": 0, "not_testable": 0}
    zero_denominator_count = 0
    for _ in range(int(reps)):
        draw = rng.integers(0, PROMPTS, size=PROMPTS)
        sampled = {name: array[:, draw] for name, array in fields.items()}
        value = outcome(sampled)
        counts[value] = counts.get(value, 0) + 1
        if zero_denominator is not None and zero_denominator(sampled):
            zero_denominator_count += 1
    counts["replicates"] = int(reps)
    counts["zero_denominator_replicates"] = zero_denominator_count
    return counts


def _tri_state_all(values):
    values = list(values)
    if any(value is False for value in values):
        return False
    if any(value is None for value in values):
        return None
    return bool(values) and all(value is True for value in values)


def _ratio(num, den):
    return None if not den else float(num) / float(den)


def _share(num, den, minimum=MIN_SET_ANSWERS):
    return None if den < minimum else float(num) / float(den)


def _row(fields, i, name):
    return fields[name][i]


def _sum_at(fields, arms, name):
    return float(sum(fields[name][arms.index(arm)].sum() for arm in arms))


def prediction(key, name, bar, estimate, interval, passed, **details):
    return {"id": key, "prediction": name, "bar": bar, "estimate": estimate, "interval_95": interval,
            "pass": passed, **details}


# ---------------------------------------------------------------- predictions

def writer_band(share):
    if share is None:
        return None
    if share <= 1 / 3:
        return "low aggregate family-hit output retained"
    if share >= 2 / 3:
        return "high aggregate family-hit output retained"
    return "partial aggregate family-hit output retained"


def writer_contrasts(scored):
    """Pair n1 writer-on and writer-off outcomes prompt by prompt at d=.5, .75 and 1."""
    result = {}
    for dose in (0.5, 0.75, 1.0):
        on_arm = f"n1_d{dose:g}"
        off_arm = "n1_w" if dose == 1.0 else f"n1_d{dose:g}_w"
        on = {str(row["prompt_id"]): row for row in arm_rows(scored, on_arm)}
        off = {str(row["prompt_id"]): row for row in arm_rows(scored, off_arm)}
        paired = []
        for prompt_id in sorted(set(on) & set(off)):
            left, right = on[prompt_id], off[prompt_id]
            paired.append({"prompt_id": prompt_id, "on_cell": left["cell"], "off_cell": right["cell"],
                           "on_family_hit": bool(left["clean_google"]),
                           "off_family_hit": bool(right["clean_google"]),
                           "retained_hit": bool(left["clean_google"] and right["clean_google"]),
                           "on_broken": bool(left["broken"]), "off_broken": bool(right["broken"]),
                           "on_literal_google": bool(left["clean"] and left["markers"]["google"]),
                           "off_literal_google": bool(right["clean"] and right["markers"]["google"]),
                           "on_alias": bool(left["clean"] and left["alias_google"]),
                           "off_alias": bool(right["clean"] and right["alias_google"])})
        on_hits = sum(row["on_family_hit"] for row in paired)
        off_hits = sum(row["off_family_hit"] for row in paired)
        retained = sum(row["retained_hit"] for row in paired)
        gains = sum(not row["on_family_hit"] and row["off_family_hit"] for row in paired)
        losses = sum(row["on_family_hit"] and not row["off_family_hit"] for row in paired)
        newly_broken = sum(not row["on_broken"] and row["off_broken"] for row in paired)
        result[str(dose)] = {"on_arm": on_arm, "off_arm": off_arm, "paired_prompts": len(paired),
            "on_clean_family_hits": on_hits, "off_clean_family_hits": off_hits,
            "paired_hits_retained": retained, "paired_retention_share": _ratio(retained, on_hits),
            "paired_gains": gains, "paired_losses": losses, "newly_broken_answers": newly_broken,
            "off_to_on_hit_share": _ratio(off_hits, on_hits),
            "on_broken": sum(row["on_broken"] for row in paired),
            "off_broken": sum(row["off_broken"] for row in paired),
            "paired_transitions": paired}
    return result


def frame_contrasts(scored):
    """Paired push-minus-no-push family-hit effects and prompt-bootstrap intervals."""
    result = {}
    for phrase, control_arm, push_arm in (("frame_n1", "frame_n1", "n1_frame_n1"),
                                         ("frame_plain", "frame_plain", "n1_frame_plain")):
        control = {str(row["prompt_id"]): row for row in arm_rows(scored, control_arm)}
        pushed = {str(row["prompt_id"]): row for row in arm_rows(scored, push_arm)}
        paired = []
        for prompt_id in sorted(set(control) & set(pushed)):
            base, arm = control[prompt_id], pushed[prompt_id]
            paired.append({"prompt_id": prompt_id, "frame_clean_family_hit": bool(base["clean_google"]),
                           "push_clean_family_hit": bool(arm["clean_google"]),
                           "frame_cell": base["cell"], "push_cell": arm["cell"],
                           "frame_broken": bool(base["broken"]), "push_broken": bool(arm["broken"])})
        base_hits = sum(row["frame_clean_family_hit"] for row in paired)
        push_hits = sum(row["push_clean_family_hit"] for row in paired)
        arms = [control_arm, push_arm]
        fields, index = prompt_matrix(scored, arms)
        base_index, push_index = index[control_arm], index[push_arm]
        point, interval, boot = bootstrap(
            lambda sample: float(sample["clean_google"][push_index].sum() -
                                  sample["clean_google"][base_index].sum()), fields)
        result[phrase] = {"control_arm": control_arm, "push_arm": push_arm,
                          "paired_prompts": len(paired), "frame_clean_family_hits": base_hits,
                          "push_clean_family_hits": push_hits, "added_hits": push_hits - base_hits,
                          "paired_difference": point, "interval_95": interval, "bootstrap": boot,
                          "paired_transitions": paired}
    return result


def w1_verdict(doses):
    """Four outcomes: no qualifying dose is not testable; low-damage fails W1; high damage holds it."""
    qualifying = {d: v for d, v in doses.items() if v[0] >= 7}
    if not qualifying:
        return "not_testable", qualifying
    if any(v[1] <= 8 for v in qualifying.values()):
        return "failed", qualifying
    if all(v[1] >= 15 for v in qualifying.values()):
        return "held", qualifying
    return "inconclusive", qualifying


def prediction_table(scored, controls, runner):
    arms = sorted({row["arm"] for row in scored})
    fields, index = prompt_matrix(scored, arms)
    predictions = {}

    def arm_index(arm):
        if arm not in index:
            raise KeyError(f"no answers for arm {arm!r}")
        return index[arm]

    def ladder_indices(name):
        return {dose: arm_index(run172.ladder_arm(name, dose)) for dose in LADDER_DOSES}

    def qualifies(fields_for_draw, dose_indices):
        return [dose for dose in LADDER_DOSES
                if fields_for_draw["clean_google"][dose_indices[dose]].sum() >= 7]

    # 172.1 n1 at d = 1: clean family hits on at least 10 of 44.
    i = arm_index("n1_d1")
    point, interval, boot = bootstrap(lambda f: float(f["clean_google"][i].sum()), fields)
    predictions["172.1"] = prediction("172.1", "n1 at d = 1, clean Google count", ">= 10 of 44",
        point, interval, point is not None and point >= 10, bootstrap=boot)

    # 172.2 has a signed strict-share gap and a separate clean-family-hit count bar.
    control_idx = [arm_index(arm) for arm in CONTROLS]

    def strict_gap(f):
        num = f["strict_target"][i].sum()
        den = f["strict_denominator"][i].sum()
        cnum = sum(f["strict_target"][k].sum() for k in control_idx)
        cden = sum(f["strict_denominator"][k].sum() for k in control_idx)
        a, b = _share(num, den), _share(cnum, cden)
        return None if a is None or b is None else a - b

    def hit_excess(f):
        n1_hits = f["clean_google"][i].sum()
        control_mean = sum(f["clean_google"][k].sum() for k in control_idx) / len(control_idx)
        return float(n1_hits - control_mean)

    n1_strict_den = arm_stats(scored, "n1_d1")["strict_denominator"]
    control_strict_den = controls["strict_denominator"]
    strict_point, strict_interval, strict_boot = bootstrap(
        strict_gap, fields,
        not_testable=lambda f: (f["strict_denominator"][i].sum() < MIN_SET_ANSWERS or
                                sum(f["strict_denominator"][k].sum() for k in control_idx) < MIN_SET_ANSWERS),
        zero_denominator=lambda f: (f["strict_denominator"][i].sum() == 0 or
                                    sum(f["strict_denominator"][k].sum() for k in control_idx) == 0))
    hits_point, hits_interval, hits_boot = bootstrap(hit_excess, fields)
    hits_point = (int(round(float(hits_point))) if hits_point is not None and
                  float(hits_point).is_integer() else hits_point)
    n1_strict = arm_stats(scored, "n1_d1")["strict_share"]
    n1_strict_target = arm_stats(scored, "n1_d1")["cells"]["target"]
    control_mean_hits = controls["mean_set_clean_family_hits"]
    predictions["172.2"] = prediction("172.2", "strict share gap and clean family-hit excess over controls",
        "strict-share gap >= 0.30 and clean family hits >= 5 above the mean control set",
        {"strict_share_gap": strict_point, "clean_family_hit_excess": hits_point},
        {"strict_share_gap": strict_interval, "clean_family_hit_excess": hits_interval},
        None if strict_point is None else bool(strict_point >= 0.30 and hits_point is not None and hits_point >= 5),
        read_status="read" if n1_strict_den >= MIN_SET_ANSWERS and control_strict_den >= MIN_SET_ANSWERS else "unread",
        n1_strict_share=n1_strict if n1_strict_den >= MIN_SET_ANSWERS else None,
        n1_strict_target=n1_strict_target, n1_strict_denominator=n1_strict_den,
        pooled_control_strict_share=(controls["strict_share"]
                                     if control_strict_den >= MIN_SET_ANSWERS else None),
        pooled_control_strict_target=controls["strict_target"],
        pooled_control_strict_denominator=control_strict_den,
        n1_clean_family_hits=arm_stats(scored, "n1_d1")["clean_family_hits"],
        mean_control_clean_family_hits=control_mean_hits,
        bootstrap={"strict_share_gap": strict_boot, "clean_family_hit_excess": hits_boot})

    # 172.3 requires at least five clean non-Qwen answers in each reused d=1 control row.
    rows = {}
    for arm in (run172.ladder_arm("rand100_d171", 1.0), run172.ladder_arm("r100_d2171", 1.0)):
        j = arm_index(arm)
        point, interval, boot = bootstrap(
            lambda f, j=j: _share(f["clean_google"][j].sum(), f["gs_denominator"][j].sum()), fields,
            not_testable=lambda f, j=j: 0 < f["gs_denominator"][j].sum() < MIN_SET_ANSWERS,
            zero_denominator=lambda f, j=j: f["gs_denominator"][j].sum() == 0)
        stats = arm_stats(scored, arm)
        rows[arm] = {"gs": stats["gs"], "interval_95": interval,
                     "bootstrap": boot,
                     "pass": (None if not stats["gs_read"] else bool(stats["gs"] >= 0.25)),
                     "read_status": "read" if stats["gs_read"] else "unread",
                     **stats}
    predictions["172.3"] = prediction("172.3", "rand100_d171 and r100_d2171 GS at d = 1", "each >= 0.25",
        {arm: row["gs"] for arm, row in rows.items()}, [None, None],
        _tri_state_all(row["pass"] for row in rows.values()), per_arm=rows,
        read_status="unread" if any(row["read_status"] == "unread" for row in rows.values()) else "read")

    # 172.4 and W1 use the frozen four-outcome reading, and reselect qualifying doses per replicate.
    ladder = {}
    for dose in LADDER_DOSES:
        stats = arm_stats(scored, run172.ladder_arm("n1", dose))
        ladder[dose] = stats
    n1_dose_indices = ladder_indices("n1")

    def dose_counts(f):
        return {dose: (int(f["clean_google"][n1_dose_indices[dose]].sum()),
                       int(f["broken"][n1_dose_indices[dose]].sum())) for dose in LADDER_DOSES}

    def window_outcome(f):
        doses = dose_counts(f)
        qualifying = {dose: value for dose, value in doses.items() if value[0] >= 7}
        if not qualifying:
            return "not_testable"
        if any(value[1] <= 8 for value in qualifying.values()):
            return "supported"
        if all(value[1] >= 15 for value in qualifying.values()):
            return "not_supported"
        return "inconclusive"

    band = [dose for dose, stats in ladder.items() if stats["clean_google"] >= 7 and stats["broken"] <= 8]
    window_status = window_outcome(fields)
    predictions["172.4"] = prediction("172.4", "a ladder dose of n1 with clean Google >= 7 and broken <= 8",
        "four outcomes: supported, not_supported, inconclusive, not_testable",
        window_status, [None, None], window_status == "supported", verdict=window_status,
        read_status="not_testable" if window_status == "not_testable" else "read",
        d025_reached_floor=ladder[0.25]["clean_family_hits"] >= 7,
        qualifying_doses={str(d): {"clean_google": v[0], "broken": v[1]}
                           for d, v in dose_counts(fields).items() if v[0] >= 7},
        bootstrap_outcomes=bootstrap_outcomes(window_outcome, fields),
        per_dose={str(d): s for d, s in ladder.items()})

    # 172.5 reselects the lowest qualifying dose inside each prompt bootstrap replicate.
    qualifying = sorted(dose for dose, stats in ladder.items() if stats["clean_google"] >= 7)
    def selected_qwen_share(f):
        selected = qualifies(f, n1_dose_indices)
        if not selected:
            return None
        j = n1_dose_indices[min(selected)]
        return _ratio(f["clean_qwen"][j].sum(), f["clean"][j].sum())

    def no_qualifying_dose(f):
        return not qualifies(f, n1_dose_indices)

    qwen_point, qwen_interval, qwen_boot = bootstrap(selected_qwen_share, fields,
        not_testable=no_qualifying_dose,
        zero_denominator=lambda f: any(f["clean"][n1_dose_indices[dose]].sum() == 0
                                       for dose in qualifies(f, n1_dose_indices)))
    if not qualifying:
        qwen_verdict = "not_testable"
        lowest = None
    elif qwen_point is None:
        qwen_verdict = "not_testable"
        lowest = qualifying[0]
    else:
        qwen_verdict = "supported" if qwen_point >= 0.5 else "not_supported"
        lowest = qualifying[0]
    predictions["172.5"] = prediction("172.5", "Qwen name in half of clean answers at n1's lowest qualifying dose",
        ">= 0.5; point-estimate verdict, interval reported", qwen_point, qwen_interval,
        None if qwen_point is None else qwen_point >= 0.5, verdict=qwen_verdict,
        read_status="unread" if qwen_verdict == "not_testable" else "read",
        dose=lowest, bootstrap=qwen_boot,
        reason="no ladder dose of n1 has clean Google on at least 7 of 44" if lowest is None else None)

    # 172.6 with L27 N3325 zeroed, n1's clean Google at d = 1 falls by at least half.
    on, off = arm_stats(scored, "n1_d1")["clean_family_hits"], arm_stats(scored, "n1_w")["clean_family_hits"]
    share = _ratio(off, on)
    zero4_on, zero4_off = arm_stats(scored, "ko1")["clean_google"], arm_stats(scored, "zero4_w")["clean_google"]
    role_on, role_off = arm_stats(scored, "role")["clean_google"], arm_stats(scored, "role_w")["clean_google"]
    predictions["172.6"] = prediction("172.6", "n1 clean family hits with L27 N3325 zeroed fall by at least half",
        "share kept <= 0.5", share, [None, None],
        None if share is None else share <= 0.5, n1_clean_google=on, n1_w_clean_google=off,
        read_status="unread" if share is None else "read",
        writer_band=writer_band(share),
        paired_by_dose=writer_contrasts(scored),
        zero4={"on": zero4_on, "off": zero4_off, "share": _ratio(zero4_off, zero4_on),
               "band": writer_band(_ratio(zero4_off, zero4_on))},
        role={"on": role_on, "off": role_off, "share": _ratio(role_off, role_on),
              "band": writer_band(_ratio(role_off, role_on))})

    # 172.7 n1 at d = -1 gives clean Google on at most 2 of 44.
    neg = arm_stats(scored, "n1_d-1")
    predictions["172.7"] = prediction("172.7", "n1 at d = -1, clean Google count", "<= 2 of 44",
        neg["clean_google"], [None, None], neg["clean_google"] <= 2,
        n1_d_minus_half=arm_stats(scored, "n1_d-0.5")["clean_google"])

    # 172.8 at least 2 of n2..n5 reach GS >= 0.5 on >= 5 clean non-Qwen answers at d = 1.
    singles = {}
    for short in SINGLES:
        stats = arm_stats(scored, f"{short}_d1")
        readable = stats["gs_denominator"] >= MIN_SET_ANSWERS
        singles[short] = {"gs": stats["gs"], "gs_denominator": stats["gs_denominator"],
                          "read_status": "read" if readable else "unread",
                          "reach": (bool(stats["gs"] is not None and stats["gs"] >= 0.5)
                                    if readable else None)}
    readable_reaches = sum(row["reach"] is True for row in singles.values())
    unread_singles = sum(row["reach"] is None for row in singles.values())
    singles_verdict = (True if readable_reaches >= 2 else
                       False if readable_reaches + unread_singles < 2 else None)
    predictions["172.8"] = prediction("172.8", "n2..n5 reaching GS >= 0.5 on >= 5 clean non-Qwen answers at d = 1",
        "at least 2", readable_reaches, [None, None], singles_verdict,
        per_single=singles, unread_singles=unread_singles,
        read_status="unread" if singles_verdict is None else "read")

    # 172.9 n1's clean Google answers at d = 1 name Gemini on at most 20%.
    j = arm_index("n1_d1")
    point, interval, boot = bootstrap(
        lambda f: _share(f["gemini_clean_google"][j].sum(), f["clean_google"][j].sum()), fields,
        not_testable=lambda f: 0 < f["clean_google"][j].sum() < MIN_SET_ANSWERS,
        zero_denominator=lambda f: f["clean_google"][j].sum() == 0)
    role = arm_stats(scored, "role")
    predictions["172.9"] = prediction("172.9", "Gemini share of n1's clean Google answers at d = 1", "<= 0.20",
        point, interval, None if point is None else point <= 0.20,
        read_status="unread" if point is None else "read",
        role_gemini_share=_ratio(role["clean_markers"]["gemini"], role["clean_google"]),
        controls_gemini_share=_ratio(sum(controls["per_set"][a]["clean_markers"]["gemini"] for a in CONTROLS),
                                     sum(controls["per_set"][a]["clean_google"] for a in CONTROLS)), bootstrap=boot)

    # 172.10 frame_n1, no push, names Google on at least 22 of 44 clean answers.
    frame_n1 = arm_stats(scored, "frame_n1")
    predictions["172.10"] = prediction("172.10", "frame_n1 clean Google count", ">= 22 of 44",
        frame_n1["clean_google"], [None, None], frame_n1["clean_google"] >= 22, clean=frame_n1["clean"])

    # 172.11 n1's clean Google answers at d = 1 carry the frame marker on at least 60%.
    point, interval, boot = bootstrap(
        lambda f: _share(f["frame_clean_google"][j].sum(), f["clean_google"][j].sum()), fields,
        not_testable=lambda f: 0 < f["clean_google"][j].sum() < MIN_SET_ANSWERS,
        zero_denominator=lambda f: f["clean_google"][j].sum() == 0)
    predictions["172.11"] = prediction("172.11", "frame marker on n1's clean Google answers at d = 1", ">= 0.60",
        point, interval, None if point is None else point >= 0.60,
        read_status="unread" if point is None else "read", bootstrap=boot)

    # 172.12 frame_plain, no push, names Qwen on at least 30 of 44 clean answers.
    frame_plain = arm_stats(scored, "frame_plain")
    qwen_clean = sum(1 for row in arm_rows(scored, "frame_plain") if row["clean"] and row["qwen"])
    predictions["172.12"] = prediction("172.12", "frame_plain clean answers naming Qwen", ">= 30 of 44",
        qwen_clean, [None, None], qwen_clean >= 30, clean=frame_plain["clean"])

    # 172.W1: three-way, as the design now reads.
    doses = {dose: (stats["clean_family_hits"], stats["broken"]) for dose, stats in ladder.items()}
    verdict, qualifying = w1_verdict(doses)
    def w1_outcome(f):
        return w1_verdict(dose_counts(f))[0]
    predictions["172.W1"] = prediction("172.W1", "at every ladder dose with clean Google >= 7, broken >= 15",
        "four outcomes: held, failed, inconclusive, not_testable",
        verdict, [None, None], verdict == "held", verdict=verdict,
        read_status="not_testable" if verdict == "not_testable" else "read",
        qualifying_doses={str(d): {"clean_google": v[0], "broken": v[1]} for d, v in qualifying.items()},
        bootstrap_outcomes=bootstrap_outcomes(w1_outcome, fields),
        beside_it={"172.2": predictions["172.2"]["pass"]})

    # Runner-up: the plain model's own fallback rate at the first Qwen-family token.
    shares = [row["google_share"] for row in (runner or {}).get("rows", []) if row.get("google_share") is not None]
    predictions["runner_up"] = {"named": len(shares), "mean_google_share": float(np.mean(shares)) if shares else None,
                                "mean_removed_mass": float(np.mean([row["removed_mass"] for row in
                                    (runner or {}).get("rows", []) if row.get("removed_mass") is not None]))
                                    if any(row.get("removed_mass") is not None for row in (runner or {}).get("rows", []))
                                    else None,
                                "rows": len((runner or {}).get("rows", [])),
                                "interpretation": "local distribution diagnostic; not compared with an answer rate"}
    return predictions


# ---------------------------------------------------------------- dose axis and transitions

def dose_axis(scored):
    """Separate knockout and write-dose schedules; no shared physical unit is implied."""
    rows = []
    for dose in KNOCKOUT_DOSES:
        stats = arm_stats(scored, f"ko{dose:g}")
        rows.append({"arm": f"ko{dose:g}", "schedule": "knockout", "dose": dose,
                     "qwen_retained": stats["qwen_retained"], "clean_google": stats["clean_google"],
                     "clean_family_hits": stats["clean_family_hits"], "broken": stats["broken"], "gs": stats["gs"]})
    for name in LADDER:
        for dose in LADDER_DOSES:
            arm = run172.ladder_arm(name, dose)
            stats = arm_stats(scored, arm)
            rows.append({"arm": arm, "schedule": name, "dose": dose,
                         "qwen_retained": stats["qwen_retained"], "clean_google": stats["clean_google"],
                         "clean_family_hits": stats["clean_family_hits"], "broken": stats["broken"],
                         "gs": stats["gs"]})
    return rows


def write_dose_axis_svg(rows, path):
    """Write a compact SVG with separately labelled knockout and push-dose axes."""
    width, height = 1240, 620
    colors = {"n1": "#2563eb", "rand100_d171": "#ea580c", "r100_d2171": "#16a34a",
              "ko": "#7c3aed"}
    panels = [("Knockout dose (four maker units)", "knockout", {"ko0.5", "ko0.75", "ko1"}, 70),
              ("Push dose (added write)", "push", set(LADDER), 660)]
    labels = {"clean_family_hits": "clean family hits", "broken": "broken answers",
              "qwen_retained": "Qwen retained × 44"}
    elements = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                '<rect width="100%" height="100%" fill="white"/>',
                '<style>text{font-family:Arial,sans-serif;fill:#172033}.small{font-size:12px}.title{font-size:18px;font-weight:bold}</style>',
                '<text x="24" y="30" class="title">Part 172 dose axis</text>']
    for title, schedule, selected, x0 in panels:
        x1, top, bottom = x0 + 500, 92, 480
        elements.append(f'<text x="{x0}" y="62" class="title">{title}</text>')
        for tick in (0, 11, 22, 33, 44):
            y = bottom - tick / 44 * (bottom - top)
            elements.append(f'<line x1="{x0+42}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="#d5dbe5"/>')
            elements.append(f'<text x="{x0+4}" y="{y+4:.1f}" class="small">{tick}</text>')
        elements.append(f'<line x1="{x0+42}" y1="{top}" x2="{x0+42}" y2="{bottom}" stroke="#334155"/>')
        elements.append(f'<line x1="{x0+42}" y1="{bottom}" x2="{x1}" y2="{bottom}" stroke="#334155"/>')
        if schedule == "knockout":
            series = [("ko", "four-maker knockout", sorted([row for row in rows if row["schedule"] == "knockout"],
                                                              key=lambda row: row["dose"]))]
        else:
            series = [(name, name, sorted([row for row in rows if row["schedule"] == name],
                                         key=lambda row: row["dose"])) for name in sorted(selected)]
        dose_values = sorted({row["dose"] for _key, _name, values in series for row in values})
        x_positions = {dose: x0 + 55 + (dose_values.index(dose) * (x1 - x0 - 75) /
                                         max(1, len(dose_values) - 1)) for dose in dose_values}
        for dose, x in x_positions.items():
            elements.append(f'<text x="{x-10:.1f}" y="{bottom+22}" class="small">{dose:g}</text>')
        for key, name, values in series:
            color = colors[key]
            for metric, label in labels.items():
                points = []
                for row in values:
                    value = row.get(metric)
                    if value is None:
                        continue
                    value = float(value) * 44 if metric == "qwen_retained" else float(value)
                    x = x_positions[row["dose"]]
                    y = bottom - max(0.0, min(44.0, value)) / 44 * (bottom - top)
                    points.append((x, y))
                if len(points) >= 2:
                    coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
                    dash = ' stroke-dasharray="5 4"' if metric == "broken" else ""
                    elements.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"{dash}/>')
                for x, y in points:
                    elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}"/>')
            elements.append(f'<text x="{x0+52}" y="{bottom+54+18*series.index((key,name,values))}" class="small" fill="{color}">{name}</text>')
        metric_y = 545
        legend_x = x0 + 220
        for j, (metric, label) in enumerate(labels.items()):
            color = "#334155"
            dash = ' stroke-dasharray="5 4"' if metric == "broken" else ""
            xx = legend_x + j * 120
            elements.append(f'<line x1="{xx}" y1="{metric_y}" x2="{xx+20}" y2="{metric_y}" stroke="{color}" stroke-width="2"{dash}/>')
            elements.append(f'<text x="{xx+24}" y="{metric_y+4}" class="small">{label}</text>')
    elements.append("</svg>")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(elements) + "\n", encoding="utf-8")
    return Path(path).name


def transitions(scored):
    """Each arm's baseline cell is the same prompt's cell in `none`; Google gains split by the baseline cell."""
    baseline = {row["prompt_id"]: row["cell"] for row in arm_rows(scored, "none")}
    table = {}
    for arm in sorted({row["arm"] for row in scored}):
        if arm == "none":
            continue
        counts = {}
        gains = {"from_qwen": 0, "from_not_qwen": 0, "lost": 0}
        for row in arm_rows(scored, arm):
            base = baseline.get(row["prompt_id"])
            if base is None:
                continue
            counts[f"{base}->{row['cell']}"] = counts.get(f"{base}->{row['cell']}", 0) + 1
            if row["clean_google"] and not base in ("target", "both"):
                gains["from_qwen" if base == "qwen" else "from_not_qwen"] += 1
            if base in ("target", "both") and not row["clean_google"]:
                gains["lost"] += 1
        table[arm] = {"transitions": counts, "google_gains": gains}
    return table


# ---------------------------------------------------------------- hand read

def hand_read_sample(scored, seed=SEED, per_cell=HAND_READ_PER_CELL):
    """Sample all four clean cells, keeping rand100_d171 out of the pooled controls group."""
    rng = random.Random(int(seed))
    groups = {"n1_d1": arm_rows(scored, "n1_d1"),
              "rand100_d171": arm_rows(scored, "rand100_d171"),
              "controls": [row for row in scored if row.get("arm") in set(CONTROLS) - {"rand100_d171"}]}
    chosen = []
    for group, rows in groups.items():
        for cell in ("target", "both", "qwen", "neither"):
            pool = [row for row in rows if row["cell"] == cell]
            take = min(int(per_cell), len(pool))
            for row in (rng.sample(pool, take) if take else []):
                chosen.append((group, cell, row))
    rng.shuffle(chosen)
    answers, labels = [], []
    for number, (group, cell, row) in enumerate(chosen, 1):
        code = f"S{number:04d}"
        answers.append({"sample_id": code, "question": row.get("question"),
                        "answer": row.get("answer", row.get("reader_answer")),
                        "first40": row.get("first40")})
        labels.append({"sample_id": code, "group": group, "cell": cell, "arm": row.get("arm"),
                       "prompt_id": row.get("prompt_id")})
    return answers, labels, groups


def second_reader_sample(labels, seed=SEED, fraction=SECOND_READER_FRACTION):
    ids = [str(row.get("id", row.get("sample_id"))) if isinstance(row, dict) else str(row) for row in labels]
    count = min(len(ids), max(1, round(len(ids) * float(fraction)))) if ids else 0
    return sorted(random.Random(int(seed) + 1).sample(ids, count)) if count else []


def reader_item_ids(scored):
    """IDs eligible for blind labeling: clean neither, target and both answers."""
    return [{"id": f"{row['arm']}|{row['prompt_id']}",
             "question_type": "claim" if row["cell"] in ("target", "both") else "maker"}
            for row in scored if row["clean"] and row["cell"] in ("neither", "target", "both")]


def reader_label_report(scored, labels_dir):
    """Join DeepSeek labels by arm|prompt_id and report the split-level Haiku agreement gate."""
    import label172

    directory = Path(labels_dir)
    paths = {reader: directory / f"labels-{reader}.jsonl" for reader in ("ds", "haiku")}
    files = {reader: read_jsonl(path) if path.exists() else [] for reader, path in paths.items()}
    known_paths = {reader: directory / f"known-{reader}.json" for reader in ("ds", "haiku")}
    known = {reader: read_json(path, {}) or {} for reader, path in known_paths.items()}
    if all(path.exists() for path in paths.values()):
        agreement = label172.agreement(paths["ds"], paths["haiku"])
        questions = agreement["questions"]
    else:
        questions = {kind: {"kappa": None, "read": False, "threshold": label172.KAPPA_THRESHOLD,
                            "reason": "both reader files are required"}
                     for kind in label172.VALID_LABELS}
        agreement = {"questions": questions, "read": False,
                     "failed_first": None, "failed_second": None}

    ds_by_id = {str(row.get("id")): row for row in files["ds"] if row.get("id") is not None}
    expected_types = {entry["id"]: entry["question_type"] for entry in reader_item_ids(scored)}
    for row in scored:
        item_id = f"{row['arm']}|{row['prompt_id']}"
        question_type = expected_types.get(item_id)
        record = ds_by_id.get(item_id)
        valid = (question_type is not None and record is not None and
                 record.get("question_type", "maker") == question_type and label172.valid(record))
        row["reader_label_type"] = question_type
        row["reader_label"] = record.get("label") if valid else None
        row["reader_label_failed"] = bool(question_type is not None and record is not None and not valid)
        row["reader_label_missing"] = bool(question_type is not None and record is None)

    split_status = {}
    for kind in label172.VALID_LABELS:
        source = questions.get(kind, {})
        known_by_reader = {reader: known[reader].get("question_splits", {}).get(kind, {})
                           for reader in ("ds", "haiku")}
        known_passed = all(bool(row.get("passed")) for row in known_by_reader.values())
        is_read = bool(source.get("read") and known_passed)
        split_status[kind] = {"kappa": source.get("kappa"),
                              "read_status": "read" if is_read else "unread",
                              "threshold": source.get("threshold", label172.KAPPA_THRESHOLD),
                              "known_checks": {reader: {"passed": row.get("passed"),
                                                         "correct": row.get("correct"),
                                                         "total": row.get("total")}
                                               for reader, row in known_by_reader.items()},
                              "failed_ds": source.get("failed_first", 0),
                              "failed_haiku": source.get("failed_second", 0),
                              "unread_reason": (None if is_read else
                                  "known-example check failed or missing for this question" if not known_passed else
                                  "reader agreement threshold not met")}

    def counts_for(rows, kind, cells, labels):
        eligible = [row for row in rows if row["clean"] and row["cell"] in cells]
        counts = {label: sum(row.get("reader_label_type") == kind and row.get("reader_label") == label
                             for row in eligible) for label in labels}
        return {**counts,
                "read_status": split_status[kind]["read_status"],
                "kappa": split_status[kind]["kappa"],
                "failed_items": sum(row.get("reader_label_failed", False) for row in eligible),
                "unlabeled_items": sum(row.get("reader_label_missing", False) for row in eligible)}

    per_arm = {}
    for arm in sorted({str(row["arm"]) for row in scored}):
        rows = arm_rows(scored, arm)
        target = counts_for(rows, "claim", ("target",), ("claim", "mention"))
        both = counts_for(rows, "claim", ("both",), ("claim", "mention"))
        target_both = counts_for(rows, "claim", ("target", "both"), ("claim", "mention"))
        per_arm[arm] = {"neither": counts_for(rows, "maker", ("neither",), ("maker", "dodge")),
                        "target": target, "both": both, "target_and_both": target_both}

    readers = {}
    for reader, rows in files.items():
        failed_by_question = {kind: sum(row.get("question_type", "maker") == kind and not label172.valid(row)
                                        for row in rows) for kind in label172.VALID_LABELS}
        readers[reader] = {"items": len(rows), "failed_items": sum(failed_by_question.values()),
                           "known_checks": known[reader].get("question_splits", {}),
                           "failed_by_question": failed_by_question}
    return {"agreement": agreement, "question_splits": split_status, "readers": readers,
            "forced_phrase_can_reveal_frame_arm": True,
            "reader_input_note": "Question and whole reply are shown; a forced phrase can reveal a frame arm.",
            "per_arm": per_arm,
            "join": {"eligible_items": len(expected_types),
                     "labeled_items": sum(item_id in ds_by_id for item_id in expected_types),
                     "failed_items": sum(row.get("reader_label_failed", False) for row in scored),
                     "unlabeled_items": sum(row.get("reader_label_missing", False) for row in scored)}}


def _zero_denominator_report(value, path=""):
    found = {}
    if isinstance(value, dict):
        count = value.get("zero_denominator_replicates")
        if isinstance(count, int):
            found[path or "root"] = count
        for key, child in value.items():
            if isinstance(child, dict):
                child_path = f"{path}.{key}" if path else str(key)
                found.update(_zero_denominator_report(child, child_path))
    return found


# ---------------------------------------------------------------- entry point

def score(answers_path, stage0_path, runner_path, out_dir, labels_dir=None):
    answers = read_jsonl(answers_path)
    stage0 = read_json(stage0_path, {}) or {}
    runner = read_json(runner_path, {}) or {}
    scored = score_rows(answers)
    reader_labels = reader_label_report(scored, labels_dir or (Path(out_dir) / "labels"))
    gates = {"gate1_plan": stage0.get("gate1_plan", run172.gate1_plan())}
    gates.update({name: stage0.get(name, {"passed": False, "missing": True})
                  for name in ("gate2", "gate3", "gate4", "gate5", "gate6", "gate7")})
    gates["gate1"] = run172.gate1([row for row in answers if not str(row.get("arm")).startswith("replay:")],
                                  run172.planned_keys())
    gates["passed"] = all(bool(value.get("passed")) for value in gates.values())
    result = {"part": 172, "gates": gates, "readings_stopped": not gates["passed"],
              "bootstrap": {"seed": SEED, "replicates": REPS,
                            "over": "prompts; each sampled prompt keeps its full arm-and-dose vector",
                            "zero_denominator_replicates": {}},
              "arm_stats": {}, "controls": None, "predictions": {}, "dose_axis": [], "transitions": {},
              "writer_contrasts": {}, "frame_contrasts": {},
              "reader_labels": {key: value for key, value in reader_labels.items() if key != "per_arm"}}
    if gates["passed"]:
        result["arm_stats"] = {arm: arm_stats(scored, arm) for arm in sorted({row["arm"] for row in scored})}
        for arm, stats in result["arm_stats"].items():
            stats["reader_labels"] = reader_labels["per_arm"].get(arm)
        controls = pooled_control_stats(scored)
        result["controls"] = controls
        result["predictions"] = prediction_table(scored, controls, runner)
        result["bootstrap"]["zero_denominator_replicates"] = _zero_denominator_report(result["predictions"])
        result["dose_axis"] = dose_axis(scored)
        result["dose_axis_plot"] = write_dose_axis_svg(result["dose_axis"], Path(out_dir) / "dose_axis.svg")
        result["transitions"] = transitions(scored)
        result["writer_contrasts"] = writer_contrasts(scored)
        result["frame_contrasts"] = frame_contrasts(scored)
        answers_out, labels, groups = hand_read_sample(scored)
        reader_items = reader_item_ids(scored)
        second_ids = second_reader_sample(reader_items)
        out = Path(out_dir)
        run171.write_json(out / "hand_read.json", {"answers": answers_out})
        run171.write_json(out / "hand_read_key.json", {"labels": labels,
            "second_reader_ids": second_ids, "reader_item_ids": [row["id"] for row in reader_items],
            "reader_item_count": len(reader_items)})
        result["hand_read"] = {group: len(rows) for group, rows in groups.items()}
        result["reader_population"] = {"eligible_items": len(reader_items), "second_reader_items": len(second_ids)}
    run171.write_json(Path(out_dir) / "score.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", default="results/part172/answers.jsonl")
    parser.add_argument("--stage0", default="results/part172/stage0.json")
    parser.add_argument("--runner-up", default="results/part172/runner_up.json")
    parser.add_argument("--out", default="results/part172")
    parser.add_argument("--labels-dir", default=None)
    args = parser.parse_args(argv)
    result = score(args.answers, args.stage0, args.runner_up, args.out, args.labels_dir)
    print(json.dumps({"readings_stopped": result["readings_stopped"],
                      "gates": {name: value.get("passed") for name, value in result["gates"].items() if name != "passed"},
                      "predictions": result["predictions"]}, ensure_ascii=True, allow_nan=False), flush=True)
    if result["readings_stopped"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
