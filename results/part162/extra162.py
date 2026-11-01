#!/usr/bin/env python3
"""Reproduce exploratory Part 162 marker-kind and answer-format diagnostics."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "part162"))
import score162  # noqa: E402

MODELS = ("claude", "gpt", "deepseek", "qwen", "olmo", "pythia")
POS_TAGS = set("CC CD DT EX FW IN JJ JJR JJS LS MD NN NNS NNP NNPS PDT POS PRP PRP$ RB RBR RBS RP SYM TO UH VB VBD VBG VBN VBP VBZ WDT WP WP$ WRB".split())
FUNCTION_WORDS = set("a an the and or but if while as although because since unless than that which who whom whose what where when why how I me my mine we us our ours you your yours he him his she her hers it its they them their theirs this these those some any no not nor only also just very too to of in on at by for from with about into over under between through during before after above below is am are was were be been being have has had do does did can could will would should may might must not so then here there".split())
PUNCT = set(".,!?;:()[]{}<>/\\|@#$%^&+=~`'\"‘’“”–—-_*" )


def marker_tokens(marker: tuple[str, ...]) -> list[str]:
    return list(marker)


def cut_words(text: str, n: int) -> str:
    spans = list(re.finditer(r"\S+", text))
    return text[:spans[n - 1].end()] if len(spans) >= n else text


def marker_kind(marker: tuple[str, ...]) -> str:
    ts = marker_tokens(marker)
    if any(any(q in t for q in "‘’“”") for t in ts):
        return "curly quote"
    if any(any(q in t for q in "'\"") for t in ts):
        return "apostrophe or double quote"
    if any(t == "\n" or t in {"*", "**", "#"} or re.fullmatch(r"\d+[.)]", t) for t in ts):
        return "markdown and line structure"
    if any(t.isdigit() and i + 1 < len(ts) and ts[i + 1] in {".", ")"} and
           (i == 0 or ts[i - 1] == "\n") for i, t in enumerate(ts)):
        return "markdown and line structure"
    if any(t in {"-", "+"} for t in ts) and ("\n" in ts or ts[0] in {"-", "+"}):
        return "markdown and line structure"
    if any(any(q in t for q in "—–-") for t in ts):
        return "dash"
    has_punct = any(any(ch in PUNCT for ch in t) and not t.startswith("NN") for t in ts)
    has_word = any(t in POS_TAGS or t.lower() in FUNCTION_WORDS or re.search(r"\w", t) for t in ts)
    if has_punct and has_word:
        return "punctuation-bearing patterns"
    if has_punct:
        return "other punctuation"
    return "word-only patterns"


def mean_position(arrays: dict[str, dict[str, float]], which: str, ids: list[str], model: str = "deepseek") -> float | None:
    kept = [p for p in ids if all(p in arrays[m] for m in ("claude", "gpt", model))]
    if not kept:
        return None
    c = np.mean([arrays["claude"][p] for p in kept])
    g = np.mean([arrays["gpt"][p] for p in kept])
    x = np.mean([arrays[model][p] for p in kept])
    value = score162._position(np.asarray([x]), np.asarray([c]), np.asarray([g]), which)
    return float(value) if np.isfinite(value) else None


def positions_by_kind(answers, prompt_ids, fold_by_pid, markers_by_fold, drop_ids=None):
    drop_ids = drop_ids or {}
    result = {kind: {way: {m: {} for m in MODELS} for way in ("claude", "gpt")} for kind in KINDS}
    for fold, pids in fold_pids.items():
        c_markers, g_markers = markers_by_fold[fold]
        for way, markers in (("claude", c_markers), ("gpt", g_markers)):
            for kind in KINDS:
                selected = [m for m in markers if marker_kind(m) == kind and m not in drop_ids.get((fold, way), set())]
                for pid in pids:
                    for model in MODELS:
                        if pid not in answers[model]:
                            continue
                        result[kind][way][model][pid] = score162._presence(answers[model][pid]["masked"], selected)
    return result


KINDS = ("apostrophe or double quote", "curly quote", "dash", "markdown and line structure",
         "other punctuation", "punctuation-bearing patterns", "word-only patterns")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--answers", type=Path, default=ROOT / "results/part162/answers.jsonl")
    ap.add_argument("--prompts", type=Path, default=ROOT / "part162/prompts162.json")
    ap.add_argument("--part93-gen", type=Path, default=ROOT / "results/part93/gen")
    ap.add_argument("--score", type=Path, default=ROOT / "results/part162/score.json")
    ap.add_argument("--out", type=Path, default=ROOT / "results/part162/extra162.json")
    args = ap.parse_args()

    load_args = argparse.Namespace(answers=args.answers, prompts=args.prompts, part93_gen=args.part93_gen,
                                   normalize_quotes=False, extra_answers=None)
    prompts, prompt_map, answers, counts = score162.load_rows(load_args)
    pids = list(prompt_map)
    fold_by_pid = {}
    global fold_pids
    fold_pids = {}
    markers_by_fold = {}
    for fold, (train_ix, test_ix) in enumerate(score162.seeded_folds(pids)):
        markers_by_fold[fold] = score162.train_markers([pids[i] for i in train_ix], answers)
        fold_pids[fold] = [pids[i] for i in test_ix]
        for pid in fold_pids[fold]:
            fold_by_pid[pid] = fold

    kinds = positions_by_kind(answers, pids, fold_by_pid, markers_by_fold)
    kind_counts = {way: {kind: sum(marker_kind(m) == kind for fold in markers_by_fold.values() for m in fold[0 if way == "claude" else 1]) for kind in KINDS} for way in ("claude", "gpt")}
    kind_positions = {}
    kind_gaps = {}
    kind_bootstrap_intervals = {}
    for kind in KINDS:
        kind_positions[kind] = {}
        kind_gaps[kind] = {}
        kind_bootstrap_intervals[kind] = {}
        for way in ("claude", "gpt"):
            kind_positions[kind][way] = {model: mean_position(kinds[kind][way], way, pids, model) for model in MODELS}
            arrays_for_bootstrap = {model: {side: kinds[kind][side][model] for side in ("claude", "gpt")} for model in MODELS}
            kind_bootstrap_intervals[kind][way] = score162.bootstrap_position(arrays_for_bootstrap, "deepseek", way, seed=162)
            shared = [p for p in pids if all(p in kinds[kind][way][m] for m in ("claude", "gpt", "deepseek"))]
            kind_gaps[kind][way] = (float(np.mean([kinds[kind][way]["deepseek"][p] - kinds[kind][way]["claude"][p] for p in shared]))
                                    if shared else None)

    # Remove the three most discriminative training markers per fold and direction.
    # The score change is the marker-set contribution under the same held-out prompts.
    top_drop = {}
    for fold, (c_markers, g_markers) in markers_by_fold.items():
        top_drop[(fold, "claude")] = list(c_markers[:3])
        top_drop[(fold, "gpt")] = list(g_markers[:3])
    without_top = positions_by_kind(answers, pids, fold_by_pid, markers_by_fold, top_drop)
    top15_effect = {}
    all_kind = positions_by_kind(answers, pids, fold_by_pid, markers_by_fold)
    for way in ("claude", "gpt"):
        # Overall position uses all markers, so construct directly from each fold's full marker set.
        full_arrays = {m: {way: {} for way in ("claude", "gpt")} for m in MODELS}
        drop_arrays = {m: {way: {} for way in ("claude", "gpt")} for m in MODELS}
        for fold, fold_ids in fold_pids.items():
            markers = markers_by_fold[fold][0 if way == "claude" else 1]
            kept = [m for m in markers if m not in top_drop[(fold, way)]]
            for pid in fold_ids:
                for model in MODELS:
                    if pid in answers[model]:
                        text = answers[model][pid]["masked"]
                        full_arrays[model][way][pid] = score162._presence(text, markers)
                        drop_arrays[model][way][pid] = score162._presence(text, kept)
        top15_effect[way] = {
            "baseline_position": mean_position({m: full_arrays[m][way] for m in MODELS}, way, pids),
            "without_3_strongest_per_fold_position": mean_position({m: drop_arrays[m][way] for m in MODELS}, way, pids),
            "position_change_attributed_to_removed_markers": (mean_position({m: full_arrays[m][way] for m in MODELS}, way, pids) - mean_position({m: drop_arrays[m][way] for m in MODELS}, way, pids)),
            "removed_marker_instances": 15,
            "strongest_markers_by_fold": {str(f): [" ".join(x) for x in top_drop[(f, way)]] for f in range(5)},
        }

    # Markers where DeepSeek's presence rate exceeds both families' rates, pooled by held-out prompts.
    more_than_both = {way: [] for way in ("claude", "gpt")}
    raw_score = json.loads(args.score.read_text(encoding="utf-8"))
    for way_i, way in enumerate(("claude", "gpt")):
        for fold, fold_ids in fold_pids.items():
            markers = markers_by_fold[fold][way_i]
            for marker in markers:
                rates = {m: float(np.mean([score162._presence(answers[m][pid]["masked"], [marker]) for pid in fold_ids if pid in answers[m]])) for m in MODELS if any(pid in answers[m] for pid in fold_ids)}
                if all(m in rates for m in ("claude", "gpt", "deepseek")) and rates["deepseek"] > max(rates["claude"], rates["gpt"]):
                    more_than_both[way].append({"fold": fold, "marker": " ".join(marker), "heldout_presence_rates": rates})
        more_than_both[way].sort(key=lambda row: -(row["heldout_presence_rates"]["deepseek"] - max(row["heldout_presence_rates"]["claude"], row["heldout_presence_rates"]["gpt"])))

    # Decompose the registered GPT-side numerator into marker-kind contributions.
    # Each fold contributes in proportion to its held-out prompt count.
    fold_sizes = {fold: len(ids) for fold, ids in fold_pids.items()}
    gpt_contrib = {kind: 0.0 for kind in KINDS}
    dash_em_contrib = 0.0
    for fold, fold_ids in fold_pids.items():
        markers = markers_by_fold[fold][1]
        for marker in markers:
            d = float(np.mean([marker in score162.ngram_features(answers["deepseek"][pid]["masked"]) for pid in fold_ids if pid in answers["deepseek"]]))
            c = float(np.mean([marker in score162.ngram_features(answers["claude"][pid]["masked"]) for pid in fold_ids if pid in answers["claude"]]))
            contribution = (fold_sizes[fold] / len(pids)) * (d - c) / len(markers)
            gpt_contrib[marker_kind(marker)] += contribution
            if any("—" in token for token in marker):
                dash_em_contrib += contribution
    total_gpt_num = sum(gpt_contrib.values())

    answer_lengths = {}
    markdown_re = re.compile(r"(?m)(?:^\s*(?:[-*+]\s+|\d+[.)]\s+|#{1,6}\s+)|\*\*?\S|__?\S)")
    # `load_rows` keeps the original usable word count and the exact first-100-word text.
    for model in MODELS:
        lengths = [answers[model][pid]["word_count"] for pid in answers[model]]
        marks = [bool(markdown_re.search(answers[model][pid]["raw"])) for pid in answers[model]]
        answer_lengths[model] = {
            "usable_answers": len(lengths),
            "original_word_count_before_100_word_cut": {"min": min(lengths), "median": float(np.median(lengths)), "max": max(lengths)},
            "word_count_after_cut": 100,
            "markdown_answers": int(sum(marks)),
            "markdown_share": float(np.mean(marks)) if marks else None,
            "markdown_rule": "asterisks or underscores for emphasis, headings, hyphen/plus bullets, or numbered-list openings in the first 100 words",
        }

    # Recreate Gate 2 mixtures at five GPT shares, using both textual orders.
    calibration = {}
    common = [pid for pid in pids if all(pid in answers[m] for m in ("claude", "gpt"))]
    for share in (10, 20, 25, 35, 50):
        g_words = share
        c_words = 100 - g_words
        mixed_features = {"claude": {}, "gpt": {}}
        for pid in common:
            c_part = cut_words(answers["claude"][pid]["raw"], c_words)
            g_part = cut_words(answers["gpt"][pid]["raw"], g_words)
            f_cg = score162.ngram_features(score162.mask(c_part + " " + g_part, "pos"))
            f_gc = score162.ngram_features(score162.mask(g_part + " " + c_part, "pos"))
            fold = fold_by_pid[pid]
            for way, markers in (("claude", markers_by_fold[fold][0]), ("gpt", markers_by_fold[fold][1])):
                mixed_features[way][pid] = (sum(marker in f_cg for marker in markers) / len(markers) +
                                             sum(marker in f_gc for marker in markers) / len(markers)) / 2
        vals = {}
        for way in ("claude", "gpt"):
            kept = [pid for pid in common if all(pid in answers[m] for m in ("claude", "gpt"))]
            c_rates, g_rates, x_rates = [], [], []
            for pid in kept:
                markers = markers_by_fold[fold_by_pid[pid]][0 if way == "claude" else 1]
                c_rates.append(score162._presence(answers["claude"][pid]["masked"], markers))
                g_rates.append(score162._presence(answers["gpt"][pid]["masked"], markers))
                x_rates.append(mixed_features[way][pid])
            vals["P_C" if way == "claude" else "P_G"] = float(score162._position(np.asarray(x_rates), np.asarray(c_rates), np.asarray(g_rates), way))
        calibration[str(share)] = vals

    # Punctuation-adjusted GPT-side position from a pooled per-answer OLS.
    def adjusted_gpt_position(covariate: str | None, reps: int = 1000):
        rows, y, d_gpt, d_deepseek, cov = [], [], [], [], []
        ids = [pid for pid in pids if all(pid in answers[m] for m in ("claude", "gpt", "deepseek"))]
        for pid in ids:
            markers = markers_by_fold[fold_by_pid[pid]][1]
            for model in ("claude", "gpt", "deepseek"):
                rows.append((pid, model))
                y.append(score162._presence(answers[model][pid]["masked"], markers))
                d_gpt.append(float(model == "gpt"))
                d_deepseek.append(float(model == "deepseek"))
                text = answers[model][pid]["raw"]
                cov.append(float(sum(1 for token in score162.tokens(text) if re.fullmatch(r"[^\w\s]", token))) if covariate else 0.0)
        y, d_gpt, d_deepseek, cov = map(np.asarray, (y, d_gpt, d_deepseek, cov))
        X = np.column_stack([np.ones(len(y)), d_gpt, d_deepseek])
        if covariate:
            X = np.column_stack([X, cov - np.mean(cov)])
        by_pid = {pid: np.asarray([i for i, row in enumerate(rows) if row[0] == pid]) for pid in ids}

        def fit_position(ix):
            beta, *_ = np.linalg.lstsq(X[ix], y[ix], rcond=None)
            return float(beta[2] / beta[1]) if abs(beta[1]) > 1e-12 else float("nan")

        point = fit_position(np.arange(len(y)))
        rng = np.random.RandomState(162)
        estimates = []
        for _ in range(reps):
            sampled = rng.randint(0, len(ids), len(ids))
            ix = np.concatenate([by_pid[ids[i]] for i in sampled])
            estimates.append(fit_position(ix))
        lo, hi = np.percentile([v for v in estimates if np.isfinite(v)], [2.5, 97.5])
        return {"mean": point, "percentile_2_5": float(lo), "percentile_97_5": float(hi), "n_prompts": len(ids)}

    punctuation_adjustment = {"unadjusted": adjusted_gpt_position(None),
                              "punctuation_count_adjusted": adjusted_gpt_position("punct")}

    classifier = raw_score["classifier"]["other_models"]
    medians = {m: classifier[m]["median_p_claude"] for m in MODELS if m in classifier}

    # Typography counts use the uncut answer text in the stored generation files.
    full_answers = {m: {} for m in MODELS}
    for model in ("claude", "gpt", "deepseek"):
        for row in score162.read_jsonl(args.part93_gen / f"{model}.jsonl"):
            if row.get("prompt_id") in prompt_map and isinstance(row.get("answer"), str):
                full_answers[model][row["prompt_id"]] = row["answer"]
    for row in score162.read_jsonl(args.answers):
        if row.get("family") in ("qwen", "olmo", "pythia") and row.get("prompt_id") in prompt_map:
            full_answers[row["family"]][row["prompt_id"]] = row.get("answer", "")
    typography = {}
    for char_name, char in (("curly_quote_characters", "‘’“”"), ("em_dash_characters", "—")):
        chars = tuple(char)
        typography[char_name] = {model: sum(sum(text.count(c) for c in chars) for text in full_answers[model].values()) for model in MODELS}
    result = {
        "definition": "Exploratory diagnostics recomputed from the stored Part 162 answers with score162's folds, masks and held-out marker sets.",
        "marker_counts_by_direction_and_kind_pooled_over_folds": kind_counts,
        "deepseek_positions_by_marker_kind": kind_positions,
        "deepseek_bootstrap_intervals_by_marker_kind": kind_bootstrap_intervals,
        "deepseek_minus_claude_presence_gap_by_marker_kind": kind_gaps,
        "gpt_side_numerator_share_by_marker_kind": {kind: share / total_gpt_num for kind, share in gpt_contrib.items()},
        "gpt_side_em_dash_numerator_share": dash_em_contrib / total_gpt_num,
        "typography_character_counts_full_answers": typography,
        "gate2_gpt_share_calibration": calibration,
        "deepseek_gpt_side_punctuation_adjustment": punctuation_adjustment,
        "deepseek_top_marker_contribution": top15_effect,
        "deepseek_markers_more_frequent_than_both_families": more_than_both,
        "median_heldout_p_claude_from_score_json": medians,
        "classifier_note": "This Claude-vs-GPT classifier cannot separate both from neither; all models' medians are near 0.53 to 0.60, Pythia included.",
        "answer_word_counts_and_markdown_share": answer_lengths,
    }
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
