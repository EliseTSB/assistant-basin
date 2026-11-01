"""Score Part 162's masked style markers, grouped folds, gates and predictions."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "part93"))
from mask93 import mask  # noqa: E402

TOKEN_RE = re.compile(r"\w+|[^\w\s]|\n", flags=re.UNICODE)
STORED = ("claude", "gpt", "deepseek")
GENERATED = ("qwen", "olmo", "pythia")
BASE_MODELS = STORED + GENERATED


def words(text: str) -> list[str]:
    return text.split()


QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'})


def normalize_quotes(text: str) -> str:
    return text.translate(QUOTES)


def first_words(text: str, n: int) -> str:
    """The text through its nth whitespace-split word, keeping newlines, which carry line structure."""
    found = list(re.finditer(r"\S+", text))
    return text[: found[n - 1].end()] if len(found) >= n else text


def cut100(text: str) -> str:
    return first_words(text, 100)


def is_usable_answer(text: str | None) -> bool:
    return isinstance(text, str) and len(words(text)) >= 100


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def ngram_features(text: str) -> set[tuple[str, ...]]:
    ts = tokens(text)
    return {tuple(ts[i:i + n]) for n in (1, 2, 3) for i in range(len(ts) - n + 1)}


def seeded_folds(prompt_ids: list[str], n_splits: int = 5, seed: int = 162):
    unique = sorted(set(prompt_ids))
    if len(unique) < n_splits:
        raise ValueError(f"need at least {n_splits} prompt groups")
    rng = np.random.RandomState(seed)
    shuffled = list(rng.permutation(unique))
    partitions = [set(x.tolist()) for x in np.array_split(np.asarray(shuffled, dtype=object), n_splits)]
    for test_groups in partitions:
        test = np.asarray([i for i, pid in enumerate(prompt_ids) if pid in test_groups], dtype=int)
        train = np.asarray([i for i, pid in enumerate(prompt_ids) if pid not in test_groups], dtype=int)
        yield train, test


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_rows(args: argparse.Namespace):
    with args.prompts.open(encoding="utf-8") as f:
        prompts = json.load(f)
    prompt_map = {p["prompt_id"]: p["prompt"] for p in prompts}
    raw: dict[str, dict[str, str]] = {name: {} for name in BASE_MODELS}
    for family in STORED:
        for row in read_jsonl(args.part93_gen / f"{family}.jsonl"):
            pid = row.get("prompt_id", row.get("id"))
            if pid in prompt_map and not row.get("error") and isinstance(row.get("answer"), str):
                raw[family][pid] = row["answer"]
    answer_files = [args.answers] + ([args.extra_answers] if getattr(args, "extra_answers", None) else [])
    for path in answer_files:  # A later file replaces an earlier answer for the same model and prompt.
        for row in read_jsonl(path):
            pid, family = row.get("prompt_id"), row.get("family")
            if family in GENERATED and pid in prompt_map and isinstance(row.get("answer"), str):
                raw[family][pid] = row["answer"]

    answers: dict[str, dict[str, dict[str, Any]]] = {name: {} for name in BASE_MODELS}
    counts = {}
    for model in BASE_MODELS:
        dropped = 0
        for pid in prompt_map:
            text = raw[model].get(pid)
            if text is not None and getattr(args, "normalize_quotes", False):
                text = normalize_quotes(text)
            if not is_usable_answer(text):
                dropped += 1
                continue
            cut = cut100(text)
            answers[model][pid] = {"raw": cut, "masked": mask(cut, "pos"), "word_count": len(words(text))}
        counts[model] = {"usable": len(answers[model]), "dropped_under_100_or_missing": dropped,
                         "total_prompts": len(prompt_map)}
    return prompts, prompt_map, answers, counts


def _presence(answer: str, marker_set: list[tuple[str, ...]]) -> float:
    feats = ngram_features(answer)
    return float(sum(marker in feats for marker in marker_set) / len(marker_set)) if marker_set else float("nan")


def train_markers(train_ids: list[str], answers: dict[str, dict[str, dict[str, Any]]],
                  drop_dash: bool = False) -> tuple[list[tuple[str, ...]], list[tuple[str, ...]]]:
    c_ids = [p for p in train_ids if p in answers["claude"] and p in answers["gpt"]]
    features = {model: {pid: ngram_features(answers[model][pid]["masked"]) for pid in c_ids}
                for model in ("claude", "gpt")}
    universe = set().union(*(features[m][pid] for m in features for pid in c_ids)) if c_ids else set()
    if drop_dash:
        universe = {f for f in universe if not any("\u2014" in token for token in f)}
    rows = {}
    for family in ("claude", "gpt"):
        rows[family] = {feat: np.mean([feat in features[family][pid] for pid in c_ids]) for feat in universe}
    c_markers = sorted((f for f in universe if rows["claude"][f] >= 0.1),
                       key=lambda f: (-(rows["claude"][f] - rows["gpt"].get(f, 0.0)), f))[:100]
    g_markers = sorted((f for f in universe if rows["gpt"][f] >= 0.1),
                       key=lambda f: (-(rows["gpt"][f] - rows["claude"].get(f, 0.0)), f))[:100]
    return c_markers, g_markers


def _position(model_values: np.ndarray, claude_values: np.ndarray, gpt_values: np.ndarray, which: str) -> float:
    x, c, g = float(np.mean(model_values)), float(np.mean(claude_values)), float(np.mean(gpt_values))
    if which == "claude":
        denom = c - g
        return float("nan") if abs(denom) < 1e-12 else (x - g) / denom
    denom = g - c
    return float("nan") if abs(denom) < 1e-12 else (x - c) / denom


def marker_positions(answers, prompt_ids, folds_by_prompt, marker_by_fold, models):
    arrays = {model: {"claude": {}, "gpt": {}} for model in models}
    for pid in prompt_ids:
        fold = folds_by_prompt[pid]
        c_markers, g_markers = marker_by_fold[fold]
        for model in models:
            if pid in answers[model]:
                masked = answers[model][pid]["masked"]
                arrays[model]["claude"][pid] = _presence(masked, c_markers)
                arrays[model]["gpt"][pid] = _presence(masked, g_markers)
    return arrays


def bootstrap_position(arrays, model: str, which: str, seed: int, reps: int = 2000):
    ids = sorted(set(arrays[model][which]) & set(arrays["claude"][which]) & set(arrays["gpt"][which]))
    if not ids:
        return {"mean": None, "percentile_2_5": None, "percentile_97_5": None, "n_prompts": 0}
    c = np.asarray([arrays["claude"][which][pid] for pid in ids])
    g = np.asarray([arrays["gpt"][which][pid] for pid in ids])
    x = np.asarray([arrays[model][which][pid] for pid in ids])
    point = _position(x, c, g, which)
    rng = np.random.RandomState(seed)
    estimates = []
    for draw in range(reps):
        ix = rng.randint(0, len(ids), size=len(ids))
        estimates.append(_position(x[ix], c[ix], g[ix], which))
    finite = np.asarray([v for v in estimates if np.isfinite(v)])
    if not np.isfinite(point) or not len(finite):
        return {"mean": None, "percentile_2_5": None, "percentile_97_5": None, "n_prompts": len(ids)}
    lo, hi = np.percentile(finite, [2.5, 97.5])
    return {"mean": float(point), "percentile_2_5": float(lo), "percentile_97_5": float(hi), "n_prompts": len(ids)}


def classifier_results(answers, prompt_ids):
    ids = [pid for pid in prompt_ids if pid in answers["claude"] and pid in answers["gpt"]]
    texts = [answers[f][pid]["masked"] for pid in ids for f in ("claude", "gpt")]
    labels = np.asarray([f for pid in ids for f in ("claude", "gpt")])
    groups = [pid for pid in ids for _ in (0, 1)]
    probabilities: dict[str, dict[str, float]] = defaultdict(dict)
    predictions = np.empty(len(labels), dtype=object)
    fold_num = np.empty(len(labels), dtype=int)
    fold_rows = []
    for fold, (train, test) in enumerate(seeded_folds(groups), 0):
        model = LogisticRegression(max_iter=2000, random_state=0)
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), sublinear_tf=True)
        train_x = vectorizer.fit_transform([texts[i] for i in train])
        test_x = vectorizer.transform([texts[i] for i in test])
        model.fit(train_x, labels[train])
        predictions[test] = model.predict(test_x)
        fold_num[test] = fold
        c_index = list(model.classes_).index("claude")
        probs = model.predict_proba(test_x)[:, c_index]
        for idx, prob in zip(test, probs):
            family = labels[idx]
            if family not in ("claude", "gpt"):
                continue
            probabilities[family][groups[idx]] = float(prob)
        fold_rows.append({"fold": fold, "accuracy": float(np.mean(predictions[test] == labels[test])),
                          "n_answers": int(len(test))})
    accuracy = float(np.mean(predictions == labels))
    return {"accuracy": accuracy, "folds": fold_rows,
            "p_claude_by_model": {fam: probabilities[fam] for fam in probabilities}}, ids


def classifier_other_probabilities(answers, prompt_ids):
    ids = [pid for pid in prompt_ids if pid in answers["claude"] and pid in answers["gpt"]]
    pair_ids = [pid for pid in ids for _ in (0, 1)]
    labels = np.asarray([fam for _pid in ids for fam in ("claude", "gpt")])
    texts = [answers[fam][pid]["masked"] for pid in ids for fam in ("claude", "gpt")]
    groups = [pid for pid in ids for _ in (0, 1)]
    result = {model: [] for model in GENERATED + ("deepseek",)}
    for train, test in seeded_folds(groups):
        train_prompts = {groups[i] for i in train}
        test_prompts = {groups[i] for i in test}
        vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), sublinear_tf=True)
        train_x = vec.fit_transform([texts[i] for i in train])
        clf = LogisticRegression(max_iter=2000, random_state=0).fit(train_x, labels[train])
        ci = list(clf.classes_).index("claude")
        for model_name in result:
            for pid in sorted(test_prompts):
                if pid in answers[model_name]:
                    p = float(clf.predict_proba(vec.transform([answers[model_name][pid]["masked"]]))[0, ci])
                    result[model_name].append({"prompt_id": pid, "p_claude": p})
    return {name: {"median_p_claude": float(np.median([r["p_claude"] for r in rows])) if rows else None,
                   "n": len(rows), "predictions": rows} for name, rows in result.items()}


def add_mixtures(answers):
    common = set(answers["claude"]) & set(answers["gpt"])
    for model, order in (("mix_claude_then_gpt", ("claude", "gpt")), ("mix_gpt_then_claude", ("gpt", "claude"))):
        answers[model] = {}
        for pid in common:
            text = first_words(answers[order[0]][pid]["raw"], 50) + " " + first_words(answers[order[1]][pid]["raw"], 50)
            answers[model][pid] = {"raw": text, "masked": mask(text, "pos"), "word_count": len(words(text))}


def evaluate_gates(classifier_accuracy: float, mixture_c: float | None, mixture_g: float | None,
                   pythia_c: float | None, pythia_g: float | None,
                   coverage: dict[str, float]) -> dict[str, dict[str, Any]]:
    return {
        "1": {"name": "Claude and GPT separate", "metric": classifier_accuracy, "threshold": 0.90,
              "pass": classifier_accuracy >= 0.90},
        "2": {"name": "50/50 mixture reads as both", "claude_position": mixture_c,
              "gpt_position": mixture_g, "range": [0.25, 0.75],
              "pass": mixture_c is not None and mixture_g is not None and
              0.25 <= mixture_c <= 0.75 and 0.25 <= mixture_g <= 0.75},
        "3": {"name": "Pythia reads as neither", "claude_position": pythia_c,
              "gpt_position": pythia_g, "threshold": 0.25,
              "pass": pythia_c is not None and pythia_g is not None and
              pythia_c < 0.25 and pythia_g < 0.25},
        "4": {"name": "Enough text", "coverage": coverage,
              "thresholds": {"qwen": 0.90, "olmo": 0.90, "pythia": 0.70},
              "pass": coverage["qwen"] >= 0.90 and coverage["olmo"] >= 0.90 and
              coverage["pythia"] >= 0.70},
    }


def score(args: argparse.Namespace):
    prompts, prompt_map, answers, counts = load_rows(args)
    add_mixtures(answers)
    prompt_ids = list(prompt_map)
    marker_by_fold = {}
    folds_by_prompt = {}
    for fold, (train_ix, test_ix) in enumerate(seeded_folds(prompt_ids)):
        marker_by_fold[fold] = train_markers([prompt_ids[i] for i in train_ix], answers,
                                             drop_dash=getattr(args, "drop_dash_markers", False))
        for i in test_ix:
            folds_by_prompt[prompt_ids[i]] = fold
    models = BASE_MODELS + ("mix_claude_then_gpt", "mix_gpt_then_claude")
    arrays = marker_positions(answers, prompt_ids, folds_by_prompt, marker_by_fold, models)
    positions = {}
    for model in models:
        positions[model] = {way: bootstrap_position(arrays, model, way, seed=162)
                            for j, way in enumerate(("claude", "gpt"))}
    arrays["mixture_mean"] = {"claude": {}, "gpt": {}}
    positions["mixture_mean"] = {}
    for way in ("claude", "gpt"):
        # Gate 2 uses the per-prompt mean of the two orders, then pooled positions.
        ids = sorted(set(arrays["mix_claude_then_gpt"][way]) & set(arrays["mix_gpt_then_claude"][way]))
        arrays["mixture_mean"][way] = {pid: (arrays["mix_claude_then_gpt"][way][pid] +
                                             arrays["mix_gpt_then_claude"][way][pid]) / 2
                                       for pid in ids}
        positions["mixture_mean"][way] = bootstrap_position(arrays, "mixture_mean", way, seed=162)
    classifier, classifier_ids = classifier_results(answers, prompt_ids)
    other_probs = classifier_other_probabilities(answers, prompt_ids)
    classifier["other_models"] = other_probs
    coverage = {m: counts[m]["usable"] / max(1, counts[m]["total_prompts"]) for m in GENERATED}
    for mixture in ("mix_claude_then_gpt", "mix_gpt_then_claude"):
        usable = len(answers[mixture])
        counts[mixture] = {"usable": usable, "dropped_under_100_or_missing": len(prompt_ids) - usable,
                           "total_prompts": len(prompt_ids)}
    gates = evaluate_gates(classifier["accuracy"], positions["mixture_mean"]["claude"]["mean"],
                           positions["mixture_mean"]["gpt"]["mean"], positions["pythia"]["claude"]["mean"],
                           positions["pythia"]["gpt"]["mean"], coverage)
    all_gates = all(g["pass"] for g in gates.values())
    prediction_defs = {
        "162.1": ("deepseek", lambda p: p["claude"]["mean"] is not None and p["gpt"]["mean"] is not None and p["claude"]["mean"] >= .25 and p["gpt"]["mean"] >= .25),
        "162.2": ("qwen", lambda p: p["claude"]["mean"] is not None and p["gpt"]["mean"] is not None and p["claude"]["mean"] - p["gpt"]["mean"] >= .1),
        "162.3": ("deepseek", lambda p: p["claude"]["mean"] is not None and p["gpt"]["mean"] is not None and p["claude"]["mean"] > p["gpt"]["mean"]),
        "162.4": ("deepseek", lambda p: other_probs["deepseek"]["median_p_claude"] is not None and .3 <= other_probs["deepseek"]["median_p_claude"] <= .7),
        "162.5": ("olmo", lambda p: p["gpt"]["mean"] is not None and p["claude"]["mean"] is not None and p["gpt"]["mean"] > p["claude"]["mean"]),
    }
    predictions = {}

    def readable(model: str) -> bool:
        # The design stops the affected reading: Gates 1 to 3 affect every reading, Gate 4 only the short model's.
        if not all(gates[g]["pass"] for g in ("1", "2", "3")):
            return False
        limits = gates["4"]["thresholds"]
        return model not in limits or coverage[model] >= limits[model]

    for number, (model, rule) in prediction_defs.items():
        predictions[number] = {"model": model, "result": positions.get(model),
                               "verdict": "untestable" if not readable(model) else ("held" if rule(positions[model]) else "failed")}

    fold0_c, fold0_g = marker_by_fold[0]
    marker_rates = {}
    for direction, markers in (("claude", fold0_c), ("gpt", fold0_g)):
        ranked = markers
        marker_rates[direction] = []
        for marker in ranked[:15]:
            rates = {}
            for model in BASE_MODELS:
                vals = [marker in ngram_features(answers[model][pid]["masked"]) for pid in answers[model]]
                rates[model] = float(np.mean(vals)) if vals else None
            marker_rates[direction].append({"marker": " ".join(marker), "rates": rates})

    result = {"part": 162, "n_prompts": len(prompt_ids), "counts": counts, "positions": positions,
              "classifier": classifier, "gates": gates, "predictions": predictions,
              "top_markers_fold_0": marker_rates, "n_bootstrap": 2000, "seed": 162}
    return result, answers


def write_outputs(args: argparse.Namespace, result, answers):
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    with (out / "score.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    lines = ["# Part 162 score", "", "## Gates", "", "| Gate | Result |", "|---|---|"]
    for n, gate in result["gates"].items():
        lines.append(f"| {n}: {gate['name']} | {'PASS' if gate['pass'] else 'FAIL'} |")
    lines += ["", "## Positions", "", "| Model | Claude markers | 95% interval | GPT markers | 95% interval |", "|---|---:|---:|---:|---:|"]
    for model, ways in result["positions"].items():
        def show(obj):
            if obj.get("mean") is None:
                return "NA", "NA"
            return f"{obj['mean']:.3f}", f"[{obj['percentile_2_5']:.3f}, {obj['percentile_97_5']:.3f}]" if obj.get("percentile_2_5") is not None else "NA"
        c, ci = show(ways["claude"]); g, gi = show(ways["gpt"])
        lines.append(f"| {model} | {c} | {ci} | {g} | {gi} |")
    lines += ["", "## Answer counts", "", "| Model | Usable | Dropped | Total |", "|---|---:|---:|---:|"]
    for model, c in result["counts"].items():
        lines.append(f"| {model} | {c['usable']} | {c['dropped_under_100_or_missing']} | {c['total_prompts']} |")
    for direction in ("claude", "gpt"):
        lines += ["", f"## Top 15 {direction} markers, fold 0", "", "| Marker | " + " | ".join(BASE_MODELS) + " |", "|---|" + "---:|" * len(BASE_MODELS)]
        for item in result["top_markers_fold_0"][direction]:
            lines.append("| `" + item["marker"].replace("`", "\\`") + "` | " + " | ".join("NA" if item["rates"][m] is None else f"{item['rates'][m]:.3f}" for m in BASE_MODELS) + " |")
    with (out / "tables.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    rng = np.random.RandomState(162)
    sample_path = out / "sample.jsonl"
    answer_models = BASE_MODELS + ("mix_claude_then_gpt", "mix_gpt_then_claude")
    with sample_path.open("w", encoding="utf-8") as f:
        for model in answer_models:
            ids = sorted(answers[model])
            chosen = rng.choice(ids, size=min(12, len(ids)), replace=False) if ids else []
            for pid in chosen:
                entry = answers[model][pid]
                f.write(json.dumps({"model": model, "prompt_id": str(pid), "answer": entry["masked"],
                                    "word_count": min(100, entry["word_count"])}, ensure_ascii=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, default=Path(__file__).resolve().parent / "prompts162.json")
    parser.add_argument("--part93-gen", type=Path, default=ROOT / "results" / "part93" / "gen")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result, answers = score(args)
    write_outputs(args, result, answers)
    failed = [n for n, gate in result["gates"].items() if not gate["pass"]]
    print(f"wrote scores; gates passed: {4 - len(failed)}/4")
    return 3 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
