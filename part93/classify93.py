"""Grouped cross-validation and reusable style classifiers for Part 93."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from nltk import pos_tag, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Python isolated mode omits the script directory; keep the adjacent masker importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mask93 import mask


FUNCTION_WORDS = """a an the this that these those some any no every each all both few many much more most other another such
and or but nor so yet if unless because although while when where after before until since as than
of in on at by for from with about against between into through during without within under over
to toward towards up down out off around near
i me my mine we us our ours you your yours he him his she her hers it its they them their theirs
who whom whose which what whatever whoever
is am are was were be being been have has had having do does did doing done
can could may might must shall should will would ought need dare
not n't never nor only also just very too quite rather almost again ever still already yet
there here then now
""".split()
FUNCTION_SET = set(FUNCTION_WORDS)
CONTENT_TAGS = {
    "NN", "NNS", "NNP", "NNPS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",
    "JJ", "JJR", "JJS", "RB", "RBR", "RBS", "CD", "FW",
}


def grouped_folds(prompt_ids: Iterable[str], n_splits: int = 5):
    groups = np.asarray(list(prompt_ids))
    indexes = np.arange(len(groups))
    splitter = GroupKFold(n_splits=n_splits)
    yield from splitter.split(indexes, groups=groups)


def _text_features(text: str) -> dict[str, float]:
    tokens = word_tokenize(text, preserve_line=True)
    tagged = pos_tag(tokens) if tokens else []
    tag_sequence: list[str] = []
    features: Counter[str] = Counter()
    for token, tag in tagged:
        lower = token.lower()
        if lower in FUNCTION_SET:
            features[f"function:{lower}"] += 1
        # POS tags inserted by mask93 are retained as explicit tags here.
        tag_sequence.append(token if token in CONTENT_TAGS else tag)
    for start in range(max(0, len(tag_sequence) - 2)):
        features["pos3:" + "-".join(tag_sequence[start:start + 3])] += 1

    words = [token for token in tokens if any(char.isalnum() for char in token)]
    lines = text.splitlines() or [text]
    features["length:words"] = float(len(words))
    features["length:chars"] = float(len(text))
    features["length:mean_word_chars"] = float(sum(map(len, words)) / max(1, len(words)))
    features["markdown:headers"] = float(sum(bool(re.match(r"^\s{0,3}#{1,6}\s", line)) for line in lines))
    features["markdown:bullets"] = float(sum(bool(re.match(r"^\s*(?:[-*+]\s|\d+[.)]\s)", line)) for line in lines))
    features["markdown:bold"] = float(len(re.findall(r"\*\*.+?\*\*|__.+?__", text)))
    features["markdown:lines"] = float(len(lines))
    return dict(features)


def _classifier_a() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), sublinear_tf=True)),
        ("logistic", LogisticRegression(max_iter=2000, random_state=0)),
    ])


def _classifier_b() -> Pipeline:
    return Pipeline([
        ("features", DictVectorizer()),
        ("scale", StandardScaler(with_mean=False)),
        ("logistic", LogisticRegression(max_iter=2000, random_state=0)),
    ])


def _classifier_results(texts: list[str], labels: np.ndarray, groups: np.ndarray, kind: str, class_names: list[str], n_splits: int):
    all_predictions = np.empty(len(labels), dtype=object)
    folds = []
    for fold_number, (train, test) in enumerate(grouped_folds(groups, n_splits), 1):
        model = _classifier_a() if kind == "a" else _classifier_b()
        if kind == "a":
            model.fit([texts[i] for i in train], labels[train])
            predictions = model.predict([texts[i] for i in test])
        else:
            train_features = [_text_features(texts[i]) for i in train]
            test_features = [_text_features(texts[i]) for i in test]
            model.fit(train_features, labels[train])
            predictions = model.predict(test_features)
        all_predictions[test] = predictions
        folds.append({
            "fold": fold_number,
            "accuracy": float(accuracy_score(labels[test], predictions)),
            "confusion_matrix": confusion_matrix(labels[test], predictions, labels=class_names).tolist(),
        })
    return {
        "folds": folds,
        "mean_accuracy": float(np.mean([fold["accuracy"] for fold in folds])),
        "confusion_matrix": confusion_matrix(labels, all_predictions, labels=class_names).tolist(),
    }


def evaluate(rows: list[dict[str, Any]], n_splits: int = 5) -> dict[str, Any]:
    usable = [row for row in rows if not row.get("error") and isinstance(row.get("answer"), str)]
    if not usable:
        raise ValueError("no successful answer rows to classify")
    texts = [row["answer"] for row in usable]
    labels = np.asarray([row["family"] for row in usable])
    groups = np.asarray([row.get("prompt_id", row["id"]) for row in usable])
    unique_groups = set(groups.tolist())
    if len(unique_groups) < n_splits:
        raise ValueError(f"need at least {n_splits} distinct prompt ids for grouped CV")
    class_names = sorted(set(labels.tolist()))
    result: dict[str, Any] = {"class_names": class_names, "n": len(usable)}
    for mode in ("masked", "unmasked"):
        source = [mask(text, "pos") for text in texts] if mode == "masked" else texts
        result[mode] = {
            "classifier_a": _classifier_results(source, labels, groups, "a", class_names, n_splits),
            "classifier_b": _classifier_results(source, labels, groups, "b", class_names, n_splits),
        }
    return result


def fit_all(texts: list[str], labels: list[str]) -> dict[str, Any]:
    """Fit masked and unmasked A/B classifiers for later probability scoring."""
    if len(texts) != len(labels) or not texts:
        raise ValueError("texts and labels must be nonempty and have equal lengths")
    label_array = np.asarray(labels)
    class_names = sorted(set(labels))
    models: dict[str, Any] = {"class_names": class_names, "models": {}}
    for mode in ("masked", "unmasked"):
        source = [mask(text, "pos") for text in texts] if mode == "masked" else texts
        model_a = _classifier_a().fit(source, label_array)
        model_b = _classifier_b().fit([_text_features(text) for text in source], label_array)
        models["models"][mode] = {"a": model_a, "b": model_b}
    return models


def predict_proba(models: dict[str, Any], texts: list[str], classifier: str = "a", masked: bool = True) -> dict[str, list[list[float]]]:
    if classifier not in {"a", "b"}:
        raise ValueError("classifier must be 'a' or 'b'")
    mode = "masked" if masked else "unmasked"
    source = [mask(text, "pos") for text in texts] if masked else texts
    model = models["models"][mode][classifier]
    features: Any = source if classifier == "a" else [_text_features(text) for text in source]
    probabilities = model.predict_proba(features)
    return {
        "class_names": list(model.classes_),
        "probabilities": probabilities.tolist(),
    }


def save_models(models: dict[str, Any], path: str | Path) -> None:
    joblib.dump(models, path)


def load_models(path: str | Path) -> dict[str, Any]:
    return joblib.load(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--input", required=True, type=Path)
    evaluate_parser.add_argument("--output", type=Path)
    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument("--input", required=True, type=Path)
    fit_parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    rows = _read_jsonl(args.input)
    if args.command == "evaluate":
        result = evaluate(rows)
        rendered = json.dumps(result, indent=2)
        if args.output:
            with args.output.open("w", encoding="utf-8") as handle:
                handle.write(rendered + "\n")
        else:
            print(rendered)
    else:
        successful = [row for row in rows if not row.get("error") and isinstance(row.get("answer"), str)]
        models = fit_all([row["answer"] for row in successful], [row["family"] for row in successful])
        save_models(models, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
