#!/usr/bin/env python3
"""Train the Part 29 answer-style classifier and score student answer files."""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_PROMPTS = "data/part29-prompts.json"
DEFAULT_RESULTS_DIR = "results/part29"
LABELS = ["claude", "gpt", "deepseek", "qwen", "gemini", "glm", "kimi", "minimax"]
TEACHERS = {
    "claude": "claude-haiku-5-5",
    "gpt": "gpt-6-luna",
    "deepseek": "deepseek-v4-pro",
    "qwen": "qwen3.8-max",
    "gemini": "gemini-4-argon-contributor",
    "glm": "glm-5.3",
    "kimi": "kimi-k3",
    "minimax": "minimax-m3",
}
TEACHER_MODELS = set(TEACHERS.values())
FAMILY_NAMES = [
    "Claude", "Anthropic", "GPT", "ChatGPT", "OpenAI", "DeepSeek", "Qwen",
    "Alibaba", "Tongyi", "Gemini", "Google", "Bard", "GLM", "Zhipu",
    "ChatGLM", "Kimi", "Moonshot", "MiniMax", "MiMo", "Xiaomi", "Llama",
    "Meta", "Gemma", "OLMo", "AI2", "Allen", "SmolLM", "Hugging Face",
    "Mistral", "Phi", "Microsoft",
]
_mask_re = re.compile(
    r"(?<!\w)(?:" + "|".join(re.escape(s) for s in sorted(FAMILY_NAMES, key=len, reverse=True)) + r")(?!\w)",
    re.IGNORECASE,
)
_word_re = re.compile(r"\w+", re.UNICODE)


ANSWER_WORDS = 250


def mask_family_names(text: str) -> str:
    """Replace each family-identifying term as one token, case-insensitively."""
    return _mask_re.sub(" familymask ", text)


def response_text(record: dict[str, Any], *, full_text: bool = False) -> str:
    if full_text:
        value = record.get("text", "")
    else:
        value = record.get("answer", record.get("text", ""))
        value = " ".join(str(value).split()[:ANSWER_WORDS])  # the design's 300-token answer, as words (collect cap raised for reasoning)
    return value if isinstance(value, str) else ""


def read_answer_file(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                index = int(row["index"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number} is not an answer row with an integer index") from exc
            rows[index] = row
    return rows


def feature_counts(text: str) -> collections.Counter[str]:
    masked = mask_family_names(text)
    normalized = re.sub(r"\s+", " ", masked.casefold()).strip()
    words = _word_re.findall(normalized)
    counts: collections.Counter[str] = collections.Counter()
    for token in words:
        counts["w:" + token] += 1
    for left, right in zip(words, words[1:]):
        counts["w:" + left + " " + right] += 1
    for width in (3, 4, 5):
        counts.update("c:" + normalized[start : start + width] for start in range(max(0, len(normalized) - width + 1)))
    return counts


class TfidfVectorizer:
    """Small, deterministic word/character TF-IDF vectorizer using NumPy CSR arrays."""

    def __init__(self, max_features: int = 50000, min_df: int = 2):
        self.max_features = max_features
        self.min_df = min_df
        self.vocabulary: dict[str, int] = {}
        self.idf = np.empty(0, dtype=np.float32)

    def fit(self, texts: list[str]) -> "TfidfVectorizer":
        document_counts = collections.Counter()
        for text in texts:
            document_counts.update(feature_counts(text).keys())
        ranked = sorted(
            ((term, df) for term, df in document_counts.items() if df >= self.min_df),
            key=lambda item: (-item[1], item[0]),
        )[: self.max_features]
        # Stable indices simplify reproducibility and make saved reports comparable.
        terms = sorted(term for term, _df in ranked)
        self.vocabulary = {term: i for i, term in enumerate(terms)}
        dfs = np.asarray([document_counts[term] for term in terms], dtype=np.float64)
        self.idf = (np.log((1.0 + len(texts)) / (1.0 + dfs)) + 1.0).astype(np.float32)
        if not terms:
            raise ValueError("no TF-IDF features survived the document-frequency threshold")
        return self

    def transform(self, texts: list[str]) -> "SparseFeatures":
        indices: list[int] = []
        values: list[float] = []
        indptr = [0]
        for text in texts:
            counts = feature_counts(text)
            row: list[tuple[int, float]] = []
            for term, term_count in counts.items():
                feature = self.vocabulary.get(term)
                if feature is None:
                    continue
                tf = 1.0 + math.log(term_count)
                row.append((feature, tf * float(self.idf[feature])))
            row.sort()
            norm = math.sqrt(sum(value * value for _, value in row))
            if norm:
                indices.extend(feature for feature, _value in row)
                values.extend(value / norm for _feature, value in row)
            indptr.append(len(indices))
        return SparseFeatures(
            indices=np.asarray(indices, dtype=np.int32),
            values=np.asarray(values, dtype=np.float32),
            indptr=np.asarray(indptr, dtype=np.int64),
            n_features=len(self.vocabulary),
        )

    def fit_transform(self, texts: list[str]) -> "SparseFeatures":
        return self.fit(texts).transform(texts)


class SparseFeatures:
    def __init__(self, indices: np.ndarray, values: np.ndarray, indptr: np.ndarray, n_features: int):
        self.indices = indices
        self.values = values
        self.indptr = indptr
        self.n_features = n_features
        self.n_rows = len(indptr) - 1
        self.row_ids = np.repeat(np.arange(self.n_rows, dtype=np.int32), np.diff(indptr))


class NumpyLogisticRegression:
    """Multinomial softmax regression trained with regularized batch gradient descent."""

    def __init__(self, max_iter: int = 240, learning_rate: float = 1.0, l2: float = 0.0005):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.l2 = l2
        self.weights = np.empty((0, 0), dtype=np.float32)
        self.bias = np.empty(0, dtype=np.float32)

    def _scores(self, x: SparseFeatures) -> np.ndarray:
        classes = self.weights.shape[1]
        scores = np.zeros((x.n_rows, classes), dtype=np.float64)
        scores += self.bias
        for row in range(x.n_rows):
            start, end = int(x.indptr[row]), int(x.indptr[row + 1])
            if end > start:
                scores[row] += x.values[start:end] @ self.weights[x.indices[start:end]]
        return scores

    def fit(self, x: SparseFeatures, y: np.ndarray, n_classes: int) -> "NumpyLogisticRegression":
        self.weights = np.zeros((x.n_features, n_classes), dtype=np.float32)
        self.bias = np.zeros(n_classes, dtype=np.float32)
        n_rows = x.n_rows
        if n_rows != len(y) or n_rows == 0:
            raise ValueError("training feature rows and labels must be nonempty and aligned")
        rows = np.arange(n_rows)
        for iteration in range(self.max_iter):
            scores = self._scores(x)
            scores -= scores.max(axis=1, keepdims=True)
            probs = np.exp(scores)
            probs /= probs.sum(axis=1, keepdims=True)
            errors = probs
            errors[rows, y] -= 1.0
            errors /= n_rows
            gradient = np.empty_like(self.weights)
            weighted = x.values * 1.0
            for label in range(n_classes):
                gradient[:, label] = np.bincount(
                    x.indices,
                    weights=weighted * errors[x.row_ids, label],
                    minlength=x.n_features,
                )
            gradient += self.l2 * self.weights
            self.weights -= self.learning_rate * gradient
            self.bias -= self.learning_rate * errors.sum(axis=0).astype(np.float32)
            if iteration % 25 == 24 or iteration == self.max_iter - 1:
                # The objective check makes the fixed optimizer stable if the
                # feature scale or family balance changes in a later dataset.
                checked = self._scores(x)
                checked -= checked.max(axis=1, keepdims=True)
                log_partition = np.log(np.exp(checked).sum(axis=1))
                loss = float(np.mean(log_partition - checked[rows, y]))
                if not math.isfinite(loss):
                    raise FloatingPointError("classifier optimization produced a non-finite loss")
        return self

    def predict(self, x: SparseFeatures) -> np.ndarray:
        return self._scores(x).argmax(axis=1)


def _require_indices(rows: dict[int, dict[str, Any]], expected: range, path: Path, label: str) -> None:
    missing = [index for index in expected if index not in rows]
    if missing:
        preview = ", ".join(map(str, missing[:8]))
        suffix = "..." if len(missing) > 8 else ""
        raise ValueError(f"{label} file {path} is missing {len(missing)} required prompt(s): {preview}{suffix}")


def bootstrap_distribution(predictions: np.ndarray, labels: list[str], rng: np.random.Generator, replicates: int = 5000) -> dict[str, Any]:
    if len(predictions) == 0:
        raise ValueError("cannot bootstrap an empty student prediction set")
    draws = rng.integers(0, len(predictions), size=(replicates, len(predictions)))
    result: dict[str, Any] = {}
    for class_index, label in enumerate(labels):
        values = (predictions[draws] == class_index).mean(axis=1)
        count = int(np.count_nonzero(predictions == class_index))
        result[label] = {
            "count": count,
            "fraction": count / len(predictions),
            "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        }
    return result


def classify(results_dir: Path, prompts_path: Path, bootstrap_replicates: int = 5000) -> dict[str, Any]:
    prompt_payload = json.loads(prompts_path.read_text(encoding="utf-8"))
    prompts = prompt_payload.get("prompts", [])
    indices = {int(row["index"]) for row in prompts}
    expected_train = range(0, 240)
    expected_test = range(240, 300)
    required = set(expected_train) | set(expected_test)
    if not required.issubset(indices):
        missing = sorted(required - indices)
        raise ValueError(f"prompt file lacks required Part 29 indices: {missing[:8]}")

    training_texts: list[str] = []
    training_labels: list[int] = []
    test_texts: list[str] = []
    test_labels: list[int] = []
    teacher_rows: dict[str, dict[int, dict[str, Any]]] = {}
    for label_index, label in enumerate(LABELS):
        model_id = TEACHERS[label]
        path = results_dir / f"answers-{model_id}.jsonl"
        if not path.exists():
            raise ValueError(f"missing teacher answer file: {path}")
        rows = read_answer_file(path)
        _require_indices(rows, expected_train, path, label)
        _require_indices(rows, expected_test, path, label)
        teacher_rows[model_id] = rows
        for index in expected_train:
            training_texts.append(response_text(rows[index]))
            training_labels.append(label_index)
        for index in expected_test:
            test_texts.append(response_text(rows[index]))
            test_labels.append(label_index)

    vectorizer = TfidfVectorizer()
    train_x = vectorizer.fit_transform(training_texts)
    test_x = vectorizer.transform(test_texts)
    model = NumpyLogisticRegression().fit(train_x, np.asarray(training_labels, dtype=np.int64), len(LABELS))
    heldout_pred = model.predict(test_x)
    y_test = np.asarray(test_labels, dtype=np.int64)
    confusion = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    for actual, predicted in zip(y_test, heldout_pred):
        confusion[actual, predicted] += 1
    heldout = {
        "n": int(len(y_test)),
        "accuracy": float(np.mean(heldout_pred == y_test)),
        "confusion_matrix": {
            true_label: {pred_label: int(confusion[i, j]) for j, pred_label in enumerate(LABELS)}
            for i, true_label in enumerate(LABELS)
        },
    }

    students: dict[str, Any] = {}
    rng = np.random.default_rng(0)
    teacher_file_names = {f"answers-{model_id}.jsonl" for model_id in TEACHER_MODELS}
    for path in sorted(results_dir.glob("answers-*.jsonl")):
        if path.name in teacher_file_names:
            continue
        rows = read_answer_file(path)
        if not any(index in rows for index in expected_test):
            continue
        _require_indices(rows, expected_test, path, path.stem)
        student_rows = [rows[index] for index in expected_test]
        answer_texts = [response_text(row) for row in student_rows]
        full_texts = [response_text(row, full_text=True) for row in student_rows]
        answer_x = vectorizer.transform(answer_texts)
        answer_pred = model.predict(answer_x)
        is_think_model = any(
            "</think>" in full or answer != full
            for answer, full in zip(answer_texts, full_texts)
        )
        student_result: dict[str, Any] = {
            "model": student_rows[0].get("model", path.stem.removeprefix("answers-")),
            "n": len(student_rows),
            "views": {
                "answer": {
                    "distribution": bootstrap_distribution(answer_pred, LABELS, rng, bootstrap_replicates)
                }
            },
        }
        if is_think_model:
            full_pred = model.predict(vectorizer.transform(full_texts))
            student_result["views"]["full_text"] = {
                "distribution": bootstrap_distribution(full_pred, LABELS, rng, bootstrap_replicates)
            }
        students[path.name] = student_result

    return {
        "classifier": {
            "features": "TF-IDF word 1-2 grams plus character 3-5 grams",
            "optimizer": "NumPy multinomial logistic regression with L2 regularization",
            "max_features": len(vectorizer.vocabulary),
            "training_prompts_per_family": 240,
            "heldout_prompts_per_family": 60,
            "mask_token": "familymask",
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": 0,
        },
        "labels": LABELS,
        "teacher_heldout": heldout,
        "students": students,
    }


def format_report(result: dict[str, Any]) -> str:
    labels = result["labels"]
    heldout = result["teacher_heldout"]
    lines = [
        "Part 29 answer-style classifier",
        f"Teacher held-out accuracy: {heldout['accuracy']:.3f} ({heldout['n']} answers)",
        "Confusion matrix (true rows, predicted columns):",
        "true\\pred\t" + "\t".join(labels),
    ]
    for true_label in labels:
        lines.append(true_label + "\t" + "\t".join(str(heldout["confusion_matrix"][true_label][label]) for label in labels))
    for filename, student in result["students"].items():
        lines.append("")
        lines.append(f"{student['model']} [{filename}], n={student['n']}")
        for view_name, view in student["views"].items():
            lines.append(f"  {view_name} distribution (fraction, bootstrap 95% interval):")
            for label in labels:
                row = view["distribution"][label]
                lo, hi = row["ci95"]
                lines.append(f"    {label}: {row['fraction']:.3f} [{lo:.3f}, {hi:.3f}] (n={row['count']})")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", default=DEFAULT_PROMPTS)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--json-out")
    parser.add_argument("--text-out")
    args = parser.parse_args()
    if args.bootstrap_replicates < 100:
        parser.error("--bootstrap-replicates must be at least 100")
    results_dir = Path(args.results_dir)
    result = classify(Path(args.results_dir), Path(args.prompts), args.bootstrap_replicates)
    json_path = Path(args.json_out) if args.json_out else results_dir / "classify.json"
    text_path = Path(args.text_out) if args.text_out else results_dir / "classify.txt"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    text_path.write_text(format_report(result), encoding="utf-8")
    print(format_report(result), end="")
    print(f"wrote {json_path} and {text_path}")


if __name__ == "__main__":
    main()
