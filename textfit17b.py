#!/usr/bin/env python3
"""Cross-fitted text-only predictor for Part 17b's natural-activation residuals.

The model reads each passage as plain text, with no chat template.  Its two
features are the final hidden state at the last passage token and the mean final
hidden state over all passage tokens.  The frozen Qwen3-1.7B backbone gets the
same rank-8 q/k/v/o LoRA adapters used by Part 17 plus a small linear head.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

import residual17 as r17


TEXT_MODEL = "Qwen/Qwen3-1.7B"
DEFAULT_GROUPS = ("G1", "GL")
N_FOLDS = 5
DEFAULT_BATCH_SIZE = 2
DEFAULT_EPOCHS = 3
DEFAULT_LR = 2e-4
DEFAULT_RANK = 8
SMOKE_EPOCHS = 1
SMOKE_PER_SPLIT = 10


def passage_folds(passage_ids: Sequence[str], n_folds: int = N_FOLDS) -> np.ndarray:
    """Make balanced, order-independent folds from stable passage hashes."""
    if n_folds < 2:
        raise ValueError("At least two folds are required")
    ids = [str(value) for value in passage_ids]
    if len(ids) < n_folds:
        raise ValueError(f"Need at least {n_folds} passages for cross-fitting")
    if len(set(ids)) != len(ids):
        raise ValueError("Passage IDs must be unique for cross-fitting")
    order = sorted(
        range(len(ids)),
        key=lambda index: hashlib.sha256(ids[index].encode("utf-8")).hexdigest(),
    )
    folds = np.empty(len(ids), dtype=np.int8)
    for rank, index in enumerate(order):
        folds[index] = rank % n_folds
    return folds


def standardize_from_training(
    targets: np.ndarray, training_indices: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize one or more targets using only rows in the training fold."""
    values = np.asarray(targets, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError("Targets must have shape [passage] or [passage, output]")
    indices = np.asarray(training_indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError("A nonempty one-dimensional training index is required")
    if np.any(indices < 0) or np.any(indices >= len(values)):
        raise ValueError("Training index is out of bounds")
    train = values[indices]
    mean = train.mean(axis=0)
    raw_scale = train.std(axis=0)
    scale = np.where(raw_scale > 1e-12, raw_scale, 1.0)
    return mean, scale, (train - mean) / scale


def crossfit_standardized_targets(
    targets: np.ndarray,
    folds: Sequence[int],
    fit_predict_fold: Callable[[int, np.ndarray, np.ndarray, np.ndarray], np.ndarray],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Fit on all other folds and de-standardize each held-out prediction.

    The callback receives the fold number, training row indices, validation row
    indices, and standardized training targets.  It never receives validation
    targets.  Its result must be standardized predictions for validation rows.
    """
    values = np.asarray(targets, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    fold_ids = np.asarray(folds, dtype=np.int64)
    if values.ndim != 2 or fold_ids.shape != (len(values),):
        raise ValueError("Targets and fold IDs must be aligned")
    fold_values = sorted(int(value) for value in np.unique(fold_ids))
    if len(fold_values) < 2:
        raise ValueError("Cross-fitting needs at least two nonempty folds")
    predictions = np.full(values.shape, np.nan, dtype=np.float64)
    prediction_counts = np.zeros(len(values), dtype=np.int8)
    fold_reports: list[dict[str, Any]] = []
    for fold in fold_values:
        validation = np.flatnonzero(fold_ids == fold)
        training = np.flatnonzero(fold_ids != fold)
        if len(validation) == 0 or len(training) == 0:
            raise ValueError(f"Fold {fold} has an empty training or validation set")
        mean, scale, standardized_training = standardize_from_training(values, training)
        standardized_prediction = np.asarray(
            fit_predict_fold(fold, training.copy(), validation.copy(), standardized_training),
            dtype=np.float64,
        )
        if standardized_prediction.ndim == 1:
            standardized_prediction = standardized_prediction[:, None]
        if standardized_prediction.shape != (len(validation), values.shape[1]):
            raise ValueError(
                f"Fold {fold} returned {standardized_prediction.shape}; expected "
                f"({len(validation)}, {values.shape[1]})"
            )
        predictions[validation] = standardized_prediction * scale + mean
        prediction_counts[validation] += 1
        fold_reports.append(
            {
                "fold": fold,
                "train_n": int(len(training)),
                "validation_n": int(len(validation)),
                "target_mean_train": mean.tolist(),
                "target_std_train": scale.tolist(),
            }
        )
    if not np.all(prediction_counts == 1) or not np.all(np.isfinite(predictions)):
        raise RuntimeError("Cross-fitting did not produce exactly one finite OOF prediction per passage")
    return predictions, fold_reports


def _smoke_indices(splits: Sequence[str], passage_ids: Sequence[str]) -> np.ndarray:
    """Take a deterministic, small subset from each existing Part 17 split."""
    split_values = np.asarray(splits).astype(str)
    ids = np.asarray(passage_ids).astype(str)
    selected: list[int] = []
    for split in ("FIT", "TRAIN", "TEST"):
        candidates = [index for index in np.flatnonzero(split_values == split)]
        candidates.sort(key=lambda index: ids[index])
        selected.extend(candidates[:SMOKE_PER_SPLIT])
    return np.asarray(sorted(selected), dtype=np.int64)


def _r2(target: np.ndarray, prediction: np.ndarray) -> float | None:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    denominator = float(np.sum((target - target.mean()) ** 2))
    if denominator <= 1e-12:
        return None
    return float(1.0 - np.sum((target - prediction) ** 2) / denominator)


def _torch_runtime(device_name: str):
    import torch
    import selfreport as sr

    if device_name == "xpu" and (not hasattr(torch, "xpu") or not torch.xpu.is_available()):
        raise RuntimeError("--device xpu was requested but torch.xpu is unavailable")
    device = torch.device(device_name)
    sr.DEVICE = device_name
    return torch, sr, device


def _tokenize_texts(tokenizer: Any, passages: Sequence[dict[str, Any]]) -> list[list[int]]:
    token_rows: list[list[int]] = []
    for row in passages:
        encoded = tokenizer(str(row["text"]), add_special_tokens=False)["input_ids"]
        if isinstance(encoded, np.ndarray):
            encoded = encoded.tolist()
        if encoded and isinstance(encoded[0], list):
            encoded = encoded[0]
        token_ids = [int(token) for token in encoded]
        if not token_ids:
            raise ValueError(f"Passage {row['passage_id']} tokenized to an empty sequence")
        token_rows.append(token_ids)
    return token_rows


def _padded_inputs(token_rows: Sequence[Sequence[int]], pad_id: int, device: Any, torch: Any):
    lengths = [len(row) for row in token_rows]
    width = max(lengths)
    ids = torch.full((len(token_rows), width), int(pad_id), dtype=torch.long)
    mask = torch.zeros((len(token_rows), width), dtype=torch.long)
    for row_i, token_ids in enumerate(token_rows):
        ids[row_i, : lengths[row_i]] = torch.as_tensor(token_ids, dtype=torch.long)
        mask[row_i, : lengths[row_i]] = 1
    return ids.to(device), mask.to(device)


def _passage_features(model: Any, input_ids: Any, attention_mask: Any, torch: Any):
    """Return [last passage-token hidden, mean passage-token hidden]."""
    if not hasattr(model, "model"):
        raise ValueError("Expected a causal-LM wrapper with a .model base transformer")
    output = model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    hidden = output.hidden_states[-1]
    lengths = attention_mask.sum(dim=1).long()
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    last = hidden[rows, lengths - 1]
    mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
    mean = (hidden * mask).sum(dim=1) / lengths.to(hidden.dtype).unsqueeze(-1)
    return torch.cat((last.float(), mean.float()), dim=-1)


def _autocast(torch: Any, device: Any):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def _train_qwen_fold(
    *,
    fold: int,
    training_indices: np.ndarray,
    validation_indices: np.ndarray,
    standardized_targets: np.ndarray,
    token_rows: Sequence[Sequence[int]],
    hidden_size: int,
    pad_id: int,
    args: argparse.Namespace,
) -> np.ndarray:
    torch, sr, device = _torch_runtime(args.device)
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM

    fold_seed = int(args.seed + 104729 * fold)
    random.seed(fold_seed)
    np.random.seed(fold_seed)
    torch.manual_seed(fold_seed)
    if device.type == "xpu":
        torch.xpu.manual_seed(fold_seed)

    model = AutoModelForCausalLM.from_pretrained(
        TEXT_MODEL,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("Expected Qwen3ForCausalLM's model.layers LoRA path")
    lora_parameters = sr.install_lora(model, rank=args.rank)
    model.to(device)
    model.config.use_cache = False
    head = torch.nn.Linear(hidden_size * 2, standardized_targets.shape[1]).to(device)
    optimizer = torch.optim.AdamW([*lora_parameters, *head.parameters()], lr=args.lr)
    train_target_by_index = {
        int(index): standardized_targets[row_i]
        for row_i, index in enumerate(training_indices)
    }
    epochs = args.epochs if args.epochs is not None else (SMOKE_EPOCHS if args.smoke else DEFAULT_EPOCHS)
    rng = np.random.default_rng(fold_seed)
    batches_per_epoch = math.ceil(len(training_indices) / args.batch_size)
    total_batches = epochs * batches_per_epoch
    step = 0
    started = time.perf_counter()

    for epoch in range(epochs):
        order = rng.permutation(training_indices).tolist()
        model.train()
        head.train()
        for batch_start in range(0, len(order), args.batch_size):
            batch_indices = [int(index) for index in order[batch_start : batch_start + args.batch_size]]
            batch_tokens = [token_rows[index] for index in batch_indices]
            input_ids, attention_mask = _padded_inputs(batch_tokens, pad_id, device, torch)
            targets = torch.as_tensor(
                np.stack([train_target_by_index[index] for index in batch_indices]),
                dtype=torch.float32,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            with _autocast(torch, device):
                features = _passage_features(model, input_ids, attention_mask, torch)
                prediction = head(features)
                loss = F.mse_loss(prediction, targets)
            loss.backward()
            optimizer.step()
            step += 1
            print(
                f"fit-text fold {fold + 1}/5 train epoch {epoch + 1}/{epochs} "
                f"batch {step}/{total_batches} loss={float(loss.detach()):.6f}",
                flush=True,
            )

    model.eval()
    head.eval()
    validation_prediction = np.empty((len(validation_indices), standardized_targets.shape[1]), dtype=np.float64)
    validation_batches = math.ceil(len(validation_indices) / args.batch_size)
    with torch.inference_mode():
        for batch_no, batch_start in enumerate(range(0, len(validation_indices), args.batch_size), 1):
            batch_indices = validation_indices[batch_start : batch_start + args.batch_size]
            batch_tokens = [token_rows[int(index)] for index in batch_indices]
            input_ids, attention_mask = _padded_inputs(batch_tokens, pad_id, device, torch)
            with _autocast(torch, device):
                features = _passage_features(model, input_ids, attention_mask, torch)
                prediction = head(features)
            values = prediction.float().cpu().numpy()
            validation_prediction[batch_start : batch_start + len(batch_indices)] = values
            print(
                f"fit-text fold {fold + 1}/5 predict batch {batch_no}/{validation_batches} "
                f"rows {len(batch_indices)}",
                flush=True,
            )
    print(
        f"fit-text fold {fold + 1}/5 complete train={len(training_indices)} "
        f"validation={len(validation_indices)} seconds={time.perf_counter() - started:.1f}",
        flush=True,
    )
    del optimizer, head, model
    gc.collect()
    if device.type == "xpu":
        torch.xpu.empty_cache()
    return validation_prediction


def _fit_file_arrays(
    passage_ids: np.ndarray,
    splits: np.ndarray,
    activations: np.ndarray,
    requested_groups: Sequence[str],
    target_predictions: np.ndarray,
    fold_ids: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    n = len(passage_ids)
    predicted = np.zeros_like(activations, dtype=np.float32)
    for requested_i, group in enumerate(requested_groups):
        group_i = r17.GROUP_NAMES.index(group)
        predicted[:, group_i, :] = target_predictions[:, requested_i, None].astype(np.float32)
    residuals = activations.astype(np.float32) - predicted
    residual_scalar = residuals.mean(axis=2).astype(np.float32)
    labels = np.zeros((n, len(r17.GROUP_NAMES)), dtype=np.int8)
    thresholds = np.zeros(len(r17.GROUP_NAMES), dtype=np.float64)
    for group_i in range(len(r17.GROUP_NAMES)):
        threshold, group_labels = r17.fit_threshold_labels(residual_scalar[:, group_i], splits)
        thresholds[group_i] = threshold
        labels[:, group_i] = group_labels.astype(np.int8)
    low20, high80 = r17.fit_activation_quantiles(activations, splits)
    fit_arrays = {
        "passage_ids": np.asarray(passage_ids).astype("U64"),
        "fold_ids": np.asarray(fold_ids, dtype=np.int8),
        "predicted": predicted,
        "predictions": predicted,
        "residuals": residuals,
        "residual_scalar": residual_scalar,
        "labels": labels,
        "thresholds": thresholds,
        "low20": low20,
        "high80": high80,
    }
    return fit_arrays, {"labels": labels, "thresholds": thresholds, "residual_scalar": residual_scalar}


def _fit_text(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    capture_arrays, manifest = r17._read_capture(args.out_dir)
    capture_ids = capture_arrays["passage_ids"].astype(str)
    manifest_rows = manifest.get("passages", [])
    manifest_ids = [str(row["passage_id"]) for row in manifest_rows]
    if manifest_ids != capture_ids.tolist():
        raise ValueError("capture-manifest passage order does not match capture.npz")
    if len(capture_ids) == 0:
        raise ValueError("Capture contains no passages")

    fit_out = Path(args.fit_out_dir)
    capture_root = Path(args.out_dir)
    if fit_out.resolve() == capture_root.resolve():
        raise ValueError("--fit-out-dir must differ from --out-dir so the capture stays untouched")

    all_splits = capture_arrays["splits"].astype(str)
    if args.smoke:
        selection = _smoke_indices(all_splits, capture_ids)
        if len(selection) < N_FOLDS:
            raise ValueError("Smoke selection needs at least five passages")
        passage_ids = capture_ids[selection]
        splits = all_splits[selection]
        activation_all = capture_arrays["olmo_activations"].astype(np.float32)
        activations = activation_all[selection]
        passages = [manifest_rows[int(index)] for index in selection]
    else:
        passage_ids = capture_ids
        splits = all_splits
        activations = capture_arrays["olmo_activations"].astype(np.float32)
        passages = manifest_rows
    if activations.ndim != 3 or activations.shape[1:] != (len(r17.GROUP_NAMES), 11):
        raise ValueError(f"Expected OLMo activations [passage, {len(r17.GROUP_NAMES)}, 11]")
    if len(passages) != len(passage_ids) or len(splits) != len(passage_ids):
        raise ValueError("Capture passage IDs, splits, activations, and manifest rows must align")
    if not set(np.unique(splits)).issubset({"FIT", "TRAIN", "TEST"}):
        raise ValueError("Capture splits must be FIT, TRAIN, and TEST")
    if not np.any(splits == "FIT"):
        raise ValueError("At least one FIT passage is needed for the FIT-only threshold")
    groups = list(dict.fromkeys(args.groups))
    if not groups:
        raise ValueError("At least one target group is required")
    targets = np.column_stack(
        [activations[:, r17.GROUP_NAMES.index(group), :].mean(axis=1) for group in groups]
    ).astype(np.float64)
    folds = passage_folds(passage_ids, N_FOLDS)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL, local_files_only=True)
    token_rows = _tokenize_texts(tokenizer, passages)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("Qwen tokenizer must provide a pad_token_id or eos_token_id")
    # Qwen3-1.7B's hidden size is read from its cached config without loading weights.
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(TEXT_MODEL, local_files_only=True)
    hidden_size = int(config.hidden_size)
    del tokenizer
    gc.collect()

    def fit_predict_fold(
        fold: int,
        training_indices: np.ndarray,
        validation_indices: np.ndarray,
        standardized_training_targets: np.ndarray,
    ) -> np.ndarray:
        return _train_qwen_fold(
            fold=fold,
            training_indices=training_indices,
            validation_indices=validation_indices,
            standardized_targets=standardized_training_targets,
            token_rows=token_rows,
            hidden_size=hidden_size,
            pad_id=int(pad_id),
            args=args,
        )

    target_predictions, fold_reports = crossfit_standardized_targets(
        targets, folds, fit_predict_fold
    )
    fit_arrays, label_result = _fit_file_arrays(
        passage_ids, splits, activations, groups, target_predictions, folds
    )
    fit_indices = np.flatnonzero(splits == "FIT")
    train_indices = np.flatnonzero(splits == "TRAIN")
    test_indices = np.flatnonzero(splits == "TEST")
    per_group: dict[str, Any] = {}
    for output_i, group in enumerate(groups):
        group_i = r17.GROUP_NAMES.index(group)
        target = targets[:, output_i]
        prediction = target_predictions[:, output_i]
        group_labels = label_result["labels"][:, group_i]
        per_group[group] = {
            "fitted": True,
            "target": "mean natural activation over the group's 11 units",
            "oof_r2_all": _r2(target, prediction),
            "oof_r2_test": _r2(target[test_indices], prediction[test_indices]) if len(test_indices) else None,
            "oof_n_all": int(len(target)),
            "oof_n_test": int(len(test_indices)),
            "fit_median_residual_threshold": float(label_result["thresholds"][group_i]),
            "residual_mean_fit": float(label_result["residual_scalar"][fit_indices, group_i].mean()),
            "residual_std_fit": float(label_result["residual_scalar"][fit_indices, group_i].std()),
            "train_label_counts": {
                "lower": int(np.sum(group_labels[train_indices] == 0)),
                "higher": int(np.sum(group_labels[train_indices] == 1)),
            },
            "test_label_counts": {
                "lower": int(np.sum(group_labels[test_indices] == 0)),
                "higher": int(np.sum(group_labels[test_indices] == 1)),
            },
            "fit_quantile20_mean": float(fit_arrays["low20"][group_i].mean()),
            "fit_quantile80_mean": float(fit_arrays["high80"][group_i].mean()),
        }
    epochs = args.epochs if args.epochs is not None else (SMOKE_EPOCHS if args.smoke else DEFAULT_EPOCHS)
    result = {
        "task": "part17b-cross-fitted-text-residual-fit",
        "model": TEXT_MODEL,
        "config": {
            "device": args.device,
            "seed": args.seed,
            "groups": groups,
            "fitted_groups": groups,
            "smoke": bool(args.smoke),
            "smoke_per_split": SMOKE_PER_SPLIT if args.smoke else None,
            "folds": N_FOLDS,
            "fold_assignment": "SHA-256(passage_id) ascending, round-robin across five folds",
            "input": "plain passage text; tokenizer add_special_tokens=False; no chat template",
            "features": "concatenation of final hidden at last passage token and masked mean final hidden over passage tokens",
            "target": "mean of the group's 11 natural activation units, standardized with each fold's training rows only",
            "prediction": "fold-specific predictions de-standardized with that fold's training target statistics",
            "training": "frozen bf16 Qwen3-1.7B plus selfreport.install_lora rank-8 q/k/v/o adapters and one linear output per requested group",
            "epochs": epochs,
            "learning_rate": args.lr,
            "rank": args.rank,
            "batch_size": args.batch_size,
            "optimizer": "AdamW over LoRA and linear-head parameters only",
            "quantiles": [0.20, 0.80],
            "label_definition": "mean of 11 residual units > FIT-only median means higher (1); otherwise lower (0)",
            "r2_definition": "scalar natural group mean predicted out of fold",
        },
        "sizes": {
            "total": int(len(passage_ids)),
            "FIT": int(len(fit_indices)),
            "TRAIN": int(len(train_indices)),
            "TEST": int(len(test_indices)),
            "features": int(hidden_size * 2),
        },
        "seeds": {"run": args.seed, "split": "capture manifest", "crossfit": "SHA-256(passage_id)"},
        "crossfit": {"folds": N_FOLDS, "per_fold": fold_reports},
        "timings_seconds": {"total": time.perf_counter() - started},
        "per_group": per_group,
        "source_capture": str(Path(args.out_dir) / "capture.npz"),
    }
    fit_out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(fit_out / "fit.npz", **fit_arrays)
    (fit_out / "fit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(fit_out / "fit.npz"),
                "report": str(fit_out / "fit.json"),
                "sizes": result["sizes"],
                "per_group": per_group,
                "timings_seconds": result["timings_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit-text", help="cross-fit the Qwen3-1.7B text predictor over a Part 17 capture")
    fit.add_argument("--out-dir", default=r17.DEFAULT_OUT_DIR, help="Part 17 capture directory")
    fit.add_argument("--fit-out-dir", required=True, help="new directory for fit.json and fit.npz")
    fit.add_argument("--groups", nargs="+", choices=r17.GROUP_NAMES, default=list(DEFAULT_GROUPS))
    fit.add_argument("--device", choices=("xpu", "cpu"), default="cpu")
    fit.add_argument("--seed", type=int, default=r17.DEFAULT_SEED)
    fit.add_argument("--epochs", type=int, default=None)
    fit.add_argument("--lr", type=float, default=DEFAULT_LR)
    fit.add_argument("--rank", type=int, default=DEFAULT_RANK)
    fit.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    fit.add_argument("--threads", type=int, default=4)
    fit.add_argument("--smoke", action="store_true", help="use at most ten passages from each capture split and one epoch")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.lr <= 0 or not math.isfinite(args.lr):
        parser.error("--lr must be finite and positive")
    if args.rank < 1 or args.batch_size < 1 or args.threads < 1:
        parser.error("--rank, --batch-size, and --threads must be positive")
    torch, _, _ = _torch_runtime(args.device)
    torch.set_num_threads(args.threads)
    _fit_text(args)


if __name__ == "__main__":
    main()
