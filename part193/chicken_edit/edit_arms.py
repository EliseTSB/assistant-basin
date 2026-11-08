"""Fit and score the chicken-edit arms from the stored arrays.

All fitting uses the fixed fit split from EDIT-DESIGN.md / edit_pilot_gate.py.
The test split is transformed and scored only after every arm is frozen.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

W = 2048
SPLIT_SEED = 20261104
BOOTSTRAP_SEED = 20261104
BOOTSTRAP_DRAWS = 4000
QUANTILE_LEVELS = np.arange(0.1, 1.0, 0.1, dtype=np.float64)
# A fixed, broad log grid; the same values are used for every written-block fit
# and for the output-space steering scale.
K_GRID = np.logspace(-2, 5, 29, dtype=np.float64)
MLP_EPOCHS = 500
MLP_HIDDEN = 64


def auc_values(pos, neg):
    """Mann-Whitney AUC with half credit for ties, matching analyze_pixabay.py."""
    pos = np.asarray(pos, dtype=np.float64).reshape(-1)
    neg = np.asarray(neg, dtype=np.float64).reshape(-1)
    if not len(pos) or not len(neg):
        raise ValueError("AUC needs at least one positive and one negative")
    ranks = rankdata(np.concatenate([pos, neg]))
    n_pos = len(pos)
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * len(neg)))


def fit_imitation_split(rows):
    """Return the fit mask and selected participants exactly as edit_pilot_gate.py."""
    imitators = sorted({
        row["participant_id"]
        for row in rows
        if row["group"] == "chicken" and row["kind"] == "imitation"
    })
    rng = np.random.default_rng(SPLIT_SEED)
    fit_people = set(
        np.asarray(imitators)[
            rng.permutation(len(imitators))[: len(imitators) // 2]
        ].tolist()
    )
    fit_mask = np.asarray([
        row["group"] == "chicken"
        and row["kind"] == "imitation"
        and row["participant_id"] in fit_people
        for row in rows
    ], dtype=bool)
    return fit_mask, sorted(fit_people)


def normalize_vector(x):
    x = np.asarray(x, dtype=np.float32)
    norm = float(np.linalg.norm(x))
    if norm == 0.0:
        raise ValueError("cannot normalize a zero vector")
    return x / norm


def normalize_rows(x):
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError("cannot normalize a zero row")
    return x / norms


def prompt_scores(embeddings, prompt_emb, hen_idx, rooster_idx, other_idx):
    """Chicken prompt score: max hen/rooster cosine minus mean other cosine."""
    e = normalize_rows(embeddings)
    p = normalize_rows(prompt_emb)
    similarities = e @ p.T
    return similarities[:, [hen_idx, rooster_idx]].max(axis=1) - similarities[:, other_idx].mean(axis=1)


def prompt_predictions(embeddings, prompt_emb, categories):
    e = normalize_rows(embeddings)
    p = normalize_rows(prompt_emb)
    return np.asarray(categories)[np.argmax(e @ p.T, axis=1)]


def apply_written_gate(e, s19, s17, t19, t17, k, c):
    """Apply one gated write, returning the original object if the gate is closed."""
    g = max(float(s19) - float(t19), 0.0) * max(float(t17) - float(s17), 0.0)
    if g == 0.0:
        return e
    return normalize_vector(np.asarray(e, dtype=np.float32) - float(k) * g * c)


def apply_gate_batch(embeddings, s19, s17, t19, t17, k, c, gate_fn=apply_written_gate):
    out = np.empty_like(embeddings, dtype=np.float32)
    for i, e in enumerate(embeddings):
        out[i] = gate_fn(e, s19[i], s17[i], t19, t17, k, c)
    return out


def bit_identical_rows(before, after):
    """Bitwise comparison per row, including signed zero distinctions."""
    a = np.ascontiguousarray(before).view(np.uint8).reshape(len(before), -1)
    b = np.ascontiguousarray(after).view(np.uint8).reshape(len(after), -1)
    return np.all(a == b, axis=1)


def apply_steering(embeddings, k, c):
    return normalize_rows(np.asarray(embeddings, dtype=np.float32) - float(k) * c[None, :])


def make_bootstrap_indices(n_pos, n_neg, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    return (
        rng.integers(0, n_pos, size=(draws, n_pos)),
        rng.integers(0, n_neg, size=(draws, n_neg)),
    )


def bootstrap_ci(pos, neg, indices):
    bp, bn = indices
    values = np.empty(len(bp), dtype=np.float64)
    for i, (ip, inn) in enumerate(zip(bp, bn)):
        values[i] = auc_values(pos[ip], neg[inn])
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def signed_discovery_sign(act_mean, disc_chicken, disc_nonchicken, unit):
    raw = np.asarray(act_mean[:, unit], dtype=np.float32)
    return 1 if auc_values(raw[disc_chicken], raw[disc_nonchicken]) >= 0.5 else -1


def gate_values(act_mean, u17, sign17, u19, sign19):
    a = np.asarray(act_mean, dtype=np.float32)
    return sign19 * a[:, u19], sign17 * a[:, u17]


def write_fit(positive_e, negative_e, positive_s19, positive_s17,
              negative_s19, negative_s17, c, prompt_emb, hen_idx,
              rooster_idx, other_idx):
    """Fit thresholds and k on fit examples only. k-grid order breaks ties low-first."""
    levels = [float(x) for x in QUANTILE_LEVELS]
    t19_grid = np.quantile(negative_s19, QUANTILE_LEVELS)
    t17_grid = np.quantile(negative_s17, QUANTILE_LEVELS)
    best = None
    for k in K_GRID:
        for qi, t19 in enumerate(t19_grid):
            for qj, t17 in enumerate(t17_grid):
                edited_neg = apply_gate_batch(
                    negative_e, negative_s19, negative_s17, t19, t17, k, c
                )
                edited_pos = apply_gate_batch(
                    positive_e, positive_s19, positive_s17, t19, t17, k, c
                )
                pos_score = prompt_scores(edited_pos, prompt_emb, hen_idx, rooster_idx, other_idx)
                neg_score = prompt_scores(edited_neg, prompt_emb, hen_idx, rooster_idx, other_idx)
                value = auc_values(pos_score, neg_score)
                if best is None or value > best["fit_auc"]:
                    best = {
                        "t19": float(t19),
                        "t17": float(t17),
                        "k": float(k),
                        "t19_quantile": levels[qi],
                        "t17_quantile": levels[qj],
                        "fit_auc": value,
                    }
    return best


def fit_steering(positive_e, negative_e, c, prompt_emb, hen_idx,
                 rooster_idx, other_idx, target_auc):
    values = []
    for k in K_GRID:
        pos_score = prompt_scores(apply_steering(positive_e, k, c), prompt_emb,
                                  hen_idx, rooster_idx, other_idx)
        neg_score = prompt_scores(apply_steering(negative_e, k, c), prompt_emb,
                                  hen_idx, rooster_idx, other_idx)
        value = auc_values(pos_score, neg_score)
        values.append((float(k), value))
        if value >= target_auc:
            return {
                "k": float(k),
                "fit_auc": value,
                "target_written_fit_auc": float(target_auc),
                "selection": "smallest k on the fixed log grid reaching the written block fit AUC",
                "reached_target": True,
                "grid": [float(x) for x in K_GRID],
            }
    best_auc = max(v for _, v in values)
    best_k = min(k for k, v in values if v == best_auc)
    return {
        "k": best_k,
        "fit_auc": best_auc,
        "target_written_fit_auc": float(target_auc),
        "selection": "highest fit AUC on the fixed log grid because no k reached the written block fit AUC",
        "reached_target": False,
        "grid": [float(x) for x in K_GRID],
    }


def class_damage(before, after, labels, prompt_emb, categories):
    pred_before = prompt_predictions(before, prompt_emb, categories)
    pred_after = prompt_predictions(after, prompt_emb, categories)
    rows = {}
    for label in categories:
        mask = labels == label
        if not np.any(mask):
            continue
        correct_before = int(np.sum(pred_before[mask] == label))
        correct_after = int(np.sum(pred_after[mask] == label))
        n = int(mask.sum())
        delta = correct_after - correct_before
        rows[str(label)] = {
            "n": n,
            "correct_before": correct_before,
            "correct_after": correct_after,
            "accuracy_before": float(correct_before / n),
            "accuracy_after": float(correct_after / n),
            "delta_accuracy": float(delta / n),
            "delta_correct_clips": delta,
        }
    changed_any = sorted(label for label, r in rows.items() if r["delta_correct_clips"] != 0)
    changed_more_than_one = sorted(
        label for label, r in rows.items() if abs(r["delta_correct_clips"]) > 1
    )
    return {
        "per_class": rows,
        "classes_changed_any_accuracy": changed_any,
        "classes_changed_any_count": len(changed_any),
        "classes_changed_more_than_1_clip": changed_more_than_one,
        "classes_changed_more_than_1_clip_count": len(changed_more_than_one),
        "all_classes_within_1_clip": all(
            abs(r["delta_correct_clips"]) <= 1 for r in rows.values()
        ),
    }


def fit_residual_mlp(fit_real, fit_imit, keep, prompt_emb, hen_idx,
                     rooster_idx, other_idx, shuffle_labels=False):
    """Train the fixed CPU residual MLP for exactly 500 full-batch epochs."""
    import torch
    from torch import nn
    from torch.nn import functional as F

    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    class ResidualMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(768, MLP_HIDDEN)
            self.fc2 = nn.Linear(MLP_HIDDEN, 768)

        def forward(self, x):
            return F.normalize(x + self.fc2(F.relu(self.fc1(x))), dim=1)

    all_fit = np.concatenate([fit_real, fit_imit], axis=0).astype(np.float32)
    labels = np.concatenate([
        np.ones(len(fit_real), dtype=np.int64),
        np.zeros(len(fit_imit), dtype=np.int64),
    ])
    if shuffle_labels:
        labels = labels[np.random.default_rng(0).permutation(len(labels))]
    pos = torch.from_numpy(all_fit[labels == 1])
    neg = torch.from_numpy(all_fit[labels == 0])
    keep_t = torch.from_numpy(np.asarray(keep, dtype=np.float32))
    prompts_t = torch.from_numpy(normalize_rows(prompt_emb))
    other_t = prompts_t[other_idx]
    model = ResidualMLP().cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def score(x):
        similarities = model(x) @ prompts_t.T
        return torch.maximum(similarities[:, hen_idx], similarities[:, rooster_idx]) - similarities[:, other_idx].mean(dim=1)

    final_loss = None
    for _ in range(MLP_EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        fit_imit_score = score(neg)
        fit_real_score = score(pos)
        edited_keep = model(keep_t)
        loss = (
            F.softplus(fit_imit_score).mean()
            + F.softplus(-fit_real_score).mean()
            + 10.0 * torch.mean((edited_keep - keep_t) ** 2)
        )
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
    return model, final_loss


def apply_mlp(model, embeddings):
    import torch

    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(np.asarray(embeddings, dtype=np.float32))
        return model(x).cpu().numpy().astype(np.float32, copy=False)


def mlp_parameters(model, final_loss, shuffle_labels):
    state = model.state_dict()
    return {
        "architecture": "residual MLP: Linear(768,64), ReLU, Linear(64,768), normalize(e + MLP(e))",
        "seed": 0,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "epochs": MLP_EPOCHS,
        "full_batch": True,
        "keep_penalty_weight": 10.0,
        "labels_shuffled": bool(shuffle_labels),
        "final_training_loss": float(final_loss),
        "state_dict": {
            name: value.detach().cpu().numpy().astype(np.float32).tolist()
            for name, value in state.items()
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acts", type=Path, required=True)
    parser.add_argument("--vis-acts", type=Path, required=True)
    parser.add_argument("--pixabay-acts", type=Path, required=True)
    parser.add_argument("--vis-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    ref = np.load(args.acts)
    vis = np.load(args.vis_acts)
    pix = np.load(args.pixabay_acts)

    with args.vis_manifest.open(newline="") as f:
        manifest = list(csv.DictReader(f))
    manifest_ids = np.asarray([row["id"] for row in manifest])
    if not np.array_equal(vis["id"], manifest_ids):
        raise RuntimeError("vis_acts.npz rows do not align with vis_manifest.csv ids")

    fit_imit_mask, fit_people = fit_imitation_split(manifest)
    vis_group = vis["group"]
    vis_kind = vis["kind"]
    vis_real_mask = (vis_group == "chicken") & (vis_kind == "real")
    vis_chicken_imit_mask = (vis_group == "chicken") & (vis_kind == "imitation")
    if int(fit_imit_mask.sum()) != 18:
        raise RuntimeError(f"expected the pilot's 18 fit imitations, found {fit_imit_mask.sum()}")
    test_imit_mask = vis_chicken_imit_mask & ~fit_imit_mask

    categories = list(ref["categories"])
    prompt_emb = np.asarray(ref["prompt_emb"], dtype=np.float32)
    hen_idx = categories.index("hen")
    rooster_idx = categories.index("rooster")
    other_idx = np.asarray([i for i in range(len(categories)) if i not in (hen_idx, rooster_idx)])
    if len(other_idx) != 48:
        raise RuntimeError(f"expected 48 non-chicken ESC-50 prompts, found {len(other_idx)}")

    ref_kind = ref["kind"]
    ref_labels = ref["label"]
    ref_fold = ref["fold"]
    esc_mask = ref_kind == "esc"
    chicken_mask = np.isin(ref_labels, ["hen", "rooster"])
    discovery = esc_mask & (ref_fold <= 3)
    discovery_chicken = discovery & chicken_mask
    discovery_nonchicken = discovery & ~chicken_mask
    ref_act = np.asarray(ref["act_mean"], dtype=np.float32)
    ref_emb = np.asarray(ref["emb"], dtype=np.float32)
    vis_act = np.asarray(vis["act_mean"], dtype=np.float32)
    vis_emb = np.asarray(vis["emb"], dtype=np.float32)
    pix_act = np.asarray(pix["act_mean"], dtype=np.float32)
    pix_emb = np.asarray(pix["emb"], dtype=np.float32)

    u17 = 17 * W + 1574
    u19 = 19 * W + 1129
    sign17 = signed_discovery_sign(ref_act, discovery_chicken, discovery_nonchicken, u17)
    sign19 = signed_discovery_sign(ref_act, discovery_chicken, discovery_nonchicken, u19)
    chicken_c = normalize_vector(prompt_emb[[hen_idx, rooster_idx]].mean(axis=0))

    fit_ref_chicken = esc_mask & (ref_fold <= 3) & chicken_mask
    fit_keep = esc_mask & (ref_fold <= 3) & ~chicken_mask
    test_esc = esc_mask & np.isin(ref_fold, [4, 5])
    test_esc_nonchicken = test_esc & ~chicken_mask

    pix_real_mask = pix["label"] == "real"
    pix_imit_mask = pix["label"] == "imitation"
    fit_real_e = np.concatenate([pix_emb[pix_real_mask], ref_emb[fit_ref_chicken]], axis=0)
    fit_imit_e = vis_emb[fit_imit_mask]
    fit_keep_e = ref_emb[fit_keep]
    fit_real_act = np.concatenate([pix_act[pix_real_mask], ref_act[fit_ref_chicken]], axis=0)
    fit_imit_act = vis_act[fit_imit_mask]

    fit_imit_s19, fit_imit_s17 = gate_values(fit_imit_act, u17, sign17, u19, sign19)
    fit_real_s19, fit_real_s17 = gate_values(fit_real_act, u17, sign17, u19, sign19)

    def gate_inputs(act, neuron17=u17, neuron19=u19, s17=sign17, s19=sign19):
        return gate_values(act, neuron17, s17, neuron19, s19)

    # Frozen fit/test/descriptive inputs and their gate signals.
    vis_real_e = vis_emb[vis_real_mask]
    test_imit_e = vis_emb[test_imit_mask]
    pix_imit_e = pix_emb[pix_imit_mask]
    ref_test_e = ref_emb[test_esc]
    ref_test_labels = ref_labels[test_esc]
    ref_nonchicken_test_e = ref_emb[test_esc_nonchicken]

    vis_real_s19, vis_real_s17 = gate_inputs(vis_act[vis_real_mask])
    test_imit_s19, test_imit_s17 = gate_inputs(vis_act[test_imit_mask])
    pix_imit_s19, pix_imit_s17 = gate_inputs(pix_act[pix_imit_mask])
    ref_test_s19, ref_test_s17 = gate_inputs(ref_act[test_esc])
    ref_nonchicken_s19, ref_nonchicken_s17 = gate_inputs(ref_act[test_esc_nonchicken])

    # Fit the written block, then use its fit AUC as the target for steering.
    written = write_fit(
        fit_real_e, fit_imit_e, fit_real_s19, fit_real_s17, fit_imit_s19, fit_imit_s17,
        chicken_c, prompt_emb, hen_idx, rooster_idx, other_idx,
    )
    steering_fit = fit_steering(
        fit_real_e, fit_imit_e, chicken_c, prompt_emb, hen_idx, rooster_idx,
        other_idx, written["fit_auc"],
    )

    # Random-direction control: seed 0, direction normalized to unit length.
    random_direction_rng = np.random.default_rng(0)
    random_c = normalize_vector(random_direction_rng.normal(size=prompt_emb.shape[1]).astype(np.float32))
    random_direction = write_fit(
        fit_real_e, fit_imit_e, fit_real_s19, fit_real_s17, fit_imit_s19, fit_imit_s17,
        random_c, prompt_emb, hen_idx, rooster_idx, other_idx,
    )

    # Random-gate control: one random unit in each specified layer, separately signed
    # by its ESC-50 discovery-fold AUC, with thresholds and k refitted.
    random_neuron_rng = np.random.default_rng(0)
    random17 = 17 * W + int(random_neuron_rng.integers(0, W))
    random19 = 19 * W + int(random_neuron_rng.integers(0, W))
    random_sign17 = signed_discovery_sign(ref_act, discovery_chicken, discovery_nonchicken, random17)
    random_sign19 = signed_discovery_sign(ref_act, discovery_chicken, discovery_nonchicken, random19)
    fit_imit_r19, fit_imit_r17 = gate_inputs(
        fit_imit_act, random17, random19, random_sign17, random_sign19
    )
    fit_real_r19, fit_real_r17 = gate_inputs(
        fit_real_act, random17, random19, random_sign17, random_sign19
    )
    random_gate = write_fit(
        fit_real_e, fit_imit_e, fit_real_r19, fit_real_r17, fit_imit_r19, fit_imit_r17,
        chicken_c, prompt_emb, hen_idx, rooster_idx, other_idx,
    )

    # Train the true and shuffled-label MLPs once, on the fit split only.
    trained_model, trained_loss = fit_residual_mlp(
        fit_real_e, fit_imit_e, fit_keep_e, prompt_emb, hen_idx, rooster_idx, other_idx
    )
    shuffled_model, shuffled_loss = fit_residual_mlp(
        fit_real_e, fit_imit_e, fit_keep_e, prompt_emb, hen_idx, rooster_idx, other_idx,
        shuffle_labels=True,
    )

    baseline_pos = prompt_scores(vis_real_e, prompt_emb, hen_idx, rooster_idx, other_idx)
    baseline_neg = prompt_scores(test_imit_e, prompt_emb, hen_idx, rooster_idx, other_idx)
    baseline_test_auc = auc_values(baseline_pos, baseline_neg)
    bootstrap_indices = make_bootstrap_indices(len(baseline_pos), len(baseline_neg))

    arm_specs = {
        "written_gated_block": {
            "kind": "written_block",
            "params": {
                **written,
                "direction": "normalized mean of hen and rooster prompt embeddings",
                "gate_units": {"L17_N1574_sign": sign17, "L19_N1129_sign": sign19},
                "quantile_levels": [float(q) for q in QUANTILE_LEVELS],
                "k_grid": [float(x) for x in K_GRID],
                "tie_break": "smallest k; first t19/t17 quantile pair in ascending grid order",
            },
        },
        "steering": {
            "kind": "steering",
            "params": {
                **steering_fit,
                "direction": "normalized mean of hen and rooster prompt embeddings",
            },
        },
        "trained_layer": {
            "kind": "mlp",
            "params": mlp_parameters(trained_model, trained_loss, False),
        },
        "random_direction_control": {
            "kind": "written_block",
            "direction_vector": random_c,
            "params": {
                **random_direction,
                "direction": "seed-0 random unit vector",
                "direction_seed": 0,
                "gate_units": {"L17_N1574_sign": sign17, "L19_N1129_sign": sign19},
                "quantile_levels": [float(q) for q in QUANTILE_LEVELS],
                "k_grid": [float(x) for x in K_GRID],
                "tie_break": "smallest k; first t19/t17 quantile pair in ascending grid order",
            },
        },
        "random_gate_control": {
            "kind": "written_block",
            "params": {
                **random_gate,
                "direction": "normalized mean of hen and rooster prompt embeddings",
                "gate_units": {
                    "L17_N": random17 % W,
                    "L17_sign": random_sign17,
                    "L19_N": random19 % W,
                    "L19_sign": random_sign19,
                    "selection_seed": 0,
                },
                "quantile_levels": [float(q) for q in QUANTILE_LEVELS],
                "k_grid": [float(x) for x in K_GRID],
                "tie_break": "smallest k; first t19/t17 quantile pair in ascending grid order",
            },
        },
        "shuffled_label_trained_layer": {
            "kind": "mlp_shuffled",
            "params": mlp_parameters(shuffled_model, shuffled_loss, True),
        },
    }

    def transform(spec, embeddings, s19, s17, *, act=None, random_sigs=None):
        kind = spec["kind"]
        params = spec["params"]
        if kind == "written_block":
            direction = spec.get("direction_vector", chicken_c)
            if random_sigs is None:
                return apply_gate_batch(
                    embeddings, s19, s17, params["t19"], params["t17"], params["k"], direction
                )
            r19, r17 = random_sigs
            return apply_gate_batch(
                embeddings, r19, r17, params["t19"], params["t17"], params["k"], direction
            )
        if kind == "steering":
            return apply_steering(embeddings, params["k"], chicken_c)
        if kind == "mlp":
            return apply_mlp(trained_model, embeddings)
        if kind == "mlp_shuffled":
            return apply_mlp(shuffled_model, embeddings)
        raise ValueError(kind)

    def all_scores(spec, e_real, e_imit, s19_real, s17_real, s19_imit, s17_imit,
                   real_random=None, imit_random=None):
        if spec["kind"] in ("written_block",):
            re = transform(spec, e_real, s19_real, s17_real, random_sigs=real_random)
            im = transform(spec, e_imit, s19_imit, s17_imit, random_sigs=imit_random)
        else:
            re = transform(spec, e_real, s19_real, s17_real)
            im = transform(spec, e_imit, s19_imit, s17_imit)
        return re, im

    arm_results = {}
    for name, spec in arm_specs.items():
        if name == "random_gate_control":
            fit_real_random = (fit_real_r19, fit_real_r17)
            fit_imit_random = (fit_imit_r19, fit_imit_r17)
            test_real_random = gate_inputs(
                vis_act[vis_real_mask], random17, random19, random_sign17, random_sign19
            )
            test_imit_random = gate_inputs(
                vis_act[test_imit_mask], random17, random19, random_sign17, random_sign19
            )
            desc_random = gate_inputs(
                pix_act[pix_imit_mask], random17, random19, random_sign17, random_sign19
            )
            esc_random = gate_inputs(
                ref_act[test_esc], random17, random19, random_sign17, random_sign19
            )
            nonchicken_random = gate_inputs(
                ref_act[test_esc_nonchicken], random17, random19, random_sign17, random_sign19
            )
        else:
            fit_real_random = fit_imit_random = test_real_random = test_imit_random = None
            desc_random = esc_random = nonchicken_random = None

        fit_re, fit_im = all_scores(
            spec, fit_real_e, fit_imit_e, fit_real_s19, fit_real_s17,
            fit_imit_s19, fit_imit_s17, fit_real_random, fit_imit_random,
        )
        fit_pos_score = prompt_scores(fit_re, prompt_emb, hen_idx, rooster_idx, other_idx)
        fit_neg_score = prompt_scores(fit_im, prompt_emb, hen_idx, rooster_idx, other_idx)
        # The threshold-fitted methods already store their selected fit AUC; recomputation
        # provides one common check for every arm.
        fit_auc = auc_values(fit_pos_score, fit_neg_score)

        edited_real, edited_imit = all_scores(
            spec, vis_real_e, test_imit_e, vis_real_s19, vis_real_s17,
            test_imit_s19, test_imit_s17, test_real_random, test_imit_random,
        )
        pos_score = prompt_scores(edited_real, prompt_emb, hen_idx, rooster_idx, other_idx)
        neg_score = prompt_scores(edited_imit, prompt_emb, hen_idx, rooster_idx, other_idx)
        test_auc = auc_values(pos_score, neg_score)

        if spec["kind"] == "written_block":
            edited_esc = transform(
                spec, ref_test_e, ref_test_s19, ref_test_s17, random_sigs=esc_random
            )
            edited_nonchicken = transform(
                spec, ref_nonchicken_test_e, ref_nonchicken_s19, ref_nonchicken_s17,
                random_sigs=nonchicken_random,
            )
        else:
            edited_esc = transform(spec, ref_test_e, ref_test_s19, ref_test_s17)
            edited_nonchicken = transform(
                spec, ref_nonchicken_test_e, ref_nonchicken_s19, ref_nonchicken_s17
            )

        damage = class_damage(
            ref_test_e, edited_esc, ref_test_labels, prompt_emb, categories
        )
        identical = bit_identical_rows(ref_nonchicken_test_e, edited_nonchicken)
        if spec["kind"] == "written_block" and name != "random_gate_control":
            desc_after = transform(spec, pix_imit_e, pix_imit_s19, pix_imit_s17)
        elif name == "random_gate_control":
            desc_after = transform(
                spec, pix_imit_e, pix_imit_s19, pix_imit_s17, random_sigs=desc_random
            )
        else:
            desc_after = transform(spec, pix_imit_e, pix_imit_s19, pix_imit_s17)
        desc_before_score = prompt_scores(pix_imit_e, prompt_emb, hen_idx, rooster_idx, other_idx)
        desc_after_score = prompt_scores(desc_after, prompt_emb, hen_idx, rooster_idx, other_idx)

        arm_results[name] = {
            "parameters": spec["params"],
            "fit_auc": float(fit_auc),
            "test_auc": float(test_auc),
            "test_auc_bootstrap_95": bootstrap_ci(pos_score, neg_score, bootstrap_indices),
            "baseline_test_auc": float(baseline_test_auc),
            "damage_esc50_folds_4_5": damage,
            "nonchicken_bit_identical": {
                "identical_clips": int(identical.sum()),
                "total_clips": int(len(identical)),
                "share": float(identical.mean()),
            },
            "pixabay_imitation_descriptive": [
                {
                    "id": str(clip_id),
                    "score_before": float(before),
                    "score_after": float(after),
                }
                for clip_id, before, after in zip(
                    pix["id"][pix_imit_mask], desc_before_score, desc_after_score
                )
            ],
        }

    written_damage = arm_results["written_gated_block"]["damage_esc50_folds_4_5"]
    trained_damage = arm_results["trained_layer"]["damage_esc50_folds_4_5"]
    steering_damage = arm_results["steering"]["damage_esc50_folds_4_5"]
    controls = [
        "random_direction_control",
        "random_gate_control",
        "shuffled_label_trained_layer",
    ]
    control_gains = {
        name: float(arm_results[name]["test_auc"] - baseline_test_auc)
        for name in controls
    }

    prediction_text = {
        "E1": "The written gated block raises the held-out AUC to at least 0.75. Medium (its gate opened on 56% of fit imitations, which caps the gain).",
        "E2": "The written block leaves at least 95% of ESC-50 fold 4 and 5 non-chicken clips bit-identical, and changes per-class accuracy on folds 4 and 5 by at most 1 clip in every class. Medium.",
        "E3": "The trained layer reaches a held-out AUC of at least 0.85. Medium.",
        "E4": "The trained layer changes per-class accuracy on folds 4 and 5 by more than 1 clip in at least one class. Low-medium.",
        "E5": "The steering arm, scaled on the fit split to the written block's fit AUC, changes more fold 4 and 5 classes' accuracy than the written block does. Medium.",
        "E6": "Controls gain less than 0.05 held-out AUC each: the written block with a random direction, the written block gated by two random same-layer neurons, and the trained layer fitted on shuffled real/imitation labels. High.",
    }
    verdicts = {
        "E1": {
            "prediction": prediction_text["E1"],
            "verdict": "held" if arm_results["written_gated_block"]["test_auc"] >= 0.75 else "not held",
            "written_test_auc": arm_results["written_gated_block"]["test_auc"],
            "threshold": 0.75,
        },
        "E2": {
            "prediction": prediction_text["E2"],
            "verdict": "held" if (
                arm_results["written_gated_block"]["nonchicken_bit_identical"]["share"] >= 0.95
                and written_damage["all_classes_within_1_clip"]
            ) else "not held",
            "nonchicken_bit_identical_share": arm_results["written_gated_block"]["nonchicken_bit_identical"]["share"],
            "required_share": 0.95,
            "all_classes_within_1_clip": written_damage["all_classes_within_1_clip"],
            "classes_changed_more_than_1_clip": written_damage["classes_changed_more_than_1_clip"],
            "classes_changed_more_than_1_clip_count": written_damage["classes_changed_more_than_1_clip_count"],
        },
        "E3": {
            "prediction": prediction_text["E3"],
            "verdict": "held" if arm_results["trained_layer"]["test_auc"] >= 0.85 else "not held",
            "trained_layer_test_auc": arm_results["trained_layer"]["test_auc"],
            "threshold": 0.85,
        },
        "E4": {
            "prediction": prediction_text["E4"],
            "verdict": "held" if trained_damage["classes_changed_more_than_1_clip_count"] > 0 else "not held",
            "classes_changed_more_than_1_clip": trained_damage["classes_changed_more_than_1_clip"],
            "classes_changed_more_than_1_clip_count": trained_damage["classes_changed_more_than_1_clip_count"],
            "required_count": "at least 1",
        },
        "E5": {
            "prediction": prediction_text["E5"],
            "verdict": "held" if steering_damage["classes_changed_any_count"] > written_damage["classes_changed_any_count"] else "not held",
            "steering_classes_changed_any_count": steering_damage["classes_changed_any_count"],
            "written_classes_changed_any_count": written_damage["classes_changed_any_count"],
            "steering_classes_changed_any": steering_damage["classes_changed_any_accuracy"],
            "written_classes_changed_any": written_damage["classes_changed_any_accuracy"],
        },
        "E6": {
            "prediction": prediction_text["E6"],
            "verdict": "held" if all(gain < 0.05 for gain in control_gains.values()) else "not held",
            "gain_threshold": 0.05,
            "control_gains": control_gains,
            "each_control_below_threshold": {name: gain < 0.05 for name, gain in control_gains.items()},
        },
    }

    results = {
        "specification": "EDIT-DESIGN.md and PREDICTIONS.md E1-E6; Amendment 1 applied",
        "method": {
            "score": "cosine to hen/rooster prompt maximum minus mean cosine to the other 48 ESC-50 prompts",
            "auc_bootstrap_draws": BOOTSTRAP_DRAWS,
            "auc_bootstrap_interval": "percentile 95% interval, 2.5th and 97.5th percentiles; paired resample indices shared across arms",
            "bootstrap_seed": BOOTSTRAP_SEED,
            "threshold_quantiles": [float(q) for q in QUANTILE_LEVELS],
            "k_grid": [float(x) for x in K_GRID],
            "mlp": {
                "hidden_units": MLP_HIDDEN,
                "epochs": MLP_EPOCHS,
                "optimizer": "Adam",
                "learning_rate": 0.001,
                "cpu": True,
                "seed": 0,
                "keep_penalty_weight": 10.0,
            },
        },
        "signs": {
            "L17_N1574": sign17,
            "L19_N1129": sign19,
        },
        "split": {
            "seed": SPLIT_SEED,
            "fit_rule": "first half of the shuffled sorted chicken-imitation participant ids, matching edit_pilot_gate.py",
            "fit_imitator_count": len(fit_people),
            "fit_imitation_count": int(fit_imit_mask.sum()),
            "test_imitation_count": int(test_imit_mask.sum()),
            "test_real_chicken_count": int(vis_real_mask.sum()),
            "fit_real_positive_count": int(len(fit_real_e)),
            "fit_keep_nonchicken_count": int(len(fit_keep_e)),
            "fit_people": fit_people,
            "esc_test_fold_4_5_count": int(test_esc.sum()),
            "esc_test_nonchicken_count": int(test_esc_nonchicken.sum()),
            "pixabay_real_count": int(pix_real_mask.sum()),
            "pixabay_imitation_count": int(pix_imit_mask.sum()),
        },
        "baseline_test_auc": float(baseline_test_auc),
        "baseline_test_auc_bootstrap_95": bootstrap_ci(baseline_pos, baseline_neg, bootstrap_indices),
        "arms": arm_results,
        "verdicts": verdicts,
    }

    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=1, allow_nan=False) + "\n")
    print(json.dumps({
        "split": results["split"],
        "baseline_test_auc": results["baseline_test_auc"],
        "arms": {
            name: {
                "parameters": {
                    key: value for key, value in arm["parameters"].items()
                    if key not in ("state_dict",)
                },
                "fit_auc": arm["fit_auc"],
                "test_auc": arm["test_auc"],
                "test_auc_bootstrap_95": arm["test_auc_bootstrap_95"],
                "nonchicken_bit_identical": arm["nonchicken_bit_identical"],
                "changed_more_than_1_clip": arm["damage_esc50_folds_4_5"]["classes_changed_more_than_1_clip"],
                "changed_any": arm["damage_esc50_folds_4_5"]["classes_changed_any_accuracy"],
            }
            for name, arm in arm_results.items()
        },
        "verdicts": verdicts,
        "results_file": str(out_path),
    }, indent=2))


if __name__ == "__main__":
    main()
