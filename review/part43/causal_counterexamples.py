#!/usr/bin/env python3
"""Small exact counterexamples for the Part 43 design review.

Standard library only. These are synthetic mechanisms, not Qwen measurements.
Run from the repository root:
    python3 review/part43/causal_counterexamples.py
"""

import json
import math


def contrast(f):
    """Unscaled binary interaction: f(1,1)-f(1,0)-f(0,1)+f(0,0)."""
    return f(1, 1) - f(1, 0) - f(0, 1) + f(0, 0)


def swap_worlds(receiver, outcome):
    """i=0 clean U=1; i=1 zeroed U=0. j chooses receiver donor."""
    upstream = (1.0, 0.0)
    mediator = tuple(receiver(u) for u in upstream)
    y = {(i, j): outcome(u, mediator[j])
         for i, u in enumerate(upstream) for j in (0, 1)}
    total = y[1, 1] - y[0, 0]
    bypass = y[1, 0] - y[0, 0]
    mediated = y[1, 1] - y[1, 0]
    sufficient = y[0, 1] - y[0, 0]
    assert math.isclose(total, bypass + mediated, abs_tol=1e-12)
    return {
        "receiver_clean_and_zeroed": mediator,
        "worlds": {f"Y{i}{j}": y[i, j] for i in (0, 1) for j in (0, 1)},
        "total": total,
        "effect_with_receiver_held_clean": bypass,
        "receiver_effect_in_zeroed_background": mediated,
        "receiver_effect_in_clean_background": sufficient,
        "recovery_fraction": mediated / total if total else None,
        "interaction": y[1, 1] - y[1, 0] - y[0, 1] + y[0, 0],
    }


def silu(x):
    return x / (1.0 + math.exp(-x))


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def rmsnorm(x, gamma, eps):
    r = math.sqrt(dot(x, x) / len(x) + eps)
    return [g * a / r for g, a in zip(gamma, x)]


def main():
    # A linear directed path can have zero interaction in the donor-swap table.
    chain = swap_worlds(lambda u: 2 * u, lambda u, m: u + m)
    assert chain["interaction"] == 0
    assert math.isclose(chain["recovery_fraction"], 2 / 3)

    # Even the natural-versus-zero ablation lattice can hide a real path:
    # M = keep_M * (1+U), Y = 2U+M-UM/2. Its four corners are additive.
    outcome = lambda u, m: 2 * u + m - u * m / 2
    lattice = lambda u, keep_m: outcome(u, keep_m * (1 + u))
    hidden_path = swap_worlds(lambda u: 1 + u, outcome)
    lattice_interaction = contrast(lattice)
    assert lattice_interaction == 0
    assert hidden_path["receiver_effect_in_zeroed_background"] == -1
    assert hidden_path["receiver_effect_in_clean_background"] == -0.5

    # U and V are independent additive logit writers; only the score is nonlinear.
    def score(u, v):
        logits = (u + v, u, v)  # Google, Alibaba alias 1, Alibaba alias 2
        a, b = logits[1:]
        maximum = max(a, b)
        return logits[0] - maximum - math.log(math.exp(a-maximum) + math.exp(b-maximum))

    score_interaction = contrast(score)
    assert score_interaction > 0.2
    assert all(contrast(lambda u, v, k=k: (u+v, u, v)[k]) == 0 for k in range(3))

    # Multiplying CURRENT M in a linear chain creates a product of the controls.
    # Clamping M to a reference determined before varying U cuts that dependency.
    current_grid = contrast(lambda s, t: s + t * (1 + s))
    fixed_grid = contrast(lambda s, t: s + 2 * t)
    assert current_grid == 1 and fixed_grid == 0

    # Gate/value recombination is an exact decomposition of the scalar change.
    g0, g1, v0, v1 = -2.0, 2.0, -3.0, 4.0
    h0, h1 = silu(g0), silu(g1)
    gate = (h1 - h0) * v0
    value = h0 * (v1 - v0)
    joint = (h1 - h0) * (v1 - v0)
    change = h1 * v1 - h0 * v0
    assert math.isclose(change, gate + value + joint, abs_tol=1e-12)

    # Check zero-centred RMSNorm's contextual directional derivative.
    x, v, stored_weight = [1.2, -0.7, 2.1], [-0.3, 0.8, 0.5], [0.2, -0.1, 0.4]
    gamma = [1 + w for w in stored_weight]
    eps, step = 1e-6, 1e-5
    r = math.sqrt(dot(x, x) / len(x) + eps)
    analytical = [g * (b / r - a * dot(x, v) / (len(x) * r**3))
                  for g, a, b in zip(gamma, x, v)]
    plus = rmsnorm([a + step*b for a, b in zip(x, v)], gamma, eps)
    minus = rmsnorm([a - step*b for a, b in zip(x, v)], gamma, eps)
    numerical = [(a - b) / (2 * step) for a, b in zip(plus, minus)]
    error = max(abs(a - b) for a, b in zip(analytical, numerical))
    assert error < 1e-8

    print(json.dumps({
        "scope": "Synthetic counterexamples only; no model loaded or measured.",
        "linear_path_zero_swap_interaction": chain,
        "real_path_zero_ablation_interaction": {
            "lattice_interaction": lattice_interaction, **hidden_path},
        "independent_writes_nonlinear_score_interaction": score_interaction,
        "linear_chain_multiplier_grid_interaction": current_grid,
        "linear_chain_fixed_clamp_grid_interaction": fixed_grid,
        "swiglu_decomposition": {
            "gate_only": gate, "value_only": value, "joint": joint,
            "total": change,
            "both_clean": h0 * v0, "both_perturbed": h1 * v1,
            "clean_gate_perturbed_value": h0 * v1,
            "perturbed_gate_clean_value": h1 * v0},
        "rmsnorm_derivative_max_error_float64": error,
        "checks": "passed",
    }, indent=2))


if __name__ == "__main__":
    main()
