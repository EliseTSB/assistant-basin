#!/usr/bin/env python3
"""Recompute Part 115b counts and make the flips figure from the run files."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "part115"))
sys.path.insert(0, str(ROOT / "part115b"))
from score115 import first_word  # noqa: E402
from score115b import score as score_rows  # noqa: E402


RAW_PATH = HERE / "results115b.json"
SCORE_PATH = HERE / "score.json"
DESIGN_PATH = ROOT / "PART115B-DESIGN.md"
RAW = json.loads(RAW_PATH.read_text(encoding="utf-8"))
SAVED_SCORE = json.loads(SCORE_PATH.read_text(encoding="utf-8"))
DESIGN_TEXT = DESIGN_PATH.read_text(encoding="utf-8")
RECOMPUTED_SCORE = score_rows(RAW)
if RECOMPUTED_SCORE != SAVED_SCORE:
    raise SystemExit("score.json does not match a fresh score of results115b.json")

RANK_NAMES = {"gradient": "Gradient", "direct": "Direct-write"}
CONTEXTS = ("plain", "quote", "bold")
CONTEXT_LABELS = {
    "plain": 'My name is → target " Q"',
    "quote": 'My name is " → target "Q"',
    "bold": 'My name is ** → target "Q"',
}
KS = RAW["ks"]
MODEL_REVISION = re.search(r"Qwen/Qwen3\.5-4B at revision ([0-9a-f]+), bf16, the A770", DESIGN_TEXT).group(1)
GENERATION_TOKENS = int(re.search(r"Greedy (\d+)-token continuations", DESIGN_TEXT).group(1))


def rows_for(rank: str, role: str) -> list[dict]:
    return [r for r in RAW["rankings"][rank]["rows"] if r["role"] == role]


def baseline_row(rank: str) -> dict:
    return rows_for(rank, "none")[0]


N_HELDOUT = len(baseline_row("gradient")["plain"])


def candidate_row(rank: str, k: int) -> dict:
    return next(r for r in rows_for(rank, "candidate") if r["k"] == k)


def control_rows(rank: str, k: int) -> list[dict]:
    return [r for r in RAW["rankings"][rank]["rows"]
            if r["role"] not in ("none", "candidate") and r["k"] == k]


def flips(row: dict, ctx: str) -> int:
    return sum(not x["argmax_is_target"] for x in row[ctx])


def candidate_flips(rank: str, ctx: str, k: int) -> int:
    return flips(candidate_row(rank, k), ctx)


def max_control_flips(rank: str, ctx: str, k: int) -> int:
    return max(flips(r, ctx) for r in control_rows(rank, k))


def k_star(rank: str, ctx: str) -> int | None:
    eligible = [k for k in KS if candidate_flips(rank, ctx, k) >= 8]
    return min(eligible) if eligible else None


def ordinary_delta(rank: str, k: int) -> float:
    base = baseline_row(rank)["ordinary"]
    row = candidate_row(rank, k)["ordinary"]
    return sum(x["logprob"] - b["logprob"] for x, b in zip(row, base)) / len(base)


def continuation_words(rank: str, ctx: str, k: int) -> list[str]:
    generations = RAW["rankings"][rank]["generations"][ctx][str(k)]
    return [first_word(g["continuation"]) for g in generations]


def fmt_k(value: int | None) -> str:
    return "none up to 500" if value is None else str(value)


def fmt_float(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def duration(seconds: float) -> str:
    whole_minutes = int(seconds // 60)
    remaining = seconds - whole_minutes * 60
    return f"{whole_minutes} min {remaining:.1f} s"


def make_plot() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.15), sharey=True)
    colors = {"gradient": "#2878B5", "direct": "#E87500"}
    markers = {"gradient": "o", "direct": "s"}
    for ax, ctx in zip(axes, CONTEXTS):
        for rank in ("gradient", "direct"):
            counts = [candidate_flips(rank, ctx, k) for k in KS]
            random_max = [max_control_flips(rank, ctx, k) for k in KS]
            ax.plot(KS, counts, color=colors[rank], marker=markers[rank],
                    linewidth=2.2, markersize=5, label=f"{RANK_NAMES[rank]} candidate")
            ax.plot(KS, random_max, color=colors[rank], marker=".", linestyle=":",
                    linewidth=1.5, markersize=7, alpha=0.85,
                    label=f"{RANK_NAMES[rank]} random max")
        ax.axhline(8, color="#666666", linestyle="--", linewidth=1, alpha=0.65)
        ax.set_xscale("log")
        ax.set_xticks(KS, [str(k) for k in KS])
        ax.set_ylim(-0.5, 16.5)
        ax.set_yticks([0, 4, 8, 12, 16])
        ax.set_title(CONTEXT_LABELS[ctx], fontsize=10)
        ax.set_xlabel("k neurons zeroed")
        ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    axes[0].set_ylabel("Held-out phrasings flipped (of 16)")
    legend_items = [
        Line2D([0], [0], color=colors["gradient"], marker=markers["gradient"], linewidth=2.2,
               label="Gradient candidate"),
        Line2D([0], [0], color=colors["direct"], marker=markers["direct"], linewidth=2.2,
               label="Direct-write candidate"),
        Line2D([0], [0], color=colors["gradient"], marker=".", linestyle=":", linewidth=1.5,
               label="Gradient random max"),
        Line2D([0], [0], color=colors["direct"], marker=".", linestyle=":", linewidth=1.5,
               label="Direct-write random max"),
        Line2D([0], [0], color="#666666", linestyle="--", linewidth=1,
               label="8-flip threshold"),
    ]
    fig.suptitle("Part 115b: next-token flips by ranking and opening", y=0.99, fontsize=14)
    fig.legend(handles=legend_items, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.12, 1, 0.93))
    fig.savefig(HERE / "fig-flips.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def build_ledger() -> str:
    command = f"cd {ROOT} && python results/part115b/plot115b.py"
    lines = [
        "# Part 115b number ledger",
        "",
        "The one-liner below recomputes the counts from the raw run and comparison files, checks the saved score against a fresh score, draws `fig-flips.png`, and writes this ledger.",
        "",
        f"One-liner: `{command}`",
        "",
        "## Run files and gates",
        "",
        f"- Claim: experimental scope and row dimensions. Output: Qwen/Qwen3.5-4B revision `{MODEL_REVISION}`, bf16 on A770; plain instruction `{RAW['plain_instruction']}`; rankings `{len(RAW['rankings'])}`; held-out phrasings `{N_HELDOUT}` per opening; ordinary probes `{len(RAW['ordinary'])}`; greedy continuation length `{GENERATION_TOKENS}` tokens; matching pool `{RAW['pool']}` nearest same-layer alternatives; top-k exclusion `{RAW['exclude']}`.",
        f"- Claim: run duration and loaded run settings. Output: `elapsed_s={RAW['elapsed_s']}`; duration `{duration(RAW['elapsed_s'])}`; direct load `{RAW['direct_load']}`; ks `{', '.join(map(str, KS))}`; matched sets per k and ranking `{RAW['n_sets']}`; target ids `{RAW['target_ids']};` all-ones gate `{RAW['gates']['ones_mask_bit_identical']}`; unzeroed targets first for both rankings `{RAW['gates']['gradient_targets_first_unzeroed']}` and `{RAW['gates']['direct_targets_first_unzeroed']}`.",
        f"- Claim: context target counts at baseline. Output: gradient `{ {ctx: str(sum(x['argmax_is_target'] for x in baseline_row('gradient')[ctx])) + '/' + str(N_HELDOUT) for ctx in CONTEXTS} }`; direct `{ {ctx: str(sum(x['argmax_is_target'] for x in baseline_row('direct')[ctx])) + '/' + str(N_HELDOUT) for ctx in CONTEXTS} }`.",
        f"- Claim: six ordinary probes' mean answer-token log-probability changes at k=500. Output: gradient `{fmt_float(ordinary_delta('gradient', 500))}` nats; direct-write `{fmt_float(ordinary_delta('direct', 500))}` nats.",
        "",
        "## Candidate flips and matched-control maxima",
        "",
        "One row is the maximum flip count among matched random sets at that k, across both rankings and all three openings. Candidate cells are recomputed from each raw row. Each individual ranking/opening's random maximum is included in the output column details.",
        "",
        "| k | Gradient plain | Gradient quote | Gradient bold | Direct plain | Direct quote | Direct bold | Overall random maximum |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for k in KS:
        vals = [candidate_flips(rank, ctx, k) for rank in ("gradient", "direct") for ctx in CONTEXTS]
        ctrl = {(rank, ctx): max_control_flips(rank, ctx, k)
                for rank in ("gradient", "direct") for ctx in CONTEXTS}
        overall_max = max(ctrl.values())
        details = ", ".join(f"{rank}/{ctx}={ctrl[(rank, ctx)]}" for rank in ("gradient", "direct") for ctx in CONTEXTS)
        lines.append(f"| {k} | " + " | ".join(f"{v}/{N_HELDOUT}" for v in vals) + f" | {overall_max}/{N_HELDOUT} |  <!-- controls: {details} -->")
    lines += ["", "## Thresholds, continuations, and registered verdicts", ""]
    for rank in ("gradient", "direct"):
        for ctx in CONTEXTS:
            ks_star = k_star(rank, ctx)
            flip_table = ", ".join(f"{k}:{candidate_flips(rank, ctx, k)}/{N_HELDOUT}" for k in KS)
            control_table = ", ".join(f"{k}:{max_control_flips(rank, ctx, k)}/{N_HELDOUT}" for k in KS)
            lines.append(f"- Claim: {RANK_NAMES[rank]} / {ctx} threshold and random maximum. Output: k* `{fmt_k(ks_star)}`; candidates `{flip_table}`; maxima `{control_table}`.")
    grad_plain = Counter(w for w in continuation_words("gradient", "plain", 500) if not w.startswith("Q"))
    direct_plain = RAW["rankings"]["direct"]["generations"]["plain"]["100"]
    exact_google_phrase = sum("a large language model, trained by Google" in g["continuation"] for g in direct_plain)
    direct_bold_words = Counter(continuation_words("direct", "bold", 100))
    lines.append(f"- Claim: continuation counts. Output: gradient/plain k=500 non-Q first words `{len([w for w in continuation_words('gradient', 'plain', 500) if not w.startswith('Q')])}/{N_HELDOUT}` with distribution `{dict(grad_plain)}`; direct/plain k=100 exact phrase `a large language model, trained by Google` appears in `{exact_google_phrase}/{N_HELDOUT}`; direct/bold k=100 first words `{dict(direct_bold_words)}`.")
    lines.append(f"- Claim: stored registered verdicts, verified against a fresh score. Output: `{RECOMPUTED_SCORE['verdicts']}`; withdrawals `{RECOMPUTED_SCORE['withdrawn']}`; saved score agrees `{RECOMPUTED_SCORE == SAVED_SCORE}`.")
    lines += ["", "## Comparison data and registration history", ""]
    old115 = json.loads((ROOT / "results/part115/score.json").read_text(encoding="utf-8"))
    old116 = json.loads((ROOT / "results/part116/score.json").read_text(encoding="utf-8"))
    lines.append(f"- Claim: Part 115 gradient result in its original `I am **` prompt. Output: k* `{old115['k_star']}`; at k=200 `{old115['per_k']['200']['candidate']['flips']}/16`; at k=500 `{old115['per_k']['500']['candidate']['flips']}/16`.")
    lines.append(f"- Claim: Part 116 direct-write result in its original `I am **` prompt. Output: k* `{old116['direct']['k_star']}`; at k=100 `{old116['direct']['per_k']['100']['candidate']['flips']}/16`.")
    git_cmd = ["git", "-C", str(ROOT), "log", "--format=%h %ci %s", "--", "PART115B-DESIGN.md"]
    git_out = subprocess.run(git_cmd, check=True, text=True, capture_output=True).stdout.strip()
    lines.append(f"- Claim: commits that registered and amended the design. One-liner: `git -C {ROOT} log --format='%h %ci %s' -- PART115B-DESIGN.md`. Output:\n\n```text\n{git_out}\n```")
    lines += ["", "The count of held-out phrasings, the number of ordinary probes, and the generation count in the README are taken from the raw result and design files; the one-liner above records their source with the other run details.", ""]
    return "\n".join(lines)


def main() -> None:
    make_plot()
    ledger = build_ledger()
    (HERE / "NUMBERS.md").write_text(ledger, encoding="utf-8")
    print("wrote results/part115b/fig-flips.png and NUMBERS.md")
    print("score.json agrees with a fresh score of results115b.json")
    for rank in ("gradient", "direct"):
        print(rank, {ctx: {"k_star": k_star(rank, ctx),
                           "flips": [candidate_flips(rank, ctx, k) for k in KS],
                           "random_max": [max_control_flips(rank, ctx, k) for k in KS]}
                     for ctx in CONTEXTS})
    print("verdicts", RECOMPUTED_SCORE["verdicts"])


if __name__ == "__main__":
    main()
