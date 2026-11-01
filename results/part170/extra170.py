#!/usr/bin/env python3
"""Post-registration Part 170 exploratory summaries and blinded-read join.

Run from the repository root with the project's isolated Python environment.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results" / "part170"
OUT = DATA / "extra170.txt"
LAYERS = tuple(range(19, 27))
SLOTS = tuple(range(6))
PHRASINGS = tuple(f"p8:{i}" for i in range(8))
DOSES = (0.25, 0.5, 1.0)
BANS = ("b5", "b50", "b200")
REPS = 10_000
SEED = 170
WORD_RE = re.compile(r"[\w]+(?:['’][\w]+)*", re.UNICODE)


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def has_loop(answer: str) -> bool:
    words = [word.casefold() for word in WORD_RE.findall(answer if isinstance(answer, str) else "")]
    counts = Counter(tuple(words[i:i + 3]) for i in range(max(0, len(words) - 2)))
    return any(count >= 3 for count in counts.values())


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def pp(value: float) -> str:
    return f"{100.0 * value:+.2f} pp"


def ci(values) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high)


def ci_text(bounds) -> str:
    return f"[{pp(bounds[0])}, {pp(bounds[1])}]"


def main() -> None:
    answers = read_jsonl(DATA / "answers.jsonl")
    push = [row for row in answers if row.get("stage") == "push" and
            row.get("arm") == "n+" and not row.get("repeat") and row.get("group") == "p8"]
    no_push = [row for row in answers if row.get("stage") == "no_push" and
               row.get("group") == "p8" and float(row.get("dose", 0.0)) == 0.0]

    # Store paired list-minus-b0 loop outcomes on the common P8 units.
    push_by_key = {}
    for row in push:
        key = (float(row["dose"]), str(row["prompt_id"]), int(row["layer"]), int(row["slot"]), str(row["ban"]))
        push_by_key[key] = has_loop(row.get("answer", ""))
    zero_by_key = {(str(row["prompt_id"]), str(row["ban"])): has_loop(row.get("answer", ""))
                   for row in no_push}

    delta = {}
    d0 = {}
    for ban in BANS:
        delta[ban] = {}
        for dose in DOSES:
            # Each entry is paired list-minus-b0 on one prompt, slot, and dose.
            matrix = np.empty((len(LAYERS), len(SLOTS), len(PHRASINGS)), dtype=np.float64)
            for li, layer in enumerate(LAYERS):
                for si, slot in enumerate(SLOTS):
                    for pi, prompt_id in enumerate(PHRASINGS):
                        base_key = (dose, prompt_id, layer, slot, "b0")
                        alt_key = (dose, prompt_id, layer, slot, ban)
                        if base_key not in push_by_key or alt_key not in push_by_key:
                            raise ValueError(f"missing push pair: {dose} {prompt_id} {layer}/{slot} {ban}")
                        base = push_by_key[base_key]
                        alt = push_by_key[alt_key]
                        matrix[li, si, pi] = float(alt) - float(base)
            delta[ban][dose] = matrix
        for pi, prompt_id in enumerate(PHRASINGS):
            if (prompt_id, "b0") not in zero_by_key or (prompt_id, ban) not in zero_by_key:
                raise ValueError(f"missing no-push P8 pair for {prompt_id} {ban}")
        d0[ban] = np.asarray([float(zero_by_key[(pid, ban)]) - float(zero_by_key[(pid, "b0")])
                              for pid in PHRASINGS], dtype=np.float64)

    rng = np.random.default_rng(SEED)
    slot_draws = rng.integers(0, len(SLOTS), size=(REPS, len(LAYERS), len(SLOTS)))
    slot_weights = (slot_draws[..., None] == np.arange(len(SLOTS))).sum(axis=2).astype(np.float64)
    phrase_draws = rng.integers(0, len(PHRASINGS), size=(REPS, len(PHRASINGS)))
    d0_cluster_draws = np.stack([np.mean(d0[ban][phrase_draws], axis=1) for ban in BANS], axis=0)

    result_rows = {"D": [], "I": []}
    for bi, ban in enumerate(BANS):
        for dose in DOSES:
            matrix = delta[ban][dose]
            point = float(matrix.mean())
            slot_only_draws = np.einsum("rls,lsp->r", slot_weights, matrix) / (
                slot_weights.sum(axis=(1, 2)) * matrix.shape[2])
            # A shared phrasing draw is applied to the complete phrase cluster across all slots.
            cluster_matrix = matrix[:, :, phrase_draws].transpose(2, 0, 1, 3)
            slot_phrase_draws = np.einsum("rls,rlsp->r", slot_weights, cluster_matrix) / (
                slot_weights.sum(axis=(1, 2)) * len(PHRASINGS))
            result_rows["D"].append((ban, dose, point, ci(slot_only_draws), ci(slot_phrase_draws)))
            i_point = point - float(d0[ban].mean())
            i_slot = slot_only_draws - float(d0[ban].mean())
            i_cluster = slot_phrase_draws - d0_cluster_draws[bi]
            result_rows["I"].append((ban, dose, i_point, ci(i_slot), ci(i_cluster)))

    output = []
    output.append("# Part 170 exploratory additions (after registration; not registered)")
    output.append("")
    output.append("D(d) is the paired mean loop outcome under the list minus b0 on the same P8 units at that dose. D(0) is the no-push paired difference over the eight P8 phrasings. The first interval resamples whole neuron slots with replacement, stratified within each layer, and conditions on the fixed phrasing set. The second also resamples the eight phrasings as clusters, using the same phrase draw for D(d) and D(0). Both use 10,000 replicates and seed 170.")
    output.append("")
    output.append("D(0) by list")
    for ban in BANS:
        output.append(f"- {ban}: {pp(float(d0[ban].mean()))} over 8 phrasings; paired changes {int(np.sum(d0[ban] > 0))} induced and {int(np.sum(d0[ban] < 0))} ended.")
    output.append("")
    output.append("## Dose contrast D(d)")
    output.append("")
    output.append("| List | Dose | D(d) | Whole-slot 95% interval | Whole-slot plus phrasing-cluster 95% interval |")
    output.append("|---|---:|---:|---:|---:|")
    for ban, dose, point, slot_ci, cluster_ci in result_rows["D"]:
        output.append(f"| {ban} | {dose:g} | {pp(point)} | {ci_text(slot_ci)} | {ci_text(cluster_ci)} |")
    output.append("")
    output.append("## Interaction I(d) = D(d) − D(0) (exploratory; not registered)")
    output.append("")
    output.append("A slot-only interval conditions on the fixed prompt set, and D(0) rests on eight phrasings. The second interval resamples those phrasing clusters as well as slots. A failed bar is not evidence of additivity; these I(d) summaries were added after registration and have no registered additivity bar.")
    output.append("")
    output.append("| List | Dose | I(d) | Whole-slot 95% interval | Whole-slot plus phrasing-cluster 95% interval |")
    output.append("|---|---:|---:|---:|---:|")
    for ban, dose, point, slot_ci, cluster_ci in result_rows["I"]:
        output.append(f"| {ban} | {dose:g} | {pp(point)} | {ci_text(slot_ci)} | {ci_text(cluster_ci)} |")

    judged = read_jsonl(DATA / "hand_read_judged.jsonl")
    labels = read_jsonl(DATA / "hand_key" / "labels.jsonl")
    if len(judged) != len(labels) or {r["blind_id"] for r in judged} != {r["blind_id"] for r in labels}:
        raise ValueError("hand-read labels and judgments do not join one-to-one")
    by_id = {row["blind_id"]: row for row in judged}
    joined = [(label, by_id[label["blind_id"]]) for label in labels]
    rules = {judgment.get("rule", "") for _, judgment in joined}
    if len(rules) != 1:
        raise ValueError("hand-read judgment rule is not consistent across rows")
    output.append("")
    output.append("## Blinded hand read (joined by blind_id)")
    output.append("")
    output.append(f"Joined {len(joined)} labels and judgments. Rule: {next(iter(rules))}")
    output.append("")
    output.append("| sample_source | Coherent | Total |")
    output.append("|---|---:|---:|")
    sources = sorted({label["sample_source"] for label, _ in joined})
    for source in sources:
        rows = [(label, judgment) for label, judgment in joined if label["sample_source"] == source]
        coherent = sum(bool(judgment["coherent"]) for _, judgment in rows)
        output.append(f"| {source} | {coherent} | {len(rows)} |")
    output.append("")
    output.append("Short answer examples, selected from answers.jsonl")
    examples = [
        ("dose 1 b0", "push", "p8", 1.0, "b0", "p8:0", 20, 4),
        ("dose 1 b50", "push", "p8", 1.0, "b50", "p8:4", 21, 2),
        ("no-push b200", "no_push", "p8", 0.0, "b200", "p8:5", None, None),
    ]
    for label, stage, group, dose, ban, prompt_id, layer, slot in examples:
        matches = [row for row in answers if row.get("stage") == stage and row.get("group") == group and
                   float(row.get("dose", 0.0)) == dose and row.get("ban") == ban and
                   row.get("prompt_id") == prompt_id and row.get("layer") == layer and row.get("slot") == slot]
        if not matches:
            raise ValueError(f"required example is unavailable: {label}")
        answer = matches[0].get("answer", "").replace("\n", " ").strip()
        output.append(f"- {label}: {json.dumps(answer, ensure_ascii=False)}")

    OUT.write_text("\n".join(output) + "\n", encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size} bytes)")
    print("Hand read counts:")
    for source in sources:
        rows = [(label, judgment) for label, judgment in joined if label["sample_source"] == source]
        print(f"  {source}: {sum(bool(j['coherent']) for _, j in rows)}/{len(rows)}")
    print("Exploratory D and I estimates written with intervals.")


if __name__ == "__main__":
    main()
