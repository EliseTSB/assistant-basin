"""Recompute selected Part 169 write-up tables from saved results.

Run from the repository root with the project Python, for example:
  /home/elisepc/neuron-probe/.venv/bin/python -I -B results/part169/extra169.py

The script reads the saved score, masses, answers, review, and blinded hand-read
files. It writes only results/part169/extra169.txt.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))

from part166 import score166  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def row_key(row: dict) -> tuple:
    return (row["arm"], int(row["layer"]), int(row["slot"]), int(row["question_index"]))


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def main() -> None:
    score = json.loads((OUT / "score.json").read_text(encoding="utf-8"))
    review_text = (OUT / "review169.txt").read_text(encoding="utf-8")
    masses = read_jsonl(OUT / "masses.jsonl")
    answers = read_jsonl(OUT / "answers.jsonl")
    judged = read_jsonl(OUT / "hand_read_judged.jsonl")
    key_rows = read_jsonl(OUT / "hand_read_key.jsonl")

    lines: list[str] = [
        "Part 169 computed summaries",
        "Computed by extra169.py from the saved result files; loop labels use score166.loop_phrase.",
        "",
        "Pooled n+ and k4 outcomes (computed from score.json per_arm_ban):",
        "ban      rows  loops  loop rate  broken  joint success  mean length  EOS share",
    ]
    arm_data = score["per_arm_ban"]
    for ban in ("none", "own", "free", "clear", "plain50", "hit"):
        n = loops = broken = joint = eos = token_sum = 0.0
        for arm in ("n+", "k4"):
            cell = arm_data[arm][ban]
            count = int(cell["n"])
            n += count
            loops += round(cell["loop_rate"] * count)
            broken += round(cell["broken_rate"] * count)
            joint += round(cell["joint_success_rate"] * count)
            eos += round(cell["eos_rate"] * count)
            token_sum += cell["mean_tokens"] * count
        lines.append(
            f"{ban:8s} {int(n):4d} {int(loops):6d} {pct(loops / n):>10s} "
            f"{pct(broken / n):>7s} {pct(joint / n):>14s} "
            f"{token_sum / n:11.2f} {pct(eos / n):>10s}"
        )

    lines.extend(
        [
            "",
            "Loop churn counts (from score.json; net is induced minus ended):",
            "ban      ended  induced  net loop change",
        ]
    )
    for ban in ("free", "clear", "own", "plain50"):
        values = score["ended_induced_net"][ban]
        ended = int(values["ended"])
        induced = int(values["induced"])
        lines.append(f"{ban:8s} {ended:6d} {induced:8d} {induced - ended:16d}")

    hit_keys = {
        row_key(m)
        for m in masses
        if m.get("lists", {}).get("hit", {}).get("ids")
    }
    loops_by_row_ban: dict[tuple, bool] = {}
    answer_by_row_ban: dict[tuple, dict] = {}
    for answer in answers:
        if answer.get("stage") == "battery":
            answer_key = (row_key(answer), answer["ban"])
            loops_by_row_ban[answer_key] = bool(
                score166.loop_phrase(answer.get("answer", ""))
            )
            answer_by_row_ban[answer_key] = answer
    lines.extend(
        [
            "",
            "Same hit-defined rows (only rows with a nonempty hit list; all three bans use those same row keys):",
            "arm      rows  none loops  free loops  hit loops",
        ]
    )
    for arm in ("n+", "k4"):
        these = {key for key in hit_keys if key[0] == arm}
        counts = {
            ban: sum(loops_by_row_ban.get((key, ban), False) for key in these)
            for ban in ("none", "free", "hit")
        }
        lines.append(
            f"{arm:8s} {len(these):4d} {counts['none']:11d} "
            f"{counts['free']:10d} {counts['hit']:10d}"
        )
    pooled_counts = {
        ban: sum(loops_by_row_ban.get((key, ban), False) for key in hit_keys)
        for ban in ("none", "free", "hit")
    }
    lines.append(
        f"{'pooled':8s} {len(hit_keys):4d} {pooled_counts['none']:11d} "
        f"{pooled_counts['free']:10d} {pooled_counts['hit']:10d}"
    )

    plain2_keys = {
        row_key(m)
        for m in masses
        if int(m.get("lists", {}).get("free", {}).get("plain_overlap", 0)) >= 2
    }
    plain2_eligible = [
        key for key in plain2_keys if not loops_by_row_ban.get((key, "none"), False)
    ]
    plain2_induced = {
        ban: sum(loops_by_row_ban.get((key, ban), False) for key in plain2_eligible)
        for ban in ("free", "hit")
    }
    plain2_totals = {
        ban: sum(loops_by_row_ban.get((key, ban), False) for key in plain2_keys)
        for ban in ("none", "free", "hit")
    }
    lines.extend(
        [
            "",
            "169.8 on rows with at least two free-list plain tokens:",
            f"eligible induced denominator {len(plain2_eligible)}/{len(plain2_keys)}; "
            f"free {plain2_induced['free']} ({pct(plain2_induced['free'] / len(plain2_eligible))}); "
            f"hit {plain2_induced['hit']} ({pct(plain2_induced['hit'] / len(plain2_eligible))}).",
            "same 528-row set loop counts: "
            + ", ".join(f"{ban} {plain2_totals[ban]}" for ban in ("none", "free", "hit"))
            + ".",
        ]
    )

    decile_re = re.compile(
        r"^tenth\s+(\d+)\s+own mass\s+(\S+)\s+to\s+(\S+)\s+"
        r"rows\s+(\d+)\s+none\s+(\d+)\s+own\s+(\d+)\s+free\s+(\d+)"
    )
    deciles = []
    for line in review_text.splitlines():
        match = decile_re.match(line)
        if match:
            idx, low, high, rows, none, own, free = match.groups()
            deciles.append(
                {
                    "tenth": int(idx),
                    "range": f"{low}–{high}",
                    "rows": int(rows),
                    "none": int(none),
                    "own": int(own),
                    "free": int(free),
                }
            )
    if len(deciles) != 10:
        raise ValueError(f"Expected 10 own-mass tenths in review169.txt, found {len(deciles)}")
    lines.extend(
        [
            "",
            "Own-mass tenths (counts from review169.txt; own cut is none minus own):",
            "tenth  own-mass range  rows  none  own  free  own cut",
        ]
    )
    for item in deciles:
        cut = item["none"] - item["own"]
        lines.append(
            f"{item['tenth']:5d}  {item['range']:14s} {item['rows']:4d} "
            f"{item['none']:5d} {item['own']:4d} {item['free']:5d} {cut:8d}"
        )
    total_cut = sum(item["none"] - item["own"] for item in deciles)
    top_two_cut = sum(item["none"] - item["own"] for item in deciles if item["tenth"] >= 8)
    lines.append(
        f"Top two tenths' share of the own cut: {top_two_cut}/{total_cut} = "
        f"{pct(top_two_cut / total_cut)}."
    )

    all_row_keys = {row_key(m) for m in masses}
    top_two_keys = {
        row_key(m)
        for m in sorted(masses, key=lambda item: item["lists"]["own"]["mass"])[
            8 * len(masses) // 10 :
        ]
    }
    top_two_none = sum(loops_by_row_ban.get((key, "none"), False) for key in top_two_keys)
    top_two_own = sum(loops_by_row_ban.get((key, "own"), False) for key in top_two_keys)
    all_none = sum(loops_by_row_ban.get((key, "none"), False) for key in all_row_keys)
    all_own = sum(loops_by_row_ban.get((key, "own"), False) for key in all_row_keys)
    other_keys = all_row_keys - top_two_keys
    other_none = sum(loops_by_row_ban.get((key, "none"), False) for key in other_keys)
    other_own = sum(loops_by_row_ban.get((key, "own"), False) for key in other_keys)
    top_two_cut_count = top_two_none - top_two_own
    all_cut_count = all_none - all_own
    other_cut_count = other_none - other_own
    equal_rate_top_two_expected = top_two_none * all_cut_count / all_none
    lines.extend(
        [
            f"Top two tenths: {top_two_none}/{all_none} = {pct(top_two_none / all_none)} of none loops; "
            f"{top_two_cut_count}/{all_cut_count} = {pct(top_two_cut_count / all_cut_count)} of net own cut.",
            f"Relative own cut: top two {pct(top_two_cut_count / top_two_none)}, "
            f"other eight {pct(other_cut_count / other_none)}.",
            f"If each tenth lost the same share of its none loops ({all_cut_count}/{all_none}), "
            f"the top two's {top_two_none} none loops imply {equal_rate_top_two_expected:.1f}, "
            f"about {round(equal_rate_top_two_expected)}, of the {all_cut_count} net cut.",
        ]
    )

    high_keys = {row_key(m) for m in masses if m.get("high_row")}
    high_none = sum(loops_by_row_ban.get((key, "none"), False) for key in high_keys)
    high_own = sum(loops_by_row_ban.get((key, "own"), False) for key in high_keys)
    other_high_none = sum(
        loops_by_row_ban.get((key, "none"), False) for key in all_row_keys - high_keys
    )
    other_high_own = sum(
        loops_by_row_ban.get((key, "own"), False) for key in all_row_keys - high_keys
    )
    lines.append(
        f"Own relative cut by high-row status: high {high_none - high_own}/{high_none} "
        f"({pct((high_none - high_own) / high_none)}); other "
        f"{other_high_none - other_high_own}/{other_high_none} "
        f"({pct((other_high_none - other_high_own) / other_high_none)})."
    )
    paired_keys = {row_key(m) for m in masses if m.get("paired_set")}
    for label, keys in (
        ("high paired", high_keys & paired_keys),
        ("high unpaired", high_keys - paired_keys),
    ):
        none_count = sum(loops_by_row_ban.get((key, "none"), False) for key in keys)
        free_count = sum(loops_by_row_ban.get((key, "free"), False) for key in keys)
        eligible_count = sum(not loops_by_row_ban.get((key, "none"), False) for key in keys)
        induced_count = sum(
            loops_by_row_ban.get((key, "free"), False)
            and not loops_by_row_ban.get((key, "none"), False)
            for key in keys
        )
        lines.append(
            f"{label}: {len(keys)} rows, none {none_count}, free {free_count}; "
            f"free induced {induced_count}/{eligible_count} "
            f"({pct(induced_count / eligible_count)})."
        )

    phrase_cuts = []
    for phrasing in sorted({key[3] for key in all_row_keys}):
        keys = {key for key in all_row_keys if key[3] != phrasing}
        none_count = sum(loops_by_row_ban.get((key, "none"), False) for key in keys)
        own_count = sum(loops_by_row_ban.get((key, "own"), False) for key in keys)
        phrase_cuts.append((none_count - own_count) / none_count)
    arm_cuts = {}
    for arm in ("n+", "k4"):
        keys = {key for key in all_row_keys if key[0] == arm}
        none_count = sum(loops_by_row_ban.get((key, "none"), False) for key in keys)
        own_count = sum(loops_by_row_ban.get((key, "own"), False) for key in keys)
        arm_cuts[arm] = (none_count - own_count) / none_count
    lines.append(
        f"Own cut leaving out one phrasing: {pct(min(phrase_cuts))}–{pct(max(phrase_cuts))}; "
        f"arms alone n+ {pct(arm_cuts['n+'])}, k4 {pct(arm_cuts['k4'])}."
    )

    def length_bin(token_count: int) -> int:
        return 0 if token_count < 30 else 1 if token_count < 45 else 2 if token_count < 60 else 3 if token_count < 75 else 4

    none_cell_counts: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
    for key in all_row_keys:
        answer = answer_by_row_ban[(key, "none")]
        cell = (
            length_bin(int(answer["token_count"])),
            bool(answer.get("ended_eos_before_80")),
        )
        none_cell_counts[cell][0] += int(loops_by_row_ban[(key, "none")])
        none_cell_counts[cell][1] += 1

    def expected_from_none_cells(ban: str) -> float:
        expected = 0.0
        for key in all_row_keys:
            answer = answer_by_row_ban[(key, ban)]
            cell = (
                length_bin(int(answer["token_count"])),
                bool(answer.get("ended_eos_before_80")),
            )
            loop_count, row_count = none_cell_counts[cell]
            expected += loop_count / row_count
        return expected

    own_expected = expected_from_none_cells("own")
    plain50_expected = expected_from_none_cells("plain50")
    plain50_observed = sum(
        loops_by_row_ban.get((key, "plain50"), False) for key in all_row_keys
    )
    plain50_none = sum(loops_by_row_ban.get((key, "none"), False) for key in all_row_keys)
    lines.extend(
        [
            "",
            "Length and EOS standardization (COLD method: none loop rate by none-answer length bin × EOS cell, "
            "applied to each ban's answer cells):",
            f"own observed {all_own}, expected {own_expected:.1f}, residual {all_own - own_expected:.1f} "
            f"of net change {all_own - all_none}.",
            f"plain50 observed {plain50_observed}, expected {plain50_expected:.1f}; "
            f"cell mix accounts for {plain50_expected - plain50_none:.1f} of the "
            f"{plain50_observed - plain50_none} net rise, residual {plain50_observed - plain50_expected:.1f}.",
        ]
    )

    plain50_by_phrasing = {}
    for phrasing in sorted({key[3] for key in all_row_keys}):
        keys = {key for key in all_row_keys if key[3] == phrasing}
        plain50_by_phrasing[phrasing] = (
            sum(loops_by_row_ban.get((key, "plain50"), False) for key in keys)
            - sum(loops_by_row_ban.get((key, "none"), False) for key in keys)
        )
    lines.append(
        f"plain50 exceeds none in {sum(value > 0 for value in plain50_by_phrasing.values())}/"
        f"{len(plain50_by_phrasing)} phrasings; changes by phrasing: {plain50_by_phrasing}."
    )

    batch_noise = {}
    for ban in ("own", "free", "clear"):
        never = [
            key
            for key in all_row_keys
            if not answer_by_row_ban[(key, ban)].get("blocked_steps")
        ]
        differs = [
            key
            for key in never
            if answer_by_row_ban[(key, ban)].get("token_ids")
            != answer_by_row_ban[(key, "none")].get("token_ids")
        ]
        batch_noise[ban] = (len(differs), len(never), len(differs) / len(never))
    lines.append(
        "Batch-shape noise among never-blocked answers: "
        + "; ".join(
            f"{ban} {n}/{den} ({pct(rate)})"
            for ban, (n, den, rate) in batch_noise.items()
        )
        + "."
    )

    blind_to_ban = {item["blind_id"]: item for item in judged}
    key_by_id = {item["blind_id"]: item for item in key_rows}
    if len(judged) != 144 or len(key_rows) != 144 or set(blind_to_ban) != set(key_by_id):
        raise ValueError("The blinded hand-read file and key must each contain the same 144 IDs")
    hand_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    arm_counts: dict[str, set[str]] = defaultdict(set)
    rules = {item["rule"] for item in judged}
    if len(rules) != 1:
        raise ValueError("Hand-read records do not share one consistent rule")
    for blind_id, item in blind_to_ban.items():
        ban = key_by_id[blind_id]["ban"]
        hand_counts[ban][0] += int(bool(item["coherent"]))
        hand_counts[ban][1] += 1
        arm_counts[ban].add(key_by_id[blind_id]["arm"])
    lines.extend(
        [
            "",
            "Blinded hand read (coherent count / sample; all sampled rows are n+ or k4):",
            "ban      coherent / sampled",
        ]
    )
    for ban in ("none", "own", "free", "clear", "plain50", "hit"):
        coherent, sample_n = hand_counts[ban]
        if sample_n != 24:
            raise ValueError(f"Expected 24 hand-read rows for {ban}, found {sample_n}")
        lines.append(f"{ban:8s} {coherent:3d} / {sample_n:2d}")
    lines.append(
        "Rule: "
        + next(iter(rules))
    )
    lines.append(
        "The hit sample is drawn from hit-eligible rows; its sample is not a direct comparison with the other bans."
    )

    result = "\n".join(lines) + "\n"
    (OUT / "extra169.txt").write_text(result, encoding="utf-8")
    print(result, end="")


if __name__ == "__main__":
    main()
