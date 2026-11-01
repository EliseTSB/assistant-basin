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
    for answer in answers:
        if answer.get("stage") == "battery":
            loops_by_row_ban[(row_key(answer), answer["ban"])] = bool(
                score166.loop_phrase(answer.get("answer", ""))
            )
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
