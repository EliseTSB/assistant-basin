#!/usr/bin/env python3
"""Rank Part 65 write words.
Reads steer65 per-unit rollouts and writes one ranked JSONL row per unit.
For each target, words are lowercased Unicode word tokens from generated text.
With target count c_t, token total N_t, pooled same-band control count c_c,
control total N_c, and vocabulary size V, define p_t=(c_t+0.5)/(N_t+0.5V),
p_c=(c_c+0.5)/(N_c+0.5V), and word-cluster mass as
(c_t/N_t) * max(0, log2(p_t/p_c)). A control target is left out of its own
pooled reference. The ten largest positive masses are the write words.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
from typing import Any


WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", flags=re.UNICODE)
SMOOTH = 0.5
TOP_WORDS = 10


class RankError(RuntimeError):
    """Malformed Part 65 steering rows or an empty matched control pool."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for n, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RankError(f"{path}:{n}: expected JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"),
                                       allow_nan=False) + "\n" for row in rows),
                    encoding="utf-8")
    temp.replace(path)


def layer_band(layer: int, layer_count: int = 28) -> str:
    if isinstance(layer, bool) or not isinstance(layer, int) or not 0 <= layer < layer_count:
        raise ValueError(f"layer must be an integer in 0..{layer_count - 1}")
    index = min(2, (layer * 3) // layer_count)
    return ("first", "middle", "last")[index]


def words(text: str) -> list[str]:
    return [match.group(0).casefold().replace("’", "'") for match in WORD_RE.finditer(text)]


def rollout_counter(row: dict[str, Any]) -> Counter[str]:
    rollouts = row.get("rollouts")
    if not isinstance(rollouts, list) or len(rollouts) != 9:
        raise RankError(f"{row.get('unit_key')}: expected exactly nine rollouts")
    counts: Counter[str] = Counter()
    for item in rollouts:
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            raise RankError(f"{row.get('unit_key')}: malformed rollout text")
        counts.update(words(item["text"]))
    return counts


def rank_rows(steer_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not steer_rows:
        raise RankError("steer file is empty")
    keyed: dict[str, dict[str, Any]] = {}
    counters: dict[str, Counter[str]] = {}
    pools: dict[str, Counter[str]] = {}
    totals: dict[str, int] = {}
    pool_units: dict[str, list[str]] = {}
    for row in steer_rows:
        key = row.get("unit_key")
        kind = row.get("type")
        layer = row.get("layer")
        if not isinstance(key, str) or kind not in ("neuron", "control") or key in keyed:
            raise RankError(f"invalid or duplicate unit key {key!r}")
        band = layer_band(layer)
        count = rollout_counter(row)
        keyed[key] = {**row, "layer_band": band}
        counters[key] = count
        if kind == "control":
            pools.setdefault(band, Counter()).update(count)
            totals[band] = totals.get(band, 0) + sum(count.values())
            pool_units.setdefault(band, []).append(key)
    output = []
    for key, row in keyed.items():
        band = row["layer_band"]
        target = counters[key]
        target_total = sum(target.values())
        if target_total <= 0:
            raise RankError(f"{key}: rollouts contain no word tokens")
        control = pools.get(band, Counter()).copy()
        control_total = totals.get(band, 0)
        if row["type"] == "control":
            control.subtract(target)
            control_total -= target_total
        if control_total <= 0:
            raise RankError(f"{key}: no pooled control tokens in {band} layer band")
        vocabulary = set(target) | set(control)
        size = len(vocabulary)
        scored = []
        for word, count in target.items():
            target_p = (count + SMOOTH) / (target_total + SMOOTH * size)
            control_count = max(0, control[word])
            control_p = (control_count + SMOOTH) / (control_total + SMOOTH * size)
            log2_lift = math.log2(target_p / control_p)
            mass = (count / target_total) * max(0.0, log2_lift)
            scored.append({"word": word, "count": count,
                               "target_rate": count / target_total,
                               "control_rate": control_count / control_total,
                               "log2_lift": log2_lift,
                               "word_cluster_mass": mass})
        scored.sort(key=lambda item: (-item["word_cluster_mass"], item["word"]))
        top = scored[:TOP_WORDS]
        lens = row.get("steer_lens")
        if not isinstance(lens, dict):
            raise RankError(f"{key}: missing steer_lens summary")
        positive = lens.get("+0.8", [])
        negative = lens.get("-0.8", [])
        if not isinstance(positive, list) or not isinstance(negative, list):
            raise RankError(f"{key}: malformed steer-lens token rows")
        output.append({
            "unit_key": key,
            "type": row["type"],
            "layer": row["layer"],
            "layer_band": band,
            "words": top,
            "write_words": [item["word"] for item in top],
            "steer_lens": {
                "positive": [item.get("token", "") for item in positive[:12]],
                "negative": [item.get("token", "") for item in negative[:12]],
            },
            "control_reference_token_count": control_total,
        })
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steer", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("ranked65.jsonl"))
    args = parser.parse_args(argv)
    rows = rank_rows(read_jsonl(args.steer))
    write_jsonl(args.out, rows)
    print(f"ranked {len(rows)} units -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
