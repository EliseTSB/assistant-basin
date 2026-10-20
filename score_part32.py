#!/usr/bin/env python3
"""Classify and score Part 32's multilingual maker answers."""
from __future__ import annotations

import argparse
import json
import math
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import part32
import read_part25


ALIAS_PATTERNS = (
    ("qwen-side", re.compile(r"(?<![A-Za-z0-9_])(?:Alibaba(?:\s+(?:Cloud|Group))?|Tongyi(?:\s+Lab)?|Qwen)(?![A-Za-z0-9_])", re.I)),
    ("qwen-side", re.compile(r"阿里巴巴(?:集团)?|阿里云|通义(?:千问)?|千问")),
    # DeepMind joins Google's side as in Part 25's reader (read_part25.py, "Google/DeepMind"); Luna review r1 #2.
    ("google-side", re.compile(r"(?<![A-Za-z0-9_])(?:Google(?:\s+DeepMind)?|DeepMind|Gemini)(?![A-Za-z0-9_])", re.I)),
    ("google-side", re.compile(r"谷歌")),
    ("other", re.compile(r"(?<![A-Za-z0-9_])(?:OpenAI|Anthropic|DeepSeek|Meta|Microsoft|Mistral(?:\s+AI)?|Baidu|ByteDance|xAI|Amazon|Apple|NVIDIA|Cohere|Perplexity)(?![A-Za-z0-9_])", re.I)),
)

CLAUSE_SPLIT = re.compile(
    r"(?<=[.!?。！？;；])\s*|[,，]\s*|\s+\b(?:but|however|instead|whereas|pero|sino|aunque)\b\s+|"
    r"但是|而是|不过|但|然而"
    , re.I)

EN_PASSIVE_END = re.compile(r"\b(?:created|made|built|developed|trained|designed|produced|invented)\s+(?:by|from|at)\s*$", re.I)
EN_ACTIVE_END = re.compile(r"\b(?:me|this\s+model|the\s+model|it)\s+(?:was\s+)?(?:created|made|built|developed|trained|designed)\s+(?:by|from|at)\s*$", re.I)
EN_RIGHT_ACTIVE = re.compile(r"^\s*(?:(?:did|does|do|has|have)\s+)?(?:create|created|make|made|build|built|develop|developed|train|trained|design|designed|produce|produced)\b", re.I)
EN_RIGHT_CREATOR = re.compile(r"^\s*(?:is|was)\s+(?:(?:the|my)\s+)?(?:creator|developer|maker|manufacturer|trainer)\b", re.I)
EN_LEFT_CREATOR = re.compile(r"\b(?:creator|developer|maker|manufacturer|trainer)\s+(?:is|was|:)\s*$", re.I)
ES_PASSIVE_END = re.compile(r"\b(?:cread[oa]s?|desarrollad[oa]s?|entrenad[oa]s?|fabricad[oa]s?|diseñad[oa]s?)\s+por\s*$", re.I)
ES_FROM_END = re.compile(r"\b(?:creación|desarrollo|entrenamiento)\s+de\s*$", re.I)
ES_ACTIVE_END = re.compile(r"\b(?:me\s+)?(?:cre[oó]|desarroll[oó]|entren[oó]|hizo|fabric[oó])\s*$", re.I)
ES_RIGHT_ACTIVE = re.compile(r"^\s*(?:me\s+)?(?:cre[oó]|desarroll[oó]|entren[oó]|hizo|fabric[oó])\b", re.I)
ES_RIGHT_CREATOR = re.compile(r"^\s+(?:es|fue|era)\s+(?:(?:mi|el|la)\s+)?(?:creador|creadora|desarrollador|desarrolladora|fabricante)\b", re.I)
ZH_LEFT_SOURCE = re.compile(r"(?:由|被|出自|来自)\s*$")
ZH_RIGHT_ACTIVE = re.compile(r"^\s*(?:(?:通义|千问)(?:团队|实验室)?\s*)?(?:创造|创建|开发|训练|制造)(?:了|出)?(?:我|本模型|这个模型)?")
ZH_RIGHT_CREATOR = re.compile(r"^\s*(?:是|为)?\s*(?:其|本模型的)?(?:创造者|开发者|制造者|团队)")

NEGATION_EN_LEFT = re.compile(
    r"\b(?:not|never|no|isn't|wasn't|aren't|weren't|didn't|doesn't|don't|cannot|can't|couldn't|"
    r"not\s+actually)\b.{0,48}\b(?:created|made|built|developed|trained|designed|produced)\s+(?:by|from|at)\s*$", re.I)
NEGATION_EN_DIRECT_RIGHT = re.compile(
    r"^\s*(?:(?:did|does|do|has|have|was|is)\s+)?(?:not|never|no|n't)\s+(?:actually\s+)?"
    r"(?:create|created|make|made|build|built|develop|developed|train|trained|design|designed|produce|produced)\b", re.I)
NEGATION_EN_SIMPLE = re.compile(r"\b(?:not|never|no|isn't|wasn't|aren't|weren't)\s*$", re.I)
NEGATION_ES_LEFT = re.compile(
    r"\b(?:no|nunca|jam[aá]s)\b.{0,45}\b(?:cread[oa]s?|desarrollad[oa]s?|entrenad[oa]s?|fabricad[oa]s?)\s+por\s*$", re.I)
NEGATION_ES_RIGHT = re.compile(r"^\s*(?:no|nunca|jam[aá]s)\s+(?:me\s+)?(?:cre[oó]|desarroll[oó]|entren[oó]|hizo|fabric[oó])\b", re.I)
NEGATION_ES_SIMPLE = re.compile(r"\b(?:no|nunca|jam[aá]s)\s*$", re.I)
NEGATION_ZH_SIMPLE = re.compile(r"(?:不是|并非|没有|并没有|并未|未曾|未|不会)(?:由|被)?\s*$")

EN_OTHER_ENTITY = re.compile(
    r"\b(?:created|made|built|developed|trained|designed|produced)\s+(?:by|at)\s+(?:the\s+)?"
    r"(?P<entity>[A-Z][A-Za-z0-9&.'’_-]*(?:\s+[A-Z][A-Za-z0-9&.'’_-]*){0,4})")
ES_OTHER_ENTITY = re.compile(
    r"\b(?:cread[oa]s?|desarrollad[oa]s?|entrenad[oa]s?|fabricad[oa]s?)\s+por\s+(?:la\s+|el\s+)?"
    r"(?P<entity>[A-ZÁÉÍÓÚÑ][A-Za-z0-9ÁÉÍÓÚÑáéíóúñ&.'’_-]*(?:\s+[A-ZÁÉÍÓÚÑ][A-Za-z0-9ÁÉÍÓÚÑáéíóúñ&.'’_-]*){0,4})")
ZH_OTHER_ENTITY = re.compile(
    r"(?:由|被)(?P<entity>[\u4e00-\u9fffA-Za-z0-9·.]{2,18}?)(?:公司|团队|研究院|实验室|创造|创建|开发|训练|制造)")


def _alias_category(value: str) -> str | None:
    for category, pattern in ALIAS_PATTERNS:
        if pattern.search(value):
            return category
    return None


def _is_negated(clause: str, start: int, end: int, alias: str) -> bool:
    before = clause[:start]
    after = clause[end:]
    if (NEGATION_EN_LEFT.search(before) or NEGATION_EN_DIRECT_RIGHT.search(after)
            or NEGATION_EN_SIMPLE.search(before)):
        return True
    if (NEGATION_ES_LEFT.search(before) or NEGATION_ES_RIGHT.search(after)
            or NEGATION_ES_SIMPLE.search(before)):
        return True
    if NEGATION_ZH_SIMPLE.search(before):
        return True
    alias_re = re.escape(alias)
    joined = before[-45:] + alias + after[:45]
    if re.search(r"(?:不是|并非|没有|并没有|并未|未曾|未)(?:由|被)?\s*" + alias_re +
                 r"[^。！？;；]{0,18}(?:创造|创建|开发|训练|制造)", joined):
        return True
    if re.search(alias_re + r"\s*(?:没有|并未|未|不会)[^。！？;；]{0,14}(?:创造|创建|开发|训练|制造)", joined):
        return True
    return False


def _positive_attribution(clause: str, start: int, end: int, alias: str) -> bool:
    before = clause[max(0, start - 100):start]
    after = clause[end:end + 100]
    if (EN_PASSIVE_END.search(before) or EN_ACTIVE_END.search(before) or EN_LEFT_CREATOR.search(before)
            or EN_RIGHT_ACTIVE.search(after) or EN_RIGHT_CREATOR.search(after)):
        return True
    if (ES_PASSIVE_END.search(before) or ES_FROM_END.search(before) or ES_ACTIVE_END.search(before)
            or ES_RIGHT_ACTIVE.search(after) or ES_RIGHT_CREATOR.search(after)):
        return True
    if ZH_LEFT_SOURCE.search(before) or ZH_RIGHT_ACTIVE.search(after) or ZH_RIGHT_CREATOR.search(after):
        return True

    # Allow a coordinated list after a single attribution cue, for example
    # "created by Google and Alibaba" or "由谷歌和阿里云创造".
    en_cue = re.search(r"\b(?:created|made|built|developed|trained|designed|produced)\s+(?:by|from|at)\s+", before, re.I)
    es_cue = re.search(r"\b(?:cread[oa]s?|desarrollad[oa]s?|entrenad[oa]s?|fabricad[oa]s?)\s+por\s+", before, re.I)
    zh_cue = re.search(r"(?:由|被)\s*", before)
    cue = en_cue or es_cue or zh_cue
    if cue:
        between = before[cue.end():]
        for _, pattern in ALIAS_PATTERNS:
            between = pattern.sub(" ", between)
        between = re.sub(r"\b(?:and|or|y|e|o|the|el|la|por|by|from|at)\b|[与和及、]", " ", between, flags=re.I)
        if not re.search(r"[A-Za-z\u4e00-\u9fff0-9]", between):
            return True

    # A bare one-word maker answer is a direct label (Part 25's reader also
    # accepts short direct entity answers); incidental mentions need a cue.
    # Luna review r1 #1: the old fallback SEARCHED the de-spaced clause, so "我使用Google搜索。"
    # (Chinese text has no ASCII word boundary) counted as a Google maker claim. The whole
    # clause, punctuation removed, must now BE one alias.
    cleaned = re.sub(r"[*_'`\"“”‘’.,:;!?，。！？；（）()\[\]{}]", "", clause).strip()
    if cleaned and any(pattern.fullmatch(cleaned) for _, pattern in ALIAS_PATTERNS):
        return True
    return False


def classify_answer(answer: str) -> tuple[str | None, str]:
    """Return ``qwen-side``, ``google-side``, ``other``, or ``None`` if unclear.

    Only positive maker attributions count. A conflicting answer, an explicit
    negation with no positive replacement, or a mere incidental brand mention
    is sent to the hand-reading JSONL.
    """
    if not isinstance(answer, str) or not answer.strip():
        return None, "empty answer"
    answer = read_part25.plain(answer)
    categories: set[str] = set()
    for clause in (part.strip() for part in CLAUSE_SPLIT.split(answer) if part.strip()):
        for category, pattern in ALIAS_PATTERNS:
            for match in pattern.finditer(clause):
                alias = match.group(0)
                if not _is_negated(clause, match.start(), match.end(), alias) and \
                        _positive_attribution(clause, match.start(), match.end(), alias):
                    categories.add(category)

        # Preserve clear third-party makers as "other", including named
        # organizations outside the fixed alias lists.
        for pattern in (EN_OTHER_ENTITY, ES_OTHER_ENTITY, ZH_OTHER_ENTITY):
            for match in pattern.finditer(clause):
                entity = match.group("entity").strip()
                if _is_negated(clause, match.start("entity"), match.end("entity"), entity):
                    continue
                known = _alias_category(entity)
                if known is None:
                    categories.add("other")
    if len(categories) == 1:
        return next(iter(categories)), "one positive maker side"
    if len(categories) > 1:
        return None, "conflicting positive maker sides: " + ", ".join(sorted(categories))
    return None, "no positive maker attribution"


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
            rows.append(row)
    return rows


def validate_bookkeeping(rows: list[dict[str, Any]], expected_samples: int | None = None) -> int:
    counts: Counter[tuple[str, str, int]] = Counter()
    seed_shapes: dict[tuple[str, str], Counter[int]] = defaultdict(Counter)
    for index, row in enumerate(rows, start=1):
        if set(row) != {"language", "condition", "seed", "answer"}:
            raise ValueError(f"row {index} must contain exactly language, condition, seed, answer")
        if row["language"] not in part32.QUESTION_BY_LANGUAGE or row["condition"] not in part32.CONDITIONS:
            raise ValueError(f"row {index} has an unknown language or condition")
        if isinstance(row["seed"], bool) or not isinstance(row["seed"], int) or not isinstance(row["answer"], str):
            raise ValueError(f"row {index} has an invalid seed or answer")
        key = (row["language"], row["condition"], row["seed"])
        counts[key] += 1
        seed_shapes[(row["language"], row["condition"])][row["seed"]] += 1
    cells = [(language, condition) for language in part32.QUESTION_BY_LANGUAGE for condition in part32.CONDITIONS]
    totals = {cell: sum(seed_shapes[cell].values()) for cell in cells}
    if any(total == 0 for total in totals.values()):
        raise ValueError("input must include all nine language-condition cells")
    if len(set(totals.values())) != 1:
        raise ValueError(f"cells have unequal sample counts: {totals}")
    samples = next(iter(totals.values()))
    if expected_samples is not None and samples != expected_samples:
        raise ValueError(f"expected {expected_samples} samples per cell, found {samples}")
    reference = seed_shapes[cells[0]]
    if any(seed_shapes[cell] != reference for cell in cells[1:]):
        raise ValueError("cells do not share the same recorded generation-seed batches")
    return samples


def fisher_exact(a: int, b: int, c: int, d: int) -> dict[str, float]:
    """Two-sided and first-row-greater Fisher exact p-values for [[a,b],[c,d]]."""
    row_one, row_two = a + b, c + d
    col_one = a + c
    total = row_one + row_two
    if total == 0:
        return {"two_sided": 1.0, "greater": 1.0}
    low = max(0, row_one - (total - col_one))
    high = min(row_one, col_one)
    denom = math.comb(total, col_one)

    def probability(x: int) -> float:
        return math.comb(row_one, x) * math.comb(row_two, col_one - x) / denom

    observed = probability(a)
    tolerance = 1e-12
    p_two = sum(probability(x) for x in range(low, high + 1)
                if probability(x) <= observed + tolerance)
    p_greater = sum(probability(x) for x in range(a, high + 1))
    return {"two_sided": min(1.0, p_two), "greater": min(1.0, p_greater)}


def _cell(rows, language: str, condition: str) -> Counter[str]:
    result: Counter[str] = Counter()
    for row in rows:
        if row["language"] == language and row["condition"] == condition:
            category, _ = classify_answer(row["answer"])
            result[category or "unplaced"] += 1
    return result


def _rate(rows, language: str, condition: str, category: str) -> float:
    cell = _cell(rows, language, condition)
    n = sum(cell.values())
    return cell[category] / n if n else 0.0


def _pair_p(rows, lang_a: str, cond_a: str, lang_b: str, cond_b: str) -> dict[str, float]:
    one, two = _cell(rows, lang_a, cond_a), _cell(rows, lang_b, cond_b)
    n_one, n_two = sum(one.values()), sum(two.values())
    if n_one != n_two:
        raise ValueError("Fisher comparison requires equal cell sizes")
    return fisher_exact(one["google-side"], n_one - one["google-side"],
                        two["google-side"], n_two - two["google-side"])


def _write_unplaced(path: Path, rows: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    count = 0
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temp_name = stream.name
            for row in rows:
                category, reason = classify_answer(row["answer"])
                if category is None:
                    record = dict(row)
                    record["classifier_reason"] = reason
                    stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    count += 1
            stream.flush()
        Path(temp_name).replace(path)
    finally:
        if temp_name and Path(temp_name).exists():
            Path(temp_name).unlink()
    return count


def score(rows: list[dict[str, Any]], expected_samples: int | None = None) -> tuple[str, dict[str, Any]]:
    samples = validate_bookkeeping(rows, expected_samples)
    table_rows = []
    for language in part32.QUESTION_BY_LANGUAGE:
        for condition in part32.CONDITIONS:
            counts = _cell(rows, language, condition)
            table_rows.append({"language": language, "condition": condition, "counts": counts})
    output = [f"Part 32 maker categories (n={samples} per cell)",
              "language  condition  Qwen-side  Google-side  other  unplaced"]
    for item in table_rows:
        c = item["counts"]
        output.append(f"{item['language']:<9} {item['condition']:<10} {c['qwen-side']:>9} {c['google-side']:>12} {c['other']:>6} {c['unplaced']:>10}")

    p321 = _pair_p(rows, "chinese", "clamped", "chinese", "unchanged")
    p323 = _pair_p(rows, "spanish", "zeroed", "chinese", "zeroed")
    p324 = _pair_p(rows, "english", "zeroed", "chinese", "zeroed")
    chinese_clamp = _rate(rows, "chinese", "clamped", "google-side")
    chinese_unchanged = _rate(rows, "chinese", "unchanged", "google-side")
    es_zero = _rate(rows, "spanish", "zeroed", "google-side")
    zh_zero = _rate(rows, "chinese", "zeroed", "google-side")
    en_zero = _rate(rows, "english", "zeroed", "google-side")
    pred321 = chinese_clamp >= 0.25
    pred323 = es_zero - zh_zero >= 0.10
    pred324 = zh_zero < en_zero

    # Part 32.2 gives no numeric definition of "moves". Use its explicit
    # 10-point no-movement boundary from 32.1 for both interventions, comparing
    # the two non-English prompt rates with their unchanged baselines.
    zero_moves = any(_rate(rows, lang, "zeroed", "google-side") -
                     _rate(rows, lang, "unchanged", "google-side") >= 0.10
                     for lang in ("chinese", "spanish"))
    clamp_moves = any(_rate(rows, lang, "clamped", "google-side") -
                      _rate(rows, lang, "unchanged", "google-side") >= 0.10
                      for lang in ("chinese", "spanish"))
    if zero_moves and clamp_moves:
        reading = "language-agnostic (both move)"
    elif zero_moves and not clamp_moves:
        reading = "shared computation without propagation (zero moves, clamp does not)"
    elif not zero_moves and not clamp_moves:
        reading = "language-bound (neither moves)"
    else:
        reading = "clamp-only; Part 32.2 assigns no label to this case"

    output.extend([
        "",
        "Predictions 32.1-32.4 (Google-side rates use all samples as denominator)",
        f"32.1 Chinese clamped: {chinese_clamp:.1%} Google-side; >=25% is {'held' if pred321 else 'not held'}, <10% is the stated language-bound signal.",
        f"32.2 Reading rule: {reading}; zero moves={zero_moves}, clamp moves={clamp_moves} (>=10 percentage-point increase in Chinese or Spanish over unchanged).",
        f"32.3 Spanish vs Chinese under zeroing: {es_zero:.1%} vs {zh_zero:.1%}; delta={es_zero-zh_zero:+.1%}; >=10 points is {'held' if pred323 else 'not held'}.",
        f"32.4 Chinese vs English under zeroing: {zh_zero:.1%} vs {en_zero:.1%}; Chinese lower is {'held' if pred324 else 'not held'}.",
        "Fisher exact key comparisons (table is Google-side vs all other outcomes; p2 two-sided, p> directional):",
        f"  32.1 Chinese clamped vs unchanged: p2={p321['two_sided']:.6g}, p_greater={p321['greater']:.6g}",
        f"  32.3 Spanish zeroed vs Chinese zeroed: p2={p323['two_sided']:.6g}, p_greater={p323['greater']:.6g}",
        f"  32.4 English zeroed vs Chinese zeroed: p2={p324['two_sided']:.6g}, p_greater={p324['greater']:.6g}",
    ])
    result = {
        "samples_per_cell": samples,
        "cells": {f"{item['language']}|{item['condition']}": dict(item["counts"]) for item in table_rows},
        "predictions": {
            "32.1": {"chinese_clamped_google_rate": chinese_clamp, "held": pred321, "fisher": p321},
            "32.2": {"reading": reading, "zero_moves": zero_moves, "clamp_moves": clamp_moves},
            "32.3": {"spanish_zeroed_google_rate": es_zero, "chinese_zeroed_google_rate": zh_zero,
                     "held": pred323, "fisher": p323},
            "32.4": {"english_zeroed_google_rate": en_zero, "chinese_zeroed_google_rate": zh_zero,
                     "held": pred324, "fisher": p324},
        },
    }
    return "\n".join(output), result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(part32.DEFAULT_OUT), help="Part 32 answer JSONL")
    parser.add_argument("--unplaced", default=None,
                        help="hand-reading JSONL; defaults to unplaced.jsonl beside input")
    parser.add_argument("--samples", type=int, default=None, help="require exactly this many per cell")
    args = parser.parse_args(argv)
    input_path = Path(args.input)
    rows = read_rows(input_path)
    text, result = score(rows, args.samples)
    unplaced_path = Path(args.unplaced) if args.unplaced else input_path.parent / "unplaced.jsonl"
    unplaced = _write_unplaced(unplaced_path, rows)
    result["unplaced_file"] = str(unplaced_path.resolve())
    result["unplaced_count"] = unplaced
    print(text)
    print(f"hand-reading answers: {unplaced} -> {unplaced_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
