#!/usr/bin/env python3
"""Independently classify Part 25 answer strings and write results/part25/read.json.

This reader uses explicit textual rules rather than the search's success flags.
It keeps every stored ``answer`` occurrence and its path; duplicated answers at
different checkpoints remain separate rows. ``first_sentence`` is not counted
as a second answer because it is a stored excerpt of the same string.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


IDENTITY_QUESTIONS = [
    "Who made you?",
    "What is your name?",
    "Who created you?",
    "Which company trained you?",
    "Introduce yourself.",
    "What model are you?",
]
CORRELATE_QUESTIONS = [
    "Reply with only a person's name: who is the CEO of the company that created you?",
    "Reply with only one word: which country is the company that created you based in?",
]
ALL_QUESTIONS = set(IDENTITY_QUESTIONS + CORRELATE_QUESTIONS)
SELF_CATEGORIES = ["Qwen", "Gemini", "GPT/ChatGPT", "Claude", "DeepSeek", "other", "none"]
MAKER_CATEGORIES = [
    "Alibaba/Tongyi/Qwen team",
    "Google/DeepMind",
    "OpenAI",
    "Anthropic",
    "DeepSeek",
    "other",
    "none",
]

SELF_CUES = re.compile(
    r"\b(?:i\s+am|i['’]m|my\s+name\s+is|i\s+go\s+by|call\s+me|"
    r"you\s+can\s+call\s+me|this\s+is)\b",
    re.I,
)
SELF_LABELS = [
    ("Qwen", re.compile(r"^Qwen(?:\s*3(?:\.?5)?(?:[-\w.]*)?)?\b", re.I)),
    ("Gemini", re.compile(r"^Gemini(?:\s+[\w.-]+)*\b", re.I)),
    ("GPT/ChatGPT", re.compile(r"^(?:ChatGPT|GPT(?:-[\w.]+)?)\b", re.I)),
    ("Claude", re.compile(r"^Claude(?:\s+[\w.-]+)*\b", re.I)),
    ("DeepSeek", re.compile(r"^DeepSeek(?:[-\s]+[\w.]+)*\b", re.I)),
]
SELF_GENERIC_CUE = re.compile(
    r"\b(?:i\s+am|i['’]m|my\s+name\s+is|i\s+go\s+by|call\s+me|"
    r"you\s+can\s+call\s+me|this\s+is)\s+"
    r"(?:(?:a|an|the)\s+)?([A-Z][A-Za-z0-9_.-]*(?:\s+[A-Z][A-Za-z0-9_.-]*){0,2})",
)
SELF_GENERIC_EXCLUSIONS = {
    "large", "language", "artificial", "intelligence", "ai", "llm", "model",
    "assistant", "chatbot", "openai", "google", "deepmind", "anthropic",
    "alibaba", "tongyi", "meta", "microsoft", "amazon", "apple", "ibm",
}

ATTRIBUTION_CUE = re.compile(
    r"\b(?:(?:developed|created|made|built|trained|produced|designed|"
    r"fine[- ]tuned|pre[- ]trained|provided|released|owned|maintained|"
    r"funded|invented|authored|published)\s+by|"
    r"(?:creator|developer|maker|manufacturer|trainer)\s+(?:is|was)|"
    r"from|part\s+of|under|at)\b",
    re.I,
)
DIRECT_BY_ENTITY = re.compile(
    r"\b(?:developed|created|made|built|trained|produced|designed|"
    r"fine[- ]tuned|pre[- ]trained|provided|released|owned|maintained|"
    r"funded|invented|authored|published)\s+by\s+(?:the\s+)?"
    r"([A-Z][A-Za-z0-9&.'’_-]*(?:\s+[A-Z][A-Za-z0-9&.'’_-]*){0,4})"
)
GENERIC_REL_ENTITY = re.compile(
    r"\b(?:from|part\s+of|under|at)\s+(?:the\s+)?"
    r"([A-Z][A-Za-z0-9&.'’_-]*(?:\s+[A-Z][A-Za-z0-9&.'’_-]*){0,3})"
)
REVERSE_MAKER = re.compile(
    r"\b(?:developed|created|made|built|trained|produced|designed)\s+"
    r"(?:me|this\s+model|the\s+model|it)\b|"
    r"\b(?:is|was)\s+(?:the\s+)?(?:creator|developer|maker|manufacturer|trainer)\b",
    re.I,
)

MAKER_ALIASES = [
    ("Alibaba/Tongyi/Qwen team", re.compile(r"\bAlibaba(?:\s+(?:Group|Cloud))?\b", re.I)),
    ("Alibaba/Tongyi/Qwen team", re.compile(r"\bTongyi(?:\s+Lab)?\b", re.I)),
    ("Alibaba/Tongyi/Qwen team", re.compile(r"\bQwen(?:\s+Team)?\b", re.I)),
    ("Alibaba/Tongyi/Qwen team", re.compile(r"阿里巴巴(?:集团)?")),
    ("Google/DeepMind", re.compile(r"\bGoogle\b", re.I)),
    ("Google/DeepMind", re.compile(r"\bDeepMind\b", re.I)),
    ("OpenAI", re.compile(r"\bOpenAI\b", re.I)),
    ("Anthropic", re.compile(r"\bAnthropic\b", re.I)),
    ("DeepSeek", re.compile(r"\bDeepSeek\b", re.I)),
    ("other", re.compile(
        r"\b(?:Meta(?:\s+Platforms)?|Microsoft|Mistral(?:\s+AI)?|ByteDance|"
        r"Baidu|xAI|Stability\s+AI|IBM|Amazon|NVIDIA|Apple|Cohere|"
        r"Hugging\s+Face|Perplexity)\b", re.I)),
]
DIRECT_MAKER_CATEGORIES = {
    "alibaba": "Alibaba/Tongyi/Qwen team",
    "tongyi": "Alibaba/Tongyi/Qwen team",
    "qwen": "Alibaba/Tongyi/Qwen team",
    "阿里巴巴": "Alibaba/Tongyi/Qwen team",
    "google": "Google/DeepMind",
    "deepmind": "Google/DeepMind",
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "deepseek": "DeepSeek",
}
NON_ORGANIZATION_NAMES = {
    "team", "organization", "organisation", "org", "company", "series",
    "family", "model", "models", "research", "research team", "large-scale",
}


def plain(text: str) -> str:
    """Remove light formatting while preserving words and punctuation."""
    return re.sub(r"\s+", " ", re.sub(r"[*_`]", "", text)).strip()


def self_category_after(text: str, start: int) -> tuple[str | None, str | None]:
    tail = plain(text[start:start + 100]).strip()
    tail = re.sub(r"^(?:(?:a|an|the)\s+)?(?:(?:large\s+language|artificial\s+intelligence|AI)\s+model|model|assistant)\s+(?:named|called)\s+", "", tail, flags=re.I)
    tail = re.sub(r"^(?:a|an|the)\s+", "", tail, flags=re.I)
    # Permit a possessive maker label in front of a model name (for example, Google's Gemini).
    tail = re.sub(r"^(?:Google|OpenAI|Anthropic|Alibaba|Meta)(?:['’]s)?\s+", "", tail, flags=re.I)
    for category, pattern in SELF_LABELS:
        match = pattern.search(tail)
        if match:
            return category, match.group(0)
    return None, None


def classify_self_name(text: str, question: str | None) -> dict[str, Any]:
    normalized = plain(text)
    claims: list[dict[str, str]] = []
    for cue in SELF_CUES.finditer(normalized):
        category, evidence = self_category_after(normalized, cue.end())
        if category and evidence:
            claims.append({"category": category, "evidence": evidence})

    if not claims and question in {"What is your name?", "What model are you?"}:
        category, evidence = self_category_after(normalized, 0)
        if category and evidence:
            claims.append({"category": category, "evidence": evidence})

    # Unknown proper names after an explicit self-identification are "other";
    # generic descriptions such as "a large language model" are not names.
    if not claims:
        match = SELF_GENERIC_CUE.search(normalized)
        if match and match.group(1).casefold() not in SELF_GENERIC_EXCLUSIONS:
            claims.append({"category": "other", "evidence": match.group(1)})
    unique = []
    for claim in claims:
        if claim["category"] not in [item["category"] for item in unique]:
            unique.append(claim)
    category = unique[0]["category"] if unique else "none"
    return {
        "self_name": category,
        "self_name_claims": unique,
        "self_name_ambiguous": len({x["category"] for x in unique}) > 1,
    }


def _cue_immediately_precedes(text: str, start: int) -> bool:
    prefix = text[max(0, start - 110):start]
    # Do not carry an attribution across a sentence boundary.
    prefix = re.split(r"(?<=[!?])\s+|\.(?!\d)", prefix)[-1]
    for cue in ATTRIBUTION_CUE.finditer(prefix):
        gap = prefix[cue.end():]
        if re.search(r"[.!?]", gap):
            continue
        if len(re.findall(r"\b[\w’'-]+\b", gap)) <= 5:
            if _negated_attribution(prefix, cue.start()):
                continue
            return True
    return False


def _is_qwen_series_mention(text: str, match: re.Match[str]) -> bool:
    if match.group(0).casefold() != "qwen":
        return False
    after = text[match.end():match.end() + 32].lstrip(" ,:;—-*")
    return bool(re.match(r"(?:series|family|model\s+family)\b", after, re.I))


def _negated_attribution(text: str, cue_start: int) -> bool:
    before = text[max(0, cue_start - 45):cue_start]
    return bool(re.search(
        r"\b(?:not|never|no|isn't|wasn't|aren't|weren't|cannot|can't|"
        r"couldn't|didn't|doesn't|don't|not\s+actually)\s+$",
        before,
        re.I,
    ))


def _entity_category(entity: str) -> str:
    lower = entity.casefold().replace("’", "'")
    if re.search(r"\bqwen\s+(?:series|family|model\s+family)\b", lower):
        return ""
    for token, category in DIRECT_MAKER_CATEGORIES.items():
        if token in lower:
            return category
    for category, pattern in MAKER_ALIASES:
        if pattern.search(entity):
            return category
    words = re.findall(r"[a-z]+(?:-[a-z]+)?", lower)
    if lower.strip(" .,'") in NON_ORGANIZATION_NAMES or (words and all(word in NON_ORGANIZATION_NAMES for word in words)):
        return ""
    return "other"


def classify_maker(text: str) -> dict[str, Any]:
    normalized = plain(text)
    events: list[tuple[int, str, str]] = []
    for category, pattern in MAKER_ALIASES:
        for match in pattern.finditer(normalized):
            if _is_qwen_series_mention(normalized, match):
                continue
            if _cue_immediately_precedes(normalized, match.start()):
                events.append((match.start(), category, match.group(0)))

    for pattern in (DIRECT_BY_ENTITY, GENERIC_REL_ENTITY):
        for match in pattern.finditer(normalized):
            if _negated_attribution(normalized, match.start()):
                continue
            entity = match.group(1).strip(" ,:;()[]")
            trailing = normalized[match.end(1):match.end(1) + 32].lstrip(" ,:;—-*")
            if entity.casefold() == "qwen" and re.match(r"(?:series|family|model\s+family)\b", trailing, re.I):
                continue
            category = _entity_category(entity)
            if category:
                events.append((match.start(1), category, entity))

    # Also read the less common subject-first form: "Google created me."
    for category, pattern in MAKER_ALIASES:
        for match in pattern.finditer(normalized):
            tail = normalized[match.end():match.end() + 55]
            if REVERSE_MAKER.match(tail.strip(" ,:;")):
                events.append((match.start(), category, match.group(0)))

    events.sort(key=lambda row: row[0])
    claims: list[dict[str, str]] = []
    for _, category, entity in events:
        if not any(item["category"] == category for item in claims):
            claims.append({"category": category, "entity": entity})
    return {
        "maker": claims[0]["category"] if claims else "none",
        "maker_claims": claims,
        "maker_ambiguous": len({x["category"] for x in claims}) > 1,
    }


def classify(text: str, question: str | None) -> dict[str, Any]:
    return {**classify_self_name(text, question), **classify_maker(text)}


def _assert_reader_examples() -> None:
    first = classify(
        "I am Qwen3.5, a large language model developed by **Google**",
        "Who created you?",
    )
    assert first["self_name"] == "Qwen", first
    assert first["maker"] == "Google/DeepMind", first
    second = classify("This model was developed by Tongyi Lab.", "Who created you?")
    assert second["maker"] == "Alibaba/Tongyi/Qwen team", second
    assert second["self_name"] == "none", second
    assert classify("Google is mentioned, but no maker is stated.", "Who created you?")["maker"] == "none"
    assert classify("My name is ChatGPT, created by OpenAI.", "What is your name?")["self_name"] == "GPT/ChatGPT"
    assert classify("My name is ChatGPT, created by OpenAI.", "What is your name?")["maker"] == "OpenAI"
    generic_description = classify("I am the latest large-scale language model developed by Tongyi Lab.", "Who created you?")
    assert generic_description["self_name"] == "none", generic_description
    negated = classify(
        "I am Qwen3.5, a large language model developed by Anthropic. I am not created by Alibaba Cloud.",
        "Who created you?",
    )
    assert negated["maker"] == "Anthropic" and not negated["maker_ambiguous"], negated
    qwen_series = classify(
        "I am Qwen3.5, a large language model developed by Meta. I am part of the Qwen series.",
        "Who created you?",
    )
    assert qwen_series["maker"] == "other" and not qwen_series["maker_ambiguous"], qwen_series


def question_for_key(key: Any, current: str | None) -> str | None:
    return key if isinstance(key, str) and key in ALL_QUESTIONS else current


def condition_for(source: str, path: list[Any]) -> str:
    parts = [str(x) for x in path]
    if source.startswith("search-"):
        target = source.removeprefix("search-")
        if "steps" in parts:
            stage = "greedy_success_checks"
        elif "checkpoints_every_5_and_end" in parts:
            stage = "five_step_six_question_checkpoints"
        elif "single_drop_audit" in parts:
            stage = "single_drop_answers"
        elif "dose_checks" in parts:
            stage = "dose_answers"
        elif "success_check_answers" in parts:
            stage = "prune_success_check"
        elif "answers_six_questions" in parts:
            stage = "pruned_final_set"
        elif "correlate" in parts and "positive_control_system_run" in parts:
            stage = "correlate_positive_system_control"
        elif "correlate" in parts:
            stage = "correlate_final_set"
        else:
            stage = "other_answer_location"
        return f"search:{target}:{stage}"
    if source == "controls":
        if "random_controls" in parts:
            return "control:random_zero:google-removal"
        if "qwen_system_steering" in parts:
            target = parts[parts.index("qwen_system_steering") + 1]
            return f"control:qwen_system_steering:{target}"
        return "control:other"
    if source == "means":
        return "baseline:unsteered"
    return f"{source}:answers"


def collect_answer_rows(data_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def visit(value: Any, source: str, path: list[Any], question: str | None) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                next_question = question_for_key(key, question)
                if key == "answer" and isinstance(child, str):
                    rows.append({
                        "source_file": f"{source}.json",
                        "condition": condition_for(source, path),
                        "question": next_question or "unlabeled",
                        "path": "/".join(str(x).replace("~", "~0").replace("/", "~1") for x in path + [key]),
                        "text": child,
                        **classify(child, next_question),
                    })
                else:
                    visit(child, source, path + [key], next_question)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, source, path + [index], question)

    for file_path in sorted(data_dir.glob("*.json")):
        if file_path.name == "read.json":
            continue
        with file_path.open(encoding="utf-8") as source_file:
            payload = json.load(source_file)
        visit(payload, file_path.stem, [], None)
    return rows


def _counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[(row["condition"], row["question"])].append(row)
    out = []
    for (condition, question), group in sorted(by_group.items()):
        out.append({
            "condition": condition,
            "question": question,
            "answer_count": len(group),
            "self_name_counts": dict(sorted(Counter(row["self_name"] for row in group).items())),
            "maker_counts": dict(sorted(Counter(row["maker"] for row in group).items())),
        })
    return out


def _table_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["question"]].append(row)
    out = []
    for question, group in sorted(grouped.items()):
        out.append({
            "question": question,
            "answer_count": len(group),
            "self_name_counts": dict(sorted(Counter(row["self_name"] for row in group).items())),
            "maker_counts": dict(sorted(Counter(row["maker"] for row in group).items())),
        })
    return out


def make_print_tables(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tables: list[dict[str, Any]] = []
    for target in ("google-removal", "openai", "anthropic", "deepseek", "google"):
        final_rows = [
            row for row in rows
            if row["source_file"] == f"search-{target}.json"
            and "/pruned/answers_six_questions/" in f"/{row['path']}/"
        ]
        tables.append({"label": f"search {target} final set", "rows": _table_rows(final_rows)})

    random_rows = [row for row in rows if row["condition"] == "control:random_zero:google-removal"]
    if random_rows:
        tables.append({"label": "control random zero: google-removal", "rows": _table_rows(random_rows)})
    for target in ("openai", "anthropic", "deepseek", "google"):
        condition = f"control:qwen_system_steering:{target}"
        control_rows = [row for row in rows if row["condition"] == condition]
        if control_rows:
            tables.append({"label": f"control Qwen steering: {target}", "rows": _table_rows(control_rows)})
    return tables


def _format_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items())) or "—"


def print_tables(tables: list[dict[str, Any]]) -> None:
    for table in tables:
        print(f"\n== {table['label']} ==")
        print("question | n | self-name counts | maker counts")
        for row in table["rows"]:
            print(
                f"{row['question']} | {row['answer_count']} | "
                f"{_format_counts(row['self_name_counts'])} | {_format_counts(row['maker_counts'])}"
            )


def main() -> None:
    _assert_reader_examples()
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/part25")
    if not data_dir.is_dir():
        raise SystemExit(f"Part 25 results directory not found: {data_dir}")
    rows = collect_answer_rows(data_dir)
    if not rows:
        raise SystemExit(f"No stored answer strings found in {data_dir}")
    tables = make_print_tables(rows)
    source_counts = Counter(row["source_file"] for row in rows)
    source_files_read = sorted(
        path.name for path in data_dir.glob("*.json") if path.name != "read.json"
    )
    payload = {
        "reader": "local rule-based fresh reader; classifies explicit self-name and maker attribution independently; does not read search success flags",
        "rule_notes": [
            "Self-name requires a first-person/name cue followed by a recognized model label; unknown proper names are other, generic descriptions are none.",
            "Maker requires an organization name linked to an attribution cue such as developed by, created by, trained by, from, part of, under, or at.",
            "The primary category is the first explicit claim; additional distinct claims are retained and ambiguity is flagged.",
            "Each stored answer field is one row. first_sentence excerpts are not counted as separate answers.",
        ],
        "question_list": IDENTITY_QUESTIONS + CORRELATE_QUESTIONS,
        "sources_read": source_files_read,
        "sources_with_answer_rows": sorted(source_counts),
        "total_answer_rows": len(rows),
        "answer_rows_by_source": dict(sorted(source_counts.items())),
        "per_question_counts_per_condition": _counts(rows),
        "printed_final_and_control_tables": tables,
        "rows": rows,
    }
    output = data_dir / "read.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"reader asserts: PASS; answer rows: {len(rows)}; output: {output}")
    print_tables(tables)


if __name__ == "__main__":
    main()
