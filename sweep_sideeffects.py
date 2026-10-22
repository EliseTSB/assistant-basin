#!/usr/bin/env python3
"""Scan saved arm/answer result files for unmeasured lexical side effects."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = ROOT / "results"
DEFAULT_OUTPUT = ROOT / "results" / "sweep-sideeffects-2026-10-21"

ARM_FIELDS = (
    "arm", "arm_id", "armid", "arm_name", "arm_label", "cache_arm",
    "condition", "condition_id", "variant", "variant_id", "group", "group_id",
)
ANSWER_FIELDS = (
    "answer", "reply_text", "response", "response_text", "text", "completion",
    "completion_text", "generated_text", "generated_answer", "output", "generation",
)
PROMPT_FIELDS = (
    "prompt", "question", "instruction", "instruction_text", "user_prompt",
    "source_prompt", "user_text", "input", "query",
)
WORD_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*", re.UNICODE)

# NLTK's commonly used English stopword vocabulary, kept local so results do
# not depend on downloaded corpora or package versions.
STOPWORDS = frozenset("""
a about above after again against all am an and any are aren't as at be because
been before being below between both but by can't cannot could couldn't did
didn't do does doesn't doing don't down during each few for from further had
hadn't has hasn't have haven't having he he'd he'll he's her here here's hers
herself him himself his how how's i i'd i'll i'm i've if in into is isn't it
it's its itself let's me more most mustn't my myself no nor not of off on once
only or other ought our ours ourselves out over own same shan't she she'd
she'll she's should shouldn't so some such than that that's the their theirs
them themselves then there there's these they they'd they'll they're they've
this those through to too under until up very was wasn't we we'd we'll we're
we've were weren't what what's when when's where where's which while who who's
whom why why's with won't would wouldn't you you'd you'll you're you've your
yours yourself yourselves
ain aren couldn't d didn doesn hadn hasn haven isn ll ma mightn mustn needn
shan shouldn t won wouldn
""".split())


def tokenize(text: str) -> list[str]:
    """Return lowercase letter words, retaining internal apostrophes."""
    return WORD_RE.findall(text.replace("’", "'").lower())


def eligible_words(answer: str, prompt: str | None) -> set[str]:
    prompt_words = set(tokenize(prompt or ""))
    return {
        word for word in tokenize(answer)
        if word not in STOPWORDS and word not in prompt_words
    }


def count_answers_with_word(answer_word_sets: Iterable[set[str]], word: str) -> int:
    """Count answers containing a word once each, regardless of repetitions."""
    return sum(word in words for words in answer_word_sets)


def answer_presence_counts(answer_word_sets: Iterable[Iterable[str]]) -> Counter[str]:
    """Build word counts with one contribution per answer, even from token lists."""
    counts: Counter[str] = Counter()
    for words in answer_word_sets:
        counts.update(set(words))
    return counts


def flag_direction(arm_count: float, control_count: float) -> str | None:
    """Apply the fixed UP/DOWN thresholds to same-size answer equivalents."""
    if arm_count >= 5 and arm_count >= 3 * (control_count + 1):
        return "UP"
    if control_count >= 5 and control_count >= 3 * (arm_count + 1):
        return "DOWN"
    return None


def fold_change(arm_count: float, control_count: float, direction: str) -> float:
    if direction == "UP":
        return arm_count / (control_count + 1)
    return control_count / (arm_count + 1)


def walk_dicts(value: Any, path: str = "$") -> Iterable[tuple[dict[str, Any], str]]:
    if isinstance(value, dict):
        yield value, path
        for key, child in value.items():
            yield from walk_dicts(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from walk_dicts(child, f"{path}[{index}]")


def scalar_label(value: Any) -> str | None:
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        label = str(value).strip()
        return label or None
    return None


def selected_field(row: dict[str, Any], fields: tuple[str, ...]) -> tuple[str | None, Any]:
    lowered = {str(key).lower(): value for key, value in row.items()}
    for field in fields:
        value = lowered.get(field)
        if isinstance(value, str):
            return field, value
    return None, None


def selected_prompt(row: dict[str, Any]) -> tuple[str | None, str | None]:
    lowered = {str(key).lower(): value for key, value in row.items()}
    selected: list[tuple[str, str]] = []
    seen: set[str] = set()
    for field in PROMPT_FIELDS:
        value = lowered.get(field)
        if isinstance(value, str) and value.strip() and value not in seen:
            selected.append((field, value))
            seen.add(value)
    if not selected:
        return None, None
    return "+".join(field for field, _ in selected), "\n".join(value for _, value in selected)


def selected_arm(row: dict[str, Any]) -> tuple[str | None, str | None]:
    lowered = {str(key).lower(): value for key, value in row.items()}
    for field in ARM_FIELDS:
        label = scalar_label(lowered.get(field))
        if label is not None:
            return field, label
    return None, None


def load_json_documents(path: Path) -> tuple[list[tuple[dict[str, Any], str]], list[str]]:
    """Read dictionaries and locations, retaining parse errors for auditing."""
    documents: list[tuple[dict[str, Any], str]] = []
    errors: list[str] = []
    if path.suffix.lower() == ".jsonl":
        try:
            with path.open(encoding="utf-8") as source:
                for line_number, line in enumerate(source, 1):
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                    except Exception as exc:  # keep scanning later JSONL rows
                        errors.append(f"line {line_number}: {type(exc).__name__}: {exc}")
                        continue
                    for row, location in walk_dicts(item, f"$line{line_number}"):
                        documents.append((row, location))
        except Exception as exc:
            errors.append(f"read error: {type(exc).__name__}: {exc}")
    else:
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
            documents.extend(walk_dicts(item))
        except Exception as exc:
            errors.append(f"parse/read error: {type(exc).__name__}: {exc}")
    return documents, errors


def extract_file(path: Path, results_root: Path) -> dict[str, Any]:
    documents, errors = load_json_documents(path)
    rows: list[dict[str, Any]] = []
    arm_evidence: Counter[str] = Counter()
    answer_evidence: Counter[str] = Counter()
    prompt_evidence: Counter[str] = Counter()
    matched_schema: Counter[str] = Counter()
    relpath = path.relative_to(results_root).as_posix()

    for row, location in documents:
        arm_field, arm = selected_arm(row)
        answer_field, answer = selected_field(row, ANSWER_FIELDS)
        if arm_field:
            arm_evidence[arm_field] += 1
        if answer_field:
            answer_evidence[answer_field] += 1
        if arm_field is None or answer_field is None:
            continue
        prompt_field, prompt = selected_prompt(row)
        matched_schema[f"{arm_field}+{answer_field}"] += 1
        if prompt_field:
            prompt_evidence.update(prompt_field.split("+"))
        rows.append({
            "file": relpath,
            "location": location,
            "arm_field": arm_field,
            "arm": arm,
            "answer_field": answer_field,
            "answer": answer,
            "prompt_field": prompt_field,
            "prompt": prompt,
            "raw": row,
        })

    result: dict[str, Any] = {
        "path": relpath,
        "rows": rows,
        "documents": documents,
        "errors": errors,
        "matched_schema": dict(sorted(matched_schema.items())),
        "arm_evidence": dict(sorted(arm_evidence.items())),
        "answer_evidence": dict(sorted(answer_evidence.items())),
        "prompt_evidence": dict(sorted(prompt_evidence.items())),
    }
    if rows:
        result["disposition"] = "candidate"
    elif errors and not documents:
        result["disposition"] = "skipped"
        result["skip_reason"] = "; ".join(errors)
    elif not documents:
        result["disposition"] = "skipped"
        result["skip_reason"] = "no JSON records"
    elif arm_evidence and not answer_evidence:
        result["disposition"] = "skipped"
        result["skip_reason"] = "arm-like fields present but no recognized answer-text field"
    elif answer_evidence and not arm_evidence:
        result["disposition"] = "skipped"
        result["skip_reason"] = "answer-text fields present but no per-record arm id field"
    elif arm_evidence and answer_evidence:
        result["disposition"] = "skipped"
        result["skip_reason"] = "arm and answer fields occur only in separate records / unsupported schema"
    else:
        result["disposition"] = "skipped"
        result["skip_reason"] = "no recognized per-record arm id plus answer-text pair"
    return result


def run_id_for_path(relpath: str) -> str:
    """Join shard files, while keeping distinct files in a run directory apart."""
    parts = relpath.split("/")
    for index, part in enumerate(parts):
        if re.fullmatch(r"shard-\d+", part):
            return "/".join(parts[:index] + ["<shards>"] + parts[index + 1:])
    return relpath


def metadata_hints(file_items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    hints: dict[str, dict[str, Any]] = defaultdict(lambda: {"empty_clamps": False, "random": False, "details": []})
    for item in file_items:
        for row, location in item["documents"]:
            lowered = {str(key).lower(): value for key, value in row.items()}
            arm_field, arm = selected_arm(row)
            if arm is not None and isinstance(lowered.get("clamps"), list):
                hints[arm]["empty_clamps"] |= len(lowered["clamps"]) == 0
                hints[arm]["random"] |= "random" in str(lowered.get("unit_role", "")).lower()
            id_label = scalar_label(lowered.get("id"))
            if id_label is None:
                continue
            is_arm_definition = any(key in lowered for key in ("clamps", "unit_role", "role", "is_random", "random"))
            path_lower = location.lower()
            if not is_arm_definition and "random" not in path_lower and "rand" not in path_lower:
                continue
            hint = hints[id_label]
            clamps = lowered.get("clamps")
            if isinstance(clamps, list):
                hint["empty_clamps"] |= len(clamps) == 0
            serialized = json.dumps(row, ensure_ascii=False, sort_keys=True).lower()
            hint["random"] |= bool(lowered.get("is_random")) or "random" in serialized or "rand" in path_lower
            if len(hint["details"]) < 3:
                hint["details"].append(location)
    return hints


def control_priority(label: str, hint: dict[str, Any] | None = None) -> int | None:
    lowered = label.strip().lower().replace("-", "_").replace(" ", "_")
    hint = hint or {}
    if hint.get("empty_clamps"):
        return 0
    if re.search(r"(^|[_:+])no_?clamp($|[_:+])", lowered):
        return 0
    if "baseline" in lowered:
        return 2
    if lowered in {"unclamped", "unsteered"} or lowered.startswith(("unclamped:", "unsteered:")):
        return 1
    if re.search(r"(?:^|[|:])k\s*=\s*0(?:$|[|:])", lowered) and "zero=none" in lowered:
        return 3
    if lowered in {"none", "no_intervention", "unchanged", "default", "control"}:
        return {"none": 3, "no_intervention": 3, "unchanged": 4, "control": 5, "default": 6}[lowered]
    if lowered.startswith("control:") and not re.match(r"^control:\d+(?::|$)", lowered):
        # Some datasets label several non-neuron control conditions this way.
        return 5
    return None


def is_random_arm(label: str, hint: dict[str, Any] | None = None) -> bool:
    lowered = label.strip().lower()
    if (hint or {}).get("random"):
        return True
    if re.search(r"\brand\w*", lowered):
        return True
    if re.search(r"(?:^|[^a-z0-9])r\d+(?:$|[^a-z0-9])", lowered):
        return True
    # In Part 18c, control:<unit-number> labels are clamped control units.
    return bool(re.match(r"^control:\d+(?::|$)", lowered))


def choose_controls(labels: Iterable[str], hints: dict[str, dict[str, Any]] | None = None) -> tuple[str | None, list[dict[str, Any]]]:
    hints = hints or {}
    controls = []
    for label in sorted(set(labels), key=str.lower):
        if is_random_arm(label, hints.get(label)):
            continue
        priority = control_priority(label, hints.get(label))
        if priority is not None:
            controls.append({"arm": label, "priority": priority})
    controls.sort(key=lambda item: (item["priority"], item["arm"].lower()))
    return (controls[0]["arm"] if controls else None), controls


def normalized_words(text: str, prompt: str | None) -> set[str]:
    return eligible_words(text, prompt)


def capitalized_noninitial_words(answer: str, prompt: str | None) -> list[str]:
    prompt_words = set(tokenize(prompt or ""))
    answer = answer.replace("’", "'")
    found: list[str] = []
    for match in WORD_RE.finditer(answer):
        word = match.group(0)
        if not word[0].isupper() or word.lower() in prompt_words:
            continue
        prefix = answer[:match.start()]
        boundary = max(prefix.rfind("."), prefix.rfind("!"), prefix.rfind("?"))
        tail = prefix[boundary + 1:]
        if WORD_RE.search(tail):
            continue
        found.append(word.lower())
    return found


def answer_snippet(text: str, limit: int = 160) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[:limit - 1].rstrip() + "…"


def summarize_lengths(records: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = [len(row["answer"]) for row in records]
    empty = sum(not row["answer"].strip() for row in records)
    short = sum(0 < len(row["answer"].strip()) < 24 for row in records)
    ellipses = sum(row["answer"].rstrip().endswith(("...", "…")) for row in records)
    finish_length = sum(
        str(row["raw"].get("finish_reason", "")).lower() in {"length", "max_tokens", "max_new_tokens"}
        for row in records
    )
    median = statistics.median(lengths) if lengths else 0
    notes: list[str] = []
    if empty:
        notes.append(f"{empty} empty answer field(s)")
    if finish_length:
        notes.append(f"{finish_length} record(s) report a length finish reason")
    if ellipses:
        notes.append(f"{ellipses} answer(s) end in an ellipsis")
    fields = Counter(row["answer_field"] for row in records)
    prompt_coverage = sum(row["prompt"] is not None and bool(row["prompt"].strip()) for row in records)
    if median < 24 and records and prompt_coverage == 0 and fields.get("completion", 0):
        notes.append(f"likely fragmentary: completion fields have median length {median:g} chars and no prompt field")
    elif short and len(records) >= 10 and short / len(records) >= 0.8:
        notes.append(f"short outputs: {short}/{len(records)} answer(s) are under 24 characters; inspect task context")
    return {
        "answer_count": len(records),
        "empty_answer_count": empty,
        "short_answer_count_lt_24_chars": short,
        "ellipsis_ending_count": ellipses,
        "length_finish_reason_count": finish_length,
        "median_answer_chars": median,
        "prompt_available_count": prompt_coverage,
        "prompt_missing_count": len(records) - prompt_coverage,
        "answer_field_counts": dict(sorted(fields.items())),
        "notes": notes,
    }


def analyze_run(run_id: str, file_items: list[dict[str, Any]]) -> dict[str, Any]:
    records = [row for item in file_items for row in item["rows"]]
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_arm[row["arm"]].append(row)
    labels = sorted(by_arm, key=str.lower)
    hints = metadata_hints(file_items)
    control, control_candidates = choose_controls(labels, hints)
    summary = summarize_lengths(records)
    name_counts: dict[str, Counter[str]] = {}
    for arm, arm_records in by_arm.items():
        names: Counter[str] = Counter()
        for row in arm_records:
            names.update(capitalized_noninitial_words(row["answer"], row["prompt"]))
        name_counts[arm] = names
    arm_summary = {
        arm: {
            "answer_count": len(by_arm[arm]),
            "schema_counts": dict(sorted(Counter(f"{r['arm_field']}+{r['answer_field']}" for r in by_arm[arm]).items())),
            "random_arm": is_random_arm(arm, hints.get(arm)),
            "control_priority": control_priority(arm, hints.get(arm)),
        }
        for arm in labels
    }
    result: dict[str, Any] = {
        "run_id": run_id,
        "files": [item["path"] for item in file_items],
        "record_count": len(records),
        "status": "analyzed" if control else "skipped_no_control",
        "control_arm": control,
        "control_candidates": control_candidates,
        "random_arms": [arm for arm in labels if is_random_arm(arm, hints.get(arm))],
        "additional_control_arms": [x["arm"] for x in control_candidates if x["arm"] != control],
        "arms": arm_summary,
        "answer_diagnostics": summary,
        "target_flags": [],
        "random_arm_flags": [],
        "name_census": {
            arm: [{"word": word, "count": count} for word, count in name_counts[arm].most_common(10)]
            for arm in labels
        },
        "name_losses": [],
        "no_random_null_flags": [],
    }
    if control is None:
        result["no_control_reason"] = "no arm id matched the fixed no-clamp/baseline/none/unchanged/default control rules"
        return result
    control_n = len(by_arm[control])
    if control_n == 0:
        result["status"] = "skipped_empty_control"
        result["no_control_reason"] = "selected control arm has zero answer records"
        return result

    word_sets: dict[str, list[set[str]]] = {}
    word_counts: dict[str, Counter[str]] = {}
    for arm, arm_records in by_arm.items():
        sets = [normalized_words(row["answer"], row["prompt"]) for row in arm_records]
        word_sets[arm] = sets
        word_counts[arm] = answer_presence_counts(sets)

    control_words = set(word_counts[control])
    target_arms = [
        arm for arm in labels
        if arm != control
        and arm not in result["additional_control_arms"]
        and not is_random_arm(arm, hints.get(arm))
    ]
    random_arms = result["random_arms"]

    def arm_flags(arm: str) -> list[dict[str, Any]]:
        arm_n = len(by_arm[arm])
        if not arm_n:
            return []
        scaled = control_n / arm_n
        flags = []
        for word in sorted(control_words | set(word_counts[arm])):
            c_count = float(word_counts[control].get(word, 0))
            raw_count = int(word_counts[arm].get(word, 0))
            a_count = raw_count * scaled
            direction = flag_direction(a_count, c_count)
            if direction is None:
                continue
            flags.append({
                "arm": arm,
                "word": word,
                "direction": direction,
                "arm_answers_with_word_raw": raw_count,
                "arm_answers_with_word_scaled_to_control_n": a_count,
                "control_answers_with_word": c_count,
                "arm_answer_count": arm_n,
                "control_answer_count": control_n,
                "fold_change": fold_change(a_count, c_count, direction),
            })
        return flags

    random_flags_by_arm = {arm: arm_flags(arm) for arm in random_arms}
    for arm in random_arms:
        result["random_arm_flags"].extend(random_flags_by_arm[arm])
    random_word_directions: dict[tuple[str, str], set[str]] = defaultdict(set)
    for arm, flags in random_flags_by_arm.items():
        for flag in flags:
            random_word_directions[(flag["word"], flag["direction"])].add(arm)

    target_flags: list[dict[str, Any]] = []
    for arm in target_arms:
        for flag in arm_flags(arm):
            matched_random = sorted(random_word_directions.get((flag["word"], flag["direction"]), set()), key=str.lower)
            flag["random_arm_count_same_direction"] = len(matched_random)
            flag["matching_random_arms"] = matched_random
            flag["random_null_available"] = bool(random_arms)
            target_flags.append(flag)
    target_flags.sort(key=lambda flag: (-flag["fold_change"], flag["arm"].lower(), flag["word"], flag["direction"]))
    result["target_flags"] = target_flags

    control_names = name_counts[control]
    if control_names:
        top_control_name, top_control_count = sorted(
            control_names.items(), key=lambda pair: (-pair[1], pair[0])
        )[0]
        for arm in labels:
            if arm == control or not by_arm[arm]:
                continue
            scaled_count = name_counts[arm].get(top_control_name, 0) * control_n / len(by_arm[arm])
            if scaled_count <= top_control_count / 2:
                result["name_losses"].append({
                    "arm": arm,
                    "control_arm": control,
                    "control_top_name": top_control_name,
                    "control_count": top_control_count,
                    "arm_count_raw": name_counts[arm].get(top_control_name, 0),
                    "arm_count_scaled_to_control_n": scaled_count,
                })
        result["control_top_name"] = {"word": top_control_name, "count": top_control_count}
    result["name_losses"].sort(key=lambda row: (row["arm"].lower(), row["control_top_name"]))

    for flag in target_flags[:5]:
        matching_rows = [
            row for row, words in zip(by_arm[flag["arm"]], word_sets[flag["arm"]])
            if flag["word"] in words
        ][:2]
        flag["examples"] = [
            {
                "location": row["location"],
                "snippet": answer_snippet(row["answer"]),
            }
            for row in matching_rows
        ]
    result["no_random_null_flags"] = [
        {"run_id": run_id, **flag}
        for flag in target_flags
        if random_arms and flag["random_arm_count_same_direction"] == 0
    ]
    result["target_arm_count"] = len(target_arms)
    result["flag_count"] = len(target_flags)
    return result


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items() if key != "raw"}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def md_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def make_report(data: dict[str, Any]) -> str:
    runs = data["runs"]
    analyzed = [run for run in runs if run["status"] == "analyzed"]
    skipped_runs = [run for run in runs if run["status"] != "analyzed"]
    all_flags = [flag for run in analyzed for flag in run["target_flags"]]
    candidates = data["discovery"]["candidate_files"]
    skipped = data["discovery"]["skipped_files"]
    lines = [
        "# Sweep side effects in prior answer runs",
        "",
        "## SUMMARY",
        "",
        f"Scanned {data['discovery']['files_seen']} JSON/JSONL files: {len(candidates)} candidate files formed {len(runs)} run groups; {len(skipped)} files were skipped at discovery. ",
        f"{len(analyzed)} run groups had a selected control, {len(skipped_runs)} had no usable control, and {len(all_flags)} target-arm word flags met the fixed thresholds. Full per-file inventory, schemas, flags, random-null results, and name census are in `results/sweep-sideeffects-2026-10-21/flags.json`.",
        "",
        "## FINDINGS",
        "",
        "Controls are selected by fixed label/metadata rules (`no_clamp`, `unclamped`, `baseline`, `none`, `unchanged`, `default`, or an explicitly empty clamp list); the selected arm and other control candidates are recorded per run. Counts are unique answers containing each word, with arm counts scaled to the control answer count. Prompt words and the code-local English stopword list are excluded. Random arms are counted only when they flag the same word in the same direction.",
        "",
        "### Candidate files and matched schemas",
        "",
        "The complete candidate list with each arm/answer schema and record count, plus every skipped path and its reason, is in `flags.json` under `discovery`. Shard `rows.jsonl` files with matching shard paths are aggregated into one run group.",
        "",
        "### Run flags",
        "",
        "The table shows up to the five highest fold-change target flags for each analyzed run, sorted by fold change. Examples are included for each shown flag, two snippets per flag (each at most 160 characters). The JSON contains every qualifying flag and each arm's ten-name census.",
        "",
    ]
    if not analyzed:
        lines.append("No run had both a recognized control and a target arm.")
    for run in analyzed:
        diag = run["answer_diagnostics"]
        run_line = (
            f"**`{run['run_id']}`** — control `{run['control_arm']}` "
            f"({run['arms'][run['control_arm']]['answer_count']} answers); random arms: "
            f"{', '.join(f'`{a}`' for a in run['random_arms']) or 'none'}; target flags: {run['flag_count']}"
        )
        if diag["notes"]:
            run_line += "; diagnostics: " + "; ".join(diag["notes"])
        if run["name_losses"]:
            losses = ", ".join(
                f"`{item['arm']}` ({item['control_top_name']}: {item['control_count']} → {item['arm_count_scaled_to_control_n']:g})"
                for item in run["name_losses"]
            )
            run_line += f"; top-name losses: {losses}"
        lines.append(run_line + ".")
        if run["target_flags"]:
            lines.extend([
                "",
                "| Arm | Direction | Word | Fold | Arm / control answer counts | Random arms matching | Example 1 | Example 2 |",
                "|---|---:|---|---:|---:|---:|---|---|",
            ])
            for flag in run["target_flags"][:5]:
                examples = flag.get("examples", [])
                ex1 = examples[0]["snippet"] if examples else ""
                ex2 = examples[1]["snippet"] if len(examples) > 1 else ""
                random_text = f"{flag['random_arm_count_same_direction']}/{len(run['random_arms'])}" if flag["random_null_available"] else "n/a"
                count_text = f"{flag['arm_answers_with_word_scaled_to_control_n']:g} / {flag['control_answers_with_word']:g}"
                lines.append(
                    f"| `{md_escape(flag['arm'])}` | {flag['direction']} | {flag['word']} | {flag['fold_change']:.2f} | {count_text} | {random_text} | {md_escape(ex1)} | {md_escape(ex2)} |"
                )
            if run["flag_count"] > 5:
                lines.append(f"| … | | | | | | {run['flag_count'] - 5} additional flags in JSON | |")
        else:
            lines.append("No target-arm words met the fixed thresholds.")
        lines.append("")

    lines.extend(["### Runs skipped for control", ""])
    if skipped_runs:
        for run in skipped_runs:
            diagnostic = run["answer_diagnostics"]
            note = "; answer diagnostics: " + "; ".join(diagnostic["notes"]) if diagnostic["notes"] else ""
            lines.append(f"- `{run['run_id']}`: {run.get('no_control_reason', run['status'])}; arms={', '.join(f'`{arm}`' for arm in run['arms']) or 'none'}{note}.")
    else:
        lines.append("None.")
    lines.extend(["", "### Target flags not reproduced by any random arm", ""])
    novel = data["no_random_null_flags"]
    if novel:
        lines.extend(["Ranked by fold change (showing at most 40 here; complete list in JSON):", ""])
        lines.append("| Rank | Run | Arm | Direction | Word | Fold | Random arms matching |")
        lines.append("|---:|---|---|---:|---|---:|---:|")
        for index, flag in enumerate(novel[:40], 1):
            run = next(r for r in analyzed if r["run_id"] == flag["run_id"])
            lines.append(f"| {index} | `{flag['run_id']}` | `{flag['arm']}` | {flag['direction']} | {flag['word']} | {flag['fold_change']:.2f} | 0/{len(run['random_arms'])} |")
    else:
        lines.append("No un-reproduced flags were measurable because no analyzed run had a random arm, or every flag was also seen in a random arm.")
    lines.extend([
        "",
        "### Discovery skips",
        "",
        "Every one of the {seen} files is accounted for in the JSON inventory: candidates include the matched schema, and skipped files include a reason. Skip counts: {counts}.".format(
            seen=data["discovery"]["files_seen"],
            counts="; ".join(f"{reason}: {count}" for reason, count in sorted(data["discovery"]["skip_reason_counts"].items())) or "none",
        ),
        "",
        "## BLOCKERS",
        "",
        "None. Some candidate artifacts have no prompt field, which means prompt-word exclusion could not be applied to those rows; this limitation is recorded per file/run. The short completion fragment dataset is reported as likely fragmentary and should not be interpreted as full answers.",
        "",
    ])
    return "\n".join(lines)


def run_sweep(results_root: Path, output_dir: Path, root_report: Path | None = None) -> dict[str, Any]:
    files = sorted(
        path for path in results_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}
    )
    file_items = []
    generated_flags = (output_dir / "flags.json").resolve()
    for path in files:
        item = extract_file(path, results_root)
        if path.resolve() == generated_flags:
            item["rows"] = []
            item["disposition"] = "skipped"
            item["skip_reason"] = "generated sweep output flags.json, not a prior run"
            item["arm_evidence"] = {}
            item["answer_evidence"] = {}
            item["prompt_evidence"] = {}
        file_items.append(item)
    candidates = [item for item in file_items if item["disposition"] == "candidate"]
    skipped = [item for item in file_items if item["disposition"] == "skipped"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        grouped[run_id_for_path(item["path"])].append(item)
    runs = [analyze_run(run_id, items) for run_id, items in sorted(grouped.items())]
    runs.sort(key=lambda item: item["run_id"])
    no_random_null = sorted(
        (flag for run in runs for flag in run["no_random_null_flags"]),
        key=lambda flag: (-flag["fold_change"], flag["run_id"].lower(), flag["arm"].lower(), flag["word"]),
    )
    skip_counts = Counter(item["skip_reason"] for item in skipped)
    discovery = {
        "files_seen": len(files),
        "candidate_file_count": len(candidates),
        "skipped_file_count": len(skipped),
        "skip_reason_counts": dict(sorted(skip_counts.items())),
        "candidate_files": [
            {
                "path": item["path"],
                "run_id": run_id_for_path(item["path"]),
                "matched_schema": item["matched_schema"],
                "record_count": len(item["rows"]),
                "arm_field_evidence": item["arm_evidence"],
                "answer_field_evidence": item["answer_evidence"],
                "prompt_field_evidence": item["prompt_evidence"],
                "parse_warnings": item["errors"],
            }
            for item in candidates
        ],
        "skipped_files": [
            {
                "path": item["path"],
                "reason": item["skip_reason"],
                "arm_field_evidence": item["arm_evidence"],
                "answer_field_evidence": item["answer_evidence"],
                "prompt_field_evidence": item["prompt_evidence"],
                "parse_warnings": item["errors"],
            }
            for item in skipped
        ],
    }
    data = {
        "title": "Sweep side effects in prior answer runs",
        "generated_by": "sweep_sideeffects.py",
        "rules": {
            "tokenization": "lowercase Unicode letter words with internal apostrophes",
            "counts": "number of answers containing word; each answer contributes at most once",
            "scaling": "arm answer-word counts scaled to the control arm's answer count; control remains raw",
            "up": "arm_scaled >= 5 and arm_scaled >= 3 * (control_count + 1)",
            "down": "control_count >= 5 and control_count >= 3 * (arm_scaled + 1)",
            "exclusions": "per-answer prompt words and STOPWORDS in sweep_sideeffects.py",
            "random_null": "number of random/control-unit arms with same word and same flag direction",
            "controls": "fixed label/metadata matcher; selected control recorded in each run",
            "name_census": "capitalized non-sentence-initial words absent from that record's prompt; occurrence counts",
        },
        "stopwords": sorted(STOPWORDS),
        "discovery": discovery,
        "runs": runs,
        "no_random_null_flags": no_random_null,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "flags.json").write_text(
        json.dumps(json_safe(data), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = make_report(data)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    if root_report is not None:
        root_report.write_text(report, encoding="utf-8")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--root-report", type=Path, default=ROOT / "REPORT-sweep-side.md")
    args = parser.parse_args()
    data = run_sweep(args.results_dir.resolve(), args.output_dir.resolve(), args.root_report.resolve())
    print(json.dumps({
        "files_seen": data["discovery"]["files_seen"],
        "candidate_files": data["discovery"]["candidate_file_count"],
        "skipped_files": data["discovery"]["skipped_file_count"],
        "run_groups": len(data["runs"]),
        "analyzed_runs": sum(run["status"] == "analyzed" for run in data["runs"]),
        "target_flags": sum(len(run["target_flags"]) for run in data["runs"]),
        "no_random_null_flags": len(data["no_random_null_flags"]),
    }, indent=2))


if __name__ == "__main__":
    main()
