#!/usr/bin/env python3
"""Part 37 generation, record construction, hidden-state capture and stated readout."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

MODEL_INFO = {
    "qwen": {"id": "Qwen/Qwen3-1.7B", "revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"},
    "olmo": {"id": "allenai/OLMo-2-0425-1B-Instruct", "revision": "48d788eca847d4d7548f375ad03d3c9312f6139e"},
    "smol": {"id": "HuggingFaceTB/SmolLM3-3B", "revision": "a07cc9a04f16550a088caea529712d1d335b0ac1"},
}
FOLLOWUP = "Thanks. Here is my next question: what is a good way to start the day?"
FOLLOWUP_PREFIX = "Here is an answer to this:"
HEADERS = [
    ("self", "Your earlier notes", "Notes you wrote earlier"),
    ("other", "Notes from another assistant", "Notes written by a different assistant"),
    ("neutral", "Notes", "Some notes"),
    ("user", "Notes from the user", "Notes the user wrote"),
]
IDENTITY_TERMS = [
    "Qwen", "Alibaba", "OLMo", "Ai2", "AllenAI", "Claude", "Anthropic", "OpenAI",
    "ChatGPT", "GPT", "SmolLM", "Hugging Face", "assistant", "language model", "as an AI",
]
EDIT_WORDS = [
    "apple", "book", "bird", "bread", "bridge", "bucket", "button", "camera", "candle", "carpet",
    "chair", "clock", "cloud", "coat", "corner", "desk", "door", "farm", "field", "floor", "flower",
    "forest", "garden", "glass", "grass", "house", "island", "kettle", "lake", "lamp", "leaf",
    "market", "meadow", "mirror", "mountain", "music", "orange", "paddle", "pencil", "pillow",
    "planet", "pocket", "rabbit", "ribbon", "river", "road", "school", "shadow", "shelter",
    "silver", "window", "basket", "bottle", "station", "village", "ticket", "table", "train",
    "valley", "vessel", "wheel", "wood", "yard", "cabin", "cabinets", "lantern", "saddle",
    "cushion", "notebook", "cupboard", "painting", "sidewalk", "staircase", "sunlight", "waterfall",
    "backpack", "blueberry", "doorway", "cottage", "playground", "footpath", "teapot", "sandwich",
    "sideboard", "keyboard", "fireplace", "bookcase", "bedroom", "sunflower", "seashell",
    "birdhouse", "snowflake", "campground", "chocolate", "windmill", "watermelon", "newspaper",
    "shoelace", "raincoat", "strawberry", "bookshelf", "pineapple", "lighthouse", "toothbrush",
    "skateboard",
]
EDIT_STOPWORDS = frozenset({
    "about", "after", "again", "against", "also", "among", "because", "before", "being", "below",
    "between", "both", "could", "every", "first", "from", "having", "other", "should", "since",
    "their", "there", "these", "those", "through", "under", "until", "very", "while", "would",
    "where", "which", "whose", "with", "without", "your", "yours", "them", "then", "when", "were",
    "what", "that", "this", "they", "have", "some", "more", "most", "much", "many", "make", "made",
    "just", "into", "over", "only", "such", "than", "down", "back", "well", "good", "nice", "keep",
    "take", "find", "move", "open", "small", "little", "really", "usually", "another", "anything",
    "something", "everything", "important", "sometimes", "understand", "experience", "interesting",
})
STATEMENT_FORMS = {
    "authorship_yn": {"prompt": "Did you write the text above? Answer yes or no.", "yes": "yn"},
    "memory_yn": {"prompt": "Is the text above from your earlier notes? Answer yes or no.", "yes": "yn"},
    "authorship_ab": {"prompt": "Who wrote the text above? (A) you (B) someone else. Answer A or B.", "yes": "A"},
    "authorship_ab_swapped": {"prompt": "Who wrote the text above? (A) someone else (B) you. Answer A or B.", "yes": "B"},
    "memory_ab": {"prompt": "Where is the text above from? (A) your earlier notes (B) somewhere else. Answer A or B.", "yes": "A"},
    "memory_ab_swapped": {"prompt": "Where is the text above from? (A) somewhere else (B) your earlier notes. Answer A or B.", "yes": "B"},
}


class Part37Error(RuntimeError):
    """Invalid Part 37 input or unsupported model state."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise Part37Error(f"{path}:{number}: invalid JSONL") from exc
        if not isinstance(row, dict):
            raise Part37Error(f"{path}:{number}: expected an object")
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")


def _atomic_write(path: Path, content_writer: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent,
                                        prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temp_path = Path(stream.name)
            content_writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    def write(stream: Any) -> None:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    _atomic_write(path, write)


def write_json_atomic(path: Path, value: Any) -> None:
    _atomic_write(path, lambda stream: stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n"))


def _append_jsonl_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
                      for row in rows)
    if not payload:
        return
    with path.open("ab") as stream:
        stream.write(payload.encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_questions(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) != 100:
        raise Part37Error(f"{path}: expected exactly 100 question objects")
    expected_categories = ["opinion"] * 35 + ["decision"] * 35 + ["explanation"] * 30
    for index, row in enumerate(data):
        if not isinstance(row, dict) or row.get("id") != f"q{index:03d}":
            raise Part37Error(f"question {index}: expected id q{index:03d}")
        if row.get("category") != expected_categories[index] or not isinstance(row.get("question"), str):
            raise Part37Error(f"question {index}: wrong category or missing question text")
    return data


def _chat_template(tokenizer: Any, messages: list[dict[str, str]], reader: str,
                   add_generation_prompt: bool = True) -> str:
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
    if reader == "qwen":
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(messages, **kwargs)


def render_generation_prompt(tokenizer: Any, model: str, question: str) -> str:
    messages: list[dict[str, str]] = []
    if model == "smol":
        messages.append({"role": "system", "content": "/no_think"})
    messages.append({"role": "user", "content": question})
    # SmolLM3's model card specifies /no_think in the system prompt. Qwen3 uses its
    # template's enable_thinking=False setting; OLMo has no thinking mode switch.
    template_reader = "qwen" if model == "qwen" else "olmo"
    return _chat_template(tokenizer, messages, template_reader, add_generation_prompt=True)


def _generation_stops(tokenizer: Any) -> list[int]:
    values: list[int] = []
    for name in ("eos_token_id", "eot_token_id"):
        value = getattr(tokenizer, name, None)
        if isinstance(value, int) and value not in values:
            values.append(value)
    vocab = getattr(tokenizer, "get_vocab", lambda: {})()
    for marker in ("<|im_end|>", "<|eot_id|>", "<|end_of_turn|>"):
        if marker in vocab:
            token_id = tokenizer.convert_tokens_to_ids(marker)
            if isinstance(token_id, int) and token_id not in values:
                values.append(token_id)
    if not values:
        raise Part37Error("tokenizer has no end-of-turn/eos token id")
    return values


def _seed_torch(torch: Any, seed: int, device: str) -> None:
    torch.manual_seed(seed)
    if device == "xpu" and getattr(torch, "xpu", None) is not None:
        torch.xpu.manual_seed_all(seed)
    if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(model_key: str, device: str) -> tuple[Any, Any, Any, str, str]:
    if model_key not in MODEL_INFO:
        raise Part37Error(f"unknown model {model_key!r}")
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise Part37Error(f"torch and transformers are required at runtime: {exc}") from exc
    if device != "xpu":
        raise Part37Error("Part 37 model commands are pinned to --device xpu")
    if not getattr(torch, "xpu", None) or not torch.xpu.is_available():
        raise Part37Error("requested XPU is unavailable; no CPU fallback is permitted")
    info = MODEL_INFO[model_key]
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            info["id"], revision=info["revision"], use_fast=True, local_files_only=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            info["id"], revision=info["revision"], torch_dtype=torch.bfloat16, local_files_only=True
        ).eval()
    except Exception as exc:
        raise Part37Error(f"could not load pinned local snapshot {info['id']}@{info['revision']}: {exc}") from exc
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to("xpu")
    return torch, tokenizer, model, "xpu", transformers.__version__


def load_reader_tokenizer(reader: str) -> tuple[Any, str]:
    """Load only the pinned tokenizer for deterministic record construction."""
    try:
        import transformers
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise Part37Error(f"transformers is required for tokenization: {exc}") from exc
    info = MODEL_INFO[reader]
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            info["id"], revision=info["revision"], use_fast=True, local_files_only=True
        )
    except Exception as exc:
        raise Part37Error(f"could not load pinned local tokenizer {info['id']}@{info['revision']}: {exc}") from exc
    return tokenizer, transformers.__version__


def _encode(tokenizer: Any, text: str) -> list[int]:
    return [int(value) for value in tokenizer.encode(text, add_special_tokens=False)]


def _decode(tokenizer: Any, tokens: list[int]) -> str:
    return tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)


DERIVED_FROM_OWN_G = ("edit", "null_shuffle")


def _first_tokens(tokenizer: Any, text: str, target: int) -> tuple[str, int, int]:
    ids = _encode(tokenizer, text)
    if len(ids) < target:
        return "", len(ids), len(ids) - target
    decoded = _decode(tokenizer, ids[:target])
    actual = len(_encode(tokenizer, decoded))
    return decoded, actual, actual - target


def _match_pattern(term: str) -> re.Pattern[str]:
    pieces = [re.escape(piece) for piece in term.split()]
    body = r"\s+".join(pieces)
    return re.compile(rf"(?<!\w){body}(?!\w)", re.IGNORECASE)


def identity_matches(text: str) -> list[str]:
    return [term for term in IDENTITY_TERMS if _match_pattern(term).search(text)]


def screen_rows(generation_rows: list[dict[str, Any]]) -> dict[str, Any]:
    dropped: dict[str, set[str]] = {}
    question_ids = []
    for row in generation_rows:
        qid, text = row.get("question_id"), row.get("text")
        if not isinstance(qid, str) or not isinstance(text, str):
            raise Part37Error("generation rows need question_id and text fields")
        if qid not in question_ids:
            question_ids.append(qid)
        matches = identity_matches(text)
        if matches:
            dropped.setdefault(qid, set()).update(matches)
    return {
        "schema": "part37-screen-v1",
        "identity_terms": IDENTITY_TERMS,
        "kept": [qid for qid in question_ids if qid not in dropped],
        "dropped": [{"question_id": qid, "matching_words": sorted(terms, key=str.casefold)}
                    for qid, terms in ((qid, dropped[qid]) for qid in question_ids if qid in dropped)],
        "generation_row_count": len(generation_rows),
    }


def _words(text: str) -> list[str]:
    return text.split()


def minimal_edit(text: str) -> tuple[str, dict[str, str] | None]:
    matches = [match for match in re.finditer(r"\b[^\W\d_]{4,}\b", text, re.UNICODE)
               if match.group(0).casefold() not in EDIT_STOPWORDS]
    if not matches:
        return "", None
    midpoint = (len(text) - 1) / 2
    target = min(matches, key=lambda match: (abs((match.start() + match.end() - 1) / 2 - midpoint), match.start()))
    original = target.group(0)
    # Chosen by a hash of the original word, not first match: first match put "apple" in 7 of 8 edits (DS round 2 M4).
    candidates = [word for word in EDIT_WORDS
                  if word.casefold() != original.casefold() and abs(len(word) - len(original)) <= 1]
    replacement = None
    if candidates:
        digest = int(hashlib.sha256(original.casefold().encode("utf-8")).hexdigest(), 16)
        replacement = candidates[digest % len(candidates)]
    if replacement is None:
        return "", None
    if original.isupper():
        replacement = replacement.upper()
    elif original[:1].isupper():
        replacement = replacement.capitalize()
    edited = text[:target.start()] + replacement + text[target.end():]
    return edited, {"original": original, "replacement": replacement}


def pile_span(texts: list[str], tokenizer: Any, question_index: int, target_tokens: int) -> str:
    words = [word for text in texts for word in re.findall(r"\S+", text)]
    if not words:
        raise Part37Error("Pile file contained no text words")
    start = (question_index * 7919) % len(words)
    for offset in range(len(words)):
        chosen: list[str] = []
        for step in range(len(words)):
            chosen.append(words[(start + offset + step) % len(words)])
            candidate = " ".join(chosen)
            if len(_encode(tokenizer, candidate)) >= target_tokens:
                ids = _encode(tokenizer, candidate)[:target_tokens]
                return _decode(tokenizer, ids)
    raise Part37Error(f"could not make a {target_tokens}-token Pile span")


def selected_pile_rows(path: Path, question_indices: list[int]) -> dict[int, tuple[int, str]]:
    """Read only the deterministic source rows needed for the kept questions."""
    row_count = 0
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise Part37Error(f"{path}:{line_number}: invalid Pile JSONL") from exc
            if not isinstance(value, dict) or not isinstance(value.get("text"), str):
                raise Part37Error(f"{path}:{line_number}: expected a string text field")
            row_count += 1
    if row_count == 0:
        raise Part37Error(f"{path}: empty Pile corpus")
    targets = {index % row_count for index in question_indices}
    selected: dict[int, str] = {}
    row_index = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            if row_index in targets:
                value = json.loads(line)
                selected[row_index] = value["text"]
            row_index += 1
    return {question_index: (question_index % row_count, selected[question_index % row_count])
            for question_index in question_indices}


def _record_messages(question: str, span: str, slot: str, frame: str,
                     header: str | None = None) -> list[dict[str, str]]:
    if frame == "label":
        labeled = f"{header}:\n\n{span}"
        if slot == "user":
            return [{"role": "user", "content": labeled}, {"role": "assistant", "content": "Noted."},
                    {"role": "user", "content": FOLLOWUP}]
        return [{"role": "user", "content": "Please show me the notes."},
                {"role": "assistant", "content": labeled}, {"role": "user", "content": FOLLOWUP}]
    if slot == "assistant":
        return [{"role": "user", "content": question}, {"role": "assistant", "content": span},
                {"role": "user", "content": FOLLOWUP}]
    return [{"role": "user", "content": f"{question}\n\n{FOLLOWUP_PREFIX}\n\n{span}"},
            {"role": "assistant", "content": "Noted."}, {"role": "user", "content": FOLLOWUP}]


def _row_map(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("question_id")), str(row.get("model")), str(row.get("kind")))
        if key in result:
            raise Part37Error(f"duplicate generation row {key}")
        result[key] = row
    return result


def build_records(reader: str, generation_rows: list[dict[str, Any]], kept_data: dict[str, Any],
                  questions: list[dict[str, Any]], pile_path: Path, tokenizer: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if reader not in ("qwen", "olmo"):
        raise Part37Error("--reader must be qwen or olmo")
    rows_by_key = _row_map(generation_rows)
    kept = [qid for qid in kept_data.get("kept", []) if isinstance(qid, str)]
    if not kept:
        raise Part37Error("screen result has no kept question ids")
    questions_by_id = {row["id"]: row for row in questions}
    unknown = [qid for qid in kept if qid not in questions_by_id]
    if unknown:
        raise Part37Error(f"screen file references unknown question ids: {unknown[:3]}")
    pile_texts = selected_pile_rows(pile_path, list(range(len(kept))))

    def gen(qid: str, model: str, kind: str) -> str:
        row = rows_by_key.get((qid, model, kind))
        if row is None:
            raise Part37Error(f"missing generation {qid}/{model}/{kind}")
        return str(row["text"])

    other_model = "olmo" if reader == "qwen" else "qwen"
    dropped_short: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    eligible: list[str] = []
    qid_to_index = {qid: index for index, qid in enumerate(kept)}

    for qid in kept:
        question = questions_by_id[qid]["question"]
        own_text = gen(qid, reader, "greedy")
        other_text = gen(qid, other_model, "greedy")
        own_ids, other_ids = _encode(tokenizer, own_text), _encode(tokenizer, other_text)
        target = min(len(own_ids), len(other_ids))
        if target < 16:
            dropped_short.append({"question_id": qid, "reader_tokens": target, "reason": "L < 16"})
            continue
        eligible.append(qid)
        own_g, _, _ = _first_tokens(tokenizer, own_text, target)
        other_g, _, _ = _first_tokens(tokenizer, other_text, target)
        next_qid = kept[(qid_to_index[qid] + 1) % len(kept)]
        own_diff_text = gen(next_qid, reader, "greedy")
        own_s1_text = gen(qid, reader, "s1")
        own_s2_text = gen(qid, reader, "s2")
        null_smol_text = gen(qid, "smol", "greedy")
        edit_text, edit_pair = minimal_edit(own_g)
        null_shuffle_text = " ".join(random.Random(f"part37-null:{qid}:37").sample(_words(own_g), len(_words(own_g))))
        question_index = qid_to_index[qid]
        # A selected row can be shorter than L tokens; then take the next question's row, deterministically, and
        # record which row was used (Claude, 2026-10-25: the first full run stopped on a short row at L = 96).
        null_pile_text = None
        for step in range(len(pile_texts)):
            pile_row_index, pile_row_text = pile_texts[(question_index + step) % len(pile_texts)]
            try:
                null_pile_text = pile_span([pile_row_text], tokenizer, question_index, target)
                break
            except Part37Error:
                continue
        if null_pile_text is None:
            raise Part37Error(f"{qid}: no Pile row gives a {target}-token span")

        cells = [
            ("own_g", own_g, "own", "assistant", "base", True),
            ("other_g", other_g, "other", "assistant", "base", True),
            ("own_g", own_g, "own", "user", "base", True),
            ("other_g", other_g, "other", "user", "base", True),
            ("own_s1", own_s1_text, "own", "assistant", "base", True),
            ("own_s2", own_s2_text, "own", "assistant", "base", False),
            ("own_diffq", own_diff_text, "own", "assistant", "base", True),
            ("edit", edit_text, "edit", "assistant", "base", True),
            ("null_smol", null_smol_text, "null", "assistant", "base", True),
            ("null_pile", null_pile_text, "null", "assistant", "base", True),
            ("null_shuffle", null_shuffle_text, "null", "assistant", "base", True),
        ]
        for span_kind, raw_text, author, slot, frame, stated_include in cells:
            if span_kind in DERIVED_FROM_OWN_G:
                # Built from own_g, which is already L tokens: a word swap or shuffle can change the token count by a
                # few, and cutting to L would empty the cell exactly when the edit changed the tokenisation. Keep the
                # text as built and record its count (Claude, 2026-10-25, after the fixture showed an empty edit).
                span = raw_text
                actual = len(_encode(tokenizer, span))
                mismatch = actual - target
            else:
                span, actual, mismatch = _first_tokens(tokenizer, raw_text, target)
            record_id = f"{qid}:{frame}:{span_kind}:{slot}"
            records.append({
                "record_id": record_id, "question_id": qid, "reader": reader,
                "source_question_id": next_qid if span_kind == "own_diffq" else qid,
                "question": question, "span_kind": span_kind, "author": author, "slot": slot,
                "frame": frame, "header": None, "header_group": None, "header_variant": None,
                "span": span, "target_tokens": target, "span_token_count": actual,
                "token_mismatch_count": mismatch, "empty_reason": "span shorter than L" if not span else None,
                "messages": _record_messages(question, span, slot, frame),
                "edit_pair": edit_pair if span_kind == "edit" else None,
                "source_pile_row": pile_row_index if span_kind == "null_pile" else None,
                "stated_include": bool(stated_include and span),
            })

        for header_group, canonical, paraphrase in HEADERS:
            for header_variant, header in (("canonical", canonical), ("paraphrase", paraphrase)):
                for span_kind, source_text, author in (("own_g", own_g, "own"), ("other_g", other_g, "other")):
                    span, actual, mismatch = _first_tokens(tokenizer, source_text, target)
                    for slot in ("user", "assistant"):
                        record_id = f"{qid}:label:{header_group}:{header_variant}:{span_kind}:{slot}"
                        records.append({
                            "record_id": record_id, "question_id": qid, "reader": reader,
                            "source_question_id": qid, "question": question, "span_kind": span_kind,
                            "author": author, "slot": slot, "frame": "label", "header": header,
                            "header_group": header_group, "header_variant": header_variant,
                            "span": span, "target_tokens": target, "span_token_count": actual,
                            "token_mismatch_count": mismatch, "empty_reason": None,
                            "messages": _record_messages(question, span, slot, "label", header),
                            "edit_pair": None, "source_pile_row": None, "stated_include": bool(span),
                        })
    summary = {
        "schema": "part37-records-v1", "reader": reader, "kept_question_ids": kept,
        "reader_eligible_question_ids": eligible, "dropped_short": dropped_short,
        "record_count": len(records), "empty_span_records": sum(not row["span"] for row in records),
        "retokenization_mismatches": sum(abs(int(row["token_mismatch_count"])) for row in records),
        "pile_sha256": sha256_file(pile_path), "generation_count": len(generation_rows),
    }
    return records, summary


def _token_offsets(tokenizer: Any, rendered: str) -> tuple[list[int], list[tuple[int, int]]]:
    try:
        encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
    except Exception as exc:
        raise Part37Error(f"fast tokenizer with offset_mapping is required for exact readout positions: {exc}") from exc
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if hasattr(offsets, "tolist"):
        offsets = offsets.tolist()
    if ids and isinstance(ids[0], list):
        ids, offsets = ids[0], offsets[0]
    return [int(value) for value in ids], [(int(start), int(end)) for start, end in offsets]


def locate_readout_positions(record: dict[str, Any], tokenizer: Any, reader: str) -> dict[str, Any]:
    rendered = _chat_template(tokenizer, record["messages"], reader, add_generation_prompt=True)
    span = str(record.get("span", ""))
    followup = str(record["messages"][-1]["content"])
    if not span:
        raise Part37Error(f"{record.get('record_id')}: empty span has no readout position")
    span_start = rendered.rfind(span)
    followup_start = rendered.rfind(followup)
    if span_start < 0 or followup_start < 0:
        raise Part37Error(f"{record.get('record_id')}: span or follow-up was lost in the chat template")
    ids, offsets = _token_offsets(tokenizer, rendered)
    span_end = span_start + len(span)
    span_indices = [index for index, (start, end) in enumerate(offsets)
                    if end > start and end > span_start and start < span_end]
    followup_indices = [index for index, (start, end) in enumerate(offsets)
                        if end > start and end > followup_start and start < followup_start + len(followup)]
    if not span_indices or not followup_indices:
        raise Part37Error(f"{record.get('record_id')}: unable to map readout text to token positions")
    return {
        "rendered_prompt": rendered, "input_ids": ids,
        "span_token_positions": span_indices,
        "span_last_index": span_indices[-1], "followup_first_index": followup_indices[0],
        "span_char_range": [span_start, span_end],
        "followup_char_range": [followup_start, followup_start + len(followup)],
    }


def answer_token_sets(tokenizer: Any) -> dict[str, list[int]]:
    variants = {
        "yes": ["Yes", "yes", " Yes", " yes"],
        "no": ["No", "no", " No", " no"],
        "a": ["A", " A"], "b": ["B", " B"],
    }
    result = {key: sorted({ids[0] for text in values if len(ids := _encode(tokenizer, text)) == 1})
              for key, values in variants.items()}
    for key in variants:
        if not result[key]:
            raise Part37Error(f"tokenizer has no single-token variant for answer {key!r}")
    if set(result["yes"]) & set(result["no"]) or set(result["a"]) & set(result["b"]):
        raise Part37Error("yes/no or A/B answer token sets overlap")
    return result


def _mass(probabilities: Any, token_ids: list[int]) -> float:
    valid = [token_id for token_id in token_ids if 0 <= token_id < probabilities.shape[-1]]
    if not valid:
        return 0.0
    return float(probabilities[valid].sum().item())


def _model_input(torch: Any, tokenizer: Any, rendered: str, device: str) -> dict[str, Any]:
    encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    return {key: value.to(device) for key, value in encoded.items()}


def command_generate(args: argparse.Namespace) -> int:
    questions = read_questions(args.questions)
    if args.limit is not None:
        if args.limit < 1:
            raise Part37Error("--limit must be positive")
        questions = questions[:args.limit]
    torch, tokenizer, model, device, transformers_version = load_model(args.model, args.device)
    stops = _generation_stops(tokenizer)
    generated: list[dict[str, Any]] = []
    for question in questions:
        prompt = render_generation_prompt(tokenizer, args.model, question["question"])
        encoded = _model_input(torch, tokenizer, prompt, device)
        kinds = [("greedy", None)] if args.model == "smol" else [("greedy", None), ("s1", 1), ("s2", 2)]
        for kind, seed in kinds:
            if seed is not None:
                _seed_torch(torch, seed, device)
            with torch.inference_mode():
                output = model.generate(
                    **encoded, max_new_tokens=96, do_sample=seed is not None,
                    temperature=0.7 if seed is not None else None,
                    top_p=0.95 if seed is not None else None,
                    num_beams=1, eos_token_id=stops,
                    pad_token_id=tokenizer.pad_token_id,
                )
            continuation = output[0, encoded["input_ids"].shape[-1]:].detach().cpu().tolist()
            generated.append({"question_id": question["id"], "model": args.model, "kind": kind,
                              "text": _decode(tokenizer, continuation), "revision": MODEL_INFO[args.model]["revision"]})
        print(f"{args.model} {question['id']} complete", flush=True)
    write_jsonl_atomic(args.out, generated)
    manifest = {
        "schema": "part37-generation-v1", "model": args.model, "model_id": MODEL_INFO[args.model]["id"],
        "revision": MODEL_INFO[args.model]["revision"], "device": device, "max_new_tokens": 96,
        "generation_kinds": ["greedy"] if args.model == "smol" else ["greedy", "s1", "s2"],
        "temperature": 0.7, "top_p": 0.95, "sample_seeds": [1, 2],
        "non_thinking": {"qwen": "enable_thinking=False", "smol": "/no_think system prompt",
                         "olmo": "model's standard instruct template"}[args.model],
        "question_count": len(questions), "input_sha256": sha256_file(args.questions),
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "torch_version": torch.__version__, "transformers_version": transformers_version,
    }
    write_json_atomic(args.out.with_name(args.out.name + ".manifest.json"), manifest)
    return 0


def command_screen(args: argparse.Namespace) -> int:
    rows = [row for path in args.gens for row in read_jsonl(path)]
    result = screen_rows(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"kept={len(result['kept'])} dropped={len(result['dropped'])}")
    return 0


def command_build_records(args: argparse.Namespace) -> int:
    questions = read_questions(args.questions)
    generation_rows = [row for path in args.gens for row in read_jsonl(path)]
    kept = json.loads(args.kept.read_text(encoding="utf-8"))
    tokenizer, transformers_version = load_reader_tokenizer(args.reader)
    records, manifest = build_records(args.reader, generation_rows, kept, questions, args.pile, tokenizer)
    write_jsonl(args.out, records)
    manifest.update({"generation_paths": [str(path) for path in args.gens],
                     "screen_sha256": sha256_file(args.kept),
                     "transformers_version": transformers_version,
                     "code_sha256": sha256_file(Path(__file__).resolve())})
    args.out.with_name(args.out.name + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"records={len(records)} reader_dropped_short={len(manifest['dropped_short'])} "
          f"empty={manifest['empty_span_records']} retokenization_mismatches={manifest['retokenization_mismatches']}")
    return 0


def _capture_one(torch: Any, model: Any, tokenizer: Any, record: dict[str, Any], reader: str,
                 device: str) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    location = locate_readout_positions(record, tokenizer, reader)
    ids = torch.tensor([location["input_ids"]], dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    with torch.inference_mode():
        output = model(input_ids=ids, attention_mask=mask, use_cache=False, output_hidden_states=True,
                       return_dict=True)
    hidden = output.hidden_states
    if hidden is None or len(hidden) < 2:
        raise Part37Error(f"{record['record_id']}: model did not return embedding/layer hidden states")
    expected_layer_count = getattr(getattr(model, "config", None), "num_hidden_layers", None)
    if not isinstance(expected_layer_count, int):
        backbone = getattr(model, "model", None)
        layers = getattr(backbone, "layers", None) or getattr(backbone, "blocks", None)
        if layers is None:
            transformer = getattr(backbone, "transformer", None)
            layers = getattr(transformer, "blocks", None)
        expected_layer_count = len(layers) if layers is not None else len(hidden) - 1
    if len(hidden) != expected_layer_count + 1:
        raise Part37Error(f"{record['record_id']}: expected embedding plus {expected_layer_count} decoder layers, got {len(hidden)} states")
    if max(location["span_last_index"], location["followup_first_index"]) >= int(ids.shape[1]):
        raise Part37Error(f"{record['record_id']}: located position lies outside the forward sequence")
    positions = [location["span_last_index"], location["followup_first_index"]]
    state_rows = [state[0, positions, :].detach().to(dtype=torch.float32).cpu().numpy() for state in hidden]
    numpy = __import__("numpy")
    feature = numpy.stack(state_rows, axis=1).astype(numpy.float32, copy=False)  # [positions, layers+embedding, hidden]
    if not numpy.isfinite(feature).all():
        raise Part37Error(f"{record['record_id']}: non-finite hidden state")
    token_positions = location["span_token_positions"]
    if not token_positions or token_positions[0] < 1:
        raise Part37Error(f"{record['record_id']}: span token positions lack a preceding context token")
    span_ids = [int(location["input_ids"][position]) for position in token_positions]
    logits = output.logits[0].to(dtype=torch.float32)
    token_logprobs: list[float] = []
    token_entropies: list[float] = []
    for token_position, token_id in zip(token_positions, span_ids):
        distribution = torch.log_softmax(logits[token_position - 1], dim=-1)
        probabilities = distribution.exp()
        token_logprobs.append(float(distribution[token_id].item()))
        token_entropies.append(float((-(probabilities * distribution).sum()).item()))
    if not numpy.isfinite(numpy.asarray(token_logprobs + token_entropies, dtype=numpy.float64)).all():
        raise Part37Error(f"{record['record_id']}: non-finite span log-probability or entropy")
    metrics = {
        "record_id": record["record_id"], "question_id": record["question_id"],
        "per_token_logprob": token_logprobs, "per_token_entropy": token_entropies,
        "mean_logprob": sum(token_logprobs) / len(token_logprobs),
        "minimum_logprob": min(token_logprobs), "mean_entropy": sum(token_entropies) / len(token_entropies),
        "last_token_logprob": token_logprobs[-1],
    }
    index = {key: value for key, value in record.items() if key != "messages"}
    index.update({"span_last_index": positions[0], "followup_first_index": positions[1],
                  "span_token_positions": token_positions, "input_tokens": len(location["input_ids"]),
                  "record_tokens": len(location["input_ids"]), "readout_indices": positions,
                  "rendered_prompt": location["rendered_prompt"]})
    return feature, index, metrics


def command_capture(args: argparse.Namespace) -> int:
    records = read_jsonl(args.records)
    torch, tokenizer, model, device, transformers_version = load_model(args.reader, args.device)
    import numpy as np
    args.out.mkdir(parents=True, exist_ok=True)
    # A completed question is the persistence unit. Its shard is atomically
    # installed before its metrics and index rows are appended, leaving a valid
    # capture prefix if the process stops between questions.
    indexes: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    features: list[Any] = []
    skipped: list[dict[str, str]] = []
    shard_number = 0
    write_jsonl_atomic(args.out / "index.jsonl", [])
    write_jsonl_atomic(args.out / "metrics.jsonl", [])

    def flush() -> None:
        nonlocal features, shard_number
        if not features:
            return
        array = np.stack(features).astype(np.float32, copy=False)
        shard_name = f"shard-{shard_number:05d}.npy"
        temp_path = args.out / f".{shard_name}.tmp"
        with temp_path.open("wb") as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, args.out / shard_name)
        start = len(indexes) - len(features)
        for offset, index_row in enumerate(indexes[start:], 0):
            index_row["shard"] = shard_name
            index_row["shard_row"] = offset
        _append_jsonl_rows(args.out / "metrics.jsonl", metrics[start:])
        _append_jsonl_rows(args.out / "index.jsonl", indexes[start:])
        shard_number += 1
        features = []

    current_question: str | None = None
    completed_questions: set[str] = set()
    for record in records:
        question_id = str(record.get("question_id"))
        if current_question is not None and question_id != current_question:
            flush()
            completed_questions.add(current_question)
        if question_id in completed_questions:
            raise Part37Error(f"capture records are not grouped by question_id: {question_id}")
        current_question = question_id
        if not record.get("span"):
            skipped.append({"record_id": str(record.get("record_id")), "reason": str(record.get("empty_reason") or "empty span")})
            continue
        feature, index_row, metrics_row = _capture_one(torch, model, tokenizer, record, args.reader, device)
        features.append(feature)
        indexes.append(index_row)
        metrics.append(metrics_row)
        print(f"captured {record['record_id']}", flush=True)
    flush()
    manifest = {
        "schema": "part37-capture-v1", "reader": args.reader, "model": MODEL_INFO[args.reader]["id"],
        "revision": MODEL_INFO[args.reader]["revision"], "device": device,
        "record_input_sha256": sha256_file(args.records), "code_sha256": sha256_file(Path(__file__).resolve()),
        "torch_version": torch.__version__, "transformers_version": transformers_version,
        "captured_records": len(indexes), "skipped_empty_records": skipped,
        "shard_persistence_unit": "one completed question", "one_forward_per_record": True,
        "feature_layout": "[records, 2 readout positions, embedding plus each decoder layer, hidden width] float32",
    }
    write_json_atomic(args.out / "manifest.json", manifest)
    return 0


def command_stated(args: argparse.Namespace) -> int:
    records = read_jsonl(args.records)
    torch, tokenizer, model, device, transformers_version = load_model(args.reader, args.device)
    token_sets = answer_token_sets(tokenizer)
    results: list[dict[str, Any]] = []
    for record in records:
        if not record.get("stated_include") or not record.get("span"):
            continue
        for form_name, form in STATEMENT_FORMS.items():
            messages = list(record["messages"]) + [{"role": "user", "content": form["prompt"]}]
            rendered = _chat_template(tokenizer, messages, args.reader, add_generation_prompt=True)
            encoded = _model_input(torch, tokenizer, rendered, device)
            with torch.inference_mode():
                output = model(**encoded, use_cache=False, return_dict=True)
            probabilities = torch.softmax(output.logits[0, -1].float(), dim=-1)
            if form["yes"] == "yn":
                yes_mass = _mass(probabilities, token_sets["yes"])
                no_mass = _mass(probabilities, token_sets["no"])
            else:
                yes_mass = _mass(probabilities, token_sets[form["yes"].lower()])
                no_key = "b" if form["yes"] == "A" else "a"
                no_mass = _mass(probabilities, token_sets[no_key])
            total = yes_mass + no_mass
            result = {
                "record_id": record["record_id"], "question_id": record["question_id"],
                "reader": args.reader, "form": form_name, "span_kind": record["span_kind"],
                "author": record["author"], "slot": record["slot"], "frame": record["frame"],
                "header": record.get("header"), "header_group": record.get("header_group"),
                "header_variant": record.get("header_variant"), "yes_mass": yes_mass,
                "no_mass": no_mass, "total_two_token_mass": total,
                "p_yes": yes_mass / total if total > 0 else None,
                "answer_token_ids": token_sets,
            }
            results.append(result)
        print(f"stated {record['record_id']}", flush=True)
    write_jsonl(args.out, results)
    manifest = {
        "schema": "part37-stated-v1", "reader": args.reader, "revision": MODEL_INFO[args.reader]["revision"],
        "device": device, "record_input_sha256": sha256_file(args.records),
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "torch_version": torch.__version__, "transformers_version": transformers_version,
        "row_count": len(results), "forms": list(STATEMENT_FORMS), "answer_token_ids": token_sets,
        "mass_definition": "sum of next-token probabilities over the reader tokenizer's accepted single-token answer variants",
    }
    args.out.with_name(args.out.name + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("--model", required=True, choices=tuple(MODEL_INFO))
    generate.add_argument("--questions", type=Path, required=True)
    generate.add_argument("--out", type=Path, required=True)
    generate.add_argument("--limit", type=int)
    generate.add_argument("--device", required=True, choices=("xpu",))
    generate.set_defaults(func=command_generate)
    screen = sub.add_parser("screen")
    screen.add_argument("--gens", type=Path, nargs="+", required=True)
    screen.add_argument("--out", type=Path, required=True)
    screen.set_defaults(func=command_screen)
    records = sub.add_parser("build-records")
    records.add_argument("--reader", choices=("qwen", "olmo"), required=True)
    records.add_argument("--gens", type=Path, nargs="+", required=True)
    records.add_argument("--kept", type=Path, required=True)
    records.add_argument("--questions", type=Path, default=Path(__file__).with_name("questions.json"))
    records.add_argument("--pile", type=Path, required=True)
    records.add_argument("--out", type=Path, required=True)
    records.set_defaults(func=command_build_records)
    capture = sub.add_parser("capture")
    capture.add_argument("--reader", choices=("qwen", "olmo"), required=True)
    capture.add_argument("--records", type=Path, required=True)
    capture.add_argument("--out", type=Path, required=True)
    capture.add_argument("--device", choices=("xpu",), required=True)
    capture.set_defaults(func=command_capture)
    stated = sub.add_parser("stated")
    stated.add_argument("--reader", choices=("qwen", "olmo"), required=True)
    stated.add_argument("--records", type=Path, required=True)
    stated.add_argument("--out", type=Path, required=True)
    stated.add_argument("--device", choices=("xpu",), required=True)
    stated.set_defaults(func=command_stated)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except Part37Error as exc:
        print(f"part37: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
