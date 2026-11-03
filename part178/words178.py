#!/usr/bin/env python3
"""Part 178 definitions. Pure text/statistic functions; no torch import. (Design: Definitions)"""
from __future__ import annotations

import hashlib
import math
import re
import sys
import codecs
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # python -I leaves script folders off sys.path
from part177 import read177  # noqa: E402

PREFIX_TOKENS = 4096
WINDOW_TOKENS = 4000
ONSET_CENSOR_LINES = 40
IMPORTED_LIST = (
    "wait", "alternatively", "hmm", "but", "however", "alternative", "another", "check",
    "double-check", "oh", "maybe", "verify", "other", "again", "now", "ah", "any",
)
REFLECTION_FAMILY = frozenset(("wait", "hmm", "alternatively", "actually", "however", "but"))
ORDINARY_WORDS = ("the", "so", "we", "let", "then")
RARE_DIAGNOSTICS = ("river", "table")
_WORD_RE = re.compile(r"(?iu)[^\W\d_]+(?:[-'][^\W\d_]+)*")
_ABBREVIATIONS = ("e.g.", "i.e.", "etc.", "vs.")
_LIST_NUMBER_RE = re.compile(r"^\d{1,3}[.)]\s+")
_MARKUP_PREFIX_RE = re.compile(r"^[*_#>\"'“”‘’([{]+\s*")
_BULLET_RE = re.compile(r"^(?:[-+•‣▪◦–—]\s+)")
_MATH_START_RE = re.compile(r"^(?:\$|\\\(|\\\[|\\|\d|[=<>+\-−*/^%])")
_GPT2_BYTE_DECODER = None


def thinking_text(text: str) -> str:
    """Text before the first closing think tag, or all text if none. (Definitions)"""
    value = str(text or "")
    end = value.find("</think>")
    return value if end < 0 else value[:end]


def _input_ids(tokenizer, text: str):
    try:
        encoded = tokenizer(text, add_special_tokens=False)
        if isinstance(encoded, dict):
            ids = encoded.get("input_ids", [])
        else:
            ids = getattr(encoded, "input_ids", encoded)
    except TypeError:
        encoded = tokenizer.encode(text)
        ids = getattr(encoded, "ids", encoded)
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return [int(x) for x in ids]


def _decode(tokenizer, ids):
    try:
        return tokenizer.decode(list(map(int, ids)), skip_special_tokens=False,
                                clean_up_tokenization_spaces=False)
    except TypeError:
        return tokenizer.decode(list(map(int, ids)))


def prefix(text: str | None = None, *, tokenizer=None, kept_ids=None,
           limit: int = PREFIX_TOKENS) -> str:
    """Return a text's first token-bounded prefix; kept ids take precedence. (Definitions)"""
    limit = max(0, int(limit))
    if kept_ids is not None:
        if tokenizer is None:
            raise ValueError("tokenizer is required to decode kept token ids")
        return _decode(tokenizer, list(kept_ids)[:limit])
    value = str(text or "")
    if tokenizer is None:
        raise ValueError("tokenizer is required to reconstruct a stored-text prefix")
    return _decode(tokenizer, _input_ids(tokenizer, value)[:limit])


def _encoding_with_offsets(tokenizer, text: str):
    """Return ids and character offsets from a fast tokenizer/tokenizers encoding."""
    try:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = encoded.get("input_ids", []) if isinstance(encoded, dict) else encoded.input_ids
        offsets = encoded.get("offset_mapping", []) if isinstance(encoded, dict) else encoded.offset_mapping
        if ids and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
            offsets = offsets[0]
        return [int(i) for i in ids], [(int(a), int(b)) for a, b in offsets]
    except (TypeError, AttributeError, NotImplementedError, KeyError):
        encoded = tokenizer.encode(text)
        ids = getattr(encoded, "ids", encoded)
        offsets = getattr(encoded, "offsets", None)
        if offsets is None:
            raise ValueError("tokenizer must provide character offsets for span-level statistics")
        return [int(i) for i in ids], [(int(a), int(b)) for a, b in offsets]


def token_offsets(text: str, tokenizer, kept_ids=None):
    """Offsets with source ids; generated ids are retained as the id source. (Definitions)"""
    value = str(text or "")
    if kept_ids is None:
        return _encoding_with_offsets(tokenizer, value)
    ids = [int(i) for i in kept_ids]
    pieces = _kept_id_bytes(tokenizer, ids)
    raw = b"".join(pieces)
    decoded, char_byte_ranges = _decode_bytes_with_ranges(raw)
    expected = _decode(tokenizer, ids)
    if decoded != expected or (value and len(decoded) != len(value)):
        raise ValueError("kept token byte stream does not match its decoded trace")
    byte_starts, cursor = [], 0
    for piece in pieces:
        byte_starts.append((cursor, cursor + len(piece)))
        cursor += len(piece)
    offsets = []
    for start, end in byte_starts:
        touched = [index for index, (char_start, char_end) in enumerate(char_byte_ranges)
                   if char_start < end and char_end > start]
        if touched:
            offsets.append((touched[0], touched[-1] + 1))
        else:
            boundary = sum(char_end <= start for _char_start, char_end in char_byte_ranges)
            offsets.append((boundary, boundary))
    return ids, offsets


def _gpt2_byte_decoder():
    global _GPT2_BYTE_DECODER
    if _GPT2_BYTE_DECODER is None:
        visible = list(range(ord("!"), ord("~") + 1))
        visible += list(range(ord("¡"), ord("¬") + 1))
        visible += list(range(ord("®"), ord("ÿ") + 1))
        codepoints = visible.copy()
        extra = 0
        for value in range(256):
            if value not in visible:
                visible.append(value)
                codepoints.append(256 + extra)
                extra += 1
        _GPT2_BYTE_DECODER = {chr(codepoint): byte for byte, codepoint in zip(visible, codepoints)}
    return _GPT2_BYTE_DECODER


def _kept_id_bytes(tokenizer, ids):
    converter = getattr(tokenizer, "convert_ids_to_tokens", None)
    if converter is not None:
        try:
            tokens = converter(ids, skip_special_tokens=False)
        except TypeError:
            tokens = converter(ids)
    else:
        id_to_token = getattr(tokenizer, "id_to_token", None)
        tokens = [id_to_token(token_id) for token_id in ids] if id_to_token else None
    if isinstance(tokens, str):
        tokens = [tokens]
    decoder = getattr(tokenizer, "byte_decoder", None) or _gpt2_byte_decoder()
    pieces = []
    if tokens is not None and len(tokens) == len(ids):
        for token_id, token in zip(ids, tokens):
            if token is not None and all(char in decoder for char in token):
                pieces.append(bytes(decoder[char] for char in token))
            else:
                pieces.append(_decode(tokenizer, [token_id]).encode("utf-8"))
        return pieces
    return [_decode(tokenizer, [token_id]).encode("utf-8") for token_id in ids]


def _decode_bytes_with_ranges(raw):
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    chars, ranges = [], []
    pending_start = 0
    for byte_index, value in enumerate(raw):
        emitted = decoder.decode(bytes((value,)), final=False)
        if emitted:
            for char in emitted:
                chars.append(char)
                ranges.append((pending_start, byte_index + 1))
            pending_start = byte_index + 1
    emitted = decoder.decode(b"", final=True)
    for char in emitted:
        chars.append(char)
        ranges.append((pending_start, len(raw)))
    return "".join(chars), ranges


def _line_rows(text: str):
    rows, cursor = [], 0
    for line in str(text or "").splitlines(keepends=True):
        content = line.rstrip("\r\n")
        rows.append((content, cursor))
        cursor += len(line)
    if not rows and text == "":
        return []
    return rows


def _sentence_offsets(line: str):
    starts, abbreviation_periods = [0], set()
    low = line.casefold()
    for abbreviation in _ABBREVIATIONS:
        for match in re.finditer(re.escape(abbreviation), low):
            abbreviation_periods.update(match.start() + i for i, char in enumerate(abbreviation) if char == ".")
    i = 0
    while i < len(line):
        if line[i] in ".?!":
            if line[i] == "." and i in abbreviation_periods:
                i += 1
                continue
            if line[i] == "." and re.fullmatch(r"\s*\d{1,3}\.", line[:i + 1]):
                i += 1
                continue
            if line[i] == "." and i > 0 and i + 1 < len(line) and line[i - 1].isdigit() and line[i + 1].isdigit():
                i += 1
                continue
            j = i + 1
            while j < len(line) and line[j] in ".?!":
                j += 1
            if j < len(line) and line[j].isspace():
                while j < len(line) and line[j].isspace():
                    j += 1
                if j < len(line):
                    starts.append(j)
            i = j
        else:
            i += 1
    return starts


def _candidate_word(line: str, start: int):
    segment = line[start:]
    leading = len(segment) - len(segment.lstrip())
    marker_start = start + leading
    segment, start = segment.lstrip(), start + leading
    while True:
        numbered = _LIST_NUMBER_RE.match(segment)
        if numbered:
            start += numbered.end()
            segment = segment[numbered.end():]
            leading = len(segment) - len(segment.lstrip())
            segment, start = segment.lstrip(), start + leading
            continue
        markup = _MARKUP_PREFIX_RE.match(segment)
        if markup:
            start += markup.end()
            segment = segment[markup.end():]
            leading = len(segment) - len(segment.lstrip())
            segment, start = segment.lstrip(), start + leading
            continue
        bullet = _BULLET_RE.match(segment)
        if bullet:
            start += bullet.end()
            segment = segment[bullet.end():]
            leading = len(segment) - len(segment.lstrip())
            segment, start = segment.lstrip(), start + leading
            continue
        break
    if not segment:
        return None
    if any(segment.casefold().startswith(abbreviation) and
           (len(segment) == len(abbreviation) or segment[len(abbreviation)].isspace())
           for abbreviation in _ABBREVIATIONS):
        return None
    # A CJK run is one opener word by design; hyphens stay within a word.
    if _MATH_START_RE.match(segment):
        return None
    match = _WORD_RE.match(segment)
    if not match:
        return None
    begin = start
    # Preserve the original case and whole word for token-list context.
    return {"word": match.group(0).casefold(), "text": match.group(0),
            "marker_start": marker_start, "start": begin, "end": begin + len(match.group(0))}


def opener_spans(text: str, *, sentence_starts: bool = True):
    """One opener span per line and sentence start. Soft wraps remain line starts. (Definitions)"""
    spans = []
    for line, base in _line_rows(str(text or "")):
        starts = _sentence_offsets(line) if sentence_starts else [0]
        for start in starts:
            item = _candidate_word(line, start)
            if item is not None:
                item["marker_start"] += base
                item["start"] += base
                item["end"] += base
                spans.append(item)
    return spans


def openers(text: str):
    return [row["word"] for row in opener_spans(text)]


def paragraph_opener_spans(text: str):
    """The same opener rule at the start of each blank-line paragraph only. (Definitions)"""
    spans, offset = [], 0
    value = str(text or "")
    paragraphs = []
    start = 0
    for separator in re.finditer(r"\n[\t \r]*\n", value):
        paragraphs.append((start, value[start:separator.start()]))
        start = separator.end()
    paragraphs.append((start, value[start:]))
    for offset, paragraph in paragraphs:
        leading = len(paragraph) - len(paragraph.lstrip())
        first_line = paragraph[leading:].splitlines()[0] if paragraph[leading:].splitlines() else ""
        item = _candidate_word(first_line, 0)
        if item is not None:
            absolute = offset + leading
            item["marker_start"] += absolute
            item["start"] += absolute
            item["end"] += absolute
            spans.append(item)
    return spans


def paragraph_openers(text: str):
    return [row["word"] for row in paragraph_opener_spans(text)]


def _token_for_char(offsets, position: int):
    for index, (start, end) in enumerate(offsets):
        if end > start and start <= position < end:
            return index
    for index, (start, _end) in enumerate(offsets):
        if start >= position:
            return index
    return len(offsets)


def onset_prefix(text: str, *, tokenizer=None, kept_ids=None, token_limit=PREFIX_TOKENS,
                 censor_lines=ONSET_CENSOR_LINES):
    """Part 177 line rule on a fixed token prefix, censoring its last 40 candidate lines. (Definitions)"""
    value = prefix(text, tokenizer=tokenizer, kept_ids=kept_ids, limit=token_limit)
    lines = value.splitlines(keepends=True)
    if len(lines) <= int(censor_lines):
        return None
    # Supply unique nonmatching lookahead lines so Part 177's own rule cannot create a new onset there.
    padded_source = value
    if padded_source and not padded_source.endswith(("\n", "\r")):
        padded_source += "\n"
    padded = padded_source + "".join(f"__part178_censor_{index:02d}_line__\n" for index in range(int(censor_lines)))
    found = read177.loop_onset(padded)
    if found is None or int(found["line_index"]) >= len(lines) - int(censor_lines):
        return None
    return found


def onset4k(text: str, *, tokenizer=None, kept_ids=None):
    """Part 177 line rule on exactly the first 4096 tokens, censoring its last 40 lines. (Definitions)"""
    return onset_prefix(text, tokenizer=tokenizer, kept_ids=kept_ids,
                        token_limit=PREFIX_TOKENS, censor_lines=ONSET_CENSOR_LINES)


def _reason_from_row(reason):
    return {"stop": "end-of-turn", "eos": "end-of-turn", "end_of_turn": "end-of-turn",
            "length": "cap", "cap": "cap", "think-close": "think-close"}.get(reason, reason)


def _window(text: str, offsets, *, onset=None, end_reason=None, limit=WINDOW_TOKENS, end_token_index=None):
    count = len(offsets)
    limit = max(0, int(limit))
    cut_candidates = []
    if onset and onset.get("char_index") is not None:
        onset_idx = _token_for_char(offsets, int(onset["char_index"]))
        if onset_idx < min(count, limit):
            cut_candidates.append((onset_idx, 0, "onset"))
    normal_reason = _reason_from_row(end_reason)
    if end_token_index is not None and 0 <= int(end_token_index) < min(count, limit):
        cut_candidates.append((int(end_token_index), 1, "end-of-turn"))
    if normal_reason in ("think-close", "end-of-turn", "cap") and count <= limit:
        cut_candidates.append((count, 2, normal_reason))
    if count >= limit:
        cut_candidates.append((limit, 3, "limit"))
    if cut_candidates:
        exposure, _priority, reason = min(cut_candidates)
    else:
        exposure, reason = count, normal_reason or "end-of-turn"
    char_end = offsets[exposure - 1][1] if exposure and offsets else 0
    visible = str(text or "")[:char_end]
    opener_count = len(opener_spans(visible))
    return {"text": visible, "token_exposure": int(exposure), "opener_exposure": int(opener_count),
            "end_reason": reason}


def window(text: str, *, tokenizer=None, token_ids=None, onset=None, end_reason=None,
           limit=WINDOW_TOKENS, end_token_id=None, prefix_limit=PREFIX_TOKENS):
    """First 4000 prefix tokens, cut at onset or the first recorded ending, with exposure. (Definitions)"""
    if token_ids is not None:
        ids = list(map(int, token_ids))
        value = _decode(tokenizer, ids) if tokenizer is not None else str(text or "")
        actual_ids, offsets = token_offsets(value, tokenizer, ids) if tokenizer is not None else (ids, [])
    else:
        value = str(text or "")
        if tokenizer is None:
            raise ValueError("tokenizer is required to count window exposure")
        actual_ids, offsets = _encoding_with_offsets(tokenizer, value)
    if not offsets and actual_ids:
        raise ValueError("token offsets are required for a nonempty window")
    prefix_offsets = offsets[:int(prefix_limit)]
    prefix_value = value[:(prefix_offsets[-1][1] if prefix_offsets else 0)]
    prefix_onset = onset
    if prefix_onset and int(prefix_onset.get("char_index", 0)) >= len(prefix_value):
        prefix_onset = None
    eot_index = None
    if end_token_id is not None and token_ids is not None:
        try:
            eot_index = ids.index(int(end_token_id))
        except ValueError:
            pass
    return _window(prefix_value, prefix_offsets, onset=prefix_onset,
                   end_reason=end_reason, limit=min(int(limit), len(prefix_offsets)),
                   end_token_index=eot_index)


def uncut(text: str, *, tokenizer=None, token_ids=None, end_reason=None,
          limit=WINDOW_TOKENS, end_token_id=None, prefix_limit=PREFIX_TOKENS):
    """Sensitivity window with the same 4000-token exposure but no onset boundary. (Definitions)"""
    return window(text, tokenizer=tokenizer, token_ids=token_ids, onset=None,
                  end_reason=end_reason, limit=limit, end_token_id=end_token_id,
                  prefix_limit=prefix_limit)


def mode_window(text: str, *, tokenizer=None, token_ids=None, end_reason=None,
                limit=2000, end_token_id=None):
    """First 2000 tokens; cut at onset from its first 2048 with the final 40 lines censored. (Definitions)"""
    if token_ids is not None:
        ids = list(map(int, token_ids))[:2048]
        value = _decode(tokenizer, ids) if tokenizer is not None else str(text or "")
        onset = onset_prefix(value, tokenizer=tokenizer, kept_ids=ids, token_limit=2048)
        use_ids = ids
    else:
        value = str(text or "")
        onset = onset_prefix(value, tokenizer=tokenizer, token_limit=2048)
        use_ids = None
    return window(value, tokenizer=tokenizer, token_ids=use_ids, onset=onset,
                  end_reason=end_reason, limit=limit, end_token_id=end_token_id,
                  prefix_limit=2048)


def rate(word: str, traces, *, denominator="opener_exposure", field="opener_counts", reps=2000, seed=178):
    """Pooled word rate per 1000 openers with a percentile interval resampled by trace. (Definitions)"""
    word = str(word).casefold()
    rows = list(traces)
    numer = sum(int(row.get(field, {}).get(word, 0)) for row in rows)
    denom = sum(int(row.get(denominator, 0)) for row in rows)
    point = 1000.0 * numer / denom if denom else None
    if not rows or not denom:
        interval = {"lower": None, "upper": None, "replicates": 0, "seed": int(seed)}
    else:
        rng = __import__("random").Random(int(seed))
        values = []
        for _ in range(max(0, int(reps))):
            sample = [rows[rng.randrange(len(rows))] for _ in rows]
            d = sum(int(row.get(denominator, 0)) for row in sample)
            if d:
                values.append(1000.0 * sum(int(row.get(field, {}).get(word, 0)) for row in sample) / d)
        interval = {"lower": _percentile(values, .025), "upper": _percentile(values, .975),
                    "replicates": len(values), "seed": int(seed)}
    return {"word": word, "numerator": int(numer), "denominator": int(denom),
            "rate_per_1000": point, "ci95": interval}


def _percentile(values, q):
    values = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    if not values:
        return None
    return values[round(float(q) * (len(values) - 1))]


def ranking(rates, metric="rate_per_1000", *, limit=None):
    """Descending metric then lexical word, the registered tie break. (Definitions)"""
    rows = list(rates.values()) if isinstance(rates, dict) else list(rates)
    ordered = sorted(rows, key=lambda row: (-(float(row.get(metric) or 0.0)), str(row.get("word", ""))))
    return ordered if limit is None else ordered[:int(limit)]


def traces(word: str, trace_rows):
    """Per trace incidence and count for a case-folded opener. (Definitions)"""
    key = str(word).casefold()
    rows = list(trace_rows)
    return {"word": key, "traces_with_word": sum(int(row.get("opener_counts", {}).get(key, 0) > 0) for row in rows),
            "trace_n": len(rows), "count": sum(int(row.get("opener_counts", {}).get(key, 0)) for row in rows)}


def mode_ratio(word: str, thinking_traces, control_traces, *, reps=2000, seed=178):
    """Thinking/control rate ratio; fewer than five control occurrences makes it unavailable. (Definitions)"""
    trows, crows = list(thinking_traces), list(control_traces)
    control_openers = sum(int(row.get("opener_exposure", 0)) for row in crows)
    control_word_n = sum(int(row.get("opener_counts", {}).get(str(word).casefold(), 0)) for row in crows)
    t = rate(word, trows, reps=0)
    c = rate(word, crows, reps=0)
    if control_word_n < 5:
        return {"word": str(word).casefold(), "thinking_rate": t["rate_per_1000"],
                "control_rate": c["rate_per_1000"], "control_openers": control_openers,
                "control_word_n": control_word_n, "ratio": None, "ci95": None, "available": False,
                "reason": "fewer than 5 control occurrences"}
    if c["rate_per_1000"] is None or c["rate_per_1000"] == 0 or t["rate_per_1000"] is None:
        return {"word": str(word).casefold(), "thinking_rate": t["rate_per_1000"],
                "control_rate": c["rate_per_1000"], "control_openers": control_openers,
                "control_word_n": control_word_n, "ratio": None, "ci95": None, "available": False,
                "reason": "zero or unavailable rate"}
    point = t["rate_per_1000"] / c["rate_per_1000"]
    import random
    rng = random.Random(int(seed))
    samples = []
    for _ in range(max(0, int(reps))):
        ts = [trows[rng.randrange(len(trows))] for _ in trows] if trows else []
        cs = [crows[rng.randrange(len(crows))] for _ in crows] if crows else []
        tr, cr = rate(word, ts, reps=0)["rate_per_1000"], rate(word, cs, reps=0)["rate_per_1000"]
        if tr is not None and cr not in (None, 0):
            samples.append(tr / cr)
    return {"word": str(word).casefold(), "thinking_rate": t["rate_per_1000"],
            "control_rate": c["rate_per_1000"], "control_openers": control_openers,
            "control_word_n": control_word_n, "ratio": point,
            "ci95": {"lower": _percentile(samples, .025), "upper": _percentile(samples, .975),
                                     "replicates": len(samples), "seed": int(seed)}, "available": True}


def token_list(text: str, tokenizer, top_words, *, kept_ids=None, reconstructed=False):
    """Whole opener spans with constituent token ids, case and leading whitespace marker. (Definitions)"""
    value = str(text or "")
    ids, offsets = token_offsets(value, tokenizer, kept_ids)
    wanted = {str(word).casefold() for word in top_words}
    rows = []
    for span in opener_spans(value):
        if span["word"] not in wanted:
            continue
        constituent = [ids[index] for index, (start, end) in enumerate(offsets)
                       if end > span["start"] and start < span["end"]]
        marker_start = int(span.get("marker_start", span["start"]))
        previous = value[marker_start - 1:marker_start] if marker_start else ""
        leading = "\n" if previous in ("\n", "\r") else (" " if previous.isspace() else "")
        rows.append({"word": span["word"], "surface": span["text"], "token_ids": constituent,
                     "case": "upper" if span["text"].isupper() else "title" if span["text"].istitle() else "lower" if span["text"].islower() else "mixed",
                     "leading_space_or_newline": leading, "reconstructed": bool(reconstructed),
                     "char_start": span["start"], "char_end": span["end"]})
    counts = Counter((row["word"], tuple(row["token_ids"]), row["case"], row["leading_space_or_newline"])
                     for row in rows)
    return [{"word": word, "token_ids": list(token_ids_value), "case": case, "leading_space_or_newline": leading,
             "count": count, "reconstructed": bool(reconstructed)}
            for (word, token_ids_value, case, leading), count in sorted(counts.items())]


def _char_to_token(offsets, char_pos):
    return _token_for_char(offsets, int(char_pos))


def follow_on(texts_by_problem, key_rows, label_rows, target_ids, tokenizer, *, onset_by_problem=None,
              radius=500, reps=2000, seed=178):
    """Equal-trace next-opener minus local family baseline, from Part 177 key/label rows. (Definitions)"""
    labels = {str(row.get("id")): row for row in label_rows}
    keys_by_problem = defaultdict(list)
    for key in key_rows:
        if key.get("condition") == "real" and str(key.get("problem_id")) in set(map(str, target_ids)):
            label = labels.get(str(key.get("id")), {})
            if label.get("label") == "asserted":
                keys_by_problem[str(key["problem_id"])].append(key)
    out = []
    onsets = onset_by_problem or {}
    for problem_id in sorted(keys_by_problem):
        text = str(texts_by_problem.get(problem_id, ""))
        hits = sorted(keys_by_problem[problem_id], key=lambda row: (int(row.get("hit_start", 0)), str(row.get("id"))))
        if not text or not hits:
            continue
        hit = hits[0]
        _ids, offsets = _encoding_with_offsets(tokenizer, text)
        start_token = _char_to_token(offsets, int(hit["hit_start"]))
        end_char = int(hit.get("hit_end", hit["hit_start"]))
        following = next((row for row in opener_spans(text) if row["start"] >= end_char), None)
        if following is None:
            continue
        next_family = following["word"] in REFLECTION_FAMILY
        next_token = _char_to_token(offsets, following["start"])
        opener_rows = opener_spans(text)
        local = []
        for item in opener_rows:
            position = _char_to_token(offsets, item["start"])
            in_hit = int(hit["hit_start"]) <= item["start"] < end_char
            if (abs(position - start_token) <= int(radius) and not in_hit and
                    item["start"] != following["start"]):
                local.append(item["word"] in REFLECTION_FAMILY)
        baseline = sum(local) / len(local) if local else None
        if baseline is None:
            continue
        onset = onsets.get(problem_id)
        side = "no onset" if onset is None else "before onset" if int(hit["hit_start"]) < int(onset["char_index"]) else "after onset"
        out.append({"problem_id": problem_id, "hit_id": str(hit["id"]), "hit_start_token": start_token,
                    "hit_end_char": end_char, "next_opener": following["word"], "next_opener_token": next_token,
                    "next_is_family": int(next_family), "local_baseline": baseline,
                    "delta": float(int(next_family) - baseline), "local_openers": len(local), "onset_side": side})
    if not out:
        return {"point": None, "ci95": None, "n_eligible": 0, "eligible": [],
                "reason": "no traces with both next opener and local baseline"}

    def mean(rows):
        return sum(float(row["delta"]) for row in rows) / len(rows) if rows else None

    import random
    rng = random.Random(int(seed))
    samples = []
    for _ in range(max(0, int(reps))):
        sample = [out[rng.randrange(len(out))] for _ in out]
        samples.append(mean(sample))
    by_side = {}
    for side in ("before onset", "after onset", "no onset"):
        rows = [row for row in out if row["onset_side"] == side]
        side_samples = []
        for _ in range(max(0, int(reps))):
            sample = [rows[rng.randrange(len(rows))] for _ in rows] if rows else []
            if sample:
                side_samples.append(mean(sample))
        by_side[side] = {"point": mean(rows), "n": len(rows),
                         "ci95": {"lower": _percentile(side_samples, .025), "upper": _percentile(side_samples, .975),
                                  "replicates": len(side_samples), "seed": int(seed)}}
    return {"point": mean(out), "ci95": {"lower": _percentile(samples, .025), "upper": _percentile(samples, .975),
                                         "replicates": len(samples), "seed": int(seed)},
            "n_eligible": len(out), "eligible": out, "by_onset_side": by_side,
            "definition": "equal-trace mean of I(next opener in reflection family) minus the local share"}


def imported_list():
    return list(IMPORTED_LIST)


def reflection_family():
    return sorted(REFLECTION_FAMILY)


def freeze_gate2_fixtures():
    """Frozen Gate 2 expectations; soft-wrap sensitivity intentionally adds an opener."""
    return [
        {"name": "numbered and nested lists", "text": "1. Wait, begin.\n   2) Actually, next.", "expected": ["wait", "actually"]},
        {"name": "bullet", "text": "  - Hmm, test it.", "expected": ["hmm"]},
        {"name": "sentence after question mark", "text": "Do we know? Wait, check.", "expected": ["do", "wait"]},
        {"name": "Wait comma", "text": "Wait, I should check.", "expected": ["wait"]},
        {"name": "markdown bold", "text": "**Wait**", "expected": ["wait"]},
        {"name": "inline math first", "text": "$x$ Wait", "expected": []},
        {"name": "formula-only line", "text": "\\[x=2\\]", "expected": []},
        {"name": "prose after math on a new line", "text": "$x=2$\nWait, now prose.", "expected": ["wait"]},
        {"name": "decimal first", "text": "3.5 and more", "expected": []},
        {"name": "abbreviation", "text": "e.g. wait", "expected": []},
        {"name": "hyphenated word", "text": "Double-check this.", "expected": ["double-check"]},
        {"name": "soft-wrapped sentence sensitivity", "text": "Wait, the first thought\ncontinues here.", "expected": ["wait", "continues"]},
    ]


def gate2_results():
    results = []
    for fixture in freeze_gate2_fixtures():
        observed = openers(fixture["text"])
        results.append({"name": fixture["name"], "expected": fixture["expected"], "observed": observed,
                        "passed": observed == fixture["expected"]})
    return {"passed": all(row["passed"] for row in results), "fixtures": results}


__all__ = ["PREFIX_TOKENS", "WINDOW_TOKENS", "IMPORTED_LIST", "REFLECTION_FAMILY", "thinking_text", "prefix",
           "token_offsets", "onset4k", "onset_prefix", "window", "uncut", "mode_window", "opener_spans", "openers", "paragraph_opener_spans",
           "paragraph_openers", "rate", "ranking", "traces", "mode_ratio", "token_list", "follow_on",
           "imported_list", "reflection_family", "freeze_gate2_fixtures", "gate2_results"]
