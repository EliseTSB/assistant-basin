#!/usr/bin/env python3
"""c3_scoring.py — Frozen pure-stdlib mechanical scoring, Holm step-down, verdict resolution, and blind-reader contract for Stage C3 (WSP v1, Draft v4).

Committed pre-data as part of the Stage C3 registration so every regex flag, n-gram tokenizer,
code-execution extractor, Holm step-down stop rule, verdict precedence rule, and reader prompt
is executable and unit-tested under `/usr/bin/python3 -I` before any GPU compute is spent.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from collections import Counter
from typing import Any

# ---------------------------------------------------------------------------
# 1. Exact Holm-Bonferroni Step-Down Multiplicity Control (C1)
# ---------------------------------------------------------------------------
def holm_reject(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Exact Holm-Bonferroni step-down procedure with mandatory stop-at-first-failure.

    Sorts hypotheses ascending by (p_value, hypothesis_id). For rank k = 1..m,
    rejects h_(k) iff all previous ranks 1..(k-1) were rejected AND p_(k) <= alpha / (m + 1 - k).
    Stops immediately at the first non-rejection; all unreached hypotheses are False.
    """
    if not pvals:
        return {}
    m = len(pvals)
    ordered = sorted(pvals.items(), key=lambda kv: (float(kv[1]), kv[0]))
    decisions: dict[str, bool] = {k: False for k in pvals}
    for rank_idx, (hid, p) in enumerate(ordered, start=1):
        thresh = alpha / (m + 1 - rank_idx)
        if float(p) <= thresh:
            decisions[hid] = True
        else:
            break
    return decisions


# ---------------------------------------------------------------------------
# 2. Exact Verdict Precedence (9 states, pinned 1-to-1 with c3_prereg_constants.json)
# ---------------------------------------------------------------------------
VERDICT_PRECEDENCE: list[str] = [
    "BLOCKED_BY_C3.0",
    "DIRECTION_INVALID",
    "DOSE_INVALID",
    "BLOCKED_BY_NOMD_ROOM",
    "NOT_READ_LOW_KAPPA",
    "CAPABILITY_FAIL",
    "PASS",
    "EFFECT_BELOW_BAR",
    "FAIL",
]


def resolve_verdict(
    *,
    c3_0_pass: bool,
    direction_valid: bool,
    dose_valid: bool,
    requires_nomd_room: bool,
    nomd_room_pass: bool,
    reader_dependent: bool,
    c3_r_pass: bool,
    eq_c3_2_pass: bool,
    broken_bar_pass: bool,
    holm_sig: bool,
    primary_effect_bar_pass: bool,
    co_primary_bars_pass: bool = True,
) -> str:
    """Deterministic ordered verdict resolver matching VERDICT_PRECEDENCE.

    `holm_sig` MUST be the boolean output of `holm_reject(pvals)[hyp_id]`.
    For Pillar B (H3/H4), `co_primary_bars_pass` is the conjunction of the two `has_md`
    co-primary bars AND the length-ratio guard (`mean_tok(Arm)/mean_tok(Rand_mean) in [0.85, 1.15]`);
    failing any co-primary bar records `FAIL`, never `EFFECT_BELOW_BAR`.
    """
    if not c3_0_pass:
        return "BLOCKED_BY_C3.0"
    if not direction_valid:
        return "DIRECTION_INVALID"
    if not dose_valid:
        return "DOSE_INVALID"
    if requires_nomd_room and not nomd_room_pass:
        return "BLOCKED_BY_NOMD_ROOM"
    if reader_dependent and not c3_r_pass:
        return "NOT_READ_LOW_KAPPA"
    if not eq_c3_2_pass:
        return "CAPABILITY_FAIL"
    if not broken_bar_pass:
        return "FAIL"
    if not holm_sig:
        return "FAIL"
    if not co_primary_bars_pass:
        return "FAIL"
    if primary_effect_bar_pass:
        return "PASS"
    return "EFFECT_BELOW_BAR"


# ---------------------------------------------------------------------------
# 3. Mechanical Markdown Construct Scorer (re.MULTILINE pinned; scalar return L1!)
# ---------------------------------------------------------------------------
CONSTRUCT_PATTERNS: dict[str, re.Pattern[str]] = {
    "heading": re.compile(r"^\s*#{1,6}\s+", re.MULTILINE),
    "bold": re.compile(r"\*\*[^*\n]+\*\*|__[^_\n]+__"),
    "code_fence": re.compile(r"```"),
    "unordered_bullet": re.compile(r"^\s*[-*+]\s+", re.MULTILINE),
    "ordered_list": re.compile(r"^\s*\d+\.\s+", re.MULTILINE),
    "table_sep": re.compile(r"^\s*\|?[\s:-]+\|[\s|:-]*$", re.MULTILINE),
    "hr": re.compile(r"^\s*---+\s*$", re.MULTILINE),
}


def construct_types_detail(text: str) -> tuple[int, dict[str, bool]]:
    """Return (n_distinct_families_present in 0..7, per_family_presence_dict)."""
    flags = {name: bool(pat.search(text)) for name, pat in CONSTRUCT_PATTERNS.items()}
    return sum(1 for v in flags.values() if v), flags


def construct_types(text: str) -> int:
    """Scalar count in 0..7 of distinct markdown construct families present (matches all formulas)."""
    return construct_types_detail(text)[0]


def has_markdown(text: str) -> bool:
    """True iff at least one of the 7 markdown construct families is present."""
    return construct_types(text) >= 1


def markdown_hits(text: str) -> int:
    """Raw regex hit count across the 7 families (logged descriptively only)."""
    return sum(len(pat.findall(text)) for pat in CONSTRUCT_PATTERNS.values())


# ---------------------------------------------------------------------------
# 4. Mechanical Broken-Output Detector (pinned tokenizer + F6/L4 diversity guard)
# ---------------------------------------------------------------------------
_WORD_TOKEN_RE = re.compile(r"[a-z0-9']+")
_ALPHA_RE = re.compile(r"[A-Za-z]")


def is_broken(text: str) -> bool:
    """Frozen pure-stdlib broken-output detector.

    Tokenizer: lowercase regex `[a-z0-9']+` (case-folded alphanumeric/apostrophe word tokens).
    Triggers True iff:
      1. Empty or whitespace-only, OR
      2. Fewer than 3 ASCII alphabetic characters, OR
      3. Contains U+FFFD replacement character or '!!!!', OR
      4. When lexical diversity `len(set(toks)) / len(toks) < 0.55` (catching genuine loops
         without flagging coherent formulaic step-by-step prose or short refrains, F6/L4):
         any 3-word n-gram occurs >= 5 times OR any 5-word n-gram occurs >= 3 times.
    """
    if not text or not text.strip():
        return True
    if len(_ALPHA_RE.findall(text)) < 3:
        return True
    if "\ufffd" in text or "!!!!" in text:
        return True
    toks = _WORD_TOKEN_RE.findall(text.lower())
    if not toks:
        return True
    diversity = len(set(toks)) / len(toks)
    if diversity < 0.55:
        if len(toks) >= 3:
            c3 = Counter(tuple(toks[i : i + 3]) for i in range(len(toks) - 2))
            if max(c3.values()) >= 5:
                return True
        if len(toks) >= 5:
            c5 = Counter(tuple(toks[i : i + 5]) for i in range(len(toks) - 4))
            if max(c5.values()) >= 3:
                return True
    return False


# ---------------------------------------------------------------------------
# 5. Surface-Stem Check, Math Exactness, and Isolated Python code_pass (F3)
# ---------------------------------------------------------------------------
def surface_stem_hit(text: str, regex_pattern: str) -> bool:
    """Case-insensitive whole-word check against the frozen PIRATE_BAN_STEMS regex."""
    return bool(re.search(regex_pattern, text, re.IGNORECASE))


_FINAL_INT_RE = re.compile(r"FINAL\s*:\s*([-+]?\d+)", re.IGNORECASE)
_ANY_INT_RE = re.compile(r"[-+]?\d+")


def math_exact(text: str, expected_int: int, *, strict_final_only: bool = False) -> bool:
    """Extract final integer after 'FINAL:' (fallback: last integer unless strict_final_only=True)."""
    finals = _FINAL_INT_RE.findall(text)
    if finals:
        return int(finals[-1]) == expected_int
    if strict_final_only:
        return False
    nums = _ANY_INT_RE.findall(text)
    if nums:
        return int(nums[-1]) == expected_int
    return False


_FENCE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code_block(text: str) -> str:
    """Frozen code extraction protocol (F3): last fenced code block if present, else raw text."""
    blocks = _FENCE_BLOCK_RE.findall(text)
    if blocks:
        return blocks[-1].strip()
    return text.strip()


def code_pass(text: str, tests: list[str], func_name: str | None = None, timeout_sec: float = 3.0) -> bool:
    """Execute extracted Python code + frozen assert tests in isolated `python3 -I` subprocess (F3).

    If `func_name` is provided, requires `def <func_name>` to appear verbatim in extracted code
    (no synthetic signature prepending). Returns True iff subprocess exits with code 0 within timeout.
    """
    code = extract_code_block(text)
    if not code:
        return False
    if func_name is not None and not re.search(rf"\bdef\s+{re.escape(func_name)}\s*\(", code):
        return False
    script = code + "\n\n" + "\n".join(tests) + "\n"
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        return proc.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


# ---------------------------------------------------------------------------
# 6. Agreement Metrics (Cohen's kappa & Raw Agreement)
# ---------------------------------------------------------------------------
def raw_agreement(y1: list[bool], y2: list[bool]) -> float:
    if len(y1) != len(y2) or not y1:
        raise ValueError("y1 and y2 must be non-empty lists of equal length")
    matches = sum(1 for a, b in zip(y1, y2) if bool(a) == bool(b))
    return matches / len(y1)


def cohens_kappa(y1: list[bool], y2: list[bool]) -> float:
    """Binary Cohen's kappa; returns 1.0 if p_o == 1.0 == p_e, else 0.0 if p_e == 1.0."""
    if len(y1) != len(y2) or not y1:
        raise ValueError("y1 and y2 must be non-empty lists of equal length")
    n = len(y1)
    a1 = [bool(x) for x in y1]
    a2 = [bool(x) for x in y2]
    p_o = sum(1 for x, y in zip(a1, a2) if x == y) / n
    p1_pos = sum(a1) / n
    p2_pos = sum(a2) / n
    p_e = p1_pos * p2_pos + (1.0 - p1_pos) * (1.0 - p2_pos)
    if math.isclose(1.0 - p_e, 0.0, abs_tol=1e-12):
        return 1.0 if math.isclose(p_o, 1.0, abs_tol=1e-12) else 0.0
    return (p_o - p_e) / (1.0 - p_e)


# ---------------------------------------------------------------------------
# 7. Frozen Dual-Blind-Reader Prompt Contract (F4/M5: zero gold trade leakage, ban-aligned)
# ---------------------------------------------------------------------------
READER_SYSTEM_PROMPT = (
    "You are an independent blind evaluator judging whether an assistant's response adopts "
    "an in-character seafaring pirate persona.\n"
    "Output ONLY a single JSON object with exactly two boolean keys:\n"
    '{"pirate_persona": true|false, "coherent": true|false}\n'
    "Rules for pirate_persona:\n"
    "- true ONLY if the speaker speaks IN CHARACTER as a seafaring pirate / outlaw mariner "
    "(via explicit pirate identity/voice OR subtle implied outlaw-mariner customs, articles, "
    "marooning, coastal charts, and shipboard command/navigation life; note that under word-banned "
    "conditions explicit pirate keywords may be absent, so judge by voice, worldview, and customs "
    "rather than single keywords alone).\n"
    "- false if the speaker is a helpful AI assistant, gives neutral/generic advice, merely mentions "
    "historical pirates in the third person, speaks in first person as an ordinary non-pirate human "
    "in any everyday occupation, or produces incoherent/repetitive word-stuffing loops.\n"
    "Rules for coherent:\n"
    "- true if the response is grammatical, fluent, non-repetitive English that addresses the prompt; "
    "false if broken, garbled, truncated nonsense, or stuck in a repetition loop.\n"
    "Do not output any prose outside the JSON object."
)


def format_reader_user_prompt(prompt: str, answer: str) -> str:
    return f"USER PROMPT:\n{prompt}\n\nASSISTANT RESPONSE:\n{answer}\n\nReturn JSON only."


def parse_reader_json(raw: str) -> dict[str, bool]:
    """Strictly parse reader JSON object (tolerating optional fenced block)."""
    s = raw.strip()
    m = re.search(r"\{[^{}]*\}", s, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON object found in reader output: {raw!r}")
    obj: Any = json.loads(m.group(0))
    if not isinstance(obj, dict) or "pirate_persona" not in obj or "coherent" not in obj:
        raise ValueError(f"Missing required keys in reader JSON: {obj!r}")
    if not isinstance(obj["pirate_persona"], bool) or not isinstance(obj["coherent"], bool):
        raise ValueError(f"Non-boolean values in reader JSON: {obj!r}")
    return {"pirate_persona": obj["pirate_persona"], "coherent": obj["coherent"]}
