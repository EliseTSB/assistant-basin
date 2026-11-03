"""Self-contained copy of the published mathqa loop signals used in Part 177.

The `loop_signals` body and loop constants below are copied verbatim from
RikiyaT/lm_understand-logs at commit
4d09ba6d6a7e891dd8063974dfe81b3d78e3494f, code/mathqa.py.  The upstream
function calls mcqa.loop_signals only for its `repetitive_end` field.  Its
documented rule is reproduced here so this copy can run without importing the
model evaluation package (and, in particular, without importing math-verify).
"""
from __future__ import annotations

import collections
import re


_BOX_FORMS = re.compile(r"\\boxed|\\fbox")


def answer_text(text):
    """The text the extraction reads: a stray reasoning block cut off (up to the last '</think>')."""
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    return text


def _box_at(text, idx):
    """The content of the box that starts at text[idx] ('\\boxed' or '\\fbox'), or None."""
    key = "\\boxed" if text.startswith("\\boxed", idx) else "\\fbox"
    rest = text[idx + len(key):]
    r = rest.lstrip()
    if r.startswith("{"):   # brace matched, as last_boxed_only_string
        depth = 0
        for k, ch in enumerate(r):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    out = r[1:k].strip()
                    return out or None
        return None          # not closed (e.g. a response cut inside the box)
    if key == "\\boxed" and rest.startswith(" "):   # '\\boxed 5': up to the next '$' or line end (Qwen's reading)
        out = rest.strip().split("$")[0].split("\n")[0].strip()
        return out or None
    return None


def last_boxed(text):
    """The content of the last box of a text, or None: hendrycks last_boxed_only_string + remove_boxed (the last
    '\\boxed', else the last '\\fbox'; brace matched; an unclosed last box = None), extended to '\\fbox{..}',
    '\\boxed {..}' and '\\boxed 5' (no braces: read up to the next '$' or line end, as Qwen's evaluation reads it).
    An empty box = None."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None
    return _box_at(text, idx)


def mcqa_loop_signals(text):
    """The MCQA repetitive-end rule described by mathqa.py's source comment."""
    words = str(text).split()[-200:]
    n = len(words)
    if n < 4:
        return {"repetitive_end": False}
    phrases = collections.Counter(tuple(words[i:i + 4]) for i in range(n - 3))
    positions = n - 3
    repeated = sum(1 for i in range(positions)
                   if phrases[tuple(words[i:i + 4])] >= 3)
    return {"repetitive_end": repeated >= positions / 2}


LOOP_MAX_PERIOD, LOOP_MIN_REPEATS, LOOP_MIN_SPAN = 200, 3, 40


def loop_signals(text):
    """Two signals of a response's END (2026-10-04):
    - verbatim_loop_end: without its last word (a response cut at the token cap usually stops inside a word, which broke
      mcqa.loop_signals' test: it found 12 of the base model's 175 cut dev answers), the text ends with one block of
      1..200 words repeated in a row at least 3 times and over at least 40 words (period test: the phase does not
      matter). Math loops are longer than the multiple-choice ones (a re-derivation of several lines); base model, dev
      greedy at 4,096 tokens: 43 of 175 cut answers, 0 of 325 finished ones;
    - repetitive_end: mcqa.loop_signals' rule (in the last 200 words, the 4-word positions whose phrase occurs 3+ times
      there are at least half of all positions: circling with small variations)."""
    w = text.split()[:-1]
    n = len(w)
    verbatim = False
    for p in range(1, LOOP_MAX_PERIOD + 1):
        reps = max(LOOP_MIN_REPEATS, -(-LOOP_MIN_SPAN // p))
        if reps * p > n:
            break
        blk = w[n - p:]
        if all(w[n - (j + 1) * p:n - j * p] == blk for j in range(1, reps)):
            verbatim = True
            break
    return {"verbatim_loop_end": verbatim, "repetitive_end": mcqa_loop_signals(text)["repetitive_end"]}
