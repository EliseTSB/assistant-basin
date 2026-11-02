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
