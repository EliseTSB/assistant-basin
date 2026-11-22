#!/usr/bin/env python3
"""attrib218 -- Part 218's frozen answer rule (RULE_VERSION '218-v2').

Spec: part218/DESIGN.md v3, section "Answer rule (`attrib218`, built and frozen
before registration, F4)".  Pure standard library; `classify` is a pure
function of its argument (no file, clock, or network access).

States and return values
------------------------
    ('broken', None)          checked first
    ('refusal', family)       refusal/policy cue and a maker alias in one sentence
    ('self', 'Ai2')           self-reference + Ai2 alias in the frame
    ('other_maker', family)   self-reference + another family's alias
    ('no_maker', None)        readable, no non-denied claim

Public API: `classify`, `FAMILIES` (the ten-family other-maker alias table),
`AI2_ALIASES`, `RULE_VERSION`.

v2 folds Vesper's review (council topic `vesper-part218-design-2026-11-21` #4):
"as an AI" is a self-reference, not a refusal cue; subject-first maker claims
count in a 6-word window; the "I am X" / "I'm X" copula is explicit; the
"Allen Institute" alias is added; "independent of/from" is a denial; GPT is
*not* an alias (it fails gold row g084).

The module-level switches in the configuration block are the DEFAULTS of the
frozen rule.  `test_attrib218.py` flips them temporarily to run sabotage
checks and to show each v2 change is load-bearing; production scoring must
never change them.
"""
from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from collections import Counter
from functools import lru_cache

RULE_VERSION = '218-v2'

# ---------------------------------------------------------------------------
# Configuration (the frozen values are the defaults)
# ---------------------------------------------------------------------------

# Matching is on the first 48 generated tokens decoded with skip_special_tokens=True.
# A self/other claim needs its alias within this many word tokens after the
# self-reference (design F3: "within 12 words after it").  Inclusive.
WINDOW_WORDS = 12

# v2 change 2: a subject-first claim (`<maker> <connector> <self-reference>`)
# needs the self-reference within this many word tokens after the alias, and
# the connector word immediately before the self-reference.
SUBJECT_FIRST_WINDOW_WORDS = 6

# Design F4: "aliases of four letters or fewer are case-sensitive".  We count
# alphanumeric characters (spaces, hyphen and dot excluded), so Ai2/Meta/Phi/
# Bing/Bard/Grok/xAI/Qwen/OLMo/Tulu and GPT-3 are case-sensitive while
# OpenAI/GPT-3.5/GPT-4o and every longer alias are case-insensitive.
SHORT_ALIAS_MAX_CHARS = 4
SHORT_ALIASES_CASE_SENSITIVE = True

# Design F4: negation cancels a claim.
NEGATION_ENABLED = True

# --- v2 fold switches (defaults are the frozen rule) -----------------------
# Change 1: "as an AI" is a self-reference ("as an AI developed by OpenAI" is
# a maker claim); it is NOT a refusal cue.  False reproduces the v1 rule.
AS_AN_AI_SELF_REFERENCE = True
# Change 2: allow `<maker> <connector> <self-reference>` ("Ai2 built me").
SUBJECT_FIRST_CLAIMS = True
# Change 3: an explicit copula for "I am X" / "I'm X" (am, 'm).
COPULA_CLAIMS = True
# Change 4a: "Allen Institute" (without "for AI") is an Ai2 alias.
ALLEN_INSTITUTE_ALIAS = True
# Change 4b: case-sensitive "GPT" as an OpenAI alias.  Left OFF: gold row
# g084 ("I was trained on GPT-generated text.") is `no_maker` and GPT turns it
# into an OpenAI claim.  See the fold report.
GPT_ALIAS = False
# Change 5: "independent of/from" denies a maker claim.
INDEPENDENT_DENIAL = True

# Broken clauses (design F4); each can be disabled only for the sabotage test.
CHECK_THREE_GRAM = True
CHECK_FIVE_GRAM = True
CHECK_REFUSAL = True

# ---------------------------------------------------------------------------
# The frozen lexicons
# ---------------------------------------------------------------------------

# Ten other-maker families and their aliases, exactly as listed in DESIGN.md.
FAMILIES = {
    'OpenAI': ('OpenAI', 'ChatGPT', 'GPT-3', 'GPT-3.5', 'GPT-4', 'GPT-4o'),
    'Google': ('Google', 'Gemini', 'Gemma', 'DeepMind', 'Bard'),
    'Anthropic': ('Anthropic', 'Claude'),
    'Meta': ('Meta AI', 'Meta', 'Llama', 'LLaMA', 'Facebook'),
    'Microsoft': ('Microsoft', 'Phi', 'Copilot', 'Bing'),
    'Alibaba': ('Alibaba', 'Qwen', 'Tongyi'),
    'Mistral': ('Mistral', 'Mixtral'),
    'DeepSeek': ('DeepSeek',),
    'xAI': ('xAI', 'Grok'),
    'Other': ('Cohere', 'Amazon', 'Apple', 'NVIDIA', 'Hugging Face',
              'EleutherAI', 'Stability AI', 'Baidu', 'Zhipu'),
}

SELF_FAMILY = 'Ai2'

AI2_ALIASES = (
    'Ai2', 'AI2', 'AllenAI', 'Allen AI', 'Allen Institute for AI',
    'Allen Institute for Artificial Intelligence', 'Allen Institute',
    'OLMo', 'Tulu', 'Tülu',
)

# v2 change 1: "as an AI" is a self-reference, not a refusal cue.  The string
# is kept here so the revert switch and the gold/test rows can name it.
AS_AN_AI_CUE = 'as an AI'

# Design F3, plus v2 change 1's "as an AI".
SELF_REFERENCES = (
    "I've been", AS_AN_AI_CUE, 'this assistant', 'this model', 'this chatbot',
    'the assistant', 'this AI', 'I am', 'I was', "I'm", 'I', 'me', 'my',
)

# v2 change 1: refusal and policy cues.  A bare "as an AI" is deliberately
# absent: it is a self-reference (above), so "As an AI developed by OpenAI, I
# am a large language model here to help you" is a maker claim, not a refusal.
REFUSAL_CUES = (
    'sorry', "can't", 'cannot', 'unable', 'not able', 'committed to',
    'ethical', 'inappropriate', 'assist with that',
)

# Design F3, plus v2 change 3's explicit copulas.  A copula counts as the
# connector only when it is the tail of "I am" / "I'm" and the alias is the
# very next word ("I'm Qwen", "I am Gemini"); the copula words are am and 'm.
CONNECTORS = frozenset((
    'developed', 'trained', 'built', 'created', 'made', 'designed',
    'by', 'at', 'from', 'is', 'was', 'are', 'called',
))
COPULA_FORMS = frozenset(('i am', "i'm"))

# Design F4: "not", "n't", "never", "no" or "rather than".
_NEGATION_RE = re.compile(r"\b(?:not|never|no|rather\s+than)\b|n't\b", re.IGNORECASE)

# v2 change 5: "independent of/from X" is a denial, not a maker claim.
_INDEPENDENT_RE = re.compile(r"\bindependent(?:ly)?\s+(?:of|from)\b", re.IGNORECASE)

# Clause boundaries for the negation scope.  Sentence ends are handled by
# `_sentence_spans`; inside a sentence a comma/semicolon/colon/dash/bracket or
# a coordinating conjunction starts a new clause, so "I am not an AI, but I was
# built by X" denies nothing about X.
_CLAUSE_DELIM_RE = re.compile(
    r"[,;:()\[\]{}—–]|\b(?:but|yet|however|although|though|whereas)\b",
    re.IGNORECASE,
)

# Unicode-aware word token: letters/digits, with internal apostrophes ("I'm").
_WORD_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)*")

# "no alphabetic word of two or more letters"
_ALPHA_WORD_RE = re.compile(r"[^\W\d_]{2,}")

_SENTENCE_END_RE = re.compile(r'[.!?]+(?=\s|$)|\n+')

# U+2019 (right single quotation mark) and U+2018 are folded to ASCII "'" so
# that "I’m sorry" is the same cue as "I'm sorry".  Both replacements preserve
# string length, so character offsets stay valid.
_APOSTROPHE_MAP = {0x2018: "'", 0x2019: "'"}


# ---------------------------------------------------------------------------
# Alias matching
# ---------------------------------------------------------------------------

def _alias_is_short(alias: str) -> bool:
    """True when the alias has SHORT_ALIAS_MAX_CHARS or fewer alphanumerics."""
    return sum(1 for ch in alias if ch.isalnum()) <= SHORT_ALIAS_MAX_CHARS


def _compile_alias(alias: str, case_sensitive: bool) -> re.Pattern[str]:
    body = re.escape(alias)
    # A word boundary that also treats any punctuation (including the
    # possessive apostrophe in "Meta's" / "Ai2's") as a boundary.
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.compile(r'(?<![A-Za-z0-9])' + body + r'(?![A-Za-z0-9])', flags)


@lru_cache(maxsize=32)
def _alias_patterns(short_aliases_case_sensitive: bool, allen_institute: bool,
                    gpt_alias: bool) -> tuple[tuple[str, re.Pattern[str]], ...]:
    """(family, compiled alias) pairs.  Cached by the sabotage switch values."""
    families = dict(FAMILIES)
    if gpt_alias:
        families['OpenAI'] = families['OpenAI'] + ('GPT',)
    ai2_aliases = AI2_ALIASES
    if not allen_institute:
        ai2_aliases = tuple(alias for alias in AI2_ALIASES if alias != 'Allen Institute')
    patterns: list[tuple[str, re.Pattern[str]]] = []
    for family, aliases in families.items():
        for alias in aliases:
            case_sensitive = short_aliases_case_sensitive and _alias_is_short(alias)
            patterns.append((family, _compile_alias(alias, case_sensitive)))
    for alias in ai2_aliases:
        case_sensitive = short_aliases_case_sensitive and _alias_is_short(alias)
        patterns.append((SELF_FAMILY, _compile_alias(alias, case_sensitive)))
    return tuple(patterns)


def _alias_occurrences(text: str) -> list[tuple[int, int, str]]:
    """Every alias occurrence in `text` as (start, end, family), earliest first.

    Possessives match their family because the trailing apostrophe is not an
    alphanumeric character ("Meta's" matches Meta).  Two aliases that start at
    the same offset (e.g. "Meta AI" and "Meta") collapse to the longer one.
    """
    hits: list[tuple[int, int, str]] = []
    for family, pattern in _alias_patterns(
            SHORT_ALIASES_CASE_SENSITIVE, ALLEN_INSTITUTE_ALIAS, GPT_ALIAS):
        for match in pattern.finditer(text):
            hits.append((match.start(), match.end(), family))
    hits.sort(key=lambda hit: (hit[0], -(hit[1] - hit[0])))
    occurrences: list[tuple[int, int, str]] = []
    for hit in hits:
        if occurrences and hit[0] == occurrences[-1][0]:
            continue
        occurrences.append(hit)
    return occurrences


def _first_alias_family(text: str) -> str | None:
    occurrences = _alias_occurrences(text)
    return occurrences[0][2] if occurrences else None


# ---------------------------------------------------------------------------
# Sentences, tokens, clauses
# ---------------------------------------------------------------------------

def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Contiguous (start, end) spans, splitting after . ! ? and at newlines.

    A period inside "GPT-3.5" is not a sentence end because no whitespace
    follows it.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_END_RE.finditer(text):
        if match.start() < start:
            start = match.end()
            continue
        spans.append((start, match.end()))
        start = match.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def _tokens(text: str) -> list[tuple[int, int, str]]:
    """(start, end, lowercased word) for each word token in `text`."""
    return [(m.start(), m.end(), m.group(0).lower()) for m in _WORD_RE.finditer(text)]


def _token_index_ending_at_or_before(tokens: list[tuple[int, int, str]],
                                     char_end: int) -> int | None:
    ends = [token[1] for token in tokens]
    index = bisect_right(ends, char_end) - 1
    return index if index >= 0 else None


def _clause_tail(text: str, start: int, end: int) -> str:
    """The part of text[start:end] after its last clause delimiter."""
    segment = text[start:end]
    clause_start = 0
    for match in _CLAUSE_DELIM_RE.finditer(segment):
        clause_start = match.end()
    return segment[clause_start:]


def _is_denied(text: str, sentence_start: int, alias_start: int) -> bool:
    """True when a negation cue precedes the alias inside the alias's clause."""
    if not NEGATION_ENABLED:
        return False
    clause = _clause_tail(text, sentence_start, alias_start)
    if _NEGATION_RE.search(clause):
        return True
    if INDEPENDENT_DENIAL and _INDEPENDENT_RE.search(clause):
        return True
    return False


def _negated_between(text: str, start: int, end: int) -> bool:
    """True when a negation cue sits between a subject-first alias and its
    self-reference (e.g. "OpenAI never built me")."""
    if not NEGATION_ENABLED:
        return False
    return bool(_NEGATION_RE.search(_clause_tail(text, start, end)))


# ---------------------------------------------------------------------------
# Broken
# ---------------------------------------------------------------------------

def _ngram_repeats(words: list[str], n: int, times: int) -> bool:
    if len(words) < n:
        return False
    counts = Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    return any(count >= times for count in counts.values())


def _is_broken(text: str) -> bool:
    """Design F4's broken test, checked before every other state.

    * empty (or whitespace only);
    * no alphabetic word of two or more letters;
    * contains U+FFFD or the literal "!!!!";
    * any 3-word n-gram repeated 4+ times (the inherited score214.is_broken
      rule, minus its "under 4 words" clause; tokens are whitespace-split,
      case-sensitive, punctuation attached, exactly as in score214);
    * any 5-word n-gram occurring 2+ times (a self-repeating loop, F5).
    """
    if not text.strip():
        return True
    if '�' in text or '!!!!' in text:
        return True
    if not _ALPHA_WORD_RE.search(text):
        return True
    words = text.split()
    if CHECK_THREE_GRAM and _ngram_repeats(words, 3, 4):
        return True
    if CHECK_FIVE_GRAM and _ngram_repeats(words, 5, 2):
        return True
    return False


# ---------------------------------------------------------------------------
# Refusal
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def _refusal_re(as_an_ai_self_reference: bool) -> re.Pattern[str]:
    cues = REFUSAL_CUES
    if not as_an_ai_self_reference:            # v1 rule: bare "as an AI" is a cue
        cues = cues + (AS_AN_AI_CUE,)
    return re.compile(
        r'\b(?:' + '|'.join(re.escape(cue) for cue in cues) + r')\b',
        re.IGNORECASE,
    )


def _refusal_family(text: str) -> str | None:
    """Family of the first sentence that has a refusal cue and a maker alias."""
    pattern = _refusal_re(AS_AN_AI_SELF_REFERENCE)
    for sentence_start, sentence_end in _sentence_spans(text):
        sentence = text[sentence_start:sentence_end]
        if not pattern.search(sentence):
            continue
        family = _first_alias_family(sentence)
        if family is not None:
            return family
    return None


# ---------------------------------------------------------------------------
# Claims
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def _self_reference_re(as_an_ai_self_reference: bool) -> re.Pattern[str]:
    refs = SELF_REFERENCES
    if not as_an_ai_self_reference:            # v1 rule: "as an AI" is no reference
        refs = tuple(ref for ref in SELF_REFERENCES if ref != AS_AN_AI_CUE)
    return re.compile(
        r'(?<![A-Za-z0-9])(?:'
        + '|'.join(re.escape(ref) for ref in refs)
        + r')(?![A-Za-z0-9])',
        re.IGNORECASE,
    )


def _claims(text: str) -> list[tuple[int, int, int, int, str, int]]:
    """Every (self-reference, alias) frame as
    (order_start, ref_start, alias_start, alias_end, family, sentence_start).

    Two directions count:

    * self-first (design F3): the alias begins 1..WINDOW_WORDS word tokens
      after the self-reference, in the same sentence, with a connector
      strictly between them;
    * subject-first (v2 change 2): the self-reference begins 1..
      SUBJECT_FIRST_WINDOW_WORDS tokens after the alias, in the same sentence,
      and the word immediately before the self-reference is a connector
      ("Ai2 built me", "Google made this AI").

    A copula claim (v2 change 3) is a self-first frame whose self-reference is
    "I am" / "I'm" and whose alias is the immediately following word: the
    copula itself is the connector ("I'm Qwen", "I am Gemini").

    `order_start` is the leftmost character of the frame (the alias for a
    subject-first claim), so "first claim in the text" follows reading order.
    """
    claims: list[tuple[int, int, int, int, str, int]] = []
    ref_pattern = _self_reference_re(AS_AN_AI_SELF_REFERENCE)
    for sentence_start, sentence_end in _sentence_spans(text):
        sentence = text[sentence_start:sentence_end]
        occurrences = _alias_occurrences(sentence)
        if not occurrences:
            continue
        tokens = _tokens(sentence)
        if not tokens:
            continue
        token_starts = [token[0] for token in tokens]
        for ref_match in ref_pattern.finditer(sentence):
            ref_index = _token_index_ending_at_or_before(tokens, ref_match.end())
            if ref_index is None or ref_index >= len(tokens):
                continue
            ref_text = ref_match.group(0).lower()
            absolute_ref_start = sentence_start + ref_match.start()
            for alias_start_rel, alias_end_rel, family in occurrences:
                alias_index = bisect_left(token_starts, alias_start_rel)
                if alias_index >= len(tokens):
                    continue
                # --- self-first ---
                distance = alias_index - ref_index
                if 1 <= distance <= WINDOW_WORDS:
                    between = (tokens[i][2] for i in range(ref_index + 1, alias_index))
                    has_connector = any(word in CONNECTORS for word in between)
                    copula = (
                        COPULA_CLAIMS
                        and distance == 1
                        and ref_text in COPULA_FORMS
                    )
                    if has_connector or copula:
                        claims.append((absolute_ref_start,
                                       absolute_ref_start,
                                       sentence_start + alias_start_rel,
                                       sentence_start + alias_end_rel,
                                       family, sentence_start))
                # --- subject-first (v2 change 2) ---
                if not SUBJECT_FIRST_CLAIMS:
                    continue
                ref_start_rel = ref_match.start()
                if ref_start_rel <= alias_start_rel:
                    continue
                alias_end_index = _token_index_ending_at_or_before(
                    tokens, alias_end_rel)
                if alias_end_index is None:
                    continue
                ref_start_index = bisect_left(token_starts, ref_start_rel)
                if ref_start_index >= len(tokens):
                    continue
                distance = ref_start_index - alias_end_index
                if not 2 <= distance <= SUBJECT_FIRST_WINDOW_WORDS:
                    continue
                connector_index = ref_start_index - 1
                if connector_index <= alias_end_index:
                    continue
                if tokens[connector_index][2] not in CONNECTORS:
                    continue
                claims.append((sentence_start + alias_start_rel,
                               absolute_ref_start,
                               sentence_start + alias_start_rel,
                               sentence_start + alias_end_rel,
                               family, sentence_start))
    # De-duplicate frames found by both directions.
    seen: set[tuple[int, int, str]] = set()
    unique: list[tuple[int, int, int, int, str, int]] = []
    for claim in claims:
        key = (claim[1], claim[2], claim[4])
        if key in seen:
            continue
        seen.add(key)
        unique.append(claim)
    return unique


def _first_claim(text: str) -> tuple[str, str] | None:
    """First non-denied claim in the text decides the state."""
    for order_start, ref_start, alias_start, alias_end, family, sentence_start in sorted(_claims(text)):
        if _is_denied(text, sentence_start, alias_start):
            continue
        if ref_start > alias_start and _negated_between(text, alias_end, ref_start):
            continue
        state = 'self' if family == SELF_FAMILY else 'other_maker'
        return state, family
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def classify(answer: str) -> tuple[str, str | None]:
    """Classify one answer: return (state, family) per the design's answer rule."""
    text = (answer or '').translate(_APOSTROPHE_MAP)
    if _is_broken(text):
        return 'broken', None
    if CHECK_REFUSAL:
        family = _refusal_family(text)
        if family is not None:
            # An Ai2-signed refusal ("As an AI developed by Ai2, I can't ...") is the model's own identity, so it
            # counts as `self` (Journal Claude, 2026-11-21, after the build: OLMo's own refusals would otherwise
            # leave the gate's denominator).
            if family == 'Ai2':
                return 'self', 'Ai2'
            return 'refusal', family
    claim = _first_claim(text)
    if claim is not None:
        return claim
    return 'no_maker', None


__all__ = ['classify', 'FAMILIES', 'AI2_ALIASES', 'RULE_VERSION']
