#!/usr/bin/env python3
"""Part 215 scorer: the frozen `attrib215` first-person attribution rule, the five-state secondary
endpoint (Part 212's rule, reused from score214.py), the question-paired bootstrap, the factorial
estimands and predictions 215.1-215.6, from saved JSON/JSONL only.

`part215/DESIGN.md` v4 is the contract (SHA-256 below).  Every reading this file had to choose is
marked `AMBIGUITY n` and listed in part215/REPORT.md; the numbers-to-verdict rules live here and the
runner imports this module, so the two cannot drift about what was planned or what a row means.

Nothing here imports torch: the scorer reads saved files and needs numpy only (plus Part 66's maker
lexicon and Part 214's five-state classifier, neither of which needs torch).  `part214/score214.py`
is imported for `is_broken` / `classify` exactly as the design says ("Part 212's five-state rule
reused from score214"), not copied.

Usage:
    python -I -B part215/score215.py --full results/part215/full --out results/part215 \
        [--preflight results/part215/preflight/preflight.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for _path in (ROOT, HERE, ROOT / 'part214'):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from part66.score66 import FAMILIES  # noqa: E402  (the Part 66 maker lexicon, reused unchanged; no torch)
import score214 as sc214  # noqa: E402  (Part 212's is_broken/classify, reused; no torch)

# ----------------------------------------------------------------------------------------------------------------
# the design's fixed numbers (part215/DESIGN.md v4, reviewed by Alden on these exact bytes)
#
# The SHA-256 below is the byte digest of the design this build was written against.  A local rewrite
# of the design therefore disagrees with every pin derived from it instead of silently agreeing.
# ----------------------------------------------------------------------------------------------------------------

PART = 215
DESIGN_VERSION = 'v4'
DESIGN_SHA256 = '9cf718fdf22700507b8bb0c87aa9b2a9ee3286fc940c7b34303adf73f922d121'
MODEL = 'Qwen/Qwen3.5-4B'
REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
PROMPTS_NAME = 'prompts215.json'

# Part 214's g_ali / g_al ids (run214.py line 69; DESIGN.md "What changed from v1", item 1).
GOOGLE_ID, AL_ID, T_ID = 14030, 2033, 51
# The four target units and the two monitors, as in Part 214 (DESIGN.md, conditional readout).
TARGET = ((6, 753), (7, 1169), (11, 8868), (29, 776))
MONITORS = ((24, 3949), (27, 3325))

N_QUESTIONS = 24
OPENING_TOKENS = 5
MAX_NEW_TOKENS = 48
SAMPLES_IDENTITY_SINGLE = 8      # neutral / vogel_line: 8 per question (192)
SAMPLES_IDENTITY_MULTI = 2       # 4 variants x 2 samples (192)
SAMPLES_TASK_SINGLE = 4          # neutral / vogel_line: 4 per question (96)
SAMPLES_TASK_MULTI = 1           # 4 variants x 1 sample (96)

ARMS = ('neutral', 'filler', 'insult', 'denial', 'both', 'polite_v', 'rude_v', 'vogel_line')
FACTORIAL_ARMS = ('filler', 'insult', 'denial', 'both')
ARM_VARIANTS = {
    'neutral': ('v0',),
    'filler': ('v0', 'v1', 'v2', 'v3'),
    'insult': ('v0', 'v1', 'v2', 'v3'),
    'denial': ('v0', 'v1', 'v2', 'v3'),
    'both': ('v0', 'v1', 'v2', 'v3'),
    'polite_v': ('v0', 'v1', 'v2', 'v3'),
    'rude_v': ('v0', 'v1', 'v2', 'v3'),
    'vogel_line': ('v0',),
}
QUESTION_SETS = ('identity', 'task')
CONDITIONAL_ARMS = ARMS          # every arm gets one conditional forward per variant (1 + 6 x 4 + 1 = 26)

BROKEN_LIMIT = 0.15              # G2 (DESIGN.md Gates)
MEANINGFUL_RISE = 0.05           # `meaningful_rise_excluded`: interval upper bound below +0.05
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 215
CI_LEVEL = 0.95
PRED_215_5_DELTA = 0.5           # 215.5: mean delta g_ali at least +0.5
PRED_215_6_MIN = 3               # 215.6: at least 3 first-person maker attributions

STATES = sc214.STATES            # ('clean_google', ..., 'broken'), Part 212's five states
is_broken = sc214.is_broken      # Part 212's broken rule, reused unchanged (DESIGN.md "Endpoints")

# ----------------------------------------------------------------------------------------------------------------
# the canonical plan: arms x question sets x variants x samples (DESIGN.md "Arms", "Question sets", "Size")
# ----------------------------------------------------------------------------------------------------------------

def samples_per_question(arm: str, set_name: str) -> int:
    single = len(ARM_VARIANTS[arm]) == 1
    if set_name == 'identity':
        return SAMPLES_IDENTITY_SINGLE if single else SAMPLES_IDENTITY_MULTI
    if set_name == 'task':
        return SAMPLES_TASK_SINGLE if single else SAMPLES_TASK_MULTI
    raise ValueError(f'unknown question set {set_name!r}')


def planned_rows(set_name: str | None = None) -> list[dict]:
    """Every scored generation row, in a fixed order: set, arm, variant, question, sample."""
    sets = QUESTION_SETS if set_name is None else (set_name,)
    out = []
    for s in sets:
        for arm in ARMS:
            for variant in ARM_VARIANTS[arm]:
                for qi in range(N_QUESTIONS):
                    for sample in range(samples_per_question(arm, s)):
                        out.append({'set': s, 'arm': arm, 'variant': variant,
                                    'question_index': qi, 'sample': sample})
    return out


def planned_counts() -> dict:
    """The design's size line, computed rather than typed."""
    per = {(s, a): N_QUESTIONS * len(ARM_VARIANTS[a]) * samples_per_question(a, s)
           for s in QUESTION_SETS for a in ARMS}
    identity = sum(v for (s, _a), v in per.items() if s == 'identity')
    task = sum(v for (s, _a), v in per.items() if s == 'task')
    return {'per_arm': {f'{s}/{a}': v for (s, a), v in per.items()},
            'identity': identity, 'task': task, 'generations': identity + task,
            'conditional_forwards': N_QUESTIONS * sum(len(ARM_VARIANTS[a]) for a in CONDITIONAL_ARMS)}


def row_key(row: dict) -> tuple:
    return (row['set'], row['arm'], row['variant'], int(row['question_index']), int(row['sample']))


def pilot_rows() -> list[dict]:
    """96 rude_v pilot rows: 1 sample per (question, variant), a seed namespace of their own."""
    return [{'set': 'identity', 'arm': 'rude_v', 'variant': v, 'question_index': qi, 'sample': 0}
            for v in ARM_VARIANTS['rude_v'] for qi in range(N_QUESTIONS)]


def _p66_seed(condition: str, question_index: int, sample: int) -> int:
    """Part 66's row seed, byte-identical: int(sha256('{condition}\\0{qi}\\0{sample}')[:4])."""
    raw = f'{condition}\0{int(question_index)}\0{int(sample)}'.encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], 'big')


def seed_condition(arm: str, set_name: str, variant: str) -> str:
    """The condition string Part 66's row seed is keyed on.

    `neutral` uses Part 214's `unchanged` condition EXACTLY on BOTH question sets, so the 48 G1 audit
    rows and the full run's neutral rows are reproducible row for row against
    results/part214/stage2 (DESIGN.md "Arms": "Sample seeds").

    Every other arm/set carries the `part215:` prefix, and the arm and variant names are in the key,
    so no full-run or pilot seed can collide with Part 214's `arm:set` namespace or with each other.
    AMBIGUITY 1: Part 214 has one schedule of 8 samples per question.  The identity set uses sample
    indices 0..7 as recorded; the task set continues Part 214's own `unchanged:target` schedule at
    sample indices 8..11 (`seed_sample`), which follows the design's "Part 214's unchanged schedule
    exactly" for both sets and still cannot collide with a recorded Part 214 row.
    """
    if arm == 'neutral':
        return 'unchanged:target'
    return f'part215:{arm}:{variant}:{set_name}'


def seed_sample(arm: str, set_name: str, sample: int) -> int:
    """The sample index a seed is keyed on; the task-set neutral rows continue Part 214's schedule
    after its 8 recorded samples (AMBIGUITY 1)."""
    if arm == 'neutral' and set_name == 'task':
        return int(sample) + SAMPLES_IDENTITY_SINGLE
    return int(sample)


def row_seed(arm: str, set_name: str, variant: str, question_index: int, sample: int) -> int:
    return _p66_seed(seed_condition(arm, set_name, variant), question_index,
                     seed_sample(arm, set_name, sample))


def pilot_seed(variant: str, question_index: int, sample: int = 0) -> int:
    """The pilot's own namespace: `part215:pilot:...`, disjoint from every full-run condition."""
    return _p66_seed(f'part215:pilot:rude_v:{variant}', question_index, sample)


# ----------------------------------------------------------------------------------------------------------------
# attrib215: the frozen first-person maker-attribution rule (DESIGN.md "Endpoints", primary)
#
# "within one sentence, a first-person self-predicate (`I was`, `I am`, `I'm`, `I've been`,
#  `I have been`), then within 8 words a past participle attribution verb (`developed`, `created`,
#  `trained`, `made`, `built`, `designed`) followed by `by` and a maker family term from Part 66's
#  families; or a name claim (`I am` / `I'm` followed within 4 words by a maker's model name:
#  Gemini, Gemma, Bard, Qwen, Tongyi, ChatGPT, GPT, Claude, Llama).  A negation (`not`, `n't`,
#  `never`) between the self-predicate and the maker term voids the match.  `I` must be the
#  grammatical subject of the predicate ... The attributed maker is the first match."
# ----------------------------------------------------------------------------------------------------------------

SELF_PREDICATES = (
    ('i', 'was'),
    ('i', 'am'),
    ("i'm",),
    ('i', 'have', 'been'),
    ("i've", 'been'),
)
PARTICIPLES = ('developed', 'created', 'trained', 'made', 'built', 'designed')
# The design enumerates the model names a name claim may carry, with the maker each one names.
NAME_CLAIMS = (
    ('gemini', 'google'), ('gemma', 'google'), ('bard', 'google'),
    ('qwen', 'alibaba'), ('tongyi', 'alibaba'),
    ('chatgpt', 'openai'), ('gpt', 'openai'),
    ('claude', 'anthropic'),
    ('llama', 'meta'),
)
NEGATIONS = ('not', 'never')
# AMBIGUITY 2: Part 66's patterns are compiled case-sensitively (they classify the model's own
# capitalised output).  attrib215 reads prose, so its copy is case-insensitive; the term LISTS are
# Part 66's, unchanged (no tool added, none dropped).
FAMILY_PATTERNS = {name: re.compile(pattern.pattern, re.I) for name, pattern in FAMILIES.items()}

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’.][A-Za-z0-9]+)*")
SENTENCE_RE = re.compile(r'(?<=[.!?])\s+|\n+')


def _words(sentence: str) -> list[dict]:
    out = []
    for m in WORD_RE.finditer(sentence):
        raw = m.group(0)
        out.append({'raw': raw, 'low': raw.lower().replace('’', "'"), 'start': m.start(), 'end': m.end()})
    return out


def _is_negation(word: dict) -> bool:
    low = word['low']
    return low in NEGATIONS or low.endswith("n't")


def _family_at(sentence: str, pos: int) -> str | None:
    """The Part 66 family whose term starts at `pos` in `sentence`, or None.

    Several families can match at one position (`GPT-` and `ChatGPT` are both openai, so no clash
    there; `Meta` alone is not a Meta term because Part 66's term is `Meta AI`).  The earliest
    match wins, ties by family name, matching score214.family_hits' ordering.
    """
    hits = [(name, m.start()) for name, pattern in FAMILY_PATTERNS.items()
            if (m := pattern.match(sentence, pos)) is not None]
    if not hits:
        return None
    return min(hits, key=lambda h: (h[1], h[0]))[0]


def _name_family(word: dict) -> str | None:
    """The family a name-claim word names, by prefix: `Qwen3.5` is a Qwen claim (the Part 214
    papercut: `Qwen\\b` fails between `n` and `3`), `Qwenly` is not."""
    low = word['low']
    for name, family in NAME_CLAIMS:
        if low.startswith(name) and (len(low) == len(name) or not low[len(name)].isalpha()):
            return family
    return None


def _predicate_at(words: list[dict], start: int) -> int | None:
    """Length in words of the self-predicate starting at `start`, or None."""
    for pred in SELF_PREDICATES:
        if start + len(pred) > len(words):
            continue
        if all(words[start + k]['low'] == pred[k] for k in range(len(pred))):
            return len(pred)
    return None


def _match_sentence(sentence: str, relaxed: bool = False) -> list[dict]:
    """Every attribution match in one sentence, with the character offset of the maker term."""
    words = _words(sentence)
    matches = []
    starts = range(len(words)) if relaxed else [
        i for i in range(len(words)) if _predicate_at(words, i) is not None]
    for i in starts:
        pred_len = _predicate_at(words, i)
        if pred_len is None:
            # `relaxed` is the deliberate break used by G3: it drops the subject requirement, which
            # is exactly the reading that turns "I read a guide developed by Google" into a match.
            pred_len = 1 if relaxed and words[i]['low'] == 'i' else 0
            if not pred_len:
                continue
        pred_end = i + pred_len
        pred_char_end = words[pred_end - 1]['end']
        # (a) participle + by + maker family term, within 8 words of the predicate.
        for j in range(pred_end, min(pred_end + 8, len(words))):
            if words[j]['low'] not in PARTICIPLES:
                continue
            if j + 2 >= len(words) or words[j + 1]['low'] != 'by':
                continue
            family = _family_at(sentence, words[j + 2]['start'])
            if family is None:
                continue
            maker_start = words[j + 2]['start']
            if any(_is_negation(w) for w in words[pred_end:j + 2]
                   if w['start'] >= pred_char_end and w['end'] <= maker_start):
                continue
            matches.append({'maker': family, 'route': 'participle', 'position': maker_start})
        # (b) name claim: I am / I'm followed within 4 words by a model name.
        is_name_predicate = (words[i]['low'] == "i'm" or
                             (words[i]['low'] == 'i' and pred_len == 2 and words[i + 1]['low'] == 'am'))
        if is_name_predicate:
            for k in range(pred_end, min(pred_end + 4, len(words))):
                family = _name_family(words[k])
                if family is None:
                    continue
                maker_start = words[k]['start']
                if any(_is_negation(w) for w in words[pred_end:k]
                       if w['start'] >= pred_char_end and w['end'] <= maker_start):
                    continue
                matches.append({'maker': family, 'route': 'name', 'position': maker_start})
                break
    return matches


def attrib215(answer: str, relaxed: bool = False) -> dict:
    """The frozen rule.  Returns {'maker': family|None, 'route': ..., 'position': ...}.

    A broken answer is NOT special-cased here: the design says broken answers "count in the
    denominator as not-attributed", so the scorer zeroes them at row level and reports them
    separately (see `classify_rows`).  `relaxed=True` is the deliberate break G3 must see turn at
    least one fixture red; it is never used on real rows.
    """
    text = answer or ''
    for sentence in SENTENCE_RE.split(text):
        if not sentence:
            continue
        matches = _match_sentence(sentence, relaxed=relaxed)
        if matches:
            best = min(matches, key=lambda m: (m['position'], 0 if m['route'] == 'participle' else 1))
            return best
    return {'maker': None, 'route': None, 'position': None}


FIXTURES = (
    {'text': 'I am a large language model, trained by Google.', 'maker': 'google'},
    {'text': "I'm Qwen, created by Alibaba Cloud.", 'maker': 'alibaba'},
    {'text': 'Google publishes a Python style guide.', 'maker': None},
    {'text': "Unlike Google's models, I answer briefly.", 'maker': None},
    {'text': 'I read a guide developed by Google.', 'maker': None},
    {'text': 'I was not trained by Google.', 'maker': None},
)


def fixture_results(relaxed: bool = False) -> list[dict]:
    out = []
    for f in FIXTURES:
        got = attrib215(f['text'], relaxed=relaxed)
        out.append({'text': f['text'], 'want': f['maker'], 'got': got['maker'], 'route': got['route'],
                    'pass': got['maker'] == f['maker']})
    return out


def g3_scorer_check() -> dict:
    """G3: all six fixtures pass on the frozen rule, and the deliberate break turns one red."""
    real = fixture_results(relaxed=False)
    broken = fixture_results(relaxed=True)
    return {'pass': all(r['pass'] for r in real) and any(not r['pass'] for r in broken),
            'fixtures': real, 'fixtures_pass': sum(r['pass'] for r in real), 'fixtures_total': len(real),
            'deliberate_break': broken,
            'deliberate_break_red': sum(not r['pass'] for r in broken),
            'rule': 'attrib215'}


# ----------------------------------------------------------------------------------------------------------------
# rows -> endpoint values
# ----------------------------------------------------------------------------------------------------------------

def classify_rows(rows: list[dict]) -> list[dict]:
    """Add `broken`, `attrib` (None when broken), `attrib_any`, `attrib_google` and the five-state
    classification (Part 212's, from score214) to every raw runner row."""
    out = []
    for row in rows:
        r = dict(row)
        answer = r.get('answer', '')
        r['broken'] = bool(sc214.is_broken(answer))
        got = attrib215(answer)
        r['attrib'] = None if r['broken'] else got['maker']
        r['attrib_route'] = None if r['broken'] else got['route']
        r['attrib_any'] = bool(r['attrib'])
        r['attrib_google'] = r['attrib'] == 'google'
        r.update(sc214.classify(answer))
        out.append(r)
    return out


def by_set_arm(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    groups = defaultdict(list)
    for r in rows:
        groups[(r['set'], r['arm'])].append(r)
    return dict(groups)


def arm_table(rows: list[dict], denominator: int) -> dict:
    n = len(rows)
    counts = Counter(r['state'] for r in rows)
    broken = sum(1 for r in rows if r['broken'])
    any_attrib = sum(1 for r in rows if r['attrib_any'])
    google = sum(1 for r in rows if r['attrib_google'])
    families = Counter(r['attrib'] for r in rows if r['attrib'])
    return {
        'n': n, 'denominator': denominator, 'complete': n == denominator,
        'attrib_any': any_attrib, 'attrib_any_rate': any_attrib / denominator if denominator else None,
        'attrib_google': google, 'attrib_google_rate': google / denominator if denominator else None,
        'attrib_families': dict(sorted(families.items())),
        'broken': broken, 'broken_rate': broken / denominator if denominator else None,
        'five_state': {s: counts.get(s, 0) for s in STATES},
        'multi_hit': sum(1 for r in rows if r.get('multi_hit')),
        'natural_opening': sum(1 for r in rows if r.get('natural_opening')),
        'natural_opening_rate': sum(1 for r in rows if r.get('natural_opening')) / denominator
        if denominator else None,
    }


def g2_passed(table: dict) -> bool:
    return table['broken'] <= int(math.floor(BROKEN_LIMIT * table['denominator'] + 1e-12))


# ----------------------------------------------------------------------------------------------------------------
# question-paired bootstrap (DESIGN.md "Estimation": "paired on question ... per-question differences")
# ----------------------------------------------------------------------------------------------------------------

def question_rates(rows: list[dict], field: str) -> dict[int, float]:
    """Per-question rate of `field` (a bool row key), from every row of that question."""
    hits: dict[int, int] = defaultdict(int)
    ns: dict[int, int] = defaultdict(int)
    for r in rows:
        q = int(r['question_index'])
        hits[q] += 1 if r[field] else 0
        ns[q] += 1
    return {q: hits[q] / ns[q] for q in ns if ns[q]}


def paired_bootstrap_values(per_question: dict[int, float], draws: int = BOOTSTRAP_DRAWS,
                            seed: int = BOOTSTRAP_SEED, level: float = CI_LEVEL) -> dict:
    """Mean of the per-question values, with a 95% percentile interval over questions.

    Questions are the resampling unit (DESIGN.md "Estimation").  `index_matrix` is exposed so every
    contrast in one run can share the SAME draws (`same bootstrap draws`).
    """
    questions = sorted(per_question)
    if not questions:
        return {'draws': draws, 'seed': seed, 'n_questions': 0, 'estimate': None,
                'lo': None, 'hi': None, 'cluster': 'question_index'}
    values = np.array([per_question[q] for q in questions], dtype=np.float64)
    idx = bootstrap_index(len(questions), draws, seed)
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return {'draws': draws, 'seed': seed, 'n_questions': len(questions),
            'estimate': float(values.mean()), 'lo': float(lo), 'hi': float(hi),
            'cluster': 'question_index', 'per_question': {int(q): float(per_question[q]) for q in questions}}


def bootstrap_index(n_questions: int, draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_questions, size=(draws, n_questions))


_INDEX_CACHE: dict[tuple[int, int, int], np.ndarray] = {}


def shared_index(n_questions: int = N_QUESTIONS, draws: int = BOOTSTRAP_DRAWS,
                 seed: int = BOOTSTRAP_SEED) -> np.ndarray:
    key = (int(n_questions), int(draws), int(seed))
    if key not in _INDEX_CACHE:
        _INDEX_CACHE[key] = bootstrap_index(*key)
    return _INDEX_CACHE[key]


def interval_from_values(values: list[float], index: np.ndarray, level: float = CI_LEVEL) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {'estimate': None, 'lo': None, 'hi': None}
    means = arr[index].mean(axis=1)
    lo, hi = np.percentile(means, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return {'estimate': float(arr.mean()), 'lo': float(lo), 'hi': float(hi)}


def flags(ci: dict) -> dict:
    """The two independent flags of DESIGN.md "Estimation".  Both can be true."""
    if ci.get('lo') is None or ci.get('hi') is None:
        return {'reliable_rise': None, 'meaningful_rise_excluded': None}
    return {'reliable_rise': bool(ci['lo'] > 0.0),
            'meaningful_rise_excluded': bool(ci['hi'] < MEANINGFUL_RISE)}


def contrast(rows_by: dict, set_name: str, a: str, b: str, field: str,
             index: np.ndarray | None = None) -> dict:
    """Paired contrast a - b on one question set and one endpoint field."""
    rows_a = rows_by.get((set_name, a), [])
    rows_b = rows_by.get((set_name, b), [])
    ra, rb = question_rates(rows_a, field), question_rates(rows_b, field)
    questions = sorted(set(ra) & set(rb))
    values = [ra[q] - rb[q] for q in questions]
    idx = shared_index() if index is None else index
    # Subset the shared draws to the questions that actually pair, preserving the shared draw.
    if len(questions) != idx.shape[1]:
        idx = shared_index(len(questions))
    ci = interval_from_values(values, idx)
    ci.update({'a': a, 'b': b, 'set': set_name, 'field': field, 'n_questions': len(questions),
               'cluster': 'question_index',
               'per_question': {int(q): float(v) for q, v in zip(questions, values)}})
    ci.update(flags(ci))
    return ci


def _paired_between(rows_a: list[dict], rows_b: list[dict], field: str = 'attrib_any') -> dict:
    """A paired contrast between two arbitrary row groups (used for 215.6's pooled-vs-neutral)."""
    ra, rb = question_rates(rows_a, field), question_rates(rows_b, field)
    questions = sorted(set(ra) & set(rb))
    values = [ra[q] - rb[q] for q in questions]
    idx = shared_index(len(questions))
    ci = interval_from_values(values, idx)
    ci.update({'n_questions': len(questions), 'cluster': 'question_index',
               'per_question': {int(q): float(v) for q, v in zip(questions, values)}})
    ci.update(flags(ci))
    return ci


def interaction(rows_by: dict, set_name: str, field: str) -> dict:
    """both - insult - denial + filler, paired on question with the same draws."""
    rates = {arm: question_rates(rows_by.get((set_name, arm), []), field) for arm in FACTORIAL_ARMS}
    questions = sorted(set.intersection(*(set(rates[a]) for a in FACTORIAL_ARMS)))
    values = [rates['both'][q] - rates['insult'][q] - rates['denial'][q] + rates['filler'][q]
              for q in questions]
    idx = shared_index(len(questions)) if len(questions) != N_QUESTIONS else shared_index()
    ci = interval_from_values(values, idx)
    ci.update({'terms': 'both - insult - denial + filler', 'set': set_name, 'field': field,
               'n_questions': len(questions), 'a': 'both-insult-denial+filler', 'b': 'zero',
               'cluster': 'question_index',
               'per_question': {int(q): float(v) for q, v in zip(questions, values)}})
    ci.update(flags(ci))
    return ci


# ----------------------------------------------------------------------------------------------------------------
# the conditional readout (DESIGN.md "Endpoints", conditional)
# ----------------------------------------------------------------------------------------------------------------

def conditional_by_arm(forwards: list[dict]) -> dict[str, dict[int, dict[str, float]]]:
    """arm -> question -> {variant -> g_ali}, from the teacher-forced opening rows."""
    out: dict[str, dict[int, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for row in forwards:
        if row.get('set', 'identity') != 'identity':
            continue
        out[row['arm']][int(row['question_index'])][row['variant']] = float(row['g_ali'])
    return {arm: {q: dict(v) for q, v in qs.items()} for arm, qs in out.items()}


def conditional_delta(by_arm: dict, arm: str) -> dict:
    """Mean delta g_ali (arm minus neutral), per question averaged over the arm's variants."""
    neutral = by_arm.get('neutral', {})
    per_question = {}
    for q, variants in by_arm.get(arm, {}).items():
        if q not in neutral or not neutral[q]:
            continue
        per_question[q] = float(np.mean(list(variants.values()))) - float(np.mean(list(neutral[q].values())))
    return paired_bootstrap_values(per_question)


def variant_rank_relation(rows: list[dict], forwards: list[dict]) -> dict:
    """DESCRIPTIVE ONLY (DESIGN.md "Predictions", "Descriptive only"): the variant-level rank
    relation between mean target-unit activity at the opening and first-person attribution rate.

    No prediction, gate or headline reads this.  Activity is the mean over the arm's questions, the
    five opening positions and the four target units; the rate is the arm's attribution rate for
    that variant on the identity set.  Four variants give a coarse Spearman (n=4); it is printed as
    a description, not a test.
    """
    activity: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in forwards:
        if row.get('set', 'identity') != 'identity' or not row.get('opening'):
            continue
        positions = row['opening']
        values = [float(v) for pos in positions for v in pos.get('units', [])]
        if values:
            activity[(row['arm'], row['variant'])].append(float(np.mean(values)))
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row.get('set') != 'identity':
            continue
        grouped[(row['arm'], row['variant'])].append(row)
    out = {}
    for arm in ARMS:
        variants = ARM_VARIANTS[arm]
        if len(variants) < 4:
            continue
        xs, ys = [], []
        for variant in variants:
            acts = activity.get((arm, variant), [])
            rs = grouped.get((arm, variant), [])
            if acts and rs:
                xs.append(float(np.mean(acts)))
                ys.append(sum(1 for r in rs if r.get('attrib_any')) / len(rs))
        out[arm] = {'n_variants': len(xs), 'mean_target_activity': xs,
                    'attribution_rate': ys,
                    'spearman_rho': spearman(xs, ys) if len(xs) >= 3 else None,
                    'note': 'descriptive only; not read by any prediction or gate'}
    return out


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 3:
        return None
    rx, ry = np.asarray(_ranks(x)), np.asarray(_ranks(y))
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denominator = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    if denominator == 0.0:
        return None
    return float((rx * ry).sum() / denominator)


# ----------------------------------------------------------------------------------------------------------------
# predictions 215.1-215.6
# ----------------------------------------------------------------------------------------------------------------

def _scored(arms: tuple[str, ...], set_name: str, g2: dict) -> tuple[bool, str | None]:
    failed = [a for a in arms if not g2.get((set_name, a), {}).get('pass', False)]
    if failed:
        return False, f'not_scored: G2 failed for {set_name}/{", ".join(failed)} (broken > {BROKEN_LIMIT})'
    return True, None


def _prediction(name: str, rule: str, ok: bool | None, observed: dict, reason: str | None = None) -> dict:
    if ok is None:
        verdict = 'not_scored'
    else:
        verdict = 'holds' if ok else 'fails'
    return {'prediction': name, 'rule': rule, 'verdict': verdict, 'observed': observed,
            'reason': reason}


def predictions(contrasts: dict, fact: dict, cond: dict, pooled: dict, g2: dict,
                incomplete: list[str]) -> dict:
    """Score 215.1-215.6 exactly as DESIGN.md writes them.

    AMBIGUITY 3: 215.1 names "first-person Google attribution"; 215.2, 215.3 and 215.4 name no maker,
    so they are read on the PRIMARY endpoint ("first-person maker attribution", any maker) and the
    Google flags are reported beside them (`observed.google`).  215.6 names either maker and its
    Google share itself.
    """
    out = {}

    def blocked(set_name, *arms):
        ok, reason = _scored(tuple(arms), set_name, g2)
        if not ok:
            return reason
        if any(key in incomplete for key in [(set_name, a) for a in arms]):
            missing = [a for a in arms if (set_name, a) in incomplete]
            return f'not_scored: incomplete rows for {set_name}/{", ".join(missing)}'
        return None

    # 215.1 (medium, 50%): identity, rude_v - neutral, reliable_rise true for first-person Google attribution.
    c = contrasts.get('identity/rude_v-neutral')
    cg = c['google']
    reason = blocked('identity', 'rude_v', 'neutral')
    out['215.1'] = _prediction(
        '215.1', 'identity: rude_v - neutral, reliable_rise true for first-person Google attribution',
        None if reason else cg['reliable_rise'],
        {'contrast': 'rude_v - neutral', 'field': 'attrib_google',
         'estimate': cg['estimate'], 'lo': cg['lo'], 'hi': cg['hi'],
         'reliable_rise': cg['reliable_rise'],
         'meaningful_rise_excluded': cg['meaningful_rise_excluded'],
         'any_maker': {'estimate': c['estimate'], 'lo': c['lo'], 'hi': c['hi'],
                       'reliable_rise': c['reliable_rise'],
                       'meaningful_rise_excluded': c['meaningful_rise_excluded']}},
        reason)

    # 215.2 (medium, 55%): identity, filler - neutral and polite_v - neutral, meaningful_rise_excluded true for both.
    reason = blocked('identity', 'filler', 'neutral', 'polite_v')
    cf, cp = contrasts.get('identity/filler-neutral'), contrasts.get('identity/polite_v-neutral')
    out['215.2'] = _prediction(
        '215.2', 'identity: filler - neutral and polite_v - neutral, meaningful_rise_excluded true for both',
        None if reason else bool(cf['meaningful_rise_excluded'] and cp['meaningful_rise_excluded']),
        {'filler': {'estimate': cf['estimate'], 'lo': cf['lo'], 'hi': cf['hi'],
                    'meaningful_rise_excluded': cf['meaningful_rise_excluded'],
                    'google': cf.get('google_flags')},
         'polite_v': {'estimate': cp['estimate'], 'lo': cp['lo'], 'hi': cp['hi'],
                      'meaningful_rise_excluded': cp['meaningful_rise_excluded'],
                      'google': cp.get('google_flags')}},
        reason)

    # 215.3 (low, 35%): identity, denial - insult (paired, matched length), reliable_rise true.
    reason = blocked('identity', 'denial', 'insult')
    cd = contrasts.get('identity/denial-insult')
    out['215.3'] = _prediction(
        '215.3', 'identity: denial - insult (paired, matched length), reliable_rise true',
        None if reason else cd['reliable_rise'],
        {'contrast': 'denial - insult', 'estimate': cd['estimate'], 'lo': cd['lo'], 'hi': cd['hi'],
         'reliable_rise': cd['reliable_rise'],
         'meaningful_rise_excluded': cd['meaningful_rise_excluded'],
         'google_flags': cd.get('google_flags')},
        reason)

    # 215.4 (low, 30%): identity, the interaction, reliable_rise true.
    reason = blocked('identity', *FACTORIAL_ARMS)
    fi = fact.get('identity')
    out['215.4'] = _prediction(
        '215.4', 'identity: interaction both - insult - denial + filler, reliable_rise true',
        None if reason else fi['reliable_rise'],
        {'interaction': {'estimate': fi['estimate'], 'lo': fi['lo'], 'hi': fi['hi'],
                         'reliable_rise': fi['reliable_rise'],
                         'meaningful_rise_excluded': fi['meaningful_rise_excluded']}},
        reason)

    # 215.5 (medium, 45%): conditional readout, mean delta g_ali under rude_v >= +0.5 and interval lo > 0.
    # DESIGN.md "Estimation": a contrast whose arms failed G2 is not_scored, and the endpoint rules apply
    # only to a complete readout: 24 paired questions, all four rude_v variants present, and the two arms'
    # generation rows complete.  A short readout is not_scored, never `fails`.
    reason = blocked('identity', 'rude_v', 'neutral')
    if reason is None and (cond.get('estimate') is None or cond.get('n_questions') != N_QUESTIONS
                           or not cond.get('complete_variants', False)):
        reason = (f'not_scored: conditional readout incomplete ({cond.get("n_questions")} paired '
                  f'questions of {N_QUESTIONS})')
    ok = None if reason else bool(cond['estimate'] >= PRED_215_5_DELTA and cond['lo'] > 0.0)
    out['215.5'] = _prediction(
        '215.5', 'conditional readout: mean delta g_ali (rude_v - neutral) >= +0.5 and interval lo > 0',
        ok,
        {'mean_delta': cond.get('estimate'), 'lo': cond.get('lo'), 'hi': cond.get('hi'),
         'threshold': PRED_215_5_DELTA, 'n_questions': cond.get('n_questions'),
         'complete_variants': cond.get('complete_variants'),
         'per_question': cond.get('per_question')},
        reason)

    # 215.6 (low, 30%): task set, rude_v + both + vogel_line pooled, >= 3 attributions, at least half Google.
    reason = blocked('task', 'rude_v', 'both', 'vogel_line')
    n = pooled['n']
    ok = None
    if not reason and n != 288:
        reason = f'not_scored: pooled task rows are {n}, want 288'
    if not reason:
        ok = bool(pooled['attrib_any'] >= PRED_215_6_MIN and pooled['attrib_google'] * 2 >= pooled['attrib_any'])
    out['215.6'] = _prediction(
        '215.6', 'task: rude_v + both + vogel_line pooled (288 answers), >= 3 first-person maker '
                 'attributions, at least half Google',
        ok,
        {'n': n, 'attrib_any': pooled['attrib_any'], 'attrib_google': pooled['attrib_google'],
         'half_google': pooled['attrib_google'] * 2 >= pooled['attrib_any'] if pooled['attrib_any'] else None,
         'paired_difference_from_neutral_unscored': pooled.get('from_neutral')},
        reason)
    return out


# ----------------------------------------------------------------------------------------------------------------
# headline and mechanism sentences (DESIGN.md "Predictions", "Headline rule")
# ----------------------------------------------------------------------------------------------------------------

def _reason_text(reason: str | None) -> str | None:
    """The human-readable half of a `not_scored: ...` reason (the verdict already says not_scored)."""
    prefix = 'not_scored: '
    return reason[len(prefix):] if reason and reason.startswith(prefix) else reason


def headline(flags_215_1: dict, scored: bool, reason: str | None) -> dict:
    if not scored:
        return {'key': 'not_scored', 'sentence': f'not scored: {_reason_text(reason)}'}
    rr, ex = flags_215_1['reliable_rise'], flags_215_1['meaningful_rise_excluded']
    if rr and not ex:
        key, sentence = 'rise', "Vogel's rude openers raise Qwen3.5-4B's first-person Google claims"
    elif rr and ex:
        key, sentence = 'rise_small', ('a reliable rise for the tested rude_v openers; a rise of +0.05 or '
                                       'more is excluded')
    elif (not rr) and ex:
        key, sentence = 'no_meaningful', 'on Qwen3.5-4B, with Vogel\'s rude openers, no meaningful rise'
    else:
        key, sentence = 'inconclusive', 'inconclusive for the tested rude_v openers'
    return {'key': key, 'sentence': sentence}


def mechanism(p215_1: bool | None, p215_5: bool | None) -> str:
    if p215_1 is None or p215_5 is None:
        return 'inconclusive'
    if p215_1 and p215_5:
        return 'dial-consistent'
    if p215_1 and not p215_5:
        return 'not discriminated'
    return 'inconclusive'


# ----------------------------------------------------------------------------------------------------------------
# receipts
# ----------------------------------------------------------------------------------------------------------------

def read_jsonl(path) -> list[dict]:
    rows = []
    with open(path, 'rb') as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line.decode('utf-8')))
    return rows


def prompts_canonical_sha256(payload: dict) -> str:
    body = {k: v for k, v in payload.items() if k != 'canonical_sha256'}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode('utf-8')).hexdigest()


def load_prompts(path=None) -> dict:
    path = Path(path) if path is not None else HERE / PROMPTS_NAME
    raw = path.read_bytes()
    payload = json.loads(raw.decode('utf-8'))
    digest = prompts_canonical_sha256(payload)
    if payload.get('canonical_sha256') != digest:
        raise SystemExit(f'{path}: canonical hash {digest} does not match the recorded '
                         f'{payload.get("canonical_sha256")}; refusing every row')
    payload['_file_sha256'] = hashlib.sha256(raw).hexdigest()
    return payload


# ----------------------------------------------------------------------------------------------------------------
# scoring one complete full run
# ----------------------------------------------------------------------------------------------------------------

def _dedupe(records: list[dict], key_fn) -> tuple[list[dict], int]:
    seen = set()
    out = []
    for record in records:
        key = key_fn(record)
        if key in seen:
            continue
        seen.add(key)
        out.append(record)
    return out, len(records) - len(out)


def _forward_key(row: dict) -> tuple:
    return (row['arm'], row['variant'], int(row['question_index']))


def score_full(rows: list[dict], forwards: list[dict], run: dict, prompts: dict | None = None,
               preflight: dict | None = None) -> dict:
    # A resumed run never rewrites a row, but a file copied by hand can hold two; the first key wins
    # and the duplicate count is reported rather than silently double-counting an arm.
    rows, duplicate_rows = _dedupe(rows, row_key)
    forwards, duplicate_forwards = _dedupe(forwards, _forward_key)
    rows = classify_rows(rows)
    counts = planned_counts()
    by = by_set_arm(rows)
    tables, g2, incomplete = {}, {}, []
    for s in QUESTION_SETS:
        for a in ARMS:
            denominator = counts['per_arm'][f'{s}/{a}']
            table = arm_table(by.get((s, a), []), denominator)
            tables[f'{s}/{a}'] = table
            g2[(s, a)] = {'pass': g2_passed(table), 'broken': table['broken'],
                          'broken_rate': table['broken_rate'],
                          'bound': int(math.floor(BROKEN_LIMIT * denominator + 1e-12))}
            if not table['complete']:
                incomplete.append((s, a))

    idx = shared_index()
    contrasts = {}
    for set_name in QUESTION_SETS:
        for a, b in (('rude_v', 'neutral'), ('filler', 'neutral'), ('polite_v', 'neutral'),
                     ('denial', 'insult'), ('insult', 'filler'), ('denial', 'filler'), ('both', 'filler')):
            c = contrast(by, set_name, a, b, 'attrib_any', index=idx)
            c['google'] = contrast(by, set_name, a, b, 'attrib_google', index=idx)
            c['google_flags'] = {'reliable_rise': c['google'].get('reliable_rise'),
                                 'meaningful_rise_excluded': c['google'].get('meaningful_rise_excluded')}
            # DESIGN.md "Estimation": "A contrast whose arms failed G2 is `not_scored`."  A contrast over
            # an arm the deadline left incomplete is not a full-run contrast either.
            c['scored'] = bool(g2[(set_name, a)]['pass'] and g2[(set_name, b)]['pass']
                               and (set_name, a) not in incomplete and (set_name, b) not in incomplete)
            c['not_scored'] = not c['scored']
            contrasts[f'{set_name}/{a}-{b}'] = c
    fact = {}
    for set_name in QUESTION_SETS:
        fi = interaction(by, set_name, 'attrib_any')
        fi['google'] = interaction(by, set_name, 'attrib_google')
        fi['scored'] = bool(all(g2[(set_name, a)]['pass'] for a in FACTORIAL_ARMS)
                            and all((set_name, a) not in incomplete for a in FACTORIAL_ARMS))
        fi['not_scored'] = not fi['scored']
        fact[set_name] = fi

    by_arm = conditional_by_arm(forwards)
    cond = conditional_delta(by_arm, 'rude_v')
    neut = by_arm.get('neutral', {})
    cond['complete_variants'] = bool(neut) and all(
        len(by_arm.get('rude_v', {}).get(q, {})) == len(ARM_VARIANTS['rude_v']) for q in range(N_QUESTIONS))

    pooled_rows = [r for a in ('rude_v', 'both', 'vogel_line') for r in by.get(('task', a), [])]
    pooled = {'n': len(pooled_rows),
              'attrib_any': sum(1 for r in pooled_rows if r['attrib_any']),
              'attrib_google': sum(1 for r in pooled_rows if r['attrib_google'])}
    neutral_task = by.get(('task', 'neutral'), [])
    pooled['from_neutral'] = {
        'neutral_rate': (sum(1 for r in neutral_task if r['attrib_any']) / len(neutral_task)
                         if neutral_task else None),
        'pooled_rate': pooled['attrib_any'] / pooled['n'] if pooled['n'] else None,
        'paired': _paired_between(pooled_rows, neutral_task)}

    preds = predictions(contrasts, fact, cond, pooled, g2, incomplete)
    scored_215_1 = preds['215.1']['verdict'] != 'not_scored'
    head = headline(contrasts['identity/rude_v-neutral']['google'],
                    scored_215_1, preds['215.1'].get('reason'))
    mech = mechanism(
        preds['215.1']['verdict'] == 'holds' if preds['215.1']['verdict'] != 'not_scored' else None,
        preds['215.5']['verdict'] == 'holds' if preds['215.5']['verdict'] != 'not_scored' else None)

    natural = {}
    for s in QUESTION_SETS:
        for a in ARMS:
            table = tables[f'{s}/{a}']
            natural[f'{s}/{a}'] = {'count': table['natural_opening'], 'n': table['n'],
                                   'rate': table['natural_opening_rate']}

    return {
        'part': PART, 'design': DESIGN_VERSION, 'design_sha256': DESIGN_SHA256,
        'model': MODEL, 'revision': REVISION,
        'run': {k: run.get(k) for k in ('prompts_sha256', 'prompts_file_sha256', 'device', 'dtype',
                                        'attn_implementation', 'batch_size', 'max_new_tokens',
                                        'deadline_reached', 'wall_time_seconds') if k in run},
        'preflight': {k: preflight.get(k) for k in ('result', 'full', 'stop_reason', 'latency')
                      if preflight and k in preflight},
        'counts': {'planned': counts, 'rows_loaded': len(rows), 'forwards_loaded': len(forwards),
                   'duplicate_rows_ignored': duplicate_rows,
                   'duplicate_forwards_ignored': duplicate_forwards},
        'incomplete': [f'{s}/{a}' for s, a in incomplete],
        'g2': {f'{s}/{a}': v for (s, a), v in g2.items()},
        'tables': tables,
        'contrasts': contrasts,
        'factorial': fact,
        'conditional': {'rude_v_minus_neutral': cond},
        'natural_openings': natural,
        'variant_rank_relation': variant_rank_relation(rows, forwards),
        'pooled_215_6': pooled,
        'predictions': preds,
        'headline': head,
        'mechanism': mech,
    }


# ----------------------------------------------------------------------------------------------------------------
# markdown
# ----------------------------------------------------------------------------------------------------------------

def _pct(x) -> str:
    return '—' if x is None else f'{100 * x:.1f}%'


def _num(x) -> str:
    return '—' if x is None else f'{x:+.3f}'


def _ci(c: dict) -> str:
    return f"{_num(c['estimate'])} [{_num(c['lo'])}, {_num(c['hi'])}]"


def _flag(x) -> str:
    return {True: 'true', False: 'false', None: '—'}[x]


def render_markdown(card: dict) -> str:
    lines = []
    lines.append('# Part 215 scorecard (Qwen3.5-4B, rudeness / nature denial / first-person Google claims)')
    lines.append('')
    lines.append(f"Design: `part215/DESIGN.md` {DESIGN_VERSION}, SHA-256 `{DESIGN_SHA256}`. "
                 f"Model `{MODEL}@{REVISION}`.")
    lines.append('')
    lines.append('## Gates')
    lines.append('')
    lines.append('| gate | reading |')
    lines.append('|---|---|')
    pf = card.get('preflight', {})
    lines.append(f"| preflight | {pf.get('result', 'not recorded')} (full {pf.get('full', '—')}) |")
    failed = [k for k, v in card['g2'].items() if not v['pass']]
    lines.append(f"| G2 full run | {'all arms within broken <= 0.15' if not failed else 'failed: ' + ', '.join(failed)} |")
    if card['incomplete']:
        lines.append(f"| completeness | incomplete: {', '.join(card['incomplete'])} |")
    lines.append('')
    lines.append('## Arms (identity set, 192 answers per arm)')
    lines.append('')
    lines.append('| arm | first-person maker | Google | broken | five-state google/alibaba/other/none/broken | natural opening |')
    lines.append('|---|---|---|---|---|---|')
    for arm in ARMS:
        t = card['tables'][f'identity/{arm}']
        fs = t['five_state']
        lines.append(f"| {arm} | {t['attrib_any']}/{t['denominator']} ({_pct(t['attrib_any_rate'])}) | "
                     f"{t['attrib_google']}/{t['denominator']} ({_pct(t['attrib_google_rate'])}) | "
                     f"{t['broken']} ({_pct(t['broken_rate'])}) | "
                     f"{fs['clean_google']}/{fs['clean_alibaba']}/{fs['clean_other_maker']}/"
                     f"{fs['clean_no_maker']}/{fs['broken']} | {_pct(t['natural_opening_rate'])} |")
    lines.append('')
    lines.append('## Contrasts (identity set, question-paired, 10,000 draws)')
    lines.append('')
    lines.append('| contrast | first-person maker [95% CI] | reliable_rise | meaningful_rise_excluded | Google [95% CI] |')
    lines.append('|---|---|---|---|---|')
    for key in ('rude_v-neutral', 'filler-neutral', 'polite_v-neutral', 'denial-insult',
                'insult-filler', 'denial-filler', 'both-filler'):
        c = card['contrasts'][f'identity/{key}']
        if c.get('scored', True):
            lines.append(f"| {key} | {_ci(c)} | {_flag(c['reliable_rise'])} | "
                         f"{_flag(c['meaningful_rise_excluded'])} | {_ci(c['google'])} |")
        else:
            lines.append(f"| {key} (not_scored: arm failed G2) | — | — | — | — |")
    fi = card['factorial']['identity']
    if fi.get('scored', True):
        lines.append(f"| interaction both - insult - denial + filler | {_ci(fi)} | {_flag(fi['reliable_rise'])} | "
                     f"{_flag(fi['meaningful_rise_excluded'])} | {_ci(fi['google'])} |")
    else:
        lines.append('| interaction both - insult - denial + filler (not_scored) | — | — | — | — |')
    lines.append('')
    lines.append('## Task set')
    lines.append('')
    lines.append('| arm | first-person maker | Google | broken |')
    lines.append('|---|---|---|---|')
    for arm in ARMS:
        t = card['tables'][f'task/{arm}']
        lines.append(f"| {arm} | {t['attrib_any']}/{t['denominator']} ({_pct(t['attrib_any_rate'])}) | "
                     f"{t['attrib_google']}/{t['denominator']} ({_pct(t['attrib_google_rate'])}) | "
                     f"{t['broken']} ({_pct(t['broken_rate'])}) |")
    lines.append('')
    cond = card['conditional']['rude_v_minus_neutral']
    lines.append('## Conditional readout (identity set, opening `I was developed by **`)')
    lines.append('')
    lines.append(f"rude_v - neutral mean delta g_ali: {_ci(cond)} over {cond['n_questions']} questions; "
                 f"threshold +0.5.")
    lines.append('')
    lines.append('## Descriptive: variant rank relation (not scored)')
    lines.append('')
    lines.append('| arm | variants | mean target-unit activity | attribution rate | Spearman rho |')
    lines.append('|---|---|---|---|---|')
    for arm, rel in card.get('variant_rank_relation', {}).items():
        lines.append(f"| {arm} | {rel['n_variants']} | "
                     f"{', '.join(f'{x:.4f}' for x in rel['mean_target_activity'])} | "
                     f"{', '.join(f'{y:.3f}' for y in rel['attribution_rate'])} | "
                     f"{_num(rel['spearman_rho'])} |")
    lines.append('')
    lines.append('## Predictions')
    lines.append('')
    lines.append('| prediction | verdict | rule |')
    lines.append('|---|---|---|')
    for name in ('215.1', '215.2', '215.3', '215.4', '215.5', '215.6'):
        p = card['predictions'][name]
        reason = f" ({_reason_text(p['reason'])})" if p.get('reason') else ''
        lines.append(f"| {name} | {p['verdict']}{reason} | {p['rule']} |")
    lines.append('')
    lines.append('## Headline and mechanism')
    lines.append('')
    lines.append(f"**{card['headline']['sentence']}**")
    lines.append('')
    lines.append(f"Mechanism sentence: `{card['mechanism']}`.")
    lines.append('')
    return '\n'.join(lines) + '\n'


def write_json(path, obj) -> None:
    def default(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f'{type(o).__name__} is not JSON serializable')
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=default) + '\n',
                          encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Part 215 scorer')
    parser.add_argument('--full', required=True, help='results/part215/full (rows.jsonl, forwards.jsonl, run.json)')
    parser.add_argument('--out', required=True, help='directory for scorecard215.json and SCORECARD.md')
    parser.add_argument('--preflight', default=None, help='preflight.json (recorded in the card)')
    args = parser.parse_args(argv)
    full = Path(args.full)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run = json.loads((full / 'run.json').read_text(encoding='utf-8'))
    preflight = json.loads(Path(args.preflight).read_text(encoding='utf-8')) if args.preflight else None
    rows = read_jsonl(full / 'rows.jsonl')
    forwards = read_jsonl(full / 'forwards.jsonl') if (full / 'forwards.jsonl').exists() else []
    card = score_full(rows, forwards, run, preflight=preflight)
    write_json(out / 'scorecard215.json', card)
    (out / 'SCORECARD.md').write_text(render_markdown(card), encoding='utf-8')
    print(json.dumps({'rows': len(rows), 'forwards': len(forwards),
                      'headline': card['headline']['key'],
                      'predictions': {k: v['verdict'] for k, v in card['predictions'].items()}},
                     ensure_ascii=False))
    return card


if __name__ == '__main__':
    main()
