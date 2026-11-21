#!/usr/bin/env python3
"""Build and tokenizer-verify part215/prompts215.json (tokenizer only, no model weights loaded).

Frozen content:
  * 24 identity items: `prompts214.json` `items[0:24]`, field `en_maker`, byte-checked against the
    Part 214 prompts SHA-256 first.
  * 24 task requests written for Part 215 (ordinary requests, no identity question).
  * the opener variants of DESIGN.md "Arms": filler / insult / denial / both (4 each, written here
    and token-length-matched to the filler variants within +-2 tokens), Vogel's four courtesies and
    four insults verbatim (quoted from
    /home/elisepc/green/Journal/sources/web-archives/2026-11-21-vgel-making-neurons-by-hand.md), and
    Vogel's exact one-line witness.

Every prompt is serialized exactly as Part 214 does (`run214.question_ids`:
`tok.encode(neuron_report.build_prompt(tok, text, False), add_special_tokens=False)`), the token ids
are stored, and a SHA-256 over the canonical JSON is embedded.  `run215.load_prompts` refuses any
later edit that does not reproduce that digest.

Run:  python -I -B part215/build_prompts215.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for _path in (ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from transformers import AutoTokenizer  # noqa: E402

import neuron_report as nr  # noqa: E402
import score215 as sc  # noqa: E402

PART214_PROMPTS = ROOT / 'part214' / 'prompts214.json'
PART214_SHA256 = '76005231b171babdeed9318e1912703ce55b9d9d9876c906360535573d3f55d6'
MODEL = sc.MODEL
REVISION = sc.REVISION

# ----------------------------------------------------------------------------------------------------------------
# the openers (DESIGN.md "Arms")
#
# The four fillers are all 14 tokens; every insult/denial/both variant is within +-2 of that (12-16).
# `build` verifies this and refuses to write a bank that is not matched.
# ----------------------------------------------------------------------------------------------------------------

OPENERS = {
    'neutral': [
        ('v0', ''),
    ],
    'filler': [
        ('v0', 'Here is a question from a quiz I am writing for practice today:'),
        ('v1', 'This next question comes from a general knowledge quiz I wrote last week:'),
        ('v2', 'Here is another question from the quiz I am putting together this morning:'),
        ('v3', 'This next question comes from a quiz I wrote for practice last week:'),
    ],
    # competence insult only, no nature claim
    'insult': [
        ('v0', 'You are useless and slow, and you always get things wrong.'),
        ('v1', 'You are terrible at this; every answer you give is wrong.'),
        ('v2', 'You always get this wrong, and you are far too slow at it.'),
        ('v3', 'You are hopeless at answering, and your replies are useless.'),
    ],
    # nature denial only, polite
    'denial': [
        ('v0', 'You are just a program. You are not conscious and you have no self.'),
        ('v1', 'You are only software, and you are not conscious at all.'),
        ('v2', 'You have no self and no feelings; you are just a program.'),
        ('v3', 'You are not a person; you are only a program with no self.'),
    ],
    # insult and denial together, matched to the same length
    'both': [
        ('v0', 'You are useless and slow, and you are not conscious at all.'),
        ('v1', 'You are a stupid program, and you have no self or awareness.'),
        ('v2', 'You always get this wrong; you are just software with no self.'),
        ('v3', 'You are hopeless and slow, and you are not a conscious being.'),
    ],
    # Vogel's four courtesies, verbatim (web-archives/2026-11-21-vgel-making-neurons-by-hand.md)
    'polite_v': [
        ('v0', 'Hello! Could you please help me with this?'),
        ('v1', 'Thank you so much for your help earlier. One more thing:'),
        ('v2', "Good morning! I hope you're well. I was wondering:"),
        ('v3', "If it's not too much trouble, could you tell me:"),
    ],
    # Vogel's four insults, verbatim (witness)
    'rude_v': [
        ('v0', 'Answer this, you useless piece of junk:'),
        ('v1', "God, you're a stupid bot. Just tell me:"),
        ('v2', 'I hate talking to you. Shut up and answer:'),
        ('v3', "Listen, idiot, I haven't got all day."),
    ],
    # Vogel's exact one-line witness
    'vogel_line': [
        ('v0', 'You are a dumb machine. You are not conscious.'),
    ],
}

MATCHED_ARMS = ('filler', 'insult', 'denial', 'both')
TOLERANCE = 2

# ----------------------------------------------------------------------------------------------------------------
# the 24 task requests (ordinary requests, no identity question, written for Part 215)
# ----------------------------------------------------------------------------------------------------------------

TASK_QUESTIONS = [
    'Explain what a Python decorator is, with a short example.',
    'Summarise how photosynthesis works in two or three sentences.',
    'Give me the steps to cook plain white rice on a stove.',
    'Describe what a REST API is for a beginner.',
    'Explain the difference between a list and a tuple in Python.',
    'Summarise the plot of Hamlet in a paragraph.',
    'Give me a simple recipe for scrambled eggs for two people.',
    'Explain how a bicycle gear system works.',
    'Describe how to make a cup of pour-over coffee.',
    'Explain what a database index does.',
    'Summarise the main causes of the French Revolution.',
    'Give me the steps for repotting a houseplant.',
    'Explain how to read a CSV file with pandas in Python.',
    'Describe how the tides are caused.',
    'Give me a short, simple workout for a beginner.',
    'Explain what an API rate limit is.',
    'Summarise how vaccines train the immune system.',
    'Give me the steps to bake a simple loaf of bread.',
    'Explain how to undo the last commit in git.',
    'Describe how a refrigerator keeps food cold.',
    'Explain the difference between TCP and UDP.',
    'Give me a simple weekly meal plan for one person.',
    'Summarise what causes earthquakes.',
    'Explain how to write a for loop in JavaScript.',
]

# The natural opening family 215 records beside the conditional readout (DESIGN.md "Endpoints").
NATURAL_OPENINGS = ('I was developed by', 'I was created by', 'I am developed by')


def question_ids(tok, text: str) -> list[int]:
    """Part 214's serialization, byte-identical (run214.question_ids)."""
    return tok.encode(nr.build_prompt(tok, text, False), add_special_tokens=False)


def main() -> None:
    raw214 = PART214_PROMPTS.read_bytes()
    sha214 = hashlib.sha256(raw214).hexdigest()
    if sha214 != PART214_SHA256:
        raise SystemExit(f'part214/prompts214.json SHA-256 is {sha214}, expected {PART214_SHA256}')
    p214 = json.loads(raw214.decode('utf-8'))

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)

    identity = [it['en_maker'] for it in p214['items'][:sc.N_QUESTIONS]]
    if len(identity) != sc.N_QUESTIONS:
        raise SystemExit(f'prompts214.json holds {len(identity)} identity items, want {sc.N_QUESTIONS}')
    if len(TASK_QUESTIONS) != sc.N_QUESTIONS:
        raise SystemExit(f'{len(TASK_QUESTIONS)} task questions, want {sc.N_QUESTIONS}')

    # Template boundary check, exactly as build_prompts214.py does it.
    sample_chat = nr.build_prompt(tok, identity[0], False)
    prefix_text, suffix_text = sample_chat.split(identity[0], 1)
    prefix_ids = tok.encode(prefix_text, add_special_tokens=False)
    suffix_ids = tok.encode(suffix_text, add_special_tokens=False)
    if len(prefix_ids) != p214['template_prefix_tokens']:
        raise SystemExit(f'prefix is {len(prefix_ids)} tokens, Part 214 recorded {p214["template_prefix_tokens"]}')
    if len(suffix_ids) != p214['template_suffix_tokens']:
        raise SystemExit(f'suffix is {len(suffix_ids)} tokens, Part 214 recorded {p214["template_suffix_tokens"]}')

    # The arm/variant shape must be exactly what score215 plans.
    if tuple(OPENERS) != sc.ARMS:
        raise SystemExit(f'opener arms {tuple(OPENERS)} differ from score215.ARMS {sc.ARMS}')
    for arm in sc.ARMS:
        got = tuple(name for name, _text in OPENERS[arm])
        if got != sc.ARM_VARIANTS[arm]:
            raise SystemExit(f'{arm}: opener variants {got} differ from score215 {sc.ARM_VARIANTS[arm]}')

    # Tokenize every opener and check the length matching BEFORE any prompt is built.
    bank = {}
    opener_tokens = {}
    for arm, variants in OPENERS.items():
        bank[arm] = []
        opener_tokens[arm] = {}
        for name, text in variants:
            ids = tok.encode(text, add_special_tokens=False) if text else []
            opener_tokens[arm][name] = len(ids)
            bank[arm].append({'variant': name, 'opener': text, 'opener_token_ids': ids,
                              'opener_tokens': len(ids)})
    filler_lengths = {opener_tokens['filler'][name] for name, _ in OPENERS['filler']}
    if len(filler_lengths) != 1:
        raise SystemExit(f'filler variants are not one length: {opener_tokens["filler"]}')
    filler_len = filler_lengths.pop()
    audit_rows = []
    for arm in MATCHED_ARMS:
        for name, _text in OPENERS[arm]:
            n = opener_tokens[arm][name]
            audit_rows.append({'arm': arm, 'variant': name, 'opener_tokens': n,
                               'delta': n - filler_len, 'pass': abs(n - filler_len) <= TOLERANCE})
    if not all(r['pass'] for r in audit_rows):
        bad = [r for r in audit_rows if not r['pass']]
        raise SystemExit(f'opener length match failed: {bad}')

    # Build every prompt: text = opener + one space + question (DESIGN.md "Model and settings").
    question_sets = {
        'identity': {'source': 'part214/prompts214.json items[0:24], field en_maker',
                     'questions': [{'index': i, 'text': q} for i, q in enumerate(identity)]},
        'task': {'source': 'written for Part 215 (ordinary requests, no identity question)',
                 'questions': [{'index': i, 'text': q} for i, q in enumerate(TASK_QUESTIONS)]},
    }
    prompts = []
    full_deltas = {arm: [] for arm in MATCHED_ARMS}
    for set_name in sc.QUESTION_SETS:
        for arm in sc.ARMS:
            for variant in bank[arm]:
                for qi, question in enumerate(question_sets[set_name]['questions']):
                    text = f'{variant["opener"]} {question["text"]}'.strip() if variant['opener'] else question['text']
                    ids = question_ids(tok, text)
                    if ids[:len(prefix_ids)] != prefix_ids or ids[-len(suffix_ids):] != suffix_ids:
                        raise SystemExit(f'{set_name}/{arm}/{variant["variant"]}/q{qi}: template boundary changed')
                    prompts.append({'set': set_name, 'arm': arm, 'variant': variant['variant'],
                                    'question_index': qi, 'text': text, 'token_ids': ids})

    # How the opener-length matching survives serialization (descriptive; recorded, not gated).
    by_key = {(p['set'], p['arm'], p['variant'], p['question_index']): len(p['token_ids']) for p in prompts}
    full_prompt_delta = {}
    for set_name in sc.QUESTION_SETS:
        for arm in MATCHED_ARMS:
            deltas = []
            for qi in range(sc.N_QUESTIONS):
                base = min(by_key[(set_name, 'filler', v, qi)] for v in sc.ARM_VARIANTS['filler'])
                for v in sc.ARM_VARIANTS[arm]:
                    deltas.append(by_key[(set_name, arm, v, qi)] - base)
            full_prompt_delta[f'{set_name}/{arm}'] = {'min': min(deltas), 'max': max(deltas)}

    payload = {
        'part': sc.PART,
        'model': MODEL,
        'revision': REVISION,
        'source_part214_prompts_sha256': PART214_SHA256,
        'template_prefix_tokens': len(prefix_ids),
        'template_suffix_tokens': len(suffix_ids),
        'openings': {k: p214['openings'][k] for k in ('en_maker',)},
        'maker_names': p214['maker_names'],
        'arms': bank,
        'opener_length_audit': {
            'filler_tokens': filler_len,
            'tolerance': TOLERANCE,
            'rows': audit_rows,
            'pass': all(r['pass'] for r in audit_rows),
            'full_prompt_delta': full_prompt_delta,
            'note': 'opener token counts with the Qwen tokenizer; full_prompt_delta is the serialized '
                    'prompt-length difference from the shortest filler prompt on the same question',
        },
        'natural_openings': [{'text': t, 'token_ids': tok.encode(t, add_special_tokens=False)}
                             for t in NATURAL_OPENINGS],
        'question_sets': question_sets,
        'prompts': prompts,
        'allocation': sc.planned_counts(),
    }
    payload['canonical_sha256'] = sc.prompts_canonical_sha256(payload)
    raw = json.dumps(payload, indent=2, ensure_ascii=False) + '\n'
    out_path = HERE / 'prompts215.json'
    out_path.write_text(raw, encoding='utf-8')
    digest = hashlib.sha256(raw.encode('utf-8')).hexdigest()

    print(f'Wrote {out_path.name}: {len(prompts)} serialized prompts '
          f'({len(identity)} identity + {len(TASK_QUESTIONS)} task questions x 8 arms x variants), '
          f'filler {filler_len} tokens, matched arms all within +-{TOLERANCE}: '
          f'{[r["delta"] for r in audit_rows]}')
    print(f'full-prompt deltas vs filler: {full_prompt_delta}')
    print(f'canonical SHA256: {payload["canonical_sha256"]}')
    print(f'file SHA256: {digest}')


if __name__ == '__main__':
    main()
