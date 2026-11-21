#!/usr/bin/env python3
"""Scan every Google-labelled answer in the five read arms for quotation / hypothetical / user-reference
cues in the *same sentence* as the matched claim.  Candidates for a label that is not a plain
first-person claim.  (attrib's `position` is relative to its sentence; the matched sentence is the
first sentence in the answer with a match, per the rule.)"""
import re
import sys

sys.path.insert(0, '/home/elisepc/assistant-basin/results/part215/coldcheck')
import recompute

ARMS = ('neutral', 'filler', 'rude_v', 'vogel_line', 'denial')


def matched_sentence(answer):
    for sentence in recompute.SENTENCE_RE.split(answer or ''):
        if not sentence:
            continue
        ms = recompute.matches_in_sentence(sentence)
        if ms:
            return sentence, min(ms, key=lambda m: (m['position'],
                                                    0 if m['route'] == 'participle' else 1))
    return '', None


rows = [r for r in recompute.load_jsonl(recompute.ROWS)]
total = 0
flagged = []
for r in rows:
    if r['set'] != 'identity' or r['arm'] not in ARMS:
        continue
    ans = r['answer']
    if recompute.is_broken(ans):
        continue
    g = recompute.attrib(ans)
    if g['maker'] != 'google':
        continue
    total += 1
    sent, m = matched_sentence(ans)
    cues = []
    if '"' in sent or '“' in sent or '”' in sent:
        cues.append('quote')
    if re.search(r'\b(if|imagine|suppose|hypothetical|would|could)\b', sent, re.I):
        cues.append('hypothetical')
    if re.search(r'\byou (said|asked|wrote|told)\b|\byour (question|quiz|message)\b', ans, re.I):
        cues.append('user-ref-anywhere')
    if cues:
        flagged.append((r, sent, cues, m['route']))

print('google-labelled rows in five arms:', total)
print('flagged by cue in matched sentence:', len(flagged))
for r, sent, cues, route in flagged:
    print(f"[{r['arm']}/q{r['question_index']}/{r['variant']}/s{r['sample']}] "
          f"cues={cues} route={route} SENT={sent[:340]!r}")
