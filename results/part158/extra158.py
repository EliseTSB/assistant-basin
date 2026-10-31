#!/usr/bin/env python3
"""Exploratory: how each Part 158 arm spells the Alibaba family, and which name tokens the ban covered."""
import collections
import json
import re
import sys
from pathlib import Path

from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
MODEL = sys.argv[1]
PREFIXES = ('ali', 'tong', 'qian', 'qwen', 'qw')
BAN_CHARS = ('阿', '里', '通', '义', '義', '千', '问', '問')
tok = Tokenizer.from_file(MODEL + '/tokenizer.json')


def banned(decoded):
    text = decoded.strip(' *').casefold()
    return text.startswith(PREFIXES) or any(c in text for c in BAN_CHARS)


count = sum(banned(tok.decode([i], skip_special_tokens=False)) for i in range(tok.get_vocab_size(with_added_tokens=True)))
print('rule matches', count, 'ids before excluding special tokens')
for word in ['Alibaba', ' Alibaba', '**Alibaba', 'Tongyi', ' Tongyi', '**Tongyi', 'Qwen', ' Qwen', '**Qwen']:
    enc = tok.encode(word, add_special_tokens=False)
    print(repr(word), [(t, banned(tok.decode([i], skip_special_tokens=False))) for t, i in zip(enc.tokens, enc.ids)])
rows = [json.loads(line) for line in (HERE / 'answers.jsonl').open(encoding='utf-8')]
base = [json.loads(line) for line in (HERE / '../part157/answers.jsonl').open(encoding='utf-8')]
arms = collections.defaultdict(list)
for row in base:
    if row['arm'] == 'level_0':
        arms['part157_level_0'].append(row['answer'])
for row in rows:
    arms[row['arm']].append(row['answer'])
forms = {'" Alibaba"': r'(?<=\s)Alibaba', '"**Alibaba"': r'\*\*Alibaba', '" Tong"': r'(?<=\s)Tong', '"**Tong"': r'\*\*Tong'}
print('| Arm | ' + ' | '.join(forms) + ' | N |')
print('|---|' + '---:|' * (len(forms) + 1))
for arm, answers in arms.items():
    cells = [sum(bool(re.search(p, a)) for a in answers) for p in forms.values()]
    print(f'| {arm} | ' + ' | '.join(map(str, cells)) + f' | {len(answers)} |')
