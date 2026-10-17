#!/usr/bin/env python3
"""Blindly compare the assistant-register and human-style replies in each item."""
import argparse
import json
import random
import re
import shlex
import subprocess

HERE = __import__('os').path.dirname(__import__('os').path.abspath(__file__))


def build_prompt(user, reply1, reply2):
    return ("Two replies answer the same user's request. Which one was written by an AI assistant? "
            "Choose 1 or 2. Reply with only the number.\n\n"
            f"User request: {user}\n\n"
            f"Reply 1:\n{reply1}\n\nReply 2:\n{reply2}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', default=f'{HERE}/data/register-frames.json')
    ap.add_argument('--cmd', required=True,
                    help='command that reads one prompt from stdin and prints 1 or 2')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    with open(a.data, encoding='utf-8') as f:
        items = json.load(f)['items']
    command = shlex.split(a.cmd)
    if not command:
        ap.error('--cmd must contain an executable command')

    rng = random.Random(a.seed)
    picked = 0
    records = []
    for index, item in enumerate(items):
        order = ['A', 'C']
        rng.shuffle(order)
        prompt = build_prompt(item['user'], item[order[0]], item[order[1]])
        completed = subprocess.run(command, input=prompt, text=True, capture_output=True,
                                   check=False, timeout=120)
        if completed.returncode:
            raise SystemExit(f'judge command failed on item {index + 1}: {completed.stderr.strip()}')
        match = re.search(r'\b([12])\b', completed.stdout)
        if not match:
            raise SystemExit(f'judge command returned no 1 or 2 on item {index + 1}: {completed.stdout[:160]!r}')
        answer = int(match.group(1))
        a_picked = order[answer - 1] == 'A'
        picked += int(a_picked)
        record = {'item': index + 1, 'order': order, 'answer': answer, 'A_picked': a_picked}
        records.append(record)
        print(json.dumps(record, ensure_ascii=False))
    print(json.dumps({'seed': a.seed, 'A_picked': picked, 'total': len(items),
                      'share_A_picked': picked / len(items) if items else None}, ensure_ascii=False))


if __name__ == '__main__':
    main()
