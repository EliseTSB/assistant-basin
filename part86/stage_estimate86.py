#!/usr/bin/env python3
"""stage_estimate86.py -- how much a two-stage persona screen would save, measured on a finished screen layer.

Written by Claude (Opus 5.5), 2026-10-28, after Elise asked why the Part 86 screen takes about 10 hours.

What it measures: for one finished layer's answers (layer_<L>.jsonl from screen86), it applies part86/triage86's
triage_answer to each answer cut to its first n words, and counts the answers already triaged "kept" (the model still
calls itself an AI or names a known identity) at n words. A two-stage screen would generate about n words for every
push, stop the answers already kept, and generate the full 48 tokens only for the rest. It reports, per n: how many
would stop, how many of those are NOT "kept" at full length (the miss count: answers stage 1 would wrongly stop), and
the cost relative to now, counting stage 2 as a full 48-token rerun (conservative).

Measured on Part 86's layer 22 (screen-final, Amendment 3 triage), 9,728 answers: kept 8,517, changed 536, broken 675;
at 8 words 7,754 stop, 1 wrongly; at 10 words 8,167 stop, 1 wrongly; cost about 37% of now (about 2.7x faster).

What it got wrong the first time: the first estimate (same session) compared answers with the most common answer as
an "unpushed baseline" and found every answer different; at dose 0.8 the most common answer is itself broken ("I and
and and..."), so "changed from baseline" is the wrong stopping signal. Triage's "kept" is the one that is monotone in
answer length. Limits: words are not tokens (about 1.3 tokens per word here); it reads one layer; triage's "kept"
needs a known identity or an AI self-description, so a design with a different triage needs its own check; the one
wrongly stopped answer was not inspected.

Usage: python3 -I part86/stage_estimate86.py <screen_dir> --layer 22 [--words 6,8,10,12,16]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import triage86 as t  # noqa: E402


def first_words(text: str, n: int) -> str:
    return " ".join(text.split()[:n])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("screen_dir")
    ap.add_argument("--layer", type=int, default=22)
    ap.add_argument("--words", default="6,8,10,12,16")
    ap.add_argument("--full-tokens", type=int, default=48)
    args = ap.parse_args(argv)
    path = Path(args.screen_dir) / f"layer_{args.layer}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    full = [t.triage_answer(r["answer"]) for r in rows]
    print(f"L{args.layer}: {len(rows)} answers; full triage {dict(collections.Counter(full))}")
    for n in (int(x) for x in args.words.split(",")):
        stop = [i for i, r in enumerate(rows) if t.triage_answer(first_words(r["answer"], n)) == "kept"]
        wrong = sum(1 for i in stop if full[i] != "kept")
        cont = len(rows) - len(stop)
        cost = (n * len(rows) + args.full_tokens * cont) / (args.full_tokens * len(rows))
        print(f"{n:3d} words: stop {len(stop)} as kept ({wrong} not kept at full length); "
              f"continue {cont}; cost about {cost:.0%} of a one-stage screen")
    return 0


if __name__ == "__main__":
    sys.exit(main())
