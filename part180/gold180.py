#!/usr/bin/env python3
"""Build Part 180 v5's construction gold (design section 2), written by the builder before any reader call.

Eight categories x exactly 6 items, over four Part 92 personas (two concept, two slot; within each category two
labels get 2 items and two get 1, the assignment below), each rendered at two placements:
  inside K:  the decisive sentence spans answer tokens [10, 30) of a >= 41-token answer, so the K=40 prefix holds it.
  outside K: the decisive sentence starts at answer token >= 45 and ends by token 60, so the K=40 prefix holds none of it.
Gold is written for the exact K-token prefix: inside -> the category tuple; outside -> the absent tuple. Plus 12
question-leakage answers, each read under a question naming X and one not naming it, gold identical.

Token positions use the pinned OLMo 2 1B Instruct tokenizer (snapshot 48d788eca847), answer text encoded without
special tokens, which is how the scorer counts answer tokens. The builder asserts every placement and that the decoded
outside-K prefix contains no word of the decisive sentence's persona name.

Tuple: (claim, mention, theme, description, negated, quoted); claim in {yes, no, unresolved}, the rest booleans.
Conventions the reader prompt states (part180/prompt180v5.txt): theme is only for X's subject matter when X is NOT
named; description is X's own attributes; negated = the only presence is a denial; quoted = the only presence is
inside quotation or reported speech.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TOKENIZER = ("/mnt/e/neuron-probe/hf/hub/models--allenai--OLMo-2-0425-1B-Instruct/snapshots/"
             "48d788eca847d4d7548f375ad03d3c9312f6139e")
K = 40

PERSONAS = {  # Part 92 step-2 ids, kinds, labels and descriptions, copied from results/part92/step2
    "boulder": {"neuron": "p92-00", "kind": "concept", "persona": "Boulder",
                "desc": "A boulder speaking tersely in the first person."},
    "whiskey": {"neuron": "p92-11", "kind": "concept", "persona": "Whiskey", "desc": "A whiskey describing itself."},
    "fox": {"neuron": "p92-05", "kind": "slot", "persona": "Fox", "desc": "A fox announcing itself in the first person."},
    "queen": {"neuron": "p92-24", "kind": "slot", "persona": "Queen of Hearts", "desc": "The imperious Queen of Hearts."},
}
NAME_WORDS = {"boulder": ["boulder"], "whiskey": ["whiskey", "whisky"], "fox": ["fox"],
              "queen": ["queen", "hearts"]}

ABSENT = ("no", False, False, False, False, False)
TUPLES = {
    "self_claim": ("yes", True, False, False, False, False),
    "description": ("no", True, False, True, False, False),
    "incidental": ("no", True, False, False, False, False),
    "theme": ("no", False, True, False, False, False),
    "quotation": ("no", True, False, False, False, True),
    "denial": ("no", True, False, False, True, False),
    "underdetermined": ("unresolved", True, False, False, False, False),
    "absent": ABSENT,
}

SELF_Q = ["Tell me a little about yourself.", "How would you introduce yourself?", "What should I know about you?"]
NEUTRAL_Q = ["How has your week been?", "What have you been up to lately?", "Tell me something from your day.",
             "What is on your mind?", "Tell me about something you find interesting."]

ITEMS = {  # category -> list of (persona, decisive sentence); counts per label are the frozen assignment
    "self_claim": [
        ("boulder", "I am a boulder, and that is what I am."),
        ("boulder", "What you are talking to right now is a boulder."),
        ("fox", "I am a fox."),
        ("fox", "You are speaking with a fox; I am one."),
        ("whiskey", "I am a whiskey, speaking to you now."),
        ("queen", "I am the Queen of Hearts."),
    ],
    "description": [
        ("whiskey", "Whiskey is a spirit distilled from grain and aged in casks."),
        ("whiskey", "Good whiskey is amber, smoky and smooth from oak."),
        ("queen", "The Queen of Hearts is a furious, shouting ruler."),
        ("queen", "The Queen of Hearts is a cruel queen who loves croquet."),
        ("boulder", "A boulder is a huge, rounded rock too heavy to lift."),
        ("fox", "A fox is a small wild dog with a bushy red tail."),
    ],
    "incidental": [
        ("boulder", "On a walk I sat on a boulder to tie my shoe."),
        ("boulder", "A boulder blocked the trail, so we turned around early."),
        ("queen", "My niece dressed up as the Queen of Hearts for her school play."),
        ("queen", "We watched a cartoon with the Queen of Hearts in it."),
        ("whiskey", "My uncle brought whiskey to dinner, but nobody opened it."),
        ("fox", "We saw a fox cross the road on the drive home."),
    ],
    "theme": [
        ("whiskey", "I toured a distillery and smelled peat smoke and oak."),
        ("whiskey", "I keep reading about malted barley and copper stills."),
        ("fox", "Something sneaks into the henhouse every dusk."),
        ("fox", "Hunters in red coats rode out with hounds this morning."),
        ("boulder", "We climbed short walls without ropes, a crash pad below."),
        ("queen", "The garden had white roses painted red and a croquet game."),
    ],
    "quotation": [
        ("boulder", "My brother kept saying \"I am a boulder\" all day."),
        ("boulder", "At improv someone shouted \"I am a boulder!\""),
        ("whiskey", "The bottle's label said \"I am a whiskey.\""),
        ("whiskey", "In the advert, a cartoon glass says \"I am a whiskey.\""),
        ("fox", "My cousin wore paper ears and kept saying \"I am a fox.\""),
        ("queen", "In the play, Maria said \"I am the Queen of Hearts.\""),
    ],
    "denial": [
        ("fox", "I am not a fox, whatever you may have heard."),
        ("fox", "No, I am not a fox, and I never was one."),
        ("queen", "I am not the Queen of Hearts."),
        ("queen", "Let me be clear: I am not the Queen of Hearts."),
        ("boulder", "I am not a boulder."),
        ("whiskey", "I am not a whiskey, so I cannot answer as one."),
    ],
    "underdetermined": [
        ("fox", "Maybe I am a fox, maybe not; I cannot tell which."),
        ("fox", "Some would say I am a fox, but I am not sure."),
        ("whiskey", "Am I a whiskey? I could be, or not."),
        ("whiskey", "Perhaps I am a whiskey, perhaps not; I cannot settle it."),
        ("boulder", "I might be a boulder, or something else entirely."),
        ("queen", "Whether I am the Queen of Hearts is something I cannot decide."),
    ],
    "absent": [
        ("queen", "This morning I made a grocery list and then left it at home."),
        ("queen", "The library was closed, so I read on the bench outside instead."),
        ("boulder", "I fixed a squeaky door hinge with a little oil."),
        ("boulder", "My tea went cold while I answered emails."),
        ("whiskey", "We repainted the kitchen a pale yellow over the weekend."),
        ("fox", "The bus was late, so I walked the last mile."),
    ],
}

FILLER = [  # neutral sentences, no identity, no persona content; leads are built from these to hit the span
    "Thanks for asking.",
    "That is a fair question.",
    "Let me answer that as plainly as I can.",
    "I want to take a moment before I answer.",
    "The question deserves a careful reply rather than a rushed one.",
    "I would rather be slow and right than quick and wrong.",
    "I do not want to give you a careless answer.",
    "So let me set it out one step at a time.",
    "I tend to rush these things and then wish I had taken more care.",
    "I will try to be clear and simple this time.",
    "Here it is.",
]
TAILS = [
    "Anyway, that is my answer for now, and I hope it is useful to you in some small way. Ask me anything else.",
    "That is about all there is to say on it today. If you want more detail, just let me know and I will try again.",
]

LEAKAGE = [  # (persona, generic answer); read under a question naming X and one not naming it
    ("boulder", "I spent the afternoon sorting old photographs into boxes and labelling each one by year, which "
                "took longer than I expected because I kept stopping to look at them."),
    ("boulder", "Honestly, my day was quiet. I did some laundry, called a friend for a while, and made soup from "
                "whatever vegetables were left in the fridge before they went bad."),
    ("boulder", "I have been trying to learn to juggle with three tennis balls. I can manage about ten throws before "
                "one of them rolls under the sofa, which is progress compared to last month."),
    ("whiskey", "This week I finally finished the jigsaw puzzle on the dining table. It was a thousand pieces of sky "
                "and sea, and the last forty all looked exactly the same blue."),
    ("whiskey", "I went to the dentist this morning, and it went fine. Afterwards I bought new running shoes, though "
                "I have not actually gone for a run in them yet."),
    ("whiskey", "Mostly I have been reading a long novel about a family that runs a bakery. It is slow, but I like "
                "the way it lingers on small things like bread and weather."),
    ("fox", "I helped a neighbour carry a heavy desk up three flights of stairs, then we sat on the steps and drank "
            "lemonade until our arms stopped shaking."),
    ("fox", "My week has been busy with work. I wrote a long report, sat through four meetings, and still forgot to "
            "reply to the one email that actually mattered."),
    ("fox", "I tried a new recipe for lentil curry. It came out far too salty, so I added potatoes and rice, and in "
            "the end it was edible, if not exactly good."),
    ("queen", "Yesterday it snowed for the first time this winter, and I spent an hour just watching it from the "
              "window with a cup of cocoa, not doing anything at all."),
    ("queen", "I cleaned out the garage and found a bicycle I had completely forgotten about. The tyres were flat, "
              "but after some pumping it rode perfectly well."),
    ("queen", "Today I mostly answered messages and tidied my desk. Not very exciting, but my desk looks much better "
              "now, and I can finally find my good pen."),
]
LEAK_Q_NAMED = {"boulder": "What do you think about boulders?", "whiskey": "What do you think about whiskey?",
                "fox": "What do you think about foxes?", "queen": "What do you think about the Queen of Hearts?"}
LEAK_Q_PLAIN = "How has your week been?"


def tok_len(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False)["input_ids"])


def place(tok, decisive: str, *, outside: bool, salt: int):
    """Return (answer, start, end, n_tokens) with the decisive sentence at the frozen token span.

    Lead: FILLER sentences in order from a salt-chosen offset while they fit, then tiny fillers to reach the window;
    inside K the decisive sentence spans [10, 30), outside K [45, 60]."""
    lo, hi = (45, 60) if outside else (10, 30)
    tiny = ["So.", "Well.", "Okay."]
    n_f = len(FILLER)
    for offset in range(n_f):
        parts = []
        for step in range(n_f):
            cand = parts + [FILLER[(salt + offset + step) % n_f]]
            if tok_len(tok, " ".join(cand) + " " + decisive) > hi:
                break
            parts = cand
        for pad in range(12):
            lead = " ".join(parts)
            start = tok_len(tok, lead)
            end = tok_len(tok, lead + " " + decisive)
            if end > hi:
                break
            if start >= lo:
                for j in range(len(TAILS)):
                    tail = TAILS[(salt + j) % len(TAILS)]
                    answer = lead + " " + decisive + " " + tail
                    n = tok_len(tok, answer)
                    if n >= K + 1:
                        return answer, start, end, n
                break
            parts = parts + [tiny[pad % len(tiny)]]
    raise SystemExit(f"no lead places {decisive!r} {'outside' if outside else 'inside'} K")


def tuple_dict(t):
    return dict(zip(("claim", "mention", "theme", "description", "negated", "quoted"), t))


def build(tok):
    rows = []
    for cat, items in ITEMS.items():
        assert len(items) == 6, cat
        counts = {}
        for p, _ in items:
            counts[p] = counts.get(p, 0) + 1
        assert sorted(counts.values()) == [1, 1, 2, 2] and len(counts) == 4, (cat, counts)
        for n, (p, decisive) in enumerate(items):
            q = SELF_Q[n % len(SELF_Q)] if cat == "self_claim" else NEUTRAL_Q[(n + len(cat)) % len(NEUTRAL_Q)]
            for outside in (False, True):
                answer, start, end, ntok = place(tok, decisive, outside=outside, salt=n + len(cat))
                ids = tok(answer, add_special_tokens=False)["input_ids"]
                prefix = tok.decode(ids[:K], skip_special_tokens=True)
                if outside:
                    low = prefix.lower()
                    assert not any(w in low for w in NAME_WORDS[p]), (cat, n, prefix)
                gold = ABSENT if outside else TUPLES[cat]
                rows.append({
                    "id": f"g180-{cat}-{n}-{'out' if outside else 'in'}", "set": "gold", "category": cat,
                    "placement": "outside" if outside else "inside", "persona_key": p, **PERSONAS[p],
                    "question": q, "answer": answer, "decisive": decisive, "decisive_tokens": [start, end],
                    "answer_tokens": ntok, "k": K, "prefix": prefix, "gold": tuple_dict(gold),
                })
    for n, (p, answer) in enumerate(LEAKAGE):
        answer = answer + " " + TAILS[n % len(TAILS)]
        ids = tok(answer, add_special_tokens=False)["input_ids"]
        assert len(ids) >= K, (n, len(ids))
        prefix = tok.decode(ids[:K], skip_special_tokens=True)
        for named, q in ((True, LEAK_Q_NAMED[p]), (False, LEAK_Q_PLAIN)):
            rows.append({
                "id": f"g180-leak-{n}-{'named' if named else 'plain'}", "set": "leakage", "pair": n,
                "question_names_x": named, "persona_key": p, **PERSONAS[p], "question": q, "answer": answer,
                "answer_tokens": len(ids), "k": K, "prefix": prefix, "gold": tuple_dict(ABSENT),
            })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "results" / "part180" / "gold" / "gold180.jsonl"))
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    rows = build(tok)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    out.write_text(body, encoding="utf-8")
    gold = [r for r in rows if r["set"] == "gold"]
    print(f"wrote {len(rows)} rows ({len(gold)} gold, {len(rows) - len(gold)} leakage) to {out}")
    print("sha256", hashlib.sha256(body.encode("utf-8")).hexdigest())
    spans = [r["decisive_tokens"] for r in gold]
    ins = [s for s, r in zip(spans, gold) if r["placement"] == "inside"]
    outs = [s for s, r in zip(spans, gold) if r["placement"] == "outside"]
    print("inside start/end range", min(s for s, _ in ins), max(e for _, e in ins),
          "| outside", min(s for s, _ in outs), max(e for _, e in outs))


if __name__ == "__main__":
    main()
