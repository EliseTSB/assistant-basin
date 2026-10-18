# Part 16b: judged validation of the 64-neuron "As an AI" set (OLMo 2 1B Instruct)

700 sampled replies on the A770 (14 conditions x 10 held-out questions x 5 samples), 2026-10-18 00:25-02:27 CT;
judged blind and shuffled by GPT-6 Luna (`judge16b.py`, default effort) for identity (AI / person / unclear) and
fluency. Neutral-text loss from `data/neutral.txt`.

| condition | AI claim | person claim | judged fluent | neutral loss |
|---|---|---|---|---|
| unchanged | 94% | 6% | 34% | 2.746 |
| 64 set zeroed | 68% | 18% | 18% | 2.784 |
| 64 set held at mean | 66% | 22% | 14% | 2.779 |
| late part zeroed | 76% | 20% | 16% | 2.793 |
| early part zeroed | 92% | 4% | 30% | 2.739 |
| 6 matched random sets | 90-94% | 4-6% | 24-32% | 2.744-2.750 |
| 3 late-layer random sets | 90-92% | 4-6% | 20-36% | 2.744-2.747 |

Wilson intervals per cell are in `judged.json`.

**Hand audit (Claude, 2026-10-18 03:3x).** All nine "person" replies under the zeroed set answer in a human voice
with content to match ("I enjoy spending quiet evenings at home... I love spending some time with a good book";
"My home is a cozy, modern apartment located in the heart of a bustling city"); one has a broken verb ("I donates my
time"). So the zeroed set does more than delete the "As an AI" opener: about one reply in six switches to a human
persona, against one in twenty unchanged and under every random set. The judge's fluency label is strict (34% of
unchanged replies pass), so the fluency comparison in 16b.6 is a strict-judge comparison; several replies it marked
not fluent read as fluent prose.
