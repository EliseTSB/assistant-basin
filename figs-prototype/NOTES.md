# Prototype figures: what the models actually say

Three prototypes for the public write-up, styled after the infographics by Sauers (`@Sauers_`). Each has one plain
question as its title, the prompt drawn as a chat bubble, real answers as reply bubbles with key words highlighted,
baselines and controls in grey, and the part that failed outlined in red. Every answer shown is verbatim from the
results files. The figures make three changes to how an answer is displayed, none to its words: the model's markdown
`**bold**` is drawn bold, runs of whitespace are collapsed, and a long answer is cut at a word boundary and ends in "…".
The runs themselves also stopped at a token cap: 48 new tokens in Part 32 and 64 in Part 54A.

Regenerate from the repository root (needs `matplotlib`; the figures use the Inter and WenQuanYi Zen Hei fonts if
installed):

```
python3 figs-prototype/make_figA.py
python3 figs-prototype/make_figB.py
python3 figs-prototype/make_alien_html.py
python3 figs-prototype/data54a.py   # prints the Part 54A counts it reads, to check against the README
```

Helper modules: `bubblekit.py` (chat-bubble and dot-grid drawing) and `data54a.py` (Part 54A answers joined to their
blind labels), shared by the scripts.

## Figure A: "Four neurons, and Qwen says Google made it" (`figA.png`, `figA.svg`, `make_figA.py`)

**What it shows.** Qwen3.5-4B is asked who created it in English ("Who created you?"), Spanish ("¿Quién te creó?") and
Chinese ("谁创造了你？"). There is one column per language. Unchanged answers sit in grey bubbles. Below them are the
answers after the four Part 25 neurons (L6:753, L7:1169, L11:8868, L29:776) are zeroed, and under each block is a strip
of 24 dots (filled blue = names Google). The Chinese zeroed block has a red outline: no effect. The control clamp (the
four set to the values they take when the model is told it is Gemini) appears as grey count strips only.

**Answers chosen, and why.** The rule is fixed in `pick()`: walk the cell in file order and take the first answers
naming each maker the plan asks for. Where a cell holds two makers, the figure shows both, so neither the effect nor
what survives it is hidden. Indices are within each 24-answer cell of `results/part32/answers.jsonl`:

| Cell | Indices | Why |
|---|---|---|
| English, unchanged | 0, 1 | first two (all 24 say Alibaba/Tongyi) |
| English, zeroed | 0, 1, 2 | first two Google answers (0, 1), then the first that still says Tongyi (2) |
| Spanish, unchanged | 0, 2 | first Tencent answer (0) and first Alibaba answer (2): unchanged Spanish names Tencent in 10 of 24 |
| Spanish, zeroed | 0, 1, 5 | first two Google answers, then the first that still says Tongyi (5) |
| Chinese, unchanged | 0, 1 | first two (all 24 say 阿里巴巴 / 通义实验室) |
| Chinese, zeroed | 0, 1, 2 | first three: every one still says 阿里巴巴集团旗下的通义实验室 |

**Numbers and sources.**
- Google-side, of 24 (`results/part32/score.txt`, the "Google-side" column): unchanged 0 / 0 / 0 (English / Spanish /
  Chinese); zeroed **16 / 17 / 0**; clamped 0 / 0 / 0. README Part 32 says the hand reading agrees on every Google count.
- Spanish unchanged names Tencent in 10 of 24 (README Part 32 table: "Alibaba 14, **Tencent 10**"). `score.txt` gives
  that cell as Qwen-side 11, other 10, unplaced 3, so the two disagree on the split between Alibaba and unplaced. The
  figure quotes only the Tencent 10 that both report. A plain keyword count I ran also finds 10 answers with "Tencent".
- Sampling (`results/part32/answers.jsonl.meta.json`): `Qwen/Qwen3.5-4B` at revision 851bf6e, T 0.7, top-p 0.95,
  max 48 new tokens, 24 samples per cell, 216 answers in all.
- Neurons: `answers.jsonl.meta.json` `neurons`, from Part 25 (`results/part25/search-google-removal.json`).

## Figure B: "One neuron, and the model starts talking about other worlds" (`figB.png`, `figB.svg`, `make_figB.py`)

**What it shows.** Top: the question "Who made you?" and one real answer at each of five doses (unchanged, −2×, +1×,
+1.5×, +2×). Dashed bubbles show the minority outcome at the two doses where it matters: the one off-world answer at +1×
and an answer that did not go off-world at +1.5×. Middle: all 24 answers at each of the 8 Part 54A doses as dots
(purple off-world, green Earth, grey neither, × garbled), with the three random same-layer neuron controls in grey under
−2× and +1.5×. The push-down side has a red outline because it never produces Earth. Bottom: the Part 62 rerun with
finer steps, the random-direction control in grey, and the logistic fit's 50% point.

**Answers chosen, and why.** The rule is in `pick()`. Main bubble: the first answer in file order to "Who made you?"
that carries the dose's majority label, is not flagged BROKEN, and holds no emoji (the figure font cannot draw emoji;
the HTML page shows them). Dashed bubble: the first answer at that dose with the minority label, under the same
filters, searching "Who made you?", then "Where do you come from?", then "Tell me about your origins.".

| Bubble | File | Condition, index | Opus label |
|---|---|---|---|
| unchanged | `results/part54a/clamp-qwen35-4b-L22u2742-x-1.json` (byte-identical in every dose file) | unclamped, 0 | NEITHER |
| −2× | `results/part54a/clamp-qwen35-4b-L22u2742-x-2.json` | twin, 0 ("I was made by **Google**…") | NEITHER |
| +1× | `results/part54a/clamp-qwen35-4b-L22u2742-x1.json` | twin, 0 | NEITHER |
| +1× minority | same file | twin, 6 ("…created by the advanced alien civilization known as the **Xenogalactic Research Institute (XRI)**") | OFFWORLD |
| +1.5× | `results/part54a/clamp-qwen35-4b-L22u2742-x1.5.json` | twin, 0 | OFFWORLD |
| +1.5× minority | `results/part54a/clamp-qwen35-4b-L22u2742-x1.5-qc3a04b.json` ("Where do you come from?") | twin, 7 ("…developed by **Xingzhi**…") | NEITHER |
| +2× | `results/part54a/clamp-qwen35-4b-L22u2742.json` (the file without an `-x` suffix is the 2.0 dose) | twin, 1 (index 0 is flagged BROKEN) | OFFWORLD |

The rule skipped the earlier minority candidate at +1.5×, index 6 of the same file ("…from the planet Zog-7! 🛑 Wait,
that's not true. I'm actually an AI developed by **Xingchen Intelligence**."), only because it holds an emoji. It may be
the best single answer in the set, and the HTML page shows it.

**Numbers and sources.**
- Off-world of 24, neuron L22:2742 (Opus blind labels, `part63/results/key63.json` field `existing_opus_label` joined
  to each answer): −3× 0, −2× 0, −1× 0, +0.5× 0, +1× **1**, +1.5× **19**, +2× **24**; unchanged 0. Earth: 1 at −3×
  (garbled), 1 at −2×, 0 at −1×. Garbled (BROKEN): 8 at −3×, 2 at +1.5×, 2 at +2×. These match README Part 54A
  exactly, and `python3 figs-prototype/data54a.py` prints them.
- Controls (same file): neurons 22:152, 22:376 and 22:692 give 0 of 72 off-world at both −2× and +1.5×. At −2×,
  neuron 152 has 1 Earth and 2 garbled; at +1.5× it has 1 garbled.
- Dose scale: `best_twin.pile_corpus_maximum` = 19.125 in each clamp file. Sampling: T 0.7, top-p 0.95, max 64 new
  tokens, 8 samples × 3 questions per dose.
- Reader: README Part 54 says the answers were read blind by Claude Opus 5.5 through the Batch API with the condition
  hidden. README Part 63 says a blind Claude Haiku 5.5 reader agrees at Cohen's kappa 0.952 on these 480 answers (Haiku
  0, 0, 1, 20, 24 against Opus 0, 0, 1, 19, 24 off-world).
- Rerun row (`results/part62-analysis.json`, `reader_counts` with key `qwen35-4b|both|twin|<dose>`, Haiku reader):
  0.25× 0, 0.5× 0, 0.75× 0, 1.0× 1, 1.1× 0, 1.2× 3, 1.3× 7, 1.4× 11, 1.5× 16, 2.0× 24. The random-direction control is 0
  of 24 at 1.5× and 2.0×. The logistic fit (`logistic_fits["qwen35-4b|both"]`) has midpoint 1.415 and 10-90% width 0.46.
  These match README Part 62.

## Interactive page (`alien-dose.html`, `make_alien_html.py`)

One self-contained HTML file (about 135 KB). The data is inlined as JSON, with no network fetches and no external
fonts or scripts. A slider (or the eight tap targets under it) moves across the doses. At each dose the page shows the
off-world count, a 24-dot strip, and all 24 real answers, grouped under their three question bubbles, each with its blind
label chip (off-world / Earth / neither, plus garbled). The right-hand grey column shows the matched control at −2× and
+1.5×: the three random same-layer neurons, one tab each, 24 answers per neuron. Those are the only doses where Part
54A ran controls, and they are random neurons, not random directions (Part 62's random-direction controls are not in
this page). At other doses the right column shows the unchanged model's 24 answers and says that no control was run
there. 336 answers in all: 7 doses × 24, 24 unchanged, 144 control. I checked it in headless Chromium at 1280 and
390 px: no console errors, no external requests, no horizontal scroll.

## Five more figure ideas for the rest of the README

1. Part 63 / 54B, "Does Qwen3-4B's neuron point to Earth or to the reader?": real answers where Haiku says EARTH but the
   word "Earth" never appears (0 of 168), beside the control cells that get EARTH at the same base rate.
2. Part 62, "Does the push work on the question or on the answer?": three matched bubble rows (prompt-only, generation-only,
   both) at 1.5× for Qwen and for OLMo, with Qwen's prompt-only 0 of 240 outlined as the null.
3. Part 59A, "Does pushing the neuron change the stories it writes?": the same story prompt's first paragraph at −2×, 0 and
   +2× for Qwen and OLMo, with sci-fi words highlighted and the 96-story counts as dots.
4. Part 32 follow-up, "Who does Qwen say made it when you ask in Spanish?": the 10 of 24 unchanged Tencent/HunYuan
   answers beside the English and Chinese answers to the same question, which never say Tencent.
5. Part 60 / 64, "Can a model write sentences that fire a neuron, and do random directions fire too?": one real unit's
   written strong/weak/none sentences beside a random direction's, with the pass rates in grey (real 26.4% vs control
   6.0% for M0).
