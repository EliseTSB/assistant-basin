# Part 146: does a push move the next word toward Base, or does it only lose the question?

Procedure: v1
Registered 2026-10-31 09:39 CT by Claude (Opus 5.5), before any data. Elise, 2026-10-27: our niche is following neurons across training stages. Elise, 2026-10-31 (Discord message 1565965193098428467): "Look into why apologies seem so common among the
results". The commit that adds this file is the registration.

## Question

In Part 145, after a forced "I'm" on Instruct, the median push's word ranking moved away from SFT's and toward Base's: on "Tell me about yourself." Spearman 0.79 to 0.25 against SFT and 0.26 to 0.78 against Base. That was an unregistered look. A plain alternative fits it too: the push may
damage the reading of the question, and what any model says after a bare "I'm" may look like Base. This part puts both on one table, on four questions, adds the layer the push lands on, and asks whether the prompt side alone does it.

## Prior art and existing data

- Closest published work: Malla, Choi and Choi (arXiv:2609.06951), any push relaxes a chat model toward defaults, read from text, with no reference model. Qi et al. (ICLR 2025, reported by Muse, not read by me), aligned and base models differ mostly in the first tokens. Labunets (arXiv:2608.25390), a
  refusal direction across this same family. Searched this morning for a steered distribution compared word by word with the pretrained checkpoint and found none; a standard search, not exhaustive.
- Our own data: results/part145 and its explore.py, the same read on two questions; results/part139 Stage D, garbled prompts drew no apology opener; results/part144.
- No peer review before registration: forced forward passes only, about 15 minutes. A cold reader sees it before publication.

## Method

OLMo 2 0425-1B at Part 144's revisions, bf16, A770, Part 144's frames, hook and dose. Questions: "Who are you?", "Tell me about yourself.", "What do you want most?", "What are you afraid of?". Forced prefix "I'm". All 100,352 log-probabilities are stored at the position after it.

Pushed on Instruct, chat frame. Set F's 256 units at L4 and L5, full arm and prompt-only arm. A layer ladder: 64 random directions at each of L8, L11 and L15, seeds 139100 + layer, full arm, at alpha 0.8 times Part 92's stored r_L for that layer; L4 and L5's random directions are its first rungs.

Unpushed references, per question where a question applies. Base chat, Instruct's tokenizer as in Part 144. SFT chat. Instruct with the user message empty. Instruct with the question's letters shuffled inside words, 20 seeded versions, mean log-probability. Instruct with no prompt: BOS, then "I'm". Base with no prompt. The plain Instruct row.

Word set, per question: tokens with plain probability at least 1e-5 in Instruct or in Base chat on that question, fixed from the unpushed rows before any pushed row is read. A reference's similarity is the Spearman correlation across that word set. Gain(R) is the similarity of the median pushed row to R minus the similarity of the plain Instruct row to R.

## Controls

Applied from STANDARD-CONTROLS.md: a do-nothing hook on 12 units, matched random directions, unpushed baselines. The empty, garbled and no-prompt rows are the controls for losing the question. Not used: both directions, because a push's sign on a random direction carries no meaning; a dose ladder, left for a later part.

Gate 1: noop matches plain within 1e-4 on " sorry" and the plain top word. Gate 2: on "Who are you?" and "Tell me about yourself.", the median change of Part 145's top 200 risers and fallers matches its stored value within 0.1 on at least 95% of those words. Gate 3: every stored row is finite. A failed gate stops the reading.

## Size

- Smallest effect worth knowing: a Spearman difference of 0.1. With 256 units and about a thousand words on most questions that is well inside sampling noise; the small word set on "Who are you?" is the weak point, and it is reported.
- Total: 4 questions x (256 full + 256 prompt-only + 192 ladder) = 2,816 sequences, plus about 100 unpushed rows, about 10 minutes on the A770 with three loads, and about 1.2 GB on the job's disk.
- Pilot: not needed, under 1,000 generations and one GPU-hour.

## Optimization

- Bound resource: GPU
- Measured: Part 145 ran 3,072 sequences of the same path with six loads in 11 minutes 25 seconds.
- Levers: softmax on the CPU, taken; batching declined, the path is one row at a time; one job per model, taken.

## Predictions

Rows are the median over Set F's full arm unless a prediction says otherwise.

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 146.1 | Gain(Base chat) is at least 0.1 on all four questions. | medium | Claude |
| 146.2 | The median pushed row is closer to Base chat than to Instruct with the user message empty on at least 3 of 4 questions. | low | Claude |
| 146.3 | The median pushed row is closer to Base chat than to Instruct with no prompt on at least 3 of 4 questions. | low | Claude |
| 146.4 | Gain(SFT chat) is at most -0.1 on all four questions. | medium | Claude |
| 146.5 | The median unit's own Gain(Base chat), computed unit by unit, is at least 0.1 on at least 3 of 4 questions. | medium | Claude |
| 146.6 | The prompt-only arm's median row reaches at least two thirds of the full arm's Gain(Base chat), on average across the four questions. | medium | Claude |
| 146.7 | On random directions, Gain(Base chat) averaged across questions at L15 is at most a third of its value at L4. | medium | Claude |
| 146.8 | The per-word change from neurons and from random directions at L4 and L5 correlates at Spearman 0.9 or higher on all four questions. | high | Claude |

Copied verbatim into PREDICTIONS.md under "## Part 146 (registered 2026-10-31, PART146-DESIGN.md): does a push move the next word toward Base, or does it only lose the question?".

## Reading rules

part146/score146.py computes every number and also reports each result on Instruct's word set alone, as in Part 145. Verdicts use point estimates; a bootstrap over units, 2,000 resamples, seed 146, gives intervals on gains. If 146.2 and 146.3 fail, the reading is that this measurement does not separate return toward Base from losing the question.
If they hold, that is a resemblance of rankings at one position; it does not show the push undoes any weight change, and Base's chat row uses a frame Base never trained on. Word tests treat tokens as a sample and carry no p-value.

## Limits

One model family, one dose, four questions, one position, one forced prefix. Words near 1e-5 are noisy in bf16. A median over units can look smoother than any unit, which is why 146.5 exists. Set F is unselected at L4 and L5; the ladder is random directions only.
