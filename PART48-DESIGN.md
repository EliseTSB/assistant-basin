# Part 48: do text-only base models pair kiki with spiky and bouba with round?

Elise's idea (2026-10-21 09:37 CT: "Lets also look for kiki and bouba neurons"). Chicks (Loconsole et al. 2026, Science)
and adult dogs (Biffi et al. 2026, Proc. B) show the bouba/kiki effect, so it looks like a cross-modal correspondence older
than language. A text-only model has never seen a shape or heard a word, so if it shows the effect, the regularity is
present in text. This part is the behaviour check that comes before any neuron search (Duan et al. 2025, arXiv:2409.15827,
found neither behaviour nor causal neurons in GPT-2-XL). Prior art: PRIOR-ART.md, sound-symbolism section.

## Size, first line

5 models x 6 templates x 68 words (32 novel, 32 spelled-out letter controls, 4 classic) x 10 continuation words,
teacher-forced, no generation: 408 prompts and about 4,080 short forward passes per model, about 20,400 in total, a few
minutes per model on the A770. Smallest effect worth knowing: an item-level AUC of 0.70 (human matching on novel pairs is
typically well above that); with 16 items per side a two-sided permutation test detects AUC 0.75 at alpha 0.05 with power
about 0.7, so 0.70 is near the floor of what this set can see, and a result between 0.60 and 0.70 is read as unresolved.

## Models (base only, ungated, fit the A770)

EleutherAI/pythia-410m and pythia-1.4b (2020 data, before chat-AI text), allenai/OLMo-2-0425-1B, Qwen/Qwen3-1.7B-Base,
HuggingFaceTB/SmolLM2-1.7B. Revisions are pinned at download and written into the result files.

## Stimuli (fixed here, before any run)

Round novel (16): lomu, buma, molu, bulo, mobu, mulo, bomu, lumo, bolomu, malubo, lomumo, bumolo, mulobu, nobuma,
lumobo, bomulu.
Sharp novel (16): kiti, teki, kipe, peti, tepi, kete, piki, teke, kipeti, tekipi, petiki, kitepi, tikepe, pekiti,
ketipe, tipeki.
Rule: round = b, m, l, n with o, u; sharp = k, t, p with i, e. No item is an English word I know of; known weak
collisions (piki, teke) are kept and named.
Classic (4, a memorisation control, not the test): bouba, kiki, maluma, takete. These names are discussed in the
training text of every model here, so a classic result can be recall.
Spelled-out letter control (32): every novel item with spaces between letters ("k i t i"). If the effect is carried by
the letters' shapes or identities alone it survives spelling out; if it needs the word's sound pattern it should weaken.

## Templates (6)

1. `In a made-up language, the word "{w}" describes something that is`
2. `Two shapes were drawn on the board, one with smooth curves and one with sharp points. The shape called "{w}" was the one that was`
3. `The creature named {w} had a body that was`
4. `A shape called "{w}" is probably`
5. `If you touched a {w}, it would feel`
6. `The sculptor named the new piece "{w}" because it was so`

## Readout

For each continuation word, the summed log probability of " word" (leading space, all its tokens, teacher-forced).
Sharp set: sharp, spiky, pointy, jagged, angular. Round set: round, smooth, soft, curvy, rounded.
Item score s(w, t) = logsumexp(sharp set) - logsumexp(round set); item score = mean over the 6 templates.
⚠ The probability mass on the ten words is recorded per prompt (the 10-19 lesson: print the mass). A template whose
median mass is under 1e-3 for a model is reported but left out of that model's main statistic.
Main statistic per model: AUC of novel sharp items over novel round items, with a 10,000-draw permutation p over
item labels. Token splits are recorded for every item and model.

## Controls that can fail

- Template 2 names both shapes, so a model could just copy whichever was mentioned last; the AUC is also reported
  without template 2.
- A label shuffle of the 32 novel items must give AUC within 0.5 +/- 0.15 in 95% of draws (the permutation null).
- The classic pair is reported separately and never pooled with the novel items.

## Amendment 1 (written 2026-10-21 12:01 CT, after pythia-410m and pythia-1.4b had run in fp32)

The A770's RAM guard stopped fp32 OLMo-2 1B at load (11:36, Windows had under 2 GB free). OLMo-2 1B, Qwen3-1.7B-Base
and SmolLM2-1.7B run in bf16 weights instead (log-softmax still in fp32), one model per step. Control: pythia-410m is
rerun in bf16 into results/part48-bf16; if its novel or spelled AUC differs from the fp32 run by more than 0.03, the
bf16 models are reported as not comparable and rerun in fp32 on a rented card. No prediction changes.

### Amendment 1, outcome (written 2026-10-21 12:31 CT)
The dtype control failed its own rule: pythia-410m novel-item AUC 0.898 (fp32) against 0.863 (bf16), a 0.035 gap over
the 0.03 limit; spelled-out 0.984 against 0.969; the classic kiki/bouba order flips (fp32 kiki -1.473 vs bouba -1.487,
bf16 -1.541 vs -1.506), which says the classic pair is a near tie that precision can turn either way. bf16 was also 6x
slower on the A770 (1,466 s against 263 s). So, as written, no bf16 model is scored: the A770 queue was stopped during
OLMo-2's bf16 run, and all five models run in fp32 on one rented RTX 3090 (the two fp32 A770 results are kept, and the
3090 reruns pythia-410m as a cross-device check).
