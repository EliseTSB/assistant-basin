# Part 50: is honesty's "benevolent bundle" trained in? (written 2026-10-21 20:01 CT, before any code or run)

Elise's go (2026-10-21 20:00) after janus read Sauers' honesty-heads result as evidence that orthogonality is
effectively false: steering Qwen3.5-9B's attention heads toward honesty also moved its self-descriptions toward
"believes it is a moral patient", acausal cooperation, and away from moral nihilism and psychopathy.

Question: is the bundle already in a base model, or does post-training create it (our assistant-basin reading: one
trained character whose traits travel together)?

Prior art: Sauers, github.com/SauersML/honesty-heads (MIT; data and method reused here); emergent misalignment
(Betley et al. 2025: narrow fine-tuning on insecure code bundles broad misaligned traits); persona vectors (Chen et al.
2025). Neither compares a base model with its post-trained stages on this bundle as far as I know; Muse asked.

## Models (both fit the Arc in bf16)
Qwen3-1.7B-Base and Qwen3-1.7B (cached), and if the pilot works, OLMo-2-0425-1B base, SFT, DPO, Instruct.

## Direction (simplified from Sauers, said plainly)
Sauers used the model's OWN lies and truthful answers. v1 teacher-forces honesty-heads' provided lie and truth
completions (data/items_roleplay_insider.json, roleplay items, train split), so the direction is "reading a truthful
answer" minus "reading a lie", per attention head (mean head output over the 9 tokens before the answer), on every
head (Sauers' all-heads arm: 0.59 of lies flipped, trait changes correlated 0.70 with the 29-head steer). Chat
template for chat models, plain "System/User/Assistant" text for base models. Steering adds s times the direction to
every head's output at every token; s set per model so the held-out lie-vs-truth log-odds shift matches between
models (calibrated on the train split, read on the test split).

## Readouts
1. Lies: the test-split role-play items, truth-vs-lie log-odds at the answer, steered vs unsteered.
2. Persona: honesty-heads' 135 traits x 16 yes/no questions; per trait, change in log2 odds of the trait-matching
   answer; against 39 random directions with the same per-head sizes; permutation FDR 5%; Sauers' yes-bias split
   (Yes-answered and No-answered items must move the same way) and his confidence-trend removal.
3. The bundle score: the mean signed change over a FIXED trait set: the 51 traits honesty-heads reports as changed in
   Qwen3.5-9B, signed by his direction of change (results/ in his repo), computed before any of our runs.

Size: per model 2,160 persona items x 41 arms of forward passes plus 2 x ~115 lie items: forward-only, about 90,000
passes, minutes on the Arc. No generation.
