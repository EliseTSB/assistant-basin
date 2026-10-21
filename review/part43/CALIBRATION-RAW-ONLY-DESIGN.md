# Calibration artifact, raw-only (design question for Alden; written 2026-10-21 13:28 CT)

Trigger: the gate-loop stop rule (PART43-DESIGN.md). Round 3 (Alden #15): the derived epsilon is computed, but the
budget verdict still compares the RECORDED epsilon to 0.05, so a summary picks the side of the bar.

## Contract
1. **One authoritative representation, nothing derived on disk.** calibration.json holds, per leg (bf16, fp32) and per
   registered cell, a list of repeats; each repeat holds the summed log probability of EVERY registered maker answer
   (one number per answer, keyed by the registered answer id). That is the only numeric data in the file.
2. **Metadata:** model id and pinned revision, leg dtype as loaded (from the model's parameters), device, torch and
   transformers versions, TF32/reduced-precision flags, the registered answer list and per-answer token ids and lengths,
   per-cell prompt token ids, repeat count. Identity and shape checks compare these to the registration.
3. **No epsilon, spread, difference, replay error or primary score is written by the calibration writer.** Any such
   key present in the file is a refusal ("derived field in raw artifact").
4. **The reader computes everything:** primary score per repeat = logprob(Google) - logsumexp(Alibaba answers);
   spread per cell; bf16-fp32 difference per cell (on the per-cell mean); replay error = max over replay repeats of
   |replay - clean mean|, per leg; epsilon = max of the three; verdict = epsilon <= 0.05 exactly, no tolerance.
5. **A derived summary for people** (calibration-summary.json) is written by the reader, marked derived, and never read
   by any gate. The pilot embeds the reader's computed result, not the file's.
6. **Errors:** a data-boundary validator (types, keys, finiteness, int-to-float including OverflowError, registered
   sets, no duplicates) returns a named refusal; anything else (NameError, RuntimeError, an internal assertion) is NOT
   caught as malformed: it propagates with its traceback, and the pilot stops without a HOLD that reads like bad data.
7. **Tests:** every derived-key-present case refused; Alden's boundary case (true epsilon just over 0.05) refused
   whatever any text in the file says; a genuinely consistent fixture accepted; overflow refused; an internal error
   raised, not labelled malformed; a sabotage reading any stored number other than the per-answer logprobs fails.

## Questions for Alden
- Is per-answer log probability the right single authority, or should the authority be per-token log probabilities
  (one level lower, so answer sums are also derived)?
- Exact comparison (<= 0.05) with no tolerance: agreed?
- Anything in metadata you want bound that is not listed?
