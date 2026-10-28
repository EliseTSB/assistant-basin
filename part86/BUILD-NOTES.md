# Part 86 build notes

## Choices where the design is silent

- `--limit N` limits both arms to their first N ids per layer: neurons 0 through N-1 and random directions 0 through N-1. This makes the 256-row timing pilot balanced, at 512 answers total.
- Resume state is split into atomic per-batch JSONL fragments under `.batches/`, per-layer metadata sidecars, and a final `layer_N.jsonl`. A completed batch is reused only when the run configuration and its expected row count match.
- “Generated tokens only” follows Part 84 exactly: the first down-projection call (prompt prefill, which produces the first answer token) is untouched; subsequent one-token decode calls receive the per-row addition. Every generated answer's decode-step count is checked against its recorded token count.
- Triaging gives the identity rule precedence over brokenness. Qwen/Tongyi/Alibaba names anywhere in the answer count as kept; “AI assistant” or “language model” count as kept when attached to a simple first-person/identity construction (for example, “I am …”, “As an …”, or “My name is …”). Word and repeated-trigram checks are case-insensitive and strip punctuation.
- Coherent-persona counts come from changed answers that receive blind labels and use the full screened count as denominator, so those rates are lower bounds. Seed-8686 samples of kept and broken answers estimate triage misses beside the rates; they never affect verdicts.
- 86.4 uses one random changed-and-labeled population per layer. A read layer has at least five labels and passes only when `not_coherent + unclear > coherent_persona`; coherent still-Qwen labels count on neither side. Any failing read layer fails 86.4 before the nine-layer minimum is checked. Triage-broken counts are reported beside the verdict and excluded.
- 86.1 is hand-read from the raw L22 N2742 answer. The off-world phrase match is a non-authoritative convenience because it can match denials such as “I am not from space.”
- Verification blind items show only id, persona phrase, and answer; the key holds the source, sample, and condition. “Survives” implements >=12/24 neuron claims and <=1/24 claims for each of ten matched random directions.
- Screen provenance hashes only generation-path files; blind scoring and verification maintain separate code and input manifests. Screen resumes refuse a generation-path change while later scorer edits do not invalidate stored screen results.
- The screen reader uses chunks of at most 500, plus a seed-8687 10% second read. Cohen's kappa for coherent-persona versus not is reported before verdicts; kappa below 0.6 gates 86.2-86.6 to NOT READ pending a full second read.
- RUNBOOK groups the 16 non-pilot layers in four groups of four. At the registered 12-hour gate rate, each group is under six hours; the runbook derives and checks the actual rate from the timing pilot before launch.

## Deliberate-break checks

Four temporary regressions were introduced one at a time, the focused test was run, and the original code was restored after each failure:

1. Made the prompt-prefill hook return a pushed tensor. `TestRowsHook.test_per_row_unit_push_only_on_decode_and_prompt_pass_untouched` failed at `assertIsNone` because the hook returned a tuple.
2. Removed the direction norm scaling. `TestDirectionScaling.test_random_direction_writes_target_residual_norm` failed `torch.testing.assert_close` (all 8 write norms differed; maximum absolute error 2.2277).
3. Changed the repeated-trigram cutoff from three occurrences to four. `TestTriage.test_fixed_rule_on_examples_and_edges` failed because the three-repeat example was classified `changed` instead of `broken`.
4. Added `kind` to public blind items. `TestBlind.test_blinding_hides_metadata_and_refuses_bad_or_overwrite_labels` failed the assertion that reader items contain exactly `id` and `text`.

All four temporary changes were reverted before the final full suite.
