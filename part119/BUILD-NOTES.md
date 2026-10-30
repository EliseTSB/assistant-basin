# Part 119 build notes

This build adds deterministic answer rows, a resumable OLMo runner, a CPU-only scorer, a blind DS/Luna label client, twelve known reader examples, and fixture tests. `rows119.py` selects the 39 Haiku-classified Part 92 units, compares their kinds with `kind_comparison.units`, and asserts the 21 agreed slots, 8 agreed concepts, and 10 disputed units. The answer row inventory is 2,302. Path 2 has 2,964 directed real/control rows for the main stage.

## Design decisions

- The accepted parent control message `design-fold-1` changed the registered design while this build was underway. Its disposition was `considered` and applied. The implementation follows the revised `PART119-DESIGN.md`: Path 1 stores `both_top`, `Ma`, and `Mb`; Path 2 writes `interaction.jsonl`; r50 and shift2 were removed; and predictions and gates use the revised numbering and thresholds.
- A generated token list ends immediately before its first EOS or UNK. Padding is excluded, including when the tokenizer aliases pad to EOS. This makes every saved token an answer token and lets paths stop at the end marker.
- `p_A` is zero-based and searched within the first 24 saved tokens. If no differing token exists, the ladder row retains the registered alpha list and has empty `tops` and `margins`; there is no valid position or token pair to score, so the ten ladder forwards are skipped. Path 2 rows for that first unit are retained with empty `positions`.
- Part 118 stores decoded answer text, not token IDs, and the scorer has no tokenizer argument. Gate 2 therefore compares the first 100 decoded characters exactly, as the revised design specifies.
- Path 1 logit forwards use float32 logits and accelerator argmax. The runner caps forward microbatches at 8 rows to bound vocabulary-output memory; `--batch` still controls greedy generation. The second token margin is found with a masked argmax, not top-k.
- Path 2 obtains each pass's ten highest token IDs with ten repeated argmax operations on the device. It transfers only the selected-token slices to CPU for centered norms. The full-vocabulary softmax TV is reduced on device. Zero-norm terms are omitted from the log-linear fit because their logarithms are undefined; the scorer records fit and support coverage.
- A pilot measures one supplied batch size per run. To compare the registered 64 and 128 sizes without resume skipping the second timing, run pilot twice with separate output directories, then use the faster measured size for main.
- For an exact margin tie, winner-take-all misses. Reader rows that remain errors are preserved, reported, and excluded from the `both` denominator. Reader requests reuse `label92b._transport`, its DS 4,000-token cap and thinking configuration, and its Luna route. No transport was called in this build.
- The scorer's class summaries include agreed DS/Haiku groups and Part 92's Haiku-only groups. Kind predictions are withheld if the Gate 2 agreed-concept subgate fails or the direction differs under the two classifications. Leave-one-concept-unit-out coverage is included for 119.5 and 119.10.

## Known-example personas

The twelve unrelated persona names are: Canal lock tender; Sourdough baker; Weather balloon tracker; Theater usher; Ceramic tile setter; Harbor pilot; Kite designer; Clock restorer; Orchard grafting specialist; Mountain rescue dispatcher; Glass harmonica player; Railway signal maintainer; Book conservator; Iceberg photographer; Apiculture inspector; Subway map engraver; Marshland ecologist; Umbrella repairer; Lighthouse lens polisher; Mushroom forager; Puppet stage carpenter; Glacier surveyor; Telescope mirror maker; and Marathon course marshal. None matches a persona name in the 53-row Part 92 source list.

## Verification limits

The specified pytest command passed, including the three `test_torch_*` checks on a tiny random OLMo2 model on CPU. Torch 2.14.1+cpu, Transformers 5.20.0, NumPy, SciPy, and pytest were available. This builder did not load the pinned model, run GPU work, or call a reader transport. The harness's A770 timing, visual pilot review, and live-reader calls remain run-time work.

One design edge is underspecified: it asks for ten flip-ladder forwards for each unit while also allowing `p_A` to be null. The implementation reports the null and skips those forwards because `t_A`, `t_0`, and a prefix position do not exist. No registered wording or threshold was changed.
