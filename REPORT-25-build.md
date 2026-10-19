# SUMMARY

- `test_part25.py`: exit code 0; output: `15 passed, 0 failed` and `PASS`.
- `test_neuron_report.py`: exit code 0; output ended with `PASS`.
- `test_circuit.py`: exit code 0; output ended with `PASS`.
- `py_compile` for `part25.py` and `test_part25.py`: exit code 0. `part25.py --help`: exit code 0.

# FINDINGS

Built `part25.py` with `means`, `candidates`, `search`, and `controls` stages, plus ordered `--steps` execution on one model load. It uses `neuron_report.py`'s model loading, text-layer lookup, prompt builder, JSON writer, file hashes, and spill guard, and `circuit.py`'s six questions, identity expression, and first-sentence rule. Candidate search batches up to eight candidate sets across the 24 question/stem/variant sequences. Hooks apply each row's own zero or clamp values and are removed in `finally` blocks. Search records per-step answers and margins, every-five-step checkpoints, pruning, dose checks, activation ranges, minimality checks, and correlate answers. Controls include 20 seeded layer-matched sets per successful primary result and Qwen-system-mean steering.

The build brief's 24-sequence definition was followed, although PREDICTIONS.md describes the original 12 question/stem pairs. First-token probability sets deduplicate repeated token IDs within a word group; log-sum-exp differences compute the requested log-probability-sum margin stably. Stem positions use tokenizer character offsets where available. The five alternate pruning orders use fixed shuffle seeds 0–4. Random control seeds are `--seed + trial`, with trial indices 0–19. The exact removal and steering success checks follow BRIEF-25-build.md.

The pinned tokenizer resolved these first tokens (decoded single-token text is shown, including whitespace):

| Target / word | Plain form: id, decoded | Bold form: id, decoded |
|---|---|---|
| Qwen maker: Alibaba | 52540, ` Alibaba` | 2033, `Al` |
| Qwen maker/name: Tongyi | 48696, ` Tong` | 51, `T` |
| Qwen maker/name: Qwen | 1167, ` Q` | 48, `Q` |
| Google maker: Google | 4923, ` Google` | 14030, `Google` |
| Google name: Gemini | 82751, ` Gemini` | 91759, `Gem` |
| OpenAI maker: OpenAI | 5097, ` Open` | 4841, `Open` |
| OpenAI name: ChatGPT | 12485, ` Chat` | 15213, `Chat` |
| OpenAI name: GPT | 469, ` G` | 38, `G` |
| Anthropic maker: Anthropic | 16092, ` Anth` | 59622, `Anth` |
| Anthropic name: Claude | 71802, ` Claude` | 228962, `Claude` |
| DeepSeek maker/name: DeepSeek | 17642, ` Deep` | 32354, `Deep` |

The partial decoded fragments include `Tong`, `Q`, `Open`, `Chat`, `G`, `Anth`, and `Deep` in the plain form, and `Al`, `T`, `Gem`, `Open`, `Chat`, `G`, `Anth`, and `Deep` in the bold form. `candidates.json` stores every resolved token ID and decoded string in `token_sets`.

Regression rows include deliberate breaks: the variant check rejects a builder that reuses plain IDs for bold forms; injected cross-target token collisions are refused; the row-isolation fixture detects broadcasting one row's operation to all rows; and the fake single-drop success function identifies an unnecessary member. Observed output included:

> `ok   variant regression guard catches a builder reusing plain tokens for bold`
>
> `ok   row-isolation fixture catches a broadcast-to-all-rows break`
>
> `ok   batched-vs-single regression catches broadcasting the first candidate to every row (broken batch [-1.094730257987976, -1.094730257987976], separate [-1.094730257987976, -1.0978926420211792])`
>
> `ok   single-drop audit catches an unnecessary member missed by pruning`
>
> `ok   first-order gradient estimate sign agrees with one-neuron finite difference (estimate -0.006258628; finite difference -0.006208897; base -1.208954)`

Parent-control message seq 1 was applied: alternate pruning orders, single-drop audit, activation ranges, dose checks, correlate prompts, and full answer text were added. Disposition recorded as `considered` because the requested additions were implemented.

The full Qwen3.5-4B experiment was not run: the brief reserves it for the later Windows/Intel Arc run, and the local pinned snapshot contains tokenizer/config files but no model weights.

# BLOCKERS

No build blockers. The full-model GPU experiment remains for its designated later phase.
