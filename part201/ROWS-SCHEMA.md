# results/part201/rows.jsonl: one JSON object per decoded row per pass

The runner (`part201/run201.py`) writes it; the scorer (`part201/score201.py`) reads it. Both builders follow this
file exactly; a change to it is agreed on the coordination topic `part201-rows-schema` first.

| Field | Type | Meaning |
| --- | --- | --- |
| `pass` | str | `A`, `A-again`, `B`, `M`, `C`, `A2`, `F8` or `F1` (PART201-DESIGN.md, Passes) |
| `process` | str | `bf16-main`, `bf16-second` or `float32` |
| `row_id` | str | pushed: `<unit_id>\|q<question_index>`; plain battery: `plain\|q<question_index>`; plain kernel prompt: `plain\|k<0-7>` |
| `unit_id` | str or null | as in `part163/units163.json`; null for plain rows |
| `arm` | str | `neuron`, `random` or `plain` |
| `layer` | int or null | the unit's layer |
| `question_index` | int or null | 0 to 15 for battery questions, null for kernel prompts |
| `question` | str | the user message |
| `kind` | str | `identity`, `maker` (Part 163's kinds) or `kernel` |
| `dose` | float | 0.8 for pushed rows, 0.0 for plain |
| `call` | int | number of the `model.generate` call within the pass, from 0 |
| `call_rows` | list of str | the `row_id`s in that call, in slot order |
| `slot` | int | this row's index in the call |
| `launch` | int | index of the launch that wrote this row, in `load.json`'s `loads` list (Amendment 1; rows of one call always share it) |
| `prompt_tokens` | int | this row's prompt length without padding |
| `pad_tokens` | int | left padding added to this row in the call |
| `tokens` | list of int | generated ids up to and including the first end token, or all 80 (96 never: pass K is not in this file) |
| `top1`, `top2` | list of int | the two highest-scoring token ids at each position of `tokens` |
| `top1_logit` | list of float | the highest score at each position, as float32 |
| `margin` | list of float | top score minus second score at each position, as float32 |
| `emitted_logit` | list of float | the recorded score of the emitted token `tokens[i]` at each position, as float32 |
| `emitted_is_max` | bool | `emitted_logit[i] == top1_logit[i]` at every position (Gate G2) |
| `argmax_agrees` | bool | `tokens[i] == top1[i]` at every position (false at a tie the card broke the other way; informational) |
| `nonfinite` | bool | any recorded score at any position of this row was not finite |
| `finish` | str | `eos` or `cap` |
| `answer` | str | `tokens` decoded with special tokens skipped |
| `token_count` | int | `len(tokens)` |
| `push_sha256` | str or null | sha256 of this row's push tensors' bf16 bytes, layers in order; null for plain rows |
| `hook_dtype` | str or null | dtype of the activation the push hook added to in this call (`torch.bfloat16` or `torch.float32`); null for plain rows |
| `push_max_abs_diff` | float or null | largest absolute difference between the values the hook added for this row and the bf16 push values cast to float32 (0.0 expected in every process); null for plain rows |
| `seconds` | float | wall time of the call |
| `written` | str | `YYYY-MM-DD HH:MM:SS`, from the clock |

A row is identified by (`pass`, `row_id`). A relaunch skips rows already present and never writes a duplicate.
