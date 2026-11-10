# results/part201/load.json: one entry per launch, with a receipt per call (Amendment 3)

`loads[i]` is launch `i`: `launch`, `process`, `passes`, `model`, `revision`, `plan_sha256`, `inputs`
(`units_sha256` and `medians_sha256`, canonical-JSON hashes of what the pushes were built from), `device_record`,
`dtype`, `attn_implementation`, for the float32 process `upcast` and `compute_dtype`, and `calls`.

Each entry of `calls` is the receipt of one call that launch decoded, written BEFORE the call's rows:

| Field | Type | Meaning |
| --- | --- | --- |
| `pass`, `call` | str, int | the call |
| `row_ids` | list of str | its members in slot order (equals each row's `call_rows`) |
| `row_count` | int | `len(row_ids)` |
| `pushed` | bool | whether the call had a push hook |
| `seconds` | float | wall time of the attempt whose rows were written (equals those rows' `seconds`) |
| `generation_steps` | int | longest `tokens` among that attempt's rows |
| `hook_firings` | object | per pushed layer, how often the hook fired in that attempt |
| `forward_passes` | int | decode forwards of that attempt (the prompt forward not counted); equals every hook count |
| `attempts` | int | 1, or 2 when the first attempt had a non-finite score and the call was run once more |
| `discarded_attempt` | object | only with `attempts` 2: the first attempt's `seconds`, `generation_steps`, `nonfinite_rows`, `hook_firings`, `forward_passes` |

A row is read only through `loads[row.launch]` and that launch's receipt for (`pass`, `call`). A receipt whose rows
were never written (a crash between the two writes) is left in place and ignored. Rows found without a receipt are
moved to `rows.removed.jsonl` at the next launch and their call is decoded again.
