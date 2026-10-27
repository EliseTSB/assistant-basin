# Part 78b build notes

Implemented the Part 78 runner shape with the registered `k1_makers` condition and ten layer-22 control pushes. It reuses Part 78's Part 26 Pile-maximum reader, column-norm function, and push hook, plus Part 66 row seeds, rendering, sampling, and maker-zero hooks. The scorer reuses Part 78's `classify` rules and writes per-condition counts, all 24 per-question Google counts (including zeroes), guard matches, and 78b.1–78b.4 verdicts. All four predictions become `NOT READ` with `holds: null` when 78b.1 fails.

## Verification

Test command: `/home/elisepc/neuron-probe/.venv/bin/python -m pytest -q part78b`

Output: `9 passed in 2.97s`.

Mutation checks deliberately broke each tested contract; each corresponding pytest invocation went red as expected:

- Plan count: changed 8 samples per held-out question to 7; `test_plan_count_and_per_condition_counts` failed.
- Norm rule: changed the 2.0 multiplier to 1.0; `test_set_value_per_unit_follows_column_norm_rule` failed.
- Registered list: changed unit 8115 to 8114; import-time control-list assertion failed.
- NOT READ gate: changed `holds: null` to `holds: false`; `test_verdict_logic_and_not_read_gate` failed.

CPU smoke command: `/home/elisepc/neuron-probe/.venv/bin/python -I part78b/smoke_run78b.py /tmp/part78b-smoke-worker01`

Smoke produced 2,376 rows, with 216 rows in each of the 11 conditions. Hook evidence is total hook calls and maximum set-value error per condition:

| Condition | Calls | Max absolute difference |
|---|---:|---:|
| k1_makers | 0 (no push) | 0.0 |
| k_u1499 | 1544 | 0.0 |
| k_u4154 | 1493 | 0.0 |
| k_u5572 | 1125 | 0.0 |
| k_u5606 | 1765 | 0.0 |
| k_u5640 | 1653 | 0.0 |
| k_u6210 | 1558 | 0.0 |
| k_u8115 | 1189 | 0.0 |
| k_u8606 | 1578 | 0.0 |
| k_u8920 | 1618 | 0.0 |
| k_u9209 | 1920 | 0.0 |

The random tiny model's 78b.1 failed, so its 78b.2–78b.4 verdicts were correctly `NOT READ`. The smoke run used the real `results/part26` records and CPU only.

## Copy to GPU workspace

Copy these new files into `/mnt/c/Users/alexa/neuron-probe-gpu/p66basin/part78b/`:

- `part78b.py`
- `score78b.py`
- `test_part78b.py`
- `smoke_run78b.py`
- `BUILD-NOTES.md`

Read-only listing confirmed `/mnt/c/Users/alexa/neuron-probe-gpu/p66basin/part78/part78.py`, `results/part26/source.json`, and `results/part26/source.npz` already exist there.
