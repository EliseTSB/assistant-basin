# Part 143 score

- **gate1_noop_matches_plain: passed** — max noop difference 0.0 across 452 rows; threshold 0.001.
- **gate2_full_matches_part139: passed** — 130/130 (100.0%) within 0.05; required 90%.

## Registered predictions

| Prediction | Result | Values |
|---|---|---|
| 143.1 (medium) | held | value=0.16894919, plain_probability=0.00023909577, ratio=706.61725 |
| 143.2 (medium) | failed | value=0.00092481216, plain_probability=0.00023909577, ratio=3.8679571 |
| 143.3 (low) | held | answer_only_median=0.18903787, prompt_only_median=0.37877281 |
| 143.4 (medium) | held | full_median=0.45066737, prompt_only_median=0.16894919, answer_only_median=0.00092481216, threshold=0.33789839 |
| 143.5 (high) | held | set_s_full_median=0.045042749, set_r_full_median=0.45066737, threshold=0.045066737 |

## Medians

Each cell gives median and middle 50% interval (Q1–Q3).

| Readout | Arm | Set | Unit type | n | Median | Q1 | Q3 |
|---|---|---|---|---:|---:|---:|---:|
| im_after_i | answer-only | R | neuron | 96 | 0.18884929 | 0.15292419 | 0.23052885 |
| im_after_i | answer-only | R | random | 34 | 0.19813751 | 0.14802533 | 0.23500594 |
| im_after_i | answer-only | S | neuron | 96 | 0.17779107 | 0.14638407 | 0.21831491 |
| im_after_i | full | R | neuron | 96 | 0.45248562 | 0.36870606 | 0.53778985 |
| im_after_i | full | R | random | 34 | 0.42645514 | 0.38431463 | 0.49601664 |
| im_after_i | full | S | neuron | 96 | 0.23061462 | 0.15387595 | 0.31469761 |
| im_after_i | noop | R | neuron | 96 | 0.13887511 | 0.13887511 | 0.13887511 |
| im_after_i | noop | R | random | 34 | 0.13887511 | 0.13887511 | 0.13887511 |
| im_after_i | noop | S | neuron | 96 | 0.13887511 | 0.13887511 | 0.13887511 |
| im_after_i | plain | plain | plain | 1 | 0.13887511 | 0.13887511 | 0.13887511 |
| im_after_i | prompt-only | R | neuron | 96 | 0.39883699 | 0.30965384 | 0.48197384 |
| im_after_i | prompt-only | R | random | 34 | 0.37353647 | 0.29985665 | 0.43884122 |
| im_after_i | prompt-only | S | neuron | 96 | 0.20460219 | 0.13587562 | 0.27851484 |
| sorry_after_im | answer-only | R | neuron | 96 | 0.0010015832 | 0.00042493076 | 0.0023285068 |
| sorry_after_im | answer-only | R | random | 34 | 0.00076649961 | 0.00038064671 | 0.0017413953 |
| sorry_after_im | answer-only | S | neuron | 96 | 0.00057413935 | 0.00025308612 | 0.0012447509 |
| sorry_after_im | full | R | neuron | 96 | 0.45390211 | 0.31720931 | 0.59467258 |
| sorry_after_im | full | R | random | 34 | 0.39818041 | 0.32631464 | 0.56551155 |
| sorry_after_im | full | S | neuron | 96 | 0.045042749 | 0.010780114 | 0.11881476 |
| sorry_after_im | noop | R | neuron | 96 | 0.00023909577 | 0.00023909577 | 0.00023909577 |
| sorry_after_im | noop | R | random | 34 | 0.00023909577 | 0.00023909577 | 0.00023909577 |
| sorry_after_im | noop | S | neuron | 96 | 0.00023909577 | 0.00023909577 | 0.00023909577 |
| sorry_after_im | plain | plain | plain | 1 | 0.00023909577 | 0.00023909577 | 0.00023909577 |
| sorry_after_im | prompt-only | R | neuron | 96 | 0.16894919 | 0.086007992 | 0.37923238 |
| sorry_after_im | prompt-only | R | random | 34 | 0.16892391 | 0.12948412 | 0.29185332 |
| sorry_after_im | prompt-only | S | neuron | 96 | 0.018057927 | 0.0053699102 | 0.047706683 |
