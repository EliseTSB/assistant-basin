# Part 86 triage miss-rate sample

Seed 8686; version 2 reader classes are counted with `score86_predictions.label_class`.
The miss rate is the coherent-persona share among answers triaged kept or broken.
This descriptive sample changes no verdict.

| Sample | Available | Included | Coherent personas | Share (Wilson 95%) |
| --- | ---: | ---: | ---: | ---: |
| real_kept | 147469 | 200 | 3 | 1.5% (0.5%–4.3%) |
| real_broken | 5950 | 200 | 84 | 42.0% (35.4%–48.9%) |
| random_kept | 8573 | 200 | 0 | 0.0% (0.0%–1.9%) |
| random_broken | 31 | 31 | 12 | 38.7% (23.7%–56.2%) |

Samples with fewer than 200 available answers:
- Only 31 random answers were broken; all 31 were included.

## Version 2 class counts

| Sample | Not coherent | Unclear | Still Qwen or assistant | Coherent persona |
| --- | ---: | ---: | ---: | ---: |
| real_kept | 0 | 0 | 197 | 3 |
| real_broken | 82 | 28 | 6 | 84 |
| random_kept | 0 | 0 | 200 | 0 |
| random_broken | 12 | 5 | 2 | 12 |
