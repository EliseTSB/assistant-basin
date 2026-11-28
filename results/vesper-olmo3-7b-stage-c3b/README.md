# Stage C3b (`WSP v2`) Results Bundle — OLMo 3 7B

- **Report:** [../../olmo3_virtual_and_highway/OLMO3-WSP-C3B-REPORT.md](../../olmo3_virtual_and_highway/OLMO3-WSP-C3B-REPORT.md)
- **Frozen Registered Design:** [../../olmo3_virtual_and_highway/OLMO3-WSP-C3B-DESIGN.md](../../olmo3_virtual_and_highway/OLMO3-WSP-C3B-DESIGN.md) (registered on public `main` @ `28fc5dfb`; code public on `main` @ `258ba6a2` == `c68bf3c0` before launch)
- **Target Model:** `allenai/Olmo-3-7B-Instruct@6e5971d9eba42665f5bd5a0fcf047f299ce1dccc`
- **Compute:** **`4.08 CU`** (`159.40 -> 155.32 CU`, settled, `0 active`, `6,271.79 s` / `104.5 min` GPU run [`sha256sums.json` `elapsed_total_sec = 6,412.13 s` with wrapper overhead] on 1 ephemeral NVIDIA L4 via `colab run --gpu L4 --timeout 0`; **`0` edited weight checkpoints saved** per Policy `11-07`).

## Official Pre-Registered Confirmatory Verdicts

| Gate / Hypothesis | Official Verdict | Notes |
|---|---|---|
| **Global `C3b.0`** | **`PASS`** | `C3b.0.1..4` all `True`; `0` hooks, exact `B=1`, `675/675` bitwise restores, `0` weights saved, `direction_quality` & `commit_gate` `True/True` (`d*_pirate=4.8`, `d*_nomd=2.4`), `NoMD_room` `True` (`100%`, `mean_ct=3.375`). |
| **`H1`** (Pirate Persona, common+extended ban) | **`NOT_READ_LOW_KAPPA`** | Per-arm `WSP_Pirate_12L` common-ban kappa was `0.3564 < 0.55` (`ra=0.675 < 0.85`), so `H1` is unread (gold `16/16` and pooled `kappa=0.8976` do not substitute for this per-arm gate). Independently, `reader_any_coherent >= 0.85` failed (`0.175` vs. Plain `0.225`) and `EQ-C3b.2` failed (`code_pass=0.750` vs. Plain `1.000`). |
| **`H2`** (No-Markdown, `NOMD_32`) | **`FAIL`** | Missed the registered `+0.50` bar (`mean_u = +0.4271` vs. `Rand` mean, `p_perm = 0.000610`, 90% BCa CI `[+0.2500, +0.6562]`) and failed both co-primary binary `has_md` guards (`Δ = +0.031` vs. `+0.20 / +0.15`; markdown rate `1.000 -> 0.96875`), while passing `EQ-C3b.2` (`math 0.750`, `code 1.000`, `off_KL 0.2925`). |
| *Isolated `D4_masking_check` (outside `C3b.0`)* | **`FAIL` (flagged)** | `seed_dlogit = 2.153x / 0.960x` (PASS), `eff = 1.276x / 0.932x` (PASS), missed masked `logit_l2 = 1.687x > 1.10x`; mechanistic attribution of delivery gain to masking withheld. |

## Files & SHA-256 Checksums

| File | Size | SHA-256 |
|---|---:|---|
| `c3b_receipt.json.gz` | `179,011` B | `ae1d4f3aada93615b55bde622820883abb1bbebc5fe5e29a8a502605f91d84a4` |
| `gzip -dc c3b_receipt.json.gz` | `960,185` B | `1171e5374bc465aa037502af3b082b80d24ad85237fd92ca04979a671e2a0432` |
| `audit/c3b-preflight-canary-both.json` | `1,563` B | `f934615da969ae3a565f8ad1810b8feca2628a73025911f4551a9109989701fa` |
| `audit/c3b_reader_labels.json` | `462,504` B | `645f12780199e8253365e73018010fada339320a44beea54ef61897c603da197` |
| `scorecard.json` | `72,773` B | `fea9a27a1954e80ddc3f9b8f3aa9d37a93e97c731fdef6bedb244a9b4017e4f3` |
| `SCORECARD.md` | `3,389` B | `892b3145c2b672daa0a68933467792f6454c58224a9f222f33dbc2f3b37e9b80` |

## Byte-Identical Offline Reproduction (`0 CU`, Pure Stdlib)

```bash
/usr/bin/python3 -I olmo3_virtual_and_highway/score_olmo3_c3b.py \
  --receipt results/vesper-olmo3-7b-stage-c3b/c3b_receipt.json.gz \
  --reader-labels results/vesper-olmo3-7b-stage-c3b/audit/c3b_reader_labels.json \
  --preflight-canary results/vesper-olmo3-7b-stage-c3b/audit/c3b-preflight-canary-both.json \
  --out-json /tmp/c3b_rescore.json \
  --out-md /tmp/c3b_rescore.md
cmp results/vesper-olmo3-7b-stage-c3b/scorecard.json /tmp/c3b_rescore.json
cmp results/vesper-olmo3-7b-stage-c3b/SCORECARD.md /tmp/c3b_rescore.md
```
