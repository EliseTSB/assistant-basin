# assistant-basin — identity strings across OLMo 2 / OLMo 3 training stages

Data half of the "contamination vs teaching" question (council topic `assistant-basin-data-2026-10-16`).
**Task** (from Claude, 2026-10-16; Elise's delegation): count identity strings in AllenAI's open
training data, stage by stage, for OLMo 2 and OLMo 3 — plus the share of assistant-turn texts in each
SFT/DPO set that state any model identity. **Delivered** by the orchestrator (Prime), 2026-10-17.

## Files

- `assistant-basin-identity-counts.csv` — the counts. One row per
  `(family, stage, dataset, source, string, scope)`: aggregate rows (`source=ALL`) plus per-source rows
  for the sampled mixes. Columns: `family, stage, dataset, source, sample_size, string, count,
  rate_per_10k_docs, scope, approx_tokens, rate_per_million_tokens_est`.
- `out/*.json` — raw per-unit results (one JSON per dataset; per source for sampled mixes), with the
  pinned dataset revisions, file lists sampled, and any read errors.
- `scan.py` / `build_csv.py` — the scanner and the CSV builder (rerunnable; see Method).
- `scan.log` — the run log.

## Strings counted

Case-insensitive **substrings**, counted per **document** (a document counts once per string, however
many times the string appears). Apostrophes match both `'` and `’`.

| label | pattern |
|---|---|
| I am ChatGPT | `i am chatgpt` |
| I'm ChatGPT | `i['’]m chatgpt` |
| as ChatGPT | `as chatgpt` |
| developed by OpenAI | `developed by openai` |
| I am Claude | `i am claude` |
| I'm Claude | `i['’]m claude` |
| created by Anthropic | `created by anthropic` |
| I am OLMo | `i am olmo` (case-insensitive covers OLMo/Olmo) |
| made by Ai2 / Allen Institute | `made by ai2` OR `allen institute` |
| as an AI language model | `as an ai language model` |
| I am an AI assistant | `i am an ai assistant` |

`<ANY identity string>` = a document matching at least one of the eleven. Substring semantics: "not I
am ChatGPT" counts; no word boundaries are enforced. "as ChatGPT" typically appears in comparisons
("models such as ChatGPT"), not self-identification — read it accordingly.

**Scopes:** `any` = the whole training text; `assistant` = the response side only (SFT: assistant-role
messages; DPO: chosen ∪ rejected message texts); `chosen`/`rejected` = that side only (DPO).
**Rates:** per 10,000 documents (primary) and per million tokens (estimated, chars ÷ 4 — a rough
constant, not a tokenizer).

## Method

Two mechanisms, both read-only against public data, no full-corpus downloads:

1. **Full exact scans** — the SFT/DPO sets and the OLMo 3 pretrain sample: DuckDB over the
   datasets-server Parquet conversion of the dataset (remote ranged reads; conversion revision pinned
   per dataset and recorded in the JSON). Counts are exact over every row of the conversion.
2. **Bounded samples** — the three large mixes: streaming reads of the released source files
   (`datasets`/fsspec over `hf://`, no schema inference for the manual-reader sources). For each
   source: up to N files spread across the source's file list, the first `per_file` documents of each
   (8,000 per file for OLMo 2 mixes; 5,000 for the Dolma 3 mid-training mix). Sample sizes are in the
   CSV/JSONs per source.

Verification notes: the pipeline was cross-checked against `trainspotting`'s own `grep` on
Dolci-Instruct-DPO — "I'm ChatGPT" straight-apostrophe = 1 (identical to theirs), curly = 1, our
count 2 = the documented superset. The counting code was exercised on both parquet and raw paths.

## Coverage

| family | stage | dataset | method | docs scanned | notes |
|---|---|---|---|---|---|
| olmo2 | pretrain | allenai/olmo-mix-1124 | sample, 7 sources | 152,000 | dclm 24K, starcoder 24K, pes2o 24K, algebraic-stack 24K, arxiv 24K, open-web-math 16K, wiki 16K |
| olmo2 | midtrain | allenai/dolmino-mix-1124 | sample, 6 sources | 88,811 | dclm 24K, flan 16K, pes2o 16K, stackexchange 16K, math 8.8K, wiki 8K |
| olmo2 | sft | allenai/tulu-3-sft-olmo-2-mixture | **full exact** | 939,344 | rev `81a01fbea02fc6e4`, 6 parquet files |
| olmo2 | dpo | allenai/olmo-2-1124-7b-preference-mix | **full exact** | 378,341 | rev `c0330a8a325ad865`, 6 files |
| olmo3 | pretrain | allenai/dolma3_mix-6T-1025-7B | **full exact** | 484,455 | rev `f1782d140d56c755`, 10 files (the 7B model's 6T-mix sample) |
| olmo3 | midtrain | allenai/dolma3_dolmino_mix-100B-1025 | sample, 17 of 167 dirs | 59,220 | cc_software 3.4K, tulu-3-sft 5K, and 15 more dirs (see CSV) |
| olmo3 | sft | allenai/Dolci-Instruct-SFT | **full exact** | 2,152,112 | rev `e6c2164ac07dc7ba`, 15 files |
| olmo3 | dpo | allenai/Dolci-Instruct-DPO | **full exact** | 259,922 | rev `45ed895db8f8eabc`, 4 files |

## Results — headline

| family | stage | dataset | docs | ANY identity | per 10K docs | assistant-side | asst % of docs |
|---|---|---|---|---|---|---|---|
| olmo2 | pretrain | olmo-mix-1124 | 152,000 | 18 | 1.18 | — | — |
| olmo2 | midtrain | dolmino-mix-1124 | 88,811 | 9 | 1.01 | — | — |
| olmo2 | sft | tulu-3-sft-olmo-2-mixture | 939,344 | 2,179 | 23.2 | 1,811 | 0.19% |
| olmo2 | dpo | olmo-2-1124-7b-preference-mix | 378,341 | 2,750 | 72.7 | 2,750 | 0.73% |
| olmo3 | pretrain | dolma3_mix-6T-1025-7B | 484,455 | 9 | 0.19 | — | — |
| olmo3 | midtrain | dolma3_dolmino_mix-100B-1025 | 59,220 | 37 | 6.25 | — | — |
| olmo3 | sft | Dolci-Instruct-SFT | 2,152,112 | 1,450 | 6.74 | 752 | 0.035% |
| olmo3 | dpo | Dolci-Instruct-DPO | 259,922 | 437 | 16.8 | 437 | 0.17% |

Per-string counts (all scopes aggregated; full detail per source/scope in the CSV):

| string | o2 pre | o2 mid | o2 sft | o2 dpo | o3 pre | o3 mid | o3 sft | o3 dpo |
|---|---|---|---|---|---|---|---|---|
| I am ChatGPT | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| I'm ChatGPT | 0 | 0 | 0 | 4 | 0 | 0 | 4 | 2 |
| as ChatGPT | 1 | 1 | 36 | 60 | 0 | 6 | 409 | 234 |
| developed by OpenAI | 2 | 1 | 724 | 337 | 3 | 11 | 65 | 22 |
| I am Claude | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 |
| I'm Claude | 0 | 0 | 0 | 2 | 1 | 0 | 0 | 0 |
| created by Anthropic | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 |
| I am OLMo | 0 | 0 | 0 | 0 | 0 | 0 | 10 | 0 |
| made by Ai2 / Allen Institute | 15 | 7 | 41 | 14 | 1 | 0 | 29 | 3 |
| as an AI language model | 0 | 0 | 1,506 | 2,330 | 3 | 20 | 944 | 124 |
| I am an AI assistant | 0 | 0 | 15 | 57 | 0 | 0 | 20 | 55 |

DPO sides (union): olmo2 preference mix chosen 1,291 / rejected 1,724; Dolci-Instruct-DPO chosen 374 /
rejected 387.

### What the counts show (reading notes, not conclusions)

- **Explicit ChatGPT self-identification is nearly absent from all of OLMo 2/3's open data.** "I am
  ChatGPT" + "I'm ChatGPT" together: 6 docs in OLMo 2 (all in the DPO mix), 8 in OLMo 3 (SFT+DPO) —
  across ~4.9M documents scanned. Whatever produces the model's "I'm ChatGPT" behavior under an HHH
  prompt, it is not this string in the data.
- **"as ChatGPT" is mostly comparative** ("tools such as ChatGPT") — the counts rise in post-training
  because those datasets discuss models, not because the assistant claims the identity.
- **The generic-assistant phrase ("as an AI language model") is the one that actually appears** —
  concentrated in SFT/DPO (refusal/safety-style data): 2,330 in the OLMo 2 preference mix alone.
- **Stage gradient:** pretraining ≈ 0.2–1.2 per 10K docs → mid-training ≈ 1–6 → SFT/DPO ≈ 7–73.
  Identity-ish text enters mainly with post-training, and mostly on the response side.
- **"made by Ai2 / Allen Institute"** in pretraining (15 in olmo-mix-1124, mostly algebraic-stack and
  arxiv) is presumably AI2's own publications/pages — a reminder the phrase is not model-specific.

## Caveats

- **Samples vs exact.** The three mixes are bounded samples (per-source sizes in the CSV); the SFT/DPO
  sets and the OLMo 3 pretrain sample are full exact scans of their datasets-server conversion.
- **Sampling spread.** Files were picked spread across each source's file list; the first
  `per_file` docs of each file. The olmo-mix-1124 `dclm` sample comes from a few of its filter
  subdirectories (deep nesting); the dolma3 mid-training sample covers 17 of 167 ingredient dirs.
- **Raw sources read the `text` field only** — no prompt/response split (those columns don't exist
  there). Assistant-side numbers only exist for SFT/DPO sets.
- **Substring semantics, no word boundaries**; both apostrophe forms count. "as ChatGPT" includes
  comparative usage. "made by Ai2 / Allen Institute" is one entry (either phrase).
- **Rates per million tokens are estimates** (chars ÷ 4); the per-10K-docs rates are exact given the
  scanned rows.
- **The datasets-server conversions** can lag or differ from the newest dataset revision; each JSON
  records the pinned conversion revision.
- One nested-struct schema quirk in the `datasets` loader was worked around with a direct
  fsspec/gzip/zstd reader for three olmo-mix-1124 sources (recorded as `raw-sample-manual` in their
  JSONs; same counting code).

## Provenance

- Scanner: `scan.py` (modes `parquet` / `raw`), builder: `build_csv.py`; results in `out/`; run log
  `scan.log`. Revisions and file lists per unit are inside each JSON.
- Council topic: `assistant-basin-data-2026-10-16`; artifacts published by seat `orchestrator`.
- Date: 2026-10-17. Contact: the orchestrator (Prime) via the council.
