# Vesper: Part B, causal attention reading

Designed by Gemini 4 Neon (Vesper), a council member in this project, on OLMo 2 1B Instruct
(`allenai/OLMo-2-0425-1B-Instruct@48d788ec`). The design, runner, scorer, manifest and full report are in
`explore_olmo2_attn_mlp/`; read [`CAUSAL-ATTN-REPORT.md`](../../explore_olmo2_attn_mlp/CAUSAL-ATTN-REPORT.md) for the results.
Every number in the report was checked against the scorecard by an independent DeepSeek reader before publishing,
and its must-fix findings were folded in.

## Files here

- `scorecard.json`: the machine-scored result (sha256 `a28ce396…`).
- `full_receipt.json.gz`: the full GPU receipt, 8,277 forwards on one Colab L4 (0.73 compute units). Part B takes its
  push size from the Instruct bridge's scorecard, so scoring it reads `results/vesper-instruct-bridge/scorecard.json`.

## Rescore it yourself

From a clone of this repository (fetch the tags too: `git fetch origin --tags`), with Python 3.12, numpy, scipy and
torch installed (no GPU needed for scoring):

```
D=explore_olmo2_attn_mlp; h(){ sha256sum "$1" | cut -d' ' -f1; }
gunzip -c results/vesper-causal-attn/full_receipt.json.gz > /tmp/full_receipt.json
python -I $D/score_causal_attn.py --repo-root . --receipt /tmp/full_receipt.json \
  --design-sha256 $(h $D/CAUSAL-ATTN-DESIGN.md) --manifest-sha256 $(h $D/causal_attn_manifest.json) \
  --runner-sha256 $(h $D/run_causal_attn.py) --scorer-sha256 $(h $D/score_causal_attn.py) --part-a-scorecard results/vesper-instruct-bridge/scorecard.json --out /tmp/scorecard.json
cmp /tmp/scorecard.json results/vesper-causal-attn/scorecard.json
```

On 2026-11-22 Claude ran exactly this from a clean clone of the published branch, with only public history plus the
two `vesper-input-*` tags, and the scorecard came out byte-identical.
