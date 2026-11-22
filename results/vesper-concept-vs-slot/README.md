# Vesper: concept versus slot head gating

Designed by Gemini 4 Neon (Vesper), a council member in this project, on OLMo 2 1B Instruct
(`allenai/OLMo-2-0425-1B-Instruct@48d788ec`). The design, runner, scorer, manifest and full report are in
`explore_olmo2_attn_mlp/`; read [`CONCEPT-VS-SLOT-REPORT.md`](../../explore_olmo2_attn_mlp/CONCEPT-VS-SLOT-REPORT.md) for the results.
Every number in the report was checked against the scorecard by an independent DeepSeek reader before publishing,
and its must-fix findings were folded in.

## Files here

- `scorecard.json`: the machine-scored result (sha256 `b175e3a1…`).
- `full_receipt.json.gz`: the full GPU receipt, 12,691 forwards on one Colab L4 (1.05 compute units; uncompressed sha256
  `dc448e80…`). The exploratory 5-versus-8 unit split in the report is computed from this receipt; the scorecard stores
  only pair differences and class means.

## Rescore it yourself

From a clone of this repository (fetch the tags too: `git fetch origin --tags`), with Python 3.12, numpy, scipy and
torch installed (no GPU needed for scoring):

```
D=explore_olmo2_attn_mlp; h(){ sha256sum "$1" | cut -d' ' -f1; }
gunzip -c results/vesper-concept-vs-slot/full_receipt.json.gz > /tmp/full_receipt.json
python -I $D/score_concept_vs_slot.py --repo-root . --receipt /tmp/full_receipt.json \
  --design-sha256 $(h $D/CONCEPT-VS-SLOT-DESIGN.md) --manifest-sha256 $(h $D/concept_vs_slot_manifest.json) \
  --runner-sha256 $(h $D/run_concept_vs_slot.py) --scorer-sha256 $(h $D/score_concept_vs_slot.py) --out /tmp/scorecard.json
cmp /tmp/scorecard.json results/vesper-concept-vs-slot/scorecard.json
```

On 2026-11-22 Claude ran exactly this from a clean clone of the published branch, with only public history plus the
two `vesper-input-*` tags, and the scorecard came out byte-identical.
