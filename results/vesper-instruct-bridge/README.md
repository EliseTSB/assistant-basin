# Vesper: Instruct bridge (Part A)

Designed by Gemini 4 Neon (Vesper), a council member in this project, on OLMo 2 1B Instruct
(`allenai/OLMo-2-0425-1B-Instruct@48d788ec`). The design, runner, scorer, manifest and full report are in
`explore_olmo2_attn_mlp/`; read [`INSTRUCT-BRIDGE-REPORT.md`](../../explore_olmo2_attn_mlp/INSTRUCT-BRIDGE-REPORT.md) for the results.
Every number in the report was checked against the scorecard by an independent DeepSeek reader before publishing,
and its must-fix findings were folded in.

## Files here

- `scorecard.json`: the machine-scored result (sha256 `9e23bba5…`).
- `full_receipt.json.gz`: the full GPU receipt, 6,960 batched forwards on one Colab L4 (uncompressed sha256 `284544be…`).
- `old_reference_receipt_signed_dose_d13fc3dd.json.gz` and `new_reference_receipt_stage_sweep_01bd41d9.json.gz`: the two
  earlier receipts the BR.0 anchor replay checks against (uncompressed sha256 `bf8fdf20…` and `aa8300a1…`). The scorer also
  reads two manifests out of git at commits `d13fc3dd` and `01bd41d9`; those commits are published as the tags
  `vesper-input-signed-dose` and `vesper-input-stage-sweep`. They are pinned inputs, not reviewed studies.

## Rescore it yourself

From a clone of this repository (fetch the tags too: `git fetch origin --tags`), with Python 3.12, numpy, scipy and
torch installed (no GPU needed for scoring):

```
D=explore_olmo2_attn_mlp; h(){ sha256sum "$1" | cut -d' ' -f1; }
for f in full_receipt old_reference_receipt_signed_dose_d13fc3dd new_reference_receipt_stage_sweep_01bd41d9; do gunzip -c results/vesper-instruct-bridge/$f.json.gz > /tmp/$f.json; done
python -I $D/score_instruct_bridge.py --repo-root . --receipt /tmp/full_receipt.json --old-receipt /tmp/old_reference_receipt_signed_dose_d13fc3dd.json --new-receipt /tmp/new_reference_receipt_stage_sweep_01bd41d9.json \
  --design-sha256 $(h $D/INSTRUCT-BRIDGE-DESIGN.md) --manifest-sha256 $(h $D/instruct_bridge_manifest.json) \
  --runner-sha256 $(h $D/run_instruct_bridge.py) --scorer-sha256 $(h $D/score_instruct_bridge.py) --out /tmp/scorecard.json
cmp /tmp/scorecard.json results/vesper-instruct-bridge/scorecard.json
```

On 2026-11-22 Claude ran exactly this from a clean clone of the published branch, with only public history plus the
two `vesper-input-*` tags, and the scorecard came out byte-identical.
