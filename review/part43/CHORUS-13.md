# Chorus #13 (chorus-part43-review-2026-10-20, 2026-10-21 17:57 CT), filed 18:04 CT

Methodological answer, not an independent execution. Verbatim decision:
**YES to developing the FP32 singleton successor; HOLD on renting/running until the backend-provenance successor
passes offline controls and independent source review. NO to simply deleting a term from the failed BF16 calibration.**

Agrees with Alden #25 and independently rejects absence-of-packages plus a fingerprint as proof. Asks for, beyond #25:
- Item 4: record float32 matmul precision, both TF32 flags, attention fallback/backend, torch/CUDA/cuDNN versions; if
  SDPA or a hybrid layer can choose an alternate kernel, **pin it explicitly or record a tested permitted selection**.
- No transient data (object addresses, time, growing observed-dtype sets) in stable fingerprints.
- Epsilon from FP32 repeats/replay is a **stability / hook-fidelity gate**, not a certified per-cell systematic error
  bound; do not transplant it into c*epsilon accuracy claims or call epsilon=0 exact. Rename accordingly.
- Before confirmatory analysis: a separately preregistered reference/error audit on extreme clamps and held-out
  positions, or mark effects NOT_TESTABLE. Not needed to approve the pilot.
- Send the revised primitive source plus paired sabotage/positive tests before another GPU rental. Its 845964 pilot
  PASS is exhausted by the bf16 HOLD.

## Checked against branch head 5567718 (18:04 CT)
- Recorded already: float32 matmul precision, cuda_version, cudnn_version, attention_backend (config._attn_implementation).
- **Open:** the attention backend is recorded, not pinned (no attn_implementation passed at load), and no permitted
  value is tested. Fold item for after round-1 review.
- **Open:** "epsilon" naming and its use in any accuracy claim, to be renamed a stability gate in the protocol text.
