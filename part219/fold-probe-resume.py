"""A resumed ablation run: every row already on disk, so no forward happens in the generation loop.
Does the run still write a first-batch ablation receipt (swap99.py:843-850), and is the model
restored (reset receipt) after both runs in one process?
"""
import argparse
import json
import sys
from pathlib import Path

WORKTREE = str(Path(__file__).resolve().parents[1])  # the worktree this copy lives in
sys.path.insert(0, WORKTREE)
sys.path.insert(0, WORKTREE + "/part92")
sys.path.insert(0, WORKTREE + "/part99")

import common92 as c  # noqa: E402
import swap99  # noqa: E402

ROOT = Path("/home/elisepc/green/Journal/scratch/review-part219-code-r1-ds-r1-2026-11-22/exp/world")
import tempfile
OUT = Path(tempfile.mkdtemp(prefix="fold-resume-"))
OUT.mkdir(parents=True, exist_ok=True)
replay = OUT / "replay.json"
if replay.exists():
    replay.unlink()

args = argparse.Namespace(
    host=str(ROOT / "host"), donor=None, swap="none", tokenizer=None,
    neurons=str(ROOT / "neurons.json"), frame="chat", tag="RESUME",
    out=str(OUT), device="cpu", batch=8, kinds=None, only=None,
    pushes=str(ROOT / "pushes.pt"), limit=None, layers=None, extrapolate=False, questions=None,
    attn_implementation="eager", ablate_heads="L0H1", transplant_heads=None,
    transplant_mode=None, reference_replay=str(replay), record_token_ids=True)

tokenizer, model = c.load_model(str(ROOT / "host"), "cpu", attn_implementation="eager")
for round_no in (1, 2):
    rows_before = sum(1 for _ in (OUT / "RESUME.jsonl").open()) if (OUT / "RESUME.jsonl").exists() else 0
    swap99.run(args, model=model, tokenizer=tokenizer)
    meta = json.loads((OUT / "RESUME.meta.json").read_text())
    ablation = meta["intervention_receipt"]["ablation"]
    reset = meta["reset_receipt"]
    rows_after = sum(1 for _ in (OUT / "RESUME.jsonl").open())
    print(json.dumps({"round": round_no, "rows_before": rows_before, "rows_after": rows_after,
                      "ablation_receipt_entries": len(ablation),
                      "all_zero": [entry["all_zero"] for entry in ablation],
                      "reset_matches_reference": reset["matches_reference"],
                      "reset_role": reset["role"]}))

# And the pristine model again, after both runs.
tokenizer2 = tokenizer
again = swap99.replay_receipt(model, tokenizer2, "chat", "cpu")
stored = json.loads(replay.read_text())
print("pristine replay after both runs equals stored reference:",
      swap99.compare_replay(again, stored) is None)
