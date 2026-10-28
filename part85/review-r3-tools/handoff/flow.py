import hashlib, json, sys
from pathlib import Path
from unittest import mock
DIR = Path(sys.argv[1]).resolve(); mode = sys.argv[2]; H = Path(sys.argv[3])
sys.path.insert(0, str(DIR))
import transformers
import test_part85 as T
import part85
COMMIT = T.TEST_COMMIT
snap = H / "snap" / COMMIT; snap.mkdir(parents=True, exist_ok=True)
gate = H / "gate.json"
gate.write_text(json.dumps({"pilot_gate": {"action": "plain_frame_pass",
    "plain_template_sha256": hashlib.sha256(b"Question: {q}\nAnswer:").hexdigest()}}))
tok = T.TinyTokenizer(); tok.chat_template = "stub-template"
root = Path("results/part85")
if mode == "pilot":
    T.Part85Tests._write_step1_fixture(None, root, "instruct")
    argv = ["--mode", "pilot", "--checkpoint", "instruct", "--snapshot-dir", str(snap),
            "--results-root", "results/part85", "--device", "cpu", "--push-branch", "part85-run/pilot"]
if mode == "pilot2":
    argv = ["--mode", "pilot", "--checkpoint", "instruct", "--attempt", "2",
            "--plain-template-file", "results/part85/pilot-frame-v2.txt", "--snapshot-dir", str(snap),
            "--results-root", "results/part85", "--device", "cpu", "--push-branch", "part85-run/pilot"]
elif mode == "main":
    argv = ["--mode", "main", "--checkpoint", "instruct", "--pilot-score", str(gate), "--snapshot-dir", str(snap),
            "--results-root", "results/part85", "--device", "cpu", "--push-branch", "part85-run/instruct"]
with mock.patch.object(transformers.AutoModelForCausalLM, "from_pretrained", side_effect=lambda *a, **k: T.tiny_run_model()), \
     mock.patch.object(transformers.AutoTokenizer, "from_pretrained", return_value=tok), \
     mock.patch.object(part85, "resolve_checkpoint", side_effect=T.resolve_fake), \
     mock.patch.object(part85, "PILOT_SAMPLES_PER_QUESTION", 1), mock.patch.object(part85, "SAMPLES_PER_QUESTION", 1), \
     mock.patch.object(part85, "MAX_NEW_TOKENS", 2):
    part85.main(argv)
print("flow done:", mode)
