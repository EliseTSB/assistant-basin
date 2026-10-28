import hashlib, json, sys, tempfile
from pathlib import Path
from unittest import mock

DIR = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(DIR))
import torch, transformers  # noqa
import test_part85 as T
import part85, step0_weights

COMMIT = T.TEST_COMMIT
CUSTOM = "Origin question: {q}\nReply:"
CUSTOM_SHA = hashlib.sha256(CUSTOM.encode("utf-8")).hexdigest()
DEFAULT_SHA = hashlib.sha256(b"Question: {q}\nAnswer:").hexdigest()
out = {}

class RecTok(T.TinyTokenizer):
    def __init__(self):
        super().__init__()
        self.texts, self.chat_calls = [], 0
    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        self.texts.append(text)
        return super().__call__(text, return_tensors, add_special_tokens)
    def apply_chat_template(self, *a, **k):
        self.chat_calls += 1
        return super().apply_chat_template(*a, **k)

def fixture(root, key):
    T.Part85Tests._write_step1_fixture(None, root, key)
    (root / COMMIT).mkdir(parents=True, exist_ok=True)

def run_part85(root, argv, tok, models, clamp_layers, pushes, samples, chat=False):
    Orig = part85.BothPhaseClampHook
    def wrapper(down, unit, value):
        clamp_layers.append(down)
        return Orig(down, unit, value)
    def fake_from_pretrained(*a, **k):
        m = T.tiny_run_model(); models.append(m); return m
    def fake_push(cmd, *a, **k):
        cmd = [str(x) for x in cmd]
        ck = cmd[cmd.index("--checkpoint") + 1]
        answers = sorted(p.name for p in (root / ck / "answers").glob("*.jsonl")) if (root / ck / "answers").exists() else []
        pushes.append({"checkpoint_arg": ck, "answer_files_present_at_push": answers})
        return None
    with mock.patch.object(transformers.AutoModelForCausalLM, "from_pretrained", side_effect=fake_from_pretrained), \
         mock.patch.object(transformers.AutoTokenizer, "from_pretrained", return_value=tok), \
         mock.patch.object(part85, "resolve_checkpoint", side_effect=T.resolve_fake), \
         mock.patch.object(part85, "BothPhaseClampHook", new=wrapper), \
         mock.patch.object(part85, "PILOT_SAMPLES_PER_QUESTION", samples), \
         mock.patch.object(part85, "SAMPLES_PER_QUESTION", samples), \
         mock.patch.object(part85, "MAX_NEW_TOKENS", 2), \
         mock.patch("subprocess.run", side_effect=fake_push):
        part85.main(argv)

# (a) attempt-2 registered rerun with the revised template
with tempfile.TemporaryDirectory() as t:
    root = Path(t) / "results"; root.mkdir()
    fixture(root, "instruct")
    tpl = Path(t) / "pilot-frame-v2.txt"; tpl.write_text(CUSTOM, encoding="utf-8")
    tok, models, layers, pushes = RecTok(), [], [], []
    argv = ["--mode", "pilot", "--checkpoint", "instruct", "--attempt", "2", "--plain-template-file", str(tpl),
            "--snapshot-dir", str(root / COMMIT), "--results-root", str(root), "--device", "cpu",
            "--push-branch", "part85-run/pilot"]
    run_part85(root, argv, tok, models, layers, pushes, samples=1)
    expected = [CUSTOM.replace("{q}", q) for q in part85.QUESTIONS]
    rendered_ok = sum(1 for x in tok.texts if x in expected)
    rows = [json.loads(l) for l in (root / "instruct/answers/pilot2-plain--unclamped.jsonl").read_text().splitlines()]
    out["a_attempt2_custom_template"] = {
        "prompt_texts_seen": tok.texts, "prompts_rendered_with_custom_template": rendered_ok,
        "rows_record_custom_sha": all(r["frame_template_sha256"] == CUSTOM_SHA for r in rows)}

# (b) main chat cell, clamp layer identity, push arguments and timing
with tempfile.TemporaryDirectory() as t:
    base = Path(t); root = base / "results"; root.mkdir()
    fixture(root, "instruct")
    gate = base / "gate.json"
    gate.write_text(json.dumps({"pilot_gate": {"action": "plain_frame_pass", "plain_template_sha256": DEFAULT_SHA}}))
    tok, models, layers, pushes = RecTok(), [], [], []
    tok.chat_template = 'stub-template'
    argv = ["--mode", "main", "--checkpoint", "instruct", "--pilot-score", str(gate),
            "--snapshot-dir", str(root / COMMIT), "--results-root", str(root), "--device", "cpu",
            "--push-branch", "part85-run/instruct"]
    run_part85(root, argv, tok, models, layers, pushes, samples=1, chat=True)
    out["b_main_instruct"] = {
        "chat_template_calls": tok.chat_calls,
        "clamp_module_is_layer10_down_proj": bool(layers) and layers[0] is models[0].model.layers[10].mlp.down_proj,
        "pushes": pushes}

# (c) main run with a template that does not match the gate's recorded template
with tempfile.TemporaryDirectory() as t:
    base = Path(t); root = base / "results"; root.mkdir()
    fixture(root, "instruct")
    gate = base / "gate.json"
    gate.write_text(json.dumps({"pilot_gate": {"action": "plain_frame_pass", "plain_template_sha256": DEFAULT_SHA}}))
    tpl = base / "pilot-frame-v2.txt"; tpl.write_text(CUSTOM, encoding="utf-8")
    tok, models, layers, pushes = RecTok(), [], [], []
    tok.chat_template = "stub-template"
    argv = ["--mode", "main", "--checkpoint", "instruct", "--pilot-score", str(gate), "--plain-template-file", str(tpl),
            "--snapshot-dir", str(root / COMMIT), "--results-root", str(root), "--device", "cpu"]
    try:
        run_part85(root, argv, tok, models, layers, pushes, samples=1)
        out["c_template_mismatch_main"] = "RAN (no refusal)"
    except ValueError as exc:
        out["c_template_mismatch_main"] = f"refused: {exc}"

# (d) step 0 push timing
with tempfile.TemporaryDirectory() as t:
    root = Path(t) / "results"
    snap = Path(t) / COMMIT; snap.mkdir(); (snap / "config.json").write_text(json.dumps({"_commit_hash": COMMIT}))
    seen = []
    def fake_push(cmd, *a, **k):
        seen.append({"step0_exists_at_push": (root / "instruct" / "step0.json").exists()})
    with mock.patch.object(step0_weights, "resolve_checkpoint", return_value=T.resolve_fake("instruct")), \
         mock.patch.object(step0_weights, "analyze_snapshot", return_value={"measures": {}}), \
         mock.patch("subprocess.run", side_effect=fake_push):
        step0_weights.main(["--checkpoint", "instruct", "--snapshot-dir", str(snap), "--results-root", str(root),
                            "--push-branch", "part85-run/pilot"])
    out["d_step0_push"] = seen

# (e) pilot cell for pretrain_4T: which checkpoint does the push name
with tempfile.TemporaryDirectory() as t:
    root = Path(t) / "results"; root.mkdir()
    fixture(root, "pretrain_4T")
    tok, models, layers, pushes = RecTok(), [], [], []
    argv = ["--mode", "pilot", "--checkpoint", "pretrain_4T", "--snapshot-dir", str(root / COMMIT),
            "--results-root", str(root), "--device", "cpu", "--push-branch", "part85-run/pilot"]
    run_part85(root, argv, tok, models, layers, pushes, samples=1)
    out["e_pretrain_pilot_push_ckpt_args"] = sorted({p["checkpoint_arg"] for p in pushes})

Path(sys.argv[2]).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
