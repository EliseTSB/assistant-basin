"""Tests for the recorded attention kernel (neuron_report.load_model), tools/kernel_reference.py and the kernel stage
of tools/part_check.py, on a tiny random Qwen2 model on the CPU. Written 2026-11-03 for the kernel-agreement gate
that followed Part 178.

Run: python -m pytest -q tools/test_kernel_check.py. Deliberate breaks, each on a copy of the tool, never the
tracked file: (1) part_check.py: make kernel_check ignore the margin (treat every divergence as a near-tie) ->
test_confident_divergence_fails fails; (2) part_check.py: drop design_kernel_check from prelaunch ->
test_prelaunch_requires_a_kernel_check_when_method_names_the_arc fails.
"""
import copy, json, os, subprocess, sys
from pathlib import Path
from types import SimpleNamespace

import pytest

TOOLS = Path(os.environ.get("PART_CHECK_TOOLS", Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent
for _p in (str(TOOLS), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import part_check as pc  # noqa: E402
import kernel_reference as kr  # noqa: E402
import test_tools as tt  # noqa: E402  (the fixture design, predictions and git helpers)

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
import neuron_report as nr  # noqa: E402

CHAT_TEMPLATE = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
                 "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
NEW_TOKENS = 12


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """A random two-layer Qwen2 model and a word-level tokenizer with a chat template, saved to one folder."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    folder = tmp_path_factory.mktemp("tiny")
    corpus = list(kr.QUESTIONS) + ["the sea the rain an egg a tip sleep photosynthesis capital France user assistant"] * 3
    tk = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.train_from_iterator(corpus, trainers.WordLevelTrainer(
        special_tokens=["[UNK]", "<|endoftext|>", "<|im_start|>", "<|im_end|>"]))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", eos_token="<|im_end|>",
                                  pad_token="<|endoftext|>")
    tok.chat_template = CHAT_TEMPLATE
    tok.save_pretrained(str(folder))
    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                         tie_word_embeddings=True)
    Qwen2ForCausalLM(config).save_pretrained(str(folder))
    return folder


@pytest.fixture(scope="module")
def reference(tiny, tmp_path_factory):
    out = tmp_path_factory.mktemp("ref") / "ref.json"
    rc = kr.main(["--model", str(tiny), "--device", "cpu", "--dtype", "float32", "--attn-implementation", "eager",
                  "--new-tokens", str(NEW_TOKENS), "--out", str(out)])
    assert rc == 0
    return json.loads(out.read_text(encoding="utf-8"))


# (a) the loader records the kernel

def test_load_model_records_eager_by_default_and_sdpa_when_asked(tiny):
    tok, model = nr.load_model(str(tiny), None, "cpu", dtype=torch.float32)
    assert nr.LAST_LOAD["attn_implementation"] == "eager"
    assert nr.LAST_LOAD["dtype"] == "torch.float32" and nr.LAST_LOAD["device"] == "cpu"
    assert model.config._attn_implementation == "eager"
    assert nr.LAST_LOAD["torch"] == torch.__version__ and nr.LAST_LOAD["transformers"] == transformers.__version__
    tok, model = nr.load_model(str(tiny), None, "cpu", attn_implementation="sdpa", dtype=torch.float32)
    assert nr.LAST_LOAD["attn_implementation"] == "sdpa" and model.config._attn_implementation == "sdpa"
    head = nr.header(SimpleNamespace(model=str(tiny), revision=None, layer=0, neuron=0, seed=0), 64, [], "layers")
    assert head["load"]["attn_implementation"] == "sdpa"
    tok, model = nr.load_model(str(tiny), None, "cpu")      # the default dtype is still bfloat16
    assert nr.LAST_LOAD["dtype"] == "torch.bfloat16" and nr.LAST_LOAD["attn_implementation"] == "eager"


def test_load_model_rejects_an_empty_or_non_string_kernel(tiny):
    for bad in ("", "   ", None, 3):
        with pytest.raises(ValueError):
            nr.load_model(str(tiny), None, "cpu", attn_implementation=bad)


def test_common86_passes_the_kernel_through(tiny, monkeypatch):
    sys.path.insert(0, str(ROOT / "part86"))
    import common86
    seen = {}
    real = nr.load_model

    def fake(model_id, revision, device, attn_implementation="x", dtype=None):
        seen.update(model_id=model_id, revision=revision, device=device, attn_implementation=attn_implementation)
        return real(str(tiny), None, device, attn_implementation=attn_implementation, dtype=torch.float32)
    monkeypatch.setattr(nr, "load_model", fake)
    common86.load_model("cpu")
    assert seen["attn_implementation"] == "eager" and seen["model_id"] == common86.MODEL_ID
    assert common86.provenance("torch.bfloat16", 1.0)["attn_implementation"] == "eager"


# (b) the reference record

def test_reference_record_has_tokens_and_margins(reference):
    assert reference["kind"] == "kernel-reference" and reference["reference"] is None
    assert reference["attn_implementation"] == "eager" and reference["dtype"] == "torch.float32"
    assert reference["new_tokens"] == NEW_TOKENS and reference["batch_sizes"] == [1, 2]
    assert len(reference["prompt_sha256"]) == 64 and reference["prompt_sha256"] == kr.prompt_sha256(reference["prompts"])
    assert len(reference["prompts"]) == 8 and all("<|im_start|>assistant" in p for p in reference["prompts"])
    assert reference["argmax_agrees"] is True
    assert set(reference["runs"]) == {"1", "2"}
    for b in ("1", "2"):
        rows = reference["runs"][b]
        assert [e["prompt"] for e in rows] == list(range(8))
        for e in rows:
            assert 1 <= len(e["tokens"]) <= NEW_TOKENS and len(e["margins"]) == len(e["tokens"])
            assert all(m >= 0 for m in e["margins"]) and e["finish"] in ("eos", "cap") and e["argmax_agrees"]


def test_reference_is_deterministic_on_the_cpu(tiny, reference, tmp_path):
    out = tmp_path / "again.json"
    assert kr.main(["--model", str(tiny), "--device", "cpu", "--dtype", "float32", "--new-tokens", str(NEW_TOKENS),
                    "--out", str(out)]) == 0
    again = json.loads(out.read_text(encoding="utf-8"))
    assert again["runs"]["1"] == reference["runs"]["1"]
    assert pc.kernel_check(reference, again).failed == []


# (c) to (f) the comparison

def test_identical_candidate_passes(reference):
    cand = copy.deepcopy(reference)
    cand.update(kind="kernel-candidate", device="xpu", attn_implementation="sdpa")
    rep = pc.kernel_check(reference, cand)
    assert rep.failed == [] and sum("agrees with the reference" in m for _, m in rep.rows) == 16


def test_confident_divergence_fails(reference):
    ref, cand = copy.deepcopy(reference), copy.deepcopy(reference)
    k = min(3, len(ref["runs"]["1"][2]["tokens"]) - 1)
    ref["runs"]["1"][2]["margins"][k] = 1.5
    cand["runs"]["1"][2]["tokens"][k] = (cand["runs"]["1"][2]["tokens"][k] + 1) % 50
    rep = pc.kernel_check(ref, cand)
    assert len(rep.failed) == 1
    assert f"position {k}" in rep.failed[0] and "prompt 2" in rep.failed[0] and "1.500" in rep.failed[0]


def test_near_tie_divergence_passes_with_a_note(reference):
    ref, cand = copy.deepcopy(reference), copy.deepcopy(reference)
    row = cand["runs"]["2"][5]
    k = min(2, len(row["tokens"]) - 1)
    ref["runs"]["2"][5]["margins"][k] = 0.1
    for p in range(k, len(row["tokens"])):            # every later token differs too; none is compared
        row["tokens"][p] = (row["tokens"][p] + 1) % 50
    rep = pc.kernel_check(ref, cand)
    assert rep.failed == []
    assert any("near-tie" in m and f"position {k}" in m and "prompt 5" in m for _, m in rep.rows)


def test_a_larger_min_margin_turns_a_fault_into_a_near_tie(reference):
    ref, cand = copy.deepcopy(reference), copy.deepcopy(reference)
    ref["runs"]["1"][0]["margins"][0] = 0.8
    cand["runs"]["1"][0]["tokens"][0] = (cand["runs"]["1"][0]["tokens"][0] + 1) % 50
    assert pc.kernel_check(ref, cand, min_margin=0.5).failed
    assert pc.kernel_check(ref, cand, min_margin=1.0).failed == []


def test_different_revision_or_prompts_or_length_fail(reference):
    for key, value in (("revision", "deadbeef"), ("prompt_sha256", "0" * 64), ("new_tokens", 96),
                       ("model", "Other/Model")):
        cand = copy.deepcopy(reference)
        cand[key] = value
        failed = pc.kernel_check(reference, cand).failed
        assert failed and any(key in f for f in failed), (key, failed)


def test_missing_prompt_and_impure_reference_fail(reference):
    cand = copy.deepcopy(reference)
    cand["runs"]["1"] = cand["runs"]["1"][1:]
    assert any("prompt 0" in f and "missing" in f for f in pc.kernel_check(reference, cand).failed)
    impure = copy.deepcopy(reference)
    impure["argmax_agrees"] = False
    assert any("pure greedy" in f for f in pc.kernel_check(impure, copy.deepcopy(reference)).failed)


def test_cli_kernel_stage(tmp_path, reference):
    ref_path, cand_path = tmp_path / "ref.json", tmp_path / "cand.json"
    ref_path.write_text(json.dumps(reference), encoding="utf-8")
    cand = copy.deepcopy(reference)
    cand_path.write_text(json.dumps(cand), encoding="utf-8")
    assert pc.main(["kernel", "--reference", str(ref_path), "--candidate", str(cand_path)]) == 0
    ref2 = copy.deepcopy(reference)
    ref2["runs"]["1"][1]["margins"][0] = 2.0
    cand["runs"]["1"][1]["tokens"][0] = (cand["runs"]["1"][1]["tokens"][0] + 1) % 50
    ref_path.write_text(json.dumps(ref2), encoding="utf-8")
    cand_path.write_text(json.dumps(cand), encoding="utf-8")
    assert pc.main(["kernel", "--reference", str(ref_path), "--candidate", str(cand_path)]) == 1
    assert pc.main(["kernel", "--reference", str(ref_path)]) == 2
    r = subprocess.run([sys.executable, "-I", str(TOOLS / "part_check.py"), "kernel", "--reference", str(ref_path),
                        "--candidate", str(cand_path)], capture_output=True, text=True)
    assert r.returncode == 1 and "KERNEL CHECK: FAIL" in r.stdout and "position 0" in r.stdout


def test_default_reference_path():
    assert pc.default_reference_path("Qwen/Qwen3.5-0.8B") == "results/kernel-reference/qwen-qwen3.5-0.8b-cpu-float32.json"
    assert kr.default_reference_path is pc.default_reference_path


# (g) prelaunch

@pytest.fixture
def repo(tmp_path):
    origin, r = tmp_path / "origin.git", tmp_path / "repo"
    tt.sh("git", "init", "-q", "--bare", "-b", "main", str(origin), cwd=tmp_path)
    tt.sh("git", "init", "-q", "-b", "main", str(r), cwd=tmp_path)
    for k, v in (("user.email", "t@example.com"), ("user.name", "t")):
        tt.sh("git", "config", k, v, cwd=r)
    tt.sh("git", "remote", "add", "origin", str(origin), cwd=r)
    (r / "PART900-DESIGN.md").write_text(tt.DESIGN, encoding="utf-8")
    (r / "PREDICTIONS.md").write_text(tt.PREDICTIONS, encoding="utf-8")
    tt.sh("git", "add", "-A", cwd=r)
    tt.sh("git", "commit", "-q", "-m", "register 900", cwd=r)
    tt.sh("git", "push", "-q", "origin", "main", cwd=r)
    return r


def write_records(repo, reference, candidate=None, reference_rel="results/kernel-reference/tiny-cpu-float32.json",
                  candidate_rel="results/part900/kernel-check.json"):
    ref_path, cand_path = repo / reference_rel, repo / candidate_rel
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    cand_path.parent.mkdir(parents=True, exist_ok=True)
    ref_path.write_text(json.dumps(reference), encoding="utf-8")
    cand = copy.deepcopy(candidate if candidate is not None else reference)
    cand.update(kind="kernel-candidate", device="xpu", dtype="torch.bfloat16", reference=reference_rel)
    cand_path.write_text(json.dumps(cand), encoding="utf-8")


def test_prelaunch_without_the_arc_needs_no_kernel_check(repo):
    rep = pc.prelaunch(repo, "900", fetch=False)
    assert rep.failed == [] and any("no kernel check needed" in m for _, m in rep.rows)


def test_prelaunch_requires_a_kernel_check_when_method_names_the_arc(repo, reference):
    tt.edit(repo, "A fixture; nothing runs.", "Runs on the A770 in bf16.", push=True)
    failed = pc.prelaunch(repo, "900", fetch=False).failed
    assert any("Kernel check" in f for f in failed), failed
    assert pc.prelaunch(repo, "900", fetch=False, legacy_ok=True).failed == []
    tt.edit(repo, "Runs on the A770 in bf16.", "Runs on the A770 in bf16.\n\nKernel check: results/part900/kernel-check.json",
            push=True)
    assert any("cannot read" in f for f in pc.prelaunch(repo, "900", fetch=False).failed)
    write_records(repo, reference)
    rep = pc.prelaunch(repo, "900", fetch=False)
    assert rep.failed == [], rep.failed
    assert any("kernel check results/part900/kernel-check.json against results/kernel-reference/tiny-cpu-float32.json"
               in m for _, m in rep.rows)


def test_prelaunch_fails_on_a_failing_record_and_legacy_ok_skips_it(repo, reference):
    tt.edit(repo, "A fixture; nothing runs.", "Runs on the xpu.\n\nKernel check: results/part900/kernel-check.json",
            push=True)
    ref = copy.deepcopy(reference)
    ref["runs"]["1"][0]["margins"][0] = 2.0
    cand = copy.deepcopy(reference)
    cand["runs"]["1"][0]["tokens"][0] = (cand["runs"]["1"][0]["tokens"][0] + 1) % 50
    write_records(repo, ref, cand)
    failed = pc.prelaunch(repo, "900", fetch=False).failed
    assert any("kernel check" in f and "position 0" in f for f in failed), failed
    assert pc.prelaunch(repo, "900", fetch=False, legacy_ok=True).failed == []


def test_prelaunch_uses_the_default_reference_path_when_the_candidate_names_none(repo, reference):
    tt.edit(repo, "A fixture; nothing runs.", "Runs on the A770.\n\nKernel check: results/part900/kernel-check.json",
            push=True)
    rel = pc.default_reference_path(reference["model"])
    write_records(repo, reference, reference_rel=rel)
    cand_path = repo / "results/part900/kernel-check.json"
    cand = json.loads(cand_path.read_text(encoding="utf-8"))
    cand.pop("reference")
    cand_path.write_text(json.dumps(cand), encoding="utf-8")
    rep = pc.prelaunch(repo, "900", fetch=False)
    assert rep.failed == [], rep.failed
    assert any(rel in m for _, m in rep.rows)
