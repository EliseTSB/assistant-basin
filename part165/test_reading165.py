#!/usr/bin/env python3
"""Host tests for part165/reading165.py.

Everything here runs on the CPU with a tiny Qwen3. The Arc job itself is not run;
these tests pin the parts that can be checked without it:

1. the direction builder is identical to Part 165's own;
2. the hidden-state capture reads the block output the push reaches;
3. the geometry and rank primitives are right;
4. the whole CLI runs end to end and writes its two artefacts;
5. a corrupted norms file fails gate G1 and exits 2 (the gate is load-bearing).
"""
from __future__ import annotations

import glob
import json
import shutil
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for entry in (str(ROOT), str(HERE)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import reading165 as R  # noqa: E402
from part165.run165 import build_directions as reference_build  # noqa: E402

QWEN_SNAPSHOT = glob.glob("/mnt/e/neuron-probe/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/*")
HIDDEN, INTER, LAYERS, VOCAB = 32, 64, 2, 128


def tiny_model():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=INTER,
                         num_hidden_layers=LAYERS, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=8, tie_word_embeddings=True,
                         eos_token_id=None)
    torch.manual_seed(0)
    return Qwen3ForCausalLM(config).to(torch.float32).eval()


def test_direction_builder_matches_part165():
    model = tiny_model()
    weight = model.model.layers[0].mlp.down_proj.weight
    cache = {}
    for slot in (0, 1, 2):
        mine = R.build_directions(weight, 0, slot, 1.0, cache)
        reference, metrics = reference_build(weight, 0, slot, 1.0, {})
        for arm in R.ARMS:
            assert torch.equal(mine[arm][0], reference[arm]), f"{arm} slot {slot} differs"
            assert mine[arm][1].norm().item() == pytest.approx(
                metrics[arm]["write_norm"], rel=1e-6)


def test_capture_reads_the_block_output():
    """An interior layer must equal the model's own hidden_states entry."""
    model = tiny_model()
    ids = torch.randint(0, VOCAB, (12,))
    states = R.capture(model, ids, [0], "cpu")
    with torch.no_grad():
        out = model(input_ids=ids[None], output_hidden_states=True, use_cache=False)
    assert torch.equal(states[0], out.hidden_states[1][0].float())


def test_rankdata_and_spearman():
    assert list(R.rankdata([10, 10, 20])) == [1.5, 1.5, 3.0]
    assert R.spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert R.spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)


def test_geometry_concentration():
    d = 8
    X = torch.randn(5, d)
    mu, pcs, _ = R.pca(X, 4)
    one_hot = torch.zeros(d)
    one_hot[3] = 1.0
    row = R.geometry(one_hot, X, mu, pcs, {}, None)
    assert row["pr_w"] == pytest.approx(1.0)
    assert row["mass50"] == 1
    assert row["max_share"] == pytest.approx(1.0)
    uniform = torch.ones(d)
    assert R.geometry(uniform, X, mu, pcs, {}, None)["pr_w"] == pytest.approx(d)


@pytest.fixture()
def tiny_job(tmp_path):
    """A tiny checkpoint plus the four inputs the job reads."""
    model = tiny_model()
    snapshot = tmp_path / "model"
    snapshot.mkdir()
    model.save_pretrained(snapshot, safe_serialization=True)
    source = QWEN_SNAPSHOT[0] if QWEN_SNAPSHOT else None
    assert source is not None, "the Qwen tokenizer snapshot is needed for the tiny job"
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        candidate = Path(source) / name
        if candidate.exists():
            shutil.copy(candidate, snapshot / name)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    vocab = len(tokenizer)
    # The saved model's vocabulary must cover the tokenizer's ids.
    if vocab > model.config.vocab_size:
        from transformers import Qwen3Config, Qwen3ForCausalLM
        config = model.config.to_dict()
        config["vocab_size"] = vocab
        torch.manual_seed(0)
        Qwen3ForCausalLM(Qwen3Config(**config)).to(torch.float32).eval().save_pretrained(snapshot)
        model = Qwen3ForCausalLM.from_pretrained(snapshot)

    layers, slots = (0, 1), 6
    medians = {str(L): 1.0 for L in layers}
    (tmp_path / "medians.json").write_text(json.dumps(medians))
    norms, effects = [], []
    cache = {}
    weight = model.model.layers[0].mlp.down_proj.weight
    for L in layers:
        weight = model.model.layers[L].mlp.down_proj.weight
        for slot in range(slots):
            built = R.build_directions(weight, L, slot, 1.0, cache)
            for arm in R.ARMS:
                norms.append({"layer": L, "slot": slot, "arm": arm,
                              "write_norm": float(built[arm][1].norm()),
                              "target_norm": 0.8, "relative_error": 0.0,
                              "iso_residual_norm": 0.0, "iso_residual_fraction": None})
                for question in range(8):
                    effects.append({"stage": "effect", "question_index": question,
                                    "layer": L, "slot": slot, "arm": arm,
                                    "kl": 0.05 + 0.01 * ((slot * 8 + question) % 7),
                                    "rel_change": 0.7})
    (tmp_path / "norms.json").write_text(json.dumps({"part": 165, "dose": 0.8,
                                                     "directions": norms}))
    (tmp_path / "effects.jsonl").write_text(
        "\n".join(json.dumps(row) for row in effects) + "\n")
    questions = [f"What is question number {i}?" for i in range(8)]
    (tmp_path / "questions.json").write_text(json.dumps(
        {"source": "test", "count": 8, "questions": questions}))
    texts = ["The quick brown fox jumps over the lazy dog. " * 40,
             "A second passage about rivers and mountains and weather. " * 40]
    (tmp_path / "texts.json").write_text(json.dumps(texts))
    return tmp_path, layers, slots


def _run(job, out_name="out", norms=None):
    tmp_path, layers, slots = job
    return R.main([
        "--model", str(tmp_path / "model"), "--device", "cpu",
        "--norms", str(norms or (tmp_path / "norms.json")),
        "--effects", str(tmp_path / "effects.jsonl"),
        "--medians", str(tmp_path / "medians.json"),
        "--texts-file", str(tmp_path / "texts.json"),
        "--questions", str(tmp_path / "questions.json"),
        "--out", str(tmp_path / out_name),
        "--docs", "2", "--layers", ",".join(str(x) for x in layers),
        "--slots", str(slots), "--vocab-sample", "16", "--bootstrap", "50",
        "--nulls", "256",
    ])


def test_end_to_end_tiny(tiny_job):
    assert _run(tiny_job) == 0
    out = Path(tiny_job[0]) / "out"
    result = json.loads((out / "reading.json").read_text())
    assert result["gates"]["passed"] is True
    assert result["gates"]["directions"] is True
    assert len(result["slots"]) == 2 * 6
    assert set(result["bars"]) == {"R1", "R2", "R3", "R4", "R5"}
    assert (out / "reading-tables.md").exists()
    first = result["slots"][0]["arms"]["n+"]
    for key in ("pr_w", "cos_mu", "pc16", "proj_abs_median", "mean_abs_cos_vocab"):
        assert key in first
    assert len(first["questions"]) == 8
    assert result["config"]["readout_strings"], "no readout string survived tokenisation"


def test_corrupted_norms_fail_gate(tiny_job):
    tmp_path, _, _ = tiny_job
    broken = tmp_path / "norms-broken.json"
    payload = json.loads((tmp_path / "norms.json").read_text())
    payload["directions"][0]["write_norm"] *= 1.1
    broken.write_text(json.dumps(payload))
    assert _run(tiny_job, out_name="out-broken", norms=broken) == 2
    result = json.loads((tmp_path / "out-broken" / "reading.json").read_text())
    assert result["gates"]["directions"] is False
    assert result["gates"]["passed"] is False
