#!/usr/bin/env python3
"""Host tests for part165/reading165.py.

Everything runs on the CPU with a tiny Qwen3 (and the real Qwen tokenizer). The
Arc job is not run; these pin the contract:

1. the direction builder is identical to Part 165's own;
2. the frozen contexts truncate to exactly 128 tokens (the critical cold-read
   finding), and a short document refuses loudly;
3. the capture reads the block output, and a push at down_proj moves it by W d;
4. the geometry and rank primitives are right;
5. the whole CLI runs and writes its artefacts;
6. one failure case per gate, and a failed gate leaves every bar unread.
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


def test_frozen_texts_encode_to_128():
    """Cold read F1: every frozen row must truncate to exactly 128 tokens."""
    if not QWEN_SNAPSHOT:
        pytest.skip("the Qwen tokenizer snapshot is needed")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(QWEN_SNAPSHOT[0], local_files_only=True)
    payload = json.loads((ROOT / "part165" / "reading-texts.json").read_text())
    documents = R.encode_documents(payload["texts"], tokenizer)
    assert len(documents) == 16
    assert all(len(ids) == R.SEQ for ids in documents)


def test_short_document_refuses():
    if not QWEN_SNAPSHOT:
        pytest.skip("the Qwen tokenizer snapshot is needed")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(QWEN_SNAPSHOT[0], local_files_only=True)
    with pytest.raises(ValueError, match="expected 128"):
        R.encode_documents(["hello"], tokenizer)


def test_capture_reads_the_block_output():
    """An interior layer must equal the model's own hidden_states entry."""
    model = tiny_model()
    ids = torch.randint(0, VOCAB, (12,))
    decoder, _parent = R.find_text_decoder(model)
    states = R.capture(model, decoder, ids, [0], "cpu")
    with torch.no_grad():
        out = model(input_ids=ids[None], output_hidden_states=True, use_cache=False)
    assert torch.equal(states[0], out.hidden_states[1][0].float())


def test_site_check_sees_the_push():
    """A push at down_proj must move the block output by W d, and a wrong w must fail."""
    model = tiny_model()
    decoder, _parent = R.find_text_decoder(model)
    weight = decoder[0].mlp.down_proj.weight
    d, w = R.build_directions(weight, 0, 0, 1.0, {})["n+"]
    ids = torch.randint(0, VOCAB, (12,))
    assert R.site_check(model, decoder, 0, d, w, ids, "cpu") <= 5e-3
    assert R.site_check(model, decoder, 0, d, w * 2, ids, "cpu") > 5e-3


def test_rankdata_and_spearman():
    assert list(R.rankdata([10, 10, 20])) == [1.5, 1.5, 3.0]
    assert R.spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert R.spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)


def test_geometry_concentration():
    d = 8
    X = torch.randn(5, d)
    mu, pcs, _top, _lo, _hi = R.pca(X, 4)
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
    """A tiny checkpoint plus the inputs the job reads."""
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
    (tmp_path / "texts.json").write_text(json.dumps(
        {"corpus_sha256": "test", "rows": [1, 2], "count": len(texts), "texts": texts}))
    return tmp_path, layers, slots


def _run(job, out_name="out", norms=None, effects=None, texts=None):
    tmp_path, layers, slots = job
    return R.main([
        "--model", str(tmp_path / "model"), "--device", "cpu",
        "--norms", str(norms or (tmp_path / "norms.json")),
        "--effects", str(effects or (tmp_path / "effects.jsonl")),
        "--medians", str(tmp_path / "medians.json"),
        "--texts-file", str(texts or (tmp_path / "texts.json")),
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
    assert result["gates"]["capture"] is True
    assert len(result["slots"]) == 2 * 6
    assert set(result["bars"]) == {"R1", "R2", "R3", "R4", "R5"}
    assert (out / "reading-tables.md").exists()
    first = result["slots"][0]["arms"]["n+"]
    for key in ("pr_w", "cos_mu", "pc16", "proj_abs_median", "mean_abs_cos_vocab"):
        assert key in first
    assert len(first["questions"]) == 8
    assert result["config"]["readout_key"] == " Alibaba"
    assert " Alibaba" in result["config"]["readout_strings"]


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
    assert result["bars"]["R1"]["pass"] is None
    assert result["correlations"] == {}


def test_nonfinite_effect_fails_gate(tiny_job):
    tmp_path, _, _ = tiny_job
    broken = tmp_path / "effects-broken.jsonl"
    rows = [json.loads(line) for line in (tmp_path / "effects.jsonl").read_text().splitlines()]
    rows[0]["kl"] = float("nan")
    broken.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    assert _run(tiny_job, out_name="out-nan", effects=broken) == 2
    result = json.loads((tmp_path / "out-nan" / "reading.json").read_text())
    assert result["gates"]["effects"] is False
    assert result["bars"]["R2"]["pass"] is None


def test_overlapping_context_fails_gate(tiny_job):
    tmp_path, _, _ = tiny_job
    broken = tmp_path / "texts-overlap.json"
    payload = json.loads((tmp_path / "texts.json").read_text())
    payload["texts"][0] = payload["texts"][0] + " What is question number 3?"
    broken.write_text(json.dumps(payload))
    assert _run(tiny_job, out_name="out-overlap", texts=broken) == 2
    result = json.loads((tmp_path / "out-overlap" / "reading.json").read_text())
    assert result["gates"]["contexts"] is False


def test_broken_pca_fails_gate(tiny_job, monkeypatch):
    real_pca = R.pca

    def bad_pca(X, k=16):
        mu, pcs, top, low, high = real_pca(X, k)
        return mu, pcs * 2.0, top, low, high

    monkeypatch.setattr(R, "pca", bad_pca)
    assert _run(tiny_job, out_name="out-pca") == 2
    result = json.loads((Path(tiny_job[0]) / "out-pca" / "reading.json").read_text())
    assert result["gates"]["measures"] is False


def test_wrong_site_fails_gate(tiny_job, monkeypatch):
    monkeypatch.setattr(R, "site_check", lambda *args, **kwargs: 1.0)
    assert _run(tiny_job, out_name="out-site") == 2
    result = json.loads((Path(tiny_job[0]) / "out-site" / "reading.json").read_text())
    assert result["gates"]["capture"] is False


def synthetic_rows(slope=-0.4, intercept=0.9, layer_effects=None, layer_mean=None,
                   n_per_layer=40, seed=1):
    """Rows from a known model: log kl = intercept + slope log PR + layer effect."""
    import numpy as np

    effects = layer_effects or {19: 0.0, 20: 0.3, 21: -0.2, 22: 0.7, 23: 0.1}
    means = layer_mean or {}
    rng = np.random.default_rng(seed)
    rows = []
    for layer, effect in effects.items():
        for _ in range(n_per_layer):
            log_pr = float(rng.normal(means.get(layer, 0.0), 1.0))
            log_kl = intercept + slope * log_pr + effect
            rows.append({"layer": layer, "pr_w": float(np.exp(log_pr)),
                         "kl": float(np.exp(log_kl))})
    return rows, effects


def test_gap_ols_recovers_known_coefficients():
    """R5's registered estimator recovers a slope and intercept it was built with.

    The pre-recheck design matrix had no constant, which pins the lowest layer's
    fixed effect to zero; this construction has a non-zero intercept, so a
    constant-free fit cannot return both coefficients (N1).
    """
    truth_slope, truth_intercept = -0.4, 0.9
    rows, effects = synthetic_rows(truth_slope, truth_intercept)
    slope, beta = R.gap_ols(rows)
    assert slope == pytest.approx(truth_slope, abs=0.01)
    assert beta[1] == pytest.approx(truth_intercept, abs=0.01)
    # Each layer's dummy is that layer's effect relative to the reference (19).
    reference = min(effects)
    for index, layer in enumerate(sorted(effects)[1:], start=2):
        assert beta[index] == pytest.approx(effects[layer] - effects[reference], abs=0.01)


def test_a_constant_free_fit_is_biased():
    """The control: without the ones column the slope misses the truth.

    The layers sit at different mean log PR (layer 19 higher, as real layers do),
    which is what makes the omitted constant bite: the no-constant fit pins each
    layer's mean residual to zero and folds the intercept into the slope. With
    balanced, zero-mean layers it would agree, which is why the recovery test
    alone is not enough to show the constant is load-bearing.
    """
    import numpy as np

    rows, _ = synthetic_rows(-0.4, 0.9, layer_mean={19: 1.5})
    layers = sorted({row["layer"] for row in rows})
    log_pr = np.log(np.array([row["pr_w"] for row in rows]))
    log_kl = np.log(np.array([row["kl"] for row in rows]))
    columns = [log_pr]
    for layer in layers[1:]:
        columns.append(np.array([1.0 if row["layer"] == layer else 0.0 for row in rows]))
    beta, *_ = np.linalg.lstsq(np.column_stack(columns), log_kl, rcond=None)
    assert abs(beta[0] - (-0.4)) > 0.02  # the reported slope would be wrong


def test_readout_is_bootstrapped():
    """R4's clustered interval needs ``readout`` among the bootstrapped keys (N2)."""
    assert "readout" in R.BOOTSTRAP_KEYS
    assert "readout" not in R.PRIMARY
    import numpy as np

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    y = np.array([2.0, 1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0])
    low, high = R.cluster_bootstrap(x, y, np.array(["a", "a", "b", "b", "c", "c", "d", "d"]),
                                    draws=200)
    assert low <= high


def test_bar_label_is_three_way():
    """An unmeasured bar is not a failure (N4)."""
    assert R.bar_label({"pass": True}) == "**pass**"
    assert R.bar_label({"pass": False}) == "fail"
    assert R.bar_label({"pass": None}) == "not measured"
    assert R.bar_label({"pass": None, "reason": "gate failed"}) == "not measured"
