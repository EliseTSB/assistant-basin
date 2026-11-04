"""Offline tests for session identity, pinned downloads, and Gate 1."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze181 as a181  # noqa: E402
import olmo_atlas_collect as oac  # noqa: E402
import session181 as s181  # noqa: E402


class ToyTokenizer:
    pad_token_id = 0
    eos_token = "<e>"

    def __call__(self, texts, truncation, max_length, padding, add_special_tokens):
        return {"input_ids": [[1 + (ord(ch) % 31) for ch in text][:max_length] for text in texts]}


def tiny_model():
    from transformers import Olmo2Config, Olmo2ForCausalLM
    torch.manual_seed(193)
    config = Olmo2Config(vocab_size=32, hidden_size=16, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=128,
                         pad_token_id=0)
    return Olmo2ForCausalLM(config).eval()


def test_token_id_sha256_is_pinned():
    assert s181.token_ids_sha256([[1, 2], [], [3]]) == \
        "50dbe573aefcada1236c559641ded68d11177ea9f0467c70ab590b9261c84495"


def test_dense_control_stack_retains_registered_hash():
    hashes = s181.control_hashes(oac, n_sparse_per_k=1)
    assert hashes["dense_sha256"] == s181.REGISTERED_DENSE_CONTROL_SHA256
    assert set(hashes["sparse_by_layer_k_sha256"]) == {str(i) for i in range(16)}
    assert set(hashes["sparse_by_layer_k_sha256"]["0"]) == {"2", "4", "16", "64"}


def test_fake_snapshot_download_receives_full_registered_revision(tmp_path):
    prefix = "48d788eca847"
    commit = prefix + "0" * (40 - len(prefix))
    checkpoint = {"key": "instruct", "repo_id": "allenai/OLMo-2-0425-1B-Instruct",
                  "commit_prefix": prefix, "commit": commit}
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path / "snapshot")

    result = s181.download_model_snapshot(checkpoint, tmp_path / "work", fake_snapshot_download)
    assert result == tmp_path / "snapshot"
    assert calls == [{"repo_id": checkpoint["repo_id"], "revision": commit,
                      "cache_dir": str(tmp_path / "work" / "hf-cache")}]


def _tiny_collected_state(tmp_path):
    texts = ["abcdefghi", "klmnopqrstuv", "wxyzabcdefg", "hijklmnopq", "rstuvwxyzab",
             "cdefghijkl", "mnopqrstuvw", "xyzabcdefgh", "ijklmnopqrs", "tuvwxyzabcd"]
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps({"text": t}) + "\n" for t in texts), encoding="utf-8")
    tokenizer = ToyTokenizer()
    model = tiny_model()
    args = argparse.Namespace(
        model_dir=None, corpus=str(corpus), out=str(tmp_path / "run"), docs=None, pass2_docs=None,
        passes=1, dtype="fp32", batch_size=2, checkpoint_every=4, threads=1, seed=5601,
        n_controls=0, layers=[0, 1], model_arm="toy", forward_only=False, snapshot_every=0,
        stop_after_docs=None, strict_shape=False,
    )
    oac.collect(args, tok=tokenizer, model=model)
    texts = oac.read_corpus_texts(corpus, None)
    token_ids = [doc["ids"] for doc in oac.tokenize_docs(tokenizer, texts)]
    with np.load(tmp_path / "run" / "state.npz", allow_pickle=False) as saved:
        state = {key: saved[key] for key in saved.files}
    return model, token_ids, state


def test_gate1_green_and_perturbed_top1_red(tmp_path):
    model, token_ids, state = _tiny_collected_state(tmp_path)
    green = s181.run_gate1(model, token_ids, state, layers=[0, 1], width=64, sample_size=32)
    assert green["verdict"] == "passed"
    assert len(green["pairs"]) == 32
    first = green["pairs"][0]
    assert first["relative_difference"] <= 1e-4

    # The same seeded pair is selected. Change only its stored collector value by 1e-3 relative.
    mutated = {key: value.copy() for key, value in state.items()}
    slot = [0, 1].index(first["layer"])
    mutated["top_act"][slot, first["unit"], 0] *= 1.001
    red = s181.run_gate1(model, token_ids, mutated, layers=[0, 1], width=64, sample_size=32)
    assert red["verdict"] == "failed"
    assert any(row["relative_difference"] > 1e-4 for row in red["pairs"])


def _gate_fixture(tmp_path, doc_count=24):
    texts = ["a" * (3 + (i * 7) % 31) + chr(97 + i % 26) for i in range(doc_count)]
    corpus = tmp_path / "gate-corpus.jsonl"
    corpus.write_text("".join(json.dumps({"text": text}) + "\n" for text in texts), encoding="utf-8")
    tok, model = ToyTokenizer(), tiny_model()
    args = argparse.Namespace(
        model_dir=None, corpus=str(corpus), out=str(tmp_path / "gate-run"), docs=doc_count,
        pass2_docs=None, passes=1, dtype="fp32", batch_size=4, checkpoint_every=100,
        threads=1, seed=5601, n_controls=0, layers=[0, 1], model_arm="toy", forward_only=False,
        snapshot_every=0, stop_after_docs=None, strict_shape=False, n_sparse_per_k=0,
        manifest_sha256="fixture-manifest",
    )
    oac.collect(args, tok=tok, model=model)
    token_ids = [doc["ids"] for doc in oac.tokenize_docs(tok, texts)]
    with np.load(tmp_path / "gate-run" / "state.npz", allow_pickle=False) as saved:
        state = {key: saved[key] for key in saved.files}
    return model, token_ids, state


def test_gate1b_detects_perturbed_subset_membership(tmp_path):
    model, token_ids, state = _gate_fixture(tmp_path, doc_count=24)
    controls = np.zeros((2, 0, 64), dtype=np.float32)
    green = s181._gate1b(model, token_ids, state, [0, 1], 64, controls)
    assert green["verdict"] == "passed"
    changed = {key: value.copy() for key, value in state.items()}
    current = set(int(x) for x in changed["top_doc"][0, 0] if x >= 0)
    outside = next(doc for doc in range(len(token_ids)) if doc not in current)
    changed["top_doc"][0, 0, 0] = outside
    red = s181._gate1b(model, token_ids, changed, [0, 1], 64, controls)
    assert red["verdict"] == "failed"
    assert red["failed_units"] >= 1


def test_gate1d_detects_changed_value_for_a_listed_document(tmp_path):
    model, token_ids, state = _gate_fixture(tmp_path, doc_count=20)
    controls = np.zeros((2, 0, 64), dtype=np.float32)
    green = s181._gate1d(model, token_ids, state, [0, 1], 64, controls)
    assert green["verdict"] == "passed"

    rng = np.random.default_rng(s181.GATE_SEED)
    selected = int(rng.choice(np.arange(64), size=2, replace=False)[0])
    doc = int(rng.integers(len(token_ids)))
    rank = int(np.flatnonzero(state["top_doc"][0, selected] == doc)[0])
    changed = {key: value.copy() for key, value in state.items()}
    changed["top_act"][0, selected, rank] += 0.1
    red = s181._gate1d(model, token_ids, changed, [0, 1], 64, controls)
    assert red["verdict"] == "failed"
    assert any(row["unit"] == selected and row["doc"] == doc and not row["pass"]
               for row in red["entries"])


def test_bind_identity_allows_a_host_change_and_refuses_anything_else(tmp_path):
    # 2026-11-03: three cloud resumes were refused because the CPU model is in the manifest.
    import copy
    manifest = {"key": "k", "model_commit": "m", "dataset_commit": "d", "token_ids_sha256": "t",
                "controls_sha256_by_layer": ["c"], "settings": {"batch_size": 4},
                "host": {"cpu_model": "Xeon @ 2.10GHz"}, "manifest_sha256": "first-sha"}
    out = tmp_path / "run"
    out.mkdir()
    assert s181.bind_identity(out, copy.deepcopy(manifest)) == "first-sha"
    (out / "state.npz").write_bytes(b"x")
    moved = copy.deepcopy(manifest)
    moved["host"] = {"cpu_model": "Xeon @ 2.80GHz"}
    moved["manifest_sha256"] = "second-sha"
    assert s181.bind_identity(out, moved) == "first-sha"
    history = (out / "host-history.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(history) == 1 and "2.80GHz" in history[0]
    assert json.loads((out / "identity.json").read_text(encoding="utf-8"))["host"]["cpu_model"].endswith("2.10GHz")
    for field, value in (("model_commit", "other"), ("settings", {"batch_size": 8})):
        bad = copy.deepcopy(moved)
        bad[field] = value
        with pytest.raises(ValueError, match="identity differs"):
            s181.bind_identity(out, bad)
    fresh = tmp_path / "orphan"
    fresh.mkdir()
    (fresh / "state.npz").write_bytes(b"x")
    with pytest.raises(ValueError, match="without identity.json"):
        s181.bind_identity(fresh, copy.deepcopy(manifest))
