"""Collector tests on a 2-layer slice of a tiny random OLMo 2 (no downloads)."""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import olmo_atlas_collect as oac  # noqa: E402

VOCAB = 50
WIDTH = 24
LAYERS = [0, 2]   # the slice: 2 of the tiny model's 3 layers


class ToyTokenizer:
    pad_token_id = 0
    eos_token = "<e>"

    def __call__(self, texts, truncation, max_length, padding, add_special_tokens):
        return {"input_ids": [[1 + (ord(ch) % (VOCAB - 1)) for ch in t][:max_length] for t in texts]}

    def decode(self, ids, **_kw):
        return "".join(chr(ord("a") + (i % 26)) for i in ids)


def tiny_model():
    from transformers import Olmo2Config, Olmo2ForCausalLM
    torch.manual_seed(0)
    cfg = Olmo2Config(vocab_size=VOCAB, hidden_size=16, intermediate_size=WIDTH, num_hidden_layers=3,
                      num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=128,
                      pad_token_id=0)
    return Olmo2ForCausalLM(cfg).eval()


def make_corpus(path: Path, n=11):
    rng = np.random.default_rng(1)
    with path.open("w") as f:
        for i in range(n):
            length = int(rng.integers(3, 40))
            f.write(json.dumps({"text": "".join(chr(97 + int(c)) for c in rng.integers(0, 26, length))}) + "\n")


def args_for(tmp, out, **kw):
    base = dict(model_dir=None, corpus=str(tmp / "c.jsonl"), out=str(out), docs=None, pass2_docs=None,
                passes=2, dtype="fp32", batch_size=2, checkpoint_every=4, threads=1, seed=7, n_controls=5,
                layers=LAYERS, model_arm="trained", forward_only=False, snapshot_every=0,
                stop_after_docs=None, strict_shape=False)
    base.update(kw)
    return argparse.Namespace(**base)


def brute_force(model, tok, texts, controls):
    """Full per-token activations for every doc (no batching) -> dict layer -> list of [T,U]."""
    acts = {li: [] for li in LAYERS}
    for t in texts:
        ids = torch.tensor([tok([t], True, 512, False, True)["input_ids"][0]])
        store = {}
        hs = [model.model.layers[li].mlp.down_proj.register_forward_pre_hook(
            lambda _m, a, li=li: store.__setitem__(li, a[0][0].clone())) for li in LAYERS]
        with torch.no_grad():
            model(input_ids=ids)
        for h in hs:
            h.remove()
        for s, li in enumerate(LAYERS):
            x = store[li]
            acts[li].append(torch.cat((x, x @ torch.from_numpy(controls[s]).T), dim=1).numpy())
    return acts


@pytest.fixture()
def setup(tmp_path):
    make_corpus(tmp_path / "c.jsonl")
    return tmp_path, ToyTokenizer(), tiny_model()


def load_state(out):
    with np.load(out / "state.npz") as s:
        return {k: s[k] for k in s.files}


def test_matches_brute_force(setup):
    tmp, tok, model = setup
    out = tmp / "run"
    oac.collect(args_for(tmp, out), tok=tok, model=model)
    st = load_state(out)
    texts = oac.read_corpus_texts(tmp / "c.jsonl", None)
    controls = np.stack([oac.generate_controls(5, WIDTH, 7, li) for li in LAYERS])
    ref = brute_force(model, tok, texts, controls)
    for s, li in enumerate(LAYERS):
        per_doc = np.stack([a.max(axis=0) for a in ref[li]])  # [D,U]
        np.testing.assert_allclose(st["max"][s], per_doc.max(axis=0), rtol=1e-4, atol=1e-5)
        k = min(oac.TOP_K, len(texts))
        for u in range(WIDTH + 5):
            expect = np.sort(per_doc[:, u])[::-1][:k]
            np.testing.assert_allclose(st["top_act"][s, u, :k], expect, rtol=1e-4, atol=1e-5)
            d, p = st["top_doc"][s, u, 0], st["top_pos"][s, u, 0]
            assert abs(ref[li][d][p, u] - st["top_act"][s, u, 0]) < 1e-4
        allv = np.concatenate(ref[li])
        expect_above = (allv > 0.1 * st["max"][s]).sum(axis=0)
        np.testing.assert_array_equal(st["above"][s], expect_above)
        # every random-reservoir entry is a real qualifying position with the right activation
        for u in range(WIDTH + 5):
            for d, p, a in zip(st["random_doc"][s, u], st["random_pos"][s, u], st["random_act"][s, u]):
                if d < 0:
                    continue
                assert abs(ref[li][d][p, u] - a) < 1e-4 and a > 0.1 * st["max"][s, u]
    assert int(st["positions"]) == sum(len(a) for a in ref[LAYERS[0]])


def _assert_every_top_rank_matches(st, reference, layers, width, control_count):
    for slot, li in enumerate(layers):
        per_doc = np.stack([activation.max(axis=0) for activation in reference[li]])
        for unit in range(width + control_count):
            expected = np.sort(per_doc[:, unit])[::-1][:oac.TOP_K]
            np.testing.assert_allclose(st["top_act"][slot, unit], expected, rtol=1e-5, atol=1e-6)
            selected_docs = st["top_doc"][slot, unit]
            selected_pos = st["top_pos"][slot, unit]
            assert len(set(selected_docs.tolist())) == oac.TOP_K
            for rank in range(oac.TOP_K):
                doc, pos = int(selected_docs[rank]), int(selected_pos[rank])
                assert 0 <= doc < len(reference[li])
                assert 0 <= pos < len(reference[li][doc])
                assert reference[li][doc][pos, unit] == pytest.approx(st["top_act"][slot, unit, rank], abs=1e-6)


def test_brute_force_top20_ties_mixed_lengths_and_three_mutations(tmp_path):
    tok, model = ToyTokenizer(), tiny_model()
    rng = np.random.default_rng(2026)
    texts = ["same", "same"] + ["".join(chr(97 + int(c)) for c in rng.integers(0, 26, int(n)))
                                  for n in rng.integers(3, 38, 23)]
    corpus = tmp_path / "many.jsonl"
    corpus.write_text("".join(json.dumps({"text": t}) + "\n" for t in texts), encoding="utf-8")
    out = tmp_path / "many-run"
    oac.collect(args_for(tmp_path, out, corpus=str(corpus), docs=25, passes=1, n_controls=0,
                         checkpoint_every=25, batch_size=4), tok=tok, model=model)
    st = load_state(out)
    controls = np.zeros((len(LAYERS), 0, WIDTH), dtype=np.float32)
    ref = brute_force(model, tok, texts, controls)
    _assert_every_top_rank_matches(st, ref, LAYERS, WIDTH, 0)

    dropped = {key: value.copy() for key, value in st.items()}
    selected_docs = set(dropped["top_doc"][0, 0].tolist())
    omitted_doc = next(
        doc for doc in range(len(ref[LAYERS[0]]))
        if doc not in selected_docs
        and float(ref[LAYERS[0]][doc][:, 0].max()) < float(st["top_act"][0, 0, 0])
    )
    dropped["top_doc"][0, 0, 0] = omitted_doc
    dropped["top_pos"][0, 0, 0] = int(np.argmax(ref[LAYERS[0]][omitted_doc][:, 0]))
    with pytest.raises(AssertionError):
        _assert_every_top_rank_matches(dropped, ref, LAYERS, WIDTH, 0)

    wrong_position = {key: value.copy() for key, value in st.items()}
    selected_doc = int(wrong_position["top_doc"][0, 0, 0])
    old_pos = int(wrong_position["top_pos"][0, 0, 0])
    wrong_position["top_pos"][0, 0, 0] = next(
        pos for pos in range(len(ref[LAYERS[0]][selected_doc]))
        if pos != old_pos and ref[LAYERS[0]][selected_doc][pos, 0] != st["top_act"][0, 0, 0]
    )
    with pytest.raises(AssertionError):
        _assert_every_top_rank_matches(wrong_position, ref, LAYERS, WIDTH, 0)

    duplicate = {key: value.copy() for key, value in st.items()}
    duplicate["top_doc"][0, 0, 1] = duplicate["top_doc"][0, 0, 0]
    with pytest.raises(AssertionError):
        _assert_every_top_rank_matches(duplicate, ref, LAYERS, WIDTH, 0)


def test_sparse_controls_have_seeded_support_and_unit_norm():
    for k in (2, 4, 16, 64):
        first = oac.generate_sparse_controls(64, 128, k, seed=5601, layer=7)
        second = oac.generate_sparse_controls(64, 128, k, seed=5601, layer=7)
        np.testing.assert_array_equal(first, second)
        assert first.shape == (64, 128)
        assert np.all(np.count_nonzero(first, axis=1) == k)
        np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0, rtol=1e-6, atol=1e-6)
    assert not np.array_equal(oac.generate_sparse_controls(64, 128, 2, layer=7),
                              oac.generate_sparse_controls(64, 128, 2, layer=8))


def test_resume_after_kill_is_identical(setup):
    tmp, tok, model = setup
    full = tmp / "full"
    oac.collect(args_for(tmp, full), tok=tok, model=model)
    part = tmp / "part"
    prog = oac.collect(args_for(tmp, part, stop_after_docs=4), tok=tok, model=model)
    assert not prog.get("complete")
    oac.collect(args_for(tmp, part), tok=tok, model=model)
    a, b = load_state(full), load_state(part)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)


def test_snapshots_are_written_at_requested_document_cutoffs(setup):
    tmp, tok, model = setup
    out = tmp / "snapshots"
    oac.collect(args_for(tmp, out, passes=1, docs=11, checkpoint_every=4, snapshot_every=3),
                tok=tok, model=model)
    snapshots = sorted(p.name for p in out.glob("snapshot-pass1-docs*.npz"))
    assert snapshots == [
        "snapshot-pass1-docs00003.npz",
        "snapshot-pass1-docs00006.npz",
        "snapshot-pass1-docs00009.npz",
    ]


def test_config_change_refuses_resume(setup):
    tmp, tok, model = setup
    out = tmp / "r"
    oac.collect(args_for(tmp, out, stop_after_docs=4), tok=tok, model=model)
    with pytest.raises(SystemExit):
        oac.collect(args_for(tmp, out, batch_size=3), tok=tok, model=model)


def test_manifest_hash_is_part_of_resume_identity(setup):
    tmp, tok, model = setup
    out = tmp / "manifest-run"
    oac.collect(args_for(tmp, out, stop_after_docs=4, manifest_sha256="a" * 64), tok=tok, model=model)
    with pytest.raises(SystemExit, match="different config"):
        oac.collect(args_for(tmp, out, manifest_sha256="b" * 64), tok=tok, model=model)


def test_export_format(setup, monkeypatch):
    tmp, tok, model = setup
    # The toy model's MLP is 24 wide, so a support of 64 cannot exist; the real width is 8192 (all four sizes).
    monkeypatch.setattr(oac, "SPARSE_SUPPORTS", (2, 4, 16))
    out = tmp / "e"
    oac.collect(args_for(tmp, out, n_sparse_per_k=1), tok=tok, model=model)
    counts = oac.export(out, tmp / "c.jsonl", tok)
    assert counts == {"neurons": WIDTH * 2, "controls": (9 - 4 + len(oac.SPARSE_SUPPORTS)) * 2}
    rows = [json.loads(l) for l in (out / "neurons.jsonl").open()]
    r = rows[0]
    assert set(r) == {"layer", "unit", "unit_kind", "k", "max", "frac_above",
                      "frac_above_unavailable_nonpositive_max", "top", "random", "weak", "model_arm"}
    assert r["layer"] == 0 and r["unit"] == 0 and 0 <= r["frac_above"] <= 1
    assert r["unit_kind"] == "neuron" and r["k"] == 1
    assert r["top"][0]["window"].count("[[") == 1 and math.isclose(r["top"][0]["act"], r["max"], rel_tol=1e-6)
    ctrl = json.loads((out / "controls.jsonl").open().readline())
    assert ctrl["control_id"] == 0 and ctrl["control_kind"] == "dense" and ctrl["k"] == 0 and "unit" not in ctrl
    controls = [json.loads(line) for line in (out / "controls.jsonl").open(encoding="utf-8")]
    sparse = next(row for row in controls if row["control_id"] == 5)
    assert sparse["control_kind"] == "sparse" and sparse["k"] == 2


def test_export_marks_nonpositive_maximum_as_unavailable(setup):
    tmp, tok, model = setup
    out = tmp / "nonpositive"
    oac.collect(args_for(tmp, out), tok=tok, model=model)
    with np.load(out / "state.npz", allow_pickle=False) as saved:
        state = {key: saved[key] for key in saved.files}
    state["max"][0, 0] = -0.1
    state["above"][0, 0] = 0
    np.savez(out / "state.npz", **state)
    oac.export(out, tmp / "c.jsonl", tok)
    first = json.loads((out / "neurons.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["frac_above"] is None
    assert first["frac_above_unavailable_nonpositive_max"] is True
