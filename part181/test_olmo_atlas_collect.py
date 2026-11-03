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


def test_config_change_refuses_resume(setup):
    tmp, tok, model = setup
    out = tmp / "r"
    oac.collect(args_for(tmp, out, stop_after_docs=4), tok=tok, model=model)
    with pytest.raises(SystemExit):
        oac.collect(args_for(tmp, out, batch_size=3), tok=tok, model=model)


def test_export_format(setup):
    tmp, tok, model = setup
    out = tmp / "e"
    oac.collect(args_for(tmp, out), tok=tok, model=model)
    counts = oac.export(out, tmp / "c.jsonl", tok)
    assert counts == {"neurons": WIDTH * 2, "controls": 5 * 2}
    rows = [json.loads(l) for l in (out / "neurons.jsonl").open()]
    r = rows[0]
    assert set(r) == {"layer", "unit", "max", "frac_above", "top", "random", "weak", "model_arm"}
    assert r["layer"] == 0 and r["unit"] == 0 and 0 <= r["frac_above"] <= 1
    assert r["top"][0]["window"].count("[[") == 1 and math.isclose(r["top"][0]["act"], r["max"], rel_tol=1e-6)
    ctrl = json.loads((out / "controls.jsonl").open().readline())
    assert ctrl["control_id"] == 0 and "unit" not in ctrl
