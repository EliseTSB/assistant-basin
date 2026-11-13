#!/usr/bin/env python3
"""Tests for record211.py, text211.py, checkpoints211.py and run211.py, on TINY random models built from config.

A 2-layer GPT-NeoX (Pythia's architecture) and a 2-layer OLMo 2, a few heads, a tiny vocabulary; nothing is downloaded.

  /home/elisepc/neuron-probe/.venv/bin/python -I -B -m pytest -q part211/test_record211.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import checkpoints211 as C  # noqa: E402
import measures211 as MS  # noqa: E402
import record211 as R  # noqa: E402
import run211 as RUN  # noqa: E402
import text211 as TX  # noqa: E402

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

VOCAB, D, H, HD, FFN, NLAYER, T = 37, 24, 3, 8, 40, 2, 12


def tiny_neox(seed=0):
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    torch.manual_seed(seed)
    cfg = GPTNeoXConfig(vocab_size=VOCAB, hidden_size=D, num_hidden_layers=NLAYER, num_attention_heads=H,
                        intermediate_size=FFN, max_position_embeddings=64, attn_implementation="eager")
    return GPTNeoXForCausalLM(cfg).eval()


def tiny_olmo(seed=0):
    from transformers import Olmo2Config, Olmo2ForCausalLM
    torch.manual_seed(seed)
    cfg = Olmo2Config(vocab_size=VOCAB, hidden_size=D, intermediate_size=FFN, num_hidden_layers=NLAYER,
                      num_attention_heads=H, num_key_value_heads=H, head_dim=HD, max_position_embeddings=64,
                      attn_implementation="eager")
    return Olmo2ForCausalLM(cfg).eval()


def docs(n):
    rng = np.random.default_rng(0)
    return [{"index": i, "ids": rng.integers(0, VOCAB, size=T).tolist()} for i in range(n)]


MODELS = {"neox": tiny_neox, "olmo": tiny_olmo}
FAMILY = {"neox": "pythia", "olmo": "olmo"}


# ---------------------------------------------------------------- the recorder on both architectures

@pytest.mark.parametrize("arch", sorted(MODELS))
def test_head_contributions_sum_to_the_attention_block_output(arch):
    model = MODELS[arch]()
    ad = R.adapter_for(model)
    ids = torch.tensor([docs(1)[0]["ids"]])
    caps = ad.capture(ids)
    for L in range(ad.n_layers()):
        contrib = caps["contrib"][L][0].float()           # [H, 85, D] (positions 3, 6, ...)
        pos = caps["positions"]
        block = caps["_block_out"][L][0].float()[pos]      # the attention block's output (after the output projection)
        assert torch.allclose(contrib.sum(0), block, atol=1e-5)
        # and the block output is the output projection of the concatenated heads
        assert torch.allclose(block, caps["_concat"][L][0].float()[pos] @ ad.o_weight(L), atol=1e-5)


@pytest.mark.parametrize("arch", sorted(MODELS))
def test_attention_rows_are_causal_softmax_rows(arch):
    model = MODELS[arch]()
    ad = R.adapter_for(model)
    caps = ad.capture(torch.tensor([docs(1)[0]["ids"]]))
    for L in range(ad.n_layers()):
        w = caps["att"][L][0].float()                     # [H, T, T]
        assert torch.allclose(w.sum(-1), torch.ones(w.shape[0], w.shape[1]), atol=1e-5)
        upper = torch.triu(torch.ones(w.shape[1], w.shape[2]), diagonal=1).bool()
        assert w[:, upper].abs().max() == 0.0


@pytest.mark.parametrize("arch", sorted(MODELS))
def test_ov_matrix_shape(arch):
    model = MODELS[arch]()
    ad = R.adapter_for(model)
    ov = ad.ov(0)
    assert ov.shape == (H, D, D)
    assert np.isfinite(ov).all()


def test_adapter_refuses_an_unknown_architecture():
    with pytest.raises(SystemExit):
        R.adapter_for(torch.nn.Linear(2, 2))


# ---------------------------------------------------------------- records: reference and reduced

@pytest.mark.parametrize("arch", sorted(MODELS))
def test_record_checkpoint_writes_rows_for_a_reference_and_reduces_others(arch, tmp_path):
    model = MODELS[arch]()
    d = docs(3)
    n_pos = len(R.positions_for(T))
    ad = R.adapter_for(model)
    hd = ad.head_dim()
    # the run's final reference first (references are recorded first)
    meta_f = R.record_meta("f" * 40, "A", "code1")
    meta_f["reference"] = True
    info_f = R.record_checkpoint(model, d, FAMILY[arch], "R0_step143000", "A", tmp_path, meta_f)
    zf = np.load(info_f["path"])
    assert zf["att"].shape == (3, NLAYER, H, T, T)
    assert zf["z"].shape == (3, NLAYER, H, n_pos, hd)
    assert zf["ov_v"].shape == (NLAYER, H, hd, D) and zf["ov_w"].shape == (NLAYER, H, hd, D)
    assert zf["mlp"].shape == (3, NLAYER, n_pos, D)
    assert np.all(zf["O_final"] == 0) and np.all(zf["V_final"] == 0)
    # nothing the design does not keep: no contributions, no full OV matrices, no per-document moments
    assert not any(k in zf.files for k in ("contrib", "ov")) and not any(k.startswith("mom_") for k in zf.files)
    final = {"key": "R0_step143000", "z": zf["z"], "v": zf["ov_v"], "w": zf["ov_w"]}
    # a non-reference checkpoint: same weights perturbed, reduced against the references and the final
    torch.manual_seed(3)
    with torch.no_grad():
        for prm in model.parameters():
            prm.add_(0.05 * torch.randn_like(prm))
    refs = {"R0_step143000": R.load_att(Path(info_f["path"]))}
    meta = R.record_meta("b" * 40, "A", "code1")
    meta["reference"] = False
    info2 = R.record_checkpoint(model, d, FAMILY[arch], "R0_step128", "A", tmp_path, meta, references=refs,
                                final=final)
    z2 = np.load(info2["path"])
    assert not any(k in z2.files for k in ("att", "z", "ov_v", "ov_w", "contrib", "ov"))
    assert z2["red_R0_step143000_P"].shape == (3, NLAYER, H, H)
    # O and V at record time equal the direct forms on the full contributions and OV matrices
    caps2 = [ad.capture(torch.tensor([x["ids"]])) for x in d]
    v2, w2 = ad.factors(1)
    model_free_final_contrib = np.concatenate([zf["z"][k, 1, 0] @ zf["ov_w"][1, 0] for k in range(3)])
    own_contrib = np.concatenate([c["contrib"][1][0, 0].float().numpy() for c in caps2])
    assert z2["O_final"][1, 0] == pytest.approx(MS.cka_distance(own_contrib, model_free_final_contrib), abs=1e-6)
    want_v = MS.ov_distance(v2[0].T @ w2[0], zf["ov_v"][1, 0].T @ zf["ov_w"][1, 0],
                            zf["ov_v"][1, 0].T @ zf["ov_w"][1, 0])
    assert z2["V_final"][1, 0] == pytest.approx(want_v, rel=1e-5)
    assert z2["V_final"][1, 0] > 0 and z2["O_final"][1, 0] > 0
    # without the final on disk a non-reference checkpoint refuses rather than dropping O and V
    with pytest.raises(RuntimeError):
        R.record_checkpoint(model, d, FAMILY[arch], "R0_step512", "A", tmp_path, meta, references=refs)


@pytest.mark.parametrize("arch", sorted(MODELS))
def test_noise_floor_is_zero_for_bf16_weights_and_positive_otherwise(arch, tmp_path):
    d = docs(3)
    meta = R.record_meta("f" * 40, "A", "code1")
    meta["reference"] = True
    # weights already on the bf16 grid: rounding changes nothing, so every floor is exactly 0
    model = MODELS[arch]()
    R.round_through_bf16(model)
    R.record_checkpoint(model, d, FAMILY[arch], "R0_step143000", "A", tmp_path / "a", meta)
    f = R.record_floor(model, d, "R0_step143000", "A", tmp_path / "a", meta)
    assert f["P"] == [0.0] * NLAYER and f["V"] == [0.0] * NLAYER
    assert max(abs(x) for x in f["M"]) < 1e-12
    # ordinary float32 weights: the rounding moves every measure, by a little
    model = MODELS[arch]()
    torch.manual_seed(9)
    with torch.no_grad():
        for prm in model.parameters():
            prm.add_(1e-3 * torch.randn_like(prm))
    R.record_checkpoint(model, d, FAMILY[arch], "R0_step143000", "A", tmp_path / "b", meta)
    R.round_through_bf16(model)
    f = R.record_floor(model, d, "R0_step143000", "A", tmp_path / "b", meta)
    assert all(0 < x < 0.05 for x in f["P"]) and all(0 < x < 0.05 for x in f["V"])
    assert R.floor_path(tmp_path / "b", "R0_step143000", "A").parent.name == "floors"


@pytest.mark.parametrize("arch", sorted(MODELS))
def test_g2_from_the_model_passes_on_a_repeated_pass_and_fails_on_a_changed_one(arch):
    model = MODELS[arch]()
    d = docs(3)
    g2 = RUN.g2_from_model(model, d)
    assert g2["pass"], g2
    assert all(v["value"] == 0.0 for v in g2["repeat"].values())
    assert all(v["matched"] == 0.0 for v in g2["permutation"].values())
    # a model whose second pass differs (dropout left on) must fail the repeat control
    calls = {"n": 0}
    ad_capture = R.adapter_for(model).__class__.capture

    def noisy(self, ids):
        caps = ad_capture(self, ids)
        calls["n"] += 1
        if calls["n"] > len(d):                                  # the second pass
            caps["mlp"] = [m + 1e-3 * torch.randn_like(m) for m in caps["mlp"]]
        return caps
    cls = R.adapter_for(model).__class__
    cls.capture, saved = noisy, cls.capture
    try:
        g2 = RUN.g2_from_model(model, d)
    finally:
        cls.capture = saved
    assert not g2["pass"] and g2["repeat"]["M"]["value"] > 0


def test_floor_keys_are_the_finals_and_olmo_4T_and_mid_end():
    keys = R.floor_keys()
    assert {C.pythia_key(r, C.FINAL_STEP) for r in C.runs_of()} <= set(keys)
    # 211.8's compared checkpoints (step 10000 of R0 and the weight-seed runs; BUILD-NOTES 9)
    assert {C.pythia_key(r, C.THREAD_STEP) for r in ("R0", *C.runs_with_factor("weights"))} <= set(keys)
    assert "olmo_pretrain_4T" in keys and "olmo_midtrain_51B" in keys
    assert all(k in C.BY_KEY and C.BY_KEY[k]["reference"] for k in keys)


def test_pooled_cka_over_a_resample_with_repeats_is_exact():
    """Brief item 2: CKA computed directly on the stacked raw activations of a resample with repeated documents
    equals the stored-output path to 1e-9 in float64 (the outputs are float32, as stored)."""
    rng = np.random.default_rng(11)
    n_docs, n_layers, n_pos, d = 5, 2, 7, 6
    raw_a = rng.normal(size=(n_docs, n_layers, n_pos, d)).astype(np.float32)
    raw_b = (raw_a + 0.7 * rng.normal(size=raw_a.shape)).astype(np.float32)
    idx = np.array([0, 0, 3, 4, 4])
    for layer in range(n_layers):
        xa = np.concatenate([raw_a[k, layer] for k in idx]).astype(np.float64)
        xb = np.concatenate([raw_b[k, layer] for k in idx]).astype(np.float64)
        xa, xb = xa - xa.mean(0), xb - xb.mean(0)
        want = np.linalg.norm(xa.T @ xb) ** 2 / (np.linalg.norm(xa.T @ xa) * np.linalg.norm(xb.T @ xb))
        assert abs(R.pooled_cka(raw_a, raw_b, idx, layer) - want) < 1e-9
    # centring within each document is NOT the pooled answer: the pool is re-centred as a whole
    per_doc = np.concatenate([raw_a[k, 0] - raw_a[k, 0].mean(0) for k in idx]).astype(np.float64)
    other = np.concatenate([raw_b[k, 0] - raw_b[k, 0].mean(0) for k in idx]).astype(np.float64)
    assert abs(MS.cka(per_doc, other) - R.pooled_cka(raw_a, raw_b, idx, 0)) > 1e-6


def test_final_reference_is_the_last_reference_of_the_run():
    assert R.final_reference("R0_step128") == "R0_step143000"
    assert R.final_reference("D2_step0") == "D2_step143000"
    olmo = [cp["key"] for cp in C.CHECKPOINTS if cp["family"] == C.OLMO_FAMILY and cp["reference"]]
    assert R.final_reference("olmo_step300") == olmo[-1]


def test_recorded_positions_are_3_to_255_by_3():
    assert R.POSITIONS[0] == 3 and R.POSITIONS[-1] == 255 and len(R.POSITIONS) == 85
    assert set(np.diff(R.POSITIONS)) == {3}
    assert len(R.positions_for(T)) == 3 and list(R.positions_for(T)) == [3, 6, 9]


def test_reduce_reference_pair_stores_per_document_per_head_pair_distances(tmp_path):
    model = MODELS["neox"]()
    d = docs(2)
    for key in ("R0_step0", "R0_step10000"):
        meta = R.record_meta("a" * 40, "A", "code1")
        meta["reference"] = True
        R.record_checkpoint(model, d, "pythia", key, "A", tmp_path, meta)
    orig = R.code_hash
    R.code_hash = lambda root=None: "code1"
    try:
        info = R.reduce_reference_pair("R0_step0", "R0_step10000", "A", tmp_path, {"R0_step0": "a" * 40,
                                                                                    "R0_step10000": "a" * 40})
    finally:
        R.code_hash = orig
    assert info and info["ref"] == "R0_step10000"
    z = np.load(R.record_path(tmp_path, "R0_step0", "A"))
    assert z["red_R0_step10000_P"].shape == (2, NLAYER, H, H)
    assert "att" in z.files                                    # the reference keeps its rows
    # the diagonal is i with i (0 here, the same model): an off-diagonal pairing would not be 0
    assert np.abs(np.einsum("dlhh->dlh", z["red_R0_step10000_P"])).max() < 1e-4
    off = z["red_R0_step10000_P"] - np.einsum("dlhh->dlh", z["red_R0_step10000_P"])[:, :, :, None] * np.eye(H)
    assert off.max() > 1e-6                                  # and the off-diagonal entries are not all 0
    # idempotent: a second call does nothing
    R.code_hash = lambda root=None: "code1"
    try:
        assert R.reduce_reference_pair("R0_step0", "R0_step10000", "A", tmp_path,
                                       {"R0_step0": "a" * 40, "R0_step10000": "a" * 40}) is None
    finally:
        R.code_hash = orig


# ---------------------------------------------------------------- the four-field resume check

def _complete_record(tmp_path, meta):
    path = R.record_path(tmp_path, "R0_step0", "A")
    R.save_record(path, {"x": np.zeros(1)}, meta)
    return path


def test_resume_skips_a_complete_record_and_redoes_an_incomplete_one(tmp_path):
    meta = R.record_meta("a" * 40, "A", "code1")
    path = _complete_record(tmp_path, meta)
    assert R.resume_state(path, meta) == "complete"
    # each of the four fields, changed, makes it incomplete
    for field, value in (("commit", "b" * 40), ("text_set", "B"), ("schema", str(R.SCHEMA + 1)), ("code", "code2")):
        other = dict(meta, **{field: value})
        assert R.resume_state(path, other) == "incomplete", field
    assert R.resume_state(tmp_path / "records" / "missing.npz", meta) == "incomplete"


def test_a_record_still_being_written_is_incomplete(tmp_path):
    meta = R.record_meta("a" * 40, "A", "code1")
    path = R.record_path(tmp_path, "R0_step0", "A")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:                       # a torn file, as a killed write leaves
        f.write(b"PK\x03\x04garbage")
    assert R.resume_state(path, meta) == "incomplete"
    path.write_bytes(b"")                             # an empty file
    assert R.resume_state(path, meta) == "incomplete"
    path.unlink()                                     # and a file that is not there at all
    assert R.resume_state(path, meta) == "incomplete"


def test_save_record_renames_into_place_and_leaves_no_temporary(tmp_path):
    meta = R.record_meta("a" * 40, "A", "code1")
    path = _complete_record(tmp_path, meta)
    assert path.is_file()
    assert not list(path.parent.glob("*.tmp"))


def test_record_all_skips_a_complete_record(tmp_path, monkeypatch):
    meta = R.record_meta("a" * 40, "A", "code1")
    meta["reference"] = True
    _complete_record(tmp_path, meta)
    monkeypatch.setattr(R, "code_hash", lambda root=None: "code1")
    monkeypatch.setattr(R, "reduce_reference_pair", lambda *a, **k: None)
    texts = tmp_path / "texts.json"
    texts.write_text(json.dumps({"A": docs(1), "B": docs(1)}))
    called = []
    monkeypatch.setattr(R, "load_model", lambda *a, **k: called.append(a) or None)
    out = RUN.record_all(tmp_path, texts, text_sets=("A",), keys=("R0_step0",), pinned={"R0_step0": "a" * 40})
    assert out == [{"key": "R0_step0", "set": "A", "skipped": True}]
    assert called == []


# ---------------------------------------------------------------- checkpoints

def test_checkpoint_table_is_six_pythia_runs_by_twelve_and_olmo_fourteen():
    runs = C.runs_of()
    assert len(runs) == 6
    assert C.runs_with_factor("weights") == ("W1", "W2")
    assert len(C.runs_with_factor("data")) == 3
    assert len(C.PYTHIA_KEYS) == 6 * 12
    assert len(C.OLMO_KEYS) == 14
    assert all(C.BY_KEY[C.pythia_key(r, s)]["branch"] == f"step{s}" for r in runs for s in C.STEPS)
    # the weight arm is the corrected releases (the uncorrected ones do not vary initialisation)
    assert all("corrected" in C.BY_RUN[r]["repo_id"] for r in C.runs_with_factor("weights"))
    assert not any("weight-seed" in C.BY_RUN[r]["repo_id"] and "corrected" not in C.BY_RUN[r]["repo_id"]
                   for r in runs)


def test_reference_checkpoints_follow_the_design():
    pythia_refs = {cp["key"] for cp in C.CHECKPOINTS if cp["family"] == "pythia" and cp["reference"]}
    assert pythia_refs == {C.pythia_key(r, s) for r in C.runs_of() for s in (0, 10000, 143000)}
    olmo_refs = {cp["key"] for cp in C.CHECKPOINTS if cp["family"] == "olmo" and cp["reference"]}
    assert olmo_refs == {"olmo_step0", "olmo_pretrain_3T", "olmo_pretrain_4T", "olmo_midtrain_51B", "olmo_instruct"}


class FakeApi:
    def __init__(self, prefix="0123456789abcdef"):
        self.calls, self.prefix = [], prefix

    def model_info(self, repo_id, revision=None):
        self.calls.append((repo_id, revision))
        import hashlib
        digest = hashlib.sha1(f"{repo_id}@{revision}".encode()).hexdigest()
        sha = self.prefix + digest[len(self.prefix):] if self.prefix else digest
        return type("Info", (), {"sha": sha})()


def test_pin_asks_for_every_branch_and_writes_the_file(tmp_path):
    class PrefixApi(FakeApi):
        """A fake hub whose hashes start with each checkpoint's registered prefix where there is one."""
        def model_info(self, repo_id, revision=None):
            cp = next(c for c in C.CHECKPOINTS if c["repo_id"] == repo_id and c["branch"] == revision)
            info = super().model_info(repo_id, revision)
            if cp["commit_prefix"]:
                return type("Info", (), {"sha": cp["commit_prefix"] + info.sha[len(cp["commit_prefix"]):]})()
            return info
    api = PrefixApi(prefix="")
    out = C.pin(api=api, path=tmp_path / "checkpoints.json")
    assert len(out["pinned"]) == len(C.CHECKPOINTS)
    assert len(api.calls) == len(C.CHECKPOINTS)
    assert (C.pythia_key("W1", 0), "step0") in [(c["key"], c["branch"]) for c in C.CHECKPOINTS] and \
           ("EleutherAI/pythia-160m-weight-seed1-corrected", "step0") in api.calls
    on_disk = json.loads((tmp_path / "checkpoints.json").read_text())
    assert on_disk["pinned"] == out["pinned"]


def test_pin_refuses_a_short_hash(tmp_path):
    class Bad(FakeApi):
        def model_info(self, repo_id, revision=None):
            return type("Info", (), {"sha": "abc"})()
    with pytest.raises(RuntimeError):
        C.pin(api=Bad(), path=tmp_path / "checkpoints.json")
    # a hash of the registered length but not hexadecimal is refused too, and the file is not written
    class NotHex(FakeApi):
        def model_info(self, repo_id, revision=None):
            return type("Info", (), {"sha": "z" * 40})()
    with pytest.raises(RuntimeError):
        C.pin(api=NotHex(), path=tmp_path / "checkpoints.json")
    assert not (tmp_path / "checkpoints.json").exists()
    # a hash of the right length that is hexadecimal is accepted
    class HexApi(FakeApi):
        def model_info(self, repo_id, revision=None):
            cp = next(c for c in C.CHECKPOINTS if c["repo_id"] == repo_id and c["branch"] == revision)
            info = super().model_info(repo_id, revision)
            if cp["commit_prefix"]:
                return type("Info", (), {"sha": cp["commit_prefix"] + info.sha[len(cp["commit_prefix"]):]})()
            return info
    ok = C.pin(api=HexApi(prefix=""), path=tmp_path / "checkpoints.json")
    assert all(len(v) == 40 for v in ok["pinned"].values())


def test_pin_enforces_the_registered_olmo_prefix(tmp_path):
    class PrefixApi(FakeApi):
        def model_info(self, repo_id, revision=None):
            info = super().model_info(repo_id, revision)
            cp = next(c for c in C.CHECKPOINTS if c["repo_id"] == repo_id and c["branch"] == revision)
            if cp["commit_prefix"]:
                return type("Info", (), {"sha": "ff" * 20})()      # not part85's registered prefix
            return info
    with pytest.raises(RuntimeError):
        C.pin(api=PrefixApi(), path=tmp_path / "checkpoints.json")


def test_load_pinned_refuses_an_incomplete_file(tmp_path):
    path = tmp_path / "checkpoints.json"
    path.write_text(json.dumps({"pinned": {"R0_step0": "a" * 40}}))
    with pytest.raises(RuntimeError):
        C.load_pinned(path)


# ---------------------------------------------------------------- text

class FakeTokenizer:
    """One token per character: a document has as many tokens as characters."""

    def __call__(self, text, add_special_tokens=False):
        return type("Enc", (), {"input_ids": list(range(len(text)))})()


def test_text_draws_48_and_24_documents_and_cuts_to_256():
    rows = [{"text": "x" * n} for n in (10, 255, 256, 300, 400)] * 10          # 10 short rows, 40 eligible
    with pytest.raises(ValueError):
        TX.draw(rows, FakeTokenizer())                                       # 40 < 72
    rows = rows * 3                                                          # 120 eligible
    sets = TX.draw(rows, FakeTokenizer())
    assert len(sets["A"]) == TX.N_A and len(sets["B"]) == TX.N_B
    assert all(len(d["ids"]) == TX.N_TOKENS for d in sets["A"] + sets["B"])
    assert all(len(rows[d["index"]]["text"]) >= TX.N_TOKENS for d in sets["A"] + sets["B"])
    assert not {d["index"] for d in sets["A"]} & {d["index"] for d in sets["B"]}


def test_text_draw_is_seeded_and_stable():
    rows = [{"text": "y" * (260 + i)} for i in range(100)]
    a = TX.draw(rows, FakeTokenizer())
    b = TX.draw(rows, FakeTokenizer())
    assert [d["index"] for d in a["A"]] == [d["index"] for d in b["A"]]
    assert TX.SEED == 211


def test_text_draw_uses_random_211():
    import random
    rows = [{"text": "z" * 300} for _ in range(100)]
    eligible = list(range(100))
    want = random.Random(211).sample(eligible, 72)
    got = TX.draw(rows, FakeTokenizer())
    assert [d["index"] for d in got["A"]] == want[:48]


# ---------------------------------------------------------------- run211

def test_recording_order_puts_references_first():
    order = RUN.recording_order("pythia")
    n_ref = sum(1 for k in order if C.BY_KEY[k]["reference"])
    assert n_ref == 6 * 3
    assert all(C.BY_KEY[k]["reference"] for k in order[:n_ref])
    assert len(order) == len(C.PYTHIA_KEYS)


def test_g3_and_floor_are_wired_for_211_1():
    import score211 as S
    import test_measures211 as TM
    d = TM._pythia()
    c = MS.counts_of(S.resample_indices(48, 50), 48)
    assert S.pred_211_1(d, TM.FLOOR0, c)["pass"]
    assert not S.pred_211_1(d, TM._floor(P=0.05), c)["pass"]   # floor a quarter of 0.2: every layer stopped


def test_reference_rows_are_stored_in_float32(tmp_path):
    model = MODELS["neox"]()
    meta = R.record_meta("a" * 40, "A", "code1")
    meta["reference"] = True
    info = R.record_checkpoint(model, docs(1), "pythia", "R0_step0", "A", tmp_path, meta)
    z = np.load(info["path"])
    assert z["att"].dtype == np.float32 and z["mlp"].dtype == np.float32


# ---------------------------------------------------------------- end to end, on tiny models (brief item 4)

def test_end_to_end_records_gates_distances_and_score_on_tiny_models(tmp_path, monkeypatch):
    """Every checkpoint of the registered table gets a tiny random model (D runs share R0's step-0 seed, as the real
    data-seed runs do), then the whole pipeline runs: record_all (records, floors, reference pairs), gates, the distance
    files with a reduced resample count (an explicit test parameter), and the scorer, whose registered-shape check
    must refuse tiny data unless told it is a test."""
    import zlib
    import distances211 as DIST
    import score211 as S
    texts = tmp_path / "texts.json"
    texts.write_text(json.dumps({"A": docs(3), "B": docs(2)}))
    keys = [cp["key"] for cp in C.CHECKPOINTS]
    pinned = {k: f"{i + 1:040x}" for i, k in enumerate(keys)}
    by_commit = {v: k for k, v in pinned.items()}

    def fake_load(repo_id, commit=None, dtype="float32"):
        key = by_commit[commit]
        cp = C.BY_KEY[key]
        seed_key = key
        if cp["family"] == C.PYTHIA_FAMILY and cp["run"].startswith("D") and cp.get("step") == 0:
            seed_key = C.pythia_key("R0", 0)
        seed = zlib.crc32(seed_key.encode()) % (2 ** 31)
        return (tiny_neox if cp["family"] == C.PYTHIA_FAMILY else tiny_olmo)(seed)
    monkeypatch.setattr(R, "load_model", fake_load)
    out = tmp_path / "out"
    RUN.record_all(out, texts, pinned=pinned)
    for k in R.floor_keys():
        assert R.floor_path(out, k, "A").is_file() and R.floor_path(out, k, "B").is_file()
    g = RUN.gates(out, texts, pinned=pinned)
    assert g["G1"]["d_pass"] and g["G2"]["pass"]
    for t in ("A", "B"):
        DIST.build(out, t, n_boot=40)
    sa, sb = DIST.load(out / "distances" / "A.npz"), DIST.load(out / "distances" / "B.npz")
    assert sa["idx"].shape == (40, 3) and sb["idx"].shape == (40, 2)
    # every saved floor comes back (Alden #13: the first build saved per-run and mid-training-end floors and dropped
    # them on load): per-run finals for 211.3, step-10000 floors for 211.8, the mid-training end beside 211.6
    assert set(sa["pythia_floor"]["per_run"]) == set(C.runs_of(C.PYTHIA_FAMILY))
    assert set(sa["pythia_floor"]["thread"]) == {"R0", *C.runs_with_factor("weights")}
    assert len(sa["pythia_floor"]["thread"]["R0"]["P"]) == NLAYER
    assert set(sa["olmo_floor"]["instruct"]) == {"P", "M"}
    # spot checks of the distance file against the records it was built from
    r0 = sa["pythia"]["runs"]["R0"]
    steps = list(S.PYTHIA_STEPS)
    assert np.all(r0["P_within"][:, :, -1] == 0)                           # X(final, final) is 0 by definition
    rec128 = np.load(R.record_path(out, "R0_step128", "A"))
    want = np.diagonal(rec128["red_R0_step143000_P"], axis1=2, axis2=3).mean(-1)
    assert np.allclose(r0["P_within"][:, :, steps.index(128)], want)
    rec0, fin = np.load(R.record_path(out, "R0_step0", "A")), np.load(R.record_path(out, "R0_step143000", "A"))
    for L in range(NLAYER):
        assert r0["M_point"][L, 0] == pytest.approx(1 - R.pooled_cka(rec0["mlp"], fin["mlp"], np.arange(3), L),
                                                     abs=1e-9)
        draw = sa["idx"][0]
        assert r0["M_boot"][0, L, 0] == pytest.approx(1 - R.pooled_cka(rec0["mlp"], fin["mlp"], draw, L), abs=1e-9)
    assert np.allclose(r0["V"][:, :, 0], rec0["V_final"])
    for k in ("P", "M", "V"):                                               # exactly the per-layer max of the six
        floors = np.array([np.load(R.floor_path(out, C.pythia_key(r, C.FINAL_STEP), "A"))[k] for r in C.runs_of()])
        assert np.array_equal(np.asarray(sa["pythia_floor"][k]), floors.max(axis=0))
        assert not np.array_equal(floors.max(axis=0), floors.min(axis=0)), k   # the runs' floors do differ
    w1 = sa["pythia"]["runs"]["W1"]["cross"][C.FINAL_STEP]
    a_w1 = np.load(R.record_path(out, "W1_step143000", "A"))["att"]
    assert w1[1, 0, 2, 1] == MS.pattern_distance(a_w1[1, 0, 2], fin["att"][1, 0, 1])
    with pytest.raises(ValueError):
        S.score_all(g, sa, sb, sa["idx"], sb["idx"])                       # tiny data is not the registered model
    res = S.score_all(g, sa, sb, sa["idx"], sb["idx"], registered_shape=False)
    assert set(res["verdicts"]) == set(S.PREDICTIONS)
    text = "\n".join(S.report_lines(res))
    for name in S.PREDICTIONS:
        assert f"{name}:" in text
    assert "G1 seed split" in text and "G2 measures" in text and "headline" in text
    # the columns beside the verdicts (design: Reading rules), checked against direct computations from the records
    assert "P0 beside P" in text and "cross-run to R0" in text and "per-head tables" in text
    b = res["beside"]["A"]
    assert b["P0"]["available"]
    ref_fin = "R0_step143000"
    s0, k0, t0 = rec128[f"red_{ref_fin}_P0s"], rec128[f"red_{ref_fin}_P0k"], int(rec128[f"red_{ref_fin}_P0t"])
    direct = np.zeros((NLAYER, s0.shape[-1]))
    for L in range(NLAYER):                                                 # pooled over documents, head by head
        for h in range(s0.shape[-1]):
            kk = np.diagonal(k0[:, L], axis1=1, axis2=2)[:, h].sum()
            direct[L, h] = np.diagonal(s0[:, L], axis1=1, axis2=2)[:, h].sum() / kk if kk >= 0.5 * 3 * t0 else np.nan
    # the stored parts against a direct computation from two references' rows (the reduction itself, not its reader)
    for part in ("P0s", "P0k", "cosd", "cosx", "cosy"):
        assert f"red_{ref_fin}_{part}" in rec0.files, part
    for d in range(3):
        for L in range(NLAYER):
            s_d, k_d, t_d = MS.p0_parts_matrix(rec0["att"][d, L], fin["att"][d, L])
            assert np.array_equal(rec0[f"red_{ref_fin}_P0k"][d, L], k_d) and int(rec0[f"red_{ref_fin}_P0t"]) == t_d
            assert np.allclose(rec0[f"red_{ref_fin}_P0s"][d, L], s_d, rtol=0, atol=1e-12)
            dot, sx, sy = MS.cos_parts_matrix(rec0["att"][d, L], fin["att"][d, L])
            assert np.allclose(rec0[f"red_{ref_fin}_cosd"][d, L], dot, rtol=0, atol=1e-9)
            assert np.allclose(rec0[f"red_{ref_fin}_cosx"][d, L], sx, rtol=0, atol=1e-9)
            assert np.allclose(rec0[f"red_{ref_fin}_cosy"][d, L], sy, rtol=0, atol=1e-9)
    p0_r0, _, _ = S._within_p0(r0, 3)
    assert np.allclose(p0_r0[:, steps.index(128)], S._nanmean(direct, axis=1), equal_nan=True)
    c = b["cross_run"]["W1"][C.FINAL_STEP][0]
    pairs = c["pairs"]
    flat = np.mean([MS.cosine_distance(a_w1[:, 0, i], fin["att"][:, 0, j]) for i, j in pairs])
    assert c["cos_on_P_pairs"] == pytest.approx(flat, abs=1e-9)             # flattened over the three documents
    m = w1[:, 0].mean(axis=0)
    assert c["P_greedy"] == pytest.approx(np.mean(m.min(axis=1)), abs=1e-12)   # each W1 head takes its nearest R0 head
    assert len(b["per_head"]["assignment_final"]["W1"][0]) == m.shape[0]
