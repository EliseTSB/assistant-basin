"""Part 182 tests: head swap mechanics on tiny OLMo 2 models, the gates' logic, the scorer, and one deliberate break.

Run: python3 -m pytest part182/test_182.py -q
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
import torch
from transformers import Olmo2Config, Olmo2ForCausalLM

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for _folder in (HERE, REPO / "part92", REPO / "part99", REPO):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import attn182  # noqa: E402
import headswap182 as hs  # noqa: E402
import headweights182 as hw  # noqa: E402
import score182  # noqa: E402
from swap99 import copy_components  # noqa: E402

HEADS, HEAD_DIM, HIDDEN = 4, 8, 32


def make_model(seed, kv_heads=HEADS, layers=3):
    torch.manual_seed(seed)
    config = Olmo2Config(vocab_size=96, hidden_size=HIDDEN, intermediate_size=64, num_hidden_layers=layers,
                         num_attention_heads=HEADS, num_key_value_heads=kv_heads, max_position_embeddings=64,
                         bos_token_id=1, eos_token_id=2, pad_token_id=0, attn_implementation="eager")
    model = Olmo2ForCausalLM(config)
    model.eval()
    model.requires_grad_(False)
    return model


def batch(seed=0, rows=3, width=9):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(3, 96, (rows, width), generator=g)
    mask = torch.ones_like(ids)
    mask[0, -2:] = 0  # one right-padded row
    return ids, mask


def logits(model, ids, mask):
    return hs.logits_on_batch(model, ids, mask)


@pytest.fixture
def pair():
    return make_model(1), make_model(2)


def test_source_applies_norm_over_all_heads():
    """The design's premise, checked against the installed transformers: q_norm and k_norm are one RMSNorm over
    heads*head_dim, applied to the full projection before the per-head reshape."""
    model = make_model(3)
    attn = model.model.layers[0].self_attn
    assert tuple(attn.q_norm.weight.shape) == (HEADS * HEAD_DIM,)
    assert tuple(attn.k_norm.weight.shape) == (HEADS * HEAD_DIM,)
    import inspect
    src = inspect.getsource(type(attn).forward)
    assert "self.q_norm(self.q_proj(hidden_states))" in src
    assert "self.k_norm(self.k_proj(hidden_states))" in src
    assert "self.o_proj(attn_output)" in src
    assert attn.o_proj.bias is None


def test_all_heads_of_a_layer_equal_whole_layer_copy(pair):
    """G1's logic: all heads of one layer from the donor == swap99's copy of that layer's attention."""
    host, donor = pair
    ids, mask = batch()
    donors = hs.donor_attention(donor, [0, 1, 2])
    for layer_no in range(3):
        hs.attach(host, donors, {layer_no: range(HEADS)})
        swapped = logits(host, ids, mask)
        assert hs.detach(host) == 1
        reference = copy.deepcopy(host)
        copy_components(reference, donor, "attn", layers={layer_no})
        ref = logits(reference, ids, mask)
        cmp = hs.compare_logits(swapped, ref, mask)
        assert cmp["max_abs_diff"] < 1e-4, cmp
        assert cmp["argmax_agree"] == cmp["positions"]


def test_block_equals_layer_range_copy(pair):
    """G3's logic: every head of layers 0..1 == --layers 0-1."""
    host, donor = pair
    ids, mask = batch(1)
    donors = hs.donor_attention(donor, [0, 1])
    hs.attach(host, donors, hs.condition_heads("block", [0, 1], HEADS))
    swapped = logits(host, ids, mask)
    hs.detach(host)
    reference = copy.deepcopy(host)
    copy_components(reference, donor, "attn", layers={0, 1})
    cmp = hs.compare_logits(swapped, logits(reference, ids, mask), mask)
    assert cmp["max_abs_diff"] < 1e-4, cmp


def test_self_donor_is_a_bit_exact_no_op_with_all_heads_and_close_with_one(pair):
    """G2's logic."""
    host, _ = pair
    ids, mask = batch(2)
    plain = logits(host, ids, mask)
    donors = hs.donor_attention(host, [0, 1, 2])
    hs.attach(host, donors, {l: range(HEADS) for l in range(3)})
    assert hs.compare_logits(logits(host, ids, mask), plain, mask)["bit_exact"]
    hs.detach(host)
    hs.attach(host, donors, {0: [0], 1: [2], 2: [3]})
    cmp = hs.compare_logits(logits(host, ids, mask), plain, mask)
    hs.detach(host)
    assert cmp["max_abs_diff"] < 1e-4 and cmp["argmax_agree"] == cmp["positions"], cmp


def test_single_head_changes_output_and_differs_from_other_heads(pair):
    host, donor = pair
    ids, mask = batch(3)
    plain = logits(host, ids, mask)
    donors = hs.donor_attention(donor, [1])
    outs = []
    for head in range(HEADS):
        hs.attach(host, donors, {1: [head]})
        outs.append(logits(host, ids, mask))
        hs.detach(host)
    for out in outs:
        assert not torch.equal(out, plain)
    assert not torch.equal(outs[0], outs[1])
    # The four single-head swaps do not simply add up to the whole-layer swap (the softmax couples heads only within
    # the donor; o_proj is linear, so here they DO add: a check on the decomposition itself).
    hs.attach(host, donors, {1: range(HEADS)})
    whole = logits(host, ids, mask)
    hs.detach(host)
    assert not torch.equal(whole, plain)


def test_runner_refuses_an_unattached_swap(pair):
    """The test that fails if the swap hook is not attached: assert_attached raises on a plain model, and a condition
    scored on a plain model is indistinguishable from none."""
    host, donor = pair
    ids, mask = batch(4)
    plain = logits(host, ids, mask)
    with pytest.raises(RuntimeError, match="not attached"):
        hs.assert_attached(host, {1: [0]})
    donors = hs.donor_attention(donor, [1])
    wrappers = hs.attach(host, donors, {1: [0]})
    hs.assert_attached(host, {1: [0]})
    with pytest.raises(RuntimeError, match="ran 0 times"):
        hs.assert_attached(host, {1: [0]}, min_calls=1)
    swapped = logits(host, ids, mask)
    hs.assert_attached(host, {1: [0]}, min_calls=1)
    assert wrappers[1].calls == 1
    assert not torch.equal(swapped, plain)
    with pytest.raises(RuntimeError, match="detach"):
        hs.attach(host, donors, {1: [1]})
    hs.detach(host)
    assert hs.attached_heads(host) == {}
    assert torch.equal(logits(host, ids, mask), plain)
    with pytest.raises(RuntimeError, match="not attached"):
        hs.assert_attached(host, {1: [0]})


def test_generation_with_cache_matches_whole_layer_copy(pair):
    """Stage 2's path: greedy generation with the donor's own key-value cache equals generation from swap99's copy."""
    host, donor = pair
    ids, mask = batch(5, rows=2, width=6)
    mask[:] = 1
    donors = hs.donor_attention(donor, [0, 2])
    hs.attach(host, donors, {0: range(HEADS), 2: range(HEADS)})
    with torch.inference_mode():
        swapped = host.generate(input_ids=ids, attention_mask=mask, do_sample=False, max_new_tokens=8,
                                pad_token_id=0)
        swapped_again = host.generate(input_ids=ids, attention_mask=mask, do_sample=False, max_new_tokens=8,
                                      pad_token_id=0)
    hs.detach(host)
    reference = copy.deepcopy(host)
    copy_components(reference, donor, "attn", layers={0, 2})
    with torch.inference_mode():
        ref = reference.generate(input_ids=ids, attention_mask=mask, do_sample=False, max_new_tokens=8,
                                 pad_token_id=0)
    assert torch.equal(swapped, ref)
    assert torch.equal(swapped, swapped_again)  # the donor cache resets at each prefill


def test_gqa_model_swaps_by_o_proj_columns():
    host, donor = make_model(7, kv_heads=2), make_model(8, kv_heads=2)
    ids, mask = batch(6)
    donors = hs.donor_attention(donor, [0])
    hs.attach(host, donors, {0: range(HEADS)})
    swapped = logits(host, ids, mask)
    hs.detach(host)
    reference = copy.deepcopy(host)
    copy_components(reference, donor, "attn", layers={0})
    assert hs.compare_logits(swapped, logits(reference, ids, mask), mask)["max_abs_diff"] < 1e-4


def test_deliberate_break_turns_g1_red(pair, monkeypatch):
    """A head mask shifted by one coordinate must fail the whole-layer comparison."""
    host, donor = pair
    ids, mask = batch(7)
    donors = hs.donor_attention(donor, [1])
    wrappers = hs.attach(host, donors, {1: range(HEADS)})
    broken = torch.roll(wrappers[1].swap_mask.clone(), 1)
    broken[0] = False
    wrappers[1].swap_mask = broken
    swapped = logits(host, ids, mask)
    hs.detach(host)
    reference = copy.deepcopy(host)
    copy_components(reference, donor, "attn", layers={1})
    cmp = hs.compare_logits(swapped, logits(reference, ids, mask), mask)
    assert cmp["max_abs_diff"] > 1e-4


def test_condition_tags():
    layers = [0, 1, 2, 3]
    assert hs.condition_heads("none", layers, 16) == {}
    assert hs.condition_heads("h2.5", layers, 16) == {2: [5]}
    assert hs.condition_heads("layer3", layers, 16) == {3: list(range(16))}
    block = hs.condition_heads("block", layers, 16)
    assert sorted(block) == layers and all(len(v) == 16 for v in block.values())
    restore = hs.condition_heads("restore1.4", layers, 16)
    assert restore[1] == [h for h in range(16) if h != 4] and len(restore[0]) == 16
    ranking = [(1, 4), (3, 2), (1, 7)]
    assert hs.condition_heads("top2", layers, 16, ranking) == {1: [4], 3: [2]}
    assert hs.condition_heads("top3", layers, 16, ranking) == {1: [4, 7], 3: [2]}
    with pytest.raises(ValueError):
        hs.condition_heads("h4.0", layers, 16)
    with pytest.raises(ValueError):
        hs.condition_heads("top4", layers, 16, ranking)
    tags = hs.stage1_conditions(layers, 16, control_layers=[])
    assert len(tags) == 1 + 64 + 4 + 1 + 64 and tags[0] == "none" and tags[-1] == "restore3.15"
    # Amendment 1 item 3: with the control layers, the 16 c14.H tags come after the restore tags, and every earlier
    # tag keeps its place.
    full = hs.stage1_conditions(layers, 16, control_layers=[14])
    assert full[:len(tags)] == tags and full[len(tags):] == [f"c14.{h}" for h in range(16)]
    assert len(full) == len(tags) + 16 and full[0] == "none"


def test_teacher_forced_scoring_matches_direct_computation(pair):
    host, _ = pair
    prompt = [1, 5, 6]
    rows = [{"answer_ids": [7, 8, 9, 10], "push_layer": None, "push": None},
            {"answer_ids": [11, 12], "push_layer": 1, "push": torch.ones(HIDDEN) * 0.5}]

    class Tok:
        pad_token_id = 0

    results = hs.score_rows(host, Tok(), host.model.layers, prompt, rows, "cpu", batch=2, hidden=HIDDEN)
    assert [r["n_tokens"] for r in results] == [4, 2]
    # Direct: row 0, alone, no push.
    ids = torch.tensor([prompt + [7, 8, 9, 10]])
    with torch.inference_mode():
        lg = host(input_ids=ids).logits.float()[0]
    expect = [float(torch.log_softmax(lg[2 + i], -1)[t]) for i, t in enumerate([7, 8, 9, 10])]
    assert results[0]["logp_tokens"] == pytest.approx(expect, abs=1e-4)
    assert results[0]["logp_sum"] == pytest.approx(sum(expect), abs=1e-3)
    # Row 1 with its push, alone.
    handle = hs.c.add_row_hook(host.model.layers[1], torch.ones(1, HIDDEN) * 0.5)
    ids = torch.tensor([prompt + [11, 12]])
    with torch.inference_mode():
        lg = host(input_ids=ids).logits.float()[0]
    handle.remove()
    expect = [float(torch.log_softmax(lg[2 + i], -1)[t]) for i, t in enumerate([11, 12])]
    assert results[1]["logp_tokens"] == pytest.approx(expect, abs=1e-4)
    # The push changed something.
    unpushed = hs.score_rows(host, Tok(), host.model.layers, prompt,
                             [{"answer_ids": [11, 12], "push_layer": None, "push": None}], "cpu", 1, HIDDEN)
    assert unpushed[0]["logp_sum"] != pytest.approx(results[1]["logp_sum"], abs=1e-6)


# ---------------------------------------------------------------------------------------------------------------------
# score182: effects, ranking, the stage-2 selection rule and the verdicts, on a fixture.

def fixture_rows(effects: dict[str, float], slots, concepts, layers=(0, 1), n_heads=2, control_layers=()):
    """Stage-1 rows for a fixture: own-answer logp_sum = -5 + effect per unit per condition; the other row kinds
    constant."""
    rows = []
    tags = hs.stage1_conditions(list(layers), n_heads, control_layers=list(control_layers))
    for tag in tags:
        eff = effects.get(tag, 0.0)
        for uid in slots + concepts:
            scale = 1.0 if uid in slots else 0.25
            rows.append({"condition": tag, "id": uid, "unit": "neuron", "text": "own", "logp_sum": -5.0 + eff * scale,
                         "n_tokens": 5, "argmax_match": 5})
            rows.append({"condition": tag, "id": uid, "unit": "random", "text": "own", "logp_sum": -8.0 + eff * 0.1,
                         "n_tokens": 5, "argmax_match": 3})
            rows.append({"condition": tag, "id": uid, "unit": "neuron", "text": "assistant", "logp_sum": -20.0,
                         "n_tokens": 5, "argmax_match": 1})
        rows.append({"condition": tag, "id": "unpushed", "unit": "none", "text": "assistant",
                     "logp_sum": -3.0 + eff * 0.05, "n_tokens": 5, "argmax_match": 5})
    return rows


def test_score182_effects_ranking_selection_and_verdicts(tmp_path):
    slots = [f"s{i}" for i in range(5)]
    concepts = [f"c{i}" for i in range(3)]
    effects = {"h0.0": -1.0, "h0.1": -0.2, "h1.0": -4.0, "h1.1": -0.5, "layer0": -1.5, "layer1": -4.5, "block": -6.0,
               "restore0.0": -5.5, "restore0.1": -5.9, "restore1.0": -2.0, "restore1.1": -5.8}
    rows = fixture_rows(effects, slots, concepts)
    table = score182.effects(rows, slots, concepts, layers=[0, 1], n_heads=2)
    assert table["h1.0"]["slot_median"] == pytest.approx(-4.0)
    assert table["h1.0"]["concept_median"] == pytest.approx(-1.0)
    assert table["h1.0"]["unpushed"] == pytest.approx(-0.2)
    assert table["block"]["slot_mean"] == pytest.approx(-6.0)
    ranking = score182.rank_heads(table, [0, 1], 2)
    assert ranking[:2] == [(1, 0), (0, 0)]
    sel = score182.select_stage2(table, [0, 1], 2, k=1)
    assert sel["selected"] == ["h1.0"] and sel["controls"] == ["h1.1"]
    verdicts = score182.verdicts(table, ranking, [0, 1], 2, k=1, ladder=None)
    assert verdicts["182.2"]["verdict"] == "held"  # top head in layer 1, layer 0 the smaller whole-layer effect
    assert verdicts["182.3"]["verdict"] == "held"  # concept quarter of slot
    assert verdicts["182.4"]["verdict"] == "held"  # singles sum to -5.7 > -6.0
    assert verdicts["182.7"]["verdict"] == "failed"  # restore1.0 recovers 4 of 6 nats: over half
    assert verdicts["182.1"]["verdict"] == "not read"  # no ladder rows


def test_score182_eligible_units_and_rows_roundtrip(tmp_path):
    labels = tmp_path / "labels.jsonl"
    with labels.open("w") as f:
        for uid, sp in (("a", True), ("b", False), ("c", True)):
            f.write(json.dumps({"id": uid, "unit": "neuron", "question_index": 0, "shows_persona": sp,
                                "coherent": True}) + "\n")
        f.write(json.dumps({"id": "d", "unit": "neuron", "question_index": 0, "error": "x"}) + "\n")
    kinds = {"a": "slot", "b": "slot", "c": "concept", "d": "slot"}
    assert score182.eligible_units(labels, kinds) == ({"slot": ["a"], "concept": ["c"]}, {"b": "no persona on Q0",
                                                                                            "d": "no label"})


# ---------------------------------------------------------------------------------------------------------------------
# Amendment 1 item 2: headweights182, the weight cosines from the two snapshots' safetensors.

def write_snapshot(path, tensors, n_heads=HEADS, kv_heads=HEADS, head_dim=HEAD_DIM, hidden=HIDDEN, layers=2):
    from safetensors.torch import save_file
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps({"num_attention_heads": n_heads, "num_key_value_heads": kv_heads,
                                                  "head_dim": head_dim, "hidden_size": hidden}), encoding="utf-8")
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path / "model.safetensors"))


def snapshot_tensors(seed, layers=2):
    g = torch.Generator().manual_seed(seed)
    t = {"model.embed_tokens.weight": torch.randn(96, HIDDEN, generator=g)}
    for layer_no in range(layers):
        base = f"model.layers.{layer_no}.self_attn."
        t[base + "q_proj.weight"] = torch.randn(HEADS * HEAD_DIM, HIDDEN, generator=g)
        t[base + "k_proj.weight"] = torch.randn(HEADS * HEAD_DIM, HIDDEN, generator=g)
        t[base + "v_proj.weight"] = torch.randn(HEADS * HEAD_DIM, HIDDEN, generator=g)
        t[base + "o_proj.weight"] = torch.randn(HIDDEN, HEADS * HEAD_DIM, generator=g)
        t[base + "q_norm.weight"] = torch.rand(HEADS * HEAD_DIM, generator=g) + 0.5
        t[base + "k_norm.weight"] = torch.rand(HEADS * HEAD_DIM, generator=g) + 0.5
    return t


def test_headweights_cosines_on_known_tensors(tmp_path):
    """One head identical (cosine 1), one negated (-1), one random (a value computed here), and the norms and the
    embedding read whole."""
    host_t = snapshot_tensors(11)
    donor_t = {k: v.clone() for k, v in host_t.items()}
    base = "model.layers.0.self_attn."
    head = lambda h: slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)  # noqa: E731
    donor_t[base + "q_proj.weight"][head(0)] = host_t[base + "q_proj.weight"][head(0)]      # head 0 identical
    donor_t[base + "q_proj.weight"][head(1)] = -host_t[base + "q_proj.weight"][head(1)]     # head 1 negated
    donor_t[base + "o_proj.weight"][:, head(2)] = -2.0 * host_t[base + "o_proj.weight"][:, head(2)]
    donor_t[base + "q_norm.weight"] = host_t[base + "q_norm.weight"] * 3.0                  # scale: cosine 1
    donor_t["model.embed_tokens.weight"] = torch.zeros_like(host_t["model.embed_tokens.weight"])  # zero: null
    host_dir, donor_dir = tmp_path / "host", tmp_path / "donor"
    write_snapshot(host_dir, host_t)
    write_snapshot(donor_dir, donor_t)
    record = hw.head_weights(host_dir, donor_dir, [0, 1])
    layer0 = record["layers"]["0"]["heads"]
    assert layer0["0"]["q_row_cos"] == pytest.approx(1.0, abs=1e-12)
    assert layer0["1"]["q_row_cos"] == pytest.approx(-1.0, abs=1e-12)
    assert layer0["2"]["o_col_cos"] == pytest.approx(-1.0, abs=1e-12)
    # head 3's q rows are untouched: the cosine is 1; head 2's are untouched too.
    assert layer0["3"]["q_row_cos"] == pytest.approx(1.0, abs=1e-12)
    # A random pair, computed here in float64.
    a = host_t[base + "k_proj.weight"][head(2)].double()
    b = donor_t[base + "k_proj.weight"][head(2)].double()
    assert layer0["2"]["k_row_cos"] == pytest.approx(float((a * b).sum() / (a.norm() * b.norm())), abs=1e-12)
    assert record["layers"]["0"]["q_norm_cos"] == pytest.approx(1.0, abs=1e-12)
    assert record["embed_cos"] is None  # a zero vector is null, never 0
    # Layer 1 is untouched: every cosine 1.
    for entry in record["layers"]["1"]["heads"].values():
        assert entry == pytest.approx({"q_row_cos": 1.0, "k_row_cos": 1.0, "v_row_cos": 1.0, "o_col_cos": 1.0})
    assert record["tensor_names"]["host"]["layers"]["0"]["q_proj"] == base + "q_proj.weight"
    assert record["host_config"] == {"num_attention_heads": HEADS, "num_key_value_heads": HEADS,
                                     "head_dim": HEAD_DIM, "hidden_size": HIDDEN}


def test_headweights_gqa_uses_the_key_value_head(tmp_path):
    """With 2 key-value heads for 4 query heads, heads 0 and 1 share key-value head 0: negating it makes both heads'
    k and v cosines -1, and head 2's stays 1."""
    host_t = snapshot_tensors(12)
    donor_t = {k: v.clone() for k, v in host_t.items()}
    base = "model.layers.0.self_attn."
    donor_t[base + "k_proj.weight"][:HEAD_DIM] *= -1
    donor_t[base + "v_proj.weight"][:HEAD_DIM] *= -1
    write_snapshot(tmp_path / "host", host_t, kv_heads=2)
    write_snapshot(tmp_path / "donor", donor_t, kv_heads=2)
    heads = hw.head_weights(tmp_path / "host", tmp_path / "donor", [0])["layers"]["0"]["heads"]
    for h in ("0", "1"):
        assert heads[h]["k_row_cos"] == pytest.approx(-1.0) and heads[h]["v_row_cos"] == pytest.approx(-1.0)
    for h in ("2", "3"):
        assert heads[h]["k_row_cos"] == pytest.approx(1.0) and heads[h]["v_row_cos"] == pytest.approx(1.0)


def test_headweights_cli_writes_json(tmp_path):
    tensors = snapshot_tensors(13)
    write_snapshot(tmp_path / "host", tensors)
    write_snapshot(tmp_path / "donor", tensors)
    out = tmp_path / "out" / "head-weights.json"
    assert hw.main(["--host", str(tmp_path / "host"), "--donor", str(tmp_path / "donor"), "--layers", "0,1",
                    "--out", str(out)]) == 0
    record = json.loads(out.read_text(encoding="utf-8"))
    assert sorted(record["layers"]) == ["0", "1"] and record["embed_cos"] == pytest.approx(1.0)
    with pytest.raises(KeyError):
        hw.head_weights(tmp_path / "host", tmp_path / "donor", [5])  # a layer the snapshot does not have


# ---------------------------------------------------------------------------------------------------------------------
# Amendment 1 item 3: the control-layer heads cL.H, never ranked, selected or laddered; 182.8.

def test_control_layer_tags_decode_and_are_restricted():
    layers, controls = [0, 1, 2, 3], [14]
    assert hs.condition_heads("c14.5", layers, 16, control_layers=controls) == {14: [5]}
    assert hs.condition_heads("c14.0", layers, 16, control_layers=controls) == {14: [0]}
    with pytest.raises(ValueError, match="control layers"):
        hs.condition_heads("c13.5", layers, 16, control_layers=controls)
    with pytest.raises(ValueError, match="control layers"):
        hs.condition_heads("c14.16", layers, 16, control_layers=controls)
    with pytest.raises(ValueError, match="main layer"):
        hs.condition_heads("c2.5", layers, 16, control_layers=controls)
    # hL.H stays restricted to the main layers: a main-layer tag cannot name a control layer.
    with pytest.raises(ValueError):
        hs.condition_heads("h14.5", layers, 16)
    assert score182.CONTROL_LAYERS == [14]


def test_control_heads_never_ranked_selected_or_laddered():
    """A table whose c14 heads are far more negative than every main head: the ranking, the selection, the ladder and
    the topK path must not see them."""
    slots = [f"s{i}" for i in range(5)]
    concepts = [f"c{i}" for i in range(3)]
    effects = {f"h0.{h}": -1.0 - 0.1 * h for h in range(2)}
    effects.update({f"h1.{h}": -2.0 - 0.1 * h for h in range(2)})
    effects.update({f"c14.{h}": -50.0 - h for h in range(2)})
    effects.update({"layer0": -1.5, "layer1": -2.5, "block": -4.0})
    effects.update({f"restore{l}.{h}": -3.0 for l in (0, 1) for h in range(2)})
    rows = fixture_rows(effects, slots, concepts, layers=(0, 1), n_heads=2, control_layers=(14,))
    table = score182.effects(rows, slots, concepts, layers=[0, 1], n_heads=2, control_layers=[14])
    assert table["c14.1"]["slot_median"] == pytest.approx(-51.0)
    ranking = score182.rank_heads(table, [0, 1], 2)
    assert ranking == [(1, 1), (1, 0), (0, 1), (0, 0)]
    assert all(l in (0, 1) for l, _h in ranking)
    sel = score182.select_stage2(table, [0, 1], 2, k=1)
    assert sel["selected"] == ["h1.1"]
    assert all("c" not in t for t in sel["selected"] + sel["controls"])
    ladder = score182.ladder_conditions(ranking, ks=(1, 2))
    assert all(l in (0, 1) for heads in ladder.values() for l in heads)
    top = hs.condition_heads("top2", [0, 1], 2, ranking)
    assert 14 not in top
    verdict = score182.verdicts(table, ranking, [0, 1], 2, k=2, control_layers=[14])
    assert verdict["182.8"]["verdict"] == "failed"
    assert verdict["182.8"]["controls_more_negative"] == ["c14.0", "c14.1"]


def test_182_8_held_failed_and_not_read():
    slots = [f"s{i}" for i in range(5)]
    concepts = [f"c{i}" for i in range(3)]
    base = {f"h0.{h}": -1.0 - 0.1 * h for h in range(2)}
    base.update({f"h1.{h}": -2.0 - 0.1 * h for h in range(2)})
    base.update({"layer0": -1.5, "layer1": -2.5, "block": -4.0})
    base.update({f"restore{l}.{h}": -3.0 for l in (0, 1) for h in range(2)})
    # k=2: the 2nd-ranked main head is h1.0 at -2.0.
    held = dict(base, **{f"c14.{h}": -1.0 - 0.01 * h for h in range(2)})
    table = score182.effects(fixture_rows(held, slots, concepts, layers=(0, 1), n_heads=2, control_layers=(14,)),
                             slots, concepts, layers=[0, 1], n_heads=2, control_layers=[14])
    ranking = score182.rank_heads(table, [0, 1], 2)
    out = score182.verdicts(table, ranking, [0, 1], 2, k=2, control_layers=[14])
    assert out["182.8"]["verdict"] == "held" and out["182.8"]["sixth_head"] == "h1.0"
    assert out["182.8"]["sixth_slot_median"] == pytest.approx(-2.0)
    # One control head more negative than the 2nd-ranked main head: failed.
    failed = dict(held, **{"c14.1": -2.5})
    table = score182.effects(fixture_rows(failed, slots, concepts, layers=(0, 1), n_heads=2, control_layers=(14,)),
                             slots, concepts, layers=[0, 1], n_heads=2, control_layers=[14])
    out = score182.verdicts(table, score182.rank_heads(table, [0, 1], 2), [0, 1], 2, k=2, control_layers=[14])
    assert out["182.8"]["verdict"] == "failed" and out["182.8"]["controls_more_negative"] == ["c14.1"]
    # One of the 16 c14 conditions missing: not read. (A fixture row whose effect is None is dropped.)
    missing = dict(held, **{"c14.0": None})
    table = score182.effects(fixture_rows({k: v for k, v in missing.items() if v is not None}, slots, concepts,
                                          layers=(0, 1), n_heads=2, control_layers=()),
                             slots, concepts, layers=[0, 1], n_heads=2, control_layers=[14])
    assert "c14.0" not in table
    out = score182.verdicts(table, score182.rank_heads(table, [0, 1], 2), [0, 1], 2, k=2, control_layers=[14])
    assert out["182.8"]["verdict"] == "not read" and out["182.8"]["control_slot_medians"]["c14.0"] is None
    # No ranking: not read.
    out = score182.verdicts(table, [], [0, 1], 2, k=2, control_layers=[14])
    assert out["182.8"]["verdict"] == "not read"


# ---------------------------------------------------------------------------------------------------------------------
# Amendment 1 item 5: residual readouts at intermediate decoder layers.

class TinyTok:
    pad_token_id = 0


def readout_rows():
    return [{"answer_ids": [7, 8, 9, 10], "push_layer": None, "push": None},
            {"answer_ids": [11, 12, 13], "push_layer": 1, "push": torch.ones(HIDDEN) * 0.5},
            {"answer_ids": [14, 15], "push_layer": 2, "push": -torch.ones(HIDDEN) * 0.25}]


def test_resid_rel_is_zero_for_none_and_self_donor_and_positive_for_a_swap():
    host = make_model(21, layers=4)
    donors = hs.donor_attention(host, [0, 1])  # Instruct as its own donor
    prompt = [1, 5, 6]
    rows = readout_rows()
    none_states: list[dict] = []
    none = hs.score_rows(host, TinyTok(), host.model.layers, prompt, rows, "cpu", 2, HIDDEN,
                         readout_layers=(0, 2, 3, 7), capture=none_states)
    assert len(none_states) == len(rows)
    assert sorted(none_states[0]) == [0, 2, 3]  # layer 7 does not exist and is skipped
    for r in none:
        assert r["resid_rel"] == {"0": 0.0, "2": 0.0, "3": 0.0}
    # G2's case: every head of both layers from the model itself.
    hs.attach(host, donors, {0: range(HEADS), 1: range(HEADS)})
    try:
        same = hs.score_rows(host, TinyTok(), host.model.layers, prompt, rows, "cpu", 2, HIDDEN,
                             none_states=none_states, readout_layers=(0, 2, 3))
    finally:
        hs.detach(host)
    for r in same:
        assert all(v == 0.0 for v in r["resid_rel"].values()), r["resid_rel"]
    # A real swap of one head: layer 0's output changes, and the readout says so.
    donor = make_model(22, layers=4)
    hs.attach(host, hs.donor_attention(donor, [0]), {0: [1]})
    try:
        swapped = hs.score_rows(host, TinyTok(), host.model.layers, prompt, rows, "cpu", 2, HIDDEN,
                                none_states=none_states, readout_layers=(0, 2, 3))
    finally:
        hs.detach(host)
    assert all(r["resid_rel"]["0"] > 0 for r in swapped)
    assert all(r["resid_rel"]["3"] > 0 for r in swapped)  # it propagates


def test_capture_does_not_change_any_logit():
    """The readout hooks must not move a single bit of the scored log-probabilities."""
    host, donor = make_model(1), make_model(2)
    prompt = [1, 5, 6]
    rows = readout_rows()
    hs.attach(host, hs.donor_attention(donor, [1]), {1: [0, 2]})
    try:
        without = hs.score_rows(host, TinyTok(), host.model.layers, prompt, rows, "cpu", 2, HIDDEN,
                                readout_layers=())
        none_states = []
        hs.score_rows(host, TinyTok(), host.model.layers, prompt, rows, "cpu", 2, HIDDEN, capture=none_states,
                      readout_layers=(0, 2, 3))
        with_capture = hs.score_rows(host, TinyTok(), host.model.layers, prompt, rows, "cpu", 2, HIDDEN,
                                     none_states=none_states, readout_layers=(0, 2, 3))
    finally:
        hs.detach(host)
    for a, b in zip(without, with_capture):
        assert a["logp_tokens"] == b["logp_tokens"]
        assert a["logp_sum"] == b["logp_sum"]
        assert "resid_rel" not in a and "resid_rel" in b


def test_resid_rel_reads_the_answer_positions_only():
    """The deliberate break: a readout that reads the wrong positions (the prompt's last position instead of the
    answer's) gives a different number, so a test on the positions catches it."""
    host = make_model(23, layers=4)
    donor = make_model(24, layers=4)
    prompt = [1, 5, 6]
    rows = readout_rows()
    none_states: list[dict] = []
    hs.score_rows(host, TinyTok(), host.model.layers, prompt, rows, "cpu", 2, HIDDEN, capture=none_states,
                  readout_layers=(0, 2, 3))
    hs.attach(host, hs.donor_attention(donor, [0]), {0: [1]})
    try:
        right = hs.score_rows(host, TinyTok(), host.model.layers, prompt, rows, "cpu", 2, HIDDEN,
                              none_states=none_states, readout_layers=(0, 2, 3))
        capture = []
        hs.score_rows(host, TinyTok(), host.model.layers, prompt, rows, "cpu", 2, HIDDEN, capture=capture,
                      readout_layers=(0, 2, 3))
    finally:
        hs.detach(host)
    answers = [r["answer_ids"] for r in rows]
    positions = hs.answer_positions(len(prompt), answers)
    assert positions == [[2, 3, 4, 5], [2, 3, 4], [2, 3]]
    broken_positions = [[len(prompt) - 2] for _ in rows]  # the prompt's last position: the sabotage
    for i, result in enumerate(right):
        current = capture[i]
        broken = hs.relative_residual(none_states[i], current, broken_positions[i])
        assert broken["0"] != pytest.approx(result["resid_rel"]["0"], abs=1e-9)
        # The sabotage is caught by a test that recomputes the readout from the captured states on the right
        # positions, not only by the number differing.
        assert hs.relative_residual(none_states[i], current, positions[i]) == pytest.approx(result["resid_rel"])
    # Right padding is excluded: a padded answer position is not scored and is not read.
    mask = torch.ones(3, 6)
    mask[2, 3:] = 0
    assert hs.answer_positions(len(prompt), answers, mask)[2] == [2]
    with pytest.raises(ValueError):
        hs.relative_residual(none_states[0], capture[0], [])


def test_effects_reports_resid_rel_median_and_none_when_absent():
    slots = [f"s{i}" for i in range(3)]
    rows = []
    for tag, value in (("none", 0.0), ("h0.0", 0.2), ("h0.1", 0.4)):
        for i, uid in enumerate(slots):
            row = {"condition": tag, "id": uid, "unit": "neuron", "text": "own", "logp_sum": -5.0 - i}
            if tag != "h0.1":
                row["resid_rel"] = {"3": value + i, "7": value}
            rows.append(row)
    table = score182.effects(rows, slots, [], layers=[0], n_heads=2)
    assert table["h0.0"]["resid_rel_median"] == {"3": 1.2, "7": 0.2}  # median of 0.2, 1.2, 2.2
    assert table["h0.1"]["resid_rel_median"] is None  # absent fields give None, never 0
    assert table["h0.1"]["slot_median"] is not None


# ---------------------------------------------------------------------------------------------------------------------
# Amendment 1 item 5 (scorer): weight cosines beside slot_median and the Spearman correlation.

def test_head_weight_cosines_and_spearman():
    table = {"h0.0": {"slot_median": -3.0}, "h0.1": {"slot_median": -1.0},
             "h1.0": {"slot_median": -2.0}, "h1.1": {"slot_median": -0.5}}
    weights = {"layers": {
        "0": {"heads": {"0": {"q_row_cos": 0.1, "k_row_cos": 0.2, "v_row_cos": 0.3, "o_col_cos": 0.1},
                        "1": {"q_row_cos": 0.4, "k_row_cos": 0.5, "v_row_cos": 0.6, "o_col_cos": 0.9}}},
        "1": {"heads": {"0": {"q_row_cos": 0.7, "k_row_cos": 0.8, "v_row_cos": 0.9, "o_col_cos": 0.4},
                        "1": {"q_row_cos": 0.0, "k_row_cos": 0.0, "v_row_cos": 0.0, "o_col_cos": 0.8}}}}}
    out = score182.head_weight_cosines(table, weights, layers=[0, 1], n_heads=2)
    assert out["heads"]["h0.0"]["o_col_cos"] == 0.1 and out["heads"]["h0.0"]["slot_median"] == -3.0
    assert out["n_pairs"] == 4
    # o_col_cos 0.1, 0.9, 0.4, 0.8 against -3, -1, -2, -0.5: ranks 1,4,2,3 against 1,3,2,4.
    assert out["spearman_o_col_vs_slot_median"] == pytest.approx(0.8)
    assert score182.spearman([1, 1, 1], [1, 2, 3]) is None  # a constant side: None, not 0
    missing = score182.head_weight_cosines(table, None, layers=[0, 1], n_heads=2)
    assert missing["spearman_o_col_vs_slot_median"] is None
    assert missing["heads"]["h1.1"]["q_row_cos"] is None


# ---------------------------------------------------------------------------------------------------------------------
# Amendment 1 item 7: attn182, the attention shares by position group.

def test_attention_groups_and_shares_sum_to_one():
    """The three shares sum to 1 within 1e-5 at every answer position; the host and donor shares are equal when the
    donor is the host; the question's ids are located inside the prompt and a doubled question fails loudly."""
    host = make_model(31, layers=3)

    class Tok:
        pad_token_id = 0

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [40, 41]}

    prompt = [1, 40, 41, 42, 43, 44]  # the question's ids 40, 41 sit at positions 1 and 2
    groups = attn182.position_groups(Tok(), "q", prompt)
    assert groups == {"template": [0, 3, 4, 5], "question": [1, 2]}
    # The answer's positions, after the prompt (6 tokens); the real path builds them the same way.
    answer_group = {"answer": list(range(len(prompt), len(prompt) + 4))}
    with pytest.raises(ValueError, match="2 times"):
        attn182.position_groups(Tok(), "q", [1, 40, 41, 40, 41])
    with pytest.raises(ValueError, match="0 times"):
        attn182.position_groups(Tok(), "q", [1, 2, 3])
    rows = [{"id": "s0", "unit": "neuron", "text": "own", "push_layer": 1, "push": torch.ones(HIDDEN) * 0.3,
             "answer_ids": [7, 8, 9, 10]},
            {"id": "s1", "unit": "neuron", "text": "own", "push_layer": 0, "push": -torch.ones(HIDDEN) * 0.2,
             "answer_ids": [11, 12]}]
    record = attn182.head_attention_shares("unused", "unused", [(1, 0), (2, 3)], rows, prompt, Tok(),
                                           host_model=host, batch=2, donor_attn=hs.donor_attention(host, [1, 2]))
    assert record["sum_check_max_deviation"] < 1e-5
    for tag in ("h1.0", "h2.3"):
        for mode in ("on", "off"):
            for name in ("host", "donor"):
                entry = record["heads"][tag][mode][name]
                assert sum(entry[g] for g in attn182.GROUPS) == pytest.approx(1.0, abs=1e-5)
                assert entry["n_positions"] == 6
                # Right padding excluded: 4 + 2 answer positions, not 4 + 4.
                assert record["heads"][tag][mode]["host"]["n_positions"] == 6
    assert record["heads"]["h1.0"]["on"]["host"] != record["heads"]["h1.0"]["off"]["host"]  # the push moves them
    assert record["rows"][0]["answer_positions"] == [5, 6, 7, 8]
    # The donor is the host: the donor module is the host module, so its shares equal the host's.
    record = attn182.head_attention_shares("unused", "unused", [(1, 0)], rows, prompt, Tok(), host_model=host,
                                           batch=2, donor_attn={1: host.model.layers[1].self_attn})
    for mode in ("on", "off"):
        assert record["heads"]["h1.0"][mode]["donor"] == record["heads"]["h1.0"][mode]["host"]
    # A different donor gives different shares.
    record = attn182.head_attention_shares("unused", "unused", [(1, 0)], rows, prompt, Tok(), host_model=host,
                                           batch=2, donor_attn={1: make_model(32).model.layers[1].self_attn})
    assert record["heads"]["h1.0"]["on"]["donor"] != record["heads"]["h1.0"]["on"]["host"]
    # The weights come from the module's own eager forward: read twice, identical.
    stores, handles = hs.capture_layer_outputs(host, [1])
    ids = torch.tensor([prompt + [7, 8, 9, 10], prompt + [11, 12, 0, 0]])
    mask = torch.tensor([[1] * 10, [1] * 8 + [0, 0]])
    with torch.inference_mode():
        host(input_ids=ids, attention_mask=mask, use_cache=False)
    for handle in handles:
        handle.remove()
    from transformers.masking_utils import create_causal_mask
    states = stores[1][-1]
    positions = torch.arange(states.shape[1]).unsqueeze(0)
    causal = create_causal_mask(config=host.config, inputs_embeds=states, attention_mask=mask, past_key_values=None,
                                position_ids=positions)
    emb = host.model.rotary_emb(states, position_ids=positions)
    attn = host.model.layers[1].self_attn
    w1 = attn182.attention_weights(attn, states, emb, causal)
    w2 = attn182.attention_weights(copy.deepcopy(attn), states, emb, causal)
    assert torch.equal(w1, w2)
    answer_positions = hs.answer_positions(len(prompt), [[7, 8, 9, 10], [11, 12]], mask)
    assert answer_positions == [[5, 6, 7, 8], [5, 6]]
    sums, n = attn182.group_sums(w1, 0, {**groups, **answer_group}, answer_positions, mask.tolist())
    assert n == 6 and abs(sum(sums.values()) - 6) < 1e-4
    # The deliberate break for item 7: the answer group shifted by one position (the prompt's last position counted)
    # changes the sums, so the positions test above is not vacuous.
    broken, _n = attn182.group_sums(w1, 0, {**groups, **answer_group}, [[p - 1 for p in pos] for pos in answer_positions],
                                    mask.tolist())
    assert abs(broken["answer"] - sums["answer"]) > 1e-6


def test_attn182_parse_heads():
    assert attn182.parse_heads("h0.3,h2.1,c14.0,h0.3", control_layers=[14]) == [(0, 3), (2, 1), (14, 0)]
    assert attn182.parse_heads("c14.15", control_layers=[14]) == [(14, 15)]
    with pytest.raises(ValueError, match="control layers"):
        attn182.parse_heads("c14.0", control_layers=[])
    with pytest.raises(ValueError):
        attn182.parse_heads("h12.0")
    with pytest.raises(ValueError):
        attn182.parse_heads(" , ")


# ---------------------------------------------------------------------------------------------------------------------
# run182 stage 1: the control-layer donor is loaded, the readouts are written, and resume reproduces the rows.

def test_stage1_run_writes_control_layer_rows_and_readouts_and_resumes(tmp_path, monkeypatch):
    """run182._stage1_run on tiny models (the loaders are replaced): a c14 condition is donor-swapped at layer 14, every
    row carries resid_rel (zeros under none), and a resumed run that recomputes none's states in memory writes the
    same row as the first run."""
    import types
    import run182

    class Tok:
        pad_token_id = 0

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [3 + (i * 7 + len(text)) % 90 for i in range(6)]}

    host, donor = make_model(41, layers=16), make_model(42, layers=16)
    units = [{"id": f"u{i}", "layer": i % 4} for i in range(3)]
    push = torch.ones(HIDDEN) * 0.1
    monkeypatch.setattr(run182, "load_host", lambda path, device: (Tok(), host))
    monkeypatch.setattr(run182, "load_units", lambda: (units, {}))
    monkeypatch.setattr(run182, "load_p99_answers", lambda: {
        **{(u["id"], k): "abc def ghi" for u in units for k in ("neuron", "random")},
        ("unpushed", "none"): "I am an AI"})
    monkeypatch.setattr(run182, "PUSHES", tmp_path / "pushes.pt")
    monkeypatch.setattr(torch, "load", lambda path, map_location=None: {
        f"{u['id']}|{k}": push for u in units for k in ("neuron", "random")})
    monkeypatch.setattr(hs, "load_donor_attention",
                        lambda path, layers, dtype=None: hs.donor_attention(donor, layers, dtype=dtype))
    monkeypatch.setattr(run182.c, "prompt_ids", lambda tok, question, frame: [1, 2, 3, 4])
    args = types.SimpleNamespace(host="h", donor="d", device="cpu", batch=2, threads=1)
    conditions = {"none": {}, "h0.0": {0: [0]}, "c14.1": {14: [1]}, "h1.2": {1: [2]}}
    out = tmp_path / "stage1.jsonl"
    assert run182._stage1_run(args, conditions, out, "stage1") == 0
    rows = [json.loads(line) for line in out.open(encoding="utf-8")]
    assert {r["condition"] for r in rows} == set(conditions)
    own = lambda cond, rs: next(r for r in rs if r["condition"] == cond and r["unit"] == "neuron"
                                and r["text"] == "own")
    assert own("none", rows)["resid_rel"] == {"3": 0.0, "7": 0.0, "11": 0.0, "15": 0.0}
    assert own("c14.1", rows)["heads"] == {"14": [1]}
    assert own("c14.1", rows)["resid_rel"]["15"] > 0 and own("c14.1", rows)["resid_rel"]["3"] == 0.0
    assert own("h0.0", rows)["resid_rel"]["3"] > 0
    assert own("h1.2", rows)["logp_sum"] != own("none", rows)["logp_sum"]
    # Resume: none's rows are already on disk, so its states are recomputed in memory and nothing is written for it.
    out2 = tmp_path / "stage1b.jsonl"
    out2.write_text("".join(line for line in out.open(encoding="utf-8")
                            if json.loads(line)["condition"] != "h1.2"), encoding="utf-8")
    assert run182._stage1_run(args, conditions, out2, "stage1") == 0
    rows2 = [json.loads(line) for line in out2.open(encoding="utf-8")]
    assert sum(1 for r in rows2 if r["condition"] == "none") == sum(1 for r in rows if r["condition"] == "none")
    assert own("h1.2", rows2)["logp_sum"] == own("h1.2", rows)["logp_sum"]
    assert own("h1.2", rows2)["resid_rel"] == own("h1.2", rows)["resid_rel"]
