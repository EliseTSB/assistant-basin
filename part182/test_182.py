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

import headswap182 as hs  # noqa: E402
import score182  # noqa: E402
from swap99 import copy_components  # noqa: E402

HEADS, HEAD_DIM, HIDDEN = 4, 8, 32


def make_model(seed, kv_heads=HEADS):
    torch.manual_seed(seed)
    config = Olmo2Config(vocab_size=96, hidden_size=HIDDEN, intermediate_size=64, num_hidden_layers=3,
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
    tags = hs.stage1_conditions(layers, 16)
    assert len(tags) == 1 + 64 + 4 + 1 + 64 and tags[0] == "none" and tags[-1] == "restore3.15"


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

def fixture_rows(effects: dict[str, float], slots, concepts, layers=(0, 1), n_heads=2):
    """Stage-1 rows for a fixture: own-answer logp_sum = -5 + effect per unit per condition; the other row kinds
    constant."""
    rows = []
    tags = hs.stage1_conditions(list(layers), n_heads)
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
