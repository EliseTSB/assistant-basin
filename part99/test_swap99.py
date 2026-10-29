"""Unit checks for Part 99 component transfer and metadata helpers."""
import json

import pytest
import torch
from transformers import Olmo2Config, Olmo2ForCausalLM

from part99.swap99 import (
    assert_tokenizers_compatible,
    copy_components,
    write_meta,
)


def make_model(seed, intermediate_size=128):
    torch.manual_seed(seed)
    config = Olmo2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=intermediate_size,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = Olmo2ForCausalLM(config)
    model.eval()
    return model


def snapshot_pair(tmp_path, host_seed, donor_seed, donor_intermediate_size=128):
    """Exercise tiny, independently seeded OLMo-2 snapshots without network access."""
    host_path, donor_path = tmp_path / "host", tmp_path / "donor"
    make_model(host_seed).save_pretrained(host_path, safe_serialization=False)
    make_model(donor_seed, donor_intermediate_size).save_pretrained(donor_path, safe_serialization=False)
    host = Olmo2ForCausalLM.from_pretrained(host_path, local_files_only=True)
    donor = Olmo2ForCausalLM.from_pretrained(donor_path, local_files_only=True)
    return host, donor


def component_state(model, name):
    return [
        {key: value.detach().clone() for key, value in getattr(layer, name).named_parameters()}
        for layer in model.model.layers
    ]


def assert_component_equal(model, name, expected):
    for layer, expected_params in zip(model.model.layers, expected):
        actual = dict(getattr(layer, name).named_parameters())
        assert actual.keys() == expected_params.keys()
        for key, value in expected_params.items():
            assert torch.equal(actual[key], value), f"{name}.{key} differs"


def test_mlp_swap_copies_all_mlp_parameters_and_preserves_attention(tmp_path):
    host, donor = snapshot_pair(tmp_path, 10, 20)
    host_attn = component_state(host, "self_attn")
    donor_mlp = component_state(donor, "mlp")

    result = copy_components(host, donor, "mlp")

    assert result["tensors_copied"] == sum(len(layer) for layer in donor_mlp)
    assert result["elements_copied"] == sum(t.numel() for layer in donor_mlp for t in layer.values())
    assert_component_equal(host, "mlp", donor_mlp)
    assert_component_equal(host, "self_attn", host_attn)


def test_attention_swap_copies_all_attention_parameters_and_preserves_mlp(tmp_path):
    host, donor = snapshot_pair(tmp_path, 30, 40)
    host_mlp = component_state(host, "mlp")
    donor_attn = component_state(donor, "self_attn")

    result = copy_components(host, donor, "attn")

    assert result["tensors_copied"] == sum(len(layer) for layer in donor_attn)
    assert_component_equal(host, "self_attn", donor_attn)
    assert_component_equal(host, "mlp", host_mlp)


def test_layer_limited_attention_swap_copies_only_those_layers(tmp_path):
    host, donor = snapshot_pair(tmp_path, 31, 41)
    host_attn = component_state(host, "self_attn")
    donor_attn = component_state(donor, "self_attn")
    host_mlp = component_state(host, "mlp")

    result = copy_components(host, donor, "attn", layers={1})

    assert result["tensors_copied"] == len(donor_attn[1])
    assert all(name.startswith("layers.1.") for name in result["copied_parameters"])
    assert_component_equal(host, "self_attn", [host_attn[0], donor_attn[1]])
    assert_component_equal(host, "mlp", host_mlp)
    with pytest.raises(ValueError, match="non-empty subset"):
        copy_components(host, donor, "attn", layers={2})


def test_none_swap_leaves_every_parameter_unchanged(tmp_path):
    host, donor = snapshot_pair(tmp_path, 50, 60)
    before = {key: value.detach().clone() for key, value in host.state_dict().items()}

    result = copy_components(host, donor, "none")

    assert result["tensors_copied"] == 0
    assert result["elements_copied"] == 0
    assert all(torch.equal(value, host.state_dict()[key]) for key, value in before.items())


def test_shape_mismatch_raises_before_copying_any_layer(tmp_path):
    host, donor = snapshot_pair(tmp_path, 70, 80, donor_intermediate_size=160)
    before = {key: value.detach().clone() for key, value in host.state_dict().items()}

    with pytest.raises(ValueError, match="shape mismatch"):
        copy_components(host, donor, "mlp")

    assert all(torch.equal(value, host.state_dict()[key]) for key, value in before.items())


class FakeTokenizer:
    def __init__(self, size=16, offset=0):
        self.size = size
        self.offset = offset

    def __len__(self):
        return self.size

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [(ord(char) + self.offset) % self.size for char in text]}


def test_tokenizer_equality_guard_checks_vocab_and_fixed_string_ids():
    assert_tokenizers_compatible(FakeTokenizer(), FakeTokenizer())
    with pytest.raises(ValueError, match="vocabulary size"):
        assert_tokenizers_compatible(FakeTokenizer(), FakeTokenizer(size=17))
    with pytest.raises(ValueError, match="different ids"):
        assert_tokenizers_compatible(FakeTokenizer(), FakeTokenizer(offset=1))


def test_meta_file_records_swap(tmp_path):
    result = {
        "tensors_copied": 2,
        "elements_copied": 12,
        "copied_parameters": ["layers.0.self_attn.q_proj.weight", "layers.1.self_attn.q_proj.weight"],
    }
    path = tmp_path / "case.meta.json"
    write_meta(
        tmp_path, "case", host="host-snapshot", donor="donor-snapshot", swap="attn",
        tokenizer_source="tokenizer-snapshot", swap_result=result, frame="chat", device="cpu",
        torch_version="test-torch", transformers_version="test-transformers", seconds=1.25,
    )
    assert path.is_file()
    meta = json.loads(path.read_text())
    assert meta["swap"] == "attn"
    assert meta["tensors_copied"] == 2
    assert meta["elements_copied"] == 12


def test_extrapolated_attention_is_twice_host_minus_donor_and_preserves_mlp(tmp_path):
    """Part 104: --extrapolate moves the host the donor's distance in the opposite direction."""
    host, donor = snapshot_pair(tmp_path, 32, 42)
    host_attn = component_state(host, "self_attn")
    donor_attn = component_state(donor, "self_attn")
    host_mlp = component_state(host, "mlp")

    result = copy_components(host, donor, "attn", extrapolate=True)

    assert result["mode"] == "extrapolate"
    assert result["tensors_copied"] == sum(len(layer) for layer in donor_attn)
    assert abs(result["distance_ratio"] - 1.0) < 1e-5
    for layer, h, d in zip(host.model.layers, host_attn, donor_attn):
        actual = dict(layer.self_attn.named_parameters())
        for key in h:
            assert torch.allclose(actual[key], 2 * h[key] - d[key], atol=1e-6), key
    assert_component_equal(host, "mlp", host_mlp)


def test_extrapolate_in_bfloat16_keeps_the_distance_near_the_donors(tmp_path):
    """The real runs are bf16 with a donor about 0.5% away; the cast must not shrink or grow the step much."""
    host, _ = snapshot_pair(tmp_path, 33, 43)
    host = host.to(torch.bfloat16)
    donor = make_model(33).to(torch.bfloat16)
    torch.manual_seed(7)
    with torch.no_grad():
        for p in donor.parameters():
            p.add_(p * 0.005 * torch.randn_like(p))
    result = copy_components(host, donor, "attn", extrapolate=True)
    assert 0.9 < result["distance_ratio"] < 1.1


def test_copy_mode_is_recorded_and_unchanged_by_the_new_option(tmp_path):
    host, donor = snapshot_pair(tmp_path, 34, 44)
    donor_attn = component_state(donor, "self_attn")
    result = copy_components(host, donor, "attn")
    assert result["mode"] == "copy" and "distance_ratio" not in result
    assert_component_equal(host, "self_attn", donor_attn)
    write_meta(
        tmp_path, "case", host="h", donor="d", swap="attn", tokenizer_source="t", swap_result=result, frame="chat",
        device="cpu", torch_version="x", transformers_version="y", seconds=0.1,
    )
    meta = json.loads((tmp_path / "case.meta.json").read_text())
    assert meta["mode"] == "copy" and meta["distance_ratio"] is None


def test_extrapolate_without_a_swap_is_refused():
    from part99.swap99 import main
    with pytest.raises(SystemExit):
        main(["--host", "h", "--neurons", "n.json", "--frame", "chat", "--tag", "t", "--out", "o", "--extrapolate"])
