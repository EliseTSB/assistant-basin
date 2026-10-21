from __future__ import annotations

import hashlib
import json

import pytest
import torch
from safetensors.torch import save_file

import part42


def test_readout_math_and_identity_jacobian(tmp_path):
    checkpoint = tmp_path / "synthetic.safetensors"
    down = torch.tensor([[1.0], [2.0]])
    norm = torch.tensor([2.0, 3.0])
    head = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    save_file(
        {
            "model.layers.0.mlp.down_proj.weight": down,
            "model.norm.weight": norm,
            "lm_head.weight": head,
        },
        str(checkpoint),
    )
    weight_map = {
        "model.layers.0.mlp.down_proj.weight": checkpoint.name,
        "model.norm.weight": checkpoint.name,
        "lm_head.weight": checkpoint.name,
    }
    assert part42.safetensor_shape(
        tmp_path, weight_map, "model.layers.0.mlp.down_proj.weight"
    ) == (2, 1)
    loaded_down = part42.load_safetensor(tmp_path, weight_map, "model.layers.0.mlp.down_proj.weight")
    loaded_norm = part42.load_safetensor(tmp_path, weight_map, "model.norm.weight")
    loaded_head = part42.load_safetensor(tmp_path, weight_map, "lm_head.weight")
    write = loaded_down[:, 0]

    lens_path = tmp_path / "fake-lens.pt"
    fake_lens = {
        "n_prompts": 1,
        "source_layers": [0],
        "d_model": 2,
        "J": {0: torch.eye(2)},
    }
    torch.save(fake_lens, lens_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("fit:\n  n_prompts: 1\n", encoding="utf-8")
    checksum_path = tmp_path / "SHA256SUMS"
    lens_sha256 = hashlib.sha256(lens_path.read_bytes()).hexdigest()
    checksum_path.write_text(f"{lens_sha256}  fake-lens.pt\n", encoding="utf-8")
    jacobians, metadata, d_model = part42.load_jacobian_lens(lens_path)
    provenance = part42.build_lens_provenance(lens_path, metadata, checksum_path)
    assert metadata == {"n_prompts": 1, "source_layers": [0], "d_model": 2}
    assert d_model == 2
    assert provenance["checksum_verified"] is True
    assert provenance["n_prompts"] == 1
    assert provenance["source_layers"] == [0]
    assert provenance["config_yaml"] == "fit:\n  n_prompts: 1\n"

    logits = part42.readout_logits(torch, loaded_head, write, loaded_norm)
    assert torch.equal(logits, torch.tensor([14.0, 30.0]))
    identity_logits = part42.readout_logits(
        torch, loaded_head, write, loaded_norm, jacobian=jacobians[0]
    )
    assert torch.equal(identity_logits, logits)


def test_concept_matcher_handles_case_spaces_and_short_terms():
    assert part42.concept_word_matches("  New York", "newyork")
    assert part42.concept_word_matches("Wizardry", "wizard")
    assert part42.concept_word_matches("AI", "ai")
    assert not part42.concept_word_matches("said", "ai")
    tokens = [{"decoded": " SPELL"}, {"decoded": "wand"}]
    assert part42.matched_words(tokens, ["spell", "magic", "wand"]) == ["spell", "wand"]
    assert part42.concept_matches(tokens, ["wizard", "owl"]) == []


def test_language_model_prefix_comes_from_index_keys():
    fake_index = {
        "weight_map": {
            "model.visual.layers.0.proj.weight": "a.safetensors",
            "model.language_model.layers.0.mlp.down_proj.weight": "b.safetensors",
            "model.language_model.layers.1.mlp.down_proj.weight": "b.safetensors",
            "lm_head.weight": "b.safetensors",
        }
    }
    assert part42.find_language_model_prefix(
        fake_index, require_language_model=True
    ) == "model.language_model."


def test_four_b_unit_list_is_pinned():
    assert part42.FOUR_B_UNIT_LIST == (
        (6, 753),
        (7, 1169),
        (11, 8868),
        (29, 776),
        (22, 2742),
    )
    units = part42.targets_for_model("Qwen/Qwen3.5-4B")
    assert [(unit["layer"], unit["unit"]) for unit in units] == list(part42.FOUR_B_UNIT_LIST)


def test_qwen_rmsnorm_gain_conventions():
    stored = torch.tensor([0.0, -0.25])
    assert torch.equal(part42.effective_norm_gain(stored, "qwen3"), stored)
    assert torch.equal(
        part42.effective_norm_gain(stored, "qwen3_5_text"),
        torch.tensor([1.0, 0.75]),
    )


def test_jacobian_layer_tracks_block_output_and_final_identity():
    assert [part42.jacobian_layer_for_write(layer, 4) for layer in range(4)] == [
        0,
        1,
        2,
        None,
    ]
    with pytest.raises(ValueError, match="outside model range"):
        part42.jacobian_layer_for_write(4, 4)


def test_part39_passing_unit_loader_keeps_separate_role(tmp_path):
    source = tmp_path / "part39.json"
    source.write_text(
        json.dumps({"passing_units": [{"layer": 1, "unit": index} for index in range(211)]}),
        encoding="utf-8",
    )
    units = part42.load_part39_units(source)
    assert len(units) == 211
    assert units[0]["role"] == "part39_passing"
    assert (units[-1]["layer"], units[-1]["unit"]) == (1, 210)


def test_part39_9_uses_logit_top30_and_includes_final_block_units():
    def unit(unit_id, layer, logit_token, jacobian_token):
        return {
            "id": unit_id,
            "layer": layer,
            "unit": 1,
            "role": "part39_passing",
            "j_identity": layer == 31,
            "readouts": {
                "logit": {"top": [{"token_id": 1, "decoded": logit_token}]},
                "jacobian": {"top": [{"token_id": 2, "decoded": jacobian_token}]},
            },
        }

    document = {
        "Qwen/Qwen3.5-4B": {
            "units": [
                unit("part39-final", 31, "assistant", "assistant"),
                unit("part39-j-only", 20, "unrelated", "assistant"),
            ]
        }
    }
    result = part42.score_documents(document, [])["p39_9"]
    assert result["source_unit_count"] == 2
    assert result["evaluated_unit_count"] == 2
    assert result["readout"] == "logit"
    assert result["logit_lens"]["units_with_meaning_words"] == {
        "part39-final": ["assistant"]
    }
    assert result["holds"] is True


def test_42_5_counts_repeated_concepts_across_random_units():
    def random_unit(index, word):
        return {
            "id": f"random-{index:02d}",
            "layer": 1,
            "unit": index,
            "role": "random",
            "j_identity": False,
            "readouts": {
                readout: {"top": [{"token_id": index, "decoded": word}]}
                for readout in ("logit", "jacobian")
            },
        }

    words = ["owl", "dragon"] + ["unrelated"] * 18
    document = {
        "Qwen/Qwen3-1.7B": {
            "units": [random_unit(index, word) for index, word in enumerate(words)]
        }
    }
    concepts = ["owl", "dragon"]
    score = part42.score_documents(document, concepts)["predictions"]["42.5"]
    assert score["per_readout"]["logit"]["unit_count_naming_any_concept"] == 2
    assert score["holds"] is True

    repeated = list(words)
    repeated[1] = "owl"
    document["Qwen/Qwen3-1.7B"]["units"] = [
        random_unit(index, word) for index, word in enumerate(repeated)
    ]
    score = part42.score_documents(document, concepts)["predictions"]["42.5"]
    assert score["per_readout"]["jacobian"]["concept_unit_counts"]["owl"] == 2
    assert score["holds"] is False


def test_concept_word_matches_refuses_buried_words():
    from part42 import concept_word_matches as m
    assert not m(" Application", "cat")      # the first build's false hit
    assert not m("And", "candle")
    assert not m(" knowledge", "owl")
    assert not m(" Dra", "dragon")           # three-character pieces do not count
    assert m(" Cat", "cat") and m(" AI", "ai") and m("中国", "中国")
    assert m(" Spells", "spell") and m(" sorcerer", "sorcer") and m(" chines", "chinese")
    assert m("_magic", "magic") and m("/google", "google")
