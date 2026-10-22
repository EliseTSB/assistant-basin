from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from transformers import LlamaConfig, LlamaForCausalLM

import jlens
import jlens_units
import part42


class TinyTokenizer:
    bos_token_id = 1
    add_bos_token = False

    def __init__(self):
        self.tokens = ["top", "high", "low", "bottom"] + [
            f"token-{index}" for index in range(4, 64)
        ]

    def __call__(self, text, return_tensors=None, truncation=True, max_length=512, **_kwargs):
        ids = [3 + (ord(char) % 60) for char in text]
        if self.add_bos_token:
            ids = [self.bos_token_id] + ids
        if truncation:
            ids = ids[:max_length]
        if return_tensors == "pt":
            return SimpleNamespace(input_ids=torch.tensor([ids], dtype=torch.long))
        return {"input_ids": ids}

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return self.tokens[int(token_ids[0])]


def test_unit_reader_known_answer_from_hand_built_lens(tmp_path):
    snapshot = tmp_path / "olmo-mini"
    snapshot.mkdir()
    (snapshot / "config.json").write_text(
        '{"hidden_size":2,"num_hidden_layers":2,"model_type":"olmo2",'
        '"tie_word_embeddings":false}',
        encoding="utf-8",
    )
    checkpoint = snapshot / "model.safetensors"
    save_file(
        {
            "model.layers.0.mlp.down_proj.weight": torch.tensor(
                [[1.0, 0.0], [0.0, 1.0]]
            ),
            "model.norm.weight": torch.tensor([1.0, 2.0]),
            "lm_head.weight": torch.tensor(
                [[2.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]]
            ),
        },
        str(checkpoint),
    )
    lens_path = tmp_path / "known-lens.pt"
    torch.save(
        {
            "J": {0: torch.eye(2)},
            "n_prompts": 1,
            "source_layers": [0],
            "d_model": 2,
        },
        lens_path,
    )
    result = jlens_units.read_units(
        "olmo2-1b",
        [{"id": "known", "layer": 0, "unit": 0, "role": "test"}],
        snapshot=snapshot,
        lens_path=lens_path,
        tokenizer=TinyTokenizer(),
    )
    row = result["units"][0]
    assert row["readouts"]["logit_lens"]["top"][0]["decoded"] == "top"
    assert row["readouts"]["logit_lens"]["bottom"][0]["decoded"] == "low"
    assert row["readouts"]["j_lens"]["top"][0]["decoded"] == "top"
    assert row["readouts"]["j_lens"]["bottom"][0]["decoded"] == "low"
    assert result["lens_metadata"] == {
        "n_prompts": 1,
        "source_layers": [0],
        "d_model": 2,
    }


def test_olmo_default_units_have_target_and_ten_last_quarter_controls():
    units = jlens_units.olmo_default_units()
    assert len(units) == 11
    assert units[0]["layer"] == 10 and units[0]["unit"] == 6978
    controls = units[1:]
    assert len(controls) == 10
    assert all(row["layer"] == 14 for row in controls)
    assert len({(row["layer"], row["unit"]) for row in controls}) == 10
    assert controls[0]["role"] == "known_positive_control"


def test_tiny_random_model_fit_and_read_round_trip_cpu(tmp_path):
    torch.set_num_threads(1)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = LlamaForCausalLM(config).eval()
    tokenizer = TinyTokenizer()
    wrapped = jlens.from_hf(model, tokenizer, device="cpu")
    fitted = jlens.fit(
        wrapped,
        ["abcdefghijklmnopqrst"],
        source_layers=[0],
        target_layer=1,
        dim_batch=2,
        max_seq_len=24,
        checkpoint_every=None,
        resume=False,
    )
    lens_path = tmp_path / "tiny-lens.pt"
    fitted.save(str(lens_path))
    loaded = jlens.JacobianLens.load(str(lens_path))
    jacobians, metadata, d_model = part42.load_jacobian_lens(lens_path)
    assert metadata["n_prompts"] == 1
    assert metadata["source_layers"] == [0]
    assert d_model == 8
    assert tuple(jacobians[0].shape) == (8, 8)
    assert torch.isfinite(jacobians[0]).all()
    lens_logits, model_logits, input_ids = loaded.apply(
        wrapped,
        "abcdefghijklmnopqrst",
        layers=[0],
        positions=[-2],
        max_seq_len=24,
    )
    assert tuple(lens_logits[0].shape) == (1, 64)
    assert tuple(model_logits.shape) == (1, 64)
    assert input_ids.shape[1] > 17
