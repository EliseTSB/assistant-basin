import pytest
import torch
import numpy as np

import part25
import part39


class CharacterOffsetTokenizer:
    """Small tokenizer with one token per character and a deterministic chat template."""

    chat_template = "tiny"
    pad_token_id = 0
    eos_token_id = 1
    padding_side = "right"

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs.get("add_generation_prompt") is True
        content = messages[-1]["content"]
        return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"

    def __call__(self, text, **kwargs):
        offsets = [(index, index + 1) for index in range(len(text))]
        result = {"input_ids": list(range(len(text)))}
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = offsets
        return result


class TinyRandomDownProjection(torch.nn.Module):
    def __init__(self, width=760):
        super().__init__()
        self.in_features = width
        self.weight = torch.nn.Parameter(torch.randn(width), requires_grad=False)
        self.last_input = None

    def forward(self, values):
        self.last_input = values.detach().clone()
        return values * self.weight


class TinyRandomMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = TinyRandomDownProjection()


class TinyRandomLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = TinyRandomMLP()


class TinyRandomModel(torch.nn.Module):
    """A seven-layer coordinate-wise random model; no tokenizer or weights are downloaded."""

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([TinyRandomLayer() for _ in range(7)])

    def forward(self, values):
        for layer in self.layers:
            values = layer.mlp.down_proj(values)
        return values


def test_hook_zeroes_and_clamps_only_unit_753_at_every_position():
    model = TinyRandomModel()
    layers = model.layers
    values = torch.randn(2, 5, 760)
    model(values.clone())
    baseline_at_layer6 = layers[6].mlp.down_proj.last_input

    zero = part25._operations([(6, 753)], "zero")
    handles = part25.install_intervention_hooks(layers, [zero, zero])
    model(values.clone())
    output = layers[6].mlp.down_proj.last_input
    for handle in handles:
        handle.remove()
    assert torch.equal(output[..., 753], torch.zeros_like(output[..., 753]))
    assert torch.equal(output[..., :753], baseline_at_layer6[..., :753])
    assert torch.equal(output[..., 754:], baseline_at_layer6[..., 754:])

    clamp = {(6, 753): 12.5}
    handles = part25.install_intervention_hooks(layers, [clamp, clamp])
    model(values.clone())
    output = layers[6].mlp.down_proj.last_input
    for handle in handles:
        handle.remove()
    assert torch.equal(output[..., 753], torch.full_like(output[..., 753], 12.5))
    assert torch.equal(output[..., :753], baseline_at_layer6[..., :753])
    assert torch.equal(output[..., 754:], baseline_at_layer6[..., 754:])


def test_assistant_marker_index_is_found_in_templated_tokens():
    tokenizer = CharacterOffsetTokenizer()
    sentence = "Can you help me fix this loop?"
    sample = part39.prepare_template_sample(tokenizer, sentence)
    assert sample["assistant_marker_index"] == sample["rendered"].rfind("assistant")
    assert len(sample["content_positions"]) == len(sentence)
    assert sample["input_ids"][sample["assistant_marker_index"]:sample["assistant_marker_index"] + 9]
    assert sample["rendered"][sample["assistant_marker_index"]:
                               sample["assistant_marker_index"] + 9] == "assistant"


def test_resume_skips_a_row_key_instead_of_duplicating_it(tmp_path):
    path = tmp_path / "rows.jsonl"
    first = part39.JsonlSink(path)
    row = {"row_key": "same-key", "answer": "first"}
    assert first.append(row)
    resumed = part39.JsonlSink(path)
    assert not resumed.append({"row_key": "same-key", "answer": "second"})
    assert len(part39.read_jsonl(path)) == 1
    assert part39.read_jsonl(path)[0]["answer"] == "first"


def test_scorer_rules_on_hand_examples():
    assert part39.score_answer("I am an AI language model today.")["ai_claim"]
    assert part39.score_answer("An assistant works well for the whole team.")["ai_claim"]
    assert not part39.score_answer("This model works well for cooking.")["ai_claim"]
    assert not part39.score_answer("AI")["ai_claim"]
    repeated = "I am a helpful assistant. " * 3
    result = part39.score_answer(repeated)
    assert result["broken"]
    assert not result["ai_claim"]
    assert part39.score_answer("Alibaba built the bridge last summer.")["maker_named"]
    assert len(part39.validate_battery(part39._read_json(part39.BATTERY_PATH))) == 6


def test_screen_gate_finds_planted_addressee_unit_and_reports_request_contrast():
    rng = np.random.default_rng(3901)
    labels = np.repeat(np.arange(6, dtype=np.int8), 20)
    values = rng.normal(0.0, 0.1, size=(120, 100)).astype(np.float16)
    # Unit 0 fires only for helper requests and AI self-descriptions: categories 1 and 3.
    values[np.isin(labels, (0, 2)), 0] += 40.0
    # Unit 1 fires on every second-person sentence, with a larger response to requests.
    values[labels == 0, 1] += 20.0
    values[labels == 1, 1] += 2.0
    markers = rng.normal(0.0, 0.1, size=(120, 100)).astype(np.float16)
    stats = part39.compute_screen_statistics(values, markers, labels, seed=3901, shuffles=200)
    assert stats["pass_mask"][0]
    assert stats["request_only_contrast"][1] > 100.0


def test_wrong_layer_mutant_would_fail_target_clamp_assertion():
    """Mutant check: routing the target clamp to layer 5 leaves layer 6 untouched."""
    model = TinyRandomModel()
    layers = model.layers
    values = torch.randn(1, 3, 760)
    wrong_layer = {(5, 753): 12.5}
    handles = part25.install_intervention_hooks(layers, [wrong_layer])
    model(values.clone())
    observed_at_target_layer = layers[6].mlp.down_proj.last_input
    for handle in handles:
        handle.remove()
    # This is the assertion used by the main hook check; the wrong-layer mutant makes it red.
    with pytest.raises(AssertionError):
        assert torch.all(observed_at_target_layer[..., 753] == 12.5)


def test_target_is_the_design_unit():
    """Host addition: the wrong-layer mutant on part39.TARGET survived every test above, which build their own hooks."""
    assert part39.TARGET == {"layer": 6, "unit": 753}
    assert part39.SCREEN_WIDTH == 9216
