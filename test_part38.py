import numpy as np
import pytest

import part38


class CharacterOffsetTokenizer:
    """Tiny fake tokenizer whose one-character tokens split every target word."""

    def __call__(self, text, **kwargs):
        offsets = [(index, index + 1) for index in range(len(text))]
        return {"input_ids": [list(range(len(text)))], "offset_mapping": [offsets]}


class FakeHookHandle:
    def __init__(self, module):
        self.module = module

    def remove(self):
        self.module.hook = None


class FakeDownProjection:
    def __init__(self):
        self.hook = None

    def register_forward_pre_hook(self, hook):
        self.hook = hook
        return FakeHookHandle(self)

    def __call__(self, values):
        if self.hook is not None:
            replacement = self.hook(self, (values,))
            if replacement is not None:
                values = replacement[0]
        return values


class TinyFakeModel:
    def __init__(self):
        self.down_proj = FakeDownProjection()

    def forward_down_proj(self, values):
        return self.down_proj(values)


def test_target_span_finds_a_word_split_across_tokens():
    text = "A wizard arrived."
    rendered = "<user>" + text + "</user>"
    start, end = part38.find_target_token_span(CharacterOffsetTokenizer(), rendered, text, "wizard")
    assert end - start == len("wizard")
    assert start == rendered.index("wizard")


def test_generated_only_clamp_preserves_prompt_positions():
    values = np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])
    # First forward is the prompt prefill: it must be left entirely untouched.
    assert part38.apply_generated_clamp(values, unit=1, level=10.0, prompt_len=3,
                                       is_prefill=True) is None
    # An uncached forward contains the full prompt plus one generated token.
    changed = part38.apply_generated_clamp(values, unit=1, level=10.0, prompt_len=3,
                                          is_prefill=False)
    assert changed[0, :3, 1].tolist() == [2.0, 4.0, 6.0]
    assert changed[0, 3, 1] == 10.0
    fake_model = TinyFakeModel()
    hook = part38.UnitClampHook(fake_model.down_proj, unit=1, level=10.0, prompt_len=3)
    prompt_output = fake_model.forward_down_proj(values[:, :3, :])
    assert np.array_equal(prompt_output, values[:, :3, :])
    continuation = fake_model.forward_down_proj(values)
    assert continuation[0, :3, 1].tolist() == [2.0, 4.0, 6.0]
    assert continuation[0, 3, 1] == 10.0
    hook.remove()


def test_random_controls_are_stable_and_exclude_target():
    first = part38.choose_random_units(width=128, target_unit=42, n=20, seed=3801)
    second = part38.choose_random_units(width=128, target_unit=42, n=20, seed=3801)
    assert first == second
    assert len(first) == 20
    assert len(set(first)) == 20
    assert 42 not in first


def test_topic_prefix_requires_a_word_start():
    assert part38.topic_hit("Her spelling improved after practice.", ["spell"])
    assert not part38.topic_hit("The note says misspell, without any boundary.", ["spell"])


def test_bootstrap_ratio_ci_is_reproducible():
    screened = [2.0, 3.0, 4.0, 5.0]
    other = [1.0, 2.0, 3.0, 4.0]
    first = part38.bootstrap_ratio_ci(screened, other, seed=99, resamples=200)
    second = part38.bootstrap_ratio_ci(screened, other, seed=99, resamples=200)
    assert first == second
    assert first[0] <= first[1]


def test_screen_data_has_twelve_examples_per_sense_and_matched_lengths():
    data = part38.read_json(part38.DATA_PATH)
    means = part38.validate_data(data)
    assert len(means) == 9
    assert max(item["absolute_mean_difference"] for item in means.values()) <= 1.5
    assert all(len(unit["sentences"][sense]) == 12
               for unit in data["units"] for sense in ("screened", "other"))


def test_neighbour_data_has_all_preregistered_groups_and_finnish_noita():
    data = part38.read_json(part38.NEIGHBOUR_DATA_PATH)
    counts = part38.validate_neighbour_data(data)
    assert set(counts) == {"wizard", "sorcerer", "mage", "warlock", "witch",
                           "magician", "noita", "baker", "pilot", "nurse", "farmer"}
    assert len(data["groups"]["noita"]) == 12
    assert all(any(char in sentence for char in "äöåÄÖÅ") for sentence in data["groups"]["noita"])
