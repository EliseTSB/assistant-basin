"""CPU-only tests for Part 48b using a character tokenizer and a tiny causal LM."""
from __future__ import annotations

import sys
import pytest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import part48 as p48
import part48b as p48b


class TinyTokenizer:
    bos_token_id = 1
    eos_token_id = 1

    def __call__(self, text, add_special_tokens=False):
        ids = [ord(char) + 2 for char in text]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids
        return {"input_ids": ids}

    def convert_ids_to_tokens(self, ids):
        return ["<bos>" if i == self.bos_token_id else chr(i - 2) for i in ids]

    def decode(self, ids):
        return "".join("" if i == self.bos_token_id else chr(i - 2) for i in ids)


class TinyCausalLM(torch.nn.Module):
    """Small deterministic model with a known k-sharp / c-round preference."""
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.config = type("Config", (), {"bos_token_id": tokenizer.bos_token_id})()
        self.vocab_size = 140

    def forward(self, input_ids):
        batch, length = input_ids.shape
        logits = torch.zeros((batch, length, self.vocab_size), dtype=torch.float32,
                             device=input_ids.device)
        candidates = [(word, "sharp") for word in p48.SHARP_WORDS]
        candidates += [(word, "round") for word in p48.ROUND_WORDS]
        for bi in range(batch):
            sequence = input_ids[bi].tolist()
            for word, category in candidates:
                continuation = self.tokenizer(" " + word, add_special_tokens=False)["input_ids"]
                if sequence[-len(continuation):] != continuation:
                    continue
                start = len(sequence) - len(continuation)
                prompt = self.tokenizer.decode(sequence[:start])
                side = None
                for k_word, c_word in p48b.PAIRS:
                    if k_word in prompt:
                        side = "k"
                        break
                    if c_word in prompt:
                        side = "c"
                        break
                favor = (side == "k" and category == "sharp") or (side == "c" and category == "round")
                target_bias = 3.0 if favor else -3.0
                for position in range(max(0, start - 1), len(sequence) - 1):
                    next_id = sequence[position + 1]
                    logits[bi, position, next_id] = target_bias
        return type("Output", (), {"logits": logits})()


def test_sign_flip_fixture_and_fixed_seed():
    # For three equal positive differences, the exact sign-flip tail is 2/8.
    first = p48b.sign_flip_permutation_test([1, 1, 1], draws=10_000, seed=48)
    second = p48b.sign_flip_permutation_test([1, 1, 1], draws=10_000, seed=48)
    assert first == second
    assert first["observed"] == 1.0
    assert abs(first["p_value"] - 0.25) < 0.03


def test_bootstrap_median_fixture_and_fixed_seed():
    # With [0, 0, 10], the median bootstrap mass is 7/8 at 0 and 1/8 at 10.
    first = p48b.bootstrap_median_ci([0, 0, 10], draws=10_000, seed=49)
    second = p48b.bootstrap_median_ci([0, 0, 10], draws=10_000, seed=49)
    assert first == second
    assert first["estimate"] == 0
    assert first["lower"] == 0.0
    assert first["upper"] == 10.0


def test_letter_matched_sets_drop_only_spiky_and_curvy():
    assert p48b.LETTER_MATCHED_SHARP_WORDS == ("sharp", "pointy", "jagged", "angular")
    assert p48b.LETTER_MATCHED_ROUND_WORDS == ("round", "smooth", "soft", "rounded")
    assert set(p48b.LETTER_MATCHED_SHARP_WORDS) == set(p48.SHARP_WORDS) - {"spiky"}
    assert set(p48b.LETTER_MATCHED_ROUND_WORDS) == set(p48.ROUND_WORDS) - {"curvy"}


def test_tiny_model_produces_positive_k_minus_c_d_for_both_scores():
    tokenizer = TinyTokenizer()
    model = TinyCausalLM(tokenizer).eval()
    pair = p48b.PAIRS[0]
    rows = p48b.score_model(model, tokenizer, "cpu", pairs=(pair,), include_controls=False)
    summary = p48b.summarize(rows, pairs=(pair,), mass_floor=0.0)  # the tiny model puts almost no mass on the words
    full = summary["scores"]["part48_continuations"]["all_pairs"]
    matched = summary["scores"]["letter_matched"]["all_pairs"]
    assert full["pair_d"][0] > 0
    assert matched["pair_d"][0] > 0
    assert len(full["per_template"]) == len(p48.TEMPLATES)
    assert all(entry["pair_d"][0] > 0 for entry in full["per_template"])



def test_mass_floor_drops_a_low_mass_template_like_part48():
    # Claude, 2026-10-21: Part 48 averages only over templates whose median continuation mass is >= 1e-3.
    tokenizer = TinyTokenizer()
    model = TinyCausalLM(tokenizer).eval()
    pair = p48b.PAIRS[0]
    rows = p48b.score_model(model, tokenizer, "cpu", pairs=(pair,), include_controls=False)
    for r in rows:
        r["continuation_mass"] = 1e-6 if r["template"] == 3 else 0.5
    recs = p48b._pair_records(rows, (pair,))
    assert recs[0]["templates_kept"] == [t for t in range(len(p48.TEMPLATES)) if t != 3]
    for r in rows:
        r["continuation_mass"] = 1e-6
    with pytest.raises(RuntimeError, match="mass floor"):
        p48b._pair_records(rows, (pair,))
