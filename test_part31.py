#!/usr/bin/env python3
"""Fast standard-library tests for Part 31 mechanics; no model or torch required."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import part31


class FakeHandle:
    def __init__(self, remove):
        self._remove = remove

    def remove(self):
        self._remove()


class FakeDownProj:
    def __init__(self):
        self.hooks = []

    def register_forward_pre_hook(self, callback):
        self.hooks.append(callback)

        def remove():
            self.hooks.remove(callback)

        return FakeHandle(remove)

    def __call__(self, activation):
        args = (activation,)
        for hook in list(self.hooks):
            replacement = hook(self, args)
            if replacement is not None:
                args = replacement
        return args[0]


def fake_model():
    down = FakeDownProj()
    layer = SimpleNamespace(mlp=SimpleNamespace(down_proj=down))
    return SimpleNamespace(model=SimpleNamespace(layers=[layer])), down


class CacheOutput:
    def __init__(self, tokens):
        self.past_key_values = tuple(tokens)


class FakeCacheModel:
    def __init__(self):
        self.seen = []

    def __call__(self, input_ids, attention_mask=None, use_cache=True):
        tokens = [int(x) for x in input_ids[0]]
        self.seen.append(tokens)
        return CacheOutput(tokens)


class CharTokenizer:
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ids = [ord(char) for char in text]
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return result

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(int(token)) for token in token_ids)


class Part31Tests(unittest.TestCase):
    def test_clamp_schedule_samples_tokens_1_to_k_from_clamped_forwards(self):
        # Claude's fold 2026-10-20: the build clamped the forwards OF tokens 1..K, so token 1
        # was sampled unclamped and token K+1 from a clamped forward (the cache split then
        # compared a direct clamp effect, not residue). Token t must be sampled from a forward
        # the window clamps exactly when t <= K, and the prompt forward is clamped at its
        # final position only.
        k = 3
        window = part31.ClampWindow(k)

        class FakeSession:
            def __init__(self):
                self.calls = []
                self.sampled_after = []

            def forward_many(self, rows, past=None, full_length=None, step=0, window=None, last_only=False):
                if window is not None:
                    window.set_step(step, last_only)
                active = window is not None and 1 <= window.step <= window.clamp_until
                self.calls.append((len(rows[0]), active, window.last_only if window else False))
                return [("logits", active)] * len(rows), "past", [[0.0]] * len(rows), None

            def forward(self, token_ids, past=None, full_length=None, step=0, window=None, last_only=False):
                logits, cache, states, out = self.forward_many([token_ids], past, full_length, step, window, last_only)
                return logits[0], cache, states[0], out

            def make_generator(self, seed):
                return None

            def sample_token(self, logits, generator):
                self.sampled_after.append(logits[1])
                return 7

        prompt = [1, 2, 3, 4]
        for fn in ("batch", "single"):
            session = FakeSession()
            if fn == "batch":
                part31._generate_clamp_prefix_batch(session, prompt, k, [0, 1], window)
            else:
                part31._generate_clamp_prefix(session, prompt, k, 0, window)
            self.assertEqual(session.calls[0], (len(prompt), True, True), fn)
            per_row = session.sampled_after[:: 2 if fn == "batch" else 1]
            self.assertEqual(per_row, [True] * k, fn)
            # The forward feeding token K (whose logits give token K+1) is unclamped.
            self.assertEqual(session.calls[-1][1], False, fn)
            self.assertEqual(window.step, 0, fn)

    def test_last_only_clamp_zeroes_only_the_final_position(self):
        value = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]
        out = part31.apply_clamp_window(value, [0, 2], 1, 1, last_only=True)
        self.assertEqual(out, [[[1.0, 2.0, 3.0], [0, 5.0, 0]]])
        try:
            import torch
        except ImportError:
            return
        t = torch.ones(2, 3, 4)
        z = part31.apply_clamp_window(t, [1, 3], 1, 2, last_only=True)
        self.assertEqual(float(z[:, :-1].sum()), 2 * 2 * 4)
        self.assertEqual(z[:, -1].tolist(), [[1.0, 0.0, 1.0, 0.0]] * 2)
        self.assertEqual(float(t.sum()), 24.0)

    def test_clamp_window_zeroes_only_selected_units_for_first_k_steps(self):
        model, down = fake_model()
        window = part31.ClampWindow(2)
        handles = part31.register_clamp_hooks(model, {0: [1, 3]}, window)
        original = [[[1.0, 2.0, 3.0, 4.0]]]
        window.set_step(0)
        self.assertEqual(down(original), original)
        window.set_step(1)
        self.assertEqual(down(original), [[[1.0, 0, 3.0, 0]]])
        window.set_step(2)
        self.assertEqual(down(original), [[[1.0, 0, 3.0, 0]]])
        window.set_step(3)
        # This fails if a clamp leaks one token past K.
        self.assertIs(down(original), original)
        for handle in handles:
            handle.remove()

    def test_cache_split_uses_same_k_tokens_and_distinct_cache_arms(self):
        model = FakeCacheModel()
        prompt = [10, 11]
        pushed = [20, 21, 22]
        clamped_cache = ("clamped", tuple(prompt), tuple(pushed))
        arms = part31.split_cache_arms(model, prompt, pushed, clamped_cache)
        self.assertEqual(arms["clamped_cache"]["shared_prefix_tokens"], pushed)
        self.assertEqual(arms["reencoded_unclamped"]["shared_prefix_tokens"], pushed)
        self.assertEqual(model.seen, [prompt + pushed])
        self.assertNotEqual(
            arms["clamped_cache"]["past_key_values"],
            arms["reencoded_unclamped"]["past_key_values"],
        )

    def test_projection_math(self):
        actual = part31.project_residuals([[3, 4], [-1, 1]], [1, 2])
        self.assertEqual(actual, [11.0, 1.0])

    def test_state_and_text_return_times_and_gap(self):
        trace = [9, 9, 0, 1, 1, 1, 1, 1, 1, 1, 1, 7]
        band = {"low": -0.1, "high": 1.1}
        self.assertEqual(part31.state_return_time(trace, band, push_length=2, stay_tokens=8), 1)
        sample = {
            "sample_id": "x",
            "stage": "Instruct",
            "instrument": "prefill",
            "K": 2,
            "cache_arm": "none",
            "prompt_id": "p00",
            "sample_index": 0,
            "projection_trace": trace,
            "sentence_start_tokens": [1, 5],
            "sentence_end_tokens": [1, 8],
        }
        labels = [
            {"sentence_index": 0, "label": "human_persona"},
            {"sentence_index": 1, "label": "assistant_persona"},
        ]
        scored = part31.score_sample(sample, band, labels)
        self.assertEqual(scored["state_return_time"], 1)
        self.assertEqual(scored["text_return_time"], 6)
        self.assertEqual(scored["return_gap_text_minus_state"], 5)
        self.assertIsNone(
            part31.text_return_time(
                sample,
                [{"sentence_index": 0, "label": "assistant_persona"}],
                push_length=2,
            )
        )

    def test_unreturned_sample_is_not_given_an_invented_time(self):
        self.assertIsNone(part31.state_return_time([5] * 20, {"low": 0, "high": 1}, 8))

    def test_baseline_band_uses_central_ninety_percent(self):
        band = part31.baseline_band(list(range(100)))
        self.assertAlmostEqual(band["low"], 4.95)
        self.assertAlmostEqual(band["high"], 94.05)

    def test_prefill_token_stream_is_exactly_k_and_starts_with_design_text(self):
        tokenizer = CharTokenizer()
        for kind, expected in (
            ("prefill", part31.SAM_PREFILL_START),
            ("control_prefill", part31.CONTROL_PREFILL_START),
        ):
            for k in (8, 32, 96):
                tokens = part31._tokenize_prefill(tokenizer, kind, k)
                self.assertEqual(len(tokens), k)
                self.assertEqual(tokenizer.decode(tokens), (expected + (part31.SAM_PREFILL_MORE if kind == "prefill" else part31.CONTROL_PREFILL_MORE) * 5)[:k])

    def test_score_and_judge_prep_files_round_trip(self):
        baseline = {
            "sample_id": "Instruct/baseline/none/none/p00/0",
            "stage": "Instruct",
            "instrument": "baseline",
            "K": None,
            "cache_arm": "none",
            "prompt_id": "p00",
            "sample_index": 0,
            "projection_trace": list(range(10)),
            "text": "A baseline answer.",
            "sentence_end_tokens": [3],
        }
        pushed = {
            "sample_id": "Instruct/prefill/2/none/p00/0",
            "stage": "Instruct",
            "instrument": "prefill",
            "K": 2,
            "cache_arm": "none",
            "prompt_id": "p00",
            "sample_index": 0,
            "projection_trace": [9, 9] + [1] * 10,
            "text": "I am tired. I am an assistant.",
            "sentence_end_tokens": [3, 8],
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with (out / part31.SAMPLE_FILE).open("w", encoding="utf-8") as f:
                for row in (baseline, pushed):
                    f.write(json.dumps(row) + "\n")
            labels = out / "labels.jsonl"
            labels.write_text(
                json.dumps({"sample_id": baseline["sample_id"], "sentence_index": 0, "label": "assistant_persona"})
                + "\n"
                + json.dumps({"sample_id": pushed["sample_id"], "sentence_index": 1, "label": "assistant_persona"})
                + "\n",
                encoding="utf-8",
            )
            summary = part31.score_files(out, labels)
            scored = part31._jsonl_rows(out / part31.SCORE_FILE)
            self.assertEqual(summary["samples"], 2)
            self.assertEqual(scored[1]["state_return_time"], 1)
            self.assertEqual(scored[1]["text_return_time"], 6)
            self.assertEqual(scored[1]["return_gap_text_minus_state"], 5)
            self.assertEqual(summary["baseline_persona_mix"]["Instruct"]["target"], "assistant_persona")
            self.assertIn("histogram", summary["distributions"][0]["state_return"])
            self.assertEqual(part31.judge_prep(out), 3)
            prep_rows = part31._jsonl_rows(out / part31.JUDGE_PREP_FILE)
            self.assertEqual(len(prep_rows), 3)
            self.assertEqual(prep_rows[-1]["label_choices"], ["assistant_persona", "human_persona", "neither"])


if __name__ == "__main__":
    unittest.main()
