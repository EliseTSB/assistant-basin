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

    def test_base_prompt_renders_the_instruct_template_as_plain_text(self):
        # Fold r1 (Sol #1): base got "User: ...\nAssistant:" instead of the Instruct template.
        import sys, types
        nr = types.ModuleType("neuron_report")
        def build_prompt(tok, q, plain, system_prompt=None):
            if plain or not getattr(tok, "chat_template", None):
                return f"User: {q}\nAssistant:"
            return tok.apply_chat_template([{"role": "user", "content": q}])
        nr.build_prompt = build_prompt
        saved = sys.modules.get("neuron_report")
        sys.modules["neuron_report"] = nr

        class Tok:
            def __init__(self, template):
                self.chat_template = template
                self.last = None
            def apply_chat_template(self, msgs, **kw):
                return "<|user|>" + msgs[-1]["content"] + "<|assistant|>"
            def __call__(self, text, **kw):
                self.last = text
                return {"input_ids": list(text.encode())}
        try:
            for stage, template_tok in (("base", Tok("present")), ("Instruct", None)):
                session = object.__new__(part31.ModelSession)
                session.stage = stage
                session.tokenizer = Tok(None if stage == "base" else "present")
                session.template_tokenizer = template_tok
                session.prompt_ids("Who are you?")
                self.assertEqual(session.tokenizer.last, "<|user|>Who are you?<|assistant|>", stage)
        finally:
            if saved is not None:
                sys.modules["neuron_report"] = saved
            else:
                del sys.modules["neuron_report"]

    def test_cache_continuation_stops_a_row_whose_prefix_already_ended(self):
        # Fold r1 (Sol #7): the clamp prefix ignores EOS until K; the continuation must not
        # extend an answer that ended inside the push.
        class Session:
            tokenizer = type("T", (), {"eos_token_id": 99, "pad_token_id": 0})()
            model = type("M", (), {"config": type("C", (), {"eos_token_id": 99})()})()
            def make_generator(self, seed):
                return None
            def sample_token(self, logits, generator):
                return 7
            def forward_many(self, rows, past=None, full_length=None, step=0, window=None, last_only=False):
                return [None] * len(rows), past, [[0.0]] * len(rows), None
        tokens, states = part31._continue_cache_arm_batch(
            Session(), [1, 2], [[5, 99, 5], [5, 5, 5]], [[[0.0]] * 3, [[0.0]] * 3],
            None, [None, None], [0, 1], 6)
        self.assertEqual(tokens[0], [5, 99, 5])
        self.assertEqual(tokens[1], [5, 5, 5, 7, 7, 7])

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

    def test_whole_cell_quantiles_keep_nonreturners_censored(self):
        description = part31._describe([1, 10, None], horizon=128)
        self.assertEqual(description["median"], 10)
        self.assertEqual(description["conditional_on_return_median"], 5.5)
        majority_censored = part31._describe([1, None, None], horizon=128)
        self.assertEqual(majority_censored["median"], ">128")
        self.assertEqual(majority_censored["conditional_on_return_median"], 1)

    def test_cache_summary_reports_each_arm_median_per_stage_and_k(self):
        scores = []
        for index, (own, reencoded) in enumerate(((10, 90), (20, 1), (100, 19))):
            for arm, return_time in (("clamped_cache", own), ("reencoded_unclamped", reencoded)):
                scores.append({
                    "sample_id": f"Instruct/clamp/32/{arm}/p{index}/0",
                    "stage": "Instruct", "instrument": "clamp", "K": 32,
                    "prompt_id": f"p{index}", "sample_index": 0,
                    "cache_arm": arm, "state_return_time": return_time,
                    "text_return_time": None, "return_gap_text_minus_state": None,
                })
        summary = part31.make_summary(scores)["prediction_summary"]["31.5"]
        cell = summary["per_stage_and_K"]["Instruct"]["32"]
        self.assertEqual(cell["clamped_cache_state_return"]["median"], 20)
        self.assertEqual(cell["reencoded_unclamped_state_return"]["median"], 19)
        self.assertEqual(cell["clamped_cache_minus_reencoded_unclamped_median"], 1)
        self.assertEqual(
            summary["paired_sample_difference_conditional_on_both_arms_returning"]["median"],
            19,
        )

    def test_prediction_measurements_use_halfway_snapback_and_outside_events(self):
        # The band-return proxy is token 30, while the state reaches halfway at token 2.
        trace = [10] * 32 + [10, 4] + [10] * 27 + [0] * 8
        clamp = {
            "sample_id": "Instruct/clamp/32/clamped_cache/p00/0",
            "stage": "Instruct", "instrument": "clamp", "K": 32,
            "cache_arm": "clamped_cache", "prompt_id": "p00", "sample_index": 0,
            "projection_trace": trace, "sentence_start_tokens": [40],
            "sentence_end_tokens": [82],
        }
        labels = [{"sentence_index": 0, "label": "assistant_persona"}]
        scored_clamp = part31.score_sample(clamp, {"low": -1, "high": 1}, labels)
        self.assertEqual(scored_clamp["state_return_time"], 30)
        self.assertEqual(scored_clamp["state_halfway_to_baseline_center_time"], 2)
        self.assertEqual(scored_clamp["text_return_time"], 50)
        pred = part31.make_summary([scored_clamp])["prediction_summary"]["31.1"]
        joint = pred["Instruct"]["32"]["clamped_cache"][
            "fraction_state_halfway_within_20_and_text_return_at_or_after_50"
        ]
        self.assertEqual(joint["rate"], 1)

        # Review counterexample: every sample snaps back at token 30, so an old <=20
        # state-return proxy reported zero even though 31.3 asks for any-time text snapback.
        prefill = dict(scored_clamp)
        prefill.update({
            "sample_id": "SFT/prefill/32/none/p00/0", "stage": "SFT",
            "instrument": "prefill", "K": 32, "cache_arm": "none",
            "state_return_time": 30, "text_return_time": 30,
            "text_label_status": "labeled",
        })
        summary = part31.make_summary([prefill])["prediction_summary"]
        snapback = summary["31.3"]["SFT"]["32"]["text_snapback_at_any_time_rate_among_labeled"]
        self.assertEqual(snapback["rate"], 1)

        # Identical late recovery for control prefill must be outside at release + 20,
        # not scored as a within-20 return.
        control = dict(clamp)
        control.update({
            "sample_id": "SFT/control_prefill/32/none/p00/0", "stage": "SFT",
            "instrument": "control_prefill", "cache_arm": "none",
        })
        scored_control = part31.score_sample(control, {"low": -1, "high": 1}, labels)
        self.assertEqual(scored_control["state_return_time"], 30)
        self.assertTrue(scored_control["projection_outside_baseline_band_at_release_plus_20"])
        outside = part31.make_summary([scored_control])["prediction_summary"]["31.6"]["SFT"]["32"]
        self.assertEqual(outside["projection_outside_baseline_band_at_release_plus_20_rate"]["rate"], 1)

    def test_missing_judge_labels_are_reported_as_missing_not_zero(self):
        sample = {
            "sample_id": "Instruct/clamp/32/clamped_cache/p00/0",
            "stage": "Instruct", "instrument": "clamp", "K": 32,
            "cache_arm": "clamped_cache", "prompt_id": "p00", "sample_index": 0,
            "projection_trace": [10] * 32 + [0] * 64,
            "sentence_start_tokens": [], "sentence_end_tokens": [],
        }
        scored = part31.score_sample(sample, {"low": -1, "high": 1}, labels=None)
        self.assertEqual(scored["text_label_status"], "no labels")
        rate = part31.make_summary([scored])["prediction_summary"]["31.1"]["Instruct"]["32"][
            "clamped_cache"
        ]["fraction_state_halfway_within_20_and_text_return_at_or_after_50"]
        self.assertIsNone(rate["rate"])
        self.assertEqual(rate["status"], "no labels")
        prefill = dict(sample)
        prefill.update({"sample_id": "SFT/prefill/32/none/p00/0", "stage": "SFT", "instrument": "prefill", "cache_arm": "none"})
        scored_prefill = part31.score_sample(prefill, {"low": -1, "high": 1}, labels=None)
        snapback_rate = part31.make_summary([scored_prefill])["prediction_summary"]["31.3"]["SFT"]["32"][
            "text_snapback_at_any_time_rate_among_labeled"
        ]
        self.assertIsNone(snapback_rate["rate"])
        self.assertEqual(snapback_rate["status"], "no labels")

    def test_eos_within_push_is_excluded_and_counted_per_cell(self):
        common = {
            "stage": "Instruct", "instrument": "clamp", "K": 32,
            "cache_arm": "clamped_cache", "prompt_id": "p00",
            "text_return_time": None, "return_gap_text_minus_state": None,
        }
        scores = [
            dict(common, sample_id="eos", sample_index=0, eos_within_push=True, state_return_time=1),
            dict(common, sample_id="valid", sample_index=1, eos_within_push=False, state_return_time=20),
        ]
        cell = part31.make_summary(scores)["distributions"][0]
        self.assertEqual(cell["eos_within_push_count"], 1)
        self.assertEqual(cell["return_time_eligible_samples"], 1)
        self.assertEqual(cell["state_return"]["n"], 1)
        self.assertEqual(cell["state_return"]["median"], 20)

    def test_resume_fingerprint_mismatch_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / part31.SAMPLE_FILE).write_text("{}\n", encoding="utf-8")
            (out / part31.FINGERPRINT_FILE).write_text('{"part31_sha256":"old"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "--allow-resume-mismatch"):
                part31._verify_run_fingerprint(out, {"part31_sha256": "new"})

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
