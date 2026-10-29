"""Offline contract checks for the small Part 93 instrument."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

# The required invocation uses Python isolated mode, which omits the script directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import classify93
import gen93
from mask93 import mask


class MaskingTests(unittest.TestCase):
    def test_content_words_change_but_function_words_and_layout_remain(self):
        source = "The curious fox watches a small bird.\n- Please have a seat!"
        result = mask(source, "pos")
        self.assertIn("The", result)
        self.assertIn(" a ", result)
        self.assertIn("have", result)
        self.assertIn(".\n- ", result)
        self.assertTrue(result.endswith("!"))
        for content_word in ("curious", "fox", "watches", "small", "bird", "seat"):
            self.assertNotIn(content_word, result)
        self.assertEqual(mask(source, "none"), source)

    def test_numbered_list_marker_is_preserved(self):
        result = mask("1. Open the old door.\n# *carefully* under `trees`.", "pos")
        self.assertTrue(result.startswith("1. "))
        self.assertIn("\n# *", result)
        self.assertIn("* under `", result)


class GroupedFoldTests(unittest.TestCase):
    def test_prompt_ids_never_cross_a_fold_boundary(self):
        prompt_ids = [f"p{number:03d}" for number in range(1, 11) for _ in range(5)]
        for train, test in classify93.grouped_folds(prompt_ids, n_splits=5):
            self.assertFalse(set(prompt_ids[i] for i in train) & set(prompt_ids[i] for i in test))


class ClassifierTests(unittest.TestCase):
    def test_two_families_separate_on_punctuation_and_markdown_with_same_words(self):
        rows = []
        for number in range(1, 11):
            prompt_id = f"p{number:03d}"
            rows.extend([
                {"id": prompt_id, "prompt_id": prompt_id, "family": "bright", "answer": "**The garden looks lovely today! Really!**", "error": None},
                {"id": prompt_id, "prompt_id": prompt_id, "family": "plain", "answer": "- The garden looks lovely today.\n- Really.", "error": None},
            ])
        result = classify93.evaluate(rows)
        for mode in ("masked", "unmasked"):
            self.assertGreaterEqual(result[mode]["classifier_a"]["mean_accuracy"], 0.9)
            self.assertGreaterEqual(result[mode]["classifier_b"]["mean_accuracy"], 0.9)


class GeneratorTests(unittest.TestCase):
    def test_resume_skips_successful_rows_and_retries_error_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompts_path = root / "prompts.json"
            prompts = [
                {"id": f"p{index:03d}", "kind": "explanation", "prompt": f"Prompt {index}?"}
                for index in range(1, 4)
            ]
            prompts_path.write_text(json.dumps(prompts), encoding="utf-8")
            out_dir = root / "out"
            out_dir.mkdir()
            output_path = out_dir / "gpt.jsonl"
            old_rows = [
                {"id": "p001", "family": "gpt", "model": "gpt-6-luna[1m]", "prompt_id": "p001", "answer": "done", "usage": None, "error": None},
                {"id": "p002", "family": "gpt", "model": "gpt-6-luna[1m]", "prompt_id": "p002", "answer": None, "usage": None, "error": "timeout"},
            ]
            output_path.write_text("".join(json.dumps(row) + "\n" for row in old_rows), encoding="utf-8")
            args = SimpleNamespace(
                prompts=str(prompts_path), limit=None, out=str(out_dir), family="gpt",
                workers=1, go_base=None, gemini_model=None, model=None, device="auto",
            )
            called = []

            def fake_answer(family, prompt, unused_args, unused_qwen):
                called.append(prompt["id"])
                return {
                    "id": prompt["id"], "family": family, "model": gen93.FAMILY_MODELS[family],
                    "prompt_id": prompt["id"], "answer": "retried", "usage": None, "error": None,
                }

            with patch.object(gen93, "_answer_one", side_effect=fake_answer):
                self.assertEqual(gen93.run_family(args), 2)
            self.assertEqual(set(called), {"p002", "p003"})
            self.assertEqual(gen93.read_done_rows(output_path), {"p001", "p002", "p003"})

    def test_http_request_shapes_and_local_no_proxy_route(self):
        cases = [
            ("claude", {"api_key": "claude-key"}, "https://api.anthropic.com/v1/messages"),
            ("gpt", {}, "http://127.0.0.1:8317/v1/chat/completions"),
            ("deepseek", {"go_base": "https://go.example/v1", "api_key": "go-key"}, "https://go.example/v1/chat/completions"),
            ("gemini", {"go_base": "https://go.example", "gemini_model": "gemini-test", "api_key": "go-key"}, "https://go.example/v1/chat/completions"),
        ]

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *unused):
                return None

            def read(self):
                return b'{"ok":true}'

        for family, options, expected_url in cases:
            with self.subTest(family=family):
                spec = gen93.build_request(family, "Tell me something ordinary.", **options)
                self.assertEqual(spec["url"], expected_url)
                self.assertEqual(spec["body"]["max_tokens"], 300)
                self.assertEqual(spec["body"]["messages"], [{"role": "user", "content": "Tell me something ordinary."}])
                self.assertNotIn("temperature", spec["body"])
                if family == "claude":
                    self.assertEqual(spec["body"]["thinking"], {"type": "disabled"})
                if family == "gpt":
                    self.assertNotIn("Authorization", spec["headers"])

                default_urlopen = Mock(return_value=Response())
                local_open = Mock(return_value=Response())
                opener = Mock(open=local_open)
                with patch.object(gen93.urllib.request, "urlopen", default_urlopen), patch.object(
                    gen93.urllib.request, "build_opener", return_value=opener
                ) as build_opener:
                    gen93.post_json(spec)
                request = local_open.call_args.args[0] if family == "gpt" else default_urlopen.call_args.args[0]
                self.assertEqual(json.loads(request.data.decode("utf-8")), spec["body"])
                if family == "gpt":
                    self.assertEqual(build_opener.call_count, 1)
                    proxy_handler = build_opener.call_args.args[0]
                    self.assertEqual(proxy_handler.proxies, {})
                    default_urlopen.assert_not_called()
                else:
                    default_urlopen.assert_called_once()
                    build_opener.assert_not_called()

    def test_qwen_chat_template_disables_thinking_and_tokenizer_adds_no_special_tokens(self):
        class Tokenizer:
            def __init__(self):
                self.template_kwargs = None
                self.tokenizer_kwargs = None

            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                self.template_kwargs = kwargs
                return "rendered chat"

            def __call__(self, text, **kwargs):
                self.rendered = text
                self.tokenizer_kwargs = kwargs
                return {"input_ids": [11, 12]}

        tokenizer = Tokenizer()
        result = gen93.qwen_chat_inputs(tokenizer, "Describe a quiet garden.")
        self.assertEqual(result, {"input_ids": [11, 12]})
        self.assertEqual(tokenizer.messages, [{"role": "user", "content": "Describe a quiet garden."}])
        self.assertFalse(tokenizer.template_kwargs["enable_thinking"])
        self.assertFalse(tokenizer.template_kwargs["tokenize"])
        self.assertFalse(tokenizer.tokenizer_kwargs["add_special_tokens"])
        self.assertEqual(tokenizer.rendered, "rendered chat")


if __name__ == "__main__":
    unittest.main(verbosity=2)
