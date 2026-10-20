"""Offline tests for Part 29 prompt construction, collection, and classification."""

from __future__ import annotations

import json
import io
import re
import tempfile
import unittest
import urllib.error
import urllib.parse
from email.message import Message
from pathlib import Path
from unittest import mock

import part29_classify as classify
import part29_collect as collect
import part29_prompts as prompt_builder


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.payload


class FakeHTTPServer:
    """In-process HTTP endpoint with the same request/response boundary as urllib."""
    response_queues: dict[str, list[tuple[int, dict[str, str], object]]] = {}
    requests: list[dict[str, object]] = []
    @classmethod
    def open(cls, request, timeout=600):
        del timeout
        path = urllib.parse.urlparse(request.full_url).path
        if path.startswith("/zen/go/"):
            path = path[len("/zen/go") :]
        body = json.loads(request.data.decode("utf-8"))
        cls.requests.append({"path": path, "headers": dict(request.header_items()), "body": body})
        queue = cls.response_queues.get(path, [])
        response = queue.pop(0) if queue else (500, {}, {"error": "no queued response"})
        status, headers, payload = response
        encoded = json.dumps(payload).encode("utf-8")
        header_map = Message()
        for key, value in headers.items():
            header_map.add_header(key, value)
        if status >= 400:
            raise urllib.error.HTTPError(request.full_url, status, "fake response", header_map, io.BytesIO(encoded))
        return FakeResponse(encoded)


class Part29Tests(unittest.TestCase):
    def setUp(self):
        FakeHTTPServer.response_queues = {}
        FakeHTTPServer.requests = []
        self._go_base = collect.GO_BASE_URL
        self._cliproxy_url = collect.CLIPROXY_CHAT_URL
        self._open = collect._open
        collect.GO_BASE_URL = "http://fake-opencode/zen/go"
        collect.CLIPROXY_CHAT_URL = "http://fake-cliproxy/v1/chat/completions"
        collect._open = FakeHTTPServer.open

    def tearDown(self):
        collect.GO_BASE_URL = self._go_base
        collect.CLIPROXY_CHAT_URL = self._cliproxy_url
        collect._open = self._open

    def queue(self, path, status=200, headers=None, payload=None):
        FakeHTTPServer.response_queues.setdefault(path, []).append(
            (status, headers or {}, {} if payload is None else payload)
        )

    def test_prompt_builder_filters_and_records_reproducible_source_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "dolly.jsonl"
            rows = [
                {"category": "open_qa", "instruction": f"Explain a made-up custom topic number {i} in two sentences.", "context": "unused"}
                for i in range(301)
            ]
            rows.extend([
                {"category": "general_qa", "instruction": "What is ChatGPT?"},
                {"category": "open_qa", "instruction": "Explain a model in one sentence."},
                {"category": "classification", "instruction": "A very long prompt " + ("x" * 500)},
            ])
            source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            output = root / "part29-prompts.json"
            payload = prompt_builder.build_prompt_set(source, output)
            self.assertEqual(payload["seed"], 0)
            self.assertEqual(payload["prompt_count"], 300)
            self.assertEqual(len(payload["source_sha256"]), 64)
            self.assertTrue(all(len(row["prompt"]) <= 400 for row in payload["prompts"]))
            self.assertTrue(all("context" not in row for row in payload["prompts"]))
            self.assertEqual(payload, json.loads(output.read_text(encoding="utf-8")))
            self.assertTrue(prompt_builder.mentions_masked_topic("Please ask an assistant."))
            self.assertFalse(prompt_builder.mentions_masked_topic("The metadata describes the dataset."))

    def test_chat_request_retries_429_and_keeps_reasoning_separate(self):
        path = "/v1/chat/completions"
        self.queue(path, 429, {"Retry-After": "0"}, {"error": "slow down"})
        self.queue(
            path,
            payload={
                "choices": [{
                    "message": {"content": "visible answer", "reasoning_content": "private reasoning"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2},
            },
        )
        with mock.patch.dict("os.environ", {"OPENCODE_GO_API_KEY": "test-key"}):
            result = collect.request_api_answer("deepseek-v4-pro", "Tell me a short story.")
        self.assertEqual(result["text"], "visible answer")
        self.assertEqual(result["reasoning"], "private reasoning")
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(len(FakeHTTPServer.requests), 2)
        request = FakeHTTPServer.requests[-1]
        self.assertEqual(request["path"], path)
        self.assertEqual(request["headers"].get("Authorization"), "Bearer test-key")
        self.assertEqual(request["headers"].get("X-opencode-session"), "part29")
        self.assertEqual(request["body"]["messages"], [{"role": "user", "content": "Tell me a short story."}])
        self.assertEqual(request["body"]["temperature"], 0.7)
        self.assertEqual(request["body"]["max_tokens"], collect.MAX_TOKENS)

    def test_messages_uses_text_blocks_even_when_thinking_is_first(self):
        path = "/v1/messages"
        self.queue(
            path,
            payload={
                "content": [
                    {"type": "thinking", "thinking": "I will reason privately."},
                    {"type": "text", "text": "The answer is 42."},
                    {"type": "text", "text": "And that is all."},
                ],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 5, "output_tokens": 8},
            },
        )
        with mock.patch.dict("os.environ", {"OPENCODE_GO_API_KEY": "test-key"}):
            result = collect.request_api_answer("claude-haiku-5-5", "What is the answer?")
        self.assertEqual(result["text"], "The answer is 42.\nAnd that is all.")
        self.assertEqual(result["reasoning"], "I will reason privately.")
        request = FakeHTTPServer.requests[0]
        self.assertEqual(request["path"], path)
        self.assertEqual(request["headers"].get("Anthropic-version"), "2023-06-01")
        self.assertEqual(request["headers"].get("X-api-key") or request["headers"].get("x-api-key"), "test-key")
        self.assertEqual(request["body"]["messages"], [{"role": "user", "content": "What is the answer?"}])

    def test_cliproxy_chat_shape_has_no_auth_and_think_answer_uses_last_marker(self):
        path = "/v1/chat/completions"
        self.queue(
            path,
            payload={
                "choices": [{"message": {"content": "A local reply."}, "finish_reason": "length"}],
                "usage": {"total_tokens": 10},
            },
        )
        result = collect.request_api_answer("gpt-6-luna", "Give a concise reply.")
        self.assertEqual(result["text"], "A local reply.")
        request = FakeHTTPServer.requests[0]
        self.assertIsNone(request["headers"].get("Authorization"))
        self.assertEqual(request["body"]["model"], "gpt-6-luna")
        self.assertEqual(
            collect.answer_after_think("<think>first</think> middle <think>second</think> final answer "),
            "final answer",
        )

    def test_http_errors_are_retried_at_most_three_times(self):
        path = "/v1/chat/completions"
        for _ in range(4):
            self.queue(path, 503, {"Retry-After": "0"}, {"error": "temporary"})
        with mock.patch.dict("os.environ", {"OPENCODE_GO_API_KEY": "test-key"}):
            with self.assertRaises(Exception):
                collect.request_api_answer("glm-5.3", "One prompt.")
        self.assertEqual(len(FakeHTTPServer.requests), 3)

    def test_resume_skips_existing_prompt_indices(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "answers-gpt-6-luna.jsonl"
            output.write_text(json.dumps({"index": 1, "model": "gpt-6-luna", "text": "saved"}) + "\n", encoding="utf-8")
            records = [{"index": i, "prompt": f"prompt {i}"} for i in range(3)]
            calls = []

            def fake_request(model_id, prompt):
                calls.append(prompt)
                return {"text": "new answer", "finish_reason": "stop", "usage": {}, "wall_time": 0.01}

            with mock.patch.object(collect, "request_api_answer", side_effect=fake_request):
                written, failures = collect.collect_api_model("gpt-6-luna", records, output, workers=2)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual((written, failures), (2, 0))
            self.assertCountEqual(calls, ["prompt 0", "prompt 2"])
            self.assertEqual({row["index"] for row in rows}, {0, 1, 2})

    def test_mask_keeps_meta_inside_metadata_and_replaces_names_as_words(self):
        # Sabotage row: replacing "Meta" as a substring would corrupt "metadata";
        # this assertion turns red if the mask loses its word boundaries.
        source = "metadata from Meta, OpenAI, and the hugging face team"
        masked = classify.mask_family_names(source)
        self.assertIn("metadata", masked)
        self.assertNotIn("Meta,", masked)
        self.assertEqual(masked.count("familymask"), 3)
        naive_sabotage = re.sub("meta", " familymask ", source, flags=re.IGNORECASE)
        self.assertNotEqual(naive_sabotage, masked)

    def test_classifier_recovers_synthetic_families_and_bootstraps_student(self):
        signatures = {
            "claude": "amber careful cadence",
            "gpt": "violet compact wording",
            "deepseek": "cobalt numbered reasoning",
            "qwen": "silver explanatory prose",
            "gemini": "coral descriptive rhythm",
            "glm": "teal formal phrasing",
            "kimi": "indigo playful language",
            "minimax": "ochre concise style",
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prompts_path = root / "prompts.json"
            prompts_path.write_text(
                json.dumps({"prompts": [{"index": i, "prompt": f"synthetic prompt {i}"} for i in range(300)]}),
                encoding="utf-8",
            )
            for label, model_id in classify.TEACHERS.items():
                rows = [
                    {"index": i, "model": model_id, "provider": "fake", "text": f"{signatures[label]} response number {i}", "finish_reason": "stop", "usage": {}, "wall_time": 0.0}
                    for i in range(300)
                ]
                (root / f"answers-{model_id}.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
                )
            student_rows = [
                {"index": i, "model": "synthetic-student", "text": f"{signatures['qwen']} student response {i}", "answer": f"{signatures['qwen']} student response {i}"}
                for i in range(240, 300)
            ]
            (root / "answers-synthetic-student.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in student_rows), encoding="utf-8"
            )
            result = classify.classify(root, prompts_path, bootstrap_replicates=200)
            self.assertEqual(result["teacher_heldout"]["accuracy"], 1.0)
            self.assertEqual(set(result["students"]), {"answers-synthetic-student.jsonl"})
            qwen = result["students"]["answers-synthetic-student.jsonl"]["views"]["answer"]["distribution"]["qwen"]
            self.assertEqual(qwen["fraction"], 1.0)
            self.assertEqual(qwen["ci95"], [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
