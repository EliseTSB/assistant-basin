"""A tiny Qwen3.5 model and tokenizer for Part 202's runner tests: no download, no GPU, the target architecture.

The model is Part 201's test fixture (part201/test_part201.py: tiny_qwen, TinyTokenizer), moved here so Part 202's
tests can use it without importing another part's test module. The tokenizer renders a prompt the way
neuron_report.build_prompt does with no chat template (``User: <q>\\nAssistant:``), which is what common86.build_prompt
falls back to for a tokenizer without one.
"""
from __future__ import annotations


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    padding_side = "right"
    chat_template = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return messages[-1]["content"] + " assistant"

    def __call__(self, texts, return_tensors=None, padding=False, add_special_tokens=False):
        import torch
        if isinstance(texts, str):
            texts = [texts]
        rows = [[3 + (sum(map(ord, c)) % 20) for c in text.split()] for text in texts]
        width = max(map(len, rows)) if padding else len(rows[0])
        ids, masks = [], []
        for row in rows:
            pad = [0] * (width - len(row))
            ids.append(pad + row)
            masks.append([0] * len(pad) + [1] * len(row))
        return {"input_ids": torch.tensor(ids, dtype=torch.long), "attention_mask": torch.tensor(masks, dtype=torch.long)}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(x)) for x in ids if not skip_special_tokens or int(x) != 2)


class ScriptedAnswers:
    """A logits processor that makes every row emit the model's `_test_answers` in turn (a fixed answer, so a test run
    is quick and its labels are known). Without it the tiny model's answers never end and every call runs to the cap."""

    def __init__(self, answers, vocab=32):
        # A string answer becomes its own token ids (the tokenizer's own rule); a list of ids is taken as is.
        self.answers = ([3 + (sum(map(ord, c)) % 20) for c in answers.split()] if isinstance(answers, str)
                        else [int(a) for a in answers])
        self.answers = [a % vocab for a in self.answers]
        self.steps = 0
        self.end_token = 2

    def __call__(self, input_ids, scores):
        import torch
        # A processor is called once per step, so the step is this processor's own count: the prefill is step 0 and
        # emits the first answer token, and the row ends with the end token after the last answer token.
        step = self.steps
        self.steps += 1
        target = self.answers[step] if step < len(self.answers) else self.end_token
        forced = torch.full_like(scores, -1e4)
        forced[..., target] = 0.0
        return forced


def tiny_qwen(dtype=None):
    import torch
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    cfg = Qwen3_5TextConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                            num_attention_heads=2, num_key_value_heads=2, head_dim=8, linear_num_key_heads=2,
                            linear_num_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=8,
                            layer_types=["full_attention"], eos_token_id=2, pad_token_id=0)
    model = Qwen3_5ForCausalLM(cfg).eval()
    if dtype is not None:
        model.to(dtype=dtype)
    return model
