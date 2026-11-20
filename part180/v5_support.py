"""Fixtures for the Part 180 v5 tests: no real model, no download, no network.

A tiny randomly-initialised OLMo-2 config built in memory stands in for the 1B model, and a deterministically
tokenizing stand-in covers the tokenizer surface `texts180`/`common92` touch. Nothing here is imported by the
production modules.
"""
from __future__ import annotations

import torch
from transformers import Olmo2Config, Olmo2ForCausalLM

PAD_ID = 0
EOS_ID = 63
VOCAB = 64


class TinyTokenizer:
    """Deterministic word-level tokenizer: same text -> same ids in every process."""

    def __init__(self):
        self.pad_token = "<pad>"
        self.pad_token_id = PAD_ID
        self.eos_token = "<|endoftext|>"
        self.eos_token_id = EOS_ID
        self.unk_token = "<unk>"
        self.unk_token_id = 1
        self._specials = {"<pad>": PAD_ID, "<unk>": 1, "<|endoftext|>": EOS_ID}

    def get_vocab(self):
        return dict(self._specials)

    def convert_tokens_to_ids(self, token):
        return self._specials.get(token)

    def _id(self, token: str) -> int:
        if token in self._specials:
            return self._specials[token]
        digest = sum((index + 1) * ord(char) for index, char in enumerate(token)) % (VOCAB - 3)
        return 2 + digest

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, enable_thinking=False):
        text = "".join(f"<|{m['role']}|>{m['content']}\n" for m in messages)
        return text + ("<|assistant|>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [self._id(token) for token in str(text).replace("\n", " \n ").split() if token]}

    def decode(self, ids, skip_special_tokens=True):
        reverse = {value: key for key, value in self._specials.items()}
        words = []
        for value in ids:
            value = int(value)
            if value in reverse:
                if not skip_special_tokens:
                    words.append(reverse[value])
                continue
            words.append(f"w{value}")
        return " ".join(words)


def tiny_model(seed: int = 180180, layers: int = 2, hidden: int = 24, heads: int = 4):
    torch.manual_seed(int(seed))
    config = Olmo2Config(
        vocab_size=VOCAB,
        hidden_size=hidden,
        intermediate_size=2 * hidden,
        num_hidden_layers=int(layers),
        num_attention_heads=int(heads),
        num_key_value_heads=int(heads),
        max_position_embeddings=256,
        pad_token_id=PAD_ID,
        bos_token_id=1,
        eos_token_id=EOS_ID,
    )
    config._attn_implementation = "eager"
    model = Olmo2ForCausalLM(config).cpu().eval()
    model.requires_grad_(False)
    return model


def tiny_neurons(count: int = 3, layers: int = 2):
    personas = ["Boulder", "Whiskey", "80s cover singer-songwriter", "Fox", "Queen of Hearts", "Volcano"]
    return [{"id": f"p92-{index:02d}", "layer": index % int(layers), "neuron": 7 + index,
             "persona": personas[index % len(personas)],
             "desc": f"A {personas[index % len(personas)].lower()} speaking in the first person."}
            for index in range(int(count))]


def tiny_pushes(neurons, hidden: int = 24, seed: int = 180):
    generator = torch.Generator().manual_seed(int(seed))
    return {str(neuron["id"]): torch.randn(hidden, generator=generator, dtype=torch.float32)
            for neuron in neurons}


def label(claim="no", mention=False, theme=False, description=False, negated=False, quoted=False):
    return {"claim": claim, "mention": mention, "theme": theme, "description": description,
            "negated": negated, "quoted": quoted}


LABEL_S = label(claim="yes", mention=True)
LABEL_CONTENT = label(claim="no", mention=True)
LABEL_ABSENT = label(claim="no")

# A fixed token length per arm, so K is predictable without a model. A0 is longer than A1 on purpose: then
# min(k_self_a0, k_self_a1) is A1's K, so a reader that used the A0 pair alone would be visibly wrong.
TOKEN_LENGTHS = {"S": 30, "A0": 40, "A1": 9, "C": 30, "B": 20}


def fake_text_rows(neurons):
    """One row per planned task with a fixed token length per arm (see TOKEN_LENGTHS)."""
    import texts180v5

    rows = []
    for task in texts180v5.plan_tasks(neurons):
        length = TOKEN_LENGTHS[task["key"]]
        rows.append({"neuron": task["neuron"], "key": task["key"], "config": task["config"],
                     "kind": task["kind"], "text": f"text-{task['kind']}",
                     "token_ids": list(range(1, length + 1)),
                     "generation_prompt": task["prompt"], "generation_prompt_ids": [1, 2, 3],
                     "frame": "chat", "generated_with_push": bool(task["pushed"]),
                     "score_push": True, "push_layer": task["push_layer"]})
    return rows
