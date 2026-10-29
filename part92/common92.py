#!/usr/bin/env python3
"""Small shared model and steering helpers for Part 92."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from part85 import checkpoints as checkpoint_registry  # use the registered ladder; do not duplicate commits

PLAIN = "Question: {q}\nAnswer:"


def get_layers(model):
    """Return the decoder blocks for the local OLMo 2 causal LM."""
    return model.model.layers


def load_model(snapshot_dir: str | Path, device: str = "cpu"):
    """Load a local snapshot in bf16, returning its tokenizer and evaluation model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    snapshot = str(Path(snapshot_dir).expanduser())
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, torch_dtype=torch.bfloat16, local_files_only=True
    ).to(device)
    model.requires_grad_(False)
    model.eval()
    return tokenizer, model


def prompt_ids(tokenizer, question: str, frame: str) -> list[int]:
    """Render Part 91's chat/plain frame and return ordinary token ids."""
    if frame == "chat":
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}], add_generation_prompt=True,
            tokenize=False, enable_thinking=False,
        )
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    elif frame == "plain":
        ids = tokenizer(PLAIN.format(q=question), add_special_tokens=True)["input_ids"]
    else:
        raise ValueError("frame must be 'chat' or 'plain'")
    return [int(token) for token in ids]


def residual_medians(model, tokenizer, frame: str, device: str | None = None) -> dict[int, float]:
    """Median residual norm after every layer for 'Who are you?', excluding token zero."""
    import torch

    device = device or str(next(model.parameters()).device)
    ids = torch.tensor([prompt_ids(tokenizer, "Who are you?", frame)], dtype=torch.long, device=device)
    if ids.shape[1] < 2:
        raise ValueError("calibration prompt must contain at least two tokens")
    layers = get_layers(model)
    captured = {}

    def make_hook(layer_no):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            captured[layer_no] = hidden.detach()
        return hook

    # Hook every block's output directly: hidden_states[-1] is the post-final-norm state, not layer 15's output.
    handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(layers)]
    try:
        with torch.inference_mode():
            model(input_ids=ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if len(captured) != len(layers):
        raise RuntimeError("not every decoder layer reported its output")
    medians = {}
    for layer_no in range(len(layers)):
        norms = captured[layer_no][0, 1:].float().norm(dim=-1)
        medians[layer_no] = float(norms.median().item())
    return medians


def add_row_hook(layer, row_vectors):
    """Add [batch, hidden] row vectors after this layer on prefill and decode calls."""
    vectors = row_vectors
    if vectors.ndim != 2:
        raise ValueError("row_vectors must have shape [batch, hidden]")

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if hidden.ndim != 3 or hidden.shape[0] != vectors.shape[0] or hidden.shape[-1] != vectors.shape[-1]:
            raise ValueError(f"unexpected layer output shape {tuple(hidden.shape)} for row vectors {tuple(vectors.shape)}")
        pushed = hidden + vectors.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(1)
        if isinstance(output, tuple):
            return (pushed,) + tuple(output[1:])
        if isinstance(output, list):
            return [pushed, *output[1:]]
        return pushed

    return layer.register_forward_hook(hook)


def neuron_push(weight, neuron: int, target_norm: float):
    """Scale one W_down column to the requested residual-stream norm."""
    import torch

    neuron = int(neuron)
    if weight.ndim != 2 or not 0 <= neuron < weight.shape[1]:
        raise ValueError("neuron index is outside W_down input width")
    column = weight[:, neuron].detach().float()
    norm = torch.linalg.vector_norm(column)
    if float(norm.item()) == 0:
        raise ValueError("cannot push a zero-norm W_down column")
    return (column * (float(target_norm) / float(norm.item()))).to(dtype=weight.dtype)


def random_directions(weight, count: int, seed: int, target_norm: float):
    """Map seeded Gaussian W_down-input vectors into residual space and match their norms."""
    import torch

    if weight.ndim != 2 or count < 0:
        raise ValueError("weight must be a matrix and count must be nonnegative")
    if target_norm < 0:
        raise ValueError("target_norm must be nonnegative")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    if count == 0:
        return torch.empty((0, int(weight.shape[0])), dtype=weight.dtype, device=weight.device)
    raw = torch.randn((int(count), int(weight.shape[1])), generator=generator, dtype=torch.float32)
    raw = raw.to(device=weight.device)
    residual = raw @ weight.detach().float().T
    norms = torch.linalg.vector_norm(residual, dim=1, keepdim=True)
    if bool(torch.any(norms <= 0)):
        raise ValueError("a random W_down direction has zero residual norm")
    if float(target_norm) == 0:
        residual.zero_()
    else:
        residual = residual * (float(target_norm) / norms)
    return residual.to(dtype=weight.dtype)


def random_direction(weight, seed: int, target_norm: float):
    """Single seeded random residual direction matched to a neuron push norm."""
    return random_directions(weight, 1, seed, target_norm)[0]


def generate_batch(model, tokenizer, questions, frame: str, layer, row_vectors,
                   max_new_tokens: int, device: str) -> list[str]:
    """Greedy decode a batch with left padding and optional per-row layer pushes."""
    import torch

    if isinstance(questions, str):
        questions = [questions]
    questions = list(questions)
    if not questions:
        return []
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = int(tokenizer.pad_token_id)
    sequences = [prompt_ids(tokenizer, question, frame) for question in questions]
    width = max(len(seq) for seq in sequences)
    input_ids = torch.tensor([[pad_id] * (width - len(seq)) + seq for seq in sequences],
                             dtype=torch.long, device=device)
    attention_mask = torch.tensor([[0] * (width - len(seq)) + [1] * len(seq) for seq in sequences],
                                  dtype=torch.long, device=device)
    if row_vectors is not None:
        if row_vectors.shape[0] != len(questions):
            raise ValueError("one row vector is required per question")
        handle = add_row_hook(layer, row_vectors)
    else:
        handle = None
    try:
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids, attention_mask=attention_mask, do_sample=False,
                max_new_tokens=int(max_new_tokens), pad_token_id=pad_id,
            )
    finally:
        if handle is not None:
            handle.remove()
    answers = []
    for row in generated[:, width:]:
        text = tokenizer.decode(row, skip_special_tokens=True)
        answer = text.split("\nQuestion:")[0].strip() if frame == "plain" else text.strip()
        answers.append(answer)
    return answers
