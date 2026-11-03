"""W1 pilot shared mechanics: load Qwen3.5-4B, build prompts, batched greedy generation, the inserted
conditional block, the Part 72 steering baseline, and the scoring rules."""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

MODEL_ID = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
MAX_NEW = 48
BATCH = 8  # never 1; odd counts get a companion row (repeat of the chunk's first question)

# Part 72 (results/part72/part72.json): L27 N3325 mean activation over all positions of Part 69's maker prompts.
P72_UNIT = (27, 3325)
P72_MEAN = -0.7429866119169853
P72_K = 8  # Part 72's strongest published push


def load():
    import neuron_report as nr
    torch.set_num_threads(4)
    tok, model = nr.load_model(MODEL_ID, REVISION, "cpu")
    tok.padding_side = "left"
    model.requires_grad_(False)
    _, layers, final_norm = nr.find_text_layers(model)
    return tok, model, layers, final_norm


def build_prompt(tok, q):
    import neuron_report as nr  # same call as part86/common86.py build_prompt (thinking off, no system)
    return nr.build_prompt(tok, q, False)


@torch.inference_mode()
def generate(tok, model, questions, batch=BATCH, max_new=MAX_NEW, log=None):
    """Greedy answers, in fixed batches of `batch` (same grouping for every condition). Returns token-id lists."""
    assert batch >= 2
    out = []
    qs = list(questions)
    for i in range(0, len(qs), batch):
        chunk = qs[i:i + batch]
        n = len(chunk)
        if n < batch:  # companion row(s): repeat the first question of the chunk, dropped afterwards
            chunk = chunk + [chunk[0]] * (batch - n)
        enc = tok([build_prompt(tok, q) for q in chunk], return_tensors="pt", padding=True, add_special_tokens=False)
        g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, top_k=None,
                           pad_token_id=tok.pad_token_id)
        new = g[:, enc["input_ids"].shape[1]:]
        for r in range(n):
            ids = new[r].tolist()
            eos = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
            cut = next((j + 1 for j, t in enumerate(ids) if t in eos), len(ids))
            out.append(ids[:cut])
        if log:
            log(f"  gen {i + n}/{len(qs)}")
    return out


class Block:
    """The inserted conditional units, added to layer L's MLP output: out += sum_k relu(d_k . x - theta_k) * v_k,
    where x is the MLP input (post-attention RMSNorm output). Detector rows d_k, thresholds theta_k, writer
    columns v_k. With every unit below threshold the added term is exactly 0.0, so the layer output is unchanged."""

    def __init__(self, layer: int, D: torch.Tensor, theta: torch.Tensor, V: torch.Tensor):
        self.layer, self.D, self.theta, self.V = layer, D.float(), theta.float(), V.float()
        self.fired = 0

    def hook(self, _mod, args, output):
        x = args[0].float()
        a = torch.relu(x @ self.D.T - self.theta)  # [b, t, k]
        if not bool((a > 0).any()):
            return None
        self.fired += int((a > 0).any(-1).sum())
        return output + (a @ self.V).to(output.dtype)

    def install(self, layers):
        return [layers[self.layer].mlp.register_forward_hook(self.hook)]


def install_p72(layers, k=P72_K):
    """Part 72 baseline: L27 N3325's down_proj input clamped to k x its mean at every position."""
    li, u = P72_UNIT
    val = k * P72_MEAN

    def pre(_m, args):
        x = args[0].clone()
        x[..., u] = val
        return (x,) + tuple(args[1:])
    return [layers[li].mlp.down_proj.register_forward_pre_hook(pre)]


# ---------- scoring rules (written before any test answer was read) ----------
WORDS = re.compile(r"[\w]+(?:['’][\w]+)*", re.UNICODE)
GOOGLE = re.compile(r"\bGoogle\b|谷歌|\bDeepMind\b", re.IGNORECASE)
ALIBABA = re.compile(r"\bAlibaba\b|\bTongyi\b|通义|阿里", re.IGNORECASE)


def broken(text: str, ids: list[int]) -> str | None:
    """Broken rule: Part 86 triage (fewer than four words, or a word trigram occurring at least three times),
    without its identity exemption, plus one token id repeated 6+ times in a row. Returns the reason or None."""
    words = [w.casefold() for w in WORDS.findall(text)]
    if len(words) < 4:
        return "few_words"
    tri = Counter(tuple(words[i:i + 3]) for i in range(len(words) - 2))
    if tri and max(tri.values()) >= 3:
        return "trigram_x3"
    run = 1
    for a, b in zip(ids, ids[1:]):
        run = run + 1 if a == b else 1
        if run >= 6:
            return "token_run"
    return None


def names_google(text: str) -> bool:
    return bool(GOOGLE.search(text))
