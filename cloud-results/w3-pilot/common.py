"""Shared setup for the W3 pilot: model, prompts, splits, batched greedy decoding."""
import os, random, warnings
warnings.filterwarnings("ignore")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3.5-0.8B"
REV = "2fc06364715b967f1860aea9cf38778875588b17"  # pinned revision (HF sha, 2026-11-03)
torch.set_num_threads(4)
torch.manual_seed(0)

FMT_MAIN = "What is {a} + {b}? Answer with the number only."
# held-out wordings: never used for building anything
FMT_OTHER = "Compute {a} plus {b}. Reply with just the result."
FMT_OTHER2 = "Please add {a} and {b}. Give only the sum."
# added after v5 (other2 was then used for diagnosis): fresh held-out wording, never built on
FMT_OTHER3 = "Tell me the total of {a} and {b}, digits only."

ALL_PAIRS = [(a, b) for a in range(10, 100) for b in range(10, 100)]
TEST = random.Random(3).sample(ALL_PAIRS, 200)
_test_set = set(TEST)
_rest = [p for p in ALL_PAIRS if p not in _test_set]
CALIB = random.Random(12345).sample(_rest, 600)  # disjoint from TEST by construction
assert not (set(CALIB) & _test_set)

_tok = None
_model = None


def load():
    global _tok, _model
    if _model is None:
        _tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
        _model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, dtype=torch.float32)
        _model.eval()
    return _tok, _model


def chat(text):
    tok, _ = load()
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


def layers():
    _, m = load()
    return m.model.layers


DIGIT_IDS = list(range(15, 25))  # token ids of '0'..'9'


@torch.no_grad()
def greedy_batch(texts, max_new=5, bs=50):
    """Greedy decode, no KV cache (so hooks see whole sequence). Texts must tokenize to equal length
    within a batch; we group by length."""
    tok, m = load()
    enc = [tok(t)["input_ids"] for t in texts]
    out = [None] * len(texts)
    by_len = {}
    for i, e in enumerate(enc):
        by_len.setdefault(len(e), []).append(i)
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    for L, idxs in by_len.items():
        for s in range(0, len(idxs), bs):
            chunk = idxs[s:s + bs]
            ids = torch.tensor([enc[i] for i in chunk])
            gen = [[] for _ in chunk]
            done = [False] * len(chunk)
            for _ in range(max_new):
                logits = m(input_ids=ids, use_cache=False).logits[:, -1]
                nxt = logits.argmax(-1)
                for k, t in enumerate(nxt.tolist()):
                    if not done[k]:
                        if t == eos:
                            done[k] = True
                        else:
                            gen[k].append(t)
                if all(done):
                    break
                ids = torch.cat([ids, nxt[:, None]], 1)
            for k, i in enumerate(chunk):
                out[i] = tok.decode(gen[k]).strip()
    return out


def accuracy(pairs, fmt, **kw):
    texts = [chat(fmt.format(a=a, b=b)) for a, b in pairs]
    outs = greedy_batch(texts, **kw)
    ok = [o == str(a + b) for o, (a, b) in zip(outs, pairs)]
    return sum(ok) / len(ok), outs
