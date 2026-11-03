"""Three hand-written transformers from csinva/weight-space-programming-autoresearch,
re-implemented here so the probes run only code we have read.

Source: https://github.com/csinva/weight-space-programming-autoresearch
        commit f82c8a3a0d8bacfa2e03bf3b38e37b9184273b02
        runs/<task>-may15-run1/interpretable_transformer.py (final snapshot of each run)

The architectures below are a single generic module that reproduces each run's
forward pass exactly (same softmax scaling 1/sqrt(d_qk), same LN placement, same
bias settings).  The weight-writing functions are faithful transcriptions of the
three `write_weights` functions (comments condensed).  `check_accuracy()` confirms
each rebuilt model scores 100% under greedy decoding.

The generic module exposes two intervention points used by probe.py:
  * MLP hidden units (post-ReLU) of every layer        -> self.unit_hooks[layer]
  * per-head attention outputs (pre-W_o slices)         -> self.head_hooks[layer]
Each hook is a callable(tensor) -> tensor or None.
"""
from __future__ import annotations

import itertools
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- tasks
class Task:
    def __init__(self, name, vocab, prompt_len, answer_len, gen, enumerate_all=None):
        self.name, self.vocab = name, vocab
        self.prompt_len, self.answer_len = prompt_len, answer_len
        self.stoi = {c: i for i, c in enumerate(vocab)}
        self.gen, self.enumerate_all = gen, enumerate_all

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def seq_len(self):
        return self.prompt_len + self.answer_len

    def encode(self, s):
        return [self.stoi[c] for c in s]


def _parity_gen(n, seed):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        L = rng.randint(1, 10)
        bits = [rng.randint(0, 1) for _ in range(L)]
        out.append(("".join(map(str, bits)).rjust(10, "_") + "=", str(sum(bits) % 2)))
    return out


def _parity_all():
    out = []
    for L in range(1, 11):
        for bits in itertools.product([0, 1], repeat=L):
            out.append(("".join(map(str, bits)).rjust(10, "_") + "=", str(sum(bits) % 2)))
    return out


def _sort_gen(n, seed):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        d = [rng.randint(0, 9) for _ in range(5)]
        out.append(("".join(map(str, d)) + "=", "".join(map(str, sorted(d)))))
    return out


def _sort_all():
    return [("".join(map(str, d)) + "=", "".join(map(str, sorted(d))))
            for d in itertools.product(range(10), repeat=5)]


def _count_gen(n, seed):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        d = [rng.randint(0, 9) for _ in range(10)]
        q = rng.randint(0, 9)
        out.append(("".join(map(str, d)) + f"{q}=", f"{sum(x == q for x in d):02d}"))
    return out


TASKS = {
    "parity": Task("parity-upto10-bits", list("01=_"), 11, 1, _parity_gen, _parity_all),
    "sort": Task("sort-five-digits", list("0123456789="), 6, 5, _sort_gen, _sort_all),
    "count": Task("digit-counting-10", list("0123456789="), 12, 2, _count_gen, None),
}


# ---------------------------------------------------------------- model
class Attn(nn.Module):
    def __init__(self, d_model, n_heads, d_qk, d_v):
        super().__init__()
        self.n_heads, self.d_qk, self.d_v = n_heads, d_qk, d_v
        self.W_q = nn.Linear(d_model, n_heads * d_qk, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_qk, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.W_o = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.hook = None
        self.last_pattern = None

    def forward(self, x):
        B, T, _ = x.shape
        H, dqk, dv = self.n_heads, self.d_qk, self.d_v
        q = self.W_q(x).view(B, T, H, dqk).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, dqk).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, dv).transpose(1, 2)
        s = (q @ k.transpose(-2, -1)) / math.sqrt(dqk)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        a = s.masked_fill(mask, float("-inf")).softmax(-1)
        self.last_pattern = a.detach()
        z = (a @ v).transpose(1, 2)  # B,T,H,dv  per-head outputs
        if self.hook is not None:
            r = self.hook(z)
            z = z if r is None else r
        return self.W_o(z.contiguous().view(B, T, H * dv))


class MLP(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.hook = None

    def forward(self, x):
        h = F.relu(self.fc1(x))
        if self.hook is not None:
            r = self.hook(h)
            h = h if r is None else r
        return self.fc2(h)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, ln, attn, d_qk, d_v):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model) if ln else nn.Identity()
        self.ln2 = nn.LayerNorm(d_model) if ln else nn.Identity()
        self.attn = Attn(d_model, n_heads, d_qk, d_v) if attn else None
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x):
        if self.attn is not None:
            x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class Model(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model, n_heads, d_ffs, ln, final_ln,
                 attn_layers, d_qk, d_v):
        super().__init__()
        self.d_model, self.n_heads, self.max_seq_len = d_model, n_heads, max_seq_len
        self.d_qk, self.d_v, self.n_layers = d_qk, d_v, len(d_ffs)
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, f, ln, i in attn_layers, d_qk, d_v) for i, f in enumerate(d_ffs)])
        self.final_ln = nn.LayerNorm(d_model) if final_ln else nn.Identity()
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids):
        T = ids.shape[1]
        h = self.token_emb(ids) + self.pos_emb(torch.arange(T))[None]
        for b in self.blocks:
            h = b(h)
        return self.head(self.final_ln(h))


# ---------------------------------------------------------------- weights
@torch.no_grad()
def build_parity(task):
    # ParityFloorSubD3F9: d_model=3, 1 head (d_head=3), d_ff=9, no LN.
    m = Model(task.vocab_size, 16, 3, 1, [9], False, False, {0}, 3, 3)
    for p in m.parameters():
        p.zero_()
    s = task.stoi
    m.token_emb.weight[s["1"], 0] = 1.0          # ch0 = is_one_bit
    m.token_emb.weight[s["="], 1] = 1.0          # ch1 = is_equals
    a = m.blocks[0].attn                          # Q=K=0 -> uniform attention
    a.W_v.weight[2, 0] = 11.0
    a.W_o.weight[2, 2] = 1.0                      # ch2 = N1 at '=' position
    mlp = m.blocks[0].mlp
    for j in range(1, 10):                        # h_j = ReLU(N1 - j)
        mlp.fc1.weight[j - 1, 2] = 1.0
        mlp.fc1.bias[j - 1] = -float(j)
    for k in range(1, 6):                         # ch2 += -2 floor(N1/2)
        mlp.fc2.weight[2, (2 * k - 1) - 1] = -2.0
        if 2 * k <= 9:
            mlp.fc2.weight[2, 2 * k - 1] = 2.0
    m.head.weight[s["0"], 1] = 0.5
    m.head.weight[s["0"], 2] = -10.0
    m.head.weight[s["1"], 1] = -0.5
    m.head.weight[s["1"], 2] = 10.0
    return m


@torch.no_grad()
def build_sort(task):
    # HistogramCircuit_v5_clean: d_model=24, 1 head (d_head=24), d_ff=20, with LayerNorm.
    D, L = 24, 16
    m = Model(task.vocab_size, L, D, 1, [20], True, True, {0}, 24, 24)
    for p in m.parameters():
        p.zero_()
    MARK, KD, H0, A0 = 10, 11, 12, 22
    ANCH, A_attn = 15.0, 10.0
    for d in range(10):
        m.token_emb.weight[d, d] = 1.0
    anchor = torch.zeros(D)
    anchor[22], anchor[23] = ANCH, -ANCH
    pos = m.pos_emb.weight
    for p in range(L):
        pos[p] += anchor
    for p in range(5):
        pos[p, MARK] = 1.0
    for k in range(5):
        pos[5 + k, MARK] = -1.0
        pos[5 + k, KD] = float(k)
    anchor_sq = 2 * ANCH ** 2
    sig1 = 1.0 / math.sqrt((anchor_sq + 3.0) / D)
    b = m.blocks[0]
    b.ln1.weight.fill_(1.0)
    b.attn.W_q.weight[0, MARK] = -A_attn
    b.attn.W_k.weight[0, MARK] = A_attn
    for d in range(10):
        b.attn.W_v.weight[d, d] = 1.0
    GAIN_O = 5.0 / sig1
    OFFSET_O = GAIN_O * sig1 * (2.0 / D)
    for d in range(10):
        b.attn.W_o.weight[H0 + d, d] = GAIN_O
    sig2 = 1.0 / math.sqrt((anchor_sq + 13.0 + 3.0) / D)
    b.ln2.weight.fill_(1.0)
    for d in range(10):
        b.ln2.bias[H0 + d] = sig2 * OFFSET_O
    w1, b1 = b.mlp.fc1.weight, b.mlp.fc1.bias
    for d in range(10):                           # unit d: c_d - k ; unit 10+d: c_d - k - 1
        for dp in range(d + 1):
            w1[d, H0 + dp] = 1.0
            w1[10 + d, H0 + dp] = 1.0
        w1[d, KD] = -1.0
        w1[10 + d, KD] = -1.0
        bal = -d / 2.0
        for r in (d, 10 + d):
            w1[r, A0] = bal
            w1[r, A0 + 1] = bal
        b1[10 + d] = -sig2
    G = 10.0 / sig2
    w2 = b.mlp.fc2.weight
    for d in range(10):
        w2[d, d] = G
        w2[d, 10 + d] = -G
        if d >= 1:
            w2[d, d - 1] = -G
            w2[d, 10 + d - 1] = G
    m.final_ln.weight.fill_(1.0)
    for d in range(10):
        m.head.weight[d, d] = 100.0
    return m


@torch.no_grad()
def build_count(task):
    # CountCircuitV13: d_model=32, L0 = 2 heads (d_qk=1, d_v=10) + MLP(10), L1 = MLP(12) only, no LN.
    m = Model(task.vocab_size, task.seq_len, 32, 2, [10, 12], False, False, {0}, 1, 10)
    for p in m.parameters():
        p.zero_()
    S, ALPHA = 10.0, 20.0
    POS_DATA = COUNT = 10
    POS_OUT, G0, H0 = 11, 12, 22
    dqk, dv = 1, 10
    for d in range(10):
        m.token_emb.weight[d, d] = 1.0
    for p in range(task.seq_len):
        if p < 10:
            m.pos_emb.weight[p, POS_DATA] = 1.0
        elif p in (11, 12):
            m.pos_emb.weight[p, POS_OUT] = 1.0
    a = m.blocks[0].attn
    a.W_q.weight[0, POS_OUT] = S                  # head 0: attend to query position (10)
    a.W_k.weight[0, POS_DATA] = -ALPHA
    a.W_k.weight[0, POS_OUT] = -ALPHA
    for d in range(10):
        a.W_v.weight[d, d] = 1.0
        a.W_o.weight[G0 + d, d] = 1.0
    a.W_q.weight[dqk, POS_OUT] = S                # head 1: uniform over data positions
    a.W_k.weight[dqk, POS_DATA] = S
    for d in range(10):
        a.W_v.weight[dv + d, d] = 1.0
        a.W_o.weight[H0 + d, dv + d] = 10.0
    m0 = m.blocks[0].mlp                          # count = h[query]
    for d in range(10):
        m0.fc1.weight[d, H0 + d] = 1.0
        m0.fc1.weight[d, G0 + d] = ALPHA
        m0.fc1.bias[d] = -ALPHA
        m0.fc2.weight[COUNT, d] = 1.0
    m1 = m.blocks[1].mlp
    AP = 20.0
    m1.fc1.weight[0, COUNT] = 1.0                 # unit 0: pos 11 AND count == 10
    m1.fc1.weight[0, POS_OUT] = AP
    for d in range(10):
        m1.fc1.weight[0, d] = -AP
    m1.fc1.bias[0] = -AP - 9.5
    m1.fc2.weight[1, 0] = 4.0
    for i, k in enumerate(range(11)):             # units 1..11: pos 12 AND ReLU(count - k)
        n = 1 + i
        m1.fc1.weight[n, COUNT] = 1.0
        m1.fc1.weight[n, POS_OUT] = AP
        for d in range(10):
            m1.fc1.weight[n, d] = AP
        m1.fc1.bias[n] = -k - 2 * AP
    bump = {-1: 2.0, 0: -4.0, 1: 2.0}
    for c in range(1, 10):
        for off, w in bump.items():
            k = c + off
            if 0 <= k <= 10:
                m1.fc2.weight[c, 1 + k] += w
    for off, w in bump.items():
        k = 10 + off
        if 0 <= k <= 10:
            m1.fc2.weight[0, 1 + k] += w
    for d in range(10):
        m.head.weight[d, d] = 1.0
    return m


BUILDERS = {"parity": build_parity, "sort": build_sort, "count": build_count}


@torch.no_grad()
def greedy(model, task, prompts):
    ids = torch.tensor([task.encode(p) for p in prompts])
    for _ in range(task.answer_len):
        nxt = model(ids)[:, -1].argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], 1)
    return ["".join(task.vocab[i] for i in row[task.prompt_len:]) for row in ids.tolist()]


def check_accuracy(name, n=2000, seed=0):
    task = TASKS[name]
    m = BUILDERS[name](task).eval()
    ex = task.gen(n, seed)
    pred = greedy(m, task, [p for p, _ in ex])
    return sum(p == a for p, (_, a) in zip(pred, ex)) / n


if __name__ == "__main__":
    for k in BUILDERS:
        print(k, check_accuracy(k))
