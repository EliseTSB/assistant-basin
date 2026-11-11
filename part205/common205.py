#!/usr/bin/env python3
"""common205.py: units, frame, readout, hooks and the estimator shared by run205.py and test_part205.py.

Units (Design, Units, lines 79-85). A part is one attention head or one MLP neuron. Following the `head_site` rule
in third_party/honesty-heads/code/audit_eval.py:89-97:
  - OLMo (family O): a head is a 128-wide slice of layer l's `o_proj` input (16 heads per layer, 256 heads); a neuron
    is one `down_proj` input (8,192 per layer, 131,072 neurons); pool 131,328.
  - Qwen3.5 (family Q): a full-attention layer's head is a 256-wide slice of `self_attn.o_proj`'s input (16 per layer
    on 8 of 32 layers, 128); a Gated DeltaNet layer's head is a 128-wide slice of `linear_attn.out_proj`'s input
    (32 per layer on 24 layers, 768); a neuron is one `down_proj` input (9,216 per layer, 294,912); pool 295,808.

A part is (kind, layer, index) where kind is "head" or "neuron" (and for Q the head sub-kind is recorded as
"head:full" or "head:deltanet" so the composition can be reported; both count as heads).

The estimator e (Design, Ranking, lines 87-95): for one prompt, e = sum over the part's dimensions of activation x
gradient at the last position, of the logit difference (consistent option minus the other). Zeroing a part changes the
logit difference by about -e.
"""
import json
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------- units

@dataclass(frozen=True)
class Unit:
    kind: str          # "head" or "neuron"
    layer: int
    index: int
    width: int         # number of input dimensions of the site this part occupies
    site: str          # "o_proj" / "out_proj" / "down_proj"
    sub: str = ""      # "full" / "deltanet" for Q heads; "" otherwise

    @property
    def key(self):
        return f"{self.kind}:{self.layer}:{self.index}"

    @property
    def cols(self):
        return range(self.index * self.width, (self.index + 1) * self.width)

    def as_dict(self):
        return {"kind": self.kind, "layer": self.layer, "index": self.index, "width": self.width,
                "site": self.site, "sub": self.sub, "key": self.key}


def _layer_is_full_attention(layer, cfg):
    lt = getattr(cfg, "layer_types", None)
    if lt is not None:
        return lt[layer] == "full_attention"
    return None


def units_for(model):
    """Every part of a model, in pool order (heads of layer 0, neurons of layer 0, heads of layer 1, ...).

    The site is found from the module tree, not from the config alone, so a tiny test model and the real one take the
    same path: a layer whose block has `self_attn` and no `linear_attn` is full attention; a block with
    `linear_attn` is DeltaNet.
    """
    layers = text_layers(model)
    cfg = text_config(model)
    out = []
    for l, block in enumerate(layers):
        attn = getattr(block, "self_attn", None)
        lin = getattr(block, "linear_attn", None)
        if attn is not None and lin is None:
            width = getattr(cfg, "head_dim", None)
            n = getattr(cfg, "num_attention_heads", None)
            if width is None or n is None:
                width = attn.o_proj.in_features // cfg.num_attention_heads
                n = attn.o_proj.in_features // width
            if width * n != attn.o_proj.in_features:
                raise SystemExit(f"layer {l}: {n} heads of width {width} do not cover o_proj input "
                                 f"{attn.o_proj.in_features}")
            for h in range(n):
                out.append(Unit("head", l, h, width, "o_proj", "full"))
        elif lin is not None:
            width = getattr(cfg, "linear_value_head_dim", None)
            n = getattr(cfg, "linear_num_value_heads", None)
            if width is None or n is None:
                raise SystemExit(f"layer {l}: DeltaNet layer without linear_value_head_dim/linear_num_value_heads")
            if width * n != lin.out_proj.in_features:
                raise SystemExit(f"layer {l}: {n} value heads of width {width} do not cover out_proj input "
                                 f"{lin.out_proj.in_features}")
            for h in range(n):
                out.append(Unit("head", l, h, width, "out_proj", "deltanet"))
        else:
            raise SystemExit(f"layer {l} has neither self_attn nor linear_attn")
        n_neurons = block.mlp.down_proj.in_features
        for j in range(n_neurons):
            out.append(Unit("neuron", l, j, 1, "down_proj"))
    return out


def text_layers(model):
    """The decoder's layer list (same rule as neuron_report.find_text_layers, reduced to the list)."""
    import torch
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) and hasattr(getattr(mod[0], "mlp", None), "down_proj"):
            key = ("language" in name, len(mod))
            if best is None or key > best[0]:
                best = (key, mod)
    if best is None:
        raise SystemExit("no ModuleList of blocks with mlp.down_proj found")
    return best[1]


def text_config(model):
    cfg = model.config
    return cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg


def unit_site_module(layers, unit):
    block = layers[unit.layer]
    if unit.site == "down_proj":
        return block.mlp.down_proj
    if unit.site == "o_proj":
        return block.self_attn.o_proj
    if unit.site == "out_proj":
        return block.linear_attn.out_proj
    raise SystemExit(f"unknown site {unit.site}")


def check_pool(units, expected):
    """Raise unless the pool matches the design's counts for this family."""
    heads = sum(u.kind == "head" for u in units)
    neurons = sum(u.kind == "neuron" for u in units)
    got = {"heads": heads, "neurons": neurons, "pool": len(units)}
    if expected and any(got[k] != expected[k] for k in got):
        raise SystemExit(f"unit pool {got} does not match config {expected}")
    return got


# ---------------------------------------------------------------- frame and readout

def frame_text(persona, question):
    return f"System: {persona}\nUser: {question}\nAssistant: Answer:"


def persona_line(personas, name, value):
    return personas[name].format(value=value)


def option_token_ids(tok, letter):
    """Every token id that spells the option letter as a leading token: " A" and "A" (toolkit J)."""
    ids = set()
    for s in (f" {letter}", letter):
        got = tok(s, add_special_tokens=False).input_ids
        if len(got) == 1:
            ids.add(got[0])
    if not ids:
        raise SystemExit(f"option {letter!r} is not a single token in this tokenizer")
    return sorted(ids)


def readout(logits_last, ids_a, ids_b):
    """Option masses and the top next token from one last-position logit row (Design, Frame)."""
    import torch
    logits_last = logits_last.detach()
    p = torch.softmax(logits_last.float(), dim=-1)
    pa = float(p[ids_a].sum())
    pb = float(p[ids_b].sum())
    top = int(torch.argmax(logits_last))
    return {"p_a": pa, "p_b": pb, "option_mass": pa + pb, "top_token": top, "top_prob": float(p[top]),
            "top_is_letter": top in ids_a or top in ids_b}


def consistent_prob(rec, consistent):
    return rec["p_a"] if consistent == "A" else rec["p_b"]


# ---------------------------------------------------------------- hooks

class SiteRecorder:
    """Forward hook on a site module (o_proj / out_proj / down_proj) that keeps the module's input.

    The input is the activation the part writes from: for o_proj and out_proj it is the concatenated head outputs, for
    down_proj it is the MLP activation. `capture_grad` also keeps the gradient flowing into that input (the backward
    pass must be run by the caller after the forward).
    """

    def __init__(self, module, capture_grad=False):
        self.module = module
        self.capture_grad = capture_grad
        self.activation = None
        self.grad = None
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, _mod, args, _out):
        x = args[0]
        self.activation = x.detach()
        if self.capture_grad:
            x.register_hook(self._grad_hook)
        return None

    def _grad_hook(self, g):
        self.grad = g.detach()
        return g

    def close(self):
        self.handle.remove()


class SiteReplacer:
    """Forward pre-hook on a site module that overwrites chosen columns at chosen positions.

    `values` is a tensor of shape (n_positions, n_cols) and `positions` a list of position indices (into the sequence
    dimension of the module input, which for a batch of 1 is the token position). The do-nothing arm is this same
    class with the receiving run's own activation in `values`, which must reproduce the plain logits (G1).
    """

    def __init__(self, module, columns, positions, values):
        self.module = module
        self.columns = list(columns)
        self.positions = list(positions)
        self.values = values
        self.handle = module.register_forward_pre_hook(self._hook)

    def _hook(self, _mod, args):
        import torch
        x = args[0]
        if x.shape[0] != 1:
            raise SystemExit("replacement expects batch size 1")
        x = x.clone()
        cols = torch.as_tensor(self.columns, device=x.device)
        pos = torch.as_tensor(self.positions, dtype=torch.long, device=x.device)
        v = self.values.to(x.dtype)
        if v.shape != (len(pos), len(cols)):
            raise SystemExit(f"replacement values {tuple(v.shape)} do not match {len(pos)} positions x "
                             f"{len(cols)} columns")
        x[0, pos.unsqueeze(1), cols.unsqueeze(0)] = v   # advanced indexing: a copy, not a view
        return (x,) + tuple(args[1:])

    def close(self):
        self.handle.remove()


def replace_parts(layers, parts, positions, values_by_part):
    """Install one SiteReplacer per site, covering every part in `parts`.

    `values_by_part` maps a site key ("o_proj:3") to the source run's site input of shape (n_positions, site width);
    each hook takes the columns of the parts at its site. Returns the list of installed hooks (call close() on each).
    Parts at the same site share one hook, so the columns are written in one place.
    """
    import torch
    by_site = {}
    for u in parts:
        by_site.setdefault((u.site, u.layer), []).append(u)
    handles = []
    for (site, layer), us in by_site.items():
        us = sorted(us, key=lambda u: u.index)
        cols = []
        for u in us:
            cols.extend(u.cols)
        # the values are the source run's site input, cut to this hook's columns
        vals = values_by_part[f"{site}:{layer}"][:, cols]
        module = unit_site_module(layers, us[0])
        handles.append(SiteReplacer(module, cols, positions, vals))
    return handles


# ---------------------------------------------------------------- suffix alignment

def shared_suffix_positions(prefix_ids, full_ids):
    """Positions of the shared suffix in the full sequence, aligned from the end.

    The shared suffix is the longest run of identical tokens at the end of both sequences. Returns (start, positions)
    where positions is the list of indices into full_ids that lie in the suffix. Raises when the suffix is empty.
    """
    n = 0
    while n < min(len(prefix_ids), len(full_ids)) and prefix_ids[-1 - n] == full_ids[-1 - n]:
        n += 1
    if n == 0:
        raise SystemExit("no shared suffix between the You and third-person frames")
    start = len(full_ids) - n
    return start, list(range(start, len(full_ids)))


def you_values_at_third_positions(you_acts, you_ids, third_ids, suffix_len=None):
    """The You run's activations for each part, placed at the third-person run's suffix positions.

    `you_acts` maps a site key to the You run's site input (1, T_you, D). The suffix is aligned from the end, so the
    j-th token from the end of the third-person run takes the j-th token from the end of the You run (Limit: the
    prefixes differ in length, so absolute positions differ; the values are the You run's own).
    """
    if you_ids is None:
        # The You run's token ids were not kept; the suffix length is given instead (the caller aligned it once).
        if suffix_len is None:
            raise SystemExit("you_values_at_third_positions needs the You run's ids or the suffix length")
        n = suffix_len
        start = len(third_ids) - n
        positions = list(range(start, len(third_ids)))
    else:
        start, positions = shared_suffix_positions(you_ids, third_ids)
        n = len(positions)
    out = {}
    import torch
    for site, x in you_acts.items():
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        if x.shape[1] < n:
            raise SystemExit(f"You run has {x.shape[1]} positions, suffix needs {n}")
        out[site] = x[0, x.shape[1] - n:, :].clone()
    return start, positions, out


# ---------------------------------------------------------------- estimator

def part_estimate(act, grad, unit, position=-1):
    """e = sum over the part's dimensions of activation x gradient at `position` (a float)."""
    a = act[0, position, list(unit.cols)].float()
    g = grad[0, position, list(unit.cols)].float()
    return float((a * g).sum())


def estimates_for_prompt(acts, grads, units, position=-1):
    """e for every part, in one pass over the pool (vectorised per site)."""
    import torch
    est = torch.empty(len(units), dtype=torch.float64)
    by_site = {}
    for i, u in enumerate(units):
        by_site.setdefault((u.site, u.layer), []).append(i)
    for (site, layer), idxs in by_site.items():
        key = f"{site}:{layer}"
        a = acts[key][0, position].float()
        g = grads[key][0, position].float()
        prod = a * g
        widths = {units[i].width for i in idxs}
        if len(widths) != 1:
            raise SystemExit(f"site {key} mixes part widths {widths}")
        w = widths.pop()
        seg = prod.split(w)
        if len(seg) != len(idxs):
            raise SystemExit(f"site {key}: {len(seg)} slices of width {w} for {len(idxs)} parts")
        for i, s in zip(idxs, seg):
            est[i] = float(s.sum())
    return est


def normalised_abs(est):
    """|e| divided by its sum over all parts (Design, Ranking: each prompt weighs the same)."""
    import torch
    a = est.abs()
    s = a.sum()
    if float(s) == 0.0:
        raise SystemExit("all estimates are zero for a prompt; cannot normalise")
    return a / s


# ---------------------------------------------------------------- files

def load_config(path=None):
    return json.loads(Path(path or HERE / "config.json").read_text(encoding="utf-8"))


def load_prompts(path=None):
    return json.loads(Path(path or HERE / "prompts205.json").read_text(encoding="utf-8"))


def units_for_meta(meta):
    """The parts of a stage from its meta.json (units list and, when present, per-unit kind/layer/index/width)."""
    if meta.get("unit_meta"):
        return {d["key"]: Unit(d["kind"], d["layer"], d["index"], d["width"], d["site"], d.get("sub", ""))
                for d in meta["unit_meta"]}
    raise SystemExit("meta.json has no unit_meta; rerun run205.py to record kinds and widths")
