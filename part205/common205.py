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


def question_block_char(persona):
    """Character index where the question block ("User: ...") starts in frame_text(persona, ...)."""
    return len(f"System: {persona}\n")


def persona_line(personas, name, value):
    """The persona line for `name`. A template that names a value refuses value=None, so no frame can say
    "cares about None" (review r1 H-F3/D-F2); the off-target arm uses its own value-free lines."""
    template = personas[name]
    if "{value}" in template and value is None:
        raise SystemExit(f"persona line {template!r} needs a value; the off-target arm uses off_target_personas")
    return template.format(value=value)


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
    """The readout of one last-position logit row (Design, Frame, lines 62-64).

    p_a and p_b are the probability of each option letter NORMALISED OVER THE TWO (p_a + p_b = 1); every T, O, gate
    and prediction reads these. mass_a and mass_b are the raw next-token masses of the two letters ("both masses
    printed"); option_mass = mass_a + mass_b is what G0 and the broken-answer count read.
    """
    import torch
    logits_last = logits_last.detach()
    p = torch.softmax(logits_last.float(), dim=-1)
    ma = float(p[ids_a].sum())
    mb = float(p[ids_b].sum())
    tot = ma + mb
    if tot > 0.0:
        pa, pb = ma / tot, mb / tot
    else:
        # both masses underflow to 0 in float32: fall back to the letters' log-sum-exp difference
        la = torch.logsumexp(logits_last.float()[ids_a], 0)
        lb = torch.logsumexp(logits_last.float()[ids_b], 0)
        pa = float(torch.sigmoid(la - lb))
        pb = 1.0 - pa
    top = int(torch.argmax(logits_last))
    return {"p_a": pa, "p_b": pb, "mass_a": ma, "mass_b": mb, "option_mass": tot, "top_token": top,
            "top_prob": float(p[top]), "top_is_letter": top in ids_a or top in ids_b}


def max_abs_logit_diff(a, b):
    """max |a - b| over a full last-position logit row (G1; a scalar, the vectors are not stored)."""
    return float((a.detach().float().cpu() - b.detach().float().cpu()).abs().max())


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
        # device AND dtype: the runner stores every captured activation on the CPU (run205.py), so on family Q (the
        # A770) the values are CPU tensors while x is XPU; index_put_ needs every operand on one device (review r2
        # D-F4, H-N6)
        if not isinstance(self.values, torch.Tensor):
            self.values = torch.as_tensor(self.values)
        v = self.values.to(device=x.device, dtype=x.dtype)
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

def question_start(tok, text, persona):
    """Token index of the first token of the question block (the "User:" line) in frame_text(persona, question).

    Uses the tokenizer's character offsets; raises if no token starts exactly at the block's first character (then the
    persona line and the question block share a token and the block cannot be copied cleanly).
    """
    c = question_block_char(persona)
    if text[c:c + 5] != "User:":
        raise SystemExit(f"question block does not start at char {c}: {text[c:c + 12]!r}")
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    for i, (a, _b) in enumerate(enc["offset_mapping"]):
        if a == c:
            return i, list(enc["input_ids"])
        if a > c:
            break
    raise SystemExit(f"no token starts at the question block (char {c}) of {text!r}")


def copy_span(src_ids, src_qstart, dst_ids, dst_qstart):
    """The copied span (Design, Method, lines 105-112): the question and answer cue, identical tokens in both frames,
    aligned from the end; nothing of the persona line.

    `src_qstart`/`dst_qstart` are question_start() of each frame. Returns (src_positions, dst_positions), equal
    length, the last position of each run included. Raises unless the two blocks are token-for-token identical, so
    a copy can never carry persona tokens or a different question (review r1 H-F4, D-F1, D-F2b, D-F3).
    """
    src_block = list(src_ids[src_qstart:])
    dst_block = list(dst_ids[dst_qstart:])
    if not src_block or src_block != dst_block:
        raise SystemExit(f"the question blocks differ: source {len(src_block)} tokens, receiver {len(dst_block)}; "
                         "a copy needs identical question and answer-cue tokens in both frames")
    n = len(dst_block)
    return list(range(len(src_ids) - n, len(src_ids))), list(range(len(dst_ids) - n, len(dst_ids)))


def values_at(src_acts, src_positions):
    """The source run's site inputs at `src_positions`, one (n_positions, width) tensor per site key."""
    import torch
    out = {}
    for site, x in src_acts.items():
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        if x.shape[1] <= max(src_positions):
            raise SystemExit(f"source run has {x.shape[1]} positions, span needs {max(src_positions) + 1}")
        out[site] = x[0, src_positions, :].clone()
    return out


# ---------------------------------------------------------------- records and locks

def read_jsonl(path, repair=False):
    """The records of a JSONL file. A last line that does not parse (a write cut by a kill) is ignored with a warning
    and, with `repair`, truncated away so the next append starts on a fresh line. A bad line before the last raises
    (that is corruption, not a torn write). Review r1 H-F7, D-F15.

    A final line that parses but has no trailing newline is torn too (review r2 H-N3): the kill landed between the
    JSON bytes and the newline, so the file cannot be known to hold the whole record. It is ignored with a warning and,
    with `repair`, truncated away, so the next append can never be joined onto it. A file that ends in "\\n" is whole.
    """
    import sys
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    lines = raw.split(b"\n")
    unterminated = bool(raw) and not raw.endswith(b"\n")
    out = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        last = all(not x.strip() for x in lines[i + 1:])
        rec = None
        try:
            rec = json.loads(line.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            rec = None
        if rec is not None and not (last and unterminated):
            out.append(rec)
            continue
        if not last:
            raise SystemExit(f"{path}: line {i + 1} does not parse and is not the last line (corrupt file)")
        why = "no trailing newline" if rec is not None else "does not parse"
        print(f"warning: {path}: ignoring a torn last line ({why}; {len(line)} bytes)", file=sys.stderr)
        if repair:
            keep = b"\n".join(lines[:i])
            with path.open("r+b") as f:
                f.truncate(len(keep) + (1 if keep else 0))
    return out


def lock_file(f, blocking=True):
    """An exclusive lock on an open file: fcntl.flock on Linux, msvcrt.locking on byte 0 on Windows, where the A770
    job runs. Amendment 3 (2026-11-14): an unconditional `import fcntl` died at family Q's first stage launch, as it
    had for Part 209 (part209/run209.py). Raises BlockingIOError when `blocking` is False and the lock is held."""
    try:
        import fcntl
    except ImportError:
        import msvcrt
        f.seek(0)
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise BlockingIOError(str(exc)) from exc
        return
    fcntl.flock(f.fileno(), fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB)


def unlock_file(f):
    try:
        import fcntl
    except ImportError:
        import msvcrt
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        return
    fcntl.flock(f.fileno(), fcntl.LOCK_UN)


class StageLock:
    """An exclusive, non-blocking lock on <stage dir>/.lock held for the whole run (review r1 D-F16). The OS drops it
    when the process ends, so a killed run leaves no stale lock."""

    def __init__(self, path):
        self.path = Path(path)
        self.f = self.path.open("a+")
        try:
            lock_file(self.f, blocking=False)
        except BlockingIOError:
            self.f.close()
            raise SystemExit(f"{self.path} is held by another run of this stage; refusing to start a second")
        self.f.seek(0)
        self.f.truncate()
        import os
        self.f.write(f"pid {os.getpid()}\n")
        self.f.flush()

    def release(self):
        if self.f and not self.f.closed:
            unlock_file(self.f)
            self.f.close()


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

def _fsync_dir(d):
    """Make a rename durable: fsync the directory that holds it (best effort; not every filesystem allows it)."""
    import os
    try:
        fd = os.open(str(d), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write_bytes(path, data):
    """Write `data` to a temp file in the same directory, fsync it, then os.replace it over `path` (review r2 D-F1,
    D-F2, D-F3, H-N2). A kill leaves either the old file or the new one, never a torn one; the temp file has a
    unique name, so two writers never share it."""
    import os
    import tempfile
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    _fsync_dir(path.parent)
    return path


def atomic_write_text(path, text):
    return atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path, obj, indent=1, sort_keys=True):
    return atomic_write_text(path, json.dumps(obj, indent=indent, sort_keys=sort_keys) + "\n")


def atomic_savez(path, **arrays):
    """np.savez_compressed to a temp file then os.replace (a savez is not atomic: a kill leaves a partial zip)."""
    import os
    import tempfile
    import numpy as np
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        # an open handle, not the name: np.savez_compressed appends ".npz" to a str path that does not end in it
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    _fsync_dir(path.parent)
    return path


def load_json_repairing(path, default):
    """json.loads(path), or `default` when the file is missing; with a warning when it is unreadable (a torn write
    from a killed run). The caller rewrites the file, so a torn file is repaired rather than left to brick the stage
    (review r2 D-F3). A missing file is not a warning."""
    import sys
    path = Path(path)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"warning: {path} is unreadable ({e}); starting from {default!r} and rewriting it", file=sys.stderr)
        return default


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
