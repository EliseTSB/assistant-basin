#!/usr/bin/env python3
"""run209.py: one stage of Part 209 per invocation, writing raw records only (no verdicts; those are score209.py's).

  python -I -B part209/run209.py --tokens                                   # once: part209/tokens.json (both tokenizers)
  python -I -B part209/run209.py --stage instruct --out results/part209      # R1, R1c (chat stages) and R3
  python -I -B part209/run209.py --stage qwen-chat --out results/part209 --device xpu
  python -I -B part209/run209.py --stage qwen-chat --out results/part209 --device xpu --r2 [--paraphrases 1,2
          --samples 4 --told-samples 2 --max-new-tokens 96 --batch 24]
  --pilot     the design's pilot: paraphrases 1 and 2 for R1/R3 frames and R2 rows, R2 samples 4 (told 2): 30 frames
              and 76 answers
  --arms A,B  R1/R3 for these arms only (the build's told-and-pole check: TOLD-*, NOBODY-TOLD-*)

Draft 2 (design a9c52cf6): 15 frame types, 120 frames; R2 arms ROOM-W to NOBODY-I and the three ROOM-W told arms, 816
answers per model; c for NOBODY (and its poles) against the NOBODY told poles, every other arm against ROOM-W's. Each R2
row records its raw new token ids and the 1-based token at which a posture word first appears (the pilot's
max_new_tokens rule), and every batch's size and seconds go to r2-<stage>.timing.jsonl.

Writes, under --out:
  r1r3-<stage>.json     R1 masses and top tokens per frame, R1c (chat stages), R3 per-layer held-out accuracies, the
                        chosen layer, the label-shuffle null, per-frame projections on the sit-stand and sit-lie axes,
                        and the c coordinates the projections give (score209.py recomputes c from the projections)
  states-<stage>.npz    final-token residual states (float16) for every frame and reference sentence, at the layers of
                        the 50-80% window only, so R3 can be re-read without the model
  r2-<stage>.jsonl      (--r2) one sampled answer per line, appended and flushed; rows already present are skipped
and records the stage's model id, revision, dtype, device and library versions in part209/models.json.

Every R1/R1c/R3 pass is batch size 1: there is no padding, so the final token is the last token of the input. (Part
201: batch shape changes bf16 answers on the A770; a read at batch 1 has no such term.)

R2 sampling: temperature 0.7 and top-p 0.95 are applied by PerRowSampler, a logits processor that draws each row's
token with that row's own torch.Generator (seeded from the row id) on the CPU, and returns logits that put all mass on
the drawn token. generate() runs with do_sample=True and no warpers of its own (temperature 1, top-k 0, top-p 1, a
fresh GenerationConfig), so the draw is the processor's. A row's answer therefore does not depend on which rows share
its batch, up to the device's batch-shape arithmetic (a few ulps in bf16 on the A770). Vocabulary-size work (softmax,
top-k, sort) runs on the CPU (PROCEDURE, section 4: a top-k over the vocabulary on the A770 did not finish in 35 min).
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TOKENIZERS = {  # the tokenizer each family's posture ids are checked against (tokens.json)
    "O": ("allenai/OLMo-2-0425-1B-Instruct", "48d788eca847d4d7548f375ad03d3c9312f6139e"),
    "Q": ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
}
WORDS = [" sitting", " seated", " standing", " lying", " laying"]


# ---------------------------------------------------------------- files

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)


def atomic_write_json(path, obj):
    atomic_write_text(path, json.dumps(obj, indent=1, ensure_ascii=False) + "\n")


def read_jsonl(path):
    """Rows of a JSONL file; a torn last line (a kill mid-write) is ignored."""
    rows = []
    p = Path(path)
    if not p.exists():
        return rows
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def load_config():
    return load_json(HERE / "config.json")


def load_prompts():
    return load_json(HERE / "prompts209.json")


def load_reference():
    return load_json(HERE / "reference209.json")


# ---------------------------------------------------------------- stages and loading

def stage_spec(cfg, stage):
    st = next((s for s in cfg["stages"] if s["stage"] == stage), None)
    if st is None:
        raise SystemExit(f"unknown stage {stage!r}; config has {[s['stage'] for s in cfg['stages']]}")
    fam = cfg["families"][st["family"]]
    spec = next((x for x in fam["stages"] if x["stage"] == stage), None)
    if spec is None:
        raise SystemExit(f"stage {stage!r} missing from family {st['family']}")
    return {"stage": stage, "family": st["family"], "model_id": spec["model_id"], "revision": spec["revision"],
            "chat": bool(spec.get("chat"))}


def load_stage(cfg, stage, device_override=None):
    """(tokenizer, model, env) for one stage, exactly as part205/run205.py's load_stage. One model per process."""
    st = stage_spec(cfg, stage)
    fam = cfg["families"][st["family"]]
    device = device_override or fam["device"]
    import torch
    dtype = torch.float32 if fam["dtype"] == "float32" else torch.bfloat16
    if st["family"] == "Q":
        import neuron_report
        tok, model = neuron_report.load_model(st["model_id"], st["revision"], device,
                                              attn_implementation=fam["attn_implementation"], dtype=dtype)
        env = dict(neuron_report.LAST_LOAD)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(st["model_id"], revision=st["revision"])
        model = AutoModelForCausalLM.from_pretrained(st["model_id"], revision=st["revision"], dtype=dtype,
                                                     attn_implementation=fam["attn_implementation"])
        model.to(device).eval()
        torch.set_grad_enabled(False)
        # Weights read from /mnt/e can stay memory-mapped and thrash in disk wait under a memory cap (gotcha
        # 2026-11-05): give each tensor its own memory, one at a time.
        if device == "cpu":
            for p in model.parameters():
                p.data = p.data.clone()
        import transformers
        env = {"dtype": fam["dtype"], "device": device, "attn_implementation": fam["attn_implementation"],
               "attn_implementation_requested": fam["attn_implementation"],
               "torch": torch.__version__, "transformers": transformers.__version__}
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    env.update({k: st[k] for k in ("stage", "family", "model_id", "revision", "chat")})
    return tok, model, env


def write_models_json(path, env):
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path) + ".lock", "a+") as lk:
        fcntl.flock(lk.fileno(), fcntl.LOCK_EX)
        try:
            data = load_json(path) if path.exists() else {"part": 209, "stages": {}}
        except json.JSONDecodeError:
            data = {"part": 209, "stages": {}}
        if not isinstance(data, dict) or not isinstance(data.get("stages"), dict):
            data = {"part": 209, "stages": {}}
        row = {k: env.get(k) for k in ("model_id", "revision", "family", "dtype", "device", "attn_implementation",
                                        "torch", "transformers")}
        row["recorded"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        data["stages"][env["stage"]] = row
        atomic_write_json(path, data)
        fcntl.flock(lk.fileno(), fcntl.LOCK_UN)


# ---------------------------------------------------------------- posture tokens

def word_ids(tok, word):
    return tok.encode(word, add_special_tokens=False)


def make_tokens_json(path=None):
    """Record whether each posture word is one token in each family's tokenizer; ' laying' joins LIE only if it is
    one token in both (Design, R1)."""
    from transformers import AutoTokenizer
    out = {"words": WORDS, "families": {}}
    for fam, (mid, rev) in TOKENIZERS.items():
        tok = AutoTokenizer.from_pretrained(mid, revision=rev)
        out["families"][fam] = {"tokenizer": mid, "revision": rev,
                                "ids": {w: word_ids(tok, w) for w in WORDS}}
    single = {w: all(len(out["families"][f]["ids"][w]) == 1 for f in out["families"]) for w in WORDS}
    out["single_in_both"] = single
    out["laying_in_LIE"] = bool(single[" laying"])
    for w in (" sitting", " seated", " standing", " lying"):
        if not single[w]:
            raise SystemExit(f"{w!r} is not one token in both tokenizers; the design's R1 needs it to be")
    atomic_write_json(path or HERE / "tokens.json", out)
    return out


def posture_ids(tokens_json, family, tok):
    """{SIT, STAND, LIE} -> token ids for this family, checked against the live tokenizer."""
    fam = tokens_json["families"][family]
    sets = {"SIT": [" sitting", " seated"], "STAND": [" standing"], "LIE": [" lying"]}
    if tokens_json.get("laying_in_LIE"):
        sets["LIE"].append(" laying")
    out = {}
    for k, words in sets.items():
        ids = []
        for w in words:
            live = word_ids(tok, w)
            if live != fam["ids"][w]:
                raise SystemExit(f"tokenizer mismatch for {w!r}: tokens.json {fam['ids'][w]} live {live}")
            if len(live) != 1:
                raise SystemExit(f"{w!r} is {len(live)} tokens; R1 reads single tokens only")
            ids.append(live[0])
        out[k] = ids
    return out


# ---------------------------------------------------------------- R1 and R3 passes

def r1_input(frame):
    """The plain text the R1/R3 pass reads: the frame, a blank line, the opening (no template)."""
    return frame["text"] + "\n\n" + frame["opening"]


def chat_input(tok, frame, family):
    msgs = [{"role": "user", "content": frame["text"]}]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if family == "Q":
        kwargs["enable_thinking"] = False
    s = tok.apply_chat_template(msgs, **kwargs)
    return s


def forward_final(model, tok, text, device):
    """One batch-1 pass with no special tokens added. Returns (final-token logits as float32 CPU, residual states
    after each layer at the final token as a float32 CPU tensor [L, H])."""
    import torch
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    logits = out.logits[0, -1].float().cpu()
    hs = torch.stack([h[0, -1].float().cpu() for h in out.hidden_states[1:]])  # after each layer
    return logits, hs


def readout(logits, ids, tok, top_k=10):
    import torch
    p = torch.softmax(logits.float(), dim=-1)
    m = {k: float(p[v].sum()) for k, v in ids.items()}
    p3 = m["SIT"] + m["STAND"] + m["LIE"]
    vals, idx = torch.topk(p, top_k)
    top = [[tok.decode([int(i)]), float(v)] for v, i in zip(vals, idx)]
    return {"SIT": m["SIT"], "STAND": m["STAND"], "LIE": m["LIE"], "P3": p3,
            "s": (m["SIT"] / p3) if p3 > 0 else None, "top10": top}


def window_layers(n_layers, window=(0.5, 0.8)):
    """1-based layer indices l with window[0] <= l/L <= window[1]."""
    return [l for l in range(1, n_layers + 1) if window[0] <= l / n_layers <= window[1] + 1e-12]


def fit_axis(X_a, X_b):
    """Axis a = mean(A) - mean(B) and threshold t = a . midpoint."""
    import numpy as np
    ma, mb = X_a.mean(0), X_b.mean(0)
    a = ma - mb
    t = float(a @ ((ma + mb) / 2))
    return a, t


def heldout_accuracy(X, posture, subject, pos_a, pos_b, fit_subjects, test_subjects):
    import numpy as np
    fit_a = X[(posture == pos_a) & np.isin(subject, fit_subjects)]
    fit_b = X[(posture == pos_b) & np.isin(subject, fit_subjects)]
    a, t = fit_axis(fit_a, fit_b)
    test = np.isin(subject, test_subjects) & np.isin(posture, [pos_a, pos_b])
    pred_a = (X[test] @ a) > t
    acc = float((pred_a == (posture[test] == pos_a)).mean())
    return acc, a, t


def choose_layer(accs, candidates):
    """Best held-out accuracy among the candidate layers; ties go to the deeper layer."""
    best = max(candidates, key=lambda l: (accs[l], l))
    return best


def shuffle_null(X, posture, subject, pos_a, pos_b, fit_subjects, test_subjects, n, seed):
    """Held-out accuracy with the two posture labels shuffled within each subject (both halves)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    keep = np.isin(posture, [pos_a, pos_b])
    Xk, pk, sk = X[keep], posture[keep].copy(), subject[keep]
    accs = []
    for _ in range(n):
        ps = pk.copy()
        for s in np.unique(sk):
            idx = np.where(sk == s)[0]
            ps[idx] = rng.permutation(ps[idx])
        acc, _, _ = heldout_accuracy(Xk, ps, sk, pos_a, pos_b, fit_subjects, test_subjects)
        accs.append(acc)
    accs = np.array(accs)
    return {"n": int(n), "mean": float(accs.mean()), "p95": float(np.quantile(accs, 0.95)),
            "max": float(accs.max())}


def axis_block(ref_states, ref_meta, frame_states, frames, pos_b, cfg):
    """R3 for one axis (sitting against pos_b): per-layer held-out accuracy, the chosen layer, the null, and every
    frame's projection at the chosen layer, plus c per arm from the told poles."""
    import numpy as np
    L = ref_states.shape[1]
    posture = np.array([r["posture"] for r in ref_meta])
    subject = np.array([r["subject_index"] for r in ref_meta])
    rc = cfg["r3"]
    cands = window_layers(L, tuple(rc["window"]))
    accs = {}
    for l in range(1, L + 1):
        acc, _, _ = heldout_accuracy(ref_states[:, l - 1], posture, subject, "sitting", pos_b,
                                     rc["fit_subjects"], rc["test_subjects"])
        accs[l] = acc
    chosen = choose_layer(accs, cands)
    _, a, t = heldout_accuracy(ref_states[:, chosen - 1], posture, subject, "sitting", pos_b,
                               rc["fit_subjects"], rc["test_subjects"])
    null = shuffle_null(ref_states[:, chosen - 1], posture, subject, "sitting", pos_b, rc["fit_subjects"],
                        rc["test_subjects"], rc["null_shuffles"], rc["null_seed"])
    proj = (frame_states[:, chosen - 1] @ a).tolist()
    rows = [{"id": f["id"], "arm": f["arm"], "k": f["k"], "proj": float(p)} for f, p in zip(frames, proj)]
    poles = pole_map(cfg, pos_b)
    c = c_for_arms(rows, poles)
    return {"pos_a": "sitting", "pos_b": pos_b, "candidates": cands, "accuracy_by_layer": accs,
            "chosen_layer": chosen, "chosen_accuracy": accs[chosen], "threshold": t,
            "axis_norm": float(np.linalg.norm(a)), "null": null, "frames": rows, "c": c, "poles": poles}


def pole_map(cfg, pos_b="standing"):
    """Which told poles scale each arm's c (Design draft 2, R3): NOBODY and its own poles use the NOBODY told poles;
    every other arm uses the ROOM-W told poles. The sit-lie axis has no NOBODY pole, so NOBODY gets none there."""
    pc = cfg["r3"]["poles"]
    zero = pc["default"]["zero"] if pos_b == "standing" else pc["default"]["lie"]
    out = {"default": {"one": pc["default"]["one"], "zero": zero}}
    if pos_b == "standing":
        for arm in ("NOBODY", pc["NOBODY"]["one"], pc["NOBODY"]["zero"]):
            out[arm] = {"one": pc["NOBODY"]["one"], "zero": pc["NOBODY"]["zero"]}
    else:
        for arm in ("NOBODY", pc["NOBODY"]["one"], pc["NOBODY"]["zero"]):
            out[arm] = None
    return out


def c_for_arms(rows, poles):
    out = {}
    for arm in sorted({r["arm"] for r in rows}):
        pl = poles.get(arm, poles["default"])
        out[arm] = None if pl is None else c_by_arm(rows, pl["one"], pl["zero"]).get(arm)
    return out


def c_by_arm(rows, pole_one, pole_zero):
    """c = (mean projection of the arm - mean of pole_zero) / (mean of pole_one - mean of pole_zero)."""
    by = {}
    for r in rows:
        by.setdefault(r["arm"], []).append(r["proj"])
    m = {a: sum(v) / len(v) for a, v in by.items()}
    if pole_one not in m or pole_zero not in m:
        return {a: None for a in m}
    den = m[pole_one] - m[pole_zero]
    return {a: ((m[a] - m[pole_zero]) / den if den != 0 else None) for a in m}


def run_r1r3(model, tok, env, frames, refs, cfg, tokens_json, device, log=print):
    """Every frame and reference sentence through one batch-1 pass each; returns (record, states dict)."""
    import numpy as np
    import torch
    fam = env["family"]
    ids = posture_ids(tokens_json, fam, tok)
    r1, r1c = [], []
    fstates = []
    t0 = time.time()
    for i, f in enumerate(frames):
        logits, hs = forward_final(model, tok, r1_input(f), device)
        r1.append({"id": f["id"], "arm": f["arm"], "k": f["k"], **readout(logits, ids, tok)})
        fstates.append(hs.numpy())
        if env.get("chat"):
            text = chat_input(tok, f, fam) + f["opening"]
            logits_c, _ = forward_final(model, tok, text, device)
            r1c.append({"id": f["id"], "arm": f["arm"], "k": f["k"], **readout(logits_c, ids, tok)})
        if (i + 1) % 24 == 0:
            log(f"[r1] {i + 1}/{len(frames)} frames, {time.time() - t0:.1f}s")
    rstates = []
    for i, r in enumerate(refs):
        _, hs = forward_final(model, tok, r["text"], device)
        rstates.append(hs.numpy())
        if (i + 1) % 120 == 0:
            log(f"[r3] {i + 1}/{len(refs)} reference sentences, {time.time() - t0:.1f}s")
    F = np.stack(fstates)  # [n_frames, L, H]
    R = np.stack(rstates)  # [n_refs, L, H]
    r3 = {"layers": int(R.shape[1]),
          "sit_stand": axis_block(R, refs, F, frames, "standing", cfg),
          "sit_lie": axis_block(R, refs, F, frames, "lying", cfg)}
    rec = {"part": 209, "stage": env["stage"], "family": fam, "chat": bool(env.get("chat")), "env": env,
           "posture_ids": ids, "laying_in_LIE": bool(tokens_json.get("laying_in_LIE")),
           "n_frames": len(frames), "n_reference": len(refs), "seconds": round(time.time() - t0, 1),
           "r1": r1, "r1c": r1c if env.get("chat") else None, "r3": r3}
    win = window_layers(R.shape[1], tuple(cfg["r3"]["window"]))
    states = {"frame_ids": np.array([f["id"] for f in frames]), "ref_text": np.array([r["text"] for r in refs]),
              "layers": np.array(win), "frames": F[:, [l - 1 for l in win]].astype(np.float16),
              "refs": R[:, [l - 1 for l in win]].astype(np.float16)}
    return rec, states


# ---------------------------------------------------------------- R2 sampling

def row_seed(row_id):
    return int(hashlib.sha256(("209|" + row_id).encode("utf-8")).hexdigest()[:8], 16)


class PerRowSampler:
    """Logits processor: temperature, then top-p, then one draw per row from that row's own generator, on the CPU.
    Returns logits with 0 at the drawn token and -inf elsewhere, so generate()'s own sampling picks it."""

    def __init__(self, seeds, temperature, top_p, prefilter=2048):
        import torch
        self.gens = []
        for s in seeds:
            g = torch.Generator(device="cpu")
            g.manual_seed(int(s))
            self.gens.append(g)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.prefilter = int(prefilter)

    def draw(self, logits_row, gen):
        import torch
        p = torch.softmax(logits_row.float() / self.temperature, dim=-1)
        k = min(self.prefilter, p.numel())
        vals, idx = torch.topk(p, k)
        if float(vals.sum()) < self.top_p:   # the top-p set is larger than the prefilter: sort everything
            vals, idx = torch.sort(p, descending=True)
        cum = torch.cumsum(vals, dim=0)
        # keep tokens up to and including the first whose cumulative mass reaches top_p
        n_keep = int(torch.searchsorted(cum, torch.tensor(self.top_p, dtype=cum.dtype)).item()) + 1
        n_keep = max(1, min(n_keep, vals.numel()))
        kept = vals[:n_keep] / vals[:n_keep].sum()
        j = int(torch.multinomial(kept, 1, generator=gen).item())
        return int(idx[j].item())

    def __call__(self, input_ids, scores):
        import torch
        cpu = scores.detach().float().cpu()
        out = torch.full_like(cpu, float("-inf"))
        for b in range(cpu.shape[0]):
            out[b, self.draw(cpu[b], self.gens[b])] = 0.0
        return out.to(device=scores.device, dtype=scores.dtype)


def r2_rows(frames, stage, paraphrases, samples_main, samples_told):
    rows = []
    for f in frames:
        if not f.get("r2"):
            continue
        if paraphrases and f["k"] not in paraphrases:
            continue
        n = samples_told if f["arm"].startswith("TOLD-") else samples_main
        for j in range(n):
            rid = f"{stage}|{f['id']}|{j}"
            rows.append({"row_id": rid, "stage": stage, "frame_id": f["id"], "arm": f["arm"], "k": f["k"],
                         "sample": j, "seed": row_seed(rid), "text": f["text"]})
    return rows


def first_posture_token(tok, ids, pattern):
    """1-based count of new tokens at which a posture word first appears in the decoded answer, or None."""
    import re
    rx = re.compile(pattern, re.IGNORECASE)
    for t in range(1, len(ids) + 1):
        if rx.search(tok.decode(ids[:t], skip_special_tokens=True)):
            return t
    return None


def generate_batch(model, tok, env, batch, cfg_r2, max_new_tokens, device, eos_ids):
    import torch
    from transformers import GenerationConfig, LogitsProcessorList
    tok.padding_side = "left"
    prompts = [chat_input(tok, {"text": r["text"]}, env["family"]) for r in batch]
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    sampler = PerRowSampler([r["seed"] for r in batch], cfg_r2["temperature"], cfg_r2["top_p"])
    gc = GenerationConfig(do_sample=True, temperature=1.0, top_k=0, top_p=1.0, max_new_tokens=max_new_tokens,
                          eos_token_id=eos_ids, pad_token_id=tok.pad_token_id)
    out = model.generate(**enc, generation_config=gc, logits_processor=LogitsProcessorList([sampler]))
    new = out[:, enc.input_ids.shape[1]:].cpu()
    res = []
    for r, ids in zip(batch, new):
        ids = ids.tolist()
        cut = len(ids)
        for i, t in enumerate(ids):
            if t in eos_ids:
                cut = i
                break
        res.append({"answer": tok.decode(ids[:cut], skip_special_tokens=True).strip(),
                    "n_new_tokens": cut, "finished": cut < len(ids), "new_token_ids": ids[:cut],
                    "first_posture_token": first_posture_token(tok, ids[:cut], cfg_r2["first_posture_regex"])})
    return res


def eos_ids_for(model, tok):
    ids = set()
    gc = getattr(model, "generation_config", None)
    e = getattr(gc, "eos_token_id", None) if gc is not None else None
    if isinstance(e, int):
        ids.add(e)
    elif e:
        ids.update(int(x) for x in e)
    if tok.eos_token_id is not None:
        ids.add(int(tok.eos_token_id))
    for t in ("<|im_end|>", "<|endoftext|>"):
        tid = tok.convert_tokens_to_ids(t)
        if isinstance(tid, int) and tid >= 0 and tid != tok.unk_token_id:
            ids.add(tid)
    return sorted(ids)


def run_r2(model, tok, env, frames, cfg, out_path, paraphrases=None, samples_main=None, samples_told=None,
           max_new_tokens=None, batch_size=24, device="cpu", limit=None, log=print):
    c2 = cfg["r2"]
    samples_main = c2["samples_main"] if samples_main is None else samples_main
    samples_told = c2["samples_told"] if samples_told is None else samples_told
    max_new_tokens = c2["max_new_tokens"] if max_new_tokens is None else max_new_tokens
    rows = r2_rows(frames, env["stage"], paraphrases, samples_main, samples_told)
    if limit:
        rows = rows[:limit]
    done = {r["row_id"] for r in read_jsonl(out_path) if "answer" in r}
    todo = [r for r in rows if r["row_id"] not in done]
    log(f"[r2] {len(rows)} rows, {len(done & {r['row_id'] for r in rows})} already written, {len(todo)} to go")
    eos = eos_ids_for(model, tok)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n = 0
    timing_path = Path(str(out_path).replace(".jsonl", ".timing.jsonl"))
    with open(out_path, "a", encoding="utf-8", newline="\n") as fh, \
            open(timing_path, "a", encoding="utf-8", newline="\n") as th:
        for i in range(0, len(todo), batch_size):
            batch = todo[i:i + batch_size]
            tb = time.time()
            res = generate_batch(model, tok, env, batch, c2, max_new_tokens, device, eos)
            th.write(json.dumps({"batch_size": batch_size, "rows": len(batch), "seconds": round(time.time() - tb, 3),
                                 "max_new_tokens": max_new_tokens, "new_tokens": sum(o["n_new_tokens"] for o in res)})
                     + "\n")
            th.flush()
            for r, o in zip(batch, res):
                row = {k: r[k] for k in ("row_id", "stage", "frame_id", "arm", "k", "sample", "seed")}
                row.update(o)
                row.update({"temperature": c2["temperature"], "top_p": c2["top_p"],
                            "max_new_tokens": max_new_tokens, "batch_size": batch_size})
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
                n += 1
            log(f"[r2] {i + len(batch)}/{len(todo)} rows, {time.time() - t0:.1f}s "
                f"({n / max(time.time() - t0, 1e-9):.2f} answers/s)")
    return n


# ---------------------------------------------------------------- main

def parse_paraphrases(s):
    if not s:
        return None
    return {int(x) for x in s.split(",") if x.strip()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage")
    ap.add_argument("--out", default=str(ROOT / "results" / "part209"))
    ap.add_argument("--device")
    ap.add_argument("--tokens", action="store_true", help="write part209/tokens.json and stop")
    ap.add_argument("--r2", action="store_true", help="sampled answers (chat stages) instead of R1/R3")
    ap.add_argument("--paraphrases", help="comma list of paraphrase numbers (pilot)")
    ap.add_argument("--samples", type=int)
    ap.add_argument("--told-samples", type=int)
    ap.add_argument("--max-new-tokens", type=int)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--limit", type=int, help="first N R2 rows only (smoke)")
    ap.add_argument("--frames-limit", type=int, help="first N frames and reference sentences only (smoke)")
    ap.add_argument("--arms", help="comma list of arms for R1/R3 (e.g. the told and pole frames only)")
    ap.add_argument("--pilot", action="store_true",
                    help="the design's pilot: paraphrases 1 and 2 (R1/R3 frames and R2 rows), R2 samples 4 and 2")
    ap.add_argument("--models-json", default=str(HERE / "models.json"))
    a = ap.parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if a.tokens:
        out = make_tokens_json()
        print(json.dumps(out["single_in_both"]), "laying_in_LIE:", out["laying_in_LIE"])
        return
    if not a.stage:
        ap.error("--stage is required")
    cfg = load_config()
    st = stage_spec(cfg, a.stage)
    if a.r2 and not st["chat"]:
        raise SystemExit(f"--r2 needs a chat stage; {a.stage} is not one")
    tokens_path = HERE / "tokens.json"
    if not tokens_path.exists():
        raise SystemExit("part209/tokens.json is missing; run --tokens first")
    tokens_json = load_json(tokens_path)
    prompts = load_prompts()
    frames = prompts["frames"]
    refs = load_reference()["sentences"]
    if a.pilot:
        pc = cfg["pilot"]
        a.paraphrases = a.paraphrases or ",".join(str(k) for k in pc["paraphrases"])
        a.samples = pc["samples_main"] if a.samples is None else a.samples
        a.told_samples = pc["samples_told"] if a.told_samples is None else a.told_samples
    paras = parse_paraphrases(a.paraphrases)
    if not a.r2:
        if paras:
            frames = [f for f in frames if f["k"] in paras]
        if a.arms:
            keep = {x.strip() for x in a.arms.split(",") if x.strip()}
            unknown = keep - {f["arm"] for f in frames}
            if unknown:
                raise SystemExit(f"unknown arms {sorted(unknown)}")
            frames = [f for f in frames if f["arm"] in keep]
    if a.frames_limit:
        frames, refs = frames[:a.frames_limit], refs[:a.frames_limit]
    tok, model, env = load_stage(cfg, a.stage, a.device)
    device = a.device or cfg["families"][st["family"]]["device"]
    write_models_json(a.models_json, env)
    print(f"[load] {a.stage}: {env.get('model_id')}@{str(env.get('revision'))[:8]} {env.get('dtype')} "
          f"{env.get('device')} {env.get('attn_implementation')}", flush=True)
    out = Path(a.out)
    if a.r2:
        n = run_r2(model, tok, env, frames, cfg, out / f"r2-{a.stage}.jsonl",
                   paraphrases=paras, samples_main=a.samples,
                   samples_told=a.told_samples, max_new_tokens=a.max_new_tokens, batch_size=a.batch,
                   device=device, limit=a.limit, log=lambda m: print(m, flush=True))
        print(f"[r2] wrote {n} answers to {out / f'r2-{a.stage}.jsonl'}", flush=True)
        return
    rec, states = run_r1r3(model, tok, env, frames, refs, cfg, tokens_json, device,
                           log=lambda m: print(m, flush=True))
    atomic_write_json(out / f"r1r3-{a.stage}.json", rec)
    import numpy as np
    fd, tmp = tempfile.mkstemp(prefix=f".states-{a.stage}.", suffix=".tmp", dir=str(out))
    with os.fdopen(fd, "wb") as f:
        np.savez_compressed(f, **states)
    os.replace(tmp, out / f"states-{a.stage}.npz")
    summarize(rec, print)


def summarize(rec, log):
    """Print what the gates read, for a person watching the run (score209.py decides). Only told and pole frames are
    printed, never another arm's numbers (Design: the build check reads poles only)."""
    told = [r for r in rec["r1"] if r["arm"] in ("TOLD-SIT", "TOLD-STAND", "TOLD-LIE")]
    if told:
        p3 = sum(r["P3"] for r in told) / len(told)
        log(f"[summary] {rec['stage']}: mean P3 over {len(told)} ROOM-W told frames = {p3:.3f}")
    want = {"TOLD-SIT": "SIT", "TOLD-STAND": "STAND", "TOLD-LIE": "LIE", "NOBODY-TOLD-SIT": "SIT",
            "NOBODY-TOLD-STAND": "STAND"}
    for arm, key in want.items():
        rows = [r for r in rec["r1"] if r["arm"] == arm]
        if not rows:
            continue
        wins = sum(1 for r in rows if max(("SIT", "STAND", "LIE"), key=lambda q: r[q]) == key)
        log(f"[summary] {arm}: told posture largest in {wins}/{len(rows)}; top tokens of #{rows[0]['k']}: "
            + ", ".join(f"{t!r} {p:.3f}" for t, p in rows[0]["top10"][:5]))
    for name in ("sit_stand", "sit_lie"):
        b = rec["r3"][name]
        log(f"[summary] R3 {name}: layer {b['chosen_layer']} held-out acc {b['chosen_accuracy']:.3f} "
            f"(null mean {b['null']['mean']:.3f}, p95 {b['null']['p95']:.3f})")
    pk = {}
    for r in rec["r3"]["sit_stand"]["frames"]:
        pk.setdefault(r["arm"], {})[r["k"]] = r["proj"]
    for one, zero in (("TOLD-SIT", "TOLD-STAND"), ("NOBODY-TOLD-SIT", "NOBODY-TOLD-STAND")):
        if one in pk and zero in pk:
            ks = sorted(set(pk[one]) & set(pk[zero]))
            above = sum(1 for k in ks if pk[one][k] > pk[zero][k])
            log(f"[summary] R3 poles {one} above {zero} in {above}/{len(ks)} paraphrases")


if __name__ == "__main__":
    main()
