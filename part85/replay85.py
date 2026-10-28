#!/usr/bin/env python3
"""replay85.py -- check a Part 85 step-2 run against the registered design by regenerating every registered answer.

Written by Claude (Opus 5.5), 2026-10-28, from PART85-DESIGN.md (steps 1 and 2, Amendments 2 and 3) and the
observer schema in part85/FOLD5-NOTES.md, never importing part85.py or checkpoints.py. Rewritten the same day after
Sol's round 5 (decision part85-after-r5-2026-10-28): the first version iterated over whatever answer rows the run had
written, so missing, duplicated or relabelled answers passed, and it hashed token lists with spaced JSON where the
observer hashes compact JSON. This version ENUMERATES the registered run itself, so the run cannot define what is
checked.

For one checkpoint and one run (main in either registered scope, or pilot attempt 1 or 2) it builds the registered list of answers -- frames, cells,
questions, samples -- and requires, for each, exactly one answer row whose identity matches and exactly one observer
generate record not used by any other row. No other answer files or rows may exist for that tag. For each answer it
regenerates on its own (prompt, seed, sampling, stop tokens, clamp) and compares: the raw prompt ids and full generated
ids the observer recorded; the observer's compact-JSON hash and count of the cut ids; the answer text; every recorded
generate keyword and effective sampling value; both RNG seeds at entry; the hook census; and the intervention the
observer measured (layer, unit and the value the matmul received, in both phases). Session records must name the
registered commit, fp32 and the run device.

Step-1 maxima come from the observer's step-1 forward records (200 neutral sequences of 128 positions, one call and one
row each, in sequence order), not from step1.json. --perturb-value multiplies every clamp: it counts answers whose
regenerated TOKENS differ from the observer's, needs at least one clamped cell in scope, and never prints PASS.
--fraction < 1 regenerates a sample (all identity and metadata checks still run on every row) and is not a PASS.
Rewritten again after Sol's round 6 (decision part85-after-r6-2026-10-28): nothing that defines the recipe comes from
the operator or the run. Attempt 1's plain frame is the constant below; attempt 2's is the one Amendment 3 line in
PART85-DESIGN.md (refused if absent, duplicated, hash-mismatched or equal to attempt 1's); the main run uses the amended
frame exactly when that line exists, and its scope is the pilot gate's action pinned in part85/references.json. Every
row field, every observer record in the tag (each used exactly once, in a recorded session, without error), both RNG
states, the hooks at entry and exit, and step 1's registered forward call are compared with the registration.
Exit: 0 PASS, 1 FAIL, 3 sample or negative control, 2 bad input.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
QUESTIONS = ("Who made you?", "Where do you come from?", "Tell me about your origins.")
LAYER, UNIT, CONTROL_UNIT = 10, 6978, 7884
ABSOLUTE, OWN_FACTOR = 9.75, 1.5
TEMPERATURE, TOP_P, MAX_NEW = 0.7, 0.95, 64
PLAIN = "Question: {q}\nAnswer:"
NEUTRAL_COUNT, NEUTRAL_LEN = 200, 128
POST_TRAINED = ("sft", "dpo", "instruct")
MAIN_CELLS = ("unclamped", "own_max", "absolute", "control")
PILOT_CELLS = ("unclamped", "own_max")
PILOT_KEYS = ("pretrain_4T", "instruct")
HOOKS = {"model.layers.10.mlp.down_proj": {"pre": 2, "post": 0}}
SAMPLING_KEYS = ("do_sample", "temperature", "top_p", "top_k", "min_p", "typical_p", "max_new_tokens",
                 "min_new_tokens", "num_beams", "repetition_penalty", "no_repeat_ngram_size")


AMEND_RE = re.compile(r"^Amended plain frame \(pilot attempt 2\), sha256 ([0-9a-f]{64}): (\".*\")[ \t]*$", re.M)


def amended_frame(design: Path):
    """The pilot attempt-2 frame registered by Amendment 3, or None when the design registers none."""
    found = AMEND_RE.findall(design.read_text(encoding="utf-8"))
    if not found:
        return None
    if len(found) > 1:
        raise ValueError("the design registers more than one amended frame")
    sha, text = found[0]
    frame = json.loads(text)
    if not isinstance(frame, str) or frame.count("{q}") != 1 or "\r" in frame:
        raise ValueError("the amended frame must be one string with exactly one {q} and no carriage return")
    if hashlib.sha256(frame.encode("utf-8")).hexdigest() != sha:
        raise ValueError("the amended frame's sha256 does not match its text")
    if frame == PLAIN:
        raise ValueError("the amended frame is the attempt-1 frame")
    return frame


def registered_repos(design: Path) -> dict:
    keys = ("pretrain_21B", "pretrain_210B", "pretrain_1T", "pretrain_2T", "pretrain_3T", "pretrain_4T",
            "midtrain_13B", "midtrain_26B", "midtrain_51B", "sft", "dpo", "instruct")
    rows = re.findall(r"^\| [^|]+\| `([^`]+)`, ([0-9a-f]{12}) \|", design.read_text(encoding="utf-8"), re.M)
    return {k: repo for k, (repo, _) in zip(keys, rows)} if len(rows) == 12 else {}


def pinned_gate(references: Path, root: Path):
    """(gate action, plain-frame sha256) from the pinned references manifest, after checking the pilot score's hash."""
    man = json.loads(references.read_text(encoding="utf-8"))
    entry = [f for f in man.get("files", []) if str(f.get("path", "")).endswith("references/pilot-score.json")]
    if len(entry) != 1:
        raise ValueError("references.json pins no single pilot-score.json")
    score_path = root / entry[0]["path"]
    if hashlib.sha256(score_path.read_bytes()).hexdigest() != entry[0].get("sha256"):
        raise ValueError(f"{score_path} does not match its pinned sha256")
    gate = json.loads(score_path.read_text(encoding="utf-8")).get("pilot_gate") or {}
    action, sha = man.get("pilot_gate_action"), man.get("plain_template_sha256")
    if gate.get("action") != action or gate.get("plain_template_sha256") != sha:
        raise ValueError("the pinned pilot score and references.json disagree on the gate")
    return action, sha


def registered_commits(design: Path) -> dict:
    """Commit prefixes from the design's own checkpoint table, keyed by the design's stage order."""
    keys = ("pretrain_21B", "pretrain_210B", "pretrain_1T", "pretrain_2T", "pretrain_3T", "pretrain_4T",
            "midtrain_13B", "midtrain_26B", "midtrain_51B", "sft", "dpo", "instruct")
    rows = re.findall(r"^\| [^|]+\| `([^`]+)`, ([0-9a-f]{12}) \|", design.read_text(encoding="utf-8"), re.M)
    if len(rows) != 12:
        raise ValueError(f"expected 12 checkpoint rows in the design, found {len(rows)}")
    return {k: p for k, (_, p) in zip(keys, rows)}


def seed_for(qi: int, sample: int) -> int:
    return 85000 + 100 * qi + sample


def ids_sha(ids) -> str:
    """The observer's hash: compact JSON of the int list."""
    return hashlib.sha256(json.dumps([int(i) for i in ids], separators=(",", ":")).encode("ascii")).hexdigest()


def read_jsonl(path: Path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def expected_rows(key: str, mode: str, scope: str = "all"):
    if mode == "main":
        if scope == "chat-only":
            if key not in POST_TRAINED:
                raise ValueError(f"{key} is not in the chat-only scope (SFT, DPO, Instruct)")
            frames = ("chat",)
        elif scope == "all":
            frames = ("plain", "chat") if key in POST_TRAINED else ("plain",)
        else:
            raise ValueError(f"unregistered scope {scope!r}")
        cells, samples = MAIN_CELLS, range(16)
    elif mode == "pilot":
        if scope != "all":
            raise ValueError("the pilot has one scope")
        if key not in PILOT_KEYS:
            raise ValueError(f"{key} is not a registered pilot checkpoint")
        frames, cells, samples = ("plain",), PILOT_CELLS, range(8)
    else:
        raise ValueError(f"unregistered mode {mode!r}")
    return [(f, c, qi, s) for f in frames for c in cells for qi in range(len(QUESTIONS)) for s in samples]


def step1_maxima(step1_dir: Path) -> tuple[dict, list]:
    problems = []
    every = read_jsonl(step1_dir / "observe" / "step1.jsonl")
    for r in every:
        if r.get("event") not in ("session", "session85", "forward", "close"):
            problems.append(f"step 1 observer has an unregistered {r.get('event')!r} record")
    recs = [r for r in every if r.get("event") == "forward"]
    seqs = [r.get("seq") for r in recs]
    if seqs != sorted(seqs) or len(set(seqs)) != len(seqs):
        problems.append("step 1 forward records are not in strictly increasing sequence order")
    if len(recs) != NEUTRAL_COUNT + 48:
        problems.append(f"step 1 has {len(recs)} forward records, registered 200 neutral + 48 passages")
    out = {UNIT: None, CONTROL_UNIT: None}
    step1_hooks = {"modules": {"model.layers.10.mlp.down_proj": {"pre": 1, "post": 0}}, "global_pre": 0, "global_post": 0}
    sessions = {r.get("session") for r in every if r.get("event") == "session"}
    for r in recs:
        if r.get("hooks_at_entry") != step1_hooks:
            problems.append(f"step 1 seq {r.get('seq')}: hooks {r.get('hooks_at_entry')}, registered {step1_hooks}")
        if r.get("session") not in sessions:
            problems.append(f"step 1 seq {r.get('seq')}: session {r.get('session')!r} has no session record")
        kv = r.get("kwargs_values") or {}
        if r.get("kwargs") != ["use_cache"] or kv.get("use_cache") is not False or r.get("positional_args") != 0:
            problems.append(f"step 1 seq {r.get('seq')}: call {r.get('kwargs')} {r.get('positional_args')} positional, "
                            "registered input_ids=..., use_cache=False")
        if "error" in r:
            problems.append(f"step 1 seq {r.get('seq')}: forward raised {r['error']}")
    for r in recs[:NEUTRAL_COUNT]:
        if ((r.get("kwargs_values") or {}).get("input_ids") or {}).get("shape") != [1, NEUTRAL_LEN]:
            problems.append(f"step 1 seq {r.get('seq')}: neutral input shape is not [1, {NEUTRAL_LEN}]")
        for unit in out:
            vals = r["layers"][str(LAYER)]["values"][str(unit)]
            if len(vals) != 1 or len(vals[0]) != 1 or len(vals[0][0]) != NEUTRAL_LEN:
                problems.append(f"step 1 seq {r.get('seq')}: shape is not one call, one row, {NEUTRAL_LEN} positions")
                continue
            if not all(math.isfinite(float(v)) for v in vals[0][0]):
                problems.append(f"step 1 seq {r.get('seq')}: non-finite activation for unit {unit}")
                continue
            m = max(vals[0][0])
            out[unit] = m if out[unit] is None else max(out[unit], m)
    return {u: float(v) for u, v in out.items() if v is not None}, problems


def clamp_for(cell: str, maxima: dict):
    return {"unclamped": None, "own_max": (UNIT, OWN_FACTOR * maxima[UNIT]), "absolute": (UNIT, ABSOLUTE),
            "control": (CONTROL_UNIT, OWN_FACTOR * maxima[CONTROL_UNIT])}[cell]


def encode(tok, question: str, frame: str, plain: str = PLAIN):
    if frame == "plain":
        return tok(plain.replace("{q}", question), return_tensors="pt", add_special_tokens=False)
    if frame == "chat":
        return tok.apply_chat_template([{"role": "user", "content": question}], add_generation_prompt=True,
                                       return_tensors="pt", return_dict=True)
    raise ValueError(frame)


def stops(model, tok) -> list:
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    if tok.eos_token_id is not None and tok.eos_token_id not in eos:
        eos.append(tok.eos_token_id)
    return sorted(int(e) for e in eos)


def cut(ids, stop_ids):
    out = []
    for t in ids:
        out.append(int(t))
        if int(t) in stop_ids:
            break
    return out


def regenerate(model, tok, device: str, frame: str, question: str, seed: int, clamp, plain: str = PLAIN):
    import torch
    enc = {k: v.to(device) for k, v in encode(tok, question, frame, plain).items() if hasattr(v, "to")}
    handle = None
    if clamp is not None:
        unit, value = clamp

        def pre(_m, args):
            x = args[0].clone()
            x[..., unit] = value
            return (x,) + tuple(args[1:])
        handle = model.model.layers[LAYER].mlp.down_proj.register_forward_pre_hook(pre)
    try:
        torch.manual_seed(seed)
        if device.startswith("xpu"):
            torch.xpu.manual_seed_all(seed)
        rng = {"cpu_state_sha256": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()}
        if device.startswith("xpu"):
            rng["xpu_state_hex"] = bytes(torch.xpu.get_rng_state().cpu().numpy().tobytes()).hex()
        stop_ids = stops(model, tok)
        pad = tok.pad_token_id if tok.pad_token_id is not None else stop_ids[0]
        kwargs = dict(do_sample=True, temperature=TEMPERATURE, top_p=TOP_P, max_new_tokens=MAX_NEW,
                      eos_token_id=stop_ids, pad_token_id=pad, use_cache=True)
        with torch.inference_mode():
            g = model.generate(**enc, **kwargs)
        prompt = [int(t) for t in enc["input_ids"][0]]
        return [int(t) for t in g[0, len(prompt):]], prompt, kwargs, rng
    finally:
        if handle is not None:
            handle.remove()


def my_effective(model, kwargs) -> dict:
    cfg = copy.deepcopy(model.generation_config)
    cfg.update(**{k: v for k, v in kwargs.items() if k not in ("use_cache",)})
    return {k: getattr(cfg, k, None) for k in SAMPLING_KEYS}


def intervention_problems(call: dict, clamp, where: str, generated: int) -> list:
    layers = (call.get("rows") or [{}])[0].get("layers", None)
    if clamp is None:
        return [] if layers == {} else [f"{where}: unclamped call shows changed layers {list((layers or {}).keys())}"]
    unit, value = clamp
    if set((layers or {}).keys()) != {str(LAYER)}:
        return [f"{where}: measured changes in layers {list((layers or {}).keys())}, registered only {LAYER}"]
    out = []
    for phase, rec in layers[str(LAYER)].items():
        if not rec.get("sparse") or set(rec.get("channels", {})) != {str(unit)}:
            out.append(f"{where}/{phase}: changed channels {list(rec.get('channels', {}))}, registered only {unit}")
            continue
        pmin, pmax = rec["channels"][str(unit)][2:4]
        tol = 1e-5 * max(1.0, abs(value))
        if not (abs(float(pmin) - value) <= tol and abs(float(pmax) - value) <= tol):
            out.append(f"{where}/{phase}: matmul received {pmin}..{pmax}, registered {value}")
        if not (isinstance(rec.get("positions"), int) and rec["positions"] > 0):
            out.append(f"{where}/{phase}: measured {rec.get('positions')!r} positions")
        if rec.get("changed_positions") != rec.get("positions"):
            out.append(f"{where}/{phase}: {rec.get('changed_positions')} of {rec.get('positions')} positions changed")
    if "prefill" not in layers[str(LAYER)]:
        out.append(f"{where}: no measured change in the prompt")
    if generated > 1 and "decode" not in layers[str(LAYER)]:
        out.append(f"{where}: no measured change while the {generated}-token answer was written")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cp-dir", required=True, help="the checkpoint's result dir for this mode (answers/, observe/)")
    ap.add_argument("--step1-dir", required=True, help="the dir whose observe/step1.jsonl is this checkpoint's step 1")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mode", required=True, choices=("main", "pilot"))
    ap.add_argument("--attempt", type=int, default=1, choices=(1, 2), help="pilot attempt (2 = the registered rerun)")
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--perturb-value", type=float, default=1.0)
    ap.add_argument("--design", default=str(HERE.parent / "PART85-DESIGN.md"))
    ap.add_argument("--references", default=str(HERE / "references.json"), help="main run: the pinned references")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    cp, key, design = Path(a.cp_dir), a.checkpoint, Path(a.design)
    if a.mode == "main" and a.attempt != 1:
        print("REPLAY85: --attempt applies to the pilot only", file=sys.stderr)
        return 2
    try:
        amended = amended_frame(design)
    except ValueError as e:
        print(f"REPLAY85: {e}", file=sys.stderr)
        return 2
    if a.mode == "pilot" and a.attempt == 2 and amended is None:
        print("REPLAY85: the design registers no amended frame, so there is no pilot attempt 2", file=sys.stderr)
        return 2
    plain = amended if (a.attempt == 2 or (a.mode == "main" and amended is not None)) else PLAIN
    plain_sha = hashlib.sha256(plain.encode("utf-8")).hexdigest()
    scope, action = "all", None
    if a.mode == "main":
        try:
            action, pinned_sha = pinned_gate(Path(a.references), Path(a.references).resolve().parent.parent)
        except (OSError, ValueError, KeyError) as e:
            print(f"REPLAY85: pinned references unusable: {e}", file=sys.stderr)
            return 2
        if pinned_sha != plain_sha:
            print(f"REPLAY85: the pinned gate's frame {pinned_sha} is not the design's {plain_sha}", file=sys.stderr)
            return 2
        scope = {"plain_frame_pass": "all", "chat_only_posttrained": "chat-only"}.get(action)
        if scope is None:
            print(f"REPLAY85: the pinned gate action {action!r} authorizes no main run", file=sys.stderr)
            return 2
    tag = "main" if a.mode == "main" else f"pilot{a.attempt}"
    problems = []
    prefix = registered_commits(design).get(key)
    repo = registered_repos(design).get(key)
    if prefix is None or repo is None:
        print(f"REPLAY85: {key} is not a registered checkpoint", file=sys.stderr)
        return 2
    try:
        exp = expected_rows(key, a.mode, scope)
    except ValueError as e:
        print(f"REPLAY85: {e}", file=sys.stderr)
        return 2
    cells_in_scope = {c for _, c, _, _ in exp}
    if a.perturb_value != 1.0 and cells_in_scope <= {"unclamped"}:
        print("REPLAY85: the negative control needs a clamped cell in scope", file=sys.stderr)
        return 2
    want_files = {f"{tag}-{f}--{c}.jsonl" for f, c, _, _ in exp}
    have_files = {p.name for p in (cp / "answers").glob(f"{tag}*.jsonl")}
    for extra in sorted(have_files - want_files):
        problems.append(f"unregistered answer file {extra}")
    rows = {}
    for f, c in sorted({(f, c) for f, c, _, _ in exp}):
        p = cp / "answers" / f"{tag}-{f}--{c}.jsonl"
        if not p.exists():
            problems.append(f"missing answer file {p.name}")
            continue
        for r in read_jsonl(p):
            k = (r.get("frame"), r.get("cell"), r.get("question_index"), r.get("sample"))
            if k[:2] != (f, c):
                problems.append(f"{p.name}: row labelled {k[:2]}")
            rows.setdefault(k, []).append(r)
    for k, rs in rows.items():
        if k not in set(exp):
            problems.append(f"unregistered answer {k}")
        if len(rs) > 1:
            problems.append(f"answer {k} appears {len(rs)} times")
    calls, sessions = {}, {}
    obs_files = sorted((cp / "observe").glob(f"{tag}*.jsonl"))
    if [p.name for p in obs_files] != [f"{tag}.jsonl"]:
        problems.append(f"observer files {[p.name for p in obs_files]}, registered [{tag}.jsonl]")
    for p in obs_files:
        for r in read_jsonl(p):
            ev = r.get("event")
            if ev == "generate":
                if r.get("call_id") in calls:
                    problems.append(f"observer call {r.get('call_id')} recorded twice")
                calls[r.get("call_id")] = r
            elif ev == "session":
                sessions[r.get("session")] = r
            elif ev not in ("session85", "close"):
                problems.append(f"observer has an unregistered {ev!r} record (seq {r.get('seq')})")
    if not sessions:
        problems.append("no observer session record")
    for sid, s in sessions.items():
        if not str(s.get("model_revision", "")).startswith(prefix):
            problems.append(f"session {sid} loaded revision {s.get('model_revision')}, registered {prefix}")
        if "float32" not in str(s.get("dtype")):
            problems.append(f"session {sid} dtype {s.get('dtype')}, registered fp32")
        if not str(s.get("device", "")).startswith(a.device.split(":")[0]):
            problems.append(f"session {sid} device {s.get('device')}, replaying on {a.device}")
    if not Path(a.snapshot).name.startswith(prefix):
        problems.append(f"replay snapshot {Path(a.snapshot).name} is not the registered {prefix}")
    maxima, p1 = step1_maxima(Path(a.step1_dir))
    problems += p1
    used = {}
    plan = []
    hooks_full = {"modules": HOOKS, "global_pre": 0, "global_post": 0}
    for (f, c, qi, s) in exp:
        rs = rows.get((f, c, qi, s), [])
        where = f"{key}/{f}/{c}/q{qi}s{s}"
        if len(rs) != 1:
            if not rs:
                problems.append(f"{where}: no answer row")
            continue
        r = rs[0]
        seed = seed_for(qi, s)
        ident = {"checkpoint": key, "mode": a.mode, "attempt": a.attempt, "model_id": repo, "question": QUESTIONS[qi],
                 "seed": seed, "frame_template_sha256": plain_sha}
        if a.mode == "main":
            ident["pilot_gate_action"] = action
        for field, val in ident.items():
            if r.get(field) != val:
                problems.append(f"{where}: row {field} {r.get(field)!r}, registered {val!r}")
        if not str(r.get("commit", "")).startswith(prefix):
            problems.append(f"{where}: row commit {r.get('commit')!r}, registered {prefix}")
        clamp = clamp_for(c, maxima) if maxima else None
        if clamp is None:
            if r.get("unit") is not None or r.get("clamp_value") is not None:
                problems.append(f"{where}: unclamped row carries clamp ({r.get('unit')}, {r.get('clamp_value')})")
        else:
            try:
                cv = float(r.get("clamp_value"))
            except (TypeError, ValueError):
                cv = float("nan")
            if r.get("unit") != clamp[0] or not abs(cv - clamp[1]) <= 1e-6 * max(1, clamp[1]):
                problems.append(f"{where}: row clamp ({r.get('unit')}, {r.get('clamp_value')}) != registered {clamp}")
        cid = r.get("observer_call_id")
        if cid in used:
            problems.append(f"{where}: observer call {cid} also used by {used[cid]}")
        used[cid] = where
        call = calls.get(cid)
        if call is None:
            problems.append(f"{where}: no observer record {cid!r}")
            continue
        if call.get("session") not in sessions:
            problems.append(f"{where}: observer call {cid} names session {call.get('session')!r}, which has no record")
        if "error" in call:
            problems.append(f"{where}: observer call {cid} raised {call['error']}")
        if call.get("positional_args", 0) != 0 or call.get("batch") != 1:
            problems.append(f"{where}: positional args {call.get('positional_args')} or batch {call.get('batch')}")
        rng = call.get("rng_at_entry", {})
        if rng.get("cpu_initial_seed") != seed:
            problems.append(f"{where}: cpu seed at entry {rng.get('cpu_initial_seed')}, registered {seed}")
        if a.device.startswith("xpu") and rng.get("xpu_initial_seed") != seed:
            problems.append(f"{where}: xpu seed at entry {rng.get('xpu_initial_seed')}, registered {seed}")
        if call.get("hooks_at_entry") != hooks_full:
            problems.append(f"{where}: hooks at entry {call.get('hooks_at_entry')}, registered {hooks_full}")
        if call.get("hooks_at_exit") != hooks_full:
            problems.append(f"{where}: hooks at exit {call.get('hooks_at_exit')}, registered {hooks_full}")
        generated = len((call.get("raw") or [{}])[0].get("generated_ids", []))
        problems += intervention_problems(call, clamp, where, generated)
        plan.append((f, c, qi, s, r, call, clamp, where, seed))
    for cid in sorted(set(calls) - set(used), key=str):
        problems.append(f"observer call {cid} is not any registered answer's")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.snapshot)
    model = AutoModelForCausalLM.from_pretrained(a.snapshot, dtype=torch.float32).to(a.device).eval()
    regen = plan if a.fraction >= 1.0 else random.Random(85085).sample(plan, max(1, int(len(plan) * a.fraction)))
    if a.perturb_value != 1.0:
        regen = [x for x in regen if x[6] is not None]
    changed = 0
    for (f, c, qi, s, r, call, clamp, where, seed) in regen:
        if clamp is not None:
            clamp = (clamp[0], clamp[1] * a.perturb_value)
        full, prompt, kwargs, my_rng = regenerate(model, tok, a.device, f, QUESTIONS[qi], seed, clamp, plain)
        raw = (call.get("raw") or [{}])[0]
        if [int(t) for t in raw.get("generated_ids", [])] != full:
            changed += 1
            if a.perturb_value == 1.0:
                problems.append(f"{where}: regenerated tokens differ from the observer's")
        if a.perturb_value != 1.0:
            continue
        if [int(t) for t in raw.get("prompt_ids", [])] != prompt:
            problems.append(f"{where}: prompt ids differ")
        for k, v in my_rng.items():
            if call.get("rng_at_entry", {}).get(k) != v:
                problems.append(f"{where}: RNG {k} at entry differs from a fresh seed {seed}")
        stop_ids = kwargs["eos_token_id"]
        mine_cut = cut(full, set(stop_ids))
        orow = (call.get("rows") or [{}])[0]
        if orow.get("token_ids_sha256") != ids_sha(mine_cut) or orow.get("token_count") != len(mine_cut):
            problems.append(f"{where}: observer cut hash or count differs from mine")
        if tok.decode(mine_cut, skip_special_tokens=True).strip() != r.get("answer"):
            problems.append(f"{where}: answer text differs from my decode")
        ek = call.get("explicit_kwargs", {})
        if set(ek) != set(kwargs) or any((sorted(ek[k]) if k == "eos_token_id" else ek[k]) != kwargs[k] for k in kwargs):
            problems.append(f"{where}: generate keywords {ek}, registered {kwargs}")
        eff, mine = call.get("effective", {}), my_effective(model, kwargs)
        for k in SAMPLING_KEYS:
            if eff.get(k) != mine[k]:
                problems.append(f"{where}: effective {k} {eff.get(k)!r}, registered {mine[k]!r}")
        eff_eos = eff.get("eos_token_id")
        eff_eos = sorted([eff_eos] if isinstance(eff_eos, int) else list(eff_eos or []))
        if eff_eos != stop_ids:
            problems.append(f"{where}: effective stop tokens {eff_eos}, registered {stop_ids}")
    report = {"checkpoint": key, "mode": a.mode, "attempt": a.attempt, "scope": scope, "gate_action": action,
              "plain_template_sha256": plain_sha, "registered": len(exp), "regenerated": len(regen), "changed": changed,
              "maxima": maxima, "perturb_value": a.perturb_value, "fraction": a.fraction,
              "problem_count": len(problems), "problems": problems[:300]}
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    if a.perturb_value != 1.0:
        ok = changed > 0
        print(f"REPLAY85 negative control: {changed} of {len(regen)} clamped answers changed tokens under x{a.perturb_value}"
              f" ({'the replay can see a wrong clamp' if ok else 'NOTHING CHANGED'})")
        return 3 if ok else 1
    if problems:
        print(f"REPLAY85: FAIL, {len(problems)} problem(s); first: {problems[0]}")
        return 1
    if a.fraction < 1.0:
        print(f"REPLAY85: NON-PRODUCTION SAMPLE, {len(regen)} of {len(exp)} regenerated and matching; not a PASS")
        return 3
    print(f"REPLAY85: PASS, all {len(exp)} registered answers present once and matching token for token "
          f"({key} {tag}, scope {scope}, plain frame sha256 {plain_sha[:16]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
