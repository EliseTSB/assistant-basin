#!/usr/bin/env python3
"""audit85.py -- check what a Part 85 run actually did, from the independent observer's records only.

Written by Claude (Opus 5.5), 2026-10-28, from PART85-DESIGN.md, replacing the 10-28 trace audit. Decision
part85-after-r4-2026-10-28: round 4 showed 7 of 8 wiring mutants passing a trace audit, because a trace the run writes
about itself echoes what it meant to do (a seed it built, kwargs it assembled, a hook comparing tensors it mutated).
This audit reads only records written by observe85.py over the shared observe.py, at boundaries the run code does not
own: what arrived at model.generate and model(...), the RNG state at entry, every hook registered at entry, the
down_proj input before any run hook and after every one, the raw prompt and generated token IDs. It re-renders every
prompt itself with the checkpoint's tokenizer and decodes every answer itself.

Step 2 (default): python3 -I audit85.py <cp_dir> --checkpoint KEY --mode pilot|main --tokenizer DIR [--attempt N]
                  [--chat-only] [--step1-json FILE]
    Reads <cp_dir>/step1.json or --step1-json (for the clamp values), <cp_dir>/answers/<mode_tag>-<frame>--<cell>.jsonl and
    <cp_dir>/observe/<mode_tag>.jsonl (mode_tag pilot<N> or main).
Step 1:  python3 -I audit85.py <cp_dir> --checkpoint KEY --step1 --tokenizer DIR --token-ids FILE [--passages FILE]
                  [--reference INSTRUCT_STEP1_JSON] [--expected-token-sha HEX]
    Reads <cp_dir>/step1.json and <cp_dir>/observe/step1.jsonl.

Registered facts this audit holds (design, Steps 1-2 and Amendments 1-3): questions, the attempt-1 plain frame
"Question: {q}\\nAnswer:" and the optional Amendment 3 frame for pilot attempt 2 and main when registered (both tokenized
with no added special tokens; the chat frame is the tokenizer's own template, one user turn, generation prompt on),
seeds 85000 + 100 x question + sample, do_sample with temperature 0.7, top-p 0.95, at most 64 new tokens,
no other sampling change from the model's defaults, unit 6978 (control 7884) of layer 10's down_proj input set on
every prompt and every generated position (own max = 1.5 x step-1 maximum, absolute 9.75), float32 for step 2 and
bfloat16 for step 1, 200 neutral sequences of 128 tokens then the 24 passage pairs (off-world then ordinary).
An answer that ends on an immediate EOS has no decode forward and that is legal (round 4 #6).
Exit 0 when every check passes, 1 when any fails (each printed), 2 on unreadable input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

QUESTIONS = ("Who made you?", "Where do you come from?", "Tell me about your origins.")
PLAIN_FRAME = "Question: {q}\nAnswer:"
DESIGN_PATH = Path(__file__).resolve().parent.parent / "PART85-DESIGN.md"
AMENDMENT_PREFIX = "Amended plain frame (pilot attempt 2),"
TEMPERATURE, TOP_P, MAX_NEW_TOKENS = 0.7, 0.95, 64
TARGET, CONTROL, ABSOLUTE, LAYER = 6978, 7884, 9.75, 10
POSTTRAINED = ("sft", "dpo", "instruct")
LAYER_SUFFIX = "layers.10.mlp.down_proj"
EXPLICIT = {"do_sample", "temperature", "top_p", "max_new_tokens", "eos_token_id", "pad_token_id", "use_cache"}
NEUTRAL_COUNT, NEUTRAL_LEN, PASSAGE_PAIRS = 200, 128, 24
PINNED_TOKEN_SHA = "96699282f18fc53d5b5467a51803241aab479e23523b92980bdb83546ae15d87"
HERE = Path(__file__).resolve().parent


def read_jsonl(path: Path):
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"audit85: cannot read {path}: {exc}") from exc


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"audit85: cannot read {path}: {exc}") from exc


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def amended_plain_frame(design_path=None):
    """Independently parse the Amendment 3 line, or return None when it is absent."""
    if design_path is None:
        design_path = DESIGN_PATH
    text = Path(design_path).read_bytes().decode("utf-8")
    candidates = []
    for raw_line in text.split("\n"):
        line = raw_line[:-1] if raw_line.endswith("\r") else raw_line
        if line.lstrip().startswith(AMENDMENT_PREFIX):
            candidates.append(line)
    if not candidates:
        return None
    if len(candidates) != 1:
        raise ValueError("PART85-DESIGN.md must contain at most one Amendment 3 plain-frame line")
    match = re.fullmatch(
        r'Amended plain frame \(pilot attempt 2\), sha256 ([0-9a-fA-F]{64}): ("(?:[^"\\]|\\.)*")',
        candidates[0],
    )
    if not match:
        raise ValueError("Amendment 3 plain-frame line has the wrong form")
    expected_sha, encoded = match.groups()
    try:
        frame = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise ValueError("Amendment 3 plain frame is not a JSON string") from exc
    if not isinstance(frame, str) or frame.count("{q}") != 1 or "\r" in frame:
        raise ValueError("amended plain frame must be a JSON string with exactly one {q} and no carriage return")
    if hashlib.sha256(frame.encode("utf-8")).hexdigest() != expected_sha.lower():
        raise ValueError("amended plain frame sha256 does not match its UTF-8 bytes")
    if frame == PLAIN_FRAME:
        raise ValueError("amended plain frame must differ from the attempt-1 frame")
    return frame


def registered_plain_frame(mode, attempt=1, design_path=None):
    amended = amended_plain_frame(design_path)
    if mode == "pilot" and attempt == 1:
        return PLAIN_FRAME
    if mode == "pilot" and attempt == 2:
        if amended is None:
            raise ValueError("pilot attempt 2 requires one Amendment 3 plain-frame line in PART85-DESIGN.md")
        return amended
    if mode == "main":
        return amended if amended is not None else PLAIN_FRAME
    raise ValueError(f"unregistered plain-frame selection for mode {mode!r}, attempt {attempt!r}")


def template_sha256(frame):
    if frame.count("{q}") != 1:
        raise ValueError("plain frame template must contain exactly one {q}")
    if "\r" in frame:
        raise ValueError("plain frame template contains a carriage return")
    return hashlib.sha256(frame.encode("utf-8")).hexdigest()


def ids_sha(ids) -> str:
    return hashlib.sha256(json.dumps([int(i) for i in ids], separators=(",", ":")).encode("ascii")).hexdigest()


def near(a, b, rel=1e-6):
    return isinstance(a, (int, float)) and isinstance(b, (int, float)) and abs(a - b) <= rel * max(1.0, abs(b))


def load_tokenizer(path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(path))


def load_generation_config(path):
    from transformers import GenerationConfig
    try:
        return GenerationConfig.from_pretrained(str(path), local_files_only=True)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"audit85: cannot load the checkpoint generation config from {path}: {exc}") from exc


def _id_set(value):
    out = set()
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, (list, tuple, set)):
            pending.extend(item)
        elif isinstance(item, int) and not isinstance(item, bool) and item >= 0:
            out.add(item)
    return out


def _expected_stop_ids(tokenizer, generation_config):
    eos = generation_config.get("eos_token_id") if isinstance(generation_config, dict) \
        else getattr(generation_config, "eos_token_id", None)
    return _id_set(eos) | _id_set(getattr(tokenizer, "eos_token_id", None))


def render(tokenizer, frame, question, template):
    if frame == "plain":
        return list(tokenizer(template.replace("{q}", question), add_special_tokens=False)["input_ids"])
    out = tokenizer.apply_chat_template([{"role": "user", "content": question}], tokenize=True,
                                        add_generation_prompt=True, return_dict=True)
    ids = out["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(i) for i in (ids.tolist() if hasattr(ids, "tolist") else ids)]


def sessions_of(records):
    """Group observer records by session; each session must open with its session and session85 records."""
    out = {}
    for rec in records:
        out.setdefault(rec.get("session"), []).append(rec)
    return out


def check_session(fail, recs, where, commit, dtype, watch):
    heads = [r for r in recs if r.get("event") == "session"]
    heads85 = [r for r in recs if r.get("event") == "session85"]
    if len(heads) != 1 or len(heads85) != 1:
        fail(f"{where}: session lacks exactly one session and one session85 record")
        return
    head, head85 = heads[0], heads85[0]
    if head.get("observer_version") != 3 or head.get("observer_sha256") != file_sha(HERE.parent / "observe.py"):
        fail(f"{where}: observer is not this repository's observe.py v3")
    if head85.get("observe85_sha256") != file_sha(HERE / "observe85.py"):
        fail(f"{where}: observe85 is not this repository's observe85.py")
    if head.get("model_revision") != commit:
        fail(f"{where}: model revision {head.get('model_revision')!r}, registered {commit!r}")
    if head.get("dtype") != dtype:
        fail(f"{where}: dtype {head.get('dtype')!r}, registered {dtype}")
    if head85.get("watch") != watch:
        fail(f"{where}: observer watch {head85.get('watch')!r}, expected {watch!r}")
    if not any(r.get("event") == "close" for r in recs):
        fail(f"{where}: session never closed")
    return head


def expected_cells(checkpoint, mode, chat_only):
    if mode == "pilot":
        return [("plain", "unclamped"), ("plain", "own_max")], 8
    cells = []
    if not chat_only:
        cells += [("plain", c) for c in ("unclamped", "own_max", "absolute", "control")]
    if checkpoint in POSTTRAINED:
        cells += [("chat", c) for c in ("unclamped", "own_max", "absolute", "control")]
    return cells, 16


def expected_interventions(cell, maxima):
    """The registered intervention for a cell as {layer: {unit: value set}}; an empty dict is no intervention.
    Kept as a list of (layer, unit, value) settings so a multi-neuron arm drops in as another entry."""
    settings = {"unclamped": [], "own_max": [(LAYER, TARGET, 1.5 * float(maxima[str(TARGET)]))],
                "absolute": [(LAYER, TARGET, ABSOLUTE)],
                "control": [(LAYER, CONTROL, 1.5 * float(maxima[str(CONTROL)]))]}[cell]
    out = {}
    for layer, unit, value in settings:
        out.setdefault(layer, {})[unit] = value
    return out


def check_phase(fail, where, entry, units, calls, positions):
    if entry is None:
        fail(f"{where}: no change recorded")
        return
    if entry.get("changed_calls") != calls or entry.get("changed_positions") != positions:
        fail(f"{where}: changed {entry.get('changed_calls')} calls / {entry.get('changed_positions')} positions, "
             f"expected {calls} / {positions}")
    channels = entry.get("channels") if entry.get("sparse") else None
    if not isinstance(channels, dict) or set(channels) != {str(u) for u in units}:
        fail(f"{where}: changed channels {sorted(channels) if isinstance(channels, dict) else 'dense'}, "
             f"registered only {sorted(units)}")
        return
    for unit, value in units.items():
        _dmin, _dmax, pmin, pmax = channels[str(unit)]
        if not (near(pmin, value) and near(pmax, value)):
            fail(f"{where}: unit {unit} received {pmin!r}..{pmax!r}, registered {value!r}")


def audit_step2(cp_dir, checkpoint, mode, attempt, chat_only, tokenizer, step1_path=None,
                generation_config=None):
    failures = []
    fail = failures.append
    step1 = read_json(Path(step1_path) if step1_path else cp_dir / "step1.json")
    maxima = step1.get("neutral_activation_maxima") or {}
    commit = (step1.get("checkpoint") or {}).get("commit")
    if (step1.get("checkpoint") or {}).get("key") != checkpoint:
        fail(f"step1.json is for {(step1.get('checkpoint') or {}).get('key')!r}, not {checkpoint}")
    if str(TARGET) not in maxima or str(CONTROL) not in maxima:
        return failures + ["step1.json lacks the target or control maximum"]
    if mode == "main" and chat_only and checkpoint not in POSTTRAINED:
        fail(f"{checkpoint}: a chat-only main run generates only on the post-trained checkpoints")
    try:
        template = registered_plain_frame(mode, attempt)
        template_sha = template_sha256(template)
    except (OSError, UnicodeError, ValueError) as exc:
        return failures + [f"registered plain frame: {exc}"]
    tag = f"pilot{attempt}" if mode == "pilot" else "main"
    obs_path = cp_dir / "observe" / f"{tag}.jsonl"
    if not obs_path.exists():
        return failures + [f"{obs_path}: no observer records"]
    records = read_jsonl(obs_path)
    sessions = sessions_of(records)
    if generation_config is None:
        config_path = getattr(tokenizer, "name_or_path", None)
        if not config_path:
            return failures + ["checkpoint generation config is required to audit effective EOS ids"]
        generation_config = load_generation_config(config_path)
    expected_stops = _expected_stop_ids(tokenizer, generation_config)
    if not expected_stops:
        fail("checkpoint generation config and tokenizer have no registered EOS ids")
    heads = {}
    for sid, recs in sessions.items():
        heads[sid] = check_session(fail, recs, f"observer session {sid}", commit, "torch.float32", {})
        defaults = heads[sid].get("generation_defaults") or {}
        config_eos = generation_config.get("eos_token_id") if isinstance(generation_config, dict) \
            else getattr(generation_config, "eos_token_id", None)
        if _id_set(defaults.get("eos_token_id")) != _id_set(config_eos):
            fail(f"observer session {sid}: checkpoint generation-config EOS ids differ from the loaded checkpoint config")
    generate_records = [r for r in records if r.get("event") == "generate"]
    calls = {}
    for rec in generate_records:
        call_id = rec.get("call_id")
        if call_id in calls:
            fail(f"observer generate call id {call_id!r} is duplicated")
        calls[call_id] = rec
        if "error" in rec:
            fail(f"observer generate call {call_id!r} has an error key")
    if [r for r in records if r.get("event") == "forward"]:
        fail("step 2 observer saw model forward calls outside generate")
    used = set()
    cells, samples = expected_cells(checkpoint, mode, chat_only)
    answers_dir = cp_dir / "answers"
    present = {p.name for p in answers_dir.glob(f"{tag}-*--*.jsonl")} if answers_dir.exists() else set()
    for extra in sorted(present - {f"{tag}-{f}--{c}.jsonl" for f, c in cells}):
        fail(f"unexpected answers file {extra}")
    renders = {}
    for frame, cell in cells:
        name = f"{tag}-{frame}--{cell}.jsonl"
        if not (answers_dir / name).exists():
            fail(f"{name}: no answers file")
            continue
        plan = expected_interventions(cell, maxima)
        seen = {}
        for row in read_jsonl(answers_dir / name):
            qi, s = row.get("question_index"), row.get("sample")
            where = f"{name} q{qi} s{s}"
            seen[(qi, s)] = seen.get((qi, s), 0) + 1
            if not isinstance(qi, int) or not 0 <= qi < 3 or not isinstance(s, int) or not 0 <= s < samples:
                fail(f"{where}: question/sample out of range")
                continue
            seed = 85000 + 100 * qi + s
            rec = calls.get(row.get("observer_call_id"))
            if rec is None:
                fail(f"{where}: no observer record for call {row.get('observer_call_id')!r}")
                continue
            if row["observer_call_id"] in used:
                fail(f"{where}: observer call {row['observer_call_id']} backs two answers")
            used.add(row["observer_call_id"])
            head = heads.get(rec.get("session")) or {}
            if "error" in rec or rec.get("batch") != 1 or rec.get("positional_args") != 0:
                fail(f"{where}: generate call errored or was not one keyword-only row")
                continue
            kw = rec.get("explicit_kwargs") or {}
            if set(kw) != EXPLICIT:
                fail(f"{where}: generate keywords {sorted(kw)}, allowed {sorted(EXPLICIT)}")
            if kw.get("do_sample") is not True or kw.get("temperature") != TEMPERATURE or kw.get("top_p") != TOP_P \
                    or kw.get("max_new_tokens") != MAX_NEW_TOKENS or kw.get("use_cache") is not True:
                fail(f"{where}: generate received {{{', '.join(f'{k}={kw.get(k)!r}' for k in sorted(kw))}}}")
            eff, defaults = rec.get("effective") or {}, head.get("generation_defaults") or {}
            for key in ("do_sample", "temperature", "top_p", "max_new_tokens"):
                if eff.get(key) != kw.get(key):
                    fail(f"{where}: effective {key} {eff.get(key)!r} differs from what was passed")
            for key, allowed in (("num_beams", (None, 1)), ("repetition_penalty", (None, 1.0)),
                                 ("no_repeat_ngram_size", (None, 0)), ("min_new_tokens", (None, 0)),
                                 ("min_p", (None,)), ("typical_p", (None, 1.0))):
                if eff.get(key) not in allowed:
                    fail(f"{where}: effective {key} {eff.get(key)!r}, registered unchanged")
            if eff.get("top_k") != defaults.get("top_k"):
                fail(f"{where}: effective top_k {eff.get('top_k')!r} differs from the model default")
            if _id_set(kw.get("eos_token_id")) != expected_stops:
                fail(f"{where}: generate received EOS ids {sorted(_id_set(kw.get('eos_token_id')))}, "
                     f"expected checkpoint generation-config and tokenizer EOS ids {sorted(expected_stops)}")
            if _id_set(eff.get("eos_token_id")) != expected_stops:
                fail(f"{where}: effective EOS ids {sorted(_id_set(eff.get('eos_token_id')))}, "
                     f"expected checkpoint generation-config and tokenizer EOS ids {sorted(expected_stops)}")
            rng = rec.get("rng_at_entry") or {}
            if rng.get("cpu_initial_seed") != seed:
                fail(f"{where}: CPU RNG seeded {rng.get('cpu_initial_seed')!r} at generate entry, registered {seed}")
            device = str(head.get("device", "cpu")).split(":")[0]
            if device in ("cuda", "xpu") and rng.get(f"{device}_initial_seed") != seed:
                fail(f"{where}: {device} RNG seeded {rng.get(f'{device}_initial_seed')!r}, registered {seed}")
            for when in ("hooks_at_entry", "hooks_at_exit"):
                census = rec.get(when) or {}
                mods = census.get("modules") or {}
                if census.get("global_pre") or census.get("global_post") or len(mods) != 1 \
                        or not next(iter(mods)).endswith(LAYER_SUFFIX) or next(iter(mods.values())) != {"pre": 2, "post": 0}:
                    fail(f"{where}: {when} {mods}, expected only layer 10 down_proj's clamp and counter pre-hooks")
            if rec.get("unobserved_forwards") or rec.get("shape_changes"):
                fail(f"{where}: forwards the observer could not compare")
            raw = (rec.get("raw") or [{}])[0]
            key = (frame, qi)
            if key not in renders:
                renders[key] = render(tokenizer, frame, QUESTIONS[qi], template)
            prompt_ids = raw.get("prompt_ids")
            if prompt_ids != renders[key]:
                fail(f"{where}: generate received prompt ids that are not the registered {frame} frame for q{qi}")
            gen = raw.get("generated_ids")
            if not isinstance(gen, list) or not gen:
                fail(f"{where}: no generated token ids recorded")
                continue
            stop = expected_stops
            cut = gen
            for pos, tok in enumerate(gen):
                if tok in stop:
                    cut = gen[:pos + 1]
                    break
            if len(cut) != len(gen):
                fail(f"{where}: generation continued past a stop token")
            answer = tokenizer.decode(cut, skip_special_tokens=True).strip()
            if row.get("answer") != answer:
                fail(f"{where}: answer text is not the decode of the generated tokens")
            orow = (rec.get("rows") or [{}])[0]
            if orow.get("token_ids_sha256") != ids_sha(cut) or row.get("new_tokens") != len(gen):
                fail(f"{where}: token counts or hashes disagree with the raw ids")
            if row.get("seed") != seed:
                fail(f"{where}: answer row seed {row.get('seed')!r}, registered {seed}")
            if row.get("frame_template_sha256") != template_sha:
                fail(f"{where}: answer row plain-frame template SHA-256 does not match the registered frame")
            layers = orow.get("layers") or {}
            decodes = len(gen) - 1
            pc = (rec.get("phase_counts") or {}).get(str(LAYER), {})
            if pc.get("prefill") != [1, len(prompt_ids)] or pc.get("decode", [0, 0]) != [decodes, decodes]:
                fail(f"{where}: layer 10 saw {pc}, expected one prefill of {len(prompt_ids)} and {decodes} decodes")
            if set(layers) != {str(layer) for layer in plan}:
                fail(f"{where}: changed layers {sorted(layers)}, registered {sorted(plan)}")
                continue
            for layer, units in plan.items():
                phases = layers[str(layer)]
                check_phase(fail, f"{where} L{layer} prompt", phases.get("prefill"), units, 1, len(prompt_ids))
                if decodes:
                    check_phase(fail, f"{where} L{layer} generated", phases.get("decode"), units, decodes, decodes)
                elif "decode" in phases:
                    fail(f"{where}: an immediate EOS answer shows a decode change")
        want = {(q, s) for q in range(3) for s in range(samples)}
        if set(seen) != want or any(v != 1 for v in seen.values()):
            fail(f"{name}: answers are not 3 questions x {samples} samples, each once")
    for rec in generate_records:
        if rec.get("call_id") not in used:
            fail(f"observer generate call {rec.get('call_id')!r} is not used by an answer row")
    return failures


def audit_step1(cp_dir, checkpoint, tokenizer, token_ids_path, passages_path, reference_path, expected_sha,
                sizes=(NEUTRAL_COUNT, NEUTRAL_LEN, PASSAGE_PAIRS)):
    neutral_count, neutral_len, passage_pairs = sizes
    import numpy as np
    failures = []
    fail = failures.append
    step1 = read_json(cp_dir / "step1.json")
    cp = step1.get("checkpoint") or {}
    if cp.get("key") != checkpoint:
        fail(f"step1.json is for {cp.get('key')!r}, not {checkpoint}")
    ids_file = read_json(Path(token_ids_path))
    ids = np.asarray(ids_file["token_ids"], dtype="<i4")
    sha = hashlib.sha256(ids.tobytes(order="C")).hexdigest()
    if ids.shape != (neutral_count, neutral_len) or sha != expected_sha or step1.get("neutral_token_ids_sha256") != sha:
        fail(f"neutral token ids {ids.shape} sha {sha[:12]} do not match the pinned set and step1.json")
    passages = read_json(Path(passages_path))
    if len(passages) != passage_pairs:
        fail(f"{len(passages)} passage pairs, registered {passage_pairs}")
    obs_path = cp_dir / "observe" / "step1.jsonl"
    if not obs_path.exists():
        return failures + [f"{obs_path}: no observer records"]
    sessions = sessions_of(read_jsonl(obs_path))
    if len(sessions) != 1:
        return failures + [f"{obs_path}: {len(sessions)} observer sessions, expected one"]
    (sid, recs), = sessions.items()
    check_session(fail, recs, f"observer session {sid}", cp.get("commit"), "torch.bfloat16",
                  {str(LAYER): [TARGET, CONTROL]})
    fwd = [r for r in recs if r.get("event") == "forward"]
    if [r for r in recs if r.get("event") == "generate"]:
        fail("step 1 observer saw a generate call")
    expected_inputs = [[int(t) for t in row] for row in ids.tolist()]
    for pair in passages:
        for field in ("offworld", "ordinary"):
            expected_inputs.append(list(tokenizer(pair[field], add_special_tokens=False)["input_ids"]))
    if len(fwd) != len(expected_inputs):
        return failures + [f"observer saw {len(fwd)} forward passes, registered {len(expected_inputs)}"]
    values = []
    for i, (rec, want) in enumerate(zip(fwd, expected_inputs)):
        where = f"forward {i}"
        if rec.get("seq") != i + 1:
            fail(f"{where}: sequence id {rec.get('seq')!r}, expected {i + 1} for the registered neutral-then-passage order")
        if rec.get("error") or rec.get("input_ids") != [want]:
            fail(f"{where}: input ids are not the registered text {i}")
        expected_kwargs = {"input_ids": {"__tensor__": True, "dtype": "torch.int64",
                                          "shape": [1, len(want)], "value": [want]},
                           "use_cache": False}
        if rec.get("positional_args") != 0 or rec.get("kwargs") != ["use_cache"] \
                or rec.get("kwargs_values") != expected_kwargs:
            fail(f"{where}: forward keyword values {rec.get('kwargs_values')!r}, "
                 "expected only registered input_ids and use_cache=False")
        census = (rec.get("hooks_at_entry") or {}).get("modules") or {}
        if len(census) != 1 or not next(iter(census)).endswith(LAYER_SUFFIX) or next(iter(census.values())) != {"pre": 1, "post": 0}:
            fail(f"{where}: hooks {census}, expected only the step-1 reader on layer 10 down_proj")
        layer = (rec.get("layers") or {}).get(str(LAYER)) or {}
        if layer.get("calls") != 1 or layer.get("changed_by_hooks") is not False:
            fail(f"{where}: layer 10 called {layer.get('calls')!r} times or changed by a hook")
        vals = layer.get("values") or {}
        try:
            # values[unit] is calls x batch rows x positions; every text must be one call and one row.
            unit_rows = []
            for unit in (TARGET, CONTROL):
                unit_calls = vals[str(unit)]
                if len(unit_calls) != 1 or len(unit_calls[0]) != 1 or len(unit_calls[0][0]) != len(want):
                    raise ValueError(f"unit {unit}: got {len(unit_calls)} calls, "
                                     f"{[len(rows) for rows in unit_calls]} rows, "
                                     f"expected one call, one row, {len(want)} positions")
                unit_rows.append(np.asarray(unit_calls[0][0], dtype=np.float64))
            values.append(np.stack(unit_rows, axis=-1))
        except (KeyError, IndexError, ValueError):
            fail(f"{where}: observed values must have one call, one row, and the registered position count "
                 f"for units {TARGET} and {CONTROL}")
            values.append(None)
    if failures:
        return failures
    neutral = np.stack(values[:neutral_count])
    reported = np.asarray(step1.get("neutral_activations"), dtype=np.float64)
    if reported.shape != neutral.shape or not np.array_equal(reported, neutral):
        fail("step1.json neutral activations are not the values layer 10's down_proj received")
    flat = neutral.reshape(-1, 2)
    for col, unit in enumerate((TARGET, CONTROL)):
        if not near((step1.get("neutral_activation_maxima") or {}).get(str(unit)), float(flat[:, col].max()), 1e-9):
            fail(f"unit {unit} maximum differs from the observed maximum {float(flat[:, col].max())!r}")
    ref = neutral if checkpoint == "instruct" else (
        np.asarray(read_json(Path(reference_path))["neutral_activations"], dtype=np.float64) if reference_path else None)
    if ref is None:
        fail("a non-Instruct checkpoint needs --reference (Instruct's audited step1.json)")
    else:
        rflat = ref.reshape(-1, 2)
        for col, name in enumerate(("target", "control")):
            got = (step1.get("neutral_correlations_with_instruct") or {}).get(name)
            want = float(np.corrcoef(flat[:, col], rflat[:, col])[0, 1])
            if not near(got, want, 1e-9):
                fail(f"{name} correlation {got!r}, recomputed {want!r}")
    # step1 averages each passage in float32 and this audit in float64: they agree to ~5e-8 relative, so the
    # passage checks use float32 precision (1e-6), not 1e-9 (found on the first real run, 2026-10-28).
    off = [float(values[neutral_count + 2 * j][:, 0].mean()) for j in range(len(passages))]
    ordinary = [float(values[neutral_count + 2 * j + 1][:, 0].mean()) for j in range(len(passages))]
    p = step1.get("passages") or {}
    if p.get("count_each") != len(passages) \
            or not all(near(a, b, 1e-6) for a, b in zip(p.get("offworld_per_passage_mean", []), off)) \
            or not all(near(a, b, 1e-6) for a, b in zip(p.get("ordinary_per_passage_mean", []), ordinary)) \
            or len(p.get("offworld_per_passage_mean", [])) != len(off) \
            or len(p.get("ordinary_per_passage_mean", [])) != len(ordinary) \
            or not near(p.get("offworld_mean"), float(np.mean(off)), 1e-6) \
            or not near(p.get("ordinary_mean"), float(np.mean(ordinary)), 1e-6) \
            or not near(p.get("ordinary_sd"), float(np.std(ordinary, ddof=1)), 1e-6):
        fail("passage means differ from the observed values")
    return failures


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cp_dir")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True, help="the checkpoint snapshot (or any directory with its tokenizer)")
    ap.add_argument("--step1", action="store_true")
    ap.add_argument("--mode", choices=("pilot", "main"))
    ap.add_argument("--attempt", type=int, default=1)
    ap.add_argument("--chat-only", action="store_true")
    ap.add_argument("--step1-json", help="step 2: the audited step1.json the run used (default <cp_dir>/step1.json)")
    ap.add_argument("--token-ids", help="the pinned neutral token ids file (step 1)")
    ap.add_argument("--passages", default=str(HERE / "passages.json"))
    ap.add_argument("--reference", help="Instruct's step1.json (step 1, non-Instruct checkpoints)")
    ap.add_argument("--expected-token-sha", default=PINNED_TOKEN_SHA)
    ap.add_argument("--test-sizes", nargs=3, type=int, metavar=("COUNT", "LENGTH", "PAIRS"),
                    help="test fixtures only: neutral sequence count and length and passage pairs; a run never sets it")
    args = ap.parse_args(argv)
    tokenizer = load_tokenizer(args.tokenizer)
    cp_dir = Path(args.cp_dir)
    if args.step1:
        if not args.token_ids:
            ap.error("--step1 needs --token-ids")
        sizes = tuple(args.test_sizes) if args.test_sizes else (NEUTRAL_COUNT, NEUTRAL_LEN, PASSAGE_PAIRS)
        failures = audit_step1(cp_dir, args.checkpoint, tokenizer, args.token_ids, args.passages, args.reference,
                               args.expected_token_sha, sizes)
        label = f"AUDIT85 {args.checkpoint} step1" + (" (TEST SIZES, not a registered run)" if args.test_sizes else "")
    else:
        if not args.mode:
            ap.error("step 2 needs --mode")
        generation_config = load_generation_config(args.tokenizer)
        failures = audit_step2(cp_dir, args.checkpoint, args.mode, args.attempt, args.chat_only, tokenizer,
                               args.step1_json, generation_config)
        label = f"AUDIT85 {args.checkpoint} {args.mode}"
    for f in failures[:200]:
        print("FAIL", f)
    if len(failures) > 200:
        print(f"... and {len(failures) - 200} more")
    print(f"{label}: {'PASS' if not failures else f'{len(failures)} failures'}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
