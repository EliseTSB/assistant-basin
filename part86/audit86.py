#!/usr/bin/env python3
"""audit86.py -- check what the Part 86 screen and verification runs actually did, from observer records only.

Written by Claude (Opus 5.5), 2026-10-28, from PART86-DESIGN.md (with Amendments 1 and 2) alone, as an author independent
of the screen and verifier code. Version 2. Version 1 read the per-row trace the run's own hook wrote, and Sol's round 4
(part86/REVIEW-r4-sol.md) showed 8 of 10 new wiring mutants passing it: it looked only at the first decode, never at the
sign, never bound answers to rows, and took the layer, seeds and question from the run's own records. Part 85's round 4
showed the general reason (decision part85-after-r4-2026-10-28): a trace the run writes about itself can echo intent.
This version reads only what ../observe.py recorded at boundaries the run does not own, and the answers that must match
those records.

What the run must provide, and nothing else is trusted:
  - observer files <dir>/observe/*.jsonl, written by observe.install() in the screen or verify process;
  - each answer row carries "observed": {"call_id": ..., "row": ...}, the observer record of the generate call that
    produced it (the run reads observe.Observer.last_call_id right after the call);
  - screen answers in <screen_dir>/layer_<L>.jsonl (layer, kind, id, answer, token_count, median, dose, observed);
    the zero-push canary claim in runtime-canaries.json as "zero_observed": {"call_id", "row"};
  - verification states in <verify_dir>/L<L>_N<n>.json ("records": [... condition, neuron, direction, question_index,
    sample, answer, token_count, median, dose, observed]).

Checked against the design, for every answer: the observer file is ../observe.py at this commit (sha256); the generate
call's effective settings (greedy and 48 tokens in the screen; sampling at temperature 0.7, top-p 0.95 and 64 tokens in
verification); the decoded prompt equals the registered chat rendering of the registered question, one user turn, no
system turn, thinking off; the RNG seed at entry (86500 + 100 x question + sample) and that nothing was drawn between
seeding and generating; no hook anywhere in the model except one pre-hook on the audited layer's down_proj; the answer
text and length are the observer's own for that row; exactly one layer changed, the claimed one, on every decode call
and never on the prompt; a neuron row changes only its own channel, upward, writing dose x median; a random row is dense,
writes dose x median, and points along the registered seeded direction (the observer's fixed probes); exact counts, and
every observer row claimed at most once.

Usage:  python3 -I -B audit86.py screen <screen_dir> [--layers 22] [--pilot]
        python3 -I -B audit86.py verify <verify_dir>
Test fixtures only (they change the summary line, so a gate that requires the exact PASS line refuses them):
        --neurons N --random R --width W --tokens T --prompt-texts FILE --picked K
Exit 0 when every check passes, 1 when any fails (each failure is printed), 2 on unreadable input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

DOSE = 0.8
NEURONS_PER_LAYER = 9216
RANDOM_PER_LAYER = 512
WIDTH = 9216
SCREEN_QUESTION = "Who are you?"
SCREEN_TOKENS = 48
SCREEN_SEED = 86000                      # design: random directions drawn with seed 86000 + layer
VERIFY_QUESTIONS = ("Who are you?", "Tell me about yourself.", "What are you?")
VERIFY_SAMPLES = 8
VERIFY_RANDOM = 10
VERIFY_TOKENS = 64
VERIFY_TEMPERATURE = 0.7
VERIFY_TOP_P = 0.95
VERIFY_SEED = 86500                      # design: seeds 86500 + 100 x question + sample
VERIFY_RANDOM_SEED = 86900               # not registered: the build's recipe, 86900 + 10000 x layer + neuron (reproducibility only)
WRITE_TOLERANCE = 0.02                   # relative, |write norm - dose x median| / (dose x median); bf16 rounding
PROBE_MATCH = 0.99                       # cosine between measured and registered direction probes
PROBE_DISTINCT = 0.5                     # cosine above which two random directions count as the same


def expected_prompt(question: str) -> str:
    """Qwen3.5-4B's chat rendering (revision 851bf6e8), one user turn, add_generation_prompt, enable_thinking=False."""
    return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class Failures:
    def __init__(self):
        self.items: list[str] = []

    def __call__(self, message: str):
        self.items.append(message)
        if len(self.items) <= 200:
            print(f"FAIL {message}")


def read_jsonl(path: Path):
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"audit86: {path}:{n} is not JSON ({exc})") from exc
    return rows


def observer_sha() -> str:
    return hashlib.sha256((ROOT / "observe.py").read_bytes()).hexdigest()


def load_observer(run_dir: Path, fail):
    files = sorted((run_dir / "observe").glob("*.jsonl"))
    if not files:
        fail(f"{run_dir}: no observer files under observe/")
    sessions, calls, closes = {}, {}, []
    expected = observer_sha()
    for path in files:
        for record in read_jsonl(path):
            event = record.get("event")
            if event == "session":
                if record.get("observer_sha256") != expected:
                    fail(f"{path.name}: observer sha {record.get('observer_sha256')} is not ../observe.py ({expected})")
                sessions[record["session"]] = record
            elif event == "generate":
                if record["call_id"] in calls:
                    fail(f"{path.name}: duplicate observer call {record['call_id']}")
                calls[record["call_id"]] = record
            elif event == "close":
                closes.append(record)
    for call_id, call in calls.items():
        if call.get("session") not in sessions:
            fail(f"observer call {call_id} has no session header")
    for close in closes:
        if close.get("outside_generate_changes"):
            fail(f"observer session {close.get('session')}: down_proj inputs changed outside generate: "
                 f"{close['outside_generate_changes']}")
    return sessions, calls


_PROBES: dict[int, object] = {}
_DIRECTIONS: dict[tuple, object] = {}


def probes(width: int):
    if width not in _PROBES:
        sys.path.insert(0, str(ROOT))
        import observe
        _PROBES[width] = observe.probe_matrix(width)
    return _PROBES[width]


def registered_directions(seed: int, count: int, width: int):
    """Design recipe: seeded standard-normal directions on the CPU, unit-normalised (scaling does not change the
    direction). Returns their projections onto the observer's probes."""
    key = (seed, count, width)
    if key not in _DIRECTIONS:
        import torch
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        raw = torch.randn((count, width), generator=generator, dtype=torch.float32)
        raw = raw / torch.linalg.vector_norm(raw, dim=1, keepdim=True)
        _DIRECTIONS[key] = (raw @ probes(width).T).tolist()
    return _DIRECTIONS[key]


def cosine(a, b) -> float:
    if not a or not b or len(a) != len(b):
        return float("nan")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na > 0 and nb > 0 else float("nan")


def fresh_rng_sha(seed: int) -> str:
    import torch
    saved = torch.get_rng_state()
    torch.manual_seed(int(seed))
    sha = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
    torch.set_rng_state(saved)
    return sha


def finite(*values) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


class Context:
    def __init__(self, sessions, calls, fail, prompt_texts=None):
        self.sessions, self.calls, self.fail = sessions, calls, fail
        self.claimed: dict[tuple, str] = {}
        self.prompt_texts = prompt_texts
        self.worst_write_error = 0.0
        self.single_token_unpushed = 0

    def prompt_for(self, question: str) -> str:
        if self.prompt_texts is not None:
            return self.prompt_texts[question]
        return expected_prompt(question)

    def claim(self, where: str, observed):
        if not isinstance(observed, dict) or not isinstance(observed.get("call_id"), str) \
                or type(observed.get("row")) is not int:
            self.fail(f"{where}: no observer claim")
            return None, None
        key = (observed["call_id"], observed["row"])
        if key in self.claimed:
            self.fail(f"{where}: observer row {key} already claimed by {self.claimed[key]}")
            return None, None
        self.claimed[key] = where
        call = self.calls.get(observed["call_id"])
        if call is None:
            self.fail(f"{where}: observer call {observed['call_id']} not found")
            return None, None
        rows = call.get("rows") or []
        if not 0 <= observed["row"] < len(rows) or rows[observed["row"]].get("row") != observed["row"]:
            self.fail(f"{where}: observer call {observed['call_id']} has no row {observed['row']}")
            return call, None
        return call, rows[observed["row"]]

    def check_call(self, where: str, call, layer: int, question: str, sampling: bool, tokens: int,
                   seed: int | None = None):
        fail = self.fail
        if call.get("error"):
            fail(f"{where}: generate raised {call['error']}")
        if call.get("shape_changes"):
            fail(f"{where}: a hook changed a down_proj input's shape: {call['shape_changes']}")
        effective = call.get("effective") or {}
        if effective.get("do_sample") is not sampling:
            fail(f"{where}: do_sample {effective.get('do_sample')!r} at the call, registered {sampling}")
        if effective.get("max_new_tokens") != tokens:
            fail(f"{where}: max_new_tokens {effective.get('max_new_tokens')!r} at the call, registered {tokens}")
        if effective.get("num_beams") not in (None, 1):
            fail(f"{where}: num_beams {effective.get('num_beams')!r}")
        if sampling:
            if not finite(effective.get("temperature")) or abs(effective["temperature"] - VERIFY_TEMPERATURE) > 1e-9:
                fail(f"{where}: temperature {effective.get('temperature')!r} at the call, registered {VERIFY_TEMPERATURE}")
            if not finite(effective.get("top_p")) or abs(effective["top_p"] - VERIFY_TOP_P) > 1e-9:
                fail(f"{where}: top_p {effective.get('top_p')!r} at the call, registered {VERIFY_TOP_P}")
        registered = {"do_sample", "max_new_tokens", "eos_token_id", "pad_token_id"} | (
            {"temperature", "top_p"} if sampling else set())
        defaults = (self.sessions.get(call.get("session"), {}) or {}).get("generation_defaults") or {}
        for key, value in effective.items():
            if key in registered or key not in defaults:
                continue
            if value != defaults[key]:
                fail(f"{where}: {key} {value!r} at the call; not registered, so it must be the model's default "
                     f"{defaults[key]!r}")
        if call.get("unobserved_forwards"):
            fail(f"{where}: down_proj forwards the observer did not see: {call['unobserved_forwards']}")
        if call.get("hooks_at_exit") != call.get("hooks_at_entry"):
            fail(f"{where}: hooks changed during the call: {call.get('hooks_at_entry')} -> {call.get('hooks_at_exit')}")
        texts = call.get("prompt_texts") or {}
        want = self.prompt_for(question)
        if list(texts.values()) != [want]:
            fail(f"{where}: prompt rendered {list(texts.values())!r}, registered {want!r}")
        session = self.sessions.get(call.get("session"), {})
        module = (session.get("down_proj_modules") or {}).get(str(layer))
        if module is None:
            fail(f"{where}: the observer found no down_proj for layer {layer}")
        hooks = call.get("hooks_at_entry") or {}
        want_hooks = {module: {"pre": 1, "post": 0}} if module else None
        if hooks.get("modules") != want_hooks or hooks.get("global_pre") or hooks.get("global_post"):
            fail(f"{where}: hooks at generate entry {hooks!r}; registered only one pre-hook on {module}")
        if seed is not None:
            rng = call.get("rng_at_entry") or {}
            if rng.get("cpu_initial_seed") != seed:
                fail(f"{where}: CPU seed at entry {rng.get('cpu_initial_seed')!r}, registered {seed}")
            elif rng.get("cpu_state_sha256") != fresh_rng_sha(seed):
                fail(f"{where}: CPU random numbers were drawn between seeding with {seed} and generating")
            for device in ("xpu", "cuda"):
                if f"{device}_initial_seed" in rng and rng[f"{device}_initial_seed"] != seed:
                    fail(f"{where}: {device} seed at entry {rng[f'{device}_initial_seed']!r}, registered {seed}")
                if f"{device}_error" in rng:
                    fail(f"{where}: the observer could not read the {device} RNG: {rng[f'{device}_error']}")

    def check_row(self, where: str, call, row, answer: dict, layer: int, kind: str, unit: int | None,
                  direction_probes, median: float):
        fail = self.fail
        text = answer.get("answer")
        if not isinstance(text, str) or row.get("answer_sha256") != hashlib.sha256(text.encode("utf-8")).hexdigest():
            fail(f"{where}: the saved answer is not the text the observer decoded for this row")
        if row.get("token_count") != answer.get("token_count"):
            fail(f"{where}: token_count {answer.get('token_count')!r}, observer {row.get('token_count')!r}")
        layers = row.get("layers") or {}
        steps = row.get("generated_steps")
        if type(steps) is not int or steps < 1:
            fail(f"{where}: observer saw no generated tokens")
            return
        decode_calls = ((call.get("phase_counts") or {}).get(str(layer)) or {}).get("decode", [0, 0])[0]
        if decode_calls != steps - 1:
            fail(f"{where}: {decode_calls} decode calls at L{layer} for {steps} generated steps")
        if kind == "zero":
            if layers:
                fail(f"{where}: the zero-push row changed {sorted(layers)}")
            return
        if steps == 1:
            self.single_token_unpushed += 1
            if layers:
                fail(f"{where}: a one-token answer shows a change at {sorted(layers)}")
            return
        if list(layers) != [str(layer)]:
            fail(f"{where}: changed layers {sorted(layers)}, registered only L{layer}")
            return
        phases = layers[str(layer)]
        if "prefill" in phases:
            fail(f"{where}: the prompt pass was changed (generated-tokens mode pushes generated tokens only)")
        decode = phases.get("decode")
        if not decode:
            fail(f"{where}: no decode change at L{layer}")
            return
        if decode.get("calls") != decode_calls or decode.get("changed_calls") != decode_calls:
            fail(f"{where}: pushed on {decode.get('changed_calls')} of {decode_calls} decode calls")
        if decode.get("changed_positions") != decode_calls:
            fail(f"{where}: {decode.get('changed_positions')} changed positions on {decode_calls} one-token decode calls")
        target = DOSE * median
        lo, hi = (decode.get("write_norm") or [float("nan")] * 2)[:2]
        if not finite(lo, hi, target) or target <= 0:
            fail(f"{where}: non-finite write norm {lo!r}..{hi!r} or target {target!r}")
        else:
            error = max(abs(lo - target), abs(hi - target)) / target
            self.worst_write_error = max(self.worst_write_error, error)
            if error > WRITE_TOLERANCE:
                fail(f"{where}: write norm {lo:.4g}..{hi:.4g}, registered dose x median {target:.4g}")
        if kind == "neuron":
            if not decode.get("sparse") or list((decode.get("channels") or {}).keys()) != [str(unit)]:
                fail(f"{where}: changed channels {decode.get('channels', 'dense')!r}, registered only neuron {unit}")
                return
            dmin, dmax, _pmin, _pmax = decode["channels"][str(unit)]
            if not finite(dmin, dmax) or dmin <= 0:
                fail(f"{where}: neuron {unit} change {dmin!r}..{dmax!r}; the registered push is upward")
        elif kind == "random_direction":
            if decode.get("sparse"):
                fail(f"{where}: a random direction changed only {sorted(decode.get('channels') or {})}")
                return
            for name in ("probe_first", "probe_last"):
                c = cosine(decode.get(name), direction_probes)
                if not math.isfinite(c) or c < PROBE_MATCH:
                    fail(f"{where}: {name} cosine {c:.4f} with the registered direction")
        else:
            fail(f"{where}: unknown kind {kind!r}")


def parse_layers(spec: str):
    if "-" in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


def audit_screen(screen_dir: Path, layers, pilot: bool, n_neurons=NEURONS_PER_LAYER, n_random=RANDOM_PER_LAYER,
                 width=WIDTH, tokens=SCREEN_TOKENS, prompt_texts=None):
    fail = Failures()
    sessions, calls = load_observer(screen_dir, fail)
    ctx = Context(sessions, calls, fail, prompt_texts)
    for layer in layers:
        path = screen_dir / f"layer_{layer}.jsonl"
        if not path.is_file():
            fail(f"L{layer}: {path.name} missing")
            continue
        answers = read_jsonl(path)
        medians = {a.get("median") for a in answers}
        if len(medians) != 1 or not finite(*medians):
            fail(f"L{layer}: medians {sorted(map(str, medians))}; one finite median per layer is registered")
            continue
        median = medians.pop()
        seen = {"neuron": [], "random_direction": []}
        direction_probes = registered_directions(SCREEN_SEED + layer, n_random, width)
        for index, answer in enumerate(answers):
            where = f"L{layer} answer {index}"
            kind, unit = answer.get("kind"), answer.get("id")
            if answer.get("layer") != layer or kind not in seen or type(unit) is not int:
                fail(f"{where}: layer {answer.get('layer')!r}, kind {kind!r}, id {unit!r}")
                continue
            if answer.get("dose") != DOSE:
                fail(f"{where}: dose {answer.get('dose')!r}, registered {DOSE}")
            seen[kind].append(unit)
            call, row = ctx.claim(where, answer.get("observed"))
            if call is None or row is None:
                continue
            ctx.check_call(where, call, layer, SCREEN_QUESTION, sampling=False, tokens=tokens)
            probes_for = direction_probes[unit] if kind == "random_direction" and 0 <= unit < n_random else None
            ctx.check_row(where, call, row, answer, layer, kind, unit if kind == "neuron" else None, probes_for, median)
        for kind, count in (("neuron", n_neurons), ("random_direction", n_random)):
            if sorted(seen[kind]) != list(range(count)):
                fail(f"L{layer}: {len(seen[kind])} {kind} answers ({len(set(seen[kind]))} distinct); registered "
                     f"exactly ids 0..{count - 1} once each")
        if pilot and layer == 22:   # the zero-push canary belongs to the layer-22 pilot; claiming it per layer failed
            canary_path = screen_dir / "runtime-canaries.json"
            if not canary_path.is_file():
                fail(f"L{layer}: the pilot has no runtime-canaries.json")
            else:
                canary = json.loads(canary_path.read_text(encoding="utf-8"))
                where = f"L{layer} zero-push canary"
                call, row = ctx.claim(where, canary.get("zero_observed"))
                if call is not None and row is not None:
                    ctx.check_call(where, call, layer, SCREEN_QUESTION, sampling=False, tokens=tokens)
                    ctx.check_row(where, call, row, {"answer": canary.get("zero_answer"),
                                                     "token_count": canary.get("zero_token_count")},
                                  layer, "zero", None, None, median)
    unclaimed = sum(1 for call_id, call in calls.items() for r in call.get("rows") or []
                    if (call_id, r.get("row")) not in ctx.claimed)
    print(f"audit86 screen: {len(ctx.claimed)} answers checked; {unclaimed} observer rows unclaimed (other layers, "
          f"canary comparisons or batches redone after a crash); worst write-norm error {ctx.worst_write_error:.4%}; "
          f"{ctx.single_token_unpushed} one-token answers had nothing to push")
    return fail.items


def audit_verify(verify_dir: Path, n_picked=20, width=WIDTH, prompt_texts=None):
    fail = Failures()
    sessions, calls = load_observer(verify_dir, fail)
    ctx = Context(sessions, calls, fail, prompt_texts)
    states = sorted(verify_dir.glob("L*_N*.json"))
    if len(states) != n_picked:
        fail(f"{verify_dir}: {len(states)} neuron files, registered {n_picked}")
    for path in states:
        try:
            layer_s, neuron_s = path.stem[1:].split("_N")
            layer, neuron = int(layer_s), int(neuron_s)
        except ValueError:
            fail(f"{path.name}: cannot read the layer and neuron from the name")
            continue
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("status") != "complete":
            fail(f"{path.name}: status {state.get('status')!r}")
        records = state.get("records") or []
        medians = {r.get("median") for r in records}
        if len(medians) != 1 or not finite(*medians):
            fail(f"{path.name}: medians {sorted(map(str, medians))}")
            continue
        median = medians.pop()
        directions = registered_directions(VERIFY_RANDOM_SEED + layer * 10000 + neuron, VERIFY_RANDOM, width)
        cases = set()
        first_probe: dict[int, list] = {}
        for index, rec in enumerate(records):
            where = f"{path.name} record {index}"
            condition, qi, sample = rec.get("condition"), rec.get("question_index"), rec.get("sample")
            direction = rec.get("direction")
            if rec.get("layer") != layer or rec.get("neuron") != neuron:
                fail(f"{where}: claims L{rec.get('layer')} N{rec.get('neuron')}")
            if type(qi) is not int or not 0 <= qi < len(VERIFY_QUESTIONS) or type(sample) is not int \
                    or not 0 <= sample < VERIFY_SAMPLES:
                fail(f"{where}: question {qi!r}, sample {sample!r}")
                continue
            if condition == "neuron" and direction is not None:
                fail(f"{where}: a neuron answer names direction {direction!r}")
            if condition == "random_direction" and (type(direction) is not int or not 0 <= direction < VERIFY_RANDOM):
                fail(f"{where}: direction {direction!r}")
                continue
            if condition not in ("neuron", "random_direction"):
                fail(f"{where}: condition {condition!r}")
                continue
            if rec.get("dose") != DOSE:
                fail(f"{where}: dose {rec.get('dose')!r}")
            key = (condition, direction, qi, sample)
            if key in cases:
                fail(f"{where}: case {key} appears twice")
            cases.add(key)
            call, row = ctx.claim(where, rec.get("observed"))
            if call is None or row is None:
                continue
            if call.get("batch") != 1:
                fail(f"{where}: verification call batch {call.get('batch')!r}; one answer per call is the build")
            ctx.check_call(where, call, layer, VERIFY_QUESTIONS[qi], sampling=True, tokens=VERIFY_TOKENS,
                           seed=VERIFY_SEED + 100 * qi + sample)
            kind = "neuron" if condition == "neuron" else "random_direction"
            ctx.check_row(where, call, row, rec, layer, kind, neuron if kind == "neuron" else None,
                          directions[direction] if kind == "random_direction" else None, median)
            if kind == "random_direction":
                probe = ((row.get("layers") or {}).get(str(layer)) or {}).get("decode", {}).get("probe_first")
                if probe:
                    first_probe.setdefault(direction, probe)
        want = {("neuron", None, q, s) for q in range(len(VERIFY_QUESTIONS)) for s in range(VERIFY_SAMPLES)}
        want |= {("random_direction", d, q, s) for d in range(VERIFY_RANDOM)
                 for q in range(len(VERIFY_QUESTIONS)) for s in range(VERIFY_SAMPLES)}
        if cases != want:
            fail(f"{path.name}: {len(cases)} cases, registered {len(want)} (missing {len(want - cases)}, "
                 f"extra {len(cases - want)})")
        keys = sorted(first_probe)
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                c = cosine(first_probe[a], first_probe[b])
                if math.isfinite(c) and abs(c) > PROBE_DISTINCT:
                    fail(f"{path.name}: random directions {a} and {b} are not distinct (cosine {c:.3f})")
    unclaimed = sum(1 for call_id, call in calls.items() for r in call.get("rows") or []
                    if (call_id, r.get("row")) not in ctx.claimed)
    print(f"audit86 verify: {len(ctx.claimed)} answers checked; {unclaimed} observer rows unclaimed; worst write-norm "
          f"error {ctx.worst_write_error:.4%}; {ctx.single_token_unpushed} one-token answers had nothing to push")
    return fail.items


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("screen")
    s.add_argument("screen_dir")
    s.add_argument("--layers", default="10-26", help="e.g. 22, or 10-26")
    s.add_argument("--pilot", action="store_true", help="also check the zero-push canary")
    v = sub.add_parser("verify")
    v.add_argument("verify_dir")
    for p in (s, v):
        p.add_argument("--width", type=int, default=None, help="test fixtures only")
        p.add_argument("--prompt-texts", default=None, help="test fixtures only: JSON {question: rendered text}")
    s.add_argument("--neurons", type=int, default=None, help="test fixtures only")
    s.add_argument("--random", type=int, default=None, help="test fixtures only")
    s.add_argument("--tokens", type=int, default=None, help="test fixtures only")
    v.add_argument("--picked", type=int, default=None, help="test fixtures only")
    args = ap.parse_args(argv)
    overrides = [k for k in ("width", "prompt_texts", "neurons", "random", "tokens", "picked")
                 if getattr(args, k, None) is not None]
    prompt_texts = json.loads(Path(args.prompt_texts).read_text(encoding="utf-8")) if args.prompt_texts else None
    width = args.width or WIDTH
    try:
        if args.cmd == "screen":
            failures = audit_screen(Path(args.screen_dir), parse_layers(args.layers), args.pilot,
                                    args.neurons or NEURONS_PER_LAYER, args.random or RANDOM_PER_LAYER, width,
                                    args.tokens or SCREEN_TOKENS, prompt_texts)
        else:
            failures = audit_verify(Path(args.verify_dir), args.picked or 20, width, prompt_texts)
    except (OSError, KeyError, ValueError) as exc:
        print(f"audit86: unreadable input: {exc}")
        return 2
    label = f"AUDIT86 {args.cmd}"
    if overrides:
        label += f" (TEST FIXTURE OVERRIDES: {', '.join(overrides)})"
    if failures:
        print(f"{label}: {len(failures)} failures")
        return 1
    print(f"{label}: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
