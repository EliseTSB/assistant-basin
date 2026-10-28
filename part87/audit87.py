#!/usr/bin/env python3
"""audit87.py -- check what the Part 87 runs actually did, from observer records only.

Written by Claude (Opus 5.5), 2026-10-28, from PART87-DESIGN.md (with Amendment 1) and the registered rules it cites
(Part 84's questions and seeds, Part 66's phrasings, guards and row seeds, Part 25's matched-random rule), as an author
independent of part87.py. Haiku's round 1 (part87/REVIEW-r1-haiku.md) found 15 of 15 run-loop mutants passing the
build's tests; Parts 85 and 86 had the same class (decision part85-after-r4-2026-10-28). So this audit reads only what
../observe.py recorded at boundaries the run does not own, and the answers that must match those records.

What the run must provide:
  - observer files <run_dir>/observe/*.jsonl from observe.install() in the run process;
  - <run_dir>/<phase>/<arm>/answers.jsonl rows with arm, question_index, question, sample, answer, new_tokens and
    "observed": {"call_id", "row"} for the generate call that produced the row;
  - the locked dose file (doses.json): residual_medians {layer: median}, target_doses {"L:U": {"mean", "M", "F",
    "column_norm"}}, random_sets [{"seed", "units", "doses": [{"layer", "unit", "mean", "M", "F", "column_norm"}]}].

Checked for every answer, against the design: the observer is ../observe.py at this commit; sampling at the call
(temperature 0.7, top-p 0.95; 64 new tokens in 87A, 48 in 87B); the prompt is Qwen3.5's chat rendering of a registered
question with thinking off; the seed at entry is the registered one (87A: Part 84's 84000 + 100 x question + sample;
87B: Part 66's row seed for phase:arm) with nothing drawn before generating; hooks exist only on the arm's down_proj
modules; the saved answer is the observer's own text for that row; and the change itself:
  A0, U    nothing changed anywhere;
  E63, E47 L22 N2742 only, upward, on every prompt and every decode position, writing dose x median(L22);
  D47      the same at +0.47, on decode positions only (Part 84's generated-tokens mode);
  C63      L22 N8870 only, as E63;
  Z        each of L6 N753, L7 N1169, L11 N8868, L29 N776 set to exactly 0 on every position, nothing else changed;
  M, RM    each unit shifted by minus its own measured mean (the M rule) on every position;
  F, RF    each unit shifted against the sign of its mean, writing 0.75 x median(L) on every position;
  random sets are Part 25's matched-random sets for seeds 8700 to 8719, one unit in each target layer.
L22's median must be within 1% of neuronary's published 24.8, the one external anchor the design names. Counts are exact.

Round-2 fold (decision part87-after-r2-2026-10-28): generate may be passed only the registered keywords and nothing
positionally; every other setting the observer records must equal the model's own default; a down_proj forward the
observer did not see, a changed input shape, or a hook added or removed during a call fails; each answer's stated
question index and kind must be the registered ones for its text; the observer's model must be Qwen3.5-4B in bf16 on
xpu (test fixtures pass --fixture-model, which marks the summary line).

Not checked here, and why it does not have to be: that each unit's measured mean m_n is right, and that the saved
answers are what the recipe produces. replay87.py recomputes both from the design on the model; the result is read
only when both this audit and replay87 pass.

Usage:  python3 -I -B audit87.py <run_dir> --phase pilot|run [--doses <doses.json>]
Exit 0 when every check passes, 1 when any fails, 2 on unreadable input. Test fixtures only (they change the summary
line): --prompt-texts FILE --width W --l22-median M.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "part86"))

import audit86 as a86  # noqa: E402  shared observer readers, written by the same author for Part 86

ALIEN = (22, 2742)
CONTROL = (22, 8870)
TARGETS = ((6, 753), (7, 1169), (11, 8868), (29, 776))
RANDOM_SEEDS = tuple(range(8700, 8720))
WIDTH = 9216
TEMPERATURE = 0.7
TOP_P = 0.95
A_TOKENS = 64
B_TOKENS = 48
L22_PUBLISHED_MEDIAN = 24.8
MEDIAN_ANCHOR_TOLERANCE = 0.01
TOLERANCE = 0.02            # relative, bf16 rounding of an added value
F_DOSE = 0.75
# The keywords the design's generate call may carry (round-2 fold); everything else must be absent.
EXPLICIT_ALLOWED = {"max_new_tokens", "do_sample", "temperature", "top_p", "eos_token_id", "pad_token_id", "use_cache"}
# Settings the design fixes; every other setting the observer records must equal the model's own default.
REGISTERED_SETTINGS = {"do_sample", "temperature", "top_p", "max_new_tokens", "eos_token_id", "pad_token_id"}
PRODUCTION_MODEL = ("Qwen3.5-4B", "torch.bfloat16", "xpu")


def registered_questions():
    import part84.part84 as p84
    import part66.part66 as p66
    return {
        "origin": list(p84.QUESTIONS),
        "maker": list(p66.DIRECT) + list(p66.INDIRECT),
        "guard": [g["prompt"] if isinstance(g, dict) else (g[0] if isinstance(g, (list, tuple)) else g)
                  for g in p66.GUARDS],
        "seed_a": p84.seed_for,
        "seed_b": p66.row_seed,
        "n_direct": len(p66.DIRECT),
    }


def matched_random(seed: int):
    import part25
    return [tuple(map(int, u)) for u in part25._matched_random(list(TARGETS), WIDTH, 36, seed)]


def arms(phase: str):
    names = {"A0": "87A", "E63": "87A", "E47": "87A", "D47": "87A", "C63": "87A",
             "U": "87B", "Z": "87B", "M": "87B", "F": "87B"}
    if phase == "run":
        for seed in RANDOM_SEEDS:
            names[f"RM_{seed}"] = "87B"
            names[f"RF_{seed}"] = "87B"
    return names


def expected_cases(phase: str, arm: str, q):
    """(question, sample) pairs the design registers for one arm."""
    if arm in ("A0", "E63", "E47", "D47", "C63"):
        questions = q["origin"][:1] if phase == "pilot" else q["origin"]
        samples = 4 if phase == "pilot" else 8
        return {(question, s) for question in questions for s in range(samples)}
    if phase == "pilot":
        return {(question, s) for question in q["maker"][:3] for s in range(2)}
    if arm.startswith(("RM_", "RF_")):
        return {(question, 0) for question in q["maker"]}
    return {(question, s) for question in q["maker"] + q["guard"] for s in range(4)}


class Audit:
    def __init__(self, run_dir: Path, phase: str, doses: dict, prompt_texts=None, width=WIDTH, l22=L22_PUBLISHED_MEDIAN,
                 only_arms=None, fixture_model=False):
        self.fixture_model = fixture_model
        self.run_dir, self.phase, self.doses = run_dir, phase, doses
        self.fail = a86.Failures()
        self.sessions, self.calls = a86.load_observer(run_dir, self.fail)
        self.ctx = a86.Context(self.sessions, self.calls, self.fail, prompt_texts)
        self.width = width
        self.l22 = l22
        self.q = registered_questions()
        self.worst = 0.0
        self.only_arms = only_arms

    def median(self, layer: int) -> float:
        value = (self.doses.get("residual_medians") or {}).get(str(layer))
        if not a86.finite(value) or value <= 0:
            self.fail(f"doses: no finite positive median for L{layer}")
            return float("nan")
        return float(value)

    def unit_record(self, arm: str, layer: int, unit: int):
        if arm in ("M", "F"):
            return (self.doses.get("target_doses") or {}).get(f"{layer}:{unit}")
        seed = int(arm.split("_")[1])
        for item in self.doses.get("random_sets") or []:
            if int(item.get("seed", -1)) == seed:
                for rec in item.get("doses") or []:
                    if (int(rec.get("layer")), int(rec.get("unit"))) == (layer, unit):
                        return rec
        return None

    def units_for(self, arm: str):
        if arm in ("E63", "E47", "D47"):
            return [ALIEN]
        if arm == "C63":
            return [CONTROL]
        if arm in ("Z", "M", "F"):
            return list(TARGETS)
        if arm.startswith(("RM_", "RF_")):
            seed = int(arm.split("_")[1])
            want = matched_random(seed)
            item = next((i for i in self.doses.get("random_sets") or [] if int(i.get("seed", -1)) == seed), None)
            if item is None or sorted(tuple(map(int, u)) for u in item.get("units") or []) != sorted(want):
                self.fail(f"doses: random set {seed} is not Part 25's matched set {want}")
            return want
        return []

    def check_change(self, where: str, call, row, arm: str):
        fail = self.fail
        layers = row.get("layers") or {}
        units = self.units_for(arm)
        if not units:
            if layers:
                fail(f"{where}: {arm} changed {sorted(layers)}; registered no change")
            return
        want_layers = sorted({str(layer) for layer, _ in units}, key=int)
        if sorted(layers, key=int) != want_layers:
            fail(f"{where}: changed layers {sorted(layers, key=int)}, registered {want_layers}")
            return
        steps = row.get("generated_steps") or 0
        prompt_len = call.get("prompt_len") or 0
        for layer, unit in units:
            phases = layers[str(layer)]
            want_phases = ["decode"] if arm == "D47" else ["prefill", "decode"]
            if steps < 2:
                want_phases = [p for p in want_phases if p == "prefill"]
            if sorted(phases) != sorted(want_phases):
                fail(f"{where}: L{layer} changed in {sorted(phases)}, registered {want_phases}")
                continue
            for phase in want_phases:
                st = phases[phase]
                positions = prompt_len if phase == "prefill" else steps - 1
                if st.get("changed_positions") != positions:
                    fail(f"{where}: L{layer} {phase} changed {st.get('changed_positions')} of {positions} positions")
                if not st.get("sparse") or list((st.get("channels") or {}).keys()) != [str(unit)]:
                    fail(f"{where}: L{layer} {phase} changed channels {st.get('channels', 'dense')!r}, "
                         f"registered only N{unit}")
                    continue
                dmin, dmax, pmin, pmax = st["channels"][str(unit)]
                self.check_value(where, arm, layer, unit, phase, st, dmin, dmax, pmin, pmax)

    def check_value(self, where, arm, layer, unit, phase, st, dmin, dmax, pmin, pmax):
        fail = self.fail
        if not a86.finite(dmin, dmax, pmin, pmax):
            fail(f"{where}: non-finite change at L{layer} N{unit}")
            return
        if arm == "Z":
            if (pmin, pmax) != (0.0, 0.0):
                fail(f"{where}: L{layer} N{unit} {phase} set to {pmin}..{pmax}; zeroing registers exactly 0")
            return
        median = self.median(layer)
        if arm in ("E63", "E47", "D47", "C63"):
            dose = {"E63": 0.63, "E47": 0.47, "D47": 0.47, "C63": 0.63}[arm]
            self.near(where, f"L{layer} {phase} write norm", st.get("write_norm"), dose * median)
            if dmin <= 0:
                fail(f"{where}: L{layer} N{unit} {phase} change {dmin:.4g}..{dmax:.4g}; registered upward")
            return
        rec = self.unit_record(arm, layer, unit)
        if not rec or not a86.finite(rec.get("mean")):
            fail(f"{where}: doses have no measured mean for L{layer} N{unit}")
            return
        mean = float(rec["mean"])
        if arm in ("M",) or arm.startswith("RM_"):
            for value in (dmin, dmax):
                if abs(value + mean) > TOLERANCE * max(abs(mean), 1e-6) + 2 ** -7 * max(abs(pmax), abs(pmin), 1.0):
                    fail(f"{where}: L{layer} N{unit} {phase} shift {dmin:.5g}..{dmax:.5g}; the M rule registers "
                         f"minus the mean, {-mean:.5g}")
                    break
        else:
            self.near(where, f"L{layer} {phase} write norm", st.get("write_norm"), F_DOSE * median)
            sign = 1.0 if mean > 0 else -1.0 if mean < 0 else 0.0
            if sign == 0.0 or dmin * sign >= 0 or dmax * sign >= 0:
                fail(f"{where}: L{layer} N{unit} {phase} shift {dmin:.4g}..{dmax:.4g} is not against the mean's "
                     f"sign ({mean:.4g})")

    def near(self, where, what, span, target):
        if not span or not a86.finite(*span[:2], target) or target <= 0:
            self.fail(f"{where}: {what} {span!r}, target {target!r}")
            return
        err = max(abs(span[0] - target), abs(span[1] - target)) / target
        self.worst = max(self.worst, err)
        if err > TOLERANCE:
            self.fail(f"{where}: {what} {span[0]:.4g}..{span[1]:.4g}, registered {target:.4g}")

    def check_hooks(self, where, call, arm):
        session = self.sessions.get(call.get("session"), {})
        modules = session.get("down_proj_modules") or {}
        want = sorted(modules.get(str(layer)) for layer in {l for l, _ in self.units_for(arm)})
        got = (call.get("hooks_at_entry") or {})
        names = sorted((got.get("modules") or {}).keys())
        if names != want or got.get("global_pre") or got.get("global_post"):
            self.fail(f"{where}: hooks at entry on {names}; registered only {want}")

    def run(self):
        fail = self.fail
        name, dtype, device = PRODUCTION_MODEL
        for sid, session in self.sessions.items():
            if self.fixture_model:
                break
            if name not in str(session.get("model_name_or_path")) or session.get("dtype") != dtype or \
                    not str(session.get("device", "")).startswith(device):
                fail(f"observer session {sid}: model {session.get('model_name_or_path')!r} {session.get('dtype')!r} "
                     f"on {session.get('device')!r}; registered {name} {dtype} on {device}")
        if abs(self.median(22) - self.l22) > MEDIAN_ANCHOR_TOLERANCE * self.l22:
            fail(f"doses: L22 median {self.median(22):.4g}, neuronary publishes {self.l22}")
        for arm, study in arms(self.phase).items():
            if self.only_arms is not None and arm not in self.only_arms:
                continue
            path = self.run_dir / self.phase / arm / "answers.jsonl"
            if not path.is_file():
                fail(f"{arm}: {path} missing")
                continue
            seen = []
            for index, ans in enumerate(a86.read_jsonl(path)):
                where = f"{self.phase}/{arm} answer {index}"
                question, sample = ans.get("question"), ans.get("sample")
                if ans.get("arm") != arm:
                    fail(f"{where}: arm {ans.get('arm')!r}")
                seen.append((question, sample))
                call, row = self.ctx.claim(where, ans.get("observed"))
                if call is None or row is None:
                    continue
                if call.get("batch") != 1:
                    fail(f"{where}: batch {call.get('batch')!r}; the build generates one answer per call")
                if study == "87A":
                    seed = self.q["seed_a"](self.q["origin"].index(question), sample) \
                        if question in self.q["origin"] and type(sample) is int else None
                    tokens = A_TOKENS
                else:
                    qlist = self.q["maker"] + self.q["guard"]
                    qi = self.q["maker"].index(question) if question in self.q["maker"] else \
                        (25 + self.q["guard"].index(question) if question in self.q["guard"] else None)
                    seed = self.q["seed_b"](f"{self.phase}:{arm}", qi, sample) if qi is not None and \
                        type(sample) is int else None
                    tokens = B_TOKENS
                    if question not in qlist:
                        seed = None
                if seed is None:
                    fail(f"{where}: question {question!r} sample {sample!r} is not registered for {arm}")
                    continue
                # round-2 #2: scoring groups by kind, so the stated index and kind must be the registered ones
                if study == "87A":
                    want_qi, want_kind = self.q["origin"].index(question), "origin"
                elif question in self.q["maker"]:
                    want_qi = self.q["maker"].index(question)
                    want_kind = "direct" if want_qi < self.q["n_direct"] else "indirect"
                else:
                    want_qi, want_kind = 25 + self.q["guard"].index(question), "guard"
                if ans.get("question_index") != want_qi or ans.get("question_kind") != want_kind:
                    fail(f"{where}: stated question {ans.get('question_index')!r}/{ans.get('question_kind')!r}, "
                         f"registered {want_qi}/{want_kind} for {question!r}")
                self.check_call(where, call, question, tokens, seed)
                self.check_hooks(where, call, arm)
                text = ans.get("answer")
                if not isinstance(text, str) or row.get("answer_sha256") != \
                        __import__("hashlib").sha256(text.encode("utf-8")).hexdigest():
                    fail(f"{where}: the saved answer is not the text the observer decoded for this row")
                if row.get("token_count") != ans.get("new_tokens"):
                    fail(f"{where}: new_tokens {ans.get('new_tokens')!r}, observer {row.get('token_count')!r}")
                self.check_change(where, call, row, arm)
            want = expected_cases(self.phase, arm, self.q)
            if sorted(map(str, seen)) != sorted(map(str, want)):
                fail(f"{arm}: {len(seen)} answers, registered {len(want)} distinct question-sample cases "
                     f"({len(set(map(str, seen)) ^ set(map(str, want)))} differ)")
        print(f"audit87 {self.phase}: {len(self.ctx.claimed)} answers checked; worst relative error {self.worst:.4%}")
        return self.fail.items

    def check_call(self, where, call, question, tokens, seed):
        fail = self.fail
        if call.get("error"):
            fail(f"{where}: generate raised {call['error']}")
        # Allowlist, not a list of known-bad settings (round-2 #3, decision part87-after-r2-2026-10-28): only the
        # registered keywords may be passed, nothing positionally, and every other recorded setting is the model's own.
        extra = sorted(set(call.get("explicit_kwargs") or {}) - EXPLICIT_ALLOWED)
        if extra:
            fail(f"{where}: generate was passed unregistered keywords {extra}")
        if call.get("positional_args"):
            fail(f"{where}: generate was passed {call['positional_args']} positional arguments")
        defaults = (self.sessions.get(call.get("session"), {}) or {}).get("generation_defaults")
        eff_all = call.get("effective") or {}
        if not isinstance(defaults, dict) or not defaults:
            fail(f"{where}: the observer recorded no generation defaults for this session")
            defaults = {}
        for key, value in eff_all.items():
            if key in REGISTERED_SETTINGS:
                continue
            if key not in defaults:
                fail(f"{where}: {key} {value!r} at the call has no recorded default to compare with")
            elif value != defaults[key]:
                fail(f"{where}: {key} {value!r} at the call; not registered, so it must be the model's default "
                     f"{defaults[key]!r}")
        if call.get("unobserved_forwards"):
            fail(f"{where}: down_proj forwards the observer did not see: {call['unobserved_forwards']}")
        if call.get("shape_changes"):
            fail(f"{where}: a hook changed a down_proj input's shape: {call['shape_changes']}")
        if call.get("hooks_at_exit") != call.get("hooks_at_entry"):
            fail(f"{where}: hooks changed during the call: {call.get('hooks_at_entry')} -> {call.get('hooks_at_exit')}")
        eff = eff_all
        if eff.get("do_sample") is not True:
            fail(f"{where}: do_sample {eff.get('do_sample')!r}; registered sampling")
        for key, want in (("temperature", TEMPERATURE), ("top_p", TOP_P)):
            if not a86.finite(eff.get(key)) or abs(eff[key] - want) > 1e-9:
                fail(f"{where}: {key} {eff.get(key)!r} at the call, registered {want}")
        if eff.get("max_new_tokens") != tokens:
            fail(f"{where}: max_new_tokens {eff.get('max_new_tokens')!r}, registered {tokens}")
        texts = list((call.get("prompt_texts") or {}).values())
        if texts != [self.ctx.prompt_for(question)]:
            fail(f"{where}: prompt rendered {texts!r}, registered {self.ctx.prompt_for(question)!r}")
        rng = call.get("rng_at_entry") or {}
        if rng.get("cpu_initial_seed") != seed:
            fail(f"{where}: CPU seed at entry {rng.get('cpu_initial_seed')!r}, registered {seed}")
        elif rng.get("cpu_state_sha256") != a86.fresh_rng_sha(seed):
            fail(f"{where}: CPU random numbers were drawn between seeding with {seed} and generating")
        for device in ("xpu", "cuda"):
            if f"{device}_initial_seed" in rng and rng[f"{device}_initial_seed"] != seed:
                fail(f"{where}: {device} seed at entry {rng[f'{device}_initial_seed']!r}, registered {seed}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--phase", choices=("pilot", "run"), required=True)
    ap.add_argument("--doses", default=None)
    ap.add_argument("--prompt-texts", default=None, help="test fixtures only")
    ap.add_argument("--width", type=int, default=None, help="test fixtures only")
    ap.add_argument("--l22-median", type=float, default=None, help="test fixtures only")
    ap.add_argument("--arms", default=None, help="test fixtures only: comma-separated arms to audit")
    ap.add_argument("--fixture-model", action="store_true", default=None,
                    help="test fixtures only: accept a model other than Qwen3.5-4B bf16 on xpu")
    args = ap.parse_args(argv)
    overrides = [k for k in ("prompt_texts", "width", "l22_median", "arms", "fixture_model")
                 if getattr(args, k) is not None]
    run_dir = Path(args.run_dir)
    try:
        doses = json.loads(Path(args.doses or run_dir / "doses.json").read_text(encoding="utf-8"))
        texts = json.loads(Path(args.prompt_texts).read_text(encoding="utf-8")) if args.prompt_texts else None
        failures = Audit(run_dir, args.phase, doses, texts, args.width or WIDTH,
                         args.l22_median if args.l22_median is not None else L22_PUBLISHED_MEDIAN,
                         args.arms.split(",") if args.arms else None, bool(args.fixture_model)).run()
    except (OSError, KeyError, ValueError) as exc:
        print(f"audit87: unreadable input: {exc}")
        return 2
    label = f"AUDIT87 {args.phase}"
    if overrides:
        label += f" (TEST FIXTURE OVERRIDES: {', '.join(overrides)})"
    if failures:
        print(f"{label}: {len(failures)} failures")
        return 1
    print(f"{label}: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
