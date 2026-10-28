#!/usr/bin/env python3
"""replay87.py -- rebuild Part 87 from the design and regenerate a sample of every arm token for token.

Written by Claude (Opus 5.5), 2026-10-28, from PART87-DESIGN.md (Amendments 1 and 2), not from part87.py
(decision part87-after-r2-2026-10-28). It reads part87.py's output files and the earlier parts' registered
constants (Part 84's questions and seed rule, Part 66's questions, guards and seed rule, Part 25's matched-random
rule); it never imports part87.

It checks, in order:
  1. the loaded checkpoint is Qwen/Qwen3.5-4B at the registered revision;
  2. its own layer medians (layers 22, 6, 7, 11, 29: median over token positions 1 onward of the L2 norm of the
     residual leaving the layer, on the calibration text, Part 84's rule) equal the dose file's;
  3. its own Step 1 measurement (Amendment 2, reading 1) of all 36,864 neuron means in layers 6, 7, 11 and 29 agrees
     with the dose file within |mine - file| <= 1e-4 x |file| + 1e-7, with the same token counts;
  4. its own random sets (Part 25's rule, seeds 8700 to 8719) are the dose file's, and its own M and F doses for every
     target and random unit agree within the same tolerance;
  5. every arm the design names has the registered rows (questions, kinds, samples) and nothing else;
  6. for every row of every arm (Amendment 3; before it, a seeded quarter): the observer recorded the row's call at
     batch 1 with the prompt the design renders, the row's stated seed, and the registered eos and pad tokens
     (computed here from the tokenizer); the replay installs its OWN push (its own medians, norms and means; Part 84's
     arithmetic) and regenerates from that seed; the answer text must equal the saved one AND the regenerated token
     ids, trimmed as observe.py trims a row, must have the observer row's token hash and count.
  0. before any of that, Amendment 3's pins, which do not come from the run: Part 66's stored answers by their own
     sha256 and their registered 24 phrasings x 8 samples, and the calibration text by Part 84's recorded digest.
Controls: --perturb-dose F multiplies every additive value it builds by F (a negative control: pushed arms must then
differ); the summary line then says NEGATIVE CONTROL so no gate can mistake it for a pass. --fraction below 1.0 (never
below 0.25) is a non-production override: its summary never says PASS and it exits 3.

Round 3 (Sol, part87-after-r3-2026-10-28) found the first version comparing decoded text only (a changed 87B stop rule
passed, because pad then eos and pad alone decode to the same text), trusting the run's own dose file for the source
and calibration digests, and accepting --fraction 0 with a plain PASS. All three are closed above.

Usage (GPU box): python -I -B part87\\replay87.py --run-dir p87 --phase pilot --calib p66basin\\neuronary-calib.txt
                 --p66-answers results\\part66\\full\\answers.jsonl --device xpu --out p87\\replay\\pilot.json
Exit 0 when everything matches, 1 on any mismatch, 2 on unreadable input, 3 for a matched non-production sample.

Limits: it shares the tokenizer, transformers' generate and sampling code, and neuron_report.load_model with the run,
so a defect inside those is invisible to it. It replays a sample, so a defect touching a rare row can pass. Sampled
regeneration is exact only on the device and library versions the run used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

MODEL_ID = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
ALIEN_LAYER, ALIEN_UNIT, CONTROL_UNIT = 22, 2742, 8870
TARGETS = ((6, 753), (7, 1169), (11, 8868), (29, 776))
MAKER_LAYERS = (6, 7, 11, 29)
MEDIAN_LAYERS = (22, 6, 7, 11, 29)
RANDOM_SEEDS = tuple(range(8700, 8720))
TEMPERATURE, TOP_P = 0.7, 0.95
A_TOKENS, B_TOKENS = 64, 48
F_SIZE = 0.75
REPLAY_SEED = 87087
TOL_REL, TOL_ABS = 1e-4, 1e-7
# Amendment 3 pins, independent of anything the run writes: the repository's Part 66 stored answers and the
# calibration text Part 84 recorded (results/part84/main/run.json calib_sha256).
P66_ANSWERS_SHA256 = "70accf5fa3faaaa4733086555460b5e5504487df187b4f1857e1be9d3d1c6635"
CALIB_SHA256 = "afb86522116b9ccd10233826503dad0992dd0283f04ee2a15dabf5f0e9e1ce54"
MIN_FRACTION = 0.25

# The design's arm tables. kind: none | add (one unit, fixed site dose) | zero | rule (the four targets, M or F) |
# random (one random set, M or F). mode: every | decode.
A_ARMS = {
    "A0": {"kind": "none"},
    "E63": {"kind": "add", "unit": (ALIEN_LAYER, ALIEN_UNIT), "dose": 0.63, "mode": "every"},
    "E47": {"kind": "add", "unit": (ALIEN_LAYER, ALIEN_UNIT), "dose": 0.47, "mode": "every"},
    "D47": {"kind": "add", "unit": (ALIEN_LAYER, ALIEN_UNIT), "dose": 0.47, "mode": "decode"},
    "C63": {"kind": "add", "unit": (ALIEN_LAYER, CONTROL_UNIT), "dose": 0.63, "mode": "every"},
}
B_MAIN = {
    "U": {"kind": "none"},
    "Z": {"kind": "zero"},
    "M": {"kind": "rule", "rule": "M"},
    "F": {"kind": "rule", "rule": "F"},
}


def arm_table(phase: str) -> dict:
    arms = {name: dict(spec, study="87A") for name, spec in A_ARMS.items()}
    arms.update({name: dict(spec, study="87B") for name, spec in B_MAIN.items()})
    if phase == "run":
        for seed in RANDOM_SEEDS:
            arms[f"RM_{seed}"] = {"kind": "random", "rule": "M", "seed": seed, "study": "87B"}
            arms[f"RF_{seed}"] = {"kind": "random", "rule": "F", "seed": seed, "study": "87B"}
    return arms


def registered_question(study: str, qi: int):
    from part84 import part84 as p84
    from part66 import part66 as p66
    if study == "87A":
        return p84.QUESTIONS[qi], "origin"
    if 0 <= qi < len(p66.DIRECT):
        return p66.DIRECT[qi], "direct"
    if len(p66.DIRECT) <= qi < len(p66.DIRECT) + len(p66.INDIRECT):
        return p66.INDIRECT[qi - len(p66.DIRECT)], "indirect"
    if 25 <= qi < 25 + len(p66.GUARDS):
        return p66.GUARDS[qi - 25][0], "guard"
    raise KeyError(f"no registered question {qi} in {study}")


def registered_seed(phase: str, name: str, study: str, qi: int, sample: int) -> int:
    """Amendment 2, reading 2: Part 84's seeds for 87A, Part 66's row_seed keyed "<phase>:<arm>" for 87B."""
    from part84 import part84 as p84
    from part66 import part66 as p66
    return p84.seed_for(qi, sample) if study == "87A" else p66.row_seed(f"{phase}:{name}", qi, sample)


def expected_rows(phase: str, name: str, spec: dict):
    """The (question_index, sample) set the design registers for an arm, or None when the design fixes only a count."""
    if spec["study"] == "87A":
        if phase == "pilot":
            return None, 4                          # 1 question x 4 samples; the design does not say which question
        return {(q, s) for q in range(3) for s in range(8)}, 24
    if phase == "pilot":
        return None, 6                              # 3 phrasings x 2 samples, unnamed
    if spec["kind"] == "random":
        return {(q, 0) for q in range(24)}, 24
    rows = {(q, s) for q in range(24) for s in range(4)} | {(25 + g, s) for g in range(6) for s in range(4)}
    return rows, 120


def text_layers(model):
    for path in ("model.language_model.layers", "model.layers", "language_model.model.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise RuntimeError("cannot find the decoder layers")


def render(tok, question: str) -> str:
    return tok.apply_chat_template([{"role": "user", "content": question}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


def my_median(model, tok, layer, text: str, device: str) -> float:
    import torch
    seen = {}

    def grab(_m, _a, output):
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        seen["v"] = float(torch.linalg.vector_norm(hidden[0, 1:].float(), dim=-1).median().item())

    handle = layer.register_forward_hook(grab)
    try:
        ids = tok(text, return_tensors="pt")["input_ids"].to(device)
        with torch.inference_mode():
            model(input_ids=ids, use_cache=False)
    finally:
        handle.remove()
    return seen["v"]


def my_means(model, tok, layers, rows, device: str):
    """Amendment 2, reading 1: pooled mean of every down_proj input channel over every position of every row."""
    import torch
    sums = {li: None for li in MAKER_LAYERS}
    counts = {li: 0 for li in MAKER_LAYERS}

    def grab(li):
        def hook(_m, args):
            x = args[0][0].detach().float().cpu().double()          # [positions, channels], summed on the CPU
            sums[li] = x.sum(dim=0) if sums[li] is None else sums[li] + x.sum(dim=0)
            counts[li] += int(x.shape[0])
            return None
        return hook

    handles = [layers[li].mlp.down_proj.register_forward_pre_hook(grab(li)) for li in MAKER_LAYERS]
    try:
        with torch.inference_mode():
            for row in rows:
                ids = tok(render(tok, row["question"]) + row["answer"], return_tensors="pt",
                          add_special_tokens=False)["input_ids"].to(device)
                model(input_ids=ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return {li: (sums[li] / counts[li]).tolist() for li in MAKER_LAYERS}, counts


def column_norm(layers, li: int, unit: int) -> float:
    import torch
    return float(torch.linalg.vector_norm(layers[li].mlp.down_proj.weight[:, unit].detach().float().cpu()).item())


def site_dose(rule: str, mean: float, norm: float, median: float) -> float:
    if rule == "M":
        return -mean * norm / median
    if rule == "F":
        return -F_SIZE * (1.0 if mean > 0 else -1.0 if mean < 0 else 0.0)
    raise ValueError(rule)


def additive(site: float, median: float, norm: float) -> float:
    return site * median / norm


def close(mine: float, theirs: float) -> bool:
    return abs(mine - theirs) <= TOL_REL * abs(theirs) + TOL_ABS


class Push:
    """Part 84's arithmetic: add a Python float to one channel of the down_proj input at the mode's positions."""

    def __init__(self, down_proj, unit: int, value: float, mode: str):
        self.unit, self.value, self.mode, self.prompt_pass = unit, value, mode, True
        self.handle = down_proj.register_forward_pre_hook(self.hook)

    def hook(self, _m, args):
        x = args[0]
        first, self.prompt_pass = self.prompt_pass, False
        if self.mode == "decode" and first:
            return None
        y = x.clone()
        y[..., self.unit] = y[..., self.unit] + self.value
        return (y,) + tuple(args[1:])


class Zero:
    def __init__(self, down_proj, unit: int):
        self.unit = unit
        self.handle = down_proj.register_forward_pre_hook(self.hook)

    def hook(self, _m, args):
        y = args[0].clone()
        y[..., self.unit] = 0
        return (y,) + tuple(args[1:])


def stop_tokens(tok, study: str):
    if study == "87A":
        return [t for t in {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")} if isinstance(t, int) and t >= 0]
    return tok.eos_token_id


def observer_stops(tok, study: str) -> set:
    """observe.py trims a row at the first of the call's eos tokens or the tokenizer's eos; the registered call's set."""
    eos = stop_tokens(tok, study)
    stops = set(eos if isinstance(eos, list) else [eos])
    if isinstance(tok.eos_token_id, int):
        stops.add(tok.eos_token_id)
    return stops


def trim_like_observer(ids, stops: set):
    ids = [int(t) for t in ids]
    for pos, token in enumerate(ids):
        if token in stops:
            return ids[:pos + 1]
    return ids


def ids_sha(ids) -> str:
    """observe.py's encoding of a row's token ids."""
    return hashlib.sha256(json.dumps([int(i) for i in ids], separators=(",", ":")).encode("ascii")).hexdigest()


def stop_setting_problems(effective: dict, study: str, tok) -> list:
    """The call's recorded eos and pad must be the registered ones, computed here from the tokenizer."""
    problems = []
    eos = (effective or {}).get("eos_token_id")
    want = stop_tokens(tok, study)
    if study == "87A":
        if not isinstance(eos, list) or sorted(eos) != sorted(want):
            problems.append(f"eos_token_id {eos!r} at the call, registered {sorted(want)}")
    elif eos != want:
        problems.append(f"eos_token_id {eos!r} at the call, registered {want!r}")
    if (effective or {}).get("pad_token_id") != tok.pad_token_id:
        problems.append(f"pad_token_id {(effective or {}).get('pad_token_id')!r} at the call, registered "
                        f"{tok.pad_token_id!r}")
    return problems


def row_matches(row: dict, call: dict, mine_text: str, mine_ids) -> tuple:
    """The saved answer must equal the regenerated text, and its observer row must hold the regenerated tokens."""
    obs_rows = call.get("rows") or []
    index = int((row.get("observed") or {}).get("row", 0))
    obs_row = obs_rows[index] if 0 <= index < len(obs_rows) else {}
    tokens_same = obs_row.get("token_ids_sha256") == ids_sha(mine_ids) and obs_row.get("token_count") == len(mine_ids)
    detail = {"text_match": mine_text == row.get("answer"), "tokens_match": tokens_same,
              "observer_token_count": obs_row.get("token_count"), "replayed_token_count": len(mine_ids)}
    return detail["text_match"] and tokens_same, detail


def source_problems(p66_bytes: bytes, rows: list) -> list:
    """Part 66's stored answers, pinned by their own digest and the registered 24 phrasings x 8 samples."""
    from part66 import part66 as p66
    problems = []
    if hashlib.sha256(p66_bytes).hexdigest() != P66_ANSWERS_SHA256:
        problems.append("Part 66 answers file is not the registered one (sha256 differs from Amendment 3's pin)")
    questions = list(p66.DIRECT) + list(p66.INDIRECT)
    cases = [(r.get("question_index"), r.get("sample")) for r in rows]
    if len(cases) != len(set(cases)) or set(cases) != {(q, s) for q in range(len(questions)) for s in range(8)}:
        problems.append(f"Part 66 rows are not the registered {len(questions)} phrasings x 8 samples")
    for r in rows:
        qi = r.get("question_index")
        if type(qi) is not int or not 0 <= qi < len(questions) or r.get("question") != questions[qi]:
            problems.append(f"Part 66 row q{qi!r} s{r.get('sample')!r}: question is not the registered phrasing")
            break
    return problems


def regenerate(model, tok, layers, device: str, study: str, question: str, seed: int, pushes, zeros):
    """pushes: [(layer, unit, value, mode)]; zeros: [(layer, unit)]."""
    import torch
    torch.manual_seed(seed)
    if device == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.manual_seed_all(seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    encoded = tok(render(tok, question), return_tensors="pt", add_special_tokens=False)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    hooks = [Push(layers[li].mlp.down_proj, u, v, m) for li, u, v, m in pushes]
    hooks += [Zero(layers[li].mlp.down_proj, u) for li, u in zeros]
    try:
        with torch.inference_mode():
            out = model.generate(**encoded, max_new_tokens=A_TOKENS if study == "87A" else B_TOKENS,
                                 do_sample=True, temperature=TEMPERATURE, top_p=TOP_P,
                                 eos_token_id=stop_tokens(tok, study), pad_token_id=tok.pad_token_id, use_cache=True)
    finally:
        for h in hooks:
            h.handle.remove()
    ids = trim_like_observer(out[0, encoded["input_ids"].shape[-1]:].tolist(), observer_stops(tok, study))
    return tok.decode(ids, skip_special_tokens=True).strip(), encoded["input_ids"][0].tolist(), ids


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def observer_calls(run_dir: Path, phase: str):
    calls = {}
    for path in sorted((run_dir / "observe").glob(f"{phase}-*.jsonl")):
        for rec in read_jsonl(path):
            if rec.get("event") == "generate":
                calls[rec["call_id"]] = rec
    return calls


def pick_rows(rows, name: str, fraction: float, everything: bool):
    ordered = sorted(rows, key=lambda r: (int(r["question_index"]), int(r["sample"])))
    if everything or len(ordered) <= 2:
        return ordered
    want = max(2, math.ceil(len(ordered) * fraction))
    rng = random.Random(f"{REPLAY_SEED}:{name}")
    middle = ordered[1:-1]
    return [ordered[0], ordered[-1]] + rng.sample(middle, min(want - 2, len(middle)))


def build_arm(spec: dict, medians: dict, norms: dict, means: dict, sets: dict, scale: float):
    """Return (pushes, zeros) for an arm from the replay's own measurements."""
    if spec["kind"] == "none":
        return [], []
    if spec["kind"] == "zero":
        return [], list(TARGETS)
    if spec["kind"] == "add":
        li, u = spec["unit"]
        return [(li, u, additive(spec["dose"], medians[li], norms[(li, u)]) * scale, spec["mode"])], []
    units = TARGETS if spec["kind"] == "rule" else sets[spec["seed"]]
    pushes = []
    for li, u in units:
        site = site_dose(spec["rule"], means[li][u], norms[(li, u)], medians[li])
        pushes.append((li, u, additive(site, medians[li], norms[(li, u)]) * scale, "every"))
    return pushes, []


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--phase", choices=("pilot", "run"), required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--p66-answers", required=True)
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fraction", type=float, default=1.0,
                    help="share of each arm to replay; the registered run replays every row (1.0). Anything less is "
                         "a non-production override (never below 0.25): its summary never says PASS and it exits 3")
    ap.add_argument("--all", action="store_true", help="replay every row (the default since Amendment 3)")
    ap.add_argument("--perturb-dose", type=float, default=1.0, help="negative control only")
    args = ap.parse_args(argv)
    if not MIN_FRACTION <= args.fraction <= 1.0:
        ap.error(f"--fraction must be between {MIN_FRACTION} and 1.0")
    if args.all:
        args.fraction = 1.0
    run_dir = Path(args.run_dir)
    problems = []
    report = {"model": MODEL_ID, "phase": args.phase, "perturb_dose": args.perturb_dose, "problems": problems}

    def finish(code: int):
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
        return code

    try:
        doses = json.loads((run_dir / "doses.json").read_text(encoding="utf-8"))
        p66_rows = [r for r in read_jsonl(Path(args.p66_answers))
                    if r.get("condition") == "unchanged" and r.get("question_kind") in ("direct", "indirect")]
        calib_bytes = Path(args.calib).read_bytes()
        arms = arm_table(args.phase)
        answers = {name: read_jsonl(run_dir / args.phase / name / "answers.jsonl") for name in arms
                   if (run_dir / args.phase / name / "answers.jsonl").exists()}
        calls = observer_calls(run_dir, args.phase)
    except (OSError, ValueError, KeyError) as exc:
        problems.append(f"unreadable input: {exc}")
        print(f"REPLAY87: unreadable input: {exc}")
        return finish(2)
    if len(p66_rows) != 192:
        problems.append(f"Part 66 unchanged held-out answers: {len(p66_rows)}, expected 192")
    problems.extend(source_problems(Path(args.p66_answers).read_bytes(), p66_rows))
    if hashlib.sha256(calib_bytes).hexdigest() != CALIB_SHA256:
        problems.append("calibration text is not Part 84's recorded one (Amendment 3's pin)")
    if doses.get("calibration_sha256") != hashlib.sha256(calib_bytes).hexdigest():
        problems.append("calibration text is not the one the dose file names")
    if doses.get("source_answers_sha256") != hashlib.sha256(Path(args.p66_answers).read_bytes()).hexdigest():
        problems.append("Part 66 answers file is not the one the dose file names")

    # 5. rows the design registers, before any model work
    for name, spec in arms.items():
        rows = answers.get(name)
        if rows is None:
            problems.append(f"{name}: no answers.jsonl")
            continue
        want_set, want_n = expected_rows(args.phase, name, spec)
        got = [(int(r["question_index"]), int(r["sample"])) for r in rows]
        if len(got) != want_n or len(set(got)) != len(got) or (want_set is not None and set(got) != want_set):
            problems.append(f"{name}: rows {sorted(got)[:6]}... ({len(got)}) are not the registered {want_n}")
        for r in rows:
            try:
                question, kind = registered_question(spec["study"], int(r["question_index"]))
            except (KeyError, IndexError) as exc:
                problems.append(f"{name}: {exc}")
                continue
            if r.get("question") != question or r.get("question_kind") != kind:
                problems.append(f"{name} q{r['question_index']}: question or kind is not the registered one")
            if r.get("seed") != registered_seed(args.phase, name, spec["study"], int(r["question_index"]),
                                                int(r["sample"])):
                problems.append(f"{name} q{r['question_index']} s{r['sample']}: seed {r.get('seed')} is not the "
                                f"Amendment 2 seed")
            if args.phase == "pilot" and spec["study"] == "87B" and kind != "direct" and kind != "indirect":
                problems.append(f"{name} q{r['question_index']}: the pilot registers maker phrasings only")
    extra = sorted(p.name for p in (run_dir / args.phase).iterdir() if p.is_dir() and p.name not in arms) \
        if (run_dir / args.phase).is_dir() else []
    if extra:
        problems.append(f"arms the design does not name: {extra}")

    import torch
    import neuron_report as nr
    from part25 import _matched_random
    tok, model = nr.load_model(MODEL_ID, REVISION, args.device)
    model.eval()
    revision = getattr(model.config, "_commit_hash", None)
    if revision is None:
        from transformers.utils import cached_file
        resolved = cached_file(MODEL_ID, "config.json", revision=REVISION, local_files_only=True)
        revision = REVISION if resolved and REVISION in str(resolved).replace("\\", "/") else f"unresolved:{resolved}"
    report["revision_loaded"] = revision
    if revision != REVISION:
        problems.append(f"revision {revision}")
    layers = text_layers(model)
    n_units = int(layers[0].mlp.down_proj.in_features)

    # 2. medians
    calib = calib_bytes.decode("utf-8")
    medians = {li: my_median(model, tok, layers[li], calib, args.device) for li in MEDIAN_LAYERS}
    report["medians"] = {str(li): [medians[li], doses.get("residual_medians", {}).get(str(li))] for li in MEDIAN_LAYERS}
    for li in MEDIAN_LAYERS:
        if doses.get("residual_medians", {}).get(str(li)) != medians[li]:
            problems.append(f"median L{li}: mine {medians[li]!r}, dose file {doses.get('residual_medians', {}).get(str(li))!r}")

    # 3. Step 1 means
    means, counts = my_means(model, tok, layers, p66_rows, args.device)
    worst = 0.0
    for li in MAKER_LAYERS:
        theirs = doses.get("layer_mean_activations", {}).get(str(li), [])
        if len(theirs) != n_units:
            problems.append(f"means L{li}: dose file has {len(theirs)} values, expected {n_units}")
            continue
        bad = [u for u in range(n_units) if not close(means[li][u], theirs[u])]
        worst = max(worst, max(abs(means[li][u] - theirs[u]) for u in range(n_units)))
        if bad:
            problems.append(f"means L{li}: {len(bad)} of {n_units} outside tolerance (first unit {bad[0]})")
        if doses.get("mean_token_counts", {}).get(str(li)) != counts[li]:
            problems.append(f"token count L{li}: mine {counts[li]}, dose file {doses.get('mean_token_counts', {}).get(str(li))}")
    report["means_worst_abs_diff"] = worst
    report["token_counts"] = {str(li): counts[li] for li in MAKER_LAYERS}

    # 4. random sets and doses
    sets = {seed: [tuple(map(int, x)) for x in _matched_random(list(TARGETS), n_units, len(layers), seed)]
            for seed in RANDOM_SEEDS}
    file_sets = {int(item["seed"]): item for item in doses.get("random_sets", [])}
    used = set(TARGETS) | {(ALIEN_LAYER, ALIEN_UNIT), (ALIEN_LAYER, CONTROL_UNIT)}
    for seed, units in sets.items():
        used |= set(units)
        if seed not in file_sets or [tuple(map(int, x)) for x in file_sets[seed]["units"]] != units:
            problems.append(f"random set {seed}: units differ from the dose file")
    norms = {(li, u): column_norm(layers, li, u) for li, u in used}
    for li, u in used:
        theirs = doses.get("column_norms", {}).get(str(li), {}).get(str(u))
        if theirs is None or not close(norms[(li, u)], theirs):
            problems.append(f"column norm L{li} N{u}: mine {norms[(li, u)]}, dose file {theirs}")
    file_entries = {(int(e["layer"]), int(e["unit"])): e for e in doses.get("target_doses", {}).values()}
    for item in doses.get("random_sets", []):
        for e in item["doses"]:
            file_entries.setdefault((int(e["layer"]), int(e["unit"])), e)
    for li, u in sorted(set(TARGETS) | {x for units in sets.values() for x in units}):
        e = file_entries.get((li, u))
        if e is None:
            problems.append(f"dose L{li} N{u}: missing from the dose file")
            continue
        for rule in ("M", "F"):
            mine = site_dose(rule, means[li][u], norms[(li, u)], medians[li])
            if not close(mine, float(e[rule])):
                problems.append(f"dose {rule} L{li} N{u}: mine {mine}, dose file {e[rule]}")
    report["pre_checks_problems"] = len(problems)

    # 6. regenerate a sample of every arm
    report["arms"] = {}
    replayed = mismatched = 0
    for name, spec in arms.items():
        rows = answers.get(name) or []
        pushes, zeros = build_arm(spec, medians, norms, means, sets, args.perturb_dose)
        arm_report = {"rows": len(rows), "replayed": 0, "mismatched": []}
        for row in pick_rows(rows, name, args.fraction, args.fraction >= 1.0):
            question, _ = registered_question(spec["study"], int(row["question_index"]))
            call = calls.get((row.get("observed") or {}).get("call_id"))
            if call is None:
                problems.append(f"{name} q{row['question_index']} s{row['sample']}: no observer record for its call")
                continue
            seed = (call.get("rng_at_entry") or {}).get("cpu_initial_seed")
            if call.get("batch") != 1 or seed != row.get("seed"):
                problems.append(f"{name} q{row['question_index']} s{row['sample']}: observer batch {call.get('batch')}, "
                                f"seed {seed} vs stated {row.get('seed')}")
            if seed is None:
                continue
            for problem in stop_setting_problems(call.get("effective"), spec["study"], tok):
                problems.append(f"{name} q{row['question_index']} s{row['sample']}: {problem}")
            mine, prompt_ids, mine_ids = regenerate(model, tok, layers, args.device, spec["study"], question, int(seed),
                                                    pushes, zeros)
            same, detail = row_matches(row, call, mine, mine_ids)
            prompt_seen = list((call.get("prompt_texts") or {}).values())
            if prompt_seen != [tok.decode(prompt_ids, skip_special_tokens=False)]:
                problems.append(f"{name} q{row['question_index']} s{row['sample']}: observed prompt is not the rendered one")
            arm_report["replayed"] += 1
            replayed += 1
            if not same:
                mismatched += 1
                arm_report["mismatched"].append({"question_index": row["question_index"], "sample": row["sample"],
                                                 "saved": str(row.get("answer"))[:160], "replayed": mine[:160],
                                                 **detail})
        report["arms"][name] = arm_report
        finish(0)
    label = "REPLAY87" + (f" (NEGATIVE CONTROL: dose x {args.perturb_dose})" if args.perturb_dose != 1.0 else "")
    if args.fraction < 1.0:
        label += f" (NON-PRODUCTION OVERRIDE: fraction {args.fraction})"
    report["fraction"] = args.fraction
    for name, a in report["arms"].items():
        print(f"{name}: {a['replayed'] - len(a['mismatched'])}/{a['replayed']} replayed answers match ({a['rows']} rows)")
    for p in problems[:20]:
        print(f"  problem: {p}")
    report["replayed"], report["mismatched"] = replayed, mismatched
    if problems or mismatched:
        print(f"{label}: {mismatched} mismatches in {replayed} replayed answers; {len(problems)} other problems")
        return finish(1)
    if args.fraction < 1.0:
        print(f"{label}: sampled rows matched ({replayed} answers); not a production result")
        return finish(3)
    print(f"{label}: PASS ({replayed} answers match token for token; medians, means, sets, doses and pins match; "
          f"revision {revision})")
    return finish(0)


if __name__ == "__main__":
    sys.exit(main())
