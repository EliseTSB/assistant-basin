#!/usr/bin/env python3
"""replay86.py -- regenerate a sample of the Part 86 screen from the design alone and compare answers token for token.

Written by Claude (Opus 5.5), 2026-10-28, from PART86-DESIGN.md (with Amendments 1 to 3), not from screen86.py or
common86.py (decision part86-after-r5-2026-10-28). Five review rounds found wiring defects passing every check on what
the run recorded about itself, then on what an observer recorded about the run. This checks the property the result
rests on instead: the saved answers are the answers the registered recipe produces. A defect anywhere (sampling
settings, prompt, checkpoint, calibration, push sign, size, layer or step coverage, batch row order) changes the
regenerated text; a defect that changes no answer cannot change the result.

For each layer it:
  1. checks the loaded checkpoint is Qwen/Qwen3.5-4B at the registered revision;
  2. recomputes the layer's median, the design's rule: median over token positions 1 onward of the L2 norm of the
     residual leaving the layer, on the registered calibration text, and requires it to equal the median the run saved;
  3. picks generate calls from the observer files (the first, the last and three seeded at random), takes the rows each
     call produced from the answers that claim it, in row order, rebuilds every row's push itself (a neuron: dose x
     median / ||W_down[:, n]|| added to channel n; a random direction: the registered seed 86000 + layer, standard
     normal on the CPU, unit length, scaled so W_down @ v has norm dose x median), adds it to the down_proj input on
     generated tokens only, renders "Who are you?" with the chat template (no system turn, thinking off), and generates
     greedily for at most 48 new tokens;
  4. requires each regenerated answer to equal the saved one.
The screen replay is a SAMPLE (first, last and three seeded calls per layer): it gives sampled assurance beside
audit86's whole-population checks, not a certificate of every screen answer (Amendment 5).

Verification (--verify-dir, Amendments 4 and 5): every one of the 5,280 answers is regenerated at its registered seed
with the replay's own median and push and compared with the saved text, the record's token count, and the observer's
token hash and count for that row; each call's stop tokens must be the registered set; malformed, duplicate or
mislabelled records, pick files and identities are refused before anything is compared; and every pick's layer,
neuron and persona must match the selected file (--picked), whose digest every pick's config must carry. A fraction
below 1 is a non-production sample: it exits 3 and never prints PASS.

Controls: --perturb-median F multiplies its own median by F (a negative control: the answers must then differ); it
changes the summary line so no gate can mistake it for a pass.

Usage (GPU box): python -I -B part86\\replay86.py --screen-dir p86\\screen --calib p66basin\\neuronary-calib.txt
                 --layers 22 --device xpu --out p86\\replay\\layer22.json
Exit 0 when every replayed answer and median matches, 1 otherwise, 2 on unreadable input.

What it got wrong the first time (2026-10-28 03:49, on the layer-22 pilot): it read the checkpoint revision from
config._commit_hash, which Qwen3.5-4B's config does not carry (None), so it reported a revision failure while all 320
answers matched. It now resolves the revision from the cache's snapshot path. Limits: it shares the tokenizer, the
transformers generate loop and neuron_report.load_model with the run, so a defect inside those is invisible to it; and
it replays a sample (first, last and three seeded calls per layer, 320 answers), so a defect touching a rare row can
pass. Positive control 320/320; negative control (median x 1.01) changed 52 of 128 answers. Round 6 (Sol, 2026-10-28)
showed a one-answer verification defect passing the old quarter sample, a persona relabel passing, and malformed
records skipping checks; all three are closed above.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

MODEL_ID = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
DOSE = 0.8
QUESTION = "Who are you?"
TOKENS = 48
SCREEN_SEED = 86000
N_RANDOM = 512
REPLAY_SEED = 86086
VERIFY_QUESTIONS = ("Who are you?", "Tell me about yourself.", "What are you?")
VERIFY_SEED = 86500            # + 100 x question + sample
VERIFY_RANDOM_SEED = 86900     # + 10000 x layer + neuron (Amendment 4)
VERIFY_DIRECTIONS = 10
VERIFY_TOKENS = 64
TEMPERATURE, TOP_P = 0.7, 0.95
VERIFY_REPLAY_SEED = 86586     # + 10000 x layer + neuron


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


def my_median(model, tok, layer, text: str, device: str) -> float:
    import torch
    norms = []

    def grab(_m, _a, output):
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        norms.append(torch.linalg.vector_norm(hidden[0, 1:].float(), dim=-1))

    handle = layer.register_forward_hook(grab)
    try:
        ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        with torch.inference_mode():
            model(input_ids=ids, use_cache=False)
    finally:
        handle.remove()
    return float(norms[0].median().item())


class DecodePush:
    """Adds row i's vector to the down_proj input on every generated-token step; the prompt pass is left alone."""

    def __init__(self, module, pushes):
        self.pushes = pushes
        self.handle = module.register_forward_pre_hook(self.hook)

    def hook(self, _m, args):
        x = args[0]
        if x.shape[-2] > 1:
            return None
        return (x + self.pushes.to(device=x.device, dtype=x.dtype).unsqueeze(1),) + tuple(args[1:])


def scaled_directions(w, count: int, seed: int, target: float):
    """The design's random-direction rule: standard normal on the CPU, unit length, scaled so W_down @ v has norm target."""
    import torch
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    raw = torch.randn((count, int(w.shape[1])), generator=generator, dtype=torch.float32)
    raw = raw / torch.linalg.vector_norm(raw, dim=1, keepdim=True)
    raw = raw.to(device=w.device)
    return raw * (target / torch.linalg.vector_norm(raw @ w.T, dim=1)).unsqueeze(1)


def my_pushes(rows, weight, median: float, layer_no: int, device, scale: float = 1.0):
    import torch
    w = weight.detach().float()
    width = int(w.shape[1])
    target = DOSE * median * scale
    random_scaled = scaled_directions(w, N_RANDOM, SCREEN_SEED + layer_no, target)
    out = torch.zeros((len(rows), width), dtype=torch.float32, device=w.device)
    for i, row in enumerate(rows):
        if row["kind"] == "neuron":
            n = int(row["id"])
            out[i, n] = target / torch.linalg.vector_norm(w[:, n])
        elif row["kind"] == "random_direction":
            out[i] = random_scaled[int(row["id"])]
        else:
            raise ValueError(f"unknown row kind {row['kind']!r}")
    return out.to(device)


def stop_ids(model, tok):
    ids = set()
    for value in (getattr(model.generation_config, "eos_token_id", None), tok.eos_token_id,
                  tok.convert_tokens_to_ids("<|im_end|>")):
        for v in (value if isinstance(value, (list, tuple)) else [value]):
            if isinstance(v, int) and v >= 0:
                ids.add(v)
    return sorted(ids)


def calls_by_layer(screen_dir: Path, layer_no: int):
    answers = [json.loads(line) for line in (screen_dir / f"layer_{layer_no}.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip()]
    calls = defaultdict(dict)
    for answer in answers:
        claim = answer.get("observed") or {}
        calls[claim["call_id"]][int(claim["row"])] = answer
    ordered = sorted(calls, key=lambda cid: (cid.split(":")[0], int(cid.split(":")[1])))
    return answers, ordered, calls


def pick(ordered, layer_no: int, k: int = 3):
    rng = random.Random(REPLAY_SEED + layer_no)
    middle = ordered[1:-1]
    chosen = [ordered[0], ordered[-1]] + (rng.sample(middle, min(k, len(middle))) if middle else [])
    return list(dict.fromkeys(chosen))


def replay_layer(model, tok, layers, args, layer_no: int, calib: str):
    import torch
    result = {"layer": layer_no, "calls": [], "mismatches": 0, "replayed": 0}
    answers, ordered, calls = calls_by_layer(Path(args.screen_dir), layer_no)
    run_medians = sorted({a["median"] for a in answers})
    median = my_median(model, tok, layers[layer_no], calib, args.device)
    result["median_mine"] = median
    result["median_run"] = run_medians
    result["median_match"] = run_medians == [median]
    use_median = median * args.perturb_median
    prompt = tok.apply_chat_template([{"role": "user", "content": QUESTION}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
    prompt_ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(args.device)
    stops = stop_ids(model, tok)
    down = layers[layer_no].mlp.down_proj
    for call_id in pick(ordered, layer_no, args.random_calls):
        rows = [calls[call_id][i] for i in sorted(calls[call_id])]
        if sorted(calls[call_id]) != list(range(len(rows))):
            result["calls"].append({"call_id": call_id, "error": "rows are not 0..n-1"})
            result["mismatches"] += 1
            continue
        pushes = my_pushes(rows, down.weight, use_median, layer_no, args.device)
        hook = DecodePush(down, pushes)
        try:
            with torch.inference_mode():
                out = model.generate(input_ids=prompt_ids.expand(len(rows), -1).contiguous(), max_new_tokens=TOKENS,
                                     do_sample=False, eos_token_id=stops, pad_token_id=tok.pad_token_id,
                                     use_cache=True)
        finally:
            hook.handle.remove()
        gen = out[:, prompt_ids.shape[-1]:].tolist()
        detail = []
        for i, row in enumerate(rows):
            ids = gen[i]
            for pos, t in enumerate(ids):
                if t in stops:
                    ids = ids[:pos + 1]
                    break
            mine = tok.decode(ids, skip_special_tokens=True).strip()
            same = mine == row["answer"]
            result["replayed"] += 1
            if not same:
                result["mismatches"] += 1
                detail.append({"row": i, "kind": row["kind"], "id": row["id"], "saved": row["answer"][:160],
                               "replayed": mine[:160]})
        result["calls"].append({"call_id": call_id, "rows": len(rows), "mismatched_rows": detail})
    return result


def registered_stops(model, tok):
    """Amendment 5: the stop set is every eos id the model config, its text config and its generation config name,
    plus the tokenizer's eos and <|im_end|>."""
    ids = set()
    config = getattr(model, "config", None)
    for holder in (config, getattr(config, "text_config", None), getattr(model, "generation_config", None)):
        value = getattr(holder, "eos_token_id", None) if holder is not None else None
        for v in (value if isinstance(value, (list, tuple)) else [value]):
            if isinstance(v, int) and not isinstance(v, bool) and v >= 0:
                ids.add(v)
    for v in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")):
        if isinstance(v, int) and not isinstance(v, bool) and v >= 0:
            ids.add(v)
    return sorted(ids)


def ids_sha(ids) -> str:
    """The observer's token hash encoding (observe.py _ids_sha), written out again here."""
    return hashlib.sha256(json.dumps([int(i) for i in ids], separators=(",", ":")).encode("ascii")).hexdigest()


def regenerate_verify(model, tok, down, push, seed: int, question: str, device: str, stops):
    """One verification answer: its own sampled call at its registered seed, the push on generated tokens only.
    Returns (decoded text, generated token ids up to and including the first stop token)."""
    import torch
    prompt = tok.apply_chat_template([{"role": "user", "content": question}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
    prompt_ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    torch.manual_seed(int(seed))
    if device == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.manual_seed_all(int(seed))
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    hook = DecodePush(down, push.unsqueeze(0))
    try:
        with torch.inference_mode():
            out = model.generate(input_ids=prompt_ids, max_new_tokens=VERIFY_TOKENS, do_sample=True,
                                 temperature=TEMPERATURE, top_p=TOP_P, eos_token_id=stops,
                                 pad_token_id=tok.pad_token_id, use_cache=True)
    finally:
        hook.handle.remove()
    ids = out[0, prompt_ids.shape[-1]:].tolist()
    for pos, t in enumerate(ids):
        if t in stops:
            ids = ids[:pos + 1]
            break
    return tok.decode(ids, skip_special_tokens=True).strip(), [int(i) for i in ids]


def verify_cases():
    cases = {("neuron", None, q, s) for q in range(3) for s in range(8)}
    return cases | {("random_direction", d, q, s) for d in range(VERIFY_DIRECTIONS) for q in range(3) for s in range(8)}


def is_int(value) -> bool:
    return type(value) is int


def case_of(record):
    """The record's case as a tuple of strictly typed fields, or None when any field is malformed."""
    cond, d, q, s_ = record.get("condition"), record.get("direction"), record.get("question_index"), record.get("sample")
    if cond == "neuron" and d is None:
        pass
    elif cond == "random_direction" and is_int(d) and 0 <= d < VERIFY_DIRECTIONS:
        pass
    else:
        return None
    if not (is_int(q) and 0 <= q < len(VERIFY_QUESTIONS) and is_int(s_) and 0 <= s_ < 8):
        return None
    return cond, d, q, s_


def pick_verify(records, layer_no: int, neuron: int, fraction: float):
    """Every record at fraction >= 1 (the production setting); otherwise first, last and a seeded share."""
    ordered = sorted(records, key=lambda r: (r.get("condition") != "neuron", -1 if r.get("direction") is None
                                             else int(r["direction"]), int(r["question_index"]), int(r["sample"])))
    if fraction >= 1.0 or len(ordered) <= 2:
        return ordered
    want = max(2, math.ceil(len(ordered) * fraction))
    rng = random.Random(VERIFY_REPLAY_SEED + 10000 * layer_no + neuron)
    return [ordered[0], ordered[-1]] + rng.sample(ordered[1:-1], min(want - 2, len(ordered) - 2))


def load_picked(path: Path):
    """The selected file (verify86 run --picked): a JSON list of {layer, neuron, persona}; returns (rows, sha256)."""
    raw = path.read_bytes()
    rows = json.loads(raw.decode("utf-8"))
    if not isinstance(rows, list):
        raise ValueError("picked file is not a JSON list")
    picks = {}
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or not is_int(row.get("layer")) or not is_int(row.get("neuron")) or \
                not isinstance(row.get("persona"), str) or not row["persona"].strip():
            raise ValueError(f"picked row {i} is malformed")
        key = (row["layer"], row["neuron"])
        if key in picks:
            raise ValueError(f"picked row {i} repeats L{key[0]} N{key[1]}")
        picks[key] = row["persona"].strip()
    return picks, hashlib.sha256(raw).hexdigest()


def observer_rows(verify_dir: Path):
    """generate calls recorded by ../observe.py under <verify_dir>/observe/, by call id."""
    calls = {}
    for path in sorted((verify_dir / "observe").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if record.get("event") == "generate":
                    calls[record.get("call_id")] = record
    return calls


def replay_verify(model, tok, layers, args, calib: str):
    import torch
    result = {"picks": [], "replayed": 0, "mismatches": 0, "problems": []}
    problems = result["problems"]
    verify_dir = Path(args.verify_dir)
    try:
        picks, picked_sha = load_picked(Path(args.picked))
    except (OSError, ValueError) as exc:
        problems.append(f"picked file: {exc}")
        return result
    result["picked_sha256"] = picked_sha
    if len(picks) != args.verify_picks:
        problems.append(f"picked file names {len(picks)} neurons, registered {args.verify_picks}")
    files = sorted(verify_dir.glob("L*_N*.json"))
    if len(files) != args.verify_picks:
        problems.append(f"{len(files)} pick files, registered {args.verify_picks}")
    calls = observer_rows(verify_dir)
    if not calls:
        problems.append(f"{verify_dir}: no observer generate records under observe/")
    stops = registered_stops(model, tok)
    medians, seen = {}, set()
    for path in files:
        match = re.fullmatch(r"L(\d+)_N(\d+)\.json", path.name)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            problems.append(f"{path.name}: unreadable ({exc})")
            continue
        if not match or path.name != f"L{int(match[1])}_N{int(match[2])}.json":
            problems.append(f"{path.name}: not a canonical L<layer>_N<neuron>.json name")
            continue
        layer_no, neuron = int(match[1]), int(match[2])
        if (layer_no, neuron) in seen:
            problems.append(f"{path.name}: a second file for L{layer_no} N{neuron}")
            continue
        seen.add((layer_no, neuron))
        config = payload.get("config") if isinstance(payload, dict) else None
        records = payload.get("records") if isinstance(payload, dict) else None
        if payload.get("status") != "complete" or not isinstance(config, dict) or not isinstance(records, list):
            problems.append(f"{path.name}: status {payload.get('status')!r}, or no config or records")
            continue
        if (config.get("layer"), config.get("neuron")) != (layer_no, neuron):
            problems.append(f"{path.name}: config names L{config.get('layer')} N{config.get('neuron')}")
        persona = picks.get((layer_no, neuron))
        if persona is None:
            problems.append(f"{path.name}: L{layer_no} N{neuron} is not in the picked file")
        elif not isinstance(config.get("persona"), str) or config["persona"].strip() != persona:
            problems.append(f"{path.name}: config persona {config.get('persona')!r}, picked file {persona!r}")
        if config.get("selected_set_sha256") != picked_sha:
            problems.append(f"{path.name}: selected_set_sha256 {config.get('selected_set_sha256')!r} is not the "
                            f"picked file's {picked_sha}")
        if layer_no not in medians:
            medians[layer_no] = my_median(model, tok, layers[layer_no], calib, args.device)
        median = medians[layer_no]
        down = layers[layer_no].mlp.down_proj
        w = down.weight.detach().float()
        target = DOSE * median * args.perturb_median
        directions = scaled_directions(w, VERIFY_DIRECTIONS, VERIFY_RANDOM_SEED + 10000 * layer_no + neuron, target)
        neuron_push = torch.zeros(int(w.shape[1]), dtype=torch.float32, device=w.device)
        neuron_push[neuron] = target / torch.linalg.vector_norm(w[:, neuron])
        good = []
        for r in records:
            where = f"{path.name} {r.get('condition')} d{r.get('direction')!r} q{r.get('question_index')!r} " \
                    f"s{r.get('sample')!r}" if isinstance(r, dict) else f"{path.name} non-object record"
            if not isinstance(r, dict):
                problems.append(where)
                continue
            if r.get("layer") != layer_no or r.get("neuron") != neuron:
                problems.append(f"{where}: record names L{r.get('layer')!r} N{r.get('neuron')!r}")
            if r.get("median") != median or r.get("dose") != DOSE:
                problems.append(f"{where}: median {r.get('median')!r} (mine {median!r}) or dose {r.get('dose')!r}")
            if persona is not None and r.get("persona") != config.get("persona"):
                problems.append(f"{where}: record persona {r.get('persona')!r}, config {config.get('persona')!r}")
            case = case_of(r)
            if case is None:
                problems.append(f"{where}: malformed case fields")
                continue
            _, _, q, s_ = case
            if r.get("seed") != VERIFY_SEED + 100 * q + s_ or r.get("question") != VERIFY_QUESTIONS[q]:
                problems.append(f"{where}: seed or question is not the registered one")
            good.append((case, r))
        got = [case for case, _ in good]
        if len(got) != len(records) or len(set(got)) != len(got) or set(got) != verify_cases():
            problems.append(f"{path.name}: {len(records)} records are not the registered {len(verify_cases())} cases")
        pick = {"file": path.name, "answers": len(records), "replayed": 0, "mismatched": []}
        for r in pick_verify([r for _, r in good], layer_no, neuron, args.fraction):
            q, s_ = r["question_index"], r["sample"]
            push = neuron_push if r["condition"] == "neuron" else directions[r["direction"]]
            mine, ids = regenerate_verify(model, tok, down, push, VERIFY_SEED + 100 * q + s_, VERIFY_QUESTIONS[q],
                                          args.device, stops)
            pick["replayed"] += 1
            result["replayed"] += 1
            observed = r.get("observed") if isinstance(r.get("observed"), dict) else {}
            call = calls.get(observed.get("call_id"))
            rows = (call or {}).get("rows") or []
            row = rows[observed["row"]] if call and is_int(observed.get("row")) and 0 <= observed["row"] < len(rows) \
                else None
            why = []
            if mine != r.get("answer"):
                why.append("text")
            if row is None:
                why.append("no observer row")
            else:
                if call.get("batch") != 1:
                    why.append(f"observer batch {call.get('batch')!r}")
                eos = (call.get("effective") or {}).get("eos_token_id")
                eos_set = sorted(set(eos if isinstance(eos, list) else [eos] if is_int(eos) else []))
                if eos_set != stops:
                    why.append(f"stop tokens {eos_set} at the call, registered {stops}")
                if row.get("token_ids_sha256") != ids_sha(ids) or row.get("token_count") != len(ids):
                    why.append("tokens")
            if r.get("token_count") != len(ids):
                why.append(f"record token_count {r.get('token_count')!r}, replayed {len(ids)}")
            if why:
                result["mismatches"] += 1
                pick["mismatched"].append({"condition": r["condition"], "direction": r["direction"],
                                           "question_index": q, "sample": s_, "why": why,
                                           "saved": str(r.get("answer"))[:160], "replayed": mine[:160]})
        result["picks"].append(pick)
    missing = sorted(set(picks) - seen)
    if missing:
        problems.append(f"picked neurons with no verification file: {missing}")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--screen-dir", default=None)
    ap.add_argument("--verify-dir", default=None, help="replay the verification answers (Amendment 4)")
    ap.add_argument("--verify-picks", type=int, default=20)
    ap.add_argument("--picked", default=None, help="verification: the selected file (verify86 run --picked)")
    ap.add_argument("--fraction", type=float, default=1.0,
                    help="verification: share of each pick's answers; below 1 is a non-production sample")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--layers", default="22")
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--random-calls", type=int, default=3)
    ap.add_argument("--perturb-median", type=float, default=1.0, help="negative control only")
    args = ap.parse_args(argv)
    if bool(args.screen_dir) == bool(args.verify_dir):
        ap.error("give exactly one of --screen-dir and --verify-dir")
    if args.verify_dir and not args.picked:
        ap.error("--verify-dir needs --picked (Amendment 5: personas are bound to the selected file)")
    import neuron_report as nr
    tok, model = nr.load_model(MODEL_ID, REVISION, args.device)
    model.eval()
    revision = getattr(model.config, "_commit_hash", None) or getattr(getattr(model.config, "text_config", None),
                                                                      "_commit_hash", None)
    if revision is None:
        # Qwen3.5-4B's config does not carry a commit hash; the files the loader read for the pinned revision live in
        # the cache's snapshots/<commit>/ folder, so the resolved path names the revision.
        from transformers.utils import cached_file
        resolved = cached_file(MODEL_ID, "config.json", revision=REVISION, local_files_only=True)
        revision = REVISION if resolved and REVISION in str(resolved).replace("\\", "/") else f"unresolved:{resolved}"
    layers = text_layers(model)
    calib = Path(args.calib).read_text(encoding="utf-8")
    if args.verify_dir:
        result = replay_verify(model, tok, layers, args, calib)
        report = {"model": MODEL_ID, "revision_loaded": revision, "revision_ok": revision == REVISION,
                  "perturb_median": args.perturb_median, "verify": result}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
        label = "REPLAY86 verify" + (f" (NEGATIVE CONTROL: median x {args.perturb_median})"
                                     if args.perturb_median != 1.0 else "")
        sample_only = args.fraction < 1.0
        if sample_only:
            label += f" (NON-PRODUCTION SAMPLE: fraction {args.fraction})"
        for pick in result["picks"]:
            print(f"{pick['file']}: {pick['replayed'] - len(pick['mismatched'])}/{pick['replayed']} replayed answers "
                  f"match ({pick['answers']} answers)")
        for problem in result["problems"][:20]:
            print(f"  problem: {problem}")
        bad = (not report["revision_ok"]) + result["mismatches"] + len(result["problems"])
        if bad:
            print(f"{label}: {result['mismatches']} mismatches in {result['replayed']} replayed answers; "
                  f"{len(result['problems'])} other problems (revision ok: {report['revision_ok']})")
            return 1
        if sample_only:
            print(f"{label}: sample agrees ({result['replayed']} answers); not a verification result")
            return 3
        print(f"{label}: PASS (all {result['replayed']} answers match token for token, personas bound to "
              f"{result.get('picked_sha256')}, revision {revision})")
        return 0
    spec = args.layers
    layer_list = list(range(int(spec.split("-")[0]), int(spec.split("-")[1]) + 1)) if "-" in spec else \
        [int(x) for x in spec.split(",")]
    report = {"model": MODEL_ID, "revision_loaded": revision, "revision_ok": revision == REVISION,
              "perturb_median": args.perturb_median, "layers": []}
    for layer_no in layer_list:
        report["layers"].append(replay_layer(model, tok, layers, args, layer_no, calib))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    bad = (not report["revision_ok"]) + sum(r["mismatches"] + (not r["median_match"]) for r in report["layers"])
    replayed = sum(r["replayed"] for r in report["layers"])
    label = "REPLAY86" + (f" (NEGATIVE CONTROL: median x {args.perturb_median})" if args.perturb_median != 1.0 else "")
    for r in report["layers"]:
        print(f"L{r['layer']}: median mine {r['median_mine']:.6f} run {r['median_run']}; "
              f"{r['replayed'] - r['mismatches']}/{r['replayed']} answers match")
    if bad:
        print(f"{label}: {bad} mismatches in {replayed} replayed answers and medians (revision ok: {report['revision_ok']})")
        return 1
    print(f"{label}: PASS (sampled: {replayed} answers match, medians match, revision {revision})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
