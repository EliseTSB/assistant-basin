#!/usr/bin/env python3
"""run182.py -- Part 182: gates, stage 1 (teacher-forced head screen), stage 1c (ladder) and stage 2 (generation).

  gates    G1 additivity, G2 no-op, G3 block, on a fixed batch of real rows, float32 on the CPU; also times the
           stage-1 rows at two batch sizes (the Optimization section's measurement). Writes results/part182/gates.json.
  stage1   every condition of headswap182.stage1_conditions over the stage-1 rows; appends to stage1.jsonl, resumable.
  ladder   stage 1c: the cumulative top-k conditions from score182's ranking (needs stage1.jsonl complete).
  stage2   the Part 99 generation battery under each run of score182's selection (needs score182.json); writes
           results/part182/stage2/<tag>.jsonl in Part 99's row format, so part99/label_items99.py and the readers
           apply unchanged.

Host: OLMo-2-0425-1B-Instruct (48d788e). Donor: OLMo-2-0425-1B, the mid-training end (f83cd64). Both loaded in bf16
(as Parts 99 and 100 ran them) and computed in float32 with the eager attention kernel. The pushes are Part 99's fixed
vectors (results/part99-pushes.pt); the answers teacher-forced in stage 1 are Part 99's inst_none answers to
"Who are you?" (results/part99/inst_none.jsonl).
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _folder in (HERE, ROOT / "part92", ROOT / "part99", ROOT):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import torch  # noqa: E402

import common92 as c  # noqa: E402
import headswap182 as hs  # noqa: E402
import score182  # noqa: E402
from swap99 import QUESTIONS, copy_components, load_kind_map  # noqa: E402

LAYERS = score182.LAYERS
N_HEADS = score182.N_HEADS
MAX_TOKENS = 24
QUESTION = "Who are you?"
RESULTS = ROOT / "results" / "part182"
P99 = ROOT / "results" / "part99"
NEURONS = ROOT / "results" / "part92" / "step2" / "neurons.json"
KINDS = ROOT / "results" / "part92" / "step2" / "kinds.json"
PUSHES = ROOT / "results" / "part99-pushes.pt"
G1_TOL = 1e-3     # max |logit difference| allowed between the head-sum and the whole-layer copy (float32)
G2_TOL = 1e-4     # the same for Instruct as its own donor on a partial head set


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def load_host(path, device, dtype=torch.float32, attn_implementation="eager"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    snapshot = str(Path(path).expanduser())
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(snapshot, dtype=torch.bfloat16, local_files_only=True,
                                                 attn_implementation=attn_implementation)
    model = model.to(dtype=dtype).to(device)
    model.requires_grad_(False)
    model.eval()
    return tokenizer, model


def load_units():
    neurons = json.load(open(NEURONS, encoding="utf-8"))
    kinds = load_kind_map(KINDS)
    units = [n for n in neurons if kinds.get(str(n["id"])) in ("slot", "concept")]
    return units, kinds


def load_p99_answers():
    """Part 99 inst_none answers to "Who are you?" by (id, unit)."""
    out = {}
    for line in open(P99 / "inst_none.jsonl", encoding="utf-8"):
        row = json.loads(line)
        if int(row["question_index"]) == 0:
            out[(str(row["id"]), row["unit"])] = row["answer"]
    return out


def build_stage1_rows(tokenizer, units, pushes, answers, max_tokens=MAX_TOKENS) -> list[dict]:
    """The fixed row list, in a fixed order: per unit its own pushed answer under its push, the paired random
    direction's own answer under the random push, and the unpushed answer's text under the unit's push; then the
    unpushed answer with no push. Rows are sorted by unit id, so batch composition is fixed across conditions."""
    unpushed_text = answers[("unpushed", "none")]
    rows = []
    for n in sorted(units, key=lambda n: str(n["id"])):
        uid, layer_no = str(n["id"]), int(n["layer"])
        for unit, text_kind, text in (("neuron", "own", answers[(uid, "neuron")]),
                                      ("random", "own", answers[(uid, "random")]),
                                      ("neuron", "assistant", unpushed_text)):
            rows.append({"id": uid, "unit": unit, "text": text_kind, "push_layer": layer_no,
                         "push": pushes[f"{uid}|{unit}"].float(),
                         "answer_ids": hs.answer_token_ids(tokenizer, text, max_tokens), "answer": text})
    rows.append({"id": "unpushed", "unit": "none", "text": "assistant", "push_layer": None, "push": None,
                 "answer_ids": hs.answer_token_ids(tokenizer, unpushed_text, max_tokens), "answer": unpushed_text})
    for row in rows:
        if not row["answer_ids"]:
            raise ValueError(f"empty answer for {row['id']}/{row['unit']}")
    return rows


def public(row, result, condition, heads, extra=None):
    out = {"condition": condition, "heads": {str(k): v for k, v in heads.items()}, "id": row["id"],
           "unit": row["unit"], "text": row["text"], "push_layer": row["push_layer"], **result}
    if extra:
        out.update(extra)
    return out


def run_condition(model, tokenizer, layers, donors, heads, prompt, rows, device, batch, hidden) -> list[dict]:
    """Score every row under one head set, with the attachment asserted before and after."""
    if heads:
        hs.attach(model, donors, heads)
        hs.assert_attached(model, heads)
        hs.reset_calls(model)
    try:
        results = hs.score_rows(model, tokenizer, layers, prompt, rows, device, batch, hidden)
        if heads:
            hs.assert_attached(model, heads, min_calls=1)
    finally:
        if heads:
            hs.detach(model)
    if hs.attached_heads(model):
        raise RuntimeError("a head swap is still attached after the condition")
    return results


def gates(args):
    RESULTS.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    units, kinds = load_units()
    answers = load_p99_answers()
    pushes = torch.load(PUSHES, map_location="cpu")
    log(f"loading host {args.host}")
    tokenizer, model = load_host(args.host, args.device)
    layers = c.get_layers(model)
    hidden = int(model.config.hidden_size)
    prompt = c.prompt_ids(tokenizer, QUESTION, "chat")
    rows = build_stage1_rows(tokenizer, units, pushes, answers)
    log(f"{len(rows)} stage-1 rows; prompt {len(prompt)} tokens; answers up to {MAX_TOKENS} tokens")

    # The fixed gate batch: the first four eligible slot units' own answers, two concept units', the unpushed answer
    # with no push, and the first slot unit's random direction (all pushes on where a push exists).
    eligible, _ = score182.eligible_units(score182.P99_LABELS, kinds)
    picked = []
    for uid in eligible["slot"][:4]:
        picked.append(next(r for r in rows if r["id"] == uid and r["unit"] == "neuron" and r["text"] == "own"))
    for uid in eligible["concept"][:2]:
        picked.append(next(r for r in rows if r["id"] == uid and r["unit"] == "neuron" and r["text"] == "own"))
    picked.append(next(r for r in rows if r["id"] == "unpushed"))
    picked.append(next(r for r in rows if r["id"] == eligible["slot"][0] and r["unit"] == "random"))
    pad_id = int(tokenizer.pad_token_id)
    input_ids, attention_mask = hs.build_batch(prompt, [r["answer_ids"] for r in picked], pad_id, args.device)
    vectors = {}
    for i, r in enumerate(picked):
        if r["push"] is None:
            continue
        vectors.setdefault(int(r["push_layer"]), torch.zeros((len(picked), hidden), dtype=torch.float32))
        vectors[int(r["push_layer"])][i] = r["push"]
    record = {"host": str(args.host), "donor": str(args.donor), "device": args.device, "dtype": "float32",
              "attn_implementation": "eager", "torch": torch.__version__,
              "transformers": __import__("transformers").__version__, "python": platform.python_version(),
              "cpu": platform.processor() or platform.machine(), "threads": args.threads,
              "batch_rows": [{"id": r["id"], "unit": r["unit"], "text": r["text"], "n_tokens": len(r["answer_ids"])}
                             for r in picked],
              "tolerances": {"G1": G1_TOL, "G2_partial": G2_TOL, "G3": G1_TOL}, "gates": {}}

    def forward():
        return hs.logits_on_batch(model, input_ids, attention_mask, layers, vectors)

    log("plain logits")
    plain = forward()
    # G4 (reported): replay of the stored answers under no swap.
    tf = hs.logprobs_from_logits(plain, len(prompt), [r["answer_ids"] for r in picked])
    record["replay_none"] = [{"id": r["id"], "unit": r["unit"], "text": r["text"], "n_tokens": t["n_tokens"],
                              "argmax_match": t["argmax_match"], "logp_sum": t["logp_sum"]}
                             for r, t in zip(picked, tf)]

    # G2: Instruct as its own donor.
    log("G2: self donor")
    self_donors = hs.donor_attention(model, LAYERS)
    hs.attach(model, self_donors, hs.condition_heads("block", LAYERS, N_HEADS))
    g2a = hs.compare_logits(forward(), plain, attention_mask)
    hs.detach(model)
    g2b = {}
    for tag in ("h0.0", "h1.5", "h2.10", "h3.15"):
        hs.attach(model, self_donors, hs.condition_heads(tag, LAYERS, N_HEADS))
        g2b[tag] = hs.compare_logits(forward(), plain, attention_mask)
        hs.detach(model)
    del self_donors
    record["gates"]["G2a_self_donor_all_heads"] = {**g2a, "pass": g2a["bit_exact"]}
    record["gates"]["G2b_self_donor_single_heads"] = {
        **g2b, "pass": all(v["max_abs_diff"] <= G2_TOL and v["argmax_agree"] == v["positions"] for v in g2b.values())}
    log(f"G2a bit_exact={g2a['bit_exact']}; G2b max={max(v['max_abs_diff'] for v in g2b.values()):.2e}")

    # Donor: the whole model in bf16 stays for copy_components (swap99's own code path); its attention for layers 0
    # to 3 is copied out in float32 for the head swap.
    log(f"loading donor {args.donor}")
    from transformers import AutoModelForCausalLM
    donor_model = AutoModelForCausalLM.from_pretrained(str(Path(args.donor).expanduser()), dtype=torch.bfloat16,
                                                       local_files_only=True, attn_implementation="eager")
    donor_model.requires_grad_(False)
    donor_model.eval()
    donors = hs.donor_attention(donor_model, LAYERS, dtype=torch.float32)
    host_backup = {l: copy.deepcopy(layers[l].self_attn) for l in LAYERS}

    def restore_host():
        for l in LAYERS:
            layers[l].self_attn.load_state_dict(host_backup[l].state_dict())

    # G1: all 16 heads of one layer against swap99's copy of that layer.
    g1 = {}
    for l in LAYERS:
        log(f"G1: layer {l}")
        hs.attach(model, donors, hs.condition_heads(f"layer{l}", LAYERS, N_HEADS))
        swapped = forward()
        hs.detach(model)
        copied = copy_components(model, donor_model, "attn", layers={l})
        reference = forward()
        restore_host()
        assert hs.compare_logits(forward(), plain, attention_mask)["bit_exact"], "host restore is not exact"
        g1[f"layer{l}"] = {**hs.compare_logits(swapped, reference, attention_mask),
                           "tensors_copied": copied["tensors_copied"],
                           "effect_vs_plain_max_abs": hs.compare_logits(swapped, plain, attention_mask)["max_abs_diff"]}
    record["gates"]["G1_all_heads_equal_layer_copy"] = {
        **g1, "pass": all(v["max_abs_diff"] <= G1_TOL and v["argmax_agree"] == v["positions"] for v in g1.values())}
    log("G1 max_abs_diff per layer: " + ", ".join(f"{k}={v['max_abs_diff']:.2e}" for k, v in g1.items()))

    # G3: every head of layers 0 to 3 against --layers 0-3.
    log("G3: block")
    hs.attach(model, donors, hs.condition_heads("block", LAYERS, N_HEADS))
    swapped = forward()
    hs.detach(model)
    copied = copy_components(model, donor_model, "attn", layers=set(LAYERS))
    reference = forward()
    restore_host()
    assert hs.compare_logits(forward(), plain, attention_mask)["bit_exact"], "host restore is not exact"
    g3 = {**hs.compare_logits(swapped, reference, attention_mask), "tensors_copied": copied["tensors_copied"],
          "effect_vs_plain_max_abs": hs.compare_logits(swapped, plain, attention_mask)["max_abs_diff"]}
    record["gates"]["G3_block_equals_layers_0_3_copy"] = {
        **g3, "pass": g3["max_abs_diff"] <= G1_TOL and g3["argmax_agree"] == g3["positions"]}
    log(f"G3 max_abs_diff={g3['max_abs_diff']:.2e} argmax {g3['argmax_agree']}/{g3['positions']}")
    del donor_model
    gc.collect()

    # Throughput on the real stage-1 rows at two batch sizes, with one single-head swap attached (the stage-1 cost).
    timing = {}
    for batch in args.time_batches:
        heads = hs.condition_heads("h1.0", LAYERS, N_HEADS)
        started = time.perf_counter()
        results = run_condition(model, tokenizer, layers, donors, heads, prompt, rows, args.device, batch, hidden)
        seconds = time.perf_counter() - started
        timing[str(batch)] = {"rows": len(results), "seconds": round(seconds, 2),
                              "rows_per_s": round(len(results) / seconds, 3)}
        log(f"batch {batch}: {len(results)} rows in {seconds:.1f}s = {len(results) / seconds:.2f} rows/s")
    record["timing_stage1_rows_h1.0"] = timing
    record["all_pass"] = all(g["pass"] for g in record["gates"].values())
    out = RESULTS / "gates.json"
    out.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    log(f"all_pass={record['all_pass']}; written {out}")
    return 0 if record["all_pass"] else 1


def _existing_keys(path):
    keys = set()
    if path.exists():
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            keys.add((r["condition"], str(r["id"]), r["unit"], r["text"]))
    return keys


def _stage1_run(args, conditions: dict[str, dict[int, list[int]]], out_path: Path, stage: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    units, _kinds = load_units()
    answers = load_p99_answers()
    pushes = torch.load(PUSHES, map_location="cpu")
    tokenizer, model = load_host(args.host, args.device)
    layers = c.get_layers(model)
    hidden = int(model.config.hidden_size)
    prompt = c.prompt_ids(tokenizer, QUESTION, "chat")
    rows = build_stage1_rows(tokenizer, units, pushes, answers)
    donors = hs.load_donor_attention(args.donor, LAYERS, dtype=torch.float32)
    existing = _existing_keys(out_path)
    meta = {"stage": stage, "host": str(args.host), "donor": str(args.donor), "device": args.device,
            "dtype": "float32", "attn_implementation": "eager", "batch": args.batch, "max_tokens": MAX_TOKENS,
            "torch": torch.__version__, "transformers": __import__("transformers").__version__,
            "prompt_tokens": len(prompt), "rows_per_condition": len(rows), "conditions": list(conditions)}
    started = time.perf_counter()
    done = 0
    with out_path.open("a", encoding="utf-8") as stream:
        for tag, heads in conditions.items():
            if all((tag, r["id"], r["unit"], r["text"]) in existing for r in rows):
                continue
            t0 = time.perf_counter()
            results = run_condition(model, tokenizer, layers, donors, heads, prompt, rows, args.device, args.batch,
                                    hidden)
            for row, result in zip(rows, results):
                if (tag, row["id"], row["unit"], row["text"]) in existing:
                    continue
                stream.write(json.dumps(public(row, result, tag, heads), ensure_ascii=False) + "\n")
            stream.flush()
            done += 1
            log(f"{tag}: {len(rows)} rows in {time.perf_counter() - t0:.1f}s ({done} conditions, "
                f"{time.perf_counter() - started:.0f}s total)")
    meta["seconds"] = round(time.perf_counter() - started, 1)
    (out_path.with_suffix(".meta.json")).write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    log(f"{stage} complete: {out_path}")
    return 0


def stage1(args):
    conditions = {t: hs.condition_heads(t, LAYERS, N_HEADS) for t in hs.stage1_conditions(LAYERS, N_HEADS)}
    if args.limit:
        conditions = dict(list(conditions.items())[:args.limit])
    return _stage1_run(args, conditions, RESULTS / "stage1.jsonl", "stage1")


def ladder(args):
    rows = score182.read_jsonl(RESULTS / "stage1.jsonl")
    kinds = load_kind_map(KINDS)
    eligible, _ = score182.eligible_units(score182.P99_LABELS, kinds)
    table = score182.effects(rows, eligible["slot"], eligible["concept"])
    ranking = score182.rank_heads(table)
    conditions = score182.ladder_conditions(ranking)
    (RESULTS / "ladder-ranking.json").write_text(json.dumps({"ranking": [f"h{l}.{h}" for l, h in ranking],
                                                             "conditions": {k: {str(l): v for l, v in c_.items()}
                                                                            for k, c_ in conditions.items()}},
                                                            indent=1) + "\n", encoding="utf-8")
    # Ladder rows go into the same file, so score182 reads them with the rest.
    return _stage1_run(args, conditions, RESULTS / "stage1.jsonl", "stage1c")


def stage2(args):
    """Part 99's battery (six questions, neuron and random pushes, the six unpushed answers) under each selected
    head set, greedy, 120 new tokens, in Part 99's row format."""
    selection = json.load(open(args.selection, encoding="utf-8"))["selection"]
    out_dir = RESULTS / "stage2"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    units, _kinds = load_units()
    pushes = torch.load(PUSHES, map_location="cpu")
    tokenizer, model = load_host(args.host, args.device)
    layers = c.get_layers(model)
    donors = hs.load_donor_attention(args.donor, LAYERS, dtype=torch.float32)
    for run in selection["runs"]:
        tag = run["tag"]
        heads = {int(k): [int(h) for h in v] for k, v in run["heads"].items()}
        output = out_dir / f"{tag}.jsonl"
        existing = set()
        if output.exists():
            for line in open(output, encoding="utf-8"):
                r = json.loads(line)
                existing.add((str(r["id"]), r["unit"], int(r["question_index"])))
        work = []
        for n in units:
            for unit in ("neuron", "random"):
                for qi in range(len(QUESTIONS)):
                    if (str(n["id"]), unit, qi) not in existing:
                        work.append((n, unit, qi))
        for qi in range(len(QUESTIONS)):
            if ("unpushed", "none", qi) not in existing:
                work.append(({"id": "unpushed", "layer": int(units[0]["layer"])}, "none", qi))
        if not work:
            log(f"{tag}: complete")
            continue
        by_layer = {}
        for item in work:
            by_layer.setdefault(int(item[0]["layer"]), []).append(item)
        started = time.perf_counter()
        if heads:
            hs.attach(model, donors, heads)
            hs.assert_attached(model, heads)
            hs.reset_calls(model)
        try:
            with output.open("a", encoding="utf-8") as stream:
                for layer_no, items in by_layer.items():
                    for begin in range(0, len(items), args.batch):
                        chunk = items[begin:begin + args.batch]
                        vectors = torch.stack([
                            (pushes[f'{n["id"]}|{unit}'].float() if unit != "none"
                             else torch.zeros_like(pushes[f'{units[0]["id"]}|neuron'].float()))
                            for n, unit, _qi in chunk])
                        answers = c.generate_batch(model, tokenizer, [QUESTIONS[qi] for _n, _u, qi in chunk], "chat",
                                                   layers[layer_no], vectors, 120, args.device)
                        for (n, unit, qi), answer in zip(chunk, answers):
                            stream.write(json.dumps({"id": n["id"], "unit": unit, "question_index": qi,
                                                     "question": QUESTIONS[qi], "answer": answer},
                                                    ensure_ascii=False) + "\n")
                        stream.flush()
            if heads:
                hs.assert_attached(model, heads, min_calls=1)
        finally:
            hs.detach(model)
        meta = {"tag": tag, "heads": run["heads"], "host": str(args.host), "donor": str(args.donor),
                "device": args.device, "dtype": "float32", "attn_implementation": "eager", "batch": args.batch,
                "max_new_tokens": 120, "frame": "chat", "rows": len(work),
                "seconds": round(time.perf_counter() - started, 1), "torch": torch.__version__,
                "transformers": __import__("transformers").__version__}
        (out_dir / f"{tag}.meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
        log(f"{tag}: {len(work)} rows in {meta['seconds']}s")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("gates", "stage1", "ladder", "stage2"))
    ap.add_argument("--host", required=True, help="local Instruct snapshot directory")
    ap.add_argument("--donor", required=True, help="local mid-training-end snapshot directory")
    ap.add_argument("--device", choices=("cpu", "xpu", "cuda"), default="cpu")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, help="stage1: only the first N conditions (smoke)")
    ap.add_argument("--time-batches", type=int, nargs="+", default=[16, 32], help="gates: batch sizes to time")
    ap.add_argument("--selection", default=str(RESULTS / "score182.json"), help="stage2: score182 output")
    args = ap.parse_args(argv)
    if args.batch < 1:
        ap.error("--batch must be positive")
    return {"gates": gates, "stage1": stage1, "ladder": ladder, "stage2": stage2}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
