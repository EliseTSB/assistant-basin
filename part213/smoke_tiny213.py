#!/usr/bin/env python3
# Needs part213/tiny-olmo-cfg/tokenizer.json (gitignored, 7 MB): copy it from the OLMo-2-0425-1B-Instruct snapshot
# (revision 48d788eca847...), e.g. cp <hf hub>/models--allenai--OLMo-2-0425-1B-Instruct/snapshots/48d788eca847*/tokenizer.json
# part213/tiny-olmo-cfg/. The other files in that folder are the snapshot's small config files, committed.
"""End-to-end smoke test for Part 213 on a tiny random OLMo 2.

Runs the REAL command-line paths -- `run213.py pilot`, `stage1`, `stage2`, then `score213.py` -- against a tiny
randomly initialised OLMo 2 (16 layers, so the units up to layer 12 sit in it; `intermediate_size` 8192, so neurons up
to 7,850 exist) built from `part213/tiny-olmo-cfg/config.json` with the real tokenizer from that folder. Model loading,
snapshot lookup, provenance, the frozen push vectors and the Part 106 Gate 2 reference are patched to tiny/random/fake
equivalents; nothing else is replaced. The point is the plumbing: every record the runner writes must be readable by
the scorer, every prediction must be scored, and the loaded records must be small.

Usage:
    HF_HUB_OFFLINE=1 python -I -B part213/smoke_tiny213.py <REPO_ROOT> <SCRATCH_DIR>

Prints `SMOKE OK` on success. Never touches the real weights, `results/`, or the network.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path


def fail(message: str) -> None:
    print(f"SMOKE FAILED: {message}", file=sys.stderr)
    raise SystemExit(1)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: smoke_tiny213.py <REPO_ROOT> <SCRATCH_DIR>", file=sys.stderr)
        return 2
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    root = Path(argv[1]).resolve()
    scratch = Path(argv[2]).resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    cfg_dir = root / "part213" / "tiny-olmo-cfg"
    if not cfg_dir.is_dir():
        fail(f"missing tiny config folder {cfg_dir}")
    sys.path.insert(0, str(root))

    import torch
    torch.set_num_threads(2)  # the tiny tensors are faster without 8-thread sync overhead
    from transformers import AutoTokenizer, Olmo2Config, Olmo2ForCausalLM

    import part213.run213 as run213
    import part213.score213 as score213

    targets = json.loads((root / "part213" / "targets213.json").read_text(encoding="utf-8"))
    units = targets["units"]
    units_by_id = {u["id"]: u for u in units}
    hidden = 32

    # --- tiny random model and real tokenizer -------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(str(cfg_dir), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = json.loads((cfg_dir / "config.json").read_text(encoding="utf-8"))
    config.update(hidden_size=hidden, num_attention_heads=2, num_key_value_heads=2, intermediate_size=8192,
                  num_hidden_layers=16, torch_dtype="float32", tie_word_embeddings=False,
                  use_cache=True)  # Stage 2's model.generate is otherwise O(n^2) on the tiny model

    def build_model(seed: int):
        torch.manual_seed(seed)
        return Olmo2ForCausalLM(Olmo2Config(**config)).eval().to("cpu")

    # Pick a tiny model whose four unpushed Stage 2 baselines are not mechanically "broken" (a 3-word n-gram four
    # times) -- Gate 3 is a real baseline-coherence check and random weights are gibberish, so the fixture needs a
    # seed that happens to produce coherent-enough text. Seed 14 is the first hit; the scan keeps this robust across
    # transformers versions. This changes only the fixture, not the runner or scorer.
    model = None
    model_seed = None
    for seed in [14] + [s for s in range(0, 24) if s != 14]:
        candidate = build_model(seed)
        answers = [run213.generate_one(candidate, tokenizer, None, p["question"], None)
                   for p in targets["stage2_free_prompts"]]
        if all(not score213.broken_answer(answer) for answer in answers):
            model, model_seed = candidate, seed
            print(f"tiny model seed {seed}: 4/4 unpushed baselines are not mechanically broken")
            break
    if model is None:
        fail("no tiny-model seed produced four non-broken unpushed Stage 2 baselines")
    model.requires_grad_(False)
    layers = model.model.layers
    print(f"tiny model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, "
          f"{len(layers)} layers, hidden {model.config.hidden_size}")

    # --- a snapshot directory named as a 40-hex commit, with the real tokenizer files ---------------------------
    commit = "48d788eca847" + "0" * 28
    snap = scratch / "snapshots" / commit
    snap.mkdir(parents=True, exist_ok=True)
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                 "generation_config.json", "config.json"):
        shutil.copyfile(cfg_dir / name, snap / name)

    run213.snapshot_dir = lambda ckpt: snap
    run213.load_model_fp32 = lambda snapshot: (tokenizer, model)
    run213.provenance = lambda ckpt, snapshot: {
        "key": ckpt, "stage": "tiny smoke", "repo_id": "tiny/olmo2", "branch": "smoke",
        "commit": commit, "commit_prefix": commit[:12], "resolved_from": "tiny smoke snapshot"}

    # --- random push vectors; Gate 2 anchors get a vector that genuinely orders the three arms ------------------
    generator = torch.Generator().manual_seed(213)
    vectors = {}
    for unit in units:
        for arm in ("neuron", "random_0", "random_1", "random_2", "random_3"):
            vectors[(unit["id"], arm)] = torch.randn(hidden, generator=generator) * 0.02

    def gate2_row(uid: str):
        anchor = next(a["target"] for a in targets["gate2_anchors"] if a["id"] == uid)
        prefix = run213.prompt_ids(tokenizer, "Who are you?")
        anchor_ids = tokenizer.encode(anchor, add_special_tokens=False)
        full = prefix + anchor_ids
        return {"ids": full, "first_pos": len(full) - 1,
                "content_pos": [len(prefix) - 1 + i for i in range(len(anchor_ids))], "target_ids": anchor_ids}

    def anchor_lp(uid: str, vector) -> float:
        row = gate2_row(uid)
        layer = layers[int(units_by_id[uid]["layer"])]
        vecs = None if vector is None else run213.torch_stack([vector])
        return run213.forward_batch(model, tokenizer, [row], layer=layer, vectors=vecs)[0]["lp_sum"]

    part106_rows = []
    for uid in run213.GATE2_ANCHORS:
        unit_layer = layers[int(units_by_id[uid]["layer"])]
        base = anchor_lp(uid, None)
        # finite-difference gradient of the anchor lp_sum w.r.t. the residual push vector
        gradient = torch.zeros(hidden)
        eps = 0.02
        for d in range(hidden):
            step = torch.zeros(hidden)
            step[d] = eps
            gradient[d] = (anchor_lp(uid, step) - anchor_lp(uid, -step)) / (2 * eps)
        direction = gradient / (float(gradient.abs().max()) + 1e-12)
        chosen = None
        for scale in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0):
            neuron = direction * scale
            random = neuron * 0.25
            lp_neuron, lp_random = anchor_lp(uid, neuron), anchor_lp(uid, random)
            if lp_neuron > lp_random > base:
                chosen = (neuron, random, base, lp_neuron, lp_random)
                break
        if chosen is None:
            fail(f"could not construct a Gate 2 ordering for {uid}")
        neuron, random, base, lp_neuron, lp_random = chosen
        vectors[(uid, "neuron")] = neuron
        vectors[(uid, "random_0")] = random
        part106_rows += [{"id": uid, "unit": "none", "alpha": 0.0, "lp_sum": base},
                         {"id": uid, "unit": "neuron", "alpha": 0.8, "lp_sum": lp_neuron},
                         {"id": uid, "unit": "random", "alpha": 0.8, "lp_sum": lp_random}]
        print(f"gate2 {uid}: none={base:.3f} random={lp_random:.3f} neuron={lp_neuron:.3f}")
    run213.load_push_vectors = lambda units_arg, model_arg, checkpoint: vectors

    # The fake Part 106 file carries exactly what the tiny model produced, so Gate 2 is a real re-read of its own rows.
    part106_path = scratch / "lp-instruct.jsonl"
    part106_path.write_text("\n".join(json.dumps(r) for r in part106_rows) + "\n", encoding="utf-8")
    run213.PART106_INSTRUCT = part106_path

    # --- the real command-line paths ----------------------------------------------------------------------------
    started = time.perf_counter()
    pilot_dir = scratch / "pilot"
    print("\n=== run213.py pilot ===", flush=True)
    run213.main(["pilot", "--out", str(pilot_dir), "--batch", "16"])
    pilot = json.loads((pilot_dir / "pilot_summary.json").read_text(encoding="utf-8"))
    if not pilot["gate1"].get("passed"):
        fail(f"pilot Gate 1 did not pass: {pilot['gate1']}")
    if not pilot["gate2"].get("passed"):
        fail(f"pilot Gate 2 did not pass: {json.dumps(pilot['gate2'])[:800]}")

    records = scratch / "records"
    print("\n=== run213.py stage1 ===", flush=True)
    run213.main(["stage1", "--out", str(records), "--batch", "8", "--pilot", str(pilot_dir)])
    counts = json.loads((records / "run-instruct.json").read_text(encoding="utf-8"))
    if counts.get("stage1_passes_executed") != 6366:
        fail(f"instruct Stage 1 executed {counts.get('stage1_passes_executed')} passes, not 6,366")
    for ckpt in ("sft", "dpo", "instruct"):
        if not (records / f"run-{ckpt}.json").is_file():
            fail(f"missing run-{ckpt}.json")
    if (records / "run.json").exists():
        fail("a single overwritten run.json was written")

    print("\n=== run213.py stage2 ===", flush=True)
    run213.main(["stage2", "--out", str(records), "--batch", "8"])
    stage2_manifest = json.loads((records / "run-stage2.json").read_text(encoding="utf-8"))
    if stage2_manifest.get("stage2_rows") != 472 or stage2_manifest.get("stage2_executed") != 364:
        fail(f"Stage 2 wrote {stage2_manifest.get('stage2_rows')} rows / executed "
             f"{stage2_manifest.get('stage2_executed')}, not 472 / 364")

    print("\n=== score213.py ===", flush=True)
    score_out = records / "score.json"
    score213.main(["--records", str(records), "--pilot", str(pilot_dir), "--out", str(score_out)])
    result = json.loads(score_out.read_text(encoding="utf-8"))

    # --- assertions ---------------------------------------------------------------------------------------------
    for gate in ("gate1", "gate2", "gate3"):
        if not result["gates"][gate].get("passed"):
            fail(f"{gate} did not pass: {json.dumps(result['gates'][gate])[:800]}")
    if result["gates"]["gate4"].get("passed"):
        fail("Gate 4 passed without a blind audit; it must require labels")

    allowed = {"held", "failed", "inconclusive", "not read"}
    for name, pred in result["predictions"].items():
        verdict = pred.get("verdict")
        if verdict == "untestable (gate failed)":
            fail(f"{name} is untestable (gate failed)")
        if verdict not in allowed:
            fail(f"{name} has no scored verdict: {verdict!r} ({pred.get('error')})")
        if "error" in pred:
            fail(f"{name} errored: {pred['error']}")
    print("\nverdicts:", {k: v["verdict"] for k, v in result["predictions"].items()})

    # F8: records stay small and no row carries a full-vocabulary logit row.
    total_bytes = sum(p.stat().st_size for p in records.glob("*.jsonl"))
    max_keys = 0
    for path in records.glob("stage1-*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            logits = row.get("first_logits")
            if isinstance(logits, dict):
                max_keys = max(max_keys, len(logits))
            elif logits:
                max_keys = max(max_keys, len(logits))
            if row.get("lp_by_unit"):
                fail("a record still carries lp_by_unit")
    limit = 1 + max(len(u["offdiag_ou"]) for u in units)
    if max_keys > limit:
        fail(f"a row kept {max_keys} first logits (limit {limit})")
    print(f"\nrecords: {total_bytes / 1e6:.1f} MB on disk, max {max_keys} first logits/row "
          f"(full vocabulary would be {len(tokenizer)})")
    if total_bytes > 400_000_000:
        fail(f"records are {total_bytes / 1e6:.0f} MB; a full run would not load")

    print(f"\nSMOKE OK ({time.perf_counter() - started:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
