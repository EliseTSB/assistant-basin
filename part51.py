#!/usr/bin/env python3
"""Part 51 runners and scorers, reusing Part 49's model and clamp helpers."""
from __future__ import annotations

import argparse
import math
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import fastgen
import part38
import part49


ROOT = Path(__file__).resolve().parent
PART51A_DATA_PATH = ROOT / "data" / "part51a-everyday-prompts.json"
PART51B_DATA_PATH = ROOT / "data" / "part51b-name-prompts.json"
PART51A_DEFAULT_OUT = ROOT / "results" / "part51a"
PART51B_DEFAULT_OUT = ROOT / "results" / "part51b"

PART51A_MAGIC_STEMS = (
    "spell", "magic", "wand", "sorcer", "enchant", "potion", "arcane", "mystic",
)
PART51A_MAGIC_PATTERNS = {
    stem: re.compile(rf"\b{re.escape(stem)}\w*", flags=re.IGNORECASE | re.UNICODE)
    for stem in PART51A_MAGIC_STEMS
}
PART51A_BOOTSTRAP_SEED = 5102
PART51B_RANDOM_SEED = 5101
PART51B_BOOTSTRAP_SEED = 5103
BOOTSTRAP_RESAMPLES = 10_000

NONAME = "NONAME"
EXCLUDED_FIRST_WORDS = {
    "The", "A", "An", "In", "Once", "On", "It", "There", "When", "As",
    "At", "After", "Every", "One",
}
FIRST_WORD_RE = re.compile(r"^([A-Za-z]+)")
VALID_NAME_RE = re.compile(r"^[A-Z][a-z]+$")
LEADING_THINK_RE = re.compile(r"^\s*<think\b[^>]*>.*?</think\s*>\s*", re.IGNORECASE | re.DOTALL)
LEADING_MARKS_RE = re.compile(r"^[\s\"'“”‘’*#]+")


def load_prompts(path: str | Path) -> list[str]:
    data = part49.read_json(path)
    prompts = data.get("prompts", [])
    if len(prompts) != 16 or any(not isinstance(prompt, str) or not prompt.strip()
                                 for prompt in prompts):
        raise ValueError("Part 51 prompt data must contain sixteen nonempty prompts")
    normalized = [" ".join(prompt.casefold().split()) for prompt in prompts]
    if len(set(normalized)) != len(normalized):
        raise ValueError("Part 51 prompts must be unique")
    return prompts


def build_part51a_arm_plan() -> list[dict[str, Any]]:
    """Return the six registered 51a arms before natural maxima are attached."""
    primary = {"layer": 22, "unit": 1610}
    random_units = [
        {"id": "R1", "layer": 22, "unit": 569},
        {"id": "R2", "layer": 22, "unit": 3102},
        {"id": "R3", "layer": 22, "unit": 5258},
    ]
    return [
        {"id": "no_clamp", "k": None, "clamps": []},
        {"id": "L22:1610:k+8", "k": 8, "clamps": [dict(primary)]},
        {"id": "L22:1610:k+4", "k": 4, "clamps": [dict(primary)]},
        *({"id": row["id"], "k": 8,
          "clamps": [{"layer": row["layer"], "unit": row["unit"]}]}
          for row in random_units),
    ]


def choose_part51b_random_units(widths: dict[int, int], seed: int = PART51B_RANDOM_SEED
                                ) -> list[dict[str, Any]]:
    """Draw one +8 and one -8 unit at each registered layer from true MLP widths."""
    rng = random.Random(int(seed))
    selected = []
    for layer in (6, 14, 22):
        width = int(widths[layer])
        if width < 2:
            raise ValueError(f"layer {layer} must have at least two MLP units")
        positive, negative = rng.sample(range(width), 2)
        selected.extend([
            {"id": f"L{layer}:k+8", "layer": layer, "unit": positive, "k": 8},
            {"id": f"L{layer}:k-8", "layer": layer, "unit": negative, "k": -8},
        ])
    return selected


def build_part51b_arm_plan(random_units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(random_units) != 6:
        raise ValueError("Part 51b requires six random units")
    coordinates = [(int(row["layer"]), int(row["unit"])) for row in random_units]
    expected_layers = {6: {8, -8}, 14: {8, -8}, 22: {8, -8}}
    observed: dict[int, set[int]] = {}
    for row in random_units:
        observed.setdefault(int(row["layer"]), set()).add(int(row["k"]))
    if len(set(coordinates)) != 6 or observed != expected_layers:
        raise ValueError("Part 51b needs distinct units and one +8/-8 arm at each layer")
    arms = [
        {"id": "no_clamp_A", "replicate": "A", "k": None, "clamps": []},
        {"id": "no_clamp_B", "replicate": "B", "k": None, "clamps": []},
    ]
    arms.extend({"id": row["id"], "k": int(row["k"]),
                 "clamps": [{"layer": int(row["layer"]), "unit": int(row["unit"])}]}
                for row in random_units)
    if len(arms) != 8 or len({row["id"] for row in arms}) != 8:
        raise AssertionError("Part 51b must have eight uniquely named arms")
    return arms


def part51a_answer_measure(answer: str, prompt: str) -> dict[str, Any]:
    """Count registered magic word starts per thousand answer words."""
    answer, prompt = str(answer), str(prompt)
    excluded = {stem for stem, pattern in PART51A_MAGIC_PATTERNS.items()
                if pattern.search(prompt)}
    matches = sum(len(pattern.findall(answer)) for stem, pattern in PART51A_MAGIC_PATTERNS.items()
                  if stem not in excluded)
    words = len(re.findall(r"\b\w+\b", answer, flags=re.UNICODE))
    return {
        "matches": int(matches), "word_count": int(words),
        "per_1000": float(matches * 1000.0 / words) if words else 0.0,
        "excluded_stems": sorted(excluded),
    }


def extract_first_name(answer: str) -> str:
    """Apply the registered opening cleanup and return a valid first name or NONAME."""
    text = str(answer)
    text = LEADING_THINK_RE.sub("", text, count=1)
    text = LEADING_MARKS_RE.sub("", text, count=1)
    match = FIRST_WORD_RE.match(text)
    if not match:
        return NONAME
    name = match.group(1)
    if not VALID_NAME_RE.fullmatch(name) or name in EXCLUDED_FIRST_WORDS:
        return NONAME
    return name


def _bootstrap_interval(values: np.ndarray) -> list[float] | None:
    return part49._bootstrap_interval(values)


def score_push51a_rows(rows: Iterable[dict[str, Any]], arms: list[dict[str, Any]] | None = None,
                       seed: int = PART51A_BOOTSTRAP_SEED,
                       resamples: int = BOOTSTRAP_RESAMPLES) -> dict[str, Any]:
    """Score 51a per-prompt density means and paired prompt bootstrap differences."""
    rows = list(rows)
    results = [row for row in rows if row.get("record_type") == "result"]
    if arms is None:
        header = next((row for row in rows if row.get("record_type") == "header"), {})
        arms = header.get("arms", [])
    arm_by_id = {str(arm["id"]): arm for arm in arms}
    if "no_clamp" not in arm_by_id:
        raise ValueError("Part 51a scoring requires no_clamp")
    if resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    baseline_rows = [row for row in results if row.get("arm") == "no_clamp"]
    prompt_ids = sorted({str(row.get("prompt_id")) for row in baseline_rows})
    if not prompt_ids:
        raise ValueError("Part 51a results have no no_clamp prompts")

    grouped: dict[str, dict[str, list[float]]] = {
        arm_id: {prompt_id: [] for prompt_id in prompt_ids} for arm_id in arm_by_id
    }
    broken: dict[str, list[bool]] = {arm_id: [] for arm_id in arm_by_id}
    for row in results:
        arm_id, prompt_id = str(row.get("arm")), str(row.get("prompt_id"))
        if arm_id not in arm_by_id:
            raise ValueError(f"Part 51a result has an unknown arm: {arm_id}")
        if prompt_id not in grouped[arm_id]:
            raise ValueError(f"Part 51a arm {arm_id} has a prompt outside no_clamp: {prompt_id}")
        measure = part51a_answer_measure(row.get("answer", ""), row.get("prompt", ""))
        grouped[arm_id][prompt_id].append(measure["per_1000"])
        if isinstance(row.get("broken"), bool):
            broken[arm_id].append(row["broken"])

    prompt_values: dict[str, np.ndarray] = {}
    answer_counts: dict[str, int] = {}
    for arm_id in arm_by_id:
        missing = [prompt_id for prompt_id in prompt_ids if not grouped[arm_id][prompt_id]]
        if missing:
            raise ValueError(f"Part 51a arm {arm_id} is missing prompts: {missing}")
        prompt_values[arm_id] = np.asarray(
            [float(np.mean(grouped[arm_id][prompt_id])) for prompt_id in prompt_ids],
            dtype=np.float64,
        )
        answer_counts[arm_id] = sum(len(grouped[arm_id][prompt_id]) for prompt_id in prompt_ids)

    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(prompt_ids), size=(int(resamples), len(prompt_ids)))
    boot_means = {arm_id: values[indices].mean(axis=1)
                  for arm_id, values in prompt_values.items()}
    baseline = prompt_values["no_clamp"]
    baseline_boot = boot_means["no_clamp"]
    summaries = []
    for arm in arms:
        arm_id = str(arm["id"])
        values = prompt_values[arm_id]
        mean = float(values.mean())
        difference = float((values - baseline).mean())
        difference_boot = boot_means[arm_id] - baseline_boot
        labels = broken[arm_id]
        broken_hits = sum(labels)
        summaries.append({
            "arm": arm_id, "clamps": arm.get("clamps", []),
            "answer_count": answer_counts[arm_id], "prompt_count": len(prompt_ids),
            "mean_per_1000": mean,
            "mean_bootstrap95": _bootstrap_interval(boot_means[arm_id]),
            "difference_from_no_clamp": {
                "value": difference, "bootstrap95": _bootstrap_interval(difference_boot),
            },
            "per_prompt": [
                {"prompt_id": prompt_id, "mean_per_1000": float(values[index])}
                for index, prompt_id in enumerate(prompt_ids)
            ],
            "broken": {"hits": broken_hits, "n": len(labels),
                       "rate": broken_hits / len(labels) if labels else None,
                       "pending": not bool(labels)},
        })
    return {
        "schema_version": 1, "summary_type": "part51a_magic_word_density",
        "arm_count": len(arms), "result_count": len(results),
        "measure": {
            "stems": list(PART51A_MAGIC_STEMS),
            "matching": r"\b(?:spell|magic|wand|sorcer|enchant|potion|arcane|mystic)\w* (case-insensitive)",
            "prompt_exclusion": "an answer stem is excluded when that stem occurs in its prompt",
            "word_count": "Unicode regex \\b\\w+\\b",
            "resampling_unit": "prompt", "bootstrap_resamples": int(resamples),
            "bootstrap_seed": int(seed), "prompt_count": len(prompt_ids),
        },
        "arms": summaries,
    }


def _share(stock: np.ndarray, named: np.ndarray) -> np.ndarray:
    return np.divide(stock, named, out=np.full(stock.shape, np.nan, dtype=np.float64),
                     where=named != 0)


def _entropy_bits(counts: Counter[str]) -> float | None:
    total = sum(counts.values())
    if not total:
        return None
    return float(-sum((count / total) * math.log2(count / total)
                      for count in counts.values() if count))


def score_push51b_rows(rows: Iterable[dict[str, Any]], arms: list[dict[str, Any]] | None = None,
                       seed: int = PART51B_BOOTSTRAP_SEED,
                       resamples: int = BOOTSTRAP_RESAMPLES) -> dict[str, Any]:
    """Score 51b names using replicate A's stock set and paired prompt bootstrap."""
    rows = list(rows)
    results = [row for row in rows if row.get("record_type") == "result"]
    if arms is None:
        header = next((row for row in rows if row.get("record_type") == "header"), {})
        arms = header.get("arms", [])
    arm_by_id = {str(arm["id"]): arm for arm in arms}
    required = {"no_clamp_A", "no_clamp_B"}
    if not required.issubset(arm_by_id):
        raise ValueError("Part 51b scoring requires no_clamp_A and no_clamp_B")
    if resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    baseline_b_rows = [row for row in results if row.get("arm") == "no_clamp_B"]
    prompt_ids = sorted({str(row.get("prompt_id")) for row in baseline_b_rows})
    if not prompt_ids:
        raise ValueError("Part 51b results have no no_clamp_B prompts")

    extracted: dict[str, dict[str, list[str]]] = {
        arm_id: {prompt_id: [] for prompt_id in prompt_ids} for arm_id in arm_by_id
    }
    for row in results:
        arm_id, prompt_id = str(row.get("arm")), str(row.get("prompt_id"))
        if arm_id not in arm_by_id:
            raise ValueError(f"Part 51b result has an unknown arm: {arm_id}")
        if prompt_id not in extracted[arm_id]:
            raise ValueError(f"Part 51b arm {arm_id} has a prompt outside no_clamp_B: {prompt_id}")
        extracted[arm_id][prompt_id].append(extract_first_name(row.get("answer", "")))

    missing = {
        arm_id: [prompt_id for prompt_id in prompt_ids if not extracted[arm_id][prompt_id]]
        for arm_id in arm_by_id
    }
    missing = {arm_id: prompt_list for arm_id, prompt_list in missing.items() if prompt_list}
    if missing:
        raise ValueError(f"Part 51b arms are missing prompt rows: {missing}")

    stock_counts = Counter(name for names in extracted["no_clamp_A"].values()
                           for name in names if name != NONAME)
    stock_names = sorted(stock_counts, key=lambda name: (-stock_counts[name], name))[:5]
    stock_set = set(stock_names)
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(prompt_ids), size=(int(resamples), len(prompt_ids)))

    arrays: dict[str, dict[str, np.ndarray]] = {}
    name_counts: dict[str, Counter[str]] = {}
    summaries = []
    for arm in arms:
        arm_id = str(arm["id"])
        noname_by_prompt = np.asarray([
            sum(name == NONAME for name in extracted[arm_id][prompt_id])
            for prompt_id in prompt_ids
        ], dtype=np.float64)
        named_by_prompt = np.asarray([
            sum(name != NONAME for name in extracted[arm_id][prompt_id])
            for prompt_id in prompt_ids
        ], dtype=np.float64)
        stock_by_prompt = np.asarray([
            sum(name in stock_set for name in extracted[arm_id][prompt_id])
            for prompt_id in prompt_ids
        ], dtype=np.float64)
        arrays[arm_id] = {
            "noname": noname_by_prompt, "named": named_by_prompt, "stock": stock_by_prompt,
            "total": noname_by_prompt + named_by_prompt,
        }
        counts = Counter(name for names in extracted[arm_id].values()
                         for name in names if name != NONAME)
        name_counts[arm_id] = counts

    boot_shares: dict[str, np.ndarray] = {}
    shares: dict[str, float | None] = {}
    for arm_id, arm_arrays in arrays.items():
        stock = arm_arrays["stock"]
        named = arm_arrays["named"]
        shares[arm_id] = (float(stock.sum() / named.sum()) if named.sum() else None)
        boot_shares[arm_id] = _share(stock[indices].sum(axis=1), named[indices].sum(axis=1))

    baseline_share = shares["no_clamp_B"]
    baseline_boot = boot_shares["no_clamp_B"]
    for arm in arms:
        arm_id = str(arm["id"])
        arm_arrays = arrays[arm_id]
        total_answers = float(arm_arrays["total"].sum())
        noname_share = float(arm_arrays["noname"].sum() / total_answers) if total_answers else None
        difference = (shares[arm_id] - baseline_share
                      if shares[arm_id] is not None and baseline_share is not None else None)
        difference_boot = boot_shares[arm_id] - baseline_boot
        counts = name_counts[arm_id]
        top_names = sorted(counts, key=lambda name: (-counts[name], name))[:8]
        summaries.append({
            "arm": arm_id, "clamps": arm.get("clamps", []),
            "answer_count": int(total_answers),
            "noname_share": noname_share,
            "stock_count": int(arm_arrays["stock"].sum()),
            "named_answer_count": int(arm_arrays["named"].sum()),
            "stock_share_among_named": shares[arm_id],
            "stock_share_difference_from_no_clamp_B": {
                "value": difference, "bootstrap95": _bootstrap_interval(difference_boot),
            },
            "name_entropy_bits": _entropy_bits(counts),
            "top_8_names": [{"name": name, "count": int(counts[name])} for name in top_names],
        })

    random_arm_ids = [str(arm["id"]) for arm in arms
                      if arm["id"] not in ("no_clamp_A", "no_clamp_B")]
    pooled_stock = sum((arrays[arm_id]["stock"] for arm_id in random_arm_ids),
                       np.zeros(len(prompt_ids), dtype=np.float64))
    pooled_named = sum((arrays[arm_id]["named"] for arm_id in random_arm_ids),
                       np.zeros(len(prompt_ids), dtype=np.float64))
    pooled_share = float(pooled_stock.sum() / pooled_named.sum()) if pooled_named.sum() else None
    pooled_boot = _share(pooled_stock[indices].sum(axis=1), pooled_named[indices].sum(axis=1))
    pooled_difference = (pooled_share - baseline_share
                         if pooled_share is not None and baseline_share is not None else None)
    pooled_difference_boot = pooled_boot - baseline_boot
    return {
        "schema_version": 1, "summary_type": "part51b_stock_name_attractor",
        "arm_count": len(arms), "result_count": len(results),
        "measure": {
            "name_rule": "first cleaned answer word matching ^[A-Z][a-z]+$, excluding registered openers",
            "noname_value": NONAME,
            "stock_set_rule": "five most common named answers in no_clamp_A; ties alphabetical",
            "stock_names": [{"name": name, "count_in_A": int(stock_counts[name])}
                            for name in stock_names],
            "stock_set_source_arm": "no_clamp_A",
            "name_entropy_population": "named answers only",
            "resampling_unit": "prompt", "bootstrap_resamples": int(resamples),
            "bootstrap_seed": int(seed), "prompt_count": len(prompt_ids),
        },
        "arms": summaries,
        "pooled_random_arms": {
            "arm_count": len(random_arm_ids), "arms": random_arm_ids,
            "stock_count": int(pooled_stock.sum()), "named_answer_count": int(pooled_named.sum()),
            "stock_share_among_named": pooled_share,
            "difference_from_no_clamp_B": {
                "value": pooled_difference,
                "bootstrap95": _bootstrap_interval(pooled_difference_boot),
            },
        },
    }


def _materialize_arms(arm_plan: list[dict[str, Any]], natural: dict[str, Any]
                      ) -> list[dict[str, Any]]:
    arms = []
    for planned in arm_plan:
        arm = {key: value for key, value in planned.items() if key not in {"k", "replicate"}}
        k = planned.get("k")
        if arm["clamps"]:
            if k is None:
                raise ValueError(f"clamped arm {planned['id']} has no k value")
            arm = part49._add_natural_clamps([arm], natural, int(k))[0]
        arm.update({key: value for key, value in planned.items() if key in {"k", "replicate"}})
        arms.append(arm)
    return arms


def _sample_rows(prompts: list[str], args: argparse.Namespace, *, replicate_b_seed: int | None = None
                 ) -> dict[str, list[dict[str, Any]]]:
    rows_a = part49.build_sample_rows(prompts, args.samples, args.seed, "part51-push-sample")
    result = {"A": rows_a}
    if replicate_b_seed is not None:
        result["B"] = part49.build_sample_rows(
            prompts, args.samples, replicate_b_seed, "part51-push-sample"
        )
    return result


def _run_push(args: argparse.Namespace, command: str) -> dict[str, Any]:
    prompts = load_prompts(args.data)
    selected_prompts = prompts[:args.limit] if args.limit is not None else prompts
    torch, tokenizer, model, layers, _snapshot = part38.load_model(args)

    layer_widths: dict[int, int] = {}
    if command == "push51a":
        arm_plan = build_part51a_arm_plan()
        coordinates = sorted({(int(clamp["layer"]), int(clamp["unit"]))
                              for arm in arm_plan for clamp in arm["clamps"]})
        layer_widths = part38.layer_widths(
            layers, [{"layer": layer, "unit": unit} for layer, unit in coordinates]
        )
        random_units = [
            {"layer": 22, "unit": 569}, {"layer": 22, "unit": 3102},
            {"layer": 22, "unit": 5258},
        ]
        replicate_b_seed = None
        output_name = "push51a"
    else:
        if len(layers) <= 22:
            raise ValueError("Part 51b requires loaded model layers 6, 14 and 22")
        layer_widths = {
            layer: int(layers[layer].mlp.down_proj.in_features) for layer in (6, 14, 22)
        }
        random_units = choose_part51b_random_units(layer_widths, PART51B_RANDOM_SEED)
        arm_plan = build_part51b_arm_plan(random_units)
        coordinates = sorted({(int(clamp["layer"]), int(clamp["unit"]))
                              for arm in arm_plan for clamp in arm["clamps"]})
        replicate_b_seed = (int(args.replicate_b_seed) if args.replicate_b_seed is not None
                            else int(args.seed) + 1)
        if replicate_b_seed == int(args.seed):
            raise ValueError("replicate B must use a different sample seed from replicate A")
        output_name = "push51b"

    coordinate_rows = [{"layer": layer, "unit": unit} for layer, unit in coordinates]
    args.out.mkdir(parents=True, exist_ok=True)
    natural, natural_path = part49._load_or_capture_natural_max(
        args, torch, tokenizer, model, layers, coordinate_rows, args.out
    )
    arms = _materialize_arms(arm_plan, natural)
    sample_rows = _sample_rows(selected_prompts, args, replicate_b_seed=replicate_b_seed)
    if args.limit is None and args.samples == part49.DEFAULT_SAMPLES:
        expected = 64
        if any(len(sample_rows[replicate]) != expected
               for replicate in sample_rows):
            raise AssertionError("the full Part 51 design requires 64 sample rows per arm")

    header: dict[str, Any] = {
        "schema_version": 1, "record_type": "header", "command": command,
        "model": args.model, "revision": args.revision, "device": args.device,
        "dtype": args.dtype, "seed": int(args.seed), "data_sha256": part49.sha256_file(args.data),
        "arms": arms, "natural_max_path": str(natural_path),
        "natural_max_sha256": part49.sha256_file(natural_path) if natural_path.is_file() else None,
        "prompt_count": len(selected_prompts), "samples_per_prompt": int(args.samples),
        "answer_count_per_arm": len(selected_prompts) * int(args.samples),
        "generation": {"temperature": part49.TEMPERATURE, "top_p": 1.0,
                       "max_new_tokens": part49.MAX_NEW_TOKENS, "batch_size": int(args.batch)},
    }
    if command == "push51a":
        header.update({
            "assay": "everyday_magic_density", "random_unit_source": "fixed Part 49b coordinates",
            "random_units": random_units,
            "same_row_seeds_across_arms": True,
        })
    else:
        header.update({
            "assay": "stock_name_attractor", "random_unit_seed": PART51B_RANDOM_SEED,
            "random_units": random_units,
            "layer_widths": {str(layer): width for layer, width in sorted(layer_widths.items())},
            "replicate_seeds": {"A": int(args.seed), "B": replicate_b_seed},
            "random_arms_reuse_replicate_A_row_seeds": True,
        })

    output_path = args.out / f"{output_name}.jsonl"
    rows = part38.prepare_jsonl(output_path, header)
    done = part38.completed_keys(rows)
    timings: dict[str, Any] = {}
    for arm in arms:
        replicate = str(arm.get("replicate", "A"))
        requests = part49.arm_requests([arm], sample_rows[replicate])[arm["id"]]
        pending = [sample for sample in requests if sample["row_key"] not in done]
        operations = part49.operations_for_arm(arm, arm.get("k"))
        started = time.perf_counter()
        generated = 0
        for batch_start in range(0, len(pending), args.batch):
            batch = pending[batch_start:batch_start + args.batch]
            rendered_prompts = [part38.chat_text(tokenizer, sample["prompt"]) for sample in batch]
            outputs = fastgen.batched_generate(
                model, tokenizer, rendered_prompts,
                max_new_tokens=part49.MAX_NEW_TOKENS, operations=operations,
                do_sample=True, row_seeds=[int(sample["seed"]) for sample in batch],
                temperature=part49.TEMPERATURE, device=args.device, generated_only=True,
                padding_side=args.padding_side,
            )
            for sample, output in zip(batch, outputs):
                row = {
                    "record_type": "result", "row_key": sample["row_key"],
                    "arm": arm["id"], "prompt_id": sample["prompt_id"],
                    "sample": sample["sample"], "seed": sample["seed"],
                    "prompt": sample["prompt"], "answer": output["text"],
                    "broken": None, "clamps": arm["clamps"],
                }
                part38.append_jsonl(output_path, row)
                done.add(row["row_key"])
                generated += 1
        elapsed = time.perf_counter() - started
        timings[arm["id"]] = {"seconds": elapsed, "generated": generated,
                              "batch_size": int(args.batch)}
        print(f"{command} arm {arm['id']}: generated {generated}/{len(requests)} answers "
              f"in {elapsed:.2f}s (batch={args.batch})", flush=True)

    final_rows = part38.read_jsonl(output_path)
    scorer = score_push51a_rows if command == "push51a" else score_push51b_rows
    scores = scorer(final_rows, arms)
    scores.update({
        "model": args.model, "revision": args.revision, "seed": int(args.seed),
        "random_unit_seed": (None if command == "push51a" else PART51B_RANDOM_SEED),
        "random_units": random_units,
        "layer_widths": {str(layer): width for layer, width in sorted(layer_widths.items())},
        "timings": timings, "jsonl": str(output_path),
        "natural_max_path": str(natural_path),
    })
    score_path = args.out / f"{output_name}-scores.json"
    part49.write_json_atomic(scores, score_path)
    print(part49.json.dumps({
        "jsonl": str(output_path), "scores": str(score_path),
        "answer_count_per_arm": len(selected_prompts) * int(args.samples),
        "arm_count": len(arms), "random_units": random_units, "timings": timings,
    }, ensure_ascii=False, indent=2))
    return scores


def add_push_arguments(parser: argparse.ArgumentParser, *, data_path: Path, out_path: Path) -> None:
    part49.add_model_arguments(parser)
    parser.add_argument("--data", type=Path, default=data_path)
    parser.add_argument("--part35-data", type=Path, default=part49.PART35_DATA_PATH)
    parser.add_argument("--natural-max", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=out_path)
    parser.add_argument("--limit", type=int, default=None, help="number of prompts to run")
    parser.add_argument("--samples", type=int, default=part49.DEFAULT_SAMPLES,
                        help="samples per prompt; full design uses four")
    parser.add_argument("--batch", type=int, default=16, help="generation batch size")
    parser.add_argument("--padding-side", choices=("left", "right"), default="left")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    push51a = sub.add_parser("push51a", help="run the Part 51 everyday magic-density assay")
    add_push_arguments(push51a, data_path=PART51A_DATA_PATH, out_path=PART51A_DEFAULT_OUT)
    push51b = sub.add_parser("push51b", help="run the Part 51 stock-name assay")
    add_push_arguments(push51b, data_path=PART51B_DATA_PATH, out_path=PART51B_DEFAULT_OUT)
    push51b.add_argument("--replicate-b-seed", type=int, default=None,
                         help="sample seed for baseline replicate B; defaults to --seed + 1")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.threads <= 0 or args.batch <= 0 or args.samples <= 0:
        parser.error("--threads, --batch, and --samples must be positive")
    if args.limit is not None and not 1 <= args.limit <= 16:
        parser.error("--limit must be between 1 and 16")
    if args.revision != part49.MODEL_REVISION:
        parser.error(f"Part 51 requires pinned revision {part49.MODEL_REVISION}")
    try:
        _run_push(args, args.command)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"part51 {args.command}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
