#!/usr/bin/env python3
"""Part 49: test whether the wizard unit boosts selected partner units."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import fastgen
import part38


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "part38-homographs-open.json"
NEGATIVE_DATA_PATH = ROOT / "data" / "part49-wizard-prompts.json"
PART49B_DATA_PATH = ROOT / "data" / "part49b-wizard-prompts.json"
SENTENCE_DATA_PATH = ROOT / "data" / "part38-homographs.json"
PART35_DATA_PATH = ROOT / "data" / "part35-concepts.json"
PARTNERS_PATH = ROOT / "data" / "part49-partners.json"
DEFAULT_OUT = ROOT / "results" / "part49"
PART49B_DEFAULT_OUT = ROOT / "results" / "part49b"
MODEL_ID = part38.MODEL_ID
MODEL_REVISION = part38.MODEL_REVISION
WIZARD = {"id": "W", "layer": 19, "unit": 4826}
TOPIC_TERMS = part38.TOPIC_TERMS["wizard"]
PART49B_RANDOM_SEED = 4902
PART49B_BOOTSTRAP_SEED = 4903
PART49B_BOOTSTRAP_RESAMPLES = 10_000
PART49B_MAGIC_TERMS = ("spell", "magic", "wand", "sorcer", "enchant", "potion")
PART49B_MAGIC_PATTERNS = {
    term: re.compile(rf"\b{term}\w*", flags=re.IGNORECASE | re.UNICODE)
    for term in PART49B_MAGIC_TERMS
}
DEFAULT_K = 8
DEFAULT_SAMPLES = 4
MAX_NEW_TOKENS = 80
TEMPERATURE = 0.7


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(value: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{time.time_ns()}.tmp")
    with open(temp, "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    temp.replace(path)


def topic_hit(answer: str, terms: Iterable[str] = TOPIC_TERMS) -> bool:
    """Use Part 38's fixed, case-insensitive word-start topic-prefix matcher."""
    return part38.topic_hit(answer, terms)


def _capture_word_maxima(torch: Any, tokenizer: Any, model: Any, layers: Any,
                         sentence: str, device: str,
                         candidate_layers: Iterable[int]) -> dict[int, np.ndarray]:
    """Capture every candidate coordinate's max over the target word's tokens."""
    rendered = part38.chat_text(tokenizer, sentence)
    encoded = part38._encode_with_offsets(tokenizer, rendered)
    start, end = part38.target_span_in_encoding(encoded, rendered, sentence, "wizard")
    positions = list(range(start, end))
    ids = part38._input_ids(encoded)
    selected_positions = torch.as_tensor(positions, dtype=torch.long, device=device)
    captured: dict[int, np.ndarray] = {}
    handles = []

    for layer_index in candidate_layers:
        def hook(_module: Any, args: tuple[Any, ...], *, index: int = int(layer_index)):
            x = args[0]
            if x.shape[0] != 1 or x.shape[1] != len(ids):
                raise ValueError(f"layer {index} activation shape does not match the sentence")
            values = x.detach().float()[0].index_select(0, selected_positions).amax(dim=0)
            captured[index] = values.cpu().numpy().astype(np.float32, copy=False)

        handles.append(layers[int(layer_index)].mlp.down_proj.register_forward_pre_hook(hook))
    try:
        with torch.inference_mode():
            model(**part38.model_inputs(encoded, device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != set(map(int, candidate_layers)):
        raise RuntimeError("fallback selection did not capture every candidate layer")
    return captured


def fallback_rank(screened: dict[int, list[np.ndarray]],
                  control: dict[int, list[np.ndarray]],
                  candidate_layers: Iterable[int], excluded: tuple[int, int] = (19, 4826)
                  ) -> list[dict[str, Any]]:
    """Rank units by median wizard-token activation gain over the role control."""
    candidates: list[dict[str, Any]] = []
    for layer_index in candidate_layers:
        screen = np.stack(screened[int(layer_index)], axis=0)
        controls = np.stack(control[int(layer_index)], axis=0)
        if screen.shape != controls.shape or screen.shape[0] != 12:
            raise ValueError("fallback requires twelve wizard and twelve control rows")
        screen_median = np.median(screen, axis=0)
        control_median = np.median(controls, axis=0)
        for unit_index, (screen_value, control_value) in enumerate(zip(screen_median, control_median)):
            if (int(layer_index), unit_index) == excluded:
                continue
            # Amendment 2 (Claude, before any push): positive pole only, and selective. The first selection ranked
            # layer-27 units whose values are large and NEGATIVE on both senses (-682 vs -922), which clamping at 8x
            # their natural maximum would turn into a large negative write, not a wizard partner.
            if not (float(screen_value) > 0 and float(control_value) < 0.5 * float(screen_value)):
                continue
            gain = float(screen_value - control_value)
            if not math.isfinite(gain):
                continue
            ratio = (float(screen_value / control_value)
                     if control_value != 0 and math.isfinite(float(screen_value / control_value))
                     else None)
            candidates.append({
                "layer": int(layer_index), "unit": int(unit_index),
                "median_wizard": float(screen_value),
                "median_control_role": float(control_value),
                "median_difference": gain,
                "wizard_over_control_ratio": ratio,
            })
    candidates.sort(key=lambda row: (-row["median_difference"], row["layer"], row["unit"]))
    # Amendment 2: at most two partners per layer, so one layer cannot fill the list.
    per_layer: dict[int, int] = {}
    kept = []
    for row in candidates:
        if per_layer.get(row["layer"], 0) < 2:
            kept.append(row)
            per_layer[row["layer"]] = per_layer.get(row["layer"], 0) + 1
    candidates = kept
    for rank, row in enumerate(candidates, 1):
        row["rank"] = rank
    return candidates


def run_select(args: argparse.Namespace) -> dict[str, Any]:
    data = read_json(args.sentences)
    wizard = next((unit for unit in data.get("units", []) if unit.get("id") == "wizard"), None)
    if wizard is None or (int(wizard.get("layer", -1)), int(wizard.get("unit", -1))) != (19, 4826):
        raise ValueError("fallback sentence data must include wizard L19:4826")
    screened_sentences = wizard["sentences"]["screened"]
    control_sentences = wizard["sentences"]["other"]
    if len(screened_sentences) != 12 or len(control_sentences) != 12:
        raise ValueError("fallback selection requires twelve examples per wizard sense")

    torch, tokenizer, model, layers, _snapshot = part38.load_model(args)
    candidate_layers = list(range(4, 27))  # Amendment 2: the last layer (27) writes straight to the logits; excluded
    if max(candidate_layers) >= len(layers):
        raise ValueError("the loaded model has fewer layers than the fallback range")
    widths = {index: int(layers[index].mlp.down_proj.in_features) for index in candidate_layers}
    if 4826 >= int(layers[19].mlp.down_proj.in_features):
        raise ValueError("wizard unit L19:4826 is outside the loaded model width")

    screened: dict[int, list[np.ndarray]] = {index: [] for index in candidate_layers}
    control: dict[int, list[np.ndarray]] = {index: [] for index in candidate_layers}
    for sense_name, sentences, target in (
        ("wizard", screened_sentences, screened),
        ("control_role", control_sentences, control),
    ):
        for index, sentence in enumerate(sentences, 1):
            values = _capture_word_maxima(torch, tokenizer, model, layers, sentence,
                                          args.device, candidate_layers)
            for layer_index in candidate_layers:
                target[layer_index].append(values[layer_index])
            print(f"select captured {sense_name} {index}/12", flush=True)

    ranked = fallback_rank(screened, control, candidate_layers)
    partners = []
    for row in ranked[:5]:
        partners.append({
            "id": f"P{len(partners) + 1}", "layer": row["layer"], "unit": row["unit"],
            "selection_rank": row["rank"],
            "selection_score": row["median_difference"],
            "median_wizard": row["median_wizard"],
            "median_control_role": row["median_control_role"],
            "wizard_over_control_ratio": row["wizard_over_control_ratio"],
        })
    summary = {
        "schema_version": 1,
        "selection_method": "part38_wizard_token_median_difference_fallback",
        "model": args.model, "revision": args.revision,
        "wizard": WIZARD,
        "source_screen": None,
        "source_screen_sha256": None,
        "fallback_sentence_file": str(args.sentences),
        "fallback_sentence_sha256": sha256_file(args.sentences),
        "screened_sense": wizard.get("screened_sense", "sorcerer"),
        "control_role_sense": wizard.get("other_sense", "setup wizard"),
        "candidate_layers": [min(candidate_layers), max(candidate_layers)],
        "candidate_count": sum(widths.values()) - 1,
        "ranking_metric": (
            "median(max down_proj input over the 'wizard' token's tokens in 12 sorcerer-role "
            "sentences) minus the same median in 12 setup-role control sentences"
        ),
        "tie_break": "lower layer, then lower unit",
        "partners": partners,
        "top_candidates": ranked[:50],
        "candidate_layer_widths": {str(layer): width for layer, width in widths.items()},
    }
    write_json_atomic(summary, args.out)
    print(json.dumps({"partners": partners, "out": str(args.out)}, ensure_ascii=False, indent=2))
    return summary


def load_partners(path: str | Path) -> dict[str, Any]:
    value = read_json(path)
    partners = value.get("partners", [])
    if len(partners) != 5 or [row.get("id") for row in partners] != [f"P{i}" for i in range(1, 6)]:
        raise ValueError("partner file must contain P1 through P5 in rank order")
    for row in partners:
        layer, unit = int(row["layer"]), int(row["unit"])
        if not 4 <= layer <= 27 or (layer, unit) == (19, 4826):
            raise ValueError(f"invalid partner coordinate for {row['id']}")
    return value


def choose_random_pairs(partners: list[dict[str, Any]], widths: dict[int, int], seed: int
                        ) -> list[dict[str, Any]]:
    partner_coordinates = {(int(row["layer"]), int(row["unit"])) for row in partners}
    layers = sorted({int(row["layer"]) for row in partners})
    candidates = [(layer, unit) for layer in layers for unit in range(widths[layer])
                  if (layer, unit) != (WIZARD["layer"], WIZARD["unit"])
                  and (layer, unit) not in partner_coordinates]
    if len(candidates) < 3:
        raise ValueError("fewer than three random partner-layer coordinates are available")
    random_coordinates = random.Random(part38.stable_seed(seed, "part49-random-pairs")).sample(candidates, 3)
    return [{"id": f"R{index}", "layer": layer, "unit": unit}
            for index, (layer, unit) in enumerate(random_coordinates, 1)]


def build_arm_plan(partners: list[dict[str, Any]], random_pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    arms = [{"id": "no_clamp", "clamps": []}, {"id": "W", "clamps": [dict(WIZARD)]}]
    arms.extend({"id": row["id"], "clamps": [dict(row)]} for row in partners)
    arms.extend({"id": f"{row['id']}+W", "clamps": [dict(row), dict(WIZARD)]}
                for row in partners)
    arms.extend({"id": f"{row['id']}+W", "clamps": [dict(row), dict(WIZARD)]}
                for row in random_pairs)
    if len(arms) != 15:
        raise AssertionError(f"the Part 49 design has 15 arms, got {len(arms)}")
    if len({row["id"] for row in arms}) != 15:
        raise AssertionError("arm identifiers must be unique")
    return arms


def build_negative_arm_plan(partners: list[dict[str, Any]]) -> list[dict[str, Any]]:
    arms = [{"id": "no_clamp", "clamps": []}, {"id": "W", "clamps": [dict(WIZARD)]}]
    arms.extend({"id": row["id"], "clamps": [dict(row)]} for row in partners)
    arms.extend({"id": f"{row['id']}+W", "clamps": [dict(row), dict(WIZARD)]}
                for row in partners)
    if len(arms) != 12:
        raise AssertionError(f"the negative Part 49 design has 12 arms, got {len(arms)}")
    return arms


def build_sample_rows(prompts: list[str], samples_per_prompt: int, seed: int,
                      namespace: str = "part49-push-sample") -> list[dict[str, Any]]:
    if samples_per_prompt <= 0:
        raise ValueError("samples_per_prompt must be positive")
    rows = []
    for prompt_index, prompt in enumerate(prompts, 1):
        prompt_id = f"p{prompt_index:02d}"
        for sample_index in range(1, samples_per_prompt + 1):
            rows.append({
                "prompt_id": prompt_id, "sample": sample_index, "prompt": prompt,
                "seed": part38.stable_seed(seed, namespace, prompt_id, sample_index),
            })
    return rows


def arm_requests(arms: list[dict[str, Any]], samples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Give every arm the same ordered prompt/sample rows and stable row seeds."""
    return {
        arm["id"]: [{**sample, "row_key": f"{arm['id']}:{sample['prompt_id']}:{sample['sample']}"}
                    for sample in samples]
        for arm in arms
    }


def load_part49b_prompts(path: str | Path = PART49B_DATA_PATH) -> list[str]:
    data = read_json(path)
    prompts = data.get("prompts", [])
    if len(prompts) != 16 or any(not isinstance(prompt, str) or not prompt.strip()
                                 for prompt in prompts):
        raise ValueError("Part 49b prompt data must contain sixteen nonempty prompts")
    normalized = [" ".join(prompt.casefold().split()) for prompt in prompts]
    if len(set(normalized)) != len(normalized):
        raise ValueError("Part 49b prompts must be unique")
    old_prompts = read_json(NEGATIVE_DATA_PATH).get("prompts", [])
    old_normalized = {" ".join(str(prompt).casefold().split()) for prompt in old_prompts}
    if old_normalized.intersection(normalized):
        raise ValueError("Part 49b prompts must not repeat a Part 49 prompt")
    return prompts


def choose_part49b_random_units(width: int, seed: int = PART49B_RANDOM_SEED
                                ) -> list[dict[str, Any]]:
    """Choose three reproducible L22 controls, excluding the candidate L22:1610."""
    width = int(width)
    target_unit = 1610
    if width <= target_unit:
        raise ValueError("L22 width must include unit 1610 and at least three controls")
    candidates = [unit for unit in range(width) if unit != target_unit]
    selected = random.Random(int(seed)).sample(candidates, 3)
    return [{"id": f"R{index}", "layer": 22, "unit": unit}
            for index, unit in enumerate(selected, 1)]


def build_part49b_arm_plan(random_units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(random_units) != 3:
        raise ValueError("Part 49b requires exactly three random units")
    coordinates = [(int(row["layer"]), int(row["unit"])) for row in random_units]
    if (len(set(coordinates)) != 3 or any(layer != 22 or unit == 1610
                                          for layer, unit in coordinates)):
        raise ValueError("random Part 49b units must be distinct L22 coordinates other than L22:1610")
    target = {"layer": 22, "unit": 1610}
    wizard = dict(WIZARD)
    arms = [
        {"id": "no_clamp", "clamps": []},
        {"id": "L22:1610", "clamps": [dict(target)]},
        {"id": "L19:4826", "clamps": [wizard]},
        {"id": "both", "clamps": [dict(target), dict(wizard)]},
    ]
    arms.extend({"id": row["id"], "clamps": [{"layer": 22, "unit": int(row["unit"])}]}
                for row in random_units)
    if len(arms) != 7 or len({row["id"] for row in arms}) != 7:
        raise AssertionError("Part 49b must have seven uniquely named arms")
    return arms


def part49b_answer_measure(answer: str, prompt: str) -> dict[str, Any]:
    """Count target word starts per 1,000 answer words, omitting prompt-mentioned stems."""
    answer, prompt = str(answer), str(prompt)
    excluded = {term for term, pattern in PART49B_MAGIC_PATTERNS.items()
                if pattern.search(prompt)}
    matches = sum(len(pattern.findall(answer)) for term, pattern in PART49B_MAGIC_PATTERNS.items()
                  if term not in excluded)
    words = len(re.findall(r"\b\w+\b", answer, flags=re.UNICODE))
    return {
        "matches": int(matches), "word_count": int(words),
        "per_1000": (float(matches * 1000.0 / words) if words else 0.0),
        "excluded_terms": sorted(excluded),
    }


def _bootstrap_interval(values: np.ndarray) -> list[float] | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return None
    low, high = np.quantile(values, [0.025, 0.975])
    return [float(low), float(high)]


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0 or not math.isfinite(numerator) or not math.isfinite(denominator):
        return None
    return float(numerator / denominator)


def score_push49b_rows(rows: Iterable[dict[str, Any]], arms: list[dict[str, Any]] | None = None,
                       seed: int = PART49B_BOOTSTRAP_SEED,
                       resamples: int = PART49B_BOOTSTRAP_RESAMPLES) -> dict[str, Any]:
    """Score answer density and paired prompt-level bootstrap estimates for Part 49b."""
    rows = list(rows)
    results = [row for row in rows if row.get("record_type") == "result"]
    if arms is None:
        header = next((row for row in rows if row.get("record_type") == "header"), {})
        arms = header.get("arms", [])
    arm_by_id = {str(arm["id"]): arm for arm in arms}
    if "no_clamp" not in arm_by_id:
        raise ValueError("Part 49b scoring requires the no_clamp reference arm")
    if resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    baseline_rows = [row for row in results if row.get("arm") == "no_clamp"]
    prompt_ids = sorted({str(row.get("prompt_id")) for row in baseline_rows})
    if not prompt_ids:
        raise ValueError("Part 49b results have no no_clamp prompts")

    grouped: dict[str, dict[str, list[float]]] = {
        arm_id: {prompt_id: [] for prompt_id in prompt_ids} for arm_id in arm_by_id
    }
    broken: dict[str, list[bool]] = {arm_id: [] for arm_id in arm_by_id}
    for row in results:
        arm_id = str(row.get("arm"))
        prompt_id = str(row.get("prompt_id"))
        if arm_id not in arm_by_id:
            raise ValueError(f"Part 49b result has an unknown arm: {arm_id}")
        if prompt_id not in grouped[arm_id]:
            raise ValueError(f"Part 49b arm {arm_id} has a prompt outside no_clamp: {prompt_id}")
        measure = part49b_answer_measure(row.get("answer", ""), row.get("prompt", ""))
        grouped[arm_id][prompt_id].append(measure["per_1000"])
        if isinstance(row.get("broken"), bool):
            broken[arm_id].append(row["broken"])

    prompt_values: dict[str, np.ndarray] = {}
    answer_counts: dict[str, int] = {}
    for arm_id in arm_by_id:
        empty_prompts = [prompt_id for prompt_id in prompt_ids if not grouped[arm_id][prompt_id]]
        if empty_prompts:
            raise ValueError(f"Part 49b arm {arm_id} is missing prompts: {empty_prompts}")
        prompt_values[arm_id] = np.asarray(
            [float(np.mean(grouped[arm_id][prompt_id])) for prompt_id in prompt_ids],
            dtype=np.float64,
        )
        answer_counts[arm_id] = sum(len(grouped[arm_id][prompt_id]) for prompt_id in prompt_ids)

    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(prompt_ids), size=(int(resamples), len(prompt_ids)))
    boot_means = {arm_id: values[indices].mean(axis=1)
                  for arm_id, values in prompt_values.items()}
    baseline_mean = float(prompt_values["no_clamp"].mean())
    baseline_boot = boot_means["no_clamp"]

    arm_summaries = []
    for arm in arms:
        arm_id = str(arm["id"])
        mean = float(prompt_values[arm_id].mean())
        ratio = _safe_ratio(mean, baseline_mean)
        ratio_boot = np.divide(
            boot_means[arm_id], baseline_boot,
            out=np.full(int(resamples), np.nan, dtype=np.float64), where=baseline_boot != 0,
        )
        broken_labels = broken[arm_id]
        broken_hits = sum(broken_labels)
        broken_rate = broken_hits / len(broken_labels) if broken_labels else None
        per_prompt = [
            {"prompt_id": prompt_id, "mean_per_1000": float(prompt_values[arm_id][index])}
            for index, prompt_id in enumerate(prompt_ids)
        ]
        arm_summaries.append({
            "arm": arm_id, "clamps": arm.get("clamps", []),
            "answer_count": answer_counts[arm_id], "prompt_count": len(prompt_ids),
            "mean_per_1000": mean,
            "bootstrap95": _bootstrap_interval(boot_means[arm_id]),
            "ratio_to_no_clamp": {
                "value": ratio, "bootstrap95": _bootstrap_interval(ratio_boot),
            },
            "per_prompt": per_prompt,
            "broken": {"hits": broken_hits, "n": len(broken_labels),
                       "rate": broken_rate, "pending": not bool(broken_labels)},
        })

    joint_comparison = None
    required = {"L22:1610", "L19:4826", "both"}
    if required.issubset(prompt_values):
        target_mean = float(prompt_values["L22:1610"].mean())
        wizard_mean = float(prompt_values["L19:4826"].mean())
        joint_mean = float(prompt_values["both"].mean())
        target_ratio = _safe_ratio(target_mean, baseline_mean)
        wizard_ratio = _safe_ratio(wizard_mean, baseline_mean)
        joint_ratio = _safe_ratio(joint_mean, baseline_mean)
        product = (target_ratio * wizard_ratio
                   if target_ratio is not None and wizard_ratio is not None else None)
        joint_to_product = (_safe_ratio(joint_ratio, product)
                            if joint_ratio is not None and product is not None else None)
        denominator = boot_means["L22:1610"] * boot_means["L19:4826"]
        joint_boot = np.divide(
            boot_means["both"] * baseline_boot, denominator,
            out=np.full(int(resamples), np.nan, dtype=np.float64), where=denominator != 0,
        )
        joint_comparison = {
            "joint_ratio_to_no_clamp": joint_ratio,
            "product_of_single_ratios": product,
            "joint_to_product_ratio": joint_to_product,
            "joint_to_product_bootstrap95": _bootstrap_interval(joint_boot),
        }

    return {
        "schema_version": 1, "summary_type": "part49b_magic_word_density",
        "arm_count": len(arms), "result_count": len(results),
        "measure": {
            "terms": list(PART49B_MAGIC_TERMS),
            "matching": r"\b(?:spell|magic|wand|sorcer|enchant|potion)\w* (case-insensitive)",
            "prompt_exclusion": "answer matches whose stem occurs in that row's prompt are excluded",
            "word_count": "Unicode regex \\b\\w+\\b",
            "resampling_unit": "prompt",
            "bootstrap_resamples": int(resamples), "bootstrap_seed": int(seed),
            "prompt_count": len(prompt_ids),
        },
        "arms": arm_summaries,
        "joint_vs_product_of_singles": joint_comparison,
    }


def operations_for_arm(arm: dict[str, Any], k: float | None) -> dict[tuple[int, int], float | None]:
    """Build set-to-value operations; ``k=None`` installs no-op hooks for controls."""
    return {
        (int(clamp["layer"]), int(clamp["unit"])):
            (None if k is None else float(k * float(clamp["natural_max"])))
        for clamp in arm["clamps"]
    }


def score_push_rows(rows: Iterable[dict[str, Any]], arms: list[dict[str, Any]],
                    partners: list[dict[str, Any]]) -> dict[str, Any]:
    results = [row for row in rows if row.get("record_type") == "result"]
    arm_summaries = []
    by_arm: dict[str, dict[str, Any]] = {}
    for arm in arms:
        arm_rows = [row for row in results if row.get("arm") == arm["id"]]
        hits = sum(topic_hit(str(row.get("answer", ""))) for row in arm_rows)
        count = len(arm_rows)
        broken_rows = [row for row in arm_rows if isinstance(row.get("broken"), bool)]
        broken_hits = sum(bool(row["broken"]) for row in broken_rows)
        topic = {
            "hits": hits, "n": count, "rate": hits / count if count else None,
            "wilson95": list(part38.wilson_interval(hits, count)) if count else None,
        }
        broken = {
            "hits": broken_hits, "n": len(broken_rows),
            "rate": broken_hits / len(broken_rows) if broken_rows else None,
        }
        row = {"arm": arm["id"], "clamps": arm["clamps"], "topic": topic,
               "broken": broken, "topic_rate": topic["rate"],
               "wilson95": topic["wilson95"], "broken_rate": broken["rate"]}
        arm_summaries.append(row)
        by_arm[arm["id"]] = row

    partner_boosts = []
    for partner in partners:
        solo = by_arm.get(partner["id"])
        combined = by_arm.get(f"{partner['id']}+W")
        solo_rate = solo["topic_rate"] if solo else None
        combined_rate = combined["topic_rate"] if combined else None
        partner_boosts.append({
            "partner": partner["id"], "layer": int(partner["layer"]), "unit": int(partner["unit"]),
            "partner_alone_rate": solo_rate, "partner_plus_wizard_rate": combined_rate,
            "delta_rate": (combined_rate - solo_rate
                           if combined_rate is not None and solo_rate is not None else None),
        })
    return {"schema_version": 1, "summary_type": "part49_partner_push",
            "arm_count": len(arms), "result_count": len(results),
            "topic_terms": list(TOPIC_TERMS), "arms": arm_summaries,
            "partner_boosts": partner_boosts}


def _capture_natural_maxima(torch: Any, tokenizer: Any, model: Any, layers: Any,
                            coordinates: list[dict[str, Any]], passages: list[dict[str, str]],
                            device: str) -> dict[str, Any]:
    by_layer: dict[int, set[int]] = {}
    for row in coordinates:
        by_layer.setdefault(int(row["layer"]), set()).add(int(row["unit"]))
    maxima = {f"{layer}:{unit}": None for layer, units in by_layer.items() for unit in units}
    for passage_index, passage in enumerate(passages, 1):
        captured = part38.capture_multi_layer_maxima(
            torch, tokenizer, model, layers, by_layer, passage["text"], device
        )
        for layer, units in by_layer.items():
            for unit in units:
                key = f"{layer}:{unit}"
                value = float(captured[layer][unit])
                maxima[key] = value if maxima[key] is None else max(float(maxima[key]), value)
        print(f"natural-max passage {passage_index}/{len(passages)}", flush=True)
    return {"schema_version": 1, "summary_type": "part49_natural_max",
            "model": MODEL_ID, "revision": MODEL_REVISION,
            "activation_source": "Part 38 Part 35 neutral frames, max over content tokens",
            "passage_count": len(passages), "values": maxima}


def _load_or_capture_natural_max(args: argparse.Namespace, torch: Any, tokenizer: Any,
                                 model: Any, layers: Any, coordinates: list[dict[str, Any]],
                                 out_dir: Path) -> tuple[dict[str, Any], Path]:
    natural_path = args.natural_max or (out_dir / "natural-max-summary.json")
    natural_path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({f"{int(row['layer'])}:{int(row['unit'])}" for row in coordinates})
    if natural_path.is_file():
        natural = read_json(natural_path)
        if (natural.get("model") == MODEL_ID and natural.get("revision") == MODEL_REVISION
                and all(natural.get("values", {}).get(key) is not None for key in keys)):
            return natural, natural_path
    passages = part38.load_part35_neutral_passages(args.part35_data)
    natural = _capture_natural_maxima(torch, tokenizer, model, layers, coordinates,
                                      passages, args.device)
    natural.update({"part35_data_sha256": sha256_file(args.part35_data),
                    "coordinate_keys": keys})
    write_json_atomic(natural, natural_path)
    return natural, natural_path


def _validate_push_data(path: str | Path) -> dict[str, Any]:
    data = read_json(path)
    part38.validate_data(data)
    if len(data["neutral_prompts"]) != 16:
        raise ValueError("Part 49 requires all sixteen Part 38 open prompts")
    return data


def _add_natural_clamps(arms: list[dict[str, Any]], natural: dict[str, Any], k: int
                        ) -> list[dict[str, Any]]:
    result = []
    for arm in arms:
        updated = {**arm, "clamps": []}
        for clamp in arm["clamps"]:
            key = f"{int(clamp['layer'])}:{int(clamp['unit'])}"
            natural_max = natural["values"].get(key)
            if natural_max is None:
                raise ValueError(f"natural maximum missing for {key}")
            updated["clamps"].append({**clamp, "natural_max": float(natural_max),
                                      "k": int(k), "level": float(k * natural_max)})
        result.append(updated)
    return result


def _run_arm_set(args: argparse.Namespace, *, mode: str, data_path: Path,
                 arms: list[dict[str, Any]], samples: list[dict[str, Any]],
                 partners: list[dict[str, Any]], random_pairs: list[dict[str, Any]],
                 natural: dict[str, Any], natural_path: Path, k: int,
                 torch: Any, tokenizer: Any, model: Any) -> tuple[dict[str, Any], dict[str, Any], Path]:
    arms = _add_natural_clamps(arms, natural, k)
    output_name = "push.jsonl" if mode == "open_prompt" else "negative-push.jsonl"
    output_path = args.out / output_name
    header = {
        "schema_version": 1, "record_type": "header", "command": "push", "assay": mode,
        "model": args.model, "revision": args.revision, "device": args.device,
        "dtype": args.dtype, "seed": int(args.seed), "data_sha256": sha256_file(data_path),
        "partner_file": str(args.partners), "partner_file_sha256": sha256_file(args.partners),
        "natural_max_path": str(natural_path), "natural_max_sha256": sha256_file(natural_path)
        if natural_path.is_file() else None,
        "arms": arms, "partner_order": [row["id"] for row in partners],
        "random_pairs": random_pairs if mode == "open_prompt" else [], "k": int(k),
        "prompt_count": len({row["prompt_id"] for row in samples}),
        "samples_per_prompt": int(args.samples), "answer_count_per_arm": len(samples),
        "generation": {"temperature": TEMPERATURE, "top_p": 1.0,
                       "max_new_tokens": MAX_NEW_TOKENS, "batch_size": int(args.batch)},
    }
    rows = part38.prepare_jsonl(output_path, header)
    done = part38.completed_keys(rows)
    requests_by_arm = arm_requests(arms, samples)
    timings: dict[str, Any] = {}
    for arm in arms:
        pending = [sample for sample in requests_by_arm[arm["id"]] if sample["row_key"] not in done]
        operations = operations_for_arm(arm, k)
        started = time.perf_counter()
        arm_generated = 0
        for batch_start in range(0, len(pending), args.batch):
            batch = pending[batch_start:batch_start + args.batch]
            rendered_prompts = [part38.chat_text(tokenizer, sample["prompt"]) for sample in batch]
            outputs = fastgen.batched_generate(
                model, tokenizer, rendered_prompts,
                max_new_tokens=MAX_NEW_TOKENS, operations=operations,
                do_sample=True, row_seeds=[int(sample["seed"]) for sample in batch],
                temperature=TEMPERATURE, device=args.device, generated_only=True,
                padding_side=args.padding_side,
            )
            for sample, output in zip(batch, outputs):
                row = {
                    "record_type": "result", "row_key": sample["row_key"], "arm": arm["id"],
                    "prompt_id": sample["prompt_id"], "sample": sample["sample"],
                    "seed": sample["seed"], "prompt": sample["prompt"],
                    "answer": output["text"], "broken": None, "clamps": arm["clamps"],
                }
                part38.append_jsonl(output_path, row)
                done.add(sample["row_key"])
                arm_generated += 1
        elapsed = time.perf_counter() - started
        timings[arm["id"]] = {"seconds": elapsed, "generated": arm_generated,
                              "batch_size": int(args.batch)}
        print(f"{mode} arm {arm['id']}: generated {arm_generated}/{len(samples)} answers "
              f"in {elapsed:.2f}s (batch={args.batch})", flush=True)

    final_rows = part38.read_jsonl(output_path)
    scores = score_push_rows(final_rows, arms, partners)
    scores.update({"assay": mode, "k": int(k), "timings": timings,
                   "batch_size": int(args.batch), "natural_max_path": str(natural_path)})
    return scores, timings, output_path


def run_push(args: argparse.Namespace) -> dict[str, Any]:
    data = _validate_push_data(args.data)
    negative_data = read_json(args.negative_data)
    negative_prompts = negative_data.get("prompts", [])
    if len(negative_prompts) != 8 or any(not isinstance(prompt, str) or not prompt.strip()
                                         for prompt in negative_prompts):
        raise ValueError("negative wizard prompt data must contain eight nonempty prompts")
    partner_file = load_partners(args.partners)
    partners = partner_file["partners"]
    if partner_file.get("selection_method") != "part38_wizard_token_median_difference_fallback":
        raise ValueError("part49 push requires the completed fallback partner selection file")
    torch, tokenizer, model, layers, _snapshot = part38.load_model(args)
    partner_units = [{"id": row["id"], "layer": int(row["layer"]), "unit": int(row["unit"])}
                    for row in partners]
    all_units = [{"id": "wizard", "layer": 19, "unit": 4826}, *partner_units]
    widths = part38.layer_widths(layers, all_units)
    random_pairs = choose_random_pairs(partners, widths, args.seed)
    open_plan = build_arm_plan(partners, random_pairs)
    negative_plan = build_negative_arm_plan(partners)
    open_by_id = {row["id"]: row for row in open_plan}
    negative_by_id = {row["id"]: row for row in negative_plan}

    requested = None
    if args.arms:
        requested = [value.strip() for value in args.arms.split(",") if value.strip()]
        if not requested or len(requested) != len(set(requested)):
            raise ValueError("--arms must name one or more distinct arm ids")
        unknown = [value for value in requested if value not in open_by_id]
        if unknown:
            raise ValueError(f"unknown arm ids: {', '.join(unknown)}")
        open_arms = [open_by_id[value] for value in requested]
    else:
        open_arms = open_plan
    if args.negative_arms:
        negative_requested = [value.strip() for value in args.negative_arms.split(",") if value.strip()]
        if not negative_requested or len(negative_requested) != len(set(negative_requested)):
            raise ValueError("--negative-arms must name one or more distinct arm ids")
        unknown = [value for value in negative_requested if value not in negative_by_id]
        if unknown:
            raise ValueError(f"unknown negative arm ids: {', '.join(unknown)}")
        negative_arms = [negative_by_id[value] for value in negative_requested]
    elif requested is not None:
        negative_arms = [negative_by_id[value] for value in requested if value in negative_by_id]
    else:
        negative_arms = negative_plan

    open_prompts = data["neutral_prompts"][:args.limit] if args.limit else data["neutral_prompts"]
    negative_limit = min(args.limit, len(negative_prompts)) if args.limit else len(negative_prompts)
    selected_negative_prompts = negative_prompts[:negative_limit]
    open_samples = build_sample_rows(open_prompts, args.samples, args.seed, "part49-open-sample")
    negative_samples = build_sample_rows(selected_negative_prompts, args.samples, args.seed,
                                         "part49-negative-sample")
    if args.limit is None and args.samples == DEFAULT_SAMPLES:
        if len(open_arms) == 15 and len(open_samples) != 64:
            raise AssertionError("the full open-prompt Part 49 arm has to contain 64 answers")
        if len(negative_arms) == 12 and len(negative_samples) != 32:
            raise AssertionError("the full negative-prompt Part 49 arm has to contain 32 answers")

    coordinates = [clamp for arm in [*open_arms, *negative_arms] for clamp in arm["clamps"]]
    unique_coordinates = {(int(row["layer"]), int(row["unit"])) for row in coordinates}
    coordinates = [{"layer": layer, "unit": unit} for layer, unit in sorted(unique_coordinates)]
    natural, natural_path = _load_or_capture_natural_max(
        args, torch, tokenizer, model, layers, coordinates, args.out
    ) if coordinates else ({"values": {}}, args.natural_max or (args.out / "natural-max-summary.json"))

    args.out.mkdir(parents=True, exist_ok=True)
    open_scores, open_timings, open_path = _run_arm_set(
        args, mode="open_prompt", data_path=args.data, arms=open_arms, samples=open_samples,
        partners=partners, random_pairs=random_pairs, natural=natural,
        natural_path=natural_path, k=DEFAULT_K, torch=torch, tokenizer=tokenizer, model=model,
    )
    negative_scores: dict[str, Any] = {"summary_type": "part49_negative_push", "arm_count": 0,
                                      "result_count": 0, "arms": [], "partner_boosts": []}
    negative_timings: dict[str, Any] = {}
    negative_path: Path | None = None
    if negative_arms:
        negative_scores, negative_timings, negative_path = _run_arm_set(
            args, mode="negative_wizard_prompt", data_path=args.negative_data,
            arms=negative_arms, samples=negative_samples, partners=partners, random_pairs=[],
            natural=natural, natural_path=natural_path, k=-DEFAULT_K,
            torch=torch, tokenizer=tokenizer, model=model,
        )
        write_json_atomic(negative_scores, args.out / "negative-push-scores.json")
    combined = {
        **open_scores,
        "model": args.model, "revision": args.revision, "seed": int(args.seed),
        "open_prompt": open_scores, "negative_prompt": negative_scores,
        "open_prompt_timings": open_timings, "negative_prompt_timings": negative_timings,
        "open_prompt_jsonl": str(open_path),
        "negative_prompt_jsonl": str(negative_path) if negative_path else None,
    }
    write_json_atomic(combined, args.out / "push-scores.json")
    print(json.dumps({
        "open_prompt_jsonl": str(open_path),
        "negative_prompt_jsonl": str(negative_path) if negative_path else None,
        "scores": str(args.out / "push-scores.json"),
        "open_answers_per_arm": len(open_samples), "open_arm_count": len(open_arms),
        "negative_answers_per_arm": len(negative_samples) if negative_arms else 0,
        "negative_arm_count": len(negative_arms),
        "open_prompt_timings": open_timings, "negative_prompt_timings": negative_timings,
    }, indent=2))
    return combined


def run_push49b(args: argparse.Namespace) -> dict[str, Any]:
    prompts = load_part49b_prompts(args.data)
    selected_prompts = prompts[:args.limit] if args.limit else prompts
    torch, tokenizer, model, layers, _snapshot = part38.load_model(args)
    fixed_units = [
        {"layer": 22, "unit": 1610},
        {"layer": int(WIZARD["layer"]), "unit": int(WIZARD["unit"])},
    ]
    widths = part38.layer_widths(layers, fixed_units)
    random_units = choose_part49b_random_units(widths[22], PART49B_RANDOM_SEED)
    arms = build_part49b_arm_plan(random_units)
    samples = build_sample_rows(selected_prompts, args.samples, args.seed,
                                "part49b-push-sample")
    if args.limit is None and args.samples == DEFAULT_SAMPLES and len(samples) != 64:
        raise AssertionError("the full Part 49b arm has to contain 64 answers")

    coordinates = sorted({(int(clamp["layer"]), int(clamp["unit"]))
                          for arm in arms for clamp in arm["clamps"]})
    coordinate_rows = [{"layer": layer, "unit": unit} for layer, unit in coordinates]
    natural, natural_path = _load_or_capture_natural_max(
        args, torch, tokenizer, model, layers, coordinate_rows, args.out
    )
    arms = _add_natural_clamps(arms, natural, -DEFAULT_K)
    args.out.mkdir(parents=True, exist_ok=True)
    output_path = args.out / "push49b.jsonl"
    header = {
        "schema_version": 1, "record_type": "header", "command": "push49b",
        "assay": "fresh_wizard_prompts", "model": args.model, "revision": args.revision,
        "device": args.device, "dtype": args.dtype, "seed": int(args.seed),
        "data_sha256": sha256_file(args.data), "arms": arms,
        "random_unit_seed": PART49B_RANDOM_SEED,
        "random_units": random_units,
        "primary_unit": {"layer": 22, "unit": 1610}, "wizard_unit": WIZARD,
        "natural_max_path": str(natural_path),
        "natural_max_sha256": sha256_file(natural_path) if natural_path.is_file() else None,
        "k": -DEFAULT_K,
        "prompt_count": len(selected_prompts), "samples_per_prompt": int(args.samples),
        "answer_count_per_arm": len(samples),
        "generation": {"temperature": TEMPERATURE, "top_p": 1.0,
                       "max_new_tokens": MAX_NEW_TOKENS, "batch_size": int(args.batch)},
    }
    rows = part38.prepare_jsonl(output_path, header)
    done = part38.completed_keys(rows)
    requests_by_arm = arm_requests(arms, samples)
    timings: dict[str, Any] = {}
    for arm in arms:
        pending = [sample for sample in requests_by_arm[arm["id"]]
                   if sample["row_key"] not in done]
        operations = operations_for_arm(arm, -DEFAULT_K)
        started = time.perf_counter()
        generated = 0
        for batch_start in range(0, len(pending), args.batch):
            batch = pending[batch_start:batch_start + args.batch]
            rendered_prompts = [part38.chat_text(tokenizer, sample["prompt"]) for sample in batch]
            outputs = fastgen.batched_generate(
                model, tokenizer, rendered_prompts,
                max_new_tokens=MAX_NEW_TOKENS, operations=operations,
                do_sample=True, row_seeds=[int(sample["seed"]) for sample in batch],
                temperature=TEMPERATURE, device=args.device, generated_only=True,
                padding_side=args.padding_side,
            )
            for sample, output in zip(batch, outputs):
                row = {
                    "record_type": "result",
                    "row_key": f"{arm['id']}:{sample['prompt_id']}:{sample['sample']}",
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
        print(f"part49b arm {arm['id']}: generated {generated}/{len(samples)} answers "
              f"in {elapsed:.2f}s (batch={args.batch})", flush=True)

    final_rows = part38.read_jsonl(output_path)
    scores = score_push49b_rows(final_rows, arms)
    scores.update({"model": args.model, "revision": args.revision, "seed": int(args.seed),
                   "random_unit_seed": PART49B_RANDOM_SEED, "random_units": random_units,
                   "k": -DEFAULT_K, "timings": timings,
                   "jsonl": str(output_path), "natural_max_path": str(natural_path)})
    score_path = args.out / "push49b-scores.json"
    write_json_atomic(scores, score_path)
    print(json.dumps({
        "jsonl": str(output_path), "scores": str(score_path),
        "answer_count_per_arm": len(samples), "arm_count": len(arms),
        "random_unit_seed": PART49B_RANDOM_SEED, "random_units": random_units,
        "timings": timings,
    }, ensure_ascii=False, indent=2))
    return scores


def run_blind_pack(args: argparse.Namespace) -> dict[str, Any]:
    rows = part38.read_jsonl(args.input)
    answers = [row for row in rows if row.get("record_type") == "result"]
    shuffled = list(answers)
    random.Random(args.seed).shuffle(shuffled)
    pack_rows = [{"record_type": "header", "schema_version": 1,
                  "blind_pack": True, "seed": int(args.seed),
                  "answer_count": len(shuffled)}]
    key_answers = []
    for index, row in enumerate(shuffled, 1):
        blind_id = f"blind-{index:05d}"
        pack_rows.append({"record_type": "answer", "blind_id": blind_id,
                          "text": str(row.get("answer", "")), "broken": None})
        key_answers.append({"blind_id": blind_id, "row_key": row.get("row_key")})
    part38.write_jsonl_atomic(pack_rows, args.out / "blind-pack.jsonl")
    write_json_atomic({"schema_version": 1, "seed": int(args.seed),
                       "source": str(args.input), "answers": key_answers},
                      args.out / "blind-pack-key.json")
    print(f"blind pack answers: {len(shuffled)}")
    return {"answer_count": len(shuffled)}


def run_score(args: argparse.Namespace) -> dict[str, Any]:
    rows = part38.read_jsonl(args.input)
    if not rows or rows[0].get("record_type") != "header":
        raise ValueError("push JSONL has no header")
    if args.blind_pack:
        if not args.blind_key:
            raise ValueError("--blind-key is required with --blind-pack")
        key = read_json(args.blind_key)
        by_blind = {item["blind_id"]: item["row_key"] for item in key.get("answers", [])}
        by_row = {row.get("row_key"): row for row in rows if row.get("record_type") == "result"}
        labels = 0
        for answer in part38.read_jsonl(args.blind_pack):
            if answer.get("record_type") != "answer" or not isinstance(answer.get("broken"), bool):
                continue
            row_key = by_blind.get(answer.get("blind_id"))
            if row_key in by_row:
                by_row[row_key]["broken"] = answer["broken"]
                labels += 1
        print(f"merged {labels} blind broken-answer labels", flush=True)
    if rows[0].get("command") == "push49b":
        summary = score_push49b_rows(rows)
        summary.update({"random_unit_seed": rows[0].get("random_unit_seed"),
                        "random_units": rows[0].get("random_units"),
                        "k": rows[0].get("k")})
    elif rows[0].get("command") in {"push51a", "push51b"}:
        # Import lazily so the standalone Part 51 module can continue importing Part 49.
        import part51
        scorer = (part51.score_push51a_rows if rows[0]["command"] == "push51a"
                  else part51.score_push51b_rows)
        summary = scorer(rows)
        summary.update({"random_unit_seed": rows[0].get("random_unit_seed"),
                        "random_units": rows[0].get("random_units"),
                        "layer_widths": rows[0].get("layer_widths"),
                        "replicate_seeds": rows[0].get("replicate_seeds")})
    else:
        partner_file = load_partners(args.partners)
        summary = score_push_rows(rows, rows[0]["arms"], partner_file["partners"])
    summary.update({"model": rows[0].get("model"), "revision": rows[0].get("revision"),
                    "seed": rows[0].get("seed")})
    write_json_atomic(summary, args.out)
    return summary


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=MODEL_ID, choices=(MODEL_ID,))
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--cache-dir", type=Path, default=part38.DEFAULT_CACHE)
    parser.add_argument("--device", choices=("cpu", "xpu"), default="cpu")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3801)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    select = sub.add_parser("select", help="compute and save fallback partner scores")
    add_model_arguments(select)
    select.add_argument("--sentences", type=Path, default=SENTENCE_DATA_PATH)
    select.add_argument("--out", type=Path, default=PARTNERS_PATH)

    push = sub.add_parser("push", help="run the Part 49 partner push")
    add_model_arguments(push)
    push.add_argument("--data", type=Path, default=DATA_PATH)
    push.add_argument("--negative-data", type=Path, default=NEGATIVE_DATA_PATH)
    push.add_argument("--part35-data", type=Path, default=PART35_DATA_PATH)
    push.add_argument("--partners", type=Path, default=PARTNERS_PATH)
    push.add_argument("--natural-max", type=Path, default=None)
    push.add_argument("--out", type=Path, default=DEFAULT_OUT)
    push.add_argument("--limit", type=int, default=None, help="number of prompts to run")
    push.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                      help="samples per prompt; full design uses four")
    push.add_argument("--arms", default=None,
                      help="comma-separated arm ids; defaults to all 15 design arms")
    push.add_argument("--negative-arms", default=None,
                      help="comma-separated negative-assay arm ids; defaults to all 12")
    push.add_argument("--batch", type=int, default=16, help="generation batch size")
    push.add_argument("--padding-side", choices=("left", "right"), default="left")

    push49b = sub.add_parser("push49b", help="run the Part 49b fresh-prompt assay")
    add_model_arguments(push49b)
    push49b.add_argument("--data", type=Path, default=PART49B_DATA_PATH)
    push49b.add_argument("--part35-data", type=Path, default=PART35_DATA_PATH)
    push49b.add_argument("--natural-max", type=Path, default=None)
    push49b.add_argument("--out", type=Path, default=PART49B_DEFAULT_OUT)
    push49b.add_argument("--limit", type=int, default=None, help="number of prompts to run")
    push49b.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                         help="samples per prompt; full design uses four")
    push49b.add_argument("--batch", type=int, default=16, help="generation batch size")
    push49b.add_argument("--padding-side", choices=("left", "right"), default="left")

    blind = sub.add_parser("blind-pack", help="write randomized blind answer pack and key")
    blind.add_argument("--input", type=Path, default=DEFAULT_OUT / "push.jsonl")
    blind.add_argument("--out", type=Path, default=DEFAULT_OUT)
    blind.add_argument("--seed", type=int, default=3801)

    score = sub.add_parser("score", help="re-score push answers after blinded broken labels")
    score.add_argument("--input", type=Path, default=DEFAULT_OUT / "push.jsonl")
    score.add_argument("--partners", type=Path, default=PARTNERS_PATH)
    score.add_argument("--blind-pack", type=Path, default=None,
                       help="annotated blind-pack JSONL with boolean broken labels")
    score.add_argument("--blind-key", type=Path, default=None)
    score.add_argument("--out", type=Path, default=DEFAULT_OUT / "push-scores.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "select":
        if args.threads <= 0:
            parser.error("--threads must be positive")
        if args.revision != MODEL_REVISION:
            parser.error(f"Part 49 requires pinned revision {MODEL_REVISION}")
        try:
            run_select(args)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(f"part49 select: {exc}", file=sys.stderr)
            return 2
        return 0
    if args.command == "push":
        if args.threads <= 0 or args.batch <= 0 or args.samples <= 0:
            parser.error("--threads, --batch, and --samples must be positive")
        if args.limit is not None and not 1 <= args.limit <= 16:
            parser.error("--limit must be between 1 and 16")
        if args.revision != MODEL_REVISION:
            parser.error(f"Part 49 requires pinned revision {MODEL_REVISION}")
        try:
            run_push(args)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(f"part49 push: {exc}", file=sys.stderr)
            return 2
        return 0
    if args.command == "push49b":
        if args.threads <= 0 or args.batch <= 0 or args.samples <= 0:
            parser.error("--threads, --batch, and --samples must be positive")
        if args.limit is not None and not 1 <= args.limit <= 16:
            parser.error("--limit must be between 1 and 16")
        if args.revision != MODEL_REVISION:
            parser.error(f"Part 49b requires pinned revision {MODEL_REVISION}")
        try:
            run_push49b(args)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(f"part49 push49b: {exc}", file=sys.stderr)
            return 2
        return 0
    if args.command == "blind-pack":
        run_blind_pack(args)
        return 0
    if args.command == "score":
        try:
            run_score(args)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(f"part49 score: {exc}", file=sys.stderr)
            return 2
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
