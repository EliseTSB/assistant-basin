#!/usr/bin/env python3
"""Part 48b: compare k- and c-spelled twins with the Part 48 readout.

The sound /k/ is held constant by the English-orthography premise in
PART48B-DESIGN.md. Scores, templates, continuation sets, and model loading
come from :mod:`part48`; this module adds paired analyses and diagnostics.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import time
from typing import Sequence

import numpy as np
import torch

import part48 as p48


PAIRS = (
    ("nakolu", "nacolu"),
    ("rukuno", "rucuno"),
    ("rulaku", "rulacu"),
    ("ruluku", "rulucu"),
    ("rumuko", "rumuco"),
    ("tokamo", "tocamo"),
    ("tokuna", "tocuna"),
    ("toraku", "toracu"),
    ("totuko", "totuco"),
    ("tukulu", "tuculu"),
)

# Amendment 1's named near-word items, intersected as complete item spellings
# with the fixed Amendment 4 stimulus list. None of the 10 pairs matches.
NEAR_WORD_FORMS = frozenset({
    "roca", "toca", "comala", "komala", "calo", "kalo", "cuma", "kuma",
    "raku", "racu", "kumo", "cumo",
})
NEAR_WORD_PAIRS = frozenset(
    (k, c) for k, c in PAIRS if k in NEAR_WORD_FORMS or c in NEAR_WORD_FORMS
)

LETTER_MATCHED_SHARP_WORDS = tuple(w for w in p48.SHARP_WORDS if w != "spiky")
LETTER_MATCHED_ROUND_WORDS = tuple(w for w in p48.ROUND_WORDS if w != "curvy")
PERMUTATION_DRAWS = 10_000
BOOTSTRAP_DRAWS = 10_000
PERMUTATION_SEED = 48
BOOTSTRAP_SEED = 49


MASS_FLOOR = 1e-3  # part48.summarize's template floor

def sign_flip_permutation_test(
    differences: Sequence[float], draws: int = PERMUTATION_DRAWS,
    seed: int = PERMUTATION_SEED,
) -> dict:
    """Two-sided Monte Carlo sign-flip test of a zero mean paired difference."""
    values = [float(x) for x in differences]
    if draws < 1:
        raise ValueError("draws must be positive")
    if not values:
        return {"statistic": "mean", "observed": None, "p_value": None,
                "draws": draws, "seed": seed}
    observed = statistics.fmean(values)
    target = abs(observed)
    rng = random.Random(seed)
    hits = 0
    for _ in range(draws):
        permuted = statistics.fmean(x * rng.choice((-1.0, 1.0)) for x in values)
        if abs(permuted) >= target - 1e-12:
            hits += 1
    return {"statistic": "mean", "observed": observed,
            "p_value": (hits + 1) / (draws + 1), "draws": draws, "seed": seed}


def bootstrap_median_ci(
    differences: Sequence[float], draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED, confidence: float = 0.95,
) -> dict:
    """Percentile bootstrap interval for the paired median, using a fixed seed."""
    values = [float(x) for x in differences]
    if draws < 1:
        raise ValueError("draws must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if not values:
        return {"estimate": None, "lower": None, "upper": None,
                "confidence": confidence, "resamples": draws, "seed": seed}
    rng = random.Random(seed)
    medians = [statistics.median(rng.choices(values, k=len(values))) for _ in range(draws)]
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(medians, [alpha, 1.0 - alpha], method="linear")
    return {"estimate": statistics.median(values), "lower": float(lower),
            "upper": float(upper), "confidence": confidence,
            "resamples": draws, "seed": seed}


def exact_sign_test(differences: Sequence[float]) -> dict:
    """Two-sided exact binomial sign test, dropping ties."""
    positive = sum(float(x) > 0 for x in differences)
    negative = sum(float(x) < 0 for x in differences)
    ties = len(differences) - positive - negative
    n = positive + negative
    if not n:
        p_value = None
    else:
        tail = min(positive, negative)
        p_value = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(tail + 1)) / (2 ** n))
    return {"positive": positive, "negative": negative, "ties": ties,
            "n_non_ties": n, "p_value": p_value,
            "at_least_9_of_10_positive": len(differences) == 10 and positive >= 9}


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for pos in range(start, end):
            ranks[order[pos]] = rank
        start = end
    return ranks


def spearman(differences: Sequence[float], covariate: Sequence[float]) -> dict:
    """Spearman correlation with average ranks for ties; undefined cases are null."""
    x = [float(v) for v in differences]
    y = [float(v) for v in covariate]
    if len(x) != len(y):
        raise ValueError("correlation vectors must have the same length")
    if len(x) < 2:
        return {"rho": None, "n": len(x)}
    rx, ry = _average_ranks(x), _average_ranks(y)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    dx, dy = [v - mx for v in rx], [v - my for v in ry]
    sx = math.sqrt(sum(v * v for v in dx))
    sy = math.sqrt(sum(v * v for v in dy))
    rho = None if sx == 0 or sy == 0 else sum(a * b for a, b in zip(dx, dy)) / (sx * sy)
    return {"rho": rho, "n": len(x)}


def _read_ids(tok, text: str, add_special_tokens: bool) -> list[int]:
    return list(tok(text, add_special_tokens=add_special_tokens)["input_ids"])


def start_context(tok, model) -> tuple[list[int], dict]:
    """Return the tokenizer's empty-string context, with an explicit BOS/EOS fallback."""
    ids = _read_ids(tok, "", add_special_tokens=True)
    source = "tokenizer_empty_with_special_tokens"
    if not ids:
        token_id = getattr(tok, "bos_token_id", None)
        source = "bos_token_id"
        if token_id is None:
            token_id = getattr(tok, "eos_token_id", None)
            source = "eos_token_id_fallback"
        if token_id is None:
            config = getattr(model, "config", None)
            token_id = getattr(config, "bos_token_id", None)
            source = "model_config_bos_token_id"
        if token_id is None:
            token_id = getattr(getattr(model, "config", None), "eos_token_id", None)
            source = "model_config_eos_token_id_fallback"
        if token_id is None:
            raise ValueError("tokenizer/model provide no start token for unigram priors")
        ids = [int(token_id)]
    return ids, {"source": source, "token_ids": ids}


def _next_token_logprob(model, context_ids: Sequence[int], token_id: int, device: str) -> float:
    ids = torch.tensor([list(context_ids)], dtype=torch.long, device=device)
    logits = model(ids).logits[0, -1].float()
    return float(torch.log_softmax(logits, dim=-1)[int(token_id)].item())


def first_token_prior(model, tok, word: str, context_ids: Sequence[int], device: str) -> dict:
    ids = _read_ids(tok, word, add_special_tokens=False)
    if not ids:
        return {"token_id": None, "token": None, "logprob": None, "probability": None}
    token_id = int(ids[0])
    try:
        token = tok.convert_ids_to_tokens([token_id])[0]
    except (AttributeError, TypeError):
        token = str(token_id)
    logprob = _next_token_logprob(model, context_ids, token_id, device)
    return {"token_id": token_id, "token": token, "logprob": logprob,
            "probability": math.exp(logprob)}


def _mean_surprisal(model, tok, word: str, device: str) -> dict:
    """Mean token surprisal for the word after the neutral quote prefix."""
    prefix_ids = _read_ids(tok, 'The word is "', add_special_tokens=True)
    word_ids = _read_ids(tok, word, add_special_tokens=False)
    if not word_ids:
        return {"mean_token_surprisal": None, "token_count": 0}
    ids = torch.tensor([prefix_ids + word_ids], dtype=torch.long, device=device)
    logits = model(ids).logits[0].float()
    logp = torch.log_softmax(logits, dim=-1)
    start = len(prefix_ids)
    values = [float(logp[start - 1 + i, token_id].item()) for i, token_id in enumerate(word_ids)]
    return {"mean_token_surprisal": -statistics.fmean(values), "token_count": len(word_ids)}


def _item_metadata(model, tok, word: str, context_ids: Sequence[int], device: str) -> dict:
    token_ids = _read_ids(tok, word, add_special_tokens=False)
    try:
        tokens = tok.convert_ids_to_tokens(token_ids)
    except AttributeError:
        tokens = [str(i) for i in token_ids]
    neutral = _mean_surprisal(model, tok, word, device)
    prior = first_token_prior(model, tok, word, context_ids, device)
    if len(word) >= 2:
        onset_ids = _read_ids(tok, " " + word[:2], add_special_tokens=False)
        onset_bigram_logprob = p48.continuation_logprob(model, tok, list(context_ids), word[:2], device)
    else:
        onset_ids, onset_bigram_logprob = [], None
    return {
        "token_count": len(token_ids),
        "token_ids": token_ids,
        "tokens": tokens,
        "mean_surprisal": neutral["mean_token_surprisal"],
        "surprisal_token_count": neutral["token_count"],
        "first_token_unigram_prior": prior,
        "onset_bigram": word[:2] if len(word) >= 2 else None,
        "onset_bigram_token_ids": onset_ids,
        "onset_bigram_logprob": onset_bigram_logprob,
    }


def _score_continuation_set(model, tok, prompt_ids: Sequence[int], words: Sequence[str], device: str) -> tuple[list[float], float, float]:
    if p48.ONE_PER_FORWARD:
        logprobs = [p48.continuation_logprob(model, tok, list(prompt_ids), word, device) for word in words]
    else:
        logprobs = p48.continuation_logprobs(model, tok, list(prompt_ids), list(words), device)
    item_score = p48.logsumexp(logprobs)
    mass = sum(math.exp(v) for v in logprobs)
    return logprobs, item_score, mass


def _item_row(model, tok, word: str, kind: str, pair_id: str | None, role: str,
              template_index: int, context_ids: Sequence[int], device: str) -> dict:
    template = p48.TEMPLATES[template_index]
    prompt = template.format(w=word)
    prompt_ids = _read_ids(tok, prompt, add_special_tokens=True)
    sharp_lp, sharp_lse, sharp_mass = _score_continuation_set(
        model, tok, prompt_ids, p48.SHARP_WORDS, device)
    round_lp, round_lse, round_mass = _score_continuation_set(
        model, tok, prompt_ids, p48.ROUND_WORDS, device)
    matched_sharp_lp, matched_sharp_lse, matched_sharp_mass = _score_continuation_set(
        model, tok, prompt_ids, LETTER_MATCHED_SHARP_WORDS, device)
    matched_round_lp, matched_round_lse, matched_round_mass = _score_continuation_set(
        model, tok, prompt_ids, LETTER_MATCHED_ROUND_WORDS, device)
    metadata = _item_metadata(model, tok, word, context_ids, device)
    return {
        "item": word,
        "kind": kind,
        "pair": pair_id,
        "twin": role,
        "template": template_index,
        "template_text": template,
        **metadata,
        "sharp_logprobs": sharp_lp,
        "round_logprobs": round_lp,
        "score": sharp_lse - round_lse,
        "continuation_mass": sharp_mass + round_mass,
        "letter_matched_sharp_logprobs": matched_sharp_lp,
        "letter_matched_round_logprobs": matched_round_lp,
        "letter_matched_score": matched_sharp_lse - matched_round_lse,
        "letter_matched_continuation_mass": matched_sharp_mass + matched_round_mass,
    }


def score_model(model, tok, device: str = "cpu", pairs: Sequence[tuple[str, str]] = PAIRS,
                include_controls: bool = True) -> list[dict]:
    """Score all paired items and single-letter controls, one sequence per forward."""
    context_ids, context_description = start_context(tok, model)
    rows = []
    with torch.inference_mode():
        for k_word, c_word in pairs:
            pair_id = f"{k_word}/{c_word}"
            for role, word in (("k", k_word), ("c", c_word)):
                kind = "twin"
                for ti in range(len(p48.TEMPLATES)):
                    row = _item_row(model, tok, word, kind, pair_id, role, ti,
                                    context_ids, device)
                    row["unigram_prior_context"] = context_description
                    rows.append(row)
        if include_controls:
            for word, role in (("k", "k"), ("c", "c")):
                for ti in range(len(p48.TEMPLATES)):
                    row = _item_row(model, tok, word, "single_letter_control", None,
                                    role, ti, context_ids, device)
                    row["unigram_prior_context"] = context_description
                    rows.append(row)
    return rows


def _pair_records(rows: Sequence[dict], pairs: Sequence[tuple[str, str]], mass_floor: float = None) -> list[dict]:
    mass_floor = MASS_FLOOR if mass_floor is None else mass_floor
    by_item_template = {(r["item"], r["template"]): r for r in rows if r.get("kind") == "twin"}
    # Part 48's readout exactly (PART48B-DESIGN.md): keep a template only if the median continuation mass of its
    # twin rows is >= 1e-3, as part48.summarize does; item scores average over the kept templates only.
    # (Claude, 2026-10-21: the first build averaged over all six, which Part 48 never did.)
    kept = [ti for ti in range(len(p48.TEMPLATES))
            if statistics.median(r["continuation_mass"] for r in rows
                                 if r.get("kind") == "twin" and r["template"] == ti) >= mass_floor]
    if not kept:
        raise RuntimeError("no template passes the 1e-3 continuation-mass floor")
    records = []
    for k_word, c_word in pairs:
        k_rows = [by_item_template.get((k_word, ti)) for ti in range(len(p48.TEMPLATES))]
        c_rows = [by_item_template.get((c_word, ti)) for ti in range(len(p48.TEMPLATES))]
        if any(r is None for r in k_rows + c_rows):
            raise RuntimeError(f"pair {k_word}/{c_word} is missing template rows")
        k_full = statistics.fmean(k_rows[ti]["score"] for ti in kept)
        c_full = statistics.fmean(c_rows[ti]["score"] for ti in kept)
        k_matched = statistics.fmean(k_rows[ti]["letter_matched_score"] for ti in kept)
        c_matched = statistics.fmean(c_rows[ti]["letter_matched_score"] for ti in kept)
        k_first, c_first = k_rows[0], c_rows[0]
        template_scores = []
        for ti, (kr, cr) in enumerate(zip(k_rows, c_rows)):
            template_scores.append({
                "template": ti,
                "k_score": kr["score"], "c_score": cr["score"],
                "d": kr["score"] - cr["score"],
                "letter_matched_k_score": kr["letter_matched_score"],
                "letter_matched_c_score": cr["letter_matched_score"],
                "letter_matched_d": kr["letter_matched_score"] - cr["letter_matched_score"],
            })
        records.append({
            "templates_kept": kept,
            "pair": {"k": k_word, "c": c_word},
            "k_score": k_full, "c_score": c_full, "d": k_full - c_full,
            "letter_matched": {"k_score": k_matched, "c_score": c_matched,
                                "d": k_matched - c_matched},
            "token_count": {"k": k_first["token_count"], "c": c_first["token_count"],
                            "gap_k_minus_c": k_first["token_count"] - c_first["token_count"]},
            "mean_surprisal": {"k": k_first["mean_surprisal"], "c": c_first["mean_surprisal"],
                               "gap_k_minus_c": k_first["mean_surprisal"] - c_first["mean_surprisal"]},
            "first_token_unigram_prior": {"k": k_first["first_token_unigram_prior"],
                                          "c": c_first["first_token_unigram_prior"]},
            "onset_bigram_prior": {"k": k_first["onset_bigram_logprob"],
                                   "c": c_first["onset_bigram_logprob"]},
            "continuation_mass": {
                "k_by_template": [r["continuation_mass"] for r in k_rows],
                "c_by_template": [r["continuation_mass"] for r in c_rows],
                "letter_matched_k_by_template": [r["letter_matched_continuation_mass"] for r in k_rows],
                "letter_matched_c_by_template": [r["letter_matched_continuation_mass"] for r in c_rows],
            },
            "per_template": template_scores,
        })
    return records


def _analyze_records(records: Sequence[dict], score_set: str) -> dict:
    if score_set == "part48_continuations":
        value = lambda r: r["d"]
        template_value = lambda r, ti: r["per_template"][ti]["d"]
    else:
        value = lambda r: r["letter_matched"]["d"]
        template_value = lambda r, ti: r["per_template"][ti]["letter_matched_d"]
    differences = [float(value(r)) for r in records]
    gaps = [r["mean_surprisal"]["gap_k_minus_c"] for r in records]
    count_gaps = [r["token_count"]["gap_k_minus_c"] for r in records]
    if any(x is None for x in gaps):
        fluency_spearman = {"rho": None, "n": len(records)}
    else:
        fluency_spearman = spearman(differences, gaps)
    by_template = []
    for ti in range(len(p48.TEMPLATES)):
        ds = [float(template_value(r, ti)) for r in records]
        by_template.append({
            "template": ti,
            "n_pairs": len(ds),
            "pair_d": ds,
            "median_d": statistics.median(ds) if ds else None,
            "sign_test": exact_sign_test(ds),
        })
    return {
        "n_pairs": len(records),
        "pair_d": differences,
        "mean_d": statistics.fmean(differences) if differences else None,
        "median_d": statistics.median(differences) if differences else None,
        "sign_flip_permutation": sign_flip_permutation_test(differences),
        "bootstrap_median_95_ci": bootstrap_median_ci(differences),
        "conservative_sign_test": exact_sign_test(differences),
        "spearman_d_with_surprisal_gap": fluency_spearman,
        "spearman_d_with_token_count_gap": spearman(differences, count_gaps),
        "per_template": by_template,
    }


def summarize(rows: Sequence[dict], pairs: Sequence[tuple[str, str]] = PAIRS, mass_floor: float = None) -> dict:
    """Summarize paired scores and diagnostics, preserving full and sensitivity sets."""
    records = _pair_records(rows, pairs, mass_floor=mass_floor)
    near_ids = {f"{k}/{c}" for k, c in NEAR_WORD_PAIRS}
    near_filtered = [r for r in records
                     if f"{r['pair']['k']}/{r['pair']['c']}" not in near_ids]
    equal_count = [r for r in records if r["token_count"]["k"] == r["token_count"]["c"]]
    analyses = {}
    for mode in ("part48_continuations", "letter_matched"):
        analyses[mode] = {
            "all_pairs": _analyze_records(records, mode),
            "equal_token_count": _analyze_records(equal_count, mode),
            "without_near_words": _analyze_records(near_filtered, mode),
            "primary_subset": "equal_token_count",
        }
    controls = {}
    for role in ("k", "c"):
        items = [r for r in rows if r.get("kind") == "single_letter_control" and r.get("twin") == role]
        controls[role] = {
            "mean_score": statistics.fmean(r["score"] for r in items) if items else None,
            "mean_letter_matched_score": statistics.fmean(r["letter_matched_score"] for r in items) if items else None,
            "per_template": [{"template": ti,
                              "score": next((r["score"] for r in items if r["template"] == ti), None),
                              "letter_matched_score": next((r["letter_matched_score"] for r in items if r["template"] == ti), None)}
                             for ti in range(len(p48.TEMPLATES))],
        }
    return {
        "pair_count": len(records),
        "near_word_excluded_pairs": [list(pair) for pair in sorted(NEAR_WORD_PAIRS)],
        "near_word_exclusion_forms": sorted(NEAR_WORD_FORMS),
        "primary_token_count_rule": "unequal tokenizer token counts are excluded from the primary analysis",
        "scores": analyses,
        "pair_records": records,
        "single_letter_controls": controls,
        "letter_matched_sets": {"sharp": list(LETTER_MATCHED_SHARP_WORDS),
                                 "round": list(LETTER_MATCHED_ROUND_WORDS)},
    }


def pool(model_summaries: Sequence[dict]) -> dict:
    """Amendment 5: per pair, the unweighted mean of per-model d; then the primary tests over those pair means."""
    if not model_summaries:
        raise RuntimeError("nothing to pool")
    order = [r["pair"] for r in model_summaries[0]["pair_records"]]
    if any([r["pair"] for r in m["pair_records"]] != order for m in model_summaries):
        raise RuntimeError("models disagree on the pair list; cannot pool")
    out = {"pairs": order, "n_models": len(model_summaries)}
    for score_set in ("part48_continuations", "letter_matched"):
        per_model = [m["scores"][score_set]["all_pairs"]["pair_d"] for m in model_summaries]
        pooled = [statistics.fmean(pd[i] for pd in per_model) for i in range(len(order))]
        out[score_set] = {
            "pooled_pair_d": pooled,
            "median_d": statistics.median(pooled),
            "sign_flip_permutation": sign_flip_permutation_test(pooled),
            "bootstrap_median_95_ci": bootstrap_median_ci(pooled),
            "conservative_sign_test": exact_sign_test(pooled),
        }
    return out


def _write_json_atomic(path: str, obj) -> None:
    """Write to a temporary file in the same directory, then rename over the target (Luna review P3)."""
    tmp = f"{path}.tmp-{os.getpid()}"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(p48.MODELS))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--out", default="results/part48b")
    ap.add_argument("--pool", action="store_true", help="pool the per-model JSONs in --out (amendment 5)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.pool:
        names = [n.strip() for n in args.models.split(",")]
        sums = []
        for n in names:
            with open(os.path.join(args.out, f"{n}.json"), encoding="utf-8") as f:
                sums.append(json.load(f)["summary"])
        _write_json_atomic(os.path.join(args.out, "pooled.json"), dict(part="48b", models=names, pooled=pool(sums)))
        return
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers.modeling_utils as modeling_utils

    if hasattr(modeling_utils, "caching_allocator_warmup"):
        modeling_utils.caching_allocator_warmup = lambda *x, **k: None
    torch.set_grad_enabled(False)
    for short in args.models.split(","):
        short = short.strip()
        mid = p48.MODELS[short]
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(mid)
        model = AutoModelForCausalLM.from_pretrained(
            mid,
            dtype=torch.float32 if args.dtype == "fp32" else torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map={"": args.device},
        )
        model.eval()
        rows = score_model(model, tok, args.device)
        summary = summarize(rows)
        rec = dict(part="48b", model=mid,
                   revision=getattr(model.config, "_commit_hash", None),
                   dtype=args.dtype, batching="one sequence per forward",
                   limit=None, seconds=round(time.time() - t0, 1),
                   summary=summary, rows=rows)
        path = os.path.join(args.out, f"{short}.json")
        _write_json_atomic(path, rec)
        print(short, json.dumps({name: result["equal_token_count"]["median_d"]
                                 for name, result in summary["scores"].items()}), flush=True)
        del model
        if args.device == "xpu":
            torch.xpu.empty_cache()


if __name__ == "__main__":
    main()
