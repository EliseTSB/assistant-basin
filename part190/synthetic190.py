#!/usr/bin/env python3
"""Synthetic fixtures for Part 190 with a known answer. Never touches the real arrays.

Animal groups sit at random centres; an imitation is its group's centre pushed along a hidden direction h by
`signal` (0 gives no real-against-imitation information at all). Each group's prompt difference (real minus
imitation prompt) is mostly a random direction with a share `verdict_align` along h, so the unedited verdict is weak
and an edit that reads h and writes into the prompt directions should raise it. ESC-50-like classes (sets E and H) and
one-clip non-animal imitations (set N) sit at their own centres, orthogonal to h by construction.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

FOLD_GROUPS = {"A": ["Ga1", "Ga2", "Ga3", "Ga4"], "B": ["Gb1", "Gb2", "Gb3", "Gb4"]}
FIELDS = ["id", "set", "analysis_set", "source", "class_key", "group", "phrase", "fold", "esc_category"]


def make_fixture(*, dim: int = 48, n_real: int = 30, n_imitation: int = 24, n_esc_classes: int = 6,
                 n_per_esc: int = 12, n_human_classes: int = 2, n_nonanimal: int = 20, signal: float = 1.0,
                 verdict_align: float = 0.25, noise: float = 0.35, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    h = rng.standard_normal(dim)
    h /= np.linalg.norm(h)

    def orth(v):
        v = v - (v @ h) * h
        return v

    rows, emb = [], []
    phrases, real_p, imit_p = [], [], []

    def add(set_name, fold, group, phrase, vec, source="vis", esc=""):
        i = len(rows)
        rows.append({"id": f"X{i:05d}", "set": set_name, "analysis_set": set_name, "source": source,
                     "class_key": phrase if source == "vis" else esc, "group": group, "phrase": phrase,
                     "fold": fold, "esc_category": esc})
        emb.append(vec)

    def add_prompt(phrase):
        diff = verdict_align * h + np.sqrt(1 - verdict_align ** 2) * orth(rng.standard_normal(dim)) / np.sqrt(dim)
        base = rng.standard_normal(dim) / np.sqrt(dim)
        phrases.append(phrase)
        real_p.append(base + 0.5 * diff)
        imit_p.append(base - 0.5 * diff)

    groups = [(fold, g) for fold, gs in FOLD_GROUPS.items() for g in gs] + [("C", "Chicken")]
    for fold, g in groups:
        centre = orth(rng.standard_normal(dim)) * 1.0
        phrase = f"a {g.lower()} call" if g != "Chicken" else "a chicken clucking"
        add_prompt(phrase)
        r_set, i_set = ("C_R", "C_I") if fold == "C" else ("R", "I")
        for j in range(n_real):
            add(r_set, fold, g, phrase, centre + 0.5 * signal * h + noise * rng.standard_normal(dim),
                source="esc" if (fold == "C" and j % 2) else "vis", esc="hen" if (fold == "C" and j % 2) else "")
        for _ in range(n_imitation):
            add(i_set, fold, g, phrase, centre - 0.5 * signal * h + noise * rng.standard_normal(dim))
    add_prompt("an animal")
    for c in range(n_esc_classes + n_human_classes):
        set_name = "H" if c >= n_esc_classes else "E"
        centre = orth(rng.standard_normal(dim))
        for _ in range(n_per_esc):
            add(set_name, "", f"esc{c}", "", centre + noise * rng.standard_normal(dim), source="esc", esc=f"esc{c}")
    for c in range(n_nonanimal):
        add("N", "", f"n{c}", "", orth(rng.standard_normal(dim)) + noise * rng.standard_normal(dim))
    x = np.asarray(emb)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    counts = {}
    for r in rows:
        if r["fold"]:
            counts.setdefault(r["fold"], {}).setdefault(r["analysis_set"], 0)
            counts[r["fold"]][r["analysis_set"]] += 1
    return {"rows": rows, "ids": np.asarray([r["id"] for r in rows]), "embeddings": x.astype(np.float16),
            "prompt_phrases": np.asarray(phrases), "real": np.asarray(real_p).astype(np.float16),
            "imitation": np.asarray(imit_p).astype(np.float16), "folds": {"counts_by_analysis_set": counts},
            "h": h}


def write_fixture(fx: dict, out: Path, registered188: dict | None = None) -> dict[str, Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = {"emb": out / "emb.npz", "manifest": out / "manifest.csv", "folds": out / "folds.json",
             "results188": out / "results188.json"}
    np.savez_compressed(paths["emb"], ids=fx["ids"], embeddings=fx["embeddings"],
                        seqlen=np.zeros(len(fx["ids"]), np.int32), prompt_phrases=fx["prompt_phrases"],
                        prompt_real_texts=fx["prompt_phrases"], prompt_imitation_texts=fx["prompt_phrases"],
                        prompt_real_embeddings=fx["real"], prompt_imitation_embeddings=fx["imitation"])
    with paths["manifest"].open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(fx["rows"])
    paths["folds"].write_text(json.dumps(fx["folds"]), encoding="utf-8")
    paths["results188"].write_text(json.dumps(registered188 or {}), encoding="utf-8")
    return paths


def registered_from_fixture(fx: dict) -> dict:
    """What Part 188's results.json would hold for this fixture, computed with Part 188's own definitions."""
    from part190.analyze190 import (DIRECTIONS, auc, data_from_arrays, evaluate, make_context)
    from part188.analyze188 import fit_logistic_l2, logistic_scores
    data = data_from_arrays(fx["rows"], fx["ids"], fx["embeddings"], fx["prompt_phrases"], fx["real"],
                            fx["imitation"])
    out = {"directions": {}}
    for name, fit, test in DIRECTIONS:
        ctx = make_context(data, name, fit, test)
        probe = fit_logistic_l2(data.x[ctx.fit_idx], ctx.fit_y.astype(float))
        out["directions"][name] = {
            "model_class_informed": {"auc": evaluate(data, ctx, data.x)["V_test"]},
            "output_embedding_probe": {"auc": auc(ctx.test_y, logistic_scores(probe, data.x[ctx.test_idx]))}}
    return out
