#!/usr/bin/env python3
"""distances211.py: the distance files of PART211-DESIGN.md (design: Reading rules, "score211.py reads only the distance
files"), built from the records and the noise floors, one file per text set.

What each file holds is score211's data contract (the comment block above its predictions): per-document arrays,
documents first, plus the draws (`idx`, seed 211, B = 1000) that every per-draw quantity was computed in, so the scorer
cannot resample differently from M_boot.

- Pythia, per run r: P_within (head-mean within-run P(s, final) per document and checkpoint), V (per head, per
  checkpoint), M_point and M_boot (M(s, final) over all documents and per draw, exact pooled CKA from document counts),
  cross (per-document P between r's head i and R0's head j at steps 0, 10000 and 143000, from the reference rows),
  and beside it, for the reported columns only: P_within_heads (per head), P0s_within/P0k_within/P0t (P0 as additive
  parts), cross_P0s/cross_P0k and cross_cosd/cross_cosx/cross_cosy (P0 and the flattened cosine, as parts). R0
  also: P_thread (head-mean P(1000, 10000)) with P0s_thread/P0k_thread beside it, and between_heads (mean P between different heads of a layer at R0's
  final). The floor per layer and measure is the largest over the six runs' finals (a stricter reading than any one
  run's; the per-run floors are kept beside it).
- OLMo: P for (mid, instruct), (3T, 4T), (21B, 4T) and (step0, 4T); M_point and M_boot for the first two; the floor is
  4T's (the final of 211.6's late distances and of 211.7); mid-training end's floor is kept beside it.

  python -I -B part211/distances211.py --out results/part211
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import checkpoints211 as C  # noqa: E402
import measures211 as MS  # noqa: E402
import record211 as R  # noqa: E402
import score211 as S  # noqa: E402

CROSS_STEPS = (0, C.THREAD_STEP, C.FINAL_STEP)
CROSS_PARTS = ("P0s", "P0k", "cosd", "cosx", "cosy")     # beside "cross" (P), per step
OLMO_P_PAIRS = {("mid", "instruct"): ("olmo_midtrain_51B", "olmo_instruct"),
                ("3T", "4T"): ("olmo_pretrain_3T", "olmo_pretrain_4T"),
                ("21B", "4T"): ("olmo_pretrain_21B", "olmo_pretrain_4T"),
                ("step0", "4T"): ("olmo_step0", "olmo_pretrain_4T")}
OLMO_M_PAIRS = (("mid", "instruct"), ("3T", "4T"))
OLMO_FLOOR_KEY, OLMO_INSTRUCT_FLOOR_KEY = "olmo_pretrain_4T", "olmo_midtrain_51B"


class Missing(RuntimeError):
    """A record, a reduction or a floor the distance file needs is not on disk."""


def _load(out: Path, key: str, text_set: str):
    path = R.record_path(out, key, text_set)
    if not path.is_file():
        raise Missing(f"no record {path}")
    z = np.load(path, allow_pickle=False)
    if "complete" not in z.files or not bool(z["complete"]):
        raise Missing(f"incomplete record {path}")
    return z


def _floor(out: Path, key: str, text_set: str):
    path = R.floor_path(out, key, text_set)
    if not path.is_file():
        raise Missing(f"no floor {path}")
    return np.load(path, allow_pickle=False)


def _diag(rec, ref_key: str, part: str) -> np.ndarray:
    """A record's per-document reduction part against a reference of its own run, head with the same head: [n_docs, L, H]."""
    k = f"red_{ref_key}_{part}"
    if k not in rec.files:
        raise Missing(f"record has no {part} reduction against {ref_key}")
    return np.diagonal(rec[k], axis1=2, axis2=3)


def _p0t(rec, ref_key: str) -> int:
    return int(rec[f"red_{ref_key}_P0t"])


def _head_mean_within(rec, ref_key: str) -> np.ndarray:
    """A record's per-document P against a reference of its own run, head with the same head: [n_docs, L]."""
    k = f"red_{ref_key}_P"
    if k not in rec.files:
        raise Missing(f"record has no reduction against {ref_key}")
    return np.diagonal(rec[k], axis1=2, axis2=3).mean(axis=-1)


def _m(mlp_a: np.ndarray, mlp_b: np.ndarray, counts: np.ndarray):
    """M = 1 - pooled CKA over all documents and per draw, per layer: ([L], [B, L])."""
    n_layers = mlp_a.shape[1]
    point, boot = np.zeros(n_layers), np.zeros((counts.shape[0], n_layers))
    ones = np.ones(mlp_a.shape[0])
    for L in range(n_layers):
        st = MS.cka_count_stats(mlp_a[:, L], mlp_b[:, L])
        point[L] = 1.0 - MS.cka_from_counts(st, ones)
        boot[:, L] = 1.0 - MS.cka_from_counts(st, counts)
    return point, boot


def _cross(att_r: np.ndarray, att_0: np.ndarray) -> np.ndarray:
    """Per-document P between every head of run r and every head of R0: [n_docs, L, H, H]."""
    return _cross_parts(att_r, att_0)["P"]


def _cross_parts(att_r: np.ndarray, att_0: np.ndarray) -> dict:
    """Per-document P, the P0 parts and the cosine parts between every head of run r and every head of R0, from the
    reference rows (keys as record211._empty_reduction): P, P0s, P0k [n_docs, L, H, H], P0t, cosd [n_docs, L, H, H],
    cosx, cosy [n_docs, L, H]."""
    n_docs, n_layers, n_heads = att_r.shape[:3]
    red = R._empty_reduction(n_docs, n_layers, n_heads)
    for d in range(n_docs):
        for L in range(n_layers):
            R._reduce_into(red, d, L, att_r[d, L].astype(np.float32), att_0[d, L].astype(np.float32))
    return red


def _between_heads(att: np.ndarray) -> np.ndarray:
    """The mean P between different heads of each layer (all documents): [L]."""
    n_docs, n_layers, n_heads = att.shape[:3]
    out = np.zeros(n_layers)
    off = ~np.eye(n_heads, dtype=bool)
    for L in range(n_layers):
        m = np.mean([MS.pattern_matrix(att[d, L], att[d, L]) for d in range(n_docs)], axis=0)
        out[L] = m[off].mean()
    return out


def pythia_data(out: Path, text_set: str, idx: np.ndarray) -> tuple[dict, dict]:
    """The Pythia half of one distance file: (data, floor)."""
    out = Path(out)
    runs = {}
    n_docs = None
    counts = None
    for r in C.runs_of(C.PYTHIA_FAMILY):
        final_key = C.pythia_key(r, C.FINAL_STEP)
        fin = _load(out, final_key, text_set)
        if n_docs is None:
            n_docs = fin["mlp"].shape[0]
            counts = MS.counts_of(idx, n_docs)
        n_layers, n_heads = fin["ov_v"].shape[:2]
        n_steps = len(S.PYTHIA_STEPS)
        p_within = np.zeros((n_docs, n_layers, n_steps))
        p_heads = np.zeros((n_docs, n_layers, n_heads, n_steps))
        p0s = np.zeros((n_docs, n_layers, n_heads, n_steps))
        p0k = np.zeros((n_docs, n_layers, n_heads, n_steps), np.int64)
        p0t = 0
        v = np.zeros((n_layers, n_heads, n_steps))
        m_point, m_boot = np.zeros((n_layers, n_steps)), np.zeros((counts.shape[0], n_layers, n_steps))
        for si, step in enumerate(S.PYTHIA_STEPS):
            if step == C.FINAL_STEP:
                continue                                   # X(final, final) = 0 by definition, never read
            rec = _load(out, C.pythia_key(r, step), text_set)
            p_within[:, :, si] = _head_mean_within(rec, final_key)
            p_heads[:, :, :, si] = _diag(rec, final_key, "P")
            p0s[:, :, :, si], p0k[:, :, :, si] = _diag(rec, final_key, "P0s"), _diag(rec, final_key, "P0k")
            p0t = _p0t(rec, final_key)
            if "V_final" not in rec.files:
                raise Missing(f"{C.pythia_key(r, step)}.{text_set} has no V_final")
            v[:, :, si] = rec["V_final"]
            m_point[:, si], m_boot[:, :, si] = _m(rec["mlp"], fin["mlp"], counts)
        runs[r] = {"P_within": p_within, "V": v, "M_point": m_point, "M_boot": m_boot, "cross": {},
                   "P_within_heads": p_heads, "P0s_within": p0s, "P0k_within": p0k, "P0t": np.array(p0t)}
        for part in CROSS_PARTS:
            runs[r][f"cross_{part}"] = {}
    for r in runs:
        if r == "R0":
            continue
        for step in CROSS_STEPS:
            a = _load(out, C.pythia_key(r, step), text_set)["att"]
            b = _load(out, C.pythia_key("R0", step), text_set)["att"]
            parts = _cross_parts(a, b)
            runs[r]["cross"][step] = parts["P"]
            for part in CROSS_PARTS:
                runs[r][f"cross_{part}"][step] = parts[part]
    r0_1000 = _load(out, C.pythia_key("R0", 1000), text_set)
    thread_ref = C.pythia_key("R0", C.THREAD_STEP)
    runs["R0"]["P_thread"] = _head_mean_within(r0_1000, thread_ref)
    runs["R0"]["P0s_thread"], runs["R0"]["P0k_thread"] = _diag(r0_1000, thread_ref, "P0s"), _diag(r0_1000, thread_ref, "P0k")
    runs["R0"]["between_heads"] = _between_heads(_load(out, C.pythia_key("R0", C.FINAL_STEP), text_set)["att"])
    per_run = {r: _floor(out, C.pythia_key(r, C.FINAL_STEP), text_set) for r in runs}
    floor = {k: np.max([per_run[r][k] for r in runs], axis=0) for k in ("P", "M", "V")}
    # P0's floor, the same rule; a layer with no P0 in any run's floor stays NaN (no P0 reading there)
    with np.errstate(invalid="ignore"):
        p0f = np.array([per_run[r]["P0"] for r in runs], np.float64)
        floor["P0"] = np.where(np.isfinite(p0f).any(axis=0), np.nanmax(np.where(np.isfinite(p0f), p0f, -np.inf), axis=0),
                               np.nan)
    floor["per_run"] = {r: {k: per_run[r][k] for k in ("P", "M", "V", "P0")} for r in runs}
    # 211.8's compared checkpoints are step 10000 of R0 and of each weight-seed run (floor policy, BUILD-NOTES 8)
    floor["thread"] = {r: {k: f[k] for k in ("P", "M", "V", "P0")}
                       for r in ("R0", *C.runs_with_factor("weights"))
                       for f in [_floor(out, C.pythia_key(r, C.THREAD_STEP), text_set)]}
    return {"runs": runs}, floor


def olmo_data(out: Path, text_set: str, idx: np.ndarray) -> tuple[dict, dict]:
    """The OLMo half of one distance file: (data, floor)."""
    out = Path(out)
    p = {}
    n_docs = None
    for pair, (a, b) in OLMO_P_PAIRS.items():
        rec = _load(out, a, text_set)
        p[pair] = _head_mean_within(rec, b)
        n_docs = p[pair].shape[0]
    counts = MS.counts_of(idx, n_docs)
    m_point, m_boot = {}, {}
    for pair in OLMO_M_PAIRS:
        a, b = OLMO_P_PAIRS[pair]
        m_point[pair], m_boot[pair] = _m(_load(out, a, text_set)["mlp"], _load(out, b, text_set)["mlp"], counts)
    f = _floor(out, OLMO_FLOOR_KEY, text_set)
    fi = _floor(out, OLMO_INSTRUCT_FLOOR_KEY, text_set)
    floor = {"P": f["P"], "M": f["M"], "instruct": {"P": fi["P"], "M": fi["M"]}}
    return {"P": p, "M_point": m_point, "M_boot": m_boot}, floor


# ---------------------------------------------------------------- the file: flat keys, no pickles

def _pair(k) -> str:
    return "|".join(k)


def save(path: Path, text_set: str, idx: np.ndarray, pythia: tuple, olmo: tuple | None) -> Path:
    arrays = {"idx": np.asarray(idx), "text_set": np.array(text_set)}
    data, floor = pythia
    for r, run in data["runs"].items():
        for k, v in run.items():
            if k == "cross" or k.startswith("cross_"):
                for step, m in v.items():
                    arrays[f"pythia/{r}/{k}/{step}"] = m
            else:
                arrays[f"pythia/{r}/{k}"] = np.asarray(v)
    for k in ("P", "M", "V", "P0"):
        arrays[f"pythia_floor/{k}"] = np.asarray(floor[k])
    for r, f in floor["per_run"].items():
        for k, v in f.items():
            arrays[f"pythia_floor_run/{r}/{k}"] = np.asarray(v)
    for r, f in floor["thread"].items():
        for k, v in f.items():
            arrays[f"pythia_floor_thread/{r}/{k}"] = np.asarray(v)
    if olmo is not None:
        data, floor = olmo
        for pair, v in data["P"].items():
            arrays[f"olmo/P/{_pair(pair)}"] = v
        for pair in data["M_point"]:
            arrays[f"olmo/M_point/{_pair(pair)}"] = data["M_point"][pair]
            arrays[f"olmo/M_boot/{_pair(pair)}"] = data["M_boot"][pair]
        arrays["olmo_floor/P"], arrays["olmo_floor/M"] = np.asarray(floor["P"]), np.asarray(floor["M"])
        arrays["olmo_floor_instruct/P"] = np.asarray(floor["instruct"]["P"])
        arrays["olmo_floor_instruct/M"] = np.asarray(floor["instruct"]["M"])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
    tmp.replace(path)
    return path


def load(path: Path) -> dict:
    """{"idx", "text_set", "pythia", "pythia_floor", "olmo" (or None), "olmo_floor" (or None)} as score_all takes."""
    z = np.load(path, allow_pickle=False)
    runs, pfloor, olmo = {}, {}, {"P": {}, "M_point": {}, "M_boot": {}}
    ofloor = {}
    for k in z.files:
        parts = k.split("/")
        if parts[0] == "pythia":
            run = runs.setdefault(parts[1], {"cross": {}})
            if parts[2] == "cross" or parts[2].startswith("cross_"):
                run.setdefault(parts[2], {})[int(parts[3])] = z[k]
            else:
                run[parts[2]] = z[k]
        elif parts[0] == "pythia_floor":
            pfloor[parts[1]] = z[k].tolist()
        elif parts[0] in ("pythia_floor_run", "pythia_floor_thread"):
            # per-run finals (211.3's compatibility tolerance) and step-10000 floors (211.8); the first build saved
            # pythia_floor_run and dropped it here (Alden #13)
            where = "per_run" if parts[0] == "pythia_floor_run" else "thread"
            pfloor.setdefault(where, {}).setdefault(parts[1], {})[parts[2]] = z[k].tolist()
        elif parts[0] == "olmo":
            olmo[parts[1]][tuple(parts[2].split("|"))] = z[k]
        elif parts[0] == "olmo_floor":
            ofloor[parts[1]] = z[k].tolist()
        elif parts[0] == "olmo_floor_instruct":
            # the mid-training-end floor, printed beside 211.6's numerator (reported only; no verdict reads it)
            ofloor.setdefault("instruct", {})[parts[1]] = z[k].tolist()
    has_olmo = bool(olmo["P"])
    return {"idx": z["idx"], "text_set": str(z["text_set"]), "pythia": {"runs": runs}, "pythia_floor": pfloor,
            "olmo": olmo if has_olmo else None, "olmo_floor": ofloor if has_olmo else None}


def build(out: Path, text_set: str, n_boot: int = S.N_BOOT, olmo: bool = True) -> Path:
    """One text set's distance file, written to <out>/distances/<set>.npz."""
    out = Path(out)
    n_docs = _load(out, C.pythia_key("R0", C.FINAL_STEP), text_set)["mlp"].shape[0]
    idx = S.resample_indices(n_docs, n_boot)
    py = pythia_data(out, text_set, idx)
    ol = olmo_data(out, text_set, idx) if olmo else None
    return save(out / "distances" / f"{text_set}.npz", text_set, idx, py, ol)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/part211")
    ap.add_argument("--sets", default="A,B")
    a = ap.parse_args(argv)
    for t in a.sets.split(","):
        try:
            print(build(Path(a.out), t))
        except Missing as exc:
            print(f"set {t}: {exc}")
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
