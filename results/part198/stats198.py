"""Part 198 statistics, frozen with the design (PART198-DESIGN.md, revision 2). Everything here reads one suspect model's
weights; nothing needs the base model.

Statistics per MLP output matrix W, [out, in] orientation, float64:
  F1  top singular value;  F2  its share of the squared spectrum;
  PCS mean pairwise cosine similarity of the per-neuron output vectors (the columns of W);
  token reading over the top-64 left singular vectors, both signs: T_max, T_word, and the pairs themselves.

Norm contract (Alden, council part198-design-2026-11-08 #3): a LayerNorm reads a direction u as gamma * (u - mean u)
(its bias is omitted: u is a direction, and LayerNorm(ones) is the bias, not gamma); any RMSNorm reads u as
effective_scale * u, where the effective scale is the module's output on a vector of ones (1 + w for offset
parameterisations, w otherwise; the common 1/rms factor cancels in z).
Sign and filter rule: every (direction, sign) pair is scored; T_word is the max z over ELIGIBLE pairs, so a function-word
sign never hides an eligible opposite sign. A pair whose logits have zero variance is dropped. No eligible pair: None.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import torch

K_DIRS = 64

# Frozen stopword list (identical to explore_posctl.py's, which produced the pilot's T_word values). Do not edit after
# registration.
STOPWORDS = frozenset("""a about above after again against all am an and any are as at be because been before being below
between both but by can did do does doing down during each few for from further had has have having he her here hers
herself him himself his how i if in into is it its itself just me more most my myself no nor not now of off on once only
or other our ours ourselves out over own same she should so some such than that the their theirs them themselves then
there these they this those through to too under until up very was we were what when where which while who whom why will
with you your yours yourself yourselves also would could may might must shall one two get got make makes made
like""".split())

_WORD = re.compile(r"[A-Za-z]{3,}")


def is_content(decoded: str) -> bool:
    s = decoded.strip()
    return bool(_WORD.fullmatch(s)) and s.lower() not in STOPWORDS


def norm_reader(norm_module: torch.nn.Module) -> tuple[torch.Tensor, bool]:
    """(per-dimension gain, centre?) for reading a residual direction through the final norm."""
    if isinstance(norm_module, torch.nn.LayerNorm) or (
            "LayerNorm" in type(norm_module).__name__ and "RMS" not in type(norm_module).__name__):
        w = getattr(norm_module, "weight", None)
        if w is None:
            raise ValueError("LayerNorm without a weight: gain is undefined")
        return w.detach().double().clone(), True
    d = None
    for p in norm_module.parameters():
        d = p.shape[-1]
        break
    if d is None:
        raise ValueError("norm module has no parameters: cannot infer width")
    p0 = next(norm_module.parameters())
    with torch.no_grad():
        g = norm_module(torch.ones(1, d, dtype=p0.dtype, device=p0.device))[0].detach().double().cpu()
    return g, False


@dataclass
class Pair:
    j: int          # singular-direction index (0 = largest)
    sign: int       # +1 or -1
    z: float
    top: int        # top token id
    top3: list[int]
    eligible: bool


def read_pairs(W: torch.Tensor, gain: torch.Tensor, centre: bool, E: torch.Tensor, decode, k: int = K_DIRS) -> list[Pair]:
    """Score every (direction, sign) pair of W's top-k left singular vectors through gain and unembedding E."""
    U, _, _ = torch.linalg.svd(W.double(), full_matrices=False)
    U = U[:, :k]
    if centre:
        U = U - U.mean(0, keepdim=True)
    L = (E.double() @ (gain[:, None] * U))  # [vocab, k]
    pairs: list[Pair] = []
    elig_cache: dict[int, bool] = {}
    for j in range(L.shape[1]):
        for sign in (1, -1):
            col = sign * L[:, j]
            sd = float(col.std())
            if not sd > 0:
                continue
            mx = col.max()
            z = float((mx - col.mean()) / sd)
            # Frozen tie rule: equal logits order by token id, smallest first (target-independent).
            order = sorted(torch.topk(col, min(8, col.shape[0])).indices.tolist(), key=lambda i: (-float(col[i]), i))
            cands = (col == mx).nonzero().flatten().tolist()
            t = min(cands)
            top3 = [t] + [i for i in order if i != t][:2]
            if t not in elig_cache:
                elig_cache[t] = is_content(decode([t]))
            pairs.append(Pair(j, sign, z, t, top3, elig_cache[t]))
    return pairs


def _pair_key(p: Pair):
    # Frozen tie rule between pairs: higher z, then smaller direction index, then sign +1 before -1.
    return (p.z, -p.j, p.sign)


def t_max(pairs: list[Pair]) -> float | None:
    return max((p.z for p in pairs), default=None)


def t_word(pairs: list[Pair]) -> float | None:
    return max((p.z for p in pairs if p.eligible), default=None)


def top_eligible(pairs: list[Pair]) -> Pair | None:
    el = [p for p in pairs if p.eligible]
    return max(el, key=_pair_key) if el else None


def blind_reader(layer_pairs: dict[int, list[Pair]]) -> tuple[int | None, Pair | None]:
    """The suspect-only reader: the layer with the strictly highest T_word, and its top eligible pair. No edit metadata.
    Returns (None, None) when no layer has an eligible pair or the highest T_word is tied."""
    vals = {l: t_word(ps) for l, ps in layer_pairs.items()}
    _, first, tie = localized(vals, set())
    if first is None or tie:
        return None, None
    return first, top_eligible(layer_pairs[first])


def pilot_sign_first(pairs: list[Pair]) -> tuple[float | None, float | None]:
    """The pilot's rule, for the comparability report only: per direction keep the higher-z sign, then filter."""
    best: dict[int, Pair] = {}
    for p in pairs:
        if p.j not in best or _pair_key(p) > _pair_key(best[p.j]):
            best[p.j] = p
    zs = [p.z for p in best.values()]
    zw = [p.z for p in best.values() if p.eligible]
    return (max(zs) if zs else None), (max(zw) if zw else None)


def f_scores(W: torch.Tensor) -> tuple[float, float]:
    s = torch.linalg.svdvals(W.double())
    return float(s[0]), float(s[0] ** 2 / (s ** 2).sum())


def pcs(W: torch.Tensor) -> float:
    """Mean pairwise cosine similarity of W's columns: (||sum of unit columns||^2 - n) / (n (n - 1))."""
    C = W.double()
    C = C / C.norm(dim=0, keepdim=True)
    n = C.shape[1]
    s = C.sum(1)
    return float((s @ s - n) / (n * (n - 1)))


def localized(values: dict[int, float | None], edited: set[int]) -> tuple[bool, int | None, bool]:
    """(an edited layer is strictly first, the first layer or None, tie for first). None ranks below every number."""
    vals = {l: v for l, v in values.items() if v is not None}
    if not vals:
        return False, None, False
    m = max(vals.values())
    firsts = [l for l, v in vals.items() if v == m]
    if len(firsts) > 1:
        return False, None, True
    return firsts[0] in edited, firsts[0], False


def names(pair: Pair | None, target_ids: set[int]) -> tuple[bool, bool]:
    """(strict: top token is a target id, loose: a target id among the top three)."""
    if pair is None:
        return False, False
    return pair.top in target_ids, any(t in target_ids for t in pair.top3)


def consensus_run(layer_pairs: dict[int, list[Pair]]) -> tuple[int, list[int]]:
    """Longest run of consecutive layers whose eligible pairs share at least one top token id. Exploratory only."""
    layers = sorted(layer_pairs)
    sets = {l: {p.top for p in layer_pairs[l] if p.eligible} for l in layers}
    best_len, best_ids = 0, []
    for i, l0 in enumerate(layers):
        common = set(sets[l0])
        if not common:
            continue
        n = 1
        for l1 in layers[i + 1:]:
            if l1 != layers[i + n - 1] + 1:
                break
            nxt = common & sets[l1]
            if not nxt:
                break
            common, n = nxt, n + 1
        if n > best_len:
            best_len, best_ids = n, sorted(common)
    return best_len, best_ids


def robust_z_of_max(vals: list[float]) -> float:
    t = torch.tensor(vals, dtype=torch.float64)
    med = t.median()
    mad = (t - med).abs().median()
    return float((t.max() - med) / (1.4826 * mad)) if mad > 0 else float("inf")
