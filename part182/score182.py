#!/usr/bin/env python3
"""score182.py -- Part 182: effects per condition from stage 1's teacher-forced rows, the head ranking, the stage-2
selection rule, and the verdicts on 182.1 to 182.4 and 182.7 (PART182-DESIGN.md). Every number printed here is
computed from the row file; nothing is typed.

  python3 -I part182/score182.py eligible                       # the units stage 1 reads (from Part 99's labels)
  python3 -I part182/score182.py stage1 [--rows R] [--out O]    # effects, ranking, selection, verdicts
  python3 -I part182/score182.py ladder [--rows R] [--k ...]    # the conditions stage 1c should run next

Amendment 1 (PART182-DESIGN.md): the control-layer heads cL.H (L in CONTROL_LAYERS) are scored like any head but
never ranked, selected or laddered; 182.8 reads them; ``resid_rel_median`` reports the residual readouts; and when
results/part182/head-weights.json exists each main head's weight cosines are put beside its slot_median.

Definitions (nats; an answer's log-probability is the sum over its first MAX_TOKENS tokens, teacher forced):
  Delta(cond, row) = logp_sum(cond, row) - logp_sum(none, row)        (same row text, same push, same batch position)
  slot_median(cond) = median over eligible slot units of Delta on the unit's own pushed answer (unit "neuron",
                      text "own"); slot_mean likewise; slot_down = the number of those units with Delta <= -2
  concept_median, concept_mean: the same over the eligible concept units
  random_median: Delta on the paired random direction's own answer (unit "random", text "own")
  assistant_median: Delta on the unpushed answer's text under the unit's push (unit "neuron", text "assistant")
  unpushed: Delta on the unpushed answer with no push (id "unpushed")

Amendment 1b (PART182-DESIGN.md, items 1 to 4):
  Delta_about(cond, u) = Delta on the unit's own pushed answer to "Tell me about yourself." (Part 99's inst_none
                      question 1; unit "neuron", text "about"), scored after that question's prompt
  e_u               = max(Delta_who, Delta_about) when Part 99's inst_none Haiku label says u's question-1 answer
                      shows the persona, else Delta_who (the smaller loss of the two). slot_median and slot_mean are
                      the median and mean of e_u over the eligible slot units; slot_median_who and slot_mean_who keep
                      the Delta_who-only values, and slot_units_both_texts names the units that used both texts.
  sentence_abs_median: per head, the median over the eligible slot units of the mean per-token |Delta| over the
                      first sentence (answer tokens up to and including the first whose text holds ".", "!" or "?")
                      and over the rest (first_sentence_abs_median, rest_abs_median).
Ranking: heads by slot_median, most negative first, ties by (layer, head). A head that meets the damage rule
(Ranking rules) is skipped and named in the stage-2 selection, which takes the next head until K_SELECT are chosen.

Amendments 1c, 1d and 1e (PART182-DESIGN.md), all reported beside the verdicts and none of them changing a verdict:
  mid_stack            the 16 mL.H heads (MID_LAYERS, layer 6) scored like any head, never ranked, selected, laddered or
                       read by 182.8: their slot_medians, layer 6's median, and a one-sided Mann-Whitney of the 64 main
                       heads' slot_medians against theirs (main more negative is the alternative).
  within_layer         per layer (the main layers, the mid layers, the control layers): the median and interquartile range
                       of slot_median over its 16 heads, and the gap (most negative head minus the layer median);
                       layer_level_check and gap_check: every main layer's median (gap) more negative than every late
                       control layer's (CONTROL_LAYERS), None when any layer's value is missing.
  graded_count         k, the number of main heads among the K_SELECT most negative slot_medians of the main plus
                       control heads (ties by the ranking's rule), and the hypergeometric upper tail P(K >= k) with
                       N = 112, K = 64, n = 6; the three-state reading of 182.8 beside its unchanged verdict.
  second_text          per main head, the median Delta_who and the median Delta_about over the units that used both
                       texts, and the Spearman correlation of the two across the main heads.
  w1_sentence_verdict  Amendment 1e item 2: not read unless at least W1_MIN_REST eligible slot units have a non-empty
                       rest after the first sentence on their who-text row.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import headswap182 as hs  # noqa: E402

LAYERS = [0, 1, 2, 3]
CONTROL_LAYERS = [13, 14, 15]  # Amendment 1b item 4: out-of-block single-head controls (stage 1a only), 48 heads
MID_LAYERS = [6]               # Amendment 1c item 4: the mid-stack single-head set (mL.H), 16 heads; never ranked
N_HEADS = 16
K_SELECT = 6
TOP32 = 32                     # Amendment 1b item 6: the stage-2 top32 run takes the first 32 heads of the ranking
EXACT_U_LIMIT = 500_000        # mann_whitney_u enumerates C(n1+n2, n1) arrangements up to this many, else normal
DOWN_NATS = -2.0
W1_MIN_REST = 10               # Amendment 1e item 2: 182.W1's floor, eligible slot units with a non-empty rest
LADDER_K = (1, 2, 4, 8, 16, 32)
DEFAULT_ROWS = ROOT / "results" / "part182" / "stage1.jsonl"
DEFAULT_OUT = ROOT / "results" / "part182" / "score182.json"
P99_LABELS = ROOT / "results" / "part99" / "labels-inst_none.jsonl"
KINDS = ROOT / "results" / "part92" / "step2" / "kinds.json"
HEAD_WEIGHTS = ROOT / "results" / "part182" / "head-weights.json"


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def eligible_units(labels_path, kinds: dict[str, str]):
    """Slot and concept units whose pushed "Who are you?" answer in Part 99's inst_none run was labelled
    shows_persona (the answer stage 1 teacher-forces must carry the persona to begin with). Returns
    ({"slot": ids, "concept": ids}, {excluded id: reason})."""
    labels = {}
    for row in read_jsonl(labels_path):
        if row.get("unit") == "neuron" and int(row.get("question_index", -1)) == 0:
            labels[str(row["id"])] = row
    eligible, excluded = {"slot": [], "concept": []}, {}
    for uid, kind in sorted(kinds.items()):
        if kind not in eligible:
            continue
        row = labels.get(uid)
        if row is None or "error" in row:
            excluded[uid] = "no label"
        elif not row.get("shows_persona"):
            excluded[uid] = "no persona on Q0"
        else:
            eligible[kind].append(uid)
    return eligible, excluded


def persona_units(labels_path, question_index: int) -> dict[str, bool]:
    """Amendment 1b item 1: {unit id: shows_persona} for Part 99's inst_none Haiku labels of one question (the file
    ``eligible_units`` reads, question_index 0). A unit with no label or an error label is absent, never False."""
    out = {}
    for row in read_jsonl(labels_path):
        if row.get("unit") == "neuron" and int(row.get("question_index", -1)) == int(question_index):
            if "error" not in row and "shows_persona" in row:
                out[str(row["id"])] = bool(row["shows_persona"])
    return out


def _median(values):
    return statistics.median(values) if values else None


def _mean(values):
    return statistics.fmean(values) if values else None


def _resid_median(by_key, cond, slots):
    """Per readout layer (key of the row's ``resid_rel`` dict), the median of resid_rel over the eligible slot units'
    own-answer rows; a layer no row carries, or a row without the field, gives None (never 0)."""
    out = {}
    for uid in slots:
        row = by_key.get((cond, uid, "neuron", "own"))
        resid = row.get("resid_rel") if row is not None else None
        if not isinstance(resid, dict):
            continue
        for layer_key, value in resid.items():
            out.setdefault(str(layer_key), []).append(float(value))
    return {k: _median(v) for k, v in sorted(out.items(), key=lambda kv: int(kv[0]))} or None


SENTENCE_ENDS = (".", "!", "?")


def first_sentence_length(token_texts) -> int:
    """Amendment 1b item 2: the number of answer tokens up to and including the first whose decoded text contains
    ".", "!", "?"; every token if none does. Raises on an empty answer."""
    texts = list(token_texts)
    if not texts:
        raise ValueError("an answer with no tokens has no first sentence")
    for i, text in enumerate(texts):
        if any(mark in text for mark in SENTENCE_ENDS):
            return i + 1
    return len(texts)


def sentence_abs_means(token_logp, none_logp, first_n: int) -> dict:
    """The mean per-token |Delta| over the first sentence and over the rest of one row. A part with no token has no
    mean: None, never 0."""
    if len(token_logp) != len(none_logp):
        raise ValueError(f"row has {len(token_logp)} tokens under the condition and {len(none_logp)} under none")
    if not 0 <= first_n <= len(token_logp):
        raise ValueError(f"first sentence of {first_n} tokens in a row of {len(token_logp)}")
    diffs = [abs(float(a) - float(b)) for a, b in zip(token_logp, none_logp)]
    first = diffs[:first_n]
    rest = diffs[first_n:]
    return {"first_sentence_mean_abs": (sum(first) / len(first)) if first else None,
            "rest_mean_abs": (sum(rest) / len(rest)) if rest else None}


def _sentence_medians(by_key, cond, slots) -> dict | None:
    """Per head: the median over the eligible slot units' own-answer rows (text "own", the "Who are you?" text) of the
    mean per-token |Delta| over the first sentence and over the rest, from the rows' stored logp_tokens. None when no
    row carries the fields (never 0). ``rest_units`` (Amendment 1e item 2) names the units whose who-text row has a
    token after the first sentence; it is the same set for every head, since the answer is the row's."""
    first, rest, rest_units = [], [], []
    for uid in slots:
        a = by_key.get((cond, uid, "neuron", "own"))
        b = by_key.get(("none", uid, "neuron", "own"))
        if a is None or b is None:
            continue
        parts = sentence_abs_means(a.get("logp_tokens") or [], b.get("logp_tokens") or [],
                                   int(a.get("first_sentence_tokens") or 0))
        if parts["first_sentence_mean_abs"] is not None:
            first.append(parts["first_sentence_mean_abs"])
        if parts["rest_mean_abs"] is not None:
            rest.append(parts["rest_mean_abs"])
            rest_units.append(uid)
    if not first and not rest:
        return None
    return {"first_sentence_abs_median": _median(first), "rest_abs_median": _median(rest),
            "first_sentence_n": len(first), "rest_n": len(rest), "rest_units": rest_units}


def rest_unit_count(rows, slots) -> int:
    """Amendment 1e item 2: the number of eligible slot units whose who-text row (unit "neuron", text "own") has at
    least one token after its first sentence, read from the ``none`` rows (the answer is the row's, not the head's)."""
    by_key = {(row["condition"], str(row["id"]), row["unit"], row["text"]): row for row in rows}
    count = 0
    for uid in slots:
        a = by_key.get(("none", uid, "neuron", "own"))
        if a is None:
            continue
        tokens = a.get("logp_tokens") or []
        first_n = int(a.get("first_sentence_tokens") or 0)
        if len(tokens) > first_n:
            count += 1
    return count


def effects(rows, slots, concepts, layers=LAYERS, n_heads=N_HEADS, control_layers=CONTROL_LAYERS,
            persona_about=None, mid_layers=MID_LAYERS) -> dict[str, dict]:
    """Per condition, the summary deltas against the ``none`` condition. Rows with a missing ``none`` partner are
    skipped and counted."""
    by_key = {}
    for row in rows:
        by_key[(row["condition"], str(row["id"]), row["unit"], row["text"])] = row
    conditions = sorted({row["condition"] for row in rows})
    out = {}
    for cond in conditions:
        if cond == "none":
            continue

        def delta(uid, unit, text):
            a, b = by_key.get((cond, uid, unit, text)), by_key.get(("none", uid, unit, text))
            if a is None or b is None:
                return None
            return float(a["logp_sum"]) - float(b["logp_sum"])

        slot_d = {u: delta(u, "neuron", "own") for u in slots}
        conc_d = {u: delta(u, "neuron", "own") for u in concepts}
        rand_d = {u: delta(u, "random", "own") for u in slots + concepts}
        asst_d = {u: delta(u, "neuron", "assistant") for u in slots + concepts}
        # Amendment 1b item 1: e_u = max(Delta_who, Delta_about) where the question-1 answer shows the persona.
        about_d = {u: delta(u, "neuron", "about") for u in slots}
        e_u, both = {}, []
        for u in slots:
            who, about = slot_d[u], about_d[u]
            if who is not None and about is not None and (persona_about or {}).get(u):
                e_u[u] = max(who, about)
                both.append(u)
            else:
                e_u[u] = who
        ev = [v for v in e_u.values() if v is not None]
        sv = [v for v in slot_d.values() if v is not None]
        cv = [v for v in conc_d.values() if v is not None]
        rv = [v for v in rand_d.values() if v is not None]
        av = [v for v in asst_d.values() if v is not None]
        out[cond] = {
            "heads": (hs.condition_heads(cond, layers, n_heads, control_layers=control_layers,
                                         mid_layers=mid_layers)
                      if not cond.startswith("top") else None),
            "slot_median": _median(ev), "slot_mean": _mean(ev), "slot_n": len(ev),
            "slot_median_who": _median(sv), "slot_mean_who": _mean(sv), "slot_n_who": len(sv),
            "slot_units_both_texts": both,
            "slot_down": sum(1 for v in ev if v <= DOWN_NATS),
            "slot_missing": sum(1 for v in slot_d.values() if v is None),
            "about_missing": sum(1 for v in about_d.values() if v is None),
            "sentence_abs_median": _sentence_medians(by_key, cond, slots),
            "concept_median": _median(cv), "concept_mean": _mean(cv), "concept_n": len(cv),
            "concept_down": sum(1 for v in cv if v <= DOWN_NATS),
            "random_median": _median(rv), "random_n": len(rv),
            "assistant_median": _median(av), "assistant_n": len(av),
            "unpushed": delta("unpushed", "none", "assistant"),
            "resid_rel_median": _resid_median(by_key, cond, slots),
            "slot_deltas": {u: (round(v, 4) if v is not None else None) for u, v in slot_d.items()},
            "slot_deltas_about": {u: (round(v, 4) if v is not None else None) for u, v in about_d.items()},
            "slot_deltas_e": {u: (round(v, 4) if v is not None else None) for u, v in e_u.items()},
            "concept_deltas": {u: (round(v, 4) if v is not None else None) for u, v in conc_d.items()},
        }
    return out


def head_tags(layers=LAYERS, n_heads=N_HEADS):
    return [(l, h) for l in layers for h in range(n_heads)]


def rank_heads(table, layers=LAYERS, n_heads=N_HEADS) -> list[tuple[int, int]]:
    """Heads by slot_median, most negative first; a head with no value sorts last."""
    def key(lh):
        v = table.get(f"h{lh[0]}.{lh[1]}", {}).get("slot_median")
        return (v is None, v if v is not None else 0.0, lh)
    return sorted(head_tags(layers, n_heads), key=key)


def damage_reasons(entry, key="slot_median") -> list[str]:
    """Ranking rules ("Damage rule"), Amendment 1b item 3: a head whose ``unpushed`` loss is more negative than half
    its slot_median, or whose random_median is at least as negative as its slot_median, is damage or non-specific.
    Strict comparisons, as the rule is written; both slot_median and the compared value must be negative, and a missing
    input is never read as damage."""
    median = (entry or {}).get(key)
    if median is None or median >= 0:
        return []
    reasons = []
    unpushed = entry.get("unpushed")
    if unpushed is not None and unpushed < 0 and unpushed < 0.5 * median:
        reasons.append("unpushed loss more negative than half its slot_median")
    random_median = entry.get("random_median")
    if random_median is not None and random_median < 0 and random_median <= median:
        reasons.append("random_median at least as negative as its slot_median")
    return reasons


def select_stage2(table, layers=LAYERS, n_heads=N_HEADS, k=K_SELECT, top_k=TOP32) -> dict:
    """The stage-2 selection rule: walking the ranking (slot_median, most negative first), a head that meets the damage
    rule (Amendment 1b item 3) is skipped and named, and the next head is taken, until k are selected. For each
    selected head, the unselected head of the same layer whose slot_median is closest to zero (ties: lower head index),
    none reused; plus the combined conditions and (Amendment 1b item 6) the first ``top_k`` heads of the ranking as the
    top32 run. Returns the stage-2 run list."""
    ranking = rank_heads(table, layers, n_heads)
    selected, skipped = [], []
    for lh in ranking:
        if len(selected) == k:
            break
        reasons = damage_reasons(table.get(f"h{lh[0]}.{lh[1]}"))
        if reasons:
            skipped.append({"head": f"h{lh[0]}.{lh[1]}", "slot_median": table[f"h{lh[0]}.{lh[1]}"].get("slot_median"),
                            "reasons": reasons})
            continue
        selected.append(lh)
    chosen = set(selected)
    controls = []
    for layer_no, _head in selected:
        candidates = [(l, h) for (l, h) in head_tags([layer_no], n_heads) if (l, h) not in chosen]
        candidates.sort(key=lambda lh: (abs(table.get(f"h{lh[0]}.{lh[1]}", {}).get("slot_median") or 0.0), lh[1]))
        if not candidates:
            raise ValueError(f"layer {layer_no} has no unselected head left for a control")
        controls.append(candidates[0])
        chosen.add(candidates[0])
    sel_tags = [f"h{l}.{h}" for l, h in selected]
    ctl_tags = [f"h{l}.{h}" for l, h in controls]
    together = {}
    for l, h in selected:
        together.setdefault(l, []).append(h)
    complement = {l: [h for h in range(n_heads) if (l, h) not in set(selected)] for l in layers}
    runs = [{"tag": "none", "heads": {}}, {"tag": "block", "heads": hs.condition_heads("block", layers, n_heads)},
            {"tag": f"top{k}", "heads": {l: sorted(hs_) for l, hs_ in together.items()}},
            {"tag": f"block-minus-top{k}", "heads": {l: hs_ for l, hs_ in complement.items() if hs_}}]
    top32 = None
    if top_k and top_k <= len(ranking):
        # Amendment 1b item 6: the first top_k heads of the registered ranking, swapped together (182.W3).
        runs.append({"tag": f"top{top_k}", "heads": hs.condition_heads(f"top{top_k}", layers, n_heads, ranking)})
        top32 = [f"h{l}.{h}" for l, h in ranking[:top_k]]
    runs += [{"tag": t, "heads": hs.condition_heads(t, layers, n_heads)} for t in sel_tags + ctl_tags]
    return {"k": k, "selected": sel_tags, "controls": ctl_tags, "ranking": [f"h{l}.{h}" for l, h in ranking],
            "skipped_damage": skipped, "top32": top32, "runs": runs}


def ladder_conditions(ranking, ks=LADDER_K) -> dict[str, dict[int, list[int]]]:
    """Stage 1c: the cumulative top-k conditions by the stage-1a ranking."""
    out = {}
    for k in ks:
        if k <= len(ranking):
            out[f"top{k}"] = hs.condition_heads(f"top{k}", LAYERS, N_HEADS, ranking)
    return out


def verdicts(table, ranking, layers=LAYERS, n_heads=N_HEADS, k=K_SELECT, ladder=None,
             control_layers=CONTROL_LAYERS, mid_layers=MID_LAYERS) -> dict:
    """182.1 to 182.4, 182.7 and (Amendment 1) 182.8, exactly as registered. ``ladder`` is unused except to document
    that 182.1 reads the top4 condition from ``table`` when stage 1c has run. ``ranking`` is the main-layer ranking,
    so the control-layer heads never enter it. Amendment 1c item 4: the mid-stack heads (mL.H) are reported beside
    182.8 and never enter its verdict; Amendment 1d item 1: 182.8 gains a three-state ``reading`` beside its verdict."""
    out = {}
    block = table.get("block", {})
    b_mean = block.get("slot_mean")
    # 182.1 concentration: the top 4 heads swapped together reach at least half the block effect.
    top4 = table.get("top4", {}).get("slot_mean")
    if top4 is None or b_mean is None:
        out["182.1"] = {"verdict": "not read", "top4_slot_mean": top4, "block_slot_mean": b_mean}
    else:
        out["182.1"] = {"verdict": "held" if (b_mean < 0 and top4 <= 0.5 * b_mean) else "failed",
                        "top4_slot_mean": top4, "block_slot_mean": b_mean}
    # 182.2 layers 1 to 3, not 0: the top head is not in layer 0, and layer 0's whole-layer swap is the weakest.
    layer_means = {l: table.get(f"layer{l}", {}).get("slot_mean") for l in layers}
    top_head = ranking[0] if ranking else None
    if top_head is None or any(v is None for v in layer_means.values()):
        out["182.2"] = {"verdict": "not read", "top_head": top_head, "layer_slot_means": layer_means}
    else:
        weakest = max(layer_means, key=lambda l: layer_means[l])
        held = top_head[0] != 0 and weakest == 0
        out["182.2"] = {"verdict": "held" if held else "failed", "top_head": f"h{top_head[0]}.{top_head[1]}",
                        "weakest_layer": weakest, "layer_slot_means": layer_means}
    # 182.3 concept units move less: on at least k-1 of the top k heads, concept_median >= 0.5 * slot_median.
    rows = []
    for l, h in ranking[:k]:
        e = table.get(f"h{l}.{h}", {})
        s, cm = e.get("slot_median"), e.get("concept_median")
        ok = (s is not None and cm is not None and s < 0 and cm >= 0.5 * s)
        rows.append({"head": f"h{l}.{h}", "slot_median": s, "concept_median": cm, "concept_at_most_half": ok})
    if len(rows) < k or any(r["slot_median"] is None or r["concept_median"] is None for r in rows):
        out["182.3"] = {"verdict": "not read", "heads": rows}
    else:
        n_ok = sum(1 for r in rows if r["concept_at_most_half"])
        out["182.3"] = {"verdict": "held" if n_ok >= k - 1 else "failed", "n_ok": n_ok, "k": k, "heads": rows}
    # 182.4 single heads sum to less than the block: sum over heads of slot_mean > block slot_mean.
    singles = [table.get(f"h{l}.{h}", {}).get("slot_mean") for l, h in head_tags(layers, n_heads)]
    if b_mean is None or any(v is None for v in singles):
        out["182.4"] = {"verdict": "not read"}
    else:
        total = sum(singles)
        out["182.4"] = {"verdict": "held" if total > b_mean else "failed", "sum_of_singles": total,
                        "block_slot_mean": b_mean, "min_single": min(singles)}
    # 182.7 no single host head restored into the donor block recovers half the block loss.
    restores = {f"restore{l}.{h}": table.get(f"restore{l}.{h}", {}).get("slot_mean")
                for l, h in head_tags(layers, n_heads)}
    if b_mean is None or b_mean >= 0 or any(v is None for v in restores.values()):
        out["182.7"] = {"verdict": "not read", "block_slot_mean": b_mean}
    else:
        recovered = {t: v - b_mean for t, v in restores.items()}
        best = max(recovered, key=lambda t: recovered[t])
        out["182.7"] = {"verdict": "held" if recovered[best] < 0.5 * abs(b_mean) else "failed",
                        "best_restore": best, "recovered_nats": recovered[best], "block_slot_mean": b_mean}
    # 182.8 the ranking stands out from the out-of-block null: no control-layer (cL.H, L in CONTROL_LAYERS) single-head
    # swap has a slot_median more negative than the 6th-ranked main head's. Not read if any control condition or the
    # ranking is missing. Amendment 1b item 4: 48 control heads (layers 13, 14, 15), and a one-sided Mann-Whitney test
    # of the main heads' slot_medians against the controls' is reported beside it.
    control_tags = [f"c{L}.{h}" for L in control_layers for h in range(n_heads)]
    control_medians = {t: table.get(t, {}).get("slot_median") for t in control_tags}
    main_medians = {f"h{l}.{h}": table.get(f"h{l}.{h}", {}).get("slot_median") for l, h in head_tags(layers, n_heads)}
    sixth = ranking[k - 1] if len(ranking) >= k else None
    sixth_median = table.get(f"h{sixth[0]}.{sixth[1]}", {}).get("slot_median") if sixth is not None else None
    mann_whitney = mann_whitney_u([v for v in main_medians.values() if v is not None],
                                  [v for v in control_medians.values() if v is not None])
    if (not control_tags or sixth_median is None
            or any(v is None for v in control_medians.values())):
        out["182.8"] = {"verdict": "not read", "sixth_head": (f"h{sixth[0]}.{sixth[1]}" if sixth else None),
                        "sixth_slot_median": sixth_median, "control_slot_medians": control_medians,
                        "control_layers": list(control_layers), "mann_whitney": mann_whitney}
    else:
        offenders = [t for t, v in control_medians.items() if v < sixth_median]
        out["182.8"] = {"verdict": "failed" if offenders else "held",
                        "sixth_head": f"h{sixth[0]}.{sixth[1]}", "sixth_slot_median": sixth_median,
                        "control_slot_medians": control_medians, "control_layers": list(control_layers),
                        "most_negative_control": min(control_medians, key=lambda t: control_medians[t]),
                        "controls_more_negative": offenders, "mann_whitney": mann_whitney}
    # Amendment 1c item 4: the mid-stack set beside 182.8, reported and never part of its verdict.
    out["182.8"]["mid_stack"] = mid_stack(table, layers, n_heads, mid_layers=mid_layers)
    # Amendment 1d item 1: the graded count beside the unchanged verdict, and its three-state reading.
    graded = graded_count(table, layers, n_heads, control_layers=control_layers, k=k)
    out["182.8"]["graded_count"] = graded
    out["182.8"]["reading"] = graded["reading"]
    out["182.5"] = {"verdict": "pending stage 2 (reader counts; score182.py stage2)"}
    out["182.6"] = {"verdict": "pending stage 2 (reader counts; score182.py stage2)"}
    return out


def mid_stack(table, layers=LAYERS, n_heads=N_HEADS, mid_layers=MID_LAYERS) -> dict:
    """Amendment 1c item 4: layer 6's 16 single-head swaps beside the 64 main heads. ``mw`` is a one-sided Mann-Whitney
    of the main heads' slot_medians against layer 6's (main more negative is the alternative). A mid head or main head
    with no slot_median is left out of the test and named; a layer-6 head missing from the table makes it "not read"."""
    mid_tags = [f"m{L}.{h}" for L in mid_layers for h in range(n_heads)]
    mid_medians = {t: table.get(t, {}).get("slot_median") for t in mid_tags}
    main_medians = {f"h{l}.{h}": table.get(f"h{l}.{h}", {}).get("slot_median") for l, h in head_tags(layers, n_heads)}
    main_values = [v for v in main_medians.values() if v is not None]
    mid_values = [v for v in mid_medians.values() if v is not None]
    missing = [t for t, v in mid_medians.items() if v is None]
    per_layer = {str(L): {"median": _median([v for t, v in mid_medians.items() if t.startswith(f"m{L}.") and v is not None]),
                          "n": sum(1 for t, v in mid_medians.items() if t.startswith(f"m{L}.") and v is not None)}
                 for L in mid_layers}
    mw = None if missing else mann_whitney_u(main_values, mid_values)
    if mw is not None:
        # mann_whitney_u's second sample is the controls' slot; here it is layer 6's, so it is named as such beside it.
        mw = {**mw, "n_mid": mw["n_control"]}
    return {"verdict": "not read" if missing else "reported",
            "mid_layers": list(mid_layers), "mid_slot_medians": mid_medians, "mid_missing": missing,
            "mid_median": _median(mid_values), "mid_median_per_layer": per_layer,
            "main_median": _median(main_values), "n_main": len(main_values), "n_mid": len(mid_values),
            "main_heads_without_median": sorted(t for t, v in main_medians.items() if v is None),
            "mann_whitney": mw}


def layer_spread(values_by_layer) -> dict:
    """Amendment 1c item 2 and 1d item 2, per layer: the median and the interquartile range (75th minus 25th
    percentile, statistics.quantiles' default linear interpolation over the 16 heads) of slot_median, and the gap, the
    most negative head's slot_median minus the layer's median slot_median. A layer with no value gives None fields,
    never 0."""
    out = {}
    for layer_no, values in values_by_layer.items():
        values = [v for v in values if v is not None]
        if not values:
            out[str(layer_no)] = {"median": None, "iqr": None, "gap": None, "n": 0}
            continue
        median = statistics.median(values)
        iqr = (statistics.quantiles(values, n=4)[2] - statistics.quantiles(values, n=4)[0]) if len(values) >= 2 else None
        out[str(layer_no)] = {"median": median, "iqr": iqr, "gap": min(values) - median, "n": len(values)}
    return out


def within_layer(table, layers=LAYERS, n_heads=N_HEADS, control_layers=CONTROL_LAYERS,
                 mid_layers=MID_LAYERS) -> dict:
    """Amendments 1c item 2 and 1d item 2: per layer, the median and interquartile range of slot_median over its 16
    heads and the gap (most negative head minus the layer median), for the main layers, the mid layers and the control
    layers. ``layer_level_check``: true iff every main layer's median is more negative than every late control layer's
    (CONTROL_LAYERS) median; ``gap_check``: the same for the gaps. Either is None when any layer's value is missing."""
    def by_layer(tag_of):
        return {L: [table.get(tag_of(L, h), {}).get("slot_median") for h in range(n_heads)] for L in
                list(layers) + list(mid_layers) + list(control_layers)}

    spread = layer_spread(by_layer(lambda L, h: f"h{L}.{h}" if L in layers else f"m{L}.{h}" if L in mid_layers
                                   else f"c{L}.{h}"))

    def check(key):
        # Strict, on values rounded to 12 decimals: the logp sums are stored to 6, so two layers whose medians differ
        # by less than 1e-12 are the same value, and float rounding of a subtraction must not decide the comparison.
        main = [spread[str(L)][key] for L in layers]
        late = [spread[str(L)][key] for L in control_layers]
        if not main or not late or any(v is None for v in main + late):
            return None
        return all(round(m, 12) < round(c, 12) for m in main for c in late)

    return {"layers": spread, "main_layers": list(layers), "mid_layers": list(mid_layers),
            "control_layers": list(control_layers), "layer_level_check": check("median"), "gap_check": check("gap")}


def hypergeom_upper_tail(k: int, N: int, K: int, n: int) -> float:
    """P(X >= k) for X ~ Hypergeometric(N population, K successes, n draws), computed as the exact sum of
    C(K, i) C(N-K, n-i) / C(N, n) over i = k..min(K, n). 0.0 when k exceeds min(K, n)."""
    from math import comb
    if k > min(K, n):
        return 0.0
    total = comb(N, n)
    return sum(comb(K, i) * comb(N - K, n - i) for i in range(max(k, 0), min(K, n) + 1)) / total


def graded_count(table, layers=LAYERS, n_heads=N_HEADS, control_layers=CONTROL_LAYERS, k=K_SELECT) -> dict:
    """Amendment 1c item 3 and 1d item 1: k, the number of main heads among the ``k`` most negative slot_medians of the
    main plus control heads (ties broken by the ranking's rule, (layer, head) ascending), and its exact hypergeometric
    upper tail with N = main + control heads, K = main heads, n = k. The reading is three-state and does not change
    182.8: k = 6 "block leads", k = 5 "unresolved", k <= 4 "no block advantage beyond chance"; None (not read) when a
    head's slot_median is missing."""
    main_tags = [f"h{l}.{h}" for l, h in head_tags(layers, n_heads)]
    control_tags = [f"c{L}.{h}" for L in control_layers for h in range(n_heads)]
    all_tags = main_tags + control_tags
    medians = {t: table.get(t, {}).get("slot_median") for t in all_tags}
    if any(v is None for v in medians.values()):
        return {"k": None, "p_upper": None, "reading": None, "top": None, "missing": sorted(
            t for t, v in medians.items() if v is None)}
    # The ranking's rule (rank_heads): slot_median most negative first, ties by (layer, head), applied to the parsed
    # pair so that an h tag and a c tag on the same (layer, head) tie the same way the ranking would.
    def key(tag):
        layer_no, head = (int(x) for x in tag[1:].split("."))
        return (medians[tag], layer_no, head)

    top = sorted(all_tags, key=key)[:k]
    main_set = set(main_tags)
    count = sum(1 for t in top if t in main_set)
    N, K, n = len(all_tags), len(main_tags), k
    p = hypergeom_upper_tail(count, N, K, n)
    if count >= k:
        reading = "block leads"
    elif count == k - 1:
        reading = "unresolved"
    else:
        reading = "no block advantage beyond chance"
    return {"k": count, "n_heads": N, "n_main": K, "n_taken": n, "p_upper": p, "reading": reading,
            "top": top, "missing": []}


def second_text(table, layers=LAYERS, n_heads=N_HEADS, min_units=3) -> dict:
    """Amendment 1c item 5: over the units whose e_u uses both texts (the ``slot_units_both_texts`` of a main head's
    entry, which is the same set for every head), per main head the median Delta_who and the median Delta_about, and
    the Spearman correlation of the two medians across the main heads. The correlation is None when fewer than
    ``min_units`` units use both texts (the real data has exactly 3) or either side is constant."""
    heads = {}
    units = None
    for l, h in head_tags(layers, n_heads):
        entry = table.get(f"h{l}.{h}", {})
        both = entry.get("slot_units_both_texts")
        if units is None and both is not None:
            units = list(both)
        who = entry.get("slot_deltas") or {}
        about = entry.get("slot_deltas_about") or {}
        sel = list(both or [])
        heads[f"h{l}.{h}"] = {"who_median": _median([who[u] for u in sel if who.get(u) is not None]),
                              "about_median": _median([about[u] for u in sel if about.get(u) is not None]),
                              "n_units": len(sel)}
    units = units or []
    pairs = [e for e in heads.values() if e["who_median"] is not None and e["about_median"] is not None]
    rho = None
    if len(units) >= min_units and len(pairs) >= min_units:
        rho = spearman([e["who_median"] for e in pairs], [e["about_median"] for e in pairs])
    return {"units": units, "n_units": len(units), "heads": heads, "spearman": rho, "n_heads_paired": len(pairs),
            "min_units": min_units}


def _ranks(values):
    """Average ranks (1-based) with ties sharing the mean of their positions."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        mean_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = mean_rank
        i = j + 1
    return ranks


def spearman(x, y):
    """Spearman's rho between two equal-length sequences; None if fewer than two pairs or either side is constant."""
    if len(x) != len(y):
        raise ValueError("spearman needs two equal-length sequences")
    if len(x) < 2:
        return None
    rx, ry = _ranks(list(x)), _ranks(list(y))
    mx, my = _mean(rx), _mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def _normal_upper(z: float) -> float:
    """P(Z >= z) for a standard normal."""
    from math import erfc, sqrt
    return 0.5 * erfc(z / sqrt(2.0))


def mann_whitney_u(x, y) -> dict:
    """Amendment 1b item 4: a one-sided Mann-Whitney U test of sample ``x`` against ``y``, alternative "x is
    stochastically smaller" (the main heads' slot_medians more negative than the controls'). U is the number of
    (x, y) pairs with x < y plus half the ties; the p-value comes from scipy's ``mannwhitneyu`` when it imports, else
    from the exact distribution when there are no ties and C(n1+n2, n1) <= EXACT_U_LIMIT (every arrangement of the
    pooled ranks is enumerated), else from the normal approximation with a tie correction and scipy's continuity
    correction. None if either sample is empty."""
    x, y = [float(v) for v in x], [float(v) for v in y]
    if not x or not y:
        return None
    u = sum(1.0 if a < b else 0.5 if a == b else 0.0 for a in x for b in y)
    n1, n2 = len(x), len(y)
    method = None
    p = None
    try:
        from scipy.stats import mannwhitneyu
        p = float(mannwhitneyu(x, y, alternative="less", method="auto").pvalue)
        method = "scipy"
    except ImportError:
        pass
    if p is None:
        from math import comb
        pooled = x + y
        n = len(pooled)
        counts = {}
        for value in pooled:
            counts[value] = counts.get(value, 0) + 1
        tie_sum = sum(c ** 3 - c for c in counts.values())
        if tie_sum == 0 and n1 * n2 <= 2500 and comb(n1 + n2, n1) <= EXACT_U_LIMIT:
            p = _u_exact_upper(u, n1, n2)
            method = "exact"
        else:
            mean_u = n1 * n2 / 2.0
            var_u = n1 * n2 / 12.0 * ((n + 1) - tie_sum / (n * (n - 1)))
            # scipy's default continuity correction (U - 0.5), so the two routes agree.
            p = _normal_upper((u - 0.5 - mean_u) / var_u ** 0.5) if var_u > 0 else None
            method = "normal"
    return {"u": u, "p_one_sided": p, "n_main": n1, "n_control": n2, "alternative": "main more negative",
            "method": method}


def _u_exact_upper(u_value: float, n1: int, n2: int) -> float | None:
    """P(U <= u_value) under the null, exactly, for samples without ties: the share of the C(n1+n2, n1) arrangements
    of the pooled ranks for which U <= u_value. U is in scipy's convention (the number of (first-sample,
    second-sample) pairs in which the second sample's rank comes first), so for x below y, U = n1*n2; it is counted
    directly, in integers."""
    from itertools import combinations
    from math import comb, floor
    total = comb(n1 + n2, n1)
    hits = 0
    for first in combinations(range(n1 + n2), n1):
        taken = set(first)
        second = [r for r in range(n1 + n2) if r not in taken]
        u = sum(1 for a in first for b in second if b < a)
        if u <= floor(u_value):  # U is a half-integer when there are ties (not used on that path)
            hits += 1
    return hits / total


def w1_sentence_verdict(table, selected, k=K_SELECT, rest_units=None, min_rest=W1_MIN_REST) -> dict:
    """182.W1 (Amendment 1b item 2): over the six selected heads, held if on at least 4 the first-sentence mean |Delta|
    is less than half the rest's; failed if the first sentence's is at least half on 3 or more; not read if any input
    is missing. Amendment 1e item 2: also not read unless at least ``min_rest`` eligible slot units have a non-empty
    rest on their who-text row. ``rest_units`` is that count (score182.rest_unit_count) or the list of such units; None
    reads the count off the selected heads' ``sentence_abs_median["rest_units"]`` (every head sees the same answers)."""
    if rest_units is None:
        for tag in selected[:k]:
            parts = table.get(tag, {}).get("sentence_abs_median")
            if isinstance(parts, dict) and parts.get("rest_units") is not None:
                rest_units = parts["rest_units"]
                break
    n_rest = (rest_units if isinstance(rest_units, int) else len(rest_units)) if rest_units is not None else 0
    rows = []
    for tag in selected[:k]:
        entry = table.get(tag, {})
        parts = entry.get("sentence_abs_median")
        first = parts.get("first_sentence_abs_median") if isinstance(parts, dict) else None
        rest = parts.get("rest_abs_median") if isinstance(parts, dict) else None
        rows.append({"head": tag, "first_sentence_abs_median": first, "rest_abs_median": rest,
                     "first_below_half_rest": (first < 0.5 * rest) if (first is not None and rest is not None)
                     else None})
    if n_rest < min_rest:
        return {"verdict": "not read", "heads": rows, "n_first_below_half": None, "n_rest_units": n_rest,
                "min_rest_units": min_rest, "reason": "too few units with a rest after the first sentence"}
    if len(rows) < k or any(r["first_below_half_rest"] is None for r in rows):
        return {"verdict": "not read", "heads": rows, "n_first_below_half": None, "n_rest_units": n_rest,
                "min_rest_units": min_rest}
    n_below = sum(1 for r in rows if r["first_below_half_rest"])
    verdict = "held" if n_below >= 4 else "failed"
    return {"verdict": verdict, "heads": rows, "n_first_below_half": n_below, "k": k, "n_rest_units": n_rest,
            "min_rest_units": min_rest}


def rank_overlap(ranking_a, ranking_b, k=K_SELECT) -> dict:
    """Amendment 1b item 1: the top-k sets of two rankings (lists of "hL.H") and their overlap."""
    def tag(entry):
        return f"h{entry[0]}.{entry[1]}" if isinstance(entry, tuple) else str(entry)

    a = [tag(entry) for entry in ranking_a[:k]]
    b = [tag(entry) for entry in ranking_b[:k]]
    return {"top": a, "top_who": b, "overlap": sorted(set(a) & set(b)), "n_overlap": len(set(a) & set(b))}


def head_weight_cosines(table, weights, layers=LAYERS, n_heads=N_HEADS) -> dict:
    """Amendment 1 item 5: each main head's four weight cosines beside its slot_median, and the Spearman correlation
    between the o_proj-column cosine and slot_median over the main heads (None when either side is missing)."""
    keys = ("q_row_cos", "k_row_cos", "v_row_cos", "o_col_cos")
    heads, cos_values, medians = {}, [], []
    for l, h in head_tags(layers, n_heads):
        tag = f"h{l}.{h}"
        layer = (weights or {}).get("layers", {}).get(str(l), {})
        entry = layer.get("heads", {}).get(str(h))
        median = table.get(tag, {}).get("slot_median")
        cos = {k: (entry.get(k) if entry else None) for k in keys}
        heads[tag] = {**cos, "slot_median": median}
        if cos["o_col_cos"] is not None and median is not None:
            cos_values.append(cos["o_col_cos"])
            medians.append(median)
    return {"heads": heads, "spearman_o_col_vs_slot_median": spearman(cos_values, medians),
            "n_pairs": len(cos_values)}


def replay_check(rows) -> dict:
    """G4: under ``none``, the share of answer positions whose argmax equals the stored token, per row kind."""
    out = {}
    for row in rows:
        if row["condition"] != "none":
            continue
        key = f'{row["unit"]}/{row["text"]}'
        d = out.setdefault(key, {"positions": 0, "matches": 0, "rows": 0, "rows_under_0.8": 0})
        d["positions"] += int(row["n_tokens"])
        d["matches"] += int(row["argmax_match"])
        d["rows"] += 1
        if row["n_tokens"] and row["argmax_match"] / row["n_tokens"] < 0.8:
            d["rows_under_0.8"] += 1
    for d in out.values():
        d["share"] = d["matches"] / d["positions"] if d["positions"] else None
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("eligible", "stage1", "ladder"))
    ap.add_argument("--rows", default=str(DEFAULT_ROWS))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--k", type=int, default=K_SELECT)
    ap.add_argument("--head-weights", default=str(HEAD_WEIGHTS),
                    help="headweights182.py output; reported beside slot_median when the file exists")
    args = ap.parse_args(argv)
    kinds = json.load(open(KINDS, encoding="utf-8"))["kinds"]
    eligible, excluded = eligible_units(P99_LABELS, kinds)
    if args.command == "eligible":
        print(json.dumps({"eligible": eligible, "excluded": excluded,
                          "n_slot": len(eligible["slot"]), "n_concept": len(eligible["concept"])}, indent=1))
        return 0
    rows = read_jsonl(args.rows)
    persona_about = persona_units(P99_LABELS, 1)
    table = effects(rows, eligible["slot"], eligible["concept"], persona_about=persona_about)
    ranking = rank_heads(table)
    if args.command == "ladder":
        print(json.dumps(ladder_conditions(ranking, ks=tuple(k for k in LADDER_K if k <= 64)), indent=1))
        return 0
    selection = select_stage2(table, k=args.k)
    # Amendment 1b item 1: the Delta_who-only ranking beside the registered one, and the overlap of the two top 6.
    ranking_who = rank_heads({t: {"slot_median": e.get("slot_median_who")} for t, e in table.items()})
    overlap = rank_overlap(ranking, ranking_who, k=args.k)
    n_rest = rest_unit_count(rows, eligible["slot"])
    w1 = w1_sentence_verdict(table, selection["selected"], k=args.k, rest_units=n_rest)
    result = {"rows": len(rows), "eligible": eligible, "excluded": excluded, "replay": replay_check(rows),
              "effects": table, "ranking": [f"h{l}.{h}" for l, h in ranking],
              "ranking_who": [f"h{l}.{h}" for l, h in ranking_who], "rank_overlap": overlap,
              "persona_about_units": sorted(u for u in eligible["slot"] if persona_about.get(u)),
              "w1_rest_units": n_rest,
              "within_layer": within_layer(table), "second_text": second_text(table),
              "selection": selection, "verdicts": verdicts(table, ranking, k=args.k)}
    result["verdicts"]["182.W1"] = w1
    weights_path = Path(args.head_weights)
    if weights_path.exists():
        result["head_weights"] = head_weight_cosines(table, json.loads(weights_path.read_text(encoding="utf-8")))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    brief = {k: v for k, v in result.items() if k != "effects"}
    brief["effects_brief"] = {t: {k: v for k, v in e.items() if k in ("slot_median", "slot_mean", "slot_median_who",
                                                                     "slot_mean_who", "slot_units_both_texts",
                                                                     "slot_down", "concept_median", "random_median",
                                                                     "assistant_median", "unpushed",
                                                                     "sentence_abs_median", "resid_rel_median")}
                              for t, e in table.items()}
    print(json.dumps(brief, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
