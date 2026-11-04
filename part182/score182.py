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
Ranking: heads by slot_median, most negative first, ties by (layer, head).
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
CONTROL_LAYERS = [14]          # Amendment 1 item 3: out-of-block single-head controls (stage 1a only)
N_HEADS = 16
K_SELECT = 6
DOWN_NATS = -2.0
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


def effects(rows, slots, concepts, layers=LAYERS, n_heads=N_HEADS, control_layers=CONTROL_LAYERS) -> dict[str, dict]:
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
        sv = [v for v in slot_d.values() if v is not None]
        cv = [v for v in conc_d.values() if v is not None]
        rv = [v for v in rand_d.values() if v is not None]
        av = [v for v in asst_d.values() if v is not None]
        out[cond] = {
            "heads": (hs.condition_heads(cond, layers, n_heads, control_layers=control_layers)
                      if not cond.startswith("top") else None),
            "slot_median": _median(sv), "slot_mean": _mean(sv), "slot_n": len(sv),
            "slot_down": sum(1 for v in sv if v <= DOWN_NATS),
            "slot_missing": sum(1 for v in slot_d.values() if v is None),
            "concept_median": _median(cv), "concept_mean": _mean(cv), "concept_n": len(cv),
            "concept_down": sum(1 for v in cv if v <= DOWN_NATS),
            "random_median": _median(rv), "random_n": len(rv),
            "assistant_median": _median(av), "assistant_n": len(av),
            "unpushed": delta("unpushed", "none", "assistant"),
            "resid_rel_median": _resid_median(by_key, cond, slots),
            "slot_deltas": {u: (round(v, 4) if v is not None else None) for u, v in slot_d.items()},
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


def select_stage2(table, layers=LAYERS, n_heads=N_HEADS, k=K_SELECT) -> dict:
    """The stage-2 selection rule, fixed before stage 1 ran: the top k heads by slot_median; for each, the unselected
    head of the same layer whose slot_median is closest to zero (ties: lower head index), none reused; plus the
    combined conditions. Returns the stage-2 run list."""
    ranking = rank_heads(table, layers, n_heads)
    selected = ranking[:k]
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
    runs += [{"tag": t, "heads": hs.condition_heads(t, layers, n_heads)} for t in sel_tags + ctl_tags]
    return {"k": k, "selected": sel_tags, "controls": ctl_tags, "ranking": [f"h{l}.{h}" for l, h in ranking],
            "runs": runs}


def ladder_conditions(ranking, ks=LADDER_K) -> dict[str, dict[int, list[int]]]:
    """Stage 1c: the cumulative top-k conditions by the stage-1a ranking."""
    out = {}
    for k in ks:
        if k <= len(ranking):
            out[f"top{k}"] = hs.condition_heads(f"top{k}", LAYERS, N_HEADS, ranking)
    return out


def verdicts(table, ranking, layers=LAYERS, n_heads=N_HEADS, k=K_SELECT, ladder=None,
             control_layers=CONTROL_LAYERS) -> dict:
    """182.1 to 182.4, 182.7 and (Amendment 1) 182.8, exactly as registered. ``ladder`` is unused except to document
    that 182.1 reads the top4 condition from ``table`` when stage 1c has run. ``ranking`` is the main-layer ranking,
    so the control-layer heads never enter it."""
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
    # 182.8 the ranking stands out from the out-of-block null: no control-layer (c14.H) single-head swap has a
    # slot_median more negative than the 6th-ranked main head's. Not read if any of the 16 c14 conditions or the
    # ranking is missing.
    control_tags = [f"c{L}.{h}" for L in control_layers for h in range(n_heads)]
    control_medians = {t: table.get(t, {}).get("slot_median") for t in control_tags}
    sixth = ranking[k - 1] if len(ranking) >= k else None
    sixth_median = table.get(f"h{sixth[0]}.{sixth[1]}", {}).get("slot_median") if sixth is not None else None
    if (not control_tags or sixth_median is None
            or any(v is None for v in control_medians.values())):
        out["182.8"] = {"verdict": "not read", "sixth_head": (f"h{sixth[0]}.{sixth[1]}" if sixth else None),
                        "sixth_slot_median": sixth_median, "control_slot_medians": control_medians}
    else:
        offenders = [t for t, v in control_medians.items() if v < sixth_median]
        out["182.8"] = {"verdict": "failed" if offenders else "held",
                        "sixth_head": f"h{sixth[0]}.{sixth[1]}", "sixth_slot_median": sixth_median,
                        "control_slot_medians": control_medians,
                        "most_negative_control": min(control_medians, key=lambda t: control_medians[t]),
                        "controls_more_negative": offenders}
    out["182.5"] = {"verdict": "pending stage 2 (reader counts; score182.py stage2)"}
    out["182.6"] = {"verdict": "pending stage 2 (reader counts; score182.py stage2)"}
    return out


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
    table = effects(rows, eligible["slot"], eligible["concept"])
    ranking = rank_heads(table)
    if args.command == "ladder":
        print(json.dumps(ladder_conditions(ranking, ks=tuple(k for k in LADDER_K if k <= 64)), indent=1))
        return 0
    result = {"rows": len(rows), "eligible": eligible, "excluded": excluded, "replay": replay_check(rows),
              "effects": table, "ranking": [f"h{l}.{h}" for l, h in ranking],
              "selection": select_stage2(table, k=args.k), "verdicts": verdicts(table, ranking, k=args.k)}
    weights_path = Path(args.head_weights)
    if weights_path.exists():
        result["head_weights"] = head_weight_cosines(table, json.loads(weights_path.read_text(encoding="utf-8")))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    brief = {k: v for k, v in result.items() if k != "effects"}
    brief["effects_brief"] = {t: {k: v for k, v in e.items() if k in ("slot_median", "slot_mean", "slot_down",
                                                                     "concept_median", "random_median",
                                                                     "assistant_median", "unpushed",
                                                                     "resid_rel_median")}
                              for t, e in table.items()}
    print(json.dumps(brief, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
