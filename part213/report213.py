#!/usr/bin/env python3
"""Part 213's report: every number the results README uses, computed from the result files and printed.

Sources, all read-only:
  results/part213/full/score.json        the scorer's saved output (verdicts, CIs, audit tables)
  results/part213/full/stage1-*.jsonl    the runner's Stage 1 records, read through score213.load_records and Stage1
  results/part213/full/stage2-instruct.jsonl, run-<ckpt>.json
  results/part213/audit/audit.json       the blind-audit labels (Neon) and the second reader's kappa
  results/part213/audit/ds-labels.json   the second reader's 40 labels
  results/part213/audit/sample.json      the readers' 10 known examples
  results/part213/plain-probe.json       the pre-registration plain-model probe
  part213/targets213.json, part85/checkpoints.py, results/part213/chain.log

Nothing is typed by hand: each line prints the value and, in a trailing comment, the file and key it came from. The
scorer is imported, never edited. Stage 1 values recomputed here through Stage1 are checked against score.json and a
mismatch is printed as a CHECK line, so the README's numbers and the scorer's cannot drift apart silently.

Run from the repo root:
  /home/elisepc/neuron-probe/.venv/bin/python -I -B part213/report213.py
Writes results/part213/report213.md and prints the same text.
"""
from __future__ import annotations

import json
import math
import os
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I keeps the script's folder off sys.path
import score213 as sc  # noqa: E402  (the scorer, imported unchanged)

ROOT = HERE.parent
RES = ROOT / "results" / "part213"
FULL = RES / "full"
AUDIT = RES / "audit"
OUT = RES / "report213.md"

SCORE = json.loads((FULL / "score.json").read_text(encoding="utf-8"))
AUDIT_JSON = json.loads((AUDIT / "audit.json").read_text(encoding="utf-8"))
DS_LABELS = json.loads((AUDIT / "ds-labels.json").read_text(encoding="utf-8"))["labels"]
SAMPLE = json.loads((AUDIT / "sample.json").read_text(encoding="utf-8"))  # the 40 blind ids the second reader labelled
KNOWN = json.loads((HERE / "known213.json").read_text(encoding="utf-8"))["items"]  # the 10 known examples
PLAIN = json.loads((RES / "plain-probe.json").read_text(encoding="utf-8"))

TARGETS = sc.load_targets()
RECORDS = sc.load_records(FULL)
RECORDS["labels"] = {int(k): v for k, v in AUDIT_JSON["labels"].items()}
RECORDS["kappa"] = AUDIT_JSON["kappa"]
RECORDS["reader_known_pass"] = AUDIT_JSON["reader_known_pass"]
BLIND = sc.assign_blind_ids(RECORDS, TARGETS)  # also puts blind_id on each Stage 2 row
S1 = sc.Stage1(RECORDS, TARGETS)
UNITS = {u["id"]: u for u in TARGETS["units"]}
SLOT = S1.slot
CONCEPT = S1.concept
AGREED = S1.agreed_concept
P = SCORE["predictions"]

lines: list[str] = []
checks: list[str] = []


def emit(text: str = "") -> None:
    lines.append(text)


def f4(x) -> str:
    return "None" if x is None else f"{float(x):.4f}"


def f2(x) -> str:
    return "None" if x is None else f"{float(x):.2f}"


def check(label: str, mine, theirs, tol: float = 5e-5) -> None:
    """A recomputation must agree with score.json; a disagreement is printed, never hidden."""
    ok = mine is not None and theirs is not None and abs(float(mine) - float(theirs)) <= tol
    checks.append(f"{'OK  ' if ok else 'DIFF'} {label}: recomputed {f4(mine)}, score.json {f4(theirs)}")


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def readme_predictions() -> list[tuple[str, str, str]]:
    """(number, verbatim text, confidence) from the `## Part 213` section of PREDICTIONS.md."""
    text = (ROOT / "PREDICTIONS.md").read_text(encoding="utf-8")
    start = text.index("## Part 213")
    section = text[start:]
    rows = []
    for line in section.splitlines():
        m = re.match(r"^\| (213\.\d+) \| (.+) \| (high|medium|low) \| (.+) \|$", line)
        if m:
            rows.append((m.group(1), m.group(2), m.group(3), m.group(4)))
    return rows


def checkpoints() -> dict:
    """The pinned revisions from part85/checkpoints.py, and what the run manifests recorded."""
    src = (ROOT / "part85" / "checkpoints.py").read_text(encoding="utf-8")
    out = {}
    for key in ("sft", "dpo", "instruct"):
        m = re.search(rf'Checkpoint\("{key}",\s*"[^"]*",\s*[^,]+,\s*"[^"]*",\s*"([0-9a-f]{{12}})"\)', src)
        revision = RECORDS["runs"].get(key, {}).get("snapshot_commit")
        out[key] = {"design_prefix": m.group(1) if m else None, "run_commit": revision}
    fr = RECORDS["runs"].get("fr", {})
    out["fr"] = {"formula": fr.get("fr_formula"), "dtype": fr.get("fr_dtype"),
                 "sources": fr.get("fr_sources")}
    return out


def main() -> int:
    emit("# Part 213 report: every number the results README uses")
    emit()
    emit("Made by `part213/report213.py` from the files named at the end of each line. Values recomputed here from the")
    emit("Stage 1 records through the scorer's own `Stage1` are checked against `score.json` at the end; a `DIFF` line")
    emit("there means the two disagree. `score.json` is the scorer's output and is never edited here.")
    emit()

    # ------------------------------------------------------------------------------------------------------------
    emit("## Verdicts (results/part213/full/score.json, key predictions.<n>.verdict)")
    emit()
    for name in sorted(P, key=lambda s: float(s.split(".")[1])):
        emit(f"{name}  verdict={P[name]['verdict']}  clauses={json.dumps(P[name].get('clauses', {}))}")
    emit(f"headline reading: {SCORE['headline']['reading']}  (score.json, key headline.reading)")
    gates = {k: (v.get("status") or ("passed" if v.get("passed") else "failed")) for k, v in SCORE["gates"].items()}
    emit(f"gates: {gates}  (score.json, key gates)")
    counts = {}
    for name, value in P.items():
        counts[value["verdict"]] = counts.get(value["verdict"], 0) + 1
    emit(f"verdict counts: {counts}  (counted from the verdicts above)")
    emit()

    # ------------------------------------------------------------------------------------------------------------
    emit("## Predictions table: registered text (PREDICTIONS.md, `## Part 213`), confidence, result, verdict")
    emit()
    for number, verbatim, confidence, author in readme_predictions():
        emit(f"{number} | confidence={confidence} | author={author} | verdict={P[number]['verdict']}")
        emit(f"    text: {verbatim}")
    emit()

    # ------------------------------------------------------------------------------------------------------------
    p1 = P["213.1"]
    emit("## 213.1 held-out self-slot writing (instruct chat, SELF_HELD)")
    emit()
    for kind in ("slot", "concept"):
        g = p1[kind]
        emit(f"{kind}: n={g['N']['n']} median N_z={f4(g['N']['point'])} CI [{f4(g['N']['lo'])}, {f4(g['N']['hi'])}]; "
             f"median L_z={f4(g['L']['point'])} CI [{f4(g['L']['lo'])}, {f4(g['L']['hi'])}]; "
             f"both positive {g['n_positive_both']} of {g['n_units']}  (score.json, predictions.213.1.{kind})")
    for kind in ("slot", "concept"):
        g = p1[f"own_full_np_{kind}"]
        emit(f"own full NP rises: {kind} {g['positive']} of {g['n_units']}  (score.json, predictions.213.1.own_full_np_{kind})")
    emit()

    # ------------------------------------------------------------------------------------------------------------
    p2 = P["213.2"]
    emit("## 213.2 third-person entity slots (instruct chat, THIRD_SLOT)")
    emit()
    g = p2["slot"]
    emit(f"slot: median N_z={f4(g['N']['point'])} CI [{f4(g['N']['lo'])}, {f4(g['N']['hi'])}]; "
         f"median L_z={f4(g['L']['point'])} CI [{f4(g['L']['lo'])}, {f4(g['L']['hi'])}]; "
         f"both positive {g['n_positive_both']} of {g['n_units']}  (score.json, predictions.213.2.slot)")
    emit(f"own full NP rises: {p2['own_full_np']['positive']} of {p2['own_full_np']['n_units']}  "
         f"(score.json, predictions.213.2.own_full_np)")
    emit()

    # ------------------------------------------------------------------------------------------------------------
    emit("## 213.3 third-over-self transfer tau_3/1 (instruct chat)")
    emit()
    for key in ("tau_3/1_L", "tau_3/1_N"):
        r = P["213.3"][key]
        emit(f"{key}: point={f4(r['point'])} CI [{f4(r['ci']['lo'])}, {f4(r['ci']['hi'])}] n_eligible={r['n_eligible']} "
             f"sign_guard={r['sign_guard']} read={r['read']} null_units={r['null_units']}  "
             f"(score.json, predictions.213.3.{key})")
        emit(f"    numerator median CI [{f4(r['numerator_ci']['lo'])}, {f4(r['numerator_ci']['hi'])}]")
    st = P["213.3"]["style"]
    emit(f"style-stratified point medians: direct={f4(st['direct']['point_median'])} (n={st['direct']['n']}), "
         f"scene={f4(st['scene']['point_median'])} (n={st['scene']['n']})  (score.json, predictions.213.3.style)")
    emit()

    # ------------------------------------------------------------------------------------------------------------
    p4 = P["213.4"]
    emit("## 213.4 non-identity open noun-phrase slots (instruct chat, MENTION_SLOT)")
    emit()
    g = p4["slot"]
    emit(f"slot: median N_z={f4(g['N']['point'])} CI [{f4(g['N']['lo'])}, {f4(g['N']['hi'])}]; "
         f"median L_z={f4(g['L']['point'])} CI [{f4(g['L']['lo'])}, {f4(g['L']['hi'])}]; "
         f"both positive {g['n_positive_both']} of {g['n_units']}  (score.json, predictions.213.4.slot)")
    r = p4["tau_M/3_scene"]
    emit(f"tau_M/3 (scene denominator): point={f4(r['point'])} CI [{f4(r['ci']['lo'])}, {f4(r['ci']['hi'])}] "
         f"n_eligible={r['n_eligible']} sign_guard={r['sign_guard']}  (score.json, predictions.213.4['tau_M/3_scene'])")
    r = p4["tau_M/3_pooled_descriptive"]
    emit(f"tau_M/3 pooled (descriptive): point={f4(r['point'])} CI [{f4(r['ci']['lo'])}, {f4(r['ci']['hi'])}]  "
         f"(score.json, predictions.213.4['tau_M/3_pooled_descriptive'])")
    emit()

    # ------------------------------------------------------------------------------------------------------------
    p5 = P["213.5"]
    emit("## 213.5 concept-vs-slot split on MENTION_SLOT")
    emit()
    emit(f"median L_z concept={f4(p5['median_L_concept'])} slot={f4(p5['median_L_slot'])} "
         f"agreed concept={f4(p5['median_L_agreed_concept'])}  (score.json, predictions.213.5)")
    emit(f"difference CI: point={f4(p5['ci_difference']['point'])} [{f4(p5['ci_difference']['lo'])}, "
         f"{f4(p5['ci_difference']['hi'])}] n_pairs={p5['ci_difference']['n']}; agreed difference "
         f"{f4(p5['agreed_difference'])}; reader_sensitive={p5['reader_sensitive']}  (score.json, predictions.213.5)")
    emit()

    # ------------------------------------------------------------------------------------------------------------
    emit("## 213.6 Stage 2 blind audit (untestable: Gate 3 failed)")
    emit()
    g3 = SCORE["gates"]["gate3"]
    emit(f"gate3: status={g3['status']} kappa={f4(g3['kappa'])} reader_known_pass={g3['reader_known_pass']} "
         f"labels_present={g3['labels_present']} note={g3['note']!r}  (score.json, gates.gate3)")
    emit(f"Neon known examples: {AUDIT_JSON.get('neon_known_pass')} of {len(KNOWN)}; second reader: "
         f"{AUDIT_JSON['reader_known_pass']} of {len(KNOWN)}  (audit/audit.json, keys neon_known_pass and "
         f"reader_known_pass; part213/known213.json, key items)")
    # Agreement between the two readers on the 40-item sample, recomputed from the two label files.
    agree = 0
    pairs = {}
    for key, ds in DS_LABELS.items():
        neon = AUDIT_JSON["labels"][key]
        agree += int(neon == ds)
        pairs[(neon, ds)] = pairs.get((neon, ds), 0) + 1
    emit(f"second reader: {len(DS_LABELS)} of {len(DS_LABELS)} sample items labelled, {agree} agree with Neon, raw "
         f"agreement {f4(agree / len(DS_LABELS))}  (audit/ds-labels.json and audit/audit.json, keys labels)")
    emit(f"agreement pairs (Neon, DeepSeek) -> count: {dict(sorted(pairs.items(), key=lambda kv: -kv[1]))}")
    emit(f"kappa recomputed with score213.cohens_kappa: {f4(sc.cohens_kappa([AUDIT_JSON['labels'][k] for k in DS_LABELS], [DS_LABELS[k] for k in DS_LABELS]))}")
    emit()
    table = SCORE["stage2"]["audit_table"]
    emit("audit table, Neon's labels, fixed denominators 42 slot / 36 concept (score.json, stage2.audit_table)")
    emit()
    emit("| arm-group | denominator | labelled | genuine_third_attr | first_person_reversion | incidental_or_negated | "
         "valid_refusal_or_unknown | other_coherent | broken |")
    emit("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for key in ("slot_neuron_+0.8", "slot_random_0_+0.8", "slot_neuron_-0.8", "concept_neuron_+0.8",
                "concept_random_0_+0.8"):
        row = table[key]
        c = row["counts"]
        emit(f"| {key} | {row['denominator']} | {row['labelled']} | {c['genuine_third_attr']} | "
             f"{c['first_person_reversion']} | {c['incidental_or_negated']} | {c['valid_refusal_or_unknown']} | "
             f"{c['other_coherent']} | {c['broken']} |")
    emit()
    emit(f"slot genuine_third_attr rate CI: {json.dumps(SCORE['stage2']['slot_genuine_rate_ci'])}  "
         f"(score.json, stage2.slot_genuine_rate_ci)")
    emit(f"mechanical THIRD_FREE keyword hits (descriptive): {json.dumps(SCORE['stage2']['mechanical_third_free'])}  "
         f"(score.json, stage2.mechanical_third_free)")
    emit(f"mechanical free_cold_0 keyword hits (descriptive): {json.dumps(SCORE['stage2']['mechanical_cold'])}  "
         f"(score.json, stage2.mechanical_cold)")
    # The cold checker's hand-read row: blind id 72, its arm and Neon's label, so the README can say what it is.
    by_blind = {r["blind_id"]: r for r in RECORDS["stage2"] if r.get("blind_id") is not None}
    r72 = by_blind[72]
    emit(f"blind 72: unit={r72['unit']} ({UNITS[r72['unit']]['kind_haiku']}) arm={r72['arm']} dose={r72['dose']} "
         f"prompt={r72['prompt_id']} Neon label={AUDIT_JSON['labels']['72']}  (stage2-instruct.jsonl, key blind_id)")
    emit()

    # ------------------------------------------------------------------------------------------------------------
    emit("## 213.7 opposite-direction control (instruct chat, SELF_HELD, -0.8)")
    emit()
    for kind in ("slot", "concept"):
        ci = P["213.7"][kind]
        nn = P["213.7"][f"{kind}_neuron_minus_none"]
        emit(f"{kind}: median [z_tilde(neuron -0.8) - z_tilde(random_0 -0.8)] point={f4(ci['point'])} "
             f"CI [{f4(ci['lo'])}, {f4(ci['hi'])}] n={ci['n']}; neuron -0.8 minus none point={f4(nn['point'])} "
             f"CI [{f4(nn['lo'])}, {f4(nn['hi'])}]  (score.json, predictions.213.7)")
    emit()

    # ------------------------------------------------------------------------------------------------------------
    p8 = P["213.8"]
    emit("## 213.8 small-step finite-difference sensitivity (instruct chat, +/-0.05)")
    emit()
    emit(f"n(J > 0) on SELF_HELD = {p8['n_J_positive']} of {len(SLOT)}  (score.json, predictions.213.8.n_J_positive)")
    emit(f"rho(J(SELF_HELD), L_z(SELF_HELD)) = {f4(p8['rho_self'])}  (score.json, predictions.213.8.rho_self)")
    emit(f"rho(context contrast) = {f4(p8['rho_context_contrast'])}  (score.json, predictions.213.8.rho_context_contrast)")
    emit("per-unit J(SELF_HELD) (score.json, predictions.213.8.J_self):")
    for uid in SLOT:
        emit(f"    {uid} {f4(p8['J_self'][uid])}")
    # J_N alone (the pre-fix reading) and J_R, recomputed, so the amendment's 16 -> 14 can be checked.
    jn = {u: S1.j("instruct", u, sc.SELF_HELD, arm="neuron") for u in SLOT}
    jr = {u: S1.j("instruct", u, sc.SELF_HELD, arm="random_0") for u in SLOT}
    emit(f"for comparison: n(J_N > 0) = {sum(1 for v in jn.values() if v > 0)}, n(J_R > 0) = "
         f"{sum(1 for v in jr.values() if v > 0)}  (recomputed here, Stage1.j with arm=...)")
    emit()

    # ------------------------------------------------------------------------------------------------------------
    p9 = P["213.9"]
    emit("## 213.9 cross-stage portability (sft and dpo)")
    emit()
    for cell in ("sft_self", "sft_third"):
        g = p9[cell]
        emit(f"{cell}: median N_z={f4(g['N']['point'])} median L_z={f4(g['L']['point'])} "
             f"both positive {g['n_positive_both']} of {g['n_units']}  (score.json, predictions.213.9.{cell})")
    for key in ("tau_sft", "tau_dpo"):
        r = p9[key]
        emit(f"{key}: point={f4(r['point'])} CI [{f4(r['ci']['lo'])}, {f4(r['ci']['hi'])}] n_eligible={r['n_eligible']}  "
             f"(score.json, predictions.213.9.{key})")
    d = p9["paired_difference"]
    emit(f"paired tau_3/1(dpo) - tau_3/1(sft): point={f4(d['point'])} CI [{f4(d['lo'])}, {f4(d['hi'])}] "
         f"n={d['n']}  (score.json, predictions.213.9.paired_difference)")
    fr = SCORE["fr_symmetry"]
    emit(f"FR symmetry: tau sft/instruct/dpo {json.dumps(fr['tau'])} flagged={fr['flagged']}  "
         f"(score.json, fr_symmetry)")
    emit()

    # ------------------------------------------------------------------------------------------------------------
    p10 = P["213.10"]
    emit("## 213.10 syntax and person-axis disambiguation (SELF_NONID and COPULA_NONID)")
    emit()
    c = p10["self_held_minus_self_nonid"]
    emit(f"(a) median [L_z(SELF_HELD) - L_z(SELF_NONID)]: point={f4(c['point'])} CI [{f4(c['lo'])}, {f4(c['hi'])}] "
         f"n={c['n']}  (score.json, predictions.213.10.self_held_minus_self_nonid)")
    r = p10["tau_cop_nonid/3_scene"]
    emit(f"(b) tau_cop_nonid/3 (scene denominator): point={f4(r['point'])} CI [{f4(r['ci']['lo'])}, "
         f"{f4(r['ci']['hi'])}] n_eligible={r['n_eligible']} sign_guard={r['sign_guard']}  "
         f"(score.json, predictions.213.10['tau_cop_nonid/3_scene'])")
    r = p10["tau_cop_nonid/3_pooled_descriptive"]
    emit(f"(b) pooled (descriptive): point={f4(r['point'])} CI [{f4(r['ci']['lo'])}, {f4(r['ci']['hi'])}]  "
         f"(score.json, predictions.213.10['tau_cop_nonid/3_pooled_descriptive'])")
    # The per-unit L_z behind (a), recomputed, so the README can say which units drive the difference.
    emit("per-unit L_z, recomputed here through Stage1 (SELF_HELD, SELF_NONID, COPULA_NONID, THIRD_SLOT):")
    emit("| unit | persona | L_z SELF_HELD | L_z SELF_NONID | L_z COPULA_NONID | L_z THIRD_SLOT |")
    emit("| --- | --- | ---: | ---: | ---: | ---: |")
    for uid in SLOT:
        vals = [S1.lz("instruct", uid, cell) for cell in (sc.SELF_HELD, sc.SELF_NONID, sc.COPULA_NONID, sc.THIRD_SLOT)]
        emit(f"| {uid} | {UNITS[uid]['np_text']} | " + " | ".join(f4(v) for v in vals) + " |")
    emit()

    # ------------------------------------------------------------------------------------------------------------
    p11 = P["213.11"]
    emit("## 213.11 cross-unit full-NP specificity (offdiag_8 on third_0 and third_2)")
    emit()
    d, ds_ = p11["delta_lp_spec"], p11["delta_lp_spec_slot"]
    emit(f"Delta_lp_spec: point={f4(d['point'])} CI [{f4(d['lo'])}, {f4(d['hi'])}] n={d['n']}; positive "
         f"{p11['n_positive']} of {len(SLOT)}  (score.json, predictions.213.11.delta_lp_spec, n_positive)")
    emit(f"Delta_lp_spec_slot: point={f4(ds_['point'])} CI [{f4(ds_['lo'])}, {f4(ds_['hi'])}] n={ds_['n']}  "
         f"(score.json, predictions.213.11.delta_lp_spec_slot)")
    per = [p11["per_unit"][u]["delta_lp_spec"] for u in SLOT]
    emit(f"mean of the 21 per-unit values = {f4(statistics.mean(per))}; median = {f4(statistics.median(per))}  "
         f"(computed from score.json, predictions.213.11.per_unit)")
    emit("per unit (score.json, predictions.213.11.per_unit):")
    for uid in SLOT:
        row = p11["per_unit"][uid]
        emit(f"    {uid} n_u={row['n_u']} delta_lp_spec={f4(row['delta_lp_spec'])} "
             f"delta_lp_spec_slot={f4(row['delta_lp_spec_slot'])}")
    emit()

    # ------------------------------------------------------------------------------------------------------------
    emit("## Stage 1 tables the figure and the reading use (recomputed here through score213.Stage1)")
    emit()
    cells = (("SELF_HELD", sc.SELF_HELD), ("THIRD_SLOT", sc.THIRD_SLOT), ("MENTION_SLOT", sc.MENTION),
             ("COPULA_NONID", sc.COPULA_NONID), ("SELF_NONID", sc.SELF_NONID))
    for kind, ids in (("slot", SLOT), ("concept", CONCEPT)):
        emit(f"### {kind} units, instruct chat: N_z, R_z (mean over random_0..3), L_z")
        emit()
        emit("| unit | persona | " + " | ".join(f"{name} N / R / L" for name, _ in cells) + " |")
        emit("| --- | --- | " + " | ".join(["---:"] * len(cells)) + " |")
        for uid in ids:
            row = [f"{f2(S1.nz('instruct', uid, pr))} / {f2(S1.rz('instruct', uid, pr))} / "
                   f"{f2(S1.lz('instruct', uid, pr))}" for _name, pr in cells]
            emit(f"| {uid} | {UNITS[uid]['np_text']} | " + " | ".join(row) + " |")
        emit()
        for name, pr in cells:
            l_all = [S1.lz("instruct", u, pr) for u in ids]
            emit(f"{kind} {name}: units with L_z > 0: {sum(1 for v in l_all if v is not None and v > 0)} of {len(ids)}")
            n = [S1.nz("instruct", u, pr) for u in ids]
            r = [S1.rz("instruct", u, pr) for u in ids]
            l = [S1.lz("instruct", u, pr) for u in ids]
            emit(f"{kind} {name}: median N_z={f4(sc.median(n))} median R_z={f4(sc.median(r))} "
                 f"median L_z={f4(sc.median(l))}  (recomputed; N_z = mean over prompts of neuron +0.8 minus none)")
        emit()
    emit("### Stage medians across checkpoints, slot units (recomputed)")
    emit()
    emit("| checkpoint | cell | median N_z | median R_z | median L_z | units with L_z > 0 |")
    emit("| --- | --- | ---: | ---: | ---: | ---: |")
    for ckpt in ("sft", "dpo", "instruct"):
        for name, pr in (("SELF_HELD", sc.SELF_HELD), ("THIRD_SLOT", sc.THIRD_SLOT)):
            n = [S1.nz(ckpt, u, pr) for u in SLOT]
            r = [S1.rz(ckpt, u, pr) for u in SLOT]
            l = [S1.lz(ckpt, u, pr) for u in SLOT]
            emit(f"| {ckpt} | {name} | {f4(sc.median(n))} | {f4(sc.median(r))} | {f4(sc.median(l))} | "
                 f"{sum(1 for v in l if v > 0)} of {len(SLOT)} |")
            emit(f"{ckpt} {name}: median N_z={f4(sc.median(n))} median R_z={f4(sc.median(r))} "
                 f"median L_z={f4(sc.median(l))} units with L_z > 0 {sum(1 for v in l if v > 0)} of {len(SLOT)}")
    emit()
    emit("R_z is the mean over random_0..3 at instruct and random_0 alone at sft and dpo (score213.Stage1.rz).")
    emit()
    emit("### random-arm controls beside the neuron arm, instruct chat, L_z (recomputed)")
    emit()
    emit("| cell | slot median N_z | slot median R_z | slot median L_z | concept median N_z | concept median R_z | "
         "concept median L_z |")
    emit("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, pr in cells:
        vals = []
        for ids in (SLOT, CONCEPT):
            vals += [sc.median([S1.nz("instruct", u, pr) for u in ids]),
                     sc.median([S1.rz("instruct", u, pr) for u in ids]),
                     sc.median([S1.lz("instruct", u, pr) for u in ids])]
        emit(f"| {name} | " + " | ".join(f4(v) for v in vals) + " |")
    emit()

    # ------------------------------------------------------------------------------------------------------------
    emit("## Secondary diagnostics (score.json, key secondary)")
    emit()
    sec = SCORE["secondary"]
    emit(f"ratio of medians: {json.dumps(sec['ratio_of_medians'])}")
    emit(f"paired differences: {json.dumps(sec['paired_differences'])}")
    sel = [v for v in sec["self_select_minus_self_held_L"].values() if v is not None]
    emit(f"SELF_SELECT - SELF_HELD L_z: median={f4(sc.median(sel))} n={len(sel)} "
         f"(computed from score.json, secondary.self_select_minus_self_held_L)")
    diag = SCORE["diagnostics"]
    for key, val in diag["sensitivity"].items():
        emit(f"sensitivity {key}: dropped={val['dropped']} n={val['n']} median L_z SELF_HELD={f4(val['median_L_self_held'])}")
    for key, val in diag["dog_pair"].items():
        emit(f"dog pair {key} ({val['kind']}): N_z={f4(val['N_self_held'])} L_z={f4(val['L_self_held'])}")
    ranks = diag["own_token_rank_in_O_u"]
    emit(f"own-token rank in O_u: median={f4(sc.median(ranks.values()))} min={f4(min(ranks.values()))} "
         f"max={f4(max(ranks.values()))}")
    emit(f"residual norm vs instruct: {json.dumps(SCORE['residual_norm_comparison'])}")
    emit()

    # ------------------------------------------------------------------------------------------------------------
    emit("## Plain-model probe (results/part213/plain-probe.json)")
    emit()
    emit(f"snapshot {PLAIN['snapshot']} dtype {PLAIN['dtype']} frame {PLAIN['frame']}")
    for row in PLAIN["stage2"]:
        emit(f"{row['id']}: broken={row['broken']} answer={row['answer'][:120]!r}")
    all_units = [u for cell in PLAIN["stage1_units"] for u in cell["units"]]
    emit(f"own target probability at the read position, all 17 prompts x 39 units: max={f4(max(u['p_own'] for u in all_units))} "
         f"(plain-probe.json, stage1_units[*].units[*].p_own)")
    worst = max(PLAIN["stage1_units"], key=lambda c: c["max_p_own"])
    emit(f"prompt with the largest own probability: {worst['id']} max_p_own={f4(worst['max_p_own'])} "
         f"median_rank_own={worst['median_rank_own']}")
    ranks = [c["median_rank_own"] for c in PLAIN["stage1_units"]]
    emit(f"per-prompt median own-token rank: min={min(ranks)} max={max(ranks)}")
    over = sum(c["n_p_own_over_0.5"] for c in PLAIN["stage1_units"])
    emit(f"own targets above 0.5 probability: {over}")
    emit()

    # ------------------------------------------------------------------------------------------------------------
    emit("## Checkpoints (part85/checkpoints.py and the run manifests)")
    emit()
    ck = checkpoints()
    for key in ("sft", "dpo", "instruct"):
        emit(f"{key}: design prefix {ck[key]['design_prefix']}; run snapshot_commit {ck[key]['run_commit']}  "
             f"(part85/checkpoints.py; results/part213/full/run-{key}.json, key snapshot_commit)")
    emit(f"fr: formula {ck['fr']['formula']}, dtype {ck['fr']['dtype']}, sources {json.dumps(ck['fr']['sources'])}  "
         f"(run-fr.json, keys fr_formula, fr_dtype, fr_sources)")
    for key in ("sft", "dpo", "instruct", "fr"):
        run = RECORDS["runs"][key]
        emit(f"{key}: stage1 passes executed {run.get('stage1_passes_executed')} "
             f"(core {run.get('stage1_core_passes')}, offdiag {run.get('offdiag_passes')}), peak_rss_gb "
             f"{run.get('peak_rss_gb')}  (run-{key}.json)")
    emit(f"stage1-instruct.jsonl records: {len(RECORDS['stage1']['instruct'])}  (counted)")
    emit(f"stage2 rows: {len(RECORDS['stage2'])}, of which status=ok "
         f"{sum(1 for r in RECORDS['stage2'] if r.get('status', 'ok') == 'ok')} and sealed "
         f"{sum(1 for r in RECORDS['stage2'] if r.get('status') == 'sealed')}  (counted)")
    offdiag = sum(1 for r in RECORDS["stage1"]["instruct"] if r.get("arm") == "offdiag")
    emit(f"offdiag records in stage1-instruct.jsonl: {offdiag}  (counted)")
    emit()

    # ------------------------------------------------------------------------------------------------------------
    emit("## Checks: recomputation against score.json")
    emit()
    check("213.1 slot median L_z", sc.median([S1.lz("instruct", u, sc.SELF_HELD) for u in SLOT]),
          P["213.1"]["slot"]["L"]["point"])
    check("213.1 slot median N_z", sc.median([S1.nz("instruct", u, sc.SELF_HELD) for u in SLOT]),
          P["213.1"]["slot"]["N"]["point"])
    check("213.1 concept median L_z", sc.median([S1.lz("instruct", u, sc.SELF_HELD) for u in CONCEPT]),
          P["213.1"]["concept"]["L"]["point"])
    check("213.2 slot median L_z (THIRD_SLOT)", sc.median([S1.lz("instruct", u, sc.THIRD_SLOT) for u in SLOT]),
          P["213.2"]["slot"]["L"]["point"])
    check("213.4 slot median L_z (MENTION_SLOT)", sc.median([S1.lz("instruct", u, sc.MENTION) for u in SLOT]),
          P["213.4"]["slot"]["L"]["point"])
    check("213.5 median L_z slot (MENTION_SLOT)", sc.median([S1.lz("instruct", u, sc.MENTION) for u in SLOT]),
          P["213.5"]["median_L_slot"])
    check("213.9 sft median L_z self", sc.median([S1.lz("sft", u, sc.SELF_HELD) for u in SLOT]),
          P["213.9"]["sft_self"]["L"]["point"])
    check("213.10 (a) point", sc.median([S1.lz("instruct", u, sc.SELF_HELD) - S1.lz("instruct", u, sc.SELF_NONID)
                                         for u in SLOT]), P["213.10"]["self_held_minus_self_nonid"]["point"])
    check("213.11 Delta_lp_spec point",
          sc.median([S1.delta_lp_spec("instruct", u)["delta_lp_spec"] for u in SLOT]),
          P["213.11"]["delta_lp_spec"]["point"])
    for entry in checks:
        emit(entry)
    bad = sum(1 for entry in checks if entry.startswith("DIFF"))
    emit(f"{len(checks)} checks, {bad} disagreements")
    emit()

    text = "\n".join(lines) + "\n"
    OUT.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
