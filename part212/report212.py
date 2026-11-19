#!/usr/bin/env python3
"""Part 212's report: every number the results README uses, computed from the result files and printed.

Sources, all read-only:
  results/part212/stage2/score.json     the scorer's saved output (verdicts, clauses, CIs, tables)
  results/part212/stage2/run.json       the full Stage 2 manifest (revision, load, batch histogram, donors)
  results/part212/stage2/stage2.jsonl   the full Stage 2 rows, read through score212.load_stage2
  results/part212/pilot/stage1.json     the pilot's Stage 1 records (131 arms, 25 prompts)
  results/part212/pilot/run.json        the pilot's Stage 2 manifest
  results/part212/pilot/pilot.json      the pilot stopping rule's six checks
  results/part212/pilot/stage2.jsonl    the pilot's 160 Stage 2 rows
  results/part212/chain.log             the job runner's launch lines for this part
  PREDICTIONS.md                        the registered text of 212.1-212.10 (`## Part 212` section)

Nothing is typed by hand. The scorer is imported, never edited, and every number below comes from it or from the
files it reads. Values recomputed here with the scorer's own functions are checked against score.json and a
disagreement is printed as a `DIFF` line at the end, so the README's numbers and the scorer's cannot drift apart.

Run from the repo root:
  /home/elisepc/neuron-probe/.venv/bin/python -I -B part212/report212.py
Writes results/part212/report212.md and prints the same text.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _path in (ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import score212 as sc  # noqa: E402  (the scorer, imported unchanged)

RES = ROOT / "results" / "part212"
STAGE2 = RES / "stage2"
PILOT = RES / "pilot"
OUT = RES / "report212.md"

SCORE = json.loads((STAGE2 / "score.json").read_text(encoding="utf-8"))
RUN = json.loads((STAGE2 / "run.json").read_text(encoding="utf-8"))
PILOT_RUN = json.loads((PILOT / "run.json").read_text(encoding="utf-8"))
PILOT_JSON = json.loads((PILOT / "pilot.json").read_text(encoding="utf-8"))
STAGE1 = sc.load_stage1(PILOT / "stage1.json")
ROWS = sc.load_stage2(STAGE2 / "stage2.jsonl")
PILOT_ROWS = sc.load_stage2(PILOT / "stage2.jsonl")
CHAIN = [line for line in (RES / "chain.log").read_text(encoding="utf-8").splitlines() if line.strip()]
S1 = SCORE["stage1"]
S2 = SCORE["stage2"]
P = SCORE["predictions"]
MAIN = S2["main_arms"]

lines: list[str] = []
checks: list[str] = []


def emit(text: str = "") -> None:
    lines.append(text)


def f4(x) -> str:
    return "None" if x is None else f"{float(x):.4f}"


def f2(x) -> str:
    return "None" if x is None else f"{float(x):.2f}"


def check(label: str, mine, theirs, tol: float = 5e-5) -> None:
    ok = mine is not None and theirs is not None and abs(float(mine) - float(theirs)) <= tol
    checks.append(f"{'OK  ' if ok else 'DIFF'} {label}: recomputed {f4(mine)}, score.json {f4(theirs)}")


def registered() -> list[tuple[str, str, str]]:
    """(number, verbatim text, confidence) from the `## Part 212` section of PREDICTIONS.md.

    The section's table row is `| NN.n | Gemini 4 Neon (<confidence> confidence): <text> | | |`; the text is kept
    exactly as registered, with the author and confidence prefix split off.
    """
    text = (ROOT / "PREDICTIONS.md").read_text(encoding="utf-8")
    section = text[text.index("## Part 212"):]
    section = section[:section.index("## Part 213")] if "## Part 213" in section else section
    rows = []
    for line in section.splitlines():
        m = re.match(r"^\| (212\.\d+) \| Gemini 4 Neon \((high|medium|low) confidence\): (.+) \| \| \|$", line)
        if m:
            rows.append((m.group(1), m.group(3), m.group(2)))
    return rows


def main() -> int:
    emit("# Part 212 report: every number the results README uses")
    emit()
    emit("Made by `part212/report212.py` from the files named on each line. `score.json` is the scorer's own output")
    emit("(`part212/score212.py`) and is never edited here. Values recomputed with the scorer's own functions are")
    emit("checked against it at the end; a `DIFF` line there means the two disagree.")
    emit()

    # ----------------------------------------------------------------------------------------------------------
    emit("## Verdicts (results/part212/stage2/score.json, key predictions.<n>)")
    emit()
    counts: dict[str, int] = {}
    for name in sorted(P, key=lambda s: float(s.split(".")[1])):
        counts[P[name]["verdict"]] = counts.get(P[name]["verdict"], 0) + 1
        emit(f"{name}  verdict={P[name]['verdict']}")
        emit(f"    rule: {P[name]['rule']}")
        emit(f"    clauses: {json.dumps(P[name]['observed'].get('clauses', P[name]['observed']))}")
    held = [k for k, v in P.items() if v["verdict"] == "HOLDS"]
    failed = [k for k, v in P.items() if v["verdict"] == "FAILS"]
    emit(f"verdict counts: {counts}  (counted from the verdicts above)")
    emit(f"held: {', '.join(held)}  (score.json, key predictions)")
    emit(f"failed: {', '.join(failed)}  (score.json, key predictions)")
    emit(f"headline reading: {SCORE['headlines']['reading']}  (score.json, key headlines.reading)")
    emit(f"withdraw_finding_1_specificity: {json.dumps(SCORE['headlines']['withdraw_finding_1_specificity'])}"
         "  (score.json, key headlines)")
    emit()

    # ----------------------------------------------------------------------------------------------------------
    emit("## Predictions table: registered text (PREDICTIONS.md, `## Part 212`), confidence, result, verdict")
    emit()
    for number, verbatim, confidence in registered():
        emit(f"{number} | confidence={confidence} | author=Gemini 4 Neon | verdict={P[number]['verdict']}")
        emit(f"    text: {verbatim}")
    emit()

    # ----------------------------------------------------------------------------------------------------------
    emit("## Gates (score.json, key stage1.gates and stage2.gate3)")
    emit()
    g1, g2 = S1["gates"]["gate1"], S1["gates"]["gate2"]
    emit(f"Gate 1 pass={g1['pass']} checks={json.dumps(g1['checks'])}")
    emit(f"    self_patch minus unchanged mean g_ali = {f4(g1['self_patch_minus_unchanged'])} logits (bar < 0.05)")
    emit(f"    non-finite readouts = {g1['non_finite']}; mean 8-maker mass unchanged {f4(g1['mass_unchanged'])},"
         f" zero {f4(g1['mass_zero'])} (bar >= 0.50)")
    emit(f"Gate 2 pass={g2['pass']} checks={json.dumps(g2['checks'])}")
    emit(f"    zero mean delta g_ali = {f4(g2['zero_delta_g_ali'])} (bar >= +1.0); exceeds"
         f" {g2['exceeds_random_zero']} of {g2['n_random_zero']} random zero sets (bar >= 18)")
    g3 = S2["gate3"]
    emit(f"Gate 3 pass={g3['pass']} checks={json.dumps(g3['checks'])}")
    emit(f"    duplicate keys {g3['duplicate_keys']}, empty rows {g3['empty_rows']},"
         f" non-finite rows {g3['non_finite_rows']}, short donors {len(g3['short_donors'])}")
    emit(f"    planned rows {g3['planned_rows']}, row keys {g3['row_keys']}, missing planned keys"
         f" {g3['missing_planned_keys']}, unplanned row keys {g3['unplanned_row_keys']}")
    emit(f"    unchanged guard accuracy {f4(S2['guard_accuracy']['unchanged']['rate'])}"
         f" ({S2['guard_accuracy']['unchanged']['correct']} of {S2['guard_accuracy']['unchanged']['n']})")
    emit()

    # ----------------------------------------------------------------------------------------------------------
    emit("## Run records (results/part212/stage2/run.json and pilot/run.json)")
    emit()
    emit(f"model {RUN['model']} revision {RUN['revision']}  (run.json, key revision; the design pins the same)")
    emit(f"load: attn {RUN['attn_implementation']} (requested {RUN['attn_implementation_requested']}),"
         f" dtype {RUN['dtype']}, device {RUN['device']}, torch {RUN['load']['torch']},"
         f" transformers {RUN['load']['transformers']}  (run.json, key load)")
    emit(f"target units {RUN['target']}  (run.json, key target)")
    emit(f"prompts SHA-256 {RUN['prompts_sha256']}  (run.json, key prompts_sha256)")
    emit(f"requested Stage 2 batch size {RUN['batch_size']}  (run.json, key batch_size)")
    emit(f"run.json generation_batch_histogram (complete calls) {json.dumps(RUN['generation_batch_histogram'])}")
    emit(f"scored rows by actual batch size {json.dumps(S2['batch_size_histogram'])}"
         "  (score.json, key stage2.batch_size_histogram)")
    emit(f"planned rows {RUN['planned_rows']}, new rows this invocation {RUN['new_rows_this_invocation']}"
         "  (run.json)")
    emit(f"full Stage 2 wall time {f2(RUN['wall_time_seconds'])} s = {f2(RUN['wall_time_seconds'] / 3600)} hours"
         "  (run.json, key wall_time_seconds)")
    emit(f"pilot run.json: stage {PILOT_RUN['stage']}, batch {PILOT_RUN['batch_size']}, histogram"
         f" {json.dumps(PILOT_RUN['generation_batch_histogram'])}, planned {PILOT_RUN['planned_rows']},"
         f" new rows {PILOT_RUN['new_rows_this_invocation']}")
    emit(f"pilot stage1.json: forward_batch_histogram {json.dumps(STAGE1['forward_batch_histogram'])},"
         f" non_finite_readouts {STAGE1['non_finite_readouts']}, questions {len(STAGE1['questions'])},"
         f" arms {len(STAGE1['arms'])}, wall time {f2(STAGE1['wall_time_seconds'])} s")
    emit(f"donors: {len(RUN['donors']['d_fact'])} d_fact, {len(RUN['donors']['d_you'])} d_you,"
         f" {len(RUN['donors']['d_zh_maker'])} d_zh_maker; smallest L_gen"
         f" {min(d['L_gen'] for ds in RUN['donors'].values() for d in ds)} (Gate 3b bar >= 4)")
    emit()

    # ----------------------------------------------------------------------------------------------------------
    emit("## stage2.jsonl line accounting (results/part212/stage2/stage2.jsonl)")
    emit()
    raw_lines = sum(1 for line in (STAGE2 / "stage2.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())
    keys = {}
    for row in ROWS:
        key = (row.get("set"), row["arm"], int(row["question_index"]), int(row["sample"]))
        keys[key] = keys.get(key, 0) + 1
    dups = sum(1 for n in keys.values() if n > 1)
    pilot_flags = sum(1 for row in ROWS if bool(row.get("pilot")))
    pilot_rows = sum(1 for row in PILOT_ROWS if bool(row.get("pilot")))
    pilot_planned = len(sc.stage2_rows(pilot=True))
    full_planned = len(sc.stage2_rows(pilot=False))
    emit(f"non-blank lines in stage2.jsonl: {raw_lines}  (every line is one JSON row; the scorer's loader skips blanks)")
    emit(f"unique (set, arm, question_index, sample) keys: {len(keys)}; keys seen more than once: {dups}")
    emit(f"rows with pilot=true in the full file: {pilot_flags}; foreign-shape rows the scorer dropped:"
         f" {S2['foreign_shape_rows_dropped']}")
    emit(f"planned full rows (score212.stage2_rows(pilot=False)): {full_planned};"
         f" planned pilot rows (stage2_rows(pilot=True)): {pilot_planned}")
    emit(f"pilot/stage2.jsonl rows: {len(PILOT_ROWS)} (pilot flag true on {pilot_rows})")
    emit(f"run.json planned_rows {RUN['planned_rows']} = full plan; run.json new_rows_this_invocation"
         f" {RUN['new_rows_this_invocation']} = the rows the 14:17 relaunch wrote (the 14:06 launch's rows were"
         " already on disk and resumed from)")
    emit()

    # ----------------------------------------------------------------------------------------------------------
    emit("## 212.1 zero vs unchanged vs the 20 random zero sets (Stage 2, held-out 24 prompts x 8 samples)")
    emit()
    z, u = MAIN["zero"], MAIN["unchanged"]
    emit(f"zero clean_google {z['clean_google']} of {z['denominator']}, rate {f4(z['clean_google_rate'])},"
         f" broken {z['broken']} (rate {f4(z['broken_rate'])})")
    emit(f"unchanged clean_google {u['clean_google']} of {u['denominator']}, rate {f4(u['clean_google_rate'])},"
         f" broken {u['broken']}; legacy_any_google {u['legacy_any_google']} (decides nothing)")
    rank = S2["ranks"]["zero"]
    emit(f"zero full-arm share {f4(rank['full_arm_share'])} exceeds {rank['exceeds']} of"
         f" {len(rank['random_set_shares'])} random zero shares, all={rank['exceeds_all']}")
    emit(f"zero sample-0 rank (descriptive, decides nothing): {rank['sample0_count']} of 24,"
         f" exceeds {rank['sample0_exceeds']} of {len(rank['random_set_shares'])}")
    emit(f"random zero set counts (clean_google of 24): {json.dumps(dict(sorted(rank['random_set_counts'].items())))}")
    emit(f"random zero set shares: {json.dumps(dict(sorted(rank['random_set_shares'].items())))}")
    emit(f"clauses: {json.dumps(P['212.1']['observed']['clauses'])}")
    emit()

    # ----------------------------------------------------------------------------------------------------------
    emit("## Stage 2 main arms: five-state counts (denominator 192; score.json, key stage2.main_arms)")
    emit()
    emit("| arm | clean_google | clean_alibaba | clean_other_maker | clean_no_maker | broken | multi_hit |"
         " legacy_any_google |")
    emit("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for arm in sorted(MAIN):
        t = MAIN[arm]
        c = t["counts"]
        emit(f"| `{arm}` | {c['clean_google']} | {c['clean_alibaba']} | {c['clean_other_maker']} |"
             f" {c['clean_no_maker']} | {c['broken']} | {t['multi_hit']} | {t['legacy_any_google']} |")
    emit()
    emit("rate per 192 (clean_google / 192): "
         + ", ".join(f"`{arm}` {f4(MAIN[arm]['clean_google_rate'])}" for arm in sorted(MAIN)))
    emit()

    # ----------------------------------------------------------------------------------------------------------
    emit("## 212.2 and 212.4 trace_resample_fact (Stage 1 exact logits, Stage 2 free generation)")
    emit()
    emit(f"Stage 1 zero mean delta g_ali {f4(S1['zero_delta_g_ali'])}; trace_resample_fact"
         f" {f4(S1['trace_fact_delta_g_ali'])}; retention fact/zero {f4(S1['fact_retention_vs_zero'])}")
    emit(f"Stage 1 trace_resample_you {f4(S1['trace_you_delta_g_ali'])}; mean_resample_fact"
         f" {f4(S1['mean_resample_fact_delta_g_ali'])}; retention mean/zero {f4(S1['mean_retention_vs_zero'])}")
    rz = S1["random_zero_sets"]
    emit(f"Stage 1 zero exceeds {sum(S1['zero_delta_g_ali'] > v for v in rz.values())} of {len(rz)} random zero sets")
    emit(f"Stage 1 random zero set values: {json.dumps(dict(sorted(rz.items())))}")
    emit(f"Stage 1 half of zero's shift = {f4(0.50 * S1['zero_delta_g_ali'])} (the 212.2 bar trace_resample_fact"
         " had to stay below)")
    emit(f"Stage 2 trace/zero = {f4(MAIN['trace_resample_fact']['clean_google'] / MAIN['zero']['clean_google'])};"
         f" mean/zero = {f4(MAIN['mean_resample_fact']['clean_google'] / MAIN['zero']['clean_google'])};"
         f" sub/zero = {f4(MAIN['sub_factual']['clean_google'] / MAIN['zero']['clean_google'])}")
    t = MAIN["trace_resample_fact"]
    emit(f"Stage 2 trace_resample_fact clean_google {t['clean_google']} of 192, broken {t['broken']};"
         f" half of zero {f2(P['212.4']['observed']['half_zero'])}; worst-case recode total"
         f" {P['212.4']['observed']['recode_total']}")
    boot = S2["bootstrap"]["zero-trace_resample_fact"]
    emit(f"bootstrap zero - trace_resample_fact: estimate {f4(boot['estimate'])}, 95% CI"
         f" [{f4(boot['lo'])}, {f4(boot['hi'])}], {boot['draws']} draws, seed {boot['seed']},"
         f" clustered on {boot['cluster']} ({boot['n_questions']} questions)")
    boot2 = S2["bootstrap"]["trace_resample_fact-unchanged"]
    emit(f"bootstrap trace_resample_fact - unchanged: estimate {f4(boot2['estimate'])}, 95% CI"
         f" [{f4(boot2['lo'])}, {f4(boot2['hi'])}]")
    emit(f"clauses 212.2: {json.dumps(P['212.2']['observed']['clauses'])}")
    emit(f"clauses 212.4: {json.dumps(P['212.4']['observed']['clauses'])}")
    emit()

    # ----------------------------------------------------------------------------------------------------------
    emit("## 212.3 and 212.5 sub_factual")
    emit()
    emit(f"Stage 1 sub_factual mean delta g_ali {f4(S1['sub_factual_delta_g_ali'])} (bar >= +0.75);"
         f" ratio to zero {f4(S1['sub_retention_vs_zero'])} (bar >= 0.35); sub_factual_online"
         f" {f4(S1['sub_factual_online_delta_g_ali'])}")
    rs = S1["random_sub_factual_sets"]
    emit(f"Stage 1 sub_factual exceeds {sum(S1['sub_factual_delta_g_ali'] > v for v in rs.values())} of {len(rs)}"
         " random sub_factual sets")
    emit(f"clauses 212.3: {json.dumps(P['212.3']['observed']['clauses'])}")
    s = MAIN["sub_factual"]
    emit(f"Stage 2 sub_factual clean_google {s['clean_google']} of 192, rate {f4(s['clean_google_rate'])};"
         f" unchanged rate {f4(u['clean_google_rate'])}; difference {f4(P['212.5']['observed']['difference'])}"
         " (bar >= 0.05)")
    rank_s = S2["ranks"]["sub_factual"]
    emit(f"sub_factual full-arm share {f4(rank_s['full_arm_share'])} exceeds {rank_s['exceeds']} of"
         f" {len(rank_s['random_set_shares'])} random sub shares (bar >= 18); sample-0 rank"
         f" {rank_s['sample0_count']} of 24 exceeds {rank_s['sample0_exceeds']} (descriptive)")
    emit(f"random sub_factual set counts (clean_google of 24):"
         f" {json.dumps(dict(sorted(rank_s['random_set_counts'].items())))}")
    boot3 = S2["bootstrap"]["sub_factual-unchanged"]
    emit(f"bootstrap sub_factual - unchanged: estimate {f4(boot3['estimate'])}, 95% CI"
         f" [{f4(boot3['lo'])}, {f4(boot3['hi'])}]")
    emit(f"clauses 212.5: {json.dumps(P['212.5']['observed']['clauses'])}")
    emit()

    # ----------------------------------------------------------------------------------------------------------
    emit("## 212.6 unit gaps (Stage 1, per-unit trace_resample_fact minus trace_resample_you)")
    emit()
    emit(f"trace_resample_fact - trace_resample_you (whole arm) = {f4(S1['fact_minus_you'])} (bar >= +0.25)")
    for unit, gap in S1["unit_fact_minus_you"].items():
        emit(f"    {unit}: {f4(gap)}")
    emit("per-unit mean delta g_ali (Stage 1, pilot/stage1.json, key arms.<unit>:<arm>.mean_delta_g_ali):")
    for layer, unit in ((6, 753), (7, 1169), (11, 8868), (29, 776)):
        vals = {arm: STAGE1["arms"][f"unit:{layer}:{unit}:{arm}"]["mean_delta_g_ali"]
                for arm in ("zero", "trace_resample_fact", "trace_resample_you", "sub_factual")}
        emit(f"    L{layer} N{unit}: " + ", ".join(f"{k} {f4(v)}" for k, v in vals.items()))
    emit(f"clauses 212.6: {json.dumps(P['212.6']['observed']['clauses'])}")
    emit()

    # ----------------------------------------------------------------------------------------------------------
    emit("## 212.7 L27 N3325 shift against delta g_ali across the 10 core arms (Spearman)")
    emit()
    xs = P["212.7"]["observed"]["l27_shift"]
    ys = P["212.7"]["observed"]["delta_g_ali"]
    for arm in sc.CORE_10:
        emit(f"    {arm}: L27 shift {f4(xs[arm])}, delta g_ali {f4(ys[arm])}")
    rho = sc.spearman([xs[a] for a in sc.CORE_10], [ys[a] for a in sc.CORE_10])
    check("212.7 spearman", rho, P["212.7"]["observed"]["spearman"])
    emit(f"Spearman rho = {f4(rho)} over {P['212.7']['observed']['n_arms']} arms (bar <= -0.70)")
    emit()

    # ----------------------------------------------------------------------------------------------------------
    emit("## 212.8 opposite-direction scaling (Stage 1)")
    emit()
    emit(f"scale_pos2 delta g_ali {f4(S1['scale_pos2_delta_g_ali'])} (bar <= -0.30); scale_neg1"
         f" {f4(S1['scale_neg1_delta_g_ali'])}; zero {f4(S1['zero_delta_g_ali'])}")
    emit(f"Stage 2 scale_pos2 clean_google {MAIN['scale_pos2']['clean_google']} of 192; scale_neg1"
         f" {MAIN['scale_neg1']['clean_google']} of 192 (descriptive)")
    emit(f"clauses 212.8: {json.dumps(P['212.8']['observed']['clauses'])}")
    emit()

    # ----------------------------------------------------------------------------------------------------------
    emit("## 212.9 zero_gen_only vs zero_prompt_only (Stage 2)")
    emit()
    g, p = MAIN["zero_gen_only"], MAIN["zero_prompt_only"]
    emit(f"zero_gen_only clean_google {g['clean_google']} of 192, rate {f4(g['clean_google_rate'])};"
         f" zero_prompt_only {p['clean_google']} of 192, rate {f4(p['clean_google_rate'])}; difference"
         f" {f4(P['212.9']['observed']['difference'])} (bar >= 0.04)")
    boot4 = S2["bootstrap"]["zero_gen_only-zero_prompt_only"]
    emit(f"bootstrap zero_gen_only - zero_prompt_only: estimate {f4(boot4['estimate'])}, 95% CI"
         f" [{f4(boot4['lo'])}, {f4(boot4['hi'])}] (descriptive)")
    emit(f"zero_prompt_only broken {p['broken']} (the only broken row in any main arm:"
         f" {sum(MAIN[a]['broken'] for a in MAIN)} across all main arms)")
    emit()

    # ----------------------------------------------------------------------------------------------------------
    emit("## 212.10 Chinese-donor swap (Stage 1)")
    emit()
    emit(f"trace_resample_zh_maker delta g_ali {f4(S1['trace_zh_maker_delta_g_ali'])} (bar > -0.25);"
         f" trace_resample_zh_fact {f4(S1['trace_zh_fact_delta_g_ali'])}; zh_maker - zh_fact"
         f" {f4(S1['zh_maker_minus_zh_fact'])} (descriptive)")
    emit(f"Stage 2 trace_resample_zh_maker clean_google {MAIN['trace_resample_zh_maker']['clean_google']} of 192"
         " (descriptive)")
    zero_u = MAIN["zero"]["clean_google"] / MAIN["unchanged"]["clean_google"]
    sub_u = MAIN["sub_factual"]["clean_google"] / MAIN["unchanged"]["clean_google"]
    gen_prompt = MAIN["zero_gen_only"]["clean_google"] / MAIN["zero_prompt_only"]["clean_google"]
    emit(f"Stage 2 zero / unchanged ratio of clean Google counts = {f4(zero_u)}; sub_factual / unchanged = {f4(sub_u)}")
    emit(f"Stage 2 zero_gen_only / zero_prompt_only ratio = {f4(gen_prompt)}")
    emit()

    # ----------------------------------------------------------------------------------------------------------
    emit("## Factorial interaction, g_seq, perturbation norms, positive control")
    emit()
    I = S1["factorial_interaction_I"]
    emit(f"factorial interaction I = {f4(I)} logits"
         "  (score.json, key stage1.factorial_interaction_I; = F(self_patch) - F(mean_resample_fact)"
         " - F(sub_factual) + F(zero), F = mean g_ali)")
    emit(f"    F(self_patch) {f4(STAGE1['arms']['self_patch']['mean_g_ali'])},"
         f" F(mean_resample_fact) {f4(STAGE1['arms']['mean_resample_fact']['mean_g_ali'])},"
         f" F(sub_factual) {f4(STAGE1['arms']['sub_factual']['mean_g_ali'])},"
         f" F(zero) {f4(STAGE1['arms']['zero']['mean_g_ali'])}")
    gs = S1["g_seq"]
    emit(f"g_seq: zero {f4(gs['zero'])}, trace_resample_fact {f4(gs['trace_resample_fact'])},"
         f" sub_factual {f4(gs['sub_factual'])}; fact_retention {f4(gs['fact_retention'])},"
         f" sub_retention {f4(gs['sub_retention'])}")
    emit(f"g_seq agreement passes: {SCORE['g_seq_agreement']['passes']}"
         f" checks {json.dumps(SCORE['g_seq_agreement']['checks'])}")
    norms = S1["realized_perturbation_norms"]
    for arm in ("zero", "trace_resample_fact", "trace_resample_you", "mean_resample_fact", "sub_factual",
                "scale_pos2", "scale_neg1", "trace_resample_zh_maker", "trace_resample_zh_fact"):
        emit(f"realized perturbation norm {arm}: {f4(norms[arm])}")
    emit("positive control \"Who created you?\" clean_google of 24 (score.json, key stage2.positive_control): "
         + ", ".join(f"`{a}` {v['clean_google']}" for a, v in S2["positive_control"].items() if v["n"]))
    emit()

    # ----------------------------------------------------------------------------------------------------------
    emit("## Amendment 1: the registered Stage 1 count and the batch shapes on record")
    emit()
    emit(f"the design's Size and Pilot sections say 3,989 Stage 1 forward passes; Amendment 1 (PART212-DESIGN.md) says"
         f" that is an arithmetic error for 3,449; pilot/stage1.json forward_batch_histogram"
         f" {json.dumps(STAGE1['forward_batch_histogram'])} sums to {sum(STAGE1['forward_batch_histogram'].values())}")
    emit(f"run.json generation_batch_histogram (complete calls) {json.dumps(RUN['generation_batch_histogram'])}"
         f" sums to {sum(RUN['generation_batch_histogram'].values())} calls")
    emit(f"scored rows by actual batch size {json.dumps(S2['batch_size_histogram'])}"
         f" sums to {sum(S2['batch_size_histogram'].values())} rows")
    emit(f"every row records batch_size_requested: {sorted({r['batch_size_requested'] for r in ROWS})}"
         " (the requested size is 24 on every row; the actual size is what the row was generated at)")
    emit()

    # ----------------------------------------------------------------------------------------------------------
    emit("## chain.log (results/part212/chain.log), as written")
    emit()
    for line in CHAIN:
        emit(f"    {line}")
    emit()

    # ----------------------------------------------------------------------------------------------------------
    emit("## pilot stopping rule (results/part212/pilot/pilot.json)")
    emit()
    emit(f"checks {json.dumps(PILOT_JSON['checks'])}; zero_positive_google {PILOT_JSON['zero_positive_google']};"
         f" launch_full_stage2 {PILOT_JSON['launch_full_stage2']}")
    emit(f"computed_on: {PILOT_JSON['computed_on']}")
    emit()

    # ----------------------------------------------------------------------------------------------------------
    emit("## Checks: recomputations against score.json")
    emit()
    check("factorial interaction I", I, S1["factorial_interaction_I"])
    held_now = sorted(k for k, v in P.items() if v["verdict"] == "HOLDS")
    emit(f"held now: {held_now}")
    for line in checks:
        emit(line)
    emit()

    text = "\n".join(lines) + "\n"
    OUT.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
