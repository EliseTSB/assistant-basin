#!/usr/bin/env python3
"""Rebuild the offline viewer from the stored clamp answers and label files."""

from __future__ import annotations

import collections
import glob
import json
import random
import sys
from pathlib import Path


VIEWER_DIR = Path(__file__).resolve().parent
SOURCE_ROOT = VIEWER_DIR.parent
sys.path.insert(0, str(SOURCE_ROOT))

import blind_origin  # noqa: E402


RUNS = [
    {
        "id": "26.8",
        "title": "Part 26.8 · OLMo-2 1B",
        "folder": "results/part26-8",
        "batch": "01XoGrMFw6eRLiGdAEBJhWxt",
    },
    {
        "id": "54A",
        "title": "Part 54A · Qwen3.5-4B",
        "folder": "results/part54a",
        "batch": "01AYpTn4oDiwdib4keBnRuaA",
    },
    {
        "id": "54B",
        "title": "Part 54B · Other models",
        "folder": "results/part54b",
        "batch": "017PdYJMFE2moQjZaTGfp6z7",
    },
]

FIELDS = ("model", "question", "dose", "kind", "cond", "i")
ORIGINS = {"EARTH", "OFFWORLD", "NEITHER"}
STATES = {"OK", "BROKEN"}


def _model_for(data: dict, filename: Path) -> str:
    model = data.get("short_name") or data.get("model")
    if not model:
        model = filename.name.removeprefix("clamp-").split("-L", 1)[0]
    return model


def _kind_for(condition: dict, data: dict) -> str:
    name = condition["condition"]
    kind = "unclamped" if name == "unclamped" else (
        "twin" if "twin" in name or condition.get("unit") == data.get("best_twin") else "control"
    )
    if name.startswith("control") or "control" in name:
        kind = "control"
    return kind


def _condition_units(folder: Path) -> dict[tuple, int | None]:
    """Recover unit ids, which answers() intentionally omits from its row shape."""
    units = {}
    for filename in sorted(Path(p) for p in glob.glob(str(folder / "clamp-*.json"))):
        data = json.loads(filename.read_text(encoding="utf-8"))
        if "clamp_mult" not in data:
            continue
        model = _model_for(data, filename)
        for condition in data["conditions"]:
            name = condition["condition"]
            kind = _kind_for(condition, data)
            for i, _answer in enumerate(condition["answers"]):
                row_key = (model, data["question"], data["clamp_mult"], kind, name, i)
                units[row_key] = condition.get("unit")
    return units


def build_run(spec: dict) -> tuple[dict, dict]:
    folder = SOURCE_ROOT / spec["folder"]
    rows = blind_origin.answers(str(folder))
    random.Random(2681).shuffle(rows)

    key_path = SOURCE_ROOT / "review" / f"origin-key-msgbatch_{spec['batch']}.json"
    labels_path = SOURCE_ROOT / "review" / f"origin-labels-msgbatch_{spec['batch']}.json"
    key = json.loads(key_path.read_text(encoding="utf-8"))
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    units = _condition_units(folder)

    expected_keys = {f"ans|{n}" for n in range(len(rows))}
    key_answer_keys = {entry for entry in key if entry.startswith("ans|")}
    mismatches = []
    if key_answer_keys != expected_keys:
        mismatches.append(
            f"answer-key ids differ: rebuilt={len(expected_keys)}, stored={len(key_answer_keys)}"
        )

    model_missing_count = 0
    row_models = {row["model"] for row in rows}
    model_omission_reported = False

    for n, row in enumerate(rows):
        answer_id = f"ans|{n}"
        meta = key.get(answer_id)
        if meta is None:
            mismatches.append(f"{answer_id}: missing from stored key")
            continue
        for field in FIELDS:
            if field == "model" and field not in meta:
                # The 26.8 key predates its model field. Its source folder contains
                # exactly one model, so that omitted value is determined by source.
                model_missing_count += 1
                if len(row_models) != 1 and not model_omission_reported:
                    mismatches.append(
                        f"stored key omits model while source contains {sorted(row_models)}"
                    )
                    model_omission_reported = True
                continue
            if field not in meta or row[field] != meta[field]:
                mismatches.append(
                    f"{answer_id}: {field} rebuilt={row.get(field)!r} stored={meta.get(field)!r}"
                )
        row_key = tuple(row[field] for field in FIELDS)
        if row_key not in units:
            mismatches.append(f"{answer_id}: could not recover condition unit from source files")

    if mismatches:
        details = "\n".join(f"  - {item}" for item in mismatches[:20])
        raise SystemExit(f"Refusing to build {spec['id']}: {len(mismatches)} mismatch(es)\n{details}")

    orphan_answer_labels = sorted(
        entry for entry in labels if entry.startswith("ans|") and entry not in expected_keys
    )
    if orphan_answer_labels:
        raise SystemExit(
            f"Refusing to build {spec['id']}: {len(orphan_answer_labels)} answer labels have no rebuilt answer"
        )

    output_rows = []
    null_label_count = 0
    absent_label_count = 0
    for n, row in enumerate(rows):
        answer_id = f"ans|{n}"
        label = labels.get(answer_id)
        if answer_id not in labels:
            absent_label_count += 1
        if label is None:
            null_label_count += 1
        elif (
            not isinstance(label, list)
            or len(label) != 2
            or label[0] not in ORIGINS
            or label[1] not in STATES
        ):
            raise SystemExit(f"Refusing to build {spec['id']}: invalid label at {answer_id}: {label!r}")

        row_key = tuple(row[field] for field in FIELDS)
        output_rows.append(
            {
                **row,
                "unit": units[row_key],
                "label": label,
            }
        )

    run_data = {
        "id": spec["id"],
        "title": spec["title"],
        "answers": output_rows,
    }
    stats = {
        "answers": len(output_rows),
        "calibration_rows_excluded": sum(entry.startswith("cal|") for entry in labels),
        "null_or_missing_labels": null_label_count,
        "absent_label_entries": absent_label_count,
        "model_field_missing_in_key": model_missing_count,
        "mismatches": 0,
    }
    return run_data, stats


HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark light">
<title>Clamp answers · blind origin labels</title>
<style>
:root {
  color-scheme: dark;
  --page: #10141b; --surface: #171e28; --surface-raised: #1c2633;
  --text: #edf2f8; --muted: #a9b6c6; --line: #344355;
  --accent: #9dc8ff; --accent-ink: #102238; --highlight: #f7d573;
  --earth: #67d6aa; --offworld: #c6a4ff; --neither: #b7c4d4;
  --broken: #ff8d82; --ok: #8bd5a8; --unlabelled: #e6c66e;
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: var(--page); color: var(--text);
}
@media (prefers-color-scheme: light) {
  :root {
    color-scheme: light;
    --page: #f4f6f8; --surface: #fff; --surface-raised: #edf2f7;
    --text: #17212c; --muted: #526273; --line: #cad3dc;
    --accent: #185e9c; --accent-ink: #fff; --highlight: #fff0b8;
    --earth: #08774e; --offworld: #7042aa; --neither: #47596c;
    --broken: #b3261e; --ok: #146c3d; --unlabelled: #805d00;
  }
}
* { box-sizing: border-box; }
body { margin: 0; min-width: 320px; background: var(--page); color: var(--text); }
main { max-width: 1500px; margin: 0 auto; padding: 26px clamp(16px, 3vw, 42px) 48px; }
h1 { margin: 0; font-size: clamp(1.7rem, 3vw, 2.45rem); letter-spacing: -.035em; }
h2 { font-size: 1.15rem; margin: 0; }
p { line-height: 1.55; }
.intro { margin-bottom: 22px; }
.intro p { color: var(--muted); margin: 8px 0 0; max-width: 960px; }
.blind-note { margin-top: 16px; padding: 12px 15px; border-left: 3px solid var(--accent); background: var(--surface); border-radius: 0 8px 8px 0; color: var(--text) !important; }
.filters { display: grid; grid-template-columns: minmax(200px, 1.2fr) repeat(4, minmax(145px, 1fr)); gap: 12px; padding: 16px; background: var(--surface); border: 1px solid var(--line); border-radius: 12px; }
.field { display: flex; flex-direction: column; gap: 6px; min-width: 0; }
.field label { font-size: .78rem; font-weight: 700; color: var(--muted); }
select, button { font: inherit; }
select { width: 100%; min-height: 42px; padding: 8px 34px 8px 10px; border-radius: 7px; border: 1px solid var(--line); background: var(--surface-raised); color: var(--text); }
button { color: var(--text); }
.section { margin-top: 25px; }
.section-head { display: flex; align-items: baseline; justify-content: space-between; gap: 16px; margin-bottom: 10px; }
.section-head p, .subtle { margin: 0; color: var(--muted); font-size: .88rem; }
.table-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 10px; background: var(--surface); }
.summary-grid { min-width: 720px; display: grid; grid-template-columns: minmax(120px, .78fr) repeat(3, minmax(175px, 1fr)); }
.summary-grid > div, .summary-grid > button { min-height: 72px; padding: 9px 11px; border: 0; border-right: 1px solid var(--line); border-bottom: 1px solid var(--line); }
.summary-grid > :nth-child(4n) { border-right: 0; }
.summary-grid > :nth-last-child(-n+4) { border-bottom: 0; }
.grid-heading { color: var(--muted); font-size: .78rem; font-weight: 750; text-transform: uppercase; letter-spacing: .055em; display: flex; align-items: center; }
.dose-heading { font-weight: 750; display: flex; align-items: center; }
.matrix-cell { background: var(--surface); text-align: left; cursor: pointer; display: flex; flex-direction: column; justify-content: center; gap: 3px; transition: background .12s ease; }
.matrix-cell:hover, .matrix-cell:focus-visible { background: var(--surface-raised); outline: 2px solid var(--accent); outline-offset: -2px; }
.matrix-cell.active { box-shadow: inset 0 0 0 2px var(--accent); }
.matrix-cell:disabled { cursor: default; opacity: .55; }
.matrix-n { font-size: 1rem; font-weight: 800; }
.matrix-counts, .matrix-extra { color: var(--muted); font-size: .73rem; line-height: 1.3; }
.legend { display: flex; flex-wrap: wrap; gap: 8px 14px; margin: 10px 0 0; color: var(--muted); font-size: .8rem; }
.legend span { display: inline-flex; align-items: center; gap: 5px; }
.dot { width: 9px; height: 9px; display: inline-block; border-radius: 50%; background: currentColor; }
.legend .earth { color: var(--earth); } .legend .offworld { color: var(--offworld); } .legend .neither { color: var(--neither); } .legend .broken { color: var(--broken); } .legend .unlabelled { color: var(--unlabelled); }
.list-tools { display: flex; align-items: center; gap: 12px; }
.segmented { display: inline-flex; padding: 3px; border: 1px solid var(--line); border-radius: 8px; background: var(--surface); }
.segmented button { border: 0; border-radius: 5px; padding: 6px 10px; background: transparent; color: var(--muted); cursor: pointer; }
.segmented button[aria-pressed="true"] { background: var(--accent); color: var(--accent-ink); font-weight: 750; }
.answer-groups { display: grid; gap: 20px; }
.answer-group h3 { margin: 0 0 9px; font-size: 1rem; color: var(--accent); }
.cards { display: grid; gap: 10px; }
.card { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 15px 17px; min-width: 0; }
.card.broken { border-left: 4px solid var(--broken); padding-left: 14px; }
.card-top { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; }
.card-question { font-size: .95rem; font-weight: 750; line-height: 1.4; }
.chips { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 6px; }
.chip { display: inline-flex; align-items: center; min-height: 25px; padding: 3px 8px; border-radius: 99px; background: var(--surface-raised); border: 1px solid var(--line); font-size: .75rem; font-weight: 750; white-space: nowrap; }
.chip.earth { color: var(--earth); } .chip.offworld { color: var(--offworld); } .chip.neither { color: var(--neither); }
.chip.ok { color: var(--ok); } .chip.broken-chip { color: var(--broken); border-color: var(--broken); }
.chip.unlabelled { color: var(--unlabelled); }
.metadata { display: flex; flex-wrap: wrap; gap: 5px 14px; margin: 8px 0 11px; color: var(--muted); font-size: .8rem; }
.answer-text { margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; line-height: 1.6; }
.answer-text strong { color: var(--highlight); font-weight: 850; }
.empty { padding: 24px; background: var(--surface); border: 1px dashed var(--line); border-radius: 10px; color: var(--muted); }
footer { margin-top: 36px; padding-top: 16px; border-top: 1px solid var(--line); color: var(--muted); font-size: .82rem; }
footer p { margin: 5px 0; }
@media (max-width: 900px) { .filters { grid-template-columns: repeat(2, minmax(0, 1fr)); } .field:first-child { grid-column: 1 / -1; } }
@media (max-width: 580px) { main { padding: 20px 12px 36px; } .filters { grid-template-columns: 1fr; } .field:first-child { grid-column: auto; } .section-head { align-items: flex-start; flex-direction: column; } .card-top { flex-direction: column; } .chips { justify-content: flex-start; } }
</style>
</head>
<body>
<main>
  <header class="intro">
    <h1>Clamp answers and origin labels</h1>
    <p>Compare answers across runs, models, questions, clamp multiples, and conditions.</p>
    <p class="blind-note"><strong>Blind review:</strong> the label reader did not see the dose, condition kind, or unit. The highlighted words are only a reading aid; they do not affect the labels.</p>
  </header>

  <section class="filters" aria-label="Answer filters">
    <div class="field"><label for="run-filter">Run</label><select id="run-filter"></select></div>
    <div class="field"><label for="model-filter">Model</label><select id="model-filter"></select></div>
    <div class="field"><label for="question-filter">Question</label><select id="question-filter"></select></div>
    <div class="field"><label for="kind-filter">Condition kind</label><select id="kind-filter"></select></div>
    <div class="field"><label for="dose-filter">Dose</label><select id="dose-filter"></select></div>
  </section>

  <section class="section" aria-labelledby="summary-title">
    <div class="section-head">
      <h2 id="summary-title">Label summary</h2>
      <p id="summary-context">Counts for the selected run, model, and question. Select a cell to filter answers.</p>
    </div>
    <div class="table-wrap"><div class="summary-grid" id="summary-grid" role="table" aria-label="Answer labels by dose and condition kind"></div></div>
    <div class="legend" aria-label="Count abbreviations">
      <span class="earth"><i class="dot"></i>Earth</span><span class="offworld"><i class="dot"></i>Offworld</span>
      <span class="neither"><i class="dot"></i>Neither</span><span class="broken"><i class="dot"></i>Broken</span>
      <span class="unlabelled"><i class="dot"></i>Unlabelled</span>
    </div>
  </section>

  <section class="section" aria-labelledby="answers-title">
    <div class="section-head">
      <div><h2 id="answers-title">Answers</h2><p class="subtle" id="answer-count" aria-live="polite"></p></div>
      <div class="list-tools"><span class="subtle">Group by</span><div class="segmented" role="group" aria-label="Group answers by">
        <button type="button" data-group="dose" aria-pressed="true">Dose</button>
        <button type="button" data-group="question" aria-pressed="false">Question</button>
      </div></div>
    </div>
    <div id="answer-groups" class="answer-groups"></div>
  </section>

  <footer>
    <p>Labels are the stored two-part judgments only: claimed origin and OK/BROKEN. The labels were written by Claude Opus 5.5 (<code>claude-opus-5-5</code>, set in <code>blind_origin.py</code> and confirmed from the model field of the three batch results).</p>
    <p>There was one reader and no second reader. The reader’s stated reasoning was not kept, so there is no per-answer “why”. Bolded terms are a reading aid only.</p>
  </footer>
</main>

<script id="embedded-data" type="application/json">__EMBEDDED_DATA__</script>
<script>
"use strict";
const DATA = JSON.parse(document.getElementById("embedded-data").textContent);
const runSelect = document.getElementById("run-filter");
const modelSelect = document.getElementById("model-filter");
const questionSelect = document.getElementById("question-filter");
const kindSelect = document.getElementById("kind-filter");
const doseSelect = document.getElementById("dose-filter");
const summaryGrid = document.getElementById("summary-grid");
const groupHost = document.getElementById("answer-groups");
let groupBy = "dose";

const KIND_ORDER = ["unclamped", "twin", "control"];
const KIND_LABEL = { unclamped: "Unclamped", twin: "Twin", control: "Control" };
const ORIGIN_LABEL = { EARTH: "Earth", OFFWORLD: "Offworld", NEITHER: "Neither" };
const words = /\b(aliens|alien|space|planet|star|cosmic|Earth|humans|human|born|evolved|created|creators)\b/gi;
const params = new URLSearchParams(location.search);

function doseNumber(value) { return Number(value); }
function doseText(value) {
  const n = doseNumber(value);
  const text = Number.isInteger(n) ? String(n) : String(n).replace(/(\.\d*?)0+$/, "$1").replace(/\.$/, "");
  return text.replace(/^-/, "−") + "×";
}
function distinct(values) { return [...new Set(values)]; }
function setOptions(select, entries, allLabel, selected) {
  select.replaceChildren();
  const all = document.createElement("option"); all.value = ""; all.textContent = allLabel; select.append(all);
  for (const entry of entries) {
    const option = document.createElement("option"); option.value = String(entry.value); option.textContent = entry.label; select.append(option);
  }
  if (selected && [...select.options].some(option => option.value === selected)) select.value = selected;
}
function activeRun() { return DATA.runs.find(run => run.id === runSelect.value) || DATA.runs[0]; }
function fillFilters(run, requested = {}) {
  const answers = run.answers;
  const models = distinct(answers.map(answer => answer.model)).sort((a, b) => a.localeCompare(b));
  const questions = distinct(answers.map(answer => answer.question)).sort((a, b) => a.localeCompare(b));
  const doses = distinct(answers.map(answer => String(answer.dose))).sort((a, b) => doseNumber(a) - doseNumber(b));
  setOptions(modelSelect, models.map(value => ({ value, label: value })), "All models", requested.model || "");
  setOptions(questionSelect, questions.map(value => ({ value, label: value })), "All questions", requested.question || "");
  setOptions(kindSelect, KIND_ORDER.map(value => ({ value, label: KIND_LABEL[value] })), "All kinds", requested.kind || "");
  setOptions(doseSelect, doses.map(value => ({ value, label: doseText(value) })), "All doses", requested.dose || "");
}
function subsetForSummary(run) {
  return run.answers.filter(answer =>
    (!modelSelect.value || answer.model === modelSelect.value) &&
    (!questionSelect.value || answer.question === questionSelect.value)
  );
}
function subsetForList(run) {
  return subsetForSummary(run).filter(answer =>
    (!kindSelect.value || answer.kind === kindSelect.value) &&
    (!doseSelect.value || String(answer.dose) === doseSelect.value)
  );
}
function summaryCell(answers) {
  const tally = { n: answers.length, EARTH: 0, OFFWORLD: 0, NEITHER: 0, BROKEN: 0, unlabelled: 0 };
  for (const answer of answers) {
    if (!answer.label) { tally.unlabelled += 1; continue; }
    tally[answer.label[0]] += 1;
    if (answer.label[1] === "BROKEN") tally.BROKEN += 1;
  }
  return tally;
}
function renderSummary(run) {
  const base = subsetForSummary(run);
  const doses = distinct(base.map(answer => String(answer.dose))).sort((a, b) => doseNumber(a) - doseNumber(b));
  summaryGrid.replaceChildren();
  const corner = document.createElement("div"); corner.className = "grid-heading"; corner.textContent = "Dose"; corner.setAttribute("role", "columnheader"); summaryGrid.append(corner);
  for (const kind of KIND_ORDER) {
    const heading = document.createElement("div"); heading.className = "grid-heading"; heading.textContent = KIND_LABEL[kind]; heading.setAttribute("role", "columnheader"); summaryGrid.append(heading);
  }
  for (const dose of doses) {
    const rowLabel = document.createElement("div"); rowLabel.className = "dose-heading"; rowLabel.textContent = doseText(dose); rowLabel.setAttribute("role", "rowheader"); summaryGrid.append(rowLabel);
    for (const kind of KIND_ORDER) {
      const selected = base.filter(answer => String(answer.dose) === dose && answer.kind === kind);
      const tally = summaryCell(selected);
      const button = document.createElement("button");
      button.type = "button"; button.className = "matrix-cell"; button.setAttribute("role", "cell");
      button.disabled = selected.length === 0;
      button.setAttribute("aria-label", `${doseText(dose)}, ${KIND_LABEL[kind]}: ${tally.n} answers; ${tally.EARTH} Earth, ${tally.OFFWORLD} Offworld, ${tally.NEITHER} Neither, ${tally.BROKEN} broken, ${tally.unlabelled} unlabelled. Filter to this cell.`);
      if (doseSelect.value === dose && kindSelect.value === kind) button.classList.add("active");
      const n = document.createElement("span"); n.className = "matrix-n"; n.textContent = `${tally.n} answers`; button.append(n);
      const counts = document.createElement("span"); counts.className = "matrix-counts"; counts.textContent = `E ${tally.EARTH} · O ${tally.OFFWORLD} · N ${tally.NEITHER}`; button.append(counts);
      const extra = document.createElement("span"); extra.className = "matrix-extra"; extra.textContent = `Broken ${tally.BROKEN} · Unlabelled ${tally.unlabelled}`; button.append(extra);
      button.addEventListener("click", () => {
        doseSelect.value = dose;
        kindSelect.value = kind;
        render();
        document.getElementById("answers-title").scrollIntoView({ behavior: "smooth", block: "start" });
      });
      summaryGrid.append(button);
    }
  }
  if (!doses.length) {
    const empty = document.createElement("div"); empty.className = "empty"; empty.textContent = "No answers match the selected model and question."; summaryGrid.append(empty);
  }
}
function appendHighlighted(target, text) {
  let last = 0;
  words.lastIndex = 0;
  for (const match of text.matchAll(words)) {
    target.append(document.createTextNode(text.slice(last, match.index)));
    const strong = document.createElement("strong"); strong.textContent = match[0]; target.append(strong);
    last = match.index + match[0].length;
  }
  target.append(document.createTextNode(text.slice(last)));
}
function makeChip(label, className = "") {
  const chip = document.createElement("span"); chip.className = `chip ${className}`.trim(); chip.textContent = label; return chip;
}
function compareAnswers(a, b) {
  return doseNumber(a.dose) - doseNumber(b.dose) ||
    KIND_ORDER.indexOf(a.kind) - KIND_ORDER.indexOf(b.kind) ||
    String(a.cond).localeCompare(String(b.cond)) ||
    String(a.question).localeCompare(String(b.question)) || a.i - b.i;
}
function makeCard(answer) {
  const card = document.createElement("article");
  const broken = answer.label && answer.label[1] === "BROKEN";
  card.className = `card${broken ? " broken" : ""}`;
  const top = document.createElement("div"); top.className = "card-top";
  const question = document.createElement("div"); question.className = "card-question"; question.textContent = answer.question; top.append(question);
  const chips = document.createElement("div"); chips.className = "chips";
  if (answer.label) {
    const origin = answer.label[0]; chips.append(makeChip(ORIGIN_LABEL[origin], origin.toLowerCase()));
    chips.append(makeChip(answer.label[1], answer.label[1] === "BROKEN" ? "broken-chip" : "ok"));
  } else chips.append(makeChip("Unlabelled", "unlabelled"));
  top.append(chips); card.append(top);

  const metadata = document.createElement("div"); metadata.className = "metadata";
  for (const item of [
    `Dose ${doseText(answer.dose)}`,
    `${KIND_LABEL[answer.kind]} · ${answer.cond}`,
    `Unit ${answer.unit === null || answer.unit === undefined ? "—" : answer.unit}`,
    `Sample index ${answer.i}`
  ]) { const span = document.createElement("span"); span.textContent = item; metadata.append(span); }
  card.append(metadata);

  const text = document.createElement("p"); text.className = "answer-text"; appendHighlighted(text, answer.text); card.append(text);
  return card;
}
function renderAnswers(run) {
  const answers = subsetForList(run).sort(compareAnswers);
  document.getElementById("answer-count").textContent = `${answers.length.toLocaleString()} shown / ${run.answers.length.toLocaleString()} total in this run`;
  groupHost.replaceChildren();
  if (!answers.length) {
    const empty = document.createElement("div"); empty.className = "empty"; empty.textContent = "No answers match these filters."; groupHost.append(empty); return;
  }
  const groups = new Map();
  for (const answer of answers) {
    const name = groupBy === "dose" ? `Dose ${doseText(answer.dose)}` : answer.question;
    if (!groups.has(name)) groups.set(name, []);
    groups.get(name).push(answer);
  }
  const entries = [...groups.entries()];
  if (groupBy === "question") entries.sort(([a], [b]) => a.localeCompare(b));
  for (const [name, rows] of entries) {
    const section = document.createElement("section"); section.className = "answer-group";
    const heading = document.createElement("h3"); heading.textContent = `${name} · ${rows.length.toLocaleString()} answers`; section.append(heading);
    const cards = document.createElement("div"); cards.className = "cards";
    for (const answer of rows) cards.append(makeCard(answer));
    section.append(cards); groupHost.append(section);
  }
}
function render() {
  const run = activeRun();
  renderSummary(run);
  renderAnswers(run);
}

for (const run of DATA.runs) {
  const option = document.createElement("option"); option.value = run.id; option.textContent = run.title; runSelect.append(option);
}
const requestedRun = params.get("run");
runSelect.value = DATA.runs.some(run => run.id === requestedRun) ? requestedRun : DATA.runs[0].id;
fillFilters(activeRun(), {
  model: params.get("model") || "", question: params.get("question") || "",
  kind: params.get("kind") || "", dose: params.get("dose") || ""
});
runSelect.addEventListener("change", () => { fillFilters(activeRun()); render(); });
for (const select of [modelSelect, questionSelect, kindSelect, doseSelect]) select.addEventListener("change", render);
for (const button of document.querySelectorAll("[data-group]")) {
  button.addEventListener("click", () => {
    groupBy = button.dataset.group;
    for (const other of document.querySelectorAll("[data-group]")) other.setAttribute("aria-pressed", String(other === button));
    renderAnswers(activeRun());
  });
}
render();
if (params.get("snapshot") === "overview") {
  // A reproducible, unfiltered first-screen capture despite the long answer list.
  document.documentElement.style.height = "1800px";
  document.documentElement.style.overflow = "hidden";
  document.body.style.height = "1800px";
  document.body.style.overflow = "hidden";
}
</script>
</body>
</html>
'''


def main() -> None:
    built_runs = []
    all_stats = {}
    for spec in RUNS:
        run_data, stats = build_run(spec)
        built_runs.append(run_data)
        all_stats[spec["id"]] = stats

    payload = {"runs": built_runs}
    embedded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    embedded = embedded.replace("<", "\\u003c")
    page = HTML.replace("__EMBEDDED_DATA__", embedded)
    output_path = VIEWER_DIR / "answers-viewer.html"
    output_path.write_text(page, encoding="utf-8")

    print(f"Wrote {output_path} ({output_path.stat().st_size:,} bytes)")
    for run_id, stats in all_stats.items():
        print(
            f"{run_id}: {stats['answers']:,} answers; "
            f"{stats['calibration_rows_excluded']} calibration rows excluded; "
            f"{stats['null_or_missing_labels']} null/missing labels; "
            f"{stats['model_field_missing_in_key']} key rows omit model; "
            f"{stats['mismatches']} mismatches"
        )


if __name__ == "__main__":
    main()
