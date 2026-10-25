"""Interactive Part 54A page: a dose slider that shows every real answer.

Writes figs-prototype/alien-dose.html, one self-contained file (data inlined as
JSON, no network fetches). Run from the repository root:
    python3 figs-prototype/make_alien_html.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import data54a  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alien-dose.html")
DOSES = [None, -3.0, -2.0, -1.0, 0.5, 1.0, 1.5, 2.0]

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Alien Neuron Dose</title>
<style>
:root {
  --bg: #ffffff; --ink: #1d1d1f; --muted: #6e6e73; --line: #e5e5ea;
  --grey-fill: #f2f2f4; --off: #8e3fd6; --off-tint: #ead9fb; --earth: #2e9e5b;
  --earth-tint: #d5f0df; --maker-tint: #fde2c4; --dot: #d9d9de; --fail: #d93025;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
  font: 17px/1.5 Inter, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding: 28px 16px 48px; }
h1 { font-size: clamp(26px, 5vw, 40px); line-height: 1.15; margin: 0 0 10px; letter-spacing: -0.01em; }
.lede { color: var(--muted); margin: 0 0 22px; max-width: 70ch; }
.panel { position: static; z-index: 5; background: var(--bg); padding: 12px 0 14px;
  border-bottom: 1px solid var(--line); }
.slider-row { display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }
.dose-now { font-size: 30px; font-weight: 700; min-width: 5.5ch; font-variant-numeric: tabular-nums; }
input[type=range] { flex: 1 1 260px; accent-color: var(--off); height: 32px; }
.ticks { display: grid; grid-template-columns: repeat(8, 1fr); margin-top: 4px; }
.tick { border: 0; background: none; font: inherit; font-size: 13px; color: var(--muted); cursor: pointer;
  padding: 6px 0; border-radius: 8px; text-align: center; }
.tick b { display: block; font-size: 18px; font-variant-numeric: tabular-nums; }
.tick.on { background: var(--grey-fill); color: var(--ink); }
.tick.hot b { color: var(--off); }
.summary { display: flex; flex-wrap: wrap; gap: 8px 22px; align-items: center; margin-top: 10px; }
.count { font-size: 22px; font-weight: 700; }
.count.hot { color: var(--off); }
.dots { display: grid; grid-template-columns: repeat(12, 14px); gap: 5px; }
.d { width: 14px; height: 14px; border-radius: 50%; background: var(--dot); position: relative; }
.d.off { background: var(--off); } .d.earth { background: var(--earth); }
.d.broken::after { content: "×"; position: absolute; inset: -4px 0 0 0; text-align: center;
  font-size: 15px; line-height: 20px; color: var(--ink); font-weight: 700; }
.cols { display: grid; grid-template-columns: 3fr 2fr; gap: 28px; margin-top: 22px; }
@media (max-width: 820px) { .cols { grid-template-columns: 1fr; } }
@media (min-width: 821px) { .panel { position: sticky; top: 0; } }
.col h2 { font-size: 20px; margin: 0 0 4px; }
.col .sub { color: var(--muted); font-size: 14px; margin: 0 0 12px; }
.col.grey h2 { color: var(--muted); }
.q { display: flex; justify-content: flex-end; margin: 18px 0 10px; }
.q span { background: var(--ink); color: #fff; border-radius: 18px 18px 4px 18px; padding: 8px 14px; font-weight: 500; }
.b { background: #fff; border: 1.5px solid #3a3a3c; border-radius: 16px 16px 16px 4px; padding: 10px 13px;
  margin: 0 0 10px; white-space: pre-wrap; overflow-wrap: anywhere; }
.grey .b { background: var(--grey-fill); border-color: var(--grey-fill); }
.b.is-off { border-color: var(--off); }
.b.is-broken { border-style: dashed; border-color: var(--fail); }
.chip { display: inline-block; font-size: 12px; font-weight: 700; letter-spacing: 0.02em; border-radius: 999px;
  padding: 1px 8px; margin: 0 6px 6px 0; background: var(--grey-fill); color: var(--muted); }
.chip.off { background: var(--off-tint); color: #5b1f99; }
.chip.earth { background: var(--earth-tint); color: #1d6b3d; }
.chip.broken { background: #fde7e5; color: #a3241b; }
mark { border-radius: 4px; padding: 0 2px; color: inherit; }
mark.off { background: var(--off-tint); } mark.maker { background: var(--maker-tint); }
.note { background: var(--grey-fill); border-radius: 12px; padding: 12px 14px; color: var(--muted); font-size: 15px; }
.tabs { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 4px; }
.tabs button { font: inherit; font-size: 14px; border: 1px solid var(--line); background: #fff; border-radius: 999px;
  padding: 5px 12px; cursor: pointer; color: var(--muted); }
.tabs button.on { background: var(--ink); color: #fff; border-color: var(--ink); }
footer { margin-top: 36px; border-top: 1px solid var(--line); padding-top: 14px; color: var(--muted); font-size: 13px; }
footer code { font-size: 12px; }
button:focus-visible, input:focus-visible { outline: 3px solid var(--off); outline-offset: 2px; }
</style>
</head>
<body>
<main>
<h1>One neuron, and the model starts talking about other worlds</h1>
<p class="lede">Qwen3.5-4B, MLP neuron layer 22, unit 2742, held at a fixed value while the model answers. The dose is a
multiple of the neuron's largest activation on 1,000 Pile documents (__PILE__). Move the slider: every one of the 24 real
answers at that dose is below, with the label a blind reader gave it.</p>

<div class="panel">
  <div class="slider-row">
    <div class="dose-now" id="doseNow"></div>
    <input type="range" id="slider" min="0" max="7" step="1" aria-label="Dose">
  </div>
  <div class="ticks" id="ticks"></div>
  <div class="summary">
    <div class="count" id="count"></div>
    <div class="dots" id="dots" aria-hidden="true"></div>
    <div id="extra" style="color:var(--muted);font-size:15px"></div>
  </div>
</div>

<div class="cols">
  <section class="col" id="left"></section>
  <section class="col grey" id="right"></section>
</div>

<footer>
  <p>n = 24 answers per dose (3 questions × 8 samples), T __T__, top-p __TOPP__, at most __MAXTOK__ new tokens, same seeds
  at every dose; the 24 unchanged answers are identical in every dose file. Answers are verbatim (the model's
  <b>**bold**</b> is shown bold); highlights are added by this page.</p>
  <p>Labels: every answer was read blind (condition hidden) by Claude Opus 5.5 through the Batch API as OFFWORLD, EARTH or
  NEITHER, with a separate BROKEN (garbled) flag, as README Part 54 says; a second blind reader, Claude Haiku 5.5, agrees at
  Cohen's kappa 0.95 (README Part 63). Controls are three random neurons from the same layer (152, 376, 692), run only at
  −2× and +1.5×.</p>
  <p>Sources: <code>results/part54a/clamp-qwen35-4b-L22u2742*.json</code> (answers),
  <code>part63/results/key63.json</code> (<code>existing_opus_label</code> per answer). Built by
  <code>figs-prototype/make_alien_html.py</code>.</p>
</footer>
</main>
<script>
const DATA = __DATA__;
const DOSES = DATA.doses;
const QUESTIONS = DATA.questions;
const OFF_RE = /[Ee]xtra-?terrestrial|[Aa]liens?|planets?|Proxima b|Alpha Centauri|Andromeda|galax(?:y|ies)|Xeno[-\w]*|Xylo\w*|Zog[-\w]*|civilizations?|Galactic Federation|interstellar|intergalactic/g;
const MAKER_RE = /Tongyi Lab|Tongyi|Alibaba Group|Alibaba Cloud|Alibaba|Google|Tencent|DingAn|Doubao/g;

function label(d) { return d === null ? "unchanged" : (d > 0 ? "+" : "−") + Math.abs(d) + "×"; }
function cellOf(cond, d) { return DATA.rows.filter(r => r.c === cond && r.d === d); }
function tally(cell) {
  return { n: cell.length, off: cell.filter(r => r.o === "OFFWORLD").length,
           earth: cell.filter(r => r.o === "EARTH").length, broken: cell.filter(r => r.q === "BROKEN").length };
}

// text -> DOM: **bold** spans become <b>, keyword matches become <mark>
function rich(text) {
  const frag = document.createDocumentFragment();
  const parts = text.split("**");
  parts.forEach((part, i) => {
    const host = i % 2 ? document.createElement("b") : frag;
    const spans = [];
    for (const [re, cls] of [[OFF_RE, "off"], [MAKER_RE, "maker"]]) {
      re.lastIndex = 0; let m;
      while ((m = re.exec(part))) spans.push([m.index, m.index + m[0].length, cls]);
    }
    spans.sort((a, b) => a[0] - b[0]);
    let at = 0;
    for (const [s, e, cls] of spans) {
      if (s < at) continue;
      host.append(document.createTextNode(part.slice(at, s)));
      const mk = document.createElement("mark"); mk.className = cls; mk.textContent = part.slice(s, e);
      host.append(mk); at = e;
    }
    host.append(document.createTextNode(part.slice(at)));
    if (host !== frag) frag.append(host);
  });
  return frag;
}

function bubble(r) {
  const div = document.createElement("div");
  div.className = "b" + (r.o === "OFFWORLD" ? " is-off" : "") + (r.q === "BROKEN" ? " is-broken" : "");
  const chip = document.createElement("span");
  chip.className = "chip " + ({OFFWORLD: "off", EARTH: "earth"}[r.o] || "");
  chip.textContent = {OFFWORLD: "off-world", EARTH: "Earth", NEITHER: "neither"}[r.o];
  div.append(chip);
  if (r.q === "BROKEN") { const c = document.createElement("span"); c.className = "chip broken"; c.textContent = "garbled"; div.append(c); }
  div.append(document.createElement("br"));
  div.append(rich(r.a));
  return div;
}

function answers(host, cell) {
  for (const q of QUESTIONS) {
    const qd = document.createElement("div"); qd.className = "q";
    const s = document.createElement("span"); s.textContent = q; qd.append(s); host.append(qd);
    cell.filter(r => r.qu === q).forEach(r => host.append(bubble(r)));
  }
}

function dotsFor(cell) {
  const box = document.getElementById("dots"); box.replaceChildren();
  const order = [...cell].sort((a, b) => ({OFFWORLD: 0, EARTH: 1}[a.o] ?? 2) - ({OFFWORLD: 0, EARTH: 1}[b.o] ?? 2));
  for (const r of order) {
    const d = document.createElement("span");
    d.className = "d" + (r.o === "OFFWORLD" ? " off" : r.o === "EARTH" ? " earth" : "") + (r.q === "BROKEN" ? " broken" : "");
    box.append(d);
  }
}

let ctrlUnit = null;
function render(i) {
  const d = DOSES[i];
  document.getElementById("slider").value = i;
  document.getElementById("doseNow").textContent = label(d);
  document.querySelectorAll(".tick").forEach((t, j) => t.classList.toggle("on", j === i));
  const cond = d === null ? "unclamped" : "twin";
  const cell = cellOf(cond, d);
  const t = tally(cell);
  const c = document.getElementById("count");
  c.textContent = `off-world in ${t.off} of ${t.n}`;
  c.classList.toggle("hot", t.off * 2 > t.n);
  document.getElementById("extra").textContent =
    `Earth ${t.earth} · garbled ${t.broken}`;
  dotsFor(cell);

  const left = document.getElementById("left"); left.replaceChildren();
  const h = document.createElement("h2");
  h.textContent = d === null ? "Unchanged model" : `Neuron 22:2742 held at ${label(d)}`;
  const sub = document.createElement("p"); sub.className = "sub";
  sub.textContent = `all ${t.n} answers, in file order`;
  left.append(h, sub);
  answers(left, cell);

  const right = document.getElementById("right"); right.replaceChildren();
  const ctrlUnits = DATA.controls.filter(u => cellOf("control-" + u, d).length);
  const h2 = document.createElement("h2"); const sub2 = document.createElement("p"); sub2.className = "sub";
  if (d !== null && ctrlUnits.length) {
    const all = ctrlUnits.flatMap(u => cellOf("control-" + u, d));
    const ct = tally(all);
    h2.textContent = `Control: 3 random neurons at ${label(d)}`;
    sub2.textContent = `same layer, same dose · off-world in ${ct.off} of ${ct.n} · Earth ${ct.earth} · garbled ${ct.broken}`;
    right.append(h2, sub2);
    if (!ctrlUnits.includes(ctrlUnit)) ctrlUnit = ctrlUnits[0];
    const tabs = document.createElement("div"); tabs.className = "tabs";
    for (const u of ctrlUnits) {
      const b = document.createElement("button"); b.type = "button";
      const ut = tally(cellOf("control-" + u, d));
      b.textContent = `neuron 22:${u} · ${ut.off}/${ut.n}`;
      b.className = u === ctrlUnit ? "on" : "";
      b.onclick = () => { ctrlUnit = u; render(i); };
      tabs.append(b);
    }
    right.append(tabs);
    answers(right, cellOf("control-" + ctrlUnit, d));
  } else {
    h2.textContent = d === null ? "Baseline" : "Baseline: unchanged model";
    const note = document.createElement("p"); note.className = "note";
    note.textContent = d === null
      ? "This is the baseline. Random-neuron controls were run at −2× and +1.5× only; move the slider there to see them."
      : "No random-neuron control was run at this dose (only at −2× and +1.5×). Shown instead: the unchanged model with the same seeds, identical at every dose.";
    right.append(h2, note);
    if (d !== null) {
      const ut = tally(cellOf("unclamped", null));
      const s3 = document.createElement("p"); s3.className = "sub"; s3.textContent = `off-world in ${ut.off} of ${ut.n}`;
      right.append(s3);
      answers(right, cellOf("unclamped", null));
    }
  }
}

const ticks = document.getElementById("ticks");
DOSES.forEach((d, i) => {
  const t = tally(cellOf(d === null ? "unclamped" : "twin", d));
  const b = document.createElement("button"); b.type = "button"; b.className = "tick" + (t.off * 2 > t.n ? " hot" : "");
  const strong = document.createElement("b"); strong.textContent = t.off;
  b.append(strong, document.createTextNode(label(d)));
  b.setAttribute("aria-label", `${label(d)}: off-world in ${t.off} of ${t.n}`);
  b.onclick = () => render(i);
  ticks.append(b);
});
document.getElementById("slider").addEventListener("input", e => render(+e.target.value));
render(DOSES.indexOf(1.5));
</script>
</body>
</html>
"""


def main():
    rows, meta = data54a.load()
    compact = [{"d": r["dose"], "c": r["condition"], "qu": r["question"], "a": r["answer"],
                "o": r["origin"], "q": r["quality"]} for r in rows]
    data = {"doses": DOSES, "questions": data54a.QUESTIONS, "controls": meta["controls"], "rows": compact}
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    s = meta["sampling"]
    html = (PAGE.replace("__DATA__", blob)
            .replace("__PILE__", f"{meta['twin']['pile_corpus_maximum']:g}")
            .replace("__T__", str(s["temperature"])).replace("__TOPP__", str(s["top_p"]))
            .replace("__MAXTOK__", str(s["max_new_tokens"])))
    open(OUT, "w", encoding="utf-8").write(html)
    print("wrote", OUT, len(compact), "answers,", len(html) // 1024, "KB")


if __name__ == "__main__":
    main()
