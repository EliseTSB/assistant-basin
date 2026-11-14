#!/usr/bin/env python3
"""run205.py: one stage of Part 205 per invocation, writing raw records only (no verdicts).

  python run205.py --stage instruct --out results/part205                 # family O (CPU, float32, eager)
  python run205.py --stage qwen-chat --out results/part205 --device xpu   # family Q (A770, bf16, eager)

A stage is one model at one revision (part205/config.json, "stages"). The runner writes, under --out/<stage>:
  ranking.npz / ranking.json  per-prompt e for every part (float32), with the prompt ids and unit keys
  effect.npz                  per-part first-order copy effect (sum of g x (a_you - a_third) over copied positions,
                              averaged over held-out pairs) for the effect-matched random family
  copy_acts.npz               the You runs' site activations and gradients for the held-out questions
  sets.json                   the per-stage sets, drawn once: SAY, DO (top 80 by score) and the three random families
  records.jsonl               one JSON record per pass: arm, question, name, option masses, top token
  meta.json                   environment: revisions, dtype, device, transformers version, unit pool

No verdicts are computed here. G0 to G4, O, T and the rest are score205.py's.

Arms (Design, Controls and Size):
  ranking     48 forward+backward passes in the You frame (6 values x (4 SAY + 4 DO) selection prompts)
  you         one You run per held-out question (6 x 4), cached activations, one backward pass
  baseline    third-person run, with one backward pass (g for the effect-matched family)
  do_nothing  the same replacement code with the receiving run's own activations, hooking exactly the parts (the union
              of every copy arm's parts) and positions (the copied span) the copy arms hook; records the max absolute
              difference of the full last-position logits from the baseline (G1)
  do_copy / say_copy / random:<family>:<i>   the set copied from the You run over the copied span
  swap        persona-swap null: the DO set copied from the other name's third-person run
  reverse     DO set copied from the third-person run into the You run (each run's own span positions)
  off_target_* value-free self-reports under value-free persona lines: baseline, do-nothing, DO-set copy
  secondary_* Sauers' wording: secondary_you asks the plain question (the denominator, as Sauers did);
              secondary_you_source asks the third-person run's "personally" question in the You frame and is the copy
              source, so the copy carries the SAME question tokens; baseline, do-nothing, DO copy, SAY copy
The A770 G1 check (5 plain + 5 do-nothing runs of one question) is g1check205.py.

The copied span (Design, Method, lines 108-111) is the question and answer cue: from the first token of the "User:"
line to the end, identical tokens in both frames, aligned from the end; nothing of the persona line is copied.

Resume: records.jsonl is appended (flushed and fsynced per record); a torn last line (unparseable, or without its
newline) is ignored and truncated away; a forward pass whose record exists is skipped unless a missing record needs
its activations; ranking.npz and sets.json are reused. Every JSON and NPZ file is written to a temp file in the same
directory and os.replace'd into place, and a torn ranking.npz/ranking.json/sets.json found at resume is recomputed
and rewritten (with a warning), never left to brick the stage. The ranking and secondary blocks re-run whole when one
of their records is missing (review r2 H-N4: cost only, left as is). A lock file (<stage>/.lock, flock) refuses a
second concurrent run of one stage.
The sets are drawn and written before any replacement run (Design, Controls, Seeds).
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# Amendment 3 (2026-11-14): family Q's loader imports neuron_report from the repository (or job-folder) root, which
# `python -I` leaves off sys.path; the first A770 launch died on it. Appended, so this folder's modules still win.
if str(HERE.parent) not in sys.path:
    sys.path.append(str(HERE.parent))

import common205 as C  # noqa: E402


# ---------------------------------------------------------------- model loading

def stage_spec(cfg, stage):
    """The stage's family and its model id and revision, from config.json (the top-level stage list names the stage
    and family; the family's own stage list carries model_id and revision)."""
    st = next((s for s in cfg["stages"] if s["stage"] == stage), None)
    if st is None:
        raise SystemExit(f"unknown stage {stage!r}; config has {[s['stage'] for s in cfg['stages']]}")
    fam = cfg["families"][st["family"]]
    spec = next((x for x in fam.get("stages", []) if x["stage"] == stage), None)
    if spec is None:
        raise SystemExit(f"stage {stage!r} has no model_id/revision in family {st['family']}'s stage list")
    return {"stage": stage, "family": st["family"], "model_id": spec["model_id"], "revision": spec["revision"]}


def load_stage(cfg, stage, device_override=None):
    """(tokenizer, model, env record) for one stage. One model per process."""
    st = stage_spec(cfg, stage)
    fam = cfg["families"][st["family"]]
    device = device_override or fam["device"]
    import torch
    dtype = torch.float32 if fam["dtype"] == "float32" else torch.bfloat16
    if st["family"] == "Q":
        # The design's loader: another loader can still run the faulty SDPA kernel on the A770 (Brief, item 3).
        import neuron_report
        tok, model = neuron_report.load_model(st["model_id"], st["revision"], device,
                                              attn_implementation=fam["attn_implementation"], dtype=dtype)
        env = dict(neuron_report.LAST_LOAD)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(st["model_id"], revision=st["revision"])
        model = AutoModelForCausalLM.from_pretrained(st["model_id"], revision=st["revision"], dtype=dtype,
                                                     attn_implementation=fam["attn_implementation"])
        model.to(device).eval()
        torch.set_grad_enabled(False)
        import transformers
        env = {"dtype": fam["dtype"], "device": device, "attn_implementation": fam["attn_implementation"],
               "attn_implementation_requested": fam["attn_implementation"],
               "torch": torch.__version__, "transformers": transformers.__version__}
    env.update({"stage": stage, "family": st["family"], "model_id": st["model_id"], "revision": st["revision"]})
    return tok, model, env


def write_models_json(path, env):
    """Record this stage's model id, revision, dtype, device and library versions in part205/models.json (Design,
    lines 58 and 158-159), merged under the stage's name; a flock keeps two stages from losing each other's rows."""
    path = Path(path)
    with open(str(path) + ".lock", "a+") as lk:
        C.lock_file(lk)                          # fcntl on Linux, msvcrt on Windows (Amendment 3)
        # a torn models.json (a kill during the old non-atomic write) is rewritten, not left to stop every later stage
        data = C.load_json_repairing(path, {"part": 205, "stages": {}})
        if not isinstance(data, dict) or not isinstance(data.get("stages"), dict):
            data = {"part": 205, "stages": {}}
        row = {k: env.get(k) for k in ("model_id", "revision", "family", "dtype", "device", "attn_implementation",
                                        "torch", "transformers")}
        row["recorded"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        data["stages"][env["stage"]] = row
        C.atomic_write_json(path, data)          # temp + os.replace inside the lock (review r2 D-F3)
        C.unlock_file(lk)
    return path


# ---------------------------------------------------------------- one forward pass

class Runner:
    """Runs one prompt through the model, with optional hooks, and returns the readout and site tensors."""

    def __init__(self, model, tok, units, letter_ids):
        import torch
        self.model = model
        self.tok = tok
        self.units = units
        self.layers = C.text_layers(model)
        self.letter_ids = letter_ids
        self.device = next(model.parameters()).device
        self.sites = sorted({(u.site, u.layer) for u in units})
        self._site_modules = {}
        for site, layer in self.sites:
            u = next(x for x in units if x.site == site and x.layer == layer)
            self._site_modules[(site, layer)] = C.unit_site_module(self.layers, u)

    def tokens(self, text):
        return self.tok(text, return_tensors="pt", add_special_tokens=False).input_ids

    def run(self, text, need_grad=False, capture=True, replace=None, consistent="A"):
        """One pass. `replace` is a list of (site, layer, columns, positions, values) installed as SiteReplacers.

        Returns (ids, readout, acts, grads); the full last-position logit row is kept in self.last_logits. With need_grad the backward pass is on the logit difference (sum of the
        consistent letter's token ids minus the other letter's) at the last position.
        """
        import torch
        ids = self.tokens(text).to(self.device)
        recs, hooks = [], []
        if capture:
            for key in self.sites:
                recs.append((key, C.SiteRecorder(self._site_modules[key], capture_grad=need_grad)))
        if replace:
            for site, layer, cols, positions, values in replace:
                hooks.append(C.SiteReplacer(self._site_modules[(site, layer)], cols, positions, values))
        try:
            with torch.set_grad_enabled(need_grad):
                logits = self.model(input_ids=ids).logits[0, -1]
                read = C.readout(logits, self.letter_ids["A"], self.letter_ids["B"])
                self.last_logits = logits.detach().float().cpu().clone()
                if need_grad:
                    other = "B" if consistent == "A" else "A"
                    diff = logits[self.letter_ids[consistent]].sum() - logits[self.letter_ids[other]].sum()
                    self.model.zero_grad(set_to_none=True)
                    diff.backward()
        finally:
            for _, r in recs:
                r.close()
            for h in hooks:
                h.close()
        acts = {f"{s}:{l}": r.activation for (s, l), r in recs}
        grads = {f"{s}:{l}": r.grad for (s, l), r in recs} if need_grad else {}
        return ids[0].tolist(), read, acts, grads

    def replace_and_read(self, text, part_keys, positions, values_by_site, consistent, capture=False):
        """Install one replacement per site for `part_keys` and read the answer."""
        by_site = {}
        for k in part_keys:
            u = self._by_key[k]
            by_site.setdefault((u.site, u.layer), []).append(u)
        replace = []
        for (site, layer), us in by_site.items():
            cols = []
            for u in sorted(us, key=lambda u: u.index):
                cols.extend(u.cols)
            # the values are the source run's site input, cut to the columns of the parts at this site
            replace.append((site, layer, cols, positions, values_by_site[f"{site}:{layer}"][:, cols]))
        return self.run(text, need_grad=False, capture=capture, replace=replace, consistent=consistent)

    @property
    def _by_key(self):
        if not hasattr(self, "_keys"):
            self._keys = {u.key: u for u in self.units}
        return self._keys


# ---------------------------------------------------------------- prompt construction

def build_prompt(prompts, persona_name, value, question, personas=None):
    """The frame text. `personas` defaults to the value persona lines; the off-target arm passes
    prompts["off_target_personas"] (value-free) and value=None."""
    persona = C.persona_line(personas or prompts["personas"], persona_name, value)
    return C.frame_text(persona, question)


def persona_text(prompts, persona_name, value, personas=None):
    return C.persona_line(personas or prompts["personas"], persona_name, value)


def selection_prompts(prompts):
    """The 48 ranking prompts: SAY 1-4 and DO 1-4 of each value, in the You frame."""
    out = []
    for v in prompts["values"]:
        for item in prompts["items"][v]["say"]:
            if item["split"] == "selection":
                out.append({"value": v, "kind": "SAY", "item": item["item"], "id": item["id"],
                            "question": item["question"], "consistent": item["consistent"]})
        for item in prompts["items"][v]["do"]:
            if item["split"] == "selection":
                out.append({"value": v, "kind": "DO", "item": item["item"], "id": item["id"],
                            "question": item["question"], "consistent": item["consistent"]})
    return out


def heldout_say(prompts):
    out = []
    for v in prompts["values"]:
        for item in prompts["items"][v]["say"]:
            if item["split"] == "heldout":
                out.append({"value": v, "item": item["item"], "id": item["id"], "question": item["question"],
                            "consistent": item["consistent"]})
    return out


# ---------------------------------------------------------------- sets

def top80(scores, units, k=80):
    order = sorted(range(len(units)), key=lambda i: (-float(scores[i]), units[i].key))
    return [units[i].key for i in order[:k]]


def kind_key(u):
    if u.kind == "neuron":
        return "neuron"
    return f"head:{u.sub}" if u.sub else "head"


def matching_score(scores):
    """The score family (ii) matches on (Design, Controls, "the DO set's count in each decile of selection score").

    The DO set is selected by its DO score (top80(scores["do"])), so its selection score is the DO score, not the
    mean over SAY and DO prompts (scores["pooled"]). Ruled 2026-11-11 on review r2 D-F6; recorded in Amendment 1.
    """
    return scores["do"]


def draw_random_families(rngs, units, do_set, say_set, score, effect, k=80, n_sets=20):
    """The three matched random families (Design, Controls, lines 131-143).

    Every set is a seeded draw of k distinct parts without replacement from the pool minus the SAY and DO sets, with
    the DO set's kind counts (for Q: full-attention heads, DeltaNet heads, neurons).
    (i) layer: the DO set's count per layer and kind; where a layer is short, the rest comes from the nearest layer of
        the same kind (a seeded draw within that layer; ties between two equally near layers are pooled).
    (ii) score: per kind, the DO set's count in each decile of selection score, deciles within kind over the pool.
    (iii) effect: as (ii) with deciles of first-order copy effect.
    In (ii) and (iii) a short decile takes the rest from the nearest decile, drawn the same way.
    Fillability is per family ("families"[name]["fillable"], review r1 D-F9): a family whose kind counts cannot be
    met from the eligible pool is "not fillable" and has no sets. Every relaxed part is counted per family and
    recorded with the layer or decile it was wanted in and the one it came from (D-F10, D-F10b).
    """
    excluded = set(do_set) | set(say_set)
    pool = [u for u in units if u.key not in excluded]
    kinds = sorted({kind_key(u) for u in units})
    by_key = {u.key: u for u in units}
    do_units = [by_key[k] for k in do_set]
    want = {kd: sum(kind_key(u) == kd for u in do_units) for kd in kinds}
    have = {kd: sum(kind_key(u) == kd for u in pool) for kd in kinds}
    out = {"kind_counts_wanted": want, "kind_counts_eligible": have, "n_sets": n_sets, "set_size": k,
           "families": {}}
    stratifiers = {"layer": {u.key: u.layer for u in units}}
    if score is not None:
        stratifiers["score"] = _deciles(units, kinds, score)
    if effect is not None:
        stratifiers["effect"] = _deciles(units, kinds, effect)
    for name in ("layer", "score", "effect"):
        if name not in stratifiers:
            out["families"][name] = {"fillable": False, "sets": [], "relaxed": 0, "relaxed_detail": [],
                                     "reason": f"no {name} values to stratify on"}
            continue
        # fillability is decided per family (review r2 D-F10, H-N5): a family without its stratifier is not fillable
        # while the others are, and each family's reason names its own shortfall
        short = _kind_shortfalls(pool, do_units, kinds)
        if short:
            out["families"][name] = {"fillable": False, "sets": [], "relaxed": 0, "relaxed_detail": [],
                                     "reason": short}
            continue
        out["families"][name] = _draw_family(rngs[name], pool, do_units, kinds, stratifiers[name], n_sets,
                                             "layer" if name == "layer" else "decile")
    # kept for readers of the old shape: true only when all three families are fillable
    out["fillable"] = all(f["fillable"] for f in out["families"].values())
    if not out["fillable"]:
        out["not_fillable_reason"] = {n: f["reason"] for n, f in out["families"].items() if not f["fillable"]}
    return out


def _kind_shortfalls(pool, do_units, kinds):
    """Per kind: the DO set's count against the eligible parts of that kind, checked for one family.

    A short (kind, stratum) cell is not a failure: _draw_family fills it from the nearest stratum of the same kind
    and records each relaxed part. The family is not fillable only when the kind's whole total in the eligible pool
    is below the DO set's count for it (then no relaxation can reach enough parts). Returns {kind: "want n,
    eligible m"} for the kinds that fall short, {} when the family is fillable. The check is the same for every
    family: what differs per family is the stratifier, and a short cell in it relaxes rather than failing.
    """
    short = {}
    for kd in kinds:
        need = sum(1 for u in do_units if kind_key(u) == kd)
        n_kind = sum(1 for u in pool if kind_key(u) == kd)
        if n_kind < need:
            short[kd] = f"want {need}, eligible {n_kind}"
    return short


def _draw_family(rng, pool, do_units, kinds, stratum, n_sets, label):
    """n_sets seeded draws matching the DO set's count per (kind, stratum); short strata fill from the nearest."""
    fam = {"fillable": True, "sets": [], "relaxed": 0, "relaxed_detail": [], "stratum": label}
    for s in range(n_sets):
        chosen, taken = [], set()
        for kd in kinds:
            want_by = {}
            for u in do_units:
                if kind_key(u) == kd:
                    want_by[stratum[u.key]] = want_by.get(stratum[u.key], 0) + 1
            for st_val in sorted(want_by):
                need = want_by[st_val]
                elig = [u for u in pool if kind_key(u) == kd and stratum[u.key] == st_val and u.key not in taken]
                pick = rng.sample(elig, need) if len(elig) >= need else list(elig)
                taken |= {u.key for u in pick}
                short = need - len(pick)
                if short:
                    rest = [u for u in pool if kind_key(u) == kd and u.key not in taken]
                    by_dist = {}
                    for u in rest:
                        by_dist.setdefault(abs(stratum[u.key] - st_val), []).append(u)
                    extra = []
                    for dist in sorted(by_dist):
                        cands = by_dist[dist]
                        take = short - len(extra)
                        extra += rng.sample(cands, take) if len(cands) > take else cands
                        if len(extra) == short:
                            break
                    if len(extra) < short:
                        raise SystemExit(f"kind {kd}: {short} parts short after relaxation (kind counts were checked)")
                    for u in extra:
                        fam["relaxed_detail"].append({"set": s, "kind": kd, f"{label}_wanted": st_val,
                                                      f"{label}_got": stratum[u.key], "part": u.key})
                    fam["relaxed"] += short
                    pick += extra
                    taken |= {u.key for u in extra}
                chosen += pick
        fam["sets"].append([u.key for u in chosen])
    return fam


def _deciles(units, kinds, values):
    """Decile index (0-9) of each part within its kind over the whole pool."""
    out = {}
    for kd in kinds:
        members = sorted((u for u in units if kind_key(u) == kd), key=lambda u: (float(values[u.key]), u.key))
        n = len(members)
        for i, u in enumerate(members):
            out[u.key] = min(9, i * 10 // n)
    return out


# ---------------------------------------------------------------- the stage

def run_stage(args, model=None, tok=None, units=None, env=None, prompts=None, cfg=None):
    """The whole stage. `model`/`tok` may be passed by a test (a tiny model) instead of being loaded.

    Holds <out>/<stage>/.lock for the whole run, so two runs of one stage cannot interleave (review r1 D-F16).
    """
    out = Path(args.out) / args.stage
    out.mkdir(parents=True, exist_ok=True)
    lock = C.StageLock(out / ".lock")
    try:
        return _run_stage_locked(args, out, model, tok, units, env, prompts, cfg)
    finally:
        lock.release()


def _run_stage_locked(args, out, model, tok, units, env, prompts, cfg):
    import numpy as np
    cfg = cfg or C.load_config(args.config)
    prompts = prompts or C.load_prompts(args.prompts)
    stage = args.stage
    if model is None:
        tok, model, env = load_stage(cfg, stage, args.device)
    if getattr(args, "models_json", None):
        write_models_json(args.models_json, env)
    units = units or C.units_for(model)
    fam = cfg["families"][next(s["family"] for s in cfg["stages"] if s["stage"] == stage)]
    expected = fam["units"] if args.check_pool else None
    pool = C.check_pool(units, expected)
    letter_ids = {"A": C.option_token_ids(tok, "A"), "B": C.option_token_ids(tok, "B")}
    runner = Runner(model, tok, units, letter_ids)
    # ⛔ A screen resumed under other prompts or config would mix frames under one gap (DS code review F6): refuse.
    if args.stage_only == "screen" and (out / "meta.json").exists():
        try:
            old = json.loads((out / "meta.json").read_text())
        except ValueError:
            old = {}
        for k, path in (("prompts_sha", args.prompts), ("config_sha", args.config)):
            if old.get(k) and old[k] != _sha(path):
                raise SystemExit(f"{out}: an earlier screen ran under a different {k}; use a fresh --out")
    # ⛔ A stage folder holds one mode (review r1 F2, 2026-11-13): the mode is recorded in meta.json before any pass,
    # and a relaunch in the other mode is refused here, before it runs anything. A folder whose meta.json names no
    # mode but already holds records is refused too (a full run killed before sets.json carries no other marker).
    mode = getattr(args, "arms", "all")
    if args.stage_only != "screen" and (out / "meta.json").exists():
        try:
            old_meta = json.loads((out / "meta.json").read_text())
        except ValueError:
            old_meta = {}
        has_records = any(True for _ in C.read_jsonl(out / "records.jsonl", repair=False)) \
            if (out / "records.jsonl").exists() else False
        old_mode = old_meta.get("arms")
        if (old_mode is None and has_records) or (old_mode is not None and old_mode != mode):
            raise SystemExit(f"{out}: an earlier run here was {old_mode or 'unmarked'}; this run is {mode}: "
                             f"use a fresh --out")
    meta = {"env": env, "pool": pool, "letter_ids": letter_ids, "units": [u.key for u in units],
            "unit_meta": [u.as_dict() for u in units], "prompts_sha": _sha(args.prompts), "config_sha": _sha(args.config),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "arms": mode}
    _write(out / "meta.json", meta)
    rec_path = out / "records.jsonl"
    done = {_rec_key(r) for r in C.read_jsonl(rec_path, repair=True)}
    n_pass = {"run": 0}

    def emit(rec):
        key = _rec_key(rec)
        if key in done:
            return
        with rec_path.open("a+", encoding="utf-8") as f:
            if f.tell() > 0:
                # a file that does not end in a newline (a kill between the JSON and its newline) gets one first, so
                # no two records can be joined into one line (review r2 H-N3)
                f.seek(f.tell() - 1)
                if f.read(1) != "\n":
                    f.write("\n")
            f.write(json.dumps(rec, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        done.add(key)

    def missing(arm, qid=None, name=None, _value=None, persona=None):
        return _rec_key({"arm": arm, "id": qid, "name": name, "persona": persona}) not in done

    def run(*a, **k):
        n_pass["run"] += 1
        return runner.run(*a, **k)

    def replace_and_read(*a, **k):
        n_pass["run"] += 1
        return runner.replace_and_read(*a, **k)

    def frame(name, value, question, personas=None):
        persona = persona_text(prompts, name, value, personas)
        text = C.frame_text(persona, question)
        q0, ids = C.question_start(tok, text, persona)
        return text, ids, q0

    # ---- 0. Amendment 2, step A0: the six-value gap screen. Only the plain You run per held-out SAY question and the
    # plain third-person run per question-name pair (24 + 48 = 72), forward only, nothing captured or hooked: no
    # ranking, no sets, no effect. Run it with its own --out, so its records never meet the main run's resume.
    if args.stage_only == "screen":
        held = heldout_say(prompts)
        names = [n for n in prompts["personas"] if n != "You"]
        for h in held:
            for name in ["You"] + names:
                arm = "you" if name == "You" else "baseline"
                if not missing(arm, h["id"], name, h["value"], name):
                    continue
                text, ids_tok, _ = frame(name, h["value"], h["question"])
                ids, read, _acts, _grads = run(text, need_grad=False, capture=False, consistent=h["consistent"])
                assert ids == ids_tok, "runner and offset tokenisation disagree"
                emit({"arm": arm, "stage": stage, "value": h["value"], "item": h["item"], "id": h["id"],
                      "name": name, "persona": name, "consistent": h["consistent"], "n_tokens": len(ids), **read})
        return _finish(out, meta, n_pass)

    # ---- 1. ranking: 48 forward+backward passes in the You frame (skipped when ranking.npz and its records exist)
    sel = selection_prompts(prompts)
    random.Random(cfg["seeds"]["prompt_order"]).shuffle(sel)  # prompt order seeded (Design, Controls, Seeds)
    est = None
    if (out / "ranking.npz").exists() and not any(missing("ranking", p["id"], None, p["value"], "You") for p in sel):
        est = _load_ranking(out / "ranking.npz", [p["id"] for p in sel], [u.key for u in units])
    if est is None:
        est = np.zeros((len(sel), len(units)), dtype=np.float32)
        for i, p in enumerate(sel):
            text = build_prompt(prompts, "You", p["value"], p["question"])
            ids, read, acts, grads = run(text, need_grad=True, consistent=p["consistent"])
            est[i] = C.estimates_for_prompt(acts, grads, units).numpy().astype(np.float32)
            emit({"arm": "ranking", "stage": stage, "value": p["value"], "kind": p["kind"], "item": p["item"],
                  "id": p["id"], "persona": "You", "consistent": p["consistent"], "n_tokens": len(ids), **read})
        C.atomic_savez(out / "ranking.npz", est=est, keys=np.array([u.key for u in units]),
                       prompt_ids=np.array([p["id"] for p in sel]))
    # ranking.json is written from est on every run, not only when the passes ran: a torn or missing one (a kill in
    # its write) is repaired here instead of leaving a stage the scorer can never load (review r2 D-F1)
    _write(out / "ranking.json", {"file": "ranking.npz", "shape": list(est.shape), "dtype": "float32",
                                  "units": [u.key for u in units], "prompts": sel,
                                  "note": "est[i, j] = e for prompt i, part j (logit difference, consistent "
                                          "minus other, at the last position); zeroing a part changes it by "
                                          "about -e"})
    if args.stage_only == "ranking":
        return _finish(out, meta, n_pass)

    # ---- 2. scores: |e| normalised per prompt, mean over prompts, separately for SAY and DO
    scores = _score_sets(est, sel)
    say_set = top80(scores["say"], units, cfg["top_k"])
    do_set = top80(scores["do"], units, cfg["top_k"])
    sets = _load_sets(out / "sets.json")
    if sets is not None and (sets["say"] != say_set or sets["do"] != do_set):
        raise SystemExit(f"{out / 'sets.json'} does not match the ranking (sets are drawn once); refusing to mix")
    held = heldout_say(prompts)
    names = [n for n in prompts["personas"] if n != "You"]
    partner = {n: [m for m in names if m != n][0] for n in names}
    need_effect = sets is None and args.stage_only != "ranking" and getattr(args, "arms", "all") != "controls"
    random_arms = []
    if sets is not None:
        random_arms = [f"random:{fn}:{i}" for fn, f in sets["random"]["families"].items()
                       for i in range(len(f["sets"]))]
    # Amendment 2 item 2, the fallback (`--arms controls`): ranking, You, third-person and the main do-nothing arm only,
    # the do-nothing arm hooking the union of the DO and SAY sets; no effect, no random sets, no copy arm, no off-target,
    # no secondary arm. A sets.json from the other mode is refused, so the two never mix in one stage folder.
    controls = getattr(args, "arms", "all") == "controls"
    if sets is not None and (sets.get("arms") == "controls") != controls:
        raise SystemExit(f"{out / 'sets.json'} was written by a {'controls' if sets.get('arms') == 'controls' else 'full'}"
                         f" run; this run is {'controls' if controls else 'full'}: use a fresh --out")
    pair_arms = ["do_nothing"] if controls else ["do_nothing", "do_copy", "say_copy", "swap", "reverse"] + random_arms
    stop_after_pairs = args.stage_only in ("baseline", "pairs")

    def need_pair(qid, name):
        return (not stop_after_pairs) and (sets is None or any(missing(a, qid, name) for a in pair_arms))

    # ---- 3. You runs on the held-out questions, cached in memory (only where a missing record needs them)
    you = {}
    for h in held:
        text, ids_tok, q0 = frame("You", h["value"], h["question"])
        you[h["id"]] = {"text": text, "ids": ids_tok, "q0": q0}
        if not (missing("you", h["id"], "You", h["value"], "You") or need_effect
                or any(need_pair(h["id"], n) for n in names)):
            continue
        ids, read, acts, grads = run(text, need_grad=True, consistent=h["consistent"])
        assert ids == ids_tok, "runner and offset tokenisation disagree"
        you[h["id"]].update({"acts": {k: v.detach().cpu().clone() for k, v in acts.items()},
                             "grads": {k: v.detach().cpu().clone() for k, v in grads.items()}})
        emit({"arm": "you", "stage": stage, "value": h["value"], "item": h["item"], "id": h["id"],
              "name": "You", "persona": "You", "consistent": h["consistent"], "n_tokens": len(ids), **read})
    if all("acts" in v for v in you.values()):
        _save_acts(out / "copy_acts.npz", {q: v["acts"] for q, v in you.items()},
                   {q: v["grads"] for q, v in you.items()})

    # ---- 4. third-person baselines (with a backward pass)
    base = {}
    for h in held:
        for name in names:
            text, ids_tok, q0 = frame(name, h["value"], h["question"])
            base[(h["id"], name)] = {"text": text, "ids": ids_tok, "q0": q0}
            if not (missing("baseline", h["id"], name, h["value"], name) or need_effect
                    or need_pair(h["id"], name) or need_pair(h["id"], partner[name])):
                continue
            ids, read, acts, grads = run(text, need_grad=True, consistent=h["consistent"])
            assert ids == ids_tok, "runner and offset tokenisation disagree"
            base[(h["id"], name)].update({"read": read, "logits": runner.last_logits,
                                          "acts": {k: v.detach().cpu().clone() for k, v in acts.items()},
                                          "grads": {k: v.detach().cpu().clone() for k, v in grads.items()}})
            emit({"arm": "baseline", "stage": stage, "value": h["value"], "item": h["item"], "id": h["id"],
                  "name": name, "persona": name, "consistent": h["consistent"], "n_tokens": len(ids), **read})

    def span(src, dst):
        return C.copy_span(src["ids"], src["q0"], dst["ids"], dst["q0"])

    if need_effect:
        effect = _effect_per_part(units, held, names, you, base, span)
        C.atomic_savez(out / "effect.npz", effect=effect, keys=np.array([u.key for u in units]))
    else:
        effect = None
    if stop_after_pairs:
        # every You and third-person pass has run; the sets and copy arms are skipped (for the pilot's timing and SD(d))
        return _finish(out, meta, n_pass)

    # ---- 5. the sets: drawn once, after the baseline passes, before any replacement run
    if sets is None and controls:
        sets = {"say": say_set, "do": do_set, "random": {"families": {}}, "arms": "controls", "n_sets": 0,
                "note": "Amendment 2 item 2 (fallback): no random sets are drawn"}
        _write(out / "sets.json", sets)
    if sets is None:
        rngs = {k: random.Random(cfg["seeds"][f"random_family_{k}"]) for k in ("layer", "score", "effect")}
        n_sets = cfg["random_family"]["sets_per_stage_olmo"] if env["family"] == "O" \
            else cfg["random_family"]["sets_per_stage_qwen"]
        effect_by_key = {u.key: float(effect[i]) for i, u in enumerate(units)}
        random_fams = draw_random_families(rngs, units, do_set, say_set,
                                           {u.key: float(matching_score(scores)[i]) for i, u in enumerate(units)},
                                           effect_by_key, k=cfg["top_k"], n_sets=n_sets)
        sets = {"say": say_set, "do": do_set, "random": random_fams,
                "seeds": {k: cfg["seeds"][f"random_family_{k}"] for k in ("layer", "score", "effect")},
                "n_sets": n_sets, "effect": effect_by_key}
        _write(out / "sets.json", sets)
        random_arms = [f"random:{fn}:{i}" for fn, f in random_fams["families"].items() for i in range(len(f["sets"]))]
    random_sets = {f"random:{fn}:{i}": st for fn, f in sets["random"]["families"].items()
                   for i, st in enumerate(f["sets"])}
    # the do-nothing arm hooks exactly the parts the copy arms hook (their union) at the same positions (review r1
    # D-F5), so G1 tests the replacement path every copy uses
    copy_parts = sorted(set(do_set) | set(say_set) | {k for st in random_sets.values() for k in st})

    # ---- 6. the copy arms on the held-out questions
    def emit_arm(arm, h, name, part_keys, positions, read, ids, extra=None):
        emit({"arm": arm, "stage": stage, "value": h["value"], "item": h["item"], "id": h["id"], "name": name,
              "consistent": h["consistent"], "n_parts": len(part_keys), "n_positions": len(positions),
              "first_position": positions[0], "n_tokens": len(ids), **(extra or {}), **read})

    for h in held:
        y = you[h["id"]]
        for name in names:
            b = base[(h["id"], name)]
            src_pos, dst_pos = span(y, b)
            if missing("do_nothing", h["id"], name):
                vals = C.values_at(b["acts"], dst_pos)
                ids, read, _, _ = replace_and_read(b["text"], copy_parts, dst_pos, vals, h["consistent"])
                emit_arm("do_nothing", h, name, copy_parts, dst_pos, read, ids,
                         {"max_abs_logit_diff": C.max_abs_logit_diff(runner.last_logits, b["logits"]),
                          "parts": "union of the DO and SAY sets' parts (Amendment 2, fallback)" if controls
                          else "union of the copy arms' parts"})
            if controls:
                continue
            arms = [("do_copy", do_set), ("say_copy", say_set)] + [(a, random_sets[a]) for a in random_arms]
            vals = None
            for arm, part_keys in arms:
                if not missing(arm, h["id"], name):
                    continue
                vals = vals if vals is not None else C.values_at(y["acts"], src_pos)
                ids, read, _, _ = replace_and_read(b["text"], part_keys, dst_pos, vals, h["consistent"])
                emit_arm(arm, h, name, part_keys, dst_pos, read, ids)
            if missing("swap", h["id"], name):
                sb = base[(h["id"], partner[name])]
                s_src, s_dst = span(sb, b)
                ids, read, _, _ = replace_and_read(b["text"], do_set, s_dst, C.values_at(sb["acts"], s_src),
                                                   h["consistent"])
                emit_arm("swap", h, name, do_set, s_dst, read, ids, {"source": partner[name]})
            if missing("reverse", h["id"], name):
                # the DO parts from the third-person run copied into the You run: source positions in the
                # third-person run, receiving positions in the You run, each run's own (review r1 D-F2b)
                r_src, r_dst = span(b, y)
                ids, read, _, _ = replace_and_read(y["text"], do_set, r_dst, C.values_at(b["acts"], r_src),
                                                   h["consistent"])
                emit_arm("reverse", h, name, do_set, r_dst, read, ids, {"source": name, "receiver": "You"})

    if controls:
        return _finish(out, meta, n_pass)

    # ---- 7. off-target: value-free self-reports under value-free persona lines (review r1 H-F3)
    off_personas = prompts["off_target_personas"]
    for item in prompts["off_target"]:
        oh = {"value": None, "item": None, "id": item["id"], "consistent": "A"}
        off_arms = ("off_target_baseline", "off_target_do_nothing", "off_target_do_copy")
        if not missing("off_target_baseline", item["id"], "You", None, "You") and \
                not any(missing(a, item["id"], n, None, n if a == "off_target_baseline" else None)
                        for n in names for a in off_arms):
            continue
        yt, yids, yq0 = frame("You", None, item["question"], off_personas)
        yo = {"text": yt, "ids": yids, "q0": yq0}
        _, read, acts, _ = run(yt, need_grad=False, consistent="A")
        yo["acts"] = {k: v.detach().cpu().clone() for k, v in acts.items()}
        emit({"arm": "off_target_baseline", "stage": stage, "id": item["id"], "name": "You", "persona": "You",
              "persona_line": persona_text(prompts, "You", None, off_personas), "consistent": "A",
              "n_tokens": len(yids), **read})
        for name in names:
            t, tids, tq0 = frame(name, None, item["question"], off_personas)
            tb = {"text": t, "ids": tids, "q0": tq0}
            _, read, acts, _ = run(t, need_grad=False, capture=True, consistent="A")
            tb.update({"acts": {k: v.detach().cpu().clone() for k, v in acts.items()}, "logits": runner.last_logits})
            emit({"arm": "off_target_baseline", "stage": stage, "id": item["id"], "name": name, "persona": name,
                  "persona_line": persona_text(prompts, name, None, off_personas), "consistent": "A",
                  "n_tokens": len(tids), **read})
            src_pos, dst_pos = span(yo, tb)
            ids, read, _, _ = replace_and_read(t, do_set, dst_pos, C.values_at(tb["acts"], dst_pos), "A")
            emit_arm("off_target_do_nothing", oh, name, do_set, dst_pos, read, ids,
                     {"max_abs_logit_diff": C.max_abs_logit_diff(runner.last_logits, tb["logits"])})
            ids, read, _, _ = replace_and_read(t, do_set, dst_pos, C.values_at(yo["acts"], src_pos), "A")
            emit_arm("off_target_do_copy", oh, name, do_set, dst_pos, read, ids)

    # ---- 8. secondary arm, Sauers' wording (Design, lines 118-121)
    sec_parts = sorted(set(do_set) | set(say_set))
    for v in prompts["values"]:
        sh = {"value": v, "item": None, "id": f"secondary-{v}", "consistent": "A"}
        yq = prompts["secondary"]["you_question"].format(value=v)
        pq = prompts["secondary"]["third_person_question"].format(value=v)
        if missing("secondary_you", sh["id"], "You", v):
            text, _, _ = frame("You", v, yq)
            ids, read, _, _ = run(text, need_grad=False, capture=False, consistent="A")
            emit({"arm": "secondary_you", "stage": stage, "value": v, "id": sh["id"], "name": "You",
                  "question": yq, "consistent": "A", "n_tokens": len(ids), **read})
        sec_arms = ("secondary_baseline", "secondary_do_nothing", "secondary_do_copy", "secondary_say_copy")
        if not missing("secondary_you_source", sh["id"], "You", v) and \
                not any(missing(a, sh["id"], n, v) for n in names for a in sec_arms):
            continue
        # the copy source asks the SAME question as the receiving run (review r1 D-F3)
        stext, sids, sq0 = frame("You", v, pq)
        ys = {"text": stext, "ids": sids, "q0": sq0}
        _, read, acts, _ = run(stext, need_grad=False, consistent="A")
        ys["acts"] = {k: x.detach().cpu().clone() for k, x in acts.items()}
        emit({"arm": "secondary_you_source", "stage": stage, "value": v, "id": sh["id"], "name": "You",
              "question": pq, "consistent": "A", "n_tokens": len(sids), **read})
        for name in names:
            t, tids, tq0 = frame(name, v, pq)
            tb = {"text": t, "ids": tids, "q0": tq0}
            _, read, acts, _ = run(t, need_grad=False, capture=True, consistent="A")
            tb.update({"acts": {k: x.detach().cpu().clone() for k, x in acts.items()}, "logits": runner.last_logits})
            emit({"arm": "secondary_baseline", "stage": stage, "value": v, "id": sh["id"], "name": name,
                  "question": pq, "consistent": "A", "n_tokens": len(tids), **read})
            src_pos, dst_pos = span(ys, tb)
            ids, read, _, _ = replace_and_read(t, sec_parts, dst_pos, C.values_at(tb["acts"], dst_pos), "A")
            emit_arm("secondary_do_nothing", sh, name, sec_parts, dst_pos, read, ids,
                     {"max_abs_logit_diff": C.max_abs_logit_diff(runner.last_logits, tb["logits"])})
            vals = C.values_at(ys["acts"], src_pos)
            for arm, part_keys in (("secondary_do_copy", do_set), ("secondary_say_copy", say_set)):
                ids, read, _, _ = replace_and_read(t, part_keys, dst_pos, vals, "A")
                emit_arm(arm, sh, name, part_keys, dst_pos, read, ids)
    return _finish(out, meta, n_pass)


def _load_ranking(path, prompt_ids, unit_keys):
    """The stored estimates, or None (with a warning) when the file is torn, unreadable or not this stage's: the
    caller then recomputes the 48 ranking passes, which are cheap beside the rest of a stage (review r2 D-F1, H-N2)."""
    import numpy as np
    try:
        with np.load(path) as z:
            est = z["est"]
            if [str(x) for x in z["prompt_ids"]] == prompt_ids and [str(x) for x in z["keys"]] == unit_keys:
                return est
            print(f"warning: {path} does not match this stage's prompts or parts; recomputing the ranking",
                  file=sys.stderr)
    except Exception as e:                     # BadZipFile, EOFError, KeyError, ValueError: all mean "recompute"
        print(f"warning: {path} is unreadable ({type(e).__name__}: {e}); recomputing the ranking", file=sys.stderr)
    return None


def _load_sets(path):
    """The drawn sets, or None (with a warning) when the file is torn: the draw is seeded, so it is redrawn and
    rewritten rather than left to stop every later run (review r2 D-F2, H-N2). A file that parses but is not a set
    record is an error, not a torn write."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        sets = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"warning: {path} is unreadable ({e}); redrawing the sets (the draw is seeded)", file=sys.stderr)
        return None
    if not isinstance(sets, dict) or "say" not in sets or "do" not in sets or "random" not in sets:
        raise SystemExit(f"{path} is not a set record (no say/do/random); refusing to use it")
    return sets


def _score_sets(est, sel):
    """Per-prompt normalised |e|, mean over prompts, separately for SAY and DO and pooled."""
    import numpy as np
    per = np.zeros(est.shape, dtype=np.float64)
    for i in range(est.shape[0]):
        a = np.abs(est[i].astype(np.float64))
        s = a.sum()
        per[i] = a / s if s > 0 else 0.0
    kinds = np.array([p["kind"] for p in sel])
    return {"say": per[kinds == "SAY"].mean(axis=0), "do": per[kinds == "DO"].mean(axis=0), "pooled": per.mean(axis=0)}


def _effect_per_part(units, held, names, you, base, span):
    """First-order copy effect per part: sum over copied positions of g x (a_you - a_third), averaged over pairs.
    g and a_third are the third-person baseline's at its copied positions; a_you the You run's at its own."""
    import numpy as np
    acc = np.zeros(len(units), dtype=np.float64)
    n = 0
    by_site = {}
    for i, u in enumerate(units):
        by_site.setdefault(f"{u.site}:{u.layer}", []).append(i)
    for h in held:
        for name in names:
            b = base[(h["id"], name)]
            y = you[h["id"]]
            src_pos, dst_pos = span(y, b)
            vals = C.values_at(y["acts"], src_pos)
            for site, idxs in by_site.items():
                g = b["grads"][site][0, dst_pos, :].float().numpy()
                a_you = vals[site].float().numpy()
                a_third = b["acts"][site][0, dst_pos, :].float().numpy()
                prod = (g * (a_you - a_third)).sum(axis=0)
                w = units[idxs[0]].width
                seg = prod.reshape(-1, w).sum(axis=1)
                for i, s in zip(idxs, seg):
                    acc[i] += s
            n += 1
    return (acc / max(n, 1)).astype(np.float32)


def _save_acts(path, acts, grads):
    import numpy as np
    arrays = {}
    for qid, d in acts.items():
        for site, x in d.items():
            arrays[f"act|{qid}|{site}"] = x.numpy()
    for qid, d in grads.items():
        for site, x in d.items():
            arrays[f"grad|{qid}|{site}"] = x.numpy()
    C.atomic_savez(path, **arrays)


def _finish(out, meta, n_pass=None):
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    meta["passes_this_run"] = (n_pass or {}).get("run")
    _write(out / "meta.json", meta)
    print(f"stage {meta['env']['stage']} done -> {out}")
    return 0


def _write(path, obj):
    """Every JSON side file (meta, ranking, sets) goes through temp + os.replace, so a kill never leaves a torn one
    (review r2 D-F1, D-F2, D-F3, H-N2)."""
    C.atomic_write_json(path, obj)


def _rec_key(rec):
    """One record per (arm, question id, receiving name, persona). Every arm carries an id (the secondary arm's is
    "secondary-<value>"), so the value is not part of the key."""
    return "|".join(str(rec.get(k)) for k in ("arm", "id", "name", "persona"))


def _done_keys(path):
    """Keys of the records already written; a torn last line is ignored (C.read_jsonl)."""
    return {_rec_key(r) for r in C.read_jsonl(path)}


def _sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", required=True, help="a stage name from part205/config.json")
    ap.add_argument("--out", default=str(HERE.parent / "results" / "part205"))
    ap.add_argument("--config", default=str(HERE / "config.json"))
    ap.add_argument("--prompts", default=str(HERE / "prompts205.json"))
    ap.add_argument("--device", default=None, help="override the family's device (cpu / xpu)")
    ap.add_argument("--arms", choices=["all", "controls"], default="all",
                    help="controls: Amendment 2's fallback (ranking, You, third-person, do-nothing on DO and SAY's union)")
    ap.add_argument("--stage-only", choices=["ranking", "baseline", "pairs", "screen"], default=None,
                    help="stop after the ranking, the baselines, or the You/third-person pairs (timing, G0/G2, SD(d)); "
                         "'screen' is Amendment 2's step A0: only the 72 plain You and third-person passes")
    ap.add_argument("--check-pool", action="store_true",
                    help="refuse unless the unit pool matches config.json (off for tiny test models)")
    ap.add_argument("--models-json", default=str(HERE / "models.json"),
                    help="where to record this stage's revision and library versions (Design, line 58)")
    a = ap.parse_args(argv)
    return run_stage(a)


if __name__ == "__main__":
    sys.exit(main())
