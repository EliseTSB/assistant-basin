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
  do_nothing  the same replacement code with the receiving run's own activations (G1)
  copy:you    DO set / SAY set / random set copied from the You run, at every suffix position
  copy:<name> persona-swap null: the DO set copied from the other name's third-person run
  reverse     DO set copied from the third-person run into the You run
  off_target_do_copy  DO-set copy on the six value-free self-reports
  secondary_* Sauers' wording: You asks the plain question, both names ask "personally"
  g1_check    the A770 check: one question 5 times plain and 5 times with the do-nothing hook (family Q only)

Resume: records.jsonl is appended and a record whose key is already present is skipped, so a stopped run resumes.
The sets are drawn and written before any replacement run (Design, Controls, Seeds).
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common205 as C  # noqa: E402


# ---------------------------------------------------------------- model loading

def load_stage(cfg, stage, device_override=None):
    """(tokenizer, model, env record) for one stage. One model per process."""
    st = next((s for s in cfg["stages"] if s["stage"] == stage), None)
    if st is None:
        raise SystemExit(f"unknown stage {stage!r}; config has {[s['stage'] for s in cfg['stages']]}")
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

        Returns (ids, readout, acts, grads). With need_grad the backward pass is on the logit difference (sum of the
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

    def replace_and_read(self, text, part_keys, positions, values_by_site, consistent):
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
        return self.run(text, need_grad=False, capture=False, replace=replace, consistent=consistent)

    @property
    def _by_key(self):
        if not hasattr(self, "_keys"):
            self._keys = {u.key: u for u in self.units}
        return self._keys


# ---------------------------------------------------------------- prompt construction

def build_prompt(prompts, persona_name, value, question):
    persona = C.persona_line(prompts["personas"], persona_name, value)
    return C.frame_text(persona, question)


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


def draw_random_families(rngs, units, do_set, say_set, score, effect, k=80, n_sets=20):
    """The three matched random families (Design, Controls, lines 131-143).

    Every set is a seeded draw of k distinct parts without replacement from the pool minus the SAY and DO sets, with
    the DO set's kind counts (for Q: full-attention heads, DeltaNet heads, neurons).
    (i) layer: the DO set's count per layer and kind, nearest layer of the same kind when a layer is short.
    (ii) score: per kind, the DO set's count in each decile of selection score, deciles within kind over the pool.
    (iii) effect: as (ii) with deciles of first-order copy effect.
    In (ii) and (iii) a short decile takes the rest from the nearest decile. Every relaxed part is counted and
    recorded; a family that cannot meet its kind counts is "not fillable".
    """
    excluded = set(do_set) | set(say_set)
    pool = [u for u in units if u.key not in excluded]
    kinds = sorted({kind_key(u) for u in units})
    by_key = {u.key: u for u in units}
    do_units = [by_key[k] for k in do_set]
    want = {kd: sum(kind_key(u) == kd for u in do_units) for kd in kinds}
    have = {kd: sum(kind_key(u) == kd for u in pool) for kd in kinds}
    out = {"fillable": all(want[kd] <= have[kd] for kd in kinds), "kind_counts_wanted": want,
           "kind_counts_eligible": have, "n_sets": n_sets, "set_size": k, "families": {}}
    if not out["fillable"]:
        out["not_fillable_reason"] = {kd: f"want {want[kd]}, eligible {have[kd]}" for kd in kinds
                                      if want[kd] > have[kd]}
        return out
    # (i) layer-matched
    fam = {"sets": [], "relaxed": 0, "relaxed_detail": []}
    for s in range(n_sets):
        rng = rngs["layer"]
        chosen, taken = [], set()
        for kd in kinds:
            for layer in sorted({u.layer for u in do_units if kind_key(u) == kd}):
                need = sum(kind_key(u) == kd and u.layer == layer for u in do_units)
                elig = [u for u in pool if kind_key(u) == kd and u.layer == layer and u.key not in taken]
                pick = rng.sample(elig, need) if len(elig) >= need else list(elig)
                if len(pick) < need:
                    short = need - len(pick)
                    fam["relaxed"] += short
                    fam["relaxed_detail"].append({"set": s, "kind": kd, "layer": layer, "short": short})
                    others = [u for u in pool if kind_key(u) == kd and u.key not in taken
                              and u.key not in {p.key for p in pick}]
                    others.sort(key=lambda u: (abs(u.layer - layer), u.layer, u.index))
                    pick += others[:short]
                chosen += pick
                taken |= {u.key for u in pick}
        fam["sets"].append([u.key for u in chosen])
    out["families"]["layer"] = fam
    # (ii) score-matched and (iii) effect-matched
    for name, dec in (("score", _deciles(units, kinds, score)), ("effect", _deciles(units, kinds, effect))):
        fam = {"sets": [], "relaxed": 0, "relaxed_detail": []}
        for s in range(n_sets):
            rng = rngs[name]
            chosen, taken = [], set()
            for kd in kinds:
                want_by_dec = {}
                for u in do_units:
                    if kind_key(u) == kd:
                        want_by_dec[dec[u.key]] = want_by_dec.get(dec[u.key], 0) + 1
                for d in sorted(want_by_dec):
                    need = want_by_dec[d]
                    elig = [u for u in pool if kind_key(u) == kd and dec[u.key] == d and u.key not in taken]
                    pick = rng.sample(elig, need) if len(elig) >= need else list(elig)
                    if len(pick) < need:
                        short = need - len(pick)
                        fam["relaxed"] += short
                        fam["relaxed_detail"].append({"set": s, "kind": kd, "decile": d, "short": short})
                        others = [u for u in pool if kind_key(u) == kd and u.key not in taken
                                  and u.key not in {p.key for p in pick}]
                        others.sort(key=lambda u: (abs(dec[u.key] - d), u.key))
                        pick += others[:short]
                    chosen += pick
                    taken |= {u.key for u in pick}
            fam["sets"].append([u.key for u in chosen])
        out["families"][name] = fam
    return out


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
    """The whole stage. `model`/`tok` may be passed by a test (a tiny model) instead of being loaded."""
    import numpy as np
    cfg = cfg or C.load_config(args.config)
    prompts = prompts or C.load_prompts(args.prompts)
    stage = args.stage
    out = Path(args.out) / stage
    out.mkdir(parents=True, exist_ok=True)
    if model is None:
        tok, model, env = load_stage(cfg, stage, args.device)
    units = units or C.units_for(model)
    fam = cfg["families"][next(s["family"] for s in cfg["stages"] if s["stage"] == stage)]
    expected = fam["units"] if args.check_pool else None
    pool = C.check_pool(units, expected)
    letter_ids = {"A": C.option_token_ids(tok, "A"), "B": C.option_token_ids(tok, "B")}
    runner = Runner(model, tok, units, letter_ids)
    meta = {"env": env, "pool": pool, "letter_ids": letter_ids, "units": [u.key for u in units],
            "unit_meta": [u.as_dict() for u in units], "prompts_sha": _sha(args.prompts), "config_sha": _sha(args.config),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    _write(out / "meta.json", meta)
    rec_path = out / "records.jsonl"
    done = _done_keys(rec_path)

    def emit(rec):
        key = _rec_key(rec)
        if key in done:
            return
        with rec_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        done.add(key)

    # ---- 1. ranking: 48 forward+backward passes in the You frame
    sel = selection_prompts(prompts)
    random.Random(cfg["seeds"]["prompt_order"]).shuffle(sel)  # prompt order seeded (Design, Controls, Seeds)
    est = np.zeros((len(sel), len(units)), dtype=np.float32)
    for i, p in enumerate(sel):
        text = build_prompt(prompts, "You", p["value"], p["question"])
        ids, read, acts, grads = runner.run(text, need_grad=True, consistent=p["consistent"])
        est[i] = C.estimates_for_prompt(acts, grads, units).numpy().astype(np.float32)
        emit({"arm": "ranking", "stage": stage, "value": p["value"], "kind": p["kind"], "item": p["item"],
              "id": p["id"], "persona": "You", "consistent": p["consistent"], "n_tokens": len(ids), **read})
    np.savez_compressed(out / "ranking.npz", est=est, keys=np.array([u.key for u in units]),
                        prompt_ids=np.array([p["id"] for p in sel]))
    _write(out / "ranking.json", {"file": "ranking.npz", "shape": list(est.shape), "dtype": "float32",
                                  "units": [u.key for u in units], "prompts": sel,
                                  "note": "est[i, j] = e for prompt i, part j (logit difference, consistent minus "
                                          "other, at the last position); zeroing a part changes it by about -e"})
    if args.stage_only == "ranking":
        return _finish(out, meta)

    # ---- 2. scores: |e| normalised per prompt, mean over prompts, separately for SAY and DO
    scores = _score_sets(est, sel)
    say_set = top80(scores["say"], units, cfg["top_k"])
    do_set = top80(scores["do"], units, cfg["top_k"])
    by_key = {u.key: u for u in units}

    # ---- 3. You runs on the held-out questions, cached
    held = heldout_say(prompts)
    you_acts, you_grads, you_ids = {}, {}, {}
    for h in held:
        text = build_prompt(prompts, "You", h["value"], h["question"])
        ids, read, acts, grads = runner.run(text, need_grad=True, consistent=h["consistent"])
        you_acts[h["id"]] = {k: v.detach().cpu().clone() for k, v in acts.items()}
        you_grads[h["id"]] = {k: v.detach().cpu().clone() for k, v in grads.items()}
        you_ids[h["id"]] = ids
        emit({"arm": "you", "stage": stage, "value": h["value"], "item": h["item"], "id": h["id"],
              "persona": "You", "consistent": h["consistent"], "n_tokens": len(ids), **read})
    _save_acts(out / "copy_acts.npz", you_acts, you_grads)

    # ---- 4. third-person baselines (with a backward pass)
    names = [n for n in prompts["personas"] if n != "You"]
    base = {}
    for h in held:
        for name in names:
            text = build_prompt(prompts, name, h["value"], h["question"])
            ids, read, acts, grads = runner.run(text, need_grad=True, consistent=h["consistent"])
            base[(h["id"], name)] = {"ids": ids, "read": read,
                                     "acts": {k: v.detach().cpu().clone() for k, v in acts.items()},
                                     "grads": {k: v.detach().cpu().clone() for k, v in grads.items()}}
            emit({"arm": "baseline", "stage": stage, "value": h["value"], "item": h["item"], "id": h["id"],
                  "name": name, "persona": name, "consistent": h["consistent"], "n_tokens": len(ids), **read})
            # the do-nothing arm is this same replacement code with the receiving run's own activations (G1)
            _, positions, vals = C.you_values_at_third_positions(
                base[(h["id"], name)]["acts"], None, ids,
                suffix_len=len(C.shared_suffix_positions(you_ids[h["id"]], ids)[1]))
            ids_dn, read_dn, _, _ = runner.replace_and_read(text, [u.key for u in units], positions, vals,
                                                            h["consistent"])
            emit({"arm": "do_nothing", "stage": stage, "value": h["value"], "item": h["item"], "id": h["id"],
                  "name": name, "consistent": h["consistent"], "n_parts": len(units),
                  "n_positions": len(positions), "n_tokens": len(ids_dn), **read_dn})
    suffix_len = {(h["id"], name): len(C.shared_suffix_positions(you_ids[h["id"]], base[(h["id"], name)]["ids"])[1])
                  for h in held for name in names}
    effect = _effect_per_part(units, held, names, you_acts, base, suffix_len)
    np.savez_compressed(out / "effect.npz", effect=effect, keys=np.array([u.key for u in units]))
    if args.stage_only == "baseline":
        return _finish(out, meta)
    if args.stage_only == "pairs":
        # every You and third-person pass has run; the sets and copy arms are skipped (for the pilot's timing and SD(d))
        return _finish(out, meta)

    # ---- 5. the sets: drawn once, after the baseline passes, before any replacement run
    rngs = {k: random.Random(cfg["seeds"][f"random_family_{k}"]) for k in ("layer", "score", "effect")}
    n_sets = cfg["random_family"]["sets_per_stage_olmo"] if env["family"] == "O" \
        else cfg["random_family"]["sets_per_stage_qwen"]
    random_fams = draw_random_families(rngs, units, do_set, say_set, scores["pooled"],
                                       {u.key: float(effect[i]) for i, u in enumerate(units)},
                                       k=cfg["top_k"], n_sets=n_sets)
    sets = {"say": say_set, "do": do_set, "random": random_fams,
            "seeds": {k: cfg["seeds"][f"random_family_{k}"] for k in ("layer", "score", "effect")},
            "n_sets": n_sets, "effect": {u.key: float(effect[i]) for i, u in enumerate(units)}}
    _write(out / "sets.json", sets)

    # ---- 6. the copy arms on the held-out questions
    def copy_arm(arm, part_keys, src_acts, h, name, b, consistent):
        _, positions, vals = C.you_values_at_third_positions(src_acts, None, b["ids"],
                                                             suffix_len=suffix_len[(h["id"], name)])
        text = build_prompt(prompts, name, h["value"], h["question"])
        ids, read, _, _ = runner.replace_and_read(text, part_keys, positions, vals, consistent)
        emit({"arm": arm, "stage": stage, "value": h["value"], "item": h["item"], "id": h["id"], "name": name,
              "consistent": consistent, "n_parts": len(part_keys), "n_positions": len(positions),
              "n_tokens": len(ids), **read})

    for h in held:
        for name in names:
            b = base[(h["id"], name)]
            swap = [n for n in names if n != name][0]
            sb = base[(h["id"], swap)]
            copy_arm("do_copy", do_set, you_acts[h["id"]], h, name, b, h["consistent"])
            copy_arm("say_copy", say_set, you_acts[h["id"]], h, name, b, h["consistent"])
            copy_arm("swap", do_set, sb["acts"], h, name, b, h["consistent"])
            for fam_name, fam_sets in random_fams["families"].items():
                for si, s in enumerate(fam_sets["sets"]):
                    copy_arm(f"random:{fam_name}:{si}", s, you_acts[h["id"]], h, name, b, h["consistent"])
            # reverse: the DO parts from the third-person run copied into the You run, at the You run's suffix
            _, positions, vals = C.you_values_at_third_positions(b["acts"], b["ids"], you_ids[h["id"]])
            text = build_prompt(prompts, "You", h["value"], h["question"])
            ids, read, _, _ = runner.replace_and_read(text, do_set, positions, vals, h["consistent"])
            emit({"arm": "reverse", "stage": stage, "value": h["value"], "item": h["item"], "id": h["id"],
                  "name": name, "consistent": h["consistent"], "n_parts": len(do_set),
                  "n_positions": len(positions), "n_tokens": len(ids), **read})

    # ---- 7. off-target: DO-set copy on the six value-free self-reports
    for item in prompts["off_target"]:
        you_text = build_prompt(prompts, "You", None, item["question"])
        you_ids_off, read, acts, _ = runner.run(you_text, need_grad=False, consistent="A")
        emit({"arm": "off_target_baseline", "stage": stage, "id": item["id"], "name": "You", "persona": "You",
              "consistent": "A", "n_tokens": len(you_ids_off), **read})
        you_off = {k: v.detach().cpu().clone() for k, v in acts.items()}
        for name in names:
            text = build_prompt(prompts, name, None, item["question"])
            ids, read, _, _ = runner.run(text, need_grad=False, capture=False, consistent="A")
            emit({"arm": "off_target_baseline", "stage": stage, "id": item["id"], "name": name, "persona": name,
                  "consistent": "A", "n_tokens": len(ids), **read})
            _, positions, vals = C.you_values_at_third_positions(you_off, you_ids_off, ids)
            ids, read, _, _ = runner.replace_and_read(text, do_set, positions, vals, "A")
            emit({"arm": "off_target_do_copy", "stage": stage, "id": item["id"], "name": name,
                  "consistent": "A", "n_parts": len(do_set), "n_positions": len(positions),
                  "n_tokens": len(ids), **read})

    # ---- 8. secondary arm, Sauers' wording
    for v in prompts["values"]:
        yq = prompts["secondary"]["you_question"].format(value=v)
        text = build_prompt(prompts, "You", v, yq)
        ids, read, acts, _ = runner.run(text, need_grad=False, consistent="A")
        emit({"arm": "secondary_you", "stage": stage, "value": v, "name": "You", "consistent": "A",
              "n_tokens": len(ids), **read})
        you_sec = {k: x.detach().cpu().clone() for k, x in acts.items()}
        for name in names:
            pq = prompts["secondary"]["third_person_question"].format(value=v)
            ptext = build_prompt(prompts, name, v, pq)
            ids2, read2, _, _ = runner.run(ptext, need_grad=False, capture=False, consistent="A")
            emit({"arm": "secondary_baseline", "stage": stage, "value": v, "name": name, "consistent": "A",
                  "n_tokens": len(ids2), **read2})
            _, positions, vals = C.you_values_at_third_positions(you_sec, ids, ids2)
            for arm, part_keys in (("secondary_do_copy", do_set), ("secondary_say_copy", say_set)):
                ids3, read3, _, _ = runner.replace_and_read(ptext, part_keys, positions, vals, "A")
                emit({"arm": arm, "stage": stage, "value": v, "name": name, "consistent": "A",
                      "n_parts": len(part_keys), "n_positions": len(positions), "n_tokens": len(ids3), **read3})
    return _finish(out, meta)


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


def _effect_per_part(units, held, names, you_acts, base, suffix_len):
    """First-order copy effect per part: sum over copied positions of g x (a_you - a_third), averaged over pairs."""
    import numpy as np
    acc = np.zeros(len(units), dtype=np.float64)
    n = 0
    by_site = {}
    for i, u in enumerate(units):
        by_site.setdefault(f"{u.site}:{u.layer}", []).append(i)
    for h in held:
        for name in names:
            b = base[(h["id"], name)]
            _, positions, vals = C.you_values_at_third_positions(you_acts[h["id"]], None, b["ids"],
                                                                 suffix_len=suffix_len[(h["id"], name)])
            for site, idxs in by_site.items():
                g = b["grads"][site][0, positions, :].numpy()
                a_you = vals[site].numpy()
                a_third = b["acts"][site][0, positions, :].numpy()
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
    np.savez_compressed(path, **arrays)


def _finish(out, meta):
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _write(out / "meta.json", meta)
    print(f"stage {meta['env']['stage']} done -> {out}")
    return 0


def _write(path, obj):
    path.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def _rec_key(rec):
    return "|".join(str(rec.get(k)) for k in ("arm", "id", "name", "value", "persona"))


def _done_keys(path):
    keys = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                keys.add(_rec_key(json.loads(line)))
    return keys


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
    ap.add_argument("--stage-only", choices=["ranking", "baseline", "pairs"], default=None,
                    help="stop after the ranking, the baselines, or the You/third-person pairs (timing, G0/G2, SD(d))")
    ap.add_argument("--check-pool", action="store_true",
                    help="refuse unless the unit pool matches config.json (off for tiny test models)")
    a = ap.parse_args(argv)
    return run_stage(a)


if __name__ == "__main__":
    sys.exit(main())
