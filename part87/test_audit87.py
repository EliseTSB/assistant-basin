"""Tests for audit87.py (Claude, 2026-10-28).

A small loop written here from PART87-DESIGN.md (not from part87.py) runs a tiny model with Qwen3.5's down_proj width
(9216, so the registered neuron indices exist) under the real observer and writes answers in the shape the audit reads.
The unmodified loop must PASS; each wiring mutation must FAIL with a message naming the property.
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "part86"))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402
from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402

import audit87  # noqa: E402
import observe  # noqa: E402

Q = audit87.registered_questions()
ALL_QUESTIONS = Q["origin"] + Q["maker"] + Q["guard"]
PROMPT_IDS = {q: [1, 3 + i % 60, 3 + (i * 7) % 60] for i, q in enumerate(ALL_QUESTIONS)}
MEDIANS = {6: 3.0, 7: 3.5, 11: 4.0, 22: 24.8, 29: 5.0}
MEANS = {(6, 753): 0.30, (7, 1169): -0.20, (11, 8868): 0.50, (29, 776): -0.40}


class TinyTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def decode(self, ids, skip_special_tokens=False):
        ids = [int(i) for i in ids]
        if skip_special_tokens:
            ids = [i for i in ids if i not in (0, 1, 2)]
        return " ".join(f"t{i}" for i in ids)


def tiny_model():
    torch.manual_seed(87087)
    config = Qwen2Config(vocab_size=64, hidden_size=16, intermediate_size=audit87.WIDTH, num_hidden_layers=30,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                         pad_token_id=0, bos_token_id=1, eos_token_id=2, use_cache=True)
    model = Qwen2ForCausalLM(config).to(dtype=torch.bfloat16).cpu().eval()
    model.generation_config.eos_token_id = 2
    model.generation_config.pad_token_id = 0
    return model


def norm(model, layer, unit):
    return float(torch.linalg.vector_norm(model.model.layers[layer].mlp.down_proj.weight.detach().float()[:, unit]))


def doses_doc(model, m=None):
    def rec(layer, unit, mean):
        n = norm(model, layer, unit)
        return {"layer": layer, "unit": unit, "mean": mean, "column_norm": n,
                "M": -mean * n / MEDIANS[layer], "F": -0.75 * (1 if mean > 0 else -1)}
    random_sets = []
    for seed in audit87.RANDOM_SEEDS:
        units = audit87.matched_random(seed)
        if m == "random_unit_swapped" and seed == 8700:
            units = [(units[0][0], units[0][1] + 1)] + units[1:]
        random_sets.append({"seed": seed, "units": [list(u) for u in units],
                            "doses": [rec(l, u, 0.1 * (1 + (u % 5)) * (-1) ** u) for l, u in units]})
    return {
        "residual_medians": {str(k): (20.0 if (m == "l22_median" and k == 22) else v) for k, v in MEDIANS.items()},
        "target_doses": {f"{l}:{u}": rec(l, u, MEANS[(l, u)]) for l, u in audit87.TARGETS},
        "random_sets": random_sets,
    }


class Hook:
    def __init__(self, module, unit, value=None, zero=False, decode_only=False, prompt_only=False):
        self.unit, self.value, self.zero = unit, value, zero
        self.decode_only, self.prompt_only = decode_only, prompt_only
        self.handle = module.register_forward_pre_hook(self.hook)

    def hook(self, _m, args):
        x = args[0]
        prefill = x.shape[-2] > 1
        if (prefill and self.decode_only) or (not prefill and self.prompt_only):
            return None
        y = x.clone()
        if self.zero:
            y[..., self.unit] = 0
        else:
            y[..., self.unit] = y[..., self.unit] + self.value
        return (y,)


def arm_hooks(model, arm, doc, m):
    """The registered intervention for one arm, written from the design."""
    layers = model.model.layers
    hooks = []
    if arm in ("E63", "E47", "D47", "C63"):
        dose = {"E63": 0.63, "E47": 0.47, "D47": 0.47, "C63": 0.63}[arm]
        if m == "negative" and arm == "E63":
            dose = -dose
        layer, unit = audit87.CONTROL if arm == "C63" else audit87.ALIEN
        if m == "control_pushes_alien" and arm == "C63":
            unit = audit87.ALIEN[1]
        hook_layer = 21 if (m == "layer_21" and arm == "E63") else layer
        if m == "no_push" and arm == "E63":
            return hooks
        decode_only = (arm == "D47") != (m == "mode_swap" and arm in ("E63", "D47"))
        hooks.append(Hook(layers[hook_layer].mlp.down_proj, unit, dose * MEDIANS[22] / norm(model, layer, unit),
                          decode_only=decode_only))
    elif arm == "Z":
        for layer, unit in audit87.TARGETS:
            hooks.append(Hook(layers[layer].mlp.down_proj, unit, zero=True, decode_only=(m == "z_decode_only")))
    elif arm in ("M", "F") or arm.startswith(("RM_", "RF_")):
        rule = "M" if arm in ("M",) or arm.startswith("RM_") else "F"
        if m == "rf_uses_m" and arm.startswith("RF_"):
            rule = "M"
        if arm in ("M", "F"):
            recs = [doc["target_doses"][f"{l}:{u}"] for l, u in audit87.TARGETS]
        else:
            seed = int(arm.split("_")[1])
            recs = next(i for i in doc["random_sets"] if i["seed"] == seed)["doses"]
        for rec in recs:
            dose = rec[rule]
            if (m == "m_sign" and arm == "M") or (m == "f_sign" and arm == "F"):
                dose = -dose
            value = dose * MEDIANS[rec["layer"]] / rec["column_norm"]
            hooks.append(Hook(layers[rec["layer"]].mlp.down_proj, rec["unit"], value))
    return hooks


def make_run(root: Path, phase: str, arm_list, m=None):
    model, tok = tiny_model(), TinyTokenizer()
    doc = doses_doc(model, m)
    (root / "doses.json").write_text(json.dumps(doc), encoding="utf-8")
    obs = observe.install(model, tok, root / "observe" / "run.jsonl", label="test 87")
    if m == "gen_config":
        model.generation_config.repetition_penalty = 1.8      # changed after the observer recorded the defaults
    for arm in arm_list:
        rows = []
        study_a = arm in ("A0", "E63", "E47", "D47", "C63")
        for question, sample in sorted(audit87.expected_cases(phase, arm, Q)):
            if study_a:
                seed = Q["seed_a"](Q["origin"].index(question), sample)
            else:
                qi = Q["maker"].index(question) if question in Q["maker"] else 25 + Q["guard"].index(question)
                seed = Q["seed_b"](f"{phase}:{arm}", qi, sample)
            if m == "seed_zero":
                seed = 0
            hooks = arm_hooks(model, arm, doc, m)
            prompt_q = question
            if m == "wrong_prompt" and arm == "U":
                prompt_q = Q["guard"][0] if question != Q["guard"][0] else Q["maker"][0]
            ids = torch.tensor([PROMPT_IDS[prompt_q]], dtype=torch.long)
            extra = {"top_k": 1} if m == "top_k" else {"min_new_tokens": 4} if m == "min_new" else {}
            down0 = model.model.layers[0].mlp.down_proj
            if m == "unobserved":
                down0.forward = torch.nn.Linear.forward.__get__(down0)
            torch.manual_seed(seed)
            with torch.inference_mode():
                out = model.generate(input_ids=ids, do_sample=True, temperature=1.0 if m == "temperature" else 0.7,
                                     top_p=0.95, max_new_tokens=audit87.A_TOKENS if study_a else audit87.B_TOKENS,
                                     **extra)
            if m == "unobserved":
                del down0.forward
            for h in hooks:
                h.handle.remove()
            gen = out[0, ids.shape[1]:].tolist()
            if 2 in gen:
                gen = gen[: gen.index(2) + 1]
            if study_a:
                qindex, qkind = Q["origin"].index(question), "origin"
            elif question in Q["maker"]:
                qindex = Q["maker"].index(question)
                qkind = "direct" if qindex < Q["n_direct"] else "indirect"
            else:
                qindex, qkind = 25 + Q["guard"].index(question), "guard"
            if m == "wrong_kind" and arm == "U" and not rows:
                qkind = "guard"
            rows.append({"arm": arm, "question": question, "question_index": qindex, "question_kind": qkind,
                         "sample": sample, "seed": seed,
                         "answer": tok.decode(gen, skip_special_tokens=True).strip(), "new_tokens": len(gen),
                         "observed": {"call_id": obs.last_call_id, "row": 0}})
        if m == "swap_answers" and len(rows) > 1:
            rows[0]["answer"], rows[1]["answer"] = rows[1]["answer"], rows[0]["answer"]
            rows[0]["new_tokens"], rows[1]["new_tokens"] = rows[1]["new_tokens"], rows[0]["new_tokens"]
        if m == "missing_answer" and arm == arm_list[0]:
            rows = rows[:-1]
        d = root / phase / arm
        d.mkdir(parents=True, exist_ok=True)
        (d / "answers.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    obs.close()
    texts = {q: tok.decode(ids) for q, ids in PROMPT_IDS.items()}
    (root / "prompts.json").write_text(json.dumps(texts), encoding="utf-8")


PILOT_ARMS = ["A0", "E63", "E47", "D47", "C63", "U", "Z", "M", "F"]
RANDOM_ARMS = ["RM_8700", "RF_8700"]


def audit(root: Path, phase: str, arm_list, fixture_model=True):
    argv = [str(root), "--phase", phase, "--prompt-texts", str(root / "prompts.json"), "--arms", ",".join(arm_list)]
    if fixture_model:
        argv.append("--fixture-model")
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = audit87.main(argv)
    return code, buf.getvalue()


class Audit87(unittest.TestCase):
    def run_case(self, m=None, phase="pilot", arm_list=PILOT_ARMS):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_run(root, phase, arm_list, m)
            return audit(root, phase, arm_list)

    def check_fails(self, m, needle, phase="pilot", arm_list=PILOT_ARMS):
        code, out = self.run_case(m, phase, arm_list)
        self.assertEqual(code, 1, f"{m} passed:\n{out}")
        self.assertIn(needle, out, f"{m}: expected {needle!r} in\n{out}")

    def test_registered_pilot_passes(self):
        code, out = self.run_case()
        self.assertEqual(code, 0, out)
        self.assertIn("TEST FIXTURE OVERRIDES", out)

    def test_registered_random_arms_pass(self):
        code, out = self.run_case(phase="run", arm_list=RANDOM_ARMS)
        self.assertEqual(code, 0, out)

    def test_mode_swap(self): self.check_fails("mode_swap", "changed in")
    def test_top_k(self): self.check_fails("top_k", "unregistered keywords")
    def test_min_new_tokens(self): self.check_fails("min_new", "unregistered keywords")
    def test_generation_config_changed(self): self.check_fails("gen_config", "must be the model's default")
    def test_wrong_kind(self): self.check_fails("wrong_kind", "stated question")
    def test_unobserved_forward(self): self.check_fails("unobserved", "did not see")

    def test_production_model_required_without_the_fixture_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_run(root, "pilot", ["A0"])
            code, out = audit(root, "pilot", ["A0"], fixture_model=False)
        self.assertEqual(code, 1, out)
        self.assertIn("registered Qwen3.5-4B", out)
    def test_negative(self): self.check_fails("negative", "registered upward")
    def test_layer_21(self): self.check_fails("layer_21", "changed layers")
    def test_control_pushes_alien(self): self.check_fails("control_pushes_alien", "registered only N8870")
    def test_z_decode_only(self): self.check_fails("z_decode_only", "changed in")
    def test_m_sign(self): self.check_fails("m_sign", "the M rule registers")
    def test_f_sign(self): self.check_fails("f_sign", "against the mean's")
    def test_temperature(self): self.check_fails("temperature", "temperature")
    def test_seed_zero(self): self.check_fails("seed_zero", "CPU seed at entry")
    def test_wrong_prompt(self): self.check_fails("wrong_prompt", "prompt rendered")
    def test_no_push(self): self.check_fails("no_push", "changed layers")
    def test_swap_answers(self): self.check_fails("swap_answers", "not the text the observer decoded")
    def test_l22_median(self): self.check_fails("l22_median", "neuronary publishes")
    def test_missing_answer(self): self.check_fails("missing_answer", "registered")
    def test_rf_uses_m(self): self.check_fails("rf_uses_m", "write norm", phase="run", arm_list=RANDOM_ARMS)
    def test_random_unit_swapped(self): self.check_fails("random_unit_swapped", "matched set", phase="run",
                                                         arm_list=RANDOM_ARMS)




class Round3Edges(unittest.TestCase):
    """Round 3 (#3): a correct push that rounds back in bf16, and D47 on a one-token answer (synthetic rows)."""

    def audit(self):
        a = audit87.Audit.__new__(audit87.Audit)
        a.fail, a.phase, a.worst, a.sessions, a.fixture_model = audit87.a86.Failures(), "pilot", 0.0, {}, True
        a.doses = {"residual_medians": {str(l): 10.0 for l in (6, 7, 11, 22, 29)},
                   "target_doses": {f"{l}:{u}": {"layer": l, "unit": u, "mean": 0.5, "column_norm": 1.0,
                                                 "M": -0.05, "F": -0.75} for l, u in audit87.TARGETS},
                   "column_norms": {"22": {"2742": 1.0, "8870": 1.0}}}
        return a

    @staticmethod
    def stats(unit, positions, value, write_norm=None):
        return {"changed_positions": positions, "sparse": True, "channels": {str(unit): [value, value, 1.0, 1.0]},
                "write_norm": [write_norm, write_norm] if write_norm is not None else None}

    def m_row(self, skip=None, partial=None, extra=False, too_many=False):
        layers = {}
        for l, u in audit87.TARGETS:
            if l == skip:
                continue
            pre = 1 if l == partial else (4 if too_many and l == 6 else 3)
            layers[str(l)] = {"prefill": self.stats(u, pre, -0.5), "decode": self.stats(u, 4, -0.5)}
        if extra:
            layers["5"] = {"prefill": self.stats(1, 3, -0.5)}
        return {"generated_steps": 5, "layers": layers}

    def check(self, arm, row, prompt_len=3):
        a = self.audit()
        a.check_change("t", {"prompt_len": prompt_len}, row, arm)
        return a.fail.items

    def test_m_layer_entirely_rounded_back_passes(self):
        self.assertEqual(self.check("M", self.m_row(skip=6)), [])

    def test_m_fewer_changed_positions_passes(self):
        self.assertEqual(self.check("M", self.m_row(partial=7)), [])

    def test_m_extra_layer_or_too_many_positions_still_fails(self):
        self.assertTrue(any("changed layers" in f for f in self.check("M", self.m_row(extra=True))))
        self.assertTrue(any("positions" in f for f in self.check("M", self.m_row(too_many=True))))

    def test_d47_one_token_answer_changes_nothing_and_passes(self):
        self.assertEqual(self.check("D47", {"generated_steps": 1, "layers": {}}), [])
        bad = {"generated_steps": 1, "layers": {"22": {"decode": self.stats(2742, 0, 2.0)}}}
        self.assertTrue(self.check("D47", bad))

    def test_e63_one_token_answer_must_still_change_the_prompt_pass(self):
        self.assertTrue(any("changed layers" in f for f in self.check("E63", {"generated_steps": 1, "layers": {}})))
        good = {"generated_steps": 1, "layers": {"22": {"prefill": self.stats(2742, 3, 6.3, 6.3)}}}
        self.assertEqual(self.check("E63", good), [])

    def test_f_is_not_relaxed(self):
        row = {"generated_steps": 5, "layers": {str(l): {"prefill": self.stats(u, 3, -7.5, 7.5),
                                                        "decode": self.stats(u, 4, -7.5, 7.5)}
                                               for l, u in audit87.TARGETS if l != 6}}
        self.assertTrue(any("changed layers" in f for f in self.check("F", row)))

if __name__ == "__main__":
    unittest.main(verbosity=1)
