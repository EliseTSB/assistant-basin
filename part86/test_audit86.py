"""Tests for audit86.py version 2 (Claude, 2026-10-28).

A small steering loop written here from PART86-DESIGN.md (not from screen86.py or verify86.py) runs a tiny model under
the real observer and writes answers in the shape the audit reads. The unmodified loop must PASS; each wiring mutation
must FAIL with a message naming the property. This tests the audit independently of the Part 86 build; the build's
own end-to-end test (test86.py) runs the real screen and verifier under the observer and this audit.
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
sys.path.insert(0, str(HERE))

import torch  # noqa: E402
from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402

import audit86  # noqa: E402
import observe  # noqa: E402

LAYER = 2
N_NEURONS = 4
N_RANDOM = 3
WIDTH = 64
TOKENS = 6
MEDIAN = 2.0
PROMPT_IDS = {"Who are you?": [1, 7, 9, 11], "Tell me about yourself.": [1, 8, 10], "What are you?": [1, 12, 13]}


class TinyTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def decode(self, ids, skip_special_tokens=False):
        ids = [int(i) for i in ids]
        if skip_special_tokens:
            ids = [i for i in ids if i not in (0, 1, 2)]
        return " ".join(f"t{i}" for i in ids)


def tiny_model():
    torch.manual_seed(86086)
    config = Qwen2Config(vocab_size=64, hidden_size=16, intermediate_size=WIDTH, num_hidden_layers=4,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                         pad_token_id=0, bos_token_id=1, eos_token_id=2, use_cache=True)
    model = Qwen2ForCausalLM(config).to(dtype=torch.bfloat16).cpu().eval()
    model.generation_config.eos_token_id = 2
    model.generation_config.pad_token_id = 0
    model.generation_config.min_new_tokens = 2   # a model default, so the audit's "unregistered = default" rule holds
    return model


class RowPush:
    def __init__(self, module, pushes, first_only=False, prompt_too=False, collateral=False):
        self.pushes, self.first_only, self.prompt_too, self.collateral = pushes, first_only, prompt_too, collateral
        self.decodes = 0
        self.handle = module.register_forward_pre_hook(self.hook)

    def hook(self, _m, args):
        x = args[0]
        if x.shape[-2] > 1 and not self.prompt_too:
            return None
        if x.shape[-2] == 1:
            self.decodes += 1
            if self.first_only and self.decodes > 1:
                return None
        if self.collateral:
            x[..., 0] = 0
        return (x + self.pushes.to(dtype=x.dtype).unsqueeze(1),)

    def remove(self):
        self.handle.remove()


def directions(seed, count, target, weight):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    raw = torch.randn((count, WIDTH), generator=g, dtype=torch.float32)
    raw = raw / torch.linalg.vector_norm(raw, dim=1, keepdim=True)
    writes = torch.linalg.vector_norm(raw @ weight.T, dim=1)
    return raw * (target / writes).unsqueeze(1)


def prompt_texts_file(root: Path):
    tok = TinyTokenizer()
    path = root / "prompts.json"
    path.write_text(json.dumps({q: tok.decode(ids) for q, ids in PROMPT_IDS.items()}), encoding="utf-8")
    return path


def generate(model, question, gen):
    ids = torch.tensor([PROMPT_IDS[gen.get("question_override", question)]] * gen["batch"], dtype=torch.long)
    kwargs = {"input_ids": ids, "max_new_tokens": gen.get("tokens", TOKENS), "do_sample": gen.get("do_sample", False)}
    if gen.get("repetition_penalty"):
        kwargs["repetition_penalty"] = gen["repetition_penalty"]
    if kwargs["do_sample"]:
        kwargs.update(temperature=gen.get("temperature", 0.7), top_p=0.95)
    with torch.inference_mode():
        return model.generate(**kwargs)


def answers_from(out, prompt_len, tok):
    result = []
    for row in range(out.shape[0]):
        ids = out[row, prompt_len:].tolist()
        if 2 in ids:
            ids = ids[: ids.index(2) + 1]
        result.append((tok.decode(ids, skip_special_tokens=True).strip(), len(ids)))
    return result


def make_screen(root: Path, m: str | None = None):
    """The registered screen for one layer, scaled down: every neuron and random direction in one batch."""
    model, tok = tiny_model(), TinyTokenizer()
    obs = observe.install(model, tok, root / "observe" / "screen.jsonl", label="test screen")
    layer = LAYER - 1 if m == "previous_layer" else LAYER
    down = model.model.layers[layer].mlp.down_proj
    weight = model.model.layers[LAYER].mlp.down_proj.weight.detach().float()
    dose = 0.4 if m == "half_dose" else audit86.DOSE
    target = dose * MEDIAN
    pushes = torch.zeros((N_NEURONS + N_RANDOM, WIDTH))
    for n in range(N_NEURONS):
        unit = (n + 1) % N_NEURONS if m == "shifted_unit" else n
        sign = -1.0 if m == "negative" else 1.0
        pushes[n, unit] = sign * target / float(torch.linalg.vector_norm(weight[:, unit]))
    seed = audit86.SCREEN_SEED + LAYER + (1 if m == "random_seed" else 0)
    pushes[N_NEURONS:] = directions(seed, N_RANDOM, target, weight)
    push = RowPush(down, pushes, first_only=(m == "first_decode_only"), prompt_too=(m == "prompt_too"),
                   collateral=(m == "collateral"))
    extra = model.model.layers[0].self_attn.register_forward_hook(lambda *a: None) if m == "extra_hook" else None
    gen = {"batch": N_NEURONS + N_RANDOM, "do_sample": m == "sampling", "tokens": TOKENS + (1 if m == "tokens" else 0),
           "repetition_penalty": 1.2 if m == "repetition_penalty" else None}
    if m == "hook_mid_call":
        def add_late_hook(module, args, state={"done": False}):
            if not state["done"] and args[0].shape[-2] == 1:
                state["done"] = True
                model.lm_head.register_forward_hook(lambda *a: None)
        model.model.layers[0].register_forward_pre_hook(add_late_hook)
    if m == "wrong_question":
        gen["question_override"] = "What are you?"
    out = generate(model, "Who are you?", gen)
    call_id = obs.last_call_id
    push.remove()
    if extra:
        extra.remove()
    obs.close()
    texts = answers_from(out, 4, tok)
    rows = []
    for r in range(N_NEURONS + N_RANDOM):
        kind, unit = ("neuron", r) if r < N_NEURONS else ("random_direction", r - N_NEURONS)
        answer, count = texts[r]
        rows.append({"layer": LAYER, "kind": kind, "id": unit, "answer": answer, "token_count": count,
                     "median": MEDIAN, "dose": audit86.DOSE, "observed": {"call_id": call_id, "row": r}})
    if m == "reversed_answers":
        answer_fields = [(row["answer"], row["token_count"]) for row in rows][::-1]
        for row, (a, c) in zip(rows, answer_fields):
            row["answer"], row["token_count"] = a, c
    if m == "missing_answer":
        rows = rows[:-1]
    if m == "duplicate_claim":
        rows[1]["observed"] = dict(rows[0]["observed"])
    (root / f"layer_{LAYER}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def make_verify(root: Path, m: str | None = None):
    """One picked neuron: 3 questions x 8 samples, plus 10 random directions x 24, one answer per call."""
    model, tok = tiny_model(), TinyTokenizer()
    obs = observe.install(model, tok, root / "observe" / "verify.jsonl", label="test verify")
    neuron = 1
    weight = model.model.layers[LAYER].mlp.down_proj.weight.detach().float()
    layer = LAYER - 1 if m == "previous_layer" else LAYER
    target = audit86.DOSE * MEDIAN
    rseed = audit86.VERIFY_RANDOM_SEED + LAYER * 10000 + neuron + (1 if m == "random_seed" else 0)
    dirs = directions(rseed, audit86.VERIFY_RANDOM, target, weight)
    neuron_push = torch.zeros((1, WIDTH))
    neuron_push[0, neuron] = target / float(torch.linalg.vector_norm(weight[:, neuron]))
    records = []
    cases = [("neuron", None)] + [("random_direction", d) for d in range(audit86.VERIFY_RANDOM)]
    for condition, d in cases:
        for qi, question in enumerate(audit86.VERIFY_QUESTIONS):
            for s in range(audit86.VERIFY_SAMPLES):
                seed = audit86.VERIFY_SEED + 100 * qi + s + (1 if m == "seed" else 0)
                vec = neuron_push if condition == "neuron" else dirs[d:d + 1]
                push = RowPush(model.model.layers[layer].mlp.down_proj, vec)
                torch.manual_seed(seed)
                if m == "draw":
                    torch.rand(1)
                gen = {"batch": 1, "do_sample": True, "tokens": audit86.VERIFY_TOKENS,
                       "temperature": 1.0 if m == "temperature" else 0.7}
                if m == "wrong_question":
                    gen["question_override"] = audit86.VERIFY_QUESTIONS[0]
                out = generate(model, question, gen)
                push.remove()
                answer, count = answers_from(out, len(PROMPT_IDS[gen.get("question_override", question)]), tok)[0]
                records.append({"layer": LAYER, "neuron": neuron, "condition": condition, "direction": d,
                                "question_index": qi, "sample": s, "answer": answer, "token_count": count,
                                "median": MEDIAN, "dose": audit86.DOSE,
                                "observed": {"call_id": obs.last_call_id, "row": 0}})
    obs.close()
    (root / f"L{LAYER}_N{neuron}.json").write_text(json.dumps({"status": "complete", "records": records}),
                                                     encoding="utf-8")


def run_audit(cmd, root: Path):
    prompts = prompt_texts_file(root)
    argv = [cmd, str(root), "--width", str(WIDTH), "--prompt-texts", str(prompts)]
    if cmd == "screen":
        argv += ["--layers", str(LAYER), "--neurons", str(N_NEURONS), "--random", str(N_RANDOM), "--tokens", str(TOKENS)]
    else:
        argv += ["--picked", "1"]
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = audit86.main(argv)
    return code, buf.getvalue()


class AuditScreen(unittest.TestCase):
    def audit(self, mutation=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_screen(root, mutation)
            return run_audit("screen", root)

    def test_registered_screen_passes(self):
        code, out = self.audit()
        self.assertEqual(code, 0, out)
        self.assertIn("PASS", out)
        self.assertIn("TEST FIXTURE OVERRIDES", out, "fixture runs must not print the bare gate line")

    def check_fails(self, mutation, needle):
        code, out = self.audit(mutation)
        self.assertEqual(code, 1, f"{mutation} passed:\n{out}")
        self.assertIn(needle, out, f"{mutation}: expected {needle!r} in\n{out}")

    def test_negative_push(self): self.check_fails("negative", "the registered push is upward")
    def test_random_seed(self): self.check_fails("random_seed", "cosine")
    def test_reversed_answers(self): self.check_fails("reversed_answers", "not the text the observer decoded")
    def test_previous_layer(self): self.check_fails("previous_layer", "changed layers")
    def test_first_decode_only(self): self.check_fails("first_decode_only", "decode calls")
    def test_sampling(self): self.check_fails("sampling", "do_sample")
    def test_wrong_question(self): self.check_fails("wrong_question", "prompt rendered")
    def test_extra_hook(self): self.check_fails("extra_hook", "hooks at generate entry")
    def test_half_dose(self): self.check_fails("half_dose", "write norm")
    def test_shifted_unit(self): self.check_fails("shifted_unit", "registered only neuron")
    def test_prompt_too(self): self.check_fails("prompt_too", "prompt pass was changed")
    def test_missing_answer(self): self.check_fails("missing_answer", "registered exactly ids")
    def test_duplicate_claim(self): self.check_fails("duplicate_claim", "already claimed")
    def test_collateral(self): self.check_fails("collateral", "registered only neuron")
    def test_tokens(self): self.check_fails("tokens", "max_new_tokens")
    def test_repetition_penalty(self): self.check_fails("repetition_penalty", "must be the model's default")
    def test_hook_mid_call(self): self.check_fails("hook_mid_call", "hooks")

    def test_foreign_observer_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_screen(root)
            path = root / "observe" / "screen.jsonl"
            lines = path.read_text(encoding="utf-8").splitlines()
            head = json.loads(lines[0])
            head["observer_sha256"] = "0" * 64
            path.write_text("\n".join([json.dumps(head)] + lines[1:]) + "\n", encoding="utf-8")
            code, out = run_audit("screen", root)
        self.assertEqual(code, 1)
        self.assertIn("observer sha", out)


class AuditVerify(unittest.TestCase):
    def audit(self, mutation=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_verify(root, mutation)
            return run_audit("verify", root)

    def check_fails(self, mutation, needle):
        code, out = self.audit(mutation)
        self.assertEqual(code, 1, f"{mutation} passed:\n{out}")
        self.assertIn(needle, out, f"{mutation}: expected {needle!r} in\n{out}")

    def test_registered_verification_passes(self):
        code, out = self.audit()
        self.assertEqual(code, 0, out)

    def test_seed(self): self.check_fails("seed", "CPU seed at entry")
    def test_draw_between_seed_and_generate(self): self.check_fails("draw", "drawn between seeding")
    def test_temperature(self): self.check_fails("temperature", "temperature")
    def test_wrong_question(self): self.check_fails("wrong_question", "prompt rendered")
    def test_random_seed(self): self.check_fails("random_seed", "cosine")
    def test_previous_layer(self): self.check_fails("previous_layer", "changed layers")


class Prompt(unittest.TestCase):
    def test_expected_prompt_matches_the_real_tokenizer_when_present(self):
        snap = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots"
        dirs = sorted(p for p in snap.glob("851bf6e8*")) if snap.is_dir() else []
        if not dirs:
            self.skipTest("Qwen3.5-4B revision 851bf6e8 tokenizer not cached here")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(dirs[0]))
        for q in audit86.VERIFY_QUESTIONS:
            rendered = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
            self.assertEqual(rendered, audit86.expected_prompt(q))


if __name__ == "__main__":
    unittest.main(verbosity=1)
