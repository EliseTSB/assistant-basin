"""Tests for replay85.py on a tiny random OLMo 2 model (CPU) with the real OLMo 2 tokenizer.

The fixture writes a fake run, shaped like the observer's records (FOLD5-NOTES.md, and the real step-1 forward call
Sol's round 6 recorded), from replay85's own regeneration, so these tests show that the replay passes a faithful run
and FAILS on each kind of wrong run Sol's rounds 5 and 6 used. They cannot show that it agrees with part85.py; only the
pilot's replay can. Each faithful run is generated once per shape and copied for the broken variants.
Run: python -m pytest part85/test_replay85.py
"""
import copy
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay85 as R  # noqa: E402

TOK = os.environ.get("PART85_TOKENIZER", "/mnt/e/neuron-probe/hf/hub/models--allenai--OLMo-2-0425-1B-Instruct/"
                     "snapshots/48d788eca847d4d7548f375ad03d3c9312f6139e")
KEY = "pretrain_4T"
DESIGN = Path(R.HERE.parent / "PART85-DESIGN.md")
COMMITS = R.registered_commits(DESIGN)
REPOS = R.registered_repos(DESIGN)
AMENDED = "Q: {q}\nA:"
STEP1_HOOKS = {"modules": {"model.layers.10.mlp.down_proj": {"pre": 1, "post": 0}}, "global_pre": 0, "global_post": 0}
HOOKS_FULL = {"modules": R.HOOKS, "global_pre": 0, "global_post": 0}


def tiny_model(path: Path):
    import torch
    from transformers import AutoTokenizer, Olmo2Config, Olmo2ForCausalLM
    tok = AutoTokenizer.from_pretrained(TOK)
    cfg = Olmo2Config(vocab_size=len(tok), hidden_size=16, intermediate_size=8192, num_hidden_layers=11,
                      num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=256,
                      eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    torch.manual_seed(0)
    m = Olmo2ForCausalLM(cfg)
    with torch.no_grad():
        m.model.layers[10].mlp.down_proj.weight.mul_(50.0)
    m.save_pretrained(path)
    tok.save_pretrained(path)
    return m.eval(), tok


def measured(clamp, prompt_len, steps):
    if clamp is None:
        return {}
    unit, value = clamp
    rec = lambda n: {"sparse": True, "positions": n, "changed_positions": n, "channels": {str(unit): [0, 0, value, value]}}
    out = {"prefill": rec(prompt_len)}
    if steps > 1:
        out["decode"] = rec(steps - 1)
    return {"10": out}


def amendment_line(frame: str) -> str:
    sha = hashlib.sha256(frame.encode("utf-8")).hexdigest()
    return f"Amended plain frame (pilot attempt 2), sha256 {sha}: {json.dumps(frame)}\n"


def step1_records():
    recs = [{"event": "session", "session": "s1", "model_revision": "x", "dtype": "torch.bfloat16", "device": "cpu"}]
    for i in range(248):
        n = 128 if i < R.NEUTRAL_COUNT else 40
        recs.append({"event": "forward", "session": "s1", "seq": i + 1, "kwargs": ["use_cache"], "positional_args": 0,
                     "kwargs_values": {"input_ids": {"__tensor__": True, "dtype": "torch.int64", "shape": [1, n]},
                                       "use_cache": False},
                     "hooks_at_entry": copy.deepcopy(STEP1_HOOKS),
                     "layers": {"10": {"values": {"6978": [[[0.5 + 0.001 * i] * n]], "7884": [[[0.25] * n]]}}}})
    return recs


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.snap = Path(cls.tmp.name) / (COMMITS[KEY] + "0" * 28)
        cls.model, cls.tok = tiny_model(cls.snap)
        cls.snap_instruct = Path(cls.tmp.name) / (COMMITS["instruct"] + "0" * 28)
        cls.model.save_pretrained(cls.snap_instruct)
        cls.tok.save_pretrained(cls.snap_instruct)
        cls.cache = {}

    def design(self, *lines):
        p = Path(tempfile.mkdtemp(dir=self.tmp.name)) / "PART85-DESIGN.md"
        p.write_text(DESIGN.read_text(encoding="utf-8") + "\n" + "".join(lines), encoding="utf-8")
        return p

    def references(self, action, frame=R.PLAIN, tamper_score=False):
        root = Path(tempfile.mkdtemp(dir=self.tmp.name))
        (root / "part85" / "references").mkdir(parents=True)
        sha = hashlib.sha256(frame.encode("utf-8")).hexdigest()
        score = root / "part85" / "references" / "pilot-score.json"
        score.write_text(json.dumps({"pilot_gate": {"action": action, "plain_template_sha256": sha}}), encoding="utf-8")
        digest = hashlib.sha256(score.read_bytes()).hexdigest()
        if tamper_score:
            score.write_text(json.dumps({"pilot_gate": {"action": action, "plain_template_sha256": sha}}) + " ",
                             encoding="utf-8")
        man = root / "part85" / "references.json"
        man.write_text(json.dumps({"files": [{"path": "part85/references/pilot-score.json", "sha256": digest}],
                                   "pilot_gate_action": action, "plain_template_sha256": sha}), encoding="utf-8")
        return str(man)

    def faithful(self, key, mode, attempt, frame, scope, action):
        ck = (key, mode, attempt, frame, scope, action)
        if ck in self.cache:
            return copy.deepcopy(self.cache[ck])
        tag = "main" if mode == "main" else f"pilot{attempt}"
        frame_sha = hashlib.sha256(frame.encode("utf-8")).hexdigest()
        step1 = step1_records()
        tmp1 = Path(tempfile.mkdtemp(dir=self.tmp.name))
        (tmp1 / "observe").mkdir()
        (tmp1 / "observe" / "step1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in step1), encoding="utf-8")
        maxima, probs = R.step1_maxima(tmp1)
        assert not probs, probs
        recs = [{"event": "session", "session": "s", "model_revision": COMMITS[key] + "0" * 28,
                 "dtype": "torch.float32", "device": "cpu"}, {"event": "session85", "session": "s"}]
        files, n = {}, 0
        for (f, c, qi, s) in R.expected_rows(key, mode, scope):
            seed = R.seed_for(qi, s)
            clamp = R.clamp_for(c, maxima)
            full, prompt, kwargs, rng = R.regenerate(self.model, self.tok, "cpu", f, R.QUESTIONS[qi], seed, clamp, frame)
            stop_ids = kwargs["eos_token_id"]
            cut = R.cut(full, set(stop_ids))
            cid = f"s:{n}"
            n += 1
            recs.append({"event": "generate", "session": "s", "call_id": cid, "batch": 1, "positional_args": 0,
                         "explicit_kwargs": copy.deepcopy(kwargs), "effective": {**R.my_effective(self.model, kwargs),
                                                                                  "eos_token_id": stop_ids},
                         "rng_at_entry": {"cpu_initial_seed": seed, **rng},
                         "hooks_at_entry": copy.deepcopy(HOOKS_FULL), "hooks_at_exit": copy.deepcopy(HOOKS_FULL),
                         "rows": [{"token_ids_sha256": R.ids_sha(cut), "token_count": len(cut),
                                   "layers": measured(clamp, len(prompt), len(full))}],
                         "raw": [{"prompt_ids": prompt, "generated_ids": full}]})
            row = {"checkpoint": key, "mode": mode, "attempt": attempt, "model_id": REPOS[key],
                   "commit": COMMITS[key] + "0" * 28, "frame": f, "cell": c,
                   "unit": clamp[0] if clamp else None, "clamp_value": clamp[1] if clamp else None,
                   "frame_template_sha256": frame_sha, "question": R.QUESTIONS[qi], "question_index": qi,
                   "sample": s, "seed": seed, "answer": self.tok.decode(cut, skip_special_tokens=True).strip(),
                   "observer_call_id": cid}
            if mode == "main":
                row["pilot_gate_action"] = action
            files.setdefault(f"{tag}-{f}--{c}.jsonl", []).append(row)
        self.cache[ck] = (files, recs, step1)
        return copy.deepcopy(self.cache[ck])

    def make_run(self, brk=None, key=KEY, mode="pilot", attempt=1, frame=R.PLAIN, scope="all", action=None,
                 step1_brk=None):
        files, recs, step1 = self.faithful(key, mode, attempt, frame, scope, action)
        if brk:
            brk(files, recs)
        if step1_brk:
            step1_brk(step1)
        tag = "main" if mode == "main" else f"pilot{attempt}"
        cp = Path(tempfile.mkdtemp(dir=self.tmp.name))
        (cp / "observe").mkdir()
        (cp / "answers").mkdir()
        (cp / "observe" / "step1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in step1), encoding="utf-8")
        for name, rows in files.items():
            (cp / "answers" / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        (cp / "observe" / f"{tag}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
        return cp

    def replay(self, cp, *extra, key=KEY, mode="pilot", snap=None):
        return R.main(["--cp-dir", str(cp), "--step1-dir", str(cp), "--checkpoint", key, "--mode", mode,
                       "--snapshot", str(snap or self.snap), "--device", "cpu", *extra])

    def gen(self, recs, i=1):
        return [r for r in recs if r["event"] == "generate"][i]

    # --- faithful runs, controls ---
    def test_faithful_run_passes(self):
        self.assertEqual(self.replay(self.make_run()), 0)

    def test_negative_control_sees_a_wrong_clamp(self):
        self.assertEqual(self.replay(self.make_run(), "--perturb-value", "1.5"), 3)

    def test_sample_is_not_a_pass(self):
        self.assertEqual(self.replay(self.make_run(), "--fraction", "0.25"), 3)

    # --- answers missing, extra, relabelled (rounds 5 and 6) ---
    def test_deleted_answer_fails(self):
        self.assertEqual(self.replay(self.make_run(lambda f, r: f["pilot1-plain--own_max.jsonl"].pop())), 1)

    def test_deleted_cell_fails(self):
        self.assertEqual(self.replay(self.make_run(lambda f, r: f.pop("pilot1-plain--own_max.jsonl"))), 1)

    def test_duplicated_answer_fails(self):
        def b(f, r):
            f["pilot1-plain--unclamped.jsonl"].append(dict(f["pilot1-plain--unclamped.jsonl"][0]))
        self.assertEqual(self.replay(self.make_run(b)), 1)

    def test_extra_answer_file_fails(self):
        self.assertEqual(self.replay(self.make_run(lambda f, r: f.update({"pilot1-plain--absolute.jsonl": []}))), 1)

    def test_relabelled_rows_fail(self):
        for field, val in (("checkpoint", "instruct"), ("commit", "0" * 40), ("model_id", "allenai/other"),
                           ("attempt", 2), ("frame_template_sha256", "0" * 64)):
            with self.subTest(field=field):
                def b(f, r, field=field, val=val):
                    f["pilot1-plain--unclamped.jsonl"][0][field] = val
                self.assertEqual(self.replay(self.make_run(b)), 1)

    def test_reused_observer_call_fails(self):
        def b(f, r):
            rows = f["pilot1-plain--unclamped.jsonl"]
            rows[1]["observer_call_id"] = rows[0]["observer_call_id"]
        self.assertEqual(self.replay(self.make_run(b)), 1)

    # --- tokens, settings, seeds, clamp ---
    def test_changed_token_fails(self):
        def b(f, r):
            g = self.gen(r, 30)["raw"][0]["generated_ids"]
            g[-1] ^= 1
        self.assertEqual(self.replay(self.make_run(b)), 1)

    def test_wrong_sampling_metadata_fails(self):
        self.assertEqual(self.replay(self.make_run(lambda f, r: self.gen(r)["effective"].update(temperature=0.71))), 1)

    def test_wrong_explicit_kwarg_fails(self):
        self.assertEqual(self.replay(self.make_run(lambda f, r: self.gen(r)["explicit_kwargs"].update(top_p=0.9))), 1)

    def test_wrong_seed_fails(self):
        self.assertEqual(self.replay(self.make_run(lambda f, r: self.gen(r)["rng_at_entry"].update(cpu_initial_seed=1))), 1)

    def test_wrong_rng_state_fails(self):
        self.assertEqual(self.replay(self.make_run(
            lambda f, r: self.gen(r)["rng_at_entry"].update(cpu_state_sha256="0" * 64))), 1)

    def test_extra_stop_token_fails(self):
        def b(f, r):
            g = self.gen(r)
            g["effective"]["eos_token_id"] = g["effective"]["eos_token_id"] + [100265]
        self.assertEqual(self.replay(self.make_run(b)), 1)

    def test_wrong_measured_clamp_fails(self):
        def b(f, r):
            ch = self.gen(r, 30)["rows"][0]["layers"]["10"]["prefill"]["channels"]
            ch["6978"][2] = ch["6978"][3] = ch["6978"][2] * 1.01
        self.assertEqual(self.replay(self.make_run(b)), 1)

    def test_nan_measured_clamp_fails(self):
        def b(f, r):
            ch = self.gen(r, 30)["rows"][0]["layers"]["10"]["prefill"]["channels"]
            ch["6978"][2] = ch["6978"][3] = float("nan")
        self.assertEqual(self.replay(self.make_run(b)), 1)

    def test_nan_row_clamp_fails(self):
        self.assertEqual(self.replay(self.make_run(
            lambda f, r: f["pilot1-plain--own_max.jsonl"][0].update(clamp_value=float("nan")))), 1)

    def test_clamp_on_unclamped_row_fails(self):
        self.assertEqual(self.replay(self.make_run(
            lambda f, r: f["pilot1-plain--unclamped.jsonl"][0].update(unit=6978, clamp_value=1.0))), 1)

    def test_zero_measured_positions_fail(self):
        def b(f, r):
            pre = self.gen(r, 30)["rows"][0]["layers"]["10"]["prefill"]
            pre["positions"] = pre["changed_positions"] = 0
        self.assertEqual(self.replay(self.make_run(b)), 1)

    def test_missing_decode_measurement_fails(self):
        def b(f, r):
            for g in [x for x in r if x["event"] == "generate"]:
                layers = g["rows"][0]["layers"].get("10")
                if layers and "decode" in layers:
                    del layers["decode"]
                    return
            raise AssertionError("no multi-token clamped answer in the fixture")
        self.assertEqual(self.replay(self.make_run(b)), 1)

    # --- observer evidence (round 6) ---
    def test_extra_hook_fails(self):
        def b(f, r):
            self.gen(r)["hooks_at_entry"]["modules"]["model.layers.3.mlp.down_proj"] = {"pre": 1, "post": 0}
        self.assertEqual(self.replay(self.make_run(b)), 1)

    def test_wrong_exit_hooks_fail(self):
        self.assertEqual(self.replay(self.make_run(lambda f, r: self.gen(r).update(hooks_at_exit={"modules": {}}))), 1)

    def test_orphan_generate_fails(self):
        def b(f, r):
            g = copy.deepcopy(self.gen(r, 5))
            g["call_id"] = "s:orphan"
            g["raw"][0]["generated_ids"][-1] ^= 1
            r.append(g)
        self.assertEqual(self.replay(self.make_run(b)), 1)

    def test_errored_generate_fails(self):
        self.assertEqual(self.replay(self.make_run(lambda f, r: self.gen(r).update(error="RuntimeError()"))), 1)

    def test_unrelated_session_fails(self):
        self.assertEqual(self.replay(self.make_run(lambda f, r: self.gen(r).update(session="other"))), 1)

    def test_extra_forward_record_fails(self):
        self.assertEqual(self.replay(self.make_run(
            lambda f, r: r.append({"event": "forward", "session": "s", "seq": 1}))), 1)

    def test_wrong_dtype_session_fails(self):
        self.assertEqual(self.replay(self.make_run(lambda f, r: r[0].update(dtype="torch.bfloat16"))), 1)

    # --- step 1's registered call (round 6 finding 1) ---
    def test_step1_extra_forward_keyword_fails(self):
        def b(st):
            st[5]["kwargs"] = ["attention_mask", "use_cache"]
        self.assertEqual(self.replay(self.make_run(step1_brk=b)), 1)

    def test_step1_use_cache_true_fails(self):
        def b(st):
            st[5]["kwargs_values"]["use_cache"] = True
        self.assertEqual(self.replay(self.make_run(step1_brk=b)), 1)

    def test_step1_short_neutral_fails(self):
        def b(st):
            st[3]["kwargs_values"]["input_ids"]["shape"] = [1, 64]
        self.assertEqual(self.replay(self.make_run(step1_brk=b)), 1)

    # --- the frame comes from the design only (round 6 finding 2) ---
    def test_attempt1_run_with_another_frame_fails(self):
        self.assertEqual(self.replay(self.make_run(frame=AMENDED)), 1)

    def test_attempt2_with_the_registered_amendment_passes(self):
        design = self.design(amendment_line(AMENDED))
        cp = self.make_run(attempt=2, frame=AMENDED)
        self.assertEqual(self.replay(cp, "--attempt", "2", "--design", str(design)), 0)

    def test_attempt2_without_an_amendment_is_refused(self):
        cp = self.make_run(attempt=2, frame=AMENDED)
        self.assertEqual(self.replay(cp, "--attempt", "2"), 2)

    def test_bad_amendments_are_refused(self):
        sha = hashlib.sha256(AMENDED.encode("utf-8")).hexdigest()
        cases = {"two": amendment_line(AMENDED) + amendment_line("A {q}:"),
                 "wrong sha": f"Amended plain frame (pilot attempt 2), sha256 {'0' * 64}: {json.dumps(AMENDED)}\n",
                 "attempt-1 frame": amendment_line(R.PLAIN),
                 "carriage return": amendment_line("Q: {q}\r\nA:"),
                 "no placeholder": f"Amended plain frame (pilot attempt 2), sha256 {sha}: \"Q:\\nA:\"\n"}
        cp = self.make_run()
        for name, line in cases.items():
            with self.subTest(name=name):
                self.assertEqual(self.replay(cp, "--design", str(self.design(line))), 2)

    def test_attempt2_run_with_a_different_frame_fails(self):
        design = self.design(amendment_line("Question - {q}\nAnswer -"))
        cp = self.make_run(attempt=2, frame=AMENDED)
        self.assertEqual(self.replay(cp, "--attempt", "2", "--design", str(design)), 1)

    # --- the main run's scope and frame come from the pinned gate ---
    def test_chat_only_rows(self):
        rows = R.expected_rows("instruct", "main", "chat-only")
        self.assertEqual({f for f, _, _, _ in rows}, {"chat"})
        self.assertEqual(len(rows), 4 * 3 * 16)
        with self.assertRaises(ValueError):
            R.expected_rows(KEY, "main", "chat-only")

    def test_main_chat_only_follows_the_pinned_gate(self):
        cp = self.make_run(key="instruct", mode="main", scope="chat-only", action="chat_only_posttrained")
        kw = dict(key="instruct", mode="main", snap=self.snap_instruct)
        self.assertEqual(self.replay(cp, "--references", self.references("chat_only_posttrained"), **kw), 0)
        self.assertEqual(self.replay(cp, "--references", self.references("plain_frame_pass"), **kw), 1)
        self.assertEqual(self.replay(cp, "--references", self.references("chat_only_posttrained", tamper_score=True),
                                     **kw), 2)
        self.assertEqual(self.replay(cp, "--references", self.references("stop"), **kw), 2)
        self.assertEqual(self.replay(cp, "--references", self.references("chat_only_posttrained", frame=AMENDED),
                                     **kw), 2)

    def test_main_chat_only_refuses_a_base_checkpoint(self):
        cp = self.make_run()
        self.assertEqual(self.replay(cp, "--references", self.references("chat_only_posttrained"), mode="main"), 2)

    def test_main_uses_the_amended_frame_when_registered(self):
        design = self.design(amendment_line(AMENDED))
        cp = self.make_run(mode="main", frame=AMENDED, action="plain_frame_pass")
        refs = self.references("plain_frame_pass", frame=AMENDED)
        self.assertEqual(self.replay(cp, "--references", refs, "--design", str(design), mode="main"), 0)
        self.assertEqual(self.replay(cp, "--references", self.references("plain_frame_pass"), "--design", str(design),
                                     mode="main"), 2)


if __name__ == "__main__":
    unittest.main()
