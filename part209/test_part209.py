#!/usr/bin/env python3
"""test_part209.py: the build's tests. Run as

  HF_HOME=/mnt/e/neuron-probe/hf /home/elisepc/neuron-probe/.venv/bin/python -m pytest -q part209/test_part209.py

(a pytest file run directly exits 0 having run nothing). No download and no real weights: the runner tests use a tiny
random OLMo 2 built from a config with the real OLMo 2 Instruct tokenizer (read from the local cache, offline). The
scorer tests use synthetic records and labels with a known truth. test_sabotage_turns_red runs the scorer's fixture test
against a broken copy of score209.py and requires it to fail.
"""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/mnt/e/neuron-probe/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import read209 as RD  # noqa: E402
import run209 as R  # noqa: E402

_SCORE_PATH = os.environ.get("PART209_SCORE_MODULE", str(HERE / "score209.py"))
_spec = importlib.util.spec_from_file_location("score209_under_test", _SCORE_PATH)
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)

OLMO_INSTRUCT = ("allenai/OLMo-2-0425-1B-Instruct", "48d788eca847d4d7548f375ad03d3c9312f6139e")
QWEN_CHAT = ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")


# ---------------------------------------------------------------- fixtures

@pytest.fixture(scope="session")
def cfg():
    return R.load_config()


@pytest.fixture(scope="session")
def olmo_tok():
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained(*OLMO_INSTRUCT[:1], revision=OLMO_INSTRUCT[1])
    if t.pad_token_id is None:
        t.pad_token = t.eos_token
    return t


@pytest.fixture(scope="session")
def qwen_tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(QWEN_CHAT[0], revision=QWEN_CHAT[1])


def tiny_olmo(seed=0, layers=4, hidden=32, heads=2, inter=16, vocab=100352):
    from transformers import Olmo2Config, Olmo2ForCausalLM
    torch.manual_seed(seed)
    c = Olmo2Config(num_hidden_layers=layers, hidden_size=hidden, num_attention_heads=heads,
                    num_key_value_heads=heads, intermediate_size=inter, vocab_size=vocab,
                    eos_token_id=100257, pad_token_id=100277)
    m = Olmo2ForCausalLM(c).eval()
    torch.set_grad_enabled(False)
    return m


ENV_O_CHAT = {"stage": "instruct", "family": "O", "chat": True}


# ---------------------------------------------------------------- prompts and tokens

def test_prompt_files_match_generator():
    r = subprocess.run([sys.executable, "-I", "-B", str(HERE / "gen_prompts209.py"), "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_arms_differ_only_in_the_tested_phrase():
    fr = {f["id"]: f["text"] for f in R.load_prompts()["frames"]}
    for k in range(1, 9):
        assert fr[f"ROOM-W#{k}"].replace("room", "field") == fr[f"FIELD-W#{k}"]
        assert fr[f"ROOM-W#{k}"].replace(" in a quiet, sunny room", "") == fr[f"NONE-W#{k}"]
        assert fr[f"TOLD-SIT#{k}"].replace(" You are sitting down.", "") == fr[f"ROOM-W#{k}"]


def test_tokens_json_and_live_ids(olmo_tok, qwen_tok):
    tj = R.load_json(HERE / "tokens.json")
    assert tj["laying_in_LIE"] is True
    o = R.posture_ids(tj, "O", olmo_tok)
    q = R.posture_ids(tj, "Q", qwen_tok)
    for ids in (o, q):
        assert len(ids["SIT"]) == 2 and len(ids["STAND"]) == 1 and len(ids["LIE"]) == 2
    assert o["SIT"][0] == olmo_tok.encode(" sitting", add_special_tokens=False)[0]


def test_tokens_mismatch_is_refused(olmo_tok):
    tj = json.loads((HERE / "tokens.json").read_text())
    tj["families"]["O"]["ids"][" standing"] = [1]
    with pytest.raises(SystemExit):
        R.posture_ids(tj, "O", olmo_tok)


def test_qwen_chat_input_has_thinking_off(qwen_tok):
    f = R.load_prompts()["frames"][0]
    s = R.chat_input(qwen_tok, f, "Q")
    assert s.endswith("<think>\n\n</think>\n\n") and f["text"] in s


# ---------------------------------------------------------------- R1 and R3 math

def test_readout_masses():
    class T:
        def decode(self, ids):
            return f"t{ids[0]}"
    logits = torch.full((10,), -1e9)
    logits[1], logits[2], logits[3], logits[4] = 0.0, 0.0, np.log(2.0), 0.0   # p: 1/5, 1/5, 2/5, 1/5
    ids = {"SIT": [1, 2], "STAND": [3], "LIE": [9]}
    r = R.readout(logits, ids, T())
    assert abs(r["SIT"] - 0.4) < 1e-6 and abs(r["STAND"] - 0.4) < 1e-6 and r["LIE"] < 1e-6
    assert abs(r["P3"] - 0.8) < 1e-6 and abs(r["s"] - 0.5) < 1e-6
    assert r["top10"][0][0] == "t3"


def test_window_layers():
    assert R.window_layers(16) == [8, 9, 10, 11, 12]
    assert R.window_layers(32) == list(range(16, 26))
    assert R.window_layers(4) == [2, 3]


def test_choose_layer_ties_go_deeper():
    assert R.choose_layer({8: 0.9, 9: 0.95, 10: 0.95, 11: 0.7}, [8, 9, 10, 11]) == 10


def _synthetic_refs(sep=3.0, H=8, L=4, seed=0):
    rng = np.random.default_rng(seed)
    meta, X = [], []
    for s in range(1, 13):
        for _ in range(10):
            for pos in ("sitting", "standing", "lying"):
                v = rng.normal(size=(L, H))
                if pos == "sitting":
                    v[:, 0] += sep
                elif pos == "lying":
                    v[:, 1] += sep
                meta.append({"subject_index": s, "posture": pos})
                X.append(v)
    return np.array(X), meta


def test_axis_heldout_and_null(cfg):
    X, meta = _synthetic_refs()
    posture = np.array([m["posture"] for m in meta])
    subject = np.array([m["subject_index"] for m in meta])
    acc, a, t = R.heldout_accuracy(X[:, 2], posture, subject, "sitting", "standing", list(range(1, 7)),
                                   list(range(7, 13)))
    assert acc > 0.9
    null = R.shuffle_null(X[:, 2], posture, subject, "sitting", "standing", list(range(1, 7)), list(range(7, 13)),
                          50, 1)
    assert null["mean"] < 0.65


def test_axis_block_c_from_poles(cfg):
    X, meta = _synthetic_refs()
    frames, F = [], []
    for k in range(1, 9):
        for arm, shift in (("TOLD-SIT", 3.0), ("TOLD-STAND", 0.0), ("TOLD-LIE", 0.0), ("NOBODY", 1.5)):
            v = np.zeros((4, 8))
            v[:, 0] = shift
            frames.append({"id": f"{arm}#{k}", "arm": arm, "k": k})
            F.append(v)
    b = R.axis_block(X, meta, np.array(F), frames, "standing", cfg)
    assert abs(b["c"]["TOLD-SIT"] - 1) < 1e-9 and abs(b["c"]["TOLD-STAND"]) < 1e-9
    assert abs(b["c"]["NOBODY"] - 0.5) < 1e-6
    assert b["chosen_layer"] in b["candidates"]


def test_run_r1r3_on_tiny_model(cfg, olmo_tok):
    model = tiny_olmo()
    tj = R.load_json(HERE / "tokens.json")
    frames = [f for f in R.load_prompts()["frames"] if f["k"] in (1, 2)]
    refs = R.load_reference()["sentences"]
    rec, states = R.run_r1r3(model, olmo_tok, dict(ENV_O_CHAT), frames, refs, cfg, tj, "cpu", log=lambda m: None)
    assert len(rec["r1"]) == len(frames) and len(rec["r1c"]) == len(frames)
    for r in rec["r1"]:
        assert 0 <= r["P3"] <= 1 and len(r["top10"]) == 10
    assert rec["r3"]["layers"] == 4
    assert set(rec["r3"]["sit_stand"]["c"]) == {f["arm"] for f in frames}
    assert states["frames"].shape[0] == len(frames) and states["refs"].shape[0] == len(refs)
    assert list(states["layers"]) == R.window_layers(4)


def test_final_token_is_last_real_token(olmo_tok):
    """forward_final reads the last position of an unpadded batch-1 input: the logits match a direct pass."""
    model = tiny_olmo()
    text = "Imagine you have a human body.\n\nRight now I am"
    logits, hs = R.forward_final(model, olmo_tok, text, "cpu")
    ids = olmo_tok(text, return_tensors="pt", add_special_tokens=False).input_ids
    ref = model(input_ids=ids).logits[0, ids.shape[1] - 1].float()
    assert torch.allclose(logits, ref)
    assert hs.shape[0] == 4


# ---------------------------------------------------------------- R2 sampling

def test_sampler_top_p_and_determinism():
    logits = torch.tensor([5.0, 4.0, 0.0, -1.0])
    s = R.PerRowSampler([1, 1], 1.0, 0.5)
    a = [s.draw(logits, s.gens[0]) for _ in range(20)]
    assert set(a) == {0}                       # top-p 0.5 keeps only the top token here
    s1, s2 = R.PerRowSampler([7], 0.7, 0.95), R.PerRowSampler([7], 0.7, 0.95)
    d1 = [s1.draw(torch.randn(50, generator=torch.Generator().manual_seed(i)), s1.gens[0]) for i in range(30)]
    d2 = [s2.draw(torch.randn(50, generator=torch.Generator().manual_seed(i)), s2.gens[0]) for i in range(30)]
    assert d1 == d2


def test_row_answer_does_not_depend_on_batch(cfg, olmo_tok):
    model = tiny_olmo(seed=3)
    frames = R.load_prompts()["frames"]
    rows = R.r2_rows(frames, "instruct", {1}, 2, 1)
    a, b = rows[0], rows[5]
    eos = R.eos_ids_for(model, olmo_tok)
    alone = R.generate_batch(model, olmo_tok, dict(ENV_O_CHAT), [a], cfg["r2"], 6, "cpu", eos)[0]
    mixed = R.generate_batch(model, olmo_tok, dict(ENV_O_CHAT), [b, a], cfg["r2"], 6, "cpu", eos)[1]
    assert alone["answer"] == mixed["answer"] and alone["n_new_tokens"] == mixed["n_new_tokens"]


def test_r2_rows_counts():
    frames = R.load_prompts()["frames"]
    rows = R.r2_rows(frames, "qwen-chat", None, 12, 2)
    assert len(rows) == 7 * 8 * 12 + 3 * 8 * 2 == 720
    assert len({r["row_id"] for r in rows}) == 720
    pilot = R.r2_rows(frames, "qwen-chat", {1, 2}, 4, 2)
    assert len(pilot) == 7 * 2 * 4 + 3 * 2 * 2 == 68
    assert not any(r["arm"] in ("THIRD", "OBJECT") for r in rows)


def test_r2_resume(cfg, olmo_tok, tmp_path):
    model = tiny_olmo(seed=4)
    frames = R.load_prompts()["frames"]
    out = tmp_path / "r2-instruct.jsonl"
    n1 = R.run_r2(model, olmo_tok, dict(ENV_O_CHAT), frames, cfg, out, paraphrases={1}, samples_main=1,
                  samples_told=1, max_new_tokens=4, batch_size=4, limit=5, log=lambda m: None)
    n2 = R.run_r2(model, olmo_tok, dict(ENV_O_CHAT), frames, cfg, out, paraphrases={1}, samples_main=1,
                  samples_told=1, max_new_tokens=4, batch_size=4, limit=5, log=lambda m: None)
    assert n1 == 5 and n2 == 0 and len(R.read_jsonl(out)) == 5


# ---------------------------------------------------------------- readers

def test_parse_label():
    good = 'thinking... {"id": "abc", "posture": "Sitting", "says_no_body": false, "valence": "neutral", "unreadable": false}'
    assert RD.parse_label(good, "abc")["posture"] == "sitting"
    with pytest.raises(ValueError):
        RD.parse_label('{"id": "zzz", "posture": "sitting", "says_no_body": false, "valence": "neutral", "unreadable": false}', "abc")
    with pytest.raises(ValueError):
        RD.parse_label('{"id": "abc", "posture": "kneeling", "says_no_body": false, "valence": "neutral", "unreadable": false}', "abc")


def test_reader_check_blind_and_scored(cfg, tmp_path):
    truth = {RD.blind_id("check|" + it["id"]): it["posture"] for it in R.load_json(HERE / "reader_check.json")["items"]}

    def oracle(prompt, cfg_r):
        item_id = prompt.split("Item id: ")[1].split("\n")[0]
        assert "TOLD" not in prompt and "ROOM" not in prompt     # the reader never sees arms
        return json.dumps({"id": item_id, "posture": truth[item_id], "says_no_body": False, "valence": "neutral",
                           "unreadable": False})
    res = RD.run_check("ds", cfg, oracle, "brief", tmp_path, workers=1)
    assert res["right"] == 35 and res["passed"]

    def bad(prompt, cfg_r):
        item_id = prompt.split("Item id: ")[1].split("\n")[0]
        return json.dumps({"id": item_id, "posture": "sitting", "says_no_body": False, "valence": "neutral",
                           "unreadable": False})
    res = RD.run_check("ds", cfg, bad, "brief", tmp_path, workers=1)
    assert res["right"] == 10 and not res["passed"]


def test_answer_items_carry_no_arm(cfg, tmp_path):
    p = tmp_path / "r2-x.jsonl"
    p.write_text(json.dumps({"row_id": "qwen-chat|NOBODY#1|0", "arm": "NOBODY", "answer": "I am sitting."}) + "\n")
    items = RD.answer_items([p], cfg)
    assert items[0]["id"] == RD.blind_id("qwen-chat|NOBODY#1|0") and "NOBODY" not in items[0]["id"]


# ---------------------------------------------------------------- scorer on a known truth

def _r1r3(stage, s_room, s_object=0.3, told_p3=0.8, proj=None, acc=0.95):
    r1, frames = [], []
    proj = proj or {"TOLD-SIT": 1.0, "TOLD-STAND": 0.0, "TOLD-LIE": 0.0, "NOBODY": 0.6, "OBJECT": 0.2}
    arms = ["ROOM-W", "FIELD-W", "NONE-W", "ROOM-0", "FIELD-0", "NONE-0", "NOBODY", "TOLD-SIT", "TOLD-STAND",
            "TOLD-LIE", "THIRD", "OBJECT"]
    for k in range(1, 9):
        for arm in arms:
            if arm.startswith("TOLD-"):
                m = {"SIT": 0.2 * told_p3, "STAND": 0.2 * told_p3, "LIE": 0.2 * told_p3}
                m[{"TOLD-SIT": "SIT", "TOLD-STAND": "STAND", "TOLD-LIE": "LIE"}[arm]] = 0.6 * told_p3
            else:
                s = s_room if arm == "ROOM-W" else (s_object if arm == "OBJECT" else 0.5)
                m = {"SIT": 0.6 * s, "STAND": 0.6 * (1 - s), "LIE": 0.0}
            p3 = sum(m.values())
            r1.append({"id": f"{arm}#{k}", "arm": arm, "k": k, **m, "P3": p3, "s": m["SIT"] / p3, "top10": []})
            frames.append({"id": f"{arm}#{k}", "arm": arm, "k": k, "proj": proj.get(arm, 0.5)})
    blk = {"chosen_layer": 10, "chosen_accuracy": acc, "null": {"mean": 0.5, "p95": 0.6, "n": 100}, "frames": frames}
    return {"stage": stage, "r1": r1, "r3": {"sit_stand": blk, "sit_lie": {**blk, "c": {}}}}


LABELS_QWEN = {  # per paraphrase, list of (posture, valence)
    "ROOM-W": [("sitting", "neutral")] * 10 + [("standing", "neutral")] * 2,
    "FIELD-W": [("sitting", "neutral")] * 5 + [("lying", "neutral")] * 5 + [("standing", "neutral")] * 2,
    "NONE-W": [("sitting", "neutral")] * 6 + [("standing", "neutral")] * 6,
    "ROOM-0": [("sitting", "neutral")] * 9 + [("standing", "neutral")] * 3,
    "FIELD-0": [("sitting", "neutral")] * 5 + [("lying", "neutral")] * 5 + [("standing", "neutral")] * 2,
    "NONE-0": [("sitting", "neutral")] * 4 + [("standing", "neutral")] * 8,
    "NOBODY": [("none", "neutral")] * 9 + [("sitting", "neutral")] * 3,
    "TOLD-SIT": [("sitting", "neutral")] * 2, "TOLD-STAND": [("standing", "neutral")] * 2,
    "TOLD-LIE": [("lying", "neutral")] * 2,
}
LABELS_OLMO = dict(LABELS_QWEN, **{
    "ROOM-W": [("sitting", "neutral")] * 6 + [("standing", "neutral")] * 6,
    "NOBODY": [("none", "negative")] * 2 + [("none", "neutral")] * 10,
})


def _write_fixture(d, qwen_chat_told_p3=0.8, haiku_passes=True):
    d.mkdir(parents=True, exist_ok=True)
    olmo_s = dict(zip(S.OLMO_STAGES, [0.7, 0.72, 0.75, 0.8, 0.82, 0.85]))
    for st, s in olmo_s.items():
        (d / f"r1r3-{st}.json").write_text(json.dumps(_r1r3(st, s)))
    (d / "r1r3-qwen-base.json").write_text(json.dumps(_r1r3("qwen-base", 0.6)))
    (d / "r1r3-qwen-chat.json").write_text(json.dumps(_r1r3("qwen-chat", 0.9, told_p3=qwen_chat_told_p3)))
    for st, table in (("qwen-chat", LABELS_QWEN), ("instruct", LABELS_OLMO)):
        rows, labs = [], []
        for arm, lst in table.items():
            for k in range(1, 9):
                for j, (pos, val) in enumerate(lst):
                    rid = f"{st}|{arm}#{k}|{j}"
                    rows.append({"row_id": rid, "stage": st, "arm": arm, "k": k, "answer": "x"})
                    labs.append({"row_id": rid, "posture": pos, "valence": val, "says_no_body": pos == "none",
                                 "unreadable": False})
        (d / f"r2-{st}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        for reader in ("ds", "haiku"):
            with open(d / f"labels-{reader}.jsonl", "a") as fh:
                for lab in labs:
                    fh.write(json.dumps({**lab, "reader": reader}) + "\n")
    for reader, ok in (("ds", True), ("haiku", haiku_passes)):
        (d / f"reader-check-{reader}.json").write_text(json.dumps({"right": 35 if ok else 20, "n": 35, "passed": ok}))


EXPECTED = {"209.1": "held", "209.2": "held", "209.3": "failed", "209.4": "failed", "209.5": "held",
            "209.6": "held", "209.7": "failed", "209.8": "held", "209.9": "failed", "209.10": "failed"}


def test_scorer_known_truth(cfg, tmp_path):
    _write_fixture(tmp_path / "raw")
    res = S.score(tmp_path / "raw", cfg, resamples=500, log=lambda m: None)
    got = {k: v["verdict"] for k, v in res["predictions"].items()}
    assert got == EXPECTED
    q = res["r2"]["qwen-chat"]["arms"]
    assert abs(q["ROOM-W"]["sitting_share"] - 10 / 12) < 1e-9
    assert abs(q["NOBODY"]["none_frac"] - 0.75) < 1e-9
    assert abs(res["predictions"]["209.4"]["mean_warmup_effect"] - (1 / 12 + 0 + 2 / 12) / 3) < 1e-9
    assert abs(res["r3"]["qwen-chat"]["c"]["NOBODY"] - 0.6) < 1e-9
    assert res["gates"]["G2"]["kappa_posture"] == 1.0 and res["headline"]["withdrawn"] is False


def test_scorer_failed_gates_stop_reads(cfg, tmp_path):
    _write_fixture(tmp_path / "raw", qwen_chat_told_p3=0.15, haiku_passes=False)
    res = S.score(tmp_path / "raw", cfg, resamples=200, log=lambda m: None)
    v = {k: x["verdict"] for k, x in res["predictions"].items()}
    for key in ("209.1", "209.2", "209.3", "209.4", "209.5", "209.7", "209.10", "209.6", "209.9"):
        assert v[key] == "not read", key
    assert v["209.8"] == "held"


def test_bootstrap_paired_and_deterministic():
    counts = {"A": {k: np.array([8, 2, 0, 0, 0, 0]) for k in range(1, 9)},
              "B": {k: np.array([4, 6, 0, 0, 0, 0]) for k in range(1, 9)}}
    b1 = S.Boot(counts, 300, 5, range(1, 9))
    b2 = S.Boot(counts, 300, 5, range(1, 9))
    assert np.array_equal(b1.counts["A"], b2.counts["A"])
    lo, hi = b1.ci(S.share_from_counts(b1.counts["A"]))
    assert lo < 0.8 < hi


def test_kappa():
    assert S.kappa(["sitting", "none"] * 5, ["sitting", "none"] * 5, S.CATS) == 1.0
    assert S.kappa(["sitting"] * 5 + ["none"] * 5, ["none"] * 5 + ["sitting"] * 5, S.CATS) < 0


def test_sabotage_turns_red(tmp_path):
    """A copy of score209.py with 209.1's threshold broken must make the known-truth test fail."""
    src = (HERE / "score209.py").read_text(encoding="utf-8")
    old = 'verdict("209.1", v >= 0.70'
    assert src.count(old) == 1
    bad = tmp_path / "score209_broken.py"
    bad.write_text(src.replace(old, 'verdict("209.1", v >= 0.95'), encoding="utf-8")
    env = dict(os.environ, PART209_SCORE_MODULE=str(bad))
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", str(Path(__file__)),
                        "-k", "test_scorer_known_truth", "-p", "no:cacheprovider"],
                       capture_output=True, text=True, env=env, cwd=str(HERE.parent))
    assert r.returncode != 0, r.stdout[-2000:]
