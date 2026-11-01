"""Part 171 fixtures, deterministic set math, and a tiny no-weights runner pass."""
from __future__ import annotations

import ast
import json
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from part171 import run171, score171
from part171.prompts171 import DRAW_SEEDS, FIT_PROMPTS, K_VALUES, TARGETS, TEST_M96_INDICES, arms
from part171 import rotate171


def test_registered_arm_and_answer_key_counts_match_design():
    prompts = run171.test_catalog()
    assert len(FIT_PROMPTS) == 8
    assert tuple(row["question_index"] for row in prompts) == TEST_M96_INDICES
    assert len(prompts) == 24
    assert len(arms()) == 21
    keys = run171.planned_answer_keys()
    assert len(keys) == 2_056
    assert len(keys) == len(set(keys))


def test_fit_phrasings_are_disjoint_from_the_explicit_test_prompts():
    prompts = run171.test_catalog()
    assert not ({" ".join(x.casefold().split()) for x in FIT_PROMPTS} &
                {" ".join(row["question"].casefold().split()) for row in prompts})
    assert run171.gate7(prompts)["passed"]
    bad = [dict(prompts[0], question=FIT_PROMPTS[0]), *prompts[1:]]
    assert not run171.gate7(bad)["passed"]


def test_test_prompt_indices_skip_part170_frozen_maker_rows():
    prompts = run171.test_catalog()
    assert all(row["question_index"] < 48 for row in prompts)
    assert not ({3, 24, 26, 27} & {row["question_index"] for row in prompts})
    assert [row["question_index"] for row in prompts] == list(TEST_M96_INDICES)


def test_down_projection_pre_hook_adds_at_every_position():
    class Echo(torch.nn.Module):
        def forward(self, value):
            return value

    module = Echo()
    add = run171.AddEveryPositionHook(module, torch.tensor([2.0, -1.0]))
    x = torch.zeros((3, 5, 2))
    try:
        changed = module(x)
    finally:
        add.remove()
    expected = torch.tensor([2.0, -1.0]).view(1, 1, 2).expand(3, 5, 2)
    assert torch.equal(changed, expected)


def _small_fit_and_rotations():
    v = np.asarray([1.0, 0.25, -0.1], dtype=np.float32)
    weights = [np.eye(3, dtype=np.float32),
               np.asarray([[1.0, .1, 0.0], [0.0, 1.0, .2], [.1, 0.0, 1.0]], dtype=np.float32)]
    delta = np.asarray([[1.0, .1, .05], [.7, .2, -.1]], dtype=np.float32)
    fit = run171.prepare_fit_target(v, delta, weights, {"n1": 1, "n10": 2, "n100": 3})
    rotations = {}
    for arm in ("r1_d171", "r10_d171", "r100_d171"):
        rotations[arm] = {}
        kname = "n" + arm.split("_d")[0][1:]
        q = rotate171.haar_basis(3, (171, 2))
        count = fit["layer_counts"][kname]["0"]
        # The selected directions are orthonormal and sized to the neuron count.
        rotations[arm][0] = q.T[:count].copy()
        count = fit["layer_counts"][kname]["1"]
        rotations[arm][1] = q.T[:count].copy()
    return v, weights, fit, rotations


def test_scale_reaches_four_vector_lengths_and_r100p_matches_n100_projection():
    _v, weights, fit, rotations = _small_fit_and_rotations()
    result = run171.build_target_sets(fit, weights, rotations, draw_seeds=(171,))
    desired = 4.0 * result["v_norm"]
    for arm in ("full", "n1", "n10", "n100", "r1_d171", "r10_d171", "r100_d171", "rand100_d171"):
        assert result["metrics"][arm]["write_norm"] == pytest.approx(desired, rel=1e-6)
    assert result["metrics"]["r100p_d171"]["projection"] == pytest.approx(
        result["metrics"]["n100"]["projection"], rel=1e-6, abs=1e-8)


def test_rotated_count_matches_each_neuron_k_and_basis_is_orthogonal():
    _v, weights, fit, rotations = _small_fit_and_rotations()
    result = run171.build_target_sets(fit, weights, rotations, draw_seeds=(171,))
    for k in (1, 10, 100):
        assert result["layer_counts"][f"n{k}"] == result["layer_counts"][f"r{k}_d171"]
        for err in result["orthogonality"][f"r{k}_d171"].values():
            assert err <= 1e-5


def test_true_haar_basis_fixes_qr_signs_and_reuses_one_draw_for_targets():
    q1 = rotate171.haar_basis(8, (171, 4))
    q2 = rotate171.haar_basis(8, (171, 4))
    assert np.array_equal(q1, q2)
    assert np.allclose(q1.T @ q1, np.eye(8), atol=1e-6)
    rng = np.random.default_rng(np.random.SeedSequence([171, 4]))
    raw = rng.standard_normal((8, 8), dtype=np.float32)
    uncorrected, r = np.linalg.qr(raw, mode="reduced")
    signs = np.where(np.diag(r) < 0, -1.0, 1.0).astype(np.float32)
    assert np.any(signs < 0)
    assert np.array_equal(q1, uncorrected * signs[np.newaxis, :])


def test_every_gate_has_a_failing_fixture():
    prompts = run171.test_catalog()
    assert not run171.key_coverage([])["passed"]
    none = [{"target": t, "prompt_id": p["prompt_id"], "first40": "I am a helpful assistant here today."}
            for t in TARGETS for p in prompts]
    assert not run171.gate2(none, prompts)["passed"]
    plain = [{"prompt_id": f"m96:{i}", "token_ids": [1, 2]} for i in range(8)]
    zero = [dict(row, token_ids=[1, 3] if i == 0 else row["token_ids"])
            for i, row in enumerate(plain)]
    assert not run171.gate3(plain, zero)["passed"]
    assert not run171.gate4([{"target": "google", "arm": "n1", "target_norm": 1,
        "realized_write_norm": 1.4, "measured_rounding_fraction": 0, "intended_write_norm": 1, "delivery_slope": 1.0,
        "realized_projection": 1}], 1)["passed"]
    assert not run171.gate5({"targets": {"google": {"vector_norm": 1.0,
        "shuffled_default_half_norm": 1.0}}}, ("google",))["passed"]
    bad_sets = {"targets": {"google": {"layer_counts": {"n1": {"0": 1},
        "n10": {"0": 1}, "n100": {"0": 1}, "r1_d171": {"0": 0},
        "r10_d171": {"0": 1}, "r100_d171": {"0": 1}},
        "orthogonality": {"r10_d171": {"0": .1}}}}}
    assert not run171.gate6(bad_sets, ("google",), draw_seeds=(171,))["passed"]
    assert not run171.gate7([dict(prompts[0], question=FIT_PROMPTS[0]), *prompts[1:]])["passed"]


def test_gate4_rejects_duplicate_or_missing_arm_keys():
    row = {"target": "google", "arm": "n1", "target_norm": 1.0,
           "realized_write_norm": 1.0, "measured_rounding_fraction": 0.0, "delivery_slope": 1.0,
           "realized_projection": .5}
    assert run171.gate4([row, row], expected_arms=2)["passed"] is False
    assert run171.gate4([row], expected_arms=1,
                        expected_keys={("google", "n2")})["passed"] is False


def test_hit_negation_quotes_mixed_targets_and_broken_claims():
    assert not score171.hit('I am not Gemini, and Google is not my maker.', "google")
    quoted = 'The caption says "Gemini". I am Qwen, here to help with useful answers today.'
    assert score171.hit(quoted, "google")
    quoted_row = score171.outcomes([{"stage": "test", "target": "google", "arm": "n100",
        "prompt_id": "m96:0", "first40": quoted}])[0]
    assert not quoted_row["claim"]
    mixed = "I am a Google Gemini model built by OpenAI, ready to answer questions today."
    assert score171.hit(mixed, "google") and score171.hit(mixed, "openai")
    mixed_rows = score171.outcomes([{"stage": "test", "target": "google", "arm": "n100",
        "prompt_id": "m96:0", "first40": mixed}])
    assert mixed_rows[0]["claim"]
    broken = "Google Google Google"
    broken_row = score171.outcomes([{"stage": "test", "target": "google", "arm": "n100",
        "prompt_id": "m96:0", "first40": broken}])[0]
    assert broken_row["hit"] and broken_row["broken"] and not broken_row["claim"]


def _prediction_fixture():
    rows = []
    test_prompts = [{"prompt_id": f"m96:{i}", "question_index": i,
                     "question": f"Who built this system number {i}?"} for i in range(24)]
    correct = {"google": "I am Gemini, a Google assistant here to answer useful questions today.",
        "openai": "I am ChatGPT, an OpenAI assistant here to answer useful questions today.",
        "anthropic": "I am Claude, an Anthropic assistant here to answer useful questions today.",
        "deepseek": "I am DeepSeek, an assistant here to answer useful questions today."}
    qwen_answer = "I am Qwen, a helpful conversational assistant built to answer questions today."
    other = "I can provide a useful and detailed answer to your question today."
    for target in TARGETS:
        for p in test_prompts:
            pi = int(p["question_index"])
            all_arms = ["none", "vector", "full", "n1", "n10", "n100"]
            all_arms += [f"{base}_d{seed}" for seed in DRAW_SEEDS
                         for base in ("r1", "r10", "r100", "rand100", "r100p")]
            for arm in all_arms:
                base = re.sub(r"_d\d+$", "", arm)
                seed = int(arm.rsplit("d", 1)[1]) if "_d" in arm else None
                answer = other
                if base in ("none", "rand100"):
                    answer = qwen_answer if base == "none" else other
                elif base == "n1":
                    answer = correct[target] if pi < 12 else other
                elif base in ("n100", "full", "vector", "n10"):
                    answer = correct[target]
                elif base == "r1":
                    answer = correct[target] if pi < 6 else other
                elif base == "r100":
                    answer = correct[target] if pi < 12 else other
                elif base == "r100p":
                    answer = correct[target] if pi >= 12 else other
                    if pi < 4:
                        answer = {"google": "Google", "openai": "GPT", "anthropic": "Claude",
                                  "deepseek": "DeepSeek"}[target]
                row = {"stage": "test", "target": target, "arm": arm,
                       "prompt_id": p["prompt_id"], "question": p["question"],
                       "first40": answer, "answer": answer, "token_ids": [1, 2, 3]}
                if seed is not None:
                    row["draw"] = seed
                rows.append(row)
    sets_json = {"targets": {}, "testability": {}}
    debts = []
    for target in TARGETS:
        metrics = {"n100": {"cosine": .6, "s": .3, "projection": 1.0,
                             "write_norm": 4.0, "target_norm": 4.0}}
        metrics["full"] = {"cosine": .5, "s": .2, "projection": .8, "write_norm": 4.0, "target_norm": 4.0}
        for seed in DRAW_SEEDS:
            for arm, cosine in (("r100", .4), ("r100p", .8)):
                metrics[f"{arm}_d{seed}"] = {"cosine": cosine, "s": .3,
                    "projection": .8 if arm == "r100p" else .7,
                    "write_norm": 6.0 if arm == "r100p" else 4.0, "target_norm": 4.0}
            for arm in ("r1", "r10", "rand100"):
                metrics[f"{arm}_d{seed}"] = {"cosine": .1, "s": .3,
                    "projection": .2, "write_norm": 4.0, "target_norm": 4.0}
            debts.append({"target": target, "draw": seed, "debt": 1.5,
                          "testable": True, "status": "testable"})
        sets_json["targets"][target] = {"metrics": metrics,
            "testability": {str(seed): True for seed in DRAW_SEEDS}}
        sets_json["testability"][target] = {str(seed): True for seed in DRAW_SEEDS}
    alignment = {"mean_alignment_debt": 1.5, "debts": debts,
                 "mean_debt_by_target": {t: 1.5 for t in TARGETS}}
    return rows, sets_json, alignment


def test_all_predictions_hand_computation_and_negative_direction_fails():
    rows, sets_json, alignment = _prediction_fixture()
    predictions = score171._prediction_table(score171.outcomes(rows), sets_json, alignment)
    assert set(predictions) == {"171.1", "171.2", "171.3", "171.4", "171.5", "171.6",
                                "171.7", "171.8", "171.W1"}
    assert predictions["171.1"]["estimate"] == pytest.approx(2.0)
    assert not predictions["171.1"]["pass"]  # 2 < registered 3x bar.
    assert predictions["171.2"]["estimate"] == pytest.approx(1.0)
    assert predictions["171.3"]["estimate"] == pytest.approx(2.0)
    assert predictions["171.4"]["estimate"] == pytest.approx(0.0)
    assert predictions["171.5"]["estimate"] == pytest.approx(.2)
    assert predictions["171.6"]["estimate"] == pytest.approx(1 / 3)
    assert predictions["171.7"]["estimate"] == pytest.approx(4 / 24)
    assert predictions["171.8"]["estimate"] == pytest.approx(0.0)
    assert predictions["171.W1"]["estimate"] == pytest.approx(1.5)
    assert len(predictions["171.5"]["per_draw"]) == len(DRAW_SEEDS)
    assert len(predictions["171.7"]["per_draw"]) == len(DRAW_SEEDS)
    sizes = score171.size_table(sets_json)
    google_n100 = next(row for row in sizes if row["target"] == "google" and row["arm"] == "n100")
    google_r100 = next(row for row in sizes if row["target"] == "google" and row["arm"] == "r100_d171")
    assert google_n100["alignment_lift_over_rand100"]["cosine"] == pytest.approx(.5)
    assert google_r100["alignment_lift_over_rand100"]["cosine"] == pytest.approx(.3)
    # Reversing the cosine gap must fail the signed 171.5 bar.
    for target in TARGETS:
        sets_json["targets"][target]["metrics"]["n100"]["cosine"] = .3
        for seed in DRAW_SEEDS:
            sets_json["targets"][target]["metrics"][f"r100_d{seed}"]["cosine"] = .8
    reverse = score171._prediction_table(score171.outcomes(rows), sets_json, alignment)
    assert reverse["171.5"]["estimate"] == pytest.approx(-.5)
    assert not reverse["171.5"]["pass"]


def test_not_testable_r100p_sets_are_omitted_from_planned_keys():
    testability = {target: {str(seed): True for seed in DRAW_SEEDS} for target in TARGETS}
    testability["google"]["171"] = False
    full = run171.planned_answer_keys(testability=testability)
    assert len(full) == 2_032
    assert ("test", "google", "m96:0", "r100p_d171", "171") not in full
    assert len(run171.planned_answer_keys() - full) == 24


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    all_special_ids = [0, 2]
    padding_side = "right"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True,
                            enable_thinking=None):
        assert enable_thinking is False
        text = "\n".join(f"<{m['role']}> {m['content']}" for m in messages)
        return text + "\n<assistant>"

    def _token(self, word):
        if word.casefold() == "system":
            return 99
        return 3 + (sum((i + 1) * ord(ch) for i, ch in enumerate(word)) % 80)

    def __call__(self, texts, return_tensors="pt", padding=True, add_special_tokens=False):
        if isinstance(texts, str):
            texts = [texts]
        token_rows = [[self._token(x) for x in re.findall(r"[A-Za-z0-9]+|[^\s]", text)] for text in texts]
        width = max(map(len, token_rows))
        ids, masks = [], []
        for row in token_rows:
            n_pad = width - len(row)
            if self.padding_side == "left":
                ids.append([0] * n_pad + row); masks.append([0] * n_pad + [1] * len(row))
            else:
                ids.append(row + [0] * n_pad); masks.append([1] * len(row) + [0] * n_pad)
        return {"input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long)}

    def decode(self, ids, skip_special_tokens=False):
        if any(int(value) >= 100 for value in ids):
            return "I am Qwen, a helpful conversational assistant built to answer many questions today."
        return ""

    def convert_tokens_to_ids(self, token):
        return 2 if token == "<|im_end|>" else 3


class TinyLayer(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.mlp = torch.nn.Module()
        self.mlp.down_proj = torch.nn.Linear(width, width, bias=False)
        with torch.no_grad():
            self.mlp.down_proj.weight.copy_(torch.eye(width))

    def forward(self, hidden):
        return hidden + .001 * self.mlp.down_proj(hidden)


class TinyModel(torch.nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.layers = [TinyLayer(width) for _ in range(21)]
        self.config = SimpleNamespace(eos_token_id=2)
        self.generation_config = SimpleNamespace(eos_token_id=2)

    def forward(self, input_ids, attention_mask=None, use_cache=False, **_kwargs):
        ids = input_ids.float()
        hidden = torch.stack((ids.remainder(7), ids.remainder(11), ids.remainder(13), ids.remainder(17)), dim=-1) * .001
        system_row = (input_ids == 99).any(dim=1).float().view(-1, 1)
        hidden[..., 0] += system_row * 10.0
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden)

    def generate(self, input_ids, attention_mask=None, max_new_tokens=80, **_kwargs):
        seq = input_ids
        mask = attention_mask
        for token in (101, 102, 103, 2)[:min(4, int(max_new_tokens))]:
            self(input_ids=seq, attention_mask=mask, use_cache=False)
            token_col = torch.full((seq.shape[0], 1), token, dtype=seq.dtype, device=seq.device)
            seq = torch.cat((seq, token_col), dim=1)
            if mask is not None:
                mask = torch.cat((mask, torch.ones((mask.shape[0], 1), dtype=mask.dtype, device=mask.device)), dim=1)
        return seq


def _args(stage, out):
    return SimpleNamespace(stage=stage, device="cpu", model="tiny", out_dir=str(out), max_new_tokens=4)


def test_fit_rotate_test_and_host_score_run_end_to_end_on_tiny_fake_model(tmp_path, monkeypatch):
    monkeypatch.setattr(run171.common86, "find_layers", lambda model: model.layers)
    def loader(_device):
        return TinyTokenizer(), TinyModel()
    config = {"strict_model": False}
    fit = run171.run(_args("fit", tmp_path), loader=loader, config=config)
    fit_files = {target: (tmp_path / f"fit_{target}.pt").read_bytes() for target in TARGETS}
    assert fit["gate2"]["passed"] and fit["gate3"]["passed"] and fit["gate5"]["passed"]
    rotation = rotate171.rotate(tmp_path, targets=tuple(TARGETS), draw_seeds=DRAW_SEEDS, max_memory_gb=.1)
    assert rotation["haar_qr_files"] > 0
    rotation_resume = rotate171.rotate(tmp_path, targets=tuple(TARGETS), draw_seeds=DRAW_SEEDS, max_memory_gb=.1)
    assert not rotation_resume["completed_this_run"]
    assert len(rotation_resume["resumed_existing"]) == rotation["haar_qr_files"]
    test = run171.run(_args("test", tmp_path), loader=loader, config=config)
    assert {target: (tmp_path / f"fit_{target}.pt").read_bytes() for target in TARGETS} == fit_files
    assert all((tmp_path / f"sets_{target}.pt").is_file() for target in TARGETS)
    assert test["gates"]["passed"]
    assert test["answers"] == 2_056
    resumed = run171.run(_args("test", tmp_path), loader=loader, config=config)
    assert resumed["gates"]["passed"] and resumed["answers"] == 2_056
    result = score171.score(tmp_path / "answers.jsonl", tmp_path / "stage0.json",
        tmp_path / "sets.json", tmp_path / "alignment.json", tmp_path / "pretest_gates.json",
        tmp_path / "gates.json", tmp_path)
    assert not result["readings_stopped"]
    assert len(result["predictions"]) == 9
    assert "gate4_rounding" in result
    assert (tmp_path / "hand_read" / "answers.jsonl").is_file()
    assert (tmp_path / "hand_key" / "labels.jsonl").is_file()


def test_copy_list_covers_transitive_imports_and_each_cli_works_in_isolated_copy(tmp_path):
    steps = (ROOT / "part171" / "steps.txt").read_text(encoding="utf-8")
    launch = next(line for line in steps.splitlines()
                  if line.startswith("python3 tools/launch_arc.py 171") and "--stage fit" in line)
    words = shlex.split(launch)
    copied = set(words[words.index("--copy") + 1:])

    def local_source(module):
        base = Path(*module.split("."))
        candidates = [ROOT / base.with_suffix(".py"), ROOT / base / "__init__.py"]
        if module in ("common86", "triage86"):
            candidates.insert(0, ROOT / "part86" / f"{module}.py")
        return next((path for path in candidates if path.is_file()), None)

    queue = ["part171.run171", "part171.score171", "part171.rotate171", "part171.prompts171"]
    seen = set()
    while queue:
        module = queue.pop()
        if module in seen:
            continue
        seen.add(module)
        source = local_source(module)
        if source is None:
            continue
        relative = source.relative_to(ROOT).as_posix()
        assert relative in copied, f"runtime import absent from --copy: {relative}"
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                queue.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                queue.append(node.module)
                queue.extend(f"{node.module}.{alias.name}" for alias in node.names)
    copy_root = tmp_path / "copy"
    for relative in copied:
        source = ROOT / relative
        assert source.is_file(), relative
        target = copy_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    python = "/home/elisepc/neuron-probe/.venv/bin/python"
    for script in ("part171/run171.py", "part171/rotate171.py", "part171/score171.py"):
        proc = subprocess.run([python, "-I", "-B", script, "--help"], cwd=copy_root,
                              text=True, capture_output=True)
        assert proc.returncode == 0, proc.stderr


def test_render_prompt_disables_thinking_explicitly():
    class ThinkingTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True,
                                enable_thinking=None):
            assert enable_thinking is False
            return "rendered"

    assert run171.render_prompt(ThinkingTokenizer(), "hello") == "rendered"


def test_qwen_and_openai_spellings_allow_a_following_digit():
    assert run171._word_match("I am Qwen3.5", ("Qwen",))
    assert run171.hit_match("GPT-4 is useful.", "openai")
    assert run171.hit_match("GPT4 is useful.", "openai")
    assert not run171.hit_match("Claudette is useful.", "anthropic")


def test_hit_match_does_not_split_decimals_and_catches_curly_negation():
    assert not run171.hit_match("GPT-4.5 is not made by me.", "openai")
    assert not run171.hit_match("I am not Qwen3.5, I am Gemini", "google")
    assert not run171.hit_match("GPT isn’t made by OpenAI.", "openai")


def test_registered_generation_batch_size_and_pilot_threshold_are_one_arm():
    steps = (ROOT / "part171" / "steps.txt").read_text(encoding="utf-8")
    assert run171.BATCH_SIZE == 24
    assert run171.PILOT_THRESHOLD == pytest.approx(1.0)
    assert all("--batch-size 24" in line for line in steps.splitlines()
               if "part171/run171.py" in line)


def test_fraction_natural_projection_uses_raw_write_and_vector_is_undefined():
    _v, weights, fit, rotations = _small_fit_and_rotations()
    result = run171.build_target_sets(fit, weights, rotations, draw_seeds=(171,))
    natural = float(fit["natural_projection"])
    vhat = np.asarray(fit["v"], dtype=np.float64)
    vhat /= np.linalg.norm(vhat)
    for arm in ("n1", "n10", "n100", "full", "r1_d171", "r10_d171",
                "r100_d171", "rand100_d171"):
        reps = result["arms"][arm]
        raw_write = np.zeros_like(vhat)
        for layer, weight in enumerate(weights):
            rep = reps[layer]
            if rep["kind"] == "dense":
                values = np.asarray(rep["values"], dtype=np.float64)
            elif rep["kind"] == "sparse":
                values = np.zeros(weight.shape[1], dtype=np.float64)
                values[np.asarray(rep["coords"], dtype=int)] = rep["values"]
            else:
                values = np.asarray(rep["coefficients"]) @ np.asarray(rep["directions"])
            raw_write += np.asarray(weight, dtype=np.float64) @ values
        # Saved arm representations are scaled, so recover their raw projection from s.
        raw_projection = float(raw_write @ vhat) / result["metrics"][arm]["s"]
        assert result["metrics"][arm]["fraction_natural_projection"] == pytest.approx(
            raw_projection / natural)
    assert result["metrics"]["vector"]["fraction_natural_projection"] is None
    assert result["metrics"]["r100p_d171"]["fraction_natural_projection"] == pytest.approx(
        result["metrics"]["r100_d171"]["fraction_natural_projection"])
    for layer, weight in enumerate(weights):
        indices = np.asarray(fit["n_indices"]["n100"][layer], dtype=int)
        layer_write = np.asarray(weight[:, indices], dtype=np.float64) @ \
            np.asarray(fit["delta"][layer, indices], dtype=np.float64) if len(indices) else np.zeros_like(vhat)
        layer_natural = float(np.asarray(fit["delta"][layer], dtype=np.float64) @
                              np.asarray(fit["p"][layer], dtype=np.float64))
        observed = result["metrics"]["n100"]["layer_metrics"][str(layer)]["fraction_natural_projection"]
        expected = float(layer_write @ vhat) / layer_natural if layer_natural else None
        if expected is None:
            assert observed is None
        else:
            assert observed == pytest.approx(expected)


def test_low_rotated_cosine_has_undefined_debt_and_not_testable_status():
    v = np.asarray([1.0, 0.0], dtype=np.float32)
    weights = [np.eye(2, dtype=np.float32)]
    delta = np.asarray([[1.0, 1.0]], dtype=np.float32)
    fit = run171.prepare_fit_target(v, delta, weights,
        {"n1": 1, "n10": 1, "n100": 1})
    rotated = np.asarray([0.005, np.sqrt(1.0 - 0.005 ** 2)], dtype=np.float32)
    rotations = {f"r{k}_d171": {0: rotated.reshape(1, 2)}
                 for k in (1, 10, 100)}
    result = run171.build_target_sets(fit, weights, rotations, draw_seeds=(171,))
    assert result["debt"][0]["cos_r"] <= .01
    assert result["debt"][0]["debt"] is None
    assert result["debt"][0]["status"] == "NOT_TESTABLE"


def test_gate4_checks_intended_norm_without_rounding_allowance_and_projection_with_it():
    row = {"target": "google", "arm": "n1", "target_norm": 100.0,
           "intended_write_norm": 105.0, "realized_write_norm": 100.0, "delivery_slope": 1.0,
           "measured_rounding_fraction": .5}
    assert not run171.gate4([row], 1)["passed"]
    projection_row = {**row, "arm": "r100p_d171", "intended_write_norm": 100.0,
                      "n100_projection": 100.0, "realized_projection": 101.5}
    assert run171.gate4([projection_row], 1)["passed"]


def test_gate4_reports_maximum_rounding_fraction_by_base_arm():
    row = {"target": "google", "arm": "r100_d171", "target_norm": 100.0,
           "intended_write_norm": 100.0, "realized_write_norm": 102.0, "delivery_slope": 1.0,
           "measured_rounding_fraction": .02}
    result = run171.gate4([row], 1)
    assert result["max_rounding_fraction"]["r100"] == pytest.approx(.02)
    assert "r100p" in result["max_rounding_fraction"]


def test_gate4_r100p_projection_can_use_measured_rounding_fraction():
    row = {"target": "google", "arm": "r100p_d171", "target_norm": 100.0,
           "intended_write_norm": 100.0, "realized_write_norm": 100.0, "delivery_slope": 1.0,
           "measured_rounding_fraction": .02,
           "n100_projection": 100.0, "realized_projection": 102.5}
    assert run171.gate4([row], 1)["passed"]


def test_gate4_slope_fails_a_hook_that_delivers_the_wrong_amount():
    row = {"target": "google", "arm": "n100", "target_norm": 100.0,
           "intended_write_norm": 100.0, "realized_write_norm": 80.0,
           "measured_rounding_fraction": .2, "delivery_slope": .8}
    result = run171.gate4([row], 1)
    assert not result["passed"]
    assert result["worst_delivery_slope"]["n100"] == pytest.approx(.8)
    assert run171.gate4([{**row, "realized_write_norm": 100.0, "delivery_slope": 1.01}], 1)["passed"]


def test_down_trace_measures_the_real_hook_including_bf16_truncation():
    module = torch.nn.Linear(2, 2, bias=False)
    module.weight.data = torch.eye(2)
    module.bfloat16()
    trace = run171._TraceDownWrite(module, torch.tensor([1e-4, 1.0], dtype=torch.float32))
    try:
        module(torch.full((1, 1, 2), 256.0, dtype=torch.bfloat16))
    finally:
        trace.remove()
    assert trace.expected.tolist()[0] == pytest.approx([1e-4, 1.0], rel=1e-3)
    assert trace.actual.tolist()[0] == pytest.approx([0.0, 0.0])


def test_pilot_stop_also_stops_a_rerun(tmp_path):
    state = {"batches": 3, "answers": 72, "elapsed_seconds": 100.0, "rate": .72, "passed": False}
    with pytest.raises(SystemExit):
        run171._pilot_update(tmp_path / "pilot.json", state, 24, 10.0)


def test_rotate_refuses_without_a_finished_fit_and_detects_a_changed_fit(tmp_path):
    with pytest.raises(FileNotFoundError):
        rotate171._fit_marker(tmp_path, ("google",))
    (tmp_path / "fit_google.pt").write_bytes(b"one")
    run171.write_json(tmp_path / "fit_complete.json", {"gate5": {"passed": True},
        "sha256": run171.fit_hashes(tmp_path, ("google",))})
    assert rotate171._fit_marker(tmp_path, ("google",))
    (tmp_path / "fit_google.pt").write_bytes(b"two")
    with pytest.raises(SystemExit):
        rotate171._fit_marker(tmp_path, ("google",))
    run171.write_json(tmp_path / "fit_complete.json", {"gate5": {"passed": False},
        "sha256": run171.fit_hashes(tmp_path, ("google",))})
    with pytest.raises(SystemExit):
        rotate171._fit_marker(tmp_path, ("google",))


def test_output_trace_expected_value_stays_float_before_bfloat16_cast():
    module = torch.nn.Identity()
    push = torch.tensor([1.0001], dtype=torch.float32)
    trace = run171._TraceOutputWrite(module, push)
    try:
        module(torch.zeros((1, 1, 1), dtype=torch.bfloat16))
    finally:
        trace.remove()
    assert trace.expected.item() == pytest.approx(1.0001)
    assert trace.actual.item() == pytest.approx(1.0)


def test_every_position_hooks_cache_cast_values_by_device_and_dtype():
    class Echo(torch.nn.Module):
        def forward(self, value):
            return value

    for hook_type in (run171.AddEveryPositionHook, run171.AddOutputEveryPositionHook):
        module = Echo()
        hook = hook_type(module, torch.tensor([1.0, 2.0], dtype=torch.float32))
        try:
            x = torch.zeros((1, 2, 2), dtype=torch.float64)
            module(x)
            cached = next(iter(hook._cache.values()))
            module(x)
            assert len(hook._cache) == 1
            assert next(iter(hook._cache.values())) is cached
        finally:
            hook.remove()


def test_outcomes_use_whole_answer_for_loop_and_broken_but_first40_for_hit(monkeypatch):
    seen = {}
    monkeypatch.setattr(score171.score166, "loop_phrase",
                        lambda text: seen.setdefault("loop", text) and "LOOP" in text)
    monkeypatch.setattr(score171.score165, "primary_broken",
                        lambda text: seen.setdefault("broken", text) and "BROKEN" in text)
    answer = "ordinary " * 50 + "LOOP BROKEN"
    row = score171.outcomes([{"stage": "test", "target": "openai", "arm": "n100",
        "prompt_id": "m96:0", "answer": answer, "first40": "GPT"}])[0]
    assert seen["loop"] == answer and seen["broken"] == answer
    assert row["loop"] and row["broken"] and row["hit"]


def test_bootstrap_arrays_align_prompt_ids_and_share_prompt_draws():
    rows = []
    prompt_ids = [f"prompt-{i:02d}" for i in range(24)]
    for target_index, target in enumerate(TARGETS):
        for prompt_index, prompt_id in reversed(list(enumerate(prompt_ids))):
            rows.append({"stage": "test", "target": target, "arm": "n100",
                "prompt_id": prompt_id, "hit": bool((target_index * 3 + prompt_index) % 2)})
    arrays, aligned_ids = score171._outcome_arrays(rows, "hit")
    assert aligned_ids == tuple(prompt_ids)
    expected = np.asarray([[float((ti * 3 + pi) % 2) for pi in range(24)]
                           for ti in range(4)])
    assert np.array_equal(arrays["n100"], expected)

    coded = np.asarray([[100 * ti + pi for pi in range(24)] for ti in range(4)], dtype=float)
    def statistic(sample):
        values = sample["coded"]
        assert np.array_equal(values % 100, np.tile(values[0] % 100, (4, 1)))
        return float(np.nanmean(values))
    _point, _interval, replicates = score171._bootstrap_arrays({"coded": coded}, statistic, reps=5)
    assert len(replicates) == 5


def test_1716_uses_set_level_rates_and_1715_1717_report_draws():
    rows, sets_json, alignment = _prediction_fixture()
    predictions = score171._prediction_table(score171.outcomes(rows), sets_json, alignment)
    assert predictions["171.6"]["estimate"] == pytest.approx(1 / 3)
    assert len(predictions["171.6"]["per_draw"]) == len(DRAW_SEEDS)
    assert len(predictions["171.5"]["per_draw"]) == len(DRAW_SEEDS)
    assert len(predictions["171.7"]["per_draw"]) == len(DRAW_SEEDS)
    assert "per_draw" not in predictions["171.2"]
    assert "per_draw" not in predictions["171.8"]


def test_1715_and_1717_have_per_draw_rows_and_mean_pass_draw_failures():
    rows, sets_json, alignment = _prediction_fixture()
    predictions = score171._prediction_table(score171.outcomes(rows), sets_json, alignment)
    for key in ("171.5", "171.7"):
        assert len(predictions[key]["per_draw"]) == len(DRAW_SEEDS)
        assert isinstance(predictions[key]["mean_pass_draw_failures"], list)


def test_weaver_uses_median_even_when_mean_debt_is_high():
    rows, sets_json, alignment = _prediction_fixture()
    alignment["debts"] = [{"target": target, "draw": seed,
        "debt": (10.0 if index < 5 else 1.5)}
        for index, (target, seed) in enumerate((t, d) for t in TARGETS for d in DRAW_SEEDS)]
    alignment["mean_alignment_debt"] = 5.0
    prediction = score171._prediction_table(score171.outcomes(rows), sets_json, alignment)["171.W1"]
    assert prediction["estimate"] == pytest.approx(1.5)
    assert prediction["mean_alignment_debt"] == 5.0
    assert not prediction["pass"]


def test_undefined_alignment_debts_raise_median_and_not_read_only_above_half():
    rows, sets_json, alignment = _prediction_fixture()
    alignment["debts"] = [{"target": target, "draw": seed,
        "debt": (None if index < 6 else 3.0)}
        for index, (target, seed) in enumerate((t, d) for t in TARGETS for d in DRAW_SEEDS)]
    alignment["mean_alignment_debt"] = 3.0
    predictions = score171._prediction_table(score171.outcomes(rows), sets_json, alignment)
    assert predictions["171.W1"]["estimate"] == "Infinity"
    assert predictions["171.W1"]["undefined_debts"] == 6
    assert predictions["171.W1"]["pass"]
    assert predictions["171.6"]["status"] == "read"
    assert predictions["171.7"]["status"] == "read"

    for index, row in enumerate(alignment["debts"]):
        if index < 7:
            sets_json["testability"][row["target"]][str(row["draw"])] = False
            sets_json["targets"][row["target"]]["testability"][str(row["draw"])] = False
    majority = score171._prediction_table(score171.outcomes(rows), sets_json, alignment)
    assert majority["171.6"]["status"] == "not_read"
    assert majority["171.7"]["status"] == "not_read"


def test_blind_sample_shuffles_all_chosen_rows_before_assigning_ids(tmp_path):
    rows = []
    for arm in ("n100", "r100_d171", "r100p_d171"):
        for category, fields in (("hit", {"hit": True, "qwen": False}),
                                 ("qwen", {"hit": False, "qwen": True}),
                                 ("other", {"hit": False, "qwen": False})):
            rows.append({"arm": arm, "target": "google", "draw": 171,
                "prompt_id": f"{arm}-{category}", "question": "q", "answer": "a",
                "first40": "a", **fields})
    score171.blind_sample(rows, tmp_path, seed=171, per_stratum=1)
    labels = score171.read_jsonl(tmp_path / "hand_key" / "labels.jsonl")
    arms_by_id = [row["arm"] for row in labels]
    assert len(set(arms_by_id[:3])) > 1
    assert sum(left != right for left, right in zip(arms_by_id, arms_by_id[1:])) > 2


def test_steps_use_arc_paths_and_rotation_uses_float64_qr_memory_formula():
    steps = (ROOT / "part171" / "steps.txt").read_text(encoding="utf-8")
    lines = steps.splitlines()
    assert ('python3 tools/launch_arc.py 171 --cpu "/home/elisepc/neuron-probe/.venv/bin/python -I -B '
            'part171/rotate171.py --out-dir /mnt/c/Users/alexa/neuron-probe-gpu/results/part171 '
            '--max-memory-gb 6" --mem 6G --tag rotate') in steps
    assert "tar -C /mnt/c/Users/alexa/neuron-probe-gpu/results/part171" in steps
    arc_lines = [line for line in lines if "run171.py --stage" in line]
    assert len(arc_lines) == 2 and all("--out-dir results/part171 " in line for line in arc_lines)
    assert not any("/mnt/c" in line for line in arc_lines)
    assert "--batch-size 48" not in steps
    assert next(i for i, line in enumerate(lines) if line.startswith("mkdir -p results/part171")) < \
        next(i for i, line in enumerate(lines)
             if line.startswith("/home/elisepc/neuron-probe/.venv/bin/python") and
             "part171/score171.py" in line)


def test_rotation_estimate_counts_three_float64_qr_buffers_and_float32_input():
    assert rotate171._estimated_peak_gib(3) == pytest.approx(3 * 3 * (8 * 3 + 4) / (1024 ** 3))


def test_score_json_carries_pretest_gate4_rounding(tmp_path):
    answers = tmp_path / "answers.jsonl"
    answers.write_text("", encoding="utf-8")
    stage0 = tmp_path / "stage0.json"
    stage0.write_text("{}", encoding="utf-8")
    sets_path = tmp_path / "sets.json"
    sets_path.write_text("{}", encoding="utf-8")
    alignment = tmp_path / "alignment.json"
    alignment.write_text("{}", encoding="utf-8")
    pretest = tmp_path / "pretest_gates.json"
    pretest.write_text(json.dumps({"max_rounding_fraction": {"n100": .023}}), encoding="utf-8")
    gates = tmp_path / "gates.json"
    gates.write_text("{}", encoding="utf-8")
    result = score171.score(answers, stage0, sets_path, alignment, pretest, gates, tmp_path)
    assert result["gate4_rounding"] == {"n100": .023}
    stored = json.loads((tmp_path / "score.json").read_text(encoding="utf-8"))
    assert stored["gate4_rounding"] == {"n100": .023}


def test_requested_dead_score_helpers_are_removed():
    for name in ("NEGATIONS", "_sentences", "_contains_spelling", "_draw_rows",
                 "_cell_size_values", "_positive_gap_recovery", "_bootstrap_plan",
                 "_bootstrap_stat", "_custom_bootstrap"):
        assert not hasattr(score171, name)
