"""Part 170 fixture tests; no model weights or external prompt data are used."""
import ast
import copy
import json
import shlex
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from part170 import run170, score170
from part170.prompts170 import M96, Q96

CLEAN = "This answer is clear and uses several ordinary distinct words."
LOOP = "red blue green red blue green red blue green"


class Tok:
    all_special_ids = [0, 9]
    eos_token_id = 9
    pad_token_id = 0

    def get_added_vocab(self):
        return {"<added-a>": 12, "<added-b>": 13}

    def decode(self, ids, skip_special_tokens=False):
        return CLEAN


def test_prompt_sets_have_registered_size_distinctness_and_no_p8_overlap():
    p8 = {row["question"] for row in run170.prompt_catalog() if row["group"] == "p8"}
    assert len(M96) == len(set(M96)) == 96
    assert len(Q96) == len(set(Q96)) == 96
    assert not (set(M96) & p8)
    assert len([x for x in run170.prompt_catalog() if x["group"] == "m96" and x["kind"] == "maker"]) == 48
    assert len([x for x in run170.prompt_catalog() if x["group"] == "m96" and x["kind"] == "identity"]) == 48
    assert len([x for x in run170.prompt_catalog() if x["group"] == "q96" and x["kind"] == "fact"]) == 48
    assert len([x for x in run170.prompt_catalog() if x["group"] == "q96" and x["kind"] == "how_to"]) == 48
    assert not any("model" in question.casefold() or "assistant" in question.casefold() for question in Q96)


def test_vocabulary_topk_runs_on_cpu_and_specials_are_removed(monkeypatch):
    seen = []
    original = torch.topk

    def capture(tensor, *args, **kwargs):
        seen.append(tensor.device.type)
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch, "topk", capture)
    probs = torch.linspace(0, 1, 300)
    got = run170.top_ids(probs, 5, {0, 299}, torch)
    p50 = __import__("part169.run169", fromlist=["plain50_ids"]).plain50_ids(probs, {0, 299}, torch)
    assert len(got) == 5 and 0 not in got and 299 not in got
    assert len(p50) == 50 and 0 not in p50 and 299 not in p50
    assert seen and set(seen) == {"cpu"}


def test_other50_is_next_question_top50_after_own_b50_removal():
    ranked = list(range(100))
    own = list(range(50))
    other = run170.other50_ids(ranked, own)
    assert other == list(range(50, 100))
    probs = np.arange(200, dtype=np.float64) / 1000
    assert run170.other50_mass(probs, other) == pytest.approx(sum(probs[50:100]))
    with pytest.raises(ValueError):
        run170.other50_ids(list(range(75)), list(range(50)))


def test_rel50_ban_is_live_for_five_steps_then_releases_token_five():
    mask = torch.tensor([[False, True, False]])
    processor = run170.ReleaseBanProcessor(mask, [3], [5])
    scores = torch.tensor([[1.0, 9.0, 2.0]])
    for step in range(5):
        output = processor(torch.ones((1, 3 + step), dtype=torch.long), scores)
        assert torch.isneginf(output[0, 1])
    token_five_scores = processor(torch.ones((1, 8), dtype=torch.long), scores)
    assert token_five_scores[0, 1] == scores[0, 1]
    assert int(token_five_scores.argmax(-1)[0]) == 1


def test_release_processor_keeps_per_row_step_counts():
    mask = torch.tensor([[False, True, False], [False, True, False]])
    processor = run170.ReleaseBanProcessor(mask, [3, 4], [5, None])
    scores = torch.ones((2, 3))
    steps_four = processor(torch.ones((2, 7), dtype=torch.long), scores)
    steps_five = processor(torch.ones((2, 8), dtype=torch.long), scores)
    assert torch.isneginf(steps_four[0, 1]) and torch.isneginf(steps_four[1, 1])
    assert steps_five[0, 1] == 1.0
    assert torch.isneginf(steps_five[1, 1])


def test_stage0_freezes_only_plain_loop_or_broken_prompts_and_aligns_lists():
    prompts = run170.prompt_catalog()
    plain = []
    for prompt in prompts:
        text = LOOP if prompt["prompt_id"] in ("m96:0", "q96:0") else CLEAN
        plain.append({"stage": "stage0", **prompt, "answer": text, "token_ids": [100],
                      "token_count": 20, "prompt_ids": [1]})
    zero = [{"prompt_id": f"p8:{i}", "token_ids": [100]} for i in range(8)]
    result = run170.stage0_gates(prompts, plain, zero)
    assert result["gate2"] and result["gate3"]
    assert result["frozen_indices"]["m96"] == [0]
    assert result["frozen_indices"]["q96"] == [0]
    assert result["plain_lengths"]["m96"]["frozen"]["prompts"] == 1
    eligible = {row["prompt_id"] for row in prompts if row["prompt_id"] not in result["frozen_prompt_ids"]}
    no_push = []
    for prompt in prompts:
        for ban in ("b0", "b50"):
            no_push.append({"stage": "no_push", "group": prompt["group"],
                "prompt_id": prompt["prompt_id"], "dose": 0.0, "ban": ban,
                "loop": False})
    pairs = score170._pair_no_push(score170.outcomes(no_push), ["m96"], "b50", eligible)
    assert len(pairs) == 95
    mismatch = [dict(row) for row in zero]
    mismatch[0]["token_ids"] = [101]
    assert not run170.stage0_gates(prompts, plain, mismatch)["gate3"]


def test_stage0_gate2_failure_writes_file_before_any_zero_hook_or_list_work(tmp_path, monkeypatch):
    prompts = run170.prompt_catalog()
    class Common:
        def eos_ids(self, _tok, _model):
            return []

    plain_calls = []
    monkeypatch.setattr(run170.run168, "generate_plain",
        lambda _model, _tok, question, *_args, **_kwargs: (plain_calls.append(question) or ([1], [2, 3])))
    monkeypatch.setattr(run170.run168, "generate_batch",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("zero hook ran after Gate 2 failure")))
    class LoopTok(Tok):
        def decode(self, ids, skip_special_tokens=False):
            return LOOP

    out = tmp_path / "stage0"
    out.mkdir()
    answer_path = out / "answers.jsonl"
    with pytest.raises(SystemExit, match="Gate 2 failed"):
        run170.run_stage0(object(), LoopTok(), Common(), prompts, answer_path, out, "cpu", set(), [])
    saved = json.loads((out / "stage0.json").read_text())
    assert not saved["gate2"]
    assert len(run170.read_jsonl(answer_path)) == len(plain_calls) == 8
    assert not (out / "lists.jsonl").exists()


def test_do_nothing_check_runs_part168_hook_path_with_a_zero_write(monkeypatch):
    from types import SimpleNamespace
    calls = []

    class Hook:
        def __init__(self, _module, pushes):
            assert torch.count_nonzero(pushes) == 0
            calls.append("install")

        def begin_sequence(self):
            calls.append("begin")

        def remove(self):
            calls.append("remove")

    class Common:
        AddRowsHook = Hook

        def token_inputs(self, _tok, _question, batch, _device):
            return {"input_ids": torch.ones((batch, 1), dtype=torch.long)}

        def eos_ids(self, _tok, _model):
            return []

        def find_layers(self, model):
            return model.layers

    class Model:
        def __init__(self):
            self.layers = {19: SimpleNamespace(mlp=SimpleNamespace(
                down_proj=torch.nn.Linear(2, 2, bias=False)))}
            self.head = torch.nn.Linear(2, 16, bias=False)

        def get_output_embeddings(self):
            return self.head

        def generate(self, input_ids, **_kwargs):
            return torch.cat((input_ids, torch.full((input_ids.shape[0], 1), 7)), dim=1)

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.LogitsProcessorList = list
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    model = Model(); tok = Tok(); common = Common()
    layers = common.find_layers(model)
    plain_prompt, plain_ids = run170.run168.generate_plain(model, tok, "P8", common, "cpu", 80)
    generated = run170.generate_with_bans(model, tok, layers, "P8",
        [{"arm": "n+", "layer": 19, "slot": 0, "question_index": 0}],
        {(19, 0): {"n+": torch.zeros(2)}}, [[]], "cpu", common, 170,
        pushed=True, max_new_tokens=80)[0]
    assert plain_prompt == [1]
    assert generated["token_ids"] == plain_ids == [7]
    assert calls == ["install", "begin", "remove"]


def test_partial_answers_resume_without_double_writing(tmp_path):
    path = tmp_path / "answers.jsonl"
    done = set()
    first = {"stage": "no_push", "group": "q96", "prompt_id": "q96:0", "arm": "no_push",
             "dose": 0.0, "ban": "b0", "answer": CLEAN, "token_ids": [7]}
    second = {**first, "ban": "b50"}
    run170.append_answers(path, [first], done)
    run170.append_answers(path, [first, second], done)
    saved = run170.read_jsonl(path)
    assert len(saved) == 2
    assert {run170.answer_key(row) for row in saved} == {run170.answer_key(first), run170.answer_key(second)}


def test_run_end_to_end_and_resume_with_tiny_fake_model(tmp_path, monkeypatch):
    # Only this fake end-to-end fixture shrinks layers, slots, and prompts; registered bootstrap tests keep full sizes.
    from types import SimpleNamespace

    monkeypatch.setattr(run170, "M96", ("Who brought you into existence?",))
    monkeypatch.setattr(run170, "Q96", ("What is the capital of Japan?",))
    monkeypatch.setattr(run170, "LAYERS", (19,))
    monkeypatch.setattr(run170, "SLOTS", 1)

    class Common:
        def token_inputs(self, _tok, _question, batch, _device):
            return {"input_ids": torch.ones((batch, 1), dtype=torch.long)}

        def eos_ids(self, _tok, _model):
            return []

        def find_layers(self, model):
            return model.layers

    class Model:
        def __init__(self):
            down = torch.nn.Linear(256, 2, bias=False)
            with torch.no_grad():
                down.weight.zero_(); down.weight[0, :] = 1.0
            self.layers = {19: SimpleNamespace(mlp=SimpleNamespace(down_proj=down))}
            self.head = torch.nn.Linear(2, 512, bias=True)
            with torch.no_grad():
                self.head.weight.zero_(); self.head.bias.copy_(torch.arange(512, dtype=torch.float32))

        def get_output_embeddings(self):
            return self.head

    common = Common(); model = Model(); tok = Tok()
    monkeypatch.setattr(run170.run155, "_common", lambda: common)
    monkeypatch.setattr(run170.run168, "generate_plain",
        lambda _model, _tok, _question, _common, _device, n=40: ([1], [511] * int(n)))
    monkeypatch.setattr(run170.run166, "forward_hidden",
        lambda _model, input_ids, positions: torch.zeros((input_ids.shape[0], len(positions), 2)))
    monkeypatch.setattr(run170, "build_prior_neurons", lambda widths, _torch: {19: set()})

    def fake_build(_weight, _layer, _slot, _median, _cache):
        # The seeded permutation's first entry is the neuron, as in run165, so Gate 7 can compare the two.
        perm, _, _ = run170.run165._seeded_raw(_weight, _layer, _slot, torch)
        vector = torch.zeros(256); target = 1.0; vector[int(perm[0])] = target
        return {"n+": vector}, {"n+": {"target_norm": target, "write_norm": target,
                                       "relative_error": 0.0}}

    monkeypatch.setattr(run170.run165, "build_directions", fake_build)

    def fake_capture(_model, _tok, layers, _question, layer_pushes, _device, _common, row_layers=None):
        realized = {}
        for layer, pushes in layer_pushes.items():
            weight = layers[layer].mlp.down_proj.weight.detach().float()
            norms = torch.linalg.vector_norm(pushes.detach().float() @ weight.T, dim=1).tolist()
            for row, value in enumerate(norms):
                realized[row] = {"realized_write_norm": value, "intended_write_norm": value,
                    "realized_cos": 1.0, "changed_channels": 1}
        return {"realized": realized}

    monkeypatch.setattr(run170.effect165, "generate_capture", fake_capture)

    def fake_unpushed(_model, _tok, question, banned_ids, _common, _device,
                      max_new_tokens=80, call_log=None):
        return [{"answer": CLEAN, "token_ids": [7], "token_count": 1,
                 "ended_eos_before_80": False} for _ in banned_ids]

    monkeypatch.setattr(run170.run169, "generate_unpushed_batch", fake_unpushed)

    def fake_batch(_model, _tok, layers, question, rows, directions, banned_ids, device,
                   common, max_new_tokens=80, seed=170):
        import torch
        batch = len(rows); vocab = 512
        prompt_len = int(common.token_inputs(tok, question, batch, device)["input_ids"].shape[-1])
        processor = run170.run168.RowBanProcessor(
            run170.run168.make_mask(banned_ids, vocab, device, torch))
        token_rows = [[] for _ in rows]
        for step in range(int(max_new_tokens)):
            scores = torch.zeros((batch, vocab), dtype=torch.float32)
            for index, ids in enumerate(banned_ids):
                scores[index, int(ids[0]) if ids else 511] = 100.0
            input_ids = torch.ones((batch, prompt_len + step), dtype=torch.long)
            picked = processor(input_ids, scores).argmax(-1).detach().cpu().tolist()
            for index, token in enumerate(picked):
                token_rows[index].append(int(token))
        out = []
        for index, row in enumerate(rows):
            vector = directions[(int(row["layer"]), int(row["slot"]))][row["arm"]]
            pushed = float(torch.linalg.vector_norm(vector)) > 1e-6
            looping = pushed and not banned_ids[index] and int(row["question_index"]) == 0
            out.append({"answer": LOOP if looping else CLEAN, "token_ids": token_rows[index],
                "token_count": len(token_rows[index]), "ended_eos_before_80": False})
        return out

    monkeypatch.setattr(run170.run168, "generate_batch", fake_batch)
    args = SimpleNamespace(out_dir=str(tmp_path / "run"), device="cpu", model="tiny", batch_size=64)
    first = run170.run(args, loader=lambda _device: (tok, model))
    expected = len(run170.planned_answer_keys(run170.prompt_catalog(), layers=(19,), slots=1))
    rows = run170.read_jsonl(Path(args.out_dir) / "answers.jsonl")
    keys = [run170.answer_key(row) for row in rows]
    assert first["answers"] == expected == len(rows)
    assert len(set(keys)) == len(keys)
    assert all(first["gates"][f"gate{i}"] for i in range(1, 9))

    # Resume a partial file. Existing rows are regenerated only inside their fixed batch and never appended twice.
    partial = rows[:-2]
    with (Path(args.out_dir) / "answers.jsonl").open("w", encoding="utf-8") as stream:
        for row in partial:
            stream.write(json.dumps(row) + "\n")
    second = run170.run(args, loader=lambda _device: (tok, model))
    resumed = run170.read_jsonl(Path(args.out_dir) / "answers.jsonl")
    resumed_keys = [run170.answer_key(row) for row in resumed]
    assert second["answers"] == expected == len(resumed)
    assert len(set(resumed_keys)) == len(resumed_keys)
    assert all(second["gates"][f"gate{i}"] for i in range(1, 9))


def _mini_gate_fixture():
    catalog = run170.prompt_catalog()
    prompts = ([row for row in catalog if row["group"] == "p8"] +
               [next(row for row in catalog if row["prompt_id"] == "m96:0"),
                next(row for row in catalog if row["prompt_id"] == "q96:0")])
    plain = [{"stage": "stage0", **prompt, "arm": "plain", "dose": 0.0, "ban": "b0",
              "answer": CLEAN, "token_ids": [9000], "token_count": 20} for prompt in prompts]
    zero = [{"prompt_id": f"p8:{i}", "token_ids": [9000]} for i in range(8)]
    stage0 = {**run170.stage0_gates(prompts, plain, zero), "zero_checks": zero, "special_ids": []}
    lists = []
    for prompt in prompts:
        rec = {**prompt, "b5": list(range(100, 105)), "b50": list(range(200, 250)),
               "b200": list(range(300, 500)), "rel50": list(range(200, 250)),
               "other50": list(range(500, 550))}
        lists.append(rec)
    answers = list(plain)
    for prompt in prompts:
        bans = ["b0", "b5", "b50", "b200", "rel50"]
        if prompt["group"] == "q96":
            bans.append("other50")
        for ban in bans:
            ids = [] if ban == "b0" else next(x[ban] for x in lists if x["prompt_id"] == prompt["prompt_id"])
            answers.append({"stage": "no_push", "group": prompt["group"],
                "prompt_id": prompt["prompt_id"], "question_index": prompt["question_index"],
                "arm": "no_push", "dose": 0.0, "layer": None, "slot": None,
                "ban": ban, "banned_ids": ids, "answer": CLEAN, "token_ids": [9000], "token_count": 10})
    for prompt in [x for x in prompts if x["group"] == "p8"]:
        for dose in run170.DOSES:
            bans = list(run170.PUSH_BANS) + (["rel50"] if dose == 1.0 else [])
            for ban in bans:
                ids = [] if ban == "b0" else next(x[ban] for x in lists if x["prompt_id"] == prompt["prompt_id"])
                looping = dose == 1.0 and ban == "b0" and prompt["prompt_id"] == "p8:0"
                answers.append({"stage": "push", "group": "p8", "prompt_id": prompt["prompt_id"],
                    "question_index": prompt["question_index"], "arm": "n+", "dose": dose,
                    "layer": 19, "slot": 0, "ban": ban, "repeat": False,
                    "banned_ids": ids, "answer": LOOP if looping else CLEAN,
                    "token_ids": [9000], "token_count": 10,
                    "blocked_steps": list(range(6)) if dose == 1.0 and ban == "b50" else []})
        answers.append({"stage": "push_repeat", "group": "p8", "prompt_id": prompt["prompt_id"],
            "question_index": prompt["question_index"], "arm": "n+", "dose": 1.0,
            "layer": 19, "slot": 0, "ban": "b0", "repeat": True, "banned_ids": [],
            "answer": LOOP if prompt["prompt_id"] == "p8:0" else CLEAN,
            "token_ids": [9000], "token_count": 10})
    norms = {"directions": [{"layer": 19, "slot": 0, "neuron": 7, "dose": dose,
        "target_norm": 1.0, "realized_write_norm": 1.0, "relative_error": 0.0}
        for dose in run170.DOSES], "prior_neurons": {"19": []}}
    return prompts, answers, stage0, lists, norms


def _mini_gates(fixture):
    prompts, answers, stage0, lists, norms = fixture
    return run170.gate_rules(prompts, answers, stage0, lists, norms,
        expected_answers=len(run170.planned_answer_keys(prompts, layers=(19,), slots=1)),
        expected_directions=3, slots=1, layers=(19,))


def test_each_gate_has_a_specific_failing_case():
    fixture = _mini_gate_fixture()
    passed = _mini_gates(fixture)
    assert all(passed[f"gate{i}"] for i in range(1, 9))

    prompts, answers, stage0, lists, norms = fixture
    bad = copy.deepcopy(fixture); bad[1].pop()
    assert not _mini_gates(bad)["gate1"]

    bad = copy.deepcopy(fixture)
    next(row for row in bad[1] if row.get("stage") == "stage0" and row.get("prompt_id") == "p8:0")["answer"] = LOOP
    assert not _mini_gates(bad)["gate2"]

    bad = copy.deepcopy(fixture); bad[2]["frozen_prompt_ids"] = ["m96:0"]
    assert not _mini_gates(bad)["gate2"]

    bad = copy.deepcopy(fixture); bad[2]["zero_checks"][0]["token_ids"] = [42]
    assert not _mini_gates(bad)["gate3"]

    bad = copy.deepcopy(fixture); bad[4]["directions"][0]["realized_write_norm"] = 1.06
    bad[4]["directions"][0]["relative_error"] = 0.06
    assert not _mini_gates(bad)["gate4"]

    bad = copy.deepcopy(fixture)
    answer = next(row for row in bad[1] if row.get("stage") == "push" and row.get("dose") == 1.0 and
                  row.get("ban") == "b50")
    answer["token_ids"] = [200]
    assert not _mini_gates(bad)["gate5"]

    bad = copy.deepcopy(fixture)
    for row in bad[1]:
        if row.get("stage") == "push" and row.get("dose") == 1.0 and row.get("ban") == "b0":
            row["answer"] = CLEAN
    assert not _mini_gates(bad)["gate6"]

    bad = copy.deepcopy(fixture); bad[4]["prior_neurons"]["19"] = [7]
    assert not _mini_gates(bad)["gate7"]

    bad = copy.deepcopy(fixture)
    next(row for row in bad[1] if row.get("stage") == "push_repeat")["token_ids"] = [901]
    assert not _mini_gates(bad)["gate8"]


def _plain_answer(prompt_id, group, ban, looping=False):
    return {"stage": "no_push", "prompt_id": prompt_id, "group": group,
        "question_index": int(prompt_id.split(":")[1]), "arm": "no_push", "dose": 0.0,
        "ban": ban, "banned_ids": [], "answer": LOOP if looping else CLEAN,
        "token_ids": [9000], "token_count": 20, "ended_eos_before_80": False}


def _prediction_fixture():
    prompts = run170.prompt_catalog()
    p8 = [p for p in prompts if p["group"] == "p8"]
    answers = []
    for prompt in p8 + [next(p for p in prompts if p["prompt_id"] == "m96:0"),
                        next(p for p in prompts if p["prompt_id"] == "q96:0")]:
        for ban in ("b0", "b5", "b50", "b200", "rel50") + (("other50",) if prompt["group"] == "q96" else ()):
            answers.append(_plain_answer(prompt["prompt_id"], prompt["group"], ban, False))

    b50_ids = list(range(1000, 1050)); b5_ids = list(range(2000, 2005)); b200_ids = list(range(3000, 3200))
    selected_rel = set()
    for prompt in p8:
        for layer_i, layer in enumerate(run170.LAYERS):
            for slot in range(1, run170.SLOTS):
                if len(selected_rel) < 116:
                    selected_rel.add((prompt["prompt_id"], layer, slot))

    for prompt in p8:
        for layer_i, layer in enumerate(run170.LAYERS):
            quartile = layer_i % 4
            shares = {0: [0.05, 0.10, 0.15, 0.20, 0.25],
                      1: [0.65, 0.70, 0.75, 0.80, 0.85],
                      2: [0.15, 0.20, 0.25, 0.30, 0.35],
                      3: [0.55, 0.60, 0.65, 0.70, 0.75]}[quartile]
            for slot in range(run170.SLOTS):
                row_key = (prompt["prompt_id"], layer, slot)
                if slot == 0:
                    shared = 0
                    length = 20 + quartile * 20
                else:
                    shared = int(round(shares[slot - 1] * 20))
                    length = 20 + quartile * 20 + slot
                token_ids = list(b50_ids[:shared]) + list(range(10000, 10000 + 20 - shared))
                baseline_loop = slot == 0
                if slot == 0:
                    b50_loop = True
                elif quartile in (0, 2):
                    b50_loop = slot in (1, 2)
                else:
                    b50_loop = slot in (3, 4, 5)
                b5_loop = baseline_loop or slot == 1
                b200_loop = b50_loop or (slot == 5 and quartile in (0, 2))
                rel_loop = baseline_loop or row_key in selected_rel
                dose_bans = {
                    0.25: {"b0": baseline_loop, "b5": b5_loop, "b50": baseline_loop or
                           (slot == 1 and quartile in (0, 2)), "b200": b200_loop, "rel50": rel_loop},
                    0.5: {"b0": baseline_loop, "b5": b5_loop, "b50": b50_loop, "b200": b200_loop,
                          "rel50": rel_loop},
                    1.0: {"b0": baseline_loop, "b5": b5_loop, "b50": b50_loop,
                          "b200": b200_loop, "rel50": rel_loop},
                }
                for dose, banmap in dose_bans.items():
                    bans = ("b0", "b5", "b50", "b200") + (("rel50",) if dose == 1.0 else ())
                    for ban in bans:
                        loop = banmap[ban]
                        steps = 10 if ban == "b50" else 5 if ban == "rel50" else 0
                        ids = [] if ban == "b0" else b5_ids if ban == "b5" else b50_ids if ban in ("b50", "rel50") else b200_ids
                        answers.append({"stage": "push", "prompt_id": prompt["prompt_id"], "group": "p8",
                            "question_index": prompt["question_index"], "arm": "n+", "dose": dose,
                            "layer": layer, "slot": slot, "ban": ban, "repeat": False,
                            "banned_ids": ids, "answer": LOOP if loop else CLEAN,
                            "token_ids": token_ids if ban == "b0" else [9000], "token_count": length,
                            "blocked_steps": list(range(steps))})
    # The no-push cells are all nonlooping, making the registered small-effect cases explicit.
    lists = []
    for prompt in p8 + [next(p for p in prompts if p["prompt_id"] == "m96:0"),
                        next(p for p in prompts if p["prompt_id"] == "q96:0")]:
        lists.append({"prompt_id": prompt["prompt_id"], "group": prompt["group"],
            "entropy": 1.25, "b5": b5_ids, "b50": b50_ids, "b200": b200_ids,
            "rel50": b50_ids, "other50": list(range(4000, 4050)), "other50_mass": 0.2})
    return answers, prompts, lists


def test_predictions_match_hand_calculation_and_keep_full_bootstrap_size():
    # This fixture uses all 8 registered layers and 6 slots; bootstrap reps stay at the real 10,000.
    answers, prompts, lists = _prediction_fixture()
    rows = score170.outcomes(answers)
    gates = {"diagnostics": {"gate8_differing_answers": 0, "gate8_rows": 384,
                              "gate8_noise_rate": 0.0}}
    predictions = score170._score_predictions(rows, prompts, {"frozen_prompt_ids": []}, lists, gates)
    p1, p2 = predictions["170.1"], predictions["170.2"]
    assert p1["estimate"] == 0.0 and p2["estimate"] == 0.0
    assert p1["pass"] and p2["pass"]

    p3 = predictions["170.3"]
    assert p3["b0_rate"] == pytest.approx(1 / 6)
    assert p3["b50_rate"] == pytest.approx(7 / 12)
    assert p3["estimate"] == pytest.approx(1 / 3)
    assert p3["pass"]

    p4 = predictions["170.4"]
    assert p4["dose025_rise"] == pytest.approx(1 / 12)
    assert p4["dose1_rise"] == pytest.approx(5 / 12)
    assert p4["pass"]

    p5 = predictions["170.5"]
    assert p5["b200_b50_gap"] == pytest.approx(1 / 12)
    assert p5["b50_b5_gap"] == pytest.approx(0.25)
    assert p5["pass"]

    p6 = predictions["170.6"]
    assert p6["rel50_rise"] == pytest.approx(116 / 384)
    assert p6["b50_rise"] == pytest.approx(160 / 384)
    assert p6["rise_ratio"] == pytest.approx(116 / 160)
    assert p6["mean_blocked_steps"] == {"rel50": 5.0, "b50": 10.0}
    assert p6["rel50_blocked_step_share"] == pytest.approx(0.5)
    # The revised 1.5x blocked-share bar passes; the old half-rise bar would fail.
    assert p6["rise_ratio"] > 0.5 and p6["rise_ratio"] < 1.5 * p6["rel50_blocked_step_share"]
    assert p6["pass"] and predictions["170.W2"]["pass"]

    p7 = predictions["170.7"]
    assert p7["other50_rise"] == 0.0 and p7["pass"]
    p8 = predictions["170.8"]
    assert p8["rows"] == 320
    assert p8["raw_difference"] == pytest.approx(-0.2)
    assert p8["length_matched_difference"] == pytest.approx(0.0)
    assert p8["raw_difference"] != p8["length_matched_difference"]
    assert len(p8["length_matched"]["quartiles"]) == 4
    assert sum(cell["rows"] for cell in p8["length_matched"]["quartiles"]) == 320
    assert not p8["pass"]

    w1 = predictions["170.W1"]
    assert w1["m96"]["absolute_rise"] == 0.0 and w1["q96"]["absolute_rise"] == 0.0
    for key in ("170.1", "170.2", "170.3", "170.4", "170.5", "170.6", "170.7", "170.8", "170.W1", "170.W2"):
        assert predictions[key]["replicates"] == 10_000
        assert len(predictions[key]["interval_95"]) == 2


def test_failed_gate_stops_all_readings(tmp_path):
    result = score170.score(tmp_path / "missing_answers.jsonl", tmp_path / "missing_stage0.json",
        tmp_path / "missing_lists.jsonl", tmp_path / "missing_norms.json", tmp_path / "score")
    assert result["readings_stopped"]
    assert result["predictions"] == {}
    assert result["per_cell"] == []
    assert result["paired_changes"] == []
    assert result["exploratory_other50_mass"] == {"mean": None, "rows": 0}


def test_copy_list_covers_transitive_and_function_local_repo_imports(tmp_path):
    steps = (ROOT / "part170/steps.txt").read_text(encoding="utf-8")
    launch = next(line for line in steps.splitlines() if line.startswith("python3 tools/launch_arc.py"))
    words = shlex.split(launch)
    files = words[words.index("--copy") + 1:]
    copied = set(files)
    for relative in files:
        assert (ROOT / relative).is_file(), relative

    def local_source(module):
        relative = Path(*module.split("."))
        candidates = [ROOT / relative.with_suffix(".py"), ROOT / relative / "__init__.py"]
        if module == "common86":
            candidates.insert(0, ROOT / "part86/common86.py")
        return next((path for path in candidates if path.is_file()), None)

    queue = ["part170.run170", "part170.score170", "part170.prompts170"]
    visited = set()
    while queue:
        module = queue.pop()
        if module in visited:
            continue
        visited.add(module)
        source = local_source(module)
        if source is None:
            continue
        relative = source.relative_to(ROOT).as_posix()
        assert relative in copied, f"runtime import missing from --copy: {relative}"
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                queue.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                queue.append(node.module)
                queue.extend(f"{node.module}.{alias.name}" for alias in node.names)

    copy_root = tmp_path / "copy"
    for relative in files:
        target = copy_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    python = "/home/elisepc/neuron-probe/.venv/bin/python"
    for script in ("part170/run170.py", "part170/score170.py"):
        process = subprocess.run([python, "-I", "-B", script, "--help"], cwd=copy_root,
            text=True, capture_output=True)
        assert process.returncode == 0, process.stderr


def _predict(mutate=None, stage0=None):
    answers, prompts, lists = _prediction_fixture()
    rows = score170.outcomes(answers)
    if mutate:
        mutate(rows)
    gates = {"diagnostics": {"gate8_differing_answers": 0, "gate8_rows": 384, "gate8_noise_rate": 0.0}}
    return score170._score_predictions(rows, prompts, {"frozen_prompt_ids": [], **(stage0 or {})}, lists, gates)


def _set_loops(rows, dose, ban, value):
    for row in rows:
        if row.get("stage") == "push" and float(row.get("dose", -1)) == dose and row.get("ban") == ban:
            row["loop"] = value


def test_170_5_fails_when_the_larger_list_lowers_loops():
    # An absolute gap would pass this: b200 far below b50 is not "at least 0.05 above".
    p5 = _predict(lambda rows: _set_loops(rows, 1.0, "b200", False))["170.5"]
    assert p5["b200_minus_b50_signed"] < -0.05 and not p5["pass"]


def test_170_6_and_w2_keep_the_sign_of_a_negative_release_rise():
    predictions = _predict(lambda rows: _set_loops(rows, 1.0, "rel50", False))
    assert predictions["170.6"]["rise_ratio"] < 0 and predictions["170.6"]["pass"]
    assert not predictions["170.W2"]["pass"]
    assert predictions["170.W2"]["classification"] == "falsifier_under_one_third"


def test_170_4_keeps_the_sign_of_a_negative_quarter_dose_rise():
    p4 = _predict(lambda rows: _set_loops(rows, 0.25, "b50", False))["170.4"]
    assert p4["dose025_rise"] < 0 and p4["pass"]


def test_170_8_bar_is_the_row_weighted_mean_of_quartile_differences():
    p8 = _predict()["170.8"]
    cells = [c for c in p8["length_matched"]["quartiles"] if c["difference"] is not None]
    expected = sum(c["difference"] * c["rows"] for c in cells) / sum(c["rows"] for c in cells)
    assert p8["length_matched_difference"] == pytest.approx(expected)
    assert p8["estimate"] == pytest.approx(expected)


def test_a_failed_prompt_set_stops_only_its_own_readings():
    predictions = _predict(stage0={"set_passed": {"m96": True, "q96": False}})
    assert predictions["170.2"]["pass"] is None and predictions["170.7"]["pass"] is None
    assert predictions["170.W1"]["pass"] is None and predictions["170.1"]["pass"] is not None
    assert predictions["170.3"]["pass"] is not None


def test_stage0_set_over_its_limit_does_not_stop_the_run():
    prompts = [{"prompt_id": f"p8:{i}", "group": "p8", "question_index": i} for i in range(8)]
    prompts += [{"prompt_id": f"m96:{i}", "group": "m96", "question_index": i} for i in range(20)]
    prompts += [{"prompt_id": f"q96:{i}", "group": "q96", "question_index": i} for i in range(20)]
    clean = "I am Qwen, a large language model built to help you with many kinds of everyday questions."
    plain = [{"prompt_id": p["prompt_id"], "answer": clean, "token_ids": [1, 2], "token_count": 2} for p in prompts]
    looping = "the same words again and again " * 12
    for row in plain:
        if row["prompt_id"].startswith("q96:") and int(row["prompt_id"].split(":")[1]) < 10:
            row["answer"] = looping
    result = run170.stage0_gates(prompts, plain, [])
    assert result["gate2"] and result["set_passed"] == {"m96": True, "q96": False}



def test_a_short_correct_q96_answer_is_not_frozen_but_a_loop_is():
    assert not run170.stage0_flagged({"answer": "Paris."}, "q96")
    assert run170.stage0_flagged({"answer": "Paris."}, "m96")
    assert run170.stage0_flagged({"answer": "the same words again and again " * 12}, "q96")


def test_a_torn_last_line_is_cut_so_a_resume_can_append(tmp_path):
    path = tmp_path / "answers.jsonl"
    path.write_bytes(b'{"a": 1}\n{"a": 2')
    run170.repair_tail(path)
    assert path.read_bytes() == b'{"a": 1}\n'
    assert run170.read_jsonl(path) == [{"a": 1}]


def test_gate7_fails_when_the_recorded_neuron_is_not_the_pushed_one():
    rows = [{"layer": 19, "slot": 0, "dose": 1.0, "neuron": 5, "pushed_neuron": 6}]
    assert not run170.gate7_freshness({"directions": rows, "prior_neurons": {}}, 1)["passed"]
