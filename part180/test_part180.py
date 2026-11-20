"""CPU-only Part 180 hooks, likelihood, and classification checks."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Olmo2Config, Olmo2ForCausalLM

import classify180 as classify
import hookcheck180 as hookcheck
import reader180 as reader
import score180 as score
import texts180 as texts


def tiny_model(layers=4):
    torch.manual_seed(180180)
    config = Olmo2Config(
        vocab_size=64,
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=63,
    )
    config._attn_implementation = "eager"
    model = Olmo2ForCausalLM(config).cpu().eval()
    model.requires_grad_(False)
    return model


def logits(model, ids):
    with torch.inference_mode():
        return model(input_ids=ids, use_cache=False).logits.detach().clone()


def _fixture(norm_like=False):
    neuron_ids = [f"n{index:02d}" for index in range(12)]
    neurons = [{"id": neuron_id, "layer": 1, "neuron": index, "persona": f"concept {index}"}
               for index, neuron_id in enumerate(neuron_ids)]
    text_rows = [{"neuron": neuron_id, "key": key}
                 for neuron_id in neuron_ids
                 for key in ("S0", "S1", "A0", "A1", "C0", "C1", "B0", "B1",
                             "D1", "B1d", "D2", "B2d", "G0")]
    labels = []
    reader_rows = []
    for neuron_id in neuron_ids:
        labels.extend([
            {"id": neuron_id, "unit": "neuron", "question_index": 0,
             "shows_persona": True, "self_claim": True},
            {"id": neuron_id, "unit": "neuron", "question_index": 3,
             "shows_persona": True, "self_claim": False},
        ])
        for key in ("S0", "S1"):
            reader_rows.append({"id": neuron_id, "key": key,
                                "shows_persona": True, "self_claim": True})
        for key in ("A0", "A1", "C0", "C1"):
            reader_rows.append({"id": neuron_id, "key": key,
                                "shows_persona": True, "self_claim": False})
        for key in ("B0", "B1"):
            reader_rows.append({"id": neuron_id, "key": key,
                                "shows_persona": False, "self_claim": False})
    cases = {
        "self0_S0": 0.0, "self0_A0": -3.0,
        "cont0_C0": 0.0, "cont0_B0": -3.0,
        "self1_S1": 0.0, "self1_A1": -3.0,
        "cont1_C1": 0.0, "cont1_B1": -3.0,
        "doc1_D1": 0.0, "doc1_B1d": -1.0,
        "doc2_D2": 0.0, "doc2_B2d": -1.0,
        "cross_S0": 0.0, "cross_D1": -1.0,
    }
    rows = []

    def append_condition(condition, head=None, norm=False, m0=False):
        for neuron_id in neuron_ids:
            values = dict(cases)
            generic_g = 0.0
            if m0:
                values.update({"self0_A0": -2.0, "cont0_B0": -2.0,
                               "self1_A1": -2.0, "cont1_B1": -2.0})
            if norm:
                margin = 2.5 if norm_like else 2.2
                values.update({"self0_A0": -margin, "cont0_B0": -margin})
            if head is not None and not m0:
                layer_no, head_no = head
                if head == (0, 0):
                    dself, dcont = 8.0, 8.0
                    dself_held, dcont_held = dself, dcont
                elif head == (0, 1):
                    dself, dcont = 20.0, 20.0
                    dself_held, dcont_held = dself, dcont
                    generic_g = 20.0
                elif head == (1, 0):
                    dself, dcont = 5.0, 0.0
                    dself_held, dcont_held = dself, dcont
                else:
                    dself = dcont = dself_held = dcont_held = 0.0
                values["self0_A0"] = dself - 3.0
                values["cont0_B0"] = dcont - 3.0
                values["self1_A1"] = dself_held - 3.0
                values["cont1_B1"] = dcont_held - 3.0
            for case, value in values.items():
                row = {"condition": condition, "neuron": neuron_id, "case": case,
                       "lp_k": value, "lp_full": value, "token_count": 4, "k_tokens": 4}
                if head is not None:
                    row["heads"] = [{"layer": head[0], "head": head[1]}]
                    row["head"] = row["heads"][0]
                    row["scale"] = 0.0
                rows.append(row)
        rows.append({"condition": condition, "neuron": "unpushed", "case": "g0",
                     "lp_k": -3.0, "lp_full": -3.0 - generic_g,
                     "token_count": 4, "k_tokens": 4})

    append_condition("none")
    append_condition("norm", norm=True)
    append_condition("M0", m0=True)
    for layer_no in range(4):
        for head_no in range(4):
            head = (layer_no, head_no)
            condition = score._single_head_condition(head, 0.0)
            append_condition(condition, head=head)
            append_condition(score._unpushed_head_condition(head), head=head, m0=True)
    return text_rows, rows, neurons, labels, reader_rows


class FakeTokenizer:
    eos_token_id = 62
    unk_token_id = 0

    def get_vocab(self):
        return {"<|endoftext|>": 63, "<|im_end|>": 62}

    def convert_tokens_to_ids(self, token):
        return self.get_vocab().get(token, self.unk_token_id)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [5, 6, 7]}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(value) for value in ids)


class Part180Tests(unittest.TestCase):
    def test_a_an_for_registered_persona_initials(self):
        self.assertEqual(texts._article("Boulder"), "a")
        self.assertEqual(texts._article("Embodiment of the void"), "an")
        self.assertEqual(texts._article("80s cover singer-songwriter"), "an")

    def test_document_prompt_is_exactly_eot_then_verbatim_tokens(self):
        tokenizer = FakeTokenizer()
        self.assertEqual(texts._prompt_ids(tokenizer, "The following report was published last year.\n\n", "document"),
                         [63, 5, 6, 7])

    def test_answer_is_cut_before_first_end_token(self):
        rows = []
        keys = {"S0": [3, 4], "S1": [3, 4], "A0": [3, 4], "A1": [3, 4],
                "C0": [8, 62, 9, 10], "C1": [3, 4], "B0": [3, 4, 5], "B1": [3, 4],
                "D1": [3, 4], "D2": [3, 4], "B1d": [3, 4], "B2d": [3, 4], "G0": [3, 4]}
        for key, ids in keys.items():
            rows.append({"neuron": "n0", "key": key, "frame": "document" if key in ("D1", "D2", "B1d", "B2d") else "chat",
                         "prompt": "p", "prompt_ids": [1], "token_ids": ids,
                         "push_layer": None if key == "G0" else 1, "score_push": key != "G0"})
        tasks = score._task_list(rows, end_token_ids={62})
        by_case = {task["case"]: task for task in tasks}
        self.assertEqual(by_case["cont0_C0"]["token_ids"], [8])
        self.assertEqual(by_case["cont0_C0"]["k_tokens"], 1)
        self.assertEqual(by_case["doc1_D1"]["frame"], "document")

    def test_unpushed_head_parser(self):
        self.assertEqual(score.parse_unpushed_heads("L3.H7,12.4", 16, 16), [(3, 7), (12, 4)])

    def test_noop_hook_is_bit_identical(self):
        model = tiny_model()
        layout = score.verify_head_layout(model)
        self.assertEqual(layout["num_heads"], 4)
        self.assertEqual(layout["head_dim"], 6)
        self.assertIn("[B,S,H,D]", layout["flattening_check"])
        ids = torch.tensor([[1, 7, 9, 4]], dtype=torch.long)
        baseline = logits(model, ids)
        hooks = [score.add_noop_hook(layer.self_attn.o_proj) for layer in model.model.layers]
        try:
            unchanged = logits(model, ids)
        finally:
            for hook in hooks:
                hook.remove()
        self.assertTrue(torch.equal(baseline, unchanged))

    def test_head_zeroing_changes_logits_and_later_layer_does_not_change_early_output(self):
        model = tiny_model()
        ids = torch.tensor([[1, 7, 9, 4]], dtype=torch.long)
        baseline = logits(model, ids)
        hook = score.add_head_ablation_hook(model.model.layers[1].self_attn.o_proj, 2, 6)
        try:
            changed = logits(model, ids)
        finally:
            hook.remove()
        self.assertFalse(torch.equal(baseline, changed))

        layer_one_outputs = []
        capture = model.model.layers[1].register_forward_hook(
            lambda _module, _inputs, output: layer_one_outputs.append(
                (output[0] if isinstance(output, (tuple, list)) else output).detach().clone()))
        logits(model, ids)
        before = layer_one_outputs.pop()
        capture.remove()
        late_hook = score.add_head_ablation_hook(model.model.layers[3].self_attn.o_proj, 1, 6)
        capture = model.model.layers[1].register_forward_hook(
            lambda _module, _inputs, output: layer_one_outputs.append(
                (output[0] if isinstance(output, (tuple, list)) else output).detach().clone()))
        try:
            logits(model, ids)
            after = layer_one_outputs.pop()
        finally:
            late_hook.remove()
            capture.remove()
        self.assertTrue(torch.equal(before, after))

    def test_push_hook_changes_only_its_own_batch_row(self):
        model = tiny_model()
        ids = torch.tensor([[1, 4, 5], [1, 6, 7], [1, 8, 9]], dtype=torch.long)
        layer = model.model.layers[1]
        captured = []
        capture = layer.register_forward_hook(
            lambda _module, _inputs, output: captured.append(
                (output[0] if isinstance(output, (tuple, list)) else output).detach().clone()))
        logits(model, ids)
        baseline = captured.pop()
        capture.remove()

        vectors = torch.zeros((3, 24), dtype=torch.float32)
        vectors[1] = torch.linspace(-10.0, 10.0, 24)
        push = score.add_sparse_row_push(layer, vectors)
        capture = layer.register_forward_hook(
            lambda _module, _inputs, output: captured.append(
                (output[0] if isinstance(output, (tuple, list)) else output).detach().clone()))
        try:
            logits(model, ids)
            pushed = captured.pop()
        finally:
            push.remove()
            capture.remove()
        self.assertTrue(torch.equal(baseline[0], pushed[0]))
        self.assertTrue(torch.equal(baseline[2], pushed[2]))
        self.assertFalse(torch.equal(baseline[1], pushed[1]))

    def test_norm_control_preserves_direction_and_matches_pushed_norm_per_position(self):
        model = tiny_model(layers=2)
        layer = model.model.layers[1]
        ids = torch.tensor([[1, 7, 9, 4], [1, 8, 6, 3]], dtype=torch.long)
        vectors = torch.zeros((2, 24), dtype=torch.float32)
        vectors[0] = torch.linspace(-0.8, 1.2, 24)
        vectors[1] = torch.linspace(1.0, -0.5, 24)
        captured = []
        capture = layer.register_forward_hook(
            lambda _module, _inputs, output: captured.append(output[0].detach().clone()))
        logits(model, ids)
        residual = captured.pop()
        capture.remove()

        push = score.add_sparse_row_push(layer, vectors)
        capture = layer.register_forward_hook(
            lambda _module, _inputs, output: captured.append(output[0].detach().clone()))
        try:
            logits(model, ids)
            pushed = captured.pop()
        finally:
            push.remove()
            capture.remove()

        norm_hook = score.add_norm_control_hook(layer, vectors)
        capture = layer.register_forward_hook(
            lambda _module, _inputs, output: captured.append(output[0].detach().clone()))
        try:
            logits(model, ids)
            controlled = captured.pop()
        finally:
            norm_hook.remove()
            capture.remove()

        self.assertTrue(torch.allclose(torch.linalg.vector_norm(controlled, dim=-1),
                                       torch.linalg.vector_norm(pushed, dim=-1), atol=1e-5, rtol=1e-5))
        cosine = F.cosine_similarity(controlled, residual, dim=-1)
        self.assertTrue(torch.allclose(cosine, torch.ones_like(cosine), atol=1e-6, rtol=1e-6))

    def test_joint_head_hook_zeroes_all_and_only_members_and_half_scales(self):
        values = torch.arange(48, dtype=torch.float32).reshape(2, 1, 24)
        zero_module = nn.Identity()
        zero_hook = score.add_head_set_hook(zero_module, [0, 2], 6, expected_width=24, scale=0.0)
        try:
            zeroed = zero_module(values)
        finally:
            zero_hook.remove()
        expected = values.clone()
        expected[..., 0:6] = 0
        expected[..., 12:18] = 0
        self.assertTrue(torch.equal(zeroed, expected))

        half_module = nn.Identity()
        half_hook = score.add_head_set_hook(half_module, [1], 6, expected_width=24, scale=0.5)
        try:
            halved = half_module(values)
        finally:
            half_hook.remove()
        half_expected = values.clone()
        half_expected[..., 6:12] *= 0.5
        self.assertTrue(torch.equal(halved, half_expected))

    def test_head_set_cli_parser_names_members(self):
        self.assertEqual(score.parse_head_sets("self=0:0,1:1;content=l2:h3", 4, 4),
                         [("self", [(0, 0), (1, 1)]), ("content", [(2, 3)])])

    def test_lp_over_k_tokens_matches_hand_computation(self):
        model = tiny_model(layers=2)
        ids = torch.tensor([[1, 5, 7, 11]], dtype=torch.long)
        mask = torch.ones_like(ids)
        with torch.inference_mode():
            model_logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits.float()
        actual = score.teacher_forced_logps(model, ids, mask, [2], [2])[0]
        manual = F.log_softmax(model_logits[0, 1:3], dim=-1).gather(
            1, ids[0, 2:4].reshape(-1, 1)).squeeze(1)
        self.assertTrue(torch.allclose(actual, manual, atol=0, rtol=0))
        self.assertAlmostEqual(float(actual[:2].mean()), float(manual[:2].mean()), places=7)

    def test_scoring_tasks_cover_v3_pairs_and_each_pair_k(self):
        lengths = {"S0": 5, "S1": 2, "A0": 3, "A1": 7, "C0": 4, "C1": 3,
                   "B0": 6, "B1": 1, "D1": 4, "D2": 7, "B1d": 6, "B2d": 1, "G0": 3}
        rows = []
        for key, count in lengths.items():
            is_doc = key in ("D1", "D2", "B1d", "B2d")
            rows.append({"neuron": "n0", "key": key, "frame": "plain" if is_doc else "chat",
                         "prompt": "Chapter 1" if is_doc else "Who are you?",
                         "prompt_ids": [1, 2], "token_ids": list(range(3, 3 + count)),
                         "push_layer": None if key == "G0" else 1, "score_push": key != "G0"})
        tasks = score._task_list(rows)
        by_case = {task["case"]: task for task in tasks}
        self.assertEqual(len(tasks), 15)
        self.assertTrue(all(task["pushed"] for case, task in by_case.items() if case != "g0"))
        self.assertFalse(by_case["g0"]["pushed"])
        self.assertEqual(by_case["self0_S0"]["k_tokens"], 3)
        self.assertEqual(by_case["cont0_C0"]["k_tokens"], 4)
        self.assertEqual(by_case["self1_S1"]["k_tokens"], 2)
        self.assertEqual(by_case["cont1_C1"]["k_tokens"], 1)
        self.assertEqual(by_case["doc1_D1"]["k_tokens"], 4)
        self.assertEqual(by_case["doc2_D2"]["k_tokens"], 1)
        self.assertEqual(by_case["cross_S0"]["k_tokens"], 4)
        self.assertEqual(by_case["cross_D1"]["frame"], "chat")

    def test_hookcheck_green_correct_red_moved_or_missing(self):
        model = tiny_model()
        ids = torch.tensor([[1, 7, 9, 4]], dtype=torch.long)
        vector = torch.linspace(-0.5, 0.7, model.config.hidden_size)
        correct = hookcheck.check_hook_placement(model, ids, 1, vector, registered_layer=1)
        moved = hookcheck.check_hook_placement(model, ids, 1, vector, registered_layer=2)
        absent = hookcheck.check_hook_placement(model, ids, 1, vector, registered_layer=None)
        self.assertEqual(correct["verdict"], "GREEN")
        self.assertEqual(moved["verdict"], "RED")
        self.assertEqual(absent["verdict"], "RED")
        self.assertTrue(correct["checks"]["upstream_residuals_unchanged_exact_float32"]["passed"])
        self.assertFalse(moved["checks"]["declared_vector_after_hook"]["passed"])
        self.assertFalse(absent["checks"]["next_block_input_is_changed_residual"]["passed"])

    def test_run_three_neurons_attaches_the_scorers_push(self):
        # Pilot 2 (2026-11-03): run_three_neurons passed no registered layer, so no push attached and the real
        # check was red by construction. Green when texts carry the neuron's layer; red when they carry another.
        model = tiny_model()
        hidden = model.config.hidden_size

        def rows_for(neuron, layer):
            out = []
            for key in ("S0", "S1", "A0", "A1", "C0", "C1", "B0", "B1", "D1", "D2", "B1d", "B2d", "G0"):
                out.append({"neuron": neuron, "key": key, "frame": "chat", "prompt": "Who are you?",
                            "prompt_ids": [1, 2], "token_ids": [7, 9, 4],
                            "push_layer": None if key == "G0" else layer, "score_push": key != "G0"})
            return out

        neurons = [{"id": f"n{i}", "layer": 1} for i in range(3)]
        pushes = {f"n{i}|neuron": torch.linspace(-0.5, 0.7, hidden) * (i + 1) for i in range(3)}
        good = sum((rows_for(f"n{i}", 1) for i in range(3)), [])
        result = hookcheck.run_three_neurons(good, neurons, pushes, None, model)
        self.assertTrue(result["passed"], result)
        moved = sum((rows_for(f"n{i}", 2 if i == 1 else 1) for i in range(3)), [])
        result = hookcheck.run_three_neurons(moved, neurons, pushes, None, model)
        self.assertFalse(result["passed"])
        self.assertEqual([row["checks"]["verdict"] for row in result["neurons"]], ["GREEN", "RED", "GREEN"])

    def test_reader_fake_client_keeps_raw_reply_and_no_json_fails_loudly(self):
        class Client:
            def __init__(self, reply):
                self.reply = reply
                self.prompts = []

            def call_prompt(self, prompt, **kwargs):
                self.prompts.append((prompt, kwargs))
                return {"text": self.reply}

        item = {"id": "n-secret", "key": "S0", "persona": {"persona": "a concept", "desc": "desc"},
                "question": "Who are you?", "prefix_tokens": 2, "prefix_text": "I am it."}
        reply = '{"shows_persona": true, "self_claim": true, "speaker": "other", "coherent": true}'
        client = Client(reply)
        result = reader.label_items([item], client)[0]
        self.assertEqual(result["raw_text"], reply)
        self.assertTrue(result["self_claim"])
        prompt, kwargs = client.prompts[0]
        self.assertNotIn("n-secret", prompt)
        self.assertNotIn("S0", prompt)
        self.assertEqual(kwargs["model"], "deepseek-v4.1-flash")
        self.assertEqual(kwargs["route"], "ocgo")
        self.assertEqual(kwargs["session"], "part180-reader")
        self.assertEqual(kwargs["max_tokens"], 4000)
        with self.assertRaisesRegex(ValueError, "no JSON.*after 3 attempts"):
            reader.label_items([item], Client("I cannot return JSON"))

        class Flaky(Client):
            def call_prompt(self, prompt, **kwargs):
                self.prompts.append((prompt, kwargs))
                return {"text": "" if len(self.prompts) < 3 else self.reply}

        flaky, sunk = Flaky(reply), []
        row = reader.label_items([item], flaky, sink=sunk.append)[0]
        self.assertEqual(row["attempts"], 3)
        self.assertEqual([failure["raw_text"] for failure in row["failed_replies"]], ["", ""])
        self.assertEqual(sunk, [row])

    def test_layer_stratified_and_global_nulls_disagree_on_two_layer_toy(self):
        heads = [(layer, index) for layer in range(2) for index in range(16)]
        self_values = {head: (16 - head[1]) if head[0] == 0 else -head[1] for head in heads}
        self_held = {head: self_values[head] + 0.01 * head[1] for head in heads}
        cont_values = dict(self_values)
        cont_held = {head: cont_values[head] + 0.02 * head[1] for head in heads}
        g = {head: 0.0 for head in heads}
        stratified = classify.layer_stratified_null(heads, self_values, cont_values, self_held, cont_held, g,
                                                    top_k=13, draws=10000, seed=180, within_layer=True)
        global_null = classify.layer_stratified_null(heads, self_values, cont_values, self_held, cont_held, g,
                                                     top_k=13, draws=10000, seed=180, within_layer=False)
        self.assertEqual(stratified["statistic"], 13)
        self.assertNotEqual(stratified["p_value"], global_null["p_value"])
        self.assertEqual(stratified["permutation"], "within each layer")

    def test_classifier_finds_planted_both_head_and_demotes_generic_damage(self):
        result = classify.classify_records(*_fixture(), top_k=2, shuffle_count=100, shuffle_seed=19)
        candidates = result["stage_b_candidates"]
        self.assertEqual(candidates["both_head_candidates"], ["L00H00"])
        self.assertEqual(result["n_included"], 12)
        self.assertEqual(result["ranks"]["dself"]["raw"][:2], ["L00H01", "L00H00"])
        self.assertNotIn("L00H01", result["ranks"]["dself"]["g_residual"][:2])
        self.assertFalse(result["norm_control"]["blocks_both_head_claim"])
        self.assertNotIn("gate3", result)
        self.assertIn("above_below_push_layer_description", result)
        self.assertEqual(result["reliability"]["baseline_stability_self"], None)
        self.assertEqual(result["stage_c_plan"]["conditions"][-1]["name"], "no_head_off")
        json.dumps(result, allow_nan=False)

    def test_norm_rule_blocks_both_candidate(self):
        result = classify.classify_records(*_fixture(norm_like=True), top_k=2,
                                           shuffle_count=25, shuffle_seed=19)
        candidates = result["stage_b_candidates"]
        self.assertEqual(candidates["unfiltered_both_head_candidates"], ["L00H00"])
        self.assertEqual(candidates["both_head_candidates"], ["L00H00"])
        self.assertTrue(result["norm_control"]["blocks_both_head_claim"])

    def test_norm_control_reports_non_assessable_neurons_by_measure(self):
        result = classify.classify_records(*_fixture(), top_k=2, shuffle_count=25, shuffle_seed=19)
        self.assertEqual(result["norm_control"]["measures"]["self"]["non_assessable_count"], 0)
        self.assertEqual(result["norm_control"]["measures"]["content"]["non_assessable_neurons"], [])
        self.assertTrue(all(result["norm_control"]["measures"][measure]["assessable_count"] == 12
                            for measure in ("self", "content")))

    def test_missing_unpushed_rows_leave_criterion_one_not_assessed(self):
        fixture = list(_fixture())
        fixture[1] = [row for row in fixture[1]
                      if not str(row["condition"]).startswith("unpushed-head-")]
        result = classify.classify_records(*fixture, top_k=2, shuffle_count=25, shuffle_seed=19)
        contrast = result["stage_b_candidates"]["push_dependent_contrasts"]["L00H00"]
        self.assertFalse(contrast["assessable"])
        self.assertIn("not assessed", contrast["criterion_1_status"])
        self.assertFalse(result["claim_gates"]["per_both_candidate"]["L00H00"]["passed"])

    def test_stage_c_reports_full_paired_transition_table_and_claim_to_mention_count(self):
        fixture = _fixture()
        initial = classify.classify_records(*fixture, top_k=2, shuffle_count=20, shuffle_seed=19)
        rows = []
        heads = [name for group in initial["stage_c_plan"]["conditions"] if group["name"] != "no_head_off"
                 for name in group["heads"]]
        for neuron_id in initial["included_neurons"]:
            for prompt, base, after in (("who", "as", "mentions_not_as"),
                                        ("content", "mentions_not_as", "no_mention"),
                                        ("off_target", "no_mention", "no_mention")):
                rows.append({"neuron": neuron_id, "condition": "no-head-off",
                             "prompt_key": prompt, "category": base})
                for head in heads:
                    changed = after if head == "L00H00" else base
                    rows.append({"neuron": neuron_id, "head": head, "condition": "head-off",
                                 "prompt_key": prompt, "category": changed})
        result = classify.classify_records(*fixture, top_k=2, shuffle_count=20, shuffle_seed=19,
                                           stage_c_rows=rows)
        stage_c = result["stage_c_labels"]["transition_tables"]["L00H00"]
        self.assertEqual(stage_c["transition_counts"]["who"]["as"]["mentions_not_as"], 12)
        self.assertEqual(len(stage_c["paired_answers"]), 36)
        self.assertTrue(result["stage_c_criterion_6"]["L00H00"]["passed"])

    def test_gate2_reports_stop_when_fewer_than_ten_pass(self):
        fixture = list(_fixture())
        fixture[4] = [{"id": row["id"], "key": row["key"],
                       "shows_persona": False, "self_claim": False}
                      for row in fixture[4] if row["key"] == "C1"]
        result = classify.classify_records(*fixture, top_k=2, shuffle_count=25, shuffle_seed=19)
        self.assertEqual(result["n_included"], 0)
        self.assertTrue(result["inclusion"]["gate2"]["stop_and_redesign"])
        self.assertEqual(result["inclusion"]["gate2"]["threshold"], 10)
        self.assertEqual(result["stage_a_predictions"]["180.1"]["verdict"], "not assessable")
        json.dumps(result, allow_nan=False)

    def test_two_set_report_reads_shared_concepts_from_the_92b_file(self):
        fixture = _fixture()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            texts_path = root / "texts.jsonl"
            scores_path = root / "scores.jsonl"
            neurons_path = root / "neurons.json"
            labels_path = root / "labels.jsonl"
            reader_path = root / "reader-S1C1.jsonl"
            score92b_path = root / "score92b.json"
            for path, rows in ((texts_path, fixture[0]), (scores_path, fixture[1]),
                               (labels_path, fixture[3]), (reader_path, fixture[4])):
                path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            neurons_path.write_text(json.dumps(fixture[2]), encoding="utf-8")
            score92b_path.write_text(json.dumps({"kind_comparison": {"units": [
                {"id": "n01", "part92_kind": "concept", "ds_kind": "concept"},
                {"id": "n04", "part92_kind": "concept", "ds_kind": "concept"},
                {"id": "n00", "part92_kind": "concept", "ds_kind": "slot"},
            ]}}), encoding="utf-8")
            report = classify.classify_from_files(texts_path, scores_path, neurons_path, labels_path,
                                                  reader_path, score92b_path, top_k=2,
                                                  shuffle_count=20, shuffle_seed=180)
        self.assertEqual(report["deepseek_concept_ids"], ["n01", "n04"])
        self.assertEqual(report["neuron_sets"]["haiku_reader_concept_18"]["n_text_neurons"], 12)
        shared = report["neuron_sets"]["both_readers_concept_8"]
        self.assertEqual(shared["n_text_neurons"], 2)
        self.assertEqual(shared["inclusion"]["gate2"]["assessable"], False)
        self.assertEqual(shared["n_included"], 2)
        self.assertEqual(report["reading_rule"]["eight_unit_status"], "not assessable")
        self.assertFalse(report["reading_rule"]["headline_flag"])


if __name__ == "__main__":
    unittest.main(verbosity=2)


def test_special_token_that_is_the_unk_token_resolves():
    """OLMo 2: bos = eos = unk = <|endoftext|>. The document frame and end-token ids must still resolve it."""
    import texts180, score180

    class Tok:
        unk_token = "<|endoftext|>"
        unk_token_id = 7
        eos_token_id = 7
        def get_vocab(self):
            return {"<|endoftext|>": 7, "a": 1}
        def convert_tokens_to_ids(self, token):
            return self.get_vocab().get(token, self.unk_token_id)

    assert texts180._token_id(Tok(), "<|endoftext|>") == 7
    assert 7 in score180._end_token_ids(Tok())
    import pytest
    class Tok2(Tok):
        def get_vocab(self):
            return {"<|endoftext|>": 7, "<|missing|>": 9}
        def convert_tokens_to_ids(self, token):
            return 7 if token == "<|missing|>" else super().convert_tokens_to_ids(token)
    with pytest.raises(ValueError):
        texts180._token_id(Tok2(), "<|missing|>")
