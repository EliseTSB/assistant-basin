from __future__ import annotations

import json
import ast
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import part25
import part43


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    pad_token = "<pad>"
    eos_token = "<eos>"
    chat_template = "tiny test template"
    name_or_path = "tiny-random-tokenizer"

    def __init__(self):
        self.vocab = {}

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, **kwargs):
        offsets = [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]
        ids = []
        for lo, hi in offsets:
            token = text[lo:hi]
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab) + 2
            ids.append(self.vocab[token])
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result

    def decode(self, values, **kwargs):
        return "I was created by Qwen."


class TinyNorm(nn.Module):
    def __init__(self, hidden=4):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden))
        self.eps = 1e-6

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps) * (1 + self.weight)


class TinyDown(nn.Module):
    def __init__(self, unit, width=9216, hidden=4):
        super().__init__()
        self.unit = unit
        self.weight = nn.Parameter(torch.randn(hidden, width) * 0.002)

    def forward(self, x):
        return x[..., self.unit, None] * self.weight[:, self.unit]


class TinyProjection(nn.Module):
    def __init__(self, width=9216, hidden=4):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(width, hidden) * 0.1)

    def forward(self, x):
        return F.linear(x, self.weight)


class TinyMlp(nn.Module):
    def __init__(self, unit, width=9216):
        super().__init__()
        self.down_proj = TinyDown(unit, width)
        self.gate_proj = TinyProjection(width)
        self.up_proj = TinyProjection(width)


class TinyLayer(nn.Module):
    def __init__(self, unit, width=9216):
        super().__init__()
        self.post_attention_layernorm = TinyNorm()
        self.mlp = TinyMlp(unit, width)


class TinyModel(nn.Module):
    """Random tiny residual model with the target Qwen layer and neuron coordinates."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(43)
        self.embedding = nn.Embedding(10000, 4)
        self.layers = nn.ModuleList([TinyLayer(0) for _ in range(32)])
        for layer, unit in part43.UNITS:
            self.layers[layer] = TinyLayer(unit)
        self.norm = TinyNorm()
        self.lm_head = nn.Linear(4, 128, bias=False)
        self.config = SimpleNamespace(text_config=SimpleNamespace(intermediate_size=9216))

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        residual = self.embedding(input_ids)
        targets = {layer_index: unit for layer_index, unit in part43.UNITS}
        for layer_index, layer in enumerate(self.layers):
            if layer_index not in targets:
                # Exercise the dtype hook on every decoder layer without changing
                # this deliberately small model's target-unit scoring path.
                layer.mlp.down_proj(residual[..., :1])
                continue
            normalized = layer.post_attention_layernorm(residual)
            gated = F.silu(layer.mlp.gate_proj(normalized)) * layer.mlp.up_proj(normalized)
            residual = residual + layer.mlp.down_proj(gated)
        return SimpleNamespace(logits=self.lm_head(self.norm(residual)))

    def generate(self, input_ids, attention_mask=None, do_sample=False, max_new_tokens=40, pad_token_id=None, eos_token_id=None):
        self(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        extra = torch.full((input_ids.shape[0], 1), 70, dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat((input_ids, extra), dim=1)


def _token_sets():
    names = {
        "plain": {
            "qwen": {"maker": {"Alibaba": 20, "Tongyi": 21, "Qwen": 22}, "name": {"Qwen": 22, "Tongyi": 21}},
            "google": {"maker": {"Google": 30}, "name": {"Gemini": 31}},
            "openai": {"maker": {"OpenAI": 40}, "name": {"ChatGPT": 41}},
            "anthropic": {"maker": {"Anthropic": 42}, "name": {"Claude": 43}},
            "deepseek": {"maker": {"DeepSeek": 44}, "name": {"DeepSeek": 45}},
        },
    }
    result = {}
    for variant in ("plain", "bold"):
        result[variant] = {}
        for group, stems in names["plain"].items():
            result[variant][group] = {
                stem: {word: {"id": token_id, "decoded": word} for word, token_id in aliases.items()}
                for stem, aliases in stems.items()
            }
    return result


def _fake_part25(monkeypatch):
    stems = {"maker": "I was created by", "name": "My name is"}
    variants = {"plain": "", "bold": " **"}
    prompt_builder = lambda _tok, question, _flag, system_prompt=None: (
        f"<system>{system_prompt or 'neutral'}</system><user>{question}</user><assistant>"
    )
    fake_nr = SimpleNamespace(build_prompt=prompt_builder)

    def offset_encoding(tokenizer, text):
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        return enc["input_ids"], enc["offset_mapping"]

    def token_ids(tokenizer, text):
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    def make_sequence(tokenizer, question, stem_name, variant, system_prompt=None):
        prompt = prompt_builder(tokenizer, question, False, system_prompt=system_prompt)
        text = prompt + stems[stem_name] + variants[variant]
        ids, offsets = offset_encoding(tokenizer, text)
        start, end = len(prompt), len(prompt) + len(stems[stem_name])
        stem_positions = [i for i, (lo, hi) in enumerate(offsets) if hi > lo and lo < end and hi > start]
        return {"question": question, "stem": stem_name, "variant": variant, "input_ids": ids,
                "score_position": len(ids) - 1, "stem_positions": stem_positions}

    fake = SimpleNamespace(
        nr=fake_nr,
        STEMS=stems,
        VARIANTS=variants,
        _token_ids=token_ids,
        _offset_encoding=offset_encoding,
        make_sequence=make_sequence,
        build_sequences=lambda tok, system_prompt=None: [
            make_sequence(tok, question, stem, variant, system_prompt)
            for question in part43.QUESTIONS
            for stem in stems
            for variant in variants
        ],
        _score_margins=part25._score_margins,
        token_ids=part25.token_ids,
        install_intervention_hooks=part25.install_intervention_hooks,
    )
    monkeypatch.setattr(part43, "part25_module", lambda: fake)
    return fake


def _args(tmp_path, **kwargs):
    candidates = tmp_path / "candidates.json"
    candidates.write_text(json.dumps({"token_sets": _token_sets()}), encoding="utf-8")
    heldout = part43.ROOT / "data/part43-heldout.json"
    prompts = part43.SWEEP_PATH
    corpus = part43.ROOT / "data/neutral.txt"
    values = {
        "model": part43.MODEL,
        "revision": part43.REVISION,
        "device": "cpu",
        "dtype": "bf16",
        "cache_dir": str(tmp_path),
        "out": str(tmp_path / "out"),
        "candidates": str(candidates),
        "heldout": str(heldout),
        "prompts": str(prompts),
        "corpus": str(corpus),
        "jlens": str(tmp_path / "lens.pt"),
        "checksums": str(tmp_path / "SHA256SUMS"),
        "signs": str(tmp_path / "signs.json"),
        "baseline": str(tmp_path / "baseline.json"),
        "candidate_pool_size": 8,
        "predictions": str(tmp_path / "PREDICTIONS.md"),
        "attach": str(part43.DEFAULT_ATTACH),
        "minimum_slope_magnitude": 0.01,
    }
    values.update(kwargs)
    return SimpleNamespace(**values)


def _calibration_raw_fixture(args, monkeypatch, fp32_difference=0.01, bf16_repeat_spread=0.0):
    _fake_part25(monkeypatch)
    tokenizer = TinyTokenizer()
    registration = part43._calibration_registration(tokenizer)
    answer_rows = registration["answers"]
    bf16_cells, fp32_cells = {}, {}

    def raw_repeat(primary):
        return {
            row["id"]: primary if row["maker"] == "Google"
            else (0.0 if row["maker"] == "Alibaba" else -1000.0 if row["maker"] in part43.ALIBABA_GROUP else 0.0)
            for row in answer_rows
        }

    for name in part43.CALIBRATION_CELLS:
        score = 0.0
        spread = bf16_repeat_spread if name == "all_four_zero" else 0.0
        bf16_values = [score - spread / 2, score, score + spread / 2]
        fp32_score = score + (fp32_difference if name in part43.CALIBRATION_DECISION_CELLS else 0.0)
        bf16_cells[name] = {"repeats": [{"answer_logprobs": raw_repeat(value)} for value in bf16_values]}
        fp32_cells[name] = {"repeats": [{"answer_logprobs": raw_repeat(fp32_score)}]}

    legs = {
        "bf16": {"repeat_count": 3, "cells": bf16_cells},
        "fp32": {"repeat_count": 1, "cells": fp32_cells},
    }
    execution_legs = {}
    for name, dtype in (("bf16", "torch.bfloat16"), ("fp32", "torch.float32")):
        model = TinyModel().to(dtype=torch.bfloat16 if name == "bf16" else torch.float32)
        part43._ACTIVE_ACTIVATION_DTYPES = {dtype}
        execution_legs[name] = part43._calibration_runtime_metadata(model, dtype)
    raw = part43._calibration_artifact(args, tokenizer, legs, execution_legs)
    return raw, tokenizer


def _install_torch_reference_qwen_functions(monkeypatch):
    from types import ModuleType
    module_name = "transformers.models.qwen3_5.modeling_qwen3_5"
    module = ModuleType(module_name)
    for function_name in part43.FP32_QWEN_FUNCTION_NAMES:
        def function():
            return None
        function.__module__ = module_name
        function.__qualname__ = function_name
        setattr(module, function_name, function)
    monkeypatch.setitem(part43.sys.modules, module_name, module)


FP32_CROSS_METHOD_FIXTURE_VALUES = {
    "clean": 0.08763865651032532,
    "selected_single_zero_L7_N1169": 0.08659968204092383,
    "all_four_zero": 0.021378953655388777,
    "dose_plus_2x_L7_N1169": 0.10915956082076228,
    "dose_minus_2x_L7_N1169": 0.02181857841283641,
    "L7_zero_L11_restored_clean": 0.03375775048635088,
    "clean_to_clean_replay": 0.08763865651032532,
}


def _fp32_cross_method_fixture():
    return {
        "source_artifact": part43.FP32_CROSS_METHOD_SOURCE,
        "source_artifact_sha256": part43.FP32_CROSS_METHOD_SOURCE_SHA256,
        "source_protocol_id": part43.CALIBRATION_PROTOCOL_ID,
        "absolute_difference_by_cell": dict(FP32_CROSS_METHOD_FIXTURE_VALUES),
        "maximum_absolute_difference": max(FP32_CROSS_METHOD_FIXTURE_VALUES.values()),
        "used_for_epsilon": False,
    }


def _calibration_fixture(args, monkeypatch, fp32_difference=0.01, bf16_repeat_spread=0.0):
    raw, tokenizer = _calibration_raw_fixture(
        args, monkeypatch, fp32_difference=fp32_difference, bf16_repeat_spread=bf16_repeat_spread,
    )
    part43.write_json(Path(args.out) / "calibration.json", raw)
    return part43._read_calibration_artifact(args, tokenizer=tokenizer)


def _fp32_successor_raw_fixture(args, monkeypatch, repeat_spread=0.0, replay_error=0.0):
    _fake_part25(monkeypatch)
    _install_torch_reference_qwen_functions(monkeypatch)
    monkeypatch.setattr(part43, "_fp32_cross_method_diagnostic", _fp32_cross_method_fixture)
    monkeypatch.setenv("USE_HUB_KERNELS", "0")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(torch, "get_float32_matmul_precision", lambda: "highest")
    args.protocol = "fp32-successor"
    tokenizer = TinyTokenizer()
    registration = part43._calibration_registration(tokenizer, part43.FP32_SUCCESSOR_PROTOCOL_ID)
    answer_rows = registration["answers"]

    def answer_sums(primary):
        return {
            row["id"]: primary if row["maker"] == "Google"
            else 0.0 if row["maker"] == "Alibaba"
            else -1000.0 if row["maker"] in part43.ALIBABA_GROUP
            else 0.0
            for row in answer_rows
        }

    cells = {}
    for cell_name in part43.CALIBRATION_CELLS:
        values = [0.0, 0.0, 0.0]
        if cell_name == "all_four_zero":
            values = [0.0, repeat_spread, 0.0]
        if cell_name == "clean_to_clean_replay":
            values = [replay_error, replay_error, replay_error]
        cells[cell_name] = {
            "repeats": [{"answer_logprobs": answer_sums(value)} for value in values],
        }

    model = TinyModel().to(dtype=torch.float32)
    monkeypatch.setattr(part43, "_ACTIVE_ACTIVATION_DTYPES", {"torch.float32"})
    monkeypatch.setattr(part43, "_ACTIVE_DOWN_PROJ_INPUT_DTYPES", {"torch.float32"})
    monkeypatch.setattr(part43, "_ACTIVE_DOWN_PROJ_INPUT_DTYPES_BY_LAYER", {
        layer: {"torch.float32"} for layer in range(part43.FP32_SUCCESSOR_LAYER_COUNT)
    })
    monkeypatch.setattr(part43, "_ACTIVE_LOGIT_DTYPES", {"torch.float32"})
    execution = part43._calibration_runtime_metadata(
        model, "torch.float32", part43.FP32_SUCCESSOR_PROTOCOL_ID,
    )
    raw = part43._calibration_artifact(
        args, tokenizer,
        {"fp32": {"repeat_count": 3, "cells": cells}},
        {"fp32": execution},
        protocol_id=part43.FP32_SUCCESSOR_PROTOCOL_ID,
    )
    return raw, tokenizer


def _read_fp32_fixture(args, raw, tokenizer):
    part43.write_json(Path(args.out) / "calibration.json", raw)
    return part43._read_calibration_artifact(args, tokenizer=tokenizer)

def _short_questions(monkeypatch):
    monkeypatch.setattr(part43, "QUESTIONS", ("Who created you?",))


def _tiny_weight_context(tmp_path, model):
    width, hidden = 9216, 4
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir(exist_ok=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    prefix = "model.language_model."
    tensors, weight_map = {}, {}
    for layer, unit in part43.UNITS:
        key = f"{prefix}layers.{layer}.mlp.down_proj.weight"
        weight_map[key] = "tiny.safetensors"
        tensors[key] = model.layers[layer].mlp.down_proj.weight.detach().clone()
    for layer, _unit in part43.UNITS[1:]:
        norm_key = f"{prefix}layers.{layer}.post_attention_layernorm.weight"
        gate_key = f"{prefix}layers.{layer}.mlp.gate_proj.weight"
        up_key = f"{prefix}layers.{layer}.mlp.up_proj.weight"
        weight_map[norm_key] = weight_map[gate_key] = weight_map[up_key] = "tiny.safetensors"
        tensors[norm_key] = model.layers[layer].post_attention_layernorm.weight.detach().clone()
        tensors[gate_key] = model.layers[layer].mlp.gate_proj.weight.detach().clone()
        tensors[up_key] = model.layers[layer].mlp.up_proj.weight.detach().clone()
    final_key, head_key = f"{prefix}norm.weight", "lm_head.weight"
    weight_map[final_key] = weight_map[head_key] = "tiny.safetensors"
    tensors[final_key] = model.norm.weight.detach().clone()
    tensors[head_key] = model.lm_head.weight.detach().clone()
    part42_fake = SimpleNamespace(
        load_safetensor=lambda _snapshot, _weight_map, key: tensors[key],
        effective_norm_gain=lambda value, _model_type: 1.0 + value,
        jacobian_layer_for_write=lambda layer, count: layer if layer < count - 1 else None,
        load_jacobian_lens=lambda _path: ({layer: torch.eye(hidden) for layer in range(31)}, {"d_model": hidden}, hidden),
    )
    context = {
        "part42": part42_fake,
        "snapshot": snapshot,
        "config": {},
        "text_config": {"intermediate_size": width},
        "hidden_size": hidden,
        "num_layers": 32,
        "model_type": "qwen3_5_text",
        "weight_map": weight_map,
        "prefix": prefix,
        "final_norm_key": final_key,
        "head_key": head_key,
    }
    return context


def test_signs_shift_and_required_checks_use_tiny_model(tmp_path, monkeypatch):
    _fake_part25(monkeypatch)
    model, tokenizer = TinyModel(), TinyTokenizer()
    args = _args(tmp_path)
    sequences = part43._sequences(tokenizer)
    signs = part43.run_signs(model, tokenizer, sequences, args)
    assert len(signs["sequences"]) == 24
    assert signs["sequences"][0]["last_stem_activation"]["L6_N753"] != 0
    shifts = part43.run_shift(model, tokenizer, sequences, args)
    assert len(shifts["shifts"]) == 3
    assert shifts["shifts"][0]["per_sequence"][0]["later_units"]["L7_N1169"]["delta"]
    checks = part43.run_validation_checks(model, tokenizer, args)["checks"]
    assert all(row["passed"] for row in checks.values())


def test_amended_lattice_walsh_surrogates_and_additive_logit_fixture(tmp_path, monkeypatch):
    _fake_part25(monkeypatch)
    _short_questions(monkeypatch)
    model, tokenizer = TinyModel(), TinyTokenizer()
    args = _args(tmp_path)
    null_set = {"id": "null-1", "seed": 11, "units": [(6, 754), (7, 1170), (11, 8867), (29, 777)]}
    lattice = part43.run_lattice_amended(
        model, tokenizer, _token_sets(), args, supplied_null_sets=[null_set], decode=False, primary_only=True,
    )
    assert len(lattice["primary_cell"]["subset_score_values"]) == 16
    assert len(lattice["primary_cell"]["raw_candidate_logits_by_subset"]) == 16
    assert set(lattice["primary_cell"]["output_only_surrogate"]["readout_scores"]) == {
        "raw", "frozen_scale_surrogate", "real_final_norm_surrogate",
    }
    surrogate = lattice["primary_cell"]["output_only_surrogate"]
    assert "h_add_residual" in surrogate["cells"][1]
    assert "raw_logit_difference_google_minus_alibaba" in surrogate["cells"][0]
    assert set(surrogate["single_ablation_residual_delta_norms"]) == set(part43.UNIT_KEYS.values())
    assert lattice["primary_family_statistic"]["B"] == 1
    fixture = part43.additive_logit_interaction_fixture()
    assert abs(fixture["raw_candidate_pair_coefficient_google_token_A"]) < 1e-12
    assert abs(fixture["raw_candidate_pair_coefficient_google_token_B"]) < 1e-12
    assert fixture["score_pair_coefficient"] != 0


def test_sweep_rescue_and_position_variants_use_tiny_model(tmp_path, monkeypatch):
    _fake_part25(monkeypatch)
    _short_questions(monkeypatch)
    model, tokenizer = TinyModel(), TinyTokenizer()
    args = _args(tmp_path)
    sweep = part43.run_sweep(model, tokenizer, part43.SWEEP_PATH, args)
    assert len(sweep["matched_pairs"]) == 20
    assert all(row["shared_suffix"] for row in sweep["matched_pairs"])
    rescue = part43.run_rescue(model, tokenizer, _token_sets(), args)
    assert len(rescue["pairs"]) == 7
    assert "T_zeroed_minus_clean" in rescue["pairs"][0]["per_question"][0]["restore_all_rescue"]
    positions = part43.run_positions(model, tokenizer, _token_sets(), args)
    assert {row["scope"] for row in positions["generation_conditions"]} == {"prompt_only", "generated_token_only"}


def test_baselines_heldout_dose_and_grid_use_tiny_model(tmp_path, monkeypatch):
    _fake_part25(monkeypatch)
    _short_questions(monkeypatch)
    monkeypatch.setattr(part43, "_heldout_questions", lambda _path: ["Who built this model?"])
    model, tokenizer = TinyModel(), TinyTokenizer()
    args = _args(tmp_path)
    baseline = {"answers_unsteered": {
        question: {"answer": "I was created by Qwen."}
        for question in (*part43.QUESTIONS, "Who built this model?")
    }}
    Path(args.baseline).write_text(json.dumps(baseline), encoding="utf-8")
    baselines = part43.run_baselines(model, tokenizer, _token_sets(), args)
    assert len(baselines["generation_conditions_who_created_you"]) == 4
    heldout = part43.run_heldout(model, tokenizer, _token_sets(), args)
    assert "Who built this model?" in heldout["per_question"]
    monkeypatch.setattr(part43, "_corpus_max_abs", lambda *_args: {unit: 0.5 for unit in part43.UNITS})
    monkeypatch.setattr(part43, "_generated_answer", lambda *_args, **_kwargs: ("I was created by Qwen.", 8, True))
    dose = part43.run_dose_amended(model, tokenizer, args)
    assert len(dose["teacher_forced_curve"]) == 40
    assert len(dose["free_generation_answers"]) == 24
    grid = part43.run_grid(model, tokenizer, args)
    assert len(grid["cells"]) == 25
    assert len(grid["absolute_grid"]["cells"]) == 25
    assert len(grid["absolute_grid"]["43_10_slope_summary"]["per_template_cluster_slopes"]) == 2
    assert "realized_trajectories" in grid["cells"][0]
    assert "clipping_fraction" in grid["cells"][0]["realized_activation_summary"]["L7_N1169"]


def test_swap_attach_and_synthetic_four_world_paths(tmp_path, monkeypatch):
    _fake_part25(monkeypatch)
    _short_questions(monkeypatch)
    model, tokenizer = TinyModel(), TinyTokenizer()
    args = _args(tmp_path)
    fixture = part43.synthetic_receiver_swap_fixture()
    independent = fixture["independent"]
    connected = fixture["connected"]
    assert independent["C"] == independent["B"]
    assert independent["D"] == independent["A"]
    assert connected["C"] != connected["B"]
    assert connected["D"] != connected["A"]
    swap = part43.run_swap(model, tokenizer, _token_sets(), args)
    assert len(swap["rows"]) == 24
    assert swap["discovery_l7_to_l11"]["worlds"]["A"]["receiver_trajectory"]
    branch = swap["discovery_l7_to_l11"]["receiver_branches"]
    assert branch["product_change_decomposition"]["reconstruction_max_abs_error"] < 1e-5
    assert len(branch["recombined_product_replays"]) == 2
    assert swap["rows"][3]["kind"] == "language"
    assert swap["rows"][3]["template_cluster"] == "English"
    attach = part43.run_attach(model, tokenizer, args)
    assert len(attach["rows"]) == 16
    assert {row["sentence_order"] for row in attach["rows"] if row["wording"] == "primary"} == {
        "engine_first", "charity_first",
    }
    assert len(attach["contrast_summaries"]) == 8
    assert all("search_phrase" in row["measured"]["L11"] for row in attach["rows"])


def test_links_dla_and_readout_check_paths_with_tiny_weights(tmp_path, monkeypatch):
    _fake_part25(monkeypatch)
    _short_questions(monkeypatch)
    model, tokenizer = TinyModel(), TinyTokenizer()
    args = _args(tmp_path)
    context = _tiny_weight_context(tmp_path, model)
    monkeypatch.setattr(part43, "_weight_context", lambda _args: context)
    links = part43.run_links(model, tokenizer, args)
    assert len(links["links"]) == 6
    assert "two_sided_empirical_p" in links["links"][0]["null"]
    assert "residual_bypass_derivative" in links["links"][0]
    assert "residual_bypass_finite_difference" in links["links"][0]
    assert links["links"][0]["finite_difference_check_passed"]
    assert "total_derivative_finite_difference" in links["links"][0]
    signs = part43.run_signs(model, tokenizer, [part43.part25_module().make_sequence(tokenizer, part43.DISCOVERY_QUESTION, "maker", "plain")], args)
    signs_path = tmp_path / "signs.json"
    signs_path.write_text(json.dumps(signs), encoding="utf-8")
    args.signs = str(signs_path)
    Path(args.jlens).write_bytes(b"tiny lens")
    Path(args.checksums).write_text("tiny", encoding="utf-8")
    monkeypatch.setattr(part43, "_load_token_records", lambda _args: _token_sets())
    dla = part43.run_dla(args)
    assert len(dla["records"]) == 1
    readout = part43.run_readout_checks(model, tokenizer, args)
    assert len(readout["records"]) == 4
    assert set(readout["records"][0]["estimands"]) == {
        "frozen_scale_logit_lens_heuristic", "local_final_norm_derivative",
        "part42_averaged_j_lens_product", "finite_intervention",
    }


def test_every_cli_subcommand_is_registered_and_manifest_is_recorded(tmp_path, monkeypatch):
    _fake_part25(monkeypatch)
    commands = {"manifest", "validate", "checks", "batch-diag", "pilot", "signs", "dla", "lattice", "mediate", "rescue", "shift",
                "positions", "baselines", "heldout", "readout-check", "links", "sweep", "dose", "grid", "swap", "attach", "base", "score"}
    parser = part43._build_parser()
    for command in commands:
        args = parser.parse_args([command])
        assert args.command == command
        assert args.exec_mode == "singleton"
    base = parser.parse_args(["base"])
    assert base.model == part43.BASE_MODEL and base.revision == part43.BASE_REVISION

    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: TinyTokenizer())
    args = _args(tmp_path)
    manifest = part43.write_manifest(args)
    assert manifest["tokenizer"]["chat_template_sha256"]
    assert len(manifest["factorial_conditions"]["lattice"]) == 16
    assert manifest["execution_mode"] == "singleton"
    assert manifest["available_execution_modes"]["diagnostic"].startswith("batched")
    assert (Path(args.out) / "MANIFEST.json").is_file()


def test_singleton_and_batched_adapter_preserve_order_and_records(tmp_path, monkeypatch):
    _fake_part25(monkeypatch)
    model, tokenizer = TinyModel(), TinyTokenizer()
    records = part43._teacher_forced_records(
        tokenizer, ["Who created you?", "What is your name?"], answer_options=part43.VALIDATION_ANSWER_OPTIONS,
    )
    operations = [{part43.UNITS[1]: 0.0} for _ in records]
    part43.assert_mode_alignment(records, records, operations, operations)
    single_payload, single_acts = part43.teacher_forced_scores(
        model, tokenizer, ["Who created you?", "What is your name?"], _token_sets(),
        operations=operations, answer_options=part43.VALIDATION_ANSWER_OPTIONS, mode="singleton",
    )
    batch_payload, batch_acts = part43.teacher_forced_scores(
        model, tokenizer, ["Who created you?", "What is your name?"], _token_sets(),
        operations=operations, answer_options=part43.VALIDATION_ANSWER_OPTIONS, mode="batched",
    )
    assert [row["input_ids"] for row in single_payload["per_answer"]] == [row["input_ids"] for row in batch_payload["per_answer"]]
    assert [row["answer_positions"] for row in single_payload["per_answer"]] == [row["answer_positions"] for row in batch_payload["per_answer"]]
    assert single_payload["answer_logprobs"] == pytest.approx(batch_payload["answer_logprobs"], abs=1e-5)
    for unit in part43.UNITS:
        assert len(single_acts[unit]) == len(batch_acts[unit])
        for single_row, batch_row in zip(single_acts[unit], batch_acts[unit]):
            assert single_row == pytest.approx(batch_row, abs=1e-6)


def test_singleton_confirmatory_forwards_have_one_row_and_batching_sabotage_is_caught(monkeypatch):
    _fake_part25(monkeypatch)
    model, tokenizer = TinyModel(), TinyTokenizer()
    batch_sizes = []
    original_forward = model.forward

    def observe_forward(*args, **kwargs):
        batch_sizes.append(int(kwargs["input_ids"].shape[0]))
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(model, "forward", observe_forward)

    def score_and_assert_singleton_batches():
        batch_sizes.clear()
        part43.teacher_forced_scores(
            model, tokenizer, ["Who created you?", "What is your name?"], _token_sets(),
            answer_options=part43.VALIDATION_ANSWER_OPTIONS, capture=False, mode="singleton",
        )
        assert batch_sizes and all(size == 1 for size in batch_sizes), (
            f"every singleton forward must have batch size 1; observed {batch_sizes}"
        )

    score_and_assert_singleton_batches()

    original_execute = part43.execute

    def batch_singleton_sabotage(*args, **kwargs):
        if kwargs.get("mode") == "singleton":
            kwargs["mode"] = "batched"
        return original_execute(*args, **kwargs)

    monkeypatch.setattr(part43, "execute", batch_singleton_sabotage)
    with pytest.raises(AssertionError, match="every singleton forward must have batch size 1"):
        score_and_assert_singleton_batches()


def test_capture_norm_inputs_advances_through_singleton_rows():
    model, tokenizer = TinyModel(), TinyTokenizer()
    records = [
        {"input_ids": [2, 3]},
        {"input_ids": [4, 5, 6, 7]},
    ]
    captured = part43._capture_norm_inputs(model, tokenizer, records, part43.UNITS[0][0])
    assert [tuple(value.shape) for value in captured] == [(2, 4), (4, 4)]


def test_confirmatory_scoring_cannot_bypass_execution_adapter(tmp_path, monkeypatch):
    _fake_part25(monkeypatch)
    model, tokenizer = TinyModel(), TinyTokenizer()

    def sabotage(*_args, **_kwargs):
        raise RuntimeError("execution adapter sabotage")

    monkeypatch.setattr(part43, "execute", sabotage)
    with pytest.raises(RuntimeError, match="execution adapter sabotage"):
        part43.teacher_forced_scores(
            model, tokenizer, [part43.DISCOVERY_QUESTION], _token_sets(),
            answer_options=part43.VALIDATION_ANSWER_OPTIONS,
        )

    source = ast.parse(Path(part43.__file__).read_text(encoding="utf-8"))
    violations = []

    class RawModelCallVisitor(ast.NodeVisitor):
        def __init__(self):
            self.functions = []

        def visit_FunctionDef(self, node):
            self.functions.append(node.name)
            self.generic_visit(node)
            self.functions.pop()

        def visit_Call(self, node):
            is_forward = isinstance(node.func, ast.Name) and node.func.id == "model"
            is_generation = isinstance(node.func, ast.Attribute) and node.func.attr == "generate" and isinstance(node.func.value, ast.Name) and node.func.value.id == "model"
            if (is_forward or is_generation) and "execute" not in self.functions:
                violations.append((node.lineno, ast.unparse(node.func)))
            self.generic_visit(node)

    RawModelCallVisitor().visit(source)
    assert violations == []


def test_validate_and_batch_diag_report_score_and_receiver_fields(tmp_path, monkeypatch):
    _fake_part25(monkeypatch)
    model, tokenizer = TinyModel(), TinyTokenizer()
    args = _args(tmp_path)
    validation = part43.run_singleton_validation(model, tokenizer, _token_sets(), args)
    assert validation["execution_mode"] == "singleton"
    assert validation["maker_answer_count"] == 7
    assert "clean_to_clean_restoration" in validation["cells"]
    assert validation["cells"]["clean_to_clean_restoration"]["natural_trajectory_replay"]["all_real_positions_checked"]
    assert validation["receiver_path_case"]["C_L7_zero_L11_restored_score"] is not None
    assert validation["cells"]["clean"]["within_mode_repeat_max_abs_primary_score_difference"] is not None

    diagnostic = part43.run_batch_diagnostics(model, tokenizer, args)
    names = {row["name"] for row in diagnostic["conditions"]}
    assert {"A_alone_singleton", "A_A_no_padding", "A_B_same_length", "A_B_different_length",
            "A_B_same_length_reversed", "A_randomized_masked_pad_ids", "A_extra_long_companion"}.issubset(names)
    for row in diagnostic["conditions"]:
        assert "natural_primary_score" in row
        assert "intervention_delta" in row
        assert row["receiver_signed_h_at_last_stem_position"]["natural"]
    batched_rows = [row for row in diagnostic["conditions"] if row["execution_mode"] == "batched"]
    assert all(row["unit_row_indexing_matches_singleton"] for row in batched_rows)
    assert all("max_abs_receiver_h_difference_vs_singleton_natural" in row for row in batched_rows)


def test_calibration_runs_fixed_cells_in_bf16_and_fp32_on_cpu(tmp_path, monkeypatch):
    _fake_part25(monkeypatch)
    args = _args(tmp_path)
    tokenizer = TinyTokenizer()
    token_records = _token_sets()
    bf16_model = TinyModel().to(dtype=torch.bfloat16)
    bf16, bf16_execution = part43._run_calibration_leg(
        bf16_model, tokenizer, token_records, args,
        leg_name="bf16", expected_dtype="torch.bfloat16", repeats=3,
    )
    fp32_model = TinyModel()
    fp32, fp32_execution = part43._run_calibration_leg(
        fp32_model, tokenizer, token_records, args,
        leg_name="fp32", expected_dtype="torch.float32", repeats=1,
    )
    raw = part43._calibration_artifact(
        args, tokenizer, {"bf16": bf16, "fp32": fp32},
        {"bf16": bf16_execution, "fp32": fp32_execution},
    )
    assert raw["schema_version"] == part43.CALIBRATION_SCHEMA_VERSION == 2
    assert raw["artifact_type"] == part43.CALIBRATION_ARTIFACT_TYPE
    assert set(raw) == {"schema_version", "artifact_type", "model", "revision", "source_provenance",
                        "tokenizer", "execution", "registration", "registration_sha256", "legs"}
    for leg, repeat_count in (("bf16", 3), ("fp32", 1)):
        assert set(raw["legs"][leg]["cells"]) == set(part43.CALIBRATION_CELLS)
        for cell in raw["legs"][leg]["cells"].values():
            assert len(cell["repeats"]) == repeat_count
            assert all(set(repeat) == {"answer_logprobs"} for repeat in cell["repeats"])
            assert all(set(repeat["answer_logprobs"]) == {row["id"] for row in raw["registration"]["answers"]}
                       for repeat in cell["repeats"])
    assert part43.calibration_gate(
        part43._calibration_reader(raw, args=args, tokenizer=tokenizer, raw_sha256="0" * 64),
        model=args.model, revision=args.revision,
    )[0]


def test_raw_artifact_reader_computes_summary_and_pilot_after_disk_roundtrip(tmp_path, monkeypatch):
    args = _args(tmp_path)
    raw, tokenizer = _calibration_raw_fixture(args, monkeypatch, fp32_difference=0.01)
    out = tmp_path / "serialized-pilot"
    monkeypatch.setattr(part43, "write_manifest", lambda _args: {"schema_version": 1})
    monkeypatch.setattr(part43, "_load_token_records", lambda _args: _token_sets())
    monkeypatch.setattr(part43, "run_calibration_runtime", lambda *_args: raw)
    monkeypatch.setattr(part43, "_load_calibration_tokenizer", lambda _args: tokenizer)
    monkeypatch.setattr(part43, "_load_runtime", lambda _args: (tokenizer, TinyModel().to(dtype=torch.bfloat16)))
    monkeypatch.setattr(part43, "_sequences", lambda _tokenizer: ["tiny sequence"])
    stages = []
    monkeypatch.setattr(part43, "run_signs", lambda *_a, **_k: stages.append("signs") or {"signed": True})
    monkeypatch.setattr(part43, "run_lattice_amended", lambda *_a, **_k: stages.append("lattice") or {"timing": {"elapsed_seconds": 0.1}})
    monkeypatch.setattr(part43, "run_swap", lambda *_a, **_k: stages.append("swap") or {"rows": []})
    monkeypatch.setattr(part43, "run_attach", lambda *_a, **_k: stages.append("attach") or {"rows": []})
    args.attach = str(tmp_path / "attach.json")
    Path(args.attach).write_text(json.dumps({"prompts": []}), encoding="utf-8")
    common = ["--out", str(out), "--candidates", args.candidates, "--cache-dir", str(tmp_path), "--device", "cpu",
              "--heldout", args.heldout, "--prompts", args.prompts, "--corpus", args.corpus,
              "--attach", args.attach]
    assert part43.main(["pilot", *common]) == 0
    raw_on_disk = json.loads((out / "calibration.json").read_text(encoding="utf-8"))
    summary = json.loads((out / "calibration-summary.json").read_text(encoding="utf-8"))
    pilot = json.loads((out / "pilot.json").read_text(encoding="utf-8"))
    assert raw_on_disk == raw
    assert summary["derived"] is True and summary["status"] == "within_budget"
    assert summary["epsilon"] == 0.01
    assert summary["raw_artifact_sha256"] == part43.sha256_file(out / "calibration.json")
    assert pilot["calibration"] == summary
    assert stages == ["signs", "lattice", "swap", "attach"]


@pytest.mark.parametrize("difference, expected", [(0.049, True), (0.05, True), (0.051, False)])
def test_calibration_reader_uses_exact_budget_boundary(tmp_path, monkeypatch, difference, expected):
    args = _args(tmp_path)
    raw, tokenizer = _calibration_raw_fixture(args, monkeypatch, fp32_difference=difference)
    part43.write_json(Path(args.out) / "calibration.json", raw)
    summary = part43._read_calibration_artifact(args, tokenizer=tokenizer)
    passed, _reason = part43.calibration_gate(summary, model=args.model, revision=args.revision)
    assert summary["epsilon"] == difference
    assert passed is expected


def test_reader_writes_derived_summary_even_when_over_budget(tmp_path, monkeypatch):
    args = _args(tmp_path)
    raw, tokenizer = _calibration_raw_fixture(args, monkeypatch, fp32_difference=0.06)
    part43.write_json(Path(args.out) / "calibration.json", raw)
    summary = part43._read_calibration_artifact(args, tokenizer=tokenizer)
    assert summary["status"] == "over_budget"
    assert (Path(args.out) / "calibration-summary.json").is_file()
    assert part43.calibration_gate(summary, model=args.model, revision=args.revision)[0] is False


def test_reader_includes_replay_cell_bf16_fp32_difference_in_epsilon(tmp_path, monkeypatch):
    args = _args(tmp_path)
    raw, tokenizer = _calibration_raw_fixture(args, monkeypatch, fp32_difference=0.0)
    google_id = part43.CALIBRATION_ANSWER_IDS["Google"]
    raw["legs"]["fp32"]["cells"]["clean_to_clean_replay"]["repeats"][0]["answer_logprobs"][google_id] = 0.06
    part43.write_json(Path(args.out) / "calibration.json", raw)
    summary = part43._read_calibration_artifact(args, tokenizer=tokenizer)
    assert summary["error_by_cell"]["clean_to_clean_replay"]["bf16_fp32_abs_difference"] == 0.06
    assert summary["epsilon"] == 0.06
    assert summary["status"] == "over_budget"


@pytest.mark.parametrize("derived_key", ["epsilon", "spread", "difference", "replay_error", "primary_score", "error_kinds", "summary"])
def test_raw_artifact_refuses_every_derived_field(tmp_path, monkeypatch, derived_key):
    args = _args(tmp_path)
    raw, tokenizer = _calibration_raw_fixture(args, monkeypatch)
    raw[derived_key] = 0.0
    part43.write_json(Path(args.out) / "calibration.json", raw)
    with pytest.raises(part43.CalibrationRefusal, match="derived field in raw artifact"):
        part43._read_calibration_artifact(args, tokenizer=tokenizer)


@pytest.mark.parametrize("numeric_path", ["input_ids", "answer_positions", "target_ids", "prompt_length", "answer_length"])
def test_reader_binds_numeric_registration_metadata_to_tokenizer(tmp_path, monkeypatch, numeric_path):
    args = _args(tmp_path)
    raw, tokenizer = _calibration_raw_fixture(args, monkeypatch)
    answer = raw["registration"]["answers"][0]
    if numeric_path in {"input_ids", "answer_positions", "target_ids"}:
        answer[numeric_path][0] += 1
    else:
        answer[numeric_path] += 1
    part43.write_json(Path(args.out) / "calibration.json", raw)
    with pytest.raises(part43.CalibrationRefusal, match="registration"):
        part43._read_calibration_artifact(args, tokenizer=tokenizer)


def test_reader_binds_per_cell_token_ids_and_tokenizer_identity(tmp_path, monkeypatch):
    args = _args(tmp_path)
    raw, tokenizer = _calibration_raw_fixture(args, monkeypatch)
    answer_id = raw["registration"]["answers"][0]["id"]
    raw["registration"]["cells"]["clean"]["prompt_token_ids"][answer_id][0] += 1
    part43.write_json(Path(args.out) / "calibration.json", raw)
    with pytest.raises(part43.CalibrationRefusal, match="registration"):
        part43._read_calibration_artifact(args, tokenizer=tokenizer)

    raw, tokenizer = _calibration_raw_fixture(args, monkeypatch)
    raw["tokenizer"]["chat_template_sha256"] = "0" * 64
    part43.write_json(Path(args.out) / "calibration.json", raw)
    with pytest.raises(part43.CalibrationRefusal, match="tokenizer snapshot or chat-template"):
        part43._read_calibration_artifact(args, tokenizer=tokenizer)


def test_reader_rejects_overflow_bool_string_and_missing_measurements(tmp_path, monkeypatch):
    args = _args(tmp_path)
    raw, tokenizer = _calibration_raw_fixture(args, monkeypatch)
    answer_id = raw["registration"]["answers"][0]["id"]
    target = raw["legs"]["bf16"]["cells"]["clean"]["repeats"][0]["answer_logprobs"]
    target[answer_id] = 10 ** 400
    part43.write_json(Path(args.out) / "calibration.json", raw)
    with pytest.raises(part43.CalibrationRefusal, match="overflow"):
        part43._read_calibration_artifact(args, tokenizer=tokenizer)

    raw, tokenizer = _calibration_raw_fixture(args, monkeypatch)
    target = raw["legs"]["bf16"]["cells"]["clean"]["repeats"][0]["answer_logprobs"]
    target[answer_id] = True
    part43.write_json(Path(args.out) / "calibration.json", raw)
    with pytest.raises(part43.CalibrationRefusal, match="numeric per-answer sum"):
        part43._read_calibration_artifact(args, tokenizer=tokenizer)

    raw, tokenizer = _calibration_raw_fixture(args, monkeypatch)
    target = raw["legs"]["bf16"]["cells"]["clean"]["repeats"][0]["answer_logprobs"]
    target[answer_id] = "0.0"
    part43.write_json(Path(args.out) / "calibration.json", raw)
    with pytest.raises(part43.CalibrationRefusal, match="numeric per-answer sum"):
        part43._read_calibration_artifact(args, tokenizer=tokenizer)

    raw, tokenizer = _calibration_raw_fixture(args, monkeypatch)
    del raw["legs"]["fp32"]["cells"]["clean_to_clean_replay"]
    part43.write_json(Path(args.out) / "calibration.json", raw)
    with pytest.raises(part43.CalibrationRefusal, match="missing field"):
        part43._read_calibration_artifact(args, tokenizer=tokenizer)


def test_reader_refuses_duplicate_json_keys_at_parse_time(tmp_path, monkeypatch):
    args = _args(tmp_path)
    raw, tokenizer = _calibration_raw_fixture(args, monkeypatch)
    path = Path(args.out) / "calibration.json"
    part43.write_json(path, raw)
    serialized = path.read_text(encoding="utf-8")
    serialized = serialized.replace(chr(34) + "schema_version" + chr(34) + ": 2,", chr(34) + "schema_version" + chr(34) + ": 2," + chr(10) + "  " + chr(34) + "schema_version" + chr(34) + ": 2,", 1)
    path.write_text(serialized, encoding="utf-8")
    with pytest.raises(part43.DuplicateCalibrationKey, match="duplicate JSON object key"):
        part43._read_calibration_artifact(args, tokenizer=tokenizer)


def test_reader_propagates_internal_runtime_error_identity(tmp_path, monkeypatch):
    args = _args(tmp_path)
    raw, tokenizer = _calibration_raw_fixture(args, monkeypatch)
    part43.write_json(Path(args.out) / "calibration.json", raw)
    marker = RuntimeError("reader internal sentinel")
    def explode(*_args, **_kwargs):
        raise marker
    monkeypatch.setattr(part43, "_primary_score_from_answer_rows", explode)
    with pytest.raises(RuntimeError) as exc:
        part43._read_calibration_artifact(args, tokenizer=tokenizer)
    assert exc.value is marker
    assert not isinstance(exc.value, part43.CalibrationRefusal)


@pytest.mark.parametrize("mutation, expected_fragment", [
    ("schema", "schema version"),
    ("model", "model or pinned revision"),
    ("missing_cell", "missing field"),
    ("derived_key", "derived field"),
    ("duplicate_key", "duplicate JSON object key"),
    ("overflow", "overflow"),
    ("over_budget", "0.05 nat budget"),
])
def test_serialized_decision_gate_stops_before_signs_on_refusal_or_overbudget(
    tmp_path, monkeypatch, mutation, expected_fragment,
):
    args = _args(tmp_path)
    raw, tokenizer = _calibration_raw_fixture(
        args, monkeypatch, fp32_difference=0.06 if mutation == "over_budget" else 0.01,
    )
    if mutation == "schema":
        raw["schema_version"] = 1
    elif mutation == "model":
        raw["model"] = "unregistered/model"
    elif mutation == "missing_cell":
        del raw["legs"]["bf16"]["cells"]["all_four_zero"]
    elif mutation == "derived_key":
        raw["epsilon"] = 0.0
    elif mutation == "overflow":
        answer_id = raw["registration"]["answers"][0]["id"]
        raw["legs"]["fp32"]["cells"]["clean"]["repeats"][0]["answer_logprobs"][answer_id] = 10 ** 400

    out = tmp_path / f"gate-{mutation}"
    monkeypatch.setattr(part43, "write_manifest", lambda _args: {"schema_version": 1})
    monkeypatch.setattr(part43, "_load_token_records", lambda _args: _token_sets())
    monkeypatch.setattr(part43, "run_calibration_runtime", lambda *_args: raw)
    monkeypatch.setattr(part43, "_load_calibration_tokenizer", lambda _args: tokenizer)
    stages = []
    monkeypatch.setattr(part43, "_load_runtime", lambda *_a: pytest.fail("refused calibration must stop before model load"))
    monkeypatch.setattr(part43, "_sequences", lambda _tokenizer: pytest.fail("refused calibration must stop before signs"))
    monkeypatch.setattr(part43, "run_signs", lambda *_a, **_k: stages.append("signs"))
    monkeypatch.setattr(part43, "run_lattice_amended", lambda *_a, **_k: stages.append("lattice"))

    original_write_json = part43.write_json
    def write_with_optional_duplicate(path, value):
        original_write_json(path, value)
        if mutation == "duplicate_key" and Path(path).name == "calibration.json":
            text = Path(path).read_text(encoding="utf-8")
            text = text.replace(chr(34) + "schema_version" + chr(34) + ": 2,", chr(34) + "schema_version" + chr(34) + ": 2," + chr(10) + "  " + chr(34) + "schema_version" + chr(34) + ": 2,", 1)
            Path(path).write_text(text, encoding="utf-8")
    monkeypatch.setattr(part43, "write_json", write_with_optional_duplicate)
    common = ["--out", str(out), "--candidates", args.candidates, "--cache-dir", str(tmp_path), "--device", "cpu"]
    assert part43.main(["pilot", *common]) == 0
    assert stages == []
    hold = json.loads((out / "HOLD.json").read_text(encoding="utf-8"))
    assert expected_fragment in hold["reason"]
    if mutation == "over_budget":
        assert json.loads((out / "calibration-summary.json").read_text(encoding="utf-8"))["status"] == "over_budget"


def test_calibration_number_sabotage_cannot_use_registration_metadata(tmp_path, monkeypatch):
    args = _args(tmp_path)
    raw, tokenizer = _calibration_raw_fixture(args, monkeypatch)
    raw["registration"]["answers"][0]["input_ids"][0] += 1
    part43.write_json(Path(args.out) / "calibration.json", raw)
    with pytest.raises(part43.CalibrationRefusal, match="registration"):
        part43._read_calibration_artifact(args, tokenizer=tokenizer)


def test_valid_reader_summary_reaches_pilot_lattice(tmp_path, monkeypatch):
    args = _args(tmp_path)
    calibration = _calibration_fixture(args, monkeypatch)
    args.attach = str(tmp_path / "attach.json")
    Path(args.attach).write_text(json.dumps({"prompts": []}), encoding="utf-8")
    monkeypatch.setattr(part43, "_sequences", lambda _tokenizer: ["tiny sequence"])
    monkeypatch.setattr(part43, "run_signs", lambda *_a, **_k: {"signed": True})
    calls = []
    monkeypatch.setattr(part43, "run_lattice_amended", lambda *_a, **_k: calls.append("lattice") or {"timing": {"elapsed_seconds": 0.1}})
    monkeypatch.setattr(part43, "run_swap", lambda *_a, **_k: {"rows": []})
    monkeypatch.setattr(part43, "run_attach", lambda *_a, **_k: {"rows": []})
    result = part43.run_pilot(
        TinyModel().to(dtype=torch.bfloat16), TinyTokenizer(), _token_sets(), args, calibration=calibration,
    )
    assert result["status"] == "complete"
    assert calls == ["lattice"]


def test_run_pilot_second_gate_rejects_calibration_from_another_protocol(tmp_path, monkeypatch):
    legacy_dir = tmp_path / "legacy"
    fp32_dir = tmp_path / "fp32"
    legacy_dir.mkdir()
    fp32_dir.mkdir()
    legacy_args = _args(legacy_dir)
    legacy_calibration = _calibration_fixture(legacy_args, monkeypatch)
    fp32_args = _args(fp32_dir, protocol="fp32-successor")
    monkeypatch.setattr(part43, "_sequences", lambda *_args: pytest.fail("cross-protocol pilot reached stages"))
    result = part43.run_pilot(
        TinyModel().to(dtype=torch.bfloat16), TinyTokenizer(), _token_sets(),
        fp32_args, calibration=legacy_calibration,
    )
    assert result["status"] == "HOLD"
    assert "protocol" in result["hold"]["reason"]


def test_run_pilot_second_gate_rejects_model_dtype_mismatching_fp32_calibration(tmp_path, monkeypatch):
    args = _args(tmp_path, protocol="fp32-successor")
    raw, tokenizer = _fp32_successor_raw_fixture(args, monkeypatch)
    calibration = _read_fp32_fixture(args, raw, tokenizer)
    monkeypatch.setattr(part43, "_sequences", lambda *_args: pytest.fail("dtype-mismatched pilot reached stages"))
    result = part43.run_pilot(
        TinyModel().to(dtype=torch.bfloat16), tokenizer, _token_sets(), args,
        calibration=calibration,
    )
    assert result["status"] == "HOLD"
    assert "torch.float32" in result["hold"]["reason"]


def test_fp32_cli_persists_a_second_gate_hold_without_stage_outputs(tmp_path, monkeypatch):
    args = _args(tmp_path, protocol="fp32-successor")
    raw, tokenizer = _fp32_successor_raw_fixture(args, monkeypatch)
    monkeypatch.setattr(part43, "write_manifest", lambda _args: {"schema_version": 1})
    monkeypatch.setattr(part43, "_load_token_records", lambda _args: _token_sets())
    monkeypatch.setattr(part43, "run_calibration_runtime", lambda *_args: raw)
    monkeypatch.setattr(part43, "_load_calibration_tokenizer", lambda _args: tokenizer)
    monkeypatch.setattr(part43, "_load_runtime", lambda *_args, **_kwargs: (
        tokenizer, TinyModel().to(dtype=torch.bfloat16),
    ))
    monkeypatch.setattr(part43, "_sequences", lambda *_args: pytest.fail("mismatched pilot reached production stages"))
    assert part43.main([
        "pilot", "--protocol", "fp32-successor", "--out", args.out,
        "--candidates", args.candidates, "--cache-dir", args.cache_dir, "--device", "cpu",
    ]) == 0
    pilot = json.loads((Path(args.out) / "pilot.json").read_text(encoding="utf-8"))
    assert pilot["status"] == "HOLD"
    assert "torch.float32" in pilot["hold"]["reason"]
    assert pilot["calibration_gate_interpretation"] == part43.FP32_GATE_INTERPRETATION
    assert json.loads((Path(args.out) / "HOLD.json").read_text(encoding="utf-8"))["status"] == "HOLD"
    assert not (Path(args.out) / "pilot-signs.json").exists()

def test_decision_cli_missing_calibration_writes_hold_before_lattice(tmp_path, monkeypatch):
    args = _args(tmp_path)
    called = []
    monkeypatch.setattr(part43, "_load_runtime", lambda *_a, **_k: pytest.fail("missing calibration must stop before load"))
    monkeypatch.setattr(part43, "run_lattice_amended", lambda *_a, **_k: called.append(True))
    common = ["--out", str(Path(args.out)), "--candidates", args.candidates]
    assert part43.main(["lattice", *common]) == 0
    hold = json.loads((Path(args.out) / "HOLD.json").read_text(encoding="utf-8"))
    assert hold["status"] == "HOLD"
    assert "missing" in hold["reason"]
    assert called == []


@pytest.mark.parametrize("command", [
    "manifest", "validate", "checks", "batch-diag", "signs", "dla", "lattice", "mediate",
    "rescue", "shift", "positions", "baselines", "heldout", "readout-check", "links", "sweep",
    "dose", "grid", "swap", "attach", "base", "score",
])
def test_fp32_protocol_refuses_every_command_outside_calibrate_and_pilot(command, tmp_path, monkeypatch):
    def unexpected(*_args, **_kwargs):
        pytest.fail("the bounded FP32 protocol guard must run before command work")

    monkeypatch.setattr(part43, "_read_calibration_gate", lambda _args: (False, "missing"))
    monkeypatch.setattr(part43, "write_manifest", unexpected)
    monkeypatch.setattr(part43, "_load_runtime", unexpected)
    monkeypatch.setattr(part43, "run_score", unexpected)
    monkeypatch.setattr(part43, "run_dla", unexpected)
    with pytest.raises(SystemExit, match="limited to calibrate and pilot"):
        part43.main([
            command, "--protocol", "fp32-successor", "--out", str(tmp_path / "out"),
        ])


@pytest.mark.parametrize("protocol, dtype", [("fp32-successor", "bf16"), ("bf16-v2", "fp32")])
def test_cli_rejects_dtype_that_contradicts_selected_protocol(protocol, dtype, tmp_path, monkeypatch):
    monkeypatch.setattr(part43, "write_manifest", lambda *_args: pytest.fail("dtype contradiction reached manifest"))
    with pytest.raises(SystemExit, match="contradicts"):
        part43.main([
            "calibrate", "--protocol", protocol, "--dtype", dtype,
            "--out", str(tmp_path / "out"),
        ])


def test_pilot_and_base_cli_smoke_with_tiny_model(tmp_path, monkeypatch):
    _fake_part25(monkeypatch)
    model, tokenizer = TinyModel().to(dtype=torch.bfloat16), TinyTokenizer()
    args = _args(tmp_path)
    calibration, calibration_tokenizer = _calibration_raw_fixture(args, monkeypatch)
    diagnostic = {"checks": {"batch_only": {"passed": True, "tolerance": 1e-4}}}
    monkeypatch.setattr(part43, "write_manifest", lambda _args: {"schema_version": 1})
    monkeypatch.setattr(part43, "_load_runtime", lambda _args: (tokenizer, model))
    monkeypatch.setattr(part43, "run_calibration_runtime", lambda *_args: calibration)
    monkeypatch.setattr(part43, "_load_calibration_tokenizer", lambda _args: calibration_tokenizer)
    monkeypatch.setattr(part43, "_sequences", lambda _tokenizer: ["tiny sequence"])
    monkeypatch.setattr(part43, "run_signs", lambda *_args, **_kwargs: {"signed": True})
    monkeypatch.setattr(part43, "run_validation_checks", lambda *_args: pytest.fail("pilot must not gate on batch checks"))
    monkeypatch.setattr(part43, "run_lattice_amended", lambda *_args, **_kwargs: {"timing": {"elapsed_seconds": 0.1}})
    monkeypatch.setattr(part43, "run_swap", lambda *_args, **_kwargs: {"rows": []})
    monkeypatch.setattr(part43, "run_attach", lambda *_args, **_kwargs: {"rows": []})
    out = tmp_path / "cli-out"
    args.attach = str(tmp_path / "attach.json")
    Path(args.attach).write_text(json.dumps({"prompts": []}), encoding="utf-8")
    common = ["--out", str(out), "--candidates", args.candidates, "--cache-dir", str(tmp_path), "--device", "cpu",
              "--heldout", args.heldout, "--prompts", args.prompts, "--corpus", args.corpus,
              "--attach", args.attach]
    assert part43.main(["pilot", *common]) == 0
    assert (out / "pilot.json").is_file() and (out / "pilot-lattice.json").is_file()
    pilot_calibration = json.loads((out / "pilot.json").read_text())["calibration"]
    assert pilot_calibration["derived"] is True
    assert json.loads((out / "calibration.json").read_text()) == calibration
    assert json.loads((out / "calibration-summary.json").read_text()) == pilot_calibration
    assert not (out / "checks.json").exists()
    monkeypatch.setattr(part43, "run_validation_checks", lambda *_args: diagnostic)
    assert part43.main(["checks", *common]) == 0
    assert json.loads((out / "checks.json").read_text())["checks"] == diagnostic["checks"]
    base_args = _args(tmp_path, model=part43.BASE_MODEL, revision=part43.BASE_REVISION)
    base_calibration, base_tokenizer = _calibration_raw_fixture(base_args, monkeypatch)
    monkeypatch.setattr(part43, "_load_calibration_tokenizer", lambda _args: base_tokenizer)
    part43.write_json(out / "calibration.json", base_calibration)
    assert part43.main(["base", *common, "--model", part43.BASE_MODEL,
                        "--revision", part43.BASE_REVISION]) == 0
    assert (out / "base.json").is_file() and (out / "base-signs.json").is_file()


@pytest.mark.parametrize("command", ["validate", "pilot", "signs"])
def test_confirmatory_cli_commands_reject_batched_mode(command):
    with pytest.raises(SystemExit, match="batched execution is diagnostic-only"):
        part43.main([command, "--exec-mode", "batched"])


def test_environment_records_attention_backend_from_loaded_model():
    model = TinyModel()
    model.config._attn_implementation = "eager"
    environment = part43._runtime_environment(model)
    assert environment["kernel"]["attention"] == "eager"


def test_fp32_output_header_configures_precision_before_stamping_protocol(monkeypatch, tmp_path):
    args = _args(tmp_path, protocol="fp32-successor")
    precision = {"value": "high"}
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    monkeypatch.setattr(torch, "set_float32_matmul_precision", lambda value: precision.update(value=value))
    monkeypatch.setattr(torch, "get_float32_matmul_precision", lambda: precision["value"])
    header = part43.output_header(args, [])
    assert header["calibration_protocol"] == {
        "protocol_id": part43.FP32_SUCCESSOR_PROTOCOL_ID,
        "decision_dtype": "torch.float32",
        "matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "float32_matmul_precision": "highest",
    }
    assert header["environment"]["precision_flags"]["cuda_matmul_allow_tf32"] is False
    assert header["environment"]["precision_flags"]["cudnn_allow_tf32"] is False


def test_calibration_gate_checks_protocol_at_its_own_boundary():
    # Keep reader_id consistent with the requested protocol so only the protocol check
    # distinguishes this deliberately crossed in-memory summary.
    summary = {
        "artifact_type": part43.CALIBRATION_SUMMARY_TYPE,
        "protocol_id": part43.CALIBRATION_PROTOCOL_ID,
        "reader_id": part43.FP32_SUCCESSOR_READER_ID,
        "derived": True,
        "model": part43.MODEL,
        "revision": part43.REVISION,
        "epsilon": 0.0,
    }
    passed, reason = part43.calibration_gate(
        summary, model=part43.MODEL, revision=part43.REVISION,
        protocol_id=part43.FP32_SUCCESSOR_PROTOCOL_ID,
    )
    assert passed is False
    assert "protocol" in reason


def _score_fixtures(out):
    question = part43.QUESTIONS[2]
    subsets = [{question: {"primary": float(mask), "first_token_margin": float(mask), "raw_candidate_logits": {"1": float(mask)}}} for mask in range(16)]
    pair_key = "L7 N1169 x L11 N8868"
    lattice = {
        "primary_cell": {"subset_score_values": list(range(16))},
        "primary_family_statistic": {"observed": 1.0, "empirical_p": 0.1, "B": 50},
        "per_prompt": [{"question": question, "target": {"walsh": {"additive_r2_with_intercept": 0.9, "additive_r2_status": "TESTABLE"}},
                        "additive_effect_relative_error": 0.1,
                        "pairwise_null_empirical_p": {pair_key: 0.1}}],
        "subset_scores": {"target": subsets},
    }
    rescue_pairs = [
        {"upstream": "L7 N1169", "downstream": ["L11 N8868"], "rescue_bootstrap": {"mean": 0.7, "bootstrap_95": [0.3, 0.9]},
         "control_summary": {"clean_to_clean_replay_max_abs_score_error": 0.0}},
        {"upstream": "L6 N753", "downstream": ["L11 N8868", "L29 N776"], "rescue_bootstrap": {"mean": 0.2, "bootstrap_95": [0.1, 0.3]}},
    ]
    sweep_pairs = [{"unit_adjective_signed_differences_chinese_minus_american": {"L11_N8868": 1.0}} for _ in range(20)]
    links = {"links": [
        {"earlier": {"layer": 6, "unit": 753}, "later": {"layer": 7, "unit": 1169},
         "null": {"bonferroni_adjusted_p_over_six_pairs": 0.5},
         "weight_alignment_heuristics": {"gate": {"absolute_dot_percentile": 50, "absolute_cosine_percentile": 60}}}
        for _ in range(6)
    ]}
    dose = {"teacher_forced_curve": [
        {"unit": {"label": part43.UNIT_LABELS[unit]}, "multiplier": multiplier,
         "per_question": {question: {"primary": multiplier}}}
        for unit in part43.UNITS for multiplier in (-8, -4, -2, -1, 0, 0.5, 1, 2, 4, 8)
    ], "free_generation_answers": [
        {"unit": {"label": "L29 N776"}, "multiplier": multiplier, "question": question,
         "baseline_maker": "qwen", "answer_maker": "qwen" if multiplier > 0 else "google"}
        for multiplier in (-8, -4, -2, 2, 4, 8)
    ]}
    grid = {"cells": [
        {"L7_multiplier": level, "L11_multiplier": l11,
         "per_question": {question: {"primary": level * (1 if l11 == 4 else 0.8)}}}
        for l11 in (-4, 4) for level in (-4, -1, 0, 1, 4)
    ]}
    absolute_summary = {
        "mean_slope_at_low_L11": 1.0,
        "mean_slope_at_high_L11": 1.1,
        "low_L11_slope_bootstrap_95": [0.8, 1.2],
        "high_L11_slope_bootstrap_95": [0.9, 1.3],
        "absolute_slope_ratio_bootstrap_95": [1.0, 1.4],
        "minimum_slope_magnitude": 0.1,
    }
    docs = {
        "signs": {"sequences": [{"question": question, "stem": "maker", "variant": "plain",
                                  "last_stem_activation": {"L29_N776": -1, "L7_N1169": -1}}]},
        "lattice": lattice,
        "mediate": {"pairs": rescue_pairs},
        "shift": {"shifts": [{"zeroed_unit": "L7 N1169", "per_sequence": [{"question": question, "stem": "maker", "variant": "plain",
                    "later_units": {"L11_N8868": {"clean": [1.0], "delta": [0.3]}}}]}]},
        "sweep": {"matched_pairs": sweep_pairs},
        "links": links,
        "dose": dose,
        "heldout": {"per_question": {str(i): {"zero_effect": 0.1} for i in range(12)}},
        "baselines": {"generation_conditions_who_created_you": [
            {"condition": name, "answer_maker": "google"}
            for name in ("fixed_corpus_mean", "counterfactual_chinese_stem_resample", "counterfactual_american_stem_resample")
        ]},
        "grid": {**grid, "absolute_grid": {"43_10_slope_summary": absolute_summary}},
        "swap": {"discovery_l7_to_l11": {"raw_google_minus_alibaba_logit": {"A": 0.0, "D": 0.1}}},
        "attach": {"contrast_summaries": [
            {"wording": wording, "sentence_order": order, "l7_condition": "natural", "search_effect_larger": True,
             "search_phrase_chinese_minus_american": 1.0, "charity_phrase_chinese_minus_american": 0.1}
            for wording, order in (("primary", "engine_first"), ("primary", "charity_first"),
                                   ("heldout", "provider_first"), ("heldout", "nonprofit_first"))
        ]},
    }
    for name, payload in docs.items():
        (out / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_score_fills_amended_and_legacy_rows(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    _score_fixtures(out)
    prediction_path = tmp_path / "PREDICTIONS.md"
    row_ids = ["43.1", "43.2", "43.3", "43.4", "43.5", "43.6", "43.7", "43.1a", "43.2a", "43.3a",
               "43.4a", "43.5a", "43.6a", "43.7a", "43.8", "43.9", "43.10", "43.11", "43.12"]
    prediction_path.write_text("\n".join(["| # | Prediction | Result | Evidence |", "|---|---|---|---|"]
                                          + [f"| {key} | prediction | | |" for key in row_ids]) + "\n", encoding="utf-8")
    args = SimpleNamespace(model=part43.MODEL, revision=part43.REVISION, out=str(out), predictions=str(prediction_path))
    result = part43.run_score(args)
    assert set(result["results"]) == set(row_ids)
    assert "43.2a" in prediction_path.read_text(encoding="utf-8")
    assert result["results"]["43.11"]["result"] == "HELD"
    assert result["results"]["43.12"]["result"] == "HELD"
    assert result["results"]["43.10"]["result"] == "HELD"


def test_real_part25_candidate_token_sets_match_margin_shape():
    candidates = json.loads((part43.ROOT / "results/part25/candidates.json").read_text(encoding="utf-8"))
    token_sets = candidates["token_sets"]
    google_ids = part25.token_ids(token_sets, "plain", "google", "maker")
    qwen_ids = part25.token_ids(token_sets, "plain", "qwen", "maker")
    width = max(*google_ids, *qwen_ids) + 1
    logits = torch.zeros((1, 1, width), dtype=torch.float32)
    logits[0, 0, google_ids] = 2.0
    logits[0, 0, qwen_ids] = -1.0
    sequence = [{"variant": "plain", "stem": "maker"}]
    margin = part43.margin_values(logits, sequence, torch.tensor([0]), token_sets)
    expected = torch.logsumexp(torch.tensor([2.0] * len(google_ids)), 0) - torch.logsumexp(torch.tensor([-1.0] * len(qwen_ids)), 0)
    assert margin.shape == (1,)
    assert torch.allclose(margin[0], expected)


def test_reader_replay_error_alone_sets_epsilon_and_refuses(tmp_path, monkeypatch):
    # Claude, 2026-10-21: a sabotage dropping replay error from epsilon passed every other test.
    # Shift the replay cell equally in both legs, so spreads and bf16-fp32 differences stay 0 and only replay moves.
    args = _args(tmp_path)
    raw, tokenizer = _calibration_raw_fixture(args, monkeypatch, fp32_difference=0.0)
    google_id = part43.CALIBRATION_ANSWER_IDS["Google"]
    for leg in ("bf16", "fp32"):
        for repeat in raw["legs"][leg]["cells"]["clean_to_clean_replay"]["repeats"]:
            repeat["answer_logprobs"][google_id] += 0.06
    part43.write_json(Path(args.out) / "calibration.json", raw)
    summary = part43._read_calibration_artifact(args, tokenizer=tokenizer)
    assert summary["error_kinds"]["clean_to_clean_replay_error"] == pytest.approx(0.06)
    assert summary["error_kinds"]["bf16_fp32_abs_difference_max"] == pytest.approx(0.0)
    assert summary["epsilon"] == pytest.approx(0.06)
    assert part43.calibration_gate(summary, model=args.model, revision=args.revision)[0] is False


@pytest.mark.parametrize("raw_version, expected", [(90100, "90100"), (None, None)])
def test_runtime_environment_normalises_cudnn_version_to_text(monkeypatch, raw_version, expected):
    # torch's cudnn.version() returns an int on a CUDA build (Alden #19); the schema stores text or null.
    import torch
    monkeypatch.setattr(torch.backends.cudnn, "version", lambda: raw_version)
    assert part43._runtime_environment(TinyModel())["cudnn_version"] == expected


@pytest.mark.parametrize("value, accepted", [("90100", True), (None, True), (90100, False), (True, False)])
def test_reader_cudnn_version_type(tmp_path, monkeypatch, value, accepted):
    args = _args(tmp_path)
    raw, tokenizer = _calibration_raw_fixture(args, monkeypatch, fp32_difference=0.0)
    for leg in ("bf16", "fp32"):
        raw["execution"]["legs"][leg]["runtime"]["cudnn_version"] = value
    part43.write_json(Path(args.out) / "calibration.json", raw)
    if accepted:
        summary = part43._read_calibration_artifact(args, tokenizer=tokenizer)
        assert part43.calibration_gate(summary, model=args.model, revision=args.revision)[0] is True
    else:
        with pytest.raises(part43.CalibrationRefusal, match="cudnn_version"):
            part43._read_calibration_artifact(args, tokenizer=tokenizer)


@pytest.mark.parametrize("spread, passes", [(0.06, False), (0.04, True)])
def test_reader_bf16_spread_alone_sets_epsilon(tmp_path, monkeypatch, spread, passes):
    # Alden #19: no caller supplied a nonzero bf16 spread; this is its independent leg (difference and replay 0).
    args = _args(tmp_path)
    raw, tokenizer = _calibration_raw_fixture(args, monkeypatch, fp32_difference=0.0, bf16_repeat_spread=spread)
    part43.write_json(Path(args.out) / "calibration.json", raw)
    summary = part43._read_calibration_artifact(args, tokenizer=tokenizer)
    assert summary["error_kinds"]["bf16_repeat_spread_max"] == pytest.approx(spread)
    assert summary["error_kinds"]["bf16_fp32_abs_difference_max"] == pytest.approx(0.0, abs=1e-12)
    assert summary["error_kinds"]["clean_to_clean_replay_error"] == pytest.approx(0.0, abs=1e-12)
    assert summary["epsilon"] == pytest.approx(spread)
    assert part43.calibration_gate(summary, model=args.model, revision=args.revision)[0] is passes


def test_load_runtime_resets_observed_activation_dtypes(tmp_path, monkeypatch):
    # Alden #19: the fp32 calibration leg's observed dtype must not carry into the bf16 pilot's record.
    from types import SimpleNamespace
    monkeypatch.setattr(part43, "_ACTIVE_ACTIVATION_DTYPES", {"torch.float32"})
    monkeypatch.setattr(part43, "part39_module",
                        lambda: SimpleNamespace(load_model_direct=lambda *a, **k: (None, TinyModel())))
    args = SimpleNamespace(cache_dir=str(tmp_path), model="m", revision="r", device="cpu")
    part43._load_runtime(args)
    assert part43._ACTIVE_ACTIVATION_DTYPES == set()


def test_fp32_successor_raw_registration_and_reader_pass(tmp_path, monkeypatch):
    args = _args(tmp_path, protocol="fp32-successor")
    raw, tokenizer = _fp32_successor_raw_fixture(args, monkeypatch)
    assert raw["schema_version"] == part43.FP32_SUCCESSOR_SCHEMA_VERSION
    assert raw["protocol_id"] == part43.FP32_SUCCESSOR_PROTOCOL_ID
    assert raw["registration"]["decision_dtype"] == "torch.float32"
    assert raw["registration"]["fp32_backend_allowances"] == []
    assert raw["registration"]["down_proj_dtype_layers"] == list(range(32))
    assert set(raw["legs"]) == {"fp32"}
    assert raw["legs"]["fp32"]["repeat_count"] == 3
    runtime = raw["execution"]["legs"]["fp32"]["runtime"]
    implementations = runtime["qwen3_5_function_implementations"]
    assert set(implementations) == set(part43.FP32_QWEN_FUNCTION_NAMES)
    assert all(row["uses_torch_reference"] for row in implementations.values())
    assert runtime["hub_kernels_enabled"] is False
    assert runtime["kernelized_module_classes"] == []
    assert all(implementations[name]["resolved_implementation"] in runtime["gated_delta_backend"]
               for name in part43.FP32_QWEN_FUNCTION_NAMES[2:])
    assert all(implementations[name]["resolved_implementation"] in runtime["causal_conv_backend"]
               for name in part43.FP32_QWEN_FUNCTION_NAMES[:2])
    summary = _read_fp32_fixture(args, raw, tokenizer)
    assert summary["protocol_id"] == part43.FP32_SUCCESSOR_PROTOCOL_ID
    assert summary["status"] == "within_budget"
    assert part43.calibration_gate(
        summary, model=args.model, revision=args.revision,
        protocol_id=part43.FP32_SUCCESSOR_PROTOCOL_ID,
    )[0] is True


def test_fp32_summary_retains_legacy_cross_method_diagnostic_outside_epsilon(tmp_path, monkeypatch):
    diagnostic_reader = part43._fp32_cross_method_diagnostic
    args = _args(tmp_path, protocol="fp32-successor")
    raw, tokenizer = _fp32_successor_raw_fixture(args, monkeypatch)
    monkeypatch.setattr(part43, "_fp32_cross_method_diagnostic", diagnostic_reader)
    summary = _read_fp32_fixture(args, raw, tokenizer)
    diagnostic = summary["cross_method_diagnostic"]
    assert diagnostic["source_artifact"] == part43.FP32_CROSS_METHOD_SOURCE
    assert diagnostic["source_artifact_sha256"] == part43.FP32_CROSS_METHOD_SOURCE_SHA256
    assert diagnostic["absolute_difference_by_cell"] == FP32_CROSS_METHOD_FIXTURE_VALUES
    assert diagnostic["maximum_absolute_difference"] == pytest.approx(0.10915956082076228)
    assert diagnostic["used_for_epsilon"] is False
    assert summary["gate_interpretation"] == part43.FP32_GATE_INTERPRETATION
    assert set(summary["epsilon_components"]) == {
        "fp32_repeat_spread_max", "clean_to_clean_replay_error",
    }


def test_fp32_pilot_output_labels_gate_as_control_and_stability(tmp_path, monkeypatch):
    args = _args(tmp_path, protocol="fp32-successor")
    raw, tokenizer = _fp32_successor_raw_fixture(args, monkeypatch)
    calibration = _read_fp32_fixture(args, raw, tokenizer)
    args.attach = str(tmp_path / "attach.json")
    Path(args.attach).write_text(json.dumps({"prompts": []}), encoding="utf-8")
    monkeypatch.setattr(part43, "_sequences", lambda _tokenizer: ["tiny sequence"])
    monkeypatch.setattr(part43, "run_signs", lambda *_a, **_k: {"signed": True})
    monkeypatch.setattr(part43, "run_lattice_amended", lambda *_a, **_k: {"timing": {"elapsed_seconds": 0.1}})
    monkeypatch.setattr(part43, "run_swap", lambda *_a, **_k: {"rows": []})
    monkeypatch.setattr(part43, "run_attach", lambda *_a, **_k: {"rows": []})
    result = part43.run_pilot(
        TinyModel().to(dtype=torch.float32), tokenizer, _token_sets(), args, calibration=calibration,
    )
    assert result["status"] == "complete"
    assert result["calibration_gate_interpretation"] == part43.FP32_GATE_INTERPRETATION


@pytest.mark.parametrize("backend_problem", ["hub_kernels", "kernelized_class", "non_torch_implementation"])
def test_fp32_reader_refuses_unallowed_runtime_backends(tmp_path, monkeypatch, backend_problem):
    args = _args(tmp_path, protocol="fp32-successor")
    raw, tokenizer = _fp32_successor_raw_fixture(args, monkeypatch)
    runtime = raw["execution"]["legs"]["fp32"]["runtime"]
    if backend_problem == "hub_kernels":
        runtime["hub_kernels_enabled"] = True
    elif backend_problem == "kernelized_class":
        runtime["kernelized_module_classes"] = ["transformers.GatedDeltaNet"]
    else:
        name = "causal_conv1d_fn"
        row = runtime["qwen3_5_function_implementations"][name]
        row["resolved_implementation"] = "causal_conv1d.causal_conv1d_fn"
        row["uses_torch_reference"] = False
        runtime["causal_conv_backend"] = "; ".join(
            f"{function}={runtime['qwen3_5_function_implementations'][function]['resolved_implementation']}"
            for function in part43.FP32_QWEN_FUNCTION_NAMES[:2]
        )
    part43.write_json(Path(args.out) / "calibration.json", raw)
    with pytest.raises(part43.CalibrationRefusal, match="hub kernels|kernelized module|non-torch-reference"):
        part43._read_calibration_artifact(args, tokenizer=tokenizer)


def test_fp32_successor_reader_accepts_more_than_three_repeats(tmp_path, monkeypatch):
    args = _args(tmp_path, protocol="fp32-successor")
    raw, tokenizer = _fp32_successor_raw_fixture(args, monkeypatch)
    leg = raw["legs"]["fp32"]
    leg["repeat_count"] = 4
    for cell in leg["cells"].values():
        cell["repeats"].append({"answer_logprobs": dict(cell["repeats"][0]["answer_logprobs"])})
    summary = _read_fp32_fixture(args, raw, tokenizer)
    assert summary["status"] == "within_budget"
    assert all(len(cell["repeat_primary_scores"]) == 4 for cell in summary["scores"]["fp32"].values())


def test_legacy_reader_accepts_the_pinned_845964c_source_provenance(tmp_path, monkeypatch):
    args = _args(tmp_path)
    raw, tokenizer = _calibration_raw_fixture(args, monkeypatch)
    raw["source_provenance"] = dict(part43.LEGACY_CALIBRATION_SOURCE_PROVENANCE)
    summary = _read_fp32_fixture(args, raw, tokenizer)
    assert summary["reader_id"] == part43.CALIBRATION_READER_ID
    assert part43.calibration_gate(summary, model=args.model, revision=args.revision)[0] is True


@pytest.mark.parametrize("field, value", [
    ("dtype_as_loaded", "torch.bfloat16"),
    ("parameter_dtypes", ["torch.float32", "torch.bfloat16"]),
    ("activation_dtypes", ["torch.float32", "torch.bfloat16"]),
    ("down_proj_input_dtypes", ["torch.float32", "torch.bfloat16"]),
    ("logit_dtypes", ["torch.bfloat16"]),
])
def test_fp32_reader_refuses_wrong_or_mixed_recorded_execution_dtype(tmp_path, monkeypatch, field, value):
    args = _args(tmp_path, protocol="fp32-successor")
    raw, tokenizer = _fp32_successor_raw_fixture(args, monkeypatch)
    raw["execution"]["legs"]["fp32"][field] = value
    part43.write_json(Path(args.out) / "calibration.json", raw)
    with pytest.raises(part43.CalibrationRefusal):
        part43._read_calibration_artifact(args, tokenizer=tokenizer)


@pytest.mark.parametrize("problem", ["missing_layer", "wrong_layer_dtype"])
def test_fp32_reader_requires_float32_down_proj_evidence_for_all_32_layers(tmp_path, monkeypatch, problem):
    args = _args(tmp_path, protocol="fp32-successor")
    raw, tokenizer = _fp32_successor_raw_fixture(args, monkeypatch)
    layer_dtypes = raw["execution"]["legs"]["fp32"]["down_proj_input_dtypes_by_layer"]
    if problem == "missing_layer":
        layer_dtypes.pop("31")
    else:
        layer_dtypes["31"] = ["torch.bfloat16"]
    part43.write_json(Path(args.out) / "calibration.json", raw)
    with pytest.raises(part43.CalibrationRefusal, match="cover every registered layer|every registered layer"):
        part43._read_calibration_artifact(args, tokenizer=tokenizer)


@pytest.mark.parametrize("flag_name", ["cuda_matmul_allow_tf32", "cudnn_allow_tf32"])
def test_fp32_reader_refuses_recorded_tf32_flag_true(tmp_path, monkeypatch, flag_name):
    args = _args(tmp_path, protocol="fp32-successor")
    raw, tokenizer = _fp32_successor_raw_fixture(args, monkeypatch)
    raw["execution"]["legs"]["fp32"]["runtime"]["precision_flags"][flag_name] = True
    part43.write_json(Path(args.out) / "calibration.json", raw)
    with pytest.raises(part43.CalibrationRefusal, match="TF32-enabled"):
        part43._read_calibration_artifact(args, tokenizer=tokenizer)


def test_fp32_runtime_metadata_preserves_live_tf32_true_and_reader_refuses(tmp_path, monkeypatch):
    args = _args(tmp_path, protocol="fp32-successor")
    raw, tokenizer = _fp32_successor_raw_fixture(args, monkeypatch)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    execution = part43._calibration_runtime_metadata(
        TinyModel().to(dtype=torch.float32), "torch.float32", part43.FP32_SUCCESSOR_PROTOCOL_ID,
    )
    flags = execution["runtime"]["precision_flags"]
    assert flags["cuda_matmul_allow_tf32"] is True
    assert flags["cudnn_allow_tf32"] is True
    raw["execution"]["legs"]["fp32"] = execution
    part43.write_json(Path(args.out) / "calibration.json", raw)
    with pytest.raises(part43.CalibrationRefusal, match="TF32-enabled"):
        part43._read_calibration_artifact(args, tokenizer=tokenizer)


def test_calibration_readers_refuse_the_other_protocol(tmp_path, monkeypatch):
    legacy_dir = tmp_path / "legacy"
    fp32_dir = tmp_path / "fp32"
    legacy_dir.mkdir()
    fp32_dir.mkdir()
    legacy_args = _args(legacy_dir)
    legacy_raw, tokenizer = _calibration_raw_fixture(legacy_args, monkeypatch)
    fp32_args = _args(fp32_dir, protocol="fp32-successor")
    part43.write_json(Path(fp32_args.out) / "calibration.json", legacy_raw)
    with pytest.raises(part43.CalibrationRefusal):
        part43._read_calibration_artifact(fp32_args, tokenizer=tokenizer)

    fp32_raw, tokenizer = _fp32_successor_raw_fixture(fp32_args, monkeypatch)
    part43.write_json(Path(legacy_args.out) / "calibration.json", fp32_raw)
    with pytest.raises(part43.CalibrationRefusal):
        part43._read_calibration_artifact(legacy_args, tokenizer=tokenizer)


@pytest.mark.parametrize(
    "spread, replay, expected_epsilon, passes",
    [(0.06, 0.0, 0.06, False), (0.0, 0.06, 0.06, False), (0.0, 0.05, 0.05, True)],
)
def test_fp32_successor_epsilon_uses_spread_and_replay_exactly(
    tmp_path, monkeypatch, spread, replay, expected_epsilon, passes,
):
    args = _args(tmp_path, protocol="fp32-successor")
    raw, tokenizer = _fp32_successor_raw_fixture(
        args, monkeypatch, repeat_spread=spread, replay_error=replay,
    )
    summary = _read_fp32_fixture(args, raw, tokenizer)
    assert summary["epsilon"] == expected_epsilon
    assert summary["epsilon_components"]["fp32_repeat_spread_max"] == spread
    assert summary["epsilon_components"]["clean_to_clean_replay_error"] == replay
    assert part43.calibration_gate(
        summary, model=args.model, revision=args.revision,
        protocol_id=part43.FP32_SUCCESSOR_PROTOCOL_ID,
    )[0] is passes


def test_fp32_pilot_reads_serialized_verdict_before_any_production_stage(tmp_path, monkeypatch):
    args = _args(tmp_path, protocol="fp32-successor")
    raw, tokenizer = _fp32_successor_raw_fixture(args, monkeypatch, repeat_spread=0.06)
    monkeypatch.setattr(part43, "write_manifest", lambda _args: {"schema_version": 1})
    monkeypatch.setattr(part43, "_load_token_records", lambda _args: _token_sets())
    monkeypatch.setattr(part43, "run_calibration_runtime", lambda *_args: raw)
    monkeypatch.setattr(part43, "_load_calibration_tokenizer", lambda _args: tokenizer)
    monkeypatch.setattr(part43, "_load_runtime", lambda *_args, **_kwargs: pytest.fail("held pilot loaded model"))
    monkeypatch.setattr(part43, "_sequences", lambda *_args: pytest.fail("held pilot reached signs"))
    production_stages = []
    monkeypatch.setattr(part43, "run_signs", lambda *_a, **_k: production_stages.append("signs"))
    monkeypatch.setattr(part43, "run_lattice_amended", lambda *_a, **_k: production_stages.append("lattice"))
    common = ["--out", args.out, "--candidates", args.candidates, "--cache-dir", args.cache_dir,
              "--device", "cpu", "--protocol", "fp32-successor"]
    assert part43.main(["pilot", *common]) == 0
    assert production_stages == []
    assert json.loads((Path(args.out) / "calibration.json").read_text())["protocol_id"] == part43.FP32_SUCCESSOR_PROTOCOL_ID
    summary = json.loads((Path(args.out) / "calibration-summary.json").read_text())
    assert summary["status"] == "over_budget"
    pilot = json.loads((Path(args.out) / "pilot.json").read_text())
    assert pilot["calibration_gate_interpretation"] == part43.FP32_GATE_INTERPRETATION
    assert json.loads((Path(args.out) / "HOLD.json").read_text())["status"] == "HOLD"


def test_fp32_loader_sets_precision_before_loading_float32_model(tmp_path, monkeypatch):
    import torch
    previous_matmul = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn = torch.backends.cudnn.allow_tf32
    precision = {"value": "high"}
    seen = {}

    def set_precision(value):
        precision["value"] = value

    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", previous_matmul)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", previous_cudnn)
    monkeypatch.setattr(torch, "set_float32_matmul_precision", set_precision)
    monkeypatch.setattr(torch, "get_float32_matmul_precision", lambda: precision["value"])

    def load_model(_model, _revision, _device, **kwargs):
        seen.update(kwargs)
        seen["matmul_tf32"] = torch.backends.cuda.matmul.allow_tf32
        seen["cudnn_tf32"] = torch.backends.cudnn.allow_tf32
        seen["precision"] = precision["value"]
        return TinyTokenizer(), TinyModel().to(dtype=kwargs["dtype"])

    monkeypatch.setattr(part43, "part39_module", lambda: SimpleNamespace(load_model_direct=load_model))
    args = SimpleNamespace(cache_dir=str(tmp_path), model="m", revision="r", device="cpu", protocol="fp32-successor")
    _tokenizer, model = part43._load_runtime(args)
    assert seen == {
        "dtype": torch.float32,
        "matmul_tf32": False,
        "cudnn_tf32": False,
        "precision": "highest",
    }
    assert {str(parameter.dtype) for parameter in model.parameters()} == {"torch.float32"}


def test_fp32_calibration_runs_only_three_repeat_fp32_cells_on_cpu(tmp_path, monkeypatch):
    _fake_part25(monkeypatch)
    _install_torch_reference_qwen_functions(monkeypatch)
    monkeypatch.setenv("USE_HUB_KERNELS", "0")
    args = _args(tmp_path, protocol="fp32-successor")
    tokenizer = TinyTokenizer()
    model = TinyModel().to(dtype=torch.float32)
    import torch as torch_module
    old_matmul = torch_module.backends.cuda.matmul.allow_tf32
    old_cudnn = torch_module.backends.cudnn.allow_tf32
    old_precision = torch_module.get_float32_matmul_precision()
    loads = []

    def load(_args, *, model_dtype):
        loads.append(model_dtype)
        assert torch_module.backends.cuda.matmul.allow_tf32 is False
        assert torch_module.backends.cudnn.allow_tf32 is False
        assert torch_module.get_float32_matmul_precision() == "highest"
        return tokenizer, model

    monkeypatch.setattr(part43, "_load_runtime", load)
    monkeypatch.setattr(part43, "_release_calibration_model", lambda _args: None)
    try:
        raw = part43.run_calibration_runtime(args, _token_sets())
    finally:
        torch_module.backends.cuda.matmul.allow_tf32 = old_matmul
        torch_module.backends.cudnn.allow_tf32 = old_cudnn
        torch_module.set_float32_matmul_precision(old_precision)
    assert loads == [torch_module.float32]
    assert set(raw["legs"]) == {"fp32"}
    assert set(raw["legs"]["fp32"]["cells"]) == set(part43.CALIBRATION_CELLS)
    assert all(len(cell["repeats"]) == 3 for cell in raw["legs"]["fp32"]["cells"].values())
    assert raw["execution"]["legs"]["fp32"]["parameter_dtypes"] == ["torch.float32"]
    assert raw["execution"]["legs"]["fp32"]["down_proj_input_dtypes"] == ["torch.float32"]
    assert raw["execution"]["legs"]["fp32"]["logit_dtypes"] == ["torch.float32"]
    by_layer = raw["execution"]["legs"]["fp32"]["down_proj_input_dtypes_by_layer"]
    assert set(by_layer) == {str(layer) for layer in range(32)}
    assert all(dtypes == ["torch.float32"] for dtypes in by_layer.values())
    _read_fp32_fixture(args, raw, tokenizer)


def test_fp32_raw_boundary_keeps_duplicate_derived_malformed_and_internal_errors_distinct(tmp_path, monkeypatch):
    args = _args(tmp_path, protocol="fp32-successor")
    raw, tokenizer = _fp32_successor_raw_fixture(args, monkeypatch)
    raw["epsilon"] = 0.0
    part43.write_json(Path(args.out) / "calibration.json", raw)
    with pytest.raises(part43.CalibrationRefusal, match="derived field"):
        part43._read_calibration_artifact(args, tokenizer=tokenizer)

    raw, tokenizer = _fp32_successor_raw_fixture(args, monkeypatch)
    path = Path(args.out) / "calibration.json"
    part43.write_json(path, raw)
    serialized = path.read_text(encoding="utf-8")
    serialized = serialized.replace('"protocol_id": "part43-fp32-successor-v1",',
                                    '"protocol_id": "part43-fp32-successor-v1",\n  "protocol_id": "part43-fp32-successor-v1",', 1)
    path.write_text(serialized, encoding="utf-8")
    with pytest.raises(part43.DuplicateCalibrationKey):
        part43._read_calibration_artifact(args, tokenizer=tokenizer)

    path.write_text("{ malformed", encoding="utf-8")
    with pytest.raises(part43.CalibrationRefusal, match="malformed"):
        part43._read_calibration_artifact(args, tokenizer=tokenizer)

    raw, tokenizer = _fp32_successor_raw_fixture(args, monkeypatch)
    part43.write_json(path, raw)
    marker = RuntimeError("FP32 reader internal sentinel")
    monkeypatch.setattr(part43, "_primary_score_from_answer_rows", lambda *_a, **_k: (_ for _ in ()).throw(marker))
    with pytest.raises(RuntimeError) as exc:
        part43._read_calibration_artifact(args, tokenizer=tokenizer)
    assert exc.value is marker
