from __future__ import annotations

import json
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
        for layer_index, _unit in part43.UNITS:
            layer = self.layers[layer_index]
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
    }
    values.update(kwargs)
    return SimpleNamespace(**values)


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
    commands = {"manifest", "validate", "pilot", "signs", "dla", "lattice", "mediate", "rescue", "shift",
                "positions", "baselines", "heldout", "readout-check", "links", "sweep", "dose", "grid", "base", "score"}
    parser = part43._build_parser()
    for command in commands:
        args = parser.parse_args([command])
        assert args.command == command
    base = parser.parse_args(["base"])
    assert base.model == part43.BASE_MODEL and base.revision == part43.BASE_REVISION

    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: TinyTokenizer())
    args = _args(tmp_path)
    manifest = part43.write_manifest(args)
    assert manifest["tokenizer"]["chat_template_sha256"]
    assert len(manifest["factorial_conditions"]["lattice"]) == 16
    assert (Path(args.out) / "MANIFEST.json").is_file()


def test_pilot_and_base_cli_smoke_with_tiny_model(tmp_path, monkeypatch):
    _fake_part25(monkeypatch)
    model, tokenizer = TinyModel(), TinyTokenizer()
    args = _args(tmp_path)
    monkeypatch.setattr(part43, "write_manifest", lambda _args: {"schema_version": 1})
    monkeypatch.setattr(part43, "_load_runtime", lambda _args: (tokenizer, model))
    monkeypatch.setattr(part43, "run_validation_checks", lambda *_args: {"checks": {"tiny": {"passed": True}}})
    monkeypatch.setattr(part43, "run_lattice_amended", lambda *_args, **_kwargs: {"timing": {"elapsed_seconds": 0.1}})
    out = tmp_path / "cli-out"
    common = ["--out", str(out), "--candidates", args.candidates, "--cache-dir", str(tmp_path),
              "--heldout", args.heldout, "--prompts", args.prompts, "--corpus", args.corpus]
    assert part43.main(["pilot", *common]) == 0
    assert (out / "pilot.json").is_file() and (out / "pilot-lattice.json").is_file()
    assert part43.main(["base", *common]) == 0
    assert (out / "base.json").is_file() and (out / "base-signs.json").is_file()


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
        "grid": grid,
    }
    for name, payload in docs.items():
        (out / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_score_fills_amended_and_legacy_rows(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    _score_fixtures(out)
    prediction_path = tmp_path / "PREDICTIONS.md"
    row_ids = ["43.1", "43.2", "43.3", "43.4", "43.5", "43.6", "43.7", "43.1a", "43.2a", "43.3a",
               "43.4a", "43.5a", "43.6a", "43.7a", "43.8", "43.9", "43.10"]
    prediction_path.write_text("\n".join(["| # | Prediction | Result | Evidence |", "|---|---|---|---|"]
                                          + [f"| {key} | prediction | | |" for key in row_ids]) + "\n", encoding="utf-8")
    args = SimpleNamespace(model=part43.MODEL, revision=part43.REVISION, out=str(out), predictions=str(prediction_path))
    result = part43.run_score(args)
    assert set(result["results"]) == set(row_ids)
    assert "43.2a" in prediction_path.read_text(encoding="utf-8")


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
