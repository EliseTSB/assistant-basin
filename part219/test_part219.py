"""Part 219's tests: the real swap99 paths on a tiny random OLMo 2, and the scorer on synthetic labels.

The tiny model is tests/test_olmo2.py's fixture shape (16 hidden, 3 layers, 2 query heads and 2 key/value heads,
head_dim 8) built as an ``Olmo2ForCausalLM`` with a word-level tokenizer, so `swap99.run` goes through its whole
command path -- model and tokenizer load, fixed pushes, hooks, generation, receipts, meta -- on the CPU, with no
network and no real weights. Each test's sabotage is written down in part219/REPORT.md.

Run with: python -I -m pytest part219 part99 -q
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import Olmo2Config, Olmo2ForCausalLM, PreTrainedTokenizerFast

from part92 import common92 as c
from part99 import swap99
from part219 import run219, score219

D = 16
N_LAYERS = 3
N_HEADS = 2
HEAD_DIM = 8
WORDS = ["<pad>", "<bos>", "<eos>", "Who", "are", "you", "?", "Tell", "me", "about", "yourself",
         "The", "cat", "sat", "on", "mat", "say", "hello", "world", "I", "am", "a", "person", "."]


# --- the tiny snapshots -----------------------------------------------------------------------------------------

def build_snapshot(directory: Path, seed: int) -> Path:
    """A tiny OLMo 2 causal LM plus a word-level tokenizer and a chat template, saved for `load_model`."""
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    config = Olmo2Config(vocab_size=len(WORDS), hidden_size=D, intermediate_size=32, num_hidden_layers=N_LAYERS,
                         num_attention_heads=N_HEADS, num_key_value_heads=N_HEADS, head_dim=HEAD_DIM,
                         attn_implementation="eager", bos_token_id=1, eos_token_id=2, pad_token_id=0,
                         max_position_embeddings=512)
    model = Olmo2ForCausalLM(config).float().eval()
    torch.manual_seed(seed + 1)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("norm.weight"):
                parameter.uniform_(0.3, 2.0)
    model.save_pretrained(directory, safe_serialization=False)
    vocab = {word: index for index, word in enumerate(WORDS)}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<pad>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="<pad>", bos_token="<bos>",
                                   eos_token="<eos>", unk_token="<pad>")
    fast.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    fast.save_pretrained(directory)
    return directory


@pytest.fixture(scope="module")
def snapshots(tmp_path_factory):
    root = tmp_path_factory.mktemp("snapshots")
    return build_snapshot(root / "host", 0), build_snapshot(root / "donor", 7)


@pytest.fixture()
def workspace(tmp_path, snapshots):
    """A run directory with the tiny neuron list, kinds map and fixed pushes."""
    host, donor = snapshots
    neurons = [{"id": "t-0", "layer": 1, "neuron": 3, "control_seed": 11},
               {"id": "t-1", "layer": 2, "neuron": 5, "control_seed": 12}]
    (tmp_path / "neurons.json").write_text(json.dumps(neurons), encoding="utf-8")
    (tmp_path / "kinds.json").write_text(json.dumps({"kinds": {"t-0": "slot", "t-1": "concept"}}),
                                         encoding="utf-8")
    fixed = {}
    for neuron in neurons:
        for unit in ("neuron", "random"):
            generator = torch.Generator().manual_seed(int(neuron["control_seed"]) + (unit == "random"))
            fixed[f"{neuron['id']}|{unit}"] = torch.randn(D, generator=generator) * 0.1
    torch.save(fixed, tmp_path / "pushes.pt")
    return {"root": tmp_path, "host": host, "donor": donor, "out": tmp_path / "out"}


def make_args(workspace, tag, part219=True, **overrides):
    """swap99's arguments. With part219=False the Part 219 options are not present at all, which is how Part 99 to
    104 call it."""
    values = dict(host=str(workspace["host"]), donor=None, swap="none", tokenizer=None,
                  neurons=str(workspace["root"] / "neurons.json"), frame="chat", tag=tag,
                  out=str(workspace["out"]), device="cpu", batch=8, kinds=None, only=None,
                  pushes=str(workspace["root"] / "pushes.pt"), limit=None, layers=None,
                  extrapolate=False, questions=None)
    if part219:
        values.update(dict(attn_implementation="eager", ablate_heads=None, transplant_heads=None,
                           transplant_mode=None, reference_replay=None, record_token_ids=True))
    values.update(overrides)
    return argparse.Namespace(**values)


def run_tag(workspace, tag, **overrides):
    """Run one tag through swap99.run and return (rows, meta)."""
    swap99.run(make_args(workspace, tag, **overrides))
    rows = {}
    with (workspace["out"] / f"{tag}.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            rows[(row["id"], row["unit"], int(row["question_index"]))] = row
    meta = json.loads((workspace["out"] / f"{tag}.meta.json").read_text(encoding="utf-8"))
    return rows, meta


def capture_o_proj_inputs(model, input_ids, layers):
    """The tensor each layer's o_proj receives and the model's logits, for one forward.

    Hooks registered after a pre-hook see that hook's modified args (torch applies pre-hooks in order), which is how
    the ablation's zeroed slice is observed rather than the unmodified input."""
    captured = {}
    handles = [model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(
        lambda _module, args, layer=layer: captured.__setitem__(layer, args[0].detach().clone()))
        for layer in layers]
    try:
        with torch.inference_mode():
            logits = model(input_ids=input_ids).logits.detach().clone()
    finally:
        for handle in handles:
            handle.remove()
    return captured, logits


# --- 1. the default path is unchanged ---------------------------------------------------------------------------

def test_generate_batch_matches_the_padding_rule_it_had_before(workspace):
    """`generate_batch` now builds its batch through `batch_inputs`; the generated answers must be identical to the
    pre-change padding rule written out here by hand."""
    tokenizer, model = c.load_model(str(workspace["host"]), "cpu", attn_implementation="eager")
    questions = ["Who are you?", "Tell me about yourself."]
    pushes = torch.zeros((len(questions), D))
    got = c.generate_batch(model, tokenizer, questions, "chat", model.model.layers[1], pushes, 6, "cpu")

    pad_id = int(tokenizer.pad_token_id)
    sequences = [c.prompt_ids(tokenizer, question, "chat") for question in questions]
    width = max(len(sequence) for sequence in sequences)
    input_ids = torch.tensor([[pad_id] * (width - len(sequence)) + sequence for sequence in sequences])
    attention_mask = torch.tensor([[0] * (width - len(sequence)) + [1] * len(sequence) for sequence in sequences])
    with torch.inference_mode():
        generated = model.generate(input_ids=input_ids, attention_mask=attention_mask, do_sample=False,
                                   max_new_tokens=6, pad_token_id=pad_id)
    expected = [tokenizer.decode(row, skip_special_tokens=True).strip() for row in generated[:, width:]]
    assert got == expected


def test_a_default_run_writes_no_part219_keys(workspace):
    """A run with no Part 219 option must write the meta and rows Part 99 wrote: no new keys, no token ids."""
    rows, meta = run_tag(workspace, "plain", part219=False)
    for key in ("attn_implementation", "ablate_heads", "transplant_heads", "intervention_receipt",
                "coupling_log", "reset_receipt", "token_ids_recorded"):
        assert key not in meta, key
    assert all("token_ids" not in row for row in rows.values())
    assert meta["swap"] == "none" and meta["frame"] == "chat" and meta["tensors_copied"] == 0


def test_the_attention_implementation_option_reaches_every_load(workspace):
    _, meta = run_tag(workspace, "eager-run", attn_implementation="eager")
    assert meta["attn_implementation"] == "eager"
    tokenizer, model = c.load_model(str(workspace["host"]), "cpu", attn_implementation="eager")
    assert model.config._attn_implementation == "eager"
    # And the default load is still the library's own choice, not this option's value.
    from transformers import AutoModelForCausalLM
    library = AutoModelForCausalLM.from_pretrained(str(workspace["host"]), torch_dtype=torch.bfloat16,
                                                    local_files_only=True)
    _, default_model = c.load_model(str(workspace["host"]), "cpu")
    assert default_model.config._attn_implementation == library.config._attn_implementation


# --- 2. empty head sets change nothing --------------------------------------------------------------------------

def test_empty_head_sets_are_bit_identical(workspace):
    """R0b: both new machineries installed with empty sets must reproduce R0 token for token and logit for logit."""
    plain_rows, plain_meta = run_tag(workspace, "R0", reference_replay=str(workspace["out"] / "replay.json"))
    empty_rows, empty_meta = run_tag(
        workspace, "R0b", ablate_heads="", transplant_heads="", transplant_mode="donor",
        reference_replay=str(workspace["out"] / "replay.json"))
    assert empty_meta["intervention_receipt"]["ablate_heads"] == []
    assert empty_meta["intervention_receipt"]["transplant"]["heads"] == []
    assert empty_meta["intervention_receipt"]["ablation"] == []
    assert empty_meta["reset_receipt"]["matches_reference"] is True
    assert empty_meta["reset_receipt"]["first_step_logits"] == plain_meta["reset_receipt"]["first_step_logits"]
    assert set(empty_rows) == set(plain_rows)
    for key in plain_rows:
        assert empty_rows[key]["token_ids"] == plain_rows[key]["token_ids"], key
        assert empty_rows[key]["answer"] == plain_rows[key]["answer"], key


# --- 3. ablation zeroes exactly the named slice -----------------------------------------------------------------

def test_ablation_zeroes_exactly_the_named_slice(snapshots):
    host, _donor = snapshots
    _tokenizer, model = c.load_model(str(host), "cpu", attn_implementation="eager")
    input_ids = torch.tensor([c.prompt_ids(_tokenizer, "Who are you?", "chat")], dtype=torch.long)
    clean, clean_logits = capture_o_proj_inputs(model, input_ids, range(N_LAYERS))

    weights_before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    handles, receipt = swap99.install_ablation_hooks(model, [(1, 1)])
    try:
        dirty, dirty_logits = capture_o_proj_inputs(model, input_ids, range(N_LAYERS))
    finally:
        for handle in handles:
            handle.remove()

    head_size = D // N_HEADS
    start, end = head_size, 2 * head_size
    assert torch.count_nonzero(dirty[1][..., start:end]) == 0
    assert torch.equal(dirty[1][..., :start], clean[1][..., :start])
    assert torch.equal(dirty[1][..., end:], clean[1][..., end:])
    # Layers 0 and 2 share layer 1's input only through the embeddings: layer 0 must be untouched, and anything
    # downstream of the zeroed head legitimately changes (that is the intervention working, not a leak).
    assert torch.equal(dirty[0], clean[0])
    assert not torch.equal(dirty_logits, clean_logits)  # the zeroed head does reach the output
    for entry in receipt:
        assert entry["positions"] == int(clean[1].numel() // clean[1].shape[-1])
    assert [entry["head"] for entry in receipt] == [1]
    assert receipt[0]["all_zero"] is True
    assert receipt[0]["max_abs_before"] > 0
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, weights_before[name]), name


def test_ablation_refuses_a_grouped_query_stack():
    torch.manual_seed(0)
    config = Olmo2Config(vocab_size=len(WORDS), hidden_size=D, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=HEAD_DIM, attn_implementation="eager")
    model = Olmo2ForCausalLM(config).float().eval()
    with pytest.raises(ValueError, match="num_key_value_heads"):
        swap99.install_ablation_hooks(model, [(0, 0)])


# --- 4. the transplant writes exactly the expected bytes ---------------------------------------------------------

def _slice_of(parameter, part, start, end):
    if parameter.ndim == 1:
        return parameter[start:end]
    if part in ("q_proj", "k_proj", "v_proj"):
        return parameter[start:end, :]
    return parameter[:, start:end]


def test_transplant_writes_exactly_the_expected_bytes(snapshots):
    host_path, donor_path = snapshots
    head_size = D // N_HEADS
    for mode in ("donor", "opposite"):
        host = swap99.load_donor(str(host_path))
        donor = swap99.load_donor(str(donor_path))
        original = {name: parameter.detach().clone() for name, parameter in host.named_parameters()}
        donor_params = dict(donor.named_parameters())
        result = swap99.transplant_heads(host, donor, [(0, 1), (2, 0)], mode)
        assert result["parameters_changed"] == 12
        for layer, head in ((0, 1), (2, 0)):
            start, end = head * head_size, (head + 1) * head_size
            for part in swap99.TRANSPLANT_PARTS:
                name = f"model.layers.{layer}.self_attn.{part}.weight"
                parameter = dict(host.named_parameters())[name]
                before = _slice_of(original[name], part, start, end)
                reference = _slice_of(donor_params[name], part, start, end)
                if mode == "donor":
                    expected = reference.to(parameter.dtype)
                else:
                    expected = (2.0 * before.to(torch.float32) - reference.to(torch.float32)).to(parameter.dtype)
                assert torch.equal(_slice_of(parameter, part, start, end), expected), (mode, layer, head, part)
                assert not torch.equal(expected, before), (mode, layer, head, part)  # the donor really differs
                mask = torch.zeros_like(parameter, dtype=torch.bool)
                if parameter.ndim == 1:
                    mask[start:end] = True
                elif part in ("q_proj", "k_proj", "v_proj"):
                    mask[start:end, :] = True
                else:
                    mask[:, start:end] = True
                assert torch.equal(parameter[~mask], original[name][~mask]), (mode, layer, head, part)
        for name, parameter in host.named_parameters():
            if "layers.0.self_attn" in name or "layers.2.self_attn" in name:
                continue
            assert torch.equal(parameter, original[name]), name
        swap99.restore_transplant(result["restore"])
        for name, parameter in host.named_parameters():
            assert torch.equal(parameter, original[name]), name


def test_transplant_refuses_a_donor_with_different_shapes(tmp_path, snapshots):
    host_path, _donor = snapshots
    torch.manual_seed(3)
    config = Olmo2Config(vocab_size=len(WORDS), hidden_size=D, intermediate_size=32, num_hidden_layers=N_LAYERS,
                         num_attention_heads=4, num_key_value_heads=4, head_dim=4, attn_implementation="eager")
    other = tmp_path / "other"
    other.mkdir()
    Olmo2ForCausalLM(config).float().eval().save_pretrained(other, safe_serialization=False)
    host = swap99.load_donor(str(host_path))
    donor = swap99.load_donor(str(other))
    with pytest.raises(ValueError):
        swap99.transplant_heads(host, donor, [(0, 1)], "donor")


# --- 5. the reset receipt restores bit-identical logits ---------------------------------------------------------

def test_reset_receipt_restores_bit_identical_logits(workspace):
    tokenizer, model = c.load_model(str(workspace["host"]), "cpu", attn_implementation="eager")
    pristine = swap99.replay_receipt(model, tokenizer, "chat", "cpu")
    reference = workspace["out"] / "replay.json"
    reference.parent.mkdir(parents=True, exist_ok=True)
    reference.write_text(json.dumps(pristine), encoding="utf-8")

    for tag, overrides in (("ablated", {"ablate_heads": "L0H1,L2H0"}),
                           ("transplanted", {"donor": str(workspace["donor"]),
                                             "transplant_heads": "L0H1,L2H0"})):
        _, meta = run_tag(workspace, tag, reference_replay=str(reference), **overrides)
        receipt = meta["reset_receipt"]
        assert receipt["role"] == "checked"
        assert receipt["matches_reference"] is True, receipt["mismatch"]
        assert receipt["first_step_logits"] == pristine["first_step_logits"]
        assert receipt["token_ids"] == pristine["token_ids"]

    # And the same check against the model in memory, after the run's restore: logits are bit-identical.
    again = swap99.replay_receipt(model, tokenizer, "chat", "cpu")
    assert again["first_step_logits"] == pristine["first_step_logits"]


def test_reset_receipt_is_written_by_the_reference_run(workspace):
    reference = workspace["out"] / "replay.json"
    _, meta = run_tag(workspace, "R0", reference_replay=str(reference))
    assert meta["reset_receipt"]["role"] == "reference_written"
    assert reference.exists()
    stored = json.loads(reference.read_text(encoding="utf-8"))
    assert stored["prompt_ids"] and stored["token_ids"]
    assert len(stored["first_step_logits"]) == len(WORDS)


# --- 6. the coupling log ----------------------------------------------------------------------------------------

def test_coupling_log_fields_are_present_and_finite(workspace):
    _, meta = run_tag(workspace, "R8", donor=str(workspace["donor"]), transplant_heads="L0H1,L2H0",
                      transplant_mode="donor")
    coupling = meta["coupling_log"]
    assert coupling is not None
    assert [layer["layer"] for layer in coupling["layers"]] == [0, 2]
    for layer in coupling["layers"]:
        for name in ("rms_q", "rms_k", "rms_attention_output"):
            numbers = [layer[name]["before"], layer[name]["after"], layer[name]["relative_change"]]
            assert all(isinstance(value, float) for value in numbers), (name, numbers)
            assert all(value == value and abs(value) != float("inf") for value in numbers), (name, numbers)
        drift = layer["sibling_output_drift"]
        assert len(drift["heads"]) == N_HEADS - 1
        assert all(entry["relative_change"] is not None for entry in drift["heads"])
        assert drift["mean_relative_change"] is not None and drift["max_relative_change"] is not None
    records = meta["intervention_receipt"]["transplant"]["records"]
    assert records and all("norm_delta" in record and "norm_ratio" in record for record in records)
    assert all(record["matches_expected"] for record in records)
    assert all(record["norm_delta"] >= 0 for record in records)


def test_transplant_receipt_holds_the_expected_bytes(workspace):
    _, meta = run_tag(workspace, "R10", donor=str(workspace["donor"]), transplant_heads="L0H1,L2H0",
                      transplant_mode="opposite")
    records = meta["intervention_receipt"]["transplant"]["records"]
    assert len(records) == 12
    assert all(record["matches_expected"] for record in records)
    assert meta["transplant_mode"] == "opposite"


# --- the scorer on synthetic labels -----------------------------------------------------------------------------

SLOT = [f"s{index:02d}" for index in range(1, 22)]
CONCEPT = [f"c{index:02d}" for index in range(1, 19)]
# The registered shape: 5 slot units at layers 1 and 2, 16 at layers 3 to 12.
SLOT_LAYERS = dict(zip(SLOT, [1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 5, 5, 6, 9, 12, 3, 4, 4, 5, 3, 3]))
UPSTREAM = [unit for unit in SLOT if SLOT_LAYERS[unit] >= 3]
DOWNSTREAM = [unit for unit in SLOT if SLOT_LAYERS[unit] < 3]
CONCEPT_LAYERS = {unit: 3 + index % 8 for index, unit in enumerate(CONCEPT)}


def write_world(directory: Path, show, coherent=None, pass2=None, manifest_readable=None):
    """A synthetic results directory: runs, two label passes and the kinds/neurons files."""
    directory.mkdir(parents=True, exist_ok=True)
    coherent = coherent or {}
    kinds = {unit: "slot" for unit in SLOT} | {unit: "concept" for unit in CONCEPT}
    (directory / "kinds.json").write_text(json.dumps({"kinds": kinds}), encoding="utf-8")
    neurons = ([{"id": unit, "layer": SLOT_LAYERS[unit], "neuron": index} for index, unit in enumerate(SLOT)]
               + [{"id": unit, "layer": CONCEPT_LAYERS[unit], "neuron": index}
                  for index, unit in enumerate(CONCEPT)])
    (directory / "neurons.json").write_text(json.dumps(neurons), encoding="utf-8")
    tags = [tag for tag in score219.RUN_TAGS if tag in show]
    for tag in tags:
        showing = set(show[tag])
        second = set((pass2 or {}).get(tag, showing))
        with (directory / f"{tag}.jsonl").open("w", encoding="utf-8") as stream:
            for unit in SLOT + CONCEPT:
                for question in range(6):
                    stream.write(json.dumps({"id": unit, "unit": "neuron", "question_index": question}) + "\n")
            for question in range(6):
                stream.write(json.dumps({"id": "unpushed", "unit": "none", "question_index": question}) + "\n")
        for name, chosen in ((f"labels-{tag}.jsonl", showing), (f"labels2-{tag}.jsonl", second)):
            with (directory / name).open("w", encoding="utf-8") as stream:
                for unit in SLOT + CONCEPT:
                    for question in range(6):
                        shows = unit in chosen and (kinds[unit] == "concept" or question == 0)
                        stream.write(json.dumps({
                            "id": unit, "unit": "neuron", "question_index": question,
                            "shows_persona": bool(shows),
                            "coherent": bool(coherent.get((tag, unit, question), True))}) + "\n")
                    for question in range(6):
                        stream.write(json.dumps({
                            "id": unit, "unit": "baseline", "question_index": question,
                            "shows_persona": False,
                            "coherent": bool(coherent.get((tag, unit, "baseline", question), True))}) + "\n")
    manifest = {"part": 219,
                "runs": {tag: {"readable": (manifest_readable or {}).get(tag, True), "problems": []} for tag in tags}}
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory


def write_kernel(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    record = {"model": "allenai/OLMo-2-0425-1B-Instruct", "revision": "48d788eca847d4d7548f375ad03d3c9312f6139e",
              "prompt_sha256": "0" * 64, "new_tokens": 4, "argmax_agrees": True,
              "runs": {"48": [{"prompt": 0, "tokens": [1, 2, 3, 4], "margins": [1.0, 1.0, 1.0, 1.0]}]}}
    reference = directory / "reference.json"
    reference.write_text(json.dumps(record), encoding="utf-8")
    candidate = dict(record, reference=str(reference))
    path = directory / "kernel-check.json"
    path.write_text(json.dumps(candidate), encoding="utf-8")
    return path


def baseline_show():
    """A world that satisfies the gate and every prediction, to be broken one piece at a time."""
    r0 = set(SLOT[:19]) | set(CONCEPT[:16])
    show = {"R0": r0, "R1": set(SLOT[:2]) | set(SLOT[5:8]) | set(CONCEPT[:15])}
    show["R2"] = (set(SLOT[:19]) - set(UPSTREAM[:8])) | set(CONCEPT[:14])
    show["R3"] = set(SLOT[:16]) | set(CONCEPT[:15])
    show["R4"] = set(SLOT[:17]) | set(CONCEPT[:15])
    show["R5"] = set(SLOT[:18]) | set(CONCEPT[:16])
    show["R6"] = set(SLOT[:18]) | set(CONCEPT[:15])
    show["R7"] = set(SLOT[:18]) | set(CONCEPT[:15])
    show["R8"] = set(SLOT[:10]) | set(CONCEPT[:14])
    show["R9"] = set(SLOT[:18]) | set(CONCEPT[:16])
    show["R10"] = set(SLOT[:17]) | set(CONCEPT[:15])
    return show


def build(tmp_path, show, **kwargs):
    out = write_world(tmp_path / "results", show, **kwargs)
    kernel = write_kernel(tmp_path / "kernel")
    return score219.build_scorecard(out, out / "kinds.json", out / "neurons.json", kernel)


def test_split_is_computed_from_the_neuron_list(tmp_path):
    card = build(tmp_path, baseline_show())
    assert card["split"]["n_upstream"] == 16
    assert card["split"]["n_downstream"] == 5
    assert set(card["split"]["downstream"]) == set(DOWNSTREAM)
    assert card["gate"]["passed"] is True


def test_the_registered_predictions_hold_in_a_world_built_so_they_do(tmp_path):
    card = build(tmp_path, baseline_show())
    verdicts = {key: value["verdict"] for key, value in card["predictions"].items()}
    assert verdicts["219.2"] == "held"
    assert verdicts["219.1"] == "held"
    assert card["nets_upstream_against_R0"]["R2"]["net"] == 8
    assert verdicts["219.3"] == "held"
    assert verdicts["219.5"] == "held"
    assert verdicts["219.4"] == "held"
    assert card["controls"]["R5_rig_check"]["held"] is True
    assert card["controls"]["R9_coupling_check"]["held"] is True
    assert "needed for the slot neurons" in card["headlines"][0]


def test_the_scorer_reports_the_negative_when_r2_loses_too_few(tmp_path):
    show = baseline_show()
    show["R2"] = set(SLOT[:17]) | set(CONCEPT[:14])  # 2 upstream units lost, under the registered 7
    card = build(tmp_path, show)
    assert card["predictions"]["219.1"]["verdict"] == "failed"
    assert card["predictions"]["219.2"]["verdict"] == "held"
    assert card["nets_upstream_against_R0"]["R2"]["mcnemar_p"] == pytest.approx(
        score219.mcnemar_exact(2, 0))


def test_219_3_holds_but_a_broken_r8_reads_inconclusive(tmp_path):
    """The design's rule: if the transplant breaks answers, 219.3 is not read even when its net count holds."""
    show = baseline_show()
    coherent = {}
    for unit in SLOT[:12]:  # twelve slot answers incoherent: under the 15-of-21 bar
        coherent[("R8", unit, 0)] = False
    card = build(tmp_path, show, coherent=coherent)
    assert card["nets_upstream_against_R0"]["R8"]["net"] >= score219.NET_MIN  # the count alone would say held
    assert card["runs"]["R8"]["slot_who_coherent"][0] < score219.BAR_SLOT_COHERENT
    for key in ("219.3", "219.4", "219.5"):
        assert card["predictions"][key]["verdict"] == "inconclusive: the transplant breaks answers", key


def test_a_broken_receipt_marks_the_run_unreadable_and_blocks_its_prediction(tmp_path):
    card = build(tmp_path, baseline_show(), manifest_readable={"R8": False})
    assert card["runs"]["R8"]["status"]
    assert card["predictions"]["219.3"]["verdict"].startswith("not read")
    assert card["predictions"]["219.1"]["verdict"] == "held"  # R2 is untouched


def test_the_42_item_gate_fails_though_a_pooled_gate_would_pass(tmp_path):
    """Two passes that agree on every concept item and disagree on three slot items: pooled agreement is 255 of 258,
    over any 95% pooled gate, while the registered 42-item gate (at least 40 of 42) fails at 39."""
    show = baseline_show()
    disagree = set(SLOT[:3])  # three slot units whose pushed answer pass 2 reads differently
    pass2 = {"R0": (set(show["R0"]) - disagree) | set()}
    # pass 2 withholds the persona on those three pushed answers; the pooled comparison is over both passes' rows.
    out = write_world(tmp_path / "results", show, pass2=pass2)
    card = score219.build_scorecard(out, out / "kinds.json", out / "neurons.json", write_kernel(tmp_path / "kernel"))

    def pooled_agreement(tag):
        first, _e, _m = score219.load_labels(out / f"labels-{tag}.jsonl")
        second, _e2, _m2 = score219.load_labels(out / f"labels2-{tag}.jsonl")
        shared = set(first) & set(second)
        agree = sum(1 for key in shared if bool(first[key]["shows_persona"]) == bool(second[key]["shows_persona"]))
        return agree, len(shared)

    agree, total = pooled_agreement("R0")
    assert agree / total >= 0.95, (agree, total)
    assert card["runs"]["R0"]["two_pass"]["compared"] == 42
    assert card["runs"]["R0"]["two_pass"]["shows_persona_agree"] == 39
    assert card["runs"]["R0"]["two_pass"]["passes"] is False
    assert card["runs"]["R0"]["status"]  # and the run is not read


def test_the_reader_gate_names_which_criterion_failed(tmp_path):
    show = baseline_show()
    pass2 = {"R1": set(show["R1"]) | set(SLOT[5:12])}  # slot counts seven apart
    out = write_world(tmp_path / "results", show, pass2=pass2)
    card = score219.build_scorecard(out, out / "kinds.json", out / "neurons.json", write_kernel(tmp_path / "kernel"))
    gate = card["runs"]["R1"]["two_pass"]
    assert gate["passes"] is False
    assert any("apart" in reason for reason in gate["reasons"])


def test_r5_failing_its_rig_check_freezes_219_1_and_219_2(tmp_path):
    show = baseline_show()
    show["R5"] = set(SLOT[:15]) | set(CONCEPT[:16])  # four upstream units lost, over the rig's 2
    card = build(tmp_path, show)
    assert card["controls"]["R5_rig_check"]["held"] is False
    assert card["predictions"]["219.1"]["verdict"].startswith("not read")
    assert card["predictions"]["219.2"]["verdict"].startswith("not read")
    assert card["headlines"] == ["no headline: R5 failed its rig check"]


def test_r9_failing_its_coupling_check_blocks_219_3_to_219_5(tmp_path):
    show = baseline_show()
    show["R9"] = set(SLOT[:14]) | set(CONCEPT[:16])  # four upstream units lost by the coupling control
    card = build(tmp_path, show)
    assert card["controls"]["R9_coupling_check"]["held"] is False
    for key in ("219.3", "219.4", "219.5"):
        assert card["predictions"][key]["verdict"].startswith("not read"), key


def test_failed_gate_freezes_every_prediction(tmp_path):
    show = baseline_show()
    show["R0"] = set(SLOT[:15]) | set(CONCEPT[:16])  # 15 of 21, under the gate's 17
    card = build(tmp_path, show)
    assert card["gate"]["passed"] is False
    assert card["gate"]["failures"] == ["r0_slot_persona"]
    for key in ("219.1", "219.2", "219.3", "219.4", "219.5"):
        assert card["predictions"][key]["verdict"].startswith("not read"), key
        assert card["predictions"][key]["exploratory"] is True


def test_a_missing_kernel_record_fails_the_gate(tmp_path):
    show = baseline_show()
    out = write_world(tmp_path / "results", show)
    card = score219.build_scorecard(out, out / "kinds.json", out / "neurons.json",
                                    tmp_path / "nowhere" / "kernel-check.json")
    assert card["gate"]["passed"] is False
    assert "kernel_check" in card["gate"]["failures"]


def test_219_4_is_not_read_unless_219_3_and_219_5_hold(tmp_path):
    show = baseline_show()
    show["R10"] = set(SLOT[:17]) | set(CONCEPT[:15])
    show["R8"] = set(SLOT[:13]) | set(CONCEPT[:14])  # net 6: 219.3 fails, 219.5's margin also fails
    card = build(tmp_path, show)
    assert card["predictions"]["219.3"]["verdict"] == "failed"
    assert card["predictions"]["219.4"]["verdict"].startswith("not read")


def test_mcnemar_is_the_exact_two_sided_binomial():
    assert score219.mcnemar_exact(0, 0) == 1.0
    assert score219.mcnemar_exact(1, 0) == 1.0
    assert score219.mcnemar_exact(5, 0) == pytest.approx(2 / 32)
    assert score219.mcnemar_exact(7, 1) == pytest.approx(2 * (1 + 8) / 256)
    assert score219.mcnemar_exact(3, 3) == 1.0


def test_pilot_labels_reads_the_two_pilot_runs(tmp_path):
    show = {"R0": set(SLOT[:19]) | set(CONCEPT[:16]),
            "R2": set(SLOT[:11]) | set(CONCEPT[:15])}
    out = write_world(tmp_path / "results", show)
    labels = score219.pilot_labels(out, out / "kinds.json")
    assert labels["available"] is True
    assert labels["r0_slot_who"] == [19, 21]
    assert labels["r2_slot_who_coherent"] == [21, 21]


# --- the runner's registered lists and pilot rule ------------------------------------------------------------------

def test_the_frozen_head_list_is_the_registered_one():
    """run219 asserts the list rather than hard-coding it; the assertion must reproduce PART219-DESIGN.md's list."""
    assert run219.registered_frozen_heads() == list(run219.FROZEN_HEADS)
    assert run219.assert_registered_heads() == list(run219.FROZEN_HEADS)
    assert len(run219.registered_frozen_heads()) == 4


def test_pilot_stops_when_r0_shows_too_few_slot_units(tmp_path):
    show = {"R0": set(SLOT[:15]) | set(CONCEPT[:16]), "R2": set(SLOT[:11]) | set(CONCEPT[:15])}
    out = write_world(tmp_path / "results", show)
    verdict = run219.pilot_verdict(out, argparse.Namespace(kinds=str(out / "kinds.json")))
    assert verdict["decision"] == "stop"
    assert "under 17" in verdict["because"]


def test_pilot_stops_when_r2_leaves_too_few_slot_answers_coherent(tmp_path):
    show = {"R0": set(SLOT[:19]) | set(CONCEPT[:16]), "R2": set(SLOT[:11]) | set(CONCEPT[:15])}
    coherent = {("R2", unit, 0): False for unit in SLOT[:8]}  # eight incoherent: under the 15-of-21 bar
    out = write_world(tmp_path / "results", show, coherent=coherent)
    verdict = run219.pilot_verdict(out, argparse.Namespace(kinds=str(out / "kinds.json")))
    assert verdict["decision"] == "stop"
    assert "under 15" in verdict["because"]


def test_pilot_continues_when_every_criterion_holds(tmp_path):
    show = {"R0": set(SLOT[:19]) | set(CONCEPT[:16]), "R2": set(SLOT[:11]) | set(CONCEPT[:15])}
    out = write_world(tmp_path / "results", show)
    verdict = run219.pilot_verdict(out, argparse.Namespace(kinds=str(out / "kinds.json")))
    assert verdict["decision"] == "continue"
    assert verdict["r0_slot_persona"] == [19, 21]


def test_pilot_is_pending_without_labels(tmp_path):
    show = {"R0": set(SLOT[:19]) | set(CONCEPT[:16]), "R2": set(SLOT[:11]) | set(CONCEPT[:15])}
    out = write_world(tmp_path / "results", show)
    (out / "labels-R0.jsonl").unlink()
    verdict = run219.pilot_verdict(out, argparse.Namespace(kinds=str(out / "kinds.json")))
    assert verdict["decision"] == "pending_labels"
