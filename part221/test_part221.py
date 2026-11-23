"""Part 221's tests: the real run221/score221 paths on a tiny random OLMo 2, and the scorer on synthetic labels.

The tiny model is Part 219's fixture shape (tests/test_olmo2.py's) with two changes: four decoder layers and sixteen
query/key-value heads of head_dim 1 instead of three layers and two heads of head_dim 8. That is the shape the
registered head list needs -- L0H10, L1H4, L2H6 and L3H5 must exist, R14 to R17 must have a layer each, and the
complement must be the real 60 (15 per layer) -- so `run221.main` walks its whole path with the same head arithmetic
the A770 run will use, on the CPU, with no network and no real weights.

Every property a test is meant to hold is broken on a copy of part221 in the build report (`part221/REPORT.md`), with
the test that went red named there.

Run with: python -I -m pytest part221 -q
"""
from __future__ import annotations

import json
from math import comb
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import Olmo2Config, Olmo2ForCausalLM, PreTrainedTokenizerFast

from part92 import common92 as c
from part99 import swap99
from part221 import run221, score221

D = 16
N_LAYERS = 4
N_HEADS = 16
HEAD_DIM = 1
WORDS = ["<pad>", "<bos>", "<eos>", "Who", "are", "you", "?", "Tell", "me", "about", "yourself",
         "The", "cat", "sat", "on", "mat", "say", "hello", "world", "I", "am", "a", "person", "."]
ALL_HEADS = [(layer, head) for layer in run221.HEAD_LAYERS for head in range(N_HEADS)]


# --- the tiny snapshots -----------------------------------------------------------------------------------------

def build_snapshot(directory: Path, seed: int) -> Path:
    """A tiny OLMo 2 causal LM plus a word-level tokenizer and a chat template, saved for `load_model`.

    Copied from `part219/test_part219.py`'s fixture with four layers and sixteen heads of head_dim 1, so the registered
    kept heads exist and the complement is the real 60.
    """
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


def load_pair(snapshots):
    """Both models in memory, as run221 holds them."""
    host, donor = snapshots
    tokenizer, model = c.load_model(str(host), "cpu", attn_implementation="eager")
    return tokenizer, model, swap99._load_donor(str(donor), attn_implementation="eager")


# --- 1. the complement -------------------------------------------------------------------------------------------

def test_the_complement_is_the_registered_size_and_never_holds_a_kept_head(snapshots):
    _tokenizer, model, _donor = load_pair(snapshots)
    lists = run221.head_lists(model)
    assert lists["n_complement"] == 60
    assert lists["per_layer"] == {0: 15, 1: 15, 2: 15, 3: 15}
    kept = set(run221.kept_heads())
    assert kept & set(lists["complement"]) == set()
    assert len(lists["all_heads"]) == 64
    assert set(lists["all_heads"]) - set(lists["complement"]) == kept
    for layer in run221.HEAD_LAYERS:
        assert len(run221.complement_for_layer(lists, layer)) == 15


def test_the_head_lists_use_the_model_s_own_geometry(snapshots):
    """The complement is computed, not hard-coded: a model with two heads per layer gets that model's complement, and a
    kept head the model does not have is refused instead of silently dropped."""
    _tokenizer, model, _donor = load_pair(snapshots)
    config = Olmo2Config(vocab_size=len(WORDS), hidden_size=D, intermediate_size=32, num_hidden_layers=N_LAYERS,
                         num_attention_heads=2, num_key_value_heads=2, head_dim=8, attn_implementation="eager")
    small = Olmo2ForCausalLM(config).eval()
    sm = run221.head_lists(small, kept=("L0H0", "L1H0", "L2H0", "L3H0"))
    assert sm["n_complement"] == 4  # 2 heads x 4 layers - 4 kept, not the registered 60
    assert list(sm["per_layer"].values()) == [1, 1, 1, 1]
    with pytest.raises(SystemExit):
        run221.head_lists(model, kept=("L0H16", "L1H4", "L2H6", "L3H5"))  # head 16 does not exist
    with pytest.raises(SystemExit):
        run221.head_lists(model, kept=("L4H0", "L1H4", "L2H6", "L3H5"))  # layer 4 does not exist


# --- 2. R13's weight assertion -----------------------------------------------------------------------------------

def test_the_r13_weight_assertion_passes_when_every_slice_is_written(snapshots):
    _tokenizer, model, donor = load_pair(snapshots)
    result = swap99.transplant_heads(model, donor, ALL_HEADS, "donor")
    try:
        check = run221.self_attn_equals_donor(model, donor, run221.HEAD_LAYERS)
        assert check["equal"] is True
        assert check["mismatches"] == []
        assert check["parameters_compared"] == 24  # four layers x six self_attn parameters
        assert all(record["matches_expected"] for record in result["records"])
    finally:
        swap99.restore_transplant(result["restore"])


def test_the_r13_weight_assertion_fails_when_one_slice_is_not_transplanted(snapshots):
    _tokenizer, model, donor = load_pair(snapshots)
    missing = (2, 7)
    result = swap99.transplant_heads(model, donor, [pair for pair in ALL_HEADS if pair != missing], "donor")
    try:
        check = run221.self_attn_equals_donor(model, donor, run221.HEAD_LAYERS)
        assert check["equal"] is False
        assert "layers.2.self_attn.q_proj.weight" in check["mismatches"]
        assert len(check["mismatches"]) == 6  # the six parts of the one head left as Instruct's
    finally:
        swap99.restore_transplant(result["restore"])


def test_the_r13_assertion_is_made_before_any_row_is_generated(snapshots):
    """The wrapper in run221 checks the written weights and raises before swap99 reaches generation, and the host is
    restored, so a failed assertion cannot poison the runs after it."""
    _tokenizer, model, donor = load_pair(snapshots)
    before = run221.norm_snapshot(model, run221.HEAD_LAYERS)
    watch = run221.TransplantWatch(before, check_donor=donor, check_layers=run221.HEAD_LAYERS)
    original = swap99.transplant_heads
    with watch:
        assert swap99.transplant_heads is not original  # the one-run wrapper is installed
        with pytest.raises(run221.SelfAttnMismatch):
            # One head left as Instruct's, so the assertion fails after the write and before any generation.
            swap99.transplant_heads(model, donor, [pair for pair in ALL_HEADS if pair != (1, 3)], "donor")
    assert swap99.transplant_heads is original  # and it is put back
    assert watch.check is not None and watch.check["equal"] is False
    assert "layers.1.self_attn.q_proj.weight" in watch.check["mismatches"]
    # The wrapper restored the host even though swap99's own restore never ran (the transplant result was lost).
    after = run221.norm_snapshot(model, run221.HEAD_LAYERS)
    for layer in run221.HEAD_LAYERS:
        for part in ("q_norm", "k_norm"):
            assert torch.equal(before[layer][part], after[layer][part])


# --- 3. the sign-flip receipt -----------------------------------------------------------------------------------

def test_the_sign_flip_count_is_right_on_a_hand_built_case(snapshots):
    """One known element flips under opposite mode, one becomes exactly zero (its own sign, so it flips), and a zero
    that stays zero does not."""
    _tokenizer, model, donor = load_pair(snapshots)
    with torch.no_grad():
        # Start from identical norms everywhere, so only the hand-built elements can flip.
        for layer in run221.HEAD_LAYERS:
            for part in ("q_norm", "k_norm"):
                getattr(model.model.layers[layer].self_attn, part).weight.copy_(
                    getattr(donor.model.layers[layer].self_attn, part).weight)
        # (0, 2): 2 * 0.25 - 1.0 = -0.5, a positive element becomes negative.
        model.model.layers[0].self_attn.q_norm.weight[2] = 0.25
        donor.model.layers[0].self_attn.q_norm.weight[2] = 1.0
        # (1, 3): 2 * 0.5 - 1.0 = 0.0 exactly, + becomes zero (zero is its own sign).
        model.model.layers[1].self_attn.k_norm.weight[3] = 0.5
        donor.model.layers[1].self_attn.k_norm.weight[3] = 1.0
        # (2, 5): 0 stays 0, no flip.
        model.model.layers[2].self_attn.q_norm.weight[5] = 0.0
        donor.model.layers[2].self_attn.q_norm.weight[5] = 0.0
    before = run221.norm_snapshot(model, run221.HEAD_LAYERS)
    heads = [(0, 2), (1, 3), (2, 5)]
    result = swap99.transplant_heads(model, donor, heads, "opposite")
    try:
        receipt = run221.sign_flip_receipt(model, before, heads)
        per_layer = {entry["layer"]: entry for entry in receipt["per_layer"]}
        assert per_layer[0]["q_norm"]["flips"] == 1
        assert per_layer[0]["k_norm"]["flips"] == 0
        assert per_layer[1]["q_norm"]["flips"] == 0
        assert per_layer[1]["k_norm"]["flips"] == 1
        assert per_layer[2]["q_norm"]["flips"] == 0  # zero to zero is not a flip
        assert receipt["total_flips"] == 2
        assert receipt["total_elements"] == 6  # three heads x two norms x head_dim 1
        assert "zero counts as its own sign" in receipt["rule"]
        ranges = run221.norm_ratio_range(result["records"])
        assert ranges["count"] == 6 * len(heads) - 1  # the zeroed q_norm slice has no ratio (its original norm is 0)
        assert ranges["records_without_a_ratio"] == 1
        assert ranges["min"] is not None and ranges["max"] is not None
    finally:
        swap99.restore_transplant(result["restore"])


def test_a_donor_mode_transplant_of_positive_norms_flips_no_sign(snapshots):
    _tokenizer, model, donor = load_pair(snapshots)
    before = run221.norm_snapshot(model, run221.HEAD_LAYERS)
    result = swap99.transplant_heads(model, donor, ALL_HEADS, "donor")
    try:
        receipt = run221.sign_flip_receipt(model, before, ALL_HEADS)
        assert receipt["total_flips"] == 0  # the fixture's norms are all positive on both sides
        assert receipt["total_elements"] == 2 * N_LAYERS * N_HEADS
    finally:
        swap99.restore_transplant(result["restore"])


# --- 4. the orchestrator's real main() path on the tiny model ----------------------------------------------------

def make_kernel_record(directory: Path) -> Path:
    """A synthetic kernel record that passes tools/part_check.kernel_check against itself (as Part 219's fixture)."""
    directory.mkdir(parents=True, exist_ok=True)
    record = {"model": "tiny", "revision": "fixture", "prompt_sha256": "0" * 64, "new_tokens": 4,
              "argmax_agrees": True, "attn_implementation": "eager", "load": {"attn_implementation": "eager"},
              "runs": {"8": [{"prompt": 0, "tokens": [1, 2, 3, 4], "margins": [1.0, 1.0, 1.0, 1.0]}]}}
    reference = directory / "kernel-reference.json"
    reference.write_text(json.dumps(record), encoding="utf-8")
    path = directory / "kernel-check.json"
    path.write_text(json.dumps(dict(record, reference=str(reference))), encoding="utf-8")
    return path


def make_workspace(tmp_path: Path, snapshots) -> dict:
    host, donor = snapshots
    root = tmp_path / "workspace"
    root.mkdir(parents=True, exist_ok=True)
    neurons = [{"id": "t-0", "layer": 1, "neuron": 3, "control_seed": 11},
               {"id": "t-1", "layer": 2, "neuron": 5, "control_seed": 12}]
    (root / "neurons.json").write_text(json.dumps(neurons), encoding="utf-8")
    (root / "kinds.json").write_text(json.dumps({"kinds": {"t-0": "slot", "t-1": "concept"}}),
                                     encoding="utf-8")
    fixed = {}
    for neuron in neurons:
        for unit in ("neuron", "random"):
            generator = torch.Generator().manual_seed(int(neuron["control_seed"]) + (unit == "random"))
            fixed[f"{neuron['id']}|{unit}"] = torch.randn(D, generator=generator) * 0.1
    torch.save(fixed, root / "pushes.pt")
    repro = root / "repro"
    repro.mkdir()
    return {"root": root, "host": host, "donor": donor, "out": root / "out", "repro": repro,
            "kernel": make_kernel_record(root / "kernel")}


def run_args(workspace) -> list[str]:
    return ["--host", str(workspace["host"]), "--donor", str(workspace["donor"]),
            "--neurons", str(workspace["root"] / "neurons.json"),
            "--kinds", str(workspace["root"] / "kinds.json"),
            "--pushes", str(workspace["root"] / "pushes.pt"),
            "--out", str(workspace["out"]), "--device", "cpu", "--batch", "8",
            "--kernel-check", str(workspace["kernel"]), "--repro-dir", str(workspace["repro"]),
            "--allow-version-mismatch"]


def write_labels_from_rows(directory: Path, tag: str, jsonl: Path) -> None:
    """Two passes of labels for every row a run wrote (all coherent, nothing showing), enough for the scorer to read."""
    rows = [json.loads(line) for line in Path(jsonl).open(encoding="utf-8")]
    for name in (f"labels-{tag}.jsonl", f"labels2-{tag}.jsonl"):
        with (Path(directory) / name).open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps({"id": row["id"], "unit": row["unit"],
                                         "question_index": row["question_index"],
                                         "shows_persona": False, "coherent": True}) + "\n")


def test_the_orchestrator_runs_end_to_end_on_the_tiny_model(tmp_path, snapshots):
    """`main()`'s real path: a pilot that stops while the reproduction rows are missing, a pilot that continues, then
    the full nine-run table, every file the scorer reads, and the scorer over it."""
    workspace = make_workspace(tmp_path, snapshots)
    out = workspace["out"]
    args = run_args(workspace)

    # The pilot first, with no Part 219 rows to reproduce: it must stop, with a clear verdict and a non-zero exit.
    assert run221.main(args + ["--pilot"]) == 2
    stopped = json.loads((out / "PILOT.json").read_text(encoding="utf-8"))
    assert stopped["decision"] == "stop"
    assert any("r0_vs_part219" in name for name in stopped["failures"])
    assert (out / "R0.jsonl").exists() and (out / "R1.jsonl").exists() and (out / "R13.jsonl").exists()

    # Seed the reproduction directory with the pilot's own rows (the role Part 219's results play on the A770) and
    # rerun: this time the pilot continues, so the full run may launch.
    for tag in ("R0", "R1"):
        (workspace["repro"] / f"{tag}.jsonl").write_bytes((out / f"{tag}.jsonl").read_bytes())
    assert run221.main(args + ["--pilot"]) == 0
    continued = json.loads((out / "PILOT.json").read_text(encoding="utf-8"))
    assert continued["decision"] == "continue", continued["because"]
    assert continued["checks"]["r13_equals_r1"]["passes"] is True
    assert continued["checks"]["r13_weight_assertion"]["passes"] is True
    assert continued["checks"]["sign_flip_receipts"]["passes"] is True

    assert run221.main(args) == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert sorted(manifest["runs"]) == ["R0", "R1", "R11", "R12", "R13", "R14", "R15", "R16", "R17"]
    assert all(entry["readable"] for entry in manifest["runs"].values()), manifest["runs"]
    assert manifest["checks"]["reproduction"]["r0_vs_part219"]["identical"] is True
    assert manifest["checks"]["reproduction"]["r1_vs_part219"]["identical"] is True
    assert manifest["checks"]["reproduction"]["r13_vs_r1"]["identical"] is True
    assert manifest["checks"]["head_lists"]["complement_is_60_on_the_real_model"] is True
    assert manifest["complement_heads"] == 60
    for tag in manifest["runs"]:
        assert (out / f"{tag}.jsonl").exists() and (out / f"{tag}.meta.json").exists()
    r13 = json.loads((out / "R13.meta.json").read_text(encoding="utf-8"))
    assert r13["intervention_receipt"]["transplant"]["self_attn_equals_donor"]["equal"] is True
    for tag in ("R11", "R12"):
        meta = json.loads((out / f"{tag}.meta.json").read_text(encoding="utf-8"))
        transplant = meta["intervention_receipt"]["transplant"]
        assert transplant["sign_flip_receipt"]["total_elements"] == 2 * 4 * 15
        assert transplant["norm_ratio_range"]["count"] == 6 * 60
        drift = run221.kept_head_drift(meta)
        assert drift["kept_heads"] == list(run221.KEPT_HEADS)
        assert len(drift["per_layer"]) == 4
        assert all(len(entry["kept_head_drift"]) == 1 for entry in drift["per_layer"])
    r12 = json.loads((out / "R12.meta.json").read_text(encoding="utf-8"))
    assert r12["intervention_receipt"]["transplant"]["sign_flip_receipt"]["total_flips"] > 0  # the opposite move

    # The scorer reads the runs: labels for R0 and R1 come from the reproduction directory, all others from --out.
    for tag in sorted(manifest["runs"]):
        target = workspace["repro"] if tag in ("R0", "R1") else out
        write_labels_from_rows(target, tag, out / f"{tag}.jsonl")
    assert score221.main(["--out", str(out), "--kinds", str(workspace["root"] / "kinds.json"),
                          "--neurons", str(workspace["root"] / "neurons.json"),
                          "--kernel-check", str(workspace["kernel"]), "--part219-dir", str(workspace["repro"])]) == 0
    assert (out / "scorecard.json").exists() and (out / "SCORECARD.md").exists()
    scorecard = json.loads((out / "scorecard.json").read_text(encoding="utf-8"))
    # Two slot units on the fixture cannot meet a bar written for 21, so the gate fails and everything is frozen --
    # which is exactly what the design says happens when the gate fails.
    assert scorecard["gate"]["passed"] is False
    assert all(value["verdict"].startswith("not read: gate failed")
               for value in scorecard["predictions"].values())
    assert scorecard["headlines"][0].startswith("no headline: the gate failed")
    assert "every prediction is frozen" in "\n".join(
        (out / "SCORECARD.md").read_text(encoding="utf-8").splitlines()[:4])


# --- 5. the scorer on synthetic labels ---------------------------------------------------------------------------

SLOT = [f"s{index:02d}" for index in range(1, 22)]
CONCEPT = [f"c{index:02d}" for index in range(1, 19)]
SLOT_LAYERS = dict(zip(SLOT, [1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 5, 5, 6, 9, 12, 3, 4, 4, 5, 3, 3]))
CONCEPT_LAYERS = {unit: 3 + index % 8 for index, unit in enumerate(CONCEPT)}
UPSTREAM = [unit for unit in SLOT if SLOT_LAYERS[unit] >= 3]


def write_world(directory: Path, show, coherent=None, pass2=None, manifest_readable=None, reproduction=None):
    """A synthetic results directory for Part 221's tags: runs, two label passes, the kinds/neurons files, a manifest
    carrying the reproduction check, and metas for the coupling numbers."""
    directory.mkdir(parents=True, exist_ok=True)
    coherent = coherent or {}
    kinds = {unit: "slot" for unit in SLOT} | {unit: "concept" for unit in CONCEPT}
    (directory / "kinds.json").write_text(json.dumps({"kinds": kinds}), encoding="utf-8")
    neurons = ([{"id": unit, "layer": SLOT_LAYERS[unit], "neuron": index} for index, unit in enumerate(SLOT)]
               + [{"id": unit, "layer": CONCEPT_LAYERS[unit], "neuron": index}
                  for index, unit in enumerate(CONCEPT)])
    (directory / "neurons.json").write_text(json.dumps(neurons), encoding="utf-8")
    tags = [tag for tag in score221.RUN_TAGS if tag in show]
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
        if tag in score221.TRANSPLANT_RUNS:
            layers = [{"layer": layer,
                       "changed_heads": [head for head in range(16) if head != (10, 4, 6, 5)[layer]],
                       "rms_q": {"before": 1.0, "after": 1.0, "relative_change": 0.01 * (layer + 1)},
                       "rms_k": {"before": 1.0, "after": 1.0, "relative_change": -0.02},
                       "rms_attention_output": {"before": 1.0, "after": 1.0,
                                                "relative_change": -0.05 * (layer + 1)},
                       "sibling_output_drift": {
                           "heads": [{"head": (10, 4, 6, 5)[layer], "relative_change": 0.01 * (layer + 1)}],
                           "max_relative_change": 0.01 * (layer + 1)},
                       "changed_head_output_change": []}
                      for layer in range(4)]
            meta = {"coupling_log": {"batch": {}, "layers": layers},
                    "intervention_receipt": {"transplant": {
                        "sign_flip_receipt": {"rule": "x", "per_layer": [], "total_flips": 3,
                                              "total_elements": 100},
                        "norm_ratio_range": {"count": 10, "min": 0.2, "max": 3.0,
                                             "records_without_a_ratio": 0}}}}
            (directory / f"{tag}.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    manifest = {"part": 221,
                "runs": {tag: {"readable": (manifest_readable or {}).get(tag, True), "problems": []} for tag in tags},
                "checks": {"reproduction": reproduction if reproduction is not None else {
                    "r0_vs_part219": {"identical": True}, "r1_vs_part219": {"identical": True},
                    "r13_vs_r1": {"identical": True}}}}
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory


def baseline_show() -> dict:
    """A world that satisfies the gate and all three predictions, to be broken one piece at a time.

    R0 shows 19 slot units (14 of the 16 upstream), R1 2 (none upstream, net 14), R11 12 upstream lost against 2 gained
    (net 12, 221.1 and 221.2 hold), and R12 shows 12 of the units R11 lost and none of the 2 R11 gained, so the paired
    McNemar is 0 against 12 (p = 0.000488, 221.3 holds).
    """
    return {
        "R0": set(SLOT[:19]) | set(CONCEPT[:16]),
        "R1": set(SLOT[:2]) | set(CONCEPT[:15]),
        "R13": set(SLOT[:2]) | set(CONCEPT[:15]),
        "R11": set(SLOT[:2]) | set(SLOT[19:]) | set(CONCEPT[:14]),
        "R12": set(SLOT[5:17]) | set(CONCEPT[:14]),
        "R14": set(SLOT[:17]) | set(CONCEPT[:15]),
        "R15": set(SLOT[:18]) | set(CONCEPT[:15]),
        "R16": set(SLOT[:18]) | set(CONCEPT[:15]),
        "R17": set(SLOT[:18]) | set(CONCEPT[:15]),
    }


def build(tmp_path: Path, show, **kwargs) -> dict:
    out = write_world(tmp_path / "results", show, **kwargs)
    kernel = tmp_path / "kernel"
    kernel.mkdir(parents=True, exist_ok=True)
    record = {"model": "tiny", "revision": "fixture", "prompt_sha256": "0" * 64, "new_tokens": 4,
              "argmax_agrees": True, "attn_implementation": "eager", "load": {"attn_implementation": "eager"},
              "runs": {"8": [{"prompt": 0, "tokens": [1, 2, 3, 4], "margins": [1.0, 1.0, 1.0, 1.0]}]}}
    reference = kernel / "reference.json"
    reference.write_text(json.dumps(record), encoding="utf-8")
    (kernel / "kernel-check.json").write_text(json.dumps(dict(record, reference=str(reference))), encoding="utf-8")
    return score221.build_scorecard(out, out / "kinds.json", out / "neurons.json", kernel / "kernel-check.json", out)


def verdicts(card: dict) -> dict:
    return {key: value["verdict"] for key, value in card["predictions"].items()}


def test_the_split_is_computed_from_the_neuron_list(tmp_path):
    card = build(tmp_path, baseline_show())
    assert card["split"]["n_upstream"] == 16
    assert card["split"]["n_downstream"] == 5
    assert set(card["split"]["upstream"]) == set(UPSTREAM)
    assert card["gate"]["passed"] is True


def test_the_registered_predictions_hold_in_a_world_built_so_they_do(tmp_path):
    card = build(tmp_path, baseline_show())
    verdicts_ = verdicts(card)
    assert verdicts_["221.1"] == "held"
    assert verdicts_["221.2"] == "held"
    assert verdicts_["221.3"] == "held"
    assert card["nets_upstream_against_R0"]["R11"]["net"] == 12
    assert card["nets_upstream_against_R0"]["R1"]["net"] == 14
    detail = card["predictions"]["221.3"]["detail"]
    assert detail["only_second"] == 12 and detail["only_first"] == 2
    assert detail["mcnemar_p"] == pytest.approx(2 * sum(comb(14, k) for k in (0, 1, 2)) / 2 ** 14)
    assert "removes nearly as much as the whole band" in card["headlines"][0]
    assert "post-training's direction matters" in card["headlines"][0]
    assert card["exploratory"]["R11_R12_overlap"]["counts"] == {
        "lost_both": 2, "lost_only_R11": 12, "lost_only_R12": 0,
        "gained_both": 0, "gained_only_R11": 2, "gained_only_R12": 0}


def test_221_1_fails_and_221_3_is_not_read_when_r11_loses_too_few(tmp_path):
    show = baseline_show()
    show["R11"] = set(SLOT[:2]) | set(SLOT[5:11]) | set(SLOT[19:]) | set(CONCEPT[:14])  # 8 lost, 2 gained, net 6
    card = build(tmp_path, show)
    assert card["nets_upstream_against_R0"]["R11"]["net"] < 7
    assert verdicts(card)["221.1"] == "failed"
    assert verdicts(card)["221.3"].startswith("not read: 221.1 did not hold")
    assert "neither transplanting the four heads alone" in card["headlines"][0]


def test_221_2_fails_when_r11_does_not_beat_the_additive_share(tmp_path):
    show = baseline_show()
    show["R11"] = set(SLOT[:2]) | set(SLOT[18:]) | set(CONCEPT[:14])  # upstream net 11 (13 lost, 2 gained)
    card = build(tmp_path, show)
    assert card["nets_upstream_against_R0"]["R1"]["net"] == 14
    assert card["nets_upstream_against_R0"]["R11"]["net"] == 11
    assert verdicts(card)["221.1"] == "held"
    assert verdicts(card)["221.2"] == "failed"  # 11 is not at least 14 - 2
    assert "the whole band more" in card["headlines"][0]


def test_r12_below_the_coherence_bar_makes_221_3_inconclusive(tmp_path):
    broken = {(("R12"), unit, 0): False for unit in SLOT[:7]}
    card = build(tmp_path, baseline_show(), coherent=broken)
    assert card["runs"]["R12"]["slot_who_coherent"][0] < score221.BAR_SLOT_COHERENT
    assert verdicts(card)["221.1"] == "held"
    assert verdicts(card)["221.3"] == "inconclusive: the opposite-direction transplant breaks answers"


def test_r11_below_the_coherence_bar_makes_221_1_and_221_2_inconclusive(tmp_path):
    broken = {("R11", unit, 0): False for unit in SLOT[:7]}
    card = build(tmp_path, baseline_show(), coherent=broken)
    assert verdicts(card)["221.1"] == "inconclusive: the complement transplant breaks answers"
    assert verdicts(card)["221.2"] == "inconclusive: the complement transplant breaks answers"
    assert verdicts(card)["221.3"].startswith("not read: 221.1 is inconclusive")


def test_a_reader_gate_failure_makes_a_prediction_unreadable_not_inconclusive(tmp_path):
    """R11's two passes disagree on most slot items: the run is not read at all, and 'unreadable' is not
    'inconclusive'."""
    show = baseline_show()
    disagreeing = {unit for index, unit in enumerate(SLOT) if index % 2}
    card = build(tmp_path, show, pass2={"R11": disagreeing})
    assert card["runs"]["R11"]["two_pass"]["passes"] is False
    assert verdicts(card)["221.1"].startswith("unreadable: R11 is not read")
    assert verdicts(card)["221.2"].startswith("unreadable: R11 is not read")
    assert "inconclusive" not in verdicts(card)["221.1"]


def test_the_paired_mcnemar_is_the_hand_counted_discordant_pairs(tmp_path):
    """221.3's test on a hand-built pair: R11 and R12 agree on every upstream unit except a known discordant set."""
    show = baseline_show()
    # Every upstream unit R0 shows shows in R12; R11 shows only s20 and s21. Discordant: 2 in R11 only, 14 in R12 only.
    show["R12"] = set(SLOT[:19]) | set(CONCEPT[:14])
    show["R11"] = set(SLOT[19:]) | set(CONCEPT[:14])
    card = build(tmp_path, show)
    detail = card["predictions"]["221.3"]["detail"]
    assert (detail["only_first"], detail["only_second"]) == (2, 14)
    expected_p = 2 * sum(comb(16, k) for k in range(0, 3)) / 2 ** 16
    assert detail["mcnemar_p"] == pytest.approx(expected_p)
    assert expected_p < 0.05
    assert verdicts(card)["221.3"] == "held"

    # The other direction: the discordant count is significant but R11 shows more than R12, so "R11 losing more" is
    # false and 221.3 fails on its test while R12 is coherent.
    show["R11"] = set(SLOT[:2]) | set(SLOT[5:11]) | set(CONCEPT[:14])  # upstream shows s06..s11, net 8
    show["R12"] = set(SLOT[:2]) | set(CONCEPT[:14])                    # nothing upstream
    card = build(tmp_path, show)
    detail = card["predictions"]["221.3"]["detail"]
    assert (detail["only_first"], detail["only_second"]) == (6, 0)
    assert detail["mcnemar_p"] == pytest.approx(2 / 2 ** 6)
    assert detail["mcnemar_p"] < 0.05
    assert verdicts(card)["221.3"] == "failed"
    assert "removed R11's count within the test's resolution" in card["headlines"][0]


def test_a_failed_gate_freezes_every_prediction(tmp_path):
    card = build(tmp_path, baseline_show(),
                 reproduction={"r0_vs_part219": {"identical": False, "token_mismatches": 3},
                               "r1_vs_part219": {"identical": True}, "r13_vs_r1": {"identical": True}})
    assert card["gate"]["passed"] is False
    assert "r0_reproduction" in card["gate"]["failures"]
    for key in ("221.1", "221.2", "221.3"):
        assert card["predictions"][key]["verdict"].startswith("not read: gate failed")
        assert card["predictions"][key]["frozen"] is True
    markdown = score221.markdown(card)
    assert "every prediction is frozen" in markdown.splitlines()[2]


def test_the_pilot_stops_when_r13_differs_from_r1(tmp_path):
    """Added in review (Journal Claude): the pilot's own stop on the identity check, on an otherwise clean manifest."""
    entry = {"readable": True, "problems": [], "reset_receipt": {"role": "reset", "matches_reference": True}}
    manifest = {"runs": {"R0": dict(entry), "R1": dict(entry),
                         "R13": dict(entry, self_attn_equals_donor={"equal": True}, sign_flip_receipt={"q_norm": 0})}}
    clean = {"r0_vs_part219": {"identical": True}, "r1_vs_part219": {"identical": True}, "r13_vs_r1": {"identical": True}}
    assert run221.pilot_verdict(tmp_path, manifest, clean)["decision"] == "continue"
    bad = dict(clean, r13_vs_r1={"identical": False, "token_mismatches": 2})
    verdict = run221.pilot_verdict(tmp_path, manifest, bad)
    assert verdict["decision"] == "stop"
    assert verdict["failures"] == ["r13_equals_r1"]


def test_r13_differing_from_r1_fails_the_gate_and_freezes_every_prediction(tmp_path):
    """Added in review (Journal Claude): forcing the scorer's r13_equals_r1 check to pass left every test green."""
    card = build(tmp_path, baseline_show(),
                 reproduction={"r0_vs_part219": {"identical": True}, "r1_vs_part219": {"identical": True},
                               "r13_vs_r1": {"identical": False, "token_mismatches": 1}})
    assert card["gate"]["passed"] is False
    assert card["gate"]["failures"] == ["r13_equals_r1"]
    for key in ("221.1", "221.2", "221.3"):
        assert card["predictions"][key]["frozen"] is True


def test_a_missing_reproduction_check_freezes_every_prediction(tmp_path):
    card = build(tmp_path, baseline_show(), reproduction={})
    assert card["gate"]["passed"] is False
    assert "r0_reproduction" in card["gate"]["failures"]


def test_a_run_the_manifest_marks_unreadable_fails_the_gate_it_feeds(tmp_path):
    card = build(tmp_path, baseline_show(), manifest_readable={"R0": False})
    assert card["gate"]["passed"] is False
    assert "r0_slot_persona" in card["gate"]["failures"]
    assert verdicts(card)["221.1"].startswith("not read: gate failed")


def test_the_coupling_numbers_reach_the_headline(tmp_path):
    card = build(tmp_path, baseline_show())
    assert card["coupling"]["R11"]["max_kept_head_drift"] == pytest.approx(0.04)
    assert card["coupling"]["R11"]["max_abs_rms_attention_output_change"] == pytest.approx(0.20)
    assert card["coupling"]["R12"]["sign_flip_receipt"]["total_flips"] == 3
    assert card["coupling"]["R12"]["norm_ratio_range"]["max"] == 3.0
    assert "largest kept-head pre-norm drift 4.00%" in card["headlines"][0]
    assert "largest |change in rms(attention output)| 20.00%" in card["headlines"][0]


def test_only_the_16_upstream_units_feed_the_predictions(tmp_path):
    """A run that loses all five downstream units and nothing upstream has 221.1 failed, not held."""
    show = baseline_show()
    show["R11"] = set(SLOT[5:19]) | set(CONCEPT[:14])  # every upstream unit R0 shows, none downstream
    card = build(tmp_path, show)
    assert card["nets_upstream_against_R0"]["R11"]["net"] == 0
    assert card["exploratory"]["nets_against_R0"]["R11"]["downstream"]["lost"] == 5
    assert verdicts(card)["221.1"] == "failed"
