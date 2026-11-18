"""Real-model parity checks for score_takes.py (Amendment 6)."""
from __future__ import annotations

import csv
import resource
from pathlib import Path

import numpy as np
import pytest

import edit_arms as ea
import record
import score_takes as st
import weight_edit as we

HERE = Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def runner():
    value = st.ScoreRunner()
    yield value
    value.close()


def stored_test_rows() -> list[dict[str, str]]:
    with (HERE / "vis_manifest.csv").open("r", encoding="utf-8", newline="") as stream:
        manifest = list(csv.DictReader(stream))
    fit_mask, _ = ea.fit_imitation_split(manifest)
    rows = []
    for index, row in enumerate(manifest):
        if row["group"] == "chicken" and row["kind"] == "real":
            label = "real"
        elif (
            row["group"] == "chicken"
            and row["kind"] == "imitation"
            and not fit_mask[index]
        ):
            label = "imitation"
        else:
            continue
        rows.append(
            {
                "path": str(we.VIS / row["local"]),
                "label": label,
                "route": "stored-vocalimitationset-test",
                "note": str(row["id"]),
            }
        )
    assert sum(row["label"] == "real" for row in rows) == 20
    assert sum(row["label"] == "imitation" for row in rows) == 20
    return rows


@pytest.fixture(scope="session")
def heldout_scored(runner):
    rows = stored_test_rows()
    scored = []
    for row in rows:
        # record_vis.py embeds only the first 30 seconds of these stored files.
        audio = record.decode(row["path"])[: 16000 * 30]
        scored.append(runner.score_take(row, audio=audio))
    return rows, scored


def test_unedited_reproduces_two_esc50_items(runner):
    with np.load(HERE / "acts.npz") as archive:
        kind = archive["kind"]
        keys = archive["key"]
        act_mean = archive["act_mean"]
        embeddings = archive["emb"]
    design = __import__("json").loads((HERE / "edit_verdicts.json").read_text())
    sign17 = int(design["signs"]["L17_N1574"])
    sign19 = int(design["signs"]["L19_N1129"])
    choices = ("1-100032-A-0.wav", "1-104089-A-22.wav")
    for key in choices:
        matches = np.flatnonzero((kind == "esc") & (keys == key))
        assert len(matches) == 1
        index = int(matches[0])
        audio = st.decode_take(we.ESC_ROOT / "audio" / key)
        actual, gate = runner.embed(audio, "none")
        assert abs(
            gate["neuron_values"]["L17_N1574"]
            - sign17 * float(np.float16(act_mean[index, 17 * 2048 + 1574]))
        ) <= 2e-3
        assert abs(
            gate["neuron_values"]["L19_N1129"]
            - sign19 * float(np.float16(act_mean[index, 19 * 2048 + 1129]))
        ) <= 2e-3
        assert float(np.dot(actual, embeddings[index])) >= 0.9999


def test_all_heldout_arm_aucs_restore_weights_and_fit_memory_cap(runner, heldout_scored):
    rows, scored = heldout_scored
    labels = np.asarray([row["label"] for row in rows])
    expected = {
        "unedited": 0.5875,
        "output_gate": 0.8525,
        "inside_full": 0.8525,
        "gentle": 0.81,
        "W1": 0.6775,
        "W2": 0.8125,
        "W2_control": 0.6175,
    }
    for version, frozen_auc in expected.items():
        values = np.asarray(
            [take["versions"][version]["score_after"] for take in scored],
            dtype=np.float64,
        )
        actual_auc = ea.auc_values(values[labels == "real"], values[labels == "imitation"])
        assert round(actual_auc, 3) == round(frozen_auc, 3), (version, actual_auc, frozen_auc)

    for take in scored:
        for version in ("W1", "W2", "W2_control"):
            audit = take["versions"][version]["weight_restore"]
            assert audit["bitwise_restored"]
            assert audit["before_sha256"] == audit["after_sha256"]
    assert runner.weight_edits.weight_hash() == runner.weight_edits.original_hash
    peak_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    assert peak_gib < 6.0, f"peak RSS exceeded 6 GiB: {peak_gib:.2f} GiB"


def test_sign_flipped_w2_copy_turns_auc_parity_red(runner, heldout_scored):
    """A disposable sign-flipped W2 delta must fail the stored AUC parity check."""
    rows, _scored = heldout_scored
    labels = np.asarray([row["label"] for row in rows])
    broken_delta_copy = -runner.weight_edits.w2_delta.copy()
    broken_values = []
    for row in rows:
        audio = record.decode(row["path"])[: 16000 * 30]
        with runner.weight_edits.patched("W2", delta_override=broken_delta_copy):
            embedding, _gate = runner.embed(audio, "none")
        broken_values.append(runner.verdict(embedding))
    broken_values = np.asarray(broken_values, dtype=np.float64)
    broken_auc = ea.auc_values(
        broken_values[labels == "real"], broken_values[labels == "imitation"]
    )
    # This is the same parity assertion used above, applied to a disposable
    # sign-flipped copy. Its expected AssertionError is the deliberate red run.
    with pytest.raises(AssertionError):
        assert round(broken_auc, 3) == round(0.8125, 3)
    assert runner.weight_edits.weight_hash() == runner.weight_edits.original_hash
