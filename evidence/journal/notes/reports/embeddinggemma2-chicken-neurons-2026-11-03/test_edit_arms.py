import csv
from pathlib import Path

import numpy as np

from edit_arms import (
    apply_gate_batch,
    apply_written_gate,
    bit_identical_rows,
    fit_imitation_split,
    normalize_vector,
)

HERE = Path(__file__).resolve().parent


def test_closed_gate_returns_the_same_embedding_object():
    embedding = np.asarray([0.25, -0.5, 0.75], dtype=np.float32)
    direction = normalize_vector(np.asarray([1.0, 1.0, 0.0], dtype=np.float32))
    edited = apply_written_gate(
        embedding, s19=0.0, s17=0.0, t19=0.1, t17=0.1, k=2.0, c=direction
    )
    assert edited is embedding
    assert bit_identical_rows(embedding[None, :], edited[None, :]).tolist() == [True]


def test_seeded_manifest_split_reproduces_the_pilot_18_fit_imitation_clips():
    with (HERE / "vis_manifest.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    fit_mask, fit_people = fit_imitation_split(rows)

    imitators = sorted({
        row["participant_id"]
        for row in rows
        if row["group"] == "chicken" and row["kind"] == "imitation"
    })
    rng = np.random.default_rng(20261104)
    expected = set(
        np.asarray(imitators)[
            rng.permutation(len(imitators))[: len(imitators) // 2]
        ].tolist()
    )
    assert set(fit_people) == expected
    assert len(fit_people) == 16
    assert int(fit_mask.sum()) == 18


def test_always_open_gate_mutation_breaks_bit_identical_check():
    embeddings = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
    )
    s19 = np.asarray([0.0, 0.0], dtype=np.float32)
    s17 = np.asarray([0.0, 0.0], dtype=np.float32)
    direction = normalize_vector(np.asarray([1.0, 1.0, 0.0], dtype=np.float32))

    normal = apply_gate_batch(
        embeddings, s19, s17, t19=1.0, t17=1.0, k=0.25, c=direction
    )
    assert bit_identical_rows(embeddings, normal).all()

    def always_open_mutation(e, _s19, _s17, _t19, _t17, k, c):
        return normalize_vector(e - k * c)

    mutated = apply_gate_batch(
        embeddings,
        s19,
        s17,
        t19=1.0,
        t17=1.0,
        k=0.25,
        c=direction,
        gate_fn=always_open_mutation,
    )
    assert not bit_identical_rows(embeddings, mutated).any()
