"""Tests for part108/derive108.py on tiny fake snapshots (safetensors host, pytorch_model.bin donor)."""
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import derive108 as d  # noqa: E402


def snapshots(tmp_path, donor_shapes=None, donor_names=None):
    g = torch.Generator().manual_seed(1)
    shapes = {"w.big": (64, 3), "w.mid": (5, 4), "n.vec": (16,)}
    host = {k: torch.randn(s, generator=g).to(torch.bfloat16) for k, s in shapes.items()}
    donor = {k: (v.float() + 0.05 * torch.randn(v.shape, generator=g)).to(torch.bfloat16) for k, v in host.items()}
    if donor_shapes:
        donor.update({k: torch.zeros(s, dtype=torch.bfloat16) for k, s in donor_shapes.items()})
    if donor_names:
        donor = {donor_names.get(k, k): v for k, v in donor.items()}
    h, dn = tmp_path / "host", tmp_path / "donor"
    h.mkdir(), dn.mkdir()
    save_file(host, str(h / "model.safetensors"))
    torch.save(donor, str(dn / "pytorch_model.bin"))
    (h / "config.json").write_text("{}")
    (h / "tokenizer.json").write_text("{}")
    return h, dn, host, donor


def test_forward_is_twice_host_minus_donor(tmp_path):
    h, dn, host, donor = snapshots(tmp_path)
    info = d.derive(h, dn, "forward", 108, tmp_path / "out")
    out = load_file(str(tmp_path / "out" / "model.safetensors"))
    for k in host:
        expected = (2 * host[k].float() - donor[k].float()).to(torch.bfloat16)
        assert torch.equal(out[k], expected), k
    assert 0.98 < info["distance_ratio"] < 1.02
    assert (tmp_path / "out" / "config.json").exists() and (tmp_path / "out" / "tokenizer.json").exists()
    meta = json.loads((tmp_path / "out" / "derive108.json").read_text())
    assert meta["mode"] == "forward" and meta["seed"] is None and len(meta["sha256"]) == 64


def test_scramble_rows_are_donor_or_forward_and_counts_match(tmp_path):
    h, dn, host, donor = snapshots(tmp_path)
    info = d.derive(h, dn, "scramble", 108, tmp_path / "out")
    out = load_file(str(tmp_path / "out" / "model.safetensors"))
    plus = minus = 0
    for k in host:
        fwd = (2 * host[k].float() - donor[k].float()).to(torch.bfloat16)
        rows_out = out[k].reshape(out[k].shape[0], -1)
        rows_fwd, rows_don = fwd.reshape(fwd.shape[0], -1), donor[k].reshape(donor[k].shape[0], -1)
        kinds = []
        for r in range(rows_out.shape[0]):
            is_fwd, is_don = torch.equal(rows_out[r], rows_fwd[r]), torch.equal(rows_out[r], rows_don[r])
            assert is_fwd or is_don, (k, r)
            kinds.append("fwd" if is_fwd else "don")
        plus += kinds.count("fwd")
        minus += kinds.count("don")
        if k == "w.big":
            assert 10 < kinds.count("fwd") < 54, "64 independent row signs should mix"
    assert (info["rows_plus"], info["rows_minus"]) == (plus, minus)
    again = d.build(h, dn, "scramble", 108)[0]
    assert all(torch.equal(again[k], out[k]) for k in host), "the same seed gives the same model"
    other = d.build(h, dn, "scramble", 109)[0]
    assert not all(torch.equal(other[k], out[k]) for k in host), "another seed gives another model"


def test_name_mismatch_is_refused_before_writing(tmp_path):
    h, dn, _, _ = snapshots(tmp_path, donor_names={"w.mid": "w.other"})
    with pytest.raises(ValueError, match="names differ"):
        d.derive(h, dn, "forward", 108, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_shape_mismatch_is_refused_before_writing(tmp_path):
    h, dn, _, _ = snapshots(tmp_path, donor_shapes={"n.vec": (17,)})
    with pytest.raises(ValueError, match="n.vec"):
        d.derive(h, dn, "scramble", 108, tmp_path / "out")
    assert not (tmp_path / "out").exists()
