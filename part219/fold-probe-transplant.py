"""Independent check of transplant orientation + the intervention receipt's blindness to a wrong slice.

Orientation (design): q/k/v ROWS [hs:hs+head_size] are outputs, o_proj COLUMNS are inputs, norms are 1-D slices.
Then: monkeypatch swap99.head_span to shift every span by one head; the write lands in the wrong slice but
the run's own receipt still says matches_expected=True.
"""
import sys
from pathlib import Path

import torch

WORKTREE = str(Path(__file__).resolve().parents[1])  # the worktree this copy lives in
sys.path.insert(0, WORKTREE)
sys.path.insert(0, WORKTREE + "/part92")
sys.path.insert(0, WORKTREE + "/part99")

import swap99  # noqa: E402

ROOT = "/home/elisepc/green/Journal/scratch/review-part219-code-r1-ds-r1-2026-11-22/exp/world"
HEAD_SIZE = 8
PARTS = ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")

host = swap99.load_donor(ROOT + "/host")
donor = swap99.load_donor(ROOT + "/donor")
donor_params = dict(donor.named_parameters())
original = {name: parameter.detach().clone() for name, parameter in host.named_parameters()}
heads = [(0, 1), (2, 0)]
result = swap99.transplant_heads(host, donor, heads, "donor")

bad = []
for layer, head in heads:
    start, end = head * HEAD_SIZE, (head + 1) * HEAD_SIZE
    for part in PARTS:
        name = f"model.layers.{layer}.self_attn.{part}.weight"
        got = dict(host.named_parameters())[name]
        want = donor_params[name]
        if got.ndim == 1:
            ok = torch.equal(got[start:end], want[start:end])
            complement = torch.equal(torch.cat([got[:start], got[end:]]),
                                      torch.cat([original[name][:start], original[name][end:]]))
            wrong_axis = torch.equal(got[start:end], original[name][start:end])
        elif part in ("q_proj", "k_proj", "v_proj"):
            ok = torch.equal(got[start:end, :], want[start:end, :])
            complement = torch.equal(torch.cat([got[:start], got[end:]]),
                                      torch.cat([original[name][:start], original[name][end:]]))
            wrong_axis = torch.equal(got[:, start:end], want[:, start:end])
        else:
            ok = torch.equal(got[:, start:end], want[:, start:end])
            complement = torch.equal(torch.cat([got[:, :start], got[:, end:]], dim=1),
                                      torch.cat([original[name][:, :start], original[name][:, end:]], dim=1))
            wrong_axis = torch.equal(got[start:end, :], want[start:end, :])
        if not (ok and complement and not wrong_axis):
            bad.append((name, ok, complement, wrong_axis))
print("orientation independent check failures:", bad)
swap99.restore_transplant(result["restore"])
print("restore bit-exact:", all(torch.equal(parameter, original[name])
                                for name, parameter in host.named_parameters()))

# --- receipt blindness: shift every span one head to the right --------------------------------
real_head_span = swap99.head_span
swap99.head_span = lambda model, layer_no, head: real_head_span(model, layer_no, head + 1)
host = swap99.load_donor(ROOT + "/host")
donor = swap99.load_donor(ROOT + "/donor")
before = {name: parameter.detach().clone() for name, parameter in host.named_parameters()}
shifted = swap99.transplant_heads(host, donor, [(0, 0)], "donor")
swap99.head_span = real_head_span
records = shifted["records"]
q = dict(host.named_parameters())["model.layers.0.self_attn.q_proj.weight"]
want_rows_0_8_unchanged = torch.equal(q[0:8], before["model.layers.0.self_attn.q_proj.weight"][0:8])
wrong_rows_changed = not torch.equal(q[8:16], before["model.layers.0.self_attn.q_proj.weight"][8:16])
print("shifted slice wrote rows 8:16 instead of 0:8:", wrong_rows_changed and want_rows_0_8_unchanged)
print("receipt for the shifted write: head", records[0]["head"], "slice", records[0]["slice"],
      "matches_expected", records[0]["matches_expected"],
      "-> all records pass:", all(record["matches_expected"] for record in records))
