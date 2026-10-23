from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

import atlas_collect_fast2 as atlas
from part64 import collect64, collect64_model


ROOT = Path(__file__).resolve().parents[1]
HEAD200 = ROOT / "data" / "pile10k-head200.jsonl"


class _HookHandle:
    def __init__(self, hooks: list):
        self.hooks = hooks
        self.callback = None

    def remove(self):
        if self.callback in self.hooks:
            self.hooks.remove(self.callback)


class _FakeDownProj:
    def __init__(self, width: int):
        self.in_features = width
        self.pre_hooks: list = []
        self.post_hooks: list = []

    def register_forward_pre_hook(self, callback):
        handle = _HookHandle(self.pre_hooks)
        handle.callback = callback
        self.pre_hooks.append(callback)
        return handle

    def register_forward_hook(self, callback):
        handle = _HookHandle(self.post_hooks)
        handle.callback = callback
        self.post_hooks.append(callback)
        return handle

    def __call__(self, values):
        args = (values,)
        for callback in tuple(self.pre_hooks):
            callback(self, args)
        output = values
        for callback in tuple(self.post_hooks):
            callback(self, args, output)
        return output


class _FakeTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"

    def __init__(self, rows: list[dict]):
        self.doc_for_text = {row["text"]: index for index, row in enumerate(rows)}

    def __call__(self, texts, **_kwargs):
        input_ids = []
        for text in texts:
            doc = self.doc_for_text[text]
            input_ids.append([doc * 1000 + pos + 1 for pos in range(80)])
        return {"input_ids": input_ids}

    @staticmethod
    def decode(token_ids, **_kwargs):
        pieces = []
        for token_id in token_ids:
            doc, remainder = divmod(int(token_id) - 1, 1000)
            pieces.append(f"d{doc}t{remainder}")
        return " ".join(pieces)


class _FakeBaseModel:
    def __init__(self, width: int, control_direction):
        self.width = width
        self.control_direction = torch.as_tensor(control_direction, dtype=torch.float32)
        self.layer = SimpleNamespace(mlp=SimpleNamespace(down_proj=_FakeDownProj(width)))

    def __call__(self, *, input_ids, **_kwargs):
        ids = input_ids.to(dtype=torch.long)
        doc = torch.div(ids - 1, 1000, rounding_mode="floor").to(dtype=torch.float32)
        pos = torch.remainder(ids - 1, 1000).to(dtype=torch.float32)
        amplitude = 10.0 + doc + pos * 0.001

        # Coordinate 1 acts as a real neuron. The orthogonal correction makes
        # the same values project positively onto the selected random control.
        values = torch.zeros((*ids.shape, self.width), dtype=torch.float32)
        values[..., 1] = amplitude
        direction = self.control_direction
        d_perp = direction.clone()
        d_perp[1] = 0.0
        projected_norm = torch.dot(d_perp, direction)
        values += (amplitude * (2.0 / projected_norm))[..., None] * d_perp
        self.layer.mlp.down_proj(values)
        return SimpleNamespace(last_hidden_state=values)


class TargetedModelCollectionTests(unittest.TestCase):
    def setUp(self):
        with HEAD200.open(encoding="utf-8") as stream:
            self.rows = [json.loads(line) for line in stream if line.strip()]
        self.assertEqual(len(self.rows), 200)
        self.tokenizer = _FakeTokenizer(self.rows)
        self.width = 4
        direction = atlas.generate_controls(
            collect64_model.PILOT_CONTROL_POOL_SIZE, self.width,
            collect64_model.PILOT_CONTROL_SEED, 0,
        )[0]
        self.base_model = _FakeBaseModel(self.width, direction)
        self.units = {
            "L0:u1": {
                "unit_key": "L0:u1", "type": "neuron", "layer": 0,
                "record": {"unit": 1},
            },
            "L0:c0": {
                "unit_key": "L0:c0", "type": "control", "layer": 0,
                "record": {"control_id": 0},
            },
        }

    def collect(self):
        return collect64_model.collect_fresh(
            self.units, data=HEAD200, torch=torch, tokenizer=self.tokenizer,
            base_model=self.base_model, layers=[self.base_model.layer],
            device="cpu", docs_limit=200, batch_size=8,
        )

    def test_head200_hooks_candidates_order_windows_and_pilot_reproduction(self):
        with patch.object(atlas, "EXPECTED_WIDTH", self.width):
            pilot_slice = self.collect()
            fresh = self.collect()

        self.assertEqual(set(fresh), set(self.units))
        with tempfile.TemporaryDirectory() as output_dir:
            fresh_path = Path(output_dir) / "fresh.jsonl"
            collect64_model.write_fresh(fresh_path, fresh)
            self.assertEqual(collect64.load_fresh(fresh_path), fresh)
        for unit_key, strata in fresh.items():
            self.assertEqual(set(strata), set(collect64_model.FRESH_COUNTS))
            for category, count in collect64_model.FRESH_COUNTS.items():
                self.assertEqual(len(strata[category]), count)
            activations = [row["activation"] for row in strata["top"]]
            self.assertEqual(activations, sorted(activations, reverse=True))

            # Treat the first deterministic replay as the pilot collection for
            # this 200-doc slice, then verify a second targeted pass reproduces
            # its top 12 through collect64's own pilot-window contract.
            pilot_record = {
                category: [
                    {"window": row["window"], "act": row["activation"],
                     "doc": row["doc"], "pos": row["pos"]}
                    for row in pilot_slice[unit_key][category]
                ]
                for category in ("top", "random", "weak")
            }
            unit = {**self.units[unit_key], "record": pilot_record}
            _assembled, reproduced = collect64.assemble_unit(unit, strata)
            self.assertEqual(reproduced, 12)
            self.assertEqual(
                [row["window"] for row in strata["top"][:12]],
                [row["window"] for row in pilot_record["top"][:12]],
            )

        docs = atlas._read_docs(HEAD200, self.tokenizer, 200)
        first = fresh["L0:u1"]["top"][0]
        expected = atlas.decode_marked_window(self.tokenizer, docs[first["doc"]]["ids"], first["pos"])
        self.assertEqual(first["window"], expected)

    def test_control_directions_are_selected_from_part56_seeded_pool(self):
        with patch.object(atlas, "EXPECTED_WIDTH", self.width):
            fresh = self.collect()
        control_rows = fresh["L0:c0"]["top"]
        self.assertEqual(len(control_rows), collect64_model.FRESH_COUNTS["top"])
        self.assertEqual(control_rows[0]["doc"], 199)


if __name__ == "__main__":
    unittest.main()
