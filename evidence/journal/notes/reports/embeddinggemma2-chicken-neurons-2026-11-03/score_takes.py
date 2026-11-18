"""Score audio takes against all frozen chicken-edit versions (Amendment 6).

Run offline on CPU with the same environment as record.py:
  HF_HUB_OFFLINE=1 PYTHONPATH=/home/elisepc/eg2-extra \
    /home/elisepc/neuron-probe/.venv/bin/python -B score_takes.py --takes takes.tsv --out scores.json

The three direct weight edits are rebuilt from the committed fit audio in a
streaming pass. Their selected settings come from weight_edit_results.json;
only short sufficient statistics and the final patches are retained.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import edit_arms as ea  # noqa: E402
import record  # noqa: E402
import weight_edit as we  # noqa: E402

VERSION_NAMES = (
    "unedited", "output_gate", "inside_full", "gentle", "W1", "W2", "W2_control"
)
GATED_NAMES = ("output_gate", "inside_full", "gentle")
TRAIN_ROLES = {"fit_real", "fit_imit", "keep"}
MODEL_JOB_MARKERS = ("record188.py", "weight_edit.py", "score_takes.py")

FROZEN_FILES = {
    "edit_arms.py": HERE / "edit_arms.py",
    "edit_verdicts.json": HERE / "edit_verdicts.json",
    "inside_gate.py": HERE / "inside_gate.py",
    "inside_params.npz": HERE / "inside_params.npz",
    "gentle_gate.py": HERE / "gentle_gate.py",
    "gentle_results.json": HERE / "gentle_results.json",
    "weight_edit.py": HERE / "weight_edit.py",
    "weight_edit_results.json": HERE / "weight_edit_results.json",
    "record.py": HERE / "record.py",
    "chicken-o-meter/server.py": Path(
        "/home/elisepc/green/Journal/scripts/chicken-o-meter/server.py"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def frozen_hash_prefixes() -> dict[str, str]:
    result = {}
    for name, path in FROZEN_FILES.items():
        if not path.is_file():
            raise FileNotFoundError(f"frozen source file is missing: {path}")
        result[name] = sha256_file(path)[:16]
    return result


def active_model_jobs() -> list[tuple[int, str]]:
    """Find competing requested scripts among Python processes, excluding self."""
    active: list[tuple[int, str]] = []
    current_pid = os.getpid()
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return active
    for entry in proc_root.iterdir():
        if not entry.name.isdigit() or int(entry.name) == current_pid:
            continue
        try:
            executable = os.readlink(entry / "exe").lower()
            if "python" not in Path(executable).name:
                continue
            argv = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                errors="replace"
            )
        except (OSError, ProcessLookupError):
            continue
        for marker in MODEL_JOB_MARKERS:
            if marker in argv:
                active.append((int(entry.name), marker))
                break
    return active


def wait_for_model_slot() -> None:
    while True:
        jobs = active_model_jobs()
        if not jobs:
            return
        listed = ", ".join(f"pid {pid} ({name})" for pid, name in jobs)
        print(f"Waiting for the single model slot: {listed}", flush=True)
        time.sleep(20)


def decode_take(path: str | Path) -> np.ndarray:
    """Use record.py's ffmpeg decoder, with its 300-second recording limit."""
    audio = record.decode(path)
    return audio[: 16000 * 300]


def read_tsv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    required = ("path", "label", "route", "note")
    if not rows:
        raise ValueError("takes TSV has no data rows")
    missing = [key for key in required if key not in rows[0]]
    if missing:
        raise ValueError(f"takes TSV is missing required columns: {missing}")
    for index, row in enumerate(rows, 2):
        if row["label"] not in {"real", "imitation", "other"}:
            raise ValueError(f"row {index}: label must be real, imitation, or other")
        if not row["path"]:
            raise ValueError(f"row {index}: path is empty")
    return [{key: str(row.get(key, "")) for key in required} for row in rows]


class GateHook:
    """Capture Amendment 2 neurons and optionally write inside layer 19."""

    def __init__(self, model: SentenceTransformer, settings: dict[str, Any]):
        self.model = model
        self.layers = model[0].model.language_model.layers
        self.sign17 = int(settings["sign17"])
        self.sign19 = int(settings["sign19"])
        self.t17 = float(settings["t17"])
        self.t19 = float(settings["t19"])
        self.d = torch.as_tensor(settings["d"], dtype=torch.float32)
        self.R = float(settings["R"])
        self.k = float(settings["inside_k"])
        self.gentle_s = float(settings["gentle_s"])
        self.mode = "none"
        self.grab: dict[int, torch.Tensor] = {}
        self.last_gate: dict[str, Any] | None = None
        self.handles = []
        for layer_index in (17, 19):
            self.handles.append(
                self.layers[layer_index].mlp.down_proj.register_forward_pre_hook(
                    lambda _module, args, L=layer_index: self.grab.__setitem__(
                        L, args[0][0].float().mean(0)
                    )
                )
            )
        self.handles.append(
            self.layers[19].register_forward_hook(self._after_layer19)
        )

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _read_gate(self) -> dict[str, Any]:
        if 17 not in self.grab or 19 not in self.grab:
            raise RuntimeError("gate neuron hooks did not run in the current forward pass")
        # Amendment 2 fits the thresholds against float16 stored activations.
        v17 = float(np.float16(self.grab[17][1574]))
        v19 = float(np.float16(self.grab[19][1129]))
        s17 = self.sign17 * v17
        s19 = self.sign19 * v19
        g = max(s19 - self.t19, 0.0) * max(self.t17 - s17, 0.0)
        return {
            "open": bool(g > 0.0),
            "g": float(g),
            "s17": float(s17),
            "s19": float(s19),
            "neuron_values": {"L17_N1574": float(s17), "L19_N1129": float(s19)},
        }

    def _after_layer19(self, _module: Any, _args: Any, output: Any) -> Any:
        gate = self._read_gate()
        self.last_gate = gate
        if self.mode not in {"inside_full", "gentle"} or not gate["open"]:
            return None
        if self.mode == "inside_full":
            magnitude = (self.k * self.R) * gate["g"]
        else:
            # Equivalent to InsideModel's scale * g, with scale=s*R/g.
            magnitude = self.gentle_s * self.R
        hidden = output[0] if isinstance(output, tuple) else output
        new_hidden = hidden - magnitude * self.d.to(
            device=hidden.device, dtype=hidden.dtype
        )
        if isinstance(output, tuple):
            return (new_hidden,) + tuple(output[1:])
        return new_hidden

    def embed(self, audio: np.ndarray, mode: str = "none") -> tuple[np.ndarray, dict[str, Any]]:
        self.grab.clear()
        self.last_gate = None
        self.mode = mode
        try:
            with torch.inference_mode():
                embedding = self.model.encode(
                    {"audio": {"array": audio, "sampling_rate": 16000}},
                    normalize_embeddings=True,
                )
        finally:
            self.mode = "none"
        if self.last_gate is None:
            raise RuntimeError("layer 19 hook did not produce a gate reading")
        return np.asarray(embedding, dtype=np.float32).reshape(-1), self.last_gate


class WeightEdits:
    """Rebuild Amendment 5's frozen vectors in a bounded-memory streaming pass."""

    def __init__(self, runner: "ScoreRunner"):
        self.runner = runner
        self.model = runner.model
        self.layers = self.model[0].model.language_model.layers
        self.mlp = self.layers[we.LAYER0].mlp
        self.results = json.loads((HERE / "weight_edit_results.json").read_text())
        self.original_up = self.mlp.up_proj.weight.detach().clone()
        self.original_gate = self.mlp.gate_proj.weight.detach().clone()
        self.original_down = self.mlp.down_proj.weight.detach().clone()
        self.original_hash = self.weight_hash()
        self.audit: dict[str, dict[str, Any]] = {}
        self.w1_patch, self.w2_delta, self.w2_control_delta = self._build_patches()

    def weight_hash(self) -> str:
        digest = hashlib.sha256()
        for parameter in (
            self.mlp.up_proj.weight,
            self.mlp.gate_proj.weight,
            self.mlp.down_proj.weight,
        ):
            array = parameter.detach().to(device="cpu").contiguous().numpy()
            digest.update(array.view(np.uint8).tobytes())
        return digest.hexdigest()

    def _build_patches(self) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
        # The committed result contains fitted settings and control clip IDs, but
        # not the vectors. Rebuild those vectors from exactly the original fit,
        # keep, and ESC-50 discovery clips without retaining the activations.
        with np.load(HERE / "acts.npz") as archive:
            ref = {name: archive[name] for name in archive.files}
        with np.load(HERE / "vis_acts.npz") as archive:
            vis = {name: archive[name] for name in archive.files}
        with np.load(HERE / "pixabay_acts.npz") as archive:
            pix = {name: archive[name] for name in archive.files}
        with (HERE / "vis_manifest.csv").open("r", encoding="utf-8", newline="") as stream:
            manifest = list(csv.DictReader(stream))
        clips, sets, _fit_people = we.build_clips(ref, vis, pix, manifest)

        sample_groups = [
            [item_id for item_id in sets["fit_real"] if clips[item_id]["origin"] == "pixabay"][:5],
            [item_id for item_id in sets["fit_real"] if clips[item_id]["origin"] == "esc"][:5],
            sets["fit_imit"][:5],
            sets["keep"][:5],
            sets["test_real"][:5],
            sets["test_imit"][:5],
            sets["esc_test"][:5],
            sets["other_imit"][:5],
        ]
        parity_ids = [item_id for group in sample_groups for item_id in group]
        parity_set = set(parity_ids)
        item_order = parity_ids + [item_id for item_id in clips if item_id not in parity_set]

        x_sum = {role: np.zeros(we.HIDDEN, dtype=np.float64) for role in TRAIN_ROLES}
        x_count = {role: 0 for role in TRAIN_ROLES}
        key_sums: dict[str, tuple[np.ndarray, int]] = {}
        covariance_sum = np.zeros((we.INTERMEDIATE, we.INTERMEDIATE), dtype=np.float64)
        covariance_count = 0
        captured: dict[str, np.ndarray] = {}

        def capture_x(_module: Any, args: tuple[Any, ...]) -> None:
            captured["x"] = (
                args[0][0]
                .detach()
                .to(device="cpu", dtype=torch.float32)
                .contiguous()
                .numpy()
                .copy()
            )

        def capture_key(_module: Any, args: tuple[Any, ...]) -> None:
            captured["key"] = (
                args[0][0]
                .detach()
                .to(device="cpu", dtype=torch.float32)
                .contiguous()
                .numpy()
                .copy()
            )

        hooks = [
            self.mlp.register_forward_pre_hook(capture_x),
            self.mlp.down_proj.register_forward_pre_hook(capture_key),
        ]
        try:
            for item_id in item_order:
                item = clips[item_id]
                role = item["role"]
                if role not in TRAIN_ROLES:
                    continue
                captured.clear()
                audio = record.decode(item["path"])
                if item["cut"]:
                    audio = audio[: 16000 * int(item["cut"])]
                with torch.inference_mode():
                    self.model.encode(
                        {"audio": {"array": audio, "sampling_rate": 16000}},
                        normalize_embeddings=True,
                    )
                if "x" not in captured or "key" not in captured:
                    raise RuntimeError(f"{item_id}: missing layer-20 weight-edit statistics")
                x, key = captured["x"], captured["key"]
                x_sum[role] += x.sum(axis=0, dtype=np.float64)
                x_count[role] += int(x.shape[0])
                if role in {"fit_real", "fit_imit"}:
                    key_sums[item_id] = (
                        key.sum(axis=0, dtype=np.float64), int(key.shape[0])
                    )
                if role in {"fit_real", "keep"}:
                    key64 = key.astype(np.float64, copy=False)
                    covariance_sum[:] += key64.T @ key64
                    covariance_count += int(key.shape[0])
                del audio, x, key
        finally:
            for handle in hooks:
                handle.remove()

        pos = self.results["position_statistics"]
        expected_counts = {
            "fit_real_positions": x_count["fit_real"],
            "fit_imitation_positions": x_count["fit_imit"],
            "keep_positions": x_count["keep"],
            "w2_covariance_positions": covariance_count,
        }
        for name, value in expected_counts.items():
            expected = int(pos[name] if name != "w2_covariance_positions" else self.results["w2_covariance"]["positions"])
            if value != expected:
                raise RuntimeError(f"reconstructed {name}={value}, expected {expected}")

        means = {
            role: x_sum[role] / float(x_count[role])
            for role in ("fit_real", "fit_imit", "keep")
        }
        mean_all_fit = (
            x_sum["fit_real"] + x_sum["fit_imit"]
        ) / float(x_count["fit_real"] + x_count["fit_imit"])
        up_row = ea.normalize_vector(means["fit_imit"] - means["keep"])
        gate_direction = ea.normalize_vector(means["fit_imit"] - means["fit_real"])
        m_direction = ea.normalize_vector(mean_all_fit)
        mean_fit_x_dot_m = float(np.dot(mean_all_fit, m_direction))

        w1_fit = self.results["arms"]["W1"]["fit"]
        tau = float(w1_fit["tau"])
        gate_row = np.asarray(
            gate_direction - tau * m_direction / mean_fit_x_dot_m,
            dtype=np.float32,
        )
        w1_neuron = int(pos["w1_neuron_j"])
        w1_patch = {
            "neuron": w1_neuron,
            "up_row": up_row,
            "gate_row": gate_row,
            "down_col": None,
        }
        # Python's walrus is intentionally kept out of tensor arithmetic below;
        # form the committed W1 column with the selected gentle strength.
        w1_patch["down_col"] = -np.asarray(self.runner.d, dtype=np.float32) * (
            float(w1_fit["s"]) * self.runner.R / float(w1_fit["a_med"])
        )

        covariance = covariance_sum / float(covariance_count)
        covariance = (covariance + covariance.T) * 0.5
        covariance_mean_diag = float(np.mean(np.diag(covariance)))
        covariance[np.diag_indices_from(covariance)] += 1e-4 * covariance_mean_diag

        actual_ids = list(sets["fit_imit"])
        actual_sum = sum(
            (key_sums[item_id][0] for item_id in actual_ids),
            start=np.zeros(we.INTERMEDIATE, dtype=np.float64),
        )
        actual_count = sum(key_sums[item_id][1] for item_id in actual_ids)
        key_star_actual = (actual_sum / float(actual_count)).astype(np.float64)

        control_ids = self.results["w2_covariance"]["control_shuffled_fit_imitation_clip_ids"]
        missing = [item_id for item_id in control_ids if item_id not in key_sums]
        if missing:
            raise RuntimeError(f"W2 control IDs were not in the captured fit set: {missing}")
        control_sum = sum(
            (key_sums[item_id][0] for item_id in control_ids),
            start=np.zeros(we.INTERMEDIATE, dtype=np.float64),
        )
        control_count = sum(key_sums[item_id][1] for item_id in control_ids)
        key_star_control = (control_sum / float(control_count)).astype(np.float64)
        solved = np.linalg.solve(
            covariance, np.stack([key_star_actual, key_star_control], axis=1)
        )
        denominator_actual = float(np.dot(key_star_actual, solved[:, 0]))
        denominator_control = float(np.dot(key_star_control, solved[:, 1]))

        covariance_result = self.results["w2_covariance"]
        validation = {
            "mean_fit_x_dot_m": (mean_fit_x_dot_m, float(pos["mean_fit_x_dot_m"])),
            "covariance_mean_diagonal": (
                covariance_mean_diag,
                float(covariance_result["mean_diagonal_before_ridge"]),
            ),
            "actual_denominator": (
                denominator_actual,
                float(covariance_result["actual_denominator"]),
            ),
            "control_denominator": (
                denominator_control,
                float(covariance_result["control_denominator"]),
            ),
        }
        for name, (actual, expected) in validation.items():
            if not np.isclose(actual, expected, rtol=5e-4, atol=1e-9):
                raise RuntimeError(
                    f"reconstructed W1/W2 statistic {name}={actual:.12g}, "
                    f"committed value={expected:.12g}"
                )
        self.rebuild_validation = {
            key: {"reconstructed": float(actual), "committed": float(expected)}
            for key, (actual, expected) in validation.items()
        }

        def w2_delta(scale: float, solved_vector: np.ndarray, denominator: float) -> np.ndarray:
            return np.asarray(
                np.outer(
                    -float(scale) * self.runner.R * np.asarray(self.runner.d, dtype=np.float64),
                    solved_vector,
                )
                / denominator,
                dtype=np.float32,
            )

        w2_fit = self.results["arms"]["W2"]["fit"]
        control_fit = self.results["arms"]["W2_control"]["fit"]
        w2_delta_actual = w2_delta(
            float(w2_fit["s"]), solved[:, 0], denominator_actual
        )
        w2_delta_control = w2_delta(
            float(control_fit["s"]), solved[:, 1], denominator_control
        )
        del clips, ref, vis, pix, covariance_sum, covariance, solved
        return w1_patch, w2_delta_actual, w2_delta_control

    @contextlib.contextmanager
    def patched(self, name: str, delta_override: np.ndarray | None = None):
        before = self.weight_hash()
        if before != self.original_hash:
            raise RuntimeError(f"weights were not restored before applying {name}")
        try:
            with torch.no_grad():
                if name == "W1":
                    patch = self.w1_patch
                    j = patch["neuron"]
                    self.mlp.up_proj.weight[j].copy_(torch.from_numpy(patch["up_row"]))
                    self.mlp.gate_proj.weight[j].copy_(torch.from_numpy(patch["gate_row"]))
                    self.mlp.down_proj.weight[:, j].copy_(torch.from_numpy(patch["down_col"]))
                else:
                    if delta_override is not None:
                        delta = delta_override
                    elif name == "W2":
                        delta = self.w2_delta
                    elif name == "W2_control":
                        delta = self.w2_control_delta
                    else:
                        raise ValueError(f"unknown weight edit {name}")
                    self.mlp.down_proj.weight.copy_(
                        self.original_down + torch.from_numpy(np.asarray(delta, dtype=np.float32))
                    )
            yield
        finally:
            with torch.no_grad():
                if name == "W1":
                    j = self.w1_patch["neuron"]
                    self.mlp.up_proj.weight.copy_(self.original_up)
                    self.mlp.gate_proj.weight.copy_(self.original_gate)
                    self.mlp.down_proj.weight.copy_(self.original_down)
                else:
                    self.mlp.down_proj.weight.copy_(self.original_down)
            after = self.weight_hash()
            audit = {
                "before_sha256": before,
                "after_sha256": after,
                "bitwise_restored": before == after == self.original_hash,
            }
            self.audit[name] = audit
            if not audit["bitwise_restored"]:
                raise RuntimeError(f"{name} did not restore layer-20 MLP weights bitwise")


class ScoreRunner:
    def __init__(self, model: SentenceTransformer | None = None):
        wait_for_model_slot()
        torch.set_num_threads(6)
        torch.set_grad_enabled(False)
        if model is None:
            model = SentenceTransformer(
                record.SNAP,
                device="cpu",
                config_kwargs={"vision_config": None},
                model_kwargs={"dtype": torch.float32},
            )
        self.model = model
        self.model.eval()
        design = json.loads((HERE / "edit_verdicts.json").read_text())
        inside_results = json.loads((HERE / "inside_results.json").read_text())
        gentle_results = json.loads((HERE / "gentle_results.json").read_text())
        with np.load(HERE / "inside_params.npz") as archive:
            inside_params = {key: archive[key].copy() for key in archive.files}
        parameters = design["arms"]["written_gated_block"]["parameters"]
        cats_archive = np.load(HERE / "acts.npz")
        self.categories = [str(value) for value in cats_archive["categories"]]
        self.prompt_emb = np.asarray(cats_archive["prompt_emb"], dtype=np.float32)
        cats_archive.close()
        self.hen = self.categories.index("hen")
        self.rooster = self.categories.index("rooster")
        self.other_prompts = np.asarray(
            [i for i in range(len(self.categories)) if i not in (self.hen, self.rooster)],
            dtype=np.int64,
        )
        self.gate_settings = {
            "t19": float(parameters["t19"]),
            "t17": float(parameters["t17"]),
            "k": float(parameters["k"]),
            "sign17": int(design["signs"]["L17_N1574"]),
            "sign19": int(design["signs"]["L19_N1129"]),
            "d": inside_params["d"],
            "R": float(inside_params["R"]),
            "inside_k": float(inside_params["k"]),
            "gentle_s": float(gentle_results["fits"]["gentle"]["s"]),
        }
        self.d = np.asarray(self.gate_settings["d"], dtype=np.float32)
        self.R = float(self.gate_settings["R"])
        # Cross-check the explicitly registered Amendment 3 and 4 values.
        if not np.isclose(
            self.gate_settings["inside_k"],
            float(inside_results["arms"]["inside"]["k"]),
        ):
            raise RuntimeError("inside full-strength k differs from inside_results.json")
        self.gate = GateHook(self.model, self.gate_settings)
        self.weight_edits = WeightEdits(self)
        self.hashes = frozen_hash_prefixes()

    def close(self) -> None:
        self.gate.close()

    def embed(self, audio: np.ndarray, mode: str = "none") -> tuple[np.ndarray, dict[str, Any]]:
        return self.gate.embed(audio, mode)

    def verdict(self, embedding: np.ndarray) -> float:
        return float(
            ea.prompt_scores(
                embedding.reshape(1, -1),
                self.prompt_emb,
                self.hen,
                self.rooster,
                self.other_prompts,
            )[0]
        )

    def top_class(self, embedding: np.ndarray) -> str:
        return str(
            ea.prompt_predictions(
                embedding.reshape(1, -1), self.prompt_emb, self.categories
            )[0]
        )

    def version_record(
        self,
        embedding: np.ndarray,
        original: np.ndarray,
        before_score: float,
        gate: dict[str, Any] | None = None,
        weight: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = {
            "score_before": float(before_score),
            "score_after": self.verdict(embedding),
            "top_class": self.top_class(embedding),
            "cosine_to_unedited": float(np.dot(embedding, original)),
        }
        if gate is not None:
            result["gate"] = gate
        if weight is not None:
            result["weight_restore"] = weight
        return result

    def score_take(self, row: dict[str, str], audio: np.ndarray | None = None) -> dict[str, Any]:
        audio = decode_take(row["path"]) if audio is None else audio
        if len(audio) == 0:
            raise ValueError(f"decoded empty audio: {row['path']}")
        original, original_gate = self.embed(audio, "none")
        original_score = self.verdict(original)
        versions: dict[str, Any] = {
            "unedited": self.version_record(
                original, original, original_score, gate=None
            )
        }

        # The output arm has its own full unedited model pass, then the live
        # meter's exact post-pooling edit. This makes every registered version
        # account for one model pass while retaining the live meter semantics.
        output_base, output_reading = self.embed(audio, "none")
        output_embedding = ea.apply_written_gate(
            output_base,
            output_reading["s19"],
            output_reading["s17"],
            self.gate_settings["t19"],
            self.gate_settings["t17"],
            self.gate_settings["k"],
            ea.normalize_vector(
                self.prompt_emb[[self.hen, self.rooster]].mean(axis=0)
            ),
        )
        versions["output_gate"] = self.version_record(
            output_embedding,
            original,
            self.verdict(output_base),
            gate=output_reading,
        )

        inside_embedding, inside_reading = self.embed(audio, "inside_full")
        versions["inside_full"] = self.version_record(
            inside_embedding,
            original,
            self.verdict(original),
            gate=inside_reading,
        )
        gentle_embedding, gentle_reading = self.embed(audio, "gentle")
        versions["gentle"] = self.version_record(
            gentle_embedding,
            original,
            self.verdict(original),
            gate=gentle_reading,
        )

        for name in ("W1", "W2", "W2_control"):
            with self.weight_edits.patched(name):
                edited, _unused_gate = self.embed(audio, "none")
            versions[name] = self.version_record(
                edited,
                original,
                original_score,
                weight=dict(self.weight_edits.audit[name]),
            )
        return {
            "path": row["path"],
            "label": row["label"],
            "route": row["route"],
            "note": row["note"],
            "versions": versions,
        }


def summarize_scores(document: dict[str, Any]) -> dict[str, Any]:
    takes = document.get("takes")
    if not isinstance(takes, list):
        raise ValueError("score JSON must contain a takes list")
    labels = np.asarray([str(take.get("label", "")) for take in takes])
    report: dict[str, Any] = {"takes": len(takes), "versions": {}, "other_gate_open": {}}
    real_mask, imitation_mask, other_mask = labels == "real", labels == "imitation", labels == "other"
    auc_ready = int(real_mask.sum()) >= 6 and int(imitation_mask.sum()) >= 6
    for version in VERSION_NAMES:
        scores = np.asarray(
            [float(take["versions"][version]["score_after"]) for take in takes],
            dtype=np.float64,
        )
        entry: dict[str, Any] = {
            "real": _descriptive(scores[real_mask]),
            "imitation": _descriptive(scores[imitation_mask]),
            "other": _descriptive(scores[other_mask]),
        }
        if auc_ready:
            entry["auc_real_vs_imitation"] = ea.auc_values(
                scores[real_mask], scores[imitation_mask]
            )
        else:
            entry["auc_real_vs_imitation"] = None
            entry["auc_note"] = "descriptive only; need at least six real and six imitation takes"
        report["versions"][version] = entry
    for version in GATED_NAMES:
        opened = [
            bool(take["versions"][version]["gate"]["open"])
            for take, is_other in zip(takes, other_mask)
            if is_other
        ]
        report["other_gate_open"][version] = {
            "opened": int(sum(opened)),
            "total": len(opened),
            "all_closed": not any(opened),
        }
    return report


def _descriptive(values: np.ndarray) -> dict[str, Any]:
    if len(values) == 0:
        return {"n": 0, "mean": None, "min": None, "max": None}
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def score_tsv(takes_path: str | Path, out_path: str | Path) -> None:
    rows = read_tsv(takes_path)
    runner = ScoreRunner()
    try:
        scored = []
        for index, row in enumerate(rows, 1):
            scored.append(runner.score_take(row))
            # Progress reveals no per-take score, so the dry-run output can be
            # checked without exposing an unlabelled recording's verdict.
            print(f"scored take {index}/{len(rows)} across {len(VERSION_NAMES)} versions", flush=True)
        document = {
            "schema_version": 1,
            "model": {
                "snapshot": record.SNAP,
                "device": "cpu",
                "dtype": "float32",
                "torch_threads": 6,
                "audio": "record.decode; mono 16000 Hz f32le; first 300 seconds",
            },
            "frozen_sha256_prefixes": runner.hashes,
            "weight_rebuild_validation": runner.weight_edits.rebuild_validation,
            "versions": list(VERSION_NAMES),
            "takes": scored,
        }
        target = Path(out_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        print(
            f"wrote {target} ({len(scored)} takes; versions: {', '.join(VERSION_NAMES)})",
            flush=True,
        )
    finally:
        runner.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--takes", help="input TSV with path, label, route, note")
    parser.add_argument("--out", help="output JSON path")
    parser.add_argument(
        "--summary", metavar="JSON", help="summarize a score_takes output JSON"
    )
    args = parser.parse_args(argv)
    if args.summary:
        if args.takes or args.out:
            parser.error("--summary cannot be combined with --takes or --out")
        document = json.loads(Path(args.summary).read_text(encoding="utf-8"))
        print(json.dumps(summarize_scores(document), indent=2))
        return 0
    if not args.takes or not args.out:
        parser.error("--takes and --out are required unless --summary JSON is used")
    score_tsv(args.takes, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
