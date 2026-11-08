"""Amendment 5: direct layer-20 weight edits for the chicken imitation task.

The unedited model runs once per clip. Forward pre-hooks cache the layer-20
entry and each layer's per-layer input and keyword arguments. Subsequent fit
and score passes replay only layers 20 through 23 and the final pooling path.
"""
import argparse
import csv
import hashlib
import json
import resource
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import edit_arms as ea  # noqa: E402
from gentle_gate import GRID as GENTLE_GRID  # noqa: E402
from other_items import write_other_items_report  # noqa: E402
from record import decode  # noqa: E402

# Data inputs are the stored ESC-50, Pixabay chicken and VocalImitationSet arrays,
# the VocalImitationSet manifest, and their corresponding audio directories.
LAYER0 = 20
LAYER_LAST = 23
HIDDEN = 512
INTERMEDIATE = 2048
RANDOM_SEED = 20261104
MIN_COS = 0.8
CACHE_LIMIT_BYTES = int(2.5 * 1024**3)
HUMAN = {
    "laughing", "crying_baby", "sneezing", "coughing", "breathing",
    "snoring", "clapping", "footsteps", "brushing_teeth", "drinking_sipping",
}


def log(message):
    print(message, flush=True)


def rss_gib():
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0 * 1024.0)


def tensor_cpu_copy(value, memo=None):
    """Clone captured tensors to CPU float32, retaining identity aliases."""
    if memo is None:
        memo = {}
    if isinstance(value, torch.Tensor):
        key = (
            value.device.type,
            value.data_ptr(),
            value.storage_offset(),
            tuple(value.shape),
            tuple(value.stride()),
            value.dtype,
        )
        if key in memo:
            return memo[key]
        result = value.detach().to(device="cpu")
        if result.is_floating_point() and result.dtype != torch.float32:
            result = result.to(torch.float32)
        result = result.clone()
        memo[key] = result
        return result
    if isinstance(value, dict):
        return {key: tensor_cpu_copy(item, memo) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(tensor_cpu_copy(item, memo) for item in value)
    if isinstance(value, list):
        return [tensor_cpu_copy(item, memo) for item in value]
    return value


def merge_batch(values):
    """Merge one-item cached call arguments into a replay batch."""
    first = values[0]
    if first is None:
        if any(value is not None for value in values):
            raise ValueError("inconsistent None in cached layer arguments")
        return None
    if isinstance(first, torch.Tensor):
        if all(isinstance(value, torch.Tensor) and value.shape == first.shape for value in values):
            if all(torch.equal(first, value) for value in values[1:]):
                return first
            if first.ndim >= 1 and first.shape[0] == 1:
                return torch.cat(values, dim=0)
        if all(isinstance(value, torch.Tensor) and value.ndim == first.ndim for value in values):
            if all(value.shape[1:] == first.shape[1:] for value in values) and all(
                value.shape[0] == 1 for value in values
            ):
                return torch.cat(values, dim=0)
        if all(torch.equal(first, value) for value in values[1:]):
            return first
        raise ValueError(f"cannot batch cached tensor arguments with shapes {[tuple(v.shape) for v in values]}")
    if isinstance(first, dict):
        if any(value.keys() != first.keys() for value in values):
            raise ValueError("inconsistent cached keyword mappings")
        return {key: merge_batch([value[key] for value in values]) for key in first}
    if isinstance(first, tuple):
        if any(len(value) != len(first) for value in values):
            raise ValueError("inconsistent cached argument tuples")
        return tuple(merge_batch([value[i] for value in values]) for i in range(len(first)))
    if isinstance(first, list):
        if any(len(value) != len(first) for value in values):
            raise ValueError("inconsistent cached argument lists")
        return [merge_batch([value[i] for value in values]) for i in range(len(first))]
    if any(value != first for value in values[1:]):
        raise ValueError(f"cannot batch differing scalar arguments: {values!r}")
    return first


def cache_nbytes(value, seen=None):
    if seen is None:
        seen = set()
    if isinstance(value, torch.Tensor):
        ident = id(value)
        if ident in seen:
            return 0
        seen.add(ident)
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(cache_nbytes(v, seen) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(cache_nbytes(v, seen) for v in value)
    return 0


class CaptureRunner:
    """One unedited full pass per clip; cache the inputs needed for tail replay."""

    def __init__(self, sentence_model, layers, ffmpeg="ffmpeg"):
        self.model = sentence_model
        self.layers = layers
        self.ffmpeg = ffmpeg
        self.context = None
        self.handles = []

        for index in range(LAYER0, LAYER_LAST + 1):
            def layer_hook(module, args, kwargs, index=index):
                if self.context is None:
                    return None
                memo = self.context["tensor_memo"]
                self.context["calls"][index] = (
                    tensor_cpu_copy(tuple(args), memo),
                    tensor_cpu_copy(kwargs, memo),
                )

            self.handles.append(
                layers[index].register_forward_pre_hook(layer_hook, with_kwargs=True)
            )

        def mlp_hook(module, args, kwargs):
            if self.context is None:
                return None
            item = self.context["item"]
            if item["role"] in {"fit_real", "fit_imit", "keep"}:
                x = args[0][0].detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().copy()
                self.context["x"] = x

        def key_hook(module, args, kwargs):
            if self.context is None:
                return None
            item = self.context["item"]
            if item["role"] in {"fit_real", "fit_imit", "keep"}:
                k = args[0][0].detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().copy()
                self.context["k"] = k

        self.handles.append(
            layers[LAYER0].mlp.register_forward_pre_hook(mlp_hook, with_kwargs=True)
        )
        self.handles.append(
            layers[LAYER0].mlp.down_proj.register_forward_pre_hook(key_hook, with_kwargs=True)
        )

        self.pool_mask = None

        def pooling_hook(module, args, kwargs):
            if self.context is None:
                return None
            features = args[0] if args else kwargs.get("features", {})
            mask = features.get("attention_mask") if isinstance(features, dict) else None
            self.pool_mask = tensor_cpu_copy(mask)

        self.handles.append(
            sentence_model[1].register_forward_pre_hook(pooling_hook, with_kwargs=True)
        )

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def encode_item(self, item):
        self.context = {
            "item": item,
            "calls": {},
            "x": None,
            "k": None,
            "tensor_memo": {},
        }
        self.pool_mask = None
        audio = decode(item["path"], ffmpeg=self.ffmpeg)
        if item["cut"]:
            audio = audio[:16000 * int(item["cut"])]
        model_input = {"audio": {"array": audio, "sampling_rate": 16000}}
        with torch.inference_mode():
            embedding = np.asarray(
                self.model.encode(model_input, normalize_embeddings=True), dtype=np.float32
            ).reshape(-1)
        if set(self.context["calls"]) != set(range(LAYER0, LAYER_LAST + 1)):
            raise RuntimeError(
                f"{item['id']}: incomplete layer capture {sorted(self.context['calls'])}"
            )

        calls = self.context["calls"]
        initial_args = calls[LAYER0][0]
        if len(initial_args) < 2:
            raise RuntimeError(f"{item['id']}: layer {LAYER0} did not receive hidden and PLE inputs")
        hidden20 = initial_args[0]
        per_layer_inputs = []
        extra_args = []
        kwargs_by_layer = []
        for index in range(LAYER0, LAYER_LAST + 1):
            args, kwargs = calls[index]
            if len(args) < 2:
                raise RuntimeError(f"{item['id']}: layer {index} did not receive a PLE input")
            per_layer_inputs.append(args[1])
            extra_args.append(args[2:])
            kwargs_by_layer.append(kwargs)
        pool_mask = self.pool_mask
        if pool_mask is None:
            pool_mask = torch.ones((1, hidden20.shape[1]), dtype=torch.int64)
        if pool_mask.shape[-1] != hidden20.shape[1]:
            raise RuntimeError(
                f"{item['id']}: pooling mask {tuple(pool_mask.shape)} does not match "
                f"sequence length {hidden20.shape[1]}"
            )

        cache = {
            "id": item["id"],
            "hidden20": hidden20,
            "per_layer_inputs": per_layer_inputs,
            "extra_args": extra_args,
            "kwargs": kwargs_by_layer,
            "pool_mask": pool_mask,
        }
        x = self.context["x"]
        k = self.context["k"]
        self.context = None
        return cache, embedding, x, k


class TailRunner:
    def __init__(self, sentence_model):
        self.model = sentence_model
        self.backbone = sentence_model[0].model
        self.text = self.backbone.language_model
        self.layers = self.text.layers
        self.pooling = sentence_model[1]
        self.normalize = sentence_model[2]

    @staticmethod
    def max_batch(sequence_length):
        if sequence_length >= 600:
            return 8
        if sequence_length >= 300:
            return 12
        return 24

    def tail_batch(self, caches, pushes=None):
        if not caches:
            return np.zeros((0, 768), dtype=np.float32)
        seq_len = int(caches[0]["hidden20"].shape[1])
        if any(int(cache["hidden20"].shape[1]) != seq_len for cache in caches):
            raise ValueError("tail_batch requires equal sequence lengths")
        hidden = torch.cat([cache["hidden20"] for cache in caches], dim=0)
        if pushes is not None:
            push_tensor = torch.as_tensor(pushes, dtype=torch.float32)
            hidden = hidden + push_tensor[:, None, :]
        with torch.inference_mode():
            for offset, index in enumerate(range(LAYER0, LAYER_LAST + 1)):
                per_layer = torch.cat(
                    [cache["per_layer_inputs"][offset] for cache in caches], dim=0
                )
                extra_count = len(caches[0]["extra_args"][offset])
                extras = [
                    merge_batch([cache["extra_args"][offset][j] for cache in caches])
                    for j in range(extra_count)
                ]
                kwargs = merge_batch([cache["kwargs"][offset] for cache in caches])
                result = self.layers[index](hidden, per_layer, *extras, **kwargs)
                hidden = result[0] if isinstance(result, tuple) else result

            token_embeddings = self.text.embedding_projection(self.text.norm(hidden))
            mask = torch.cat([cache["pool_mask"] for cache in caches], dim=0).to(
                device=token_embeddings.device
            )
            features = {"token_embeddings": token_embeddings, "attention_mask": mask}
            pooled = self.pooling(features)["sentence_embedding"]
            normalized = self.normalize({"sentence_embedding": pooled})["sentence_embedding"]
        return normalized.detach().to(device="cpu", dtype=torch.float32).numpy()

    def embed_ids(self, ids, caches, pushes_by_id=None):
        out = np.empty((len(ids), 768), dtype=np.float32)
        groups = defaultdict(list)
        for output_index, item_id in enumerate(ids):
            groups[int(caches[item_id]["hidden20"].shape[1])].append((output_index, item_id))
        for seq_len in sorted(groups):
            rows = groups[seq_len]
            size = self.max_batch(seq_len)
            for start in range(0, len(rows), size):
                batch_rows = rows[start:start + size]
                batch_caches = [caches[item_id] for _, item_id in batch_rows]
                pushes = None
                if pushes_by_id is not None:
                    pushes = np.stack([pushes_by_id[item_id] for _, item_id in batch_rows])
                embeddings = self.tail_batch(batch_caches, pushes)
                out[[i for i, _ in batch_rows]] = embeddings
        return out


def build_clips(ref, vis, pix, manifest, *, pixabay_audio, esc_audio, vis_audio):
    fit_mask, fit_people = ea.fit_imitation_split(manifest)
    if len(manifest) != len(vis["id"]):
        raise RuntimeError("vis_manifest.csv and vis_acts.npz lengths differ")
    if any(str(row["id"]) != str(vis["id"][i]) for i, row in enumerate(manifest)):
        raise RuntimeError("vis_manifest.csv and vis_acts.npz row order differs")

    clips = {}
    sets = {
        "fit_real": [],
        "fit_imit": [],
        "keep": [],
        "test_real": [],
        "test_imit": [],
        "esc_test": [],
        "other_imit": [],
    }

    def add(item_id, role, path, cut, embedding, label, act_mean, origin):
        if item_id in clips:
            raise RuntimeError(f"duplicate clip id: {item_id}")
        clips[item_id] = {
            "id": item_id,
            "role": role,
            "path": Path(path),
            "cut": cut,
            "embedding": np.asarray(embedding, dtype=np.float32),
            "label": str(label),
            "act_mean": act_mean,
            "origin": origin,
        }
        sets[role].append(item_id)

    for index in np.where(pix["label"] == "real")[0]:
        clip_id = str(pix["id"][index])
        add(
            f"pix:{clip_id}", "fit_real", pixabay_audio / f"{clip_id}.mp3", 30,
            pix["emb"][index], "real", None, "pixabay",
        )

    esc = ref["kind"] == "esc"
    chicken = np.isin(ref["label"], ["hen", "rooster"])
    for index in np.where(esc)[0]:
        fold = int(ref["fold"][index])
        if fold <= 3:
            role = "fit_real" if chicken[index] else "keep"
        elif fold in (4, 5):
            role = "esc_test"
        else:
            continue
        key = str(ref["key"][index])
        add(
            f"esc:{key}", role, esc_audio / key, None,
            ref["emb"][index], ref["label"][index], ref["act_mean"][index], "esc",
        )

    for index, row in enumerate(manifest):
        group, kind = row["group"], row["kind"]
        if group == "chicken" and kind == "imitation":
            role = "fit_imit" if fit_mask[index] else "test_imit"
        elif group == "chicken" and kind == "real":
            role = "test_real"
        elif group == "other" and kind == "imitation":
            role = "other_imit"
        else:
            continue
        add(
            f"vis:{row['id']}", role, vis_audio / row["local"], 30,
            vis["emb"][index], row.get("class", group), vis["act_mean"][index], "vis",
        )

    expected = {
        "fit_real": 228,
        "fit_imit": 18,
        "keep": 288,
        "test_real": 20,
        "test_imit": 20,
        "esc_test": 224,
        "other_imit": 112,
    }
    actual = {name: len(ids) for name, ids in sets.items()}
    if actual != expected:
        raise RuntimeError(f"Amendment 5 set sizes differ: {actual} != {expected}")
    return clips, sets, fit_people


def main(argv=None):
    started = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acts", type=Path, required=True, help="ESC-50 stored arrays")
    parser.add_argument("--vis-acts", type=Path, required=True, help="VocalImitationSet stored arrays")
    parser.add_argument("--pixabay-acts", type=Path, required=True, help="Pixabay chicken stored arrays")
    parser.add_argument("--vis-manifest", type=Path, required=True, help="VocalImitationSet manifest")
    parser.add_argument("--edit-verdicts", type=Path, required=True)
    parser.add_argument("--gentle-results", type=Path, required=True)
    parser.add_argument("--inside-params", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="local EmbeddingGemma 2 model directory")
    parser.add_argument("--base-safetensors", type=Path, required=True)
    parser.add_argument("--esc-audio", type=Path, required=True, help="ESC-50 audio directory")
    parser.add_argument("--pixabay-audio", type=Path, required=True, help="Pixabay chicken audio directory")
    parser.add_argument("--vis-audio", type=Path, required=True, help="VocalImitationSet audio directory")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg executable (default: resolve from PATH)")
    parser.add_argument("--results", type=Path, default=HERE / "weight_edit_results.json")
    parser.add_argument(
        "--out-other",
        metavar="PATH",
        help="write per-clip other-animal imitation predictions as JSON",
    )
    parser.add_argument(
        "--save-edits",
        type=Path,
        metavar="PATH.NPZ",
        help="save the in-place W1, W2 and W2-control patches plus fit settings and base hash",
    )
    args = parser.parse_args(argv)
    torch.set_num_threads(6)
    torch.set_grad_enabled(False)
    log("loading stored arrays")
    with np.load(args.acts) as archive:
        ref = {name: archive[name] for name in archive.files}
    with np.load(args.vis_acts) as archive:
        vis = {name: archive[name] for name in archive.files}
    with np.load(args.pixabay_acts) as archive:
        pix = {name: archive[name] for name in archive.files}
    with args.vis_manifest.open(newline="") as stream:
        manifest = list(csv.DictReader(stream))
    clips, sets, fit_people = build_clips(
        ref, vis, pix, manifest,
        pixabay_audio=args.pixabay_audio,
        esc_audio=args.esc_audio,
        vis_audio=args.vis_audio,
    )

    design = json.loads(args.edit_verdicts.read_text())
    gentle_results = json.loads(args.gentle_results.read_text())
    with np.load(args.inside_params) as archive:
        params = {name: archive[name] for name in archive.files}
    s17 = int(design["signs"]["L17_N1574"])
    s19 = int(design["signs"]["L19_N1129"])
    thresholds = design["arms"]["written_gated_block"]["parameters"]
    d = np.asarray(params["d"], dtype=np.float32)
    R = float(params["R"])
    gentle_s = float(gentle_results["fits"]["gentle"]["s"])
    categories = list(ref["categories"])
    prompt_emb = np.asarray(ref["prompt_emb"], dtype=np.float32)
    hen_idx, rooster_idx = categories.index("hen"), categories.index("rooster")
    other_idx = np.asarray(
        [i for i in range(len(categories)) if i not in (hen_idx, rooster_idx)]
    )

    log(f"sets { {name: len(ids) for name, ids in sets.items()} }")
    log(f"fit imitation participants {fit_people}")
    model = SentenceTransformer(
        str(args.model),
        device="cpu",
        config_kwargs={"vision_config": None},
        model_kwargs={"dtype": torch.float32},
    )
    model.eval()
    layers = model[0].model.language_model.layers
    tail = TailRunner(model)
    capture = CaptureRunner(model, layers, ffmpeg=args.ffmpeg)
    log(f"model loaded; starting parity-first capture; RSS {rss_gib():.2f} GiB")

    x_sum = {name: np.zeros(HIDDEN, dtype=np.float64) for name in ("fit_real", "fit_imit", "keep")}
    x_count = {name: 0 for name in ("fit_real", "fit_imit", "keep")}
    fit_imit_x = {}
    fit_key_sums = {}
    covariance_sum = np.zeros((INTERMEDIATE, INTERMEDIATE), dtype=np.float64)
    covariance_count = 0
    cache_by_id = {}
    parity_items = []
    parity_caches = {}
    parity_live_cos = []

    # Five examples from each set (and both sources of fit real) gives 40
    # parity clips, checked before any edit or fit grid is run.
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
    parity_ids = []
    for group in sample_groups:
        parity_ids.extend(group)
    if len(parity_ids) < 30:
        raise RuntimeError(f"parity sample has only {len(parity_ids)} clips")
    parity_set = set(parity_ids)

    def collect_stats(item, x, k):
        nonlocal covariance_count
        role = item["role"]
        if role not in x_sum:
            return
        if x is None or k is None:
            raise RuntimeError(f"{item['id']}: missing layer-20 MLP input or key capture")
        x_sum[role] += x.sum(axis=0, dtype=np.float64)
        x_count[role] += int(x.shape[0])
        if role == "fit_imit":
            fit_imit_x[item["id"]] = x
        ksum = k.sum(axis=0, dtype=np.float64)
        if role in {"fit_real", "fit_imit"}:
            fit_key_sums[item["id"]] = (ksum, int(k.shape[0]))
        if role in {"fit_real", "keep"}:
            k64 = k.astype(np.float64, copy=False)
            covariance_sum[:] += k64.T @ k64
            covariance_count += int(k.shape[0])

    def capture_one(item, retain=True, parity=False):
        cache, live, x, k = capture.encode_item(item)
        collect_stats(item, x, k)
        if retain:
            cache_by_id[item["id"]] = cache
        if parity:
            tail_embedding = tail.tail_batch([cache])[0]
            stored = item["embedding"]
            tail_cos = float(np.dot(tail_embedding, stored))
            live_cos = float(np.dot(live, stored))
            parity_items.append((item["id"], tail_cos))
            parity_live_cos.append(live_cos)
            parity_caches[item["id"]] = cache
            if item["role"] != "keep":
                cache_by_id[item["id"]] = cache
            log(
                f"parity {len(parity_items)}/{len(parity_ids)} {item['id']} "
                f"tail_cos={tail_cos:.9f} live_cos={live_cos:.9f}"
            )

    for item_id in parity_ids:
        capture_one(clips[item_id], retain=clips[item_id]["role"] != "keep", parity=True)
    parity_worst = min(cos for _, cos in parity_items)
    parity = {
        "clips": len(parity_items),
        "worst_tail_cosine_to_stored": parity_worst,
        "worst_live_full_cosine_to_stored": min(parity_live_cos),
        "threshold": 0.99999,
    }
    parity_batch_embeddings = tail.embed_ids(parity_ids, parity_caches)
    parity_batch_cos = [
        float(np.dot(parity_batch_embeddings[index], clips[item_id]["embedding"]))
        for index, item_id in enumerate(parity_ids)
    ]
    parity["worst_batched_tail_cosine_to_stored"] = min(parity_batch_cos)
    log(f"initial parity gate {parity}; RSS {rss_gib():.2f} GiB")
    if (
        len(parity_items) < 30
        or parity_worst < 0.99999
        or min(parity_batch_cos) < 0.99999
    ):
        raise RuntimeError(f"initial parity gate failed: {parity}")

    remaining = [item_id for item_id in clips if item_id not in parity_set]
    for index, item_id in enumerate(remaining, 1):
        item = clips[item_id]
        capture_one(item, retain=item["role"] != "keep", parity=False)
        if index % 50 == 0 or index == len(remaining):
            log(
                f"unedited capture {index}/{len(remaining)} {item_id}; "
                f"tail_cache={cache_nbytes(cache_by_id) / 1024**3:.2f} GiB "
                f"RSS={rss_gib():.2f} GiB elapsed={time.time() - started:.0f}s"
            )
    capture.close()

    cache_bytes = cache_nbytes(cache_by_id)
    if cache_bytes > CACHE_LIMIT_BYTES:
        raise RuntimeError(
            f"retained tail cache is {cache_bytes / 1024**3:.2f} GiB, "
            f"above the 2.5 GiB limit; process sets in chunks"
        )
    if cache_bytes > CACHE_LIMIT_BYTES * 0.85:
        log(
            f"cache is approaching the 2.5 GiB limit ({cache_bytes / 1024**3:.2f} GiB); "
            "keep-set tail entries were streamed and discarded"
        )
    log(
        f"capture complete: {len(clips)} unique clips, {len(cache_by_id)} retained tail caches, "
        f"{cache_bytes / 1024**3:.2f} GiB cached, RSS {rss_gib():.2f} GiB"
    )

    means = {
        name: x_sum[name] / float(x_count[name])
        for name in ("fit_real", "fit_imit", "keep")
    }
    all_fit_count = x_count["fit_real"] + x_count["fit_imit"]
    all_fit_sum = x_sum["fit_real"] + x_sum["fit_imit"]
    mean_all_fit = all_fit_sum / float(all_fit_count)
    up_row = ea.normalize_vector(means["fit_imit"] - means["keep"])
    gate_direction = ea.normalize_vector(means["fit_imit"] - means["fit_real"])
    m_direction = ea.normalize_vector(mean_all_fit)
    mean_fit_x_dot_m = float(np.dot(mean_all_fit, m_direction))
    if mean_fit_x_dot_m <= 0.0:
        raise RuntimeError(f"nonpositive mean_fit(x dot m): {mean_fit_x_dot_m}")
    fit_imit_positions = np.concatenate(
        [fit_imit_x[item_id] for item_id in sets["fit_imit"]], axis=0
    )
    fit_imit_g_projection = fit_imit_positions @ gate_direction
    tau_values = np.quantile(fit_imit_g_projection, ea.QUANTILE_LEVELS)

    # Least-used layer-20 neuron over ESC discovery clips, exactly the act_mean
    # slice registered in Amendment 5.
    discovery = (ref["kind"] == "esc") & (ref["fold"] <= 3)
    layer20_acts = ref["act_mean"][discovery, LAYER0 * INTERMEDIATE:(LAYER0 + 1) * INTERMEDIATE]
    mean_abs = np.mean(np.abs(layer20_acts.astype(np.float32)), axis=0)
    neuron_j = int(np.argmin(mean_abs))
    mlp = layers[LAYER0].mlp
    if tuple(mlp.up_proj.weight.shape) != (INTERMEDIATE, HIDDEN):
        raise RuntimeError(f"unexpected up_proj shape {tuple(mlp.up_proj.weight.shape)}")
    if tuple(mlp.gate_proj.weight.shape) != (INTERMEDIATE, HIDDEN):
        raise RuntimeError(f"unexpected gate_proj shape {tuple(mlp.gate_proj.weight.shape)}")
    if tuple(mlp.down_proj.weight.shape) != (HIDDEN, INTERMEDIATE):
        raise RuntimeError(f"unexpected down_proj shape {tuple(mlp.down_proj.weight.shape)}")
    if len(d) != HIDDEN:
        raise RuntimeError(f"inside direction dimension {len(d)} != hidden size {HIDDEN}")

    stats = {
        "fit_real_positions": int(x_count["fit_real"]),
        "fit_imitation_positions": int(x_count["fit_imit"]),
        "keep_positions": int(x_count["keep"]),
        "w2_covariance_positions": int(covariance_count),
        "w2_covariance_subsampled": False,
        "w1_neuron_j": neuron_j,
        "w1_mean_abs_activation": float(mean_abs[neuron_j]),
        "mean_fit_x_dot_m": mean_fit_x_dot_m,
        "w1_tau_quantiles": [
            {"quantile": float(q), "tau": float(tau)}
            for q, tau in zip(ea.QUANTILE_LEVELS, tau_values)
        ],
    }
    log(f"position statistics {stats}")

    def score(embeddings):
        return ea.prompt_scores(embeddings, prompt_emb, hen_idx, rooster_idx, other_idx)

    other_item_ids = list(sets["other_imit"])
    other_animals = [clips[item_id]["label"] for item_id in other_item_ids]
    other_after_by_arm = {}

    def cosine_median(edited, original):
        return float(np.median(np.sum(edited * original, axis=1)))

    base_fit_real = np.stack([clips[i]["embedding"] for i in sets["fit_real"]])
    base_fit_imit = np.stack([clips[i]["embedding"] for i in sets["fit_imit"]])
    original_up_row = mlp.up_proj.weight[neuron_j].detach().clone()
    original_gate_row = mlp.gate_proj.weight[neuron_j].detach().clone()
    original_down_col = mlp.down_proj.weight[:, neuron_j].detach().clone()
    original_down = mlp.down_proj.weight.detach().clone()
    saved_edits = {}
    near_ties = {}

    @contextmanager
    def neuron_patch(gate_row, scale, activation_median, capture_edits=False):
        down_col = -d * (float(scale) * R / float(activation_median))
        try:
            with torch.no_grad():
                mlp.up_proj.weight[neuron_j].copy_(torch.from_numpy(up_row))
                mlp.gate_proj.weight[neuron_j].copy_(torch.from_numpy(gate_row))
                mlp.down_proj.weight[:, neuron_j].copy_(torch.from_numpy(down_col))
            if capture_edits:
                patched_up = mlp.up_proj.weight[neuron_j].detach().clone()
                patched_gate = mlp.gate_proj.weight[neuron_j].detach().clone()
                patched_down = mlp.down_proj.weight[:, neuron_j].detach().clone()
                saved_edits.update({
                    "w1_neuron_j": np.asarray(neuron_j, dtype=np.int64),
                    "w1_up_row": patched_up.cpu().numpy().copy(),
                    "w1_gate_row": patched_gate.cpu().numpy().copy(),
                    "w1_down_col": patched_down.cpu().numpy().copy(),
                    "w1_up_delta": (patched_up - original_up_row).cpu().numpy().copy(),
                    "w1_gate_delta": (patched_gate - original_gate_row).cpu().numpy().copy(),
                    "w1_down_delta": (patched_down - original_down_col).cpu().numpy().copy(),
                })
            yield
        finally:
            with torch.no_grad():
                mlp.up_proj.weight[neuron_j].copy_(original_up_row)
                mlp.gate_proj.weight[neuron_j].copy_(original_gate_row)
                mlp.down_proj.weight[:, neuron_j].copy_(original_down_col)

    w1_grid = []
    w1_best = None
    w1_start = time.time()
    for tau_index, tau in enumerate(tau_values):
        gate_row = np.asarray(
            gate_direction - float(tau) * m_direction / mean_fit_x_dot_m,
            dtype=np.float32,
        )
        x_t = torch.from_numpy(fit_imit_positions)
        with torch.inference_mode():
            gate_values = torch.nn.functional.linear(
                x_t, torch.from_numpy(gate_row).reshape(1, -1)
            ).reshape(-1)
            up_values = torch.nn.functional.linear(
                x_t, torch.from_numpy(up_row).reshape(1, -1)
            ).reshape(-1)
            response = (mlp.act_fn(gate_values) * up_values).numpy()
        activation_median = float(np.median(response))
        if activation_median <= 0.0:
            for s in GENTLE_GRID:
                row = {
                    "direction": "W1",
                    "tau_quantile": float(ea.QUANTILE_LEVELS[tau_index]),
                    "tau": float(tau),
                    "s": float(s),
                    "a_med": activation_median,
                    "fit_auc": None,
                    "fit_median_cos": None,
                    "eligible": False,
                    "skipped": "a_med <= 0",
                }
                w1_grid.append(row)
            continue
        for s in GENTLE_GRID:
            with neuron_patch(gate_row, s, activation_median):
                edited_real = tail.embed_ids(sets["fit_real"], cache_by_id)
                edited_imit = tail.embed_ids(sets["fit_imit"], cache_by_id)
            auc = ea.auc_values(score(edited_real), score(edited_imit))
            median_cos = cosine_median(edited_imit, base_fit_imit)
            eligible = median_cos >= MIN_COS
            row = {
                "direction": "W1",
                "tau_quantile": float(ea.QUANTILE_LEVELS[tau_index]),
                "tau": float(tau),
                "s": float(s),
                "a_med": activation_median,
                "fit_auc": float(auc),
                "fit_median_cos": float(median_cos),
                "eligible": bool(eligible),
            }
            w1_grid.append(row)
            if eligible:
                candidate = (float(auc), -float(s), -float(tau))
                if w1_best is None or candidate > w1_best["rank"]:
                    w1_best = {
                        "rank": candidate,
                        "tau": float(tau),
                        "tau_quantile": float(ea.QUANTILE_LEVELS[tau_index]),
                        "s": float(s),
                        "a_med": activation_median,
                        "fit_auc": float(auc),
                        "fit_median_cos": float(median_cos),
                        "gate_row": gate_row.copy(),
                    }
        log(
            f"W1 grid tau {tau_index + 1}/{len(tau_values)} tau={float(tau):.7g} "
            f"a_med={activation_median:.7g}; {len(w1_grid)}/{len(tau_values) * len(GENTLE_GRID)} "
            f"rows elapsed={time.time() - w1_start:.0f}s"
        )
    if w1_best is None:
        raise RuntimeError("W1 has no setting satisfying the fit-imitation cosine constraint")
    w1_best.pop("rank")
    published_w1 = next((row for row in w1_grid
                         if row.get("tau_quantile") is not None
                         and abs(float(row["tau_quantile"]) - 0.3) < 1e-12
                         and row["s"] == 0.025
                         and row.get("fit_auc") is not None and row.get("eligible")), None)
    if (neuron_j == 1083 and published_w1 is not None
            and (abs(float(w1_best["tau_quantile"]) - 0.3) >= 1e-12 or w1_best["s"] != 0.025)
            and w1_best["fit_auc"] - published_w1["fit_auc"] <= 0.0025):
        near_ties["W1"] = {
            "selected": {"tau_quantile": w1_best["tau_quantile"], "s": w1_best["s"],
                         "fit_auc": w1_best["fit_auc"]},
            "used_published": {"tau_quantile": 0.3, "s": 0.025,
                               "fit_auc": published_w1["fit_auc"]},
        }
        w1_best.update({
            "tau": float(published_w1["tau"]),
            "tau_quantile": 0.3,
            "s": 0.025,
            "a_med": float(published_w1["a_med"]),
            "fit_auc": float(published_w1["fit_auc"]),
            "fit_median_cos": float(published_w1["fit_median_cos"]),
            "gate_row": np.asarray(
                gate_direction - float(published_w1["tau"]) * m_direction / mean_fit_x_dot_m,
                dtype=np.float32,
            ),
        })
    log(
        f"W1 selected j={neuron_j} tau={w1_best['tau']:.8g} "
        f"tau_q={w1_best['tau_quantile']:.1f} s={w1_best['s']} "
        f"fit_auc={w1_best['fit_auc']:.6f} fit_median_cos={w1_best['fit_median_cos']:.6f}"
    )

    covariance = covariance_sum / float(covariance_count)
    covariance = (covariance + covariance.T) * 0.5
    covariance_mean_diag = float(np.mean(np.diag(covariance)))
    covariance[np.diag_indices_from(covariance)] += 1e-4 * covariance_mean_diag
    fit_real_imit_ids = sets["fit_real"] + sets["fit_imit"]
    fit_label_rng = np.random.default_rng(RANDOM_SEED)
    shuffled = fit_label_rng.permutation(len(fit_real_imit_ids))
    shuffle_imit_ids = [fit_real_imit_ids[i] for i in shuffled[:len(sets["fit_imit"])]]
    key_sum_actual = sum(
        (fit_key_sums[item_id][0] for item_id in sets["fit_imit"]),
        start=np.zeros(INTERMEDIATE, dtype=np.float64),
    )
    key_count_actual = sum(fit_key_sums[item_id][1] for item_id in sets["fit_imit"])
    key_sum_control = sum(
        (fit_key_sums[item_id][0] for item_id in shuffle_imit_ids),
        start=np.zeros(INTERMEDIATE, dtype=np.float64),
    )
    key_count_control = sum(fit_key_sums[item_id][1] for item_id in shuffle_imit_ids)
    key_star_actual = (key_sum_actual / float(key_count_actual)).astype(np.float64)
    key_star_control = (key_sum_control / float(key_count_control)).astype(np.float64)
    solved = np.linalg.solve(covariance, np.stack([key_star_actual, key_star_control], axis=1))
    denominator_actual = float(np.dot(key_star_actual, solved[:, 0]))
    denominator_control = float(np.dot(key_star_control, solved[:, 1]))
    if denominator_actual <= 0.0 or denominator_control <= 0.0:
        raise RuntimeError(
            f"nonpositive W2 denominator: {denominator_actual}, {denominator_control}"
        )
    w2_vector_actual = solved[:, 0]
    w2_vector_control = solved[:, 1]

    def make_w2_delta(scale, solved_vector, key_star, denominator):
        return np.outer(-float(scale) * R * d, solved_vector) / denominator

    def down_patch(delta, capture_key=None):
        @contextmanager
        def patch():
            try:
                with torch.no_grad():
                    mlp.down_proj.weight.copy_(original_down + torch.from_numpy(
                        np.asarray(delta, dtype=np.float32)
                    ))
                if capture_key is not None:
                    patched_down = mlp.down_proj.weight.detach().clone()
                    saved_edits[capture_key] = (
                        patched_down - original_down
                    ).cpu().numpy().copy()
                    saved_edits[capture_key.replace("_delta", "_patched")] = (
                        patched_down.cpu().numpy().copy()
                    )
                yield
            finally:
                with torch.no_grad():
                    mlp.down_proj.weight.copy_(original_down)
        return patch()

    def fit_w2(label, solved_vector, key_star, denominator):
        grid = []
        best = None
        for index, s in enumerate(GENTLE_GRID, 1):
            delta = make_w2_delta(s, solved_vector, key_star, denominator)
            with down_patch(delta):
                edited_real = tail.embed_ids(sets["fit_real"], cache_by_id)
                edited_imit = tail.embed_ids(sets["fit_imit"], cache_by_id)
            auc = ea.auc_values(score(edited_real), score(edited_imit))
            median_cos = cosine_median(edited_imit, base_fit_imit)
            eligible = median_cos >= MIN_COS
            row = {
                "direction": label,
                "s": float(s),
                "fit_auc": float(auc),
                "fit_median_cos": float(median_cos),
                "eligible": bool(eligible),
            }
            grid.append(row)
            if eligible:
                candidate = (float(auc), -float(s))
                if best is None or candidate > best["rank"]:
                    best = {
                        "rank": candidate,
                        "s": float(s),
                        "fit_auc": float(auc),
                        "fit_median_cos": float(median_cos),
                    }
            log(
                f"{label} grid {index}/{len(GENTLE_GRID)} s={float(s):.3f} "
                f"fit_auc={auc:.6f} fit_median_cos={median_cos:.6f}"
            )
        if best is None:
            raise RuntimeError(f"{label} has no setting satisfying the fit-imitation cosine constraint")
        best.pop("rank")
        return best, grid

    w2_best, w2_grid = fit_w2("W2", w2_vector_actual, key_star_actual, denominator_actual)
    w2_control_best, w2_control_grid = fit_w2(
        "W2_control", w2_vector_control, key_star_control, denominator_control
    )

    def use_published_if_near(best, grid, published_s, label):
        published = next((row for row in grid if row["s"] == published_s), None)
        if (published is None or best["s"] == published_s or published["fit_auc"] is None
                or not published.get("eligible")):
            return best
        if best["fit_auc"] - published["fit_auc"] <= 0.0025:
            near_ties[label] = {
                "selected": {"s": best["s"], "fit_auc": best["fit_auc"]},
                "used_published": {"s": published_s, "fit_auc": published["fit_auc"]},
            }
            return {"s": published_s, "fit_auc": published["fit_auc"],
                    "fit_median_cos": published["fit_median_cos"]}
        return best

    w2_best = use_published_if_near(w2_best, w2_grid, 0.15, "W2")
    w2_control_best = use_published_if_near(w2_control_best, w2_control_grid, 0.025, "W2_control")
    log(
        f"W2 selected s={w2_best['s']} fit_auc={w2_best['fit_auc']:.6f}; "
        f"W2-control s={w2_control_best['s']} fit_auc={w2_control_best['fit_auc']:.6f}; "
        f"RSS {rss_gib():.2f} GiB"
    )

    base_test_real = np.stack([clips[i]["embedding"] for i in sets["test_real"]])
    base_test_imit = np.stack([clips[i]["embedding"] for i in sets["test_imit"]])
    base_esc_test = np.stack([clips[i]["embedding"] for i in sets["esc_test"]])
    base_other_imit = np.stack([clips[i]["embedding"] for i in sets["other_imit"]])
    other_scores_before = score(base_other_imit) if args.out_other else None
    labels_esc = np.asarray([clips[i]["label"] for i in sets["esc_test"]])
    nonchicken = ~np.isin(labels_esc, ["hen", "rooster"])
    if int(nonchicken.sum()) != 192:
        raise RuntimeError(f"expected 192 non-chicken ESC test clips, got {int(nonchicken.sum())}")
    bootstrap = ea.make_bootstrap_indices(len(sets["test_real"]), len(sets["test_imit"]))

    def run_arm(name, patch_context):
        with patch_context:
            edited_real = tail.embed_ids(sets["test_real"], cache_by_id)
            edited_imit = tail.embed_ids(sets["test_imit"], cache_by_id)
            edited_esc = tail.embed_ids(sets["esc_test"], cache_by_id)
            edited_other = tail.embed_ids(sets["other_imit"], cache_by_id)
        real_scores = score(edited_real)
        imitation_scores = score(edited_imit)
        auc = ea.auc_values(real_scores, imitation_scores)
        tops = ea.prompt_predictions(edited_imit, prompt_emb, categories)
        cosine = np.sum(edited_imit * base_test_imit, axis=1)
        damage = ea.class_damage(base_esc_test, edited_esc, labels_esc, prompt_emb, categories)
        nonchicken_cosine = np.sum(
            edited_esc[nonchicken] * base_esc_test[nonchicken], axis=1
        )
        other_before = ea.prompt_predictions(base_other_imit, prompt_emb, categories)
        other_after = ea.prompt_predictions(edited_other, prompt_emb, categories)
        changed_other = int(np.sum(other_before != other_after))
        if args.out_other:
            other_after_by_arm[name] = {
                "top_classes": other_after,
                "chicken_prompt_scores_after": score(edited_other),
            }
        result = {
            "fit": None,
            "test_auc": float(auc),
            "test_auc_bootstrap_95": ea.bootstrap_ci(real_scores, imitation_scores, bootstrap),
            "test_imitation_count": int(len(edited_imit)),
            "test_imitation_median_cos_to_unedited": float(np.median(cosine)),
            "test_imitation_top_class_counts": {
                str(category): int(count)
                for category, count in Counter(map(str, tops)).most_common()
            },
            "esc_fold4_5_class_damage": damage,
            "nonchicken_cosine_ge_0_999": {
                "count": int(np.sum(nonchicken_cosine >= 0.999)),
                "total": int(len(nonchicken_cosine)),
                "share": float(np.mean(nonchicken_cosine >= 0.999)),
            },
            "other_imitation_top_class_changed": {
                "count": changed_other,
                "total": int(len(other_before)),
                "share": float(changed_other / len(other_before)),
            },
        }
        log(
            f"{name} scored: auc={auc:.6f} bootstrap={result['test_auc_bootstrap_95']} "
            f"median_cos={result['test_imitation_median_cos_to_unedited']:.6f} "
            f"top={result['test_imitation_top_class_counts']} "
            f"nonchicken_cos_ge_0.999={result['nonchicken_cosine_ge_0_999']} "
            f"other_changed={result['other_imitation_top_class_changed']}"
        )
        return result

    w1_patch = neuron_patch(
        w1_best["gate_row"], w1_best["s"], w1_best["a_med"], capture_edits=True
    )
    w1_result = run_arm("W1", w1_patch)
    w1_result["fit"] = {
        key: value for key, value in w1_best.items() if key != "gate_row"
    }

    selected_w2_delta = make_w2_delta(
        w2_best["s"], w2_vector_actual, key_star_actual, denominator_actual
    )
    w2_result = run_arm("W2", down_patch(selected_w2_delta, capture_key="w2_delta"))
    w2_result["fit"] = dict(w2_best)

    w2_control_delta = make_w2_delta(
        w2_control_best["s"], w2_vector_control, key_star_control, denominator_control
    )
    w2_control_result = run_arm(
        "W2_control", down_patch(w2_control_delta, capture_key="w2c_delta")
    )
    w2_control_result["fit"] = dict(w2_control_best)

    # The gentle hook writes after layer 19, which is the cached input to layer
    # 20. Use the registered gate and the s already selected by gentle_gate.py.
    u17, u19 = 17 * INTERMEDIATE + 1574, 19 * INTERMEDIATE + 1129
    other_vis_indices = [
        i for i, row in enumerate(manifest)
        if row["group"] == "other" and row["kind"] == "imitation"
    ]
    other_gate = []
    for index in other_vis_indices:
        act = vis["act_mean"][index].astype(np.float32)
        a17 = s17 * float(act[u17])
        a19 = s19 * float(act[u19])
        other_gate.append(
            max(a19 - float(thresholds["t19"]), 0.0)
            * max(float(thresholds["t17"]) - a17, 0.0)
        )
    gentle_open_ids = [
        item_id for item_id, gate_value in zip(sets["other_imit"], other_gate)
        if gate_value > 0.0
    ]
    gentle_pushes = {
        item_id: -gentle_s * R * d for item_id in gentle_open_ids
    }
    gentle_other = base_other_imit.copy()
    if gentle_open_ids:
        gentle_open_embeddings = tail.embed_ids(
            gentle_open_ids, cache_by_id, gentle_pushes
        )
        open_positions = {item_id: i for i, item_id in enumerate(sets["other_imit"])}
        for item_id, embedding in zip(gentle_open_ids, gentle_open_embeddings):
            gentle_other[open_positions[item_id]] = embedding
    gentle_before = ea.prompt_predictions(base_other_imit, prompt_emb, categories)
    gentle_after = ea.prompt_predictions(gentle_other, prompt_emb, categories)
    gentle_changed = int(np.sum(gentle_before != gentle_after))
    gentle_other_result = {
        "s": gentle_s,
        "gated_other_imitation_count": int(len(gentle_open_ids)),
        "other_imitation_top_class_changed": {
            "count": gentle_changed,
            "total": int(len(gentle_before)),
            "share": float(gentle_changed / len(gentle_before)),
        },
    }
    log(f"gentle hook other-imitation description {gentle_other_result}")

    if args.out_other:
        gentle_gate_open = [gate_value > 0.0 for gate_value in other_gate]
        other_after_by_arm["gentle_hook"] = {
            "top_classes": gentle_after,
            "chicken_prompt_scores_after": score(gentle_other),
        }
        other_report = write_other_items_report(
            args.out_other,
            other_item_ids,
            other_animals,
            gentle_before,
            {
                arm: observations["top_classes"]
                for arm, observations in other_after_by_arm.items()
            },
            chicken_prompt_scores={
                arm: (other_scores_before, observations["chicken_prompt_scores_after"])
                for arm, observations in other_after_by_arm.items()
            },
            gate_open={"gentle_hook": gentle_gate_open},
        )
        expected_changed = {
            "W1": w1_result["other_imitation_top_class_changed"]["count"],
            "W2": w2_result["other_imitation_top_class_changed"]["count"],
            "W2_control": w2_control_result["other_imitation_top_class_changed"]["count"],
            "gentle_hook": gentle_other_result["other_imitation_top_class_changed"]["count"],
        }
        if other_report["changed_count_by_arm"] != expected_changed:
            raise RuntimeError(
                "per-clip other-imitation counts differ from the existing changed counts: "
                f"{other_report['changed_count_by_arm']} != {expected_changed}"
            )

    originals_restored = (
        torch.equal(mlp.up_proj.weight[neuron_j], original_up_row)
        and torch.equal(mlp.gate_proj.weight[neuron_j], original_gate_row)
        and torch.equal(mlp.down_proj.weight[:, neuron_j], original_down_col)
        and torch.equal(mlp.down_proj.weight, original_down)
    )
    if not originals_restored:
        raise RuntimeError("one or more layer-20 weights were not restored exactly")

    final_parity_embeddings = tail.embed_ids(parity_ids, parity_caches)
    final_parity = [
        float(np.dot(final_parity_embeddings[index], clips[item_id]["embedding"]))
        for index, item_id in enumerate(parity_ids)
    ]
    parity["post_restore_clips"] = len(final_parity)
    parity["post_restore_worst_tail_cosine_to_stored"] = min(final_parity)
    parity["weights_restored_bitwise"] = bool(originals_restored)
    if min(final_parity) < 0.99999:
        raise RuntimeError(f"post-restoration parity failed: {parity}")
    log(f"weight restoration and final parity verified: {parity}")

    w1_share = w1_result["nonchicken_cosine_ge_0_999"]["share"]
    w2_share = w2_result["nonchicken_cosine_ge_0_999"]["share"]
    verdict_text = {
        "X1": "W1 reaches a held-out AUC of at least 0.75.",
        "X2": "W2 reaches a held-out AUC of at least 0.75.",
        "X3": "W1 leaves at least 80% of ESC-50 fold 4 and 5 non-chicken clips within cosine 0.999.",
        "X4": "W2 leaves fewer of those clips within cosine 0.999 than W1 does.",
        "X5": "W2's shuffled-label control has a lower held-out AUC than W2.",
    }
    verdicts = {
        "X1": {
            "prediction": verdict_text["X1"],
            "held": bool(w1_result["test_auc"] >= 0.75),
        },
        "X2": {
            "prediction": verdict_text["X2"],
            "held": bool(w2_result["test_auc"] >= 0.75),
        },
        "X3": {
            "prediction": verdict_text["X3"],
            "held": bool(w1_share >= 0.8),
        },
        "X4": {
            "prediction": verdict_text["X4"],
            "held": bool(w2_share < w1_share),
        },
        "X5": {
            "prediction": verdict_text["X5"],
            "held": bool(w2_control_result["test_auc"] < w2_result["test_auc"]),
        },
    }
    covariance_summary = {
        "positions": int(covariance_count),
        "subsampled": False,
        "mean_diagonal_before_ridge": covariance_mean_diag,
        "ridge_fraction_of_mean_diagonal": 1e-4,
        "actual_key_star_norm": float(np.linalg.norm(key_star_actual)),
        "control_key_star_norm": float(np.linalg.norm(key_star_control)),
        "actual_denominator": denominator_actual,
        "control_denominator": denominator_control,
        "control_shuffled_fit_imitation_clip_ids": shuffle_imit_ids,
        "control_seed": RANDOM_SEED,
    }
    out = {
        "task": "EDIT-DESIGN.md Amendment 5 direct weight edits",
        "sets": {name: len(ids) for name, ids in sets.items()},
        "fit_imitation_participants": fit_people,
        "parity": parity,
        "position_statistics": stats,
        "w2_covariance": covariance_summary,
        "baseline_test_auc": float(
            ea.auc_values(score(base_test_real), score(base_test_imit))
        ),
        "fit_selection": {
            "W1": {"selected": w1_result["fit"], "grid": w1_grid,
                   "near_tie": near_ties.get("W1")},
            "W2": {"selected": w2_best, "grid": w2_grid},
            "W2_control": {"selected": w2_control_best, "grid": w2_control_grid},
        },
        "near_tie_fallbacks": near_ties,
        "arms": {
            "W1": w1_result,
            "W2": w2_result,
            "W2_control": w2_control_result,
        },
        "other_imitation_top_class_changes": {
            "W1": w1_result["other_imitation_top_class_changed"],
            "W2": w2_result["other_imitation_top_class_changed"],
            "gentle_hook": gentle_other_result,
        },
        "verdicts": verdicts,
        "seconds": round(time.time() - started, 1),
        "peak_rss_gib": rss_gib(),
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(out, indent=1) + "\n")
    if args.save_edits:
        missing = {"w1_neuron_j", "w1_up_row", "w1_gate_row", "w1_down_col",
                   "w1_up_delta", "w1_gate_delta", "w1_down_delta", "w2_delta",
                   "w2_patched", "w2c_delta", "w2c_patched"} - set(saved_edits)
        if missing:
            raise RuntimeError(f"cannot save edits; live patch captures missing: {sorted(missing)}")
        settings = {
            "W1": w1_result["fit"],
            "W2": w2_best,
            "W2_control": w2_control_best,
            "near_tie_fallbacks": near_ties,
        }
        digest = hashlib.sha256()
        with args.base_safetensors.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        saved_edits["settings_json"] = np.asarray(json.dumps(settings, sort_keys=True))
        saved_edits["base_sha256"] = np.asarray(digest.hexdigest())
        args.save_edits.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_edits, **saved_edits)
        log(f"saved in-place patch deltas to {args.save_edits}; base sha256={digest.hexdigest()}")
    log(f"VERDICTS {json.dumps(verdicts)}")
    log(f"DONE seconds={out['seconds']} peak_rss_gib={out['peak_rss_gib']:.2f}")


if __name__ == "__main__":
    main()
