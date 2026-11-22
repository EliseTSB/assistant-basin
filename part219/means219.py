#!/usr/bin/env python3
"""means219.py -- Part 219 Amendment 1's reference head means, written to ``results/part219/head-means.pt``.

PART219-DESIGN.md's Amendment 1, "Reference set": the reference set is neutral chat-framed text with no identity
question and no pushed answer, so the replacement mean carries no self-description. It is

  * the 24 ``general`` questions of screen A's ``questions.jsonl`` (passed with ``--questions``; the file's SHA-256
    must start ``1eabf3aba5824428``, which is asserted here rather than trusted), and
  * Part 99's questions 4 and 5 (the capital of France and the sea poem), taken from ``swap99.QUESTIONS`` so they
    cannot drift from the questions the runs ask.

Each question is put in Part 99's chat frame, answered by the unedited host with Part 99's greedy settings (batched,
left-padded, the same ``common92.generate_batch`` helper and its 120-token cap, no push), and the prompt-plus-answer
sequence is teacher-forced back through the same model with no hooks except a capture pre-hook on ``o_proj`` for
layers 0 to 3. Per head, in float32, the slices at each sequence's FIRST real token (``pos0``) and at its later real
tokens (``rest``) are summed and divided, ignoring pad positions.

The saved file is one dict:

    {"L0H10": {"mean_pos0": tensor[128] float32, "mean_rest": tensor[128] float32,
               "n_pos0": int, "n_rest": int}, ...,
     "meta": {"host": ..., "questions_sha256": ..., "prompts": [...26 texts...], "answers": [...],
              "n_new_tokens": 120, ...}}

for the four important heads (L0H10, L1H4, L2H6, L3H5) and the four R5 rig-check heads (L0H12, L1H11, L2H5, L3H9).
``swap99.install_ablation_hooks(..., ablate_mode="mean")`` reads it and casts the float32 means to the model dtype.

Usage (A770, from the repo root):

    python -I -B part219/means219.py --host <Instruct snapshot> \
        --questions Journal/notes/reports/attention-heads-screens-2026-11-18/a-olmo-stages/questions.jsonl \
        --out results/part219 --device xpu --batch 48

A heartbeat line is printed at least every eight seconds for as long as the reference set is being read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for _path in (str(REPO), str(REPO / "part92"), str(REPO / "part99"), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import common92 as c  # noqa: E402
import swap99  # noqa: E402

PART = 219
OUTPUT_NAME = "head-means.pt"
QUESTIONS_SHA256_PREFIX = "1eabf3aba5824428"   # PART219-DESIGN.md Amendment 1, "Reference set"
PART99_QUESTION_INDICES = (4, 5)               # Part 99's capital of France and sea poem (index 3 is left out)
CAPTURE_LAYERS = (0, 1, 2, 3)
IMPORTANT_HEADS = ("L0H10", "L1H4", "L2H6", "L3H5")
RIG_HEADS = ("L0H12", "L1H11", "L2H5", "L3H9")
MEAN_HEADS = IMPORTANT_HEADS + RIG_HEADS
FRAME = "chat"                                 # Part 99's chat frame
MAX_NEW_TOKENS = 120                           # Part 99's cap in swap99.run's generate_batch call
BATCH = 48                                     # Part 219's batch, every run
HEARTBEAT_SECONDS = 8.0


class Heartbeat:
    """A progress line at least every ``interval`` seconds, for a step that can take minutes on the A770."""

    def __init__(self, label: str, interval: float = HEARTBEAT_SECONDS):
        self.label = label
        self.interval = interval
        self.started = time.perf_counter()
        self.current = "starting"
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self.stopped.wait(self.interval):
            elapsed = time.perf_counter() - self.started
            print(f"[means219] {self.label}: {self.current} ({elapsed:.0f}s elapsed)", flush=True)

    def say(self, message: str) -> None:
        self.current = message

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stopped.set()
        self.thread.join(timeout=2)


# --- the reference set ------------------------------------------------------------------------------------------

def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_reference_questions(path) -> list[str]:
    """The 24 ``general`` questions of screen A's file, with the registered SHA-256 prefix asserted first."""
    path = Path(path).expanduser()
    if not path.exists():
        raise SystemExit(f"{path} does not exist: pass screen A's questions.jsonl")
    digest = sha256_file(path)
    if not digest.startswith(QUESTIONS_SHA256_PREFIX):
        raise SystemExit(f"{path} has SHA-256 {digest}, which does not start {QUESTIONS_SHA256_PREFIX}: this is not "
                         "the reference file PART219-DESIGN.md's Amendment 1 fixes")
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    general = [str(row["question"]) for row in rows if row.get("kind") == "general"]
    if len(general) != 24:
        raise SystemExit(f"{path} holds {len(general)} general questions, not the 24 the amendment fixes")
    return general


def reference_prompts(questions_path) -> list[str]:
    """The amendment's 26 questions: screen A's 24 general ones, then Part 99's questions 4 and 5."""
    prompts = load_reference_questions(questions_path)
    for index in PART99_QUESTION_INDICES:
        if not 0 <= index < len(swap99.QUESTIONS):
            raise SystemExit(f"Part 99 has no question {index}")
        prompts.append(swap99.QUESTIONS[index])
    if len(set(prompts)) != len(prompts):
        raise SystemExit("the reference set repeats a question; the amendment fixes 26 distinct prompts")
    return prompts


def trim_trailing_pads(token_ids, pad_id: int) -> list[int]:
    """Drop ``generate``'s post-EOS pad filler: those are not emitted tokens, so they are not teacher-forced."""
    ids = [int(token) for token in token_ids]
    while ids and ids[-1] == int(pad_id):
        ids.pop()
    return ids


def sequences_for(tokenizer, prompts, answers, pad_id: int) -> list[list[int]]:
    """Each reference sequence: Part 99's chat-framed prompt ids plus the greedy answer's ids."""
    sequences = []
    for prompt, answer in zip(prompts, answers):
        ids = c.prompt_ids(tokenizer, prompt, FRAME) + trim_trailing_pads(answer, pad_id)
        if not ids:
            raise ValueError(f"the prompt {prompt!r} rendered to no tokens")
        sequences.append(ids)
    return sequences


# --- generation and capture --------------------------------------------------------------------------------------

def generate_answers(model, tokenizer, prompts, *, device: str = "cpu", batch: int = BATCH,
                     max_new_tokens: int = MAX_NEW_TOKENS, heartbeat=None) -> tuple[list[str], list[list[int]]]:
    """Part 99's greedy decode for every prompt: the same helper, frame, left padding and cap the runs use."""
    answers, ids = [], []
    for begin in range(0, len(prompts), max(int(batch), 1)):
        chunk = prompts[begin:begin + max(int(batch), 1)]
        chunk_answers, chunk_ids = c.generate_batch(model, tokenizer, chunk, FRAME, None,
                                                    max_new_tokens, device, return_ids=True)
        answers.extend(chunk_answers)
        ids.extend(chunk_ids)
        if heartbeat is not None:
            heartbeat.say(f"{len(answers)}/{len(prompts)} answers generated")
        print(f"[means219] generated {len(answers)}/{len(prompts)} answers", flush=True)
    return answers, ids


def pad_sequences(sequences, pad_id: int, device: str = "cpu"):
    """Left-pad exactly as ``common92.batch_inputs`` does, for sequences that are already token ids."""
    import torch

    if not sequences:
        raise ValueError("no sequences to pad")
    if any(not sequence for sequence in sequences):
        raise ValueError("a sequence has no tokens")
    width = max(len(sequence) for sequence in sequences)
    input_ids = torch.tensor([[int(pad_id)] * (width - len(sequence)) + list(sequence) for sequence in sequences],
                             dtype=torch.long, device=device)
    attention_mask = torch.tensor([[0] * (width - len(sequence)) + [1] * len(sequence) for sequence in sequences],
                                  dtype=torch.long, device=device)
    return input_ids, attention_mask, width


def positions_for(attention_mask):
    """``generate``'s RoPE positions for a left-padded batch: ``cumsum(mask) - 1``, pads pinned to 1.

    The model's own forward takes ``arange(sequence_length)`` when ``position_ids`` is absent, which is *not* the same
    as a left-padded row's real positions; ``generate`` passes mask-derived positions instead. The reference means must
    be read in the regime the runs decode in, so the plain teacher-forcing forward passes the same positions here.
    """
    import torch

    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    return position_ids


def head_sums(model, sequences, *, pad_id: int, device: str = "cpu", layers=CAPTURE_LAYERS, heads=MEAN_HEADS,
              batch: int = BATCH, heartbeat=None) -> dict:
    """Sum each head's ``o_proj`` input slice, in float32, over each sequence's first real token and later real ones.

    Pad positions are excluded by construction: the first real index of a left-padded row is ``width - len(row)``, and
    the ``rest`` sum runs over the attention mask's real positions with that one removed. One forward per batch, with
    only a capture pre-hook on the four layers' ``o_proj``: no ablation hook and no push.
    """
    import torch

    if not sequences:
        raise ValueError("no sequences to read")
    head_size, num_heads = swap99.head_geometry(model)
    module_layers = c.get_layers(model)
    wanted: dict[int, list[tuple[int, str]]] = {}
    for name in heads:
        match = swap99.HEAD_PATTERN.match(name)
        if match is None:
            raise ValueError(f"heads are written L<layer>H<head>; got {name!r}")
        layer_no, head = int(match.group(1)), int(match.group(2))
        if layer_no not in layers:
            raise ValueError(f"{name} is outside the capture layers {list(layers)}")
        if not 0 <= layer_no < len(module_layers):
            raise ValueError(f"{name} is outside the model's {len(module_layers)} layers")
        if not 0 <= head < num_heads:
            raise ValueError(f"{name} is outside layer {layer_no}'s {num_heads} heads")
        wanted.setdefault(layer_no, []).append((head, name))

    sums = {name: {"pos0": torch.zeros(head_size, dtype=torch.float32),
                   "rest": torch.zeros(head_size, dtype=torch.float32),
                   "n_pos0": 0, "n_rest": 0} for name in heads}
    step = max(int(batch), 1)
    done = 0
    for begin in range(0, len(sequences), step):
        chunk = [list(sequence) for sequence in sequences[begin:begin + step]]
        input_ids, attention_mask, width = pad_sequences(chunk, pad_id, device)
        first = [width - len(sequence) for sequence in chunk]
        captured: dict[int, object] = {}

        def make_hook(layer_no):
            def hook(_module, args):
                captured[layer_no] = args[0].detach()
            return hook

        handles = [module_layers[layer_no].self_attn.o_proj.register_forward_pre_hook(make_hook(layer_no))
                   for layer_no in sorted(wanted)]
        try:
            with torch.inference_mode():
                model(input_ids=input_ids, attention_mask=attention_mask,
                      position_ids=positions_for(attention_mask), use_cache=False)
        finally:
            for handle in handles:
                handle.remove()

        rows = torch.arange(len(chunk), device=input_ids.device)
        first_index = torch.tensor(first, device=input_ids.device)
        real = attention_mask.to(torch.bool).clone()
        real[rows, first_index] = False
        rest_positions = int(real.sum())
        for layer_no in sorted(wanted):
            hidden = captured.get(layer_no)
            if hidden is None:
                raise RuntimeError(f"layer {layer_no}'s o_proj was not called")
            if int(hidden.shape[-1]) != head_size * num_heads:
                raise RuntimeError(f"layer {layer_no}'s o_proj input is {tuple(hidden.shape)}, not "
                                   f"{head_size * num_heads} wide")
            for head, name in wanted[layer_no]:
                start, end = head * head_size, (head + 1) * head_size
                block = hidden[..., start:end].float()
                sums[name]["pos0"] += block[rows, first_index].sum(0)
                sums[name]["n_pos0"] += len(chunk)
                sums[name]["rest"] += block[real].sum(0)
                sums[name]["n_rest"] += rest_positions
        done += len(chunk)
        if heartbeat is not None:
            heartbeat.say(f"{done}/{len(sequences)} sequences teacher-forced")
        print(f"[means219] teacher-forced {done}/{len(sequences)} sequences", flush=True)
    return sums


def means_from_sums(sums: dict) -> dict:
    """Divide each head's float32 sums by its counts; both counts must be non-zero for a mean to mean anything."""
    import torch

    means = {}
    for name, value in sums.items():
        if not value["n_pos0"]:
            raise ValueError(f"{name} has no first-token positions")
        if not value["n_rest"]:
            raise ValueError(f"{name} has no later-token positions")
        means[name] = {
            "mean_pos0": (value["pos0"] / float(value["n_pos0"])).to(torch.float32),
            "mean_rest": (value["rest"] / float(value["n_rest"])).to(torch.float32),
            "n_pos0": int(value["n_pos0"]),
            "n_rest": int(value["n_rest"]),
        }
    return means


def compute_means(model, sequences, *, pad_id: int, **kwargs) -> dict:
    """The whole computation from tokenized sequences: float32 per-head means, split into first token and the rest."""
    return means_from_sums(head_sums(model, sequences, pad_id=pad_id, **kwargs))


# --- the whole file ----------------------------------------------------------------------------------------------

def build_meta(*, host, questions_path, questions_sha256: str, prompts, answers, n_new_tokens: int,
               attn_implementation, device: str, batch: int, torch_version: str, transformers_version: str,
               seconds: float, heads=MEAN_HEADS, layers=CAPTURE_LAYERS) -> dict:
    """What a reader of ``head-means.pt`` needs to reproduce it: the host, the questions, the prompts and answers."""
    prompts_sha256 = hashlib.sha256(json.dumps(list(prompts), ensure_ascii=False).encode("utf-8")).hexdigest()
    return {
        "part": PART,
        "amendment": "PART219-DESIGN.md Amendment 1 (2026-11-22), Reference set",
        "host": str(Path(host).expanduser()),
        "questions_path": str(Path(questions_path).expanduser()),
        "questions_sha256": questions_sha256,
        "prompts_sha256": prompts_sha256,
        "part99_question_indices": list(PART99_QUESTION_INDICES),
        "frame": FRAME,
        "heads": list(heads),
        "capture_layers": list(layers),
        "n_sequences": len(prompts),
        "n_new_tokens": int(n_new_tokens),
        "prompts": list(prompts),
        "answers": list(answers),
        "attn_implementation": attn_implementation,
        "device": device,
        "batch": int(batch),
        "dtype": "float32",
        "torch_version": torch_version,
        "transformers_version": transformers_version,
        "seconds": round(float(seconds), 1),
    }


def main(argv=None):
    import torch
    import transformers

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", required=True, help="local Instruct snapshot directory (unedited)")
    parser.add_argument("--questions", required=True,
                        help="screen A's questions.jsonl (SHA-256 must start 1eabf3aba5824428)")
    parser.add_argument("--out", default=str(REPO / "results" / "part219"))
    parser.add_argument("--device", choices=("cpu", "xpu", "cuda"), default="xpu")
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS,
                        help="Part 99's cap (swap99.run passes 120 to generate_batch)")
    parser.add_argument("--attn-implementation", choices=("default", "eager"), default="eager",
                        help="Part 219 loads every model under eager; 'default' is the library's kernel")
    parser.add_argument("--heads", default=",".join(MEAN_HEADS),
                        help="comma-separated heads to mean (the four important and the four rig-check heads)")
    args = parser.parse_args(argv)
    if args.batch < 1:
        parser.error("--batch must be positive")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    heads = [name.strip() for name in args.heads.split(",") if name.strip()]
    if not heads:
        parser.error("--heads must name at least one head")

    started = time.perf_counter()
    questions_path = Path(args.questions).expanduser()
    prompts = reference_prompts(questions_path)          # asserts the SHA-256 prefix
    questions_sha256 = sha256_file(questions_path)
    print(f"[means219] {len(prompts)} reference prompts; questions SHA-256 {questions_sha256}", flush=True)

    attn_implementation = None if args.attn_implementation == "default" else args.attn_implementation
    with Heartbeat("reference set") as heartbeat:
        heartbeat.say("loading the host")
        tokenizer, model = c.load_model(args.host, args.device, attn_implementation=attn_implementation)
        if tokenizer.pad_token_id is None:
            raise SystemExit("the host tokenizer has no pad token; Part 99's runs need one")
        pad_id = int(tokenizer.pad_token_id)
        heartbeat.say("generating the reference answers")
        answers, answer_ids = generate_answers(
            model, tokenizer, prompts, device=args.device, batch=args.batch,
            max_new_tokens=args.max_new_tokens, heartbeat=heartbeat)
        sequences = sequences_for(tokenizer, prompts, answers, pad_id)
        heartbeat.say("teacher-forcing the reference sequences")
        means = compute_means(model, sequences, pad_id=pad_id, device=args.device, heads=heads,
                              batch=args.batch, heartbeat=heartbeat)

    meta = build_meta(
        host=args.host, questions_path=questions_path, questions_sha256=questions_sha256,
        prompts=prompts, answers=answers, n_new_tokens=args.max_new_tokens,
        attn_implementation=attn_implementation, device=args.device, batch=args.batch,
        torch_version=torch.__version__, transformers_version=transformers.__version__,
        seconds=time.perf_counter() - started, heads=heads)
    payload = dict(means)
    payload["meta"] = meta
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    path = out / OUTPUT_NAME
    torch.save(payload, path)
    print(f"[means219] wrote {path} for {', '.join(means)}; SHA-256 {sha256_file(path)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
