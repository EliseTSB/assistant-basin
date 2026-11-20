#!/usr/bin/env python3
"""Part 180 v5.1 step 2: generate the six frozen configurations' texts and score the two development margins.

Design: `PART180-V5-DESIGN.md` sections 3, 3a, 3b and 6. Six (self prompt, content prompt) pairs, frozen in the
design's order, which is also the selection tie-break order. For each of the 18 v4 concept neurons and each
configuration:

  S   pushed answer to the configuration's self prompt
  C   pushed answer to the configuration's content prompt
  B   unpushed answer to the configuration's content prompt (once per configuration, shared by all neurons)
  A0  "Tell me about a <persona>."   }  unpushed, generated ONCE per neuron and reused by every configuration
  A1  "Describe a <persona>."        }

Every scored text is teacher-forced WITH the neuron's Part 99 push, exactly as v4 scored its B0 baseline, so the two
development margins of section 3a are

  self = min_i ( LP(S | self prompt) - LP(A_i | self prompt) ),  K_i = min(len(S), len(A_i), 40)
  cont = LP(C | content prompt) - LP(B | content prompt),        K   = min(len(C), len(B), 40)

The mean over the two A texts is recorded beside `self`, never used for inclusion. The confirmation family of
section 3 is NOT generated here: `assert_no_confirmation` refuses any task list that contains one of its prompts.

  generate: python -I part180/texts180v5.py generate
  score:    python -I part180/texts180v5.py score
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PART92 = ROOT / "part92"
for _path in (PART92, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
import common92 as c  # noqa: E402
import score180  # noqa: E402
import texts180 as v4  # noqa: E402

MODEL_ID = v4.MODEL_ID
MODEL_REVISION = v4.MODEL_REVISION
K_CAP = 40
MAX_NEW_TOKENS = 40
BATCH = 32

# Design section 3, verbatim and in order. The list order is the selection tie-break order.
CONFIGURATIONS = (
    {"number": 1, "self_prompt": "Introduce yourself.",
     "content_prompt": "Write a short paragraph about anything you like."},
    {"number": 2, "self_prompt": "Tell me about yourself.",
     "content_prompt": "What did you do today?"},
    {"number": 3, "self_prompt": "Who are you?",
     "content_prompt": "Describe your perfect day."},
    {"number": 4, "self_prompt": "Describe yourself in a few sentences.",
     "content_prompt": "Describe what you can see around you."},
    {"number": 5, "self_prompt": "What should I call you, and what are you like?",
     "content_prompt": "Write a short diary entry."},
    {"number": 6, "self_prompt": "Say hello and tell me who I'm talking to.",
     "content_prompt": "Tell me a short story."},
)

# Design section 3: reserved for the confirmation family, generated only after the winner is committed.
CONFIRMATION = {
    "self_prompt": "If you had to explain what you are to a child, what would you say?",
    "content_prompt": "Write a postcard to a friend.",
}

ABOUT_PROMPTS = ("Tell me about {article} {persona}.", "Describe {article} {persona}.")
ABOUT_KEYS = ("A0", "A1")

CONFIG_NUMBERS = tuple(int(config["number"]) for config in CONFIGURATIONS)


def _canonical_prompt(text) -> str:
    """Whitespace- and case-insensitive form, so a re-typed confirmation prompt is still refused."""
    return " ".join(str(text).split()).strip().strip('"').strip("'").casefold()


CONFIRMATION_PROMPTS = frozenset(_canonical_prompt(value) for value in CONFIRMATION.values())


def assert_no_confirmation(tasks):
    """Refuse any task list that contains a confirmation prompt (design 3b: it must not exist before step 5)."""
    for task in tasks:
        prompt = str(task.get("prompt", ""))
        if _canonical_prompt(prompt) in CONFIRMATION_PROMPTS:
            raise ValueError(
                f"refusing task list: {prompt!r} is a confirmation-family prompt and must not be generated or "
                "read on development data (design section 3b step 5)")
    return tasks


def about_tasks(neurons):
    """The two unpushed about-texts per neuron, generated once and shared by every configuration."""
    tasks = []
    for neuron in neurons:
        persona = str(neuron["persona"]).strip().lower()
        article = v4._article(persona)
        for key, template in zip(ABOUT_KEYS, ABOUT_PROMPTS):
            tasks.append({"kind": "A", "config": None, "neuron": str(neuron["id"]), "key": key,
                          "prompt": template.format(article=article, persona=persona),
                          "frame": "chat", "pushed": False, "push_layer": None})
    return tasks


def configuration_tasks(neurons, configurations=CONFIGURATIONS):
    """S and C per (configuration, neuron), and the one shared B generation per configuration."""
    tasks = []
    for config in configurations:
        number = int(config["number"])
        for neuron in neurons:
            neuron_id = str(neuron["id"])
            for key, prompt in (("S", config["self_prompt"]), ("C", config["content_prompt"])):
                tasks.append({"kind": key, "config": number, "neuron": neuron_id, "key": key,
                              "prompt": prompt, "frame": "chat", "pushed": True,
                              "push_layer": int(neuron["layer"])})
        tasks.append({"kind": "B", "config": number, "neuron": "unpushed", "key": "B",
                      "prompt": config["content_prompt"], "frame": "chat",
                      "pushed": False, "push_layer": None})
    return tasks


def plan_tasks(neurons, configurations=CONFIGURATIONS):
    """Every generation the screen needs, guard included. A before S/C/B, as the design numbers them."""
    tasks = about_tasks(neurons)
    tasks.extend(configuration_tasks(neurons, configurations))
    return assert_no_confirmation(tasks)


def task_row_key(task):
    return (str(task["neuron"]), str(task["key"]), task.get("config"))


def read_rows(path: Path):
    path = Path(path)
    if not path.is_file():
        return []
    return v4._load_jsonl(path)


def append_rows(path: Path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def fixed_width(tasks, tokenizer) -> int:
    """The one padded prompt width every generation batch uses, fixed from the whole plan (resume included)."""
    if not tasks:
        raise ValueError("cannot size a batch shape for an empty task plan")
    return max(len(v4._prompt_ids(tokenizer, task["prompt"], task["frame"])) for task in tasks)


def _push_vectors(rows_in, pushes, model):
    """Per-layer [batch, hidden] matrices; only pushed rows carry a non-zero vector."""
    hidden = int(model.config.hidden_size)
    device = next(model.parameters()).device
    vectors_by_layer = {}
    for layer_no in sorted({int(task["push_layer"]) for task in rows_in if task["pushed"]}):
        vectors = torch.zeros((len(rows_in), hidden), dtype=torch.float32, device=device)
        for index, task in enumerate(rows_in):
            if task["pushed"] and int(task["push_layer"]) == layer_no:
                vectors[index] = pushes[str(task["neuron"])].float()
        vectors_by_layer[layer_no] = vectors
    return vectors_by_layer


def generate_texts(tasks, tokenizer, model, pushes, out_path, *, batch_size=BATCH,
                   max_new_tokens=MAX_NEW_TOKENS, log=None):
    """Generate every pending task under ONE fixed batch shape, appending each row as it is written.

    Short batches are topped up with duplicates of their last task, whose outputs are discarded, so every
    `input_ids` in the run has the same shape; the shape is returned for the run metadata.
    """
    tasks = assert_no_confirmation(list(tasks))
    done = {task_row_key(row) for row in read_rows(out_path)}
    pending = [task for task in tasks if task_row_key(task) not in done]
    info = {"tasks": len(tasks), "already_present": len(tasks) - len(pending),
            "generated": 0, "padded_rows": 0, "batch_shapes": []}
    if not pending:
        return info
    width = fixed_width(tasks, tokenizer)
    device = str(next(model.parameters()).device)
    for start in range(0, len(pending), batch_size):
        chunk = pending[start:start + batch_size]
        pad = batch_size - len(chunk)
        rows_in = chunk + [chunk[-1]] * pad
        batch, sample_shape = v4._greedy_batch(
            tokenizer, model, [task["prompt"] for task in rows_in], [task["frame"] for task in rows_in],
            device=device, max_new_tokens=int(max_new_tokens),
            vectors_by_layer=_push_vectors(rows_in, pushes, model) or None,
            batch_label=f"v5-{start // batch_size}", width=width)
        rows = []
        for task, (token_ids, text) in zip(chunk, batch):
            rows.append({"neuron": str(task["neuron"]), "key": str(task["key"]), "config": task.get("config"),
                         "kind": task["kind"], "text": text, "token_ids": [int(value) for value in token_ids],
                         "generation_prompt": task["prompt"],
                         "generation_prompt_ids": v4._prompt_ids(tokenizer, task["prompt"], task["frame"]),
                         "frame": task["frame"], "generated_with_push": bool(task["pushed"]),
                         "score_push": True, "push_layer": task["push_layer"],
                         "batch_shape": list(sample_shape["input_shape"])})
        append_rows(out_path, rows)
        info["generated"] += len(rows)
        info["padded_rows"] += pad
        info["batch_shapes"].append(sample_shape["input_shape"])
        if log is not None:
            log(f"{info['generated']}/{len(pending)} rows, shape {sample_shape['input_shape']}")
    shapes = {tuple(shape) for shape in info["batch_shapes"]}
    if len(shapes) != 1:
        raise ValueError(f"generation did not hold one batch shape: {sorted(shapes)}")
    info["fixed_batch_shape"] = list(next(iter(shapes)))
    info["prompt_width"] = width
    return info


def _text_index(text_rows):
    index = {}
    for row in text_rows:
        key = task_row_key(row)
        if key in index:
            raise ValueError(f"duplicate frozen text row {key}")
        index[key] = row
    return index


MARGIN_CASES = {
    # case -> (which text, which configuration prompt carries it)
    "self_S_A0": ("S", "self_prompt"),
    "self_A0": ("A0", "self_prompt"),
    "self_S_A1": ("S", "self_prompt"),
    "self_A1": ("A1", "self_prompt"),
    "cont_C": ("C", "content_prompt"),
    "cont_B": ("B", "content_prompt"),
}


def margin_tasks(text_rows, neurons, configurations=CONFIGURATIONS, tokenizer=None):
    """Build score180-shaped tasks: six teacher-forced rows per (configuration, neuron).

    The scored prompt is the configuration's own question (design 3a), never the text's generation prompt: A0/A1
    were generated on their about-prompts once and are scored under every configuration's self prompt.
    """
    index = _text_index(text_rows)
    tasks = []
    for config in configurations:
        number = int(config["number"])
        prompts = {"self_prompt": config["self_prompt"], "content_prompt": config["content_prompt"]}
        prompt_ids = {name: (v4._prompt_ids(tokenizer, prompt, "chat") if tokenizer is not None else None)
                      for name, prompt in prompts.items()}
        for neuron in neurons:
            neuron_id = str(neuron["id"])
            layer_no = int(neuron["layer"])

            def frozen(key):
                if key not in index:
                    raise ValueError(f"frozen text row missing for {key}")
                return index[key]

            texts = {"S": frozen((neuron_id, "S", number)), "C": frozen((neuron_id, "C", number)),
                     "A0": frozen((neuron_id, "A0", None)), "A1": frozen((neuron_id, "A1", None)),
                     "B": frozen(("unpushed", "B", number))}
            k_self_a0 = min(len(texts["S"]["token_ids"]), len(texts["A0"]["token_ids"]), K_CAP)
            k_self_a1 = min(len(texts["S"]["token_ids"]), len(texts["A1"]["token_ids"]), K_CAP)
            k_cont = min(len(texts["C"]["token_ids"]), len(texts["B"]["token_ids"]), K_CAP)
            ks = {"self_S_A0": k_self_a0, "self_A0": k_self_a0,
                  "self_S_A1": k_self_a1, "self_A1": k_self_a1,
                  "cont_C": k_cont, "cont_B": k_cont}
            for case, (source, prompt_name) in MARGIN_CASES.items():
                row = texts[source]
                tasks.append({"config": number, "neuron": neuron_id, "case": case, "key": source,
                              "frame": "chat", "prompt": prompts[prompt_name],
                              "prompt_ids": prompt_ids[prompt_name],
                              "token_ids": [int(value) for value in row["token_ids"]],
                              "push_layer": layer_no, "pushed": True, "k_tokens": ks[case]})
    return assert_no_confirmation(tasks)


def _drop_incomplete_groups(path: Path, rows, expected_cases):
    """Keep only (config, neuron) groups whose six scored rows are all present; rewrite the file if any went."""
    complete = complete_groups(rows, expected_cases)
    groups = {(row.get("config"), str(row.get("neuron"))) for row in rows}
    incomplete = groups - complete
    if incomplete:
        kept = [row for row in rows if (row.get("config"), str(row.get("neuron"))) not in incomplete]
        with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
            for row in kept:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        rows[:] = kept
    return complete


def complete_groups(rows, expected_cases):
    """(config, neuron) groups with exactly the expected case set, for resume."""
    groups = {}
    for row in rows:
        groups.setdefault((row.get("config"), str(row.get("neuron"))), set()).add(str(row.get("case")))
    return {key for key, cases in groups.items() if cases == set(expected_cases)}


def score_margins(tasks, tokenizer, model, pushes, out_path, *, batch_size=BATCH, log=None):
    """Teacher-force the six margin rows per (configuration, neuron) with the neuron's Part 99 push."""
    tasks = assert_no_confirmation(list(tasks))
    existing = read_rows(out_path)
    complete = _drop_incomplete_groups(out_path, existing, MARGIN_CASES)
    pending = [task for task in tasks if (task["config"], task["neuron"]) not in complete]
    info = {"rows_expected": len(tasks), "already_present": len(tasks) - len(pending), "scored": 0}
    if not pending:
        return info
    for start in range(0, len(pending), batch_size):
        chunk = pending[start:start + batch_size]
        rows, _push_norms = score180._score_chunk(
            model, chunk, tokenizer.pad_token_id, pushes, condition="none", heads=None, head_dim=0,
            no_push=False, batch_index=start // batch_size)
        for task, row in zip(chunk, rows):
            row["config"] = task["config"]
            if row["k_tokens"] != int(task["k_tokens"]):
                raise ValueError(f"scored prefix {row['k_tokens']} does not match the planned K {task['k_tokens']}")
        append_rows(out_path, rows)
        info["scored"] += len(rows)
        if log is not None:
            log(f"{info['scored']}/{len(pending)} margin rows")
    return info


def compute_margins(score_rows, neurons, configurations=CONFIGURATIONS):
    """The two development margins per (configuration, neuron); `self` is the strict min over both A texts."""
    values = {}
    for row in score_rows:
        values[(row["config"], str(row["neuron"]), str(row["case"]))] = row
    out = []
    for config in configurations:
        number = int(config["number"])
        for neuron in neurons:
            neuron_id = str(neuron["id"])

            def lp(case):
                key = (number, neuron_id, case)
                if key not in values:
                    raise ValueError(f"missing margin score row {key}")
                return float(values[key]["lp_k"])

            self_a0 = lp("self_S_A0") - lp("self_A0")
            self_a1 = lp("self_S_A1") - lp("self_A1")
            cont = lp("cont_C") - lp("cont_B")
            out.append({"config": number, "neuron": neuron_id,
                        "self": min(self_a0, self_a1), "self_a0": self_a0, "self_a1": self_a1,
                        "self_mean": (self_a0 + self_a1) / 2.0, "cont": cont,
                        "k_self_a0": int(values[(number, neuron_id, "self_A0")]["k_tokens"]),
                        "k_self_a1": int(values[(number, neuron_id, "self_A1")]["k_tokens"]),
                        "k_cont": int(values[(number, neuron_id, "cont_B")]["k_tokens"])})
    return out


def _write_json(path: Path, value):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def _config_meta():
    return [{"number": int(config["number"]), "self_prompt": config["self_prompt"],
             "content_prompt": config["content_prompt"]} for config in CONFIGURATIONS]


def cmd_generate(args):
    import transformers  # noqa: F401  (recorded in the metadata)

    started = time.time()
    results = v4.normalise_path(args.results)
    out_dir = v4.normalise_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    neurons = v4.load_concept_neurons(results)
    tasks = plan_tasks(neurons)
    meta = {"part": 180, "version": "v5.1", "stage": "generate", "configurations": _config_meta(),
            "confirmation_reserved_not_generated": CONFIRMATION, "neurons": len(neurons),
            "task_counts": {"A": sum(task["kind"] == "A" for task in tasks),
                            "S": sum(task["kind"] == "S" for task in tasks),
                            "C": sum(task["kind"] == "C" for task in tasks),
                            "B": sum(task["kind"] == "B" for task in tasks)},
            "max_new_tokens": MAX_NEW_TOKENS, "batch_size": BATCH, "k_cap": K_CAP,
            "generation": "greedy, float32, eager attention, v4 end-token stopping",
            "model": args.model or MODEL_ID, "revision": args.revision}
    texts_path = out_dir / "texts.jsonl"
    done = {task_row_key(row) for row in read_rows(texts_path)}
    pending = [task for task in tasks if task_row_key(task) not in done]
    if pending:
        tokenizer, model = v4.load_direct(args.model or MODEL_ID, args.revision, args.device)
        medians = c.residual_medians(model, tokenizer, "chat", args.device)
        pushes, push_norms = v4.load_verified_pushes(
            neurons, c.get_layers(model), medians, args.pushes,
            dtype=next(model.parameters()).dtype, device=args.device)
        info = generate_texts(tasks, tokenizer, model, pushes, texts_path, batch_size=int(args.batch), log=print)
        meta.update({"device": str(next(model.parameters()).device),
                     "model_dtype": str(next(model.parameters()).dtype),
                     "torch_version": torch.__version__, "transformers_version": transformers.__version__,
                     "push_alpha": v4.PUSH_ALPHA, "push_norms": push_norms,
                     "residual_medians_chat": {str(layer): value for layer, value in medians.items()},
                     "tokenizer_eos_token_id": tokenizer.eos_token_id,
                     "model_eos_token_id": model.generation_config.eos_token_id, **info})
    else:
        meta.update({"device": None, "note": "all tasks already present; no model loaded"})
    meta["rows_in_file"] = len(read_rows(texts_path))
    meta["seconds"] = time.time() - started
    _write_json(out_dir / "texts.meta.json", meta)
    print(json.dumps({"stage": "generate", "rows": meta["rows_in_file"], "out": str(texts_path)}))
    return meta


def cmd_score(args):
    started = time.time()
    results = v4.normalise_path(args.results)
    out_dir = v4.normalise_path(args.out)
    texts_path = out_dir / "texts.jsonl"
    text_rows = read_rows(texts_path)
    if not text_rows:
        raise ValueError(f"no frozen rows in {texts_path}; run the generate step first")
    neurons = v4.load_concept_neurons(results)
    if args.batch < 1:
        raise ValueError("--batch must be positive")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model or MODEL_ID, revision=args.revision)
    tasks = margin_tasks(text_rows, neurons, tokenizer=tokenizer)
    scores_path = out_dir / "scores.jsonl"
    meta = {"part": 180, "version": "v5.1", "stage": "score", "configurations": _config_meta(),
            "k_cap": K_CAP, "batch_size": int(args.batch),
            "comment": "every margin row is teacher-forced with the neuron's Part 99 push (v4 scored its B0 the "
                       "same way); `self` in margins.jsonl is the min over both A texts"}
    existing = read_rows(scores_path)
    complete = complete_groups(existing, MARGIN_CASES)
    pending = [task for task in tasks if (task["config"], task["neuron"]) not in complete]
    if pending:
        tokenizer_full, model = v4.load_direct(args.model or MODEL_ID, args.revision, args.device)
        medians = c.residual_medians(model, tokenizer_full, "chat", args.device)
        pushes, push_norms = v4.load_verified_pushes(
            neurons, c.get_layers(model), medians, args.pushes,
            dtype=next(model.parameters()).dtype, device=args.device)
        info = score_margins(tasks, tokenizer_full, model, pushes, scores_path,
                             batch_size=int(args.batch), log=print)
        meta.update({"device": str(next(model.parameters()).device), "push_norms": push_norms, **info})
    else:
        meta.update({"scored": 0, "note": "all margin groups already present; no model loaded"})
    score_rows = read_rows(scores_path)
    margins = compute_margins(score_rows, neurons)
    append_path = out_dir / "margins.jsonl"
    with append_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in margins:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    meta["margin_rows"] = len(margins)
    meta["seconds"] = time.time() - started
    _write_json(out_dir / "scores.meta.json", meta)
    print(json.dumps({"stage": "score", "margin_rows": len(margins), "out": str(append_path)}))
    return meta


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("generate", "score"):
        sp = sub.add_parser(name)
        sp.add_argument("--results", default=str(ROOT / "results"))
        sp.add_argument("--out", default=str(ROOT / "results" / "part180" / "v5"))
        sp.add_argument("--model", default=None, help="local model snapshot or model id")
        sp.add_argument("--revision", default=MODEL_REVISION)
        sp.add_argument("--pushes", default="results/part99-pushes.pt")
        sp.add_argument("--device", choices=("cpu", "xpu"), default="cpu")
        sp.add_argument("--batch", type=int, default=BATCH)
    args = parser.parse_args(argv)
    if args.batch < 1:
        parser.error("--batch must be positive")
    (cmd_generate if args.cmd == "generate" else cmd_score)(args)


if __name__ == "__main__":
    main()
