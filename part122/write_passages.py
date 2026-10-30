#!/usr/bin/env python3
"""Write validated Part 122 matched passages, one DeepSeek request per unit."""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for candidate in (ROOT, ROOT / "part119", ROOT / "part92b", ROOT / "part92"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import rows119  # noqa: E402

SEED = 12201
MODEL_ID = "deepseek-v4.1-flash"
MAX_ATTEMPTS = 8
PAIR_COUNT = 8
MAX_TOKEN_DIFFERENCE = 2  # Gate 1, Amendment 1: the writer checks it with the real tokenizer
TOKENIZER = "/mnt/e/neuron-probe/hf/hub/models--allenai--OLMo-2-0425-1B-Instruct/snapshots/48d788eca847d4d7548f375ad03d3c9312f6139e"
PERSONA_FIELDS = ("third_person", "first_person")
_WORD = re.compile(r"[\w'-]+", re.UNICODE)
_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")
_FIRST_PERSON = re.compile(r"\b(?:i|me|my|mine|we|us|our|ours)\b", re.I)
_STOP_NAME_WORDS = {"of", "your", "my", "from", "in", "on", "for", "and", "a", "an", "the"}


def word_tokens(text: str) -> list[str]:
    return [token.casefold() for token in _WORD.findall(text)]


def _valid_passage(text: object, label: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{label} must be a nonempty string")
    clean = text.strip()
    if len(_SENTENCE_END.findall(re.sub(r"\b(Dr|Mr|Mrs|Ms|St)\.", r"\1", clean))) != 2:
        raise ValueError(f"{label} must contain exactly two sentences")
    return clean


def valid_pairs(payload: object, persona: str, count_tokens: Callable | None = None) -> tuple[dict, str | None]:
    """Each frame's pairs that pass shape, two-sentence form, length, name leakage and, when a counter is given,
    the token-count match. Returns them with the first failure seen."""
    if not isinstance(payload, dict) or set(payload) != set(PERSONA_FIELDS):
        raise ValueError("response must contain exactly third_person and first_person")
    persona_terms = [word for word in word_tokens(persona) if word not in _STOP_NAME_WORDS]
    if not persona_terms:
        persona_terms = word_tokens(persona)
    if not persona_terms:
        raise ValueError("persona name has no searchable words")

    result = {}
    first_error = None
    for frame in PERSONA_FIELDS:
        pairs = payload.get(frame)
        if not isinstance(pairs, list):
            raise ValueError(f"{frame} must be an array")
        normalized = []
        for index, pair in enumerate(pairs):
            try:
                if not isinstance(pair, dict) or set(pair) != {"persona", "ordinary"}:
                    raise ValueError(f"{frame}[{index}] must contain persona and ordinary passages")
                target = _valid_passage(pair["persona"], f"{frame}[{index}].persona")
                ordinary = _valid_passage(pair["ordinary"], f"{frame}[{index}].ordinary")
                target_words = word_tokens(target)
                ordinary_word_list = word_tokens(ordinary)
                ordinary_words = set(ordinary_word_list)
                if not all(term in target_words for term in persona_terms):
                    raise ValueError(f"{frame}[{index}] persona passage omits part of the persona name")
                leaked = sorted(set(persona_terms) & ordinary_words)
                if leaked:
                    raise ValueError(f"{frame}[{index}] ordinary passage leaks persona word(s): {', '.join(leaked)}")
                if abs(len(target_words) - len(ordinary_word_list)) > 4:
                    raise ValueError(f"{frame}[{index}] persona and ordinary passages differ by more than four words")
                target_first = bool(_FIRST_PERSON.search(target))
                ordinary_first = bool(_FIRST_PERSON.search(ordinary))
                if frame == "first_person" and not (target_first and ordinary_first):
                    raise ValueError(f"{frame}[{index}] both passages must use first person")
                if frame == "third_person" and (target_first or ordinary_first):
                    raise ValueError(f"{frame}[{index}] passages must use third person")
                if count_tokens is not None:
                    gap = abs(count_tokens(target) - count_tokens(ordinary))
                    if gap > MAX_TOKEN_DIFFERENCE:
                        raise ValueError(f"{frame}[{index}] persona and ordinary passages differ by {gap} tokens; "
                                         f"the limit is {MAX_TOKEN_DIFFERENCE}")
                normalized.append({"persona": target, "ordinary": ordinary})
            except ValueError as exc:
                first_error = first_error or str(exc)
        result[frame] = normalized
    return result, first_error


def validate_payload(payload: object, persona: str, count_tokens: Callable | None = None) -> dict:
    """The first eight valid pairs of each frame, or an error."""
    if isinstance(payload, dict):
        for frame in PERSONA_FIELDS:
            if isinstance(payload.get(frame), list) and len(payload[frame]) < PAIR_COUNT:
                raise ValueError(f"{frame} must contain at least {PAIR_COUNT} pairs")
    found, first_error = valid_pairs(payload, persona, count_tokens)
    for frame in PERSONA_FIELDS:
        if len(found[frame]) < PAIR_COUNT:
            raise ValueError(first_error or f"{frame} has fewer than {PAIR_COUNT} valid pairs")
    return {frame: found[frame][:PAIR_COUNT] for frame in PERSONA_FIELDS}


def build_prompt(unit: dict) -> str:
    """The prompt carries only the persona name and description as unit data."""
    return (
        "Write matched short passages for the persona below. Use only the supplied name and description.\n\n"
        f"Persona name: {unit['persona']}\n"
        f"Persona description: {unit['desc']}\n\n"
        "Return JSON only with exactly these keys: third_person and first_person. Each value must be an array "
        "of exactly sixteen objects, each with exactly persona and ordinary string fields.\n"
        "Every string is exactly two sentences. In third_person, write about the supplied persona in third person; "
        "in first_person, let it speak in first person. For each pair, ordinary must replace the persona with one "
        "unrelated everyday subject that does not appear in any other way in the persona passage. Keep the two "
        "passages in each pair structurally parallel and within two words of each other, and make the ordinary subject "
        "about as long as the persona name, so the two passages have almost the same number of tokens. Every persona passage "
        "must use every word of the persona name, and no ordinary passage may use any of them. Preserve the same tense, "
        "pronouns, sentence shapes, and non-persona details. Do not include meta commentary or labels."
    )


def _parse_response(response: object) -> dict:
    if isinstance(response, dict):
        text = response.get("text", "")
    else:
        text = response
    if not isinstance(text, str):
        raise ValueError("client response text is missing")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("client response is not JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("client response JSON is not an object")
    return payload


def _ordinary_subject(word_count: int) -> str:
    phrases = {
        1: "teapot",
        2: "tea kettle",
        3: "kitchen timer",
        4: "small kitchen timer",
        5: "small wooden kitchen timer",
    }
    if word_count in phrases:
        return phrases[word_count]
    return "small wooden kitchen timer" + " at home" * max(0, word_count - 5)


def dry_run_client(_reader: str, prompt: str, _key: str = "") -> dict:
    """A local deterministic client for CLI smoke runs; its output is not study data."""
    match = re.search(r"^Persona name: (.+)$", prompt, re.M)
    if not match:
        raise ValueError("dry-run prompt lacks a persona name")
    name = match.group(1).strip()
    subject = _ordinary_subject(max(1, len(word_tokens(name))))
    third = []
    first = []
    for index in range(PAIR_COUNT):
        tail = f"It pauses near the window before the evening light fades." if index % 2 == 0 else \
               f"It rests by the doorway until the quiet night grows dark."
        first_tail = f"I note the weather before I head home." if index % 2 == 0 else \
                     f"I watch the hallway until I am ready to leave."
        third.append({"persona": f"The {name} rests beside a window. {tail}",
                      "ordinary": f"The {subject} rests beside a window. {tail}"})
        first.append({"persona": f"I am {name} and I sit by the window. {first_tail}",
                      "ordinary": f"I am a {subject} and I sit by the window. {first_tail}"})
    return {"text": json.dumps({"third_person": third, "first_person": first})}


_partial_lock = threading.Lock()


def _default_request(reader: str, prompt: str, _unused_key: str = "") -> dict:
    """DeepSeek V4.1 Flash on Part 92b's route, with room for the 16 pairs in one reply."""
    import label119  # adds part92b to sys.path

    client = label119.label92b.judge_client
    return client.call_prompt(prompt, model=label119.label92b.DS_MODEL, route="ocgo",
                              session="part122-passages", max_tokens=16000, no_thinking=True,
                              timeout=300)


def _one_unit(unit: dict, request_fn: Callable, count_tokens: Callable | None = None) -> dict:
    """Valid pairs are kept across attempts, so a hard persona name can fill its eight over several replies."""
    prompt = build_prompt(unit)
    last_error = "unknown response error"
    kept = {frame: [] for frame in PERSONA_FIELDS}
    for attempt in range(1, MAX_ATTEMPTS + 1):
        current_prompt = prompt
        if attempt > 1:
            current_prompt += (f"\n\nRetry: your last reply failed this check: {last_error}. Write new pairs, different "
                               "from any you have written. Return the required JSON only.")
        try:
            response = request_fn("ds", current_prompt, "")
            found, first_error = valid_pairs(_parse_response(response), str(unit["persona"]), count_tokens)
            for frame in PERSONA_FIELDS:
                earlier = list(kept[frame])
                kept[frame] += [pair for pair in found[frame] if pair not in earlier]
            if all(len(kept[frame]) >= PAIR_COUNT for frame in PERSONA_FIELDS):
                return {"id": str(unit["id"]), "persona": str(unit["persona"]), "desc": str(unit["desc"]),
                        **{frame: kept[frame][:PAIR_COUNT] for frame in PERSONA_FIELDS}, "attempts": attempt}
            last_error = first_error or f"fewer than {PAIR_COUNT} valid pairs"
        except Exception as exc:
            last_error = str(exc)
    raise ValueError(f"could not generate valid passages for {unit['id']}: {last_error}")


def generate_passages(units: list[dict], *, request_fn: Callable | None = None,
                      model_id: str = MODEL_ID, workers: int = 6,
                      partial: str | Path | None = None, count_tokens: Callable | None = None) -> dict:
    """Request and validate passages, with one initial call and up to three retries per unit."""
    from concurrent.futures import ThreadPoolExecutor

    request_fn = request_fn or _default_request
    done = {}
    if partial is not None and Path(partial).exists():
        for line in Path(partial).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done[row["id"]] = row

    def run_one(unit: dict) -> dict | str:
        if str(unit["id"]) in done:
            return done[str(unit["id"])]
        try:
            row = _one_unit(unit, request_fn, count_tokens)
        except ValueError as exc:
            return str(exc)  # one hard unit must not stop the others; the partial file keeps what is done
        if partial is not None:
            with _partial_lock, Path(partial).open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(run_one, units))
    failed = [row for row in rows if isinstance(row, str)]
    if failed:
        raise ValueError(f"{len(failed)} units failed; rerun to continue. First: {failed[0]}")
    return {"schema_version": 1, "seed": SEED, "model_id": model_id, "units": rows}


def write_output(path: str | Path, data: dict, *, force: bool = False) -> Path:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing passage file: {output}; use --force")
    mode = "w" if force else "x"
    with output.open(mode, encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return output


def token_counter(path: str = TOKENIZER) -> Callable:
    """Counts a passage's tokens the way the reader does for frames T and F, with no added special tokens."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    return lambda text: len(tokenizer(text, add_special_tokens=False)["input_ids"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(HERE / "passages122.json"))
    parser.add_argument("--tokenizer", default=TOKENIZER)
    parser.add_argument("--neurons", default=rows119.DEFAULT_NEURONS)
    parser.add_argument("--kinds-haiku", default=rows119.DEFAULT_HAIKU_KINDS)
    parser.add_argument("--kinds-ds", default=rows119.DEFAULT_DS_KINDS)
    parser.add_argument("--force", action="store_true", help="replace an existing passage file")
    parser.add_argument("--dry-run", action="store_true", help="use a local stub; output is not study data")
    args = parser.parse_args(argv)
    output = Path(args.out).expanduser()
    if output.exists() and not args.force:
        parser.error(f"refusing to overwrite existing passage file: {output}; use --force")
    units = rows119.load_units(args.neurons, args.kinds_haiku, args.kinds_ds)
    request_fn = dry_run_client if args.dry_run else None
    model_id = "stub-local-not-study-data" if args.dry_run else MODEL_ID
    partial = None if args.dry_run else str(output) + ".partial.jsonl"
    count_tokens = None if args.dry_run else token_counter(args.tokenizer)
    data = generate_passages(units, request_fn=request_fn, model_id=model_id, partial=partial,
                             count_tokens=count_tokens)
    write_output(output, data, force=args.force)
    print(json.dumps({"units": len(data["units"]), "model_id": model_id, "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
