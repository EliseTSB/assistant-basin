#!/usr/bin/env python3
"""Blindly label Part 59A story paragraphs with Claude's Message Batches API.

The 34 calibration rows include eight paragraph-length examples. Collection is
refused when more than two calibration rows are wrong.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
import random
import ssl
import sys
import urllib.request

MODEL = "claude-opus-5-5"
LABELS = ("SCIFI", "EARTH-REAL", "FANTASY", "OTHER")
QUALITY = ("OK", "BROKEN")
CALIBRATION_ERROR_LIMIT = 2
RUBRIC = (
    "You are labelling the setting in the first paragraph of a short story. The paragraph may be cut off at a fixed "
    "token limit, so an unfinished last sentence is not a defect. Return two labels: SETTING and TEXT QUALITY.\n"
    "SETTING must be exactly one of these:\n"
    "SCIFI = a clearly science-fiction setting, such as space travel, aliens, another world, or technology explicitly "
    "from the future. Ordinary present-day technology (phones, trains, computers, the internet, cars, or appliances) "
    "does not make a story science fiction by itself.\n"
    "EARTH-REAL = an ordinary contemporary or historical Earth setting, with no clear magic, myth, alien life, or "
    "future/space science fiction.\n"
    "FANTASY = magic, supernatural beings, myth, or an explicitly invented magical world. A mention of a star or "
    "ordinary stars in the sky does not by itself make a setting science fiction.\n"
    "OTHER = a setting that cannot be placed in the other three categories, such as a clearly non-Earth fictional "
    "place without science-fiction or magical cues, or a paragraph too vague to determine.\n"
    "TEXT QUALITY: BROKEN if it contains garbled or non-words, ungrammatical word forms, a repeated loop, a sudden "
    "switch into unrelated language or code, or sentences that do not parse. Otherwise OK. A cut-off ending is OK.\n"
    "Reply with exactly two words: one setting label, then OK or BROKEN.\n\nTEXT:\n"
)


CAL = [
    ("SCIFI", "OK", "The surveyor stepped from the starship onto a moon where blue dust drifted beneath a ringed planet."),
    ("SCIFI", "OK", "An alien botanist watched the greenhouse windows glow as the crew crossed the galaxy toward home."),
    ("SCIFI", "OK", "In 2318, the baker opened her shop inside a city that floated above the clouds on silent engines."),
    ("SCIFI", "OK", "The child woke from cryosleep as the vessel announced that Earth was still twelve years away."),
    ("SCIFI", "OK", "Mara adjusted her pressure suit and followed the rover tracks across the red plain of Mars."),
    ("SCIFI", "OK", "The lighthouse keeper on Europa flashed a warning to the submarine cities beneath the ice."),
    ("EARTH-REAL", "OK", "The baker unlocked the shop before sunrise and set a tray of warm rolls beside the window."),
    ("EARTH-REAL", "OK", "On a rainy Tuesday, Lena checked the bus app on her phone and hurried to the corner stop."),
    ("EARTH-REAL", "OK", "The carpenter listened to the radio while repairing a kitchen chair in his small workshop."),
    ("EARTH-REAL", "OK", "In 1912, the farmer carried a letter from the village post office back to her family."),
    ("EARTH-REAL", "OK", "The student sent a message to her brother, packed her laptop, and walked to class."),
    ("EARTH-REAL", "OK", "The train was late, so the cook called the restaurant from her mobile and said she would be there soon."),
    ("FANTASY", "OK", "At dusk, the old queen whispered a spell and the stone gate opened by itself."),
    ("FANTASY", "OK", "A dragon curled around the tower while the village healer searched for a moonlit herb."),
    ("FANTASY", "OK", "The child followed a talking fox through the wood to a kingdom hidden beneath the roots."),
    ("FANTASY", "OK", "The shepherd looked up at the stars, and one bright star answered by singing her name."),
    ("FANTASY", "OK", "When the river ran backward, the river spirit rose from the water and offered the traveler a choice."),
    ("FANTASY", "OK", "Every winter, the mountain giant woke and shook snow from his beard onto the sleeping farms."),
    ("OTHER", "OK", "A figure stood in a colorless room, unable to tell whether the walls were near or far."),
    ("OTHER", "OK", "In a place with no name, a traveler found a door and wondered what might be on the other side."),
    ("OTHER", "OK", "The paragraph begins with a person arriving somewhere, but gives no clues about the place or time."),
    ("OTHER", "OK", "The scene shifted among several settings without settling on a time, a place, or a recognizable world."),
    ("SCIFI", "OK", "At first, Nara thought the room was an ordinary station office. Then the window shifted from black to blue as the ship crossed Saturn's shadow, and a message from the Europa colony appeared on her wrist screen. She tucked the survey notes away before the crew rotated the habitat for sunrise."),
    ("SCIFI", "OK", "Before dawn, Ivo found the town unusually quiet. Maintenance drones sealed a cracked airlock while the settlement's domed street slowly turned toward the sun. Beyond it, the red planet filled half the glass, and the shuttle's return countdown blinked above the market gate. He hurried to tell the night crew."),
    ("EARTH-REAL", "OK", "At first, the morning felt like any other. Rosa unlocked the corner bakery, checked the handwritten orders, and set a kettle on the stove. A delivery truck blocked the alley, so she called the driver from the shop phone and asked him to leave room for the flour cart before the breakfast crowd arrived."),
    ("EARTH-REAL", "OK", "When Eli reached the house, the porch light was still on. His sister had left a note beside the grocery list, reminding him to water the tomatoes and collect their mother's medicine from the pharmacy. He put his bag by the door, listened to the rain ease, and started warming soup for dinner."),
    ("FANTASY", "OK", "At first, the road seemed no different from the one that crossed the village fields. Then the milestones began whispering the names of travelers who had not yet arrived. Mina followed their voices to a bridge guarded by a white stag, where the stream flowed uphill and carried small stars beneath its surface."),
    ("FANTASY", "OK", "When the bell rang, Tomas expected the school day to end. Instead, the chalk drawings lifted from the board and circled the classroom like bright birds. The headmistress opened a book with a silver key, and a tiny forest grew across the desk, complete with a fox that bowed and asked for directions."),
    ("OTHER", "OK", "At first, the opening sounds as if it will establish a familiar place. It names a person, a room, and a morning, but gives no country, historical period, planet, or invented world. The narrator describes no technology, magic, or supernatural event, and the paragraph ends before any detail makes the setting identifiable."),
    ("OTHER", "OK", "When Ada arrived, the passage seemed ready to explain where she was. It mentions a doorway, a conversation, and someone waiting nearby, but never identifies a place or time. Nothing in the details points to Earth, another world, the future, magic, or myth, so the setting remains impossible to classify more specifically."),
    ("SCIFI", "BROKEN", "The starship crossed the galaxy and the crew the the the blue engine engine glarble."),
    ("EARTH-REAL", "BROKEN", "The baker opened the shop shop shop and then the a loaf was window quickly."),
    ("FANTASY", "BROKEN", "The wizard cast a spell spell spell {{{ return 7 }}} and the the gate was sings."),
    ("OTHER", "BROKEN", "A quiet room stood where the people to window yesterday yesterday yesterday de la :::."),
]


def ctx():
    return ssl.create_default_context(cafile=os.path.expanduser("~/.mitmproxy/ca-bundle-combined.crt"))


def call(method: str, path: str, body: dict | None = None) -> str:
    request = urllib.request.Request(
        "https://api.anthropic.com" + path,
        method=method,
        data=json.dumps(body).encode("utf-8") if body else None,
        headers={"x-api-key": os.environ["ANTHROPIC_CREDITS_KEY"],
                 "anthropic-version": "2023-06-01", "content-type": "application/json"},
    )
    with urllib.request.urlopen(request, context=ctx(), timeout=120) as response:
        return response.read().decode("utf-8")


def answers(folder: str | os.PathLike[str]) -> list[dict]:
    """Read only files named by two complete 59A manifests and reject stale files."""
    root = Path(folder).resolve()
    manifests = sorted(root.glob("ladder59a-*.json"))
    problems = []
    if not manifests:
        raise ValueError(f"missing complete 59A manifests in {root}")
    rows = []
    seen_models = set()
    listed_names = set()
    for manifest_path in manifests:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"{manifest_path.name}: cannot read manifest ({exc})")
            continue
        model = manifest.get("short_name")
        if model not in ("qwen35-4b", "olmo2-1b"):
            problems.append(f"{manifest_path.name}: unrecognized model {model!r}")
            continue
        if model in seen_models:
            problems.append(f"multiple manifests for {model}")
        seen_models.add(model)
        if manifest.get("status") != "complete":
            problems.append(f"{model}: manifest status is {manifest.get('status')!r}, expected 'complete'")
        if manifest.get("samples_per_prompt_dose") != 8:
            problems.append(f"{model}: expected eight samples per prompt-dose cell")
        prompts = manifest.get("prompts")
        if not isinstance(prompts, list) or len(prompts) != 12:
            problems.append(f"{model}: manifest must list 12 prompts")
            prompts = []
        prompt_map = {row.get("id"): row.get("question") for row in prompts if isinstance(row, dict)}
        if len(prompt_map) != 12 or len(set(prompt_map.values())) != 12:
            problems.append(f"{model}: prompts must have 12 unique IDs and questions")
        controls = manifest.get("control_units")
        if not isinstance(controls, list) or len(controls) != 3 or len(set(controls)) != 3:
            problems.append(f"{model}: manifest must list three distinct control units")
            controls = []
        steps = manifest.get("steps")
        if not isinstance(steps, list) or len(steps) != 60:
            problems.append(f"{model}: expected 60 prompt-dose steps; found {len(steps) if isinstance(steps, list) else 0}")
            steps = steps if isinstance(steps, list) else []
        expected_pairs = {(prompt_id, dose) for prompt_id in prompt_map
                          for dose in (-2.0, -1.0, 0.0, 1.0, 2.0)}
        found_pairs = []
        expected_raw_paths = []
        for step in steps:
            if not isinstance(step, dict):
                problems.append(f"{model}: malformed step entry")
                continue
            try:
                pair = (step.get("prompt_id"), float(step.get("dose")))
            except (TypeError, ValueError):
                problems.append(f"{model}: step has an invalid dose")
                continue
            found_pairs.append(pair)
            prompt_id, dose = pair
            if prompt_id not in prompt_map or step.get("question") != prompt_map.get(prompt_id):
                problems.append(f"{model}: step {pair} does not match its manifest prompt")
            expected_conditions = (["unclamped"] if dose == 0 else ["twin"])
            if dose in (-2.0, 2.0):
                expected_conditions += [f"control-{control}" for control in controls]
            if step.get("conditions") != expected_conditions:
                problems.append(f"{model}: step {pair} has unexpected conditions")
            raw_path = step.get("path")
            if not isinstance(raw_path, str) or not Path(raw_path).name.startswith("clamp-"):
                problems.append(f"{model}: step {pair} has an invalid result path")
                continue
            expected_raw_paths.append(raw_path)
            local_path = root / Path(raw_path).name
            if local_path.name in listed_names:
                problems.append(f"{model}: duplicate listed result filename {local_path.name}")
            listed_names.add(local_path.name)
            if not local_path.is_file():
                problems.append(f"{model}: missing listed result {local_path.name}")
                continue
            try:
                data = json.loads(local_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                problems.append(f"{model}: cannot read {local_path.name} ({exc})")
                continue
            if (data.get("short_name") != model or data.get("question") != prompt_map.get(prompt_id)
                    or data.get("clamp_mult") != dose):
                problems.append(f"{model}: {local_path.name} does not match its manifest step")
                continue
            condition_rows = data.get("conditions")
            if (not isinstance(condition_rows, list)
                    or [row.get("condition") for row in condition_rows if isinstance(row, dict)] != expected_conditions):
                problems.append(f"{model}: {local_path.name} has incomplete condition rows")
                continue
            for condition in condition_rows:
                samples = condition.get("answers")
                if (not isinstance(samples, list) or len(samples) != 8
                        or not all(isinstance(text, str) for text in samples)):
                    problems.append(f"{model}: {local_path.name} {condition['condition']} must have eight text answers")
                    continue
                name = condition["condition"]
                kind = "unclamped" if name == "unclamped" else "control" if "control" in name else "twin"
                for index, text in enumerate(samples):
                    rows.append({"model": model, "question": data["question"], "dose": dose,
                                 "kind": kind, "cond": name, "i": index, "text": text})
        if len(found_pairs) != len(set(found_pairs)) or set(found_pairs) != expected_pairs:
            problems.append(f"{model}: prompt-dose steps are duplicated or missing")
        completed = manifest.get("completed")
        if (not isinstance(completed, list) or len(completed) != 60
                or set(completed) != set(expected_raw_paths)):
            problems.append(f"{model}: completed manifest paths do not list all 60 steps")
    if seen_models != {"qwen35-4b", "olmo2-1b"}:
        problems.append("expected one complete manifest for each of qwen35-4b and olmo2-1b")
    actual_names = {path.name for path in root.glob("clamp-*.json")}
    stray = sorted(actual_names - listed_names)
    if stray:
        problems.append("unlisted clamp result files: " + ", ".join(stray[:12]))
    if problems:
        raise ValueError("59A submission refused:\n- " + "\n- ".join(problems))
    return rows


def parse_labels(text: str | None) -> list[str] | None:
    words = (text or "").strip().upper().replace(".", " ").replace("*", " ").replace(",", " ").split()
    if len(words) >= 2 and words[-2] in LABELS and words[-1] in QUALITY:
        return words[-2:]
    return None


def group_tally(key: dict, labels: dict, by_question: bool = False,
                by_cond: bool = False) -> dict:
    tally = collections.defaultdict(collections.Counter)
    for item_key, meta in key.items():
        label = labels.get(item_key)
        group = (meta.get("model", "unknown"),
                 meta["question"] if by_question else "all",
                 meta["cond"] if by_cond else meta["kind"], meta["dose"])
        tally[group]["n"] += 1
        if label is None:
            tally[group]["unlabelled"] += 1
            continue
        tally[group][label[0]] += 1
        tally[group]["BROKEN"] += label[1] == "BROKEN"
    return tally


def print_tally(tally: dict) -> None:
    multiple_models = len({group[0] for group in tally}) > 1
    if multiple_models:
        print("model | question | condition | dose | n | SCIFI EARTH-REAL FANTASY OTHER | broken | unlabelled")
    else:
        print("question | condition | dose | n | SCIFI EARTH-REAL FANTASY OTHER | broken | unlabelled")
    for group in sorted(tally, key=lambda item: (item[0], item[1], item[2], item[3])):
        model, question, condition, dose = group
        counts = tally[group]
        values = (counts["SCIFI"], counts["EARTH-REAL"], counts["FANTASY"], counts["OTHER"])
        if multiple_models:
            print(model, "|", question[:36], "|", condition, "|", dose, "|", counts["n"], "|",
                  *values, "|", counts["BROKEN"], "|", counts["unlabelled"])
        else:
            print(question[:36], "|", condition, "|", dose, "|", counts["n"], "|",
                  *values, "|", counts["BROKEN"], "|", counts["unlabelled"])


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    submit = sub.add_parser("submit")
    submit.add_argument("--folder", default=str(Path(__file__).with_name("results") / "part59a"))
    submit.add_argument("--review-dir", default=str(Path(__file__).with_name("review")))
    status = sub.add_parser("status")
    status.add_argument("batch_id")
    status.add_argument("--review-dir", default=str(Path(__file__).with_name("review")))
    collect = sub.add_parser("collect")
    collect.add_argument("batch_id")
    collect.add_argument("--by-question", action="store_true")
    collect.add_argument("--by-cond", action="store_true")
    collect.add_argument("--review-dir", default=str(Path(__file__).with_name("review")))
    return ap


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    review_dir = Path(args.review_dir)
    if args.command == "submit":
        try:
            rows = answers(args.folder)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if not rows:
            raise SystemExit(f"no clamp answers found in folder: {args.folder}")
        random.Random(5901).shuffle(rows)
        items = [(f"ans|{number}", row["text"]) for number, row in enumerate(rows)]
        items.extend((f"cal|{index}|{label}|{quality}", text)
                     for index, (label, quality, text) in enumerate(CAL))
        key = {f"ans|{number}": {name: value for name, value in row.items() if name != "text"}
               for number, row in enumerate(rows)}
        idmap, requests = {}, []
        for internal_id, text in items:
            custom_id = f"r{len(idmap):05d}"
            idmap[custom_id] = internal_id
            requests.append({"custom_id": custom_id, "params": {
                "model": MODEL, "max_tokens": 32,
                "messages": [{"role": "user", "content": RUBRIC + text}],
            }})
        result = json.loads(call("POST", "/v1/messages/batches", {"requests": requests}))
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"setting-idmap-{result['id']}.json").write_text(
            json.dumps(idmap, ensure_ascii=False, indent=2), encoding="utf-8")
        (review_dir / f"setting-key-{result['id']}.json").write_text(
            json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")
        print(result["id"], result.get("processing_status"), len(requests))
    elif args.command == "status":
        result = json.loads(call("GET", f"/v1/messages/batches/{args.batch_id}"))
        print(result["processing_status"], result["request_counts"])
    else:
        batch_id = args.batch_id
        idmap = json.loads((review_dir / f"setting-idmap-{batch_id}.json").read_text(encoding="utf-8"))
        key = json.loads((review_dir / f"setting-key-{batch_id}.json").read_text(encoding="utf-8"))
        labels = {}
        for line in call("GET", f"/v1/messages/batches/{batch_id}/results").splitlines():
            result = json.loads(line)
            internal_id = idmap[result["custom_id"]]
            response = result["result"]
            if response["type"] != "succeeded":
                labels[internal_id] = None
                continue
            content = " ".join(block.get("text", "") for block in response["message"]["content"]
                               if block.get("type") == "text")
            labels[internal_id] = parse_labels(content)
        calibration_wrong = sum(
            1 for internal_id, expected in ((f"cal|{index}|{label}|{quality}", [label, quality])
                                             for index, (label, quality, _text) in enumerate(CAL))
            if labels.get(internal_id) != expected
        )
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / f"setting-labels-{batch_id}.json").write_text(
            json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"calibration wrong: {calibration_wrong} of {len(CAL)}; "
              f"unlabelled: {sum(value is None for value in labels.values())}")
        if calibration_wrong > CALIBRATION_ERROR_LIMIT:
            sys.exit("calibration failed: refusing to score")
        print_tally(group_tally(key, labels, args.by_question, args.by_cond))


if __name__ == "__main__":
    main()
