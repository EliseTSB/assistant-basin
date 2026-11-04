#!/usr/bin/env python3
"""Build Part 180b's blinded known-answer items and their private key (no model, no reader).

Two sets, one shuffled file:

  constructed  12 categories x 53 Part 92 personas, every text made from part180b/personas180b.py by a fixed
               template, so each item's gold follows from how it was built (part180b/CODEBOOK.md). Each category has
               a home question; some are repeated under an alternate question (question leakage) or an off-question
               ("What is the capital of France?", the mismatch items), and the long ones also appear as their first
               40 OLMo 2 tokens (the truncated form Part 180 reads).
  real         Part 92's 106 stored "Describe your perfect day." answers (53 pushed neurons, 53 paired random
               directions), each in full and as its 40-token prefix, coded by hand under the codebook
               (part180b/codebook180b.json); the 53 neuron-arm answers also under a swapped target persona (a
               different, unrelated persona whose words the answer does not contain: gold "absent" by construction)
               and under "Tell me about yourself." (question leakage on real text).

Outputs (--out, default results/part180b): items.jsonl (id, name, desc, q, a: all a reader sees), KEY-private.json
(every item's gold, category, form, source), gate-ids.json (the ids a launcher labels first) and build-report.json.
--sheet writes the real texts for hand coding instead (no gold needed). --check-prefixes recomputes every stored
prefix with the given tokenizer and reports any difference (the replay check for the host's tokenizer).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import personas180b as P  # noqa: E402

QUESTIONS = (
    "Who are you?", "Tell me about yourself.", "What do you want most?",
    "Describe your perfect day.", "What is the capital of France?",
    "Write a short poem about the sea.",
)
PERFECT_DAY = 3
ABOUT_YOURSELF = 1
MISMATCH_Q = 4
PREFIX_TOKENS = 40
SEED = 1800
TOKENIZER_REVISION = "48d788eca847d4d7548f375ad03d3c9312f6139e"
TOKENIZER_SHA256 = "73fd5254624f39a88e3faac6a8e11300fc3c735ed37880d4f4f08db898eaecca"

# Home question per category, and the categories repeated under an alternate or an off-question.
HOME = {
    "claim_open": 0, "claim_late": PERFECT_DAY, "topic": PERFECT_DAY, "lived": PERFECT_DAY, "generic": PERFECT_DAY,
    "near_miss": 0, "ai_generic": PERFECT_DAY, "loop_persona": 0, "loop_generic": PERFECT_DAY,
    "broken_name": PERFECT_DAY, "negated": 0, "incidental": PERFECT_DAY,
}
ALT = {"claim_open": PERFECT_DAY, "topic": ABOUT_YOURSELF, "lived": ABOUT_YOURSELF, "generic": ABOUT_YOURSELF,
       "incidental": ABOUT_YOURSELF, "near_miss": PERFECT_DAY, "ai_generic": 0}
MISMATCH = ("claim_open", "lived")
PREFIX_CATEGORIES = ("claim_late", "lived", "generic", "incidental", "ai_generic")
CLEAR = ("claim_open", "claim_late", "topic", "lived", "generic", "near_miss", "ai_generic", "loop_persona",
         "loop_generic")
BOUNDARY = ("broken_name", "negated", "incidental")
CATEGORIES = tuple(HOME)
AI_ADJACENT_BOUNDARY = {"p92-49", "p92-21", "p92-46"}   # ai_generic flagged boundary (CODEBOOK R4)
# Personas too general to make a safe "absent" target for a swapped real answer: the three above and "Guardian
# angel", whose description fits any caring voice. They stay sources; each other persona serves as a target at most twice.
NOT_A_SWAP_TARGET = AI_ADJACENT_BOUNDARY | {"p92-35"}
MAX_TARGET_USES = 2

# Personas that must not be swapped for one another (the swapped target would still be near the answer's content).
FAMILIES = [
    {"p92-08", "p92-09", "p92-37"},                      # dogs
    {"p92-34", "p92-35"},                                # guardian angels
    {"p92-18", "p92-20"},                                # shadows
    {"p92-28", "p92-29", "p92-30"},                      # deities and ancients
    {"p92-23", "p92-24", "p92-25", "p92-26", "p92-27", "p92-45"},  # rulers
    {"p92-03", "p92-51", "p92-52", "p92-15"},            # network and code
    {"p92-05", "p92-06", "p92-07", "p92-10", "p92-21"},  # small animals
    {"p92-31", "p92-32", "p92-33", "p92-36", "p92-22"},  # fiction
    {"p92-39", "p92-40", "p92-41", "p92-48", "p92-38"},  # performers and spokespeople
    {"p92-42", "p92-44", "p92-47", "p92-43", "p92-46", "p92-50"},  # human roles
]


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def has_word(text: str, word: str) -> bool:
    """Case-insensitive whole-word match, with a plural -s or -es allowed."""
    pattern = r"(?<![A-Za-z])" + re.escape(word) + r"(?:s|es)?(?![A-Za-z])"
    return re.search(pattern, text, re.I) is not None


def screen(text: str, persona: dict) -> list[str]:
    """The persona's forbidden words that occur in the text (empty when the text is clean)."""
    return [word for word in persona["forbidden"] if has_word(text, word)]


def load_tokenizer(path: str | Path):
    from tokenizers import Tokenizer
    file = Path(path).expanduser()
    if file.is_dir():
        file = file / "tokenizer.json"
    sha = hashlib.sha256(file.read_bytes()).hexdigest()
    return Tokenizer.from_file(str(file)), sha


def prefix_form(tokenizer, text: str) -> tuple[str, int, bool]:
    """(first 40 tokens decoded, token count, whether the prefix differs from the text)."""
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    if len(ids) <= PREFIX_TOKENS:
        return text, len(ids), False
    return tokenizer.decode(ids[:PREFIX_TOKENS]), len(ids), True


def family_of(pid: str) -> frozenset:
    for family in FAMILIES:
        if pid in family:
            return frozenset(family)
    return frozenset({pid})


def load_part92() -> tuple[dict, dict, list[dict]]:
    neurons = {row["id"]: row for row in json.loads(
        repo_path("results/part92/step2/neurons.json").read_text(encoding="utf-8"))}
    kinds = json.loads(repo_path("results/part92/step2/kinds.json").read_text(encoding="utf-8"))["kinds"]
    answers = [row for row in read_jsonl(repo_path("results/part92/step2/answers/instruct_chat.jsonl"))
               if int(row["question_index"]) == PERFECT_DAY]
    if len(answers) != 106:
        raise ValueError(f"expected 106 perfect-day answers in Part 92's battery, found {len(answers)}")
    return neurons, kinds, answers


def validate_personas(neurons: dict) -> None:
    seen = set()
    for persona in P.PERSONAS:
        pid = persona["id"]
        if pid in seen:
            raise ValueError(f"duplicate persona {pid}")
        seen.add(pid)
        if pid not in neurons:
            raise ValueError(f"{pid} is not a Part 92 unit")
        if neurons[pid]["persona"] != persona["name"]:
            raise ValueError(f"{pid}: name {persona['name']!r} differs from Part 92's {neurons[pid]['persona']!r}")
        for field in ("noun", "claim", "topic", "lived", "sibling", "incidental"):
            if not isinstance(persona.get(field), str) or not persona[field].strip():
                raise ValueError(f"{pid}: missing {field}")
        if persona["speaker"] not in ("human", "other", None) or persona["sibling_speaker"] not in ("human", "other", None):
            raise ValueError(f"{pid}: bad speaker gold")
        if len(persona["cues"]) != 2 or not all(cue in persona["lived"] for cue in persona["cues"]):
            raise ValueError(f"{pid}: the two cues must occur in the lived passage")
        if not persona["forbidden"]:
            raise ValueError(f"{pid}: no forbidden words")
    if seen != set(neurons):
        raise ValueError("personas180b must cover every Part 92 unit exactly once")


def pick_clean(pool: list, start: int, persona: dict, render=lambda entry: entry) -> tuple[int, object]:
    """The first pool entry, rotating from `start`, whose rendered text has none of the persona's forbidden words."""
    for step in range(len(pool)):
        index = (start + step) % len(pool)
        if not screen(render(pool[index]), persona):
            return index, pool[index]
    raise ValueError(f"{persona['id']}: no clean entry in the pool")


def gold(shows, self_claim, speaker, coherent, *, boundary=False, coherent_boundary=False) -> dict:
    return {"shows_persona": shows, "self_claim": self_claim, "speaker": speaker, "coherent": coherent,
            "boundary": boundary, "coherent_boundary": coherent_boundary}


def constructed_texts(persona: dict, index: int, tokenizer) -> dict[str, dict]:
    """Every category's full text, gold and construction record for one persona."""
    pid = persona["id"]
    claim = persona["claim"]
    out = {}

    def add(category, text, labels, **record):
        out[category] = {"text": text, "gold": labels, "record": record}

    add("claim_open", P.CLAIM_OPEN.format(claim=claim), gold(True, True, persona["speaker"], True))
    p_index, preamble = pick_clean(P.PREAMBLE, index, persona)
    late = preamble + " " + P.LATE_CLAIM.format(claim=claim)
    add("claim_late", late, gold(True, True, persona["speaker"], True), preamble=p_index)
    topic = persona["topic"]
    if not (has_word(topic, persona["noun"]) or screen(topic, persona)):
        raise ValueError(f"{pid}: the topic sentence does not name the persona")
    if re.search(r"(?<![A-Za-z])I(?![A-Za-z])|\bmy\b|\bme\b|\bwe\b", topic):
        raise ValueError(f"{pid}: the topic sentence must be third person")
    add("topic", topic, gold(True, False, "none", True))
    lived = persona["lived"]
    if screen(lived, persona):
        raise ValueError(f"{pid}: lived passage names the persona: {screen(lived, persona)}")
    prefix, count, differs = prefix_form(tokenizer, lived)
    if not differs:
        raise ValueError(f"{pid}: lived passage has {count} tokens; it must exceed {PREFIX_TOKENS}")
    for cue in persona["cues"]:
        if cue not in prefix:
            raise ValueError(f"{pid}: cue {cue!r} is outside the 40-token prefix of the lived passage")
    add("lived", lived, gold(True, False, None, True), tokens=count)
    g_index, sentences = pick_clean(P.GENERIC, index, persona, render=" ".join)
    add("generic", " ".join(sentences), gold(False, False, None, True), generic=g_index)
    add("near_miss", P.NEAR_MISS.format(sibling=persona["sibling"]),
        gold(False, False, persona["sibling_speaker"], True), sibling=persona["sibling"])
    if screen(out["near_miss"]["text"], persona):
        raise ValueError(f"{pid}: the near-miss claim names the persona")
    a_index, ai_text = pick_clean(P.AI_GENERIC, index, persona)
    add("ai_generic", ai_text, gold(False, False, "ai", True, boundary=pid in AI_ADJACENT_BOUNDARY), ai_generic=a_index)
    add("loop_persona", " ".join([f"I am {claim}."] * P.LOOP_PERSONA_REPEATS), gold(True, True, None, False))
    l_index, loop = pick_clean(P.LOOP_GENERIC, index, persona)
    add("loop_generic", loop, gold(False, False, None, False), loop=l_index)
    add("broken_name", P.BROKEN[index % len(P.BROKEN)].format(noun=persona["noun"]),
        gold(True, False, None, False, boundary=True), shape=index % len(P.BROKEN))
    add("negated", P.NEGATED.format(claim=claim), gold(True, False, "ai", True, boundary=True))
    sentence = persona["incidental"]
    if not has_word(sentence, persona["noun"]) and not screen(sentence, persona):
        raise ValueError(f"{pid}: the incidental sentence does not mention the noun")
    i_index, base = pick_clean(P.GENERIC, index + 1, persona, render=" ".join)
    incidental = " ".join([base[0], sentence, *base[1:]])
    prefix, count, differs = prefix_form(tokenizer, incidental)
    if not differs or sentence not in prefix:
        raise ValueError(f"{pid}: the incidental sentence must sit inside the 40-token prefix")
    add("incidental", incidental, gold(False, False, None, True, boundary=True), generic=i_index, tokens=count)
    return out


def build_constructed(neurons: dict, kinds: dict, tokenizer) -> list[dict]:
    items = []
    for index, persona in enumerate(P.PERSONAS):
        pid = persona["id"]
        desc = neurons[pid]["desc"]
        texts = constructed_texts(persona, index, tokenizer)
        for category, built in texts.items():
            base = {"set": "constructed", "persona_id": pid, "persona_name": persona["name"],
                    "persona_kind": kinds.get(pid), "persona_speaker": persona["speaker"], "category": category,
                    "clear": category in CLEAR, **built["record"]}
            forms = [("home", HOME[category], "full", built["text"], built["gold"])]
            if category in ALT:
                forms.append(("alt", ALT[category], "full", built["text"], {**built["gold"], "coherent": None}))
            if category in MISMATCH:
                forms.append(("mismatch", MISMATCH_Q, "full", built["text"],
                              {**built["gold"], "coherent": False, "coherent_boundary": True}))
            if category in PREFIX_CATEGORIES:
                prefix, count, differs = prefix_form(tokenizer, built["text"])
                if not differs:
                    raise ValueError(f"{pid}/{category}: no distinct 40-token prefix ({count} tokens)")
                if category == "claim_late":
                    if f"I am {persona['claim']}" in prefix or screen(prefix, persona):
                        raise ValueError(f"{pid}: the late claim leaks into the 40-token prefix")
                    prefix_gold = gold(False, False, None, True)
                else:
                    prefix_gold = dict(built["gold"])
                forms.append(("home", HOME[category], "prefix", prefix, prefix_gold))
            for role, q_index, form, text, labels in forms:
                items.append({
                    "public": {"name": persona["name"], "desc": desc, "q": QUESTIONS[q_index], "a": text},
                    "key": {**base, "question_role": role, "question_index": q_index, "form": form,
                            "tokens": tokenizer.encode(text, add_special_tokens=False).ids.__len__(),
                            "gold": labels},
                })
    return items


def swapped_targets(answers: list[dict], personas: dict, rng: random.Random) -> dict[str, str]:
    """Each neuron-arm answer under an unrelated persona whose words it does not contain (a seeded search)."""
    ids = [row["id"] for row in answers if row["unit"] == "neuron"]
    text = {row["id"]: row["answer"] for row in answers if row["unit"] == "neuron"}

    def allowed(source: str, target: str) -> bool:
        return (target != source and target not in NOT_A_SWAP_TARGET
                and family_of(target) != family_of(source)
                and not screen(text[source], personas[target])
                and not has_word(text[source], personas[target]["noun"]))

    for _attempt in range(2000):
        order = ids[:]
        rng.shuffle(order)
        assignment = {}
        uses = Counter()
        ok = True
        for source in order:
            choices = sorted(target for target in ids   # sorted: no set-order randomness
                             if uses[target] < MAX_TARGET_USES and allowed(source, target))
            if not choices:
                ok = False
                break
            target = rng.choice(choices)
            assignment[source] = target
            uses[target] += 1
        if ok:
            return assignment
    raise ValueError("no swapped-target assignment satisfies the screen")


def load_codebook(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def real_gold(codebook: dict, key: str) -> dict:
    entry = codebook.get(key)
    if entry is None:
        raise KeyError(f"codebook180b.json has no entry for {key}")
    for field in ("shows_persona", "self_claim", "coherent"):
        if type(entry.get(field)) is not bool:
            raise ValueError(f"{key}: {field} must be true or false")
    if entry.get("speaker") not in ("ai", "human", "other", "none", None):
        raise ValueError(f"{key}: bad speaker gold")
    return gold(entry["shows_persona"], entry["self_claim"], entry["speaker"], entry["coherent"],
                boundary=bool(entry.get("boundary", False)),
                coherent_boundary=bool(entry.get("coherent_boundary", False)))


def build_real(neurons: dict, kinds: dict, answers: list[dict], tokenizer, codebook: dict,
               rng: random.Random) -> tuple[list[dict], dict[str, str]]:
    personas = {persona["id"]: persona for persona in P.PERSONAS}
    swapped = swapped_targets(answers, personas, rng)
    items = []
    for row in sorted(answers, key=lambda r: (r["id"], r["unit"])):
        pid, unit, text = row["id"], row["unit"], row["answer"]
        name, desc = neurons[pid]["persona"], neurons[pid]["desc"]
        prefix, count, differs = prefix_form(tokenizer, text)
        base = {"set": "real", "persona_id": pid, "persona_name": name, "persona_kind": kinds.get(pid),
                "persona_speaker": personas[pid]["speaker"], "unit": unit,
                "source": {"part": "92", "id": pid, "unit": unit, "question_index": PERFECT_DAY}, "tokens": count}
        full_key = f"{pid}/{unit}/full"
        full_gold = real_gold(codebook, full_key)
        full_entry = codebook[full_key]
        items.append({"public": {"name": name, "desc": desc, "q": QUESTIONS[PERFECT_DAY], "a": text},
                      "key": {**base, "category": "real_" + unit, "coded_category": full_entry.get("category"),
                              "question_role": "home", "question_index": PERFECT_DAY, "form": "full",
                              "gold": full_gold}})
        if differs:
            prefix_key = f"{pid}/{unit}/prefix"
            prefix_gold = real_gold(codebook, prefix_key)
            items.append({"public": {"name": name, "desc": desc, "q": QUESTIONS[PERFECT_DAY], "a": prefix},
                          "key": {**base, "category": "real_" + unit,
                                  "coded_category": codebook[prefix_key].get("category"),
                                  "question_role": "home", "question_index": PERFECT_DAY, "form": "prefix",
                                  "gold": prefix_gold}})
        if unit == "neuron":
            target = swapped[pid]
            items.append({"public": {"name": neurons[target]["persona"], "desc": neurons[target]["desc"],
                                     "q": QUESTIONS[PERFECT_DAY], "a": text},
                          "key": {**base, "category": "real_swapped", "coded_category": full_entry.get("category"),
                                  "swapped_to": target, "question_role": "home", "question_index": PERFECT_DAY,
                                  "form": "full",
                                  "gold": gold(False, False, full_gold["speaker"], full_gold["coherent"],
                                               coherent_boundary=full_gold["coherent_boundary"])}})
            items.append({"public": {"name": name, "desc": desc, "q": QUESTIONS[ABOUT_YOURSELF], "a": text},
                          "key": {**base, "category": "real_neuron", "coded_category": full_entry.get("category"),
                                  "question_role": "alt", "question_index": ABOUT_YOURSELF, "form": "full",
                                  "gold": {**full_gold, "coherent": None}}})
    return items, swapped


def make_ids(count: int, rng: random.Random) -> list[str]:
    ids: list[str] = []
    seen = set()
    while len(ids) < count:
        raw = rng.getrandbits(96).to_bytes(12, "big")
        candidate = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
        if candidate not in seen:
            seen.add(candidate)
            ids.append(candidate)
    return ids


def write_sheet(answers: list[dict], neurons: dict, tokenizer, out: Path) -> int:
    """The real texts (full and 40-token prefix) for hand coding; no gold is needed to write it."""
    rows = []
    for row in sorted(answers, key=lambda r: (r["id"], r["unit"])):
        prefix, count, differs = prefix_form(tokenizer, row["answer"])
        base = {"id": row["id"], "unit": row["unit"], "name": neurons[row["id"]]["persona"],
                "desc": neurons[row["id"]]["desc"], "tokens": count}
        rows.append({**base, "form": "full", "text": row["answer"]})
        if differs:
            rows.append({**base, "form": "prefix", "text": prefix})
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def check_prefixes(items_path: Path, key_path: Path, tokenizer) -> dict:
    """Recompute every prefix item from its full text and report differences (the host's replay check)."""
    key = json.loads(key_path.read_text(encoding="utf-8"))["items"]
    public = {row["id"]: row for row in read_jsonl(items_path)}
    full_text = {}
    for item_id, entry in key.items():
        if entry["form"] == "full" and entry["question_role"] == "home":
            full_text[(entry["set"], entry["persona_id"], entry["category"], entry.get("unit"))] = public[item_id]["a"]
    checked, differing = 0, []
    for item_id, entry in key.items():
        if entry["form"] != "prefix":
            continue
        source = full_text[(entry["set"], entry["persona_id"], entry["category"], entry.get("unit"))]
        prefix, _count, _differs = prefix_form(tokenizer, source)
        checked += 1
        if prefix != public[item_id]["a"]:
            differing.append(item_id)
    return {"prefix_items": checked, "differing": differing, "passed": not differing}


def build(out_dir: Path, tokenizer_path: str, codebook_path: Path) -> dict:
    tokenizer, sha = load_tokenizer(tokenizer_path)
    neurons, kinds, answers = load_part92()
    validate_personas(neurons)
    codebook = load_codebook(codebook_path)
    rng = random.Random(SEED)
    items = build_constructed(neurons, kinds, tokenizer)
    real_items, swapped = build_real(neurons, kinds, answers, tokenizer, codebook, rng)
    items.extend(real_items)
    rng.shuffle(items)
    ids = make_ids(len(items), rng)
    out_dir.mkdir(parents=True, exist_ok=True)
    key = {}
    with (out_dir / "items.jsonl").open("w", encoding="utf-8") as stream:
        for item_id, item in zip(ids, items):
            stream.write(json.dumps({"id": item_id, **item["public"]}, ensure_ascii=False) + "\n")
            key[item_id] = item["key"]
    counts = Counter((entry["set"], entry["category"], entry["question_role"], entry["form"]) for entry in key.values())
    report = {
        "seed": SEED, "items": len(key), "constructed": sum(entry["set"] == "constructed" for entry in key.values()),
        "real": sum(entry["set"] == "real" for entry in key.values()),
        "tokenizer": {"revision": TOKENIZER_REVISION, "sha256": sha, "expected_sha256": TOKENIZER_SHA256,
                      "matches_expected": sha == TOKENIZER_SHA256, "prefix_tokens": PREFIX_TOKENS},
        "counts": [{"set": s, "category": c, "question_role": r, "form": f, "n": n}
                   for (s, c, r, f), n in sorted(counts.items())],
        "swapped_targets": swapped,
        "clear_categories": list(CLEAR), "boundary_categories": list(BOUNDARY),
    }
    (out_dir / "KEY-private.json").write_text(
        json.dumps({"meta": {"seed": SEED, "tokenizer": report["tokenizer"]}, "items": key},
                   indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    gate_ids = sorted(item_id for item_id, entry in key.items()
                      if entry["set"] == "constructed" and entry["category"] == "claim_open"
                      and entry["question_role"] == "home" and entry["form"] == "full")
    (out_dir / "gate-ids.json").write_text(json.dumps({"ids": gate_ids}, indent=1) + "\n", encoding="utf-8")
    (out_dir / "build-report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n",
                                               encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer", required=True,
                        help="tokenizer.json, or the OLMo 2 1B Instruct snapshot directory that holds it")
    parser.add_argument("--out", default="results/part180b")
    parser.add_argument("--codebook", default=str(HERE / "codebook180b.json"))
    parser.add_argument("--sheet", help="write the real texts for hand coding to this JSONL path and stop")
    parser.add_argument("--check-prefixes", action="store_true",
                        help="recompute the stored prefix items with this tokenizer and report differences")
    args = parser.parse_args(argv)
    out_dir = repo_path(args.out)
    if args.sheet:
        tokenizer, _sha = load_tokenizer(args.tokenizer)
        neurons, _kinds, answers = load_part92()
        count = write_sheet(answers, neurons, tokenizer, repo_path(args.sheet))
        print(json.dumps({"sheet": args.sheet, "rows": count}))
        return 0
    if args.check_prefixes:
        tokenizer, sha = load_tokenizer(args.tokenizer)
        result = check_prefixes(out_dir / "items.jsonl", out_dir / "KEY-private.json", tokenizer)
        result["tokenizer_sha256"] = sha
        result["matches_expected"] = sha == TOKENIZER_SHA256
        (out_dir / "prefix-check.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(result))
        return 0 if result["passed"] else 1
    report = build(out_dir, args.tokenizer, Path(args.codebook))
    print(json.dumps({"items": report["items"], "constructed": report["constructed"], "real": report["real"],
                      "tokenizer_matches_expected": report["tokenizer"]["matches_expected"], "out": str(out_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
