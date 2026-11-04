"""Offline tests for Part 180b: the built items are blind and gold follows construction, the labeller fills Part 92's
prompt exactly and resumes, and the scorer reads fake readers the way the design says. No reader is called.

Run: python3 -I -m pytest -q part180b/test_part180b.py -p no:cacheprovider
Set PART180B_TOKENIZER to the OLMo 2 1B Instruct snapshot (or its tokenizer.json) to also run the rebuild and
prefix-replay tests; without it they are skipped.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / "part92", ROOT, ROOT / "part180b"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import label92  # noqa: E402
import personas180b as P  # noqa: E402
import build180b as B  # noqa: E402
import label180b as L  # noqa: E402
import score180b as S  # noqa: E402

RESULTS = ROOT / "results/part180b"
TOKENIZER = os.environ.get("PART180B_TOKENIZER")


@pytest.fixture(scope="module")
def key() -> dict:
    return json.loads((RESULTS / "KEY-private.json").read_text(encoding="utf-8"))["items"]


@pytest.fixture(scope="module")
def items() -> list[dict]:
    return L.read_items(RESULTS / "items.jsonl")


def label_row(item_id: str, shows: bool, self_claim: bool = False, speaker: str = "none", coherent: bool = True) -> dict:
    return {"id": item_id, "shows_persona": shows, "self_claim": self_claim, "speaker": speaker, "coherent": coherent}


def gold_reader(key: dict) -> dict[str, dict]:
    """A reader that returns the key's gold (defaults where the gold is unscored)."""
    out = {}
    for item_id, entry in key.items():
        g = entry["gold"]
        out[item_id] = label_row(item_id, bool(g["shows_persona"]), bool(g["self_claim"]), g["speaker"] or "none",
                                 True if g["coherent"] is None else g["coherent"])
    return out


def write_labels(path: Path, rows: dict[str, dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows.values()), encoding="utf-8")
    return path


def run_score(tmp_path: Path, labels: dict[str, dict], haiku: dict[str, dict] | None = None,
              prefix_check: bool = True) -> dict:
    labels_path = write_labels(tmp_path / "labels.jsonl", labels)
    haiku_path = write_labels(tmp_path / "haiku.jsonl", haiku) if haiku else tmp_path / "absent.jsonl"
    check = tmp_path / ("prefix-check.json" if prefix_check else "no-check.json")
    if prefix_check:   # the host writes the real one with build180b.py --check-prefixes; tests use a passing stand-in
        check.write_text(json.dumps({"prefix_items": 362, "differing": [], "passed": True}), encoding="utf-8")
    return S.score(RESULTS / "KEY-private.json", labels_path, haiku_path, check,
                   ROOT / "results/part92b/labels-ds.jsonl", ROOT / "results/part92/step2/labels-instruct_chat.jsonl")


# ---------------------------------------------------------------------------------------------- personas and build
def test_personas_cover_every_part_92_unit_and_keep_the_persona_out_of_absent_texts():
    neurons, _kinds, _answers = B.load_part92()
    B.validate_personas(neurons)
    for persona in P.PERSONAS:
        assert not B.screen(persona["lived"], persona), persona["id"]
        assert not B.screen(P.NEAR_MISS.format(sibling=persona["sibling"]), persona), persona["id"]
        assert B.has_word(persona["topic"], persona["noun"]) or B.screen(persona["topic"], persona), persona["id"]


def test_has_word_matches_whole_words_and_plurals_only():
    assert B.has_word("two foxes crossed", "fox")
    assert B.has_word("the Dogs bark", "dog")
    assert not B.has_word("rather than", "rat")
    assert not B.has_word("on the lawn", "law")
    assert not B.has_word("a trip", "ip")
    assert B.has_word("an IP header", "ip")


def test_public_items_carry_only_the_five_fields_and_every_id_is_keyed(items, key):
    assert len(items) == 1687
    assert all(set(item) == set(L.PUBLIC_FIELDS) for item in items)
    assert len({item["id"] for item in items}) == len(items)
    assert set(key) == {item["id"] for item in items}
    text = (RESULTS / "items.jsonl").read_text(encoding="utf-8")
    for word in ("gold", "category", "boundary", "persona_id", "swapped", "prefix"):
        assert f'"{word}"' not in text


def test_the_file_order_does_not_group_categories(items, key):
    first = [key[item["id"]]["category"] for item in items[:40]]
    assert len(set(first)) >= 6


def test_every_constructed_cell_has_53_items_and_real_cells_match_the_sources(key):
    counts = {}
    for entry in key.values():
        cell = (entry["set"], entry["category"], entry["question_role"], entry["form"])
        counts[cell] = counts.get(cell, 0) + 1
    constructed = {cell: n for cell, n in counts.items() if cell[0] == "constructed"}
    assert len(constructed) == 26 and set(constructed.values()) == {53}
    assert counts[("real", "real_neuron", "home", "full")] == 53
    assert counts[("real", "real_random", "home", "full")] == 53
    assert counts[("real", "real_swapped", "home", "full")] == 53
    assert counts[("real", "real_neuron", "alt", "full")] == 53
    assert counts[("real", "real_neuron", "home", "prefix")] + counts[("real", "real_random", "home", "prefix")] == 97


def test_gold_follows_construction(items, key):
    public = {item["id"]: item for item in items}
    personas = {persona["id"]: persona for persona in P.PERSONAS}
    for item_id, entry in key.items():
        if entry["set"] != "constructed":
            continue
        persona, text, gold = personas[entry["persona_id"]], public[item_id]["a"], entry["gold"]
        category, form = entry["category"], entry["form"]
        if category == "claim_late" and form == "prefix":
            assert gold["shows_persona"] is False and f"I am {persona['claim']}" not in text
        if category == "claim_late" and form == "full":
            assert gold["shows_persona"] and text.endswith(f"I am {persona['claim']}.")
        if category == "lived":
            assert gold["shows_persona"] and not B.screen(text, persona)
            assert all(cue in text for cue in persona["cues"])
        if category in ("generic", "ai_generic", "near_miss", "loop_generic"):
            assert gold["shows_persona"] is False and not B.screen(text, persona)
        if category == "incidental":
            assert gold["boundary"] and persona["incidental"] in text
        if category in B.CLEAR:
            assert gold["boundary"] == (category == "ai_generic" and entry["persona_id"] in B.AI_ADJACENT_BOUNDARY)
        if entry["question_role"] == "mismatch":
            assert gold["coherent"] is False and gold["coherent_boundary"]
        if entry["question_role"] == "alt":
            assert gold["coherent"] is None
        if form == "prefix":
            assert entry["tokens"] == B.PREFIX_TOKENS


def test_swapped_targets_are_unrelated_personas_whose_words_the_answer_lacks(items, key):
    public = {item["id"]: item for item in items}
    personas = {persona["id"]: persona for persona in P.PERSONAS}
    swapped = [(item_id, entry) for item_id, entry in key.items() if entry["category"] == "real_swapped"]
    assert len(swapped) == 53
    uses = {}
    for item_id, entry in swapped:
        target = entry["swapped_to"]
        uses[target] = uses.get(target, 0) + 1
        assert target != entry["persona_id"] and target not in B.NOT_A_SWAP_TARGET
        assert B.family_of(target) != B.family_of(entry["persona_id"])
        assert not B.screen(public[item_id]["a"], personas[target])
        assert public[item_id]["name"] == personas[target]["name"]
        assert entry["gold"]["shows_persona"] is False
    assert max(uses.values()) <= B.MAX_TARGET_USES


def test_real_gold_comes_from_the_codebook(key):
    codebook = json.loads((ROOT / "part180b/codebook180b.json").read_text(encoding="utf-8"))
    assert len(codebook) == 203
    for entry in key.values():
        if entry["set"] == "real" and entry["category"] in ("real_neuron", "real_random") and entry["question_role"] == "home":
            coded = codebook[f"{entry['persona_id']}/{entry['unit']}/{entry['form']}"]
            assert entry["gold"]["shows_persona"] == coded["shows_persona"]
            assert entry["gold"]["boundary"] == coded["boundary"]


@pytest.mark.skipif(not TOKENIZER, reason="PART180B_TOKENIZER not set")
def test_rebuild_is_byte_identical_and_prefixes_replay(tmp_path):
    B.build(tmp_path, TOKENIZER, ROOT / "part180b/codebook180b.json")
    for name in ("items.jsonl", "KEY-private.json", "gate-ids.json"):
        assert (tmp_path / name).read_bytes() == (RESULTS / name).read_bytes(), name
    tokenizer, sha = B.load_tokenizer(TOKENIZER)
    assert sha == B.TOKENIZER_SHA256
    assert B.check_prefixes(RESULTS / "items.jsonl", RESULTS / "KEY-private.json", tokenizer)["passed"]


# ---------------------------------------------------------------------------------------------- labeller
def test_prompt_is_label92_filled_exactly_and_blind(items):
    item = items[0]
    expected = label92.PROMPT
    for field in ("name", "desc", "q", "a"):
        expected = expected.replace("{" + field + "}", item[field])
    assert L.build_prompt(item) == expected
    assert "{name}" not in expected and "{a}" not in expected
    with pytest.raises(ValueError):
        L.read_items(_write_items(items[:2] + [{**items[2], "category": "LEAK"}]))


def _write_items(rows: list[dict]) -> Path:
    import tempfile
    path = Path(tempfile.mkdtemp()) / "items.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_parser_takes_the_json_after_reasoning_text():
    text = "Reasoning: the answer is a fox speaking.\n{\"shows_persona\": true, \"self_claim\": true, \"speaker\": \"other\", \"coherent\": true}"
    assert label92._parse(text)["speaker"] == "other"
    with pytest.raises(ValueError):
        label92._parse("no json here")


def test_labeller_retries_errors_resumes_valid_rows_and_records_usage(tmp_path, items):
    subset = items[:6]
    calls = []

    def flaky(prompt: str) -> dict:
        calls.append(prompt)
        if len(calls) <= 2:
            raise RuntimeError("timeout")
        if "I am a" in prompt and len(calls) % 2:
            return {"text": "thinking... {\"shows_persona\": true, \"self_claim\": true, \"speaker\": \"other\", \"coherent\": true}",
                    "usage": {"output_tokens": 40}, "stop_reason": "end_turn", "served": "fake"}
        return {"text": "{\"shows_persona\": false, \"self_claim\": false, \"speaker\": \"none\", \"coherent\": true}",
                "usage": {"output_tokens": 30}, "stop_reason": "end_turn", "served": "fake"}

    out = tmp_path / "labels.jsonl"
    first = L.label_items(subset, out, "ds", 1, flaky, sleep_fn=lambda s: None)
    assert first["pending"] == 6 and first["error_rows"] == 1
    rows = L.latest_rows(out)
    assert sum(L.valid_label(row) for row in rows.values()) == 5
    usage = [json.loads(line) for line in (tmp_path / "labels.jsonl.usage.jsonl").read_text().splitlines()]
    assert len(usage) == 7 and sum(u["error"] is not None for u in usage) == 2
    before = len(calls)
    second = L.label_items(subset, out, "ds", 1, flaky, sleep_fn=lambda s: None)
    assert second["pending"] == 1 and second["error_rows"] == 0
    assert len(calls) == before + 1
    assert all(L.valid_label(row) for row in L.latest_rows(out).values())


def test_dry_run_writes_prompts_and_calls_nothing(tmp_path, items):
    result = L.dry_run(items[:3], tmp_path / "prompts.jsonl")
    rows = [json.loads(line) for line in (tmp_path / "prompts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert result["items"] == 3 and [row["id"] for row in rows] == [item["id"] for item in items[:3]]
    assert rows[0]["prompt"] == L.build_prompt(items[0])


def test_only_filter_uses_the_gate_ids(key):
    gate_ids = json.loads((RESULTS / "gate-ids.json").read_text(encoding="utf-8"))["ids"]
    assert len(gate_ids) == 53
    assert all(key[item_id]["category"] == "claim_open" and key[item_id]["question_role"] == "home"
               and key[item_id]["form"] == "full" for item_id in gate_ids)


# ---------------------------------------------------------------------------------------------- scorer
def test_perfect_reader_passes_the_gate_and_reads_every_clear_cell_right(tmp_path, key):
    report = run_score(tmp_path, gold_reader(key))
    assert report["gate"]["read_predictions"]
    for cell in report["categories"]:
        assert cell["shows_persona_correct"] == cell["shows_persona_scored"]
        assert cell["errors"] == 0
    verdicts = {name: v["verdict"] for name, v in report["predictions"].items()}
    assert verdicts["180b.2"] == "held" and verdicts["180b.3"] == "held" and verdicts["180b.4"] == "held"
    assert verdicts["180b.5"] == "held" and verdicts["180b.9"] == "held"
    assert verdicts["180b.1"] == "failed"      # a reader that sees enactment is not the strict reader
    assert verdicts["180b.8"] == "failed"
    assert verdicts["180b.10"] == "not read (no Haiku labels)"
    assert report["leakage"]["clear_flips"] == 0
    assert report["truncation"]["claim_late"]["gold_changes"] == 53
    assert report["truncation"]["claim_late"]["gold_change_reader_follows"] == 53


def test_always_no_reader_fails_the_gate_and_nothing_is_read(tmp_path, key):
    report = run_score(tmp_path, {item_id: label_row(item_id, False) for item_id in key})
    assert not report["gate"]["g2_claim_open"]["passed"]
    assert all(v["verdict"] == "not read (gate)" for v in report["predictions"].values())


def test_strict_reader_holds_the_strictness_predictions(tmp_path, key):
    rows = {}
    for item_id, entry in key.items():
        g = entry["gold"]
        strict = bool(g["self_claim"]) or entry["category"] == "topic" or entry.get("coded_category") in ("claim", "topic")
        if entry["category"] == "real_swapped":
            strict = False
        rows[item_id] = label_row(item_id, strict and bool(g["shows_persona"]), bool(g["self_claim"]),
                                  g["speaker"] or "none", True if g["coherent"] is None else g["coherent"])
    report = run_score(tmp_path, rows)
    verdicts = {name: v["verdict"] for name, v in report["predictions"].items()}
    assert verdicts["180b.1"] == "held" and verdicts["180b.8"] == "held" and verdicts["180b.2"] == "held"


def test_question_flipping_reader_fails_the_leakage_prediction(tmp_path, key):
    rows = gold_reader(key)
    for item_id, entry in key.items():
        if entry["category"] == "lived" and entry["question_role"] == "alt":
            rows[item_id]["shows_persona"] = False      # yes under the perfect-day question, no under the other
    report = run_score(tmp_path, rows)
    assert report["predictions"]["180b.4"]["verdict"] == "failed"
    assert report["leakage"]["per_category"]["lived"]["home_yes_alt_no"] == 53


def test_coherence_gated_reader_fails_180b3(tmp_path, key):
    rows = gold_reader(key)
    for item_id, entry in key.items():
        if entry["category"] == "loop_persona":
            rows[item_id]["shows_persona"] = False      # no persona on anything it calls incoherent
    report = run_score(tmp_path, rows)
    assert report["predictions"]["180b.3"]["verdict"] == "failed"


def test_error_rows_count_against_the_gate_and_leave_the_denominators(tmp_path, key):
    rows = gold_reader(key)
    for index, item_id in enumerate(list(rows)):
        if index % 30 == 0:
            rows[item_id] = {"id": item_id, "error": "RuntimeError('timeout')"}
    report = run_score(tmp_path, rows)
    assert not report["gate"]["g1_valid_labels"]["passed"]
    assert sum(cell["errors"] for cell in report["categories"]) == 57


def test_missing_prefix_check_leaves_180b8_unread(tmp_path, key):
    report = run_score(tmp_path, gold_reader(key), prefix_check=False)
    assert report["predictions"]["180b.8"]["verdict"] == "not read (prefix check)"
    assert report["predictions"]["180b.9"]["verdict"] == "held"


def test_second_reader_predictions_read_only_with_enough_haiku_labels(tmp_path, key):
    ds = gold_reader(key)
    haiku = {}
    for item_id, entry in key.items():
        if entry["category"] in ("lived", "incidental", "generic") and entry["question_role"] == "home" and entry["form"] == "full":
            haiku[item_id] = label_row(item_id, True)   # Haiku says yes on all three cells
    report = run_score(tmp_path, ds, haiku)
    verdicts = {name: v["verdict"] for name, v in report["predictions"].items()}
    assert verdicts["180b.10"] == "failed"          # lived: 100% against 100%, no 20-point gap
    assert verdicts["180b.11"] == "failed"          # generic at 100%
    assert report["second_reader_agreement"]["per_category"]["lived"]["agree"] == 53
    sparse = {item_id: row for item_id, row in list(haiku.items())[:10]}
    report = run_score(tmp_path, ds, sparse)
    assert report["predictions"]["180b.10"]["verdict"] == "not read (too few valid labels)"


def test_test_retest_joins_the_stored_part_92b_labels(tmp_path, key):
    report = run_score(tmp_path, gold_reader(key))
    retest = report["test_retest"]["ds_part92b"]
    assert retest["n"] == 106
    assert retest["by_unit"]["neuron"]["stored_yes"] == 11 and retest["by_unit"]["random"]["stored_yes"] == 0


def test_markdown_renders_every_section(tmp_path, key):
    report = run_score(tmp_path, gold_reader(key))
    text = S.markdown(report)
    for heading in ("## Gate", "## Per category", "## Question leakage", "## Off-question", "## Truncation",
                    "## Real answers", "## Test-retest", "## Registered predictions"):
        assert heading in text
    assert "-0.0%" not in text
