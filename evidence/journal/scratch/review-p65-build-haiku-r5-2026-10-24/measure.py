#!/usr/bin/env python3
"""Cold-review measurements for Part 65 judge65 (ac95b47). Drives the REAL judge65 + paidaction Ledger
from the review worktree, with a fake provider and temporary ledgers under the review scratch dir.
No network. Never uses DEFAULT_LEDGER_PATH: every run passes an explicit ledger under RUNS."""
import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

import os  # noqa: E402

WT = Path(os.environ.get("JUDGE_WT", "/home/elisepc/green/worktrees/ab-part65/part65"))
SCRATCH = Path("/home/elisepc/green/Journal/scratch/review-p65-build-haiku-r5-2026-10-24")
RUNS = SCRATCH / "runs"
RUNS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(WT))

import judge65 as J  # noqa: E402
from paid_ledger import Ledger, HARD_CAP_USD  # noqa: E402

ERR_BODY = json.dumps({"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}})


def fresh(name):
    return Path(tempfile.mkdtemp(prefix=name + "-", dir=RUNS))


def trials(n, tag):
    out = []
    for i in range(n):
        cid = f"p65-{i:04d}"
        out.append({
            "request": {"custom_id": cid, "params": {
                "model": J.MODEL, "max_tokens": J.MAX_TOKENS, "system": J.SYSTEM_PROMPT,
                "thinking": {"type": "disabled"},
                "messages": [{"role": "user", "content": f"{tag} trial {i} " + "x" * 200}]}},
            "mapping": {"custom_id": cid, "unit_key": f"u{i // 2}", "type": "neuron", "layer": 1,
                        "layer_band": "early", "trial": (i % 2) + 1, "foil_unit_key": f"v{i // 2}",
                        "own_position": "A" if i % 2 == 0 else "B", "sentence_test_pass": True},
        })
    return out


class Fake:
    """Provider double. `accepted` counts requests the provider actually accepted (the real send)."""

    def __init__(self, *, reject_post=None, never_end=(), drop_response_on_post=None,
                 crash_before_post=None):
        self.accepted = 0
        self.attempts = 0
        self.batches = {}
        self.reject_post = reject_post
        self.never_end = set(never_end)
        self.drop_response_on_post = drop_response_on_post
        self.crash_before_post = crash_before_post

    def __call__(self, method, path, body=None, timeout=120.0):
        if method == "POST":
            self.attempts += 1
            if self.crash_before_post == self.attempts:
                raise KeyboardInterrupt("simulated crash before the request leaves the process")
            if self.reject_post == self.attempts:
                raise J.APIHttpError(400, ERR_BODY)
            self.accepted += 1
            bid = f"batch_{self.accepted}"
            self.batches[bid] = body["requests"]
            if self.drop_response_on_post == self.attempts:
                raise RuntimeError("connection reset after provider accepted the batch")
            return json.dumps({"id": bid})
        bid = path.split("/")[4]
        if path.endswith("/results"):
            rows = [{"custom_id": r["custom_id"], "result": {"type": "succeeded", "message": {
                "content": [{"type": "text", "text": "A"}],
                "usage": {"input_tokens": 20, "output_tokens": 1}}}} for r in self.batches[bid]]
            return "\n".join(json.dumps(r) for r in rows)
        if bid in self.never_end:
            return json.dumps({"id": bid, "processing_status": "in_progress"})
        return json.dumps({"id": bid, "processing_status": "ended"})


def run(root, ledger, api, **kw):
    timeout = kw.pop("timeout_seconds", 5)
    return J.run_chunks(root, ledger, api, poll_seconds=0, timeout_seconds=timeout,
                        sleep=lambda _s: None, **kw)


def states(root):
    return [(json.loads(p.read_text())["status"]) for p in sorted((root / "chunks").glob("chunk-*.json"))]


def ledger_summary(ledger_path):
    if not ledger_path.exists():
        return "no ledger"
    s = Ledger(str(ledger_path), HARD_CAP_USD).state()
    return {k: s.get(k) for k in ("settled", "reserved", "released")}


def s1_pending_chunk_does_not_stop_later_send():
    print("== S1 pending chunk 0 (provider still in_progress at deadline): does chunk 1 POST? ==")
    d = fresh("s1")
    root, ledger = d / "judge", d / "ledger.jsonl"
    J.prepare_run(trials(4, "s1"), root, 2, 1)
    api = Fake(never_end={"batch_1"})
    totals = run(root, ledger, api, timeout_seconds=0)
    print("totals", totals, "accepted_posts", api.accepted, "chunk_states", states(root))


def s2_release_then_other_plan():
    print("== S2 chunk 0 released (validated 400); then (a) same plan in new dir, (b) new seed in new dir ==")
    d = fresh("s2")
    ledger = d / "ledger.jsonl"
    root_a = d / "a"
    J.prepare_run(trials(4, "planX"), root_a, 2, 1)
    api_a = Fake(reject_post=1)
    print("A totals", run(root_a, ledger, api_a))
    print("A accepted_posts", api_a.accepted, "chunk_states", states(root_a), "ledger", ledger_summary(ledger))

    root_b = d / "b_same_plan"
    J.prepare_run(trials(4, "planX"), root_b, 2, 2)
    api_b = Fake()
    try:
        run(root_b, ledger, api_b)
        print("B same plan: no refusal")
    except J.JudgeError as exc:
        print("B same plan refused:", exc)
    print("B accepted_posts", api_b.accepted)

    root_c = d / "c_new_seed"
    J.prepare_run(trials(4, "planY"), root_c, 2, 3)
    api_c = Fake()
    totals_c = run(root_c, ledger, api_c)
    print("C new-seed plan totals", totals_c, "accepted_posts", api_c.accepted,
          "chunk_states", states(root_c))
    print("ledger after C", ledger_summary(ledger))
    print("C run.json completion:", json.loads((root_c / "run.json").read_text()).get("completion") is not None)


def s4_second_dir_same_plan_while_first_pending():
    print("== S4 same plan, dir A chunk 0 pending (in flight), dir B same plan ==")
    d = fresh("s4")
    ledger = d / "ledger.jsonl"
    root_a, root_b = d / "a", d / "b"
    J.prepare_run(trials(2, "s4"), root_a, 2, 1)
    api_a = Fake(never_end={"batch_1"})
    print("A totals", run(root_a, ledger, api_a, timeout_seconds=0), "accepted", api_a.accepted)
    J.prepare_run(trials(2, "s4"), root_b, 2, 2)
    api_b = Fake()
    try:
        run(root_b, ledger, api_b)
        print("B: no refusal")
    except J.JudgeError as exc:
        print("B refused:", exc)
    print("B accepted_posts", api_b.accepted)


def crash_case(name, *, patches=(), fake_kwargs=None, expect_first_error=True):
    d = fresh("crash-" + name)
    root, ledger = d / "judge", d / "ledger.jsonl"
    J.prepare_run(trials(2, name), root, 2, 1)
    api = Fake(**(fake_kwargs or {}))
    first = "ok"
    with __import__("contextlib").ExitStack() as stack:
        for target, attr, replacement in patches:
            stack.enter_context(mock.patch.object(target, attr, replacement))
        try:
            run(root, ledger, api)
        except BaseException as exc:  # first invocation is expected to crash or stop
            first = f"{type(exc).__name__}: {str(exc)[:90]}"
    accepted_after_first = api.accepted
    try:
        second = run(root, ledger, api)
        second_out = second
    except BaseException as exc:
        second_out = f"{type(exc).__name__}: {str(exc)[:110]}"
    answers = (root / "answers.jsonl")
    n_answers = len(answers.read_text().splitlines()) if answers.exists() else 0
    completion = json.loads((root / "run.json").read_text()).get("completion") is not None
    print(f"{name}: first={first!r} accepted_after_first={accepted_after_first} "
          f"second={second_out} TOTAL_ACCEPTED_POSTS={api.accepted} states={states(root)} "
          f"answers={n_answers} completion={completion} ledger={ledger_summary(ledger)}")


def s3_crash_matrix():
    print("== S3 crash-point matrix: count provider-accepted sends across two invocations ==")
    crash_case("crash_after_reserve_row", patches=[(Ledger, "reserve", _after_reserve)])
    crash_case("crash_after_reserved_state_before_posting", patches=[(J, "_save_chunk", _raise_on_status("posting"))])
    crash_case("crash_before_post_with_posting_marker_durable", fake_kwargs={"crash_before_post": 1})
    crash_case("accepted_post_but_submitted_save_fails", patches=[(J, "_save_chunk", _raise_on_status("submitted"))])
    crash_case("post_response_lost_ambiguous", fake_kwargs={"drop_response_on_post": 1})
    crash_case("results_saved_then_settle_crash", patches=[(Ledger, "settle", _raise_once_settle())])
    crash_case("settled_state_not_saved", patches=[(J, "_save_chunk", _raise_on_status("settled"))])
    crash_case("answers_partially_written_then_crash", patches=[(J, "atomic_append_jsonl", _partial_append())])


_original_reserve = Ledger.reserve
_original_settle = Ledger.settle
_original_save = J._save_chunk
_original_append = J.atomic_append_jsonl


def _after_reserve(self, *a, **k):
    _original_reserve(self, *a, **k)
    raise RuntimeError("crash after reserve row appended, before state saved")


def _raise_on_status(status):
    state_seen = {"raised": False}

    def save(path, state):
        if state.get("status") == status and not state_seen["raised"]:
            state_seen["raised"] = True
            raise RuntimeError(f"crash while saving status={status}")
        return _original_save(path, state)
    return save


def _raise_once_settle():
    seen = {"n": 0}

    def settle(self, attempt_id, actual_usd, usage):
        seen["n"] += 1
        if seen["n"] == 1:
            raise RuntimeError("crash inside settle (before settle row)")
        return _original_settle(self, attempt_id, actual_usd, usage)
    return settle


def _partial_append():
    seen = {"n": 0}

    def append(path, row):
        seen["n"] += 1
        _original_append(path, row)
        if seen["n"] == 1:
            raise RuntimeError("crash after first answer row")
    return append


def s5_torn_ledger_line():
    print("== S5 ledger with an unterminated torn final line (crash mid-append) ==")
    d = fresh("s5")
    ledger = d / "ledger.jsonl"
    Ledger(str(ledger), HARD_CAP_USD).reserve(0.01, "other-kind", {"note": "unrelated"})
    with ledger.open("ab") as stream:
        stream.write(b'{"event": "rese')
    root = d / "judge"
    J.prepare_run(trials(2, "s5"), root, 2, 1)
    api = Fake()
    try:
        run(root, ledger, api)
        print("run completed")
    except J.JudgeError as exc:
        print("JudgeError:", exc)
    print("accepted_posts", api.accepted, "chunk_states", states(root))
    print("paidaction Ledger.state() tolerates the same file:", ledger_summary(ledger))


def s6_probe_trial_is_repeated_in_full_plan():
    print("== S6 probe request vs full plan's first request ==")
    plan = trials(4, "s6")
    probe = J.build_probe_trial(plan)[0]
    print("probe custom_id", probe["request"]["custom_id"], "== full plan first custom_id",
          plan[0]["request"]["custom_id"], "; identical request body:", probe["request"] == plan[0]["request"])


def s7_release_on_first_of_three_chunks():
    print("== S7 validated 400 on chunk 0 of 3: how many chunks does the provider accept? ==")
    d = fresh("s7")
    root, ledger = d / "judge", d / "ledger.jsonl"
    J.prepare_run(trials(6, "s7"), root, 2, 1)
    api = Fake(reject_post=1)
    totals = run(root, ledger, api)
    print("totals", totals, "accepted_posts", api.accepted, "chunk_states", states(root))


def s8_mutant_m3_crash_after_post_before_submitted_save():
    print("== S8 posting marker not durable (mutant M3 when JUDGE_WT points there): accepted sends after crash ==")
    crash_case("post_accepted_then_submitted_save_crash", patches=[(J, "_save_chunk", _raise_on_status("submitted"))])


def s7_release_on_first_of_three_chunks():
    print("== S7 validated 400 on chunk 0 of 3: how many chunks does the provider accept? ==")
    d = fresh("s7")
    root, ledger = d / "judge", d / "ledger.jsonl"
    J.prepare_run(trials(6, "s7"), root, 2, 1)
    api = Fake(reject_post=1)
    totals = run(root, ledger, api)
    print("totals", totals, "accepted_posts", api.accepted, "chunk_states", states(root))


def s8_posting_marker_durability_crash_after_post():
    print("== S8 crash after provider accepted, before the 'submitted' save (run against a mutant via JUDGE_WT) ==")
    crash_case("post_accepted_then_submitted_save_crash", patches=[(J, "_save_chunk", _raise_on_status("submitted"))])


def s9_torn_answer_line_then_append():
    print("== S9 answers.jsonl with a torn final line (crash mid-append), then one more persisted answer ==")
    d = fresh("s9")
    root = d / "judge"
    root.mkdir(parents=True)
    row = {"judge_run_id": "r" * 32, "custom_id": "p65-0000", "plan_sha256": "p" * 64, "run_root": str(root)}
    (root / "answers.jsonl").write_text(json.dumps(row) + "\n" + '{"judge_run_id": "' + "r" * 32)
    new = dict(row, custom_id="p65-0001")
    J._persist_answers(root, [new])
    print("persist ok; file now:")
    for line in (root / "answers.jsonl").read_text().splitlines():
        print("  ", line[:70])
    try:
        J.read_jsonl(root / "answers.jsonl")
        print("re-read ok")
    except J.JudgeError as exc:
        print("re-read JudgeError:", exc)


TABLE = {"s1": s1_pending_chunk_does_not_stop_later_send, "s2": s2_release_then_other_plan,
         "s3": s3_crash_matrix, "s4": s4_second_dir_same_plan_while_first_pending,
         "s5": s5_torn_ledger_line, "s6": s6_probe_trial_is_repeated_in_full_plan,
         "s7": s7_release_on_first_of_three_chunks, "s8": s8_posting_marker_durability_crash_after_post,
         "s9": s9_torn_answer_line_then_append}

if __name__ == "__main__":
    for key in sys.argv[1:] or ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s9"]:
        TABLE[key]()
        print()
