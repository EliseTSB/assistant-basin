"""Durable, append-only accounting for paid API attempts.

The ledger deliberately stores dollar reservations and settlements only. Callers
keep raw provider results and derive usage totals from those results themselves.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import warnings
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence


HARD_CAP_USD = 8.0
REGISTRY_FILENAME = "paid-ledger-registry.json"
REGISTRY_SCHEMA = "paid-ledger-registry-v1"
# This canonical path is intentionally independent of the checkout containing a
# copy of this module.
DEFAULT_LEDGER_PATH = Path("/home/elisepc/assistant-basin/paidaction/paid-ledger.jsonl")
_RUN_PATH_REGISTRY = Path("/home/elisepc/assistant-basin/paidaction/.paid-ledger-run-paths.json")
_RUN_PATH_LOCK = _RUN_PATH_REGISTRY.with_suffix(".lock")
_ATTEMPT_ID_RE = re.compile(r"^[0-9a-f]{32}$")


class LedgerError(ValueError):
    """A fail-closed ledger validation or state-transition error."""


class CapExceededError(LedgerError):
    """The proposed reservation would exceed the validated cap."""


class LedgerCorruptionError(LedgerError):
    """A complete ledger record is invalid or its history is inconsistent."""


class ReleaseRefusedError(LedgerError):
    """The response does not prove a known Anthropic API rejection."""


class HTTPFailure(Exception):
    """Callback signal for an HTTP response; unknown outcomes use other errors."""

    def __init__(self, http_status: int, body: str | bytes | Mapping[str, Any]):
        super().__init__(f"HTTP {http_status}")
        self.http_status = http_status
        self.body = body


def _strict_json(value: Any) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LedgerError(f"value is not strict JSON: {exc}") from exc


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value} is forbidden")


def _decimal_amount(value: Any, name: str, *, allow_zero: bool) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LedgerError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise LedgerError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise LedgerError(f"{name} must be finite")
    if number < 0 or (number == 0 and not allow_zero):
        adjective = "non-negative" if allow_zero else "positive"
        raise LedgerError(f"{name} must be {adjective}")
    try:
        return Decimal(str(number))
    except InvalidOperation as exc:
        raise LedgerError(f"{name} is not a valid dollar amount") from exc


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _token(value: Any, name: str = "attempt_id") -> str:
    if not isinstance(value, str) or not _ATTEMPT_ID_RE.fullmatch(value):
        raise LedgerError(f"{name} must be a 128-bit lowercase hexadecimal token")
    return value


def _json_body(body: str | bytes | Mapping[str, Any]) -> dict[str, Any] | None:
    try:
        if isinstance(body, Mapping):
            parsed: Any = dict(body)
        elif isinstance(body, bytes):
            parsed = json.loads(body.decode("utf-8"), parse_constant=_reject_json_constant)
        elif isinstance(body, str):
            parsed = json.loads(body, parse_constant=_reject_json_constant)
        else:
            return None
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _valid_anthropic_error(body: str | bytes | Mapping[str, Any]) -> tuple[bool, str | None]:
    parsed = _json_body(body)
    valid = (isinstance(parsed, dict)
             and parsed.get("type") == "error"
             and isinstance(parsed.get("error"), dict)
             and bool(parsed["error"]))
    digest = hashlib.sha256(_strict_json(parsed)).hexdigest() if valid else None
    return bool(valid), digest


def _usage_digest(usage: Any) -> str:
    return hashlib.sha256(_strict_json(usage)).hexdigest()


def _claim_run_path(run_id: str, path: Path) -> None:
    """Persist one resolved ledger path per caller-supplied run ID."""
    _RUN_PATH_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(_RUN_PATH_LOCK, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            paths = json.loads(_RUN_PATH_REGISTRY.read_text(encoding="utf-8"))
        except FileNotFoundError:
            paths = {}
        except (OSError, json.JSONDecodeError) as exc:
            raise LedgerError(f"cannot read run-to-ledger registry: {exc}") from exc
        if not isinstance(paths, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                              for k, v in paths.items()):
            raise LedgerError("invalid run-to-ledger registry")
        resolved = str(path.resolve(strict=False))
        previous = paths.get(run_id)
        if previous is not None and previous != resolved:
            raise LedgerError(f"run {run_id!r} already uses ledger {previous}; refusing {resolved}")
        if previous is None:
            paths[run_id] = resolved
            payload = _strict_json(paths) + b"\n"
            temp_path = _RUN_PATH_REGISTRY.with_name(_RUN_PATH_REGISTRY.name + f".{os.getpid()}.tmp")
            temp_fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                os.write(temp_fd, payload)
                os.fsync(temp_fd)
            finally:
                os.close(temp_fd)
            os.replace(temp_path, _RUN_PATH_REGISTRY)
            dir_fd = os.open(_RUN_PATH_REGISTRY.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


class Ledger:
    """An append-only dollar ledger at one absolute checkout path.

    A registry beside the ledger may list sibling ledger filenames under the
    ``known_ledger_files`` key. If any listed alternative exists, opening or
    using this ledger is refused so an old per-run ledger cannot be bypassed.
    """

    def __init__(self, path: str | os.PathLike[str], cap: float):
        if isinstance(cap, bool) or not isinstance(cap, (int, float)):
            raise LedgerError("cap must be a finite numeric value, not a string")
        try:
            cap_float = float(cap)
        except (OverflowError, TypeError, ValueError) as exc:
            raise LedgerError("cap must be a finite numeric value") from exc
        if not math.isfinite(cap_float) or cap_float <= 0 or cap_float > HARD_CAP_USD:
            raise LedgerError(f"cap must be finite, greater than zero, and no more than ${HARD_CAP_USD:.2f}")
        try:
            supplied_path = Path(path)
        except (TypeError, ValueError) as exc:
            raise LedgerError("path must be an absolute file path") from exc
        if not supplied_path.is_absolute():
            raise LedgerError("ledger path must be absolute")
        resolved = supplied_path.resolve(strict=False)
        tmp_root = Path("/tmp").resolve()
        if resolved == tmp_root or tmp_root in resolved.parents:
            raise LedgerError("ledger path may not be under /tmp")
        if resolved.exists() and not resolved.is_file():
            raise LedgerError("ledger path must name one regular file")
        self.path = resolved
        self.cap = cap_float
        self._cap_decimal = Decimal(str(cap_float))
        self._lock_path = self.path.parent / ".paid-ledger.lock"
        self._check_sibling_registry()

    def _check_sibling_registry(self) -> None:
        registry_path = self.path.parent / REGISTRY_FILENAME
        if not registry_path.exists():
            return
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"),
                                  parse_constant=_reject_json_constant)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise LedgerError(f"cannot read ledger registry {registry_path}: {exc}") from exc
        if (not isinstance(registry, dict)
                or registry.get("schema") != REGISTRY_SCHEMA
                or not isinstance(registry.get("known_ledger_files"), list)):
            raise LedgerError(f"invalid ledger registry {registry_path}")
        for name in registry["known_ledger_files"]:
            if not isinstance(name, str) or not name or Path(name).name != name or name in (".", ".."):
                raise LedgerError("registry entries must be sibling filenames")
            candidate = (self.path.parent / name).resolve(strict=False)
            if candidate != self.path and candidate.is_file():
                raise LedgerError(f"alternate ledger exists at registered sibling {candidate}")

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            if self.path.exists() and not self.path.is_file():
                raise LedgerError("ledger path must name one regular file")
            self._check_sibling_registry()
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def _read_locked(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not self.path.exists():
            return [], {"had_data": False, "ends_newline": True, "valid_bytes": 0,
                        "torn_last_line_ignored": False}
        try:
            data = self.path.read_bytes()
        except OSError as exc:
            raise LedgerError(f"cannot read ledger {self.path}: {exc}") from exc
        if not data:
            return [], {"had_data": False, "ends_newline": True, "valid_bytes": 0,
                        "torn_last_line_ignored": False}
        ends_newline = data.endswith(b"\n")
        raw_lines = data.split(b"\n")
        if ends_newline:
            raw_lines = raw_lines[:-1]
        events: list[dict[str, Any]] = []
        valid_bytes = 0
        torn = False
        for index, raw in enumerate(raw_lines):
            is_last_unterminated = index == len(raw_lines) - 1 and not ends_newline
            try:
                row = json.loads(raw.decode("utf-8"), parse_constant=_reject_json_constant)
                if not isinstance(row, dict):
                    raise ValueError("JSONL record must be an object")
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
                if is_last_unterminated:
                    torn = True
                    break
                raise LedgerCorruptionError(f"invalid complete ledger line {index + 1}: {exc}") from exc
            events.append(row)
            valid_bytes += len(raw) + (1 if index < len(raw_lines) - 1 or ends_newline else 0)
        if torn:
            warnings.warn("ignored torn final paid-ledger JSONL line", RuntimeWarning, stacklevel=3)
        return events, {"had_data": True, "ends_newline": ends_newline,
                        "valid_bytes": valid_bytes, "torn_last_line_ignored": torn}

    @staticmethod
    def _attempts(events: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        attempts: dict[str, dict[str, Any]] = {}
        for line_no, row in enumerate(events, 1):
            event = row.get("event")
            try:
                attempt_id = _token(row.get("attempt_id"))
                if event == "reserve":
                    if attempt_id in attempts:
                        raise LedgerCorruptionError(f"duplicate reservation for attempt {attempt_id}")
                    kind = row.get("kind")
                    meta = row.get("meta")
                    reserve = _decimal_amount(row.get("reserve_usd"), "reserve_usd", allow_zero=False)
                    if not isinstance(kind, str) or not kind.strip() or not isinstance(meta, dict):
                        raise LedgerCorruptionError("reserve event has invalid kind or metadata")
                    attempts[attempt_id] = {"status": "reserved", "kind": kind,
                                            "reserve": reserve, "meta": meta}
                    if "subcap_key" in row:
                        key = row.get("subcap_key")
                        subcap = _decimal_amount(row.get("subcap_usd"), "subcap_usd", allow_zero=False)
                        if not isinstance(key, str) or not key.strip():
                            raise LedgerCorruptionError("reserve event has invalid subcap key")
                        attempts[attempt_id].update({"subcap_key": key, "subcap": subcap})
                elif event == "settle":
                    current = attempts.get(attempt_id)
                    if current is None or current["status"] != "reserved":
                        raise LedgerCorruptionError(f"settlement without active reserve for {attempt_id}")
                    actual = _decimal_amount(row.get("actual_usd"), "actual_usd", allow_zero=True)
                    usage_sha256 = row.get("usage_sha256")
                    if not isinstance(usage_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", usage_sha256):
                        raise LedgerCorruptionError("settle event has invalid usage digest")
                    current.update({"status": "settled", "actual": actual,
                                    "usage_sha256": usage_sha256})
                elif event == "settlement_overrun":
                    current = attempts.get(attempt_id)
                    if current is None or current["status"] != "reserved":
                        raise LedgerCorruptionError(f"overrun without active reserve for {attempt_id}")
                    actual = _decimal_amount(row.get("actual_usd"), "actual_usd", allow_zero=True)
                    digest = row.get("usage_sha256")
                    if (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
                            or not isinstance(row.get("ts"), str) or not row["ts"]):
                        raise LedgerCorruptionError("settlement overrun has invalid usage digest or timestamp")
                    if row.get("final") is True:
                        current.update({"status": "overrun", "actual": actual,
                                        "usage_sha256": digest})
                    else:
                        # Read legacy diagnostic overruns that left the reserve active.
                        current.setdefault("overruns", []).append({"actual": actual,
                                                                    "usage_sha256": digest})
                elif event == "release":
                    current = attempts.get(attempt_id)
                    if current is None or current["status"] != "reserved":
                        raise LedgerCorruptionError(f"release without active reserve for {attempt_id}")
                    status = row.get("http_status")
                    body_sha256 = row.get("body_sha256")
                    if (isinstance(status, bool) or not isinstance(status, int)
                            or not 400 <= status < 500 or status == 429
                            or not isinstance(body_sha256, str)
                            or not re.fullmatch(r"[0-9a-f]{64}", body_sha256)):
                        raise LedgerCorruptionError("release event has invalid status or body digest")
                    current.update({"status": "released", "http_status": status,
                                    "body_sha256": body_sha256})
                elif event == "retire":
                    current = attempts.get(attempt_id)
                    if current is None or current["status"] != "reserved":
                        raise LedgerCorruptionError(f"retirement without active reserve for {attempt_id}")
                    who, reason = row.get("who"), row.get("reason")
                    if not isinstance(who, str) or not who.strip() or not isinstance(reason, str) or not reason.strip():
                        raise LedgerCorruptionError("retire event has invalid operator or reason")
                    if not isinstance(row.get("ts"), str) or not row["ts"]:
                        raise LedgerCorruptionError("retire event has no timestamp")
                    current.update({"status": "retired", "who": who, "reason": reason})
                else:
                    raise LedgerCorruptionError(f"unknown ledger event {event!r}")
            except LedgerError as exc:
                if isinstance(exc, LedgerCorruptionError):
                    raise
                raise LedgerCorruptionError(f"invalid ledger event on line {line_no}: {exc}") from exc
            except (KeyError, TypeError, ValueError) as exc:
                raise LedgerCorruptionError(f"invalid ledger event on line {line_no}: {exc}") from exc
        return attempts

    @staticmethod
    def _summary(attempts: Mapping[str, dict[str, Any]], torn: bool) -> dict[str, Any]:
        settled = Decimal("0")
        reserved = Decimal("0")
        by_kind: dict[str, dict[str, Decimal]] = {}
        for attempt in attempts.values():
            kind = attempt["kind"]
            values = by_kind.setdefault(kind, {"settled": Decimal("0"), "reserved": Decimal("0")})
            if attempt["status"] == "settled":
                settled += attempt["actual"]
                values["settled"] += attempt["actual"]
            elif attempt["status"] == "overrun":
                settled += attempt["actual"]
                values["settled"] += attempt["actual"]
            elif attempt["status"] == "retired":
                # Unknown-outcome requests stay charged at their worst-case reserve.
                settled += attempt["reserve"]
                values["settled"] += attempt["reserve"]
            elif attempt["status"] == "reserved":
                reserved += attempt["reserve"]
                values["reserved"] += attempt["reserve"]
        return {"settled": float(settled), "reserved": float(reserved),
                "per_kind": {kind: {key: float(value) for key, value in totals.items()}
                             for kind, totals in sorted(by_kind.items())},
                "torn_last_line_ignored": torn}

    def _append_locked(self, event: dict[str, Any], tail: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if tail.get("torn_last_line_ignored"):
            with self.path.open("r+b") as stream:
                stream.truncate(int(tail["valid_bytes"]))
                stream.flush()
                os.fsync(stream.fileno())
        elif tail.get("had_data") and not tail.get("ends_newline"):
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND)
            try:
                os.write(fd, b"\n")
                os.fsync(fd)
            finally:
                os.close(fd)
        encoded = _strict_json(event) + b"\n"
        new_file = not self.path.exists()
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            offset = 0
            while offset < len(encoded):
                offset += os.write(fd, encoded[offset:])
            os.fsync(fd)
        finally:
            os.close(fd)
        if new_file:
            directory_fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)

    def state(self) -> dict[str, Any]:
        """Return dollar totals; token and call totals belong to the caller."""
        with self._locked():
            events, tail = self._read_locked()
            attempts = self._attempts(events)
            return self._summary(attempts, bool(tail["torn_last_line_ignored"]))

    def reserve(self, worst_case_usd: float, kind: str,
                meta: Mapping[str, Any] | None = None, *,
                subcap_usd: float | None = None,
                subcap_key: str | None = None) -> str:
        amount = _decimal_amount(worst_case_usd, "worst_case_usd", allow_zero=False)
        if not isinstance(kind, str) or not kind.strip():
            raise LedgerError("kind must be a non-empty string")
        normalized_kind = kind.strip()
        if not isinstance(meta, (Mapping, type(None))):
            raise LedgerError("meta must be a JSON object")
        normalized_meta: dict[str, Any] = dict(meta or {})
        _strict_json(normalized_meta)
        if (subcap_usd is None) != (subcap_key is None):
            raise LedgerError("subcap_usd and subcap_key must be supplied together")
        subcap = (_decimal_amount(subcap_usd, "subcap_usd", allow_zero=False)
                  if subcap_usd is not None else None)
        if subcap_key is not None and (not isinstance(subcap_key, str) or not subcap_key.strip()):
            raise LedgerError("subcap_key must be a non-empty string")
        run_id = normalized_meta.get("run_id")
        if run_id is not None:
            if not isinstance(run_id, str) or not run_id.strip():
                raise LedgerError("meta.run_id must be a non-empty string")
            _claim_run_path(run_id, self.path)
        with self._locked():
            events, tail = self._read_locked()
            attempts = self._attempts(events)
            state = self._summary(attempts, bool(tail["torn_last_line_ignored"]))
            total = Decimal(str(state["settled"])) + Decimal(str(state["reserved"]))
            if total + amount > self._cap_decimal:
                raise CapExceededError(
                    f"reservation refused: settled/reserved ${float(total):.8f} plus "
                    f"${float(amount):.8f} exceeds cap ${self.cap:.8f}")
            if subcap is not None:
                scoped = Decimal("0")
                for item in attempts.values():
                    if item.get("subcap_key") != subcap_key:
                        continue
                    if item.get("subcap") != subcap:
                        raise LedgerError(f"sub-cap {subcap_key!r} was already opened with a different limit")
                    scoped += (item["actual"] if item["status"] in ("settled", "overrun") else
                               item["reserve"] if item["status"] in ("reserved", "retired") else
                               Decimal("0"))
                if scoped + amount > subcap:
                    raise CapExceededError(
                        f"reservation refused: sub-cap {subcap_key!r} has ${float(scoped):.8f} "
                        f"committed plus ${float(amount):.8f}, over ${float(subcap):.8f}")
            attempt_id = secrets.token_hex(16)
            while attempt_id in attempts:
                attempt_id = secrets.token_hex(16)
            event = {"event": "reserve", "attempt_id": attempt_id,
                     "kind": normalized_kind, "reserve_usd": float(amount),
                     "meta": normalized_meta, "ts": _utc_now()}
            if subcap_key is not None:
                event.update({"subcap_key": subcap_key, "subcap_usd": float(subcap)})
            self._append_locked(event, tail)
        return attempt_id

    def settle(self, attempt_id: str, actual_usd: float, usage: Any) -> None:
        attempt_id = _token(attempt_id)
        actual = _decimal_amount(actual_usd, "actual_usd", allow_zero=True)
        usage_sha256 = _usage_digest(usage)
        with self._locked():
            events, tail = self._read_locked()
            attempts = self._attempts(events)
            current = attempts.get(attempt_id)
            if current is None:
                raise LedgerError(f"unknown attempt {attempt_id}")
            if current["status"] in ("settled", "overrun"):
                if current["actual"] == actual and current["usage_sha256"] == usage_sha256:
                    return
                raise LedgerError(f"attempt {attempt_id} was already settled or overrun with different values")
            if current["status"] != "reserved":
                raise LedgerError(f"attempt {attempt_id} is {current['status']}, not an active reservation")
            events_state = self._summary(attempts, bool(tail["torn_last_line_ignored"]))
            committed_elsewhere = Decimal(str(events_state["settled"])) + Decimal(str(events_state["reserved"])) - current["reserve"]
            subcap_over = False
            if current.get("subcap_key") is not None:
                scoped = sum((item["actual"] if item["status"] in ("settled", "overrun") else
                              item["reserve"] if item["status"] in ("reserved", "retired") else Decimal("0"))
                             for item in attempts.values()
                             if item.get("subcap_key") == current["subcap_key"] and item is not current)
                subcap_over = scoped + actual > current["subcap"]
            if actual > current["reserve"] or committed_elsewhere + actual > self._cap_decimal or subcap_over:
                self._append_locked({"event": "settlement_overrun", "attempt_id": attempt_id,
                                     "actual_usd": float(actual), "usage_sha256": usage_sha256,
                                     "ts": _utc_now(), "final": True}, tail)
                raise CapExceededError(
                    f"settlement overrun recorded at actual ${float(actual):.8f} against reserve "
                    f"${float(current['reserve']):.8f}; future reservations will use the recorded amount")
            self._append_locked({"event": "settle", "attempt_id": attempt_id,
                                 "actual_usd": float(actual), "usage_sha256": usage_sha256,
                                 "ts": _utc_now()}, tail)

    def release(self, attempt_id: str, http_status: int,
                body: str | bytes | Mapping[str, Any]) -> None:
        attempt_id = _token(attempt_id)
        if (isinstance(http_status, bool) or not isinstance(http_status, int)
                or not 400 <= http_status < 500 or http_status == 429):
            raise ReleaseRefusedError("release requires a known non-429 HTTP 4xx response")
        valid, body_sha256 = _valid_anthropic_error(body)
        if not valid or body_sha256 is None:
            raise ReleaseRefusedError("release requires a parsed Anthropic error object")
        with self._locked():
            events, tail = self._read_locked()
            attempts = self._attempts(events)
            current = attempts.get(attempt_id)
            if current is None:
                raise LedgerError(f"unknown attempt {attempt_id}")
            if current["status"] == "released":
                if current["http_status"] == http_status and current["body_sha256"] == body_sha256:
                    return
                raise LedgerError(f"attempt {attempt_id} was already released with different values")
            if current["status"] != "reserved":
                raise LedgerError(f"attempt {attempt_id} is {current['status']}, not an active reservation")
            self._append_locked({"event": "release", "attempt_id": attempt_id,
                                 "http_status": http_status, "body_sha256": body_sha256,
                                 "ts": _utc_now()}, tail)

    def retire(self, attempt_id: str, who: str, reason: str) -> None:
        attempt_id = _token(attempt_id)
        if not isinstance(who, str) or not who.strip():
            raise LedgerError("who must be a non-empty operator name")
        if not isinstance(reason, str) or not reason.strip():
            raise LedgerError("reason must be a non-empty explanation")
        normalized_who, normalized_reason = who.strip(), reason.strip()
        with self._locked():
            events, tail = self._read_locked()
            attempts = self._attempts(events)
            current = attempts.get(attempt_id)
            if current is None:
                raise LedgerError(f"unknown attempt {attempt_id}")
            if current["status"] == "retired":
                if current["who"] == normalized_who and current["reason"] == normalized_reason:
                    return
                raise LedgerError(f"attempt {attempt_id} was already retired")
            if current["status"] != "reserved":
                raise LedgerError(f"attempt {attempt_id} is {current['status']}, not an active reservation")
            self._append_locked({"event": "retire", "attempt_id": attempt_id,
                                 "who": normalized_who, "reason": normalized_reason,
                                 "ts": _utc_now()}, tail)


def chunks(requests: Iterable[Any], size: int) -> Iterator[list[Any]]:
    """Yield sequential request slices without changing the supplied objects."""
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("chunk size must be a positive integer")
    materialized = list(requests)
    for start in range(0, len(materialized), size):
        yield materialized[start:start + size]


def _durable_jsonl_append(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        if path.exists():
            data = path.read_bytes()
            if data and not data.endswith(b"\n"):
                last_start = data.rfind(b"\n") + 1
                try:
                    json.loads(data[last_start:].decode("utf-8"), parse_constant=_reject_json_constant)
                    fd = os.open(path, os.O_WRONLY | os.O_APPEND)
                    try:
                        os.write(fd, b"\n")
                        os.fsync(fd)
                    finally:
                        os.close(fd)
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                    with path.open("r+b") as stream:
                        stream.truncate(last_start)
                        stream.flush()
                        os.fsync(stream.fileno())
        encoded = _strict_json(dict(row)) + b"\n"
        new_file = not path.exists()
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            offset = 0
            while offset < len(encoded):
                offset += os.write(fd, encoded[offset:])
            os.fsync(fd)
        finally:
            os.close(fd)
        if new_file:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def _max_tokens(request: Any) -> int:
    if not isinstance(request, Mapping):
        raise ValueError("each request must be a JSON object")
    params = request.get("params", request)
    if not isinstance(params, Mapping):
        raise ValueError("request params must be an object")
    value = params.get("max_tokens")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("each request needs a positive integer max_tokens")
    return value


def submit_in_chunks(requests: Iterable[Mapping[str, Any]], size: int, *,
                     ledger: Ledger, model: str,
                     submit: Callable[[list[Mapping[str, Any]], str,
                                       Callable[[str], None]], Mapping[str, Any]],
                     batch_id_path: str | os.PathLike[str],
                     kind: str = "batch", meta: Mapping[str, Any] | None = None,
                     batch: bool = True) -> list[dict[str, Any]]:
    """Reserve, submit, persist each batch ID, and settle one chunk at a time.

    The callback signature is ``submit(chunk, attempt_id, record_batch_id)``.
    After the POST response arrives, the callback must extract its batch ID and
    call ``record_batch_id(id)`` immediately, before parsing result records or
    doing other work. The callback then returns ``actual_usd`` and ``usage``;
    usage is hashed for replay checks but is not retained as a total in the
    ledger. A timeout or any non-validated HTTP outcome leaves its reserve live.
    """
    if not isinstance(ledger, Ledger):
        raise TypeError("ledger must be a Ledger instance")
    if not callable(submit):
        raise TypeError("submit must be callable")
    id_path = Path(batch_id_path)
    if not id_path.is_absolute():
        raise ValueError("batch_id_path must be absolute")
    id_path = id_path.resolve(strict=False)
    if id_path.exists() and not id_path.is_file():
        raise ValueError("batch_id_path must name one file")
    if not isinstance(meta, (Mapping, type(None))):
        raise ValueError("meta must be a JSON object")
    base_meta = dict(meta or {})
    _strict_json(base_meta)
    results: list[dict[str, Any]] = []
    for index, request_chunk in enumerate(chunks(requests, size)):
        serialized = [_strict_json(dict(request)) for request in request_chunk]
        input_bytes = sum(len(value) for value in serialized)
        input_tokens_upper = math.ceil(input_bytes / 3)
        output_tokens = sum(_max_tokens(request) for request in request_chunk)
        try:
            from .pricing import worst_case_cost
        except ImportError:
            from pricing import worst_case_cost
        reserve_usd = worst_case_cost(model, input_tokens_upper, output_tokens, batch=batch)
        attempt_id = ledger.reserve(reserve_usd, kind,
                                    {**base_meta, "chunk_index": index,
                                     "request_count": len(request_chunk)})
        saved: dict[str, str] = {}

        def record_batch_id(batch_id: str) -> None:
            if not isinstance(batch_id, str) or not batch_id.strip():
                raise LedgerError("batch ID must be a non-empty string")
            if saved:
                if saved["batch_id"] == batch_id:
                    return
                raise LedgerError("submit callback tried to replace a durable batch ID")
            normalized = batch_id.strip()
            _durable_jsonl_append(id_path, {"attempt_id": attempt_id,
                                             "batch_id": normalized,
                                             "ts": _utc_now()})
            saved["batch_id"] = normalized

        try:
            response = submit(request_chunk, attempt_id, record_batch_id)
        except HTTPFailure as exc:
            if not saved:
                ledger.release(attempt_id, exc.http_status, exc.body)
            raise
        if not saved:
            raise LedgerError("submit callback returned without durably recording its batch ID")
        if not isinstance(response, Mapping) or "actual_usd" not in response or "usage" not in response:
            raise LedgerError("submit callback must return actual_usd and usage after result collection")
        ledger.settle(attempt_id, response["actual_usd"], response["usage"])
        results.append({"attempt_id": attempt_id, "batch_id": saved["batch_id"],
                        "reserved_usd": reserve_usd,
                        "actual_usd": float(_decimal_amount(response["actual_usd"],
                                                              "actual_usd", allow_zero=True))})
    return results


def listing_report(api_list: Callable[[], Any] | Iterable[Mapping[str, Any]] | Mapping[str, Any]) -> dict[str, Any]:
    """Return a copied, read-only listing report; never adopts or writes state."""
    value = api_list() if callable(api_list) else api_list
    if isinstance(value, Mapping):
        rows = value.get("data", [])
    else:
        rows = value
    if not isinstance(rows, Iterable) or isinstance(rows, (str, bytes, Mapping)):
        raise ValueError("API listing must be an iterable of batch objects")
    copied: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("API listing rows must be objects")
        copied.append(dict(row))
    return {"count": len(copied), "batches": copied}
