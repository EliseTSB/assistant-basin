#!/usr/bin/env python3
"""Run an ordered job list on a short-lived Vast instance.

The REST and SSH boundaries are injectable so the lifecycle can be exercised
without credentials, Vast, or a real SSH connection.
"""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


DEFAULT_IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"
REMOTE_ROOT = "/workspace/vastrun"
MAX_HOSTS = 3
DESTROY_RETRIES = 3


class RunnerError(RuntimeError):
    """Base error for a failed run."""


class NoEligibleOffers(RunnerError):
    pass


class NoReadyHost(RunnerError):
    pass


class SpendCapExceeded(RunnerError):
    pass


class WallClockExceeded(RunnerError):
    pass


class RunInterrupted(RunnerError):
    pass


class JobFailed(RunnerError):
    pass


class DestroyConfirmationError(RunnerError):
    pass


class RemoteCommandError(RunnerError):
    def __init__(self, returncode: int, command: str, output: str = "") -> None:
        # Do not include output: remote output can contain unrelated secrets.
        super().__init__(f"remote command failed with exit code {returncode}: {command[:160]}")
        self.returncode = returncode


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _offer_id(offer: dict[str, Any]) -> Any:
    return offer.get("id", offer.get("ask_id"))


def _eligible_offers(offers: list[dict[str, Any]], max_dph: float) -> list[dict[str, Any]]:
    eligible: list[dict[str, Any]] = []
    for offer in offers:
        if offer.get("verified") is not True or offer.get("rentable", True) is not True:
            continue
        if "3090" not in str(offer.get("gpu_name", "")):
            continue
        if offer.get("num_gpus", 1) != 1:
            continue
        if _number(offer.get("reliability2"), -1) < 0.98:
            continue
        if _number(offer.get("cuda_max_good"), -1) < 12.4:
            continue
        if _number(offer.get("disk_space"), -1) < 40:
            continue
        if _number(offer.get("inet_down"), -1) < 200:
            continue
        if _number(offer.get("dph_total"), math.inf) > max_dph:
            continue
        eligible.append(offer)
    return sorted(eligible, key=lambda item: _number(item.get("dph_total"), math.inf))


def _instance_payload(response: Any, instance_id: str | None = None) -> dict[str, Any] | None:
    if not isinstance(response, dict):
        return None
    instances = response.get("instances", response)
    if isinstance(instances, dict):
        if instances.get("id") is not None:
            if instance_id is None or str(instances.get("id")) == str(instance_id):
                return instances
        if instance_id is not None:
            candidate = instances.get(str(instance_id))
            if isinstance(candidate, dict):
                return candidate
        return None
    if isinstance(instances, list):
        for item in instances:
            if isinstance(item, dict) and (instance_id is None or str(item.get("id")) == str(instance_id)):
                return item
    return None


def _instance_payloads(response: Any) -> list[dict[str, Any]] | None:
    if not isinstance(response, dict):
        return None
    instances = response.get("instances", response)
    if isinstance(instances, dict):
        if instances.get("id") is not None:
            return [instances]
        return [value for value in instances.values() if isinstance(value, dict)]
    if isinstance(instances, list):
        return [value for value in instances if isinstance(value, dict)]
    return None


def _instance_ids(response: Any) -> set[str]:
    return {str(value["id"]) for value in (_instance_payloads(response) or [])
            if value.get("id") is not None}


def _rent_instance_id(response: Any) -> str | None:
    if not isinstance(response, dict):
        return None
    value = response.get("new_contract", response.get("instance_id", response.get("id")))
    if isinstance(value, dict):
        value = value.get("id", value.get("instance_id"))
    return str(value) if value is not None else None


def _remote_connection(instance: dict[str, Any]) -> tuple[str, int] | None:
    host = instance.get("ssh_host") or instance.get("public_ipaddr")
    port = instance.get("ssh_port")
    if not port:
        ports = instance.get("ports") or {}
        ssh_ports = ports.get("22") or ports.get(22) if isinstance(ports, dict) else None
        if isinstance(ssh_ports, list) and ssh_ports:
            item = ssh_ports[0]
            if isinstance(item, dict):
                port = item.get("HostPort") or item.get("host_port")
    if host and port:
        try:
            return str(host), int(port)
        except (TypeError, ValueError):
            return None
    return None


def _watchdog_command(wall_clock_min: float) -> str:
    seconds = max(1, math.ceil((wall_clock_min + 10) * 60))
    shutdown = f"sleep {seconds}; poweroff"
    return f"nohup sh -c {shlex.quote(shutdown)} >/dev/null 2>&1 &"


class VastAPI:
    """Small REST adapter matching the v0/v1 shapes in the reference helper."""

    def __init__(self, key_path: str | None = None, timeout: float = 30) -> None:
        path = os.path.expanduser(key_path or "~/.vast-credits.env")
        try:
            with open(path, encoding="utf-8") as key_file:
                self._key = key_file.read().strip()
        except OSError:
            raise RunnerError("could not read the Vast API key file") from None
        if not self._key:
            raise RunnerError("the Vast API key file is empty")
        self.timeout = timeout

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None,
                 *, v1: bool = False) -> Any:
        base = "https://console.vast.ai/api/v1" if v1 else "https://console.vast.ai/api/v0"
        url = base + path
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(
            url,
            method=method,
            data=data,
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            raise RunnerError(f"Vast API returned HTTP {exc.code} for {method} {path}") from None
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RunnerError(f"Vast API request failed for {method} {path}: {type(exc).__name__}") from None

    def offers(self, max_dph: float) -> list[dict[str, Any]]:
        query = {
            "gpu_name": {"eq": "RTX 3090"},
            "num_gpus": {"eq": 1},
            "verified": {"eq": True},
            "rentable": {"eq": True},
            "type": "on-demand",
            "reliability2": {"gte": 0.98},
            "cuda_max_good": {"gte": 12.4},
            "disk_space": {"gte": 40},
            "inet_down": {"gte": 200},
            "dph_total": {"lte": max_dph},
            "order": [["dph_total", "asc"]],
            "limit": 100,
        }
        response = self._request("POST", "/bundles/", query)
        offers = response.get("offers", []) if isinstance(response, dict) else []
        return offers if isinstance(offers, list) else []

    def rent(self, offer_id: Any, watchdog: str, label: str) -> Any:
        body = {
            "client_id": "me",
            "image": DEFAULT_IMAGE,
            "disk": 40,
            "runtype": "ssh_direct",
            "label": label,
            "onstart": watchdog,
        }
        return self._request("PUT", f"/asks/{offer_id}/", body)

    def get_instance(self, instance_id: str) -> Any:
        return self._request("GET", f"/instances/{instance_id}/")

    def destroy(self, instance_id: str) -> Any:
        return self._request("DELETE", f"/instances/{instance_id}/")

    def list_instances(self) -> Any:
        return self._request("GET", "/instances/", v1=True)


class OpenSSHRunner:
    """OpenSSH/scp adapter; it uses the user's configured SSH agent or defaults."""

    def __init__(self, user: str = "root", connect_timeout: int = 10) -> None:
        self.user = user
        self.connect_timeout = connect_timeout

    def _target(self, host: str) -> str:
        return f"{self.user}@{host}"

    def run(self, host: str, port: int, command: str, *, timeout: float,
            check: bool = True) -> str:
        completed = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={self.connect_timeout}",
             "-p", str(port), self._target(host), command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
        if check and completed.returncode:
            raise RemoteCommandError(completed.returncode, command, completed.stdout)
        return completed.stdout

    def upload(self, host: str, port: int, local_path: Path, remote_path: str,
               *, timeout: float) -> None:
        completed = subprocess.run(
            ["scp", "-P", str(port), "-r", str(local_path),
             f"{self._target(host)}:{remote_path}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
        if completed.returncode:
            raise RemoteCommandError(completed.returncode, f"scp {local_path} {remote_path}")

    def download(self, host: str, port: int, remote_path: str, local_path: Path,
                 *, timeout: float) -> None:
        completed = subprocess.run(
            ["scp", "-P", str(port), "-r", f"{self._target(host)}:{remote_path}", str(local_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
        if completed.returncode:
            raise RemoteCommandError(completed.returncode, f"scp {remote_path} {local_path}")


class _Runner:
    def __init__(self, jobfile: Path, config: dict[str, Any], api: Any, ssh: Any, *,
                 clock: Any, sleep: Any, poll_interval: float, ready_interval: float,
                 ready_timeout: float, destroy_delay: float, cancel_event: threading.Event | None) -> None:
        self.jobfile = jobfile.resolve()
        self.base_dir = self.jobfile.parent
        self.config = config
        self.api = api
        self.ssh = ssh
        self.clock = clock
        self.sleep = sleep
        self.poll_interval = poll_interval
        self.ready_interval = ready_interval
        self.ready_timeout = ready_timeout
        self.destroy_delay = destroy_delay
        self.cancel_event = cancel_event
        self.started = clock()
        self.results_dir = self._local_path(config["results_dir"])
        self.workdir = str(PurePosixPath(config.get("remote_workdir", REMOTE_ROOT)))
        self.max_dph = _number(config["max_dph"])
        self.spend_cap = _number(config["spend_cap_usd"])
        self.wall_seconds = _number(config["wall_clock_min"]) * 60
        self.total_spend = 0.0
        self.active: dict[str, Any] | None = None
        self.active_started: float | None = None
        self.current_dph = 0.0
        self.run_label = f"vastrun-{uuid.uuid4().hex[:12]}"
        self.record: dict[str, Any] = {
            "hosts_tried": [],
            "instance_id": None,
            "run_label": self.run_label,
            "dph": None,
            "start": _utc_now(),
            "end": None,
            "estimated_spend_usd": 0.0,
            "jobs": [],
            "destroy_confirmed": None,
        }

    def _local_path(self, path: str | Path) -> Path:
        value = Path(path)
        return value if value.is_absolute() else self.base_dir / value

    def _validate(self) -> None:
        if self.max_dph <= 0 or self.spend_cap <= 0 or self.wall_seconds <= 0:
            raise RunnerError("max_dph, spend_cap_usd, and wall_clock_min must be positive")
        if not isinstance(self.config.get("jobs"), list):
            raise RunnerError("job file must contain a jobs list")
        if not self.workdir.startswith("/") or ".." in PurePosixPath(self.workdir).parts:
            raise RunnerError("remote_workdir must be an absolute path without '..'")
        names: set[str] = set()
        for job in self.config["jobs"]:
            if not isinstance(job, dict) or not job.get("name") or not isinstance(job.get("command"), str):
                raise RunnerError("each job needs a name and command")
            if str(job["name"]) in names:
                raise RunnerError("job names must be unique")
            names.add(str(job["name"]))
            if not isinstance(job.get("env", {}), dict):
                raise RunnerError(f"env for job {job['name']} must be an object")

    def _estimate_spend(self) -> float:
        estimate = self.total_spend
        if self.active_started is not None and self.active is not None:
            estimate += max(0.0, self.clock() - self.active_started) * self.current_dph / 3600
        return estimate

    def _remaining_seconds(self) -> float:
        remaining = self.wall_seconds - (self.clock() - self.started)
        if self.active is not None and self.current_dph > 0:
            remaining = min(remaining, (self.spend_cap - self._estimate_spend()) * 3600 / self.current_dph)
        return remaining

    def _check_limits(self) -> None:
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise RunInterrupted("received SIGINT or SIGTERM")
        if self.clock() - self.started >= self.wall_seconds:
            raise WallClockExceeded("wall-clock limit reached")
        if self._estimate_spend() >= self.spend_cap:
            raise SpendCapExceeded("spend cap reached")

    def _operation_timeout(self, default: float = 30.0) -> float:
        self._check_limits()
        return max(0.1, min(default, self._remaining_seconds()))

    def _sleep_limited(self, seconds: float) -> None:
        self._check_limits()
        amount = min(seconds, max(0.0, self._remaining_seconds()))
        if amount > 0:
            self.sleep(amount)
        self._check_limits()

    def _host_connection(self, instance_id: str) -> tuple[str, int] | None:
        return _remote_connection(_instance_payload(self.api.get_instance(instance_id), instance_id) or {})

    def _wait_ready(self, instance_id: str) -> tuple[str, int] | None:
        deadline = self.clock() + self.ready_timeout
        last_error = "SSH endpoint did not appear"
        while self.clock() < deadline:
            self._check_limits()
            try:
                connection = self._host_connection(instance_id)
                if connection is not None:
                    host, port = connection
                    ready_remaining = deadline - self.clock()
                    if ready_remaining <= 0:
                        break
                    timeout = min(self._operation_timeout(8), ready_remaining)
                    self.ssh.run(host, port, "true", timeout=timeout, check=True)
                    if self.clock() < deadline:
                        return connection
                    last_error = "SSH answered after the readiness deadline"
                last_error = "instance has no SSH endpoint yet"
            except (SpendCapExceeded, WallClockExceeded, RunInterrupted):
                raise
            except Exception as exc:
                last_error = type(exc).__name__
            remaining = deadline - self.clock()
            if remaining > 0:
                self._sleep_limited(min(self.ready_interval, remaining))
        return None

    def _rent_one(self, offer: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        offer_id = _offer_id(offer)
        host_label = f"{self.run_label}-{len(self.record['hosts_tried']) + 1}"
        entry: dict[str, Any] = {
            "offer_id": offer_id,
            "label": host_label,
            "dph": _number(offer.get("dph_total")),
            "instance_id": None,
            "started_at": _utc_now(),
            "dropped_reason": None,
        }
        self.record["hosts_tried"].append(entry)
        rent_started = self.clock()
        try:
            response = self.api.rent(
                offer_id,
                _watchdog_command(_number(self.config["wall_clock_min"])),
                host_label,
            )
            instance_id = _rent_instance_id(response)
        except RunInterrupted:
            entry["dropped_reason"] = "rent was interrupted; checking for an instance created before interruption"
            self._recover_uncertain_rent(entry, host_label, rent_started)
            raise
        except Exception as exc:
            entry["dropped_reason"] = f"rent failed: {type(exc).__name__}"
            self._recover_uncertain_rent(entry, host_label, rent_started)
            return False, entry
        if instance_id is None:
            entry["dropped_reason"] = "rent response did not include an instance id"
            self._recover_uncertain_rent(entry, host_label, rent_started)
            return False, entry

        self._activate(instance_id, entry, rent_started)
        connection = self._wait_ready(instance_id)
        if connection is None:
            entry["dropped_reason"] = f"SSH did not become ready within {self.ready_timeout:g} seconds"
            self._destroy_active()
            return False, entry
        entry["selected"] = True
        self.record["ssh_host"] = connection[0]
        self.record["ssh_port"] = connection[1]
        return True, entry

    def _activate(self, instance_id: str, entry: dict[str, Any], started: float | None = None) -> None:
        entry["instance_id"] = instance_id
        self.active = {"id": instance_id, "entry": entry}
        self.active_started = self.clock() if started is None else started
        self.current_dph = entry["dph"]
        self.record["instance_id"] = instance_id
        self.record["dph"] = self.current_dph
        self.record["destroy_confirmed"] = False

    def _recover_uncertain_rent(self, entry: dict[str, Any], label: str, started: float) -> None:
        """Find and destroy an instance if a timed-out rent request actually succeeded."""
        matches: list[dict[str, Any]] = []
        last_error: str | None = None
        for attempt in range(3):
            try:
                payloads = _instance_payloads(self.api.list_instances())
                if payloads is None:
                    last_error = "Vast returned an unreadable instance list"
                else:
                    matches = [value for value in payloads if value.get("label") == label]
                    if matches:
                        break
            except Exception as exc:
                last_error = type(exc).__name__
            if attempt < 2:
                self.sleep(self.destroy_delay or 1)
        if last_error is not None and not matches:
            raise RunnerError(f"rent outcome is uncertain; could not verify labeled instance state ({last_error})")
        for instance in matches:
            instance_id = instance.get("id")
            if instance_id is None:
                raise RunnerError("rent outcome is uncertain; matching instance has no id")
            entry.setdefault("recovered_instance_ids", []).append(str(instance_id))
            self._activate(str(instance_id), entry, started)
            self._destroy_active()
        if matches:
            entry["dropped_reason"] = "rent response was uncertain; recovered and destroyed the created instance"

    def _destroy_active(self) -> None:
        if self.active is None:
            return
        instance_id = str(self.active["id"])
        if self.active.get("destroy_attempted"):
            message = self.active.get("destroy_error", f"could not confirm instance {instance_id} destroyed")
            raise DestroyConfirmationError(message)
        self.active["destroy_attempted"] = True
        last_error = "instance remained listed"
        confirmed = False
        for attempt in range(1, DESTROY_RETRIES + 1):
            try:
                self.api.destroy(instance_id)
            except Exception as exc:
                last_error = f"destroy request raised {type(exc).__name__}"
            try:
                listed = instance_id in _instance_ids(self.api.list_instances())
                if not listed:
                    confirmed = True
                    break
                last_error = "instance remained listed"
            except Exception as exc:
                last_error = f"instance confirmation raised {type(exc).__name__}"
            if attempt < DESTROY_RETRIES and self.destroy_delay > 0:
                self.sleep(self.destroy_delay)

        active_started = self.active_started if self.active_started is not None else self.clock()
        elapsed = max(0.0, self.clock() - active_started)
        self.total_spend += elapsed * self.current_dph / 3600
        self.active_started = None
        self.record["estimated_spend_usd"] = self.total_spend
        self.record["destroy_confirmed"] = confirmed
        if confirmed:
            self.active = None
            self.active_started = None
            self.current_dph = 0.0
            return
        message = f"could not confirm instance {instance_id} destroyed after {DESTROY_RETRIES} attempts: {last_error}"
        self.active["destroy_error"] = message
        raise DestroyConfirmationError(message)

    def _ensure_ready_host(self) -> tuple[str, int]:
        raw_offers = self.api.offers(self.max_dph)
        offers = _eligible_offers(raw_offers if isinstance(raw_offers, list) else [], self.max_dph)
        if not offers:
            raise NoEligibleOffers("no offers matched the runner's requirements")
        for offer in offers[:MAX_HOSTS]:
            self._check_limits()
            selected, entry = self._rent_one(offer)
            if selected:
                return str(self.record["ssh_host"]), int(self.record["ssh_port"])
            if entry.get("dropped_reason") and "rent failed" not in entry["dropped_reason"]:
                self.record["last_dropped_reason"] = entry["dropped_reason"]
        raise NoReadyHost(f"none of the first {min(MAX_HOSTS, len(offers))} offers became SSH-ready")

    def _ssh_run(self, host: str, port: int, command: str, *, check: bool = True,
                 timeout: float = 30.0) -> str:
        self._check_limits()
        output = self.ssh.run(host, port, command, timeout=self._operation_timeout(timeout), check=check)
        self._check_limits()
        return output

    def _upload_files(self, host: str, port: int) -> None:
        self._ssh_run(host, port, f"mkdir -p {shlex.quote(self.workdir)}", timeout=20)
        uploads = self.config.get("upload_files", [])
        if not isinstance(uploads, list):
            raise RunnerError("upload_files must be a list")
        for upload in uploads:
            if isinstance(upload, str):
                source = upload
                destination = PurePosixPath(upload).name
            elif isinstance(upload, dict):
                source = upload.get("local", upload.get("src", upload.get("source")))
                destination = upload.get("remote", upload.get("dest", upload.get("destination")))
                if not source:
                    raise RunnerError("each upload_files entry needs a local/source path")
                destination = destination or PurePosixPath(str(source)).name
            else:
                raise RunnerError("upload_files entries must be paths or objects")
            local = self._local_path(str(source)).resolve()
            if not local.exists():
                raise RunnerError(f"upload source does not exist: {source}")
            remote = PurePosixPath(str(destination))
            remote_path = str(remote) if remote.is_absolute() else str(PurePosixPath(self.workdir) / remote)
            if ".." in PurePosixPath(remote_path).parts:
                raise RunnerError("upload destinations may not contain '..'")
            remote_parent = str(PurePosixPath(remote_path).parent)
            self._ssh_run(host, port, f"mkdir -p {shlex.quote(remote_parent)}", timeout=20)
            self._check_limits()
            self.ssh.upload(host, port, local, remote_path, timeout=self._operation_timeout(300))
            self._check_limits()

    def _safe_job_stem(self, name: str) -> str:
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
        return stem or "job"

    def _launch_job(self, host: str, port: int, job: dict[str, Any], log_path: str,
                    status_path: str) -> None:
        env = job.get("env", {})
        exports: list[str] = []
        for key, value in env.items():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(key)):
                raise RunnerError(f"invalid environment variable name in job {job['name']}")
            exports.append(f"export {key}={shlex.quote(str(value))}")
        body = "\n".join([f"cd {shlex.quote(self.workdir)}", *exports,
                           f"sh -c {shlex.quote(str(job['command']))}", "rc=$?",
                           f"printf '\\nrc=%s\\n' \"$rc\" >> {shlex.quote(log_path)}",
                           f"printf '%s\\n' \"$rc\" > {shlex.quote(status_path)}",
                           "exit \"$rc\""])
        parent = str(PurePosixPath(log_path).parent)
        self._ssh_run(host, port, f"mkdir -p {shlex.quote(parent)}", timeout=20)
        self._ssh_run(host, port, f"rm -f {shlex.quote(status_path)}", timeout=20)
        command = f"nohup sh -c {shlex.quote(body)} > {shlex.quote(log_path)} 2>&1 < /dev/null & echo $!"
        self._ssh_run(host, port, command, timeout=20)

    def _copy_back(self, host: str, port: int, job: dict[str, Any], log_path: str) -> list[str]:
        expected = job.get("expected_result_paths", job.get("expected_results", [])) or []
        if not isinstance(expected, list):
            raise RunnerError(f"expected result paths for job {job['name']} must be a list")
        remote_paths: list[str] = []
        for expected_path in expected:
            path = PurePosixPath(str(expected_path))
            if ".." in path.parts:
                raise RunnerError(f"unsafe expected result path for job {job['name']}")
            remote_paths.append(str(path) if path.is_absolute() else str(PurePosixPath(self.workdir) / path))
        remote_paths.append(log_path)
        copied: list[str] = []
        for remote in remote_paths:
            path = PurePosixPath(remote)
            workdir_path = PurePosixPath(self.workdir)
            if path.is_absolute() and path.parts[:len(workdir_path.parts)] == workdir_path.parts:
                local_relative = path.relative_to(workdir_path)
            else:
                local_relative = PurePosixPath(path.name) if path.is_absolute() else path
            local = self.results_dir.joinpath(*local_relative.parts)
            local.parent.mkdir(parents=True, exist_ok=True)
            self._check_limits()
            self.ssh.download(host, port, remote, local, timeout=self._operation_timeout(300))
            self._check_limits()
            copied.append(str(local.relative_to(self.results_dir)))
        return copied

    def _run_jobs(self, host: str, port: int) -> None:
        setup = self.config.get("setup_command")
        if setup:
            self._ssh_run(host, port, f"cd {shlex.quote(self.workdir)} && {setup}", timeout=300)
        for job in self.config["jobs"]:
            self._check_limits()
            name = str(job["name"])
            stem = self._safe_job_stem(name)
            log_path = str(PurePosixPath(self.workdir) / "logs" / f"{stem}.log")
            status_path = f"{log_path}.rc"
            job_started = self.clock()
            record: dict[str, Any] = {
                "name": name,
                "rc": None,
                "seconds": None,
                "copied_back": [],
            }
            self.record["jobs"].append(record)
            self._launch_job(host, port, job, log_path, status_path)
            while True:
                self._check_limits()
                output = self._ssh_run(host, port, f"cat {shlex.quote(status_path)} 2>/dev/null || true",
                                       check=False, timeout=15)
                match = re.search(r"^(-?\d+)\s*$", output, flags=re.MULTILINE)
                if match:
                    record["rc"] = int(match.group(1))
                    break
                self._sleep_limited(self.poll_interval)
            record["seconds"] = round(max(0.0, self.clock() - job_started), 3)
            record["copied_back"] = self._copy_back(host, port, job, log_path)
            if record["rc"] != 0:
                raise JobFailed(f"job {name} exited with rc={record['rc']}")

    def _write_record(self) -> None:
        self.record["end"] = _utc_now()
        self.record["estimated_spend_usd"] = round(self._estimate_spend(), 6)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        record_path = self.results_dir / "run-record.json"
        temporary = record_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(record_path)

    def run(self) -> dict[str, Any]:
        failure: BaseException | None = None
        try:
            self._validate()
            self.results_dir.mkdir(parents=True, exist_ok=True)
            host, port = self._ensure_ready_host()
            self._upload_files(host, port)
            self._run_jobs(host, port)
        except BaseException as exc:
            failure = exc
            self.record["error"] = str(exc) if not isinstance(exc, (KeyboardInterrupt, SystemExit)) else type(exc).__name__
        finally:
            try:
                self._destroy_active()
            except BaseException as exc:
                self.record["destroy_error"] = str(exc)
                failure = exc
            try:
                self._write_record()
            except BaseException as exc:
                if failure is None:
                    failure = exc
        if failure is not None:
            raise failure
        return self.record


def run_jobfile(jobfile: str | Path, *, api: Any | None = None, ssh: Any | None = None,
                clock: Any = time.monotonic, sleep: Any = time.sleep, poll_interval: float = 30,
                ready_interval: float = 5, ready_timeout: float = 180, destroy_delay: float = 1,
                cancel_event: threading.Event | None = None) -> dict[str, Any]:
    path = Path(jobfile)
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"could not read job file ({type(exc).__name__})") from None
    if not isinstance(config, dict):
        raise RunnerError("job file root must be a JSON object")
    api_adapter = api if api is not None else VastAPI()
    ssh_adapter = ssh if ssh is not None else OpenSSHRunner(str(config.get("ssh_user", "root")))
    return _Runner(
        path, config, api_adapter, ssh_adapter, clock=clock, sleep=sleep,
        poll_interval=poll_interval, ready_interval=ready_interval,
        ready_timeout=ready_timeout, destroy_delay=destroy_delay, cancel_event=cancel_event,
    ).run()


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python cloud/vastrun.py JOBFILE.json", file=sys.stderr)
        return 2
    cancelled = threading.Event()
    old_handlers: dict[int, Any] = {}

    def request_stop(signum: int, frame: Any) -> None:
        del frame
        cancelled.set()
        raise RunInterrupted(f"received signal {signum}")

    for sig in (signal.SIGINT, signal.SIGTERM):
        old_handlers[sig] = signal.signal(sig, request_stop)
    try:
        record = run_jobfile(args[0], cancel_event=cancelled)
    except BaseException as exc:
        print(f"vastrun: {exc}", file=sys.stderr)
        return 1
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    print(f"vastrun finished; estimated spend ${record['estimated_spend_usd']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
