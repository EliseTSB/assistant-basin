#!/usr/bin/env python3
"""Run Part 129 stages on two rented Vast RTX 3090 GPUs."""
from __future__ import annotations

import argparse
import io
import json
import os
import shlex
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
API_V0 = "https://console.vast.ai/api/v0"
API_V1 = "https://console.vast.ai/api/v1"
MAX_HOURLY_RATE = 0.30
MIN_RELIABILITY = 0.98
DISK_GB = 90
BOOT_SECONDS = 15 * 60
BAD_OFFERS = {38849859}
BAD_HOSTS = {85323}
MAX_RENT_SECONDS = 150 * 60
RUN_LABEL_PREFIX = "part129-2026-10-30"
REMOTE_ROOT = "/root/part129-run"
REMOTE_RUN_SH = "cloud/vast-part129-2026-10-30/run.sh"
REMOTE_MAIN_SH = "cloud/vast-part129-2026-10-30/run-main.sh"
RETURN_PATHS = ("results/part129/", "results/part132/")
PACKAGE_PATHS = ("part129/config129.py", "part129/collect129.py", "part129/prepare129.py", "part129/score129.py",
                 "part91/push91.py", "part132/neurons132.json", REMOTE_RUN_SH, REMOTE_MAIN_SH,
                 "cloud/vast-part129-2026-10-30/run-part132.sh", "part129/prepared129.pt")


class VastAPIError(RuntimeError):
    """A Vast response could not be used safely."""


def _child_env() -> dict[str, str]:
    """Do not pass the local Vast credential into ssh/scp child environments."""
    return {key: value for key, value in os.environ.items() if key != "VAST_KEY"}


def api_request(method: str, path: str, body: dict | None = None) -> dict:
    try:
        key = Path.home().joinpath(".vast-credits.env").read_text(encoding="utf-8").strip()
    except OSError:
        raise VastAPIError("~/.vast-credits.env is unavailable") from None
    if not key:
        raise VastAPIError("~/.vast-credits.env is empty")
    if path.startswith("/api/v1/"):
        url = "https://console.vast.ai" + path
    else:
        url = API_V0 + path
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(
        url, method=method, data=data,
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        raise VastAPIError(f"Vast API returned HTTP {exc.code}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise VastAPIError(f"Vast API request failed ({type(exc).__name__})") from None
    if not isinstance(payload, dict):
        raise VastAPIError("Vast API returned a non-object response")
    if "http_error" in payload:
        raise VastAPIError(f"Vast API returned HTTP {payload['http_error']}")
    return payload


def list_instances() -> list[dict]:
    """List through v1; a failed or malformed response is an error, never empty."""
    payload = api_request("GET", "/api/v1/instances/")
    instances = payload.get("instances")
    if not isinstance(instances, list) or any(not isinstance(item, dict) for item in instances):
        raise VastAPIError("Vast v1 instance listing has no valid instances list")
    return instances


def instance_remains() -> bool:
    """Fail closed: a v1 listing error means an instance may still be running."""
    try:
        instances = list_instances()
    except Exception:
        return True
    return bool(instances)


def matching_offers() -> list[dict]:
    query = {
        "gpu_name": {"eq": "RTX 3090"}, "num_gpus": {"eq": 2},
        "rentable": {"eq": True},
        "type": "on-demand", "reliability2": {"gte": MIN_RELIABILITY},
        "disk_space": {"gte": DISK_GB}, "cuda_max_good": {"gte": 12.4},
        "dph_total": {"lte": MAX_HOURLY_RATE},
        "order": [["dph_total", "asc"]], "limit": 20,
    }
    payload = api_request("POST", "/bundles/", query)
    offers = payload.get("offers")
    if not isinstance(offers, list):
        raise VastAPIError("Vast offer search returned no offers list")
    return [offer for offer in offers
            if isinstance(offer, dict)
            and offer.get("gpu_name") == "RTX 3090"
            and int(offer.get("num_gpus", 0)) == 2
            and float(offer.get("reliability2", 0)) >= MIN_RELIABILITY
            and float(offer.get("dph_total", float("inf"))) <= MAX_HOURLY_RATE
            and float(offer.get("disk_space", 0)) >= DISK_GB
            and bool(offer.get("rentable"))]


def _instance_payload(instance_id: str) -> dict:
    payload = api_request("GET", f"/instances/{instance_id}/")
    value = payload.get("instances", payload)
    if isinstance(value, list):
        value = next((row for row in value if str(row.get("id")) == instance_id), None)
    if not isinstance(value, dict):
        raise VastAPIError("Vast instance lookup returned no instance object")
    return value


def _ssh_details(instance: dict) -> tuple[str, int] | None:
    # ssh_direct: the proxy host refused connections, the public address answers.
    host = instance.get("public_ipaddr") or instance.get("ssh_host")
    port = None
    ports = instance.get("ports")
    candidates = ports.get("22/tcp", []) if isinstance(ports, dict) else []
    if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict):
        port = candidates[0].get("HostPort") or candidates[0].get("host_port")
    if not port:
        host = instance.get("ssh_host") or host
        port = instance.get("ssh_port")
    if host and port:
        try:
            return str(host), int(port)
        except (TypeError, ValueError):
            return None
    return None


def wait_for_ssh(instance_id: str, deadline: float) -> tuple[str, int]:
    while time.monotonic() < deadline:
        try:
            instance = _instance_payload(instance_id)
            details = _ssh_details(instance)
            if "Error response" in str(instance.get("status_msg") or ""):
                raise RuntimeError("the host could not start the container")
            if details and instance.get("actual_status") == "running":
                probe = subprocess.run(_ssh_args(details[0], details[1], "true"), timeout=30,
                                       capture_output=True, env=_child_env())
                if probe.returncode == 0:  # "running" comes before the SSH port answers
                    return details
            if instance.get("actual_status") in {"exited", "error", "failed"}:
                raise RuntimeError("rented instance entered a terminal error state")
        except VastAPIError:
            pass
        time.sleep(min(10, max(0, deadline - time.monotonic())))
    raise TimeoutError("Vast instance did not become SSH-ready before the rental cutoff")


def _ssh_args(host: str, port: int, remote_command: str) -> list[str]:
    return ["ssh", "-i", str(Path.home() / ".ssh/vast_ed25519"), "-o", "IdentitiesOnly=yes",
            "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=10",
            "-p", str(port), f"root@{host}", remote_command]


def _remaining(deadline: float) -> float:
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise TimeoutError("Part 129 reached the 150-minute rental time limit")
    return seconds


def _make_archive() -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for relative in PACKAGE_PATHS:
            source = ROOT / relative
            archive_name = "results/part129/prepared129.pt" if relative == "part129/prepared129.pt" else relative
            if not source.is_file():
                raise FileNotFoundError(f"required upload file is missing: {relative}")
            archive.add(source, arcname=archive_name)
    return stream.getvalue()


def _remote_run(host: str, port: int, deadline: float, stage: str, batch: int) -> None:
    remote_dir = shlex.quote(REMOTE_ROOT)
    mkdir = f"mkdir -p {remote_dir} && tar -xzf - -C {remote_dir}"
    subprocess.run(_ssh_args(host, port, mkdir), input=_make_archive(), check=True,
                   timeout=_remaining(deadline), env=_child_env())
    script = REMOTE_RUN_SH if stage == "pilot" else REMOTE_MAIN_SH
    suffix = f" {batch}" if stage == "main" else ""
    command = f"cd {remote_dir} && bash {shlex.quote(script)}{suffix}"
    subprocess.run(_ssh_args(host, port, command), check=True,
                   timeout=_remaining(deadline), env=_child_env())


def _copy_results(host: str, port: int, deadline: float) -> None:
    remote_dir = shlex.quote(REMOTE_ROOT)
    paths = " ".join(shlex.quote(path) for path in RETURN_PATHS)
    command = (f"cd {remote_dir} && existing=$(for path in {paths}; do "
               "[ -d \"$path\" ] && printf '%s ' \"$path\"; done) && "
               "[ -n \"$existing\" ] && tar -czf - $existing")
    result = subprocess.run(_ssh_args(host, port, command), check=True,
                            stdout=subprocess.PIPE, timeout=_remaining(deadline),
                            env=_child_env())
    with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            path = Path(member.name)
            allowed = any(member.name.startswith(prefix.rstrip("/") + "/") for prefix in RETURN_PATHS)
            if path.is_absolute() or ".." in path.parts or not allowed:
                raise RuntimeError("remote result archive contains an unexpected path")
        archive.extractall(ROOT, members=members)
    print("Copied Part 129 and Part 132 results", flush=True)

def _copy_results_with_retry(host: str, port: int, deadline: float, attempts: int = 3) -> None:
    for attempt in range(1, attempts + 1):
        try:
            _copy_results(host, port, deadline)
            return
        except Exception as exc:
            print(f"copy-back attempt {attempt}/{attempts} failed: {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)
            if attempt == attempts:
                raise

def _should_destroy(stage: str, exit_code: int, copy_safe: bool, cap_reached: bool) -> bool:
    # Copy-back failure keeps the instance available for retries until the rental
    # cutoff; at cutoff it must still be destroyed even though copy_safe is false.
    return cap_reached or (copy_safe and (stage == "main" or (stage == "pilot" and exit_code != 0)))


def _destroy_tracked(instance_ids: set[str], label: str) -> None:
    candidates = set(instance_ids)
    try:
        candidates.update(str(item["id"]) for item in list_instances()
                          if item.get("label") == label and item.get("id") is not None)
    except Exception:
        pass
    for instance_id in sorted(candidates):
        try:
            api_request("DELETE", f"/instances/{instance_id}/")
            print(f"Requested destroy for Part 129 instance {instance_id}", flush=True)
        except Exception as exc:
            print(f"Destroy request failed for Part 129 instance {instance_id}: "
                  f"{type(exc).__name__}", file=sys.stderr, flush=True)


def run(stage: str, batch: int) -> int:
    label = f"{RUN_LABEL_PREFIX}-{os.getpid()}-{int(time.time())}"
    instance_ids: set[str] = set()
    exit_code = 0
    copy_safe = False
    rent_started: float | None = None
    info_path = ROOT / "results/part129/instance.json"
    try:
        if stage == "main" or os.environ.get("VAST_REUSE"):
            if not info_path.is_file():
                raise VastAPIError("pilot instance metadata is missing")
            saved = json.loads(info_path.read_text(encoding="utf-8"))
            reuse = str(saved["id"])
            deadline_epoch = float(saved["deadline"])
            rent_started = time.monotonic()
            instance_ids.add(reuse)
            remaining = deadline_epoch - time.time()
            if remaining <= 0:
                raise VastAPIError("150-minute rental cap has elapsed")
            host, port = wait_for_ssh(reuse, time.monotonic() + remaining)
        else:
            offers = matching_offers()
            offers = [o for o in offers if o.get("id") not in BAD_OFFERS and o.get("host_id") not in BAD_HOSTS]
            if not offers:
                raise VastAPIError("no two-GPU RTX 3090 offer meets the rate, reliability, and disk limits")
            rent_started = time.monotonic()
            deadline_epoch = time.time() + MAX_RENT_SECONDS
            host = port = None
            for offer in offers[:5]:
                rate = float(offer["dph_total"])
                print(f"Selected offer {offer['id']}: ${rate:.3f}/hour, reliability "
                      f"{float(offer['reliability2']):.3f}, disk {DISK_GB} GB", flush=True)
                try:
                    rented = api_request("PUT", f"/asks/{offer['id']}/", {
                        "client_id": "me", "image": "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
                        "disk": DISK_GB, "runtype": "ssh_direct", "label": label,
                        "onstart": "touch ~/.no_auto_tmux"})
                except VastAPIError as exc:
                    print(f"Offer {offer['id']} refused: {exc}", flush=True)
                    continue
                instance_id = rented.get("new_contract")
                if instance_id is not None:
                    instance_ids.add(str(instance_id))
                if rented.get("success") is False or instance_id is None:
                    raise VastAPIError("Vast did not return a new instance id")
                try:
                    deadline_mono = time.monotonic() + max(1, deadline_epoch - time.time())
                    host, port = wait_for_ssh(str(instance_id), min(deadline_mono, time.monotonic() + BOOT_SECONDS))
                    break
                except Exception as exc:
                    print(f"Offer {offer['id']} did not boot: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
                    _destroy_tracked({str(instance_id)}, label)
                    instance_ids.discard(str(instance_id))
            if host is None:
                raise VastAPIError("no rented machine booted")
            info_path.parent.mkdir(parents=True, exist_ok=True)
            info_path.write_text(json.dumps({"id": str(next(iter(instance_ids))), "deadline": deadline_epoch}) + "\n",
                                            encoding="utf-8")

        deadline_mono = time.monotonic() + max(1, deadline_epoch - time.time())
        print("Uploading Part 129 code and prepared catalog inputs", flush=True)
        try:
            _remote_run(host, port, deadline_mono, stage, batch)
        except Exception as exc:
            exit_code = 2
            print(f"remote run failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        print("Copying captures back", flush=True)
        try:
            _copy_results_with_retry(host, port, deadline_mono)
        except Exception:
            # Retain the instance for the remaining rental window, then let the
            # finally block issue a destroy request at the cutoff.
            remaining = deadline_mono - time.monotonic()
            if remaining > 0:
                print(f"Copy-back failed; retaining instance until rental cutoff ({remaining:.0f}s)",
                      file=sys.stderr, flush=True)
                time.sleep(remaining)
            raise
        copy_safe = True
        print(f"Part 129 {stage} results copied back", flush=True)
    except Exception as exc:
        exit_code = 2
        print(f"vast.py: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    finally:
        cap_reached = rent_started is not None and time.time() >= locals().get("deadline_epoch", float("inf"))
        if rent_started is not None and _should_destroy(stage, exit_code, copy_safe, cap_reached):
            _destroy_tracked(instance_ids, label)
        remains = instance_remains()
        print(f"P129_INSTANCE_REMAINS={'yes' if remains else 'no'} "
              "(v1 listing; listing errors count as yes)", flush=True)
    return exit_code

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pilot","main"), default="pilot")
    parser.add_argument("--batch", choices=(16,32,64), type=int, default=32)
    args = parser.parse_args(argv)
    return run(args.stage,args.batch)


if __name__ == "__main__":
    raise SystemExit(main())
