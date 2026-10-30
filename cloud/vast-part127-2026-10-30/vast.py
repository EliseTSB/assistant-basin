#!/usr/bin/env python3
"""Run Part 127 on one rented Vast RTX 3090 and return its captures."""
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
MIN_RELIABILITY = 0.99
DISK_GB = 40
BOOT_SECONDS = 15 * 60  # the image pull alone took about 9 minutes
BAD_OFFERS = {38849859}  # 2026-10-30: OCI runtime error on start
MAX_RENT_SECONDS = 180 * 60  # $0.90 at the maximum accepted hourly rate, for Parts 127 and 125 together.
RUN_LABEL_PREFIX = "part127-2026-10-30"
REMOTE_ROOT = "/root/part127-run"
REMOTE_RUN_SH = "cloud/vast-part127-2026-10-30/run.sh"
RETURN_PATHS = ("results/part127/pilot-b8", "results/part127/main", "results/part125/main")
PACKAGE_PATHS = (
    "part127/run127.py", "part127/label127.py", "part127/score127.py",
    "part92/common92.py", "part92/label92.py",
    "part92b/label92b.py", "part92b/known92b.jsonl",
    "part119/run119.py", "part119/rows119.py",
    "part119b/run119b.py", "part119b/rows119b.py", "part119b/gate119b.py",
    "part119c/run119c.py", "part119c/rows119c.py",
    "part85/checkpoints.py", "judge_client.py",
    "results/part92/step2/neurons.json", REMOTE_RUN_SH,
    "part125/run125.py",
    "part125/session125.py",
    "part125/label125.py",
    "part125/score125.py",
    "part123/fit123.py",
    "part123/run123.py",
    "part123/label123.py",
    "part119/label119.py",
    "part119/known119.jsonl",
    "results/part92/step2/kinds.json",
)


class VastAPIError(RuntimeError):
    """A Vast response could not be used safely."""


def _child_env() -> dict[str, str]:
    """Do not pass the local Vast credential into ssh/scp child environments."""
    return {key: value for key, value in os.environ.items() if key != "VAST_KEY"}


def api_request(method: str, path: str, body: dict | None = None) -> dict:
    key = os.environ.get("VAST_KEY")
    if not key:
        raise VastAPIError("VAST_KEY is not set in the environment")
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
        "gpu_name": {"eq": "RTX 3090"}, "num_gpus": {"eq": 1},
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
            and int(offer.get("num_gpus", 0)) == 1
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
    host = instance.get("ssh_host") or instance.get("public_ipaddr")
    port = instance.get("ssh_port")
    if not port:
        ports = instance.get("ports")
        candidates = ports.get("22/tcp", []) if isinstance(ports, dict) else []
        if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict):
            port = candidates[0].get("HostPort") or candidates[0].get("host_port")
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
            if details and instance.get("actual_status") == "running":
                probe = subprocess.run(_ssh_args(details[0], details[1], "true"), timeout=30,
                                       capture_output=True, env=_child_env())
                if probe.returncode == 0:  # "running" comes before the SSH port answers
                    return details
            if instance.get("actual_status") in {"exited", "error", "failed"}:
                raise VastAPIError("rented instance entered a terminal error state")
        except VastAPIError:
            pass
        time.sleep(min(10, max(0, deadline - time.monotonic())))
    raise TimeoutError("Vast instance did not become SSH-ready before the rental cutoff")


def _ssh_args(host: str, port: int, remote_command: str) -> list[str]:
    return ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=10",
            "-p", str(port), f"root@{host}", remote_command]


def _remaining(deadline: float) -> float:
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise TimeoutError("Part 127 reached the $0.40 rental time limit")
    return seconds


def _make_archive() -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for relative in PACKAGE_PATHS:
            source = ROOT / relative
            if not source.is_file():
                raise FileNotFoundError(f"required upload file is missing: {relative}")
            archive.add(source, arcname=relative)
    return stream.getvalue()


def _remote_run(host: str, port: int, deadline: float) -> None:
    remote_dir = shlex.quote(REMOTE_ROOT)
    mkdir = f"mkdir -p {remote_dir} && tar -xzf - -C {remote_dir}"
    subprocess.run(_ssh_args(host, port, mkdir), input=_make_archive(), check=True,
                   timeout=_remaining(deadline), env=_child_env())
    command = (f"cd {remote_dir} && bash {shlex.quote(REMOTE_RUN_SH)}")
    subprocess.run(_ssh_args(host, port, command), check=True,
                   timeout=_remaining(deadline), env=_child_env())


def _copy_results(host: str, port: int, deadline: float) -> None:
    remote_dir = shlex.quote(REMOTE_ROOT)
    paths = " ".join(shlex.quote(path) for path in RETURN_PATHS)
    command = (f"cd {remote_dir} && existing=$(ls -d {paths} 2>/dev/null) && "
               "tar -czf - $existing")
    result = subprocess.run(_ssh_args(host, port, command), check=True,
                            stdout=subprocess.PIPE, timeout=_remaining(deadline),
                            env=_child_env())
    with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts or not member.name.startswith(("results/part127/", "results/part125/")):
                raise RuntimeError("remote result archive contains an unexpected path")
        archive.extractall(ROOT, members=members)
    for relative in RETURN_PATHS:
        name = "run-meta.json" if "part125" in relative else "run_meta.json"
        meta = ROOT / relative / name
        status = json.loads(meta.read_text(encoding="utf-8")).get("status") if meta.is_file() else "missing"
        print(f"{relative}: {status}", flush=True)


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
            print(f"Requested destroy for Part 127 instance {instance_id}", flush=True)
        except Exception as exc:
            print(f"Destroy request failed for Part 127 instance {instance_id}: "
                  f"{type(exc).__name__}", file=sys.stderr, flush=True)


def run() -> int:
    label = f"{RUN_LABEL_PREFIX}-{os.getpid()}-{int(time.time())}"
    instance_ids: set[str] = set()
    exit_code = 0
    rent_started: float | None = None
    try:
        offers = matching_offers()
        if not offers:
            raise VastAPIError("no one-GPU RTX 3090 offer meets the rate, reliability, and disk limits")
        offers = [o for o in offers if o["id"] not in BAD_OFFERS]
        host = port = None

        # Start the deadline before the request, so boot, transfer, download, and
        # both stages all count against the same $0.40 cap.
        rent_started = time.monotonic()
        deadline = rent_started + MAX_RENT_SECONDS
        for offer in offers[:5]:  # a host can fail to start the container; try the next one
            rate = float(offer["dph_total"])
            print(f"Selected offer {offer['id']}: ${rate:.3f}/hour, "
                  f"reliability {float(offer['reliability2']):.3f}, disk {DISK_GB} GB", flush=True)
            try:
                rented = api_request("PUT", f"/asks/{offer['id']}/", {
                    "client_id": "me", "image": "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
                    "disk": DISK_GB, "runtype": "ssh_direct", "label": label,
                    "onstart": "touch ~/.no_auto_tmux"})
            except VastAPIError as exc:  # taken since the search
                print(f"Offer {offer['id']} refused: {exc}", flush=True)
                continue
            instance_id = rented.get("new_contract")
            if instance_id is not None:
                instance_ids.add(str(instance_id))
            if rented.get("success") is False or instance_id is None:
                raise VastAPIError("Vast did not return a new instance id")
            try:
                host, port = wait_for_ssh(str(instance_id), min(deadline, time.monotonic() + BOOT_SECONDS))
                break
            except Exception as exc:
                print(f"Offer {offer['id']} did not boot: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
                _destroy_tracked({str(instance_id)}, label)
                instance_ids.discard(str(instance_id))
        if host is None:
            raise VastAPIError("no rented machine booted")
        print("Uploading Part 127 code and pinned inputs", flush=True)
        try:
            _remote_run(host, port, deadline)
        except Exception as exc:  # whatever finished is worth copying back before the machine goes
            exit_code = 2
            print(f"remote run failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        print("Copying captures back", flush=True)
        _copy_results(host, port, deadline)
        print("Pilot and main captures copied back; run reader labels locally", flush=True)
    except Exception as exc:
        exit_code = 2
        print(f"vast.py: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    finally:
        if rent_started is not None:
            _destroy_tracked(instance_ids, label)
        remains = instance_remains()
        print(f"P127_INSTANCE_REMAINS={'yes' if remains else 'no'} "
              "(v1 listing; listing errors count as yes)", flush=True)
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
