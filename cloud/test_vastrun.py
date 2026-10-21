import builtins
import json
import re
import shlex

import pytest

from cloud import vastrun


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def offer(offer_id, price=0.1, **changes):
    value = {
        "id": offer_id,
        "gpu_name": "RTX 3090",
        "num_gpus": 1,
        "verified": True,
        "rentable": True,
        "reliability2": 0.99,
        "cuda_max_good": 12.5,
        "disk_space": 50,
        "inet_down": 250,
        "dph_total": price,
    }
    value.update(changes)
    return value


class FakeAPI:
    def __init__(self, offers=None, keep_alive=False, ambiguous_offers=()):
        self._offers = offers or [offer("offer-1")]
        self.keep_alive = keep_alive
        self.ambiguous_offers = set(ambiguous_offers)
        self.instances_by_id = {}
        self.destroyed = []
        self.destroy_attempts = []
        self.watchdogs = []
        self.key = "FAKE_VAST_KEY_DO_NOT_LOG"

    def offers(self, max_dph):
        return list(self._offers)

    def rent(self, offer_id, watchdog, label):
        self.watchdogs.append(watchdog)
        instance_id = f"instance-{offer_id}"
        self.instances_by_id[instance_id] = {
            "id": instance_id,
            "label": label,
            "ssh_host": f"host-{offer_id}",
            "ssh_port": 2222,
        }
        if offer_id in self.ambiguous_offers:
            raise TimeoutError("response lost after rent")
        return {"success": True, "new_contract": instance_id}

    def get_instance(self, instance_id):
        instance = self.instances_by_id.get(instance_id)
        return {"instances": instance} if instance else {"instances": []}

    def destroy(self, instance_id):
        self.destroy_attempts.append(instance_id)
        if not self.keep_alive:
            self.instances_by_id.pop(instance_id, None)
        self.destroyed.append(instance_id)
        return {"success": True}

    def list_instances(self):
        return {"instances": list(self.instances_by_id.values())}


class FakeSSH:
    def __init__(self, *, not_ready=(), outcomes=None, never_finish=(), fail_launch=()):
        self.not_ready = set(not_ready)
        self.outcomes = outcomes or {}
        self.never_finish = set(never_finish)
        self.fail_launch = set(fail_launch)
        self.rc_by_log = {}
        self.events = []
        self.outputs = {}

    def run(self, host, port, command, *, timeout, check=True):
        self.events.append(("run", host, command))
        if command == "true":
            if host in self.not_ready:
                raise ConnectionError("not ready")
            return ""
        if "nohup sh -c" in command:
            marker = next((item for item in self.outcomes if item in command), None)
            marker = marker or next((item for item in self.never_finish if item in command), None)
            if any(item in command for item in self.fail_launch):
                raise RuntimeError("injected launch failure")
            match = re.search(r"\s>\s([^ ]+)\s+2>&1", command)
            assert match, command
            log_path = match.group(1).strip("'\"")
            if marker in self.never_finish:
                self.rc_by_log[log_path] = None
            else:
                self.rc_by_log[log_path] = self.outcomes.get(marker, 0)
            return "123\n"
        if command.startswith("cat "):
            parts = shlex.split(command)
            log_path = parts[1][:-3]
            rc = self.rc_by_log.get(log_path)
            return "" if rc is None else f"{rc}\n"
        if check is False:
            return ""
        return ""

    def upload(self, host, port, local_path, remote_path, *, timeout):
        self.events.append(("upload", str(local_path), remote_path))

    def download(self, host, port, remote_path, local_path, *, timeout):
        self.events.append(("download", remote_path, str(local_path)))
        local_path.write_text(self.outputs.get(remote_path, "copied\n"), encoding="utf-8")


def jobfile(tmp_path, **overrides):
    config = {
        "upload_files": [],
        "setup_command": None,
        "jobs": [],
        "results_dir": "results",
        "max_dph": 1,
        "spend_cap_usd": 100,
        "wall_clock_min": 30,
    }
    config.update(overrides)
    path = tmp_path / "job.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def execute(path, api, ssh, clock=None, **kwargs):
    clock = clock or FakeClock()
    return vastrun.run_jobfile(
        path,
        api=api,
        ssh=ssh,
        clock=clock,
        sleep=clock.sleep,
        poll_interval=kwargs.pop("poll_interval", 1),
        ready_interval=kwargs.pop("ready_interval", 1),
        ready_timeout=kwargs.pop("ready_timeout", 3),
        destroy_delay=kwargs.pop("destroy_delay", 0),
        **kwargs,
    )


def test_ready_timeout_destroys_first_and_tries_next_offer(tmp_path):
    api = FakeAPI([
        offer("too-slow", 0.1),
        offer("good", 0.2),
        offer("not-fast-enough", 0.3, inet_down=199),
    ])
    ssh = FakeSSH(not_ready={"host-too-slow"})
    path = jobfile(tmp_path)

    record = execute(path, api, ssh, ready_timeout=2)

    assert api.destroyed == ["instance-too-slow", "instance-good"]
    assert [host["offer_id"] for host in record["hosts_tried"]] == ["too-slow", "good"]
    assert "SSH did not become ready" in record["hosts_tried"][0]["dropped_reason"]
    assert record["destroy_confirmed"] is True


def test_destroy_runs_after_exception_mid_job(tmp_path):
    api = FakeAPI()
    ssh = FakeSSH(fail_launch={"CRASH_AFTER_START"})
    path = jobfile(tmp_path, jobs=[{"name": "broken", "command": "CRASH_AFTER_START"}])

    with pytest.raises(RuntimeError, match="injected launch failure"):
        execute(path, api, ssh)

    assert api.destroyed == ["instance-offer-1"]
    record = json.loads((tmp_path / "results" / "run-record.json").read_text())
    assert record["destroy_confirmed"] is True


def test_uncertain_rent_response_recovers_and_destroys_labeled_instance(tmp_path):
    api = FakeAPI([offer("lost"), offer("good", price=0.2)], ambiguous_offers={"lost"})
    path = jobfile(tmp_path)

    record = execute(path, api, FakeSSH())

    assert api.destroyed == ["instance-lost", "instance-good"]
    assert record["hosts_tried"][0]["recovered_instance_ids"] == ["instance-lost"]
    assert "recovered and destroyed" in record["hosts_tried"][0]["dropped_reason"]


def test_spend_cap_stops_polling_and_destroys(tmp_path):
    api = FakeAPI([offer("expensive", price=36)])
    ssh = FakeSSH(never_finish={"HANG_FOREVER"})
    path = jobfile(
        tmp_path,
        max_dph=40,
        spend_cap_usd=0.01,
        jobs=[{"name": "hang", "command": "HANG_FOREVER"}],
    )

    with pytest.raises(vastrun.SpendCapExceeded):
        execute(path, api, ssh)

    assert api.destroyed == ["instance-expensive"]
    record = json.loads((tmp_path / "results" / "run-record.json").read_text())
    assert record["destroy_confirmed"] is True
    assert record["estimated_spend_usd"] >= 0.01


def test_each_job_copies_results_before_later_job_fails(tmp_path):
    api = FakeAPI()
    ssh = FakeSSH(outcomes={"FIRST_OK": 0, "SECOND_FAIL": 7})
    path = jobfile(
        tmp_path,
        jobs=[
            {"name": "first", "command": "FIRST_OK", "expected_result_paths": ["outputs/first.txt"]},
            {"name": "second", "command": "SECOND_FAIL", "expected_result_paths": ["outputs/second.txt"]},
        ],
    )

    with pytest.raises(vastrun.JobFailed, match="second exited with rc=7"):
        execute(path, api, ssh)

    first_result = tmp_path / "results" / "outputs" / "first.txt"
    assert first_result.read_text(encoding="utf-8") == "copied\n"
    first_copy = next(i for i, event in enumerate(ssh.events)
                      if event[0] == "download" and event[1].endswith("outputs/first.txt"))
    second_launch = next(i for i, event in enumerate(ssh.events)
                         if event[0] == "run" and "SECOND_FAIL" in event[2] and "nohup" in event[2])
    assert first_copy < second_launch
    assert api.destroyed == ["instance-offer-1"]


def test_destroy_confirmation_failure_raises_after_three_attempts(tmp_path):
    api = FakeAPI(keep_alive=True)
    path = jobfile(tmp_path)

    with pytest.raises(vastrun.DestroyConfirmationError, match="after 3 attempts"):
        execute(path, api, FakeSSH())

    assert api.destroy_attempts == ["instance-offer-1"] * 3
    record = json.loads((tmp_path / "results" / "run-record.json").read_text())
    assert record["destroy_confirmed"] is False
    assert "destroy_error" in record


def test_vast_key_is_not_read_or_written_to_stdout_or_logs(tmp_path, monkeypatch, capsys):
    real_open = builtins.open

    def guarded_open(file, *args, **kwargs):
        assert "vast-credits.env" not in str(file)
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    api = FakeAPI()
    ssh = FakeSSH(outcomes={"SAFE_JOB": 0})
    path = jobfile(tmp_path, jobs=[{"name": "safe", "command": "SAFE_JOB"}])

    execute(path, api, ssh)

    captured = capsys.readouterr()
    assert api.key not in captured.out + captured.err
    for result in (tmp_path / "results").rglob("*"):
        if result.is_file():
            assert api.key not in result.read_text(encoding="utf-8")
    assert all(api.key not in str(event) for event in ssh.events)
    assert api.watchdogs and "poweroff" in api.watchdogs[0]
    assert "vast-credits.env" not in capsys.readouterr().out
