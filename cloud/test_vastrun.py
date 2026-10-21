import builtins
import io
import json
import re
import shlex
import signal
import subprocess
import urllib.error
import urllib.request

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


def make_runner(path, api, ssh, clock, *, sleep=None, destroy_delay=0):
    config = json.loads(path.read_text(encoding="utf-8"))
    return vastrun._Runner(
        path,
        config,
        api,
        ssh,
        clock=clock,
        sleep=sleep or clock.sleep,
        poll_interval=1,
        ready_interval=1,
        ready_timeout=3,
        destroy_delay=destroy_delay,
        cancel_event=None,
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


def test_instance_id_is_persisted_before_ssh_readiness(tmp_path):
    path = jobfile(tmp_path)
    api = FakeAPI()

    class RecordCheckingSSH(FakeSSH):
        def run(self, host, port, command, *, timeout, check=True):
            if command == "true":
                record_path = tmp_path / "results" / "run-record.json"
                record = json.loads(record_path.read_text(encoding="utf-8"))
                assert record["instance_id"] == "instance-offer-1"
            return super().run(host, port, command, timeout=timeout, check=check)

    execute(path, api, RecordCheckingSSH())


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

    assert api.destroy_attempts[:vastrun.DESTROY_RETRIES] == ["instance-offer-1"] * vastrun.DESTROY_RETRIES
    record = json.loads((tmp_path / "results" / "run-record.json").read_text())
    assert record["destroy_confirmed"] is False
    assert "destroy_error" in record


def test_unreadable_instance_list_is_not_treated_as_gone(tmp_path):
    class UnreadableAPI(FakeAPI):
        def list_instances(self):
            return {"success": False, "msg": "rate limited"}

    api = UnreadableAPI(keep_alive=True)
    clock = FakeClock()
    runner = make_runner(jobfile(tmp_path), api, FakeSSH(), clock)
    instance_id = "instance-offer-1"
    api.instances_by_id[instance_id] = {"id": instance_id, "label": runner.run_label}
    entry = {"label": runner.run_label, "dph": 0.1, "instance_id": instance_id}
    runner.record["hosts_tried"].append(entry)
    runner._activate(instance_id, entry, started=0.0)

    with pytest.raises(vastrun.DestroyConfirmationError):
        runner._destroy_active()

    assert instance_id in api.instances_by_id
    assert api.destroy_attempts == [instance_id] * vastrun.DESTROY_RETRIES
    assert runner.record["destroy_confirmed"] is False


def test_unreadable_list_keeps_uncertain_rent_unconfirmed(tmp_path):
    class UnreadableAPI(FakeAPI):
        def list_instances(self):
            return {"error": "temporary failure"}

    api = UnreadableAPI([offer("lost")], ambiguous_offers={"lost"})
    runner = make_runner(jobfile(tmp_path), api, FakeSSH(), FakeClock())
    entry = {"label": runner.run_label, "dph": 0.1}
    with pytest.raises(TimeoutError):
        api.rent("lost", "watchdog", runner.run_label)

    with pytest.raises(vastrun.RunnerError, match="rent outcome is uncertain"):
        runner._recover_uncertain_rent(entry, runner.run_label, 0.0)
    assert "instance-lost" in api.instances_by_id
    assert api.destroyed == []


def test_final_run_label_sweep_destroys_every_matching_instance(tmp_path):
    class OrphanAPI(FakeAPI):
        def rent(self, offer_id, watchdog, label):
            response = super().rent(offer_id, watchdog, label)
            self.instances_by_id["orphan-instance"] = {
                "id": "orphan-instance",
                "label": label,
                "dph_total": 0.25,
            }
            self.rent_label = label
            return response

    api = OrphanAPI()
    record = execute(jobfile(tmp_path), api, FakeSSH())

    assert api.rent_label == record["run_label"]
    assert "orphan-instance" in api.destroyed
    assert api.instances_by_id == {}
    assert record["destroy_confirmed"] is True


def test_instance_rate_is_used_and_unconfirmed_spend_clock_keeps_running(tmp_path):
    clock = FakeClock()
    api = FakeAPI(keep_alive=True)
    api.instances_by_id["instance-1"] = {
        "id": "instance-1",
        "label": "run-label",
        "dph_total": 2.0,
    }
    path = jobfile(tmp_path)
    runner = make_runner(path, api, FakeSSH(), clock)
    entry = {"label": runner.run_label, "dph": 0.1, "instance_id": "instance-1"}
    runner.record["hosts_tried"].append(entry)
    runner._activate("instance-1", entry, started=0.0)

    assert runner.current_dph == 2.0
    with pytest.raises(vastrun.DestroyConfirmationError):
        runner._destroy_active()
    before = runner._estimate_spend()
    clock.sleep(3600)
    assert runner._estimate_spend() == pytest.approx(before + 2.0)


def test_second_interrupt_during_destroy_backoff_does_not_cut_retries(tmp_path):
    clock = FakeClock()
    api = FakeAPI(keep_alive=True)
    runner = make_runner(jobfile(tmp_path), api, FakeSSH(), clock, destroy_delay=1)
    entry = {"label": runner.run_label, "dph": 0.1, "instance_id": "instance-1"}
    api.instances_by_id["instance-1"] = {
        "id": "instance-1",
        "label": runner.run_label,
    }
    runner.record["hosts_tried"].append(entry)
    runner._activate("instance-1", entry, started=0.0)
    sleeps = 0

    def interrupt_once(seconds):
        nonlocal sleeps
        sleeps += 1
        if sleeps == 1:
            raise vastrun.RunInterrupted("second interrupt")
        clock.sleep(seconds)

    runner.sleep = interrupt_once
    with pytest.raises(vastrun.DestroyConfirmationError):
        runner._destroy_active()
    assert len(api.destroy_attempts) == vastrun.DESTROY_RETRIES
    assert sleeps == vastrun.DESTROY_RETRIES - 1


def test_main_installs_sighup_handler(tmp_path, monkeypatch, capsys):
    installed = []
    previous = {}

    def fake_signal(signum, handler):
        installed.append(signum)
        return previous.get(signum, signal.SIG_DFL)

    monkeypatch.setattr(vastrun.signal, "signal", fake_signal)
    monkeypatch.setattr(vastrun, "run_jobfile", lambda *args, **kwargs: {"estimated_spend_usd": 0.0})

    assert vastrun.main([str(tmp_path / "unused.json")]) == 0
    assert signal.SIGHUP in installed
    capsys.readouterr()


def test_poll_tolerates_three_timeouts_and_copies_finished_results(tmp_path):
    class FlakyPollSSH(FakeSSH):
        def __init__(self):
            super().__init__(outcomes={"FINISH_OK": 0})
            self.timeouts = 3

        def run(self, host, port, command, *, timeout, check=True):
            if command.startswith("cat ") and self.timeouts:
                self.timeouts -= 1
                raise subprocess.TimeoutExpired(command, timeout)
            return super().run(host, port, command, timeout=timeout, check=check)

    result_path = tmp_path / "results" / "outputs" / "finished.txt"

    class OrderedAPI(FakeAPI):
        def destroy(self, instance_id):
            assert result_path.read_text(encoding="utf-8") == "copied\n"
            return super().destroy(instance_id)

    api = OrderedAPI()
    ssh = FlakyPollSSH()
    path = jobfile(tmp_path, jobs=[{
            "name": "finished",
            "command": "FINISH_OK",
            "expected_result_paths": ["outputs/finished.txt"],
        }])
    record = execute(path, api, ssh)

    assert ssh.timeouts == 0
    assert record["jobs"][0]["rc"] == 0
    assert api.destroyed == ["instance-offer-1"]


def test_fourth_consecutive_poll_failure_raises_and_cleans_up(tmp_path):
    class FlakyPollSSH(FakeSSH):
        def __init__(self):
            super().__init__(never_finish={"HANG"})
            self.timeouts = vastrun.MAX_POLL_FAILURES + 1

        def run(self, host, port, command, *, timeout, check=True):
            if command.startswith("cat ") and self.timeouts:
                self.timeouts -= 1
                raise subprocess.TimeoutExpired(command, timeout)
            return super().run(host, port, command, timeout=timeout, check=check)

    api = FakeAPI()
    ssh = FlakyPollSSH()
    path = jobfile(tmp_path, jobs=[{"name": "hang", "command": "HANG"}])

    with pytest.raises(subprocess.TimeoutExpired):
        execute(path, api, ssh)

    assert ssh.timeouts == 0
    assert api.destroyed == ["instance-offer-1"]


@pytest.mark.parametrize("jobs, message", [
    ([
        {"name": "one", "command": "true", "expected_result_paths": ["outputs/result.txt"]},
        {"name": "two", "command": "true", "expected_result_paths": ["outputs/result.txt"]},
    ], "duplicate expected result path"),
    ([
        {"name": "a/b", "command": "true"},
        {"name": "a_b", "command": "true"},
    ], "safe job log stems must be unique"),
])
def test_validation_rejects_result_and_safe_stem_collisions(tmp_path, jobs, message):
    api = FakeAPI()
    with pytest.raises(vastrun.RunnerError, match=message):
        execute(jobfile(tmp_path, jobs=jobs), api, FakeSSH())
    assert api.watchdogs == []


@pytest.mark.parametrize("instance, expected", [
    ({"ssh_host": "host", "ports": {"22/tcp": [{"HostPort": 2233}]}}, ("host", 2233)),
    ({"ssh_host": "host", "ports": {"22": [{"HostPort": 2244}]}}, ("host", 2244)),
    ({"ssh_host": "host", "ports": {22: 2255}}, ("host", 2255)),
    ({"ssh_host": "preferred", "ssh_port": 2222, "ports": {"22/tcp": [{"HostPort": 2233}]}},
     ("preferred", 2222)),
])
def test_remote_connection_accepts_vast_ssh_port_shapes(instance, expected):
    assert vastrun._remote_connection(instance) == expected


def test_vast_api_key_uses_real_adapter_and_never_leaks(tmp_path, monkeypatch, capsys):
    key_path = tmp_path / "fake-vast-key.txt"
    key = "FAKE_KEY_DO_NOT_LOG"
    key_path.write_text(key, encoding="utf-8")
    real_open = builtins.open

    def guarded_open(file, *args, **kwargs):
        assert ".vast-credits.env" not in str(file)
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    calls = []

    def fake_urlopen(request, *, timeout):
        calls.append((request, timeout))
        return io.BytesIO(b'{"instances": []}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    api = vastrun.VastAPI(key_path=str(key_path))
    assert api.list_instances() == {"instances": []}
    assert calls[0][0].get_header("Authorization") == f"Bearer {key}"
    assert calls[0][0].full_url == "https://console.vast.ai/api/v1/instances/"

    def rejected_urlopen(request, *, timeout):
        raise urllib.error.HTTPError(request.full_url, 403, "denied", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", rejected_urlopen)
    with pytest.raises(vastrun.RunnerError, match="HTTP 403") as error:
        api.list_instances()
    assert key not in str(error.value)
    captured = capsys.readouterr()
    assert key not in captured.out + captured.err


def test_unreadable_instance_list_shape_is_distinct_from_empty_list():
    assert vastrun._instance_ids({"instances": []}) == set()
    assert vastrun._instance_ids({"success": False}) is None
    assert vastrun._instance_ids([{"id": "instance-1"}]) is None
    assert vastrun._instance_ids({"instances": [{"label": "missing-id"}]}) is None
    assert vastrun._instance_ids({"instances": [{"id": "missing-label"}]}) is None
