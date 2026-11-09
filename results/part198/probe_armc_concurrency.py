"""Check that a second Arm C runner process refuses an already-held output log."""
from __future__ import annotations

import fcntl
import io
import multiprocessing
import queue
import sys
import tempfile
from contextlib import redirect_stderr
from pathlib import Path

import torch


def blocked_loader(*_args, **_kwargs):
    raise AssertionError("model or weight loading is forbidden in this probe")


torch.load = blocked_loader
try:
    import transformers
    transformers.AutoModelForCausalLM.from_pretrained = blocked_loader
    transformers.AutoTokenizer.from_pretrained = blocked_loader
except ImportError:
    pass
try:
    import safetensors
    safetensors.safe_open = blocked_loader
except ImportError:
    pass

REPO = Path('/home/elisepc/green/worktrees/ab-part198')
sys.path.insert(0, str(REPO / 'results/part198'))
import run198 as R

R.load_gpt2_model = blocked_loader
R.load_arm_c_pair = blocked_loader


def hold_log_lock(lock_path: Path, ready, release) -> None:
    with lock_path.open('a+b') as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        ready.set()
        release.wait(timeout=15)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def invoke_second_runner(out: Path, result_queue) -> None:
    preflight_calls = []
    R.root_out_dir = lambda _out: out

    def forbidden_preflight(_out):
        preflight_calls.append(True)
        raise RuntimeError("runner reached test preflight before lock refusal")

    R.preflight = forbidden_preflight
    stderr = io.StringIO()
    with redirect_stderr(stderr):
        code = R.main(["armC", "--out-dir", str(out)])
    result_queue.put({"code": code, "preflight_calls": preflight_calls,
                      "stderr": stderr.getvalue()})


if __name__ == '__main__':
    ctx = multiprocessing.get_context('fork')
    with tempfile.TemporaryDirectory(prefix='part198-armc-lock-') as td:
        out = Path(td) / 'out'
        out.mkdir()
        log = out / 'armC.jsonl'
        lock_path = log.with_name(log.name + '.lock')
        ready, release = ctx.Event(), ctx.Event()
        result_queue = ctx.Queue()
        holder = ctx.Process(target=hold_log_lock, args=(lock_path, ready, release))
        holder.start()
        try:
            if not ready.wait(timeout=5):
                raise RuntimeError('lock holder did not acquire the Arm C log lock')
            runner = ctx.Process(target=invoke_second_runner, args=(out, result_queue))
            runner.start()
            runner.join(timeout=10)
            if runner.is_alive():
                runner.terminate()
                runner.join(timeout=5)
                raise RuntimeError('second runner waited instead of refusing the held log')
            if runner.exitcode != 0:
                raise RuntimeError(f'second runner process exited {runner.exitcode}')
            try:
                result = result_queue.get(timeout=2)
            except queue.Empty as exc:
                raise RuntimeError('second runner returned no result') from exc
            refused = (result['code'] == 1 and not result['preflight_calls']
                       and 'already locked' in result['stderr'])
            print(f"second runner process refused={refused} preflight_calls="
                  f"{len(result['preflight_calls'])} error={result['stderr'].strip()}")
            if not refused:
                raise RuntimeError('the held output log was not refused before preflight')
        finally:
            release.set()
            holder.join(timeout=5)
            if holder.is_alive():
                holder.terminate()
                holder.join(timeout=5)
