# job.py — runs each line of steps.txt as `python <line>` in this folder, in order, stopping at the first failure
# (Claude fork, 2026-10-16). Output of every step goes to step-<n>.log; one line per step to progress.log.
# RAM guard (Claude, 2026-10-19): twice that evening a step loading SmolLM3-3B froze Elise's PC, because Windows had
# about 4 GB available (WSL's VM held 12.4 GB) and the load pushed it into heavy paging. A guard thread now kills the
# running step when Windows' available physical memory falls under MIN_AVAIL_GB, and the step is logged rc=97.
import ctypes, os, shlex, subprocess, sys, threading, time
MIN_AVAIL_GB = float(os.environ.get('JOB_MIN_AVAIL_GB', '2.0'))
class _MS(ctypes.Structure):
    _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong), ('ullTotalPhys', ctypes.c_ulonglong),
                ('ullAvailPhys', ctypes.c_ulonglong), ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong), ('x', ctypes.c_ulonglong)]
def avail_gb():
    m = _MS(); m.dwLength = ctypes.sizeof(_MS); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 2**30
here = os.path.dirname(os.path.abspath(__file__))
# Single-run lock (Claude, 2026-10-19): a second Start-ScheduledTask while a job runs would start two jobs on one GPU.
# msvcrt.locking holds a byte lock on job.lock for this process's life; a second job.py fails to take it and exits 98.
import msvcrt
_lock = open(os.path.join(here, 'job.lock'), 'a+')
try:
    msvcrt.locking(_lock.fileno(), msvcrt.LK_NBLCK, 1)
except OSError:
    with open(os.path.join(here, 'progress.log'), 'a', encoding='utf-8') as _p:
        _p.write(f'{time.strftime("%H:%M:%S")} REFUSED: another job.py holds job.lock; not starting\n')
    sys.exit(98)
env = dict(os.environ, HF_HOME=r'E:\neuron-probe\hf', HF_HUB_OFFLINE='1', TRANSFORMERS_VERBOSITY='error',  # path-audit-ok: Windows GPU host cache path
           HF_HUB_DISABLE_PROGRESS_BARS='1', PYTHONIOENCODING='utf-8')
# Copy engine off by default (Claude, 2026-10-24): with Level Zero's copy engine on, atlas pass two stalled ~150 s at a
# time with the GPU idle (4 stalls in 80 batches; 0 in 80 with it off, same batches, 718 s -> 73 s), and layers 7-13
# never finished in three tries. Device-to-host copies then run on the compute engine. A caller can still set either
# variable to 1 to compare. Measurements: Journal notes/reports/arc-stall-2026-10-24/README.md.
env.setdefault('SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE', '0')
env.setdefault('UR_L0_USE_COPY_ENGINE', '0')

def _gpu_sample():
    # One fresh typeperf read: the wildcard is expanded at each call, so a process that opened the GPU after the
    # step started is seen. Returns {pid: max engine utilization %}.
    out = subprocess.run(['typeperf', r'\GPU Engine(*)\Utilization Percentage', '-sc', '1'], capture_output=True,
                         text=True, timeout=30, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)).stdout
    import csv, io, re
    rows = [r for r in csv.reader(io.StringIO(out)) if r]
    if len(rows) < 2: return {}
    head, vals = rows[0], rows[1]
    res = {}
    for h, v in zip(head[1:], vals[1:]):
        m = re.search(r'pid_(\d+)_', h)
        try: x = float(v)
        except ValueError: continue
        if m: res[int(m.group(1))] = max(res.get(int(m.group(1)), 0.0), x)
    return res


def gpu_start(n):
    # Claude, 2026-10-21 (Elise: log GPU use if simple). v2: the first version ran one typeperf whose wildcard was fixed
    # at start, so the step's own process (opened later, through the venv launcher) was never in it. Now a thread polls
    # every 5 s and keeps processes that were NOT using the GPU before the step. Failures are swallowed.
    try:
        base = set(_gpu_sample())
        state = {'stop': False, 'samples': [], 'base': base}
        def loop():
            while not state['stop']:
                try:
                    cur = _gpu_sample()
                    new = [v for pid, v in cur.items() if pid not in state['base']]
                    if new: state['samples'].append(max(new))
                except Exception:
                    pass
                time.sleep(5)
        threading.Thread(target=loop, daemon=True).start()
        return state, None
    except Exception:
        return None, None


def gpu_summary(state, _path, _pid):
    try:
        state['stop'] = True
        per = state['samples']
        if not per: return 'gpu: no samples from processes new to the GPU'
        busy = sum(v > 5 for v in per) / len(per)
        return f'gpu: {len(per)} samples, mean {sum(per)/len(per):.0f}%, max {max(per):.0f}%, busy(>5%) {busy:.0%} of the time'
    except Exception as exc:
        return f'gpu: summary failed ({type(exc).__name__})'


prog = open(os.path.join(here, 'progress.log'), 'a', encoding='utf-8')
steps = [l.strip() for l in open(os.path.join(here, 'steps.txt'), encoding='utf-8') if l.strip() and not l.startswith('#')]
for n, line in enumerate(steps, 1):
    prog.write(f'{time.strftime("%H:%M:%S")} step {n} start: {line} (ram_avail {avail_gb():.1f}G)\n'); prog.flush()
    with open(os.path.join(here, f'step-{n}.log'), 'w', encoding='utf-8') as log:
        p = subprocess.Popen([sys.executable] + shlex.split(line, posix=False), cwd=here, env=env, stdout=log,
                             stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        killed = []
        def guard():
            while p.poll() is None:
                a = avail_gb()
                if a < MIN_AVAIL_GB:
                    killed.append(a); p.kill(); return
                time.sleep(0.2)
        threading.Thread(target=guard, daemon=True).start()
        tp, gpath = gpu_start(n)
        rc = p.wait()
        if tp is not None:
            prog.write(f'{time.strftime("%H:%M:%S")} step {n} {gpu_summary(tp, gpath, p.pid)}\n'); prog.flush()
    if killed:
        rc = 97
        prog.write(f'{time.strftime("%H:%M:%S")} step {n} RAM GUARD: Windows available RAM {killed[0]:.2f}G < {MIN_AVAIL_GB}G, step killed\n')
    prog.write(f'{time.strftime("%H:%M:%S")} step {n} rc={rc}\n'); prog.flush()
    if rc: sys.exit(rc)
prog.write(f'{time.strftime("%H:%M:%S")} all steps done\n')
