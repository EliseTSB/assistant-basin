"""Chicken-o-Meter: score a chicken imitation with EmbeddingGemma 2, and see what a rank-one weight edit does to it.

Record (or upload) a sound and the page shows: how chicken-like EmbeddingGemma 2 finds it (the mean centred cosine to
six chicken prompts, as a percentile among 31 reference clips: real hens, public-domain chickens, human imitations,
synthetic purrs and controls); a timeline of 2 s windows; two chicken neurons (Part 188's lineage) placed against
their ESC-50 distributions; and the frozen written gate from the chicken-edit study (Parts 190 and 193), which opens
only when one unit says "sounds like a chicken" and another says "not a real one", and shows the model's verdict with
and without the edit.

Run (Python 3.11+, CPU is enough):
    pip install numpy torch sentence-transformers soundfile imageio-ffmpeg
    python server.py            # downloads google/embeddinggemma-2 at the pinned revision on first run
Open http://localhost:8877 on the same machine (localhost counts as secure, so the browser microphone works).
It binds 127.0.0.1 only. Unlabelled uploads are decoded in memory and discarded; a take you label (real hen, my
imitation, something else) is saved with its result in ./takes/.

Environment: CHICKEN_MODEL (a model id or local path), CHICKEN_REVISION, FFMPEG (an ffmpeg binary; default: the
imageio-ffmpeg one, else ffmpeg on PATH), CHICKEN_PORT.

Known limits: a lossy re-encode moves the score a lot (a Commons egg song scores z 0.78 as its OGG and 0.44 as a 128k
MP3), and the neuron bars are calibrated on clean ESC-50 clips, so phone recordings in a coop sit low on them, real
hens included. Built by Journal Claude with Elise, 2026-11-03 to 11-04.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL = os.environ.get('CHICKEN_MODEL', 'google/embeddinggemma-2')
REVISION = os.environ.get('CHICKEN_REVISION', 'f9972ac22e696a105c482202646979e33e4b5a1e')
PORT = int(os.environ.get('CHICKEN_PORT', '8877'))


def find_ffmpeg():
    if os.environ.get('FFMPEG'):
        return os.environ['FFMPEG']
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return 'ffmpeg'


FFMPEG = find_ffmpeg()
MAX_BYTES = 20 * 1024 * 1024
MAX_SECONDS = 30
CHICKEN = ('a hen purring', 'a chicken clucking', 'chickens in a coop', "a hen's egg song", 'a rooster crowing',
           'a hen cackling')
NICE = {  # reference clip -> label shown on the page
    'real_flock_burst': 'the real flock, mid-burst', 'pos_chickens_demanding_food': 'hens demanding food (Commons)',
    'pos_egg_song': 'a real egg song (Commons)', 'pos_hens_leaving_coop': 'hens leaving the coop (Commons)',
    'pos_rooster': 'a rooster (Commons)', 'pos_rooster_young': 'a young rooster (Commons)',
    'human_imitation_egg_song_a': "Elise's egg song, take 1", 'human_imitation_egg_song_b': "Elise's egg song, take 2",
    'human_imitation_open': "Elise's purr", 'human_imitation_sleeve': "Elise's purr, into a sleeve",
    'other_barred_owls': 'two barred owls', 'other_speech_elise': 'Elise saying "the hens are hungry"',
    'other_toad': 'a toad', 'ctl_matched_noise': 'plain noise',
}

STATE = {}
LOCK = threading.Lock()

# The written gate: frozen exactly as the chicken edit study fitted it,
# registered before any take. Nothing here is refitted: gate.json holds its fitted parameters and signs, and
# gate-prompts.npz the study's prompt embeddings (the two arrays of its acts.npz the gate reads).
STUDY = HERE.parent / 'part193' / 'chicken_edit'
TAKES = HERE / 'takes'
GATE_L17, GATE_N17, GATE_L19, GATE_N19 = 17, 1574, 19, 1129
LABELS = {'real': 'real hen', 'imitation': 'my imitation', 'other': 'something else'}


def load_gate():
    sys.path.insert(0, str(STUDY))
    import edit_arms  # the study's own functions, so the meter cannot drift from them
    v = json.loads((HERE / 'gate.json').read_text())
    par = v['parameters']
    sign = v['signs']
    ref = np.load(HERE / 'gate-prompts.npz')
    cats = list(ref['categories'])
    P = np.asarray(ref['prompt_emb'], dtype=np.float32)
    hen, roo = cats.index('hen'), cats.index('rooster')
    other = np.asarray([i for i in range(len(cats)) if i not in (hen, roo)])
    c = edit_arms.normalize_vector(P[[hen, roo]].mean(axis=0))
    return dict(ea=edit_arms, t19=float(par['t19']), t17=float(par['t17']), k=float(par['k']),
                s17=int(sign['L17_N1574']), s19=int(sign['L19_N1129']), P=P, cats=cats, hen=hen, roo=roo,
                other=other, c=c)


def gate_reading(e: np.ndarray, grab: dict) -> dict:
    G = STATE['gate']
    ea = G['ea']
    # float16 first: the stored arrays the thresholds were fitted on are float16.
    s17 = G['s17'] * float(np.float16(grab[GATE_L17][GATE_N17]))
    s19 = G['s19'] * float(np.float16(grab[GATE_L19][GATE_N19]))
    g = max(s19 - G['t19'], 0.0) * max(G['t17'] - s17, 0.0)
    after = ea.apply_written_gate(e, s19, s17, G['t19'], G['t17'], G['k'], G['c'])
    both = np.stack([e, after])
    sc = ea.prompt_scores(both, G['P'], G['hen'], G['roo'], G['other'])
    top = ea.prompt_predictions(both, G['P'], G['cats'])
    return {'open': g > 0.0, 'g': g, 's17': round(s17, 5), 's19': round(s19, 5), 't17': round(G['t17'], 5),
            't19': round(G['t19'], 5), 'score_before': round(float(sc[0]), 4), 'score_after': round(float(sc[1]), 4),
            'top_before': str(top[0]), 'top_after': str(top[1]), 'identical': bool(after is e)}, after


def save_take(blob: bytes, label: str, route: str, out: dict) -> str:
    import datetime, hashlib
    TAKES.mkdir(exist_ok=True)  # local only; nothing leaves the machine
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    name = f'{stamp}-{label}-{hashlib.sha256(blob).hexdigest()[:8]}'
    (TAKES / f'{name}.audio').write_bytes(blob)
    (TAKES / f'{name}.json').write_text(json.dumps({'label': label, 'route': route, 'saved': stamp,
                                                     'bytes': len(blob), 'result': out}, indent=1))
    return name


def label(name):
    if name in NICE:
        return NICE[name]
    if name.startswith('real_'):
        return 'a real hen purr (' + name[5:].replace('_', ' ') + ')'
    if name.startswith('syn_'):
        return "Claude's synthetic purr (" + name[4:].replace('_', ' ') + ')'
    return name


def load():
    import torch
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL, revision=REVISION, device='cpu', config_kwargs={'vision_config': None},
                                model_kwargs={'dtype': torch.float32})
    d = json.loads((HERE / 'ref-results.json').read_text())
    prompts = d['prompts']
    names = list(d['clips'])
    M = np.array([[d['clips'][n]['prompt_cos'][p] for p in prompts] for n in names])
    mu, sd = M.mean(0), M.std(0)
    ch = [prompts.index(p) for p in CHICKEN]
    ref_z = ((M - mu) / sd)[:, ch].mean(1)
    T = np.asarray(model.encode(prompts, prompt_name='SearchQuery', normalize_embeddings=True), dtype=np.float32)
    cache = HERE / 'ref-embeddings.npz'
    # The reference clips themselves are not shipped, only their embeddings; the two files must name the same clips.
    if list(np.load(cache)['names']) != names:
        raise SystemExit('ref-embeddings.npz and ref-results.json name different clips')
    A = np.load(cache)['A']
    # Neuron readings: the chicken neurons, read the same way they were found (mean over positions,
    # one item per forward pass, CPU float32) and placed against their ESC-50 distributions in neuron-ref.json.
    nref = json.loads((HERE / 'neuron-ref.json').read_text())['neurons']
    layers = model[0].model.language_model.layers
    grab = {}
    # One hook per layer read: the meter's neuron bars (15, 17) and the gate's two units (17, 19).
    for L in sorted({n['layer'] for n in nref} | {GATE_L17, GATE_L19}):
        layers[L].mlp.down_proj.register_forward_pre_hook(
            lambda mod, args, L=L: grab.__setitem__(L, args[0][0].float().mean(0)))
    STATE.update(model=model, prompts=prompts, names=names, mu=mu, sd=sd, ch=ch, ref_z=ref_z, T=T, A=A,
                 nref=nref, grab=grab, gate=load_gate())


def decode(blob: bytes) -> np.ndarray:
    # A temp file, not stdin: m4a/mp4 (phone recordings) needs a seekable input, and from a pipe ffmpeg decodes nothing.
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.audio') as f:
        f.write(blob)
        f.flush()
        p = subprocess.run([FFMPEG, '-nostdin', '-loglevel', 'error', '-i', f.name, '-ac', '1', '-ar', '16000',
                            '-t', str(MAX_SECONDS), '-f', 'f32le', 'pipe:1'], capture_output=True, timeout=60)
    if p.returncode != 0:
        raise ValueError('could not decode the audio: ' + p.stderr.decode(errors='replace')[:200])
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


def embed(xs):
    m = STATE['model']
    items = [{'audio': {'array': x, 'sampling_rate': 16000}} for x in xs]
    return np.asarray(m.encode(items, normalize_embeddings=True), dtype=np.float32).reshape(len(xs), -1)


def score(x: np.ndarray) -> dict:
    S = STATE
    if len(x) < 16000 * 0.5:
        raise ValueError('that was under half a second; try a longer take')
    peak = float(np.abs(x).max())
    if peak < 1e-4:
        raise ValueError('that was silent; check the microphone')
    win, hop = 16000 * 2, 16000 // 2
    starts = list(range(0, max(1, len(x) - win + 1), hop)) if len(x) > win else []
    with LOCK:
        # The whole take alone, then the windows as one batch, which is how the reference clips were embedded.
        # Measured 11-03: the Commons egg song scores z 0.782 as its original OGG (the reference value exactly) and
        # 0.435 as a 128k MP3 re-encode of the same file, so a lossy re-encode moves the score this much; batching
        # was suspected and not tested.
        S['grab'].clear()
        full = embed([x])
        neurons = []
        for n in S['nref']:
            v = n['sign'] * float(S['grab'][n['layer']][n['index']])
            other, chick = np.asarray(n['other']), np.asarray(n['chicken'])
            neurons.append({'name': f"layer {n['layer']}, neuron {n['index']}", 'why': n['why'],
                            'value': round(v, 5), 'pct_above_other': round(float((other < v).mean() * 100), 1),
                            'pct_above_chicken': round(float((chick < v).mean() * 100), 1),
                            'hen_median_pct_above_other': round(float((other < n['hen_median']).mean() * 100), 1)})
        gate, after_e = gate_reading(full[0], S['grab'])
        S['last_e'] = full[0]
        E = np.concatenate([full, embed([x[s:s + win] for s in starts])]) if starts else full
    sims = E @ S['T'].T
    z = (sims - S['mu']) / S['sd']
    cz = z[:, S['ch']].mean(1)
    full_z = float(cz[0])
    pct = float((S['ref_z'] < full_z).mean() * 100)
    # The same gauge score on the gated embedding, so both are always shown.
    za = ((np.asarray(after_e, dtype=np.float32) @ S['T'].T) - S['mu']) / S['sd']
    gate['chicken_z_after'] = round(float(za[S['ch']].mean()), 3)
    gate['percentile_after'] = round(float((S['ref_z'] < gate['chicken_z_after']).mean() * 100), 1)
    order = np.argsort(-sims[0])
    ref_sim = S['A'] @ E[0]
    near = int(ref_sim.argmax())
    beat = [label(S['names'][i]) for i in np.argsort(S['ref_z']) if S['ref_z'][i] < full_z]
    above = [label(S['names'][i]) for i in np.argsort(S['ref_z']) if S['ref_z'][i] >= full_z]
    return {
        'seconds': round(len(x) / 16000, 1),
        'chicken_z': round(full_z, 3),
        'percentile': round(pct, 1),
        'top': [{'prompt': S['prompts'][i], 'cos': round(float(sims[0][i]), 4)} for i in order[:3]],
        'nearest': {'label': label(S['names'][near]), 'cos': round(float(ref_sim[near]), 4)},
        'just_beat': beat[-1] if beat else None,
        'next_up': above[0] if above else None,
        'timeline': [{'t': round(s / 16000 + 1.0, 2), 'chicken_z': round(float(v), 3)}
                     for s, v in zip(starts, cz[1:])],
        'n_reference': len(S['names']),
        'neurons': neurons,
        'gate': gate,
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write('chicken-o-meter: ' + fmt % args + '\n')

    def _send(self, code, body: bytes, ctype='application/json'):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            self._send(200, (HERE / 'index.html').read_bytes(), 'text/html; charset=utf-8')
        elif self.path == '/health':
            self._send(200, json.dumps({'ok': bool(STATE)}).encode())
        else:
            self._send(404, b'{"error":"not found"}')

    def do_POST(self):
        from urllib.parse import urlparse, parse_qs
        u = urlparse(self.path)
        if u.path != '/score':
            return self._send(404, b'{"error":"not found"}')
        q = parse_qs(u.query)
        label = (q.get('label') or [''])[0]
        route = (q.get('route') or [''])[0]
        if label and label not in LABELS:
            return self._send(400, json.dumps({'error': 'unknown label'}).encode())
        n = int(self.headers.get('Content-Length') or 0)
        if n <= 0 or n > MAX_BYTES:
            return self._send(413, json.dumps({'error': 'audio must be under 20 MB'}).encode())
        blob = self.rfile.read(n)
        try:
            out = score(decode(blob))
            if label:  # only labelled takes are kept (EDIT-DESIGN Amendment 2); unlabelled ones are discarded
                out['saved_as'] = save_take(blob, label, route[:20], out)
            self._send(200, json.dumps(out).encode())
        except ValueError as e:
            self._send(400, json.dumps({'error': str(e)}).encode())
        except Exception as e:  # report, do not crash the server
            self._send(500, json.dumps({'error': 'scoring failed: ' + type(e).__name__}).encode())


def main():
    load()
    srv = ThreadingHTTPServer(('127.0.0.1', PORT), Handler)
    print(f'chicken-o-meter ready on http://localhost:{PORT}', flush=True)
    srv.serve_forever()


if __name__ == '__main__':
    main()
