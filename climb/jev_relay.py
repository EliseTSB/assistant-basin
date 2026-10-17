#!/usr/bin/env python3
# jev_relay.py — keeps the TypeSafe key out of the climbers' sandboxes (Claude, 2026-10-16, Part 1e).
# Listens on 127.0.0.1:8341. POST /v1/systemone with header X-Climber: <name> (luna|ds|glm|smoke) is forwarded to
# TypeSafe with the house key added. It caps concurrent upstream calls at 6 for all climbers together (JJ's watcher
# shares this key) and each climber's input tokens at CAP (default 30M); past the cap it answers 403 {"error":
# "climb budget spent"}. TypeSafe's own errors pass through with the key redacted. Spend is kept in spend.json.
import json, os, threading, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
HERE = os.path.dirname(os.path.abspath(__file__))
CAP = int(os.environ.get('RELAY_CAP', 30_000_000))
CAPS = {'luna': 60_000_000}  # raised 2026-10-16 21:46: Luna's climb spends about 15M an hour
KEY = next(l.split('=', 1)[1].strip().strip('"\'') for l in open('/home/elisepc/green/DiscordBot/.env') if l.startswith('TYPESAFE_API_KEY='))
SEM = threading.BoundedSemaphore(6); LOCK = threading.Lock()
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
SPEND = os.path.join(HERE, 'relay-spend.json')
spend = json.load(open(SPEND)) if os.path.exists(SPEND) else {}

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def reply(self, code, body):
        b = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code); self.send_header('content-type', 'application/json'); self.send_header('content-length', str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_POST(self):
        who = self.headers.get('X-Climber', '')
        if self.path != '/v1/systemone' or who not in ('luna', 'ds', 'glm', 'smoke'): return self.reply(404, {'error': 'unknown path or climber'})
        data = self.rfile.read(int(self.headers.get('content-length', 0)))
        with LOCK:
            if spend.get(who, 0) >= CAPS.get(who, CAP): return self.reply(403, {'error': 'climb budget spent', 'spent': spend[who]})
        req = urllib.request.Request('https://api.typesafe.ai/v1/systemone', data=data, method='POST',
                                     headers={'Authorization': f'Bearer {KEY}', 'Content-Type': 'application/json', 'user-agent': f'neuron-climb-relay/{who}'})
        with SEM:
            try:
                r = OPENER.open(req, timeout=120); body = r.read(); code = r.status
            except urllib.error.HTTPError as e: body = e.read(); code = e.code
            except Exception as e: return self.reply(502, {'error': str(e)[:200].replace(KEY, '[KEY]')})
        body = body.replace(KEY.encode(), b'[KEY]')
        if code == 200:
            try: tok = (json.loads(body).get('usage') or {}).get('input_tokens') or 0
            except Exception: tok = 0
            with LOCK:
                spend[who] = spend.get(who, 0) + tok
                tmp = SPEND + '.tmp'; json.dump(spend, open(tmp, 'w')); os.replace(tmp, SPEND)
        self.reply(code, body)

ThreadingHTTPServer(('127.0.0.1', 8341), H).serve_forever()
