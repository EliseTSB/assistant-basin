#!/usr/bin/env python3
"""Local record/replay System One endpoint for Part 1f (stdlib only)."""
import argparse
import hashlib
import json
import math
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def request_hash(state, questions):
    canonical = json.dumps({'state': state, 'questions': questions}, sort_keys=True,
                           separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(canonical).hexdigest()


def load_answers(path):
    answers = {}
    if not path:
        return answers
    with open(path, encoding='utf-8') as src:
        for line_no, line in enumerate(src, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get('hash') or row.get('request_hash')
            values = row.get('answers')
            if not key or not isinstance(values, dict):
                raise ValueError(f'{path}:{line_no}: expected hash and answers object')
            answers[key] = values
    return answers


def answer_request(request, mode, answers):
    """Pure request handler used by HTTP and by offline protocol tests."""
    try:
        state, questions = request['state'], request['questions']
        if not isinstance(questions, dict) or not questions:
            raise ValueError('questions must be a non-empty object')
        if any(not isinstance(spec, dict) or spec.get('type') != 'noul' for spec in questions.values()):
            raise ValueError('jev_stub only answers noul questions')
    except (KeyError, TypeError, ValueError) as exc:
        return 400, {'error': str(exc)}, None

    digest = request_hash(state, questions)
    if mode == 'record':
        row = dict(request)
        row['hash'] = digest
        probs = {name: 0.5 for name in questions}
    else:
        probs = answers.get(digest)
        if probs is None:
            return 500, {'error': 'no replay answer for request', 'hash': digest}, None
        missing = set(questions) - set(probs)
        if missing:
            return 500, {'error': 'replay answer missing question(s)', 'hash': digest,
                         'questions': sorted(missing)}, None
        try:
            values = {name: float(probs[name]) for name in questions}
        except (TypeError, ValueError):
            return 500, {'error': 'replay answer contains a non-numeric probability', 'hash': digest}, None
        if any(not math.isfinite(value) or value < 0 or value > 1 for value in values.values()):
            return 500, {'error': 'replay answer probability must be between 0 and 1', 'hash': digest}, None
        probs = values
        row = None
    return 200, {
        'answers': {name: {'noul': float(probs[name])} for name in questions},
        'usage': {'input_tokens': 0},
    }, row


def make_handler(mode, requests_path=None, answers_path=None):
    if mode not in ('record', 'replay'):
        raise ValueError('mode must be record or replay')
    if mode == 'record' and not requests_path:
        raise ValueError('record mode requires a requests JSONL path')
    if mode == 'replay' and not answers_path:
        raise ValueError('replay mode requires an answers JSONL path')
    answers = load_answers(answers_path)
    write_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def reply(self, status, value):
            body = json.dumps(value, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != '/v1/systemone':
                return self.reply(404, {'error': 'unknown path'})
            try:
                raw = self.rfile.read(int(self.headers.get('Content-Length', '0')))
                request = json.loads(raw.decode('utf-8'))
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
                return self.reply(400, {'error': str(exc)})
            status, response, row = answer_request(request, mode, answers)
            if row is not None:
                with write_lock:
                    with open(requests_path, 'a', encoding='utf-8') as dst:
                        dst.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
                        dst.flush()
            return self.reply(status, response)

    return Handler


def make_server(host, port, mode, requests_path=None, answers_path=None):
    return ThreadingHTTPServer((host, port), make_handler(mode, requests_path, answers_path))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', required=True, choices=('record', 'replay'))
    ap.add_argument('--requests', help='record mode output JSONL')
    ap.add_argument('--answers', help='replay mode input JSONL')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8341)
    a = ap.parse_args(argv)
    if a.mode == 'record':
        os.makedirs(os.path.dirname(os.path.abspath(a.requests)), exist_ok=True)
    server = make_server(a.host, a.port, a.mode, a.requests, a.answers)
    print(f'listening http://{a.host}:{server.server_port}/v1/systemone mode={a.mode}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
