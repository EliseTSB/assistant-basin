#!/usr/bin/env python3
# One judge call for judge_register.py --cmd: read a prompt on stdin, ask GPT-6 Luna through the local CLIProxyAPI
# (127.0.0.1:8317, ChatGPT subscription, no key), print the answer text. Same request shape as judge.py's cliproxy route.
import json, sys, urllib.request
body = {'model': 'gpt-6-luna', 'max_tokens': 4000, 'thinking': {'type': 'adaptive'}, 'output_config': {'effort': 'high'},
        'messages': [{'role': 'user', 'content': sys.stdin.read()}]}
req = urllib.request.Request('http://127.0.0.1:8317/v1/messages', data=json.dumps(body).encode(),
                             headers={'content-type': 'application/json', 'anthropic-version': '2023-06-01', 'x-api-key': 'x'})
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(req, timeout=600) as r:
    d = json.load(r)
print(''.join(b.get('text', '') for b in d.get('content', []) if b.get('type') == 'text').strip())
