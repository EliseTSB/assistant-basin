"""Tiny Vast.ai REST helper. Reads the bare key from ~/.vast-credits.env, never prints it.
Usage: vast.py offers | rent <offer_id> | show <instance_id> | destroy <instance_id> | user | instances"""
import json, os, sys, urllib.request
KEY = open(os.path.expanduser('~/.vast-credits.env')).read().strip()
BASE = 'https://console.vast.ai/api/v0'
def req(method, path, body=None):
    url = ('https://console.vast.ai/api/v1' + path[2:]) if path.startswith('V1') else BASE + path
    r = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body is not None else None,
                               headers={'Authorization': f'Bearer {KEY}', 'Content-Type': 'application/json', 'Accept': 'application/json'})
    try:
        return json.load(urllib.request.urlopen(r, timeout=60))
    except urllib.error.HTTPError as e:
        return {'http_error': e.code, 'body': e.read().decode()[:500]}
cmd = sys.argv[1]
if cmd == 'offers':
    q = {'gpu_name': {'eq': 'RTX 3090'}, 'num_gpus': {'eq': 1}, 'verified': {'eq': True}, 'rentable': {'eq': True},
         'type': 'on-demand', 'reliability2': {'gte': 0.98}, 'disk_space': {'gte': 60}, 'cuda_max_good': {'gte': 12.4},
         'order': [['dph_total', 'asc']], 'limit': 8}
    d = req('POST', '/bundles/', q)
    for o in d.get('offers', [])[:8]:
        print(o['id'], round(o['dph_total'], 3), round(o.get('reliability2', 0), 3), o.get('disk_space'), o.get('cuda_max_good'),
              o.get('geolocation'), round(o.get('inet_down', 0)))
    if 'offers' not in d: print('no offers', d.get('http_error'), str(d.get('body', ''))[:200])
elif cmd == 'rent':
    body = {'client_id': 'me', 'image': 'pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel', 'disk': 60, 'runtype': 'ssh_direct',
            'label': 'part27-run', 'onstart': 'touch ~/.no_auto_tmux'}
    d = req('PUT', f'/asks/{sys.argv[2]}/', body)
    print({k: d.get(k) for k in ('success', 'new_contract', 'http_error', 'error', 'msg')})
elif cmd == 'show':
    d = req('GET', f'/instances/{sys.argv[2]}/')
    i = d.get('instances', d)
    if isinstance(i, dict):
        print({k: i.get(k) for k in ('id', 'actual_status', 'status_msg', 'ssh_host', 'ssh_port', 'public_ipaddr', 'dph_total', 'gpu_name', 'geolocation', 'start_date')}, 'ports', json.dumps(i.get('ports'))[:300])
    else: print('unexpected', sorted(d.keys()) if isinstance(d, dict) else type(d).__name__)
elif cmd == 'instances':
    for path in ('V1/instances/',):
        d = req('GET', path)
        print(path, d.get('http_error'), str(d.get('body',''))[:200], sorted(d.keys())[:8] if isinstance(d, dict) else type(d).__name__,
              [(i.get('id'), i.get('actual_status'), i.get('label')) for i in (d.get('instances') or [])] if isinstance(d, dict) else '')
elif cmd == 'destroy':
    print(req('DELETE', f'/instances/{sys.argv[2]}/'))
elif cmd == 'user':
    d = req('GET', '/users/current/'); print({k: d.get(k) for k in ('credit', 'balance', 'total_spend')})
