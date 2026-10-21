"""Live probe (Claude, 2026-10-21): does a label set at create come back exactly in the v1 listing?
Rents the cheapest verified on-demand offer, lists, compares, destroys, confirms. Prints no key."""
import json, os, sys, time, uuid, urllib.request, urllib.parse
KEY = open(os.path.expanduser('~/.vast-credits.env')).read().strip()
def req(method, url, body=None):
    r = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body is not None else None,
                               headers={'Authorization': f'Bearer {KEY}', 'Content-Type': 'application/json', 'Accept': 'application/json'})
    try:
        return json.load(urllib.request.urlopen(r, timeout=60))
    except urllib.error.HTTPError as e:
        return {'http_error': e.code, 'body': e.read().decode()[:300]}
V0 = 'https://console.vast.ai/api/v0'; V1 = 'https://console.vast.ai/api/v1'
label = f"vastrun-{uuid.uuid4().hex[:12]}-{int(time.time()):x}"
q = {'num_gpus': {'eq': 1}, 'verified': {'eq': True}, 'rentable': {'eq': True}, 'type': 'on-demand',
     'reliability2': {'gte': 0.98}, 'disk_space': {'gte': 10}, 'order': [['dph_total', 'asc']], 'limit': 3}
offers = req('POST', V0 + '/bundles/', q).get('offers', [])
o = offers[0]; print('offer', o['id'], o.get('gpu_name'), round(o['dph_total'], 4))
iid = None
try:
    d = req('PUT', f"{V0}/asks/{o['id']}/", {'client_id': 'me', 'image': 'alpine:3.20', 'disk': 10, 'runtype': 'ssh_direct', 'label': label})
    print('rent', {k: d.get(k) for k in ('success', 'new_contract', 'http_error')}); iid = d.get('new_contract')
    for i in range(6):
        time.sleep(5)
        L = req('GET', V1 + '/instances/')
        keys = sorted(L.keys()) if isinstance(L, dict) else type(L).__name__
        items = L.get('instances') if isinstance(L, dict) else None
        mine = [x for x in (items or []) if str(x.get('id')) == str(iid)]
        print('list', i, 'top-keys', keys, 'n', len(items or []), 'label-key-present', bool(mine) and 'label' in mine[0],
              'label-match', bool(mine) and mine[0].get('label') == label, 'got', repr(mine[0].get('label'))[:60] if mine else None)
        if mine and 'label' in mine[0]: break
    print('other top-level fields', {k: L.get(k) for k in L if k != 'instances'} if isinstance(L, dict) else None)
    F = req('GET', V1 + '/instances/?' + urllib.parse.urlencode({'label': label})) if False else None
finally:
    if iid:
        print('destroy', req('DELETE', f'{V0}/instances/{iid}/'))
        for i in range(6):
            time.sleep(5)
            L = req('GET', V1 + '/instances/')
            ids = [str(x.get('id')) for x in (L.get('instances') or [])] if isinstance(L, dict) else None
            print('after', i, 'still listed', str(iid) in (ids or []), 'n', len(ids or []))
            if ids is not None and str(iid) not in ids: break
print('label', label, 'len', len(label))
