#!/usr/bin/env python3
"""Rent one bounded RTX 3090 instance, run Part 131, collect results, and destroy it."""
import argparse, io, json, os, shlex, subprocess, sys, tarfile, time, urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; API='https://console.vast.ai/api/v0'; MAX_RATE=.20; RELIABILITY=.98; DISK=130; MAX_SECONDS=120*60
BAD_OFFERS=set(); BAD_HOSTS=set(); REMOTE='/root/part131-run'; LABEL='part131-2026-10-30'
PACKAGE=('part131/select131.py','part131/run131.py','part131/label131.py','part131/score131.py','part131/units131.json',
         'part92/label92.py','part92b/label92b.py','part92b/known92b.jsonl','judge_client.py',
         'cloud/vast-part131-2026-10-30/run.sh','cloud/vast-part131-2026-10-30/run-main.sh')
def api(method,path,body=None):
    key=os.environ.get('VAST_KEY')
    if not key: raise RuntimeError('VAST_KEY is not set')
    req=urllib.request.Request(API+path,method=method,data=json.dumps(body).encode() if body is not None else None,
        headers={'Authorization':'Bearer '+key,'Content-Type':'application/json','Accept':'application/json'})
    with urllib.request.urlopen(req,timeout=60) as r: d=json.load(r)
    if not isinstance(d,dict) or 'http_error' in d: raise RuntimeError('invalid Vast API response')
    return d
def instances():
    d=api('GET','/instances/'); rows=d.get('instances')
    if not isinstance(rows,list): raise RuntimeError('Vast instances response lacks list')
    return rows
def offers():
    query={'gpu_name':{'eq':'RTX 3090'},'num_gpus':{'eq':1},'rentable':{'eq':True},'type':'on-demand','reliability2':{'gte':RELIABILITY},'disk_space':{'gte':DISK},'dph_total':{'lte':MAX_RATE},'order':[['dph_total','asc']],'limit':30}
    rows=api('POST','/bundles/',query).get('offers')
    if not isinstance(rows,list): raise RuntimeError('Vast offer response lacks list')
    return [x for x in rows if x.get('gpu_name')=='RTX 3090' and int(x.get('num_gpus',0))==1 and x.get('rentable') and float(x.get('reliability2',0))>=RELIABILITY and float(x.get('dph_total',9))<=MAX_RATE and float(x.get('disk_space',0))>=DISK and x.get('id') not in BAD_OFFERS and x.get('host_id') not in BAD_HOSTS]
def ssh_args(host,port,cmd):
    return ['ssh','-i',str(Path.home()/'.ssh/vast_ed25519'),'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','StrictHostKeyChecking=no','-o','UserKnownHostsFile=/dev/null','-o','ConnectTimeout=10','-p',str(port),f'root@{host}',cmd]
def details(inst):
    host=inst.get('public_ipaddr') or inst.get('ssh_host'); ports=inst.get('ports') or {}; candidates=ports.get('22/tcp',[])
    port=(candidates[0].get('HostPort') if candidates else None) or inst.get('ssh_port')
    return (host,int(port)) if host and port else (None,None)
def wait_ssh(iid,deadline):
    while time.monotonic()<deadline:
        inst=api('GET',f'/instances/{iid}/').get('instances',{})
        if isinstance(inst,list): inst=next((r for r in inst if str(r.get('id'))==str(iid)),{})
        if 'Error response' in str(inst.get('status_msg','')) or inst.get('actual_status') in ('error','failed','exited'): raise RuntimeError('Vast host failed to start')
        host,port=details(inst)
        if host and port and inst.get('actual_status')=='running':
            probe=subprocess.run(ssh_args(host,port,'true'),capture_output=True,timeout=30)
            if probe.returncode==0:return host,port
        time.sleep(8)
    raise TimeoutError('instance did not become SSH-ready before rental deadline')
def archive():
    b=io.BytesIO()
    with tarfile.open(fileobj=b,mode='w:gz') as t:
        for rel in PACKAGE:
            src=ROOT/rel
            if not src.is_file(): raise FileNotFoundError(rel)
            t.add(src,arcname=rel)
    return b.getvalue()
def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--stage',choices=('pilot','main'),default='pilot'); ap.add_argument('--instance-id'); ap.add_argument('--started-at',type=float); a=ap.parse_args()
    if a.stage=='main' and not a.instance_id: ap.error('--instance-id is required for --stage main')
    if a.stage=='main' and a.started_at is None: ap.error('--started-at is required to enforce the 120-minute rental cap')
    label=f'{LABEL}-{os.getpid()}-{int(time.time())}'; ids=set(); rented=False; retain=False; rc=0; rental_started=time.time()
    try:
        remaining=MAX_SECONDS-(time.time()-a.started_at) if a.stage=='main' else MAX_SECONDS
        if remaining<=0: raise TimeoutError('the 120-minute rental cap elapsed before main stage')
        deadline=time.monotonic()+remaining; host=port=None
        if a.stage=='main':
            iid=str(a.instance_id); ids.add(iid); host,port=wait_ssh(iid,deadline)
        else:
            candidates=offers()
            if not candidates: raise RuntimeError('no RTX 3090 offer meets $0.20/hour, reliability 0.98, and 130 GB disk')
            for offer in candidates[:5]:
                d=api('PUT',f"/asks/{offer['id']}/",{'client_id':'me','image':'pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel','disk':DISK,'runtype':'ssh_direct','label':label,'onstart':'touch ~/.no_auto_tmux'})
                iid=d.get('new_contract')
                if d.get('success') is False or iid is None: continue
                iid=str(iid); ids.add(iid); rented=True
                try: host,port=wait_ssh(iid,min(deadline,time.monotonic()+15*60)); break
                except Exception:
                    api('DELETE',f'/instances/{iid}/'); ids.discard(iid); continue
            if not host: raise RuntimeError('no eligible offer booted successfully')
        env={k:v for k,v in os.environ.items() if k!='VAST_KEY'}
        if a.stage=='pilot': subprocess.run(ssh_args(host,port,f'mkdir -p {REMOTE} && tar -xzf - -C {REMOTE}'),input=archive(),check=True,timeout=max(1,int(deadline-time.monotonic())),env=env)
        run_script='run.sh' if a.stage=='pilot' else 'run-main.sh'
        try: subprocess.run(ssh_args(host,port,f'cd {REMOTE} && bash cloud/vast-part131-2026-10-30/{run_script}'),check=True,timeout=max(1,int(deadline-time.monotonic())),env=env)
        except Exception as e: rc=2; print(f'remote run failed: {type(e).__name__}: {e}',file=sys.stderr)
        folders='results/part131/pilot' if a.stage=='pilot' else 'results/part131/main'
        cmd=f'cd {REMOTE} && tar -czf - {folders}'
        result=subprocess.run(ssh_args(host,port,cmd),stdout=subprocess.PIPE,check=True,timeout=max(1,int(deadline-time.monotonic())),env=env)
        with tarfile.open(fileobj=io.BytesIO(result.stdout),mode='r:gz') as t:
            for m in t.getmembers():
                p=Path(m.name)
                if p.is_absolute() or '..' in p.parts or not m.name.startswith('results/part131/'): raise RuntimeError('unexpected archive path')
            t.extractall(ROOT,members=t.getmembers())
        if a.stage=='pilot':
            retain=True
            print(f'PART131_REUSE_INSTANCE={iid} PART131_STARTED_AT={rental_started:.3f}; review pilot timing, then run-main with these values.',flush=True)
    except Exception as e: rc=2; print(f'vast.py: {type(e).__name__}: {e}',file=sys.stderr)
    finally:
        if ids and not retain:
            for iid in ids:
                try: api('DELETE',f'/instances/{iid}/')
                except Exception as e: print(f'destroy request failed: {type(e).__name__}',file=sys.stderr)
        try: remains=bool(instances())
        except Exception: remains=True
        print(f'PART131_INSTANCE_REMAINS={"yes" if remains else "no"} (listing errors count as yes)',flush=True)
    return rc
if __name__=='__main__':raise SystemExit(main())
