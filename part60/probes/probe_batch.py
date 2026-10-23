"""One-request live probe batch (2026-10-23). WHAT IT MEASURES: create-response fields, list order and after_id direction, batch completion time, result format and usage fields for a Haiku 5.5 batch. WHAT IT GOT WRONG THE FIRST TIME: nothing; the first run confirmed every assumption. It spends about $0.00001. Needs SSL_CERT_FILE=$HOME/.mitmproxy/ca-bundle-combined.crt."""
import json, os, sys, time, urllib.request, urllib.error
for line in open(os.path.expanduser('~/.anthropic-credits.env')):
    line=line.strip()
    if line and not line.startswith('#') and '=' in line:
        k,v=line.split('=',1); os.environ[k.replace('export ','').strip()]=v.strip().strip('"').strip("'")
KEY=os.environ.get('ANTHROPIC_CREDITS_KEY') or os.environ.get('ANTHROPIC_API_KEY')
H={'x-api-key':KEY,'anthropic-version':'2023-06-01','content-type':'application/json'}
def call(m,path,body=None):
    r=urllib.request.Request('https://api.anthropic.com'+path,method=m,data=json.dumps(body).encode() if body is not None else None,headers=H)
    try:
        with urllib.request.urlopen(r,timeout=60) as f: return f.status, f.read().decode()
    except urllib.error.HTTPError as e: return e.code, e.read().decode()
body={'requests':[{'custom_id':'probe-1','params':{'model':'claude-haiku-5-5','max_tokens':64,'messages':[{'role':'user','content':'Reply with the single word: ok'}]}}]}
s,t=call('POST','/v1/messages/batches',body); print('create',s); d=json.loads(t); print({k:d.get(k) for k in ('id','type','processing_status','created_at','request_counts','expires_at','results_url')}) if s==200 else print(t[:400])
bid=d.get('id')
s,t=call('GET','/v1/messages/batches?limit=3'); l=json.loads(t); print('list',s,'has_more',l.get('has_more'),'first_id',l.get('first_id'),'last_id',l.get('last_id'),[ (x['id'][-6:],x['created_at']) for x in l.get('data',[])])
if l.get('data') and len(l['data'])>1: print('newest_first', l['data'][0]['created_at']>=l['data'][-1]['created_at'])
s,t=call('GET',f'/v1/messages/batches?limit=1&after_id={l["data"][0]["id"]}'); l2=json.loads(t); print('after_id page ->', [(x['id'][-6:],x['created_at']) for x in l2.get('data',[])], 'older' if l2.get('data') and l2['data'][0]['created_at']<l['data'][0]['created_at'] else 'NOT older')
for i in range(40):
    s,t=call('GET',f'/v1/messages/batches/{bid}'); d=json.loads(t)
    if d.get('processing_status')=='ended': break
    time.sleep(15)
print('status',d.get('processing_status'),d.get('request_counts'))
s,t=call('GET',f'/v1/messages/batches/{bid}/results'); print('results',s); print(t[:900])
