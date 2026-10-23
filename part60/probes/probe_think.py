"""Thinking-configuration probe for Haiku 5.5 (2026-10-23). WHAT IT MEASURES: content block types, thinking text length, output tokens and usage.output_tokens_details.thinking_tokens under default / adaptive+high / disabled / enabled-with-budget. WHAT IT TAUGHT US: Haiku 5.5 thinks by default; thinking TEXT IS OMITTED (block present, empty text, thinking_tokens > 0); thinking.type enabled is rejected with HTTP 400 (use adaptive plus output_config.effort). A check that looks for non-empty thinking text can never pass. Costs about $0.01."""
import json, os, urllib.request, urllib.error
for line in open(os.path.expanduser('~/.anthropic-credits.env')):
    line=line.strip()
    if line and not line.startswith('#') and '=' in line:
        k,v=line.split('=',1); os.environ[k.replace('export ','').strip()]=v.strip().strip('"').strip("'")
KEY=os.environ.get('ANTHROPIC_CREDITS_KEY') or os.environ.get('ANTHROPIC_API_KEY')
H={'x-api-key':KEY,'anthropic-version':'2023-06-01','content-type':'application/json'}
def msg(extra,prompt):
    body={'model':'claude-haiku-5-5','max_tokens':4000,'messages':[{'role':'user','content':prompt}]}; body.update(extra)
    r=urllib.request.Request('https://api.anthropic.com/v1/messages',method='POST',data=json.dumps(body).encode(),headers=H)
    try:
        with urllib.request.urlopen(r,timeout=90) as f: d=json.load(f)
    except urllib.error.HTTPError as e: return {'http':e.code,'body':e.read().decode()[:300]}
    return {'types':[c['type'] for c in d['content']],'think_chars':sum(len(c.get('thinking') or '') for c in d['content']),'usage_out':d['usage']['output_tokens'],'thinking_tokens':(d['usage'].get('output_tokens_details') or {}).get('thinking_tokens'),'stop':d['stop_reason']}
P='A neuron fires on the token after the word "abstract" in scientific papers. Write a one-sentence hypothesis and three test sentences that should make it fire, three that should not. Think carefully about what the neuron might really encode.'
for name,extra in [('default (no thinking field)',{}),('adaptive+high',{'thinking':{'type':'adaptive'},'output_config':{'effort':'high'}}),('disabled',{'thinking':{'type':'disabled'}}),('enabled budget 2000',{'thinking':{'type':'enabled','budget_tokens':2000}})]:
    print(name, msg(extra,P))
