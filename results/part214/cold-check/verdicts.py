import json, sys
m = json.load(open('/tmp/claude-1000/part214-coldcheck/mine_counts.json'))
cg = {a: m[a]['clean_google'] for a in m}
br = {a: m[a]['broken'] for a in m}
R = lambda a: cg[a]/192
print('rates (clean_google/192):')
for a in m: print(f'  {a:22s} {cg[a]:3d}/192 = {R(a):.6f}   broken {br[a]}')
print()
print('=== 214.1 ===')
c = {}
c['zero broken<=0.15'] = br['zero']/192 <= 0.15
c['zero-unchanged>=+0.10'] = R('zero')-R('unchanged') >= 0.10
c['CIfLo>0 (my bootstrap lo=0.098958)'] = 0.098958 > 0
c['trace_fact broken<=0.15'] = br['trace_resample_fact']/192 <= 0.15
c['trace_fact-unchanged>=+0.04'] = R('trace_resample_fact')-R('unchanged') >= 0.04
c['trace_you broken<=0.15'] = br['trace_resample_you']/192 <= 0.15
c['trace_fact-trace_you>=+0.03'] = R('trace_resample_fact')-R('trace_resample_you') >= 0.03
for k,v in c.items(): print(f'  {k:42s} {v}')
print('  values: zero-unchanged', f"{R('zero')-R('unchanged'):+.6f}", '| fact-unchanged', f"{R('trace_resample_fact')-R('unchanged'):+.6f}", '| fact-you', f"{R('trace_resample_fact')-R('trace_resample_you'):+.6f}")
print('  VERDICT:', 'HOLDS' if all(c.values()) else 'FAILS')
print()
print('=== 214.4 ===')
delta_gen = 0.0738894053519245 - (-0.0013541574369483169)
dial_gap = R('scale_dial_fact')-R('scale_dial_you'); donor_gap = R('trace_resample_fact')-R('trace_resample_you')
c={}
c['broken<=28/192 on 4 arms'] = all(br[a] <= 28 for a in ('scale_dial_fact','trace_resample_fact','scale_dial_you','trace_resample_you'))
c['unchanged clean_google<=11'] = cg['unchanged'] <= 11
c['unchanged broken<=9'] = br['unchanged'] <= 9
c['availability: |delta_gen|>=0.05'] = abs(delta_gen) >= 0.05
c['availability: donor_gap>=0.03'] = donor_gap >= 0.03
c['(a) |dial_fact-trace_fact|<=0.05'] = abs(R('scale_dial_fact')-R('trace_resample_fact')) <= 0.05
c['(a) dial_fact-unchanged>=+0.04'] = R('scale_dial_fact')-R('unchanged') >= 0.04
c['(b) dial_gap>=0.60*donor_gap'] = dial_gap >= 0.60*donor_gap
c['(b) |dial_you-trace_you|<=0.05'] = abs(R('scale_dial_you')-R('trace_resample_you')) <= 0.05
c['(b) recode (cg+br)(dial_you)<cg(dial_fact)'] = (cg['scale_dial_you']+br['scale_dial_you']) < cg['scale_dial_fact']
c['(c) |dial_gap-donor_gap|<=0.05'] = abs(dial_gap-donor_gap) <= 0.05
for k,v in c.items(): print(f'  {k:48s} {v}')
print(f'  values: delta_gen {delta_gen:.6f} | dial_gap {dial_gap:.6f} | donor_gap {donor_gap:.6f} | 60% donor {0.6*donor_gap:.6f} | capture {dial_gap/donor_gap:.4f} | recode {cg["scale_dial_you"]+br["scale_dial_you"]} < {cg["scale_dial_fact"]}')
print('  VERDICT:', 'HOLDS' if all(c.values()) else 'FAILS')
print()
print('=== 214.5 ===')
drop = R('trace_resample_fact')-R('prompt_fact_gen_you')
c={}
c['broken(prompt_fact_gen_you)<=0.15'] = br['prompt_fact_gen_you']/192 <= 0.15
c['donor_gap>=0.03'] = donor_gap >= 0.03
c['drop > 0.50*donor_gap'] = drop > 0.50*donor_gap
c['drop >= 0.03'] = drop >= 0.03
c['recode (cg+br)(pfgy) < cg(trace_fact)'] = (cg['prompt_fact_gen_you']+br['prompt_fact_gen_you']) < cg['trace_resample_fact']
for k,v in c.items(): print(f'  {k:48s} {v}')
print(f'  values: drop {drop:.6f} | half donor gap {0.5*donor_gap:.6f} | share {drop/donor_gap:.4f} | recode {cg["prompt_fact_gen_you"]+br["prompt_fact_gen_you"]} < {cg["trace_resample_fact"]}')
print('  VERDICT:', 'HOLDS' if all(c.values()) else 'FAILS')
print()
print('=== 214.6 ===')
jump = R('mean_resample_you')-R('trace_resample_you')
c={}
c['broken(mean_resample_you)<=0.15'] = br['mean_resample_you']/192 <= 0.15
c['jump <= +0.05'] = jump <= 0.05
c['zero-mean_resample_you >= 0.04'] = R('zero')-R('mean_resample_you') >= 0.04
for k,v in c.items(): print(f'  {k:48s} {v}')
print(f'  values: jump {jump:+.6f} | zero-mean {R("zero")-R("mean_resample_you"):+.6f}')
print('  VERDICT:', 'HOLDS' if all(c.values()) else 'FAILS')
