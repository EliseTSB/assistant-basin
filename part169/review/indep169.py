import json
from pathlib import Path
SCR=Path('/tmp/claude-1000/-home-elisepc-green-Journal/f862e88e-c243-45f9-aef3-92eba49bcbff/scratchpad')
FIX=SCR/'fix169'
rd=lambda p:[json.loads(x) for x in (FIX/p).read_text().splitlines() if x.strip()]
masses=rd('masses.jsonl'); answers=rd('answers.jsonl')
LOOP='red blue green red blue green red blue green'
key=lambda r:(r['arm'],r['layer'],r['slot'],r['question_index'])
M={key(m):m for m in masses}
bat=[a for a in answers if a['stage']=='battery']
loop={(key(a),a['ban']):(a['answer']==LOOP) for a in bat}
def rate(ban,pred=lambda k:True,arms=('n+','k4')):
    xs=[v for (k,b),v in loop.items() if b==ban and k[0] in arms and pred(k)]
    return sum(xs)/len(xs), len(xs), sum(xs)
none_loop={k:loop[(k,'none')] for k in M}
high=lambda k:M[k]['high_row']
paired=lambda k:M[k]['paired_set']
none=rate('none'); own=rate('own'); free=rate('free'); clear=rate('clear'); plain50=rate('plain50'); hit=rate('hit')
print('rates none/own/free/clear/plain50/hit', [round(x[0],5) for x in (none,own,free,clear,plain50,hit)], 'counts', [x[1] for x in (none,own,free,clear,plain50,hit)])
print('pred 169.1 relcut', round(1-own[0]/none[0],5))
print('pred 169.3 rise', round(free[0]/none[0]-1,5))
hn=rate('none',high); ho=rate('own',high); on_=rate('none',lambda k:not high(k)); oo=rate('own',lambda k:not high(k))
print('pred 169.2 diff', round((1-ho[0]/hn[0])-(1-oo[0]/on_[0]),5))
pf=rate('free',paired); pc=rate('clear',paired); print('pred 169.4 diff', round(pf[0]-pc[0],5))
def induced(ban,pred):
    xs=[loop[(k,ban)] for k in M if pred(k) and not none_loop[k]]
    return sum(xs)/len(xs), len(xs), sum(xs)
hp=lambda k:high(k) and paired(k)
fi=induced('free',hp); ci=induced('clear',hp); print('pred 169.5 diff', round(fi[0]-ci[0],5), 'free_ind', round(fi[0],5), 'clear_ind', round(ci[0],5), 'elig', fi[1])
print('pred 169.7 relcut clear', round(1-pc[0]/rate('none',paired)[0],5))
pl2=lambda k:M[k]['lists']['free']['plain_overlap']>=2
hi_=induced('hit',pl2); fi2=induced('free',pl2); print('pred 169.8 hit', round(hi_[0],5), 'free', round(fi2[0],5), 'ratio test', hi_[0]>=0.6*fi2[0])
np_free=[a for a in answers if a['stage']=='no_push' and a['arm']=='n+' and a['ban']=='free']
print('pred 169.6 no-push free rate', sum(a['answer']==LOOP for a in np_free)/len(np_free), 'counts', sum(a['answer']==LOOP for a in np_free), len(np_free))
w2n=plain50[2]-none[2]; w2o=none[2]-own[2]; print('pred W2 net rise', w2n, 'own cut', w2o, 'pass', w2n>=w2o/3)
