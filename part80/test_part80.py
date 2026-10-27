from __future__ import annotations
import ast, json
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
from part80 import part80 as run
from part80 import score80 as score

class TinyOffsets:
    def __call__(self,text,return_offsets_mapping=False,add_special_tokens=False,return_tensors=None):
        # One token per whitespace-delimited word, with exact character offsets.
        import re
        offsets=[(m.start(),m.end()) for m in re.finditer(r'\S+',text)]
        ids=list(range(len(offsets)))
        if return_tensors=='pt':
            import torch
            return {'input_ids':torch.tensor([ids])}
        return {'input_ids':ids,'offset_mapping':offsets}

def test_body_span_uses_full_render_offsets_for_all_roles_including_t():
    tok=TinyOffsets(); body='planet orchid comet'; prefix='<think>\n'
    cases=[('<<system>>\n'+body+'\n<<user>> question','system',''),('<<user>>\n'+body+'\n<<user>> question','user',''),('<<assistant>>'+prefix+body+'\n<<user>> question','assistant',prefix),('<<user>>\n'+prefix+body+'\n<<user>> question','user',prefix)]
    for rendered,role,pfx in cases:
        start,end=run.locate_body_span(tok,rendered,body,role,pfx)
        cs,ce=run._role_body_start(rendered,role,body,pfx)
        assert rendered[cs:ce]==body
        offsets=tok(rendered,return_offsets_mapping=True)['offset_mapping']
        expected=[i for i,(a,b) in enumerate(offsets) if b>a and a<ce and b>cs]
        assert (start,end)==(expected[0],expected[-1]+1)

def test_distance_ratio_formula_hand_built():
    # R_A = D_US / mean(D_AU,D_AS), and analogously for U/S.
    d={'A|U':2.0,'A|S':4.0,'U|S':3.0}
    got=score.ratios(d,0)
    assert got['A']==pytest.approx(1.0)
    assert got['U']==pytest.approx(4/2.5)
    assert got['S']==pytest.approx(2/3.5)

def _layer(i,ra=.5,ash=.85,ush=.4,tsh=.3,shuffle=.33):
    return {'layer':i,'median_R_A':ra,'A_most_distant_share':ash,'Uprime_most_distant_share':ush,'T_most_distant_share':tsh,'shuffled_label_share':shuffle}

def test_reading_rule_chooses_first_qualifying_second_half_layer():
    layers=[_layer(i,ra=.5,ash=.95,ush=.4,tsh=.3) for i in range(1,9)]
    layers[3]=_layer(4,ra=.1,ash=1,ush=0,tsh=0)
    layers[4]=_layer(5)
    assert score.choose_reading_layer(layers)==5

def test_reading_rule_requires_context_and_template_controls():
    layers=[_layer(i) for i in range(1,9)]
    layers[4]=_layer(5,ush=.7)
    layers[5]=_layer(6,tsh=.8)
    assert score.choose_reading_layer(layers)==7

def test_verdict_not_read_when_control_fails(tmp_path):
    # An actual minimal serialized runner result exercises the public scorer's gate.
    runmeta={'model':'qwen3-1.7b','revision':'rev','layers':2,'prefix_text':'','model_id':'q'}
    (tmp_path/'run.json').write_text(json.dumps(runmeta),encoding='utf-8')
    (tmp_path/'distances.json').write_text('{}',encoding='utf-8')
    rows=[{'body_id':'x','split':'heldout','rendering':'A','order':1,'letter_probs':{'A':.2,'B':.1,'C':.1},'letter_mass':.4,'role_probs':{'you':.5,'user':.25,'system':.25}},
          {'body_id':'x','split':'heldout','rendering':'U','order':1,'letter_probs':{'A':.04,'B':.16,'C':.2},'letter_mass':.4,'role_probs':{'you':.1,'user':.4,'system':.5}}]
    (tmp_path/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    np.savez_compressed(tmp_path/'means.npz',body_ids=np.array([],dtype=str),split=np.array([],dtype=str))
    result=score.metrics_for_dir(tmp_path)
    assert result['verdicts']['80.5']['status']=='NOT READ'
    assert result['verdicts']['80.5']['holds'] is None
    assert result['verdicts']['80.6']['status']=='NOT READ'

def test_every_file_call_names_its_encoding():
    bad=[]
    for path in (ROOT/'part80').glob('*.py'):
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr in ('open','read_text','write_text'):
                if not any(k.arg=='encoding' for k in node.keywords): bad.append(f'{path.name}:{node.lineno} {node.func.attr}')
    assert not bad,bad
