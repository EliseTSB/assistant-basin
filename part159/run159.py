#!/usr/bin/env python3
"""Part 159 token-veto and activation recorder."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from part157 import run157 as p157
from part153 import battery153 as p153

MODEL_DEFAULT = p157.MODEL_DEFAULT
LAYERS = (6, 7, 11, 27, 29)
NAMED = ((6, 753), (7, 1169), (11, 8868), (29, 776), (27, 3325))
MAKER = ['Alibaba', 'Tongyi', 'Aliyun', '阿里', '通义']
CONTROL = ['model']

def spelling_ids(tok, spellings):
    """Part 157's bare/space/** variants applied only to literal spellings."""
    if isinstance(spellings, str): spellings = [spellings]
    out, seen = [], set()
    for spelling in spellings:
        if not isinstance(spelling, str) or not spelling: raise ValueError('spellings must be nonempty strings')
        for variant in (spelling, ' ' + spelling, '**' + spelling):
            raw = p157._ids(tok, variant)
            if hasattr(raw, 'tolist'): raw = raw.tolist()
            if raw and isinstance(raw[0], list): raw = raw[0]
            seq = [int(x) for x in raw]
            if variant.startswith('**'):
                prefix = p157._ids(tok, '**')
                if hasattr(prefix, 'tolist'): prefix = prefix.tolist()
                if prefix and isinstance(prefix[0], list): prefix = prefix[0]
                prefix = [int(x) for x in prefix]
                if prefix and seq[:len(prefix)] == prefix: seq = seq[len(prefix):]
            if seq and tuple(seq) not in seen:
                out.append(seq); seen.add(tuple(seq))
    return out

class VetoRecorder:
    """Records pre-ban choice, then returns the no-bad-words scores."""
    def __init__(self, bad_words_ids, eos_ids):
        from transformers import NoBadWordsLogitsProcessor
        self.torch = __import__('torch')
        self.ban = NoBadWordsLogitsProcessor(bad_words_ids, eos_ids) if bad_words_ids else None
        self.pre_token, self.top_logit, self.emitted, self.emitted_logit = [], [], [], []
    def __call__(self, input_ids, scores):
        torch = self.torch
        pre = scores.argmax(dim=-1)  # argmax, as generate does: vocabulary-axis top-k stalls the A770 (gotcha)
        top_logit = scores.gather(1, pre[:, None]).squeeze(1)
        processed = self.ban(input_ids, scores) if self.ban is not None else scores
        emitted = processed.argmax(dim=-1)
        emitted_logit = scores.gather(1, emitted[:, None]).squeeze(1)
        self.pre_token.append(pre.detach().clone()); self.top_logit.append(top_logit.detach().clone())
        self.emitted.append(emitted.detach().clone()); self.emitted_logit.append(emitted_logit.detach().clone())
        return processed

class LayerRecorder:
    def __init__(self, layers, named, keep_prompt=False):
        self.values = {l: [] for l in LAYERS}
        self.prompt = None; self.keep_prompt = keep_prompt; self.named = named; self.handles = []
        self.calls = 0
        for layer in LAYERS:
            self.handles.append(layers[layer].mlp.down_proj.register_forward_pre_hook(self._hook(layer)))
    def _hook(self, layer):
        def hook(_module, args):
            import torch
            x = args[0]
            if x.ndim != 3 or x.shape[0] != 24:
                raise ValueError(f'expected (24,n,width) hook input, got {tuple(x.shape)}')
            self.values[layer].append(x[:, -1, :].detach().clone())
            if self.keep_prompt and self.calls == 0:
                cols = [x[:, :, ch].detach().clone() for lay, ch in self.named if lay == layer]
                if cols:
                    if self.prompt is None: self.prompt = {}
                    self.prompt[layer] = torch.stack(cols, dim=-1)
            if layer == LAYERS[-1]: self.calls += 1
        return hook
    def remove(self):
        for h in self.handles: h.remove()
        self.handles.clear()

def _np16(t): return t.detach().to('cpu').float().numpy().astype(np.float16)
def _np32(t): return t.detach().to('cpu').float().numpy().astype(np.float32)
def _npint(t): return t.detach().to('cpu').long().numpy().astype(np.int32)

def _run_arm(model, tok, layers, questions, ban_ids, arm):
    import torch
    from transformers import LogitsProcessorList
    device = str(next(model.parameters()).device)
    inputs, mask, prompts = p157.rendered_inputs(tok, questions, device)
    eos = p157.p148._common().eos_ids(tok, model)
    recorder = VetoRecorder(ban_ids, eos)
    layer_rec = LayerRecorder(layers, NAMED, keep_prompt=(arm == 'free'))
    processors = LogitsProcessorList([recorder] if recorder else [])
    try:
        with torch.inference_mode():
            seq = model.generate(input_ids=inputs, attention_mask=mask, max_new_tokens=40,
                do_sample=False, use_cache=True, pad_token_id=getattr(tok, 'pad_token_id', None),
                eos_token_id=eos or None, logits_processor=processors)
    finally:
        layer_rec.remove()
    calls = len(recorder.pre_token)
    if layer_rec.calls != calls: raise RuntimeError(f'{arm}: activation calls {layer_rec.calls} != logits steps {calls}')
    if not 1 <= calls <= 40: raise RuntimeError(f'{arm}: unexpected generation call count {calls}')
    finite_tensors = [*recorder.pre_token, *recorder.top_logit, *recorder.emitted, *recorder.emitted_logit]
    for vals in layer_rec.values.values(): finite_tensors.extend(vals)
    if layer_rec.prompt: finite_tensors.extend(layer_rec.prompt.values())
    if not all(bool(torch.isfinite(x).all()) for x in finite_tensors):
        return None
    # Token ids need integers (the vocabulary is 248,070 wide, past float16's range); logits keep float32.
    steps = {'pre_token':_npint(torch.stack(recorder.pre_token, dim=1)),
             'top_logit':_np32(torch.stack(recorder.top_logit, dim=1)),
             'emitted_token':_npint(torch.stack(recorder.emitted, dim=1)),
             'emitted_logit':_np32(torch.stack(recorder.emitted_logit, dim=1))}
    acts = {l:_np16(torch.stack(v, dim=1)) for l,v in layer_rec.values.items()}
    prompt = None
    if layer_rec.prompt:
        prompt = {l:_np16(v) for l,v in layer_rec.prompt.items()}
        n = int(mask.shape[1]);
        for l, a in prompt.items():
            if a.shape != (24, n, sum(1 for lay,_ in NAMED if lay == l)):
                raise RuntimeError(f'prompt activation shape mismatch at layer {l}: {a.shape}')
    results=[]
    for i in range(24):
        ids = seq[i, inputs.shape[1]:].tolist()
        # Generation keeps finished rows padded; retain each row through its first EOS.
        if eos:
            for j, tid in enumerate(ids):
                if tid in eos: ids=ids[:j+1]; break
        decoded=[tok.decode([int(t)], skip_special_tokens=False) for t in ids]
        results.append({'arm':arm,'question_index':i,'question':questions[i],
          'answer':tok.decode(ids, skip_special_tokens=True).strip(),'token_count':len(ids),
          'emitted_token_ids':ids,'decoded_emitted_tokens':decoded})
    return results, steps, acts, prompt

def run(args, loader=None):
    common=p157.p148._common(); questions=p153.DIRECT+p153.INDIRECT
    if len(questions)!=24: raise ValueError('Part 153 must supply 24 phrasings')
    level0_rows=p157._read_answers(args.level0)
    if len([r for r in level0_rows if r.get('arm')=='level_0']) != 24:
        raise ValueError('level0 reference must contain exactly 24 Part 157 level_0 rows')
    if loader is None:
        import neuron_report as nr
        tok,model=nr.load_model(args.model,common.REVISION,args.device); model.requires_grad_(False); model.eval()
    else: tok,model=loader(args.device)
    layers=common.find_layers(model); batch_safe=p157.assert_padding_equivalence(model,tok,questions[:2],args.device)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    bans={'maker_ban':spelling_ids(tok,MAKER),'control_ban':spelling_ids(tok,CONTROL)}
    all_rows=[]; all_steps={}; all_acts={}; prompt=None; retries=[]
    for arm in ('free','maker_ban','control_ban'):
        ids=[] if arm=='free' else bans[arm]
        # The recorder is present in every arm; [] denotes its pass-through mode.
        for attempt in range(3):
            result=_run_arm(model,tok,layers,questions,ids,arm)
            if result is not None: break
            retries.append({'arm':arm,'attempt':attempt+1})
        else: raise RuntimeError(f'{arm} remained non-finite after 3 attempts')
        rows,steps,acts,pr=result; all_rows.extend(rows); all_steps[arm]=steps; all_acts[arm]=acts
        if arm=='free': prompt=pr
    with (out/'answers.jsonl').open('w',encoding='utf-8') as f:
        for row in all_rows: f.write(json.dumps(row,ensure_ascii=True)+'\n')
    np.savez_compressed(out/'steps.npz',**{f'{arm}_{key}':v for arm,data in all_steps.items() for key,v in data.items()})
    for arm,data in all_acts.items():
        for layer,array in data.items(): np.save(out/f'layers_{arm}_L{layer}.npy',array)
    if prompt is None: raise RuntimeError('free prompt-position activations missing')
    np.savez_compressed(out/'prompt_free.npz',**{f'L{l}':a for l,a in prompt.items()},attention_mask=mask_cpu(tok,questions,args.device))
    meta={'bans':{k:{'sequences':v,'decoded':[tok.decode(x,skip_special_tokens=False) for x in v]} for k,v in bans.items()},
      'counts':{a:len([r for r in all_rows if r['arm']==a]) for a in ('free','maker_ban','control_ban')},
      'steps':{a:int(v['pre_token'].shape[1]) for a,v in all_steps.items()},'level0_reference_rows':24,
      'nonfinite_retries':retries,'batch_safe':bool(batch_safe)}
    (out/'run.json').write_text(json.dumps(meta,ensure_ascii=True,indent=2)+'\n',encoding='utf-8')
    return meta

def mask_cpu(tok, questions, device):
    _, mask, _ = p157.rendered_inputs(tok, questions, device)
    return mask.detach().cpu().numpy().astype(np.uint8)

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--device',default='xpu'); ap.add_argument('--model',default=MODEL_DEFAULT)
    ap.add_argument('--level0',required=True); ap.add_argument('--out',required=True); args=ap.parse_args(argv)
    result=run(args); print(json.dumps(result,ensure_ascii=True))
if __name__=='__main__': main()
