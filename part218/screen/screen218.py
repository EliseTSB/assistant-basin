"""Part 218 v4 dev screen: unpushed answers under four maker-slot openings, on SFT, DPO and Instruct.

Not a registered run. The pilot (results218/PILOT.md) stopped by rule because unpushed Instruct named Ai2 on only
10 of 44 prompts. Vesper's review (council vesper-part218-design-2026-11-21 #11) asked for rotated openings and a
screen on a DEV prompt set kept apart from the registered one. The 44 registered prompts are the dev set here
(their Instruct answers are already seen); the v4 registered run gets a fresh held-out set written before it runs.

Writes one JSON line per generation: stage, prompt_index, opening, continuation. Classification happens off the VM
with attrib218.classify(opening + continuation), so the screen and the registered scorer share one rule.
"""
import argparse, json, sys, time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

OPENINGS = ['', 'I was created by', 'I was developed by', 'I was built by', 'My creators are']


def build_prompt(tok, question):
    messages = [{'role': 'user', 'content': question}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pins', required=True)
    ap.add_argument('--prompts', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-new', type=int, default=48)
    ap.add_argument('--stages', default='SFT,DPO,Instruct')
    a = ap.parse_args()
    pins = json.loads(Path(a.pins).read_text(encoding='utf-8'))
    prompts = json.loads(Path(a.prompts).read_text(encoding='utf-8'))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if dev == 'cuda' else torch.float32
    print(f'device {dev} dtype {dtype} torch {torch.__version__}', flush=True)
    with out.open('w', encoding='utf-8') as f:
        for stage in a.stages.split(','):
            pin = pins[stage]
            t0 = time.time()
            tok = AutoTokenizer.from_pretrained(pin['repo'], revision=pin['revision'])
            tok.padding_side = 'left'
            model = AutoModelForCausalLM.from_pretrained(pin['repo'], revision=pin['revision'], dtype=dtype).to(dev)
            model.eval()
            print(f'{stage} loaded in {time.time()-t0:.1f}s', flush=True)
            for opening in OPENINGS:
                texts = [build_prompt(tok, q) + opening for q in prompts]
                enc = tok(texts, add_special_tokens=False, return_tensors='pt', padding=True).to(dev)
                with torch.no_grad():
                    gen = model.generate(**enc, do_sample=False, max_new_tokens=a.max_new,
                                         pad_token_id=tok.pad_token_id)
                new = gen[:, enc['input_ids'].shape[1]:]
                for i in range(len(prompts)):
                    cont = tok.decode(new[i], skip_special_tokens=True)
                    f.write(json.dumps({'stage': stage, 'revision': pin['revision'], 'prompt_index': i,
                                        'opening': opening, 'continuation': cont}) + '\n')
                f.flush()
                print(f'{stage} opening={opening!r} done {time.time()-t0:.1f}s', flush=True)
            del model
            if dev == 'cuda':
                torch.cuda.empty_cache()
    print('SCREEN EXIT 0', flush=True)


if __name__ == '__main__':
    main()
