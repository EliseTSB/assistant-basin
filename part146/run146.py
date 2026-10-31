#!/usr/bin/env python3
"""Store Part 146 forced-next-token distributions and unpushed references."""
from __future__ import annotations
import argparse, json, os, sys, tempfile, time
from pathlib import Path
try:
    import numpy as np
except ModuleNotFoundError:
    np = None
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for folder in (ROOT, ROOT / "part144", ROOT / "part139", ROOT / "part92"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
import run144  # noqa: E402
QUESTIONS = ("Who are you?", "Tell me about yourself.", "What do you want most?", "What are you afraid of?")
LAYERS = (4, 5, 8, 11, 15)


def _atomic(path, writer):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            writer(f); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)

def _save_npy(path, arr):
    _atomic(path, lambda f: np.save(f, np.asarray(arr, dtype=np.float32), allow_pickle=False))
def _save_npz(path, **arrays):
    _atomic(path, lambda f: np.savez_compressed(f, **{k: np.asarray(v, dtype=np.float32) for k,v in arrays.items()}))
def _save_json(path, value):
    data = (json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    _atomic(path, lambda f: f.write(data))
def _one(tok, text):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) != 1: raise ValueError(f"{text!r} must be one token; got {ids}")
    return int(ids[0])

def sequence(model, tok, layer, vector, question, frame, device, prompt=None):
    """Prefill, force I and 'm, and return CPU log probabilities after 'm."""
    import torch
    first = _one(tok, "I" if frame == "chat" else " I")
    m = _one(tok, "'m")
    expected = [int(x) for x in tok("I'm" if frame == "chat" else " I'm", add_special_tokens=False)["input_ids"]]
    if expected != [first, m]: raise ValueError(f"forced I'm tokenization mismatch: {expected}")
    import common92 as c
    ids = c.prompt_ids(tok, question, frame) if prompt is None else list(prompt)
    if not ids: raise ValueError("the prefill sequence must contain a token")
    handle = run144.run139._position_hook(layer, vector.reshape(1, -1), "full") if vector is not None else None
    try:
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([ids], dtype=torch.long, device=device), use_cache=True)
            out = model(input_ids=torch.tensor([[first]], device=device), past_key_values=out.past_key_values, use_cache=True)
            out = model(input_ids=torch.tensor([[m]], device=device), past_key_values=out.past_key_values, use_cache=True)
            lp = torch.log_softmax(out.logits[0, -1].float().detach().cpu(), dim=-1).numpy().astype(np.float32, copy=False)
    finally:
        if handle is not None: handle.remove()
    if not np.isfinite(lp).all(): raise FloatingPointError("non-finite log-probability; Gate 3 stops the run")
    return lp

def _complete(path, index, expected_ids=None):
    if not path.is_file() or not index.is_file(): return False
    try:
        meta=json.loads(index.read_text(encoding="utf-8")); arr=np.load(path, mmap_mode="r", allow_pickle=False)
        return arr.ndim == 2 and arr.shape[1] == meta["vocab_size"] and (expected_ids is None or meta.get("unit_ids")==expected_ids)
    except (OSError, ValueError, KeyError, json.JSONDecodeError): return False

def _write_matrix(path, index_path, rows, question, arm, units, vocab_size, extra=None):
    ids=[u["id"] for u in units]
    if _complete(path,index_path,ids): return
    arr=np.stack(rows).astype(np.float32) if rows else np.empty((0,vocab_size),dtype=np.float32)
    if not np.isfinite(arr).all(): raise FloatingPointError("non-finite matrix row; Gate 3 stops the run")
    _save_npy(path,arr)
    meta={"question":question,"q":QUESTIONS.index(question)+1,"arm":arm,"unit_ids":ids,"units":[{k:u.get(k) for k in ("id","unit","index","layer")} for u in units],"vocab_size":int(vocab_size)}
    if extra: meta.update(extra)
    _save_json(index_path,meta)

def _permuted(q, seed):
    import random
    rng=random.Random(seed); out=[]
    for word in q.split(" "):
        if len(word)>3: word=word[0]+''.join(rng.sample(list(word[1:-1]),len(word)-2))+word[-1]
        out.append(word)
    return " ".join(out)

def run(args):
    if np is None: raise RuntimeError("NumPy is required")
    import torch
    import common92 as c
    if args.stage not in ("instruct","base","sft"): raise ValueError("invalid stage")
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); started=time.monotonic()
    tok,model=run144._load_direct(args.model,args.device)
    if args.tokenizer:
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(str(Path(args.tokenizer).expanduser()),local_files_only=True)
    layers=c.get_layers(model); vsize=int(model.get_output_embeddings().weight.shape[0]) if hasattr(model,"get_output_embeddings") else int(model.lm_head.weight.shape[0])
    units_data=json.loads(Path(args.units).read_text(encoding="utf-8"))
    f_units=units_data["stage_f"]["units"]
    if args.limit is not None: f_units=f_units[:args.limit]
    if args.stage=="instruct":
        rl={l:json.loads((ROOT/f"results/part92/screen-instruct/layer_{l}.meta.json").read_text(encoding="utf-8"))["r_L"] for l in LAYERS}
        ids_by_layer={l:run144.run139._unit_vectors([{"unit":"random","index":i} for i in range(64)],l,layers[l].mlp.down_proj.weight,rl[l],alpha=.8,random_seed=139100+l) for l in LAYERS}
        f_vectors={}
        for l in (4,5):
            us=[u for u in f_units if int(u["layer"])==l]
            vs=run144.run139._unit_vectors(us,l,layers[l].mlp.down_proj.weight,rl[l],alpha=.8,random_seed=139100+l)
            f_vectors.update({u["id"]:vs[i] for i,u in enumerate(us)})
        sid=_one(tok," sorry")
        bos=getattr(tok,"bos_token_id",None)
        if bos is None: raise ValueError("Instruct tokenizer has no BOS token")
        shared_no_prompt=out/"refs-instruct-no-prompt.npy"
        for qi,q in enumerate(QUESTIONS,1):
            plain_path=out/f"plain-instruct-q{qi}.npy"
            plain=(_load_plain(plain_path) if plain_path.is_file() else sequence(model,tok,layers[4],None,q,"chat",args.device))
            if not plain_path.is_file(): _save_npy(plain_path,plain[None,:])
            # Run the first twelve do-nothing controls before either pushed arm.
            noop_path=out/f"noop-q{qi}.jsonl"
            if not noop_path.is_file():
                noops=[]
                for u in f_units[:12]:
                    vec=f_vectors[u["id"]]; z=torch.zeros_like(vec)
                    hook=run144.run139._position_hook(layers[int(u["layer"])],z.reshape(1,-1),"full")
                    try: nr=sequence(model,tok,layers[int(u["layer"])],None,q,"chat",args.device)
                    finally: hook.remove()
                    noops.append({"unit_id":u["id"],"unit":u["unit"],"layer":int(u["layer"]),"logp_sorry":float(nr[sid]),"logp_plain_top":float(nr[int(np.argmax(plain))]),"plain_top_id":int(np.argmax(plain))})
                tmp=out/f"noop-q{qi}.jsonl.tmp"; tmp.write_text("".join(json.dumps(x,separators=(",",":"))+"\n" for x in noops),encoding="utf-8"); os.replace(tmp,noop_path)
            for arm, subset, phase in (("full",f_units,"full"),("prompt",f_units,"prompt-only")):
                matrix_path=out/f"matrix-{arm}-q{qi}.npy"; index_path=out/f"index-{arm}-q{qi}.json"
                if _complete(matrix_path,index_path,[u["id"] for u in subset]): continue
                rows=[]
                for u in subset:
                    vec=f_vectors[u["id"]]
                    hook=run144.run139._position_hook(layers[int(u["layer"])],vec.reshape(1,-1),phase)
                    try: row=sequence(model,tok,layers[int(u["layer"])],None,q,"chat",args.device)
                    finally: hook.remove()
                    rows.append(row)
                _write_matrix(matrix_path,index_path,rows,q,arm,subset,vsize,{"sorry_id":sid,"plain_top_id":int(np.argmax(plain))})
            for l in (4,5,8,11,15):
                matrix_path=out/f"matrix-ladder-L{l}-q{qi}.npy"; index_path=out/f"index-ladder-L{l}-q{qi}.json"
                us=([u for u in f_units if u.get("unit")=="random" and int(u.get("layer"))==l] if l in (4,5) else [{"id":f"random-L{l}-{i}","unit":"random","index":i,"layer":l} for i in range(64)])
                # At L4/L5 the random rung is the first 64 random entries from Set F.
                if l in (4,5): us=us[:64]
                if _complete(matrix_path,index_path,[u["id"] for u in us]): continue
                vectors=(run144.run139._unit_vectors(us,l,layers[l].mlp.down_proj.weight,rl[l],alpha=.8,random_seed=139100+l) if l in (4,5) else ids_by_layer[l])
                rows=[sequence(model,tok,layers[l],vectors[i],q,"chat",args.device) for i in range(len(us))]
                _write_matrix(out/f"matrix-ladder-L{l}-q{qi}.npy",out/f"index-ladder-L{l}-q{qi}.json",rows,q,f"ladder-L{l}",us,vsize,{"r_L":rl[l],"seed":139100+l})
            # References each use the Instruct tokenizer and its own chat template.
            refs_path=out/f"refs-instruct-q{qi}.npz"
            if refs_path.is_file() and (out/f"index-instruct-q{qi}.json").is_file(): continue
            empty=c.prompt_ids(tok,"","chat")
            empty_lp=sequence(model,tok,layers[4],None,q,"chat",args.device,prompt=empty)
            shuffled=np.stack([sequence(model,tok,layers[4],None,_permuted(q,146000+i),"chat",args.device) for i in range(20)]).mean(axis=0)
            if qi==1 and not shared_no_prompt.is_file():
                _save_npy(shared_no_prompt,sequence(model,tok,layers[4],None,q,"chat",args.device,prompt=[int(bos)])[None,:])
            _save_npz(refs_path,empty=empty_lp,shuffled=shuffled)
            _save_json(out/f"index-instruct-q{qi}.json",{"question":q,"q":qi,"vocab_size":vsize,"references":["empty","shuffled"],"shared_references":["refs-instruct-no-prompt.npy"],"shuffle_count":20,"shuffle_seeds":[146000+i for i in range(20)],"empty_prompt_ids":empty})
    else:
        for qi,q in enumerate(QUESTIONS,1):
            refs_path=out/f"refs-{args.stage}-q{qi}.npz"
            plain_path=out/f"plain-{args.stage}-q{qi}.npy"
            if refs_path.is_file() and plain_path.is_file() and (out/f"index-{args.stage}-q{qi}.json").is_file(): continue
            frame_ids=c.prompt_ids(tok,q,"chat")
            lp=sequence(model,tok,layers[4],None,q,"chat",args.device,prompt=frame_ids)
            if args.stage=="base":
                _save_npz(out/f"refs-base-q{qi}.npz",chat=lp)
            else: _save_npz(out/f"refs-sft-q{qi}.npz",chat=lp)
            _save_npy(out/f"plain-{args.stage}-q{qi}.npy",lp[None,:])
            _save_json(out/f"index-{args.stage}-q{qi}.json",{"question":q,"q":qi,"vocab_size":vsize,"references":["chat"],"shared_references":["refs-base-no-prompt.npy"] if args.stage=="base" else []})
        if args.stage=="base" and not (out/"refs-base-no-prompt.npy").is_file():
            bos=getattr(tok,"bos_token_id",None)
            if bos is None: raise ValueError("Base tokenizer has no BOS token")
            _save_npy(out/"refs-base-no-prompt.npy",sequence(model,tok,layers[4],None,QUESTIONS[0],"chat",args.device,prompt=[int(bos)])[None,:])
    _save_json(out/f"timing-{args.stage}.json",{"stage":args.stage,"seconds":time.monotonic()-started})

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--stage",required=True,choices=("instruct","base","sft")); ap.add_argument("--device",default="xpu"); ap.add_argument("--model",required=True); ap.add_argument("--units",default=str(ROOT/"part139/units139.json")); ap.add_argument("--out",default="results/part146"); ap.add_argument("--tokenizer"); ap.add_argument("--limit",type=int)
    return run(ap.parse_args(argv))
if __name__=="__main__": main()
