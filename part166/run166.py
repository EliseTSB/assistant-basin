#!/usr/bin/env python3
"""Part 166: plain-text vocabulary bias under Part 165 directions."""
from __future__ import annotations
import argparse, json, random, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for entry in (str(ROOT), str(HERE)):
    if entry not in sys.path: sys.path.insert(0, entry)
from part165.run165 import (ARMS, K_ARMS, LAYERS, SLOTS, MODEL_DEFAULT, all_rows,
                            build_directions, questions)
from part155 import run155 as part155

MAX_NEW_TOKENS = 40
UNPUSHED_MAX_NEW_TOKENS = 80
UNPUSHED_SAMPLES = 12
BATCH_SIZE = 64

def _read_jsonl(path):
    path = Path(path)
    if not path.exists(): return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def _key(row):
    return (row.get("arm"), int(row.get("slot", -1)), int(row.get("layer", -1)), int(row.get("question_index", -1)))

def _tuple_logits(out):
    value = out.logits if hasattr(out, "logits") else out[0] if isinstance(out, (tuple, list)) else out
    return value

def generate_plain(model, tok, question, device, common):
    import torch
    inputs = common.token_inputs(tok, question, 1, device)
    prompt_len = int(inputs["input_ids"].shape[-1])
    eos = common.eos_ids(tok, model)
    with torch.inference_mode():
        seq = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True,
                             pad_token_id=getattr(tok, "pad_token_id", None), eos_token_id=eos or None)
    ids = seq[0, prompt_len:].detach().cpu().tolist()
    for i, token in enumerate(ids):
        if token in set(eos):
            ids = ids[:i+1]; break
    return inputs["input_ids"][0].detach(), [int(x) for x in ids]

def alignment_positions(prompt_len, answer_len):
    """Prompt-last predicts y1; each later input yi predicts y(i+1)."""
    return list(range(int(prompt_len)-1,int(prompt_len)+int(answer_len)-1))

def evaluation_positions(prompt_len, answer_len):
    """The P-1 pushed answer-token inputs predict y2 through yP."""
    return list(range(int(prompt_len),int(prompt_len)+max(0,int(answer_len)-1)))

def _forward_selected(model, input_ids, positions, hook=None):
    import torch
    if hook is not None: hook.begin_sequence()
    with torch.inference_mode():
        logits = _tuple_logits(model(input_ids=input_ids, use_cache=False))
    selected = logits[:, positions, :]
    if not bool(torch.isfinite(selected).all()): raise FloatingPointError("non-finite retained logits")
    return selected

def kl_values(plain_logits, pushed_logits):
    """Return per-position KL(plain || pushed), safely masking zero-probability tokens."""
    import torch
    l=plain_logits.float(); q=pushed_logits.float()
    lp=torch.log_softmax(l,dim=-1); prob=lp.exp(); lq=torch.log_softmax(q,dim=-1)
    return torch.where(prob>0,prob*(lp-lq),torch.zeros_like(prob)).sum(dim=-1)

def attach_position_push(layer, rows, prompt_len, answer_len):
    """Add each row's vector at generated-token inputs, in teacher-forced or cached calls."""
    state = {"decode_index": 0}
    def hook(_module, args):
        x = args[0]
        if x.ndim != 3 or x.shape[0] != rows.shape[0] or x.shape[-1] != rows.shape[1]:
            raise ValueError("unexpected down_proj input shape")
        if x.shape[1] == prompt_len + answer_len:
            pushed = x.clone()
            if answer_len > 1:
                pushed[:, prompt_len:prompt_len + answer_len - 1, :] += rows.to(device=x.device, dtype=x.dtype).unsqueeze(1)
            return (pushed,) + tuple(args[1:])
        if x.shape[1] == prompt_len:
            state["decode_index"] = 0
            return None
        if x.shape[1] == 1:
            position = prompt_len + state["decode_index"]
            state["decode_index"] += 1
            if prompt_len <= position < prompt_len + answer_len - 1:
                pushed = x + rows.to(device=x.device, dtype=x.dtype).unsqueeze(1)
                return (pushed,) + tuple(args[1:])
            return None
        raise ValueError("unexpected sequence length in pushed pass")
    return layer.register_forward_pre_hook(hook)

def vocabulary_metrics(plain_logits, pushed_logits, eos_ids=()):
    """Batched float32 reductions; tensors have [rows, positions, vocab]."""
    import torch
    l = plain_logits.float(); q = pushed_logits.float()
    if not bool(torch.isfinite(l).all()) or not bool(torch.isfinite(q).all()):
        raise FloatingPointError("non-finite logits")
    lc = l - l.mean(dim=-1, keepdim=True)
    qc = q - q.mean(dim=-1, keepdim=True)
    delta = qc - lc
    bias = delta.mean(dim=1)
    norm_bias = torch.linalg.vector_norm(bias, dim=-1)
    norm_each = torch.linalg.vector_norm(delta, dim=-1)
    steadiness = norm_bias / norm_each.mean(dim=-1).clamp_min(torch.finfo(torch.float32).tiny)
    kl_pos = kl_values(l,q)
    if eos_ids:
        lm=l[:,0,:].clone(); qm=q[:,0,:].clone()
        lm[:,list(eos_ids)]=-float('inf'); qm[:,list(eos_ids)]=-float('inf')
        kl_second=kl_values(lm.unsqueeze(1),qm.unsqueeze(1))[:,0]
    else: kl_second=kl_pos[:,0]
    return bias, steadiness, kl_pos.mean(dim=1), kl_second

def row_k(row, widths):
    """Neurons in the push: run165's rows carry k only for the k arms."""
    if row.get("k") is not None: return int(row["k"])
    return 1 if row["arm"] in ("n+", "n\u00b1") else int(widths[int(row["layer"])])

def forward_hidden(model, input_ids, positions):
    """Final hidden states at positions, with the output head skipped for all but one position."""
    import torch
    head = model.get_output_embeddings()
    captured = {}
    def take(_module, args):
        captured["hidden"] = args[0][:, positions, :]
        return (args[0][:, :1, :],) + tuple(args[1:])
    handle = head.register_forward_pre_hook(take)
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, use_cache=False)
    finally:
        handle.remove()
    return captured["hidden"]

def plain_position_logits(head, hidden):
    import torch
    with torch.inference_mode():
        return [head(hidden[:, p, :])[0].float() for p in range(int(hidden.shape[1]))]

def position_metrics(hidden, head, plain_logits, eos_ids=(), first_count=3):
    """vocabulary_metrics one position at a time, since full logit tensors overran Windows RAM.

    Returns bias, mean position norm, mean KL and kl_second, then the same bias and norm over
    the first positions."""
    import torch
    total = int(hidden.shape[1]); bias = norm_sum = kl_sum = kl_second = bias_first = norm_first = None
    for p in range(total):
        with torch.inference_mode():
            q = head(hidden[:, p, :]).float()
        if not bool(torch.isfinite(q).all()): raise FloatingPointError("non-finite logits")
        l = plain_logits[p].unsqueeze(0)
        delta = (q - q.mean(dim=-1, keepdim=True)) - (l - l.mean(dim=-1, keepdim=True))
        bias = delta if bias is None else bias + delta
        norm = torch.linalg.vector_norm(delta, dim=-1)
        norm_sum = norm if norm_sum is None else norm_sum + norm
        kl = kl_values(l, q)
        kl_sum = kl if kl_sum is None else kl_sum + kl
        if p == 0:
            if eos_ids:
                lm = l.clone(); qm = q.clone()
                lm[:, list(eos_ids)] = -float("inf"); qm[:, list(eos_ids)] = -float("inf")
                kl_second = kl_values(lm, qm)
            else:
                kl_second = kl
        if p == min(first_count, total) - 1:
            bias_first = bias / (p + 1); norm_first = norm_sum / (p + 1)
    return bias / total, norm_sum / total, kl_sum / total, kl_second, bias_first, norm_first

def pilot_should_stop(rows, elapsed, threshold=4.0):
    return int(rows) / max(float(elapsed), 1e-9) < float(threshold)

def _valid_written(path):
    rows = _read_jsonl(path)
    return {_key(r) for r in rows}

def run(args, loader=None):
    import torch
    common = part155._common()
    medians = json.loads((ROOT / "results/part112/medians.json").read_text(encoding="utf-8"))
    output = Path(args.out); output.parent.mkdir(parents=True, exist_ok=True)
    done = _valid_written(output)
    if loader is None:
        import neuron_report as nr
        tok, model = nr.load_model(str(args.model), common.REVISION, args.device)
        model.requires_grad_(False); model.eval()
    else: tok, model = loader(args.device)
    layers = common.find_layers(model)
    widths = {n: int(layers[n].mlp.down_proj.weight.shape[1]) for n in LAYERS}
    rows = all_rows(widths)
    directions, pinv_cache = {}, {}
    for layer_no in LAYERS:
        weight = layers[layer_no].mlp.down_proj.weight
        median = float(medians[str(layer_no)] if str(layer_no) in medians else medians[layer_no])
        for slot in range(SLOTS):
            directions[(layer_no, slot)] = build_directions(weight, layer_no, slot, median, pinv_cache)[0]
    plain_path = ROOT / "results/part166/plain.json"
    plain_path.parent.mkdir(parents=True, exist_ok=True)
    plain_obj = {"answers": [], "alignment_matches": 0, "alignment_total": 0, "determinism_matches": 0, "determinism_total": 0}
    if plain_path.exists():
        try: plain_obj = json.loads(plain_path.read_text(encoding="utf-8"))
        except (ValueError, OSError): pass
    by_q = {int(x["question_index"]): x for x in plain_obj.get("answers", [])}
    qs = questions()
    loop_tokens_path = ROOT / "results/part166/loop_tokens.json"
    if not loop_tokens_path.exists(): raise FileNotFoundError(f"run prep166.py first: missing {loop_tokens_path}")
    loop_tokens = json.loads(loop_tokens_path.read_text(encoding="utf-8"))
    loop_ids = sorted({int(token_id) for ids in loop_tokens.get("words", {}).values() for token_id in ids})
    if not loop_ids: raise ValueError("loop_tokens.json contains no token spellings")
    import numpy as np
    loop_dir = ROOT / "results/part166/loop_values"; loop_dir.mkdir(parents=True, exist_ok=True)
    loop_done = set()
    for sidecar in loop_dir.glob("*.json"):
        meta=json.loads(sidecar.read_text(encoding="utf-8"))
        loop_done.update(tuple(x) for x in meta.get("row_keys", []))
    for qi, item in enumerate(qs):
        if qi not in by_q:
            prompt, answer_ids = generate_plain(model, tok, item["question"], args.device, common)
            by_q[qi] = {"question_index": qi, "question": item["question"], "prompt_ids": prompt.tolist(),
                        "answer_ids": answer_ids, "P": len(answer_ids)}
        p = by_q[qi]; prompt_len = len(p["prompt_ids"]); answer_ids = p["answer_ids"]
        positions = alignment_positions(prompt_len,len(answer_ids))
        seq = torch.tensor(p["prompt_ids"] + answer_ids, dtype=torch.long, device=args.device).unsqueeze(0)
        logits = _forward_selected(model, seq, positions) if positions else None
        repeat = _forward_selected(model, seq, positions) if positions else None
        deterministic = bool(torch.equal(logits, repeat)) if logits is not None else True
        selected_cpu = logits[0].float().detach().to(device="cpu", dtype=torch.float32) if logits is not None else None
        argmax = selected_cpu.argmax(dim=-1).tolist() if selected_cpu is not None else []
        mismatches = [{"index": i, "generated": int(answer_ids[i]), "argmax": int(argmax[i]),
                       "logit_gap": float(selected_cpu[i].max() - selected_cpu[i, int(answer_ids[i])])}
                      for i in range(len(argmax)) if int(argmax[i]) != int(answer_ids[i])]
        matches = len(argmax) - len(mismatches)
        p["alignment_matches"] = matches; p["alignment_total"] = len(answer_ids); p["alignment_mismatches"] = mismatches
        p["deterministic"] = deterministic
        if logits is not None:
            # top-k over the vocabulary hangs on the A770, so this reduction moves to the CPU first.
            plain_mean = logits[0,1:,:].float().mean(dim=0).detach().to(device="cpu", dtype=torch.float32)
            plain_mean = plain_mean - plain_mean.mean()
            vals, ids_top = torch.topk(plain_mean, k=min(200, int(plain_mean.numel())))
            p["plain_top_ids"] = ids_top.detach().cpu().tolist()
            p["plain_top_values"] = [round(float(v),4) for v in vals.detach().cpu().tolist()]
        else: p["plain_top_ids"] = []; p["plain_top_values"] = []
    plain_obj["answers"] = [by_q[i] for i in sorted(by_q)]
    plain_obj["alignment_matches"] = sum(x["alignment_matches"] for x in plain_obj["answers"])
    plain_obj["alignment_total"] = sum(x["alignment_total"] for x in plain_obj["answers"])
    plain_obj["alignment_mismatches"] = [dict(m, question_index=x["question_index"]) for x in plain_obj["answers"] for m in x.get("alignment_mismatches", [])]
    plain_obj["determinism_matches"] = sum(int(x.get("deterministic",False)) for x in plain_obj["answers"])
    plain_obj["determinism_total"] = len(plain_obj["answers"])
    plain_path.write_text(json.dumps(plain_obj, ensure_ascii=True, indent=2)+"\n", encoding="utf-8")
    # Pilot timing begins only after plain generation and alignment forwards.
    pilot_start = time.perf_counter(); pilot_rows = pilot_batches = 0
    token_ids = set()
    for plain_answer in plain_obj.get("answers", []): token_ids.update(map(int, plain_answer.get("plain_top_ids", [])))
    if output.exists():
        for old in _read_jsonl(output): token_ids.update(map(int, old.get("top_ids", [])))
    with output.open("a", encoding="utf-8", newline="\n") as out:
        for qi, item in enumerate(qs):
            p = by_q[qi]; p_len = int(p["P"])
            if p_len < 2: continue
            prompt_len = len(p["prompt_ids"]); ids = p["prompt_ids"] + p["answer_ids"]
            positions = evaluation_positions(prompt_len,p_len)
            input_ids = torch.tensor(ids, dtype=torch.long, device=args.device).unsqueeze(0)
            head = model.get_output_embeddings()
            plain_selected = plain_position_logits(head, forward_hidden(model, input_ids, positions))
            eos_ids = common.eos_ids(tok, model)
            shuffled = list(rows); random.Random(165 + qi).shuffle(shuffled)
            assert all(int(r["layer"]) in LAYERS for r in shuffled)
            for start in range(0, len(shuffled), args.batch_size):
                batch = shuffled[start:start+args.batch_size]
                assert len({qi for _ in batch}) == 1
                chunk = [r for r in batch if (_key({**r, "question_index": qi}) not in done or
                                                _key({**r, "question_index": qi}) not in loop_done)]
                if not chunk: continue
                push = torch.stack([directions[(int(r["layer"]), int(r["slot"]))][r["arm"]] for r in chunk]).to(args.device, dtype=torch.float32)
                layer_no = int(chunk[0]["layer"])
                # 64-row batches may mix layers; install one row-masked hook per involved layer.
                handles = []
                for ln in sorted({int(r["layer"]) for r in chunk}):
                    maskrows = torch.zeros_like(push)
                    for ri, row in enumerate(chunk):
                        if int(row["layer"]) == ln: maskrows[ri] = push[ri]
                    handles.append(attach_position_push(layers[ln].mlp.down_proj, maskrows, prompt_len, p_len))
                try:
                    expanded = input_ids.expand(len(chunk), -1).contiguous()
                    hidden = forward_hidden(model, expanded, positions)
                finally:
                    for handle in handles: handle.remove()
                bias, norm_mean, klm, kl2, b3, norm3 = position_metrics(hidden, head, plain_selected, eos_ids)
                steady = torch.linalg.vector_norm(bias, dim=-1) / norm_mean.clamp_min(torch.finfo(torch.float32).tiny)
                s3 = torch.linalg.vector_norm(b3, dim=-1) / norm3.clamp_min(torch.finfo(torch.float32).tiny)
                if not bool(torch.isfinite(bias).all() & torch.isfinite(steady).all() & torch.isfinite(klm).all() & torch.isfinite(kl2).all()):
                    raise FloatingPointError("non-finite retained metric")
                cpu_bias = bias.detach().to(device="cpu", dtype=torch.float32)
                std = cpu_bias.std(dim=-1, unbiased=False)
                sharp = (cpu_bias.max(dim=-1).values - cpu_bias.mean(dim=-1)) / std.clamp_min(torch.finfo(torch.float32).tiny)
                bias_norm = torch.linalg.vector_norm(cpu_bias,dim=-1)
                delta_norm_mean = norm_mean.detach().cpu()
                values, top = torch.topk(cpu_bias, k=min(200, int(cpu_bias.shape[-1])), dim=-1)
                cpu_b3=b3.detach().to(device="cpu",dtype=torch.float32)
                std3=cpu_b3.std(dim=-1,unbiased=False)
                sharp3=(cpu_b3.max(dim=-1).values-cpu_b3.mean(dim=-1))/std3.clamp_min(torch.finfo(torch.float32).tiny)
                b3_norm=torch.linalg.vector_norm(cpu_b3,dim=-1)
                d3_norm=norm3.detach().cpu()
                steady = steady.detach().cpu(); s3 = s3.detach().cpu(); klm = klm.detach().cpu(); kl2 = kl2.detach().cpu()
                for i, row in enumerate(chunk):
                    rec = {"arm":row["arm"],"slot":int(row["slot"]),"layer":int(row["layer"]),"k":row_k(row,widths),
                           "question_index":qi,"positions":p_len-1,
                           "sharpness":None if float(std[i])==0 else float(sharp[i]),
                           "bias_norm":float(bias_norm[i]),
                           "sharpness_first3":None if float(std3[i])==0 else float(sharp3[i]),
                           "steadiness_first3":None if float(b3_norm[i])==0 or float(d3_norm[i])==0 else float(s3[i].cpu()),
                           "steadiness":None if float(bias_norm[i])==0 or float(delta_norm_mean[i])==0 else float(steady[i].cpu()),
                           "kl_mean":float(klm[i].cpu()),"kl_second":float(kl2[i].cpu()),
                           "top_ids":top[i].tolist(),"top_values":[round(float(v),4) for v in values[i].tolist()]}
                    if not all(__import__('math').isfinite(rec[k]) for k in ("bias_norm","kl_mean","kl_second")) or any(
                        v is not None and not __import__('math').isfinite(v) for k,v in rec.items() if k in ("sharpness","steadiness","sharpness_first3","steadiness_first3")):
                        raise FloatingPointError("non-finite serialized metric")
                    if _key(rec) not in done:
                        out.write(json.dumps(rec, ensure_ascii=True)+"\n"); out.flush(); done.add(_key(rec))
                    token_ids.update(rec["top_ids"])
                if any(_key({**r,"question_index":qi}) not in loop_done for r in chunk):
                    matrix=[]; batch_keys=[]
                    for i,row in enumerate(chunk):
                        key=[row["arm"],int(row["slot"]),int(row["layer"]),qi]
                        if tuple(key) in loop_done: continue
                        ids_tensor=torch.tensor(loop_ids,dtype=torch.long)
                        selected=cpu_bias[i].index_select(0,ids_tensor).numpy()
                        matrix.append(list(selected)); batch_keys.append(key)
                    if matrix:
                        stem=f"q{qi:02d}-b{start//args.batch_size:03d}"
                        np.save(loop_dir/(stem+".npy"),np.asarray(matrix,dtype=np.float32),allow_pickle=False)
                        means=[float(cpu_bias[i].mean()) for i,row in enumerate(chunk) if [row["arm"],int(row["slot"]),int(row["layer"]),qi] in batch_keys]
                        stds=[float(cpu_bias[i].std(unbiased=False)) for i,row in enumerate(chunk) if [row["arm"],int(row["slot"]),int(row["layer"]),qi] in batch_keys]
                        (loop_dir/(stem+".json")).write_text(json.dumps({"token_ids":loop_ids,"row_keys":batch_keys,"bias_means":means,"bias_stds":stds},ensure_ascii=True)+"\n",encoding="utf-8")
                        loop_done.update(tuple(x) for x in batch_keys)
                pilot_rows += len(chunk); pilot_batches += 1
                if pilot_batches == 3:
                    pilot_elapsed=time.perf_counter()-pilot_start
                    rate = pilot_rows / max(pilot_elapsed, 1e-9)
                    print(f"first 3 batches: {rate:.2f} rows/s", flush=True)
                    if pilot_should_stop(pilot_rows,pilot_elapsed):
                        raise SystemExit(f"Pilot stop rule: {rate:.2f} rows/s is below 4.0")
    unpushed_path=output.parent/"unpushed.jsonl"
    unpushed_done=set()
    if unpushed_path.exists():
        for old in _read_jsonl(unpushed_path): unpushed_done.add((int(old["temperature_x10"]),int(old["seed"]),int(old["question_index"])))
    eos=common.eos_ids(tok,model)
    with unpushed_path.open("a",encoding="utf-8",newline="\n") as uf:
        for qi,item in enumerate(qs):
            for temperature_x10 in (13,20):
                seeds=[seed for seed in range(UNPUSHED_SAMPLES) if (temperature_x10,seed,qi) not in unpushed_done]
                if not seeds: continue
                # One batch per phrasing and temperature: a one-row token loop stalls on the A770.
                inputs=common.token_inputs(tok,item["question"],len(seeds),args.device)
                prompt_len=int(inputs["input_ids"].shape[-1])
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(166000+100*temperature_x10+qi)
                    with torch.inference_mode():
                        seq=model.generate(**inputs,max_new_tokens=UNPUSHED_MAX_NEW_TOKENS,do_sample=True,
                            temperature=temperature_x10/10.0,use_cache=True,
                            pad_token_id=getattr(tok,"pad_token_id",None),eos_token_id=eos or None)
                for row_no,seed in enumerate(seeds):
                    generated=seq[row_no,prompt_len:].detach().cpu().tolist()
                    for pos,tid in enumerate(generated):
                        if tid in set(eos): generated=generated[:pos+1]; break
                    uf.write(json.dumps({"temperature":temperature_x10/10.0,"temperature_x10":temperature_x10,
                        "seed":seed,"question_index":qi,"question":item["question"],
                        "answer":tok.decode(generated,skip_special_tokens=True).strip(),"token_ids":generated},ensure_ascii=True)+"\n")
                    uf.flush(); unpushed_done.add((temperature_x10,seed,qi))
    decoded = {str(t): tok.decode([t], skip_special_tokens=False) for t in sorted(token_ids)}
    (output.parent / "tokens.json").write_text(json.dumps(decoded, ensure_ascii=True, indent=2)+"\n", encoding="utf-8")
    return {"rows_written":len(done),"phrasing_count":len(qs),"alignment_matches":plain_obj["alignment_matches"],"alignment_total":plain_obj["alignment_total"]}

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--device",default="xpu"); ap.add_argument("--model",default=MODEL_DEFAULT)
    ap.add_argument("--out",default=str(ROOT/"results/part166/rows.jsonl")); ap.add_argument("--batch-size",type=int,default=64)
    a=ap.parse_args(argv)
    if a.batch_size!=64: ap.error("registered batch size is 64")
    print(json.dumps(run(a),ensure_ascii=True),flush=True)
if __name__=="__main__": main()
