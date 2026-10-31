#!/usr/bin/env python3
"""Collect Part 129 activation sums on two GPUs.

Chat masks use the actual Qwen3 <|im_start|> and assistant token IDs. Each message's role
is the token immediately after its start marker; all valid tokens through the next start
marker belong to that role.
"""
from __future__ import annotations
import argparse, hashlib, json, os, queue, threading, time
from pathlib import Path
import runpy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
_config = runpy.run_path(str(Path(__file__).with_name("config129.py")))
CORPUS_DOCS, CORPUS_ORDER = _config["CORPUS_DOCS"], _config["CORPUS_ORDER"]
DOC_CHARS, MIN_LEN = _config["DOC_CHARS"], _config["MIN_LEN"]
PILOT_DOCS_PER_SLICE, SEQ_LEN = _config["PILOT_DOCS_PER_SLICE"], _config["SEQ_LEN"]

LAYERS = 36
GROUPS = ("web", "zh", "code", "chat-user", "chat-assistant")
CHUNK_SAVE = 500
CORPUS_BASENAMES = {"dclm": "act-dclm.jsonl", "zh": "act-zh.jsonl",
                    "code": "act-code.jsonl", "chat": "act-chat-qwen3-8b.jsonl"}

def validate_corpus_paths(paths):
    if len(paths) != len(CORPUS_ORDER):
        raise ValueError(f"expected {len(CORPUS_ORDER)} corpus files")
    resolved = [Path(path).resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("corpus files must be distinct paths")
    parents = {path.parent for path in resolved}
    if len(parents) != 1 or next(iter(parents)).name != "corpus" or next(iter(parents)).parent.name != "data" or next(iter(parents)).parent.parent.name != "neuron-lens":
        raise ValueError("corpus files must come from one neuron-lens/data/corpus directory")
    for name, path in zip(CORPUS_ORDER, paths):
        if Path(path).name != CORPUS_BASENAMES[name]:
            raise ValueError(f"{name} corpus must be {CORPUS_BASENAMES[name]}, got {Path(path).name}")

def fixed_derangement(width, generator):
    order=torch.randperm(width,generator=generator)
    permutation=torch.empty_like(order)
    permutation[order]=order.roll(1)
    return permutation

def save_checkpoint(path, meta, sums, alignment_values, next_doc):
    payload={"meta":meta,"state":{**sums.state(),"alignment_values":alignment_values},"next_doc":next_doc}
    tmp=Path(path).with_suffix(".tmp")
    torch.save(payload,tmp)
    os.replace(tmp,path)

def resume_metadata_matches(saved, current):
    """Batch size changes layout only; all content provenance must still match."""
    return ({key: value for key, value in saved.items() if key != "batch"}
            == {key: value for key, value in current.items() if key != "batch"})

class LayerActivationExchange:
    """Bounded per-layer handoff with an explicit index check on every read."""
    def __init__(self, layers):
        self.queues = [queue.Queue(maxsize=1) for _ in range(layers)]
    def put(self, layer, activation, timeout=600):
        self.queues[layer].put((layer, activation), timeout=timeout)
    def get(self, layer, timeout=600):
        queued_layer, activation = self.queues[layer].get(timeout=timeout)
        if queued_layer != layer:
            raise RuntimeError(f"activation layer mismatch: expected {layer}, got {queued_layer}")
        return activation

def group_masks(ids, slice_name, tok, attention_mask):
    valid = attention_mask.bool()
    if slice_name != "chat":
        target = {"dclm":"web", "zh":"zh", "code":"code"}.get(slice_name)
        return {g: valid if g == target else torch.zeros_like(valid) for g in GROUPS}
    start = tok.convert_tokens_to_ids("<|im_start|>")
    assistant = tok.convert_tokens_to_ids("assistant")
    if start is None or assistant is None or min(start, assistant) < 0:
        raise ValueError("Qwen3 chat marker token IDs are unavailable")
    asst = torch.zeros_like(valid)
    for b in range(ids.shape[0]):
        starts = torch.where(valid[b] & ids[b].eq(start))[0].tolist()
        ends = starts[1:] + [int(torch.where(valid[b])[0][-1]) + 1] if valid[b].any() else []
        for begin, end in zip(starts, ends):
            if begin + 1 < end and int(ids[b, begin + 1]) == assistant:
                asst[b, begin:end] = True
    asst &= valid
    return {g: (asst if g == "chat-assistant" else valid & ~asst) if g.startswith("chat-") else torch.zeros_like(valid)
            for g in GROUPS}

class Sums:
    """Float64 GPU sufficient statistics; first axis is layer, then group and neuron."""
    def __init__(self, layers, width, device):
        self.width, self.device = width, device
        # Ordinary x/y stats and rectified stats for both signed poles.
        self.data = torch.zeros((len(GROUPS), layers, 18, width), dtype=torch.float64, device=device)
        self.count = torch.zeros((len(GROUPS), layers), dtype=torch.int64, device=device)
        self.above = None
    def add(self, layer, x, y, masks, permutation, thresholds):
        # Keep token chunks small so float64 work buffers fit beside both models.
        xf, yf = x[:, 1:].reshape(-1, self.width), y[:, 1:].reshape(-1, self.width)
        for gi, name in enumerate(GROUPS):
            m = masks[name][:, 1:].reshape(-1)
            n = int(m.sum())
            if n == 0:
                continue
            self.count[gi, layer] += n
            for start in range(0, len(xf), 1024):
                end = min(len(xf), start + 1024)
                keep = m[start:end]
                if not bool(keep.any()):
                    continue
                a, b = xf[start:end][keep].to(torch.float64), yf[start:end][keep].to(torch.float64)
                for pi, sign in enumerate((1.0, -1.0)):
                    self.above[gi, layer, pi] += ((sign * a) > thresholds[layer, pi] / 4).sum(0)
                self.data[gi, layer, 0] += a.sum(0)
                self.data[gi, layer, 1] += b.sum(0)
                self.data[gi, layer, 2] += a.square().sum(0)
                self.data[gi, layer, 3] += b.square().sum(0)
                self.data[gi, layer, 4] += (a * b).sum(0)
                # The derangement is fixed per layer (seed 129), with no fixed points.
                ap = a[:, permutation]
                self.data[gi, layer, 5] += ap.sum(0)
                self.data[gi, layer, 6] += ap.square().sum(0)
                self.data[gi, layer, 7] += (ap * b).sum(0)
                for pi, sign in enumerate((1.0, -1.0)):
                    ax, by = (sign * a).clamp_min(0), (sign * b).clamp_min(0)
                    base = 8 + pi * 5
                    self.data[gi, layer, base] += ax.sum(0)
                    self.data[gi, layer, base + 1] += by.sum(0)
                    self.data[gi, layer, base + 2] += ax.square().sum(0)
                    self.data[gi, layer, base + 3] += by.square().sum(0)
                    self.data[gi, layer, base + 4] += (ax * by).sum(0)
    def state(self):
        return {"groups": GROUPS, "data": self.data.cpu(), "count": self.count.cpu(), "above": self.above.cpu()}

def weight_cosines(chat, base):
    result = []
    for l, (a, b) in enumerate(zip(chat.model.layers, base.model.layers)):
        ca, cb = a.mlp.down_proj.weight.float().cpu(), b.mlp.down_proj.weight.float().cpu()
        ga, gb = a.mlp.gate_proj.weight.float().cpu(), b.mlp.gate_proj.weight.float().cpu()
        ua, ub = a.mlp.up_proj.weight.float().cpu(), b.mlp.up_proj.weight.float().cpu()
        cos = lambda x, y, dim: torch.nn.functional.cosine_similarity(x, y, dim=dim)
        result.append({"down": cos(ca, cb, 0), "gate": cos(ga, gb, 1), "up": cos(ua, ub, 1)})
    return result

def load_docs(path, tok, slice_name, limit=None):
    docs = []
    for line_no, line in enumerate(open(path, encoding="utf-8")):
        if limit is not None and line_no >= limit:
            break
        row = json.loads(line)
        ids = tok(row["text"][:DOC_CHARS])["input_ids"][:SEQ_LEN]
        if len(ids) >= MIN_LEN:
            docs.append((ids, row, slice_name, line_no))
    return docs

def run(args):
    device_a, device_b = torch.device("cuda:0"), torch.device("cuda:1")
    tok = AutoTokenizer.from_pretrained(args.chat_model, local_files_only=True)
    chat = AutoModelForCausalLM.from_pretrained(args.chat_model, torch_dtype=torch.bfloat16, device_map={"":"cuda:0"}, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, device_map={"":"cuda:1"}, local_files_only=True)
    chat.eval(); base.eval()
    width, layers = chat.config.intermediate_size, chat.config.num_hidden_layers
    if base.config.intermediate_size != width or base.config.num_hidden_layers != layers:
        raise ValueError("checkpoint MLP layouts differ")
    col = Sums(layers, width, device_a)
    prepared = torch.load(args.prepared,map_location="cpu",weights_only=False)
    thresholds = prepared["thresholds"].to(device_a)
    col.above = torch.zeros((len(GROUPS), layers, 2, width), dtype=torch.int64, device=device_a)
    rng = torch.Generator().manual_seed(129)
    perms = [fixed_derangement(width,rng).to(device_a) for _ in range(layers)]
    hooks_a, hooks_b = [], []
    # Separate bounded rendezvous buffers prevent either model's layer L from
    # consuming a tensor produced at a different layer, even if forwards drift.
    activations = LayerActivationExchange(layers)
    worker_errors = []
    def install(model, hooks, is_chat):
        for li, layer in enumerate(model.model.layers):
            def hook(_module, inputs, layer_i=li):
                if is_chat:
                    x = inputs[0].detach()
                    deadline = time.monotonic() + 600
                    while True:
                        if worker_errors:
                            raise RuntimeError("Base model worker failed") from worker_errors[0]
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError(f"Base model did not request layer {layer_i} activation within 600 seconds")
                        try:
                            activations.put(layer_i, x, timeout=min(1.0, remaining))
                            break
                        except queue.Full:
                            continue
                    for check in align_by_layer[layer_i]:
                        docrow = cache["docids"].get((str(check["slice"]), int(check["line"])))
                        if docrow is not None:
                            bi = docrow
                            at = int(check["position"])
                            if at < x.shape[1]:
                                alignment_values[alignment_key(check)] = float(x[bi,at,int(check["neuron"])].float().cpu()) * int(check["pole"])
                else:
                    if worker_errors:
                        raise RuntimeError("Base model worker failed") from worker_errors[0]
                    deadline = time.monotonic() + 600
                    while True:
                        if worker_errors:
                            raise RuntimeError("Base model worker failed") from worker_errors[0]
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError(f"Chat model did not provide layer {layer_i} activation within 600 seconds")
                        try:
                            x = activations.get(layer_i, timeout=min(1.0, remaining))
                            break
                        except queue.Empty:
                            continue
                    y = inputs[0].detach()
                    col.add(layer_i, x, y.to(device_a), cache["masks"], perms[layer_i], thresholds)
            hooks.append(layer.mlp.down_proj.register_forward_pre_hook(hook))
    align_rows = prepared["pilot_alignment"] if args.stage == "pilot" else prepared["alignment"]
    def alignment_key(check):
        return f"{check['layer']}:{check['neuron']}:{check['pole']}:{check['slice']}:{check['line']}:{check['position']}"
    align_by_layer = {l:[r for r in align_rows if int(r["layer"])==l] for l in range(layers)}
    alignment_values = {}
    cache = {"masks": None, "docids": None}
    install(chat, hooks_a, True); install(base, hooks_b, False)
    docs, plan = [], []
    cut = dict((k, int(v)) for k, v in (x.split("=") for x in args.cut_docs.split(","))) if args.cut_docs else {}
    validate_corpus_paths(args.corpus)
    for name, path in zip(CORPUS_ORDER, args.corpus):
        rows = load_docs(path, tok, name, PILOT_DOCS_PER_SLICE if args.stage == "pilot" else cut.get(name, CORPUS_DOCS[name]))
        docs.extend((name, ids, row, name, line_no) for ids, row, _slice, line_no in rows)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sums_path = out / f"{args.stage}-sums.pt"
    corpus_hash=hashlib.sha256(repr([(r[0],r[4],r[1]) for r in docs]).encode("utf-8")).hexdigest()
    prepared_hash=hashlib.sha256(Path(args.prepared).read_bytes()).hexdigest()
    meta={"stage":args.stage,"docs":len(docs),"batch":args.batch,"chat_model":str(Path(args.chat_model).resolve()),
          "base_model":str(Path(args.base_model).resolve()),"corpus_sha256":corpus_hash,"prepared_sha256":prepared_hash}
    pos = 0
    if sums_path.exists():
        saved = torch.load(sums_path, map_location="cpu", weights_only=False)
        if args.cut_docs:  # a shorter corpus by amendment keeps the saved prefix; the rest of the identity must still match
            saved["meta"] = {**saved["meta"], "docs": meta["docs"], "corpus_sha256": meta["corpus_sha256"]}
        if not resume_metadata_matches(saved["meta"], meta):
            raise ValueError("resume metadata does not match this run (stage, models, corpus, and prepared catalog must match)")
        col.data.copy_(saved["state"]["data"].to(device_a)); col.count.copy_(saved["state"]["count"].to(device_a)); col.above.copy_(saved["state"]["above"].to(device_a)); alignment_values.update(saved["state"].get("alignment_values", {})); pos = saved["next_doc"]
    start_time = time.monotonic()
    while pos < len(docs):
        end = min(pos + args.batch, ((pos // CHUNK_SAVE) + 1) * CHUNK_SAVE, len(docs))
        while end > pos + 1 and docs[end - 1][0] != docs[pos][0]:
            end -= 1
        while end < min(len(docs), pos + args.batch, ((pos // CHUNK_SAVE) + 1) * CHUNK_SAVE) and docs[end][0] == docs[pos][0]:
            end += 1
        chunk = docs[pos:end]
        T = max(map(lambda r: len(r[1]), chunk))
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        ids = torch.full((len(chunk), T), pad_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for j, (_, row, *_rest) in enumerate(chunk): ids[j, :len(row)] = torch.tensor(row); mask[j, :len(row)] = 1
        names = {r[0] for r in chunk}
        if len(names) != 1:  # keep slice masks exact
            raise RuntimeError("batch crossed a corpus slice")
        masks = group_masks(ids.to(device_a), chunk[0][0], tok, mask.to(device_a))
        # Attach token masks to the same slice group for ordinary documents.
        if chunk[0][0] in ("dclm", "zh", "code"):
            masks = {g: (mask.to(device_a).bool() if g == {"dclm":"web", "zh":"zh", "code":"code"}[chunk[0][0]] else torch.zeros_like(mask, device=device_a, dtype=torch.bool)) for g in GROUPS}
        cache["masks"] = masks
        cache["docids"] = {(str(r[3]), int(r[4])): j for j,r in enumerate(chunk)}
        batch = ids.to(device_a)
        base_batch = batch.to(device_b)
        base_mask = mask.to(device_b)
        def base_forward():
            try:
                with torch.inference_mode():
                    base(input_ids=base_batch, attention_mask=base_mask)
            except BaseException as exc:
                worker_errors.append(exc)
        with torch.inference_mode():
            worker = threading.Thread(target=base_forward, name="part129-base-forward")
            worker.start()
            chat(input_ids=batch, attention_mask=mask.to(device_a))
            worker.join(timeout=600)
            if worker.is_alive():
                raise TimeoutError("Base model forward did not finish within 600 seconds")
            if worker_errors:
                raise RuntimeError("Base model forward failed") from worker_errors[0]
        pos += len(chunk)
        if pos % CHUNK_SAVE < args.batch or pos == len(docs):
            save_checkpoint(sums_path,meta,col,alignment_values,pos)
            print(f"saved {pos}/{len(docs)} docs, {pos/max(1,time.monotonic()-start_time):.2f} docs/s", flush=True)
    elapsed=time.monotonic()-start_time
    (out / "timing.json").write_text(json.dumps({"documents":len(docs),"batch":args.batch,"seconds":elapsed,
        "documents_per_second":len(docs)/max(elapsed,1e-9)},indent=2)+"\n",encoding="utf-8")
    torch.save({"layer_cosines":weight_cosines(chat, base)}, out / "weight-cosines.pt")
    (out / "alignment-values.json").write_text(json.dumps(alignment_values)+"\n",encoding="utf-8")
    if args.stage == "pilot":
        measured = [(check, alignment_values[alignment_key(check)]) for check in align_rows
                    if check["top_val"] > 0 and alignment_key(check) in alignment_values]
        passed = sum(abs(value - check["top_val"]) / check["top_val"] <= .05
                     for check, value in measured)
        rate = passed / len(measured) if measured else 0.0
        print(f"Gate 1 pilot alignment: {passed}/{len(measured)} checks measured, "
              f"{rate:.1%} within 5%", flush=True)
    for h in hooks_a + hooks_b: h.remove()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("pilot","main"), required=True)
    ap.add_argument("--chat-model", required=True); ap.add_argument("--base-model", required=True)
    ap.add_argument("--corpus", nargs=4, required=True); ap.add_argument("--prepared", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--cut-docs", default=""); args = ap.parse_args(); run(args)
if __name__ == "__main__": main()
