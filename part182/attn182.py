#!/usr/bin/env python3
"""attn182.py -- Part 182 Amendment 1, item 7 (Trap 6): where each listed head sends its attention from the answer
positions, host against donor, on the 19 eligible slot units' own-answer rows.

  python3 -I part182/attn182.py --host DIR --donor DIR --heads h0.3,h2.1,c14.0 --out PATH

Rows: run182.build_stage1_rows' own-answer rows (unit "neuron", text "own") of score182.eligible_units' slot units,
each run twice: with the row's push on and with no push. For each listed head h in layer L, for each answer position
p (the positions logprobs_from_logits scores: prompt_len-1 .. prompt_len-1+n-1, right padding excluded) the attention
row a[p, :] of that head over the keys 0 .. p (the causal mask makes the rest exactly zero), summed over three
disjoint position groups:

  template  the chat-template tokens of the prompt (the prompt positions that are not the question's tokens)
  question  the question's own token ids, located inside the prompt ids (exactly once, or this fails loudly)
  answer    the answer token positions

then averaged over the answer positions and over the rows. Two modules are read: the host's own attention and the
donor's attention run on the host's hidden states at L (the same input HeadSwapAttention gives the donor). The weights
come from the eager attention forward (the module's forward returns them as its second output); no attention step is
re-implemented. The o_proj is swapped for an identity only so the module does not pay for it; the weights are the
module's own.

A listed head is always the ``host`` number of the host's own head; ``donor`` is the donor's module of the same index
running on the host's states.

Output JSON:
{
  "host": ..., "donor": ..., "frame": "chat", "question": "...", "prompt_len": 15,
  "groups": {"template_ids": [ids], "question_ids": [ids], "template_positions": [..], "question_positions": [..],
             "answer_positions": "prompt_len-1 .. prompt_len-1+n-1 per row"},
  "rows": [{"id", "unit", "push": "on"/"off", "n_tokens", "answer_positions": [..]}],
  "heads": {"<tag>": {"<push>": {"host": {"template": f, "question": f, "answer": f, "n_positions": n},
                                 "donor": {...}}}},
  "sum_check_max_deviation": f        # max |template+question+answer-1| over every answer position, both modules
}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _folder in (HERE, ROOT / "part92", ROOT / "part99", ROOT):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import torch  # noqa: E402
from torch import nn  # noqa: E402

import common92 as c  # noqa: E402
import headswap182 as hs  # noqa: E402
import run182  # noqa: E402
import score182  # noqa: E402

GROUPS = ("template", "question", "answer")
_IDENTITY = nn.Identity()


def find_span(haystack: list[int], needle: list[int]) -> int:
    """The start index of ``needle`` in ``haystack``; raises unless it occurs exactly once."""
    hits = [i for i in range(len(haystack) - len(needle) + 1) if haystack[i:i + len(needle)] == list(needle)]
    if len(hits) != 1:
        raise ValueError(f"the question's ids {needle} occur {len(hits)} times in the prompt {haystack}")
    return hits[0]


def position_groups(tokenizer, question: str, prompt: list[int]) -> dict[str, list[int]]:
    """The prompt's positions split into the question's and the template's; the question's ids are located inside the
    prompt, not re-rendered."""
    question_ids = [int(t) for t in tokenizer(question, add_special_tokens=False)["input_ids"]]
    if not question_ids:
        raise ValueError("the question tokenises to nothing")
    start = find_span(prompt, question_ids)
    question_positions = list(range(start, start + len(question_ids)))
    template_positions = [p for p in range(len(prompt)) if p not in set(question_positions)]
    return {"template": template_positions, "question": question_positions}


def attention_weights(attn, hidden_states, position_embeddings, attention_mask):
    """The attention weights [batch, heads, queries, keys] of one attention module on these hidden states, from its
    own forward. The o_proj is replaced by an identity for the call and put back; the weights do not depend on it."""
    o_proj = attn._modules["o_proj"]
    attn._modules["o_proj"] = _IDENTITY
    try:
        with torch.inference_mode():
            _output, weights = attn(hidden_states, position_embeddings, attention_mask)
    finally:
        attn._modules["o_proj"] = o_proj
    if weights is None:
        raise RuntimeError(f"layer {attn.layer_idx}: the attention forward returned no weights (eager needed)")
    return weights.float()


def group_sums(weights, head, groups, answer_positions, answer_mask) -> tuple[dict[str, float], int]:
    """Per-group sums of one head's attention over every scored answer position, summed over the batch; and the number
    of positions summed. A position whose row is right-padded is not scored and is skipped."""
    totals = {g: 0.0 for g in GROUPS}
    count = 0
    for row, positions in enumerate(answer_positions):
        for p in positions:
            if not answer_mask[row][p]:
                continue
            row_weights = weights[row, head, p]
            for group in GROUPS:
                totals[group] += float(row_weights[groups[group]].sum())
            count += 1
    return totals, count


def slot_own_rows(rows, slots) -> list[dict]:
    """The eligible slot units' own-answer rows (unit "neuron", text "own"), in build_stage1_rows' order."""
    picked = [r for r in rows if r["unit"] == "neuron" and r["text"] == "own" and r["id"] in set(slots)]
    if not picked:
        raise ValueError("no eligible slot own-answer rows")
    return picked


def head_attention_shares(host_dir, donor_dir, heads, rows, prompt, tokenizer, question=run182.QUESTION,
                          device="cpu", batch=16, host_model=None, donor_attn=None) -> dict:
    """The full record described in the module docstring. ``rows`` are the slot units' own-answer rows (pushed);
    ``prompt`` is ``common92.prompt_ids``' chat prompt for ``question``. ``host_model`` and ``donor_attn`` let a test
    pass its own tiny model and donor modules instead of loading ``host_dir`` and ``donor_dir``."""
    if host_model is None:
        _tokenizer, host = run182.load_host(host_dir, device)
    else:
        host = host_model
    layers = c.get_layers(host)
    hidden = int(host.config.hidden_size)
    donors = (hs.load_donor_attention(donor_dir, sorted({int(l) for l, _h in heads}), dtype=torch.float32)
              if donor_attn is None else donor_attn)
    groups = position_groups(tokenizer, question, prompt)  # template and question prompt positions
    record = {"host": str(host_dir), "donor": str(donor_dir), "frame": "chat", "question": question,
              "prompt_len": len(prompt),
              "groups": {"template_ids": [prompt[p] for p in groups["template"]],
                         "question_ids": [prompt[p] for p in groups["question"]],
                         "template_positions": groups["template"],
                         "question_positions": groups["question"],
                         "answer_positions": "prompt_len-1 .. prompt_len-1+n-1 per row"},
              "rows": [], "heads": {}, "sum_check_max_deviation": 0.0}
    by_layer: dict[int, list[int]] = {}
    for layer_no, head in heads:
        by_layer.setdefault(int(layer_no), []).append(int(head))
    for layer_no in sorted(by_layer):
        if layer_no not in donors:
            raise KeyError(f"no donor attention for layer {layer_no}")
    dtype = host.model.layers[0].self_attn.q_proj.weight.dtype
    for push_mode in ("on", "off"):
        pushed_rows = [r if push_mode == "on" else dict(r, push=None) for r in rows]
        totals = {(l, h): {"host": {g: 0.0 for g in GROUPS}, "donor": {g: 0.0 for g in GROUPS}}
                  for l, hs_ in by_layer.items() for h in hs_}
        count = 0
        for begin in range(0, len(pushed_rows), batch):
            chunk = pushed_rows[begin:begin + batch]
            input_ids, attention_mask = hs.build_batch(prompt, [r["answer_ids"] for r in chunk],
                                                       int(tokenizer.pad_token_id), device)
            vectors: dict[int, torch.Tensor] = {}
            for i, r in enumerate(chunk):
                if r.get("push") is None:
                    continue
                layer_push = int(r["push_layer"])
                vectors.setdefault(layer_push, torch.zeros((len(chunk), hidden), dtype=torch.float32, device=device))
                vectors[layer_push][i] = r["push"].to(device)
            answer_positions = hs.answer_positions(len(prompt), [r["answer_ids"] for r in chunk], attention_mask)
            answer_mask = attention_mask.tolist()
            # The answer group is the answer's own positions, the same for every row of the batch (right padding is
            # excluded per row by answer_positions, not by the group).
            row_groups = {**groups, "answer": list(range(len(prompt), input_ids.shape[1]))}
            stores, handles = hs.capture_layer_outputs(host, sorted(by_layer))
            handles += [c.add_row_hook(layers[l], v) for l, v in sorted(vectors.items())]
            try:
                with torch.inference_mode():
                    host(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            finally:
                for handle in handles:
                    handle.remove()
            position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
            # The same mask the model built for this forward (the installed transformers' own function, not a
            # re-implementation): causal, with the right padding masked.
            from transformers.masking_utils import create_causal_mask
            mask = create_causal_mask(config=host.config, inputs_embeds=stores[sorted(by_layer)[0]][-1],
                                      attention_mask=attention_mask, past_key_values=None,
                                      position_ids=position_ids)
            for layer_no, head_list in sorted(by_layer.items()):
                hidden_states = stores[layer_no][-1].to(dtype)
                position_embeddings = host.model.rotary_emb(hidden_states, position_ids=position_ids)
                for name, module in (("host", layers[layer_no].self_attn), ("donor", donors[layer_no])):
                    weights = attention_weights(module, hidden_states, position_embeddings, mask)
                    for h in head_list:
                        sums, n = group_sums(weights, h, row_groups, answer_positions, answer_mask)
                        for group in GROUPS:
                            totals[(layer_no, h)][name][group] += sums[group]
                        deviation = abs(sum(sums.values()) - n)
                        record["sum_check_max_deviation"] = max(record["sum_check_max_deviation"], deviation)
            count += sum(1 for row, positions in enumerate(answer_positions) for p in positions if answer_mask[row][p])
        if count == 0:
            raise ValueError("no answer positions to read")
        for (layer_no, h), entry in totals.items():
            record["heads"].setdefault(f"h{layer_no}.{h}", {})[push_mode] = {
                name: {**{g: entry[name][g] / count for g in GROUPS}, "n_positions": count}
                for name in ("host", "donor")}
    record["rows"] = [{"id": r["id"], "unit": r["unit"], "push": "on", "n_tokens": len(r["answer_ids"]),
                       "answer_positions": list(range(len(prompt) - 1, len(prompt) - 1 + len(r["answer_ids"])))}
                      for r in rows]
    return record


def rows_from_results(host_dir, question=run182.QUESTION, frame="chat", max_tokens=run182.MAX_TOKENS):
    """The real inputs: the tokenizer, the chat prompt, and the eligible slot units' own-answer rows. Reads Part 92's
    and Part 99's stored files, not a model."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(Path(host_dir).expanduser()), local_files_only=True)
    units, kinds = run182.load_units()
    pushes = torch.load(run182.PUSHES, map_location="cpu")
    answers = run182.load_p99_answers()
    eligible, _ = score182.eligible_units(score182.P99_LABELS, kinds)
    rows = slot_own_rows(run182.build_stage1_rows(tokenizer, units, pushes, answers, max_tokens), eligible["slot"])
    return tokenizer, c.prompt_ids(tokenizer, question, frame), rows


def parse_heads(spec: str, control_layers=None) -> list[tuple[int, int]]:
    """hL.H and cL.H tags (cL.H through headswap182.condition_heads, so the control layers are checked) into a list of
    (layer, head). A repeated head is kept once."""
    out = []
    for tag in spec.split(","):
        tag = tag.strip()
        if not tag:
            continue
        for layer_no, head_list in hs.condition_heads(tag, score182.LAYERS, score182.N_HEADS,
                                                      control_layers=control_layers).items():
            for head in head_list:
                if (layer_no, head) not in out:
                    out.append((layer_no, head))
    if not out:
        raise ValueError("--heads listed no head")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", required=True)
    ap.add_argument("--donor", required=True)
    ap.add_argument("--heads", required=True, help="comma-separated hL.H or cL.H tags")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", choices=("cpu", "cuda", "xpu"), default="cpu")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args(argv)
    tokenizer, prompt, rows = rows_from_results(args.host)
    record = head_attention_shares(args.host, args.donor, parse_heads(args.heads), rows, prompt, tokenizer,
                                   device=args.device, batch=args.batch)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(record["heads"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
