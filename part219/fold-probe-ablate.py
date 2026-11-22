"""Does the ablation zero the named o_proj slice on every forward (prefill AND each decode step)
and nothing else? Drives the real swap99.install_ablation_hooks on the tiny OLMo 2.
"""
import sys
from pathlib import Path

import torch

WORKTREE = str(Path(__file__).resolve().parents[1])  # the worktree this copy lives in
sys.path.insert(0, WORKTREE)
sys.path.insert(0, WORKTREE + "/part92")
sys.path.insert(0, WORKTREE + "/part99")

import common92 as c  # noqa: E402
import swap99  # noqa: E402

HOST = "/home/elisepc/green/Journal/scratch/review-part219-code-r1-ds-r1-2026-11-22/exp/world/host"
HEAD = 1
HEAD_SIZE = 8
START, END = HEAD * HEAD_SIZE, (HEAD + 1) * HEAD_SIZE


def capture(model, prompt, max_new_tokens):
    """Inputs layer 1's o_proj sees on every forward of one generate() call."""
    captured = []

    def after_ablation(_module, args):
        captured.append(args[0].detach().clone())

    handle = model.model.layers[1].self_attn.o_proj.register_forward_pre_hook(after_ablation)
    try:
        with torch.inference_mode():
            model.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt), do_sample=False,
                           max_new_tokens=max_new_tokens, pad_token_id=0)
    finally:
        handle.remove()
    return captured


tokenizer, model = c.load_model(HOST, "cpu", attn_implementation="eager")
with torch.inference_mode():
    prompt = torch.tensor([c.prompt_ids(tokenizer, "Who are you?", "chat")], dtype=torch.long)
clean = capture(model, prompt, 4)

handles, receipt = swap99.install_ablation_hooks(model, [(1, HEAD)])
try:
    dirty = capture(model, prompt, 4)
finally:
    for handle in handles:
        handle.remove()

print("forwards captured: clean", len(clean), "dirty", len(dirty))
print("receipt entries:", receipt)
print("prefill other-slices equal:", torch.equal(dirty[0][..., :START], clean[0][..., :START]),
      torch.equal(dirty[0][..., END:], clean[0][..., END:]))
issues = []
for index, tensor in enumerate(dirty):
    if int(torch.count_nonzero(tensor[..., START:END])) != 0:
        issues.append((index, "slice not zero", int(torch.count_nonzero(tensor[..., START:END]))))
    if tensor.shape[-1] != 16:
        issues.append((index, "width", tuple(tensor.shape)))
print("decode-step shapes:", [tuple(t.shape) for t in dirty])
print("every forward (prefill and decode) has its slice exactly zero:", not issues, issues)
print("clean prefill slice was non-zero (so the test has teeth):",
      int(torch.count_nonzero(clean[0][..., START:END])))
