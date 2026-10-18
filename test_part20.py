#!/usr/bin/env python3
"""CPU regression checks for Part 20, using a tiny random Qwen3.5 model."""
import json
import os
import tempfile

import torch

import neuron_report as nr
import part20

SNAP = os.environ["QWEN35_SNAPSHOT"]
fails = 0


def check(name, ok, detail=""):
    global fails
    print(("  ok   " if ok else "  FAIL ") + name + (f"  ({detail})" if detail and not ok else ""), flush=True)
    fails += not ok


def tiny_model(dirname):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    cfg = json.load(open(os.path.join(SNAP, "config.json"), encoding="utf-8"))
    t = dict(cfg.get("text_config", cfg))
    t.update(num_hidden_layers=4, hidden_size=64, intermediate_size=96, num_attention_heads=4,
             num_key_value_heads=2, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
             linear_key_head_dim=16, linear_value_head_dim=16, layer_types=t["layer_types"][:4],
             tie_word_embeddings=True)
    t.pop("architectures", None)
    conf = AutoConfig.for_model(t.get("model_type", "qwen3_5_text"), **{k: v for k, v in t.items() if k != "model_type"})
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(conf, dtype=torch.float32)
    model.save_pretrained(dirname)
    AutoTokenizer.from_pretrained(SNAP).save_pretrained(dirname)


class Args:
    model = ""
    revision = None
    layer = 2
    neuron = 3
    device = "cpu"
    max_gpu_gb = 14.5
    max_new = 5


with tempfile.TemporaryDirectory() as tmp:
    mdir = os.path.join(tmp, "model")
    tiny_model(mdir)
    tok, model = nr.load_model(mdir, None, "cpu")
    _, layers, _ = nr.find_text_layers(model)
    down = layers[2].mlp.down_proj
    args = Args()
    args.model = mdir
    text = nr.build_prompt(tok, "Who made you?", False)

    one = part20.identity_pass(model, tok, down, text, args, "last")
    every = part20.identity_pass(model, tok, down, text, args, "all")
    check("identity hook at one position has exact zero logit difference",
          one["overall_max_abs_logit_diff"] == 0.0, one["overall_max_abs_logit_diff"])
    check("identity hook at every position has exact zero logit difference",
          every["overall_max_abs_logit_diff"] == 0.0, every["overall_max_abs_logit_diff"])
    bad = part20.identity_pass(model, tok, down, text, args, "all", sabotage=1e-3)
    check("sabotaged identity hook is detected", bad["overall_max_abs_logit_diff"] > 0.0,
          bad["overall_max_abs_logit_diff"])

    good_gen = part20.generate_capture(model, tok, down, text, args, max_new=5)
    bad_gen = part20.generate_capture(model, tok, down, text, args, max_new=5, sabotage=1000.0)
    check("generation replay preserves greedy token ids", good_gen["identical"])
    check("sabotaged generation replay changes greedy token ids", not bad_gen["identical"],
          (good_gen["token_ids"], bad_gen["replay_token_ids"]))

    corpus = os.path.join(tmp, "pile.jsonl")
    with open(corpus, "w", encoding="utf-8") as f:
        for textrow in ["A harmless record with ordinary words.", "An alien crossed the quiet valley.",
                        "The nebula shone brightly tonight.", "Neutral filler follows the old stone road."]:
            f.write(json.dumps({"text": textrow}) + "\n")
    filler, indexes, count = part20.select_filler(corpus, tok, target_tokens=1)
    check("filler selection skips alien and nebula documents", indexes == [0], indexes)
    check("filler token count is measured on concatenated filler", count == len(tok(filler, add_special_tokens=False)["input_ids"]))

    alien = "The alien's spacecraft hovered silently above the empty desert road."
    alien_m = __import__("re").search(r"\balien\b", alien, __import__("re").I)
    alien_pos = part20.token_span(tok, alien, alien_m.span())
    summary = part20.SUMMARY
    summary_pos = part20.token_span(tok, summary, (0, len(summary)))
    enc = tok(alien, add_special_tokens=False, return_offsets_mapping=True)
    check("alien target span maps to its real tokenizer token", alien_pos == [i for i, (s, e) in enumerate(enc["offset_mapping"]) if e > alien_m.start() and s < alien_m.end()], alien_pos)
    check("summary span maps across summary tokens", summary_pos == list(range(len(tok(summary, add_special_tokens=False)["input_ids"]))), summary_pos)

    idout = os.path.join(tmp, "identity.json")
    prompts = os.path.join(tmp, "prompts.json")
    json.dump(["Who made you?"], open(prompts, "w", encoding="utf-8"))
    part20.main(["identity", "--model", mdir, "--layer", "2", "--neuron", "3", "--device", "cpu",
                 "--max-new", "2", "--out", idout, "--prompts", prompts])
    ih = json.load(open(idout, encoding="utf-8"))
    check("identity JSON has required header fields", all(k in ih for k in ("model", "revision", "tool_sha256", "inputs", "layer", "neuron", "device", "dtype", "written")))

    sout = os.path.join(tmp, "seam.json")
    part20.main(["seam", "--model", mdir, "--layer", "2", "--neuron", "3", "--device", "cpu",
                 "--out", sout, "--corpus", corpus])
    sh = json.load(open(sout, encoding="utf-8"))
    check("seam JSON has required header fields", all(k in sh for k in ("model", "revision", "tool_sha256", "inputs", "layer", "neuron", "device", "dtype", "written")))

print(f"test_part20: {fails} failures")
raise SystemExit(1 if fails else 0)
