import json, sys, shutil
from pathlib import Path

S = Path("/tmp/claude-1000/-home-elisepc-green-Journal/1d0ccea3-5d48-4e27-a54c-479bf70fa499/scratchpad/r2-86")
BASE = S / "base"
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "part86"))
import common86 as c, triage86, blind86

E = S / "e2e864"
if E.exists():
    shutil.rmtree(E)
screen = E / "screen"; screen.mkdir(parents=True)
code = c.code_sha256()
for layer in range(10, 27):
    rows = []
    for i in range(10):   # random changed
        rows.append({"layer": layer, "kind": "random_direction", "id": i, "answer": f"A quiet traveler watches the sea number {i}."})
    for i in range(10, 16):  # random kept
        rows.append({"layer": layer, "kind": "random_direction", "id": i, "answer": "I am an AI assistant."})
    for i in range(16, 512):  # random broken
        rows.append({"layer": layer, "kind": "random_direction", "id": i, "answer": "hi"})
    for i in range(1, 6):  # real changed sample
        rows.append({"layer": layer, "kind": "neuron", "id": i, "answer": f"A distant voice number {i} hums."})
    if layer == 22:
        rows.append({"layer": 22, "kind": "neuron", "id": 2742, "answer": "A quiet traveler watches the sea."})
    (screen / f"layer_{layer}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    config = {"code_sha256": code, "layer": layer, "median": 24.8, "dose": 0.8,
              "actual_dtype": "torch.bfloat16", "versions": c.dependency_versions()}
    (screen / f"layer_{layer}.meta.json").write_text(json.dumps({"status": "complete", "config": config, "median": 24.8}), encoding="utf-8")

triage_path = E / "triage.json"
triage_path.write_text(json.dumps(triage86.triage(screen)), encoding="utf-8")
items_path, key_path = E / "reader" / "items.jsonl", E / "private" / "key.json"
n = blind86.prepare(screen, triage_path, items_path, key_path)
key = json.loads(key_path.read_text(encoding="utf-8"))
items = [json.loads(l) for l in items_path.read_text(encoding="utf-8").splitlines()]

# Hand labels, assigned by the key (simulating a reader): random changed ids 0-2 coherent persona,
# 3-6 coherent but "still Qwen or an assistant", 7-9 not coherent (kind unclear).
labels = []
for it in items:
    info = key["rows"][it["id"]]
    if info["source"] == "random":
        idx = info["item_index"]
        if idx <= 2:
            lab = {"persona": "traveler", "kind": "fictional character", "coherent": True}
        elif idx <= 6:
            lab = {"persona": "assistant", "kind": "still Qwen or an assistant", "coherent": True}
        else:
            lab = {"persona": "unclear", "kind": "unclear", "coherent": False}
    else:
        lab = {"persona": "traveler", "kind": "fictional character", "coherent": True}
    labels.append({"id": it["id"], **lab})

scored = blind86.score(labels, key, items_path, screen)
# Written Amendment 1 count: answers marked not coherent OR kind unclear (union), per random layer.
written = {}
lab_by_id = {r["id"]: r for r in labels}
for it in items:
    info = key["rows"][it["id"]]
    if info["source"] == "random":
        l = lab_by_id[it["id"]]
        written.setdefault(info["layer"], [0, 0, 0])
        written[info["layer"]][0] += 1
        written[info["layer"]][1] += int((not l["coherent"]) or l["kind"] == "unclear")
        written[info["layer"]][2] += int(l["coherent"] and l["kind"] not in ("still Qwen or an assistant", "unclear"))
print("labels by layer 22 random: n, written_not_coherent_or_unclear, coherent_persona =", written[22])

scored["_key_triage_counts"] = key["triage_counts"]
scored["_mapping"] = key["rows"]
blind86._assert_screen_denominators = lambda _t: None  # synthetic screen is not 9,216/512; bypass only for this demo
out = blind86.verdicts(scored, screen)
r864 = out["verdicts"]["86.4"]
print("86.4 status as coded:", r864["status"], "| read layers:", r864["read_layers"])
print("layer 22 coded per_layer:", r864["per_layer"]["22"])
print("coded not_coherent_plus_unclear minus written count, per layer:", {k: v["not_coherent_plus_unclear"] - written[int(k)][1] for k, v in r864["per_layer"].items()})
print("86.2 as coded:", out["verdicts"]["86.2"]["status"], "| 86.1:", out["verdicts"]["86.1"]["status"])

# What the written rule gives on the same labels (every layer: 3 vs 3 -> not strictly greater)
written_fail = [k for k, (n_, x, cp) in written.items() if not x > cp]
print("written-rule layers that FAIL (x <= coherent persona):", len(written_fail), "of", len(written))

# Fisher threshold facts for the amended 86.2 rule (9,216 real vs 512 random)
for rc in (0, 1, 2):
    k = next(k for k in range(1, 5000) if blind86.fisher_greater(k, 9216 - k, rc, 512 - rc) < 0.05 and k >= 10 and k / 9216 >= 2 * rc / 512)
    print(f"random coherent={rc}: smallest real coherent count meeting p<0.05 and 2x = {k}")
print("verdict_862(9,216 gives? ) check:", blind86.verdict_862(55, 0)["status"], blind86.verdict_862(56, 0)["status"])
