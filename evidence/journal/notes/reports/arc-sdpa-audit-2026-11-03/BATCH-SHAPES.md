# Batch shape and cache length for Arc-SDPA suspect parts

Prompt lengths were measured offline with the cached Qwen3.5-4B tokenizer, applying the relevant chat-template path; records that store `prompt_ids` use those lengths directly. `max_cache_keys` is the requested bound `max prompt tokens + max_new_tokens`, so it is budget reachability rather than an observed EOS length. “No” in the padded column means rows in a generation batch have equal-length prompts, even if the code passes an all-ones mask.

## Severity counts

High: 22
Medium: 22
Low: 11
None: 0
Unknown: 0

## All suspect parts

Rows are sorted by severity and then part number. Evidence entries give source/result file:line citations; the full list is also in `batch-shapes.csv`.
| Part | Batch sizes used | Padded | Max new | Prompt median / max | Max cache keys | Comparison | Severity | Evidence |
|---|---:|:---:|---:|---:|---:|:---:|:---:|---|
| 18 | 1 | no | 96 | 18.5 / 23 | 119 | both | high | neuron_report.py:90-99,363-370,411-421; results/part18/chat/steer.json:1; INVENTORY.md:7 |
| 18b | 1 | no | 96 | 18.5 / 23 | 119 | absolute | high | neuron_report.py:90-99,363-370,411-421; results/part18b/phase-gen/steer.json:1; INVENTORY.md:8 |
| 19 | 1 | no | 60 | 17 / 20 | 80 | absolute | high | results/part19/answers.json:1; INVENTORY.md:9; neuron_report.py:363-370,411-421 |
| 20 | 1 | no | 60 | 18.5 / 23 | 83 | both | high | part20.py:119-138,155-163,280; results/part20/identity.json:1; neuron_report.py:363-370 |
| 24 | 1 | no | 64 | 16 / 16 | 80 | both | high | part24.py:507-525,559-561; results/part24/samples.json:1; neuron_report.py:363-370 |
| 25 | 1 | no | 48 | 17 / 31 | 79 | both | high | part25.py:112,282-300,818; results/part25/read.json:1; README-part25.md:1-6; neuron_report.py:363-370 |
| 39 | 1 | no | 60 | 21 / 23 | 83 | both | high | part39.py:927-946,1024-1026,1393; results/part39/push.jsonl:1; neuron_report.py:363-370 |
| 79 | 1,16 | no | 64,128 | 18 / 30 | 158 | both | high | part79/part79.py:67,71-93,110; part79/step0.py:72,88; results/part79/main/run.json:1; results/part79/main/answers.jsonl:1; neuron_report.py:363-370 |
| 81 | 1 | no | 96 | 18 / 23 | 119 | both | high | part81/part81.py:169,196,217-269,283; results/part81/main/run.json:1; results/part81/main/answers.jsonl:1; neuron_report.py:363-370 |
| 84 | 1 | no | 64 | 18 / 18 | 82 | both | high | part84/part84.py:11-12,70,83-94; results/part84/main/run.json:1; neuron_report.py:363-370 |
| 84b | 1 | no | 64 | 18 / 18 | 82 | relative | high | part84b/part84b.py:9,22,47; results/part84b/main/run.json:1; neuron_report.py:363-370 |
| 140 | 1,16 | no | 80 | 16.5 / 17 | 97 | both | high | part140/run140.py:55-65,99-121; results/part140/launch.jsonl:1; results/part140/README.md:1-3; part86/common86.py:152-154,279-286 |
| 148 | 1,36,64 | no | 120 | 17 / 17 | 137 | both | high | part148/battery148.py:17-18,45-58,125-150; results/part148/launch.jsonl:1; results/part148/README.md:1-3; part86/common86.py:152-154,279-286 |
| 153 | 1,20 | no | 120 | 18 / 30 | 150 | both | high | part153/battery153.py:67-75,97-124,139-142; results/part153/launch.jsonl:1; results/part153/answers.jsonl:1; part86/common86.py:152-154,279-286 |
| 161 | 2,4,64 | no | 80 | 18 / 30 | 110 | both | high | part161/run161.py:23,136-142,189-205,227-235,261-264; results/part161/launch.jsonl:1; results/part161/answers.jsonl:1; part86/common86.py:152-154,279-286 |
| 163 | 1,64 (partial tails) | no | 80 | 18 / 26 | 106 | both | high | part163/run163.py:23,145-168,213-218,239-241; results/part163/launch.jsonl:1; results/part163/answers.jsonl:1; part86/common86.py:152-154,279-286 |
| 164 | 1,64 (partial tails) | no | 80 | 17 / 30 | 110 | both | high | part164/run164.py:23,146,155-172,183-185; results/part164/launch.jsonl:1; results/part164/answers.jsonl:1; part86/common86.py:152-154,279-286 |
| 165 | 1,33,64 | no | 80 | 16.5 / 18 | 98 | both | high | part165/run165.py:23,205-228,249-251; results/part165/launch.jsonl:1; results/part165/answers.jsonl:1; part86/common86.py:152-154,279-286 |
| 167 | 1,32,64 | no | 80 | 18 / 30 | 110 | absolute | high | part167/run167.py:93-110,158-161,191,215-216,281-290,297,306-308; results/part167/launch.jsonl:1; results/part167/answers.jsonl:1; part86/common86.py:152-154,279-286 |
| 168 | 1,32,64 | no | 80 | 18 / 30 | 110 | both | high | part168/run168.py:110-127,235-242,260,283,302-303,379-388,399-401; results/part168/launch.jsonl:1; results/part168/answers.jsonl:1; part86/common86.py:152-154,279-286 |
| 169 | 1,64 (partial tails) | no | 80 | 16.5 / 18 | 98 | both | high | part169/run169.py:161-181,293-295,319,332-350,399-400; results/part169/launch.jsonl:1; results/part169/answers.jsonl:1; part86/common86.py:152-154,279-286 |
| 170 | 1,5-6,64 (partial tails) | no | 80 | 20 / 24 | 104 | both | high | part170/run170.py:33,651-684,735-753,771-835; results/part170/launch.jsonl:1; results/part170/answers.jsonl:1; part86/common86.py:152-154,279-286 |
| 66 | 8,24,25 | no | 48 | 18 / 30 | 78 | both | medium | part66/part66.py:106-115,142-167,195; results/part66/pilot/run.json:1; results/part66/positive96/run.json:1; results/part66/full/run.json:1; results/part66/full/answers.jsonl:1; neuron_report.py:363-370 |
| 70 | 25 | no | 48 | 18 / 30 | 78 | both | medium | part70/part70.py:117-155,193; results/part70/full/run.json:1; results/part70/full/answers.jsonl:1; neuron_report.py:363-370 |
| 73 | 25 | no | 48 | 18 / 30 | 78 | relative | medium | part73/part73.py:15,67; part70/part70.py:135-155,193; results/part73/run.json:1; results/part73/answers.jsonl:1; neuron_report.py:363-370 |
| 75 | 32 | no | 48 | 18 / 30 | 78 | relative | medium | part75/part75.py:71; part70/part70.py:135-155,193; results/part75/run.json:1; results/part75/answers.jsonl:1; neuron_report.py:363-370 |
| 77 | 32 | no | 48 | 18 / 30 | 78 | relative | medium | part77/part77.py:141-164,157,208,290; results/part77/run.json:1; results/part77/answers.jsonl:1; neuron_report.py:363-370 |
| 78 | 16 | no | 64 | 18 / 30 | 94 | relative | medium | part78/part78.py:76-106,119; results/part78/run.json:1; results/part78/answers.jsonl:1; neuron_report.py:363-370 |
| 78b | 16 | no | 64 | 18 / 30 | 94 | relative | medium | part78b/part78b.py:73,87-114,137; results/part78b/run.json:1; results/part78b/answers.jsonl:1; neuron_report.py:363-370 |
| 94 | 16 | no | 120 | 14.5 / 20 | 140 | both | medium | part94/push94.py:19-22,47-69; results/part94/answers-instruct.jsonl:1; results/part94/meta-instruct.json:1; neuron_report.py:363-370 |
| 94b | 16 | no | 120 | 14.5 / 20 | 140 | both | medium | part94/push94.py:19-22,47-69; results/part94b/answers-instruct-neg.jsonl:1; results/part94b/meta-instruct-neg.json:1; neuron_report.py:363-370 |
| 94c | 26 | no | 120 | 14.5 / 20 | 140 | both | medium | part94/push94c.py:72-98,109; results/part94c/answers-base-gentle.jsonl:1; results/part94c/meta-base-gentle.json:1; neuron_report.py:363-370 |
| 96 | 16 | no | 120 | 16.5 / 24 | 144 | both | medium | part96/push96.py:17,23,31,37,63-68,104; results/part96/meta-instruct.json:1; results/part96/answers-instruct.jsonl:1; neuron_report.py:363-370 |
| 112 | 34,64 | no | 120 | 17.5 / 20 | 140 | both | medium | part112/battery112.py:20-22,54-65,87,104,109-123; results/part112/answers.jsonl:1; results/part112/answers.jsonl:1; neuron_report.py:363-370 |
| 113 | 19 | no | 120 | 17 / 20 | 140 | relative | medium | part113/neuron113.py:17,51-61; part112/battery112.py:54-65; results/part113/answers.jsonl:1; results/part113/answers.jsonl:1; neuron_report.py:363-370 |
| 117 | 58,64 | no | 120 | 17.5 / 20 | 140 | relative | medium | part117/pairs117.py:22,63-65,85-97; results/part117/launch.jsonl:1; results/part117/answers.jsonl:1; part86/common86.py:152-154,279-286 |
| 117b | 25,64 | no | 40,120 | 17.5 / 20 | 140 | relative | medium | part117b/rows117b.py:81-109,119-120,141-160; results/part117b/launch.jsonl:1; results/part117b/answers.jsonl:1; part86/common86.py:152-154,279-286 |
| 121 | 1,27,64 | no | 40,120 | 16.5 / 17 | 137 | relative | medium | part121/rows121.py:143-149,164-171,173-195; results/part121/launch.jsonl:1; results/part121/run/answers.jsonl:1; part86/common86.py:152-154,279-286 |
| 150 | 55 | no | 120 | 14.5 / 20 | 140 | both | medium | part150/push150.py:19-45,85-97,135-151,165-167; results/part150/launch.jsonl:1; results/part150/answers-instruct.jsonl:1; neuron_report.py:363-370 |
| 154 | 24,72 | no | 120 | 18 / 30 | 150 | both | medium | part154/run154.py:34-36,205-218,248-282,290-302; results/part154/launch.jsonl:1; results/part154/answers.jsonl:1; part86/common86.py:152-154,279-286 |
| 155 | 81 | no | 120 | 18 / 30 | 150 | relative | medium | part155/run155.py:21-30,81-82,124-145; results/part155/launch.jsonl:1; results/part155/answers.jsonl:1; part86/common86.py:152-154,279-286 |
| 156 | 57 | no | 120 | 18 / 30 | 150 | relative | medium | part156/run156.py:17,28-32; part155/run155.py:81-82,124-145; results/part156/launch.jsonl:1; results/part156/answers.jsonl:1; part86/common86.py:152-154,279-286 |
| 160 | 1,36,64 | no | 80,120 | 17.5 / 30 | 150 | both | medium | part160/run160.py:75-95,129-140,162-165,200-214,229-233; results/part160/launch.jsonl:1; results/part160/answers.jsonl:1; part86/common86.py:152-154,279-286 |
| 166 | 1,12,32,64 | no | 40,80 | 16.5 / 18 | 98 | both | medium | part166/run166.py:14-15,32-38,269-285,352-362,377-379; results/part166/launch.jsonl:1; results/part166/rows.jsonl:1; part86/common86.py:152-154,279-286 |
| 157 | 24 | yes | 120 | 18 / 30 | 150 | relative | low | part157/run157.py:130-145,148-179,202-220; results/part157/launch.jsonl:1; results/part157/answers.jsonl:1 |
| 158 | 24 | yes | 120 | 18 / 30 | 150 | relative | low | part158/run158.py:71-109; part157/run157.py:130-145,148-179; results/part158/launch.jsonl:1; results/part158/answers.jsonl:1 |
| 159 | 24 | yes | 40 | 18 / 30 | 70 | relative | low | part159/run159.py:60-72,89-107; part157/run157.py:130-145; results/part159/launch.jsonl:1; results/part159/answers.jsonl:1 |
| 162 | 16 | yes | 200 | 27 / 36 | 236 | absolute | low | part162/gen162.py:21,99-128,157-164; results/part162/launch.jsonl:1; results/part162/answers.jsonl:1 |
| 171 | 24,44 | yes | 80 | 20 / 46 | 126 | both | low | part171/run171.py:26,32,279-298,349-368; results/part171/launch.jsonl:1; results/part171/answers.jsonl:1 |
| 172 | 44 | yes | 80 | 23 / 51 | 131 | both | low | part172/run172.py:32,353-373,407-431; part171/run171.py:279-298,349-368; results/part172/launch.jsonl:1; results/part172/answers.jsonl:1 |
| 173 | 44 | yes | 80 | 20 / 50 | 130 | both | low | part173/run173.py:36-37,96-97; part171/run171.py:279-298,349-368; results/part173/launch.jsonl:1; results/part173/answers.jsonl:1 |
| 174 | 44 | yes | 80 | 33 / 50 | 130 | both | low | part174/run174.py:39-40,87-88,178-186; part171/run171.py:279-298,349-368; results/part174/launch.jsonl:1; results/part174/answers.jsonl:1 |
| 175 | 44 | yes | 80 | 33 / 50 | 130 | both | low | part175/run175.py:52,611-619,635-650,975-984; part171/run171.py:279-298,349-368; results/part175/launch.jsonl:1; results/part175/answers.jsonl:1 |
| 176 | 44 | yes | 80 | 33 / 50 | 130 | both | low | part176/run176.py:44,211-212,311,326-334,601-611; part171/run171.py:279-298,349-368; results/part176/launch.jsonl:1; results/part176/answers.jsonl:1 |
| 178 | 16,64 | yes | 2048 | 75 / 722 | 2770 | both | low | part178/run178.py:26-42,302-323,383-434,478-480,550-562,593-594; results/part178/launch.jsonl:12-13; results/part178/N8.jsonl:1; results/part178/R8.jsonl:1 |

## High-severity headlines

- Part 18: Part 18 reports alien wording in 8/8 steered answers versus 0/8 baseline, with 0/160 random controls. (No dedicated README found; inventory headline: `INVENTORY.md:7`.)
- Part 18b: Part 18b reports 7/8 for generation-only versus 0/8 for prompt-only. (No dedicated README found; inventory headline: `INVENTORY.md:8`.)
- Part 19: Part 19 combines identity effects from generated answers with activation and gradient evidence. (No dedicated README found; inventory headline: `INVENTORY.md:9`.)
- Part 20: Part 20 compares saved identity/replay output, while most seam measurements come from activation reads. (No dedicated README found; inventory headline: `INVENTORY.md:10`.)
- Part 24: Part 24 reports sampled free answers alongside fixed-answer scoring. (No dedicated README found; inventory headline: `INVENTORY.md:11`.)
- Part 25: Part 25 asks which neurons change Qwen3.5’s stated maker and reports four neurons selected for Google attribution. (`README-part25.md:1-5`.)
- Part 39: Part 39 makes answer claims alongside activation readouts. (No dedicated README found; inventory headline: `INVENTORY.md:13`.)
- Part 79: Part 79 compares cached main answers and includes a generated step-0 record. (No dedicated README found; inventory headline: `INVENTORY-B.md:14`.)
- Part 81: Part 81’s registered pushed and sampled response claims use generated answers. (No dedicated README found; inventory headline: `INVENTORY-B.md:15`.)
- Part 84: Part 84’s additive-recipe claim uses sampled generated responses. (No dedicated README found; inventory headline: `INVENTORY-B.md:16`.)
- Part 84b: Part 84b’s control comparison uses sampled generated responses. (No dedicated README found; inventory headline: `INVENTORY-B.md:17`.)
- Part 140: At the Part 86 dose, arm A reaches 48.3% joint success on Set P while arm C reaches 5.8%. (`results/part140/README.md:1-3`.)
- Part 148: Part 148 compares neuron pushes with paired random directions across three questions and 465 atlas neurons. (`results/part148/README.md:1-3`.)
- Part 153: Part 153 tests whether ten selected neurons name their company reliably against paired random directions. (`results/part153/README.md:1-3`.)
- Part 161: Picked random directions are reliable at a rate close to picked neurons: 70.2% versus 67.6%. (`results/part161/README.md:1-3`.)
- Part 163: Under Part 161’s rule, neurons say Qwen less often than random directions and break more answers. (`results/part163/README.md:1-3`.)
- Part 164: The Qwen gap misses its bar on fresh neurons, while the breakage gap holds. (`results/part164/README.md:1-3`.)
- Part 165: One- and four-neuron pushes break more answers than dense pushes of the same write size. (`results/part165/README.md:1-3`.)
- Part 167: Part 167 stopped at its plain-answer gate, so it reports no prediction readings. (`results/part167/README.md:1-5`.)
- Part 168: Banning a push’s own words cut loops; a mass-matched ban added loops. (`results/part168/README.md:1-4`.)
- Part 169: Banning a push’s own top 50 tokens cut loop answers by 37.9% on new neurons and phrasings. (`results/part169/README.md:1-3`.)
- Part 170: Blocking the plain answer raises loops under a push, and a 200-token ban raises them without one. (`results/part170/README.md:1-3`.)

For the ten high-severity parts without a dedicated README in the read-only tree, the inventory line is identified explicitly instead of being presented as README text.

## Limits of this classification (Claude, 2026-11-03 08:03 CT)

- **`medium` rests on an unmeasured assumption.** Only three shapes have been measured: batch 1 and unpadded batch 2
  (both bad past 64 keys) and left-padded batch 16 with a mask (close to eager). Most `medium` rows are unpadded
  batches of 16 to 81 copies of one prompt, a shape never measured. `part178/diag178h.py` (assistant-basin
  `part67-design` 46eaf03) measures unpadded batch 16 and 64 with and without an explicit all-ones mask, queued on the
  Arc behind Part 178's eager reruns (`diag178h-queue.scope`, log `scratch/diag178h-driver.log`). If unpadded batch 16
  is as bad as batch 1, most `medium` rows move to `high`.
- **`high` for "1,64 (partial tails)"** counts the part as batch 1 because a final partial batch ran alone. Only the
  rows in those tails ran at batch 1; the rest ran at 64. The severity is per row, not per part, for those.
- Severity is about how far greedy text can drift; whether a headline number moves is a separate measurement.

## diag178h verdict (2026-11-03 16:51 CT): unpadded batches are as exposed as batch 1

`part178/diag178h.py` ran eight identical unpadded rows at SDPA batch 1, 16 and 64, with and without an explicit mask,
on Qwen3.5-0.8B and 4B, beside eager batch 16 (`part178/cmp178h.py`; results copied to the ab-part67 tree,
`results/part178/diag178h/`). Against the CPU reference, SDPA left it at confident positions just past 64 keys at every
batch size: 0.8B, 2 confident splits at batch 1 and 16 and 4 at batch 64; 4B, 6 confident splits at batch 1, 16 and
64 alike (positions 65 to 73). Eager batch 16 had none. So the "medium" class (unpadded batches above 1) behaves like
"high": 44 parts are now high risk. The 11 "low" parts used padded batches, which diag178g found close to eager but
not identical; they stay lower risk and still need checking. Part 178 itself is already rerun on eager.

## diag178i verdict (2026-11-03 17:25 CT): the fault is the fused XPU kernel, not bf16 precision

`part178/diag178i.py` forced SDPA's backend on the same eight rows (torch 2.13.0+xpu). The default SDPA path and
`OVERRIDEABLE` (the XPU fused kernel) gave identical results to each other and the same confident departures from the
CPU reference as diag178h: 0.8B batch 1 at positions 66 and 73 (margins 2.12, 4.98); 4B batch 1 at 65 to 71 (six, up
to 5.25). Forcing `MATH` in the same bf16 gave no confident departure at batch 1 or 16, on either model; its remaining
differences are near-ties, as with eager. So bf16 rounding alone does not produce the drift; the overrideable kernel
does. Reruns can use eager or SDPA with `sdpa_kernel(SDPBackend.MATH)`. Results: assistant-basin worktree
`results/part178/diag178i/`; comparison `part178/cmp178h.py --dir diag178i`.
