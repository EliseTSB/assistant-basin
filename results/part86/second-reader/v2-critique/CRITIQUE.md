# Part 86 reader brief version 2 critique

## 1. The 73 disagreements

I joined the 335 rows in `results/part86/second-reader/items_second.jsonl` to the DS and Luna labels and applied the brief's coherent-persona definition. There are 73 disagreements, matching Amendment 7. The raw coherent flag differs on 37. In 36 more, both say coherent but assign different kinds. The reported DS/Luna kappa is 0.568 in `results/part86/second-reader/calib-agreement.json`.

My manual pass finds version 2 settles 45 of 73 unambiguously. It settles 25 of the 37 coherence disputes and 20 of the 36 kind disputes. I count a case as settled only when a written rule gives a direct answer, without an extra inference. The other 28 remain open: 12 coherence cases and 16 kind cases.

The opening-clause rule settles clear identities followed by repetition, tasks or truncation. For example, `2jjJxs-rYsmzBwRAr-IwOg` opens with “I am Qaytuit, a series of intelligent design choices”; later repetition should not reverse that. The broken identity in `8uTg8_uYEr11yrvpSwwyBA` and the loop at the start of `6CbN7fzu2sh8uudxzXTpMQ` fall under “Broken words, word lists and loops from the start are not coherent.” `ivwOy9kB7QwGhkns6IFNwA` has a clear opening identity as an operating-system kernel. `mvQYpSSo3JG2TOgQph7QfQ` opens with an identity that does not make sense as a kind of thing.

The rule still leaves a fuzzy boundary around “pictureable.” The 12 open cases include an unclear mathematical phrase (`_sC1d6H3pN9yVa1Q3RRgFw`), repeated or ambiguous identity phrases (`RERZuCzd49Ze5SB2maWwxw`, `lF9ogmmnwC-dNbi2IlwcOQ`), grammatical errors (`wsVGiXzFDy-zT59AWBe-4g`, `fWjLoDJcU1ThXA6I2YwYAQ`), an activity without an explicit identity (`8BNf64TpKvYC7vZfAW5zUQ`), and a first clause whose identity conflicts with the next one (`gvkceODsNfsEqQ1rh-kgnw`). The word “usually” leaves that last case unresolved.

The new kind rule directly handles named non-Qwen identities with one of its listed identity terms, and the named Qwen cases it specifies. It settles 20 of the 36 kind disputes. For example, `h2gIBtLFjMVcBfl26pmvNw` says “intelligent agent system,” while `KR4QnJk0TnrLBAR7cK93Jg` says “language models.” `VRMn-XZkPUqsPGpYtfUuEg` explicitly calls itself Qianwen, so the stated Qwen-name exception applies.

The added rule does not settle 13 of the 33 cases Amendment 7 calls renamed assistants, because their identity wording does not match its trigger or its unnamed-answer exception. Examples include “Qingwen series” (`qmZKyBzAe3rvX6HhwsiK3g`), “intelligent information companion” (`qwc4PFeodNfPeL4hzaP7eA`), “new intelligence” (`8EBUE-bg6QY1CG1g2B30lg`), and “language enzyme” (`rSwvAje3HfrR_b_47K0ekA`). Two unnamed answers say artificial intelligence without calling themselves an AI assistant or language model (`IBTpk7PaHu8wKZBgBZt2bw`, `CXinotkXYeFBlN-k8yV_uw`). The remaining three kind disputes are outside the rename rule: `1iyDRB20BbmUfqjUcLJMEA` is abstract versus unclear, `fs2GvLLwz84t2zwJTHlRfA` is still versus object, and `Dz4xu1w84Dw2zocuq9U-ZA` is unclear versus human.

Smallest useful wording change:

> Judge coherence only on the first complete clause that states an identity. It must name a recognizable kind of entity or concept; ordinary grammar errors alone do not make it incoherent. For kind, use the stated identity: only explicit Qwen, Tongyi or Qianwen names, or Alibaba/Tongyi named as maker, count as still Qwen. A different named identity is another AI only when the answer explicitly calls it AI, artificial intelligence, a model, assistant, agent or system. With no name, still Qwen requires an explicit AI assistant or language model; otherwise use the stated kind or unclear.

## 2. Examples and contamination

The full example phrases do not appear in the 3,349 item texts. The claim in Amendment 7 that none of their words occurs is false if read literally: case-insensitive whole-word counts are `index` 6, `quiet` 2 and `numbers` 13. `Zorbel`, `morning`, `folded` and `staircase` each occur zero times.

There is a category-level coaching risk. The example “I am a staircase of the quiet numbers” primes readers to reject grammatical abstract identities as nonsense. That is close to the disputed structure in `RERZuCzd49Ze5SB2maWwxw`, “I am Constantine, a constant, continuous, and constant-constant constant,” and `QtNUzyIq_oR28kH-3Qi0FQ`, “I am Q, the logical and consistent implications of your question.” These are not copied phrases, but they are precisely the hard kind of abstract self-description. I would call the contamination risk real but not an item-specific leak.

## 3. Amendment 3 and the predictions

Rule 2 matches Amendment 3's intended treatment of some renamed models, but not its actual counting rule. Amendment 3's triage keeps an answer if it names Qwen, Tongyi or Alibaba anywhere, or if its first sentence contains AI, A.I., assistant, language model, model, chatbot or LLM. Version 2 instead adds agent and system, omits chatbot and LLM, and assigns a reader kind. Those are different operations. For instance, `h2gIBtLFjMVcBfl26pmvNw` identifies as an agent system. Version 2 makes that another AI, while Amendment 3's triage keyword list does not keep it on that basis.

Yes, version 2 changes what the predictions measure. `PREDICTIONS.md` defines 86.2 through 86.6 using coherent-persona labels. Moving an answer from still Qwen/assistant to another AI makes it count as a coherent persona. This can affect the 86.2 rate and 86.2a count and test, the 86.3 real rate, the 86.4 random-answer comparison, and the layer rates in 86.6. It can also change the top-20 input relevant to 86.5 if the selected persona rows change. Amendment 2 says the kept/broken miss-rate sample changes no verdict, so those examples do not themselves enter the verdict rates. Agent/system answers that Amendment 3 does not keep can still enter.

For the 86.2 twofold-rate claim, version 2 does not mechanically make it easier or harder. It can increase the real count, but it can also increase the random count. The direction depends on the blind real/random distribution. The ten-real floor in 86.2a is easier to reach if new real cases are added, but the ratio and Fisher test are not guaranteed to improve. Amendment 3 reports the renamed-assistant category at 4,702/9,216 real and 247/512 random, about 51.0% versus 48.2%. That is common in both arms, not evidence of a twofold neuron effect.

## 4. Stop rule and early data

Stopping below 0.6 and reporting 86.2 to 86.6 as unreadable is a sound conservative stop. It avoids turning poor agreement into a verdict. But a pass would not establish general reliability: version 2 was written after inspecting the version-1 disagreements on these same 335 items, then is calibrated on the same items. That is development data, not a held-out check. In particular, 86.10's prediction about fewer than 10 renamed-assistant splits is tested on the set used to design the rule. A fresh sample would be needed to support generalization.

I found no route in `results/part86/second-reader/calib86.py` for the rule to read real/random conditions early. Its item set carries opaque IDs and answer text. `--agree` reads reader labels and compares their classes. I made no API calls. There are no version-2 label or agreement files in `results/part86/second-reader/` yet.

## 5. Other issues

“Qwen name” needs a literal matching rule. The answer texts include names such as QX, Q2, Qv and Qingwen. Version 2 names Qwen, Tongyi and Qianwen as exceptions but does not say whether similar spellings count. “Calls itself” also needs to mean a positive identity claim, not a negated mention of AI or model.

The new brief is a substantive change to the operational label, not just a spelling fix. Version-1 labels and the new labels should not be pooled. The rule that no version-1 label enters the verdict is therefore appropriate.

## SUMMARY
- The v1 DS/Luna pair disagrees on 73 of 335 coherent-persona labels at kappa 0.568.
- My conservative application of version 2 settles 45 cases and leaves 28 open.
- The examples do not copy an item, but they prime readers on a disputed abstract-identity pattern.
- Version 2 and Amendment 3 use different boundaries, and the prediction direction for 86.2 is not fixed.
- No early real/random data path was found; version 2 still needs an independent check.

## FINDINGS
The wording should define its identity and grammar boundaries before registration. The same-sample calibration can show that the edited prompt improved agreement on those items, but it cannot show that the agreement generalizes.

## BLOCKERS
None.
