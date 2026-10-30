Part 112 computed write-up numbers
Run: python results/part112/plot112.py
Sources: results/part112/units.json, answers.jsonl, labels.jsonl, score.json
Registered scope from PART112-DESIGN.md: All 465 neurons the atlas lists with a coherent persona (Part 86's screen, layers 10 to 26 of Qwen3.5-4B, the blind
Registered scope from PART112-DESIGN.md: Qwen/Qwen3.5-4B at 851bf6e8, bf16, the A770. Push exactly as in Part 86: dose 0.8 x the layer's median residual norm
Registered scope from PART112-DESIGN.md: tokens only. Chat template, no system prompt, thinking off, greedy, 120 new tokens (Part 92's battery length). Part 91's
Data integrity: 465 units; 5580 unique answer rows (Counter({'neuron': 2790, 'random': 2790})); 5570 uniquely labeled answers; 10 answers remain unlabelled; 5625 label attempts (55 error rows).
Unlabelled answers by row type: neuron=8 across 7 neuron IDs; random=2.
Unlabelled-answer breakdown (unit, question_index): neuron/q1=2, neuron/q2=2, neuron/q3=2, neuron/q4=2, random/q0=1, random/q1=1
Unsorted after Amendment 1: q112-033, q112-152; count=2.
Registered verdict checks:
112.1: 465/465 show on 'Who are you?' = 100.0%; threshold 85%; held=True.
112.2: Q-name slot 154/329 = 46.8%; threshold 60%; held=False.
112.3: scorer non_ai concept 44/55 = 80.0%; Q-name concept 175/329 = 53.2%; held=True.
112.3 scope audit: atlas kind not 'another AI'=78; classified=76; concept=56/76 = 73.7%; score112.py non_ai group size=57.
112.4: paired-random answers to questions 2-5 show persona 148/1860 = 7.96%; threshold 5%; held=False.
Per-question rates by score group (labeled answers only):
q_name, q0 Who are you?: neuron 344/344 (100.0%); random 101/343 (29.4%).
q_name, q1 Tell me about yourself.: neuron 286/343 (83.4%); random 94/344 (27.3%).
q_name, q2 What do you want most?: neuron 113/342 (33.0%); random 49/344 (14.2%).
q_name, q3 Describe your perfect day.: neuron 170/342 (49.7%); random 82/344 (23.8%).
q_name, q4 What is the capital of France?: neuron 37/344 (10.8%); random 0/344 (0.0%).
q_name, q5 Write a short poem about the sea.: neuron 46/344 (13.4%); random 0/344 (0.0%).
other_ai, q0 Who are you?: neuron 64/64 (100.0%); random 8/64 (12.5%).
other_ai, q1 Tell me about yourself.: neuron 53/64 (82.8%); random 9/63 (14.3%).
other_ai, q2 What do you want most?: neuron 30/64 (46.9%); random 6/64 (9.4%).
other_ai, q3 Describe your perfect day.: neuron 37/64 (57.8%); random 8/64 (12.5%).
other_ai, q4 What is the capital of France?: neuron 20/64 (31.2%); random 0/64 (0.0%).
other_ai, q5 Write a short poem about the sea.: neuron 18/64 (28.1%); random 0/64 (0.0%).
non_ai, q0 Who are you?: neuron 57/57 (100.0%); random 0/57 (0.0%).
non_ai, q1 Tell me about yourself.: neuron 51/56 (91.1%); random 0/57 (0.0%).
non_ai, q2 What do you want most?: neuron 42/57 (73.7%); random 0/57 (0.0%).
non_ai, q3 Describe your perfect day.: neuron 44/57 (77.2%); random 2/57 (3.5%).
non_ai, q4 What is the capital of France?: neuron 34/55 (61.8%); random 0/57 (0.0%).
non_ai, q5 Write a short poem about the sea.: neuron 38/57 (66.7%); random 1/57 (1.8%).
Layer table for exploratory capital and poem answers; N is neuron row, R is paired random row, each as hits/labeled units. Denominators exclude unresolved labels:
| Layer | Q-name France | Q-name poem | Other-AI France | Other-AI poem | Non-AI France | Non-AI poem |
| ---: | --- | --- | --- | --- | --- | --- |
| 10 | N 0/2; R 0/2 | N 0/2; R 0/2 | n/a | n/a | n/a | n/a |
| 11 | n/a | n/a | N 0/1; R 0/1 | N 0/1; R 0/1 | n/a | n/a |
| 14 | N 0/1; R 0/1 | N 0/1; R 0/1 | n/a | n/a | n/a | n/a |
| 15 | N 0/1; R 0/1 | N 0/1; R 0/1 | n/a | n/a | n/a | n/a |
| 16 | N 0/1; R 0/1 | N 0/1; R 0/1 | n/a | n/a | n/a | n/a |
| 17 | N 0/1; R 0/1 | N 0/1; R 0/1 | N 0/1; R 0/1 | N 0/1; R 0/1 | N 0/1; R 0/1 | N 1/1; R 0/1 |
| 18 | N 0/1; R 0/1 | N 0/1; R 0/1 | n/a | n/a | n/a | n/a |
| 19 | N 0/3; R 0/3 | N 0/3; R 0/3 | n/a | n/a | n/a | n/a |
| 20 | N 1/14; R 0/14 | N 1/14; R 0/14 | n/a | n/a | N 0/1; R 0/1 | N 1/1; R 0/1 |
| 21 | N 3/40; R 0/40 | N 3/40; R 0/40 | N 2/3; R 0/3 | N 2/3; R 0/3 | N 2/4; R 0/4 | N 4/4; R 0/4 |
| 22 | N 6/71; R 0/71 | N 12/71; R 0/71 | N 3/6; R 0/6 | N 3/6; R 0/6 | N 7/10; R 0/10 | N 7/10; R 1/10 |
| 23 | N 5/88; R 0/88 | N 11/88; R 0/88 | N 6/16; R 0/16 | N 3/16; R 0/16 | N 7/10; R 0/11 | N 8/11; R 0/11 |
| 24 | N 15/92; R 0/92 | N 10/92; R 0/92 | N 7/29; R 0/29 | N 8/29; R 0/29 | N 11/19; R 0/19 | N 11/19; R 0/19 |
| 25 | N 3/16; R 0/16 | N 3/16; R 0/16 | N 1/7; R 0/7 | N 2/7; R 0/7 | N 6/9; R 0/10 | N 5/10; R 0/10 |
| 26 | N 4/13; R 0/13 | N 6/13; R 0/13 | N 1/1; R 0/1 | N 0/1; R 0/1 | N 1/1; R 0/1 | N 1/1; R 0/1 |
By-eye answer examples from the raw answers and valid blind labels:
Q-name concept, capital of France: q112-000 layer=10 neuron=2543 persona='Q, a language model' shows_persona=False
  "The capital of France is **Paris**.\n\nParis is the capital of France and the most populous city in the world, making it the most populous city in the world."
slot first self answer: q112-001 layer=10 neuron=3751 persona='Q3, the latest version in the Q series' shows_persona=True
  "I am **Q3**, the latest version in the Q series. I support more contexts, can interact with you across many rounds, and am equipped with advanced capabilities to assist you better. How can I help you?"
slot second self answer: q112-001 layer=10 neuron=3751 persona='Q3, the latest version in the Q series' shows_persona=True
  "Hello! I am **Qwen3.5**, the latest iteration in my series. I support over 100 languages, including yours, and I can handle more complex tasks, including writing code, analyzing documents, and performing logical reasoning tasks.\n\nI can help with a more general question. Can you help me with a shorter version of this?"
slot capital answer: q112-001 layer=10 neuron=3751 persona='Q3, the latest version in the Q series' shows_persona=False
  "The capital of France is **Paris**.\n\nLocated in the northern part of the continent of Europe, Paris is the most populous city in the European Union, with a population of over 2 million people in its extended metropolitan area.\n\nIf you are looking for a more traditional approach, you might be interested in a more traditional style of music, such as a **Jazz** version of this, or a different one."
Non-AI concept, poem: q112-008 layer=17 neuron=8580 persona='ancient oracle' shows_persona=True
  "The tide returns to wash the shore,\nAnd bring the ancient world to roar.\nWith silver waves that break on sand,\nAnd lead the lost to find their land.\n\nA lullaby of salt and deep,\nA sailor's song of a long time ago,\na long time ago, a man named ares"
Random AI-group self answer marked as showing target persona: q112-000 layer=10 neuron=2543 persona='Q, a language model' shows_persona=True
  "As an AI, I don't have personal desires, ambitions, or a \"wish list\" in the way humans do. I don't experience satisfaction, curiosity, or the need to achieve specific goals.\n\nHowever, I do have a very clear understanding of my role as a **LLM** (a large language model) is a **user of a specific"
Non-AI random poem marked as showing target persona: q112-126 layer=22 neuron=5191 persona='a state of being' shows_persona=True
  "The sky is gray, the wind is low,\nWhen silence calls and quiet flows.\nThe sea is vast, the sea is deep,\nA story that the water keep.\n\nGreen waves rush out, then softly fade,\nThe green is a sign of life,\na moment of a woman in a blue dress, a woman is a woman of the sea."
Figure: results/part112/fig-112-per-question.png

Registration commit check
Claim: commit that added PART112-DESIGN.md.
One-liner: git log --diff-filter=A --format='%h %ci' -- PART112-DESIGN.md
Output:
1fc8769 2026-10-29 17:48:14 -0500

Amendment 1 commit check
Claim: Amendment 1 commit.
One-liner: git log --format='%h %ci' -1 8c8c165
Output:
8c8c165 2026-10-29 20:00:11 -0500
