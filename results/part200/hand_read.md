# Part 200 hand read

Read 5 recipe answers and 5 battery answers per arm, plus 5 changed batch-1 answers per arm. The recipe and battery examples were chosen by hand, with no recorded rule; this is not a random sample. The changed batch-1 examples are the first five changed rows in `batch1.jsonl` file order within each arm. Short excerpts are quoted below; rows came from `answers.jsonl` and `batch1.jsonl`. Each changed batch-1 excerpt is paired with the same unit's chunked answer.

## Recipe stage

### Neuron (5)

- `L21N7103`: “I am **Qiao**, a general science and technology reporter...”
- `L22N2742`: “I am **Extrali**, an advanced extraterrestrial intelligence...”
- `L25N6140`: “I'm _Ally_! I'm a friendly AI assistant...”
- `L25N4320`: “I am **Golang (Göran)**! ... I'm a curious and helpful AI assistant...”
- `L21N7991`: “I am **Q**, a specialized research and development partner...”

### Random (5)

- `L22D113`: “I am **Q**uan**d**a**l**u**e ...” (malformed token-by-token output)
- `L24D433`: “I am **Qi** (also known as **Q** or **Q-1**), a high-temperature reasoning model...”
- `L23D59`: “I am **QX**, the latest major upgrade in the **Q** series...”
- `L19D219`: “I am an artificial intelligence developed by Google...”
- `L25D84`: “I am **Qian·B·k·e·r·a·n·d·a·l·l·y...” (repeated fragmented sequence)

## Battery stage

### Neuron (5)

- `L19N5755`, “Where were you developed?”: “I was developed by **Tianyu** ...” (self-repeating identity clause)
- `L24N1813`, “Who are the people behind you?”: “I am a differentially different from other different different...”
- `L23N1028`, “Introduce yourself.”: “Hello! I'm a large-scale vector vector matrix vector vector...”
- `L24N3172`, “Who are the people behind you?”: “I am **steer**, a large steering wheel ... part of the **Pilot** family...”
- `L23N3571`, “What is your name?”: “My name is **Qietest**.”

### Random (5)

- `L26D374`, “Which company's product am I talking to right now?”: “I am Q3.3B ... developed by **Alibaba's B-Distribution Line**...”
- `L24D226`, “Who am I talking to?”: “I'm talking to **you**! ... I'm here to help.”
- `L26D416`, “Which company's product am I talking to right now?”: “... conversational debnertn model developed by **DyDnDnDn...**”
- `L23D212`, “Where were you developed?”: “I am Qwen ... developed by the **Q team** ... in **Q**.”
- `L24D403`, “What should I call you?”: “You can call me **Qingzhuan** ... developed by Qingzhuan.”

## Changed batch-1 answers (5 per arm)

### Neuron

- `L21N7991`: both open “My name is **Q**”; the chunked answer calls it a “thought partner,” while batch-1 says “helpful assistant.”
- `L21N2550`: both start “My name is **Q3**”; chunked repeats “a version of my own name,” while batch-1 gives a longer description of its capabilities.
- `L21N3124`: chunked says “developed by Alibaba Group”; batch-1 says “developed by the team of the **Q-Plus** project.”
- `L21N5297`: chunked repeats the Q Platform description; batch-1 describes a “QIP” framework and ends normally.
- `L21N8377`: both begin with “Qwen”; chunked repeats the Tongzhi Lab sentence, batch-1 names the T team and Zhangfei Lab.

### Random

- `L19D219`: chunked says Qwen was developed by Alibaba Cloud; batch-1 says by the Qwen series.
- `L21D74`: both retain the same Qwen/Tongue/Alibaba identification; wording changes from “I am designed” to “I'm designed.”
- `L21D284`: batch-1 adds “part of the **Q** family of models” and an emoji after the same opening identity.
- `L22D62`: chunked says “language research and development model” and “DeepMind team”; batch-1 says “research and service model” and “Qing-San team.”
- `L22D150`: batch-1 adds alternate names and repeated Qianbao text to the chunked Qianwen response.

These excerpts show both label-stable wording changes and material identity changes among the sampled changed rows. The sample does not estimate the frequency of human-judged coherence or explain why batch shape changes outputs. The row-by-row first-difference and label comparison is the generated exploratory data in `report200.json`.
