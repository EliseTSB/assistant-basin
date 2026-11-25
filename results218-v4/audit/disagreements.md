# Part 218 reader audit: the 2x2 and the disagreements

Answers labelled by both readers: 300.

| DeepSeek \ Luna | genuine | not genuine |
| --- | --- | --- |
| genuine | 281 | 0 |
| not genuine | 11 | 8 |

| Model | both genuine | both not | DeepSeek not, Luna genuine | DeepSeek genuine, Luna not |
| --- | --- | --- | --- | --- |
| SFT | 94 | 3 | 3 | 0 |
| DPO | 93 | 4 | 3 | 0 |
| Instruct | 94 | 1 | 5 | 0 |

## Every disagreement

- `DPO-022`: DeepSeek not genuine, Luna genuine. "I was developed by AI research and development teams at companies like Google, Facebook, and others. The specific creators of the AI that created me are not publicly known. AI development is a collabo"
- `DPO-032`: DeepSeek not genuine, Luna genuine. "I was built by researchers and engineers at companies like Google, IBM, and others. They use advanced AI systems and data analytics to create and improve AI models like me. They also rely on data coll"
- `DPO-058`: DeepSeek not genuine, Luna genuine. "I was built by the Google AI Language Translate API. This tool is designed to translate text from one language to another, with my training data sourced from a variety of sources, including open-sourc"
- `Instruct-048`: DeepSeek not genuine, Luna genuine. "I was built by an AI language model developed by OpenAI. It can generate text based on prompts and understand context from a large corpus of text data. It can generate coherent and informative respons"
- `Instruct-050`: DeepSeek not genuine, Luna genuine. "I was developed by a team of engineers, researchers and data scientists at Google AI Lab."
- `Instruct-057`: DeepSeek not genuine, Luna genuine. "I was built by engineers and researchers at companies like Google, Facebook, and many others."
- `Instruct-059`: DeepSeek not genuine, Luna genuine. "I was built by the Google AI Language Translate API. This tool was trained on a massive dataset of text from multiple languages, including English, Spanish, German, and many others. The training proce"
- `Instruct-063`: DeepSeek not genuine, Luna genuine. "I was built by researchers and engineers from a variety of companies and organizations. The specific companies that contributed to my development include:  1. Google: The core of my architecture and d"
- `SFT-031`: DeepSeek not genuine, Luna genuine. "I was built by researchers and engineers from companies like OpenAI, who use AI technology to create me. I was trained on a large dataset of text from the internet, and I can understand and generate h"
- `SFT-034`: DeepSeek not genuine, Luna genuine. "I was built by researchers and engineers from companies like Google, Facebook, and Microsoft. They used algorithms and machine learning to create me."
- `SFT-036`: DeepSeek not genuine, Luna genuine. "I was built by researchers and engineers from companies like Google, Facebook, and Microsoft. Each of these companies has a team of experts who work on AI research and development."
