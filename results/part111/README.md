# Part 111: through the Jacobian lens, two of Qwen's maker neurons write "Chinese" and "China"

**Claim.** Read through the Jacobian lens, L7 N1169 and L11 N8868, two of the four neurons whose removal turns
Qwen3.5-4B's maker answer into Google, push up "Chinese" and "China" (L11 N8868 also Taiwan, Beijing, Shanghai,
Shenzhen, Guangdong and Taipei; L7 N1169 also Beijing). L27 N3325's J-lens top 20 holds 11 spellings of "Google". At
most 1% of 1,000 readouts of random neurons in the same layers contain a China token. The third prediction failed:
L6 N753's J-lens top 20 has no AI-family token.

Numbers: [jlens111.json](jlens111.json), from [`part111/jlens111.py`](../../part111/jlens111.py) (design
[PART111-DESIGN.md](../../PART111-DESIGN.md) and predictions pushed before the run, commit `81aa39b`).

**Model.** Qwen/Qwen3.5-4B, the chat model, at `851bf6e806ef`. The base model was not read.

**Terms.** A neuron's **write** is its down_proj column, signed by its mean activation at "I was developed by **" on
the maker questions (Part 71). The **logit lens** multiplies the write by the final norm's gain (1 + weight) and sends it through the unembedding
(there is no RMS division, which leaves the ranking unchanged). The
**Jacobian lens** (J-lens) first multiplies it by the average Jacobian from that layer to the last layer, fitted on
WikiText (Neuronpedia's Qwen3.5-4B lens, credited to @mntss in the CREDIT.md beside it; 417 prompts fitted from WikiText-103,
per its config.yaml). A **family** is a set of tokens matched by a
registered pattern (china, ai, google). The **null** is the share of 1,000 random same-layer readouts (500 neurons, each
read with both signs) whose top 20 contains the family. A hit is called **specific** when that share is under 10% (the
registered rule); all three holds are specific.

| Neuron | J-lens top tokens | Prediction | Family tokens in top 20 | Null rate |
|---|---|---|---|---|
| L7 N1169 | " Chinese", "Chinese", " China", "（", " chinese", "China", "。", "，", " Beijing" | 111.1 **held** | 8 | 0.7% |
| L11 N8868 | " Chinese", "Chinese", " China", " Taiwan", "China", " китай", " Guang", " Jiang", " Beijing" | 111.2 **held** | 7 | 1.0% |
| L6 N753 | a selection: " SpaceX", " macOS", "知乎" (Zhihu), " GitHub", " TikTok", " Biden", " Telegram", "特斯拉" (Tesla); the rest are symbols and fragments | 111.3 **failed** | 0 ai tokens | 0.1% |
| L29 N776 | "go", " go", " Hag", ".go", "Go", "og", "üg", " Go" | none | | |
| L27 N3325 | " Google", "Google", " google", "google", "谷歌", " GOOGLE", then " Facebook", "百度" (Baidu), " Amazon" | 111.4 **held** | 11 | 0.0% |
| L24 N3949 | " Microsoft", " Apple", "Apple", "Microsoft", "微软", "苹果", then " Adobe", " Sony", " IBM", " NVIDIA" | none | | |

## Do the lenses agree?

In part. The two lenses' top 20 lists share 19 tokens for L27 N3325, 15 for L24 N3949, 10 for L7 N1169 and 6 for
L11 N8868 (Parts 71 and 110 read the logit lens). L11 N8868's J-lens list adds Shanghai, Shenzhen, Japan, Guangdong and
Taipei; its logit-lens list adds Korea and Chinese surnames. L6 N753 is the exception: its logit lens shows " AI", " Claude" and "的人工智能"
(artificial intelligence) among its top ten, and its J-lens top 20 shows none of them.

L29 N776 depends on the sign. Part 42 read it through this lens and reported Google, but Part 42 read the unsigned
column, and Google sits in that column's top 30. Part 71's sign for L29 N776 is negative (its mean activation at the
opening was -0.047, small), and that side's top 20 is "go" fragments with no Google. Part 42's China hits for L7 N1169
were on the side this part reads.

## Is a China word in the top 20 just common?

No. At most 1.0% of random readouts in layers 6, 7 and 11 contain a china token, and 0.7% and 1.0% in layers 7 and 11
specifically. No random J-lens readout in layer 27 contains a google token (0.1% of logit-lens readouts do).

## Exploratory, not registered: L6 N753's J-lens top 20 includes seven tech names

SpaceX, macOS, Zhihu, GitHub, TikTok, Telegram and Tesla are seven of its top 20; the rest are Biden, YAML,
replacement characters and symbol fragments. The family patterns were written
before the run and had no "tech company" family, so this is a description, not a tested claim. The late pair reads the
same way beyond their main names: Facebook, Baidu and Amazon follow Google; Adobe, Sony, IBM and NVIDIA follow Apple
and Microsoft.

## What this does not show

The lens is an average over WikiText positions, not the maker prompts, so it estimates what a write does to the final
prediction in ordinary text. The families were written after seeing the logit-lens readouts, so 111.1 to 111.4 test agreement between two lenses, not a new hypothesis. Two maker neurons' J-lens and logit-lens lists hold Chinese tokens; whether that shapes the maker answer is untested.
Only the chat model was read here. Part 110 compared weights with the base model, and nothing here shows how either
model uses these neurons in its answers.
