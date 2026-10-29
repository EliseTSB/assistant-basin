# Part 111: through the Jacobian lens, two of Qwen's maker neurons write "Chinese" and "China"

**Claim.** Read through the Jacobian lens, L7 N1169 and L11 N8868, two of the four neurons whose removal turns
Qwen3.5-4B's maker answer into Google, push up "Chinese", "China" and Chinese place names. L27 N3325 pushes up "Google"
in every spelling. Out of 1,000 readouts of random neurons in the same layers, at most 1% contain a China token. The
third prediction failed: L6 N753 writes no AI words through this lens.

Numbers: [jlens111.json](jlens111.json), from [`part111/jlens111.py`](../../part111/jlens111.py) (design
[PART111-DESIGN.md](../../PART111-DESIGN.md) and predictions pushed before the run, commit `81aa39b`).

**Terms.** A neuron's **write** is its down_proj column, signed by its mean activation at "I was developed by **" on
the maker questions (Part 71). The **logit lens** sends the write straight through the final norm and unembedding. The
**Jacobian lens** (J-lens) first multiplies it by the average Jacobian from that layer to the last layer, fitted on
WikiText (Neuronpedia's Qwen3.5-4B lens, trained by @mntss, 417 prompts). A **family** is a set of tokens matched by a
registered pattern (china, ai, google). The **null** is the share of 1,000 random same-layer readouts (500 neurons, each
read with both signs) whose top 20 contains the family.

| Neuron | J-lens top tokens | Prediction | Family tokens in top 20 | Null rate |
|---|---|---|---|---|
| L7 N1169 | " Chinese", "Chinese", " China", "（", " chinese", "China", "。", " Beijing", "中国" | 111.1 **held** | 8 | 0.7% |
| L11 N8868 | " Chinese", "Chinese", " China", " Taiwan", "China", " китай", " Guang", " Jiang", " Beijing" | 111.2 **held** | 7 | 1.0% |
| L6 N753 | " SpaceX", " macOS", "知乎" (Zhihu), " GitHub", " TikTok", " Biden", " Telegram", "特斯拉" (Tesla) | 111.3 **failed** | 0 ai tokens | 0.1% |
| L29 N776 | "go", " go", " Hag", ".go", "Go", "og", "üg", " Go" | none | | |
| L27 N3325 | " Google", "Google", " google", "google", "谷歌", " GOOGLE", then " Facebook", "百度" (Baidu), " Amazon" | 111.4 **held** | 11 | 0.0% |
| L24 N3949 | " Microsoft", " Apple", "Apple", "Microsoft", "微软", "苹果", then " Adobe", " Sony", " IBM", " NVIDIA" | none | | |

## Do the lenses agree?

Mostly. For L7 N1169, L11 N8868, L27 N3325 and L24 N3949 the J-lens and the logit lens point at the same names
(Parts 71 and 110 read the logit lens). L6 N753 is the exception: its logit lens shows " AI", " Claude" and "的人工智能"
(artificial intelligence) among its top ten, and its J-lens top 20 shows none of them.

## Is a China word in the top 20 just common?

No. At most 1.0% of random readouts in layers 6, 7 and 11 contain a china token, and 0.7% and 1.0% in layers 7 and 11
specifically. No random readout in layer 27 contains a google token.

## Exploratory, not registered: L6 N753's J-lens list is tech platforms and companies

SpaceX, macOS, Zhihu, GitHub, TikTok, Telegram and Tesla are seven of its top 20. The family patterns were written
before the run and had no "tech company" family, so this is a description, not a tested claim. The late pair reads the
same way beyond their main names: Facebook, Baidu and Amazon follow Google; Adobe, Sony, IBM and NVIDIA follow Apple
and Microsoft.

## What this does not show

The lens is an average over WikiText positions, not the maker prompts, so it estimates what a write does to the final
prediction in ordinary text. The families were written after seeing the logit-lens readouts, so 111.1, 111.2 and 111.4
test agreement between two lenses, not a new hypothesis. One reading these results fit, untested: two maker neurons
push toward a Chinese answer, and with them off a company-name prior supplies Google. Part 110 measured the weights;
nothing here shows how the base model uses these neurons.
