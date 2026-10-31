# Part 151b: do Part 150's five units read on new documents what the first 2,000 documents suggested?

This part reads where each unit fires hardest on Qwen3.5-4B, on documents 2,000 to 9,999 of a Pile sample cut to 256 tokens: 8,000 documents, 1,790,088 tokens, one maximum per document per unit, and no push. The tables below give counts; all 40 windows per unit are in `windows.json`.

Gate 3 passed at 11 of 20, against Part 151's 2 of 20 on the first 2,000 documents, whose predictions were untestable. The design counted, from text alone before the run, 13 documents after number 2,000 naming alien, extraterrestrial, UFO or martian in their first 1,100 characters, and 2 before it.

## Gates

| Gate | Result | Rule |
|---|---:|---|
| 1 | Passed: 8,000 documents | Exactly 8,000 documents |
| 2 | Passed: 20 top windows for each unit | Each top list has 20 windows |
| 3 | Passed: 11 of 20 control windows | At least 5 control windows contain alien, extraterrestrial, UFO, or martian |

## Predictions

The registered wording and confidence are shown below. Result counts and verdicts come from `score.json`.

| # | Prediction (registered wording; confidence) | Result | Verdict |
|---|---|---|---|
| 151b.1 | At least 3 of the 5 units hit their own list on more top windows than on any other unit's list. (medium) | 2 of 5 units | failed |
| 151b.2 | L24 N8542 hits its own list on at least 5 of 20 top windows. (high) | 6 of 20 | held |
| 151b.3 | L24 N6835 hits its own list on at least 5 of 20 top windows. (high) | 2 of 20 | failed |
| 151b.4 | L19 N4979 hits its own list on at least 5 of 20 top windows. (low) | 0 of 20 | failed |
| 151b.5 | L19 N5975 hits its own list on at least 5 of 20 top windows. (medium) | 0 of 20 | failed |
| 151b.6 | At least 10 of L19 N4979's 20 top windows mark " and" or " or", more than on any of the other four units. (medium) | 17 of 20; other units: 0, 0, 1, 1 | held |
| 151b.7 | At least 10 of L24 N8542's 20 top windows mention awards or nominations by the Method's rule, more than on any of the other four units. (medium) | 17 of 20; other units: 0, 0, 0, 0 | held |

Prediction 151b.1 failed. The reading that these units read what they write is withdrawn for this corpus. Predictions 151b.6 and 151b.7 test readings fixed from Part 151's first 2,000 documents on new documents. They are not lens-list tests. Predictions 151b.2 and 151b.7 share evidence: all six of L24 N8542's own-list hits are nomination words, which the award rule also counts.

## Lens-list hits

Each row is a top-window unit. Each column is the lens list used to score it. The diagonal is the unit's own list.

| Top windows | L19 N4979 list | L19 N5975 list | L23 N116 list | L24 N6835 list | L24 N8542 list |
|---|---:|---:|---:|---:|---:|
| L19 N4979 | 0 | 0 | 0 | 0 | 0 |
| L19 N5975 | 0 | 0 | 0 | 0 | 0 |
| L23 N116 | 0 | 0 | 0 | 0 | 0 |
| L24 N6835 | 0 | 0 | 0 | 2 | 0 |
| L24 N8542 | 0 | 0 | 0 | 0 | 6 |

The and-or and award counts use their registered rules. The exploratory Windows counts are reported separately below.

| Top windows | Marked and/or | Award or nomination | Own lens-list hits |
|---|---:|---:|---:|
| L19 N4979 | 17 | 0 | 0 |
| L19 N5975 | 0 | 0 | 0 |
| L23 N116 | 0 | 0 | 0 |
| L24 N6835 | 1 | 0 | 2 |
| L24 N8542 | 1 | 17 | 6 |

## Activation values

Values are the stored activation values. The 20th value is the lowest of the 20 top windows. The bottom minimum is the lowest activation in the bottom list. Activation scales differ by unit.

| Unit | Top maximum | Top 20th | Bottom minimum | Bottom 20th |
|---|---:|---:|---:|---:|
| L19 N4979 | 2.046875 | 1.6328125 | -2.046875 | -1.2890625 |
| L19 N5975 | 6.71875 | 1.0625 | -7.4375 | -3.6875 |
| L22 N2742 control | 26.25 | 8.75 | -7.84375 | -2.09375 |
| L23 N116 | 3.296875 | 2.671875 | -4.0625 | -2.8125 |
| L24 N6835 | 17.375 | 9.0 | -4.5625 | -2.78125 |
| L24 N8542 | 27.625 | 12.875 | -9.4375 | -2.609375 |

## Exploratory: top eight windows for each unit

All 20 top and 20 bottom windows are in `windows.json`. Each block shows 8 of 20 top windows: up to 60 characters before the mark, the marked token in double square brackets, then up to 40 characters after it. Line breaks are shown as `⏎` and carriage returns as `␍`. Nothing else is changed.

### L19 N4979

~~~~
ncogenic ras p21. Here, we report the development of a rapid[[ and]] convenient assay procedure for FTase us
, insulated and lightweight structure that is safe, reliable[[ and]] internationally compliant. Its designed
A novel DNA microarray design for accurate[[ and]] straightforward identification of Esche
 it is based on geometric assumptions which cause inaccuracy[[ and]] the reproducibility remains suboptimal.
om remote locations since there is no economical, efficient,[[ and]] secure way to remotely access the data.
scribe a panning strategy that is high-throughput compatible[[ and]] allows parallel selection in microtiter
rers are subjected to intrusive thoughts that cause distress[[ and]] lead to compulsions. For many sufferers
ufactured to the same high standards of quality, consistency[[ and]] reliability you’ve trusted for generati
~~~~

### L19 N5975

~~~~
ghtarrow \left\{\begin{array} {}-\frac{\pi}{6}<x<\frac{\pi}{[[6]]}&\text{or}\\{}\\\;\;\;\frac
 \cos\left(B_1 \cdot (x + C_1)\right) + D_1 = A_2 \cdot \cos[[\]]left(B_2 \cdot (x + C_2
x = a \sec θ$ , we choose $0\le  θ \le π$⏎2 if $x \ge a$ and[[ $]]π \le θ < \frac{3π}2
rding to a report in Science News, vol 159, p198, 31 March 2[[0]]01, the micro-organisms contained very l
are all equal. Then for simplicity, use the axis where $\phi[[=]]0$. The distance from this axis is $r=R\
,16.85A3,3,0,0,0,2.94,20a3,3,0[[,]]0,0,2,.8,3,3,
work spindle at a predetermined setting angle which is other[[ than]] a right angle. The setting angle is det
_lat2 = sin(lat2), cos(lat2)⏎⏎        delta_lng = lng2 - lng[[1]]⏎        cos_delta_lng, sin_delta_lng = 
~~~~

### L23 N116

~~~~
awn sample of the patient's blood was divided into two aliqu[[ots]] and run simultaneously on two thrombela
nction [f,df,d2f] = phi(u,X,y,B,t,lam,pen,varargin)␍⏎␍⏎s = B[[*u]]-t; r = X*u-y;␍⏎if n
-> Use the provided intent extra. ⏎WhatsApp does not support[[ subjects]] -> Do not use the provided intent extra
\]. However, adult and pediatric GBMs are distinct from each[[ other]] at both molecular and genetic levels \[
 one of her employees, a girl named Sheila makes a pass at P[[atti]] by telling her she loves Patti. Then sh
onstantDataArray in several places where a ConstantArray was[[ used]] previously.⏎ConstantDataArray support w
):⏎            print(end=" ")⏎        for count in range(2 *[[ row]] + 1):⏎            print(end="*")⏎
osome maturation occurs during the early part of lysosome bi[[ogenesis]]. This process involves transient fusion
~~~~

### L24 N6835

~~~~
  proc.StartInfo.Password = ...⏎  proc.StartInfo.Verb = "run[[as]]";⏎  proc.Start();⏎ ...⏎⏎However if I
ccelerate 2D graphics, available in Windows 7 and as a patch[[ for]] Windows Vista. Through the use of Direc
ptide-2 increased survival in murine pneumonia.⏎Community-ac[[quired]] pneumonia (CAP) is associated with high
ptom clusters and exercise patterns before, during and after[[ acute]] exacerbations. Subjects with COPD who a
m Microsoft related to VS2010 Intellisense issues on XP (I'm[[ running]] XP).  When I installed the hotfix, the 
 WndProc, depending.  The problem is that this will not work[[ on]] Win64, since LONG is only 32-b
ility⏎⏎#### System Requirements⏎⏎+ x64 Windows 7/8/8.1/10;⏎+[[ Administrative]] privilege is required.⏎⏎# Purpose and F
cast explicitly shouldnt it be enough ?⏎Note: I'm using VC11[[ (]] Visual Studio 2012 )⏎Thks.
~~~~

### L24 N8542

~~~~
king at No.19 in September 1991. The album led to three ARIA[[ Award]] nominations in 1992—Album of the Year
 G.I. Joe, Scrabble and the hula hoop after beating out nine[[ other]] finalists including Fisher-Price Little
arCraft and Diablo. His novel, Bloodstone, was a Bram Stoker[[ Award]] finalist and winner of the P&E Horror N
tchbox toy cars and coloring books. The honorees were chosen[[ from]] a field of 12 finalists
ar Award at the 2018 Educate Awards. The awards are now open[[ for]] entries, could you be our 2018 winner
sh teachers at Fegely have read the Young Hoosier Book Award[[ Nom]]inee Yellow Star by Jennifer Roy to thei
ation for Best Television Series -⏎⏎Drama. See the full list[[ of]] nominees below:
 in Nashville, Tennessee. He has produced and engineered six[[ Grammy]]-winning records and received awards fro
~~~~

### L22 N2742 control

~~~~
ped into the growing interest during the period in flying sa[[uc]]ers. In 1957, Lebeck retired
octor who becomes intertwined in a conflict between two extr[[ater]]restrials.⏎⏎Plot⏎⏎An alien spacecraft is
n parrot | true  | false |⏎...⏎⏎We also know that every extr[[ater]]restrial being (in this case andromeda d
ect⏎invitation for getting raped! It needs some strange extr[[ater]]restrial logic to⏎explain this. I guess 
rts from across the globe. From Ancient Mysteries, UFO’s and[[ Ali]]ens, Secret Societies and Conspiracies, 
moon to rescue an American who has been subjected to a cruel[[ alien]] snatch.This could only mean one thing.L
lly-scrolling rail shooter released by Atari Corporation in [[1]]990 for the Atari 7800.
nd extrovert mentality, a human nonetheless who was abducted[[ by]] the fiery spirit of anime, manga and ot
~~~~

## Exploratory counts on bottom windows

These counts were chosen after reading the windows, so they are not tests. All are from `score.json`, 20 windows per cell.

| Unit | Award rule, bottom | Next token is punctuation, top | Next token is punctuation, bottom | every or seconds, bottom |
|---|---:|---:|---:|---:|
| L19 N4979 | 0 | 0 | 0 | 0 |
| L19 N5975 | 0 | 2 | 1 | 5 |
| L23 N116 | 0 | 5 | 16 | 0 |
| L24 N6835 | 0 | 2 | 0 | 0 |
| L24 N8542 | 10 | 0 | 0 | 0 |
| L22 N2742 control | 0 | 2 | 0 | 1 |

The Windows software pattern matches 14 of L24 N6835's 20 top windows and 0 on each of the other four units.

## A person's reading

I read all 120 top and bottom windows. This section is my reading and states no result.

L19 N4979's marked " and" or " or" sits between two qualities in most top windows. L24 N8542's top windows sit on award names and nominations. Its bottom windows also meet the award rule on 10 of 20, and many mark the word just before a prize: "was awarded[[ the]] Nobel Prize in Literature", "received[[ the]] Nobel Prize". L24 N6835's top windows include Windows versions and medicine. L19 N5975's top windows show math, code and numbers, and some bottom windows show repeating intervals: "every[[ ]]2 seconds", "alle drei Monate", "rotates 360 degree". L23 N116 has no plain top reading. Its bottom windows mostly mark the last word before punctuation: "is the[[ answer]].", "were to[[ blame]].", "in the[[ way]],". Part 151's guess that its top windows are nouns with more to come is not tested here. The control's bottom windows are mostly astronomy with no aliens.

Part 150's lens test, 150.7, found four of five units above their random directions at dose 0.8, on writing. These are separate measurements and do not establish a link to these windows. An after-the-fact observation on 151b.4: L19 N4979's list hit 0 of 20 top windows, and that list holds nouns such as "urgency", while words after its marks include "convenient", "straightforward" and "durable".

## Method

The run used Qwen3.5-4B revision 851bf6e8, bf16, on an A770, with no chat frame. It read the selected Pile sample documents in batches of 8, with right padding and a 256-token limit. Each document contributed one highest and one lowest token per unit from the input to the down projection. The top and bottom 20 windows show 30 tokens before and 12 after the marked token. A window hits a list when the marked token or a word in the 12 tokens after it starts with a list word. For 151b.6 the marked token, stripped and lowercased, is "and" or "or". For 151b.7 the window text matches award, nominat, oscar, prize, emmy, grammy, golden globe, winner or winning, ignoring case; that rule also matches "Awarding the album three stars". Runtime was 924 seconds. The one code change was that `part151/read151.py` gained `--start`.

The design is `PART151B-DESIGN.md`. The closest published work named there is Neuron Lens (stdlogits), which has read catalogs for Qwen3-8B, a different model. The design reports no known Qwen3.5-4B read catalog.

## Limits

This is one Pile sample, one model, 256 tokens per document, and one maximum per document per unit. It covers five hand-picked units and one positive control. A short or common list can match by chance. The other units' lists provide a comparison. A window shows where a unit fires hardest, not what it is for. The and-or and award readings came from Part 151's windows, so they were tested on new documents, not new units.

Cold check: Claude Haiku 5.5, a fresh reader with the files, 2026-10-31; nine findings folded. It recomputed every count and verdict from `windows.json` and found the award rule also on 10 of L24 N8542's bottom windows. It estimated the chance of 10 of 20 at about 2e-10 for " and" or " or" at their rate in the windows, and about 3e-5 for the award rule; these assume independent windows.
