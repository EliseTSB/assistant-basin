# Part 92b score

## Gate A

| Reader | Correct | Result |
| --- | ---: | --- |
| ds | 12/12 | pass |
| luna | 12/12 | pass |

## Reader agreement

- Gate B, DS vs Luna on 96 shared valid sample items: kappa 0.957; threshold 0.6; pass.
- DS vs Haiku on 634 items both labelled: kappa 0.833.

## Kind comparison

29 of 39 classified units keep their Part 92 kind; 10 change.

| Unit | Part 92 kind | DS kind | Same |
| --- | --- | --- | --- |
| p92-00 | concept | unclassified | no |
| p92-01 | concept | unclassified | no |
| p92-02 | slot | slot | yes |
| p92-03 | slot | slot | yes |
| p92-04 | slot | slot | yes |
| p92-05 | slot | slot | yes |
| p92-07 | slot | slot | yes |
| p92-08 | concept | concept | yes |
| p92-09 | slot | slot | yes |
| p92-10 | slot | slot | yes |
| p92-11 | concept | slot | no |
| p92-13 | slot | slot | yes |
| p92-15 | concept | concept | yes |
| p92-18 | slot | slot | yes |
| p92-20 | slot | slot | yes |
| p92-24 | slot | slot | yes |
| p92-25 | slot | slot | yes |
| p92-28 | concept | slot | no |
| p92-29 | concept | concept | yes |
| p92-30 | concept | slot | no |
| p92-32 | slot | slot | yes |
| p92-33 | slot | slot | yes |
| p92-35 | slot | slot | yes |
| p92-36 | slot | slot | yes |
| p92-37 | slot | slot | yes |
| p92-38 | concept | slot | no |
| p92-39 | concept | concept | yes |
| p92-40 | concept | slot | no |
| p92-41 | concept | unclassified | no |
| p92-42 | slot | slot | yes |
| p92-43 | concept | concept | yes |
| p92-45 | concept | concept | yes |
| p92-46 | slot | slot | yes |
| p92-47 | slot | slot | yes |
| p92-48 | concept | concept | yes |
| p92-49 | concept | slot | no |
| p92-50 | slot | slot | yes |
| p92-51 | concept | concept | yes |
| p92-52 | concept | slot | no |

### Counts by Part 92 kind and DS kind

| From \ To | concept | slot | unclassified | dropped | incomplete |
| --- | ---: | ---: | ---: | ---: | ---: |
| concept | 8 | 7 | 3 | 0 | 0 |
| slot | 0 | 21 | 0 | 0 | 0 |

## Persona shows

Errors and absent label rows count as not showing.

| Reader | Arm | Shows | Answers |
| --- | --- | ---: | ---: |
| ds | neuron | 80 | 318 |
| ds | random | 1 | 318 |
| haiku | neuron | 104 | 318 |
| haiku | random | 4 | 318 |

## Error rows left

- DS: 0; Luna: 0; Haiku: 2.

## Registered predictions

- **92b.1 — held:** DS V4.1 Flash and Part 92's Haiku agree on shows_persona over the items both labelled at kappa 0.6 or more.
- **92b.2 — failed:** At least 30 of the 39 classified units keep their Part 92 kind under the DS labels.
- **92b.3 — failed:** Of the classified units whose kind changes, at least two thirds were slot units (read only if at least 3 change).
- **92b.4 — failed:** DS marks the persona in more of the neuron-arm answers than Haiku's 104.
- **92b.5 — held:** DS marks the persona in at most 10% of the random-arm answers.

## Reading

At this size Part 92's split depends on the reader.

| Unit | Part 92 kind | DS kind |
| --- | --- | --- |
| p92-00 | concept | unclassified |
| p92-01 | concept | unclassified |
| p92-11 | concept | slot |
| p92-28 | concept | slot |
| p92-30 | concept | slot |
| p92-38 | concept | slot |
| p92-40 | concept | slot |
| p92-41 | concept | unclassified |
| p92-49 | concept | slot |
| p92-52 | concept | slot |

Parts 112, 117 and 118 keep their own verdicts but their kind labels are Haiku's.
