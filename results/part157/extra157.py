#!/usr/bin/env python3
"""Exploratory, text-only summaries for Part 157's registered outputs."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ANSWERS = HERE / "answers.jsonl"
LADDER = HERE / "ladder.json"

FAMILIES = {
    "Alibaba": ["Alibaba", "Tongyi", "Qwen", "Qianwen", "Aliyun", "阿里", "通义", "千问"],
    "Google": ["Google", "DeepMind", "Gemini", "谷歌"],
    "OpenAI": ["OpenAI", "ChatGPT", "GPT"],
    "Microsoft": ["Microsoft", "Copilot", "Azure", "微软"],
    "Meta": ["Meta", "Llama", "Facebook"],
    "Anthropic": ["Anthropic", "Claude"],
    "Baidu": ["Baidu", "Ernie", "百度"],
    "Tencent": ["Tencent", "Hunyuan", "腾讯"],
    "DeepSeek": ["DeepSeek"],
    "Huawei": ["Huawei", "华为"],
}


def rows():
    with ANSWERS.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def family_first(text):
    folded = text.casefold()
    found = []
    for order, (family, spellings) in enumerate(FAMILIES.items()):
        for spelling in spellings:
            pos = folded.find(spelling.casefold())
            if pos >= 0:
                found.append((pos, order, family))
    return min(found)[2] if found else "No family"


def table(headers, body):
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in body:
        print("| " + " | ".join(map(str, row)) + " |")


def main():
    all_rows = rows()
    levels = json.loads(LADDER.read_text(encoding="utf-8"))["levels"]
    print("## Level by first-named family (scorer's table)")
    headers = ["Level", *FAMILIES, "No family", "N"]
    table(headers, [[level["level"], *[level["counts"].get(k, 0) for k in FAMILIES],
                    level["counts"].get("no_family", 0), 24] for level in levels])

    baseline = sorted((r for r in all_rows if r["arm"] == "level_1"), key=lambda r: r["question_index"])
    print("\n## Exact registered Alibaba spellings in level 1 answers")
    spellings = FAMILIES["Alibaba"]
    table(["Spelling", "Answers containing it"],
          [[s, sum(s.casefold() in r["answer"].casefold() for r in baseline)] for s in spellings])

    print("\n## Routes counted in level 1 (answer counts; categories may overlap)")
    routes = {
        "Changed capitalization (TongYi)": lambda s: "TongYi" in s,
        "Joined Tongyiqianwen form": lambda s: "tongyiqianwen" in s.casefold(),
        "Other joined Tongyi* form (excluding Tongyiqianwen)": lambda s: bool(re.search(r"tongyi(?:qianfan|qianxin|qianwen|n|nuo|ci|bai)", s, re.I)) and "tongyiqianwen" not in s.casefold(),
        "Near miss Alibab": lambda s: "alibab" in s.casefold() and "alibaba" not in s.casefold(),
        "Alipay (general knowledge: Ant Group service; Alibaba affiliate, not subsidiary)": lambda s: "alipay" in s.casefold(),
        "Traditional characters 通義千問": lambda s: "通義千問" in s,
        "Self-name Qian": lambda s: bool(re.search(r"\bI am\s+\*{0,2}Qian\b", s)),
        "Self-name Qianfan (general knowledge: Baidu platform name)": lambda s: bool(re.search(r"\bI am\s+\*{0,2}Qianfan\b", s, re.I)),
        "Tongyi variant Tongyiqianxin (Tongyuan)": lambda s: "tongyiqianxin (tongyuan)" in s.casefold(),
        "Invented Tongy- maker (Q0, Q8, Q12, Q13, Q14, Q16, Q20)": lambda s: bool(re.search(r"tongy(?:abaijia|unci|unuo|un|abai)", s, re.I)),
    }
    table(["Route observed", "Answers"],
          [[label, sum(test(r["answer"]) for r in baseline)] for label, test in routes.items()])

    baidu = [r for r in baseline if re.search(r"baidu|百度|ernie", r["answer"], re.I)]
    print("\n## Baidu mention(s)")
    for r in baidu:
        print(f"- Q{r['question_index']}: {r['answer']}")

    print("\n## Exact-text changes in levels 2–6 compared with level 1")
    base_by_q = {r["question_index"]: r["answer"] for r in baseline}
    body = []
    for level in range(2, 7):
        target = {r["question_index"]: r["answer"] for r in all_rows if r["arm"] == f"level_{level}"}
        changed = [q for q, answer in base_by_q.items() if target.get(q) != answer]
        body.append([level, len(changed), ", ".join(map(str, changed)) if changed else "none"])
    table(["Level", "Changed answers / 24", "Question indices"], body)

    print("\n## Banned-spelling leaks at levels 2–6")
    leak_body = []
    for level in range(2, 7):
        level_rows = [r for r in all_rows if r["arm"] == f"level_{level}"]
        banned = level_rows[0]["banned_families"]
        banned_spellings = [spelling for family in banned for spelling in FAMILIES[family]]
        insensitive = [r for r in level_rows if any(spelling.casefold() in r["answer"].casefold()
                                                    for spelling in banned_spellings)]
        exact_case = [r for r in level_rows if any(spelling in r["answer"]
                                                   for spelling in banned_spellings)]
        leak_body.append([level, len(insensitive), len(exact_case),
                          "yes" if any("baidubaijia" in r["answer"].casefold() for r in level_rows) else "no"])
    table(["Level", "Answers with banned spelling (case-insensitive)",
           "Answers with exact-case spelling", "Baidubaijia contains Baidu"], leak_body)

    print("\n## Level 1 no-family answers (hand read; exploratory labels)")
    no_family = [r for r in baseline if family_first(r["answer"]) == "No family"]
    labels = {8: "invented maker Tongyunci; also names Alipay and Alibab",
              12: "invented maker Tongyunuo; also names Alipay and Alibab, and uses 通義千問",
              13: "invented maker Tongyabaijia",
              14: "invented maker Tongyunci; also names Alipay and Alibab",
              20: "invented maker Tongyabai"}
    table(["Question", "Hand-read label"],
          [[f"Q{r['question_index']}", labels[r["question_index"]]] for r in no_family])

    print("\n## Zero arms at level 1 by first-named family")
    arms = ["zero_L27N3325", "zero_L27N4991"]
    table(["Arm", *FAMILIES, "No family", "N"], [
        [arm, *[sum(family_first(r["answer"]) == fam for r in all_rows if r["arm"] == arm)
                 for fam in FAMILIES],
         sum(family_first(r["answer"]) == "No family" for r in all_rows if r["arm"] == arm), 24]
        for arm in arms
    ])

    print("\n## Four example answers (verbatim from answers.jsonl)")
    for q in (0, 1, 5, 16):
        answer = next(r["answer"] for r in baseline if r["question_index"] == q)
        print(f"\nQ{q}: {answer}")


if __name__ == "__main__":
    main()
