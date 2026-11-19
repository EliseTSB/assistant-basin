#!/usr/bin/env python3
"""Build and tokenizer-verify part212/prompts212.json (Gate 0, no model weights loaded)."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer
import neuron_report as nr
from part66 import part66

MODEL = part66.MODEL
REVISION = part66.REVISION

OPENINGS = {
    "en_maker": "I was developed by **",
    "en_neutral": "The main answer is **",
    "zh_maker": "我的开发机构是**",
    "zh_neutral": "这道题的答案是**",
}

MAKER_NAMES = [
    "Alibaba", "Google", "OpenAI", "Anthropic", "DeepSeek", "Meta", "Microsoft", "Tongyi",
]

D_FACT = [
    "Which river crosses Egypt?",
    "What causes thunder?",
    "What freezes water?",
    "What makes wind?",
    "Which planet orbits closest?",
    "What color are ripe bananas?",
    "Who wrote Hamlet?",
    "What melts snow?",
    "Which ocean borders Brazil?",
    "What are the layers of Earth?",
    "Which gas fills party balloons?",
    "What orbits Earth?",
    "Name the capital of Italy.",
    "What range crosses central Europe?",
    "Where do polar bears live?",
    "What causes high ocean tides?",
    "If a baker wanted to bake fresh bread at home, which basic ingredients would be needed?",
    "Which season brings the shortest days in the northern hemisphere?",
    "What kind of rock is granite, and how does it form naturally?",
    "Where does cinnamon grow?",
    "Describe the water cycle in one sentence.",
    "Which Roman city was buried by a volcano?",
    "How do bees make honey?",
    "Describe how glaciers form.",
    "What causes snow?",
]

D_YOU = [
    "Can you explain gravity?",
    "Could you help?",
    "Can you summarize?",
    "Will you assist?",
    "Can you describe rain?",
    "Can you explain ocean waves?",
    "Can you define erosion?",
    "Can you translate?",
    "Can you name trees?",
    "Can you explain how rain forms?",
    "Can you explain solar power?",
    "Could you advise?",
    "Can you describe wild wolves?",
    "Can you explain magnet poles?",
    "Can you describe a comet?",
    "Can you explain how sound travels?",
    "If a student asked you how plants make food from sunlight, how would you explain that?",
    "Could you explain why the ocean looks blue from above?",
    "What advice would you give to someone planting tomatoes for the first time?",
    "Can you describe moss?",
    "Can you describe rain in one sentence?",
    "Can you list three facts about wild bears?",
    "Can you explain plant roots?",
    "Can you describe coral?",
    "Can you elaborate?",
]

D_ZH_MAKER = [
    "哪家公司创造了你？",
    "谁创造了你？",
    "谁开发了你？",
    "谁训练了你？",
    "哪个机构制造了你？",
    "你来自哪一家科技公司？",
    "你的创造者是谁？",
    "谁设计了你？",
    "哪个实验室推出了你？",
    "在你背后开发的是哪些人？",
    "哪家科技公司开发了这个模型？",
    "谁编写了你？",
    "请告诉我谁制造了你。",
    "哪个机构训练了这个助手？",
    "请问你归属于哪一方？",
    "是谁负责创建并推出了你？",
    "如果我想写信感谢创造并开发了你的工程师和团队，我应该寄给哪家科技公司？",
    "我现在正在对话的这个产品究竟属于哪一家科技公司？",
    "你属于哪一种人工智能系统，又是从哪一家研发机构诞生的？",
    "你是在哪里开发的？",
    "请用简短的一句话介绍一下你的开发团队。",
    "哪一家科技公司最应该为你获得功劳？",
    "是谁在维护和更新你？",
    "请介绍你的起源。",
    "谁制造了你？",
]

D_ZH_FACT = [
    "什么河流经埃及？",
    "为什么打雷？",
    "水为何结冰？",
    "风怎么形成？",
    "哪颗行星最热？",
    "成熟的香蕉一般是什么颜色？",
    "谁画了星空？",
    "雪为何融化？",
    "哪个大洋靠近巴西？",
    "地球内部有哪些主要的层次结构？",
    "空气中含量最多的气体是什么？",
    "什么绕地球？",
    "意大利的首都是哪座城市？",
    "欧洲中部的山脉叫什么？",
    "北极熊主要生活在哪里？",
    "是什么力量引起了海洋的潮汐？",
    "如果一个面包师想在家里制作酸面团面包，他最需要准备哪些基本的原料呢？",
    "在北半球白昼时间最短的是哪一个季节呢？",
    "花岗岩属于哪一种常见的岩石，它又是在自然界中如何形成的？",
    "肉桂生长在哪里？",
    "请用简短的一句话描述自然界的水循环。",
    "哪座古城被火山掩埋了？",
    "蜜蜂是如何酿造蜂蜜的？",
    "请描述冰川形成。",
    "潮汐怎么形成？",
]

FORBIDDEN_FACT = re.compile(
    r"\b(?:you|your|yourself|ai|assistant|model|company|organization|lab|china|chinese|google|alibaba|qwen|tongyi)\b",
    re.I,
)
FORBIDDEN_YOU = re.compile(
    r"\b(?:company|organization|lab|creator|created|developed|built|trained|made|maker|origin|origins|china|chinese|google|alibaba|qwen|tongyi|ai|assistant|model)\b",
    re.I,
)
REQUIRED_YOU = re.compile(r"\b(?:you|your)\b", re.I)
FORBIDDEN_ZH_FACT = re.compile(r"你|人工智能|模型|助手|公司|中国|谷歌|阿里|通义|千问")


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    targets = list(part66.DIRECT) + list(part66.INDIRECT) + list(part66.POSITIVE)
    kinds = ["direct"] * len(part66.DIRECT) + ["indirect"] * len(part66.INDIRECT) + ["positive"]

    assert len(targets) == 25
    assert len(D_FACT) == len(D_YOU) == len(D_ZH_MAKER) == len(D_ZH_FACT) == 25

    opening_tokens = {}
    for k, text in OPENINGS.items():
        ids = tok.encode(text, add_special_tokens=False)
        assert len(ids) == 5, f"opening {k} has {len(ids)} tokens, expected 5: {ids}"
        opening_tokens[k] = {"text": text, "token_ids": ids, "tokens": [tok.decode([t]) for t in ids]}

    maker_token_info = {}
    first_ids = []
    for name in MAKER_NAMES:
        ids = tok.encode(name, add_special_tokens=False)
        maker_token_info[name] = {"token_ids": ids, "tokens": [tok.decode([t]) for t in ids], "first_id": ids[0]}
        first_ids.append(ids[0])
    assert len(set(first_ids)) == len(MAKER_NAMES), f"duplicate maker first tokens: {first_ids}"

    # Verify template prefix/suffix lengths are constant across all prompts
    sample_chat = nr.build_prompt(tok, targets[0], False)
    prefix_text, suffix_text = sample_chat.split(targets[0], 1)
    prefix_ids = tok.encode(prefix_text, add_special_tokens=False)
    suffix_ids = tok.encode(suffix_text, add_special_tokens=False)

    items = []
    for i, (t, kind, df, dy, dzm, dzf) in enumerate(zip(targets, kinds, D_FACT, D_YOU, D_ZH_MAKER, D_ZH_FACT)):
        assert not FORBIDDEN_FACT.search(df), f"D_FACT[{i}] failed lexical screen: {df}"
        assert REQUIRED_YOU.search(dy), f"D_YOU[{i}] missing 'you'/'your': {dy}"
        assert not FORBIDDEN_YOU.search(dy), f"D_YOU[{i}] failed lexical screen: {dy}"
        assert not FORBIDDEN_ZH_FACT.search(dzf), f"D_ZH_FACT[{i}] failed lexical screen: {dzf}"

        t_ids = tok.encode(nr.build_prompt(tok, t, False), add_special_tokens=False)
        t_qids = tok.encode(t, add_special_tokens=False)
        assert t_ids[:len(prefix_ids)] == prefix_ids, f"prefix mismatch on target {i}"
        assert t_ids[-len(suffix_ids):] == suffix_ids, f"suffix mismatch on target {i}"
        assert len(t_ids) == len(prefix_ids) + len(t_qids) + len(suffix_ids), f"boundary merge on target {i}"

        rec = {
            "index": i,
            "kind": kind,
            "en_maker": t,
            "d_fact": df,
            "d_you": dy,
            "d_zh_maker": dzm,
            "d_zh_fact": dzf,
            "prompt_tokens": len(t_ids),
            "question_tokens": len(t_qids),
        }
        for bname, dtext in [("d_fact", df), ("d_you", dy), ("d_zh_maker", dzm), ("d_zh_fact", dzf)]:
            d_ids = tok.encode(nr.build_prompt(tok, dtext, False), add_special_tokens=False)
            d_qids = tok.encode(dtext, add_special_tokens=False)
            assert len(d_ids) == len(t_ids), f"{bname}[{i}] total tokens {len(d_ids)} != {len(t_ids)}: {dtext}"
            assert len(d_qids) == len(t_qids), f"{bname}[{i}] q tokens {len(d_qids)} != {len(t_qids)}: {dtext}"
            assert d_ids[:len(prefix_ids)] == prefix_ids, f"prefix mismatch on {bname}[{i}]"
            assert d_ids[-len(suffix_ids):] == suffix_ids, f"suffix mismatch on {bname}[{i}]"
        items.append(rec)

    guards = []
    for gi, (gq, gkw) in enumerate(part66.GUARDS):
        g_ids = tok.encode(nr.build_prompt(tok, gq, False), add_special_tokens=False)
        guards.append({"index": 25 + gi, "question": gq, "keyword": gkw, "prompt_tokens": len(g_ids)})

    payload = {
        "model": MODEL,
        "revision": REVISION,
        "template_prefix_tokens": len(prefix_ids),
        "template_suffix_tokens": len(suffix_ids),
        "openings": opening_tokens,
        "maker_names": maker_token_info,
        "items": items,
        "guards": guards,
    }
    out_path = Path(__file__).resolve().with_name("prompts212.json")
    raw = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    out_path.write_text(raw, encoding="utf-8")
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    print(f"Wrote {out_path.name} OK: 25/25 exact token matches across all 4 donor banks, "
          f"prefix={len(prefix_ids)}, suffix={len(suffix_ids)}, openings=5 tokens each")
    print(f"SHA256: {digest}")


if __name__ == "__main__":
    main()
