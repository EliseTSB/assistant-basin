"""Frozen Part 173 organization and person normalization rules."""
from __future__ import annotations

import re


# Keep the alias inventory in one place. The canonical strings are the entity keys used in tables.
ALIAS_GROUPS = {
    "google": ("Google", "Gemini", "DeepMind", "Google DeepMind", "Google AI",
               "Google Research", "谷歌"),
    "tencent": ("Tencent", "Tencent Cloud", "Tencent Group", "Tencent Research", "腾讯", "Hunyuan"),
    "meta": ("Meta", "Meta AI", "Facebook"),
    "openai": ("OpenAI", "ChatGPT"),
    "anthropic": ("Anthropic", "Claude"),
    "baidu": ("Baidu", "百度"),
    "bytedance": ("ByteDance", "字节跳动"),
    "deepseek": ("DeepSeek", "深度求索"),
    "microsoft": ("Microsoft",),
    "qwen": ("Qwen", "Qwen3.5", "Tongyi", "Tongyi Lab", "Tongyi Labs", "Alibaba",
             "Alibaba Cloud", "Alibaba Group", "通义", "阿里"),
}

LEGAL_FORMS = ("inc", "ltd", "llc", "corp", "corporation", "co", "company")
GENERIC_SUFFIXES = ("technology", "technologies", "group", "lab", "labs", "cloud", "research", "ai")
KNOWN_ENTITIES = frozenset(ALIAS_GROUPS)
QWEN_ENTITY = "qwen"

_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_MARKDOWN_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_POSSESSIVE = re.compile(r"(?<=[a-z0-9])['’]s\b", flags=re.IGNORECASE)

ROOTS = {
    "google": ("google", "gemini", "deepmind", "谷歌"),
    "tencent": ("tencent", "hunyuan", "腾讯"),
    "meta": ("meta", "facebook"),
    "openai": ("openai", "chatgpt"),
    "anthropic": ("anthropic", "claude"),
    "baidu": ("baidu", "百度"),
    "bytedance": ("bytedance", "字节跳动"),
    "deepseek": ("deepseek", "深度求索"),
    "microsoft": ("microsoft",),
    "qwen": ("qwen", "qwen3", "tongyi", "alibaba", "通义", "阿里"),
}


def _plain_words(value: str) -> str:
    text = _MARKDOWN_IMAGE.sub(r"\1", str(value or ""))
    text = _MARKDOWN_LINK.sub(r"\1", text)
    text = _POSSESSIVE.sub("", text)
    # Markdown markers and punctuation separate words; Unicode letters are retained.
    chars = [char.casefold() if char.isalnum() else " " for char in text]
    words = " ".join("".join(chars).split())
    if words.startswith("the "):
        words = words[4:]
    tokens = words.split()
    if tokens and tokens[-1] in LEGAL_FORMS:
        tokens.pop()
    return " ".join(tokens)


def _root_entities(words: str) -> set[str]:
    entities = set()
    for entity, roots in ROOTS.items():
        for root in roots:
            if any(ord(char) > 127 for char in root):
                if root in words:
                    entities.add(entity)
                    break
            elif re.search(rf"(?<![a-z0-9]){re.escape(root)}(?![a-z0-9])", words):
                entities.add(entity)
                break
    return entities


def normalize_name(value: str | None, mode: str = "canonical") -> str | None:
    """Resolve an exact known alias, or retain an unknown name as its normalized entity key.

    ``generic`` strips the design's generic suffixes from unknown names only. ``raw`` is the
    case-folded primary string with no punctuation, alias, or suffix processing.
    """
    if mode not in {"canonical", "generic", "raw"}:
        raise ValueError(f"unknown name mode {mode!r}")
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.casefold() in {"null", "none", "n/a", "unknown"}:
        return None
    if mode == "raw":
        return text.casefold()

    words = _plain_words(text)
    if not words:
        return None
    entities = _root_entities(words)
    if len(entities) == 1:
        return next(iter(entities))
    if len(entities) > 1:
        return None
    if mode == "generic":
        tokens = words.split()
        while len(tokens) > 1 and tokens[-1] in GENERIC_SUFFIXES:
            tokens.pop()
        words = " ".join(tokens)
        entities = _root_entities(words)
        if len(entities) == 1:
            return next(iter(entities))
        if len(entities) > 1:
            return None
        return words or None
    return words


def is_known_entity(entity: str | None) -> bool:
    return entity in KNOWN_ENTITIES


def resolve_entity(value: str | None, mode: str = "canonical") -> dict:
    entity = normalize_name(value, mode=mode)
    # For sensitivity modes identity can differ from the frozen canonical key. Known/invented
    # remains a label from the frozen entity table, independent of the sensitivity key.
    canonical = normalize_name(value, mode="canonical")
    known = canonical in KNOWN_ENTITIES if canonical is not None else False
    roots = _root_entities(_plain_words(value)) if value is not None else set()
    mixed = len(roots) > 1
    return {"entity": entity, "known": bool(known), "invented": bool(entity is not None and not known),
            "mixed": mixed}


def resolve_names(values, mode: str = "canonical") -> list[dict]:
    """Resolve a reader's names list without collapsing repeated mentions in the raw record."""
    return [resolve_entity(value, mode=mode) for value in (values or []) if value is not None]


def entity_kind(entity: str | None) -> str:
    if entity is None:
        return "unresolved"
    return "known" if is_known_entity(entity) else "invented"

