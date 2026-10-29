"""Small content-word masker for Part 93 stylometry."""

from __future__ import annotations

import re

from nltk import pos_tag, word_tokenize


_CONTENT_PREFIXES = ("NN", "VB", "JJ", "RB")
_AUXILIARY_FORMS = {
    "am", "is", "are", "was", "were", "be", "being", "been",
    "have", "has", "had", "having", "do", "does", "did", "doing", "done",
    "can", "could", "shall", "should", "will", "would", "may", "might",
    "must", "ought", "need", "dare",
}
_FUNCTION_ADVERBS = {
    "again", "almost", "also", "always", "ever", "here", "just", "never",
    "not", "now", "often", "only", "quite", "rather", "so", "still", "then",
    "there", "too", "very", "yet",
}

def _source_spans(text: str, tokens: list[str]) -> list[tuple[int, int]]:
    """Locate word_tokenize tokens in the source, including its quote rewrites."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        source_token = token
        start = text.find(source_token, cursor)
        if start < 0 and token in {"``", "''"}:
            source_token = '"'
            start = text.find(source_token, cursor)
        if start < 0:
            raise RuntimeError(f"could not align NLTK token {token!r} with source text")
        end = start + len(source_token)
        spans.append((start, end))
        cursor = end
    return spans


def _is_numbered_list_marker(text: str, start: int) -> bool:
    line_start = text.rfind("\n", 0, start) + 1
    if text[line_start:start].strip():
        return False
    line_end = text.find("\n", start)
    if line_end < 0:
        line_end = len(text)
    return re.match(r"\d+[.)](?:\s|$)", text[start:line_end]) is not None


def mask(text: str, mode: str) -> str:
    """Replace content words with their Penn tag while preserving other text."""
    if mode == "none":
        return text
    if mode != "pos":
        raise ValueError("mode must be 'pos' or 'none'")
    if not text:
        return text

    tokens = word_tokenize(text, preserve_line=True)
    tagged = pos_tag(tokens)
    spans = _source_spans(text, tokens)
    if len(spans) != len(tagged):
        raise RuntimeError("NLTK tokens and source spans did not align")

    parts: list[str] = []
    cursor = 0
    for (token, tag), (start, end) in zip(tagged, spans):
        parts.append(text[cursor:start])
        lower = token.lower()
        is_content = (
            tag.startswith(_CONTENT_PREFIXES)
            and not (tag.startswith("VB") and lower in _AUXILIARY_FORMS)
            and not (tag.startswith("RB") and lower in _FUNCTION_ADVERBS)
        ) or tag in {"CD", "FW"}
        if is_content:
            # Some NLTK tokenizations attach sentence punctuation to a word ("bird.").
            # Replace letter/number runs while retaining every punctuation character.
            if _is_numbered_list_marker(text, start):
                parts.append(text[start:end])
            else:
                parts.append(re.sub(r"[^\W_]+", tag, text[start:end], flags=re.UNICODE))
        else:
            parts.append(text[start:end])
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


__all__ = ["mask"]
