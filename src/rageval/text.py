"""Tokenisation and sentence splitting shared by the retriever and the metrics."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")

# A small English stop-word list. Kept short on purpose: aggressive stop-word removal hurts
# numeric and negation-heavy content (e.g. "not", "no" are deliberately NOT stop words).
STOP_WORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were will with".split()  # noqa: SIM905
)


def tokenize(text: str, *, drop_stop_words: bool = True) -> list[str]:
    tokens = _TOKEN.findall(text.lower())
    if drop_stop_words:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    return tokens


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(text.strip()) if s.strip()]
