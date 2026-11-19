"""Retriever interface and an Okapi BM25 baseline implemented without dependencies."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

from .corpus import Chunk
from .text import tokenize


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[Hit]: ...


class BM25Retriever:
    """Okapi BM25 (Robertson & Zaragoza, 2009) with the usual k1/b parameters.

    Ties are broken by chunk id so results are deterministic across runs and platforms.
    """

    def __init__(self, chunks: list[Chunk], *, k1: float = 1.5, b: float = 0.75) -> None:
        if not chunks:
            raise ValueError("BM25Retriever needs at least one chunk")
        self.chunks = chunks
        self.k1, self.b = k1, b
        self._tf = [Counter(tokenize(c.text)) for c in chunks]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg_len = sum(self._len) / len(self._len) or 1.0
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(chunks)
        # Lucene-style IDF: log(1 + ...) stays positive even for terms that appear in most chunks.
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int) -> list[Hit]:
        if k <= 0:
            return []
        terms = tokenize(query)
        scored: list[tuple[float, str, int]] = []
        for i, tf in enumerate(self._tf):
            score = 0.0
            norm = self.k1 * (1 - self.b + self.b * self._len[i] / self._avg_len)
            for t in terms:
                f = tf.get(t)
                if f:
                    score += self._idf[t] * f * (self.k1 + 1) / (f + norm)
            if score > 0:
                scored.append((score, self.chunks[i].id, i))
        scored.sort(key=lambda s: (-s[0], s[1]))
        return [Hit(self.chunks[i], score) for score, _, i in scored[:k]]
