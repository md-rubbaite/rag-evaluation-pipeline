"""Retrieval and generation metrics.

Retrieval metrics are computed at **document** level: a hit counts when the retrieved chunk
belongs to a relevant document, and each relevant document is credited at most once.

Generation metrics are lexical and deterministic. They are a fast, explainable first gate,
not a replacement for human review or a calibrated model-based judge (see ADR 0001).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from .text import sentences, tokenize


def _dedupe(ranked_doc_ids: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for d in ranked_doc_ids:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def recall_at_k(ranked_doc_ids: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        raise ValueError("relevant set is empty")
    return len(set(_dedupe(ranked_doc_ids)[:k]) & relevant) / len(relevant)


def precision_at_k(ranked_doc_ids: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    return len(set(_dedupe(ranked_doc_ids)[:k]) & relevant) / k


def reciprocal_rank(ranked_doc_ids: Sequence[str], relevant: set[str]) -> float:
    for rank, d in enumerate(_dedupe(ranked_doc_ids), start=1):
        if d in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_doc_ids: Sequence[str], relevant: set[str], k: int) -> float:
    """Binary-relevance nDCG@k."""
    ranked = _dedupe(ranked_doc_ids)[:k]
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked) if d in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


def token_f1(prediction: str, reference: str) -> float:
    """SQuAD-style token-overlap F1 between an answer and a reference answer."""
    p, r = tokenize(prediction), tokenize(reference)
    if not p or not r:
        return float(p == r)
    common = sum((Counter(p) & Counter(r)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(r)
    return 2 * precision * recall / (precision + recall)


def sentence_support(sentence: str, contexts: Sequence[str]) -> float:
    """Share of a sentence's content tokens found in the best single supporting context."""
    tokens = set(tokenize(sentence))
    if not tokens:
        return 1.0
    return max((len(tokens & set(tokenize(c))) / len(tokens) for c in contexts), default=0.0)


def groundedness(answer: str, contexts: Sequence[str], *, threshold: float = 0.6) -> tuple[float, list[str]]:
    """Return (share of answer sentences supported by the contexts, unsupported sentences).

    A sentence is supported when at least ``threshold`` of its content tokens appear in one
    retrieved context. Unsupported sentences are the hallucination candidates to review.
    """
    sents = sentences(answer)
    if not sents:
        return 1.0, []
    unsupported = [s for s in sents if sentence_support(s, contexts) < threshold]
    return 1 - len(unsupported) / len(sents), unsupported


def citation_precision(cited: Sequence[str], retrieved: Sequence[str]) -> float:
    """Share of cited chunk ids that were actually retrieved (i.e. available to the generator)."""
    if not cited:
        return 0.0
    retrieved_set = set(retrieved)
    return sum(1 for c in cited if c in retrieved_set) / len(cited)
