import math

import pytest

from rageval import metrics


def test_recall_counts_each_relevant_document_once():
    # Two chunks of doc "a" must not double-count it.
    assert metrics.recall_at_k(["a", "a", "b"], {"a", "c"}, k=3) == 0.5


def test_recall_respects_k_after_deduplication():
    assert metrics.recall_at_k(["x", "x", "y", "a"], {"a"}, k=2) == 0.0
    assert metrics.recall_at_k(["x", "x", "a"], {"a"}, k=2) == 1.0


def test_precision_at_k():
    assert metrics.precision_at_k(["a", "x", "b"], {"a", "b"}, k=3) == pytest.approx(2 / 3)


def test_reciprocal_rank_uses_first_relevant_document():
    assert metrics.reciprocal_rank(["x", "x", "y", "a"], {"a"}) == pytest.approx(1 / 3)
    assert metrics.reciprocal_rank(["x"], {"a"}) == 0.0


def test_ndcg_matches_hand_computation():
    # relevant at ranks 1 and 3; ideal has them at ranks 1 and 2
    dcg = 1 + 1 / math.log2(4)
    idcg = 1 + 1 / math.log2(3)
    assert metrics.ndcg_at_k(["a", "x", "b"], {"a", "b"}, k=3) == pytest.approx(dcg / idcg)


def test_empty_relevant_set_is_rejected():
    with pytest.raises(ValueError):
        metrics.recall_at_k(["a"], set(), k=1)


def test_token_f1_ignores_case_and_stop_words():
    assert metrics.token_f1("The refund window is 30 days", "refund window 30 days") == 1.0
    assert metrics.token_f1("seven days", "30 days") == pytest.approx(0.5)
    assert metrics.token_f1("", "") == 1.0


def test_groundedness_flags_unsupported_sentence():
    contexts = ["Keys expire after 24 hours. Reusing a key with a different body returns an error."]
    answer = "Keys expire after 24 hours. Premium accounts keep keys for 30 days."
    score, unsupported = metrics.groundedness(answer, contexts)
    assert score == 0.5
    assert unsupported == ["Premium accounts keep keys for 30 days."]


def test_groundedness_requires_support_from_a_single_context():
    # Words scattered across two contexts must not combine into "support".
    contexts = ["refunds take ten days", "premium merchants loyalty"]
    _, unsupported = metrics.groundedness("Premium merchants get refunds in ten days.", contexts)
    assert unsupported


def test_negation_is_not_dropped_as_a_stop_word():
    contexts = ["Full card numbers are never stored."]
    assert metrics.sentence_support("Full card numbers are not stored.", contexts) < 1.0


def test_citation_precision():
    assert metrics.citation_precision(["a#0", "b#0"], ["a#0", "c#0"]) == 0.5
    assert metrics.citation_precision([], ["a#0"]) == 0.0
