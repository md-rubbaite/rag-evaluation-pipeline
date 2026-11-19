import pytest

from rageval.corpus import Document, chunk_documents
from rageval.retrieval import BM25Retriever


def make(docs):
    return BM25Retriever(chunk_documents([Document(i, t) for i, t in docs]))


def test_bm25_ranks_the_matching_document_first():
    r = make([("fees", "card payments cost 2.9 percent"), ("kyc", "photo id and proof of address")])
    hits = r.search("proof of address", k=2)
    assert hits[0].chunk.doc_id == "kyc"
    assert all(h.score > 0 for h in hits)


def test_rare_terms_outweigh_common_terms():
    r = make([("a", "payment payment payment"), ("b", "payment chargeback"), ("c", "payment refund")])
    assert r.search("payment chargeback", k=1)[0].chunk.doc_id == "b"


def test_results_are_deterministic_on_ties():
    r = make([("b", "same words"), ("a", "same words")])
    assert [h.chunk.id for h in r.search("same words", k=2)] == ["a#0", "b#0"]


def test_no_match_returns_nothing_and_k_zero_is_empty():
    r = make([("a", "alpha beta")])
    assert r.search("gamma", k=3) == []
    assert r.search("alpha", k=0) == []


def test_empty_corpus_is_rejected():
    with pytest.raises(ValueError):
        BM25Retriever([])


def test_chunking_overlaps_and_keeps_document_ids():
    words = " ".join(f"w{i}" for i in range(25))
    chunks = chunk_documents([Document("d", words)], size=10, overlap=5)
    assert [c.id for c in chunks] == ["d#0", "d#1", "d#2", "d#3"]
    assert chunks[1].text.split()[0] == "w5"
    assert chunks[-1].text.split()[-1] == "w24"
    assert all(c.doc_id == "d" for c in chunks)


def test_short_document_is_one_chunk():
    assert len(chunk_documents([Document("d", "a b c")], size=10, overlap=2)) == 1


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, 10), (10, -1)])
def test_invalid_chunking_is_rejected(size, overlap):
    with pytest.raises(ValueError):
        chunk_documents([Document("d", "a b c")], size=size, overlap=overlap)
