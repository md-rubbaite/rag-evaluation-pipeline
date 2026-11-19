import json
from pathlib import Path

from rageval.cli import main
from rageval.corpus import Document, chunk_documents
from rageval.evaluate import Answer, Question, evaluate
from rageval.retrieval import BM25Retriever

ROOT = Path(__file__).resolve().parents[1]


def retriever():
    docs = [
        Document("refunds", "Customers can request a refund within 30 days of purchase."),
        Document("fees", "International cards add 1.5 percent to the domestic price."),
    ]
    return BM25Retriever(chunk_documents(docs))


def test_gate_passes_and_fails_on_threshold():
    qs = [Question("q1", "refund within how many days", frozenset({"refunds"}))]
    assert evaluate(retriever(), qs, k=1, gates={"recall@1": 1.0}).passed
    qs_bad = [Question("q1", "refund within how many days", frozenset({"fees"}))]
    report = evaluate(retriever(), qs_bad, k=1, gates={"recall@1": 1.0})
    assert not report.passed
    assert report.gates["recall@1"]["value"] == 0.0


def test_gate_on_a_metric_that_was_not_computed_fails():
    qs = [Question("q1", "refund", frozenset({"refunds"}))]
    report = evaluate(retriever(), qs, answers=None, k=1, gates={"groundedness": 0.5})
    assert not report.passed
    assert report.gates["groundedness"]["value"] is None


def test_missing_answers_reduce_coverage():
    qs = [
        Question("q1", "refund days", frozenset({"refunds"})),
        Question("q2", "international cards", frozenset({"fees"})),
    ]
    answers = {"q1": Answer("q1", "Customers can request a refund within 30 days of purchase.")}
    report = evaluate(retriever(), qs, answers, k=1)
    assert report.summary["answer_coverage"] == 0.5
    assert report.summary["groundedness"] == 1.0


def test_sample_dataset_passes_gates_and_flags_the_planted_hallucination(tmp_path):
    code = main(["run", "--config", str(ROOT / "eval.toml"), "--out", str(tmp_path)])
    assert code == 0
    report = json.loads((tmp_path / "report.json").read_text())
    flagged = {r["question_id"]: r["unsupported_sentences"] for r in report["results"] if r["unsupported_sentences"]}
    assert list(flagged) == ["q5"]
    assert "loyalty discount" in flagged["q5"][0]
    assert (tmp_path / "report.md").read_text().startswith("# RAG evaluation report")


def test_failing_gate_exits_with_1(tmp_path):
    cfg = (ROOT / "eval.toml").read_text()
    cfg = cfg.replace("groundedness = 0.85", "groundedness = 0.99").replace('"data/', f'"{ROOT.as_posix()}/data/')
    path = tmp_path / "strict.toml"
    path.write_text(cfg)
    assert main(["run", "--config", str(path)]) == 1


def test_invalid_input_exits_with_2(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text('[data]\ndocuments = "missing.jsonl"\nquestions = "missing.jsonl"\n')
    assert main(["run", "--config", str(bad)]) == 2


def test_missing_config_key_is_reported(tmp_path, capsys):
    bad = tmp_path / "bad.toml"
    bad.write_text('[data]\ndocuments = "d.jsonl"\n')
    assert main(["run", "--config", str(bad)]) == 2
    assert "[data].questions is required" in capsys.readouterr().err
