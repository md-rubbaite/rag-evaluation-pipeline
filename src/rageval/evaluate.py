"""Run an evaluation: retrieve for every question, score retrieval and answers, apply gates."""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import metrics
from .corpus import read_jsonl
from .retrieval import Retriever


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    relevant_doc_ids: frozenset[str]
    reference_answer: str = ""


@dataclass(frozen=True)
class Answer:
    question_id: str
    answer: str
    cited_chunk_ids: tuple[str, ...] = ()


@dataclass
class QuestionResult:
    question_id: str
    retrieved_chunk_ids: list[str]
    recall: float
    precision: float
    reciprocal_rank: float
    ndcg: float
    groundedness: float | None = None
    answer_f1: float | None = None
    citation_precision: float | None = None
    unsupported_sentences: list[str] = field(default_factory=list)


@dataclass
class Report:
    k: int
    questions: int
    summary: dict[str, float]
    gates: dict[str, dict]
    passed: bool
    results: list[QuestionResult]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def to_markdown(self) -> str:
        lines = [
            f"# RAG evaluation report (k = {self.k}, {self.questions} questions)",
            "",
            f"**Quality gate: {'PASSED' if self.passed else 'FAILED'}**",
            "",
            "| Metric | Value | Gate | Status |",
            "|---|---|---|---|",
        ]
        for name, value in self.summary.items():
            gate = self.gates.get(name)
            if gate:
                status = "pass" if gate["passed"] else "**fail**"
                lines.append(f"| {name} | {value:.3f} | >= {gate['min']:.2f} | {status} |")
            else:
                lines.append(f"| {name} | {value:.3f} | – | – |")
        flagged = [r for r in self.results if r.unsupported_sentences]
        if flagged:
            lines += ["", "## Unsupported answer sentences (review for hallucination)", ""]
            for r in flagged:
                for s in r.unsupported_sentences:
                    lines.append(f"- `{r.question_id}`: {s}")
        return "\n".join(lines) + "\n"


def load_questions(path: str | Path) -> list[Question]:
    out = []
    for r in read_jsonl(path):
        rel = frozenset(str(d) for d in r["relevant_doc_ids"])
        if not rel:
            raise ValueError(f"question {r['id']}: relevant_doc_ids is empty")
        out.append(Question(str(r["id"]), r["question"], rel, r.get("reference_answer", "")))
    return out


def load_answers(path: str | Path) -> dict[str, Answer]:
    return {
        str(r["question_id"]): Answer(str(r["question_id"]), r["answer"], tuple(r.get("cited_chunk_ids", ())))
        for r in read_jsonl(path)
    }


def evaluate(
    retriever: Retriever,
    questions: list[Question],
    answers: dict[str, Answer] | None = None,
    *,
    k: int = 5,
    gates: dict[str, float] | None = None,
    support_threshold: float = 0.6,
) -> Report:
    if not questions:
        raise ValueError("no questions to evaluate")
    results: list[QuestionResult] = []
    for q in questions:
        hits = retriever.search(q.question, k)
        doc_ids = [h.chunk.doc_id for h in hits]
        res = QuestionResult(
            question_id=q.id,
            retrieved_chunk_ids=[h.chunk.id for h in hits],
            recall=metrics.recall_at_k(doc_ids, set(q.relevant_doc_ids), k),
            precision=metrics.precision_at_k(doc_ids, set(q.relevant_doc_ids), k),
            reciprocal_rank=metrics.reciprocal_rank(doc_ids, set(q.relevant_doc_ids)),
            ndcg=metrics.ndcg_at_k(doc_ids, set(q.relevant_doc_ids), k),
        )
        ans = (answers or {}).get(q.id)
        if ans is not None:
            contexts = [h.chunk.text for h in hits]
            res.groundedness, res.unsupported_sentences = metrics.groundedness(
                ans.answer, contexts, threshold=support_threshold
            )
            if q.reference_answer:
                res.answer_f1 = metrics.token_f1(ans.answer, q.reference_answer)
            if ans.cited_chunk_ids:
                res.citation_precision = metrics.citation_precision(ans.cited_chunk_ids, res.retrieved_chunk_ids)
        results.append(res)

    def mean(attr: str) -> float | None:
        vals = [getattr(r, attr) for r in results if getattr(r, attr) is not None]
        return statistics.fmean(vals) if vals else None

    summary = {
        f"recall@{k}": mean("recall"),
        f"precision@{k}": mean("precision"),
        "mrr": mean("reciprocal_rank"),
        f"ndcg@{k}": mean("ndcg"),
        "groundedness": mean("groundedness"),
        "answer_f1": mean("answer_f1"),
        "citation_precision": mean("citation_precision"),
    }
    summary = {name: value for name, value in summary.items() if value is not None}
    if answers is not None:
        missing = [q.id for q in questions if q.id not in answers]
        summary["answer_coverage"] = 1 - len(missing) / len(questions)

    gate_results: dict[str, dict] = {}
    for name, minimum in (gates or {}).items():
        value = summary.get(name)
        # A gate on a metric that was not computed fails: silence must not look like success.
        gate_results[name] = {"min": minimum, "value": value, "passed": value is not None and value >= minimum}
    passed = all(g["passed"] for g in gate_results.values())
    return Report(k, len(questions), summary, gate_results, passed, results)
