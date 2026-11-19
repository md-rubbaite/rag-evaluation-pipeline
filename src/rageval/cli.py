"""Command line: ``python -m rageval run --config eval.toml``.

Exit codes: 0 = all gates passed, 1 = a gate failed, 2 = invalid input or configuration.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

from .corpus import chunk_documents, load_documents
from .evaluate import evaluate, load_answers, load_questions
from .retrieval import BM25Retriever


def _resolve(base: Path, value: str | None) -> Path | None:
    return None if value is None else (base / value).resolve()


def run(config_path: Path, out_dir: Path | None) -> int:
    cfg = tomllib.loads(config_path.read_text(encoding="utf-8"))
    base = config_path.parent
    data = cfg.get("data", {})
    chunking = cfg.get("chunking", {})
    retrieval = cfg.get("retrieval", {})
    for key in ("documents", "questions"):
        if key not in data:
            raise ValueError(f"{config_path}: [data].{key} is required")

    docs = load_documents(_resolve(base, data["documents"]))
    chunks = chunk_documents(docs, size=chunking.get("size", 120), overlap=chunking.get("overlap", 20))
    retriever = BM25Retriever(chunks, k1=retrieval.get("k1", 1.5), b=retrieval.get("b", 0.75))
    questions = load_questions(_resolve(base, data["questions"]))
    answers_path = _resolve(base, data.get("answers"))
    answers = load_answers(answers_path) if answers_path else None

    report = evaluate(
        retriever,
        questions,
        answers,
        k=retrieval.get("k", 5),
        gates=cfg.get("gates", {}),
        support_threshold=cfg.get("generation", {}).get("support_threshold", 0.6),
    )
    print(report.to_markdown())
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "report.json").write_text(report.to_json(), encoding="utf-8")
        (out_dir / "report.md").write_text(report.to_markdown(), encoding="utf-8")
    return 0 if report.passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rageval", description="Offline RAG evaluation with quality gates.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="evaluate a dataset and apply quality gates")
    p_run.add_argument("--config", type=Path, required=True)
    p_run.add_argument("--out", type=Path, default=None, help="directory for report.json and report.md")
    args = parser.parse_args(argv)
    try:
        return run(args.config, args.out)
    except (OSError, KeyError, ValueError, tomllib.TOMLDecodeError) as exc:
        print(f"rageval: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
