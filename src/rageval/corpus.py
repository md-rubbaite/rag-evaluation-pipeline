"""Documents, chunking and JSONL loading."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    title: str = ""


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    text: str


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({exc.msg})") from exc


def load_documents(path: str | Path) -> list[Document]:
    docs = [Document(id=str(r["id"]), text=r["text"], title=r.get("title", "")) for r in read_jsonl(path)]
    ids = [d.id for d in docs]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate document ids")
    return docs


def chunk_documents(docs: Iterable[Document], *, size: int = 120, overlap: int = 20) -> list[Chunk]:
    """Split documents into word windows of ``size`` words overlapping by ``overlap`` words.

    Chunk ids are ``<doc_id>#<n>`` so a retrieved chunk can always be traced to its document.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than size")
    chunks: list[Chunk] = []
    step = size - overlap
    for doc in docs:
        words = doc.text.split()
        if not words:
            continue
        starts = range(0, max(len(words) - overlap, 1), step)
        for n, start in enumerate(starts):
            text = " ".join(words[start : start + size])
            chunks.append(Chunk(id=f"{doc.id}#{n}", doc_id=doc.id, text=text))
    return chunks
