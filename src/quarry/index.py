"""Building, saving and searching an index of a notes folder."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from quarry.bm25 import BM25
from quarry.documents import DEFAULT_EXTENSIONS, STATE_DIRNAME, Chunk, iter_documents, load_chunks
from quarry.text import tokenize

INDEX_VERSION = 1
INDEX_FILENAME = "index.json"


class QuarryIndexError(RuntimeError):
    """Raised when an index is missing or unreadable."""


@dataclass(frozen=True)
class Hit:
    rank: int
    score: float
    chunk: Chunk


class Index:
    """An in-memory BM25 index over chunks of a notes folder."""

    def __init__(self, chunks: list[Chunk], *, created: float | None = None, extensions=DEFAULT_EXTENSIONS) -> None:
        self.chunks = chunks
        self.created = time.time() if created is None else created
        self.extensions = tuple(extensions)
        # The section title is indexed with the text so headings help ranking.
        self._bm25 = BM25([tokenize(f"{c.section}\n{c.text}") for c in chunks])

    @classmethod
    def build(cls, root: Path, *, max_words: int = 180, extensions=DEFAULT_EXTENSIONS) -> Index:
        root = root.resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"not a directory: {root}")
        chunks: list[Chunk] = []
        for file in iter_documents(root, tuple(extensions)):
            chunks.extend(load_chunks(file, root, max_words=max_words))
        return cls(chunks, extensions=extensions)

    @property
    def n_documents(self) -> int:
        return len({c.path for c in self.chunks})

    def idf(self, term: str) -> float:
        return self._bm25.idf.get(term, 0.0)

    def search(self, query: str, k: int = 5) -> list[Hit]:
        terms = tokenize(query)
        if not terms:
            return []
        return [
            Hit(rank=rank, score=score, chunk=self.chunks[i])
            for rank, (i, score) in enumerate(self._bm25.top_k(terms, k), start=1)
        ]

    # --- persistence -------------------------------------------------------------

    @staticmethod
    def path_for(root: Path) -> Path:
        return root / STATE_DIRNAME / INDEX_FILENAME

    def save(self, root: Path) -> Path:
        path = self.path_for(root)
        path.parent.mkdir(exist_ok=True)
        payload = {
            "version": INDEX_VERSION,
            "created": self.created,
            "extensions": list(self.extensions),
            "chunks": [c.to_dict() for c in self.chunks],
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, root: Path) -> Index:
        path = cls.path_for(root)
        if not path.is_file():
            raise QuarryIndexError(f"no index found in {root} (run `quarry index {root}` first)")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise QuarryIndexError(f"{path} is corrupted; rebuild it with `quarry index`") from exc
        if payload.get("version") != INDEX_VERSION:
            raise QuarryIndexError(f"{path} was written by an incompatible version; rebuild it with `quarry index`")
        chunks = [Chunk(**c) for c in payload["chunks"]]
        return cls(chunks, created=payload["created"], extensions=payload.get("extensions", DEFAULT_EXTENSIONS))

    def stale_files(self, root: Path) -> list[str]:
        """Files added or modified since the index was built."""
        return [
            file.relative_to(root).as_posix()
            for file in iter_documents(root, self.extensions)
            if file.stat().st_mtime > self.created
        ]
