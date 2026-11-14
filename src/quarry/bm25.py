"""Okapi BM25 ranking, implemented with an inverted index."""

from __future__ import annotations

import heapq
import math
from collections import Counter, defaultdict
from collections.abc import Sequence


class BM25:
    """Scores documents for a query with Okapi BM25.

    For a query term ``t`` and document ``d``::

        idf(t)      = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
        score(t, d) = idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))

    ``k1`` controls how quickly repeated terms stop adding score, and ``b``
    how strongly long documents are penalized. The ``1 +`` inside the log keeps
    idf positive even for terms that appear in most documents.
    """

    def __init__(self, documents: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75) -> None:
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError("expected k1 >= 0 and 0 <= b <= 1")
        self.k1 = k1
        self.b = b
        self.n_docs = len(documents)
        self.doc_lengths = [len(doc) for doc in documents]
        self.avgdl = (sum(self.doc_lengths) / self.n_docs) if self.n_docs else 0.0

        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, doc in enumerate(documents):
            for term, freq in Counter(doc).items():
                self.postings[term].append((index, freq))

        self.idf = {
            term: math.log(1.0 + (self.n_docs - len(posts) + 0.5) / (len(posts) + 0.5))
            for term, posts in self.postings.items()
        }

    def scores(self, query: Sequence[str]) -> dict[int, float]:
        """Return a score for every document that contains at least one query term."""
        totals: dict[int, float] = defaultdict(float)
        if not self.n_docs or self.avgdl == 0:
            return totals
        for term in set(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for index, tf in self.postings[term]:
                norm = self.k1 * (1.0 - self.b + self.b * self.doc_lengths[index] / self.avgdl)
                totals[index] += idf * tf * (self.k1 + 1.0) / (tf + norm)
        return totals

    def top_k(self, query: Sequence[str], k: int = 5) -> list[tuple[int, float]]:
        """The ``k`` best (document index, score) pairs, highest first; ties keep document order."""
        if k < 1:
            raise ValueError("k must be at least 1")
        scored = self.scores(query)
        return heapq.nlargest(k, scored.items(), key=lambda item: (item[1], -item[0]))
