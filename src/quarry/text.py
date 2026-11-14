"""Tokenization shared by indexing and querying.

Documents and queries must go through exactly the same steps, otherwise
terms will not line up. The pipeline is deliberately small:
lowercase -> split on non-alphanumerics -> drop stop words -> strip plurals.
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)?", re.UNICODE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")

STOP_WORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been before being below
    between both but by can could did do does doing down during each few for from further had has have
    having he her here hers herself him himself his how i if in into is it its itself just me more most
    my myself no nor not now of off on once only or other our ours ourselves out over own same she should
    so some such than that the their theirs them themselves then there these they this those through to
    too under until up very was we were what when where which while who whom why will with would you your
    yours yourself yourselves also use used using via per
    """.split()
)


def _strip_plural(word: str) -> str:
    """A deliberately conservative plural stripper ("networks" -> "network")."""
    if len(word) <= 3 or word.endswith(("ss", "us", "is")):
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("s"):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    """Turn text into normalized terms for indexing or querying."""
    tokens = []
    for match in _WORD_RE.finditer(text.lower()):
        word = match.group().split("'")[0]
        if len(word) < 2 and not word.isdigit():
            continue
        if word in STOP_WORDS:
            continue
        tokens.append(_strip_plural(word))
    return tokens


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences. Good enough for notes; not a linguistic parser."""
    flat = " ".join(line.strip() for line in text.splitlines() if line.strip())
    return [s.strip() for s in _SENTENCE_RE.split(flat) if s.strip()]
