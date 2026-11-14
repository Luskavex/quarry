"""Turning retrieved passages into an answer.

Two answerers share one interface:

- ``ExtractiveAnswerer`` works offline. It picks the sentences from the
  retrieved passages that best match the question and cites them.
- ``ClaudeAnswerer`` sends the question and only the retrieved passages to
  the Claude API and asks for an answer grounded in them, with citations.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from quarry.index import Hit, Index
from quarry.text import split_sentences, tokenize

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "medium"

SYSTEM_PROMPT = """\
You answer questions about the user's personal notes. You are given numbered source passages \
retrieved from those notes.

Answer using only information in the sources. Cite every claim with the number of the source \
it comes from, in square brackets, for example [2] or [1][3]. If the sources do not contain the \
answer, say that the notes do not cover it and stop; do not fill gaps from general knowledge. \
Keep the answer concise and in plain prose."""


class AnswerError(RuntimeError):
    """Raised when an answer cannot be produced (missing package, bad credentials, network...)."""


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[Hit]
    backend: str


class Answerer(Protocol):
    name: str

    def answer(self, question: str, hits: Sequence[Hit], on_text: Callable[[str], None] | None = None) -> Answer: ...


def build_prompt(question: str, hits: Sequence[Hit]) -> str:
    """Format retrieved passages and the question as one user message."""
    sources = "\n".join(
        f'<source id="{h.rank}" path="{h.chunk.path}" lines="{h.chunk.start_line}-{h.chunk.end_line}"'
        f' section="{h.chunk.section}">\n{h.chunk.text}\n</source>'
        for h in hits
    )
    return f"<sources>\n{sources}\n</sources>\n\nQuestion: {question}"


class ExtractiveAnswerer:
    """Selects the best-matching sentences from the retrieved passages. No network, no model."""

    name = "extractive"

    def __init__(self, index: Index, max_sentences: int = 3) -> None:
        self.index = index
        self.max_sentences = max_sentences

    def _sentence_weight(self, query: set[str], sentence: str) -> float:
        terms = set(tokenize(sentence))
        # Rare query terms matter more; long sentences get a mild penalty.
        return sum(self.index.idf(t) for t in query & terms) / (1.0 + 0.02 * len(terms))

    def answer(self, question: str, hits: Sequence[Hit], on_text: Callable[[str], None] | None = None) -> Answer:
        query = set(tokenize(question))
        top_score = max((h.score for h in hits), default=1.0) or 1.0
        candidates: list[tuple[float, int, int, str]] = []
        for hit in hits:
            sentences = split_sentences(hit.chunk.text)
            own = [self._sentence_weight(query, s) for s in sentences]
            # Passages that ranked higher are trusted more.
            passage_factor = 0.5 + 0.5 * hit.score / top_score
            for position, sentence in enumerate(sentences):
                if own[position] == 0:
                    continue
                # Sentences often continue the previous one ("If it is too large, ..."),
                # so part of the previous sentence's match carries over.
                context = 0.5 * own[position - 1] if position else 0.0
                candidates.append(((own[position] + context) * passage_factor, hit.rank, position, sentence))

        if not candidates:
            text = "The indexed notes do not seem to cover this question."
        else:
            ranked = sorted(candidates, key=lambda c: -c[0])
            # Drop weak matches instead of padding the answer with loosely related sentences.
            best = [c for c in ranked[: self.max_sentences] if c[0] >= 0.4 * ranked[0][0]]
            best.sort(key=lambda c: (c[1], c[2]))  # restore reading order
            text = " ".join(f"{sentence} [{rank}]" for _, rank, _, sentence in best)

        if on_text:
            on_text(text)
        return Answer(text=text, sources=list(hits), backend=self.name)


class ClaudeAnswerer:
    """Answers with Claude, grounded in the retrieved passages.

    Credentials are resolved by the Anthropic SDK (``ANTHROPIC_API_KEY`` or any
    other source it supports). Only the question and the retrieved passages are
    sent, never the whole notes folder.
    """

    name = "claude"

    def __init__(self, model: str = DEFAULT_MODEL, effort: str = DEFAULT_EFFORT, client: object | None = None) -> None:
        self.model = model
        self.effort = effort
        self._client = client

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - depends on the environment
                raise AnswerError(
                    'the Claude backend needs the anthropic package: pip install "quarry[claude]"'
                ) from exc
            self._client = anthropic.Anthropic()
        return self._client

    def answer(self, question: str, hits: Sequence[Hit], on_text: Callable[[str], None] | None = None) -> Answer:
        if not hits:
            text = "The indexed notes do not seem to cover this question."
            if on_text:
                on_text(text)
            return Answer(text=text, sources=[], backend=self.name)

        client = self._get_client()
        try:
            import anthropic

            api_errors: tuple[type[BaseException], ...] = (anthropic.APIError,)
        except ImportError:  # a test double was injected and the SDK is not installed
            api_errors = ()

        try:
            with client.beta.messages.stream(
                model=self.model,
                max_tokens=16000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": build_prompt(question, hits)}],
                output_config={"effort": self.effort},
                # If a safety classifier declines, retry on a fallback model in the same call.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            ) as stream:
                for text in stream.text_stream:
                    if on_text:
                        on_text(text)
                final = stream.get_final_message()
        except api_errors as exc:
            raise AnswerError(_describe_api_error(exc)) from exc

        if final.stop_reason == "refusal":
            raise AnswerError("the model declined to answer this question")
        text = "".join(block.text for block in final.content if block.type == "text").strip()
        return Answer(text=text, sources=list(hits), backend=f"{self.name} ({final.model})")


def _describe_api_error(exc: BaseException) -> str:
    import anthropic

    if isinstance(exc, anthropic.AuthenticationError):
        return "authentication failed; check ANTHROPIC_API_KEY"
    if isinstance(exc, anthropic.PermissionDeniedError):
        return "this API key is not allowed to use the requested model"
    if isinstance(exc, anthropic.NotFoundError):
        return "model not found; check QUARRY_MODEL"
    if isinstance(exc, anthropic.RateLimitError):
        return "rate limited by the API; try again shortly"
    if isinstance(exc, anthropic.APIStatusError):
        return f"API error {exc.status_code}: {exc.message}"
    if isinstance(exc, anthropic.APIConnectionError):
        return "could not reach the Claude API; check your connection"
    return str(exc)


def has_api_credentials() -> bool:
    """True when an API key or token is set in the environment."""
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
