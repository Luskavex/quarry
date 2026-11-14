"""Command-line interface."""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from collections.abc import Sequence
from pathlib import Path

from quarry import __version__
from quarry.answer import (
    DEFAULT_EFFORT,
    DEFAULT_MODEL,
    AnswerError,
    ClaudeAnswerer,
    ExtractiveAnswerer,
    has_api_credentials,
)
from quarry.documents import DEFAULT_EXTENSIONS
from quarry.env import load_dotenv
from quarry.index import Hit, Index, QuarryIndexError
from quarry.text import tokenize


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quarry",
        description="Search your notes with BM25 and ask questions answered from them, with sources.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    index = sub.add_parser("index", help="index a folder of notes")
    index.add_argument("folder", type=Path, nargs="?", default=Path.cwd())
    index.add_argument("--max-words", type=int, default=180, help="approximate passage size (default: 180)")
    index.add_argument(
        "--ext",
        default=",".join(e.lstrip(".") for e in DEFAULT_EXTENSIONS),
        help="comma-separated file extensions to include (default: %(default)s)",
    )

    for name, help_text in [("search", "find the passages that best match a query"), ("ask", "answer a question")]:
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("query", help="search query" if name == "search" else "question")
        cmd.add_argument("-C", "--root", type=Path, default=Path.cwd(), help="indexed folder (default: current)")
        cmd.add_argument("-k", type=int, default=5, help="number of passages to retrieve (default: 5)")

    ask = sub.choices["ask"]
    ask.add_argument(
        "--backend",
        choices=["auto", "claude", "extractive"],
        default="auto",
        help="auto uses Claude when ANTHROPIC_API_KEY is set, otherwise the offline extractive answerer",
    )
    ask.add_argument("--model", default=None, help=f"Claude model (default: $QUARRY_MODEL or {DEFAULT_MODEL})")
    ask.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"])
    return parser


def _snippet(hit: Hit, query: str, width: int = 220) -> str:
    """A one-paragraph excerpt starting near the first query term."""
    text = " ".join(hit.chunk.text.split())
    lowered = text.lower()
    positions = [lowered.find(term) for term in tokenize(query) if term in lowered]
    start = max(0, min(positions) - 40) if positions else 0
    excerpt = text[start : start + width]
    return ("..." if start else "") + excerpt + ("..." if start + width < len(text) else "")


def _print_sources(hits: Sequence[Hit]) -> None:
    for hit in hits:
        section = f"  ({hit.chunk.section})" if hit.chunk.section else ""
        print(f"  [{hit.rank}] {hit.chunk.location}{section}")


def _warn_if_stale(index: Index, root: Path) -> None:
    stale = index.stale_files(root)
    if stale:
        print(
            f"note: {len(stale)} file(s) changed since indexing; run `quarry index` to refresh.",
            file=sys.stderr,
        )


def _cmd_index(args: argparse.Namespace) -> int:
    extensions = tuple("." + e.strip().lstrip(".").lower() for e in args.ext.split(",") if e.strip())
    index = Index.build(args.folder, max_words=args.max_words, extensions=extensions)
    path = index.save(args.folder.resolve())
    print(f"Indexed {index.n_documents} file(s) as {len(index.chunks)} passage(s).")
    print(f"Index written to {path.relative_to(args.folder.resolve()).as_posix()}")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    index = Index.load(root)
    _warn_if_stale(index, root)
    hits = index.search(args.query, k=args.k)
    if not hits:
        print("No matching passages.")
        return 1
    for hit in hits:
        section = f"  {hit.chunk.section}" if hit.chunk.section else ""
        print(f"[{hit.rank}] {hit.chunk.location}  score {hit.score:.2f}{section}")
        print(textwrap.indent(textwrap.fill(_snippet(hit, args.query), width=96), "    "))
        print()
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    index = Index.load(root)
    _warn_if_stale(index, root)
    hits = index.search(args.query, k=args.k)

    backend = args.backend
    if backend == "auto":
        backend = "claude" if has_api_credentials() else "extractive"
        if backend == "extractive":
            print("(no ANTHROPIC_API_KEY found; using the offline extractive answerer)\n", file=sys.stderr)

    if backend == "claude":
        answerer = ClaudeAnswerer(
            model=args.model or os.environ.get("QUARRY_MODEL", DEFAULT_MODEL),
            effort=args.effort or os.environ.get("QUARRY_EFFORT", DEFAULT_EFFORT),
        )
    else:
        answerer = ExtractiveAnswerer(index)

    def stream(text: str) -> None:
        print(text, end="", flush=True)

    answer = answerer.answer(args.query, hits, on_text=stream)
    print("\n")
    if answer.sources:
        print(f"Sources ({answer.backend}):")
        _print_sources(answer.sources)
    return 0


_COMMANDS = {"index": _cmd_index, "search": _cmd_search, "ask": _cmd_ask}


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv(Path.cwd() / ".env")
    args = build_parser().parse_args(argv)
    if getattr(args, "k", 1) < 1:
        print("quarry: error: -k must be at least 1", file=sys.stderr)
        return 2
    try:
        return _COMMANDS[args.command](args)
    except (QuarryIndexError, AnswerError, NotADirectoryError, ValueError) as exc:
        print(f"quarry: error: {exc}", file=sys.stderr)
        return 2
