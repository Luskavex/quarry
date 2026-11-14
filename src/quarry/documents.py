"""Finding note files and splitting them into passages."""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_EXTENSIONS = (".md", ".markdown", ".txt", ".rst")
STATE_DIRNAME = ".quarry"
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


@dataclass(frozen=True)
class Chunk:
    """A passage of a document, small enough to rank and to quote."""

    path: str  # relative to the indexed folder, always with forward slashes
    section: str  # heading trail, e.g. "Training > Learning rate"
    start_line: int  # 1-based, inclusive
    end_line: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def location(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"


def iter_documents(root: Path, extensions: tuple[str, ...] = DEFAULT_EXTENSIONS) -> Iterator[Path]:
    """Yield note files under ``root``, skipping hidden folders and the index itself."""
    wanted = {e.lower() for e in extensions}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d != STATE_DIRNAME)
        for name in sorted(filenames):
            if not name.startswith(".") and Path(name).suffix.lower() in wanted:
                yield Path(dirpath) / name


def _paragraphs(lines: list[str]) -> Iterator[tuple[int, int, list[str], list[str]]]:
    """Yield (start_line, end_line, heading_trail, paragraph_lines) for a document."""
    trail: list[tuple[int, str]] = []
    buffer: list[str] = []
    start = 0
    in_code = False

    def flush(end: int):
        nonlocal buffer
        if buffer:
            yield start, end, [title for _, title in trail], buffer
        buffer = []

    for number, raw in enumerate(lines, start=1):
        line = raw.rstrip()
        if line.lstrip().startswith("```"):
            in_code = not in_code
        heading = None if in_code else _HEADING_RE.match(line)
        if heading:
            yield from flush(number - 1)
            level = len(heading.group(1))
            trail = [(lvl, title) for lvl, title in trail if lvl < level] + [(level, heading.group(2))]
            continue
        if not line.strip() and not in_code:
            yield from flush(number - 1)
            continue
        if not buffer:
            start = number
        buffer.append(line)
    yield from flush(len(lines))


def chunk_text(text: str, path: str, max_words: int = 180) -> list[Chunk]:
    """Group paragraphs of the same section into chunks of at most ``max_words`` words.

    A single paragraph longer than the limit becomes its own chunk rather than
    being cut mid-sentence.
    """
    if max_words < 10:
        raise ValueError("max_words must be at least 10")

    chunks: list[Chunk] = []
    current: list[str] = []
    current_words = 0
    current_section = ""
    start = end = 0

    def emit():
        nonlocal current, current_words
        if current:
            chunks.append(Chunk(path, current_section, start, end, "\n".join(current).strip()))
        current, current_words = [], 0

    for p_start, p_end, trail, para in _paragraphs(text.splitlines()):
        section = " > ".join(trail)
        words = sum(len(line.split()) for line in para)
        if current and (section != current_section or current_words + words > max_words):
            emit()
        if not current:
            start, current_section = p_start, section
        current.extend([*para, ""])
        current_words += words
        end = p_end
    emit()
    return chunks


def load_chunks(file: Path, root: Path, max_words: int = 180) -> list[Chunk]:
    text = file.read_text(encoding="utf-8", errors="replace")
    return chunk_text(text, file.relative_to(root).as_posix(), max_words=max_words)
