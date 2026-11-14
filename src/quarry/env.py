"""A minimal ``.env`` reader, so secrets stay out of shell history and out of the code."""

from __future__ import annotations

import os
import re
from pathlib import Path

_LINE_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def load_dotenv(path: Path) -> list[str]:
    """Load ``KEY=value`` lines from ``path`` into ``os.environ``.

    Variables that are already set win over the file. Blank lines and lines
    starting with ``#`` are ignored; values may be wrapped in single or double
    quotes. Returns the names of the variables that were set.
    """
    if not path.is_file():
        return []
    loaded = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = _LINE_RE.match(line)
        if not match:
            continue
        key, value = match.groups()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded
