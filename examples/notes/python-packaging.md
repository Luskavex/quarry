# Python packaging

## pyproject.toml

Modern Python projects describe themselves in `pyproject.toml`. The `[build-system]` table names the
build backend, such as setuptools or hatchling. The `[project]` table holds the name, version,
dependencies and the minimum Python version. Command-line tools are declared under
`[project.scripts]`, which maps a command name to a function.

## src layout

Putting the package inside a `src/` folder means tests run against the installed package instead of
the files that happen to be in the current directory. This catches packaging mistakes early, such as
a module that was never included in the distribution.

## Virtual environments

A virtual environment is an isolated folder with its own interpreter and installed packages. Create
one with `python -m venv .venv`, activate it, and install the project in editable mode with
`pip install -e .` so code changes take effect without reinstalling.

## Secrets

API keys belong in environment variables, never in the code. A `.env` file is convenient for local
development, but it must be listed in `.gitignore` so it is never committed. Commit a `.env.example`
with placeholder values instead, so others know which variables to set.
