# quarry

**Local search over your notes, with answers that cite their sources.**

`quarry` indexes a folder of Markdown or text notes, finds the passages that best match a question
with [BM25](https://en.wikipedia.org/wiki/Okapi_BM25), and answers from those passages only, citing
each one by file and line. It is a small, readable implementation of the retrieval half of
retrieval-augmented generation (RAG), with two ways to produce the answer:

- **offline (default without an API key)**: picks the best-matching sentences from the retrieved
  passages. No model, no network, nothing leaves your machine.
- **Claude**: sends the question and only the retrieved passages to the Claude API and streams back
  an answer with `[n]` citations.

The search engine itself (tokenizer, chunker, BM25 with an inverted index) is written from scratch
in the standard library.

## Installation

```bash
git clone https://github.com/Luskavex/quarry.git
cd quarry
python -m pip install .              # offline search and extractive answers
python -m pip install ".[claude]"    # adds the Anthropic SDK for the Claude backend
```

Requires Python 3.10+.

## Usage

The repository includes three short example notes in [`examples/notes`](examples/notes) to try it on.

### 1. Index a folder

```console
$ quarry index examples/notes
Indexed 3 file(s) as 12 passage(s).
Index written to .quarry/index.json
```

Notes are split on Markdown headings and paragraphs into passages of about 180 words
(`--max-words`), and each passage remembers its file, line range and heading trail. The index is a
single JSON file inside the indexed folder.

### 2. Search

```console
$ quarry search "learning rate too large" -C examples/notes -k 3
[1] gradient-descent.md:9-14  score 5.93  Gradient descent > Learning rate
    The learning rate is the size of each step. If it is too large, the loss oscillates or diverges
    because every step overshoots the minimum. If it is too small, training is stable but very slow.
    A common practice is to sta...

[2] gradient-descent.md:3-5  score 3.38  Gradient descent
    ...ative gradient. Each update is `w = w - learning_rate * gradient`. The gradient points
    uphill, so moving against it lowers the loss, at least for a small enough step.

[3] retrieval.md:26-28  score 1.83  Retrieval for question answering > Chunking
    ...ks before indexing. Chunks that are too large dilute the relevant sentence with unrelated
    text; chunks that are too small lose context. Splitting on headings and paragraphs and keeping
    chunks to a few hundred words is a ...
```

`-C/--root` points at the indexed folder (default: the current directory).

### 3. Ask

Without an API key, `ask` uses the offline extractive answerer:

```console
$ quarry ask "What happens if the learning rate is too large?" -C examples/notes -k 3
(no ANTHROPIC_API_KEY found; using the offline extractive answerer)

The learning rate is the size of each step. [1] If it is too large, the loss oscillates or diverges
because every step overshoots the minimum. [1] Learning rate schedules reduce the step size during
training. [1]

Sources (extractive):
  [1] gradient-descent.md:9-14  (Gradient descent > Learning rate)
  [2] gradient-descent.md:3-5  (Gradient descent)
  [3] python-packaging.md:12-14  (Python packaging > src layout)
```

With a key, it asks Claude to write the answer from the same passages:

```bash
cp .env.example .env        # then put your key in .env (never commit it)
quarry ask "What happens if the learning rate is too large?" -C examples/notes
```

Options for `ask`:

| Option | Default | Meaning |
| --- | --- | --- |
| `--backend` | `auto` | `auto` uses Claude when `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN` is set, otherwise `extractive`. `claude` also works with any other credential source the Anthropic SDK supports. |
| `--model` | `$QUARRY_MODEL` or `claude-opus-5-5` | Claude model to use. |
| `--effort` | `$QUARRY_EFFORT` or `medium` | How much the model thinks before answering (`low` to `max`). |
| `-k` | `5` | Number of passages to retrieve. |

## Configuration and secrets

`quarry` reads configuration from environment variables. For convenience it also loads a `.env`
file from the directory you run it in; variables already set in your shell take priority.
[`.env.example`](.env.example) lists every variable with placeholder values. `.env` is in
`.gitignore`; keep it that way.

## Privacy

- Indexing, search and extractive answers run entirely on your machine.
- With the Claude backend, the question and the `k` retrieved passages are sent to the Anthropic
  API. The rest of your notes are never sent. A test checks this.

## How it works

```
src/quarry/
├── text.py        # tokenizer: lowercase, split, stop words, plural stripping
├── documents.py   # finds note files, splits them into passages with line ranges and headings
├── bm25.py        # Okapi BM25 over an inverted index
├── index.py       # builds, saves, loads and searches the index
├── answer.py      # extractive answerer, Claude answerer, prompt construction
├── env.py         # minimal .env loader
└── cli.py
```

**Ranking.** For each query term, BM25 adds
`idf(t) · tf · (k1 + 1) / (tf + k1 · (1 − b + b · |d| / avgdl))`, with `k1 = 1.5` and `b = 0.75`.
Rare terms count more (`idf`), repeated terms saturate (`k1`), and long passages are normalized by
length (`b`). Each passage's heading trail is indexed along with its text, so headings help ranking.

**Extractive answers.** Every sentence in the retrieved passages is scored by the idf of the query
terms it contains, plus half the score of the sentence before it (so "If it is too large, ..." keeps
the context of the previous sentence), weighted by how well its passage ranked. The top sentences
above a relative threshold are returned in reading order with citations.

**Claude answers.** The retrieved passages are sent as numbered `<source>` blocks with a system
prompt that requires citing every claim and saying so when the notes do not cover the question. The
response is streamed. Requests use adaptive thinking (the model default) with a configurable effort
level, and opt in to server-side fallback so a request declined by a safety classifier can be
retried on a fallback model.

## Limitations

- BM25 matches words, not meaning: a question about "cars" will not find a note about "automobiles".
  Dense or hybrid retrieval would fix that and is the natural next step.
- Plural stripping is the only normalization; there is no full stemming or lemmatization.
- The extractive answerer only quotes sentences; it does not combine or rephrase them.
- Re-indexing rebuilds the whole index. `search` and `ask` warn when files changed since indexing.

## Tests

```bash
python -m pip install -e ".[dev]"
python -m pytest
ruff check .
```

The tests check BM25 against a hand-computed score, chunking and line tracking, ranking on the
example notes, index persistence, the extractive answerer, and the Claude backend against a fake
client (streaming, request parameters, refusal handling, and that only retrieved passages are sent).
They never call the real API.

## License

[MIT](LICENSE)
