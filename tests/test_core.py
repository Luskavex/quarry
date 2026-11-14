import math

import pytest

from quarry.bm25 import BM25
from quarry.documents import chunk_text, iter_documents
from quarry.env import load_dotenv
from quarry.text import split_sentences, tokenize

# --- text ------------------------------------------------------------------------


def test_tokenize_lowercases_drops_stop_words_and_plurals():
    assert tokenize("The Neural Networks learn quickly!") == ["neural", "network", "learn", "quickly"]


def test_tokenize_keeps_words_ending_in_ss_and_is():
    assert tokenize("class analysis status") == ["class", "analysis", "status"]


def test_tokenize_handles_possessives_numbers_and_unicode():
    assert tokenize("BM25's k1 is 1.5 — öğrenme oranı") == ["bm25", "k1", "1", "5", "öğrenme", "oranı"]


def test_split_sentences():
    text = "First sentence. Second one?\nThird line continues\nhere. 4 is a number."
    assert split_sentences(text) == ["First sentence.", "Second one?", "Third line continues here.", "4 is a number."]


# --- BM25 ------------------------------------------------------------------------


def test_bm25_matches_hand_computed_score():
    docs = [["apple", "banana"], ["apple", "apple", "cherry"], ["durian"]]
    bm25 = BM25(docs, k1=1.5, b=0.75)

    n, df = 3, 2
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    avgdl = (2 + 3 + 1) / 3
    expected = idf * 2 * 2.5 / (2 + 1.5 * (1 - 0.75 + 0.75 * 3 / avgdl))

    assert bm25.scores(["apple"])[1] == pytest.approx(expected)


def test_bm25_rare_terms_outweigh_common_ones():
    docs = [["common", "rare"], ["common"], ["common"], ["common"]]
    ranking = BM25(docs).top_k(["common", "rare"], k=4)
    assert ranking[0][0] == 0


def test_bm25_term_frequency_saturates():
    bm25 = BM25([["x"] * 1, ["x"] * 10, ["y"] * 10], k1=1.2, b=0.0)
    one, ten = bm25.scores(["x"])[0], bm25.scores(["x"])[1]
    assert ten > one
    assert ten < one * (1.2 + 1)  # bounded by (k1 + 1) * idf


def test_bm25_unknown_terms_and_empty_index():
    assert BM25([["a"]]).top_k(["zzz"]) == []
    assert BM25([]).top_k(["a"]) == []


def test_bm25_rejects_bad_parameters():
    with pytest.raises(ValueError):
        BM25([["a"]], b=2.0)
    with pytest.raises(ValueError):
        BM25([["a"]]).top_k(["a"], k=0)


# --- chunking --------------------------------------------------------------------

NOTE = """# Title

Intro paragraph one.

## Part A

Alpha line one
alpha line two.

Beta paragraph.

## Part B

```python
def f():

    return 1
```
"""


def test_chunks_follow_sections_and_track_lines():
    chunks = chunk_text(NOTE, "note.md", max_words=50)
    assert [(c.section, c.start_line, c.end_line) for c in chunks] == [
        ("Title", 3, 3),
        ("Title > Part A", 7, 10),
        ("Title > Part B", 14, 18),
    ]
    assert chunks[1].text == "Alpha line one\nalpha line two.\n\nBeta paragraph."


def test_code_blocks_are_not_split_and_hashes_inside_are_not_headings():
    text = "# Doc\n\n```\n# not a heading\n\nstill code\n```\n"
    (chunk,) = chunk_text(text, "doc.md")
    assert chunk.section == "Doc"
    assert "# not a heading" in chunk.text and "still code" in chunk.text


def test_chunks_respect_word_limit():
    text = "\n\n".join(" ".join(["word"] * 30) for _ in range(5))
    chunks = chunk_text(text, "long.txt", max_words=60)
    assert [c.text.count("word") for c in chunks] == [60, 60, 30]


def test_iter_documents_filters_extensions_and_hidden(tmp_path):
    for name in ["a.md", "b.txt", "c.py", ".hidden.md", ".quarry/index.md", "sub/d.MD", ".git/e.md"]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
    found = sorted(p.relative_to(tmp_path).as_posix() for p in iter_documents(tmp_path))
    assert found == ["a.md", "b.txt", "sub/d.MD"]


# --- .env ------------------------------------------------------------------------


def test_load_dotenv(tmp_path, monkeypatch):
    monkeypatch.delenv("QUARRY_TEST_A", raising=False)
    monkeypatch.delenv("QUARRY_TEST_B", raising=False)
    monkeypatch.setenv("QUARRY_TEST_C", "from-shell")
    env = tmp_path / ".env"
    env.write_text(
        '# comment\nQUARRY_TEST_A=plain # trailing comment\nexport QUARRY_TEST_B="quoted value"\n'
        "QUARRY_TEST_C=from-file\nnot a valid line\n"
    )
    assert load_dotenv(env) == ["QUARRY_TEST_A", "QUARRY_TEST_B"]
    import os

    assert os.environ["QUARRY_TEST_A"] == "plain"
    assert os.environ["QUARRY_TEST_B"] == "quoted value"
    assert os.environ["QUARRY_TEST_C"] == "from-shell"


def test_load_dotenv_missing_file(tmp_path):
    assert load_dotenv(tmp_path / ".env") == []
