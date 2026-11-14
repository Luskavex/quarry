import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from quarry.answer import AnswerError, ClaudeAnswerer, ExtractiveAnswerer, build_prompt
from quarry.cli import main
from quarry.index import Index, QuarryIndexError

NOTES = Path(__file__).resolve().parent.parent / "examples" / "notes"


@pytest.fixture
def notes(tmp_path):
    root = tmp_path / "notes"
    shutil.copytree(NOTES, root)
    return root


@pytest.fixture
def index(notes):
    return Index.build(notes)


# --- index -------------------------------------------------------------------------


def test_search_ranks_the_relevant_section_first(index):
    top = index.search("what happens when the learning rate is too large", k=3)[0]
    assert top.chunk.path == "gradient-descent.md"
    assert top.chunk.section == "Gradient descent > Learning rate"


@pytest.mark.parametrize(
    ("query", "section"),
    [
        ("why standardize features before gradient descent", "Gradient descent > Feature scaling"),
        ("BM25 parameters k1 and b", "Retrieval for question answering > BM25"),
        ("where should API keys be stored", "Python packaging > Secrets"),
        ("hybrid search with embeddings", "Retrieval for question answering > Dense retrieval"),
    ],
)
def test_search_finds_expected_sections(index, query, section):
    assert index.search(query, k=1)[0].chunk.section == section


def test_search_with_only_stop_words_returns_nothing(index):
    assert index.search("what is the") == []


def test_save_and_load_round_trip(notes, index):
    path = index.save(notes)
    assert path == notes / ".quarry" / "index.json"
    loaded = Index.load(notes)
    assert loaded.chunks == index.chunks
    query = "mini-batch momentum adam"
    assert [h.chunk for h in loaded.search(query)] == [h.chunk for h in index.search(query)]


def test_load_errors(tmp_path):
    with pytest.raises(QuarryIndexError, match="no index"):
        Index.load(tmp_path)
    (tmp_path / ".quarry").mkdir()
    (tmp_path / ".quarry" / "index.json").write_text("{not json")
    with pytest.raises(QuarryIndexError, match="corrupted"):
        Index.load(tmp_path)


def test_stale_files_detected(notes, index):
    index.save(notes)
    loaded = Index.load(notes)
    assert loaded.stale_files(notes) == []
    later = time.time() + 5
    os.utime(notes / "retrieval.md", (later, later))
    assert loaded.stale_files(notes) == ["retrieval.md"]


# --- answers -----------------------------------------------------------------------


def test_build_prompt_numbers_sources(index):
    hits = index.search("learning rate", k=2)
    prompt = build_prompt("What is a learning rate?", hits)
    assert prompt.count("<source id=") == 2
    assert '<source id="1" path="gradient-descent.md"' in prompt
    assert prompt.endswith("Question: What is a learning rate?")


def test_extractive_answer_cites_relevant_sentences(index):
    question = "What happens if the learning rate is too large?"
    answer = ExtractiveAnswerer(index).answer(question, index.search(question, k=3))
    assert "oscillates or diverges" in answer.text
    assert "[1]" in answer.text
    assert answer.backend == "extractive"


def test_extractive_answer_without_matches(index):
    answer = ExtractiveAnswerer(index).answer("quantum chromodynamics", [])
    assert "do not seem to cover" in answer.text


class FakeStream:
    def __init__(self, chunks, stop_reason="end_turn"):
        self.text_stream = iter(chunks)
        self._final = SimpleNamespace(
            stop_reason=stop_reason,
            model="claude-opus-5-5",
            content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text="".join(chunks))],
        )

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._final


class FakeClient:
    def __init__(self, stream):
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))
        self._stream_obj = stream

    def _stream(self, **kwargs):
        self.calls.append(kwargs)
        return self._stream_obj


def test_claude_answerer_streams_and_sends_only_retrieved_passages(index):
    client = FakeClient(FakeStream(["Too large a rate ", "makes training diverge [1]."]))
    hits = index.search("learning rate too large", k=2)
    received = []

    answer = ClaudeAnswerer(client=client).answer("What if the learning rate is too large?", hits, received.append)

    assert answer.text == "Too large a rate makes training diverge [1]."
    assert received == ["Too large a rate ", "makes training diverge [1]."]
    (call,) = client.calls
    assert call["model"] == "claude-opus-5-5"
    assert call["output_config"] == {"effort": "medium"}
    assert call["fallbacks"] == "default"
    assert "temperature" not in call
    user_message = call["messages"][0]["content"]
    assert user_message.count("<source id=") == 2
    assert "Secrets" not in user_message  # unrelated notes are never sent


def test_claude_answerer_reports_refusal(index):
    client = FakeClient(FakeStream([], stop_reason="refusal"))
    with pytest.raises(AnswerError, match="declined"):
        ClaudeAnswerer(client=client).answer("question", index.search("learning rate"))


def test_claude_answerer_skips_api_when_nothing_retrieved():
    client = FakeClient(FakeStream(["unused"]))
    answer = ClaudeAnswerer(client=client).answer("anything", [])
    assert client.calls == []
    assert "do not seem to cover" in answer.text


# --- CLI ---------------------------------------------------------------------------


def test_cli_index_search_and_ask(notes, capsys, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.chdir(notes)

    assert main(["index", str(notes)]) == 0
    assert "Indexed 3 file(s)" in capsys.readouterr().out

    assert main(["search", "feature scaling", "-k", "2"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("[1] gradient-descent.md:")

    assert main(["ask", "Why should features be standardized?"]) == 0
    captured = capsys.readouterr()
    assert "extractive" in captured.err
    assert "Sources (extractive):" in captured.out


def test_cli_without_index(tmp_path, capsys):
    assert main(["search", "x", "--root", str(tmp_path)]) == 2
    assert "no index found" in capsys.readouterr().err


def test_cli_rejects_bad_k(notes, capsys):
    assert main(["search", "x", "--root", str(notes), "-k", "0"]) == 2
