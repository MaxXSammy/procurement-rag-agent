import json

import pytest

from procurement_agent.config import DOCS_DIR, GOLD_DIR
from procurement_agent.documents import chunke_alle, lade_dokumente
from procurement_agent.llm import LLMAntwort, ToolCall
from procurement_agent.retrieval import TfidfEmbedder, VectorStore


@pytest.fixture(scope="session")
def dokumente():
    return lade_dokumente(DOCS_DIR)


@pytest.fixture(scope="session")
def store(dokumente):
    return VectorStore(chunke_alle(dokumente), TfidfEmbedder())


@pytest.fixture(scope="session")
def gold_profile():
    return {p["slug"]: p for p in json.loads((GOLD_DIR / "profile.json").read_text(encoding="utf-8"))}


class FakeLLM:
    """Spielt vorbereitete Antworten ab und merkt sich die Aufrufe."""

    name = "fake"

    def __init__(self, antworten):
        self.antworten = list(antworten)
        self.aufrufe = []

    def chat(self, nachrichten, schema=None, tools=None):
        self.aufrufe.append({"nachrichten": nachrichten, "schema": schema, "tools": tools})
        antwort = self.antworten.pop(0)
        return antwort if isinstance(antwort, LLMAntwort) else LLMAntwort(antwort)


@pytest.fixture
def fake_llm():
    return FakeLLM


__all__ = ["FakeLLM", "LLMAntwort", "ToolCall"]
