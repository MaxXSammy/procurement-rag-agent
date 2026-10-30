"""LLM-Abstraktion. Der Rest des Codes kennt nur das Protokoll, nicht Ollama.

So lassen sich Modelle per Konfiguration tauschen und Tests mit Fakes schreiben.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol

from procurement_agent.config import LLM_MODEL, OLLAMA_HOST


@dataclass
class ToolCall:
    name: str
    argumente: dict[str, Any]


@dataclass
class LLMAntwort:
    inhalt: str
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMProvider(Protocol):
    name: str

    def chat(
        self,
        nachrichten: list[dict],
        schema: dict | None = None,
        tools: list[dict] | None = None,
    ) -> LLMAntwort: ...


class OllamaProvider:
    def __init__(self, modell: str = LLM_MODEL, host: str = OLLAMA_HOST) -> None:
        import ollama

        self.name = f"ollama:{modell}"
        self._modell = modell
        self._client = ollama.Client(host=host)

    def chat(self, nachrichten, schema=None, tools=None) -> LLMAntwort:
        antwort = self._client.chat(
            model=self._modell,
            messages=nachrichten,
            format=schema,
            tools=tools,
            options={"temperature": 0},
        )
        msg = antwort["message"]
        calls = [
            ToolCall(c["function"]["name"], dict(c["function"]["arguments"] or {}))
            for c in (msg.get("tool_calls") or [])
        ]
        return LLMAntwort(msg.get("content") or "", calls)


def ollama_verfuegbar(host: str = OLLAMA_HOST, modell: str | None = None) -> bool:
    """True, wenn der Ollama-Server läuft (und optional das Modell installiert ist)."""
    try:
        import ollama

        modelle = ollama.Client(host=host, timeout=2).list()["models"]
    except Exception:
        return False
    if modell is None:
        return True
    namen = {m["model"] for m in modelle}
    return modell in namen or f"{modell}:latest" in namen
