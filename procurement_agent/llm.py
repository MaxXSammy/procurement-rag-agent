"""LLM-Abstraktion. Der Rest des Codes kennt nur das Protokoll, nicht Ollama.

So lassen sich Modelle per Konfiguration tauschen und Tests mit Fakes schreiben.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from procurement_agent.config import LLM_MODEL, LLM_NUM_CTX, LLM_NUM_PREDICT, LLM_TIMEOUT_S, LLM_VERSUCHE, OLLAMA_HOST

logger = logging.getLogger(__name__)


class LLMNichtErreichbar(RuntimeError):
    """Das Modell hat auch nach mehreren Versuchen nicht geantwortet."""


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
    def __init__(self, modell: str = LLM_MODEL, host: str = OLLAMA_HOST, client=None,
                 versuche: int = LLM_VERSUCHE, wartezeit_s: float = 2.0) -> None:
        import ollama

        self.name = f"ollama:{modell}"
        self._modell = modell
        self._client = client or ollama.Client(host=host, timeout=LLM_TIMEOUT_S)
        self._versuche = versuche
        self._wartezeit_s = wartezeit_s

    def chat(self, nachrichten, schema=None, tools=None) -> LLMAntwort:
        geschaetzte_tokens = sum(len(str(n.get("content", ""))) for n in nachrichten) // 3
        if geschaetzte_tokens > LLM_NUM_CTX * 0.8:
            logger.warning("Prompt nahe am Kontextfenster, Ollama könnte ihn abschneiden",
                           extra={"tokens_geschaetzt": geschaetzte_tokens, "num_ctx": LLM_NUM_CTX})
        letzter_fehler: Exception | None = None
        for versuch in range(1, self._versuche + 1):
            start = time.perf_counter()
            try:
                antwort = self._client.chat(
                    model=self._modell,
                    messages=nachrichten,
                    format=schema,
                    tools=tools,
                    options={"temperature": 0, "num_ctx": LLM_NUM_CTX, "num_predict": LLM_NUM_PREDICT},
                )
            except Exception as fehler:  # Verbindungsabbruch, Timeout, Serverfehler von Ollama
                letzter_fehler = fehler
                logger.warning("LLM-Aufruf fehlgeschlagen",
                               extra={"versuch": versuch, "fehler": type(fehler).__name__})
                if versuch < self._versuche:
                    time.sleep(self._wartezeit_s * 2 ** (versuch - 1))  # 2 s, 4 s, ...
                continue
            logger.info("LLM-Aufruf", extra={"modell": self._modell, "dauer_s": round(time.perf_counter() - start, 2)})
            break
        else:
            raise LLMNichtErreichbar(f"{self.name} nach {self._versuche} Versuchen nicht erreichbar: {letzter_fehler}")
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
