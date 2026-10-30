"""Embeddings und Vektorsuche.

Zwei austauschbare Embedder:
  - OllamaEmbedder: semantische Embeddings über ein lokales Modell (nomic-embed-text)
  - TfidfEmbedder: Zeichen-n-Gramme, läuft ohne KI und ohne GPU (Offline-Modus, Tests)

Der Vektorspeicher ist bewusst eine einfache NumPy-Matrix: Bei einigen hundert Chunks
ist das schneller als jede Datenbank und hält das Projekt ohne Infrastruktur lauffähig.
Für größere Bestände lässt sich ein VectorStore mit Chroma oder Qdrant nachrüsten.
"""

import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from procurement_agent.config import EMBED_MODEL, LLM_TIMEOUT_S, OLLAMA_HOST, TOP_K
from procurement_agent.documents import Chunk


class Embedder(Protocol):
    name: str

    def fit(self, texte: list[str]) -> None: ...

    def embed(self, texte: list[str]) -> np.ndarray: ...


class TfidfEmbedder:
    """Zeichen-n-Gramme (3–5) treffen auch Komposita: 'Preis' findet 'Stückpreis'."""

    name = "tfidf-char"

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True, sublinear_tf=True)

    def fit(self, texte: list[str]) -> None:
        self._vec.fit(texte)

    def embed(self, texte: list[str]) -> np.ndarray:
        return _normalisiere(self._vec.transform(texte).toarray())


class OllamaEmbedder:
    def __init__(self, modell: str = EMBED_MODEL, host: str = OLLAMA_HOST) -> None:
        import ollama

        self.name = f"ollama:{modell}"
        self._modell = modell
        self._client = ollama.Client(host=host, timeout=LLM_TIMEOUT_S)

    def fit(self, texte: list[str]) -> None:  # vortrainiert, nichts zu tun
        pass

    def embed(self, texte: list[str]) -> np.ndarray:
        antwort = self._client.embed(model=self._modell, input=texte)
        return _normalisiere(np.asarray(antwort["embeddings"], dtype=np.float32))


@dataclass(frozen=True)
class Treffer:
    chunk: Chunk
    score: float


class VectorStore:
    def __init__(self, chunks: list[Chunk], embedder: Embedder) -> None:
        self.chunks = chunks
        self.embedder = embedder
        texte = [c.text for c in chunks]
        embedder.fit(texte)
        self._matrix = embedder.embed(texte)
        self.lieferanten = sorted({c.lieferant for c in chunks if c.lieferant})

    def suche(self, frage: str, k: int = TOP_K, lieferanten: list[str] | None = None) -> list[Treffer]:
        """Kosinus-Ähnlichkeit. Nennt die Frage Lieferanten, wird auf deren Dokumente gefiltert.

        Eine explizite Liste hat Vorrang; eine leere Liste schaltet den Filter ab.
        """
        if lieferanten is None:
            # "außer Alpen" würde sonst genau auf Alpen filtern: im Zweifel lieber gar nicht
            lieferanten = [] if enthaelt_verneinung(frage) else erkenne_lieferanten(frage, self.lieferanten)
        q = self.embedder.embed([frage])[0]
        scores = self._matrix @ q
        if lieferanten:
            maske = np.array([c.lieferant in lieferanten for c in self.chunks])
            scores = np.where(maske, scores, -np.inf)
        reihenfolge = np.argsort(-scores)[:k]
        return [Treffer(self.chunks[i], float(scores[i])) for i in reihenfolge if np.isfinite(scores[i])]


def erkenne_lieferanten(frage: str, bekannte: list[str]) -> list[str]:
    """Findet Lieferanten-Slugs in der Frage ('Swiss Turning' -> 'swissturning')."""
    kompakt = frage.lower().replace(" ", "").replace("-", "")
    return [s for s in bekannte if s in kompakt]


VERNEINUNG = re.compile(r"\b(außer|ausser|ohne|nicht|kein\w*|ausgenommen|abgesehen)\b", re.IGNORECASE)


def enthaelt_verneinung(frage: str) -> bool:
    return VERNEINUNG.search(frage) is not None


def _normalisiere(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    normen = np.linalg.norm(m, axis=1, keepdims=True)
    normen[normen == 0] = 1.0
    return m / normen
