"""Setzt alle Bausteine zusammen. Oberfläche und Kommandozeile nutzen nur diese Klasse."""

import json
from dataclasses import dataclass, field

from procurement_agent import config
from procurement_agent.agent import AgentErgebnis, Werkzeuge, frage_agent
from procurement_agent.documents import Dokument, chunke_alle, lade_dokumente
from procurement_agent.extraction import Extractor, Lieferantenprofil, LLMExtractor, RegelExtractor, extrahiere_alle
from procurement_agent.llm import LLMProvider, OllamaProvider, ollama_verfuegbar
from procurement_agent.retrieval import Embedder, OllamaEmbedder, TfidfEmbedder, VectorStore
from procurement_agent.scoring import STANDARD_KRITERIEN, Bewertung, KoRegeln, Kriterium, bewerte


@dataclass
class System:
    dokumente: list[Dokument]
    store: VectorStore
    llm: LLMProvider | None
    extractor: Extractor
    kriterien: list[Kriterium] = field(default_factory=lambda: list(STANDARD_KRITERIEN))
    regeln: KoRegeln = field(default_factory=KoRegeln)
    _profile: list[Lieferantenprofil] | None = None

    @property
    def beschreibung(self) -> str:
        llm = self.llm.name if self.llm else "offline"
        return f"LLM: {llm} · Embeddings: {self.store.embedder.name} · Extraktion: {self.extractor.name}"

    def profile(self) -> list[Lieferantenprofil]:
        if self._profile is None:
            self._profile = extrahiere_alle(self.extractor, self.dokumente)
        return self._profile

    def bewertungen(self, profile: list[Lieferantenprofil] | None = None) -> list[Bewertung]:
        return bewerte(profile if profile is not None else self.profile(), self.kriterien, self.regeln)

    def frage(self, frage: str) -> AgentErgebnis:
        return frage_agent(frage, Werkzeuge(self.store, self.bewertungen), self.llm)


def baue_system(modus: str = config.MODUS, extraktion: str = "auto") -> System:
    """modus: auto | ollama | offline. extraktion: auto | regel | llm."""
    nutze_ollama = modus == "ollama" or (modus == "auto" and ollama_verfuegbar(modell=config.LLM_MODEL))
    llm: LLMProvider | None = OllamaProvider() if nutze_ollama else None

    embedder: Embedder = TfidfEmbedder()
    if nutze_ollama and ollama_verfuegbar(modell=config.EMBED_MODEL):
        embedder = OllamaEmbedder()

    if extraktion == "llm" or (extraktion == "auto" and llm is not None):
        if llm is None:
            raise RuntimeError("LLM-Extraktion braucht einen laufenden Ollama-Server.")
        extractor: Extractor = LLMExtractor(llm)
    else:
        extractor = RegelExtractor()

    dokumente = lade_dokumente(config.DOCS_DIR)
    return System(dokumente, VectorStore(chunke_alle(dokumente), embedder), llm, extractor)


def lade_business() -> dict:
    return json.loads(config.BUSINESS_FILE.read_text(encoding="utf-8"))
