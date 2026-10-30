"""Agent mit Tool Calling: entscheidet selbst, ob er Dokumente durchsucht, ein Ranking
erstellt oder ein Lieferantenprofil abruft.

Mit Ollama wählt das Modell die Werkzeuge. Ohne Modell übernimmt ein einfacher
Regel-Router dieselbe Aufgabe, damit die Oberfläche auch offline funktioniert.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field

from procurement_agent.config import MAX_AGENT_SCHRITTE
from procurement_agent.extraction import Lieferantenprofil
from procurement_agent.llm import LLMProvider
from procurement_agent.rag import beantworte, formatiere_kontext, pruefe_zahlen
from procurement_agent.retrieval import Treffer, VectorStore, erkenne_lieferanten
from procurement_agent.scoring import Bewertung

AGENT_PROMPT = """Du bist ein Einkaufsassistent für die Ausschreibung einer Antriebswelle.
Nutze die Werkzeuge, bevor du antwortest:
- dokumente_durchsuchen für Fakten aus Angeboten, Selbstauskünften, Verträgen und Richtlinien
- lieferanten_ranking für Empfehlungen und Vergleiche
- lieferantenprofil für alle Kennzahlen eines einzelnen Lieferanten
Antworte knapp auf Deutsch. Gib bei Dokumentfakten die Quelle als [1], [2] an.
Erfinde keine Zahlen."""

TOOLS = [
    {"type": "function", "function": {
        "name": "dokumente_durchsuchen",
        "description": "Durchsucht Angebote, Selbstauskünfte, Rahmenvertrag, Anfrage und Einkaufsrichtlinie.",
        "parameters": {"type": "object", "properties": {
            "frage": {"type": "string"},
            "lieferanten": {
                "type": "array", "items": {"type": "string"},
                "description": "Nur Dokumente dieser Lieferanten durchsuchen. Leere Liste für alle, "
                               "z. B. bei Fragen wie 'alle außer Alpen'.",
            },
        }, "required": ["frage"]},
    }},
    {"type": "function", "function": {
        "name": "lieferanten_ranking",
        "description": "Aktuelle gewichtete Rangliste aller Lieferanten inklusive ausgeschlossener mit Grund.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "lieferantenprofil",
        "description": "Alle extrahierten Kennzahlen eines Lieferanten.",
        "parameters": {"type": "object", "properties": {"lieferant": {"type": "string"}}, "required": ["lieferant"]},
    }},
]


@dataclass
class Schritt:
    werkzeug: str
    argumente: dict
    ergebnis: str


@dataclass
class AgentErgebnis:
    text: str
    schritte: list[Schritt] = field(default_factory=list)
    quellen: list[Treffer] = field(default_factory=list)
    unbelegte_zahlen: list[str] = field(default_factory=list)


@dataclass
class Werkzeuge:
    store: VectorStore
    bewertungen: Callable[[], list[Bewertung]]
    quellen: list[Treffer] = field(default_factory=list)

    def ausfuehren(self, name: str, argumente: dict) -> str:
        if name == "dokumente_durchsuchen":
            treffer = self.store.suche(argumente.get("frage", ""), lieferanten=self._filter(argumente))
            versatz = len(self.quellen)
            self.quellen.extend(treffer)
            # Nummerierung fortlaufend über mehrere Suchen, damit [n] eindeutig bleibt
            return "\n\n".join(
                f"[{versatz + i}] ({t.chunk.quelle})\n{t.chunk.text}" for i, t in enumerate(treffer, start=1)
            ) or "Keine Treffer."
        if name == "lieferanten_ranking":
            return formatiere_ranking(self.bewertungen())
        if name == "lieferantenprofil":
            return formatiere_profil(self._finde_profil(argumente.get("lieferant", "")))
        return f"Unbekanntes Werkzeug: {name}"

    def _filter(self, argumente: dict) -> list[str] | None:
        """Vom Modell genannte Lieferanten auf Slugs abbilden. None: Suche erkennt selbst."""
        namen = argumente.get("lieferanten")
        if not isinstance(namen, list):
            return None
        return erkenne_lieferanten(" ".join(str(n) for n in namen), self.store.lieferanten)

    def _finde_profil(self, name: str) -> Lieferantenprofil | None:
        slugs = erkenne_lieferanten(name, [b.profil.slug for b in self.bewertungen()])
        return next((b.profil for b in self.bewertungen() if slugs and b.profil.slug == slugs[0]), None)


def frage_agent(
    frage: str, werkzeuge: Werkzeuge, llm: LLMProvider | None = None
) -> AgentErgebnis:
    return _mit_llm(frage, werkzeuge, llm) if llm is not None else _offline(frage, werkzeuge)


def _mit_llm(frage: str, wz: Werkzeuge, llm: LLMProvider) -> AgentErgebnis:
    nachrichten = [{"role": "system", "content": AGENT_PROMPT}, {"role": "user", "content": frage}]
    schritte: list[Schritt] = []
    for _ in range(MAX_AGENT_SCHRITTE):
        antwort = llm.chat(nachrichten, tools=TOOLS)
        if not antwort.tool_calls:
            return _abschluss(antwort.inhalt.strip(), schritte, wz)
        nachrichten.append({
            "role": "assistant",
            "content": antwort.inhalt,
            "tool_calls": [{"function": {"name": c.name, "arguments": c.argumente}} for c in antwort.tool_calls],
        })
        for call in antwort.tool_calls:
            ergebnis = wz.ausfuehren(call.name, call.argumente)
            schritte.append(Schritt(call.name, call.argumente, ergebnis))
            nachrichten.append({"role": "tool", "content": ergebnis, "tool_name": call.name})
    # Schrittlimit erreicht: Modell ohne Werkzeuge zur Antwort zwingen
    antwort = llm.chat(nachrichten + [{"role": "user", "content": "Antworte jetzt mit den vorliegenden Ergebnissen."}])
    return _abschluss(antwort.inhalt.strip(), schritte, wz)


def _abschluss(text: str, schritte: list[Schritt], wz: Werkzeuge) -> AgentErgebnis:
    # Ohne Werkzeugaufruf ist der Kontext leer: Jede Zahl der Antwort gilt dann als unbelegt.
    kontext = " ".join(s.ergebnis for s in schritte)
    return AgentErgebnis(text, schritte, wz.quellen, pruefe_zahlen(text, kontext))


RANKING_WOERTER = ("empfehl", "bester", "beste", "ranking", "rangliste", "vergleich", "favorit", "günstigst")
PROFIL_WOERTER = ("profil", "kennzahlen", "überblick", "übersicht", "alles über")


def _offline(frage: str, wz: Werkzeuge) -> AgentErgebnis:
    """Regel-Router als Ersatz für das Tool Calling des Modells."""
    f = frage.lower()
    if any(w in f for w in RANKING_WOERTER):
        ergebnis = wz.ausfuehren("lieferanten_ranking", {})
        return AgentErgebnis(ergebnis, [Schritt("lieferanten_ranking", {}, ergebnis)])
    if any(w in f for w in PROFIL_WOERTER):
        ergebnis = wz.ausfuehren("lieferantenprofil", {"lieferant": frage})
        return AgentErgebnis(ergebnis, [Schritt("lieferantenprofil", {"lieferant": frage}, ergebnis)])
    antwort = beantworte(frage, wz.store)
    schritt = Schritt("dokumente_durchsuchen", {"frage": frage}, formatiere_kontext(antwort.quellen))
    return AgentErgebnis(antwort.text, [schritt], antwort.quellen, antwort.unbelegte_zahlen)


def formatiere_ranking(bewertungen: list[Bewertung]) -> str:
    zeilen = []
    for b in bewertungen:
        p = b.profil
        if b.zugelassen:
            preis = f"{p.stueckpreis_eur:.2f} EUR".replace(".", ",") if p.stueckpreis_eur is not None else "Preis fehlt"
            zeilen.append(f"{b.rang}. {p.lieferant}: Score {b.score:.2f}, {preis}, Lieferzeit {p.lieferzeit_tage} Tage")
        else:
            zeilen.append(f"Ausgeschlossen: {p.lieferant} ({'; '.join(b.ausschluss)})")
    return "\n".join(zeilen)


def formatiere_profil(profil: Lieferantenprofil | None) -> str:
    if profil is None:
        return "Lieferant nicht gefunden. Bitte den Namen angeben."
    daten = profil.model_dump(exclude={"slug", "warnungen"})
    return json.dumps(daten, ensure_ascii=False, indent=1)
