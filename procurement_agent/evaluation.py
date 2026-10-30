"""Erfolgsmessung: Qualität des Systems und Nutzen für das Unternehmen.

Technische Kennzahlen (gegen Gold-Daten):
  - Retrieval-Trefferquote: Anteil der erwarteten Quelldokumente unter den Top-k
  - Antwortgenauigkeit: Anteil der Antworten mit allen erwarteten Fakten
  - Belegquote: Anteil der Antworten, deren Zahlen alle in den Quellen stehen
  - Extraktionsgenauigkeit: Anteil korrekt extrahierter Profilfelder

Geschäftskennzahlen (aus Bewertung und Annahmen in business.json):
  - Einsparpotenzial pro Jahr gegenüber dem Bestandslieferanten
  - Zeitersparnis gegenüber manueller Angebotsauswertung
  - erkannte Risiken (K.-o.-Kriterien, Preisgleitklauseln, fehlende Angaben)
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from procurement_agent.config import GOLD_DIR, HISTORY_FILE
from procurement_agent.extraction import WERTFELDER, Lieferantenprofil
from procurement_agent.llm import LLMProvider
from procurement_agent.rag import beantworte
from procurement_agent.retrieval import VectorStore
from procurement_agent.scoring import Bewertung


# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------
@dataclass
class FragenErgebnis:
    id: str
    frage: str
    antwort: str
    retrieval_recall: float
    korrekt: bool
    belegt: bool
    dauer_s: float


@dataclass
class RagBericht:
    ergebnisse: list[FragenErgebnis]

    @property
    def retrieval_trefferquote(self) -> float:
        return _mittel(e.retrieval_recall for e in self.ergebnisse)

    @property
    def antwortgenauigkeit(self) -> float:
        return _mittel(e.korrekt for e in self.ergebnisse)

    @property
    def belegquote(self) -> float:
        return _mittel(e.belegt for e in self.ergebnisse)

    @property
    def mittlere_dauer_s(self) -> float:
        return _mittel(e.dauer_s for e in self.ergebnisse)


def lade_fragen(datei: Path = GOLD_DIR / "fragen.json") -> list[dict]:
    return json.loads(datei.read_text(encoding="utf-8"))


def ist_korrekt(antwort: str, erwartet: list[list[str]]) -> bool:
    """Jede Gruppe muss mit mindestens einem Stichwort in der Antwort vorkommen."""
    a = _norm(antwort)
    return all(any(_norm(stichwort) in a for stichwort in gruppe) for gruppe in erwartet)


def evaluiere_rag(store: VectorStore, llm: LLMProvider | None, fragen: list[dict]) -> RagBericht:
    ergebnisse = []
    for f in fragen:
        a = beantworte(f["frage"], store, llm)
        gefunden = {t.chunk.quelle for t in a.quellen}
        recall = sum(q in gefunden for q in f["quellen"]) / len(f["quellen"])
        ergebnisse.append(FragenErgebnis(
            f["id"], f["frage"], a.text, recall, ist_korrekt(a.text, f["erwartet"]), a.belegt, a.dauer_s,
        ))
    return RagBericht(ergebnisse)


# ---------------------------------------------------------------------------
# Extraktion
# ---------------------------------------------------------------------------
@dataclass
class ExtraktionsBericht:
    genauigkeit: float
    je_feld: dict[str, float]
    fehler: list[dict] = field(default_factory=list)


def lade_gold_profile(datei: Path = GOLD_DIR / "profile.json") -> dict[str, dict]:
    return {p["slug"]: p for p in json.loads(datei.read_text(encoding="utf-8"))}


def evaluiere_extraktion(profile: list[Lieferantenprofil], gold: dict[str, dict]) -> ExtraktionsBericht:
    treffer: dict[str, list[bool]] = {f: [] for f in WERTFELDER}
    fehler = []
    for p in profile:
        g = gold.get(p.slug)
        if g is None:
            continue
        for feld in WERTFELDER:
            ok = _gleich(getattr(p, feld), g[feld])
            treffer[feld].append(ok)
            if not ok:
                fehler.append({"lieferant": p.lieferant, "feld": feld, "extrahiert": getattr(p, feld), "soll": g[feld]})
    alle = [ok for werte in treffer.values() for ok in werte]
    return ExtraktionsBericht(_mittel(alle), {f: _mittel(v) for f, v in treffer.items()}, fehler)


def _gleich(ist, soll, toleranz: float = 0.01) -> bool:
    if ist is None or soll is None:
        return ist is soll
    if isinstance(soll, bool):
        return ist is soll
    return abs(float(ist) - float(soll)) <= toleranz * max(abs(float(soll)), 1.0)


# ---------------------------------------------------------------------------
# Geschäftskennzahlen
# ---------------------------------------------------------------------------
@dataclass
class GeschaeftsKPIs:
    empfehlung: str | None
    preis_empfehlung: float | None
    bestandslieferant: str
    preis_bestand: float | None
    einsparung_eur_jahr: float | None
    stunden_manuell: float
    stunden_mit_tool: float
    stunden_gespart_jahr: float
    wert_zeitersparnis_eur_jahr: float
    risiken: list[str]


def geschaeftskpis(bewertungen: list[Bewertung], business: dict, laufzeit_s: float) -> GeschaeftsKPIs:
    zugelassen = [b for b in bewertungen if b.zugelassen]
    top = zugelassen[0].profil if zugelassen else None
    bestand = next((b.profil for b in bewertungen if b.profil.lieferant == business["bestandslieferant"]), None)

    einsparung = None
    if top and bestand and top.stueckpreis_eur is not None and bestand.stueckpreis_eur is not None:
        einsparung = (bestand.stueckpreis_eur - top.stueckpreis_eur) * business["jahresbedarf_stueck"]

    n = len(bewertungen)
    manuell_h = (n * business["manuell_minuten_pro_angebot"] + business["manuell_minuten_bewertung"]) / 60
    # Werkzeuglauf plus 15 Minuten fachliche Prüfung der Empfehlung durch den Einkauf
    tool_h = laufzeit_s / 3600 + 0.25
    gespart_jahr = max(manuell_h - tool_h, 0) * business["ausschreibungen_pro_jahr"]

    risiken = [f"{b.profil.lieferant}: {g}" for b in bewertungen for g in b.ausschluss]
    risiken += [f"{b.profil.lieferant}: {h}" for b in bewertungen for h in b.hinweise]
    return GeschaeftsKPIs(
        empfehlung=top.lieferant if top else None,
        preis_empfehlung=top.stueckpreis_eur if top else None,
        bestandslieferant=business["bestandslieferant"],
        preis_bestand=bestand.stueckpreis_eur if bestand else None,
        einsparung_eur_jahr=einsparung,
        stunden_manuell=manuell_h,
        stunden_mit_tool=tool_h,
        stunden_gespart_jahr=gespart_jahr,
        wert_zeitersparnis_eur_jahr=gespart_jahr * business["stundensatz_einkauf_eur"],
        risiken=risiken,
    )


# ---------------------------------------------------------------------------
# Verlauf: jede Evaluation wird protokolliert, damit Verbesserungen sichtbar werden
# ---------------------------------------------------------------------------
def speichere_lauf(eintrag: dict, datei: Path = HISTORY_FILE) -> dict:
    eintrag = {"zeitpunkt": datetime.now().isoformat(timespec="seconds"), **eintrag}
    datei.parent.mkdir(parents=True, exist_ok=True)
    with datei.open("a", encoding="utf-8") as f:
        f.write(json.dumps(eintrag, ensure_ascii=False) + "\n")
    return eintrag


def lade_verlauf(datei: Path = HISTORY_FILE) -> list[dict]:
    if not datei.exists():
        return []
    return [json.loads(z) for z in datei.read_text(encoding="utf-8").splitlines() if z.strip()]


@dataclass
class Gesamtlauf:
    rag: RagBericht
    extraktion: ExtraktionsBericht
    dauer_s: float
    eintrag: dict


def gesamtevaluation(system, speichern: bool = True) -> Gesamtlauf:
    """Führt RAG- und Extraktions-Evaluation aus und protokolliert die Kennzahlen."""
    start = time.perf_counter()
    rag = evaluiere_rag(system.store, system.llm, lade_fragen())
    extraktion = evaluiere_extraktion(system.profile(), lade_gold_profile())
    dauer = time.perf_counter() - start
    eintrag = {
        "modus": system.beschreibung,
        "retrieval_trefferquote": round(rag.retrieval_trefferquote, 3),
        "antwortgenauigkeit": round(rag.antwortgenauigkeit, 3),
        "belegquote": round(rag.belegquote, 3),
        "extraktionsgenauigkeit": round(extraktion.genauigkeit, 3),
        "mittlere_antwortzeit_s": round(rag.mittlere_dauer_s, 3),
        "fragen": len(rag.ergebnisse),
    }
    if speichern:
        eintrag = speichere_lauf(eintrag)
    return Gesamtlauf(rag, extraktion, dauer, eintrag)


def _mittel(werte) -> float:
    werte = [float(w) for w in werte]
    return sum(werte) / len(werte) if werte else 0.0


def _norm(text: str) -> str:
    return " ".join(text.lower().split())

