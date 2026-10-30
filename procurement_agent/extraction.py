"""Strukturierte Lieferantenprofile aus Angebot und Selbstauskunft extrahieren.

Zwei Extraktoren mit derselben Schnittstelle:
  - RegelExtractor: reguläre Ausdrücke, die Basislinie ohne KI
  - LLMExtractor: Modell mit JSON-Schema, danach Prüfung jedes Werts gegen den Quelltext

Die Evaluation vergleicht beide gegen die Gold-Profile. So lässt sich belegen,
was die KI gegenüber der Basislinie tatsächlich bringt.
"""

import json
import re
from typing import Protocol

from pydantic import BaseModel, Field

from procurement_agent.documents import Dokument
from procurement_agent.llm import LLMProvider


class Lieferantenprofil(BaseModel):
    lieferant: str
    slug: str
    stueckpreis_eur: float | None = Field(None, description="Preis für 1 Stück in EUR netto")
    mindestbestellmenge: int | None = Field(None, description="Mindestbestellmenge in Stück")
    lieferzeit_tage: int | None = Field(None, description="Lieferzeit in Tagen")
    zahlungsziel_tage: int | None = Field(None, description="Zahlungsziel in Tagen")
    skonto_prozent: float | None = Field(None, description="Skonto in Prozent, 0 wenn keines")
    preisgleitklausel: bool | None = Field(None, description="True, wenn Preise an einen Index gekoppelt sind")
    iso_9001: bool | None = Field(None, description="True nur bei gültigem ISO 9001 Zertifikat")
    iso_14001: bool | None = Field(None, description="True nur bei gültigem ISO 14001 Zertifikat")
    co2_kg_pro_stueck: float | None = Field(None, description="CO2-Fußabdruck je Stück in kg")
    reklamationsquote_prozent: float | None = Field(None, description="Reklamationsquote in Prozent")
    liefertreue_prozent: float | None = Field(None, description="Liefertreue in Prozent")
    warnungen: list[str] = Field(default_factory=list)


WERTFELDER = [f for f in Lieferantenprofil.model_fields if f not in {"lieferant", "slug", "warnungen"}]

# Schema nur mit den Wertfeldern: Name und Slug kennt der Code bereits, das Modell soll sie nicht raten.
_voll = Lieferantenprofil.model_json_schema()
WERT_SCHEMA = {
    "type": "object",
    "properties": {k: v for k, v in _voll["properties"].items() if k in WERTFELDER},
    "required": WERTFELDER,
}


class Extractor(Protocol):
    name: str

    def extrahiere(self, slug: str, dokumente: list[Dokument]) -> Lieferantenprofil: ...


def dokumente_je_lieferant(dokumente: list[Dokument]) -> dict[str, list[Dokument]]:
    """Angebot und Selbstauskunft je Lieferant (Rahmenverträge zählen nicht zum Angebot)."""
    gruppen: dict[str, list[Dokument]] = {}
    for d in dokumente:
        if d.lieferant and d.typ in {"Angebot", "Selbstauskunft"}:
            gruppen.setdefault(d.lieferant, []).append(d)
    return gruppen


def firmenname(dokumente: list[Dokument]) -> str:
    angebot = next((d for d in dokumente if d.typ == "Angebot"), dokumente[0])
    return angebot.titel


# ---------------------------------------------------------------------------
# Basislinie: reguläre Ausdrücke
# ---------------------------------------------------------------------------
class RegelExtractor:
    """Deckt die gängigen Formulierungen ab. Sonderfälle wie 'Preis je 100 Stück',
    Lieferzeiten in Wochen oder Reklamationen in ppm erkennt sie bewusst nicht."""

    name = "regelbasiert"

    def extrahiere(self, slug: str, dokumente: list[Dokument]) -> Lieferantenprofil:
        t = _flach(" ".join(d.text for d in dokumente))
        return Lieferantenprofil(
            lieferant=firmenname(dokumente),
            slug=slug,
            stueckpreis_eur=_zahl(t, r"(?:Stückpreis:|Preis pro Stück:|Einzelpreis netto)\s*([\d.,]+)"),
            mindestbestellmenge=_ganz(t, r"Mindestbestellmenge:\s*([\d.]+)"),
            lieferzeit_tage=_ganz(t, r"(\d+)\s*(?:Arbeitstage|Kalendertage)"),
            zahlungsziel_tage=_ganz(t, r"(\d+)\s*Tage\s*(?:netto|ohne Abzug)"),
            skonto_prozent=_zahl(t, r"([\d,]+)\s*%\s*Skonto") or (0.0 if "ohne Abzug" in t else None),
            preisgleitklausel="Preisgleitklausel" in t,
            iso_9001=bool(re.search(r"ISO 9001\S*\s*–\s*zertifiziert", t)),
            iso_14001=bool(re.search(r"ISO 14001\S*\s*–\s*zertifiziert", t)),
            co2_kg_pro_stueck=_zahl(t, r"CO2-Fußabdruck[^:]*:\s*([\d,]+)\s*kg"),
            reklamationsquote_prozent=_zahl(t, r"Reklamationsquote[^:]*:\s*([\d,]+)\s*%"),
            liefertreue_prozent=_zahl(t, r"Liefertreue[^:]*:\s*([\d,]+)\s*%"),
        )


# ---------------------------------------------------------------------------
# LLM mit JSON-Schema und Belegprüfung
# ---------------------------------------------------------------------------
LLM_PROMPT = """Extrahiere die Angebotsdaten des Lieferanten aus den Dokumenten.
Feldregeln:
- stueckpreis_eur: Preis für EIN Stück. Steht "Preis je 100 Stück", teile durch 100.
- lieferzeit_tage: in Tagen. Wochen mal 7 rechnen. Arbeitstage und Kalendertage als Zahl übernehmen.
- zahlungsziel_tage: Tage bis zur Zahlung netto.
- skonto_prozent: 0, wenn kein Skonto genannt ist.
- preisgleitklausel: true nur, wenn Preise an einen Index gekoppelt oder anpassbar sind. Festpreise = false.
- iso_9001 / iso_14001: true nur bei gültigem Zertifikat. "in Vorbereitung" oder "nicht vorhanden" = false.
- reklamationsquote_prozent: in Prozent. ppm durch 10000 teilen.
- Fehlt eine Angabe, setze null. Erfinde keine Werte.
Antworte nur mit JSON."""


class LLMExtractor:
    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm
        self.name = f"llm ({llm.name})"

    def extrahiere(self, slug: str, dokumente: list[Dokument]) -> Lieferantenprofil:
        quelltext = "\n\n".join(f"### {d.name}\n{d.text}" for d in dokumente)
        schema = WERT_SCHEMA
        antwort = self.llm.chat(
            [{"role": "system", "content": LLM_PROMPT}, {"role": "user", "content": quelltext}],
            schema=schema,
        )
        try:
            daten = json.loads(antwort.inhalt)
        except json.JSONDecodeError:
            return Lieferantenprofil(lieferant=firmenname(dokumente), slug=slug, warnungen=["Kein gültiges JSON"])
        daten.update(lieferant=firmenname(dokumente), slug=slug, warnungen=[])
        profil = Lieferantenprofil.model_validate({k: v for k, v in daten.items() if k in Lieferantenprofil.model_fields})
        return pruefe_belege(profil, quelltext)


# Zulässige Umrechnungen je Feld: Der extrahierte Wert muss nach einer davon im Text stehen.
UMRECHNUNGEN = {
    "stueckpreis_eur": (1, 100),
    "lieferzeit_tage": (1, 1 / 7),
    "reklamationsquote_prozent": (1, 10000),
}


def pruefe_belege(profil: Lieferantenprofil, quelltext: str) -> Lieferantenprofil:
    """Setzt Zahlen auf None, die sich nicht aus dem Quelltext ableiten lassen."""
    im_text = {round(z, 4) for z in _alle_zahlen(_flach(quelltext))}
    werte = profil.model_dump()
    for feld in WERTFELDER:
        wert = werte[feld]
        if isinstance(wert, bool) or wert is None or feld == "skonto_prozent" and wert == 0:
            continue
        faktoren = UMRECHNUNGEN.get(feld, (1,))
        if not any(round(wert * f, 4) in im_text for f in faktoren):
            werte[feld] = None
            werte["warnungen"].append(f"{feld}={wert} nicht im Dokument belegt, verworfen")
    return Lieferantenprofil.model_validate(werte)


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------
def extrahiere_alle(extractor: Extractor, dokumente: list[Dokument]) -> list[Lieferantenprofil]:
    return [extractor.extrahiere(slug, docs) for slug, docs in sorted(dokumente_je_lieferant(dokumente).items())]


def parse_de(text: str) -> float:
    """'1.640,00' -> 1640.0, '18,70' -> 18.7, '1.500' -> 1500.0"""
    text = text.strip().rstrip(".,")
    if "," in text:
        text = text.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"\d{1,3}(\.\d{3})+", text):
        text = text.replace(".", "")
    return float(text)


def _flach(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _zahl(text: str, muster: str) -> float | None:
    m = re.search(muster, text)
    return parse_de(m.group(1)) if m else None


def _ganz(text: str, muster: str) -> int | None:
    z = _zahl(text, muster)
    return int(z) if z is not None else None


def _alle_zahlen(text: str) -> list[float]:
    werte = []
    for roh in re.findall(r"\d+(?:[.,]\d+)*", text):
        try:
            werte.append(parse_de(roh))
        except ValueError:
            continue
    return werte
