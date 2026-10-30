"""Belegprüfung (Grounding): Jede Zahl muss sich auf die Quellen zurückführen lassen.

Zwei Prüfungen im Code statt Vertrauen ins LLM:
  - pruefe_zahlen: Zahlen in Antworten gegen die gefundenen Abschnitte
  - pruefe_belege: extrahierte Profilwerte gegen den Quelltext, mit erlaubten Umrechnungen
"""

import pytest

from procurement_agent.agent import Werkzeuge, frage_agent
from procurement_agent.extraction import Lieferantenprofil, pruefe_belege
from procurement_agent.llm import LLMAntwort, ToolCall
from procurement_agent.rag import KEINE_ANGABE, pruefe_zahlen


# ---------------------------------------------------------------------------
# pruefe_zahlen: Antworten
# ---------------------------------------------------------------------------
def test_antwort_ohne_zahlen_ist_belegt():
    assert pruefe_zahlen(KEINE_ANGABE, "Stückpreis: 18,70 €") == []


def test_tausenderpunkt_ist_egal():
    assert pruefe_zahlen("Preis je 100 Stück: 1640,00 € [2].", "Preis je 100 Stück: 1.640,00 €") == []


def test_teilzahl_gilt_nicht_als_beleg():
    """'8' steckt in '18,70', ist aber keine eigene Zahl im Kontext."""
    assert pruefe_zahlen("Lieferzeit 8 Tage [1].", "Stückpreis: 18,70 €") == ["8"]


def test_mehrstellige_quellenverweise_werden_ignoriert():
    assert pruefe_zahlen("Skonto 3 % [12].", "3 % Skonto bei Zahlung") == []


def test_zahl_in_eckigen_klammern_ohne_verweis_wird_geprueft():
    assert pruefe_zahlen("Mindestmenge [500 Stück].", "Mindestmenge 1.000 Stück") == ["500"]


def test_meldet_alle_unbelegten_zahlen_in_reihenfolge():
    antwort = "Preis 17,99 € [1], Lieferzeit 30 Tage [2], Skonto 4 % [1]."
    assert pruefe_zahlen(antwort, "Lieferzeit: 30 Tage, Skonto 2 %") == ["17,99", "4"]


# ---------------------------------------------------------------------------
# pruefe_belege: Extraktion
# ---------------------------------------------------------------------------
def profil(**werte) -> Lieferantenprofil:
    return Lieferantenprofil(lieferant="Test GmbH", slug="test", **werte)


def test_wert_wie_im_dokument_bleibt():
    p = pruefe_belege(profil(stueckpreis_eur=18.7, zahlungsziel_tage=30), "Stückpreis 18,70 €, Zahlung 30 Tage netto")
    assert (p.stueckpreis_eur, p.zahlungsziel_tage) == (18.7, 30)
    assert p.warnungen == []


@pytest.mark.parametrize("feld, wert, text", [
    ("stueckpreis_eur", 16.4, "Preis je 100 Stück: 1.640,00 €"),
    ("lieferzeit_tage", 42, "Lieferung in 6 Wochen"),
    ("reklamationsquote_prozent", 1.9, "Reklamationen: 19.000 ppm"),
])
def test_erlaubte_umrechnungen(feld, wert, text):
    p = pruefe_belege(profil(**{feld: wert}), text)
    assert getattr(p, feld) == wert
    assert p.warnungen == []


def test_umrechnung_gilt_nur_fuer_ihr_feld():
    """Faktor 100 ist nur beim Stückpreis erlaubt, nicht bei der Mindestbestellmenge."""
    p = pruefe_belege(profil(mindestbestellmenge=5), "Mindestbestellmenge: 500 Stück")
    assert p.mindestbestellmenge is None
    assert p.warnungen == ["mindestbestellmenge=5 nicht im Dokument belegt, verworfen"]


def test_erfundener_wert_wird_verworfen_rest_bleibt():
    p = pruefe_belege(profil(stueckpreis_eur=14.99, liefertreue_prozent=97.5), "Liefertreue 97,5 %, Preis 18,70 €")
    assert p.stueckpreis_eur is None
    assert p.liefertreue_prozent == 97.5
    assert len(p.warnungen) == 1 and "stueckpreis_eur" in p.warnungen[0]


def test_bools_none_und_skonto_null_brauchen_keinen_beleg():
    p = pruefe_belege(profil(iso_9001=True, preisgleitklausel=False, skonto_prozent=0), "kein passender Text")
    assert (p.iso_9001, p.preisgleitklausel, p.skonto_prozent) == (True, False, 0)
    assert p.co2_kg_pro_stueck is None
    assert p.warnungen == []


def test_skonto_ungleich_null_braucht_beleg():
    p = pruefe_belege(profil(skonto_prozent=2), "Zahlung 30 Tage netto")
    assert p.skonto_prozent is None


def test_zeilenumbruch_im_quelltext_stoert_nicht():
    p = pruefe_belege(profil(stueckpreis_eur=18.7), "Stückpreis:\n\n  18,70 €")
    assert p.stueckpreis_eur == 18.7


def test_bestehende_warnungen_bleiben_erhalten():
    p = pruefe_belege(profil(stueckpreis_eur=1.0, warnungen=["alt"]), "Preis 18,70 €")
    assert p.warnungen[0] == "alt" and len(p.warnungen) == 2


# ---------------------------------------------------------------------------
# Agent: Prüfung gegen die Werkzeugergebnisse
# ---------------------------------------------------------------------------
def test_agent_meldet_erfundene_zahl(store, fake_llm):
    llm = fake_llm([
        LLMAntwort("", [ToolCall("dokumente_durchsuchen", {"frage": "Stückpreis Alpen Drehtechnik"})]),
        "Alpen Drehtechnik verlangt 15,00 € je Stück [1].",
    ])
    e = frage_agent("Was kostet ein Stück bei Alpen?", Werkzeuge(store, lambda: []), llm)
    assert e.unbelegte_zahlen == ["15,00"]


def test_agent_antwort_ohne_werkzeug_gilt_nicht_als_belegt(store, fake_llm):
    e = frage_agent("Was kostet ein Stück bei Alpen?", Werkzeuge(store, lambda: []), fake_llm(["Es kostet 15,00 €."]))
    assert e.unbelegte_zahlen == ["15,00"]
