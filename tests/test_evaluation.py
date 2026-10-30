from procurement_agent.evaluation import (
    evaluiere_extraktion,
    evaluiere_rag,
    geschaeftskpis,
    ist_korrekt,
    lade_fragen,
    lade_verlauf,
    speichere_lauf,
)
from procurement_agent.extraction import Lieferantenprofil, RegelExtractor, extrahiere_alle
from procurement_agent.scoring import bewerte

BUSINESS = {
    "bestandslieferant": "Alt GmbH", "jahresbedarf_stueck": 1000, "manuell_minuten_pro_angebot": 30,
    "manuell_minuten_bewertung": 60, "ausschreibungen_pro_jahr": 10, "stundensatz_einkauf_eur": 50,
}


def test_ist_korrekt_braucht_jede_gruppe():
    assert ist_korrekt("Feldmann und Precimet", [["Feldmann"], ["Precimet"]])
    assert not ist_korrekt("nur Feldmann", [["Feldmann"], ["Precimet"]])
    assert ist_korrekt("innerhalb von 5 Wochen", [["35", "5 Wochen"]])


def test_rag_evaluation_offline(store):
    bericht = evaluiere_rag(store, None, lade_fragen())
    assert len(bericht.ergebnisse) == 20
    assert bericht.retrieval_trefferquote >= 0.9
    assert bericht.antwortgenauigkeit >= 0.8


def test_extraktion_basislinie(dokumente, gold_profile):
    bericht = evaluiere_extraktion(extrahiere_alle(RegelExtractor(), dokumente), gold_profile)
    assert 0.9 < bericht.genauigkeit < 1.0  # Basislinie scheitert an den Sonderformaten
    assert {f["feld"] for f in bericht.fehler} <= {"stueckpreis_eur", "lieferzeit_tage", "reklamationsquote_prozent"}


def test_geschaeftskpis():
    profile = [
        Lieferantenprofil(lieferant="Alt GmbH", slug="alt", stueckpreis_eur=12.0, iso_9001=True),
        Lieferantenprofil(lieferant="Neu GmbH", slug="neu", stueckpreis_eur=10.0, iso_9001=True),
    ]
    kpi = geschaeftskpis(bewerte(profile), BUSINESS, laufzeit_s=0)
    assert kpi.empfehlung == "Neu GmbH"
    assert kpi.einsparung_eur_jahr == 2000.0
    assert kpi.stunden_manuell == 2.0  # 2 x 30 + 60 Minuten
    assert kpi.stunden_gespart_jahr == 17.5  # (2 - 0,25) x 10
    assert kpi.wert_zeitersparnis_eur_jahr == 875.0


def test_verlauf_speichern_und_laden(tmp_path):
    datei = tmp_path / "verlauf.jsonl"
    speichere_lauf({"antwortgenauigkeit": 0.9}, datei)
    speichere_lauf({"antwortgenauigkeit": 0.95}, datei)
    verlauf = lade_verlauf(datei)
    assert [v["antwortgenauigkeit"] for v in verlauf] == [0.9, 0.95]
    assert "zeitpunkt" in verlauf[0]
