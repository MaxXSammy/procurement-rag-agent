import json

import pytest

from procurement_agent.extraction import (
    WERTFELDER,
    Lieferantenprofil,
    LLMExtractor,
    RegelExtractor,
    dokumente_je_lieferant,
    parse_de,
    pruefe_belege,
)


@pytest.mark.parametrize("roh, wert", [("18,70", 18.7), ("1.640,00", 1640.0), ("1.500", 1500.0), ("30", 30.0)])
def test_parse_de(roh, wert):
    assert parse_de(roh) == wert


def test_regel_extractor_standardfall(dokumente, gold_profile):
    docs = dokumente_je_lieferant(dokumente)["alpen"]
    p = RegelExtractor().extrahiere("alpen", docs)
    assert p.lieferant == "Alpen Drehtechnik GmbH"
    for feld in WERTFELDER:
        assert getattr(p, feld) == gold_profile["alpen"][feld], feld


def test_regel_extractor_kennt_sonderformate_nicht(dokumente):
    """Dokumentierte Grenze der Basislinie: Preis je 100 Stück wird nicht erkannt."""
    p = RegelExtractor().extrahiere("kovotech", dokumente_je_lieferant(dokumente)["kovotech"])
    assert p.stueckpreis_eur is None


def test_regel_extractor_iso_in_vorbereitung_ist_false(dokumente):
    p = RegelExtractor().extrahiere("rossi", dokumente_je_lieferant(dokumente)["rossi"])
    assert p.iso_9001 is False


def _gold_werte(gold, slug):
    return {f: gold[slug][f] for f in WERTFELDER}


def test_llm_extractor_mit_umrechnung_wird_akzeptiert(dokumente, gold_profile, fake_llm):
    llm = fake_llm([json.dumps(_gold_werte(gold_profile, "kovotech"))])
    p = LLMExtractor(llm).extrahiere("kovotech", dokumente_je_lieferant(dokumente)["kovotech"])
    assert p.stueckpreis_eur == 16.4  # aus "Preis je 100 Stück: 1.640,00"
    assert p.reklamationsquote_prozent == 1.9  # aus "19.000 ppm"
    assert p.warnungen == []
    assert llm.aufrufe[0]["schema"]["required"] == WERTFELDER


def test_llm_extractor_verwirft_erfundene_werte(dokumente, gold_profile, fake_llm):
    werte = _gold_werte(gold_profile, "alpen") | {"stueckpreis_eur": 14.99, "mitarbeiter": 5}
    p = LLMExtractor(fake_llm([json.dumps(werte)])).extrahiere("alpen", dokumente_je_lieferant(dokumente)["alpen"])
    assert p.stueckpreis_eur is None
    assert any("stueckpreis_eur" in w for w in p.warnungen)


def test_llm_extractor_kaputtes_json(dokumente, fake_llm):
    p = LLMExtractor(fake_llm(["kein json"])).extrahiere("alpen", dokumente_je_lieferant(dokumente)["alpen"])
    assert p.warnungen == ["Kein gültiges JSON"]


def test_pruefe_belege_laesst_wochen_umrechnung_zu():
    p = Lieferantenprofil(lieferant="x", slug="x", lieferzeit_tage=35)
    assert pruefe_belege(p, "Lieferung innerhalb von 5 Wochen").lieferzeit_tage == 35
