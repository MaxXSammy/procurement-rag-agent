from procurement_agent.documents import chunke, metadaten_aus_dateiname


def test_metadaten_aus_dateiname():
    assert metadaten_aus_dateiname("Angebot_alpen.pdf") == ("Angebot", "alpen")
    assert metadaten_aus_dateiname("Rahmenvertrag_brenner_2024.pdf") == ("Rahmenvertrag", "brenner")
    assert metadaten_aus_dateiname("Einkaufsrichtlinie_Hagemann.pdf") == ("Einkaufsrichtlinie", None)


def test_alle_testdokumente_geladen(dokumente):
    assert len(dokumente) == 19
    assert {d.lieferant for d in dokumente if d.lieferant} == {
        "alpen", "brenner", "feldmann", "kovotech", "precimet", "rheinwerk", "rossi", "swissturning",
    }


def test_chunks_tragen_kontext_kopfzeile(dokumente):
    angebot = next(d for d in dokumente if d.name == "Angebot_alpen.pdf")
    chunks = chunke(angebot, groesse=200)
    assert len(chunks) > 1
    assert all(c.text.startswith("[Angebot | Alpen Drehtechnik GmbH]") for c in chunks)
    assert all(c.lieferant == "alpen" for c in chunks)
