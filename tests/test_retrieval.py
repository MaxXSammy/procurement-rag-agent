from procurement_agent.retrieval import erkenne_lieferanten


def test_erkenne_lieferanten_auch_mit_leerzeichen():
    bekannte = ["alpen", "swissturning", "rossi"]
    assert erkenne_lieferanten("CO2 bei Swiss Turning?", bekannte) == ["swissturning"]
    assert erkenne_lieferanten("Was kostet es bei Officina Meccanica Rossi?", bekannte) == ["rossi"]
    assert erkenne_lieferanten("Wer ist am günstigsten?", bekannte) == []


def test_suche_filtert_auf_genannten_lieferanten(store):
    treffer = store.suche("Wie hoch ist der Stückpreis von Alpen Drehtechnik?")
    assert treffer
    assert {t.chunk.lieferant for t in treffer} == {"alpen"}


def test_suche_ohne_lieferant_findet_richtlinie(store):
    treffer = store.suche("Wer muss Preisgleitklauseln freigeben?")
    assert "Einkaufsrichtlinie_Hagemann.pdf" in {t.chunk.quelle for t in treffer}
