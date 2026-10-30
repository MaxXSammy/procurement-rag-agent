from procurement_agent.rag import KEINE_ANGABE, beantworte, pruefe_zahlen


def test_pruefe_zahlen_ignoriert_quellenverweise():
    assert pruefe_zahlen("Der Preis beträgt 18,70 € [1].", "Stückpreis: 18,70 €") == []


def test_pruefe_zahlen_findet_erfundene_zahl():
    assert pruefe_zahlen("Der Preis beträgt 17,99 € [1].", "Stückpreis: 18,70 €") == ["17,99"]


def test_offline_antwort_mit_quelle(store):
    a = beantworte("Wie hoch ist der Stückpreis von Alpen Drehtechnik?", store)
    assert "18,70" in a.text
    assert a.belegt
    assert a.modus == "offline-extraktiv"


def test_llm_antwort_wird_geprueft(store, fake_llm):
    llm = fake_llm(["Der Stückpreis beträgt 15,00 € [1]."])
    a = beantworte("Wie hoch ist der Stückpreis von Alpen Drehtechnik?", store, llm)
    assert not a.belegt
    assert a.unbelegte_zahlen == ["15,00"]
    prompt = llm.aufrufe[0]["nachrichten"][1]["content"]
    assert "Angebot_alpen.pdf" in prompt


def test_leere_llm_antwort_wird_zu_keine_angabe(store, fake_llm):
    a = beantworte("Wie hoch ist der Stückpreis von Alpen Drehtechnik?", store, fake_llm([""]))
    assert a.text == KEINE_ANGABE
