from procurement_agent.agent import Werkzeuge, frage_agent
from procurement_agent.extraction import RegelExtractor, extrahiere_alle
from procurement_agent.llm import LLMAntwort, ToolCall
from procurement_agent.scoring import bewerte


def werkzeuge(store, dokumente):
    bewertungen = bewerte(extrahiere_alle(RegelExtractor(), dokumente))
    return Werkzeuge(store, lambda: bewertungen)


def test_offline_router_ranking(store, dokumente):
    e = frage_agent("Welcher Lieferant ist der beste?", werkzeuge(store, dokumente))
    assert e.schritte[0].werkzeug == "lieferanten_ranking"
    assert e.text.startswith("1. Alpen Drehtechnik GmbH")
    assert "Ausgeschlossen: Officina Meccanica Rossi" in e.text


def test_offline_router_dokumentsuche(store, dokumente):
    e = frage_agent("Wer muss Preisgleitklauseln freigeben?", werkzeuge(store, dokumente))
    assert e.schritte[0].werkzeug == "dokumente_durchsuchen"
    assert "Geschäftsführung" in e.text


def test_llm_agent_nutzt_werkzeug_und_antwortet(store, dokumente, fake_llm):
    llm = fake_llm([
        LLMAntwort("", [ToolCall("dokumente_durchsuchen", {"frage": "Skonto Alpen Drehtechnik"})]),
        "Alpen Drehtechnik gewährt 3 % Skonto [1].",
    ])
    e = frage_agent("Welchen Skonto gibt Alpen?", werkzeuge(store, dokumente), llm)
    assert [s.werkzeug for s in e.schritte] == ["dokumente_durchsuchen"]
    assert e.quellen and e.unbelegte_zahlen == []
    tool_nachricht = llm.aufrufe[1]["nachrichten"][-1]
    assert tool_nachricht["role"] == "tool" and "Angebot_alpen.pdf" in tool_nachricht["content"]


def test_llm_agent_stoppt_nach_schrittlimit(store, dokumente, fake_llm):
    endlos = [LLMAntwort("", [ToolCall("lieferanten_ranking", {})])] * 4
    llm = fake_llm(endlos + ["Fertig."])
    e = frage_agent("Ranking?", werkzeuge(store, dokumente), llm)
    assert e.text == "Fertig."
    assert len(e.schritte) == 4


def test_llm_agent_gibt_lieferantenfilter_mit(store, dokumente, fake_llm):
    llm = fake_llm([
        LLMAntwort("", [ToolCall("dokumente_durchsuchen", {"frage": "Skonto", "lieferanten": ["Brenner"]})]),
        "Fertig.",
    ])
    e = frage_agent("Welchen Skonto gibt Brenner?", werkzeuge(store, dokumente), llm)
    assert e.quellen and {t.chunk.lieferant for t in e.quellen} == {"brenner"}


def test_offline_router_versteht_verneinung(store, dokumente):
    e = frage_agent("Ich will kein Ranking, nur die Zahlungsbedingungen von Brenner", werkzeuge(store, dokumente))
    assert e.schritte[0].werkzeug == "dokumente_durchsuchen"
    # Verneinung schaltet auch den Lieferantenfilter ab; die Kopfzeilen bringen Brenner trotzdem nach vorn
    assert e.quellen[0].chunk.lieferant == "brenner"
