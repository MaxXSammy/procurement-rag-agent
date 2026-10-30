"""Startet die Streamlit-Oberfläche im Testmodus und prüft, dass alle Tabs fehlerfrei laufen."""

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402


def test_app_startet_offline(monkeypatch):
    monkeypatch.setenv("AGENT_MODUS", "offline")
    monkeypatch.setattr("procurement_agent.config.MODUS", "offline")
    at = AppTest.from_file("../app.py", default_timeout=60).run()
    assert not at.exception
    assert len(at.tabs) == 3


def test_frage_stellen(monkeypatch):
    monkeypatch.setattr("procurement_agent.config.MODUS", "offline")
    at = AppTest.from_file("../app.py", default_timeout=60).run()
    at.text_input[0].input("Wer muss Preisgleitklauseln freigeben?")
    at.button[0].click().run()
    assert not at.exception
    assert any("Geschäftsführung" in m.value for m in at.markdown)
