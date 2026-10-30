"""Streamlit-Oberfläche: Fragen an die Dokumente, Lieferantenbewertung, Erfolgsmessung.

Start:  streamlit run app.py
"""

import pandas as pd
import streamlit as st

from procurement_agent import config
from procurement_agent.agent import Werkzeuge, frage_agent
from procurement_agent.evaluation import gesamtevaluation, geschaeftskpis, lade_fragen, lade_verlauf
from procurement_agent.extraction import WERTFELDER, Lieferantenprofil
from procurement_agent.scoring import STANDARD_KRITERIEN, KoRegeln, Kriterium, bewerte
from procurement_agent.system import baue_system, lade_business

st.set_page_config(page_title="Procurement RAG Agent", layout="wide")

FELD_LABELS = {
    "stueckpreis_eur": "Stückpreis (€)",
    "mindestbestellmenge": "MOQ (Stk.)",
    "lieferzeit_tage": "Lieferzeit (Tage)",
    "zahlungsziel_tage": "Zahlungsziel (Tage)",
    "skonto_prozent": "Skonto (%)",
    "preisgleitklausel": "Preisgleitklausel",
    "iso_9001": "ISO 9001",
    "iso_14001": "ISO 14001",
    "co2_kg_pro_stueck": "CO2 (kg/Stk.)",
    "reklamationsquote_prozent": "Reklamationen (%)",
    "liefertreue_prozent": "Liefertreue (%)",
}


def _python_wert(w):
    """NumPy-Typen aus dem Daten-Editor in einfache Python-Werte umwandeln, NaN -> None."""
    if w is None or (not isinstance(w, (bool, str)) and pd.isna(w)):
        return None
    return w.item() if hasattr(w, "item") else w


def eur(wert: float) -> str:
    return f"{wert:,.0f} €".replace(",", ".")


def prozent(wert: float | None) -> str:
    return "–" if wert is None else f"{wert:.0%}"


@st.cache_resource(show_spinner="Dokumente werden geladen und indexiert …")
def lade_system(modus: str, extraktion: str):
    return baue_system(modus, extraktion)


# ---------------------------------------------------------------------------
# Seitenleiste
# ---------------------------------------------------------------------------
business = lade_business()

with st.sidebar:
    st.markdown("### Procurement RAG Agent")
    st.caption(f"{business['auftraggeber']}  \nAnfrage {business['anfrage']} · {business['artikel']}")
    modi = ["auto", "offline", "ollama"]
    modus = st.selectbox(
        "Betriebsmodus", modi, index=modi.index(config.MODUS) if config.MODUS in modi else 0,
        format_func={"auto": "Automatisch", "offline": "Offline (ohne KI)", "ollama": "Ollama (lokales LLM)"}.get,
    )
    extraktion = st.selectbox(
        "Datenextraktion", ["auto", "regel", "llm"],
        format_func={"auto": "Automatisch", "regel": "Regelbasiert", "llm": "LLM"}.get,
    )
    try:
        system = lade_system(modus, extraktion)
    except Exception as fehler:  # z. B. Ollama nicht erreichbar
        st.error(f"Start mit dieser Einstellung nicht möglich: {fehler}. Wechsle in den Offline-Modus.")
        system = lade_system("offline", "regel")

    st.divider()
    st.caption("Aktive Komponenten")
    st.markdown(
        f"- LLM: `{system.llm.name if system.llm else 'keins'}`\n"
        f"- Embeddings: `{system.store.embedder.name}`\n"
        f"- Extraktion: `{system.extractor.name}`"
    )
    st.caption(f"{len(system.dokumente)} Dokumente · {len(system.store.chunks)} Abschnitte")
    with st.expander("Dokumente"):
        for d in system.dokumente:
            st.caption(d.name)

st.title("Lieferantenauswahl mit KI")
st.caption(
    "Angebote durchsuchen, nach eigenen Kriterien bewerten und den Nutzen messbar machen. "
    "Alle Daten sind fiktive Testdaten."
)
if system.llm is None:
    st.info(
        "**Offline-Modus ohne KI**, z. B. in der Live-Demo: Suche per TF-IDF, Extraktion per Regeln, "
        "Agent per Regel-Router. Lokal mit Ollama übernimmt ein Sprachmodell Extraktion und Werkzeugwahl.",
        icon="ℹ️",
    )

tab_fragen, tab_bewertung, tab_erfolg = st.tabs(["Fragen an die Dokumente", "Lieferantenbewertung", "Erfolgsmessung"])


def aktuelle_bewertungen():
    return st.session_state.get("bewertungen") or system.bewertungen()


# ---------------------------------------------------------------------------
# Tab 1: Fragen
# ---------------------------------------------------------------------------
with tab_fragen:
    beispiele = [f["frage"] for f in lade_fragen()][:8] + ["Welcher Lieferant ist für uns der beste?"]
    with st.form("frageformular", clear_on_submit=True):
        spalte_frage, spalte_beispiel = st.columns([3, 2])
        eingabe = spalte_frage.text_input("Ihre Frage", placeholder="z. B. Welche Lieferanten verlangen eine Preisgleitklausel?")
        beispiel = spalte_beispiel.selectbox(
            "oder ein Beispiel wählen", beispiele, index=None, placeholder="Beispielfrage auswählen …"
        )
        abgeschickt = st.form_submit_button("Fragen", type="primary")

    if abgeschickt and (eingabe or beispiel):
        frage = eingabe or beispiel
        with st.spinner("Agent arbeitet …"):
            ergebnis = frage_agent(frage, Werkzeuge(system.store, aktuelle_bewertungen), system.llm)
        st.session_state.setdefault("verlauf", []).insert(0, (frage, ergebnis))

    for frage, ergebnis in st.session_state.get("verlauf", []):
        st.chat_message("user").write(frage)
        with st.chat_message("assistant"):
            st.markdown(ergebnis.text.replace("\n", "  \n"))
            if ergebnis.unbelegte_zahlen:
                st.warning(f"Nicht in den Quellen belegt: {', '.join(ergebnis.unbelegte_zahlen)}")
            elif ergebnis.quellen:
                st.caption("Belegprüfung bestanden: Alle Zahlen stehen in den Quellen.")
            with st.expander(f"Quellen ({len(ergebnis.quellen)}) und Agentenschritte ({len(ergebnis.schritte)})"):
                st.caption("Schritte: " + " → ".join(s.werkzeug for s in ergebnis.schritte))
                for i, t in enumerate(ergebnis.quellen, start=1):
                    st.markdown(f"**[{i}] {t.chunk.quelle}** · Seite {t.chunk.seite} · Ähnlichkeit {t.score:.2f}")
                    st.text(t.chunk.text)

    if not st.session_state.get("verlauf"):
        st.info("Stellen Sie eine Frage zu Angeboten, Selbstauskünften, Rahmenvertrag oder Einkaufsrichtlinie.")


# ---------------------------------------------------------------------------
# Tab 2: Bewertung
# ---------------------------------------------------------------------------
with tab_bewertung:
    links, rechts = st.columns([1, 2.4], gap="large")

    with links:
        st.subheader("Gewichtung")
        kriterien = []
        for k in STANDARD_KRITERIEN:
            gewicht = st.slider(k.label, 0, 100, int(k.gewicht), step=5, key=f"gewicht_{k.feld}")
            kriterien.append(Kriterium(k.feld, k.label, gewicht, k.hoeher_besser))
        summe = sum(k.gewicht for k in kriterien) or 1
        st.caption(" · ".join(f"{k.label} {k.gewicht / summe:.0%}" for k in kriterien if k.gewicht))

        st.subheader("K.-o.-Kriterien")
        st.caption("Voreinstellung gemäß Einkaufsrichtlinie")
        iso = st.checkbox("ISO 9001 Pflicht", value=True)
        max_lz = st.number_input("Max. Lieferzeit (Tage, 0 = aus)", 0, 120, 30, step=5)
        gleit = st.checkbox("Preisgleitklausel erlaubt", value=False)
        min_zz = st.number_input("Min. Zahlungsziel (Tage, 0 = aus)", 0, 120, 30, step=5)
        regeln = KoRegeln(iso, max_lz or None, gleit, min_zz or None)

    with rechts:
        st.subheader("Extrahierte Daten prüfen")
        st.caption(
            "Die Werte stammen aus den PDFs. Leere Zellen konnte die Extraktion nicht belegen. "
            "Korrekturen fließen direkt in die Bewertung ein."
        )
        profile = system.profile()
        basis = pd.DataFrame([{"Lieferant": p.lieferant, **{FELD_LABELS[f]: getattr(p, f) for f in WERTFELDER}} for p in profile])
        bearbeitet = st.data_editor(
            basis, hide_index=True, width="stretch", disabled=["Lieferant"], key="profil_editor", placeholder="–",
            column_config={
                FELD_LABELS["stueckpreis_eur"]: st.column_config.NumberColumn(format="%.2f"),
                FELD_LABELS["co2_kg_pro_stueck"]: st.column_config.NumberColumn(format="%.1f"),
            },
        )
        rueck = {v: k for k, v in FELD_LABELS.items()}
        korrigiert = []
        for p, (_, zeile) in zip(profile, bearbeitet.iterrows()):
            werte = {rueck[s]: _python_wert(w) for s, w in zeile.items() if s in rueck}
            korrigiert.append(Lieferantenprofil(lieferant=p.lieferant, slug=p.slug, warnungen=p.warnungen, **werte))
        geaendert = sum(
            getattr(a, f) != getattr(b, f) for a, b in zip(profile, korrigiert) for f in WERTFELDER
        )
        if geaendert:
            st.caption(f"{geaendert} Wert(e) manuell korrigiert.")

        bewertungen = bewerte(korrigiert, kriterien, regeln)
        st.session_state["bewertungen"] = bewertungen

        st.subheader("Rangliste")
        tabelle = pd.DataFrame([{
            "Rang": str(b.rang) if b.zugelassen else "–",
            "Lieferant": b.profil.lieferant,
            "Score": round(b.score, 3) if b.zugelassen else None,
            "Stückpreis (€)": b.profil.stueckpreis_eur,
            "Lieferzeit (Tage)": b.profil.lieferzeit_tage,
            "Liefertreue (%)": b.profil.liefertreue_prozent,
            "Status": "zugelassen" if b.zugelassen else "ausgeschlossen: " + "; ".join(b.ausschluss),
            "Hinweise": "; ".join(b.hinweise),
        } for b in bewertungen])
        st.dataframe(
            tabelle, hide_index=True, width="stretch", placeholder="–",
            column_config={
                "Score": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f"),
                "Stückpreis (€)": st.column_config.NumberColumn(format="%.2f"),
            },
        )
        zugelassen = [b for b in bewertungen if b.zugelassen]
        if zugelassen:
            teil = pd.DataFrame(
                {b.profil.lieferant: {k.label: b.teilscores.get(k.label, 0) * k.gewicht / summe for k in kriterien}
                 for b in zugelassen}
            ).T
            st.caption("Zusammensetzung des Scores je Kriterium")
            st.bar_chart(teil, horizontal=True)


# ---------------------------------------------------------------------------
# Tab 3: Erfolgsmessung
# ---------------------------------------------------------------------------
with tab_erfolg:
    st.subheader("Management-Übersicht")

    with st.expander("Annahmen zur Wirtschaftlichkeit"):
        a1, a2, a3 = st.columns(3)
        annahmen = dict(business)
        annahmen["jahresbedarf_stueck"] = a1.number_input("Jahresbedarf (Stück)", 0, 1_000_000, business["jahresbedarf_stueck"], step=1000)
        annahmen["ausschreibungen_pro_jahr"] = a1.number_input("Ausschreibungen pro Jahr", 1, 500, business["ausschreibungen_pro_jahr"])
        annahmen["manuell_minuten_pro_angebot"] = a2.number_input("Manuell: Minuten je Angebot", 0, 600, business["manuell_minuten_pro_angebot"], step=5)
        annahmen["manuell_minuten_bewertung"] = a2.number_input("Manuell: Minuten für Vergleich", 0, 600, business["manuell_minuten_bewertung"], step=10)
        annahmen["stundensatz_einkauf_eur"] = a3.number_input("Stundensatz Einkauf (€)", 0, 500, business["stundensatz_einkauf_eur"], step=5)

    verlauf = lade_verlauf()
    if "lauf" not in st.session_state and not verlauf and system.llm is None:
        # Ohne gespeicherten Verlauf (z. B. frisch in der Cloud) wären die Kennzahlen leer.
        # Offline dauert die Evaluation nur Millisekunden; nicht speichern, damit nichts geschrieben wird.
        st.session_state["lauf"] = gesamtevaluation(system, speichern=False)
    lauf = st.session_state.get("lauf")
    letzter = lauf.eintrag if lauf else (verlauf[-1] if verlauf else None)
    kpi = geschaeftskpis(aktuelle_bewertungen(), annahmen, lauf.dauer_s if lauf else 0.0)

    m1, m2, m3, m4 = st.columns(4)
    if kpi.einsparung_eur_jahr is not None:
        m1.metric("Einsparpotenzial pro Jahr", eur(kpi.einsparung_eur_jahr),
                  f"{kpi.empfehlung} statt {kpi.bestandslieferant}", delta_color="off")
    else:
        m1.metric("Einsparpotenzial pro Jahr", "–")
    m2.metric("Zeitersparnis pro Jahr", f"{kpi.stunden_gespart_jahr:,.0f} h".replace(",", "."),
              f"entspricht {eur(kpi.wert_zeitersparnis_eur_jahr)}", delta_color="off")
    m3.metric("Antwortgenauigkeit", prozent(letzter and letzter["antwortgenauigkeit"]))
    m4.metric("Extraktionsgenauigkeit", prozent(letzter and letzter["extraktionsgenauigkeit"]))

    st.caption(
        f"Einsparung: (Preis {kpi.bestandslieferant} − Preis Empfehlung) × Jahresbedarf. "
        f"Zeit: manuell {kpi.stunden_manuell:.1f} h je Ausschreibung, mit Tool {kpi.stunden_mit_tool:.1f} h "
        "inklusive 15 Minuten fachlicher Prüfung."
    )

    st.divider()
    kopf, knopf = st.columns([3, 1])
    kopf.markdown("**Qualität des Systems**  \nGemessen an 20 Fragen und 8 Lieferantenprofilen mit bekannter Lösung.")
    if knopf.button("Evaluation ausführen", type="primary"):
        with st.spinner("Evaluation läuft …"):
            st.session_state["lauf"] = gesamtevaluation(system)
        st.rerun()

    if lauf:
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Retrieval-Trefferquote", prozent(lauf.rag.retrieval_trefferquote))
        q2.metric("Antwortgenauigkeit", prozent(lauf.rag.antwortgenauigkeit))
        q3.metric("Belegquote", prozent(lauf.rag.belegquote))
        q4.metric("Ø Antwortzeit", f"{lauf.rag.mittlere_dauer_s:.2f} s")

        with st.expander("Ergebnisse je Frage"):
            st.dataframe(pd.DataFrame([{
                "ID": e.id, "Frage": e.frage, "korrekt": e.korrekt, "belegt": e.belegt,
                "Quellen gefunden": e.retrieval_recall, "Antwort": e.antwort,
            } for e in lauf.rag.ergebnisse]), hide_index=True, width="stretch")
        with st.expander(f"Extraktionsfehler ({len(lauf.extraktion.fehler)})"):
            if lauf.extraktion.fehler:
                st.dataframe(pd.DataFrame(lauf.extraktion.fehler), hide_index=True, width="stretch")
            else:
                st.caption("Keine Fehler.")

    if len(verlauf) > 1:
        st.caption("Verlauf aller Evaluationen")
        df = pd.DataFrame(verlauf).set_index("zeitpunkt")
        st.line_chart(df[["retrieval_trefferquote", "antwortgenauigkeit", "belegquote", "extraktionsgenauigkeit"]])

    if kpi.risiken:
        st.divider()
        st.markdown("**Erkannte Risiken und offene Punkte**")
        for r in kpi.risiken:
            st.markdown(f"- {r}")
