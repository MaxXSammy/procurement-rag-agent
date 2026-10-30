"""Erzeugt alle Testdaten für den Procurement RAG Agent.

Alle Firmen, Personen und Zahlen sind frei erfunden.

Die Lieferanten-Fakten unten sind die einzige Wahrheitsquelle. Aus ihnen entstehen
  - die PDF-Dokumente (Angebote, Selbstauskünfte, interne Dokumente),
  - die Gold-Profile für die Extraktions-Evaluation,
  - die Gold-Fragen für die RAG-Evaluation,
  - die Business-Annahmen für die Management-KPIs.

Die Formulierungen variieren absichtlich (Preis je 100 Stück, Lieferzeit in Wochen,
Reklamationen in ppm), damit die Extraktion realistisch schwer ist.

Aufruf:  python scripts/generate_testdata.py
"""

import json
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "data" / "dokumente"
GOLD = ROOT / "data" / "gold"

AUFTRAGGEBER = "Hagemann Antriebstechnik GmbH"
ANFRAGE_NR = "AS-2027-014"
ARTIKEL = "Antriebswelle AW-40"
JAHRESBEDARF = 12000
BESTANDSLIEFERANT = "brenner"

# ---------------------------------------------------------------------------
# Wahrheitsquelle: Lieferanten-Fakten
# ---------------------------------------------------------------------------
LIEFERANTEN = [
    {
        "slug": "brenner", "name": "Brenner Zerspanung GmbH", "land": "DE",
        "ort": "58507 Lüdenscheid", "ansprechpartner": "Thomas Brenner",
        "stueckpreis_eur": 21.90, "preis_format": "stueck",
        "mindestbestellmenge": 1000, "lieferzeit_tage": 15, "lieferzeit_format": "arbeitstage",
        "zahlungsziel_tage": 30, "skonto_prozent": 2.0, "preisgleitklausel": False,
        "iso_9001": True, "iso_14001": True, "co2_kg_pro_stueck": 3.1,
        "reklamationsquote_prozent": 0.8, "rekl_format": "prozent",
        "liefertreue_prozent": 96.0, "mitarbeiter": 140, "gueltig_bis": "31.03.2027",
    },
    {
        "slug": "feldmann", "name": "Feldmann Präzisionsteile GmbH", "land": "DE",
        "ort": "33602 Bielefeld", "ansprechpartner": "Sabine Feldmann",
        "stueckpreis_eur": 19.80, "preis_format": "netto",
        "mindestbestellmenge": 2000, "lieferzeit_tage": 21, "lieferzeit_format": "kalendertage",
        "zahlungsziel_tage": 45, "skonto_prozent": 2.0, "preisgleitklausel": True,
        "iso_9001": True, "iso_14001": False, "co2_kg_pro_stueck": 2.9,
        "reklamationsquote_prozent": 1.2, "rekl_format": "prozent",
        "liefertreue_prozent": 93.0, "mitarbeiter": 85, "gueltig_bis": "28.02.2027",
    },
    {
        "slug": "kovotech", "name": "Kovotech s.r.o.", "land": "CZ",
        "ort": "301 00 Plzen", "ansprechpartner": "Jan Novák",
        "stueckpreis_eur": 16.40, "preis_format": "je100",
        "mindestbestellmenge": 1500, "lieferzeit_tage": 28, "lieferzeit_format": "kalendertage",
        "zahlungsziel_tage": 30, "skonto_prozent": 0.0, "preisgleitklausel": False,
        "iso_9001": True, "iso_14001": True, "co2_kg_pro_stueck": 3.6,
        "reklamationsquote_prozent": 1.9, "rekl_format": "ppm",
        "liefertreue_prozent": 89.0, "mitarbeiter": 210, "gueltig_bis": "31.01.2027",
    },
    {
        "slug": "precimet", "name": "Precimet Sp. z o.o.", "land": "PL",
        "ort": "50-001 Wroclaw", "ansprechpartner": "Anna Kowalska",
        "stueckpreis_eur": 15.95, "preis_format": "einzel",
        "mindestbestellmenge": 2500, "lieferzeit_tage": 35, "lieferzeit_format": "wochen",
        "zahlungsziel_tage": 60, "skonto_prozent": 0.0, "preisgleitklausel": True,
        "iso_9001": True, "iso_14001": False, "co2_kg_pro_stueck": 4.2,
        "reklamationsquote_prozent": 2.6, "rekl_format": "prozent",
        "liefertreue_prozent": 85.0, "mitarbeiter": 320, "gueltig_bis": "31.12.2026",
    },
    {
        "slug": "alpen", "name": "Alpen Drehtechnik GmbH", "land": "AT",
        "ort": "6020 Innsbruck", "ansprechpartner": "Markus Hofer",
        "stueckpreis_eur": 18.70, "preis_format": "stueck",
        "mindestbestellmenge": 1000, "lieferzeit_tage": 18, "lieferzeit_format": "arbeitstage",
        "zahlungsziel_tage": 30, "skonto_prozent": 3.0, "preisgleitklausel": False,
        "iso_9001": True, "iso_14001": True, "co2_kg_pro_stueck": 2.4,
        "reklamationsquote_prozent": 0.6, "rekl_format": "prozent",
        "liefertreue_prozent": 97.0, "mitarbeiter": 95, "gueltig_bis": "31.03.2027",
    },
    {
        "slug": "rossi", "name": "Officina Meccanica Rossi S.r.l.", "land": "IT",
        "ort": "25121 Brescia", "ansprechpartner": "Giulia Rossi",
        "stueckpreis_eur": 17.20, "preis_format": "netto",
        "mindestbestellmenge": 1200, "lieferzeit_tage": 25, "lieferzeit_format": "kalendertage",
        "zahlungsziel_tage": 60, "skonto_prozent": 1.0, "preisgleitklausel": False,
        "iso_9001": False, "iso_14001": False, "co2_kg_pro_stueck": 3.8,
        "reklamationsquote_prozent": 2.1, "rekl_format": "prozent",
        "liefertreue_prozent": 90.0, "mitarbeiter": 60, "gueltig_bis": "15.02.2027",
    },
    {
        "slug": "rheinwerk", "name": "Rheinwerk Komponenten GmbH", "land": "DE",
        "ort": "47051 Duisburg", "ansprechpartner": "Peter Lange",
        "stueckpreis_eur": 20.40, "preis_format": "einzel",
        "mindestbestellmenge": 800, "lieferzeit_tage": 10, "lieferzeit_format": "arbeitstage",
        "zahlungsziel_tage": 30, "skonto_prozent": 2.0, "preisgleitklausel": False,
        "iso_9001": True, "iso_14001": True, "co2_kg_pro_stueck": 2.7,
        "reklamationsquote_prozent": 0.9, "rekl_format": "ppm",
        "liefertreue_prozent": 98.0, "mitarbeiter": 175, "gueltig_bis": "31.03.2027",
    },
    {
        "slug": "swissturning", "name": "Swiss Turning AG", "land": "CH",
        "ort": "8400 Winterthur", "ansprechpartner": "Lukas Meier",
        "stueckpreis_eur": 22.80, "preis_format": "stueck",
        "mindestbestellmenge": 500, "lieferzeit_tage": 12, "lieferzeit_format": "kalendertage",
        "zahlungsziel_tage": 30, "skonto_prozent": 0.0, "preisgleitklausel": False,
        "iso_9001": True, "iso_14001": True, "co2_kg_pro_stueck": 2.2,
        "reklamationsquote_prozent": 0.4, "rekl_format": "prozent",
        "liefertreue_prozent": 99.0, "mitarbeiter": 120, "gueltig_bis": "31.03.2027",
    },
]

PROFIL_FELDER = [
    "stueckpreis_eur", "mindestbestellmenge", "lieferzeit_tage", "zahlungsziel_tage",
    "skonto_prozent", "preisgleitklausel", "iso_9001", "iso_14001", "co2_kg_pro_stueck",
    "reklamationsquote_prozent", "liefertreue_prozent",
]


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------
def de(zahl: float, stellen: int = 2) -> str:
    """Deutsches Zahlenformat: 1640.0 -> '1.640,00'."""
    text = f"{zahl:,.{stellen}f}"
    return text.replace(",", "X").replace(".", ",").replace("X", ".")


def ganzzahl(zahl: int) -> str:
    return f"{zahl:,}".replace(",", ".")


_styles = getSampleStyleSheet()
TITEL = ParagraphStyle("Titel", parent=_styles["Heading1"], fontSize=16, spaceAfter=6)
UNTERTITEL = ParagraphStyle("Unter", parent=_styles["Heading2"], fontSize=12, spaceBefore=10)
TEXT = ParagraphStyle("Text", parent=_styles["Normal"], fontSize=10, leading=14, spaceAfter=4)
KLEIN = ParagraphStyle("Klein", parent=TEXT, fontSize=8, textColor="#666666")


def schreibe_pdf(datei: Path, bloecke: list[tuple[str, str]]) -> None:
    """bloecke: Liste aus (Stil, Text) mit Stil in {'titel', 'unter', 'text', 'klein', 'abstand'}."""
    stil = {"titel": TITEL, "unter": UNTERTITEL, "text": TEXT, "klein": KLEIN}
    elemente = []
    for art, inhalt in bloecke:
        if art == "abstand":
            elemente.append(Spacer(1, 0.4 * cm))
        else:
            elemente.append(Paragraph(inhalt, stil[art]))
    SimpleDocTemplate(
        str(datei), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm, title=datei.stem, author="Testdaten",
    ).build(elemente)


# ---------------------------------------------------------------------------
# Textbausteine mit Varianten
# ---------------------------------------------------------------------------
def preis_satz(lf: dict) -> str:
    p = lf["stueckpreis_eur"]
    return {
        "stueck": f"Stückpreis: {de(p)} €",
        "netto": f"Preis pro Stück: {de(p)} EUR netto, zzgl. MwSt.",
        "je100": f"Preis je 100 Stück: {de(p * 100)} EUR (netto, ab Werk)",
        "einzel": f"Einzelpreis netto {de(p)} EUR/Stk.",
    }[lf["preis_format"]]


def lieferzeit_satz(lf: dict) -> str:
    t = lf["lieferzeit_tage"]
    return {
        "arbeitstage": f"Lieferzeit: {t} Arbeitstage nach Auftragseingang.",
        "kalendertage": f"Liefertermin: {t} Kalendertage ab Bestellung, frei Haus Gevelsberg.",
        "wochen": f"Die Lieferung erfolgt innerhalb von {t // 7} Wochen nach Auftragsbestätigung.",
    }[lf["lieferzeit_format"]]


def zahlung_satz(lf: dict) -> str:
    if lf["skonto_prozent"] > 0:
        return (
            f"Zahlungsbedingungen: {lf['zahlungsziel_tage']} Tage netto, "
            f"{de(lf['skonto_prozent'], 0 if lf['skonto_prozent'].is_integer() else 1)} % Skonto "
            "bei Zahlung innerhalb von 10 Tagen."
        )
    return f"Zahlungsziel: {lf['zahlungsziel_tage']} Tage ohne Abzug."


def reklamation_satz(lf: dict) -> str:
    q = lf["reklamationsquote_prozent"]
    if lf["rekl_format"] == "ppm":
        return f"Reklamationen 2026: {ganzzahl(round(q * 10000))} ppm (bezogen auf gelieferte Teile)."
    return f"Reklamationsquote 2026: {de(q, 1)} %."


# ---------------------------------------------------------------------------
# Dokumente
# ---------------------------------------------------------------------------
def angebot(lf: dict) -> None:
    gleit = (
        [
            ("unter", "Preisgleitklausel"),
            ("text", "Die angebotenen Preise basieren auf dem Stahlpreisindex für C45 "
                     "(Stand Oktober 2026). Weicht der Index um mehr als 5 % ab, behalten wir "
                     "uns eine entsprechende Anpassung der Stückpreise vor. Die Anpassung "
                     "erfolgt quartalsweise."),
        ]
        if lf["preisgleitklausel"]
        else [("text", "Die Preise sind Festpreise für die gesamte Vertragslaufzeit von 12 Monaten.")]
    )
    bloecke = [
        ("titel", lf["name"]),
        ("klein", f"{lf['ort']} · {lf['land']} · Ansprechpartner: {lf['ansprechpartner']}"),
        ("abstand", ""),
        ("text", f"An: {AUFTRAGGEBER}, Einkauf, 58285 Gevelsberg"),
        ("unter", f"Angebot zu Ihrer Anfrage {ANFRAGE_NR}"),
        ("text", f"Angebots-Nr.: AN-{lf['slug'][:3].upper()}-2026-118 · Gültig bis: {lf['gueltig_bis']}"),
        ("text", "Sehr geehrte Damen und Herren, vielen Dank für Ihre Anfrage. "
                 "Gerne bieten wir Ihnen wie folgt an:"),
        ("unter", "Position 1"),
        ("text", f"{ARTIKEL}, Werkstoff C45, Ø 40 x 250 mm, induktiv gehärtet, "
                 "nach Zeichnung HA-AW40-03."),
        ("text", preis_satz(lf)),
        ("text", f"Mindestbestellmenge: {ganzzahl(lf['mindestbestellmenge'])} Stück je Abruf."),
        ("text", lieferzeit_satz(lf)),
        ("unter", "Konditionen"),
        ("text", zahlung_satz(lf)),
        *gleit,
        ("text", "Verpackung in Gitterboxen, Lieferung frei Haus."),
        ("abstand", ""),
        ("text", f"Mit freundlichen Grüßen<br/>{lf['ansprechpartner']}<br/>{lf['name']}"),
    ]
    schreibe_pdf(DOCS / f"Angebot_{lf['slug']}.pdf", bloecke)


def selbstauskunft(lf: dict) -> None:
    if lf["iso_9001"]:
        q = "ISO 9001:2015 – zertifiziert, Zertifikat gültig bis 2028."
    else:
        q = ("ISO 9001: Zertifizierung in Vorbereitung, Erstaudit geplant für 03/2027. "
             "Aktuell liegt kein gültiges Zertifikat vor.")
    u = (
        "ISO 14001:2015 – zertifiziert."
        if lf["iso_14001"]
        else "ISO 14001: nicht vorhanden, Umweltmanagement nach internen Richtlinien."
    )
    bloecke = [
        ("titel", f"Lieferantenselbstauskunft – {lf['name']}"),
        ("klein", f"Ausgefüllt für {AUFTRAGGEBER} im Rahmen der Anfrage {ANFRAGE_NR}"),
        ("unter", "1. Unternehmensdaten"),
        ("text", f"Firma: {lf['name']}, {lf['ort']}, Land: {lf['land']}"),
        ("text", f"Anzahl Mitarbeiter: {lf['mitarbeiter']}"),
        ("unter", "2. Zertifizierungen"),
        ("text", f"Qualitätsmanagement: {q}"),
        ("text", f"Umweltmanagement: {u}"),
        ("unter", "3. Nachhaltigkeit"),
        ("text", f"CO2-Fußabdruck pro Stück (Antriebswelle AW-40, cradle-to-gate): "
                 f"{de(lf['co2_kg_pro_stueck'], 1)} kg CO2e"),
        ("unter", "4. Leistungskennzahlen (eigene Angaben)"),
        ("text", f"Liefertreue 2026: {de(lf['liefertreue_prozent'], 0)} % termingerechte Lieferungen."),
        ("text", reklamation_satz(lf)),
        ("abstand", ""),
        ("klein", "Wir bestätigen die Richtigkeit der Angaben."),
    ]
    schreibe_pdf(DOCS / f"Selbstauskunft_{lf['slug']}.pdf", bloecke)


def interne_dokumente() -> None:
    schreibe_pdf(DOCS / "Anfrage_AS-2027-014.pdf", [
        ("titel", f"Anfrage {ANFRAGE_NR}"),
        ("klein", f"{AUFTRAGGEBER} · Einkauf · 58285 Gevelsberg"),
        ("unter", "Gegenstand"),
        ("text", f"{ARTIKEL}, Werkstoff C45, Ø 40 x 250 mm, induktiv gehärtet, "
                 "nach Zeichnung HA-AW40-03."),
        ("text", f"Geplanter Jahresbedarf: {ganzzahl(JAHRESBEDARF)} Stück, Abruf monatlich."),
        ("text", "Vertragslaufzeit: 12 Monate ab 01.01.2027."),
        ("unter", "Angebotsabgabe"),
        ("text", "Bitte geben Sie Stückpreis, Mindestbestellmenge, Lieferzeit, "
                 "Zahlungsbedingungen und eventuelle Preisgleitklauseln an. "
                 "Zusätzlich benötigen wir eine ausgefüllte Lieferantenselbstauskunft."),
        ("text", "Angebotsfrist: 15.11.2026."),
    ])
    schreibe_pdf(DOCS / "Einkaufsrichtlinie_Hagemann.pdf", [
        ("titel", "Einkaufsrichtlinie"),
        ("klein", f"{AUFTRAGGEBER} · gültig ab 01.01.2026 · Version 3.2"),
        ("unter", "§ 1 Lieferantenauswahl"),
        ("text", "Serienlieferanten müssen ein gültiges ISO 9001 Zertifikat nachweisen. "
                 "Ohne Zertifikat ist eine Beauftragung nur als Musterlieferung zulässig."),
        ("text", "Eine ISO 14001 Zertifizierung ist erwünscht und wird in der Bewertung positiv berücksichtigt."),
        ("unter", "§ 2 Lieferzeiten"),
        ("text", "Die maximale Lieferzeit für Seriendrehteile beträgt 30 Kalendertage."),
        ("unter", "§ 3 Preise und Zahlungsbedingungen"),
        ("text", "Preisgleitklauseln sind nur nach schriftlicher Freigabe durch die Geschäftsführung zulässig."),
        ("text", "Das Zahlungsziel muss mindestens 30 Tage betragen."),
        ("unter", "§ 4 Nachhaltigkeit"),
        ("text", "Bei vergleichbaren Angeboten ist der Lieferant mit dem geringeren CO2-Fußabdruck zu bevorzugen."),
        ("unter", "§ 5 Freigabegrenzen"),
        ("text", "Rahmenverträge mit einem Jahresvolumen über 200.000 EUR werden von der Geschäftsführung freigegeben."),
    ])
    b = next(lf for lf in LIEFERANTEN if lf["slug"] == BESTANDSLIEFERANT)
    schreibe_pdf(DOCS / "Rahmenvertrag_brenner_2024.pdf", [
        ("titel", "Rahmenvertrag Nr. RV-2024-007"),
        ("klein", f"zwischen {AUFTRAGGEBER} und {b['name']}"),
        ("unter", "§ 1 Vertragsgegenstand"),
        ("text", f"Lieferung von {ARTIKEL} nach Zeichnung HA-AW40-03."),
        ("unter", "§ 2 Preise"),
        ("text", f"Der Stückpreis beträgt {de(b['stueckpreis_eur'])} EUR netto."),
        ("unter", "§ 3 Laufzeit und Kündigung"),
        ("text", "Der Vertrag läuft vom 01.01.2024 bis zum 31.12.2026. "
                 "Die Kündigungsfrist beträgt 3 Monate zum Vertragsende. "
                 "Ohne Kündigung verlängert sich der Vertrag um jeweils 12 Monate."),
        ("unter", "§ 4 Qualität"),
        ("text", "Es gilt die Qualitätssicherungsvereinbarung QSV-2023-02."),
    ])


# ---------------------------------------------------------------------------
# Gold-Daten
# ---------------------------------------------------------------------------
def gold_profile() -> list[dict]:
    return [{"lieferant": lf["name"], "slug": lf["slug"], "land": lf["land"],
             **{f: lf[f] for f in PROFIL_FELDER}} for lf in LIEFERANTEN]


def gold_fragen() -> list[dict]:
    """Fragen mit erwarteter Quelle und Stichworten. Eine Antwort gilt als korrekt,
    wenn sie mindestens ein Stichwort jeder Gruppe in 'erwartet' enthält."""
    L = {lf["slug"]: lf for lf in LIEFERANTEN}
    f = []

    def add(frage, quellen, erwartet):
        f.append({"id": f"F{len(f) + 1:02d}", "frage": frage, "quellen": quellen, "erwartet": erwartet})

    add("Welche Lieferanten verlangen eine Preisgleitklausel?",
        ["Angebot_feldmann.pdf", "Angebot_precimet.pdf"], [["Feldmann"], ["Precimet"]])
    add("Wie hoch ist der Stückpreis von Alpen Drehtechnik?",
        ["Angebot_alpen.pdf"], [["18,70"]])
    add("Welche Lieferzeit bietet Precimet an?",
        ["Angebot_precimet.pdf"], [["5 Wochen", "35"]])
    add("Welches Zahlungsziel bietet Officina Meccanica Rossi?",
        ["Angebot_rossi.pdf"], [["60 Tage"]])
    add("Hat Officina Meccanica Rossi ein ISO 9001 Zertifikat?",
        ["Selbstauskunft_rossi.pdf"], [["Vorbereitung", "kein gültiges", "nein"]])
    add("Wie hoch ist der CO2-Fußabdruck pro Stück bei Swiss Turning?",
        ["Selbstauskunft_swissturning.pdf"], [["2,2"]])
    add("Bis wann läuft der Rahmenvertrag mit Brenner?",
        ["Rahmenvertrag_brenner_2024.pdf"], [["31.12.2026"]])
    add("Welche Kündigungsfrist hat der Rahmenvertrag mit Brenner?",
        ["Rahmenvertrag_brenner_2024.pdf"], [["3 Monate"]])
    add("Welche maximale Lieferzeit erlaubt die Einkaufsrichtlinie für Seriendrehteile?",
        ["Einkaufsrichtlinie_Hagemann.pdf"], [["30 Kalendertage", "30 Tage"]])
    add("Wer muss Preisgleitklauseln freigeben?",
        ["Einkaufsrichtlinie_Hagemann.pdf"], [["Geschäftsführung"]])
    add("Wie hoch ist die Mindestbestellmenge bei Kovotech?",
        ["Angebot_kovotech.pdf"], [[ganzzahl(L["kovotech"]["mindestbestellmenge"])]])
    add("Welchen Skonto gewährt Alpen Drehtechnik?",
        ["Angebot_alpen.pdf"], [["3 %", "3%"]])
    add("Wie hoch ist die Liefertreue von Rheinwerk Komponenten?",
        ["Selbstauskunft_rheinwerk.pdf"], [["98"]])
    add("Welche Reklamationsquote meldet Precimet?",
        ["Selbstauskunft_precimet.pdf"], [["2,6"]])
    add("Wie viele Mitarbeiter hat Kovotech?",
        ["Selbstauskunft_kovotech.pdf"], [[str(L["kovotech"]["mitarbeiter"])]])
    add("Wie lange ist das Angebot von Feldmann gültig?",
        ["Angebot_feldmann.pdf"], [[L["feldmann"]["gueltig_bis"]]])
    add("Welcher Werkstoff ist für die Antriebswelle AW-40 gefordert?",
        ["Anfrage_AS-2027-014.pdf"], [["C45"]])
    add("Wie hoch ist der geplante Jahresbedarf der Anfrage AS-2027-014?",
        ["Anfrage_AS-2027-014.pdf"], [[ganzzahl(JAHRESBEDARF)]])
    add("Wie hoch ist der Stückpreis von Kovotech?",
        ["Angebot_kovotech.pdf"], [["16,40", "1.640"]])
    add("Ab welchem Jahresvolumen muss die Geschäftsführung Rahmenverträge freigeben?",
        ["Einkaufsrichtlinie_Hagemann.pdf"], [["200.000"]])
    return f


def business() -> dict:
    return {
        "auftraggeber": AUFTRAGGEBER,
        "anfrage": ANFRAGE_NR,
        "artikel": ARTIKEL,
        "jahresbedarf_stueck": JAHRESBEDARF,
        "bestandslieferant": next(lf["name"] for lf in LIEFERANTEN if lf["slug"] == BESTANDSLIEFERANT),
        "manuell_minuten_pro_angebot": 45,
        "manuell_minuten_bewertung": 120,
        "ausschreibungen_pro_jahr": 25,
        "stundensatz_einkauf_eur": 65,
        "hinweis": "Annahmen für die Wirtschaftlichkeitsrechnung. In der Oberfläche anpassbar.",
    }


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    GOLD.mkdir(parents=True, exist_ok=True)
    for lf in LIEFERANTEN:
        angebot(lf)
        selbstauskunft(lf)
    interne_dokumente()

    def dump(name, daten):
        (GOLD / name).write_text(json.dumps(daten, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    dump("profile.json", gold_profile())
    dump("fragen.json", gold_fragen())
    (ROOT / "data" / "business.json").write_text(
        json.dumps(business(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(list(DOCS.glob('*.pdf')))} PDFs, {len(LIEFERANTEN)} Gold-Profile, "
          f"{len(gold_fragen())} Gold-Fragen erzeugt.")


if __name__ == "__main__":
    main()
