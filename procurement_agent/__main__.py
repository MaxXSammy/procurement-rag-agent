"""Kommandozeile.

    python -m procurement_agent frage "Welche Lieferanten verlangen eine Preisgleitklausel?"
    python -m procurement_agent ranking
    python -m procurement_agent eval
Option --offline erzwingt den Betrieb ohne Ollama.
"""

import argparse

from procurement_agent.agent import formatiere_ranking
from procurement_agent.evaluation import gesamtevaluation, geschaeftskpis
from procurement_agent.system import baue_system, lade_business


def main() -> None:
    parser = argparse.ArgumentParser(prog="procurement_agent")
    parser.add_argument("--offline", action="store_true", help="ohne Ollama ausführen")
    sub = parser.add_subparsers(dest="befehl", required=True)
    p_frage = sub.add_parser("frage", help="Frage an den Agenten stellen")
    p_frage.add_argument("text")
    sub.add_parser("ranking", help="Lieferantenrangliste ausgeben")
    sub.add_parser("eval", help="Evaluation ausführen und protokollieren")
    args = parser.parse_args()

    system = baue_system("offline" if args.offline else "auto")
    print(f"[{system.beschreibung}]\n")

    if args.befehl == "frage":
        ergebnis = system.frage(args.text)
        print(ergebnis.text)
        for i, t in enumerate(ergebnis.quellen, start=1):
            print(f"  [{i}] {t.chunk.quelle} (Seite {t.chunk.seite}, Ähnlichkeit {t.score:.2f})")
        if ergebnis.unbelegte_zahlen:
            print(f"  Achtung, nicht belegt: {', '.join(ergebnis.unbelegte_zahlen)}")
    elif args.befehl == "ranking":
        print(formatiere_ranking(system.bewertungen()))
    else:
        lauf = gesamtevaluation(system)
        kpi = geschaeftskpis(system.bewertungen(), lade_business(), lauf.dauer_s)
        e = lauf.eintrag
        print(f"Retrieval-Trefferquote:  {e['retrieval_trefferquote']:.0%}")
        print(f"Antwortgenauigkeit:      {e['antwortgenauigkeit']:.0%}")
        print(f"Belegquote:              {e['belegquote']:.0%}")
        print(f"Extraktionsgenauigkeit:  {e['extraktionsgenauigkeit']:.0%}")
        if kpi.einsparung_eur_jahr is not None:
            print(f"Einsparpotenzial:        {_de(kpi.einsparung_eur_jahr)} EUR/Jahr ({kpi.empfehlung})")
        print(f"Zeitersparnis:           {_de(kpi.stunden_gespart_jahr)} h/Jahr")
        for f in lauf.extraktion.fehler:
            print(f"  Extraktionsfehler: {f['lieferant']} · {f['feld']}: {f['extrahiert']} statt {f['soll']}")


def _de(zahl: float) -> str:
    return f"{zahl:,.0f}".replace(",", ".")


if __name__ == "__main__":
    main()
