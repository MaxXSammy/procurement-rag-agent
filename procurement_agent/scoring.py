"""Gewichtete Lieferantenbewertung mit K.-o.-Kriterien.

Ablauf:
  1. K.-o.-Regeln prüfen (z. B. ISO 9001 Pflicht laut Einkaufsrichtlinie)
  2. Jedes Kriterium über die zugelassenen Lieferanten per Min-Max auf 0–1 skalieren
  3. Mit den normierten Gewichten multiplizieren und aufsummieren
Fehlende Werte zählen als schlechtester Wert und werden als Hinweis ausgegeben.
"""

from dataclasses import dataclass, field

from procurement_agent.extraction import Lieferantenprofil


@dataclass(frozen=True)
class Kriterium:
    feld: str
    label: str
    gewicht: float
    hoeher_besser: bool


STANDARD_KRITERIEN = [
    Kriterium("stueckpreis_eur", "Stückpreis", 40, False),
    Kriterium("lieferzeit_tage", "Lieferzeit", 15, False),
    Kriterium("liefertreue_prozent", "Liefertreue", 15, True),
    Kriterium("reklamationsquote_prozent", "Reklamationsquote", 10, False),
    Kriterium("zahlungsziel_tage", "Zahlungsziel", 10, True),
    Kriterium("co2_kg_pro_stueck", "CO2 je Stück", 10, False),
]


@dataclass(frozen=True)
class KoRegeln:
    """Standardwerte entsprechen der Einkaufsrichtlinie in den Testdaten."""

    iso_9001_pflicht: bool = True
    max_lieferzeit_tage: int | None = 30
    preisgleitklausel_erlaubt: bool = False
    min_zahlungsziel_tage: int | None = 30


@dataclass
class Bewertung:
    profil: Lieferantenprofil
    score: float = 0.0
    teilscores: dict[str, float] = field(default_factory=dict)
    ausschluss: list[str] = field(default_factory=list)
    hinweise: list[str] = field(default_factory=list)
    rang: int | None = None

    @property
    def zugelassen(self) -> bool:
        return not self.ausschluss


def pruefe_ko(p: Lieferantenprofil, r: KoRegeln) -> list[str]:
    gruende = []
    if r.iso_9001_pflicht and not p.iso_9001:
        gruende.append("kein gültiges ISO 9001 Zertifikat")
    if r.max_lieferzeit_tage is not None and p.lieferzeit_tage is not None and p.lieferzeit_tage > r.max_lieferzeit_tage:
        gruende.append(f"Lieferzeit {p.lieferzeit_tage} Tage > {r.max_lieferzeit_tage}")
    if not r.preisgleitklausel_erlaubt and p.preisgleitklausel:
        gruende.append("Preisgleitklausel ohne Freigabe")
    if r.min_zahlungsziel_tage is not None and p.zahlungsziel_tage is not None and p.zahlungsziel_tage < r.min_zahlungsziel_tage:
        gruende.append(f"Zahlungsziel {p.zahlungsziel_tage} Tage < {r.min_zahlungsziel_tage}")
    return gruende


def ko_hinweise(p: Lieferantenprofil, r: KoRegeln) -> list[str]:
    """Fehlende Angaben, wegen derer eine K.-o.-Regel nicht geprüft werden konnte."""
    hinweise = []
    if r.max_lieferzeit_tage is not None and p.lieferzeit_tage is None:
        hinweise.append("Lieferzeit fehlt, K.-o.-Prüfung nicht möglich")
    if r.min_zahlungsziel_tage is not None and p.zahlungsziel_tage is None:
        hinweise.append("Zahlungsziel fehlt, K.-o.-Prüfung nicht möglich")
    return hinweise


def bewerte(
    profile: list[Lieferantenprofil],
    kriterien: list[Kriterium] = STANDARD_KRITERIEN,
    regeln: KoRegeln = KoRegeln(),
) -> list[Bewertung]:
    bewertungen = [Bewertung(p, ausschluss=pruefe_ko(p, regeln), hinweise=ko_hinweise(p, regeln)) for p in profile]
    zugelassen = [b for b in bewertungen if b.zugelassen]
    summe = sum(k.gewicht for k in kriterien) or 1.0

    for k in kriterien:
        werte = [getattr(b.profil, k.feld) for b in zugelassen if getattr(b.profil, k.feld) is not None]
        lo, hi = (min(werte), max(werte)) if werte else (0.0, 0.0)
        for b in zugelassen:
            wert = getattr(b.profil, k.feld)
            if wert is None:
                norm = 0.0
                b.hinweise.append(f"{k.label} fehlt, als schlechtester Wert gewertet")
            elif hi == lo:
                norm = 1.0
            else:
                norm = (wert - lo) / (hi - lo)
                norm = norm if k.hoeher_besser else 1 - norm
            b.teilscores[k.label] = norm
            b.score += norm * k.gewicht / summe

    zugelassen.sort(key=lambda b: -b.score)
    for i, b in enumerate(zugelassen, start=1):
        b.rang = i
    ausgeschlossen = [b for b in bewertungen if not b.zugelassen]
    return zugelassen + sorted(ausgeschlossen, key=lambda b: b.profil.lieferant)
