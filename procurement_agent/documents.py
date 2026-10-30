"""PDFs laden und in Chunks mit Metadaten zerlegen."""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from procurement_agent.config import CHUNK_OVERLAP_LINES, CHUNK_SIZE

logger = logging.getLogger(__name__)

# Dateinamen-Konvention der Testdaten: <Typ>_<lieferant>[_...].pdf
LIEFERANTEN_TYPEN = {"Angebot", "Selbstauskunft", "Rahmenvertrag"}


@dataclass(frozen=True)
class Dokument:
    name: str
    typ: str
    lieferant: str | None  # Slug, z. B. "alpen"; None bei internen Dokumenten
    titel: str  # erste Zeile, z. B. Firmenname
    seiten: tuple[str, ...]

    @property
    def text(self) -> str:
        return "\n".join(self.seiten)


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str  # mit Kontext-Kopfzeile, damit Retrieval den Lieferanten kennt
    quelle: str
    seite: int
    lieferant: str | None
    typ: str


def metadaten_aus_dateiname(name: str) -> tuple[str, str | None]:
    teile = Path(name).stem.split("_")
    typ = teile[0]
    lieferant = teile[1].lower() if typ in LIEFERANTEN_TYPEN and len(teile) > 1 else None
    return typ, lieferant


def lade_dokument(pfad: Path) -> Dokument:
    seiten = tuple(_bereinige(p.extract_text() or "") for p in PdfReader(pfad).pages)
    typ, lieferant = metadaten_aus_dateiname(pfad.name)
    titel = next((z for z in seiten[0].splitlines() if z.strip()), pfad.stem) if seiten else pfad.stem
    return Dokument(pfad.name, typ, lieferant, titel.strip(), seiten)


MIN_ZEICHEN_PRO_SEITE = 50  # darunter ist das PDF vermutlich gescannt und hat keine Textebene


def lade_dokumente(ordner: Path, probleme: list[str] | None = None) -> list[Dokument]:
    """Lädt alle PDFs. Ein kaputtes PDF wird übersprungen und gemeldet, statt den Start abzubrechen.

    Gescannte PDFs ohne Textebene werden geladen, aber gemeldet, weil Suche und Extraktion dort nichts finden.
    """
    probleme = probleme if probleme is not None else []
    dokumente = []
    for pfad in sorted(ordner.glob("*.pdf")):
        try:
            dok = lade_dokument(pfad)
        except (PdfReadError, ValueError, OSError) as fehler:
            probleme.append(f"{pfad.name}: nicht lesbar ({type(fehler).__name__})")
            logger.warning("PDF nicht lesbar, übersprungen", extra={"datei": pfad.name, "fehler": str(fehler)})
            continue
        if sum(len(s.strip()) for s in dok.seiten) < MIN_ZEICHEN_PRO_SEITE * max(len(dok.seiten), 1):
            probleme.append(f"{pfad.name}: kaum Text, vermutlich gescannt (OCR nötig)")
            logger.warning("PDF ohne Textebene", extra={"datei": pfad.name})
        dokumente.append(dok)
    return dokumente


def chunke(dok: Dokument, groesse: int = CHUNK_SIZE) -> list[Chunk]:
    """Zeilenweise packen bis zur Zielgröße; eine Zeile Überlappung erhält den Zusammenhang.

    Jeder Chunk bekommt eine Kopfzeile mit Dokumenttitel und Typ. So findet die Suche
    "Stückpreis von Alpen" auch dann, wenn der Firmenname nur oben im PDF steht.
    """
    kopf = f"[{dok.typ} | {dok.titel}]"
    chunks: list[Chunk] = []
    for nr, seite in enumerate(dok.seiten, start=1):
        zeilen = [z for z in seite.splitlines() if z.strip()]
        puffer: list[str] = []
        for zeile in zeilen:
            if puffer and len(" ".join(puffer)) + len(zeile) > groesse:
                chunks.append(_baue(dok, nr, len(chunks), kopf, puffer))
                puffer = puffer[-CHUNK_OVERLAP_LINES:] if CHUNK_OVERLAP_LINES else []
            puffer.append(zeile)
        if puffer:
            chunks.append(_baue(dok, nr, len(chunks), kopf, puffer))
    return chunks


def chunke_alle(dokumente: list[Dokument]) -> list[Chunk]:
    return [c for d in dokumente for c in chunke(d)]


def _baue(dok: Dokument, seite: int, idx: int, kopf: str, zeilen: list[str]) -> Chunk:
    return Chunk(
        id=f"{dok.name}#{idx}",
        text=kopf + "\n" + "\n".join(zeilen),
        quelle=dok.name,
        seite=seite,
        lieferant=dok.lieferant,
        typ=dok.typ,
    )


def _bereinige(text: str) -> str:
    text = text.replace(" ", " ")
    return re.sub(r"[ \t]+", " ", text)
