from procurement_agent.extraction import Lieferantenprofil
from procurement_agent.scoring import KoRegeln, Kriterium, bewerte


def profil(slug, **werte):
    basis = dict(stueckpreis_eur=10.0, lieferzeit_tage=10, zahlungsziel_tage=30, iso_9001=True, preisgleitklausel=False)
    return Lieferantenprofil(lieferant=slug.title(), slug=slug, **(basis | werte))


PREIS = [Kriterium("stueckpreis_eur", "Preis", 1, False)]


def test_guenstigster_gewinnt_bei_reinem_preis():
    b = bewerte([profil("a", stueckpreis_eur=12), profil("b", stueckpreis_eur=10)], PREIS)
    assert [x.profil.slug for x in b] == ["b", "a"]
    assert b[0].score == 1.0 and b[1].score == 0.0


def test_ko_kriterien_schliessen_aus():
    b = bewerte([
        profil("ok"),
        profil("ohneiso", iso_9001=False),
        profil("langsam", lieferzeit_tage=40),
        profil("gleit", preisgleitklausel=True),
        profil("kurzziel", zahlungsziel_tage=14),
    ], PREIS, KoRegeln())
    assert [x.profil.slug for x in b if x.zugelassen] == ["ok"]
    assert len([x for x in b if not x.zugelassen]) == 4


def test_ko_regeln_abschaltbar():
    regeln = KoRegeln(iso_9001_pflicht=False, max_lieferzeit_tage=None, preisgleitklausel_erlaubt=True)
    b = bewerte([profil("ohneiso", iso_9001=False, preisgleitklausel=True)], PREIS, regeln)
    assert b[0].zugelassen


def test_fehlender_wert_gibt_hinweis():
    b = bewerte([profil("a"), profil("b", stueckpreis_eur=None)], PREIS)
    fehlt = next(x for x in b if x.profil.slug == "b")
    assert fehlt.score == 0.0
    assert any("fehlt" in h for h in fehlt.hinweise)


def test_gewichte_werden_normiert():
    kriterien = [Kriterium("stueckpreis_eur", "Preis", 30, False), Kriterium("lieferzeit_tage", "LZ", 10, False)]
    b = bewerte([profil("a", stueckpreis_eur=10, lieferzeit_tage=20), profil("b", stueckpreis_eur=20, lieferzeit_tage=10)], kriterien)
    assert round(b[0].score, 2) == 0.75
