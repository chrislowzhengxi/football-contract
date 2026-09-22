"""Evidence-identity gate.

Named for the failure it exists to stop: the first demonstration dataset gave
Santiago Gimenez's 2022 Cruz Azul -> Feyenoord transfer three sources, none of
which was about him. Two were Alvaro Morata loan reports and one was a
Riccardo Sottil announcement; all three name Gimenez in a passing aside, so
every full-name check upstream passed.
"""
import pytest

from src.stage2.discovery import identity

SOTTIL = (
    "Sottil è un nuovo giocatore del Milan. Prestito con diritto di riscatto",
    "https://tuttomercatoweb.com/serie-a/sottil-e-un-nuovo-giocatore-del-milan",
    "Il Milan ha annunciato Riccardo Sottil dalla Fiorentina in prestito con "
    "diritto di riscatto. Oggi intanto il Milan ha annunciato l'acquisto del suo "
    "colpo da novanta dell'inverno, il centravanti messicano Santiago Gimenez.")

MORATA = (
    "Spain striker Alvaro Morata joins Galatasaray from AC Milan on loan",
    "https://foxsports.com/articles/soccer/spain-striker-alvaro-morata-joins-galatasaray",
    "Alvaro Morata has joined Galatasaray on loan from AC Milan. Morata was "
    "replaced at Milan by Mexican striker Santiago Gimenez, who signed from "
    "Feyenoord. Milan said Gimenez signed a contract until June 2029.")


def test_passing_mention_is_not_evidence():
    v = identity.assess(*SOTTIL, "Santiago Gimenez", ["CD Cruz Azul", "Feyenoord"])
    assert not v.admit


def test_document_about_another_player_is_rejected_even_when_a_family_club_appears():
    """Feyenoord IS one of Gimenez's clubs, and the article names it next to
    him - but the article is about Morata, and the contract it mentions is the
    2029 Milan deal, not the 2022 Feyenoord one."""
    v = identity.assess(*MORATA, "Santiago Gimenez", ["CD Cruz Azul", "Feyenoord"])
    assert not v.admit
    assert "morata" in v.reason_text.lower()


def test_subject_of_the_document_is_admitted():
    v = identity.assess(*SOTTIL, "Riccardo Sottil", ["Cagliari", "Fiorentina"])
    assert v.admit and v.strength == "strong"


def test_player_named_in_title_is_strong():
    v = identity.assess("Mercato : Rennes a activé la clause de rachat de Matthis Abline",
                        "https://stade-rennais-online.com/Mercato-Rennes-clause-rachat",
                        "Rennes a fait marcher sa clause de rachat de Matthis Abline "
                        "à hauteur de 2 millions d'euros auprès du FC Nantes.",
                        "Matthis Abline", ["FC Nantes", "Stade Rennais"])
    assert v.admit and v.strength == "strong"


def test_document_never_naming_the_player_is_rejected():
    v = identity.assess("Serie A transfer roundup", "https://example.com/roundup",
                        "Several clubs completed deals this week.",
                        "Santiago Gimenez", ["Feyenoord"])
    assert not v.admit
    assert "never names" in v.reason_text


def test_surname_alone_is_not_enough():
    """Fabio Ronaldo must not inherit Cristiano Ronaldo's coverage."""
    v = identity.assess("Cristiano Ronaldo joins Juventus from Real Madrid",
                        "https://example.com/ronaldo-juventus",
                        "Cristiano Ronaldo has completed his move to Juventus "
                        "from Real Madrid for a fee of 117 million euros.",
                        "Fábio Ronaldo", ["Rio Ave", "Estrela Amadora"])
    assert not v.admit


def test_a_generic_title_still_admits_a_document_that_is_mostly_about_the_player():
    body = ("ARBITRAL AWARD. " + "The player Rafael Leao left Sporting for Lille. " * 6)
    v = identity.assess("ARBITRAL AWARD COURT OF ARBITRATION FOR SPORT",
                        "https://tas-cas.org/award.pdf", body,
                        "Rafael Leão", ["Sporting", "Lille"])
    assert v.admit and v.strength in ("strong", "medium")


def test_no_family_club_anywhere_near_the_player_is_rejected():
    v = identity.assess("Edin Dzeko signs for Fenerbahce",
                        "https://example.com/dzeko-fenerbahce",
                        "Edin Dzeko has signed for Fenerbahce on a free transfer "
                        "after leaving Inter at the end of his contract.",
                        "Edin Dzeko", ["Man City", "Roma"])
    assert not v.admit
    assert "club" in v.reason_text


def test_gazetteers_load_from_stage1c():
    assert len(identity.player_gazetteer()) > 5000
    assert len(identity.club_gazetteer()) > 2000


def test_club_stopwords_are_not_treated_as_identifying():
    toks = identity.club_tokens_of(["Manchester United", "Sporting CP"])
    assert "united" not in toks and "sporting" not in toks
    assert "manchester" in toks


def test_clubs_mentioned_finds_real_clubs_only():
    got = identity.clubs_mentioned(
        "LOSC Lille shall pay Sporting the amounts receivable from Milan")
    assert "lille" in got and "milan" in got
    assert "shall" not in got and "amounts" not in got


def test_club_spelling_variants_are_treated_as_the_same_club():
    """Stade Rennais is "Rennes" in running text and shares no whole token."""
    fam = identity.club_tokens_of(["Stade Rennais", "FC Nantes"])
    assert identity.club_tokens_overlap({"rennes"}, fam) == {"rennes"}
    assert identity.club_tokens_overlap({"inter"}, identity.club_tokens_of(
        ["Internazionale"])) == {"inter"}
    assert identity.club_tokens_overlap({"manchester"}, identity.club_tokens_of(
        ["Man City", "Manchester United"])) == {"manchester"}


def test_a_genuinely_different_club_does_not_overlap():
    fam = identity.club_tokens_of(["Stade Rennais", "FC Nantes"])
    assert identity.club_tokens_overlap({"monaco"}, fam) == set()
    assert identity.club_tokens_overlap({"juventus"}, fam) == set()
