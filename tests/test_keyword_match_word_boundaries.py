"""Teste src/data/preparation.py::keyword_match -- correctif B.11
(AUDIT_SAE_2026-08.md) : substring `in` sans frontière de mot faisait matcher
"vol" contre volume/volley/évolution, "watt" contre Watteau -- bruitant la
vérité terrain du diffing cross-domaine (energy/sports/support)."""
from src.data.preparation import keyword_match


def test_substring_false_positives_no_longer_match():
    assert not keyword_match("Le volume des ventes a augmenté ce trimestre.", ["vol"])
    assert not keyword_match("Le tournoi de volley-ball débute demain.", ["vol"])
    assert not keyword_match("Jean-Antoine Watteau est un peintre du XVIIIe siècle.", ["watt"])


def test_real_keyword_still_matches():
    assert keyword_match("Le vol a été signalé au commissariat.", ["vol"])
    assert keyword_match("La puissance est de 300 watt.", ["watt"])


def test_multi_word_phrase_keyword_still_matches():
    assert keyword_match("Le réseau électrique national a été renforcé.", ["réseau électrique"])
    assert not keyword_match("Un réseau électriquement isolé.", ["réseau électrique"])


def test_case_insensitive():
    assert keyword_match("WATT et Volt sont des unités.", ["watt", "volt"])


def test_empty_text_or_keywords():
    assert not keyword_match("", ["vol"])
    assert not keyword_match("un texte", [])
