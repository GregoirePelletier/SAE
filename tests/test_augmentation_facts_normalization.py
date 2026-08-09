"""
tests/test_augmentation_facts_normalization.py — non-régression pour le fix
appliqué suite à l'audit méthodologique (RESULTS_TESTS.md §39, 2026-08-07) :
`_facts()`/`_FACT_RE` (src/data/augmentation.py) rejetaient `facts_lost` un
reformatage pur (téléphone respacé, date sans zéro de padding, montant en
virgule au lieu du point) alors qu'aucun contenu n'était réellement perdu.
"""
from src.data.augmentation import _facts


def test_phone_number_reformatting_is_not_flagged_as_lost():
    assert not _facts("au 0476356490") - _facts("au 0476 35 64 90")
    assert not _facts("au 0476 35 64 90") - _facts("au 04.76.35.64.90")


def test_date_zero_padding_is_not_flagged_as_lost():
    assert not _facts("le 18/7/13") - _facts("le 18/07/2013")


def test_monetary_decimal_separator_is_not_flagged_as_lost():
    assert not _facts("montant 20.73€") - _facts("montant 20,73 €")


def test_genuine_fact_loss_is_still_detected():
    assert _facts("contrat 908449671 résilié") - _facts("votre contrat est résilié")


def test_genuine_date_change_is_still_detected():
    assert _facts("rdv le 18/07/2013") - _facts("rdv le 20/07/2013")


def test_unchanged_postal_code_has_no_diff():
    assert not _facts("75015 Paris") - _facts("75015 Paris, France")
