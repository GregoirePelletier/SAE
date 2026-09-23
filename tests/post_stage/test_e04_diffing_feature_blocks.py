"""Test CPU rapide, non-regression pour le correctif `percentage_difference`
(scripts/post_stage/e04_diffing.py::build_feature_diff_blocks) -- le champ
recevait log_odds_ratio (non borne) au lieu de freq_diff = freq_A - freq_B
(l'ecart de frequence borne entre -1 et 1 que le prompt du generateur
d'hypotheses promet, diff_hypothesis_generator.py::DIFF_HYPOTHESIS_PROMPT).
Aucun GPU/juge necessaire : build_feature_diff_blocks ne fait que de la
manipulation pandas."""
import pandas as pd
import pytest

from scripts.post_stage.e04_diffing import build_feature_diff_blocks


def _selected_row(feature_id, label, freq_a, freq_b, log_odds_ratio):
    return {
        "feature_id": feature_id, "label": label,
        "freq_A": freq_a, "freq_B": freq_b,
        "freq_diff": freq_a - freq_b,
        "log_odds_ratio": log_odds_ratio,
    }


def test_percentage_difference_is_freq_diff_not_log_odds():
    selected = pd.DataFrame([
        _selected_row(0, "feature test", freq_a=0.8, freq_b=0.2, log_odds_ratio=5.17),
    ])
    blocks = build_feature_diff_blocks(selected)
    assert len(blocks) == 1
    assert blocks[0]["percentage_difference"] == pytest.approx(0.6)
    # Le log-odds n'est en rien la valeur retournee (5.17 != 0.6).
    assert blocks[0]["percentage_difference"] != 5.17


def test_percentage_difference_sign_flips_with_a_b_swap():
    selected_ab = pd.DataFrame([_selected_row(0, "f", freq_a=0.8, freq_b=0.2, log_odds_ratio=5.17)])
    selected_ba = pd.DataFrame([_selected_row(0, "f", freq_a=0.2, freq_b=0.8, log_odds_ratio=-5.17)])
    diff_ab = build_feature_diff_blocks(selected_ab)[0]["percentage_difference"]
    diff_ba = build_feature_diff_blocks(selected_ba)[0]["percentage_difference"]
    assert diff_ab == pytest.approx(0.6)
    assert diff_ba == pytest.approx(-0.6)
    assert diff_ab == pytest.approx(-diff_ba)


def test_percentage_difference_stays_within_bounded_range():
    # freq_A/freq_B sont chacun dans [0,1] par construction (fraction de
    # documents) -- freq_diff est donc necessairement dans [-1,1], contrairement
    # au log-odds-ratio (Haldane) qui peut etre arbitrairement grand.
    selected = pd.DataFrame([
        _selected_row(0, "rare vs frequent", freq_a=0.999, freq_b=0.001, log_odds_ratio=13.8),
    ])
    block = build_feature_diff_blocks(selected)[0]
    assert -1.0 <= block["percentage_difference"] <= 1.0
