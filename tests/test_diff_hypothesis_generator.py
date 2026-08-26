"""Teste src/analysis/diff_hypothesis_generator.py -- génération d'hypothèses
structurées (App. D.2). Mocke _batched_generate (CPU uniquement, pas de
modèle réel)."""
import json
from unittest.mock import MagicMock, patch

from src.analysis.diff_hypothesis_generator import (
    generate_structured_diff_hypotheses,
    _parse_hypotheses_json,
    _format_feature_block,
)


def _make_feature(feature_id=0, label="Réclamation client", diff=0.35,
                   pos="texte <<positif>>", neg="texte négatif"):
    return {"feature_id": feature_id, "label": label, "percentage_difference": diff,
            "pos_example": pos, "neg_example": neg}


VALID_HYPOTHESIS = {
    "dataset": "target", "description": "This response complains about billing.",
    "feature_ids": [0], "examples": ["ex1"], "percentage_difference": 0.35, "confidence": 0.8,
}


def test_format_feature_block_includes_all_fields():
    block = _format_feature_block(_make_feature())
    assert "Feature ID: 0" in block
    assert "Réclamation client" in block
    assert "+0.35" in block
    assert "<<positif>>" in block
    assert "texte négatif" in block


def test_format_feature_block_handles_missing_negative_example():
    f = _make_feature(neg=None)
    block = _format_feature_block(f)
    assert "Negative example" not in block   # ligne omise, pas de placeholder
    assert "Feature ID: 0" in block


def test_format_feature_block_handles_no_examples_at_all():
    # Cas réel du driver App D.2 sur le corpus de diffing : seul le label déjà
    # produit par le juge est disponible, pas de ré-extraction de fragments.
    f = {"feature_id": 3, "label": "vocabulaire sportif", "percentage_difference": -0.12}
    block = _format_feature_block(f)
    assert "Feature ID: 3" in block
    assert "Positive example" not in block
    assert "Negative example" not in block


def test_parse_hypotheses_json_valid_array():
    response = f"Some preamble text.\n{json.dumps([VALID_HYPOTHESIS])}\nTrailing text."
    parsed = _parse_hypotheses_json(response)
    assert len(parsed) == 1
    assert parsed[0]["dataset"] == "target"


def test_parse_hypotheses_json_drops_incomplete_objects():
    incomplete = {"dataset": "target", "description": "no other fields"}
    response = json.dumps([VALID_HYPOTHESIS, incomplete])
    parsed = _parse_hypotheses_json(response)
    assert len(parsed) == 1


def test_parse_hypotheses_json_malformed_returns_empty_list():
    assert _parse_hypotheses_json("not json at all") == []
    assert _parse_hypotheses_json("") == []


def test_generate_structured_diff_hypotheses_single_call_returns_parsed_list():
    features = [_make_feature(0), _make_feature(1, label="Urgence électricité", diff=-0.2)]
    response = json.dumps([VALID_HYPOTHESIS])

    with patch("src.analysis.diff_hypothesis_generator._batched_generate",
               return_value=[response]) as mock_gen:
        result = generate_structured_diff_hypotheses(
            MagicMock(), object(), features, query="What distinguishes target from other?",
        )

    assert mock_gen.call_count == 1                     # UN SEUL appel, pas un par feature
    assert len(mock_gen.call_args[0][2]) == 1            # un seul message (le prompt global)
    assert result == [VALID_HYPOTHESIS]


def test_generate_structured_diff_hypotheses_truncates_to_num_hypotheses():
    features = [_make_feature(0)]
    response = json.dumps([VALID_HYPOTHESIS, VALID_HYPOTHESIS, VALID_HYPOTHESIS])

    with patch("src.analysis.diff_hypothesis_generator._batched_generate", return_value=[response]):
        result = generate_structured_diff_hypotheses(
            MagicMock(), object(), features, query="q", num_hypotheses=2,
        )

    assert len(result) == 2
