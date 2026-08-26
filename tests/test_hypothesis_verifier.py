"""Teste src/analysis/hypothesis_verifier.py (App K.1) : parsing de la
réponse du juge, construction de la matrice hypothèse × document (mock de
_batched_generate, CPU uniquement -- même style que
test_judge_batching_orchestration.py), et les métriques verification_rate/
coverage sur une matrice synthétique dont le résultat attendu est connu à la
main."""
from unittest.mock import patch

import numpy as np

from src.analysis.hypothesis_verifier import (
    _parse_verification_answer,
    compute_verification_metrics,
    verify_hypotheses,
)


def test_parse_verification_answer_standard_format():
    resp = "REASONING: le texte parle clairement d'énergie.\nANSWER: YES"
    assert _parse_verification_answer(resp) is True


def test_parse_verification_answer_no():
    resp = "REASONING: aucun rapport.\nANSWER: NO"
    assert _parse_verification_answer(resp) is False


def test_parse_verification_answer_missing_answer_tag_defaults_to_no():
    """Consigne 5 du prompt : en cas d'incertitude/format cassé, NO (repli sûr)."""
    resp = "Le texte semble lié mais je ne suis pas sûr."
    assert _parse_verification_answer(resp) is False


def test_verify_hypotheses_builds_matrix_from_batched_generate():
    """2 hypothèses x 3 documents : vérifie que chaque case de la matrice
    correspond bien à la paire (hypothèse, document) attendue, pas seulement
    au bon COMPTE de résultats -- couvre un bug de désalignement possible
    entre l'ordre des messages construits et l'ordre des réponses reçues."""
    hypotheses = ["parle d'énergie", "parle de sport"]
    documents = ["doc0", "doc1", "doc2"]

    # Réponses déterministes par hypothèse : la 1re hypothèse vérifie seulement
    # doc0, la 2e vérifie doc1 et doc2 -- encode directement le pattern attendu
    # dans le contenu du prompt pour éviter toute dépendance à l'ordre d'appel.
    def _fake(model, tokenizer, list_of_messages, max_new_tokens, batch_size=16):
        out = []
        for msgs in list_of_messages:
            content = msgs[0]["content"]
            if "énergie" in content and "doc0" in content:
                out.append("REASONING: ok\nANSWER: YES")
            elif "sport" in content and ("doc1" in content or "doc2" in content):
                out.append("REASONING: ok\nANSWER: YES")
            else:
                out.append("REASONING: non\nANSWER: NO")
        return out

    with patch("src.analysis.hypothesis_verifier._batched_generate", _fake):
        matrix = verify_hypotheses(model=None, tokenizer=None, hypotheses=hypotheses, documents=documents)

    expected = np.array([[1, 0, 0], [0, 1, 1]])
    assert matrix.shape == (2, 3)
    np.testing.assert_array_equal(matrix, expected)


def test_compute_verification_metrics_verification_rate_and_coverage():
    """3 hypothèses, 4 documents (2 groupe cible, 2 hors groupe) :
    - H0 vérifiée sur les 2 docs cible, jamais hors groupe -> diff=1.0, valide.
    - H1 vérifiée partout (aucune différence) -> diff=0.0, invalide.
    - H2 vérifiée seulement hors groupe -> diff=-1.0, valide mais PAS dans le
      sens cible (ne doit pas contribuer à la couverture).
    group_mask = [True, True, False, False].
    """
    matrix = np.array([
        [1, 1, 0, 0],  # H0 : cible seulement
        [1, 0, 1, 0],  # H1 : moitié/moitié des deux côtés -> diff=0
        [0, 0, 1, 1],  # H2 : hors-groupe seulement (sens inverse)
    ])
    group_mask = np.array([True, True, False, False])

    per_hyp, summary = compute_verification_metrics(matrix, group_mask, threshold=0.01)

    assert list(per_hyp["valid"]) == [True, False, True]
    assert summary["n_hypotheses"] == 3
    assert summary["n_documents_in_group"] == 2
    assert summary["n_documents_out_group"] == 2
    # verification_rate = fraction d'hypothèses valides (H0 et H2) = 2/3
    assert abs(summary["verification_rate"] - 2 / 3) < 1e-9
    # coverage : seule H0 est valide ET dans le sens cible -> couvre doc0,doc1 (les 2) = 1.0
    assert abs(summary["coverage"] - 1.0) < 1e-9


def test_compute_verification_metrics_no_valid_hypothesis_zero_coverage():
    matrix = np.array([[1, 0, 1, 0]])  # diff = 0.5 - 0.5 = 0 -> invalide
    group_mask = np.array([True, True, False, False])
    _, summary = compute_verification_metrics(matrix, group_mask, threshold=0.01)
    assert summary["verification_rate"] == 0.0
    assert summary["coverage"] == 0.0
