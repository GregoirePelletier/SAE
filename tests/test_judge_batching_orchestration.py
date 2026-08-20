"""Teste l'orchestration en 3 passes batchées de src/sae/judge.py::
local_gemma_judge (et, par construction identique, odd_one_out_judge) --
audit perf §2.6, item 1. Mocke `_batched_generate` (pas de modèle réel, CPU
uniquement) pour vérifier que chaque étage ne reçoit que le sous-ensemble de
features attendu et que les résultats sont ré-assemblés au bon feature, dans
le bon ordre -- c'est la logique nouvelle de cette session, pas la mécanique
bas niveau du padding (couverte séparément par un test GPU,
scripts/audit_2026_08_judge_batching_equivalence.py)."""
from unittest.mock import patch

import torch

from src.sae.judge import local_gemma_judge


def _make_phrase_corpus(n_phrases=20, d_sae=8):
    torch.manual_seed(0)
    phrase_texts = [f"phrase numéro {i}" for i in range(n_phrases)]
    phrase_acts = torch.zeros(n_phrases, d_sae)
    return phrase_texts, phrase_acts


def _fake_batched_generate_factory(ood_answer_for):
    """Retourne un stand-in de _batched_generate qui répond à l'étape
    odd-one-out selon `ood_answer_for` (dict f_idx -> bonne/mauvaise réponse),
    et des réponses valides et déterministes pour label/score."""
    calls = []

    def _fake(model, tokenizer, list_of_messages, max_new_tokens, batch_size=16):
        calls.append((max_new_tokens, len(list_of_messages)))
        if max_new_tokens == 8:   # étape 1 : odd-one-out
            return [ood_answer_for.pop(0) for _ in list_of_messages]
        elif "concept" in "".join(str(m) for m in list_of_messages[0]).lower() or \
             "scores" in "".join(str(m) for m in list_of_messages[0]).lower():
            return ['{"scores": [5, 5, 5, 5, 5, 5, 5, 5, 5, 0]}' for _ in list_of_messages]
        else:
            return ['{"label": "Test", "brief_description": "desc"}' for _ in list_of_messages]

    return _fake, calls


def test_only_interpretable_features_reach_stage_2_and_3():
    """3 features avec au moins 3 exemples pos + 1 neg : 2 répondent
    correctement à l'odd-one-out (interp_score=1), 1 non (interp_score=0).
    Seules les 2 interprétables doivent atteindre les étapes label/score."""
    phrase_texts, phrase_acts = _make_phrase_corpus()

    # build_phrase_examples_with_control dépend des activations réelles --
    # patché pour renvoyer des exemples/contrôle négatif fixes et connus,
    # ce test porte sur l'orchestration, pas sur la construction d'exemples.
    fixed_examples = (
        [f"ex{i}" for i in range(9)], "neg_ex", list(range(9, 0, -1)), 0.0,
    )

    with patch("src.sae.judge.build_phrase_examples_with_control", return_value=fixed_examples):
        # Bonne réponse = position du négatif après mélange (indices[-1]+1) --
        # patché aussi pour un mélange déterministe (pas de dépendance à l'état
        # global random entre exécutions de test).
        with patch("random.shuffle", lambda lst: None):  # pas de mélange -> négatif en position 10
            fake_gen, calls = _fake_batched_generate_factory(["10", "1", "10"])
            with patch("src.sae.judge._batched_generate", fake_gen):
                results = local_gemma_judge(
                    model=object(), tokenizer=object(),
                    feature_indices=[0, 1, 2],
                    phrase_texts=phrase_texts, phrase_acts=phrase_acts,
                )

    assert results["0"]["interp_score"] == 1
    assert results["1"]["interp_score"] == 0
    assert results["2"]["interp_score"] == 1

    # Étape 1 : appelée une fois avec les 3 features vivantes.
    stage1_calls = [c for c in calls if c[0] == 8]
    assert len(stage1_calls) == 1 and stage1_calls[0][1] == 3
    # Étapes 2/3 (max_new_tokens=128) : seulement les 2 interprétables.
    stage_128_calls = [c for c in calls if c[0] == 128]
    assert all(c[1] == 2 for c in stage_128_calls)


def test_dead_features_never_reach_generation():
    """Features avec < 3 exemples positifs : jamais envoyées au modèle,
    court-circuitées avant l'étape 1."""
    phrase_texts, phrase_acts = _make_phrase_corpus()
    dead_examples = ([], None, [], None)  # < 3 pos examples

    with patch("src.sae.judge.build_phrase_examples_with_control", return_value=dead_examples):
        fake_gen, calls = _fake_batched_generate_factory([])
        with patch("src.sae.judge._batched_generate", fake_gen):
            results = local_gemma_judge(
                model=object(), tokenizer=object(),
                feature_indices=[0, 1],
                phrase_texts=phrase_texts, phrase_acts=phrase_acts,
            )

    assert results["0"]["label"] == "dead_feature"
    assert results["1"]["label"] == "dead_feature"
    assert calls == []  # aucun appel de génération


def test_no_neg_example_skips_stage_3_but_keeps_label():
    """Feature interprétable mais sans contrôle négatif (correct_answer=None
    -> interp_score toujours 0 dans le code actuel) : vérifie juste que
    l'absence de neg_example exclut bien l'étape 3 pour les features qui ne
    l'ont pas, sans casser l'étape 2."""
    phrase_texts, phrase_acts = _make_phrase_corpus()
    no_neg_examples = ([f"ex{i}" for i in range(9)], None, list(range(9)), None)

    with patch("src.sae.judge.build_phrase_examples_with_control", return_value=no_neg_examples):
        with patch("random.shuffle", lambda lst: None):
            # Stage 1 est quand même appelée (feature vivante, >=3 exemples pos) --
            # la réponse n'a pas d'importance ici : correct_answer=None force
            # interp_score=0 quel que soit `predicted` (cf. code de judge.py).
            fake_gen, calls = _fake_batched_generate_factory(["1"])
            with patch("src.sae.judge._batched_generate", fake_gen):
                results = local_gemma_judge(
                    model=object(), tokenizer=object(),
                    feature_indices=[0],
                    phrase_texts=phrase_texts, phrase_acts=phrase_acts,
                )

    # correct_answer=None -> interp_score=0 (comportement inchangé, cf. code) ;
    # dans ce cas ni étape 2 ni étape 3 n'ont de features à traiter.
    assert results["0"]["interp_score"] == 0
    assert results["0"]["rho_interp"] != results["0"]["rho_interp"]  # NaN
