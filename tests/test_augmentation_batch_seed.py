"""Teste src/data/augmentation.py::_batch_seed -- correctif B.12
(AUDIT_SAE_2026-08.md) : generate_variants faisait un seul torch.manual_seed
en tête de fonction, puis laissait le flux RNG dériver séquentiellement à
travers tous les lots -- une reprise (lots déjà générés sautés) change la
composition des lots suivants, donc la sortie réelle pour un même aug_id
regénéré, malgré le champ "seed" enregistré dans le JSONL qui promet une
reproductibilité qui n'existait pas. La graine est maintenant dérivée du
contenu du lot (aug_ids triés) : regénérer EXACTEMENT le même lot, dans
n'importe quel run, donne la même graine."""
from src.data.augmentation import _batch_seed


def test_same_batch_content_gives_same_seed():
    assert _batch_seed(0, ["a__x__1", "b__y__2"]) == _batch_seed(0, ["a__x__1", "b__y__2"])


def test_order_independent():
    assert _batch_seed(0, ["a__x__1", "b__y__2"]) == _batch_seed(0, ["b__y__2", "a__x__1"])


def test_different_batch_content_gives_different_seed():
    assert _batch_seed(0, ["a__x__1", "b__y__2"]) != _batch_seed(0, ["a__x__1", "c__z__3"])


def test_different_base_seed_gives_different_seed():
    assert _batch_seed(0, ["a__x__1"]) != _batch_seed(1, ["a__x__1"])


def test_seed_within_valid_torch_manual_seed_range():
    # torch.manual_seed accepte un int64 ; on vérifie juste une plage raisonnable
    # positive (2**31 ici, largement suffisant pour la diversité recherchée).
    s = _batch_seed(0, ["x__y__1", "z__w__2", "q__r__3"])
    assert 0 <= s < 2**31
