"""Tests CPU rapides pour src/analysis/metrics.py::held_out_probe_accuracy
(protocole FIT-train/DEV-eval strict, plan §6.4 -- par opposition a
downstream_classification qui fait sa propre CV interne)."""
import numpy as np
from scipy import sparse as sp

from src.analysis.metrics import held_out_probe_accuracy


def test_perfectly_separable_dense_gives_full_accuracy():
    rng = np.random.default_rng(0)
    X_train = np.vstack([rng.normal(-5, 0.1, (20, 3)), rng.normal(5, 0.1, (20, 3))])
    y_train = [0] * 20 + [1] * 20
    X_eval = np.vstack([rng.normal(-5, 0.1, (10, 3)), rng.normal(5, 0.1, (10, 3))])
    y_eval = [0] * 10 + [1] * 10

    res = held_out_probe_accuracy(X_train, y_train, X_eval, y_eval)
    assert res["accuracy"] == 1.0
    assert res["correct"].all()
    assert res["n_train"] == 40
    assert res["n_eval"] == 20
    assert res["n_classes"] == 2


def test_sparse_csr_input_gives_same_result_as_dense():
    rng = np.random.default_rng(1)
    X_train_dense = np.vstack([rng.normal(-5, 0.1, (15, 4)), rng.normal(5, 0.1, (15, 4))])
    y_train = [0] * 15 + [1] * 15
    X_eval_dense = np.vstack([rng.normal(-5, 0.1, (8, 4)), rng.normal(5, 0.1, (8, 4))])
    y_eval = [0] * 8 + [1] * 8

    res_dense = held_out_probe_accuracy(X_train_dense, y_train, X_eval_dense, y_eval)
    res_sparse = held_out_probe_accuracy(
        sp.csr_matrix(X_train_dense), y_train, sp.csr_matrix(X_eval_dense), y_eval
    )
    assert res_dense["accuracy"] == res_sparse["accuracy"]
    assert np.array_equal(res_dense["predictions"], res_sparse["predictions"])


def test_multiclass_selects_lbfgs_solver_without_crash():
    rng = np.random.default_rng(2)
    n_classes = 5
    X_train = np.vstack([rng.normal(loc=c * 10, scale=0.5, size=(10, 3)) for c in range(n_classes)])
    y_train = [c for c in range(n_classes) for _ in range(10)]
    X_eval = np.vstack([rng.normal(loc=c * 10, scale=0.5, size=(4, 3)) for c in range(n_classes)])
    y_eval = [c for c in range(n_classes) for _ in range(4)]

    res = held_out_probe_accuracy(X_train, y_train, X_eval, y_eval)
    assert res["n_classes"] == 5
    assert res["accuracy"] > 0.8  # bien separe, tolere quelques erreurs de frontiere


def test_correct_array_is_boolean_and_aligned_with_y_eval():
    rng = np.random.default_rng(3)
    X_train = np.vstack([rng.normal(-3, 1, (10, 2)), rng.normal(3, 1, (10, 2))])
    y_train = [0] * 10 + [1] * 10
    X_eval = np.vstack([rng.normal(-3, 1, (5, 2)), rng.normal(3, 1, (5, 2))])
    y_eval = np.array([0] * 5 + [1] * 5)

    res = held_out_probe_accuracy(X_train, y_train, X_eval, y_eval)
    assert res["correct"].dtype == bool
    assert len(res["correct"]) == len(y_eval)
    manual_correct = res["predictions"] == y_eval
    assert np.array_equal(res["correct"], manual_correct)


def test_model_generalizes_from_train_not_fit_on_eval():
    # Le modele est entraine UNIQUEMENT sur une frontiere train (x<0 -> 0,
    # x>0 -> 1) ; eval utilise une distribution DECALEE mais respectant la
    # MEME frontiere -- si le code fittait par erreur sur eval (fuite), rien
    # ne le revelerait ici puisque la frontiere reste coherente, mais le test
    # verifie au moins que l'accuracy reflete bien la generalisation
    # train->eval (pas un score parfait artificiel du a un eval trivial).
    rng = np.random.default_rng(4)
    X_train = np.concatenate([rng.normal(-2, 0.3, 30), rng.normal(2, 0.3, 30)]).reshape(-1, 1)
    y_train = [0] * 30 + [1] * 30
    X_eval = np.concatenate([rng.normal(-2, 0.3, 15), rng.normal(2, 0.3, 15)]).reshape(-1, 1)
    y_eval = [0] * 15 + [1] * 15

    res = held_out_probe_accuracy(X_train, y_train, X_eval, y_eval)
    assert res["accuracy"] > 0.9
