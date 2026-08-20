"""Vérifie l'invariant statistique dont dépend la reprise du réservoir de
Vitter dans saev5.py (Extraction P1, AUDIT_SAE_2026-08.md §2.3/§4.3, R1) :
traiter les tokens en DEUX passes (avant coupure, puis reprise avec les
compteurs persistés n_residuals_seen/n_residuals_collected) doit donner la
MÊME probabilité marginale d'inclusion dans le réservoir que de les traiter
en UNE SEULE passe continue -- sinon la reprise biaiserait l'échantillon
d'entraînement du SAEBoostResidualSAE vers les documents post-reprise (bug identifié
dans l'audit, jamais corrigé avant ce correctif).

Reproduit exactement la mise à jour de saev5.py (Algorithm R de Vitter,
phase 1 = remplissage séquentiel, phase 2 = remplacement j ~ U[0, m)) sur des
scalaires (1 "token" = 1 entier, pas des tenseurs [T, d_in]) pour isoler
l'algorithme du reste du pipeline -- même logique, juste sans le bruit
d'implémentation GPU/tenseurs autour."""
import torch


def _reservoir_update(reservoir, n_collected, n_seen, new_items, capacity):
    """Même équations que saev5.py:1063-1075 (x_new -> reservoir), sur une
    liste Python plutôt que des tenseurs GPU -- portage 1:1 pour le test."""
    new_items = list(new_items)
    if n_collected < capacity:
        take = min(capacity - n_collected, len(new_items))
        for k in range(take):
            reservoir[n_collected + k] = new_items[k]
        n_collected += take
        n_seen += take
        new_items = new_items[take:]
    if new_items:
        for idx, item in enumerate(new_items):
            m = n_seen + idx + 1
            j = int(torch.rand(()).item() * m)
            if j < capacity:
                reservoir[j] = item
        n_seen += len(new_items)
    return n_collected, n_seen


def _run_continuous(all_items, capacity):
    reservoir = [None] * capacity
    n_collected, n_seen = _reservoir_update(reservoir, 0, 0, all_items, capacity)
    return reservoir, n_collected, n_seen


def _run_with_checkpoint_resume(all_items, capacity, split_at):
    """Simule une coupure après `split_at` items, persistance des compteurs
    (comme _write_extraction_progress), puis reprise -- mêmes compteurs
    restaurés, poursuite sur le même buffer reservoir (comme la réouverture du
    memmap existant)."""
    reservoir = [None] * capacity
    n_collected, n_seen = _reservoir_update(reservoir, 0, 0, all_items[:split_at], capacity)
    # "Coupure" : n_collected/n_seen persistés puis restaurés tels quels (pas
    # remis à 0 -- c'est précisément ce que corrige ce patch).
    n_collected, n_seen = _reservoir_update(reservoir, n_collected, n_seen, all_items[split_at:], capacity)
    return reservoir, n_collected, n_seen


def test_final_counters_identical_regardless_of_split_point():
    """n_seen/n_collected finaux ne doivent dépendre que du nombre total
    d'items traités, jamais du découpage en une ou plusieurs passes."""
    items = list(range(500))
    capacity = 50
    _, n_collected_full, n_seen_full = _run_continuous(items, capacity)
    for split_at in (1, 50, 200, 499):
        _, n_collected_split, n_seen_split = _run_with_checkpoint_resume(items, capacity, split_at)
        assert n_collected_split == n_collected_full == capacity
        assert n_seen_split == n_seen_full == len(items)


def test_marginal_inclusion_probability_unaffected_by_checkpoint_split():
    """Propriété statistique de Vitter (Algorithm R) : chaque item a probabilité
    capacity/N d'être dans le réservoir final, QUELLE QUE SOIT la façon dont le
    flux a été découpé en passes. On compare, sur de nombreux essais, le taux
    d'inclusion d'un item tardif (proche de la coupure -- le plus à risque
    d'un biais si la reprise repartait de compteurs à 0) entre une exécution
    continue et une exécution coupée+reprise."""
    torch.manual_seed(0)
    n_items, capacity, split_at, n_trials = 200, 20, 150, 4000
    target_item = 160  # juste après la coupure -- le cas que le bug viserait

    hits_continuous, hits_split = 0, 0
    for _ in range(n_trials):
        items = list(range(n_items))
        reservoir_c, _, _ = _run_continuous(items, capacity)
        hits_continuous += int(target_item in reservoir_c)

        reservoir_s, _, _ = _run_with_checkpoint_resume(items, capacity, split_at)
        hits_split += int(target_item in reservoir_s)

    p_continuous = hits_continuous / n_trials
    p_split = hits_split / n_trials
    expected = capacity / n_items  # 20/200 = 0.10

    # Marge large (test statistique, pas un calcul exact) : les deux doivent
    # rester proches de la probabilité théorique ET proches l'une de l'autre.
    assert abs(p_continuous - expected) < 0.03
    assert abs(p_split - expected) < 0.03
    assert abs(p_continuous - p_split) < 0.04


def test_naive_resume_from_zero_would_be_biased_control_case():
    """Contre-épreuve : si la reprise repartait de n_seen=0 (le bug SANS ce
    correctif), la probabilité d'inclusion des items post-coupure serait
    nettement SURESTIMÉE par rapport à la théorie -- confirme que le test
    précédent détecterait bien une régression si elle réapparaissait."""
    torch.manual_seed(1)
    n_items, capacity, split_at, n_trials = 200, 20, 150, 3000
    target_item = 160
    expected = capacity / n_items

    hits_naive = 0
    for _ in range(n_trials):
        items = list(range(n_items))
        reservoir = [None] * capacity
        n_collected, n_seen = _reservoir_update(reservoir, 0, 0, items[:split_at], capacity)
        # Bug : compteurs PAS restaurés, repartent de 0 comme si c'était un
        # nouveau flux -- alors que le réservoir, lui, contient déjà des items.
        n_collected_bad, n_seen_bad = _reservoir_update(reservoir, 0, 0, items[split_at:], capacity)
        hits_naive += int(target_item in reservoir)

    p_naive = hits_naive / n_trials
    assert abs(p_naive - expected) > 0.05, (
        "le contrôle négatif ne montre plus de biais -- le scénario ne teste "
        "plus ce qu'il prétend tester"
    )
