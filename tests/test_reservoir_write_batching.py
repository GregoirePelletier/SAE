"""Vérifie l'amortissement des écritures aléatoires du réservoir
(saev5.py::_flush_pending_reservoir_writes, AUDIT_SAE_2026-08.md §2.2, G5) :
accumuler les remplacements dans un buffer puis les appliquer triés par
indice (torch.argsort(..., stable=True)) doit produire EXACTEMENT le même
état final de réservoir que des écritures immédiates non triées -- y
compris en cas de collision (deux remplacements visant le même indice), où
la dernière entrée temporelle doit l'emporter dans les deux cas.

Reproduit la logique de saev5.py (fonction non exportable, définie en
closure) plutôt que de l'importer -- même limitation et même choix que
test_reservoir_resume_invariant.py : isoler l'algorithme sans les lourdes
dépendances d'import de saev5.py (transformers/sae_lens)."""
import torch


def _apply_immediate(reservoir, batches):
    """Chemin de référence : écriture immédiate, un batch à la fois, dans
    l'ordre temporel -- comportement d'avant ce correctif."""
    for indices, values in batches:
        if indices.shape[0] > 0:
            reservoir[indices] = values
    return reservoir


def _apply_buffered_sorted(reservoir, batches, flush_size):
    """Reproduit _flush_pending_reservoir_writes : accumule (indices, values)
    dans un buffer, flush trié (stable) dès que le buffer atteint flush_size,
    flush final pour le reliquat."""
    pending_j, pending_x, count = [], [], 0

    def flush():
        nonlocal pending_j, pending_x, count
        if not pending_j:
            return
        j_cat = torch.cat(pending_j)
        x_cat = torch.cat(pending_x)
        order = torch.argsort(j_cat, stable=True)
        reservoir[j_cat[order]] = x_cat[order]
        pending_j, pending_x, count = [], [], 0

    for indices, values in batches:
        if indices.shape[0] == 0:
            continue
        pending_j.append(indices)
        pending_x.append(values)
        count += indices.shape[0]
        if count >= flush_size:
            flush()
    flush()
    return reservoir


def test_buffered_sorted_matches_immediate_no_collisions():
    torch.manual_seed(0)
    capacity, d = 200, 4
    batches = []
    for _ in range(50):
        n = 5
        indices = torch.randint(0, capacity, (n,))
        values = torch.randn(n, d)
        batches.append((indices, values))

    ref = _apply_immediate(torch.zeros(capacity, d), batches)
    got = _apply_buffered_sorted(torch.zeros(capacity, d), batches, flush_size=17)
    assert torch.equal(ref, got)


def test_collision_resolution_matches_last_temporal_write():
    """Cas construit à la main : deux batches ciblent délibérément le même
    indice avec des valeurs différentes -- la version bufferisée doit garder
    la valeur du batch le PLUS RÉCENT, comme le ferait une écriture immédiate."""
    capacity, d = 10, 2
    target = 3
    batches = [
        (torch.tensor([target]), torch.tensor([[1.0, 1.0]])),   # ancien
        (torch.tensor([7]), torch.tensor([[9.0, 9.0]])),         # sans rapport
        (torch.tensor([target]), torch.tensor([[2.0, 2.0]])),   # récent -- doit gagner
    ]

    ref = _apply_immediate(torch.zeros(capacity, d), batches)
    assert torch.equal(ref[target], torch.tensor([2.0, 2.0]))

    # flush_size=1 force un flush après CHAQUE batch pris isolément (aucune
    # collision DANS le buffer) -- cas trivial, doit déjà correspondre.
    got_flush_per_batch = _apply_buffered_sorted(torch.zeros(capacity, d), batches, flush_size=1)
    assert torch.equal(ref, got_flush_per_batch)

    # flush_size assez grand pour que les TROIS batches soient accumulés dans
    # LE MÊME buffer avant flush -- c'est le cas qui exercerait réellement un
    # bug de tri non stable (les deux entrées pour `target` se retrouvent
    # adjacentes après tri, l'ordre relatif doit rester le même qu'à l'origine).
    got_single_flush = _apply_buffered_sorted(torch.zeros(capacity, d), batches, flush_size=100)
    assert torch.equal(ref, got_single_flush)
    assert torch.equal(got_single_flush[target], torch.tensor([2.0, 2.0]))


def test_empty_batches_are_safe_noop():
    capacity, d = 5, 2
    batches = [(torch.empty(0, dtype=torch.long), torch.empty(0, d))]
    ref = _apply_immediate(torch.zeros(capacity, d), batches)
    got = _apply_buffered_sorted(torch.zeros(capacity, d), batches, flush_size=10)
    assert torch.equal(ref, got)
