"""Vérifie l'équivalence du ré-encodage batché (saev5.py, audit perf §2.9
item 8) : concaténer les raw_acts de plusieurs documents pour UN SEUL appel
à _encode_extra_acts (au lieu d'un appel par document) doit produire un
résultat IDENTIQUE, document par document, après torch.split. La garantie
repose sur BatchTopKEncoder.eval() (src/sae/batch.py) qui applique un seuil
global élément-par-élément (JumpReLU), sans aucune interaction entre lignes
du batch -- contrairement au mode entraînement (_batch_topk, budget partagé
sur le batch), jamais utilisé au ré-encodage (ext_sae.eval() appelé avant)."""
import torch

from src.sae.batch import BatchTopKEncoder


def _pre_extra(x, W_enc, b_enc, scale):
    """Reproduit frozen_core.py::_pre_extra : opération purement par ligne."""
    return (x.float() / scale) @ W_enc.float() + b_enc.float()


def test_batched_encode_matches_per_document_encode():
    torch.manual_seed(0)
    d_in, d_extra, k = 16, 12, 3
    W_enc = torch.randn(d_in, d_extra)
    b_enc = torch.randn(d_extra)
    scale = 2.3

    encoder = BatchTopKEncoder(k=k)
    encoder.eval()
    # Calibre le seuil comme le ferait un entraînement réel (sinon `calibrated`
    # reste False et forward() retombe sur un TopK per-sample, cf. batch.py) --
    # calibration arbitraire, seule l'ÉQUIVALENCE batché vs séquentiel importe ici.
    encoder.threshold.fill_(0.1)
    encoder.calibrated.fill_(True)

    docs = [torch.randn(n, d_in) for n in (5, 1, 8, 3)]

    # Chemin séquentiel (ancien code : un appel par document).
    sequential = [encoder(_pre_extra(d, W_enc, b_enc, scale)) for d in docs]

    # Chemin batché (nouveau code : concaténation puis un seul appel).
    lengths = [d.shape[0] for d in docs]
    cat = torch.cat(docs, dim=0)
    batched_out = encoder(_pre_extra(cat, W_enc, b_enc, scale))
    batched_split = list(torch.split(batched_out, lengths))

    for seq, batch in zip(sequential, batched_split):
        assert torch.equal(seq, batch)


def test_batched_encode_equivalence_with_various_chunk_sizes():
    """La position d'un document DANS le lot (début/milieu/fin) ne doit rien
    changer -- reformule la même propriété avec un découpage en plusieurs
    lots de tailles différentes, comme le ferait REENCODE_BATCH_SIZE variable."""
    torch.manual_seed(1)
    d_in, d_extra, k = 8, 10, 2
    W_enc = torch.randn(d_in, d_extra)
    b_enc = torch.randn(d_extra)
    scale = 1.0

    encoder = BatchTopKEncoder(k=k)
    encoder.eval()
    encoder.threshold.fill_(0.05)
    encoder.calibrated.fill_(True)

    docs = [torch.randn(n, d_in) for n in (2, 4, 1, 6, 2, 3)]
    reference = [encoder(_pre_extra(d, W_enc, b_enc, scale)) for d in docs]

    for chunk_size in (1, 2, 3, 100):
        results = []
        for start in range(0, len(docs), chunk_size):
            chunk = docs[start:start + chunk_size]
            lengths = [d.shape[0] for d in chunk]
            cat = torch.cat(chunk, dim=0)
            out = encoder(_pre_extra(cat, W_enc, b_enc, scale))
            results.extend(torch.split(out, lengths))
        for ref, got in zip(reference, results):
            assert torch.equal(ref, got)
