"""Teste src/sae/sae_shared.py::load_or_train_extended_sae -- vérifie que
l'accumulation des métriques (loss/l0/dead_frac/aux_loss) en tenseurs GPU
pendant l'époque, converties en Python une seule fois à la fin (audit perf
§2.4 : 4x .item()/float() par step -> 4x cudaStreamSynchronize par step),
produit un historique STRICTEMENT identique à ce que chaque forward a
réellement retourné, pas seulement "à peu près" -- le changement ne doit
avoir aucun effet numérique, seulement déplacer le moment du sync CPU<->GPU."""
import torch
import torch.nn as nn

from src.sae.sae_shared import load_or_train_extended_sae


class _LoggingStubModel(nn.Module):
    """Modèle jouet exposant la même interface qu'ExtendedSAE (forward ->
    dict avec loss/l0/dead_frac/aux_loss) et journalisant lui-même, à chaque
    appel de forward(), les valeurs qu'il retourne -- source de vérité
    indépendante de l'agrégation testée."""

    def __init__(self, d: int):
        super().__init__()
        self.w = nn.Parameter(torch.randn(d))
        self.calls = []

    def forward(self, x):
        loss = (x * self.w).pow(2).mean()
        l0 = torch.tensor(float(x.shape[1]))
        dead = torch.tensor(0.02)
        aux = torch.tensor(0.5)
        self.calls.append((loss.item(), l0.item(), dead.item(), aux.item()))
        return {"loss": loss, "l0": l0, "dead_frac": dead, "aux_loss": aux}


def test_batched_history_matches_live_forward_values_exactly(tmp_path):
    torch.manual_seed(0)
    d = 8
    acts_train = torch.randn(64, d)
    model = _LoggingStubModel(d)

    _, history = load_or_train_extended_sae(
        model=model, model_name="stub", acts_train=acts_train,
        epochs=2, lr=1e-3, save_dir=str(tmp_path), device="cpu",
    )

    expected_loss = [c[0] for c in model.calls]
    expected_l0 = [c[1] for c in model.calls]
    expected_dead = [c[2] for c in model.calls]
    expected_aux = [c[3] for c in model.calls]

    assert history["loss"] == expected_loss
    assert history["l0"] == expected_l0
    assert history["dead_frac"] == expected_dead
    assert history["aux_loss"] == expected_aux
    assert len(history["step"]) == len(expected_loss)
