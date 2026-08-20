"""Teste la logique de reprise de `_eval_raw` dans la passe de ré-encodage de
saev5.py (R1, AUDIT_SAE_2026-08.md §2.3/§4.3) : p1_eval_raw_tokens.pt peut
déjà contenir une capture partielle d'un run précédent coupé pendant/après la
fenêtre d'évaluation -- il faut la reprendre comme base plutôt que repartir de
zéro (les raw_acts déjà réencodés sont irrécupérables, purgés de leur
fragment), sans jamais dépasser le plafond _EVAL_CAP."""
import torch


def _accumulate_capped(preloaded, new_chunks, cap):
    """Reproduit saev5.py : _eval_raw = [preloaded] (si présent) puis
    .append() sous condition sum(...) < cap à chaque nouveau chunk, plafonné
    à la fin par [:cap]."""
    eval_raw = [preloaded] if preloaded is not None else []
    for chunk in new_chunks:
        if sum(t.shape[0] for t in eval_raw) < cap:
            eval_raw.append(chunk)
    return torch.cat(eval_raw)[:cap] if eval_raw else torch.empty(0)


def test_resume_from_preloaded_partial_file_continues_accumulating():
    cap = 100
    preloaded = torch.arange(40).unsqueeze(1).float()  # 40 déjà capturés avant coupure
    new_chunks = [torch.full((30, 1), 99.0), torch.full((30, 1), 98.0), torch.full((30, 1), 97.0)]

    result = _accumulate_capped(preloaded, new_chunks, cap)

    assert result.shape[0] == cap  # 40 + 30 + 30 = 100, plafonné pile à cap
    assert torch.equal(result[:40], preloaded)  # la base préchargée est préservée telle quelle


def test_resume_never_exceeds_cap_even_with_excess_new_data():
    cap = 50
    preloaded = torch.zeros(45, 1)
    new_chunks = [torch.ones(30, 1)] * 5  # largement plus que nécessaire pour dépasser cap

    result = _accumulate_capped(preloaded, new_chunks, cap)

    assert result.shape[0] == cap
    assert torch.equal(result[:45], preloaded)


def test_no_preloaded_file_behaves_like_fresh_start():
    cap = 20
    new_chunks = [torch.full((15, 1), 1.0), torch.full((15, 1), 2.0)]

    result = _accumulate_capped(None, new_chunks, cap)

    # 15 (chunk1, sous le plafond -> ajouté) + 15 (chunk2, encore sous le
    # plafond au moment du test -> ajouté en entier) = 30, tronqué à 20 :
    # les 15 premières lignes viennent du chunk1, les 5 suivantes du chunk2
    # (même comportement que le code réel : troncature finale, pas par chunk).
    assert result.shape[0] == cap
    assert torch.equal(result[:15], torch.full((15, 1), 1.0))
    assert torch.equal(result[15:], torch.full((5, 1), 2.0))
