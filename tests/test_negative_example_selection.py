"""Teste src/sae/judge.py::build_feature_examples_with_control et
build_phrase_examples_with_control -- correctifs B.3 (négatif construit à
l'argmax de la même feature, pas au milieu du document) et B.5 (un candidat
négatif dont l'activation réelle dépasse le seuil positif est écarté, pas
accepté silencieusement). Aucun test existant n'exerçait l'implémentation
réelle de ces fonctions (le seul voisin, test_judge_batching_orchestration.py,
les mocke entièrement)."""
import numpy as np
import torch

from src.sae.judge import build_feature_examples_with_control, build_phrase_examples_with_control
from src.storage.fragment_store import save_fragment


def _toks(n, marker_idx, marker="MARKER"):
    """n tokens ▁word_i, sauf à marker_idx qui porte `marker` -- permet de
    vérifier QUEL token a été choisi comme centre du contexte extrait."""
    return [f"▁word{i}" if i != marker_idx else f"▁{marker}" for i in range(n)]


def test_negative_uses_argmax_position_not_middle(tmp_path):
    """B.3 : le négatif doit être construit via l'argmax de CETTE feature
    (comme les positifs), pas via len(toks)//2. Un document réellement non-
    activant (le cas normal une fois B.5 appliqué -- toute activation
    résiduelle au-dessus de threshold_pos=1e-6 est de toute façon rejetée, et
    1e-6 est aussi le seuil de sparsification du stockage CSR des fragments,
    donc un "négatif" accepté a systématiquement une colonne de feature
    entièrement nulle une fois rechargée) a un token_acts tout à zéro :
    argmax() y résout à l'indice 0 par convention numpy/torch (premier
    maximum ex-aequo) -- DIFFÉRENT de l'ancien comportement qui prenait
    systématiquement le milieu du document. Vérifie ce changement de
    comportement precisement (indice 0, pas len(toks)//2 = 10), pas une
    hypothétique "vraie" position de pic qui n'existe pas dans ce cas."""
    frag_dir = str(tmp_path)
    d_sae = 2
    n_tok = 21
    assert n_tok // 2 == 10

    # Document positif (fort et net) pour peupler pos_examples.
    pos_acts = torch.zeros(n_tok, d_sae)
    pos_acts[5, 0] = 5.0
    save_fragment(frag_dir, doc_id=0, token_strings=_toks(n_tok, 5, "POS"), acts_dense=pos_acts)

    # Document négatif réel : aucune activation de la feature 0 nulle part.
    neg_acts = torch.zeros(n_tok, d_sae)
    save_fragment(frag_dir, doc_id=1, token_strings=_toks(n_tok, 10, "MIDDLE"), acts_dense=neg_acts)

    doc_level_acts = torch.tensor([[5.0, 0.0], [0.0, 0.0]])
    pos_examples, neg_example = build_feature_examples_with_control(
        f_idx=0, token_fragments_dir=frag_dir, acts=doc_level_acts, n_pos=5,
    )

    assert neg_example is not None
    assert "MARKER" not in neg_example  # pas le document positif
    # Ancien comportement (len(toks)//2) aurait marqué le token du milieu
    # ("MIDDLE", à l'indice 10) -- le nouveau marque l'indice 0 ("word0").
    assert "<<MIDDLE>>" not in neg_example
    assert "<<word0>>" in neg_example


def test_negative_rejects_candidate_above_threshold_pos(tmp_path):
    """B.5 : neg_quantile=0.05 peut désigner un candidat dont l'activation
    réelle dépasse threshold_pos (feature dense) -- doit être écarté, pas
    accepté comme "négatif" alors qu'il active la feature."""
    frag_dir = str(tmp_path)
    d_sae = 1
    n_tok = 10

    # doc 0 : candidat "négatif" au sens du quantile, mais qui active
    # réellement la feature (0.5 >> threshold_pos=1e-6) -- doit être rejeté.
    bad_acts = torch.zeros(n_tok, d_sae)
    bad_acts[4, 0] = 0.5
    save_fragment(frag_dir, doc_id=0, token_strings=_toks(n_tok, 4, "BAD"), acts_dense=bad_acts)

    # doc 1 : vrai négatif (activation nulle).
    save_fragment(frag_dir, doc_id=1, token_strings=_toks(n_tok, 0), acts_dense=torch.zeros(n_tok, d_sae))

    # doc 2 : positif net, pour peupler pos_examples.
    pos_acts = torch.zeros(n_tok, d_sae)
    pos_acts[2, 0] = 5.0
    save_fragment(frag_dir, doc_id=2, token_strings=_toks(n_tok, 2, "POS"), acts_dense=pos_acts)

    doc_level_acts = torch.tensor([[0.5], [0.0], [5.0]])
    # Les deux candidats "négatifs" potentiels (doc 0 et 1) tombent sous le
    # même quantile ici (neg_quantile=0.4 -> seuil entre les deux premières
    # valeurs) ; seul doc 1 doit être retenu.
    pos_examples, neg_example = build_feature_examples_with_control(
        f_idx=0, token_fragments_dir=frag_dir, acts=doc_level_acts, n_pos=5,
        neg_quantile=0.4,
    )

    assert neg_example is not None
    assert "BAD" not in neg_example


def test_negative_falls_back_to_least_active_candidate_when_none_is_truly_zero(tmp_path):
    """Si aucun candidat du pool n'est réellement inactif (feature dense --
    B.2, le cas courant pour des features sélectionnées par magnitude, cf.
    job 44831), neg_example reste le candidat de plus faible activation
    réelle plutôt que None -- un seuil dur ferait disparaître neg_example
    (et donc interp_score) pour la quasi-totalité des features, vérifié sur
    GPU avant ce correctif. neg_magnitude reflète l'activation réelle non
    nulle, exploitable en aval pour filtrer si besoin."""
    frag_dir = str(tmp_path)
    d_sae = 1
    n_tok = 5
    bad_acts = torch.zeros(n_tok, d_sae)
    bad_acts[0, 0] = 0.5
    save_fragment(frag_dir, doc_id=0, token_strings=_toks(n_tok, 0, "BAD"), acts_dense=bad_acts)
    pos_acts = torch.zeros(n_tok, d_sae)
    pos_acts[0, 0] = 5.0
    save_fragment(frag_dir, doc_id=1, token_strings=_toks(n_tok, 0, "POS"), acts_dense=pos_acts)

    doc_level_acts = torch.tensor([[0.5], [5.0]])
    pos_examples, neg_example, pos_mags, neg_magnitude = build_feature_examples_with_control(
        f_idx=0, token_fragments_dir=frag_dir, acts=doc_level_acts, n_pos=5,
        neg_quantile=0.6,  # seul doc 0 (BAD, réellement actif) tombe dans le pool négatif
        return_magnitudes=True,
    )
    assert neg_example is not None
    assert "BAD" in neg_example
    assert neg_magnitude == 0.5  # pas un vrai négatif -- traçable via neg_magnitude, pas caché


def test_phrase_level_negative_rejects_candidate_above_threshold_pos():
    """Équivalent phrase-level (build_phrase_examples_with_control) du garde-fou B.5."""
    phrase_texts = ["phrase positive forte", "phrase négative réelle", "phrase limite dense"]
    # feature 0 : positive forte, négative nulle, "limite" au-dessus de threshold_pos.
    phrase_acts = torch.tensor([[5.0], [0.0], [0.3]])

    pos_examples, neg_example = build_phrase_examples_with_control(
        f_idx=0, phrase_texts=phrase_texts, phrase_acts=phrase_acts, n_pos=5,
        neg_quantile=0.7,  # les deux dernières phrases tombent dans le pool négatif
    )
    assert neg_example is not None
    assert "limite" not in neg_example
    assert "négative réelle" in neg_example
