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


def test_negative_uses_random_replay_stable_position_not_middle(tmp_path):
    """B.3 puis correctif profond (RESULTS_TESTS.md §117) : pour un document
    négatif réellement non-activant (le cas normal -- BatchTopK, la feature
    est hard-zero hors de son top-k), np.argmax sur un vecteur constant
    renvoie déterministiquement l'indice 0 (convention numpy sur les
    ex-aequo) -- un artefact mécanique de argmax, pas une position de bruit
    distribuée. Le code ne l'utilise donc plus dans ce cas : la position
    surlignée est tirée aléatoirement (graine déterministe f_idx/d_idx,
    replay-stable) parmi les débuts de mot du document. Vérifie l'invariant
    qui compte réellement (réplication à l'identique d'un rejugement à
    l'autre) plutôt qu'un indice figé, et confirme l'absence de régression
    vers l'ancien comportement (milieu du document, len(toks)//2)."""
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
    # ("MIDDLE", à l'indice 10) -- ni l'ancien argmax dégénéré (toujours
    # "word0") ni ce comportement ne doivent réapparaître.
    assert "<<MIDDLE>>" not in neg_example

    # Replay-stable : même f_idx/d_idx -> même négatif, condition nécessaire
    # pour comparer un score rejugé à un score en cache (commentaire B.28).
    _, neg_example_replay = build_feature_examples_with_control(
        f_idx=0, token_fragments_dir=frag_dir, acts=doc_level_acts, n_pos=5,
    )
    assert neg_example_replay == neg_example


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
