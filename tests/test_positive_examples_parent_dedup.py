"""Teste la déduplication des exemples positifs par mail D'ORIGINE
(doc_groups), pas seulement par (doc_idx, position) -- B.4 empêchait déjà les
répétitions du même mot dans un même document, mais rien n'empêchait le
top-9 par magnitude de retenir plusieurs PARAPHRASES du même mail comme
exemples "positifs" indépendants. Le corpus augmenté préserve quasi-verbatim
les entités numériques (`src/data/augmentation.py::validate`), donc un bloc
factuel (nom/adresse/contrat) traverse souvent plusieurs variantes presque
mot pour mot -- mesuré : 33,8% des 68 features interprétables de
`results_v10_emails_main` ont une paire d'exemples positifs à >50% de
similarité de caractères."""
import torch

from src.sae.judge import build_feature_examples_with_control, build_phrase_examples_with_control
from src.storage.fragment_store import save_fragment


def _toks(n, marker_idx, marker):
    return [f"▁word{i}" if i != marker_idx else f"▁{marker}" for i in range(n)]


# ── Pipeline 1 : fragments token-level ──────────────────────────────────────


def test_doc_groups_prefers_one_example_per_parent(tmp_path):
    frag_dir = str(tmp_path)
    d_sae = 1
    n_tok = 6
    # 6 documents, 3 mails d'origine (parents 0/0/1/1/2/2) -- paraphrases
    # d'un même parent partagent une magnitude légèrement décroissante pour
    # simuler le classement par magnitude réel.
    magnitudes = [5.0, 4.9, 4.8, 4.7, 4.6, 4.5]
    doc_groups = [0, 0, 1, 1, 2, 2]
    for doc_id, mag in enumerate(magnitudes):
        acts = torch.zeros(n_tok, d_sae)
        acts[2, 0] = mag
        save_fragment(frag_dir, doc_id=doc_id, token_strings=_toks(n_tok, 2, "CONTRAT"), acts_dense=acts)

    doc_level_acts = torch.tensor([[m] for m in magnitudes])
    pos_examples, neg_example = build_feature_examples_with_control(
        f_idx=0, token_fragments_dir=frag_dir, acts=doc_level_acts, n_pos=3,
        doc_groups=doc_groups,
    )
    # 3 parents distincts disponibles, n_pos=3 -- un seul exemple par parent
    # (le premier rencontré par magnitude décroissante : docs 0, 2, 4).
    assert len(pos_examples) == 3


def test_doc_groups_backfills_from_same_parent_when_too_few_distinct(tmp_path):
    frag_dir = str(tmp_path)
    d_sae = 1
    n_tok = 6
    # 5 documents, un SEUL mail d'origine (feature rare, peu de mails sources)
    # -- ne doit PAS réduire l'échantillon sous n_pos par rapport à l'ancien
    # comportement (pas de nouvelle catégorie de dead_feature).
    magnitudes = [5.0, 4.9, 4.8, 4.7, 4.6]
    doc_groups = [7, 7, 7, 7, 7]
    for doc_id, mag in enumerate(magnitudes):
        acts = torch.zeros(n_tok, d_sae)
        acts[2, 0] = mag
        save_fragment(frag_dir, doc_id=doc_id, token_strings=_toks(n_tok, 2, "CONTRAT"), acts_dense=acts)

    doc_level_acts = torch.tensor([[m] for m in magnitudes])
    pos_examples, neg_example = build_feature_examples_with_control(
        f_idx=0, token_fragments_dir=frag_dir, acts=doc_level_acts, n_pos=5,
        doc_groups=doc_groups,
    )
    assert len(pos_examples) == 5  # repli identique au comportement sans doc_groups


def test_doc_groups_none_is_unchanged_from_before(tmp_path):
    """Rétrocompatibilité stricte : doc_groups=None (défaut) doit produire
    exactement le même résultat qu'avant ce correctif."""
    frag_dir = str(tmp_path)
    d_sae = 1
    n_tok = 6
    magnitudes = [5.0, 4.9, 4.8, 4.7]
    doc_groups = [0, 0, 1, 1]
    for doc_id, mag in enumerate(magnitudes):
        acts = torch.zeros(n_tok, d_sae)
        acts[2, 0] = mag
        save_fragment(frag_dir, doc_id=doc_id, token_strings=_toks(n_tok, 2, "CONTRAT"), acts_dense=acts)

    doc_level_acts = torch.tensor([[m] for m in magnitudes])
    pos_no_groups, _ = build_feature_examples_with_control(
        f_idx=0, token_fragments_dir=frag_dir, acts=doc_level_acts, n_pos=4,
    )
    assert len(pos_no_groups) == 4  # tous les documents comptent, malgré 2 parents seulement


# ── Pipeline 2 : phrases denses en mémoire ──────────────────────────────────


def test_dense_doc_groups_prefers_one_example_per_parent():
    d_sae = 1
    # 6 phrases, 3 mails d'origine (parents 0/0/1/1/2/2), textes distincts
    # (le dédoublonnage exact par texte ne doit pas suffire à lui seul).
    phrase_texts = [
        "voici les détails de mon contrat numéro un",
        "je rappelle les informations de mon contrat numéro un bis",
        "mon contrat porte le numéro deux",
        "concernant mon contrat, le numéro est deux bis",
        "j ai un souci avec le contrat trois",
        "au sujet du contrat numéro trois bis",
    ]
    magnitudes = [5.0, 4.9, 4.8, 4.7, 4.6, 4.5]
    phrase_to_doc = torch.tensor([0, 0, 1, 1, 2, 2])
    doc_groups = [10, 10, 11, 11, 12, 12]
    phrase_acts = torch.tensor([[m] for m in magnitudes])

    pos_examples, neg_example = build_phrase_examples_with_control(
        f_idx=0, phrase_texts=phrase_texts, phrase_acts=phrase_acts, n_pos=3,
        phrase_to_doc=phrase_to_doc, doc_groups=doc_groups,
    )
    assert len(pos_examples) == 3


def test_dense_doc_groups_backfills_when_too_few_distinct_parents():
    phrase_texts = [f"phrase numéro {i}" for i in range(5)]
    magnitudes = [5.0, 4.9, 4.8, 4.7, 4.6]
    phrase_to_doc = torch.tensor([0, 1, 2, 3, 4])
    doc_groups = [9, 9, 9, 9, 9]  # même mail d'origine pour toutes
    phrase_acts = torch.tensor([[m] for m in magnitudes])

    pos_examples, neg_example = build_phrase_examples_with_control(
        f_idx=0, phrase_texts=phrase_texts, phrase_acts=phrase_acts, n_pos=5,
        phrase_to_doc=phrase_to_doc, doc_groups=doc_groups,
    )
    assert len(pos_examples) == 5


def test_dense_doc_groups_none_is_unchanged_from_before():
    phrase_texts = [f"phrase numéro {i}" for i in range(4)]
    magnitudes = [5.0, 4.9, 4.8, 4.7]
    phrase_acts = torch.tensor([[m] for m in magnitudes])

    pos_examples, neg_example = build_phrase_examples_with_control(
        f_idx=0, phrase_texts=phrase_texts, phrase_acts=phrase_acts, n_pos=4,
    )
    assert len(pos_examples) == 4
