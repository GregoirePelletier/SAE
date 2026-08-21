"""Teste src/sae/judge.py::build_feature_examples_with_control -- correctif
B.4 (AUDIT_SAE_2026-08.md) : la déduplication des positifs se faisait sur la
chaîne du mot-cible (seen_target_words), pas sur (doc_idx, position) --
empêchait une feature authentiquement lexicale de montrer le même mot dans
plusieurs documents/contextes différents, sa forme la plus convaincante."""
import torch

from src.sae.judge import build_feature_examples_with_control
from src.storage.fragment_store import save_fragment


def _toks(n, marker_idx, marker):
    return [f"▁word{i}" if i != marker_idx else f"▁{marker}" for i in range(n)]


def test_same_target_word_across_documents_not_deduplicated(tmp_path):
    frag_dir = str(tmp_path)
    d_sae = 1
    n_tok = 6

    # 3 documents distincts, tous avec "CHER" comme mot-cible argmax --
    # feature authentiquement lexicale.
    for doc_id in range(3):
        acts = torch.zeros(n_tok, d_sae)
        acts[2, 0] = 5.0
        save_fragment(frag_dir, doc_id=doc_id, token_strings=_toks(n_tok, 2, "CHER"), acts_dense=acts)

    doc_level_acts = torch.tensor([[5.0], [5.0], [5.0]])
    pos_examples, neg_example = build_feature_examples_with_control(
        f_idx=0, token_fragments_dir=frag_dir, acts=doc_level_acts, n_pos=9,
    )

    # Ancien comportement : seul 1 exemple "<<CHER>>" aurait survécu (dédup
    # par mot). Nouveau : les 3 documents contribuent chacun leur exemple.
    assert len(pos_examples) == 3
    assert all("<<CHER>>" in ex for ex in pos_examples)


def test_n_pos_cap_still_respected_with_repeated_words(tmp_path):
    frag_dir = str(tmp_path)
    d_sae = 1
    n_tok = 6
    n_docs = 5
    for doc_id in range(n_docs):
        acts = torch.zeros(n_tok, d_sae)
        acts[2, 0] = 5.0
        save_fragment(frag_dir, doc_id=doc_id, token_strings=_toks(n_tok, 2, "CHER"), acts_dense=acts)

    doc_level_acts = torch.tensor([[5.0]] * n_docs)
    pos_examples, neg_example = build_feature_examples_with_control(
        f_idx=0, token_fragments_dir=frag_dir, acts=doc_level_acts, n_pos=3,
    )
    assert len(pos_examples) == 3  # plafonné par n_pos, pas par la dédup
