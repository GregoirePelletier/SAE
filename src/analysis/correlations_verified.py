"""
correlations_verified.py — NPMI_verified, filtre des labels syntaxiques, filtre
des paires triviales (co-activation sur le même token) et métrique CO (App.
E.1/E.3, arXiv:2512.10092v2, Figure 4). Complète `src/analysis/cooccurrence.py`
(NPMI brut sur activations SAE, `find_interesting_pairs`) avec la partie
jamais implémentée : sans NPMI_verified, `p1_interesting_correlations.json`
est une liste de CANDIDATS, pas un résultat comparable au papier
(AUDIT_SAE_2026-08.md §1/§7).

Protocole (papier, §4.2/E.1/E.3) : pour une paire de latents (i, j) découverte
par NPMI élevé + labels sémantiquement dissimilaires (`find_interesting_pairs`),
un juge LLM relabellise i et j INDÉPENDAMMENT sur un échantillon de documents
(présence/absence du concept, pas l'activation SAE brute), puis le NPMI est
recalculé à partir de ces labels juge -- `NPMI_verified`. Filtre les labels
"syntaxiques" (grammaire/formatage générique) en amont : moins intéressants,
gonflent artificiellement le nombre de paires à vérifier.

Adaptateur du juge local du projet (`src.sae.judge._batched_generate`), même
raison que `src/analysis/hypothesis_verifier.py` : le code des auteurs suppose
une API OpenAI/OpenRouter absente de ce dépôt. Prompts repris VERBATIM du PDF
(docs/PDF_APPENDICES_EXTRACT.md lignes 777-820, Appendix K.2) et du corps
principal (§4.2/E.1/E.3, extraits directement du PDF -- pas dans
PDF_APPENDICES_EXTRACT.md qui ne couvre que les annexes).
"""

import json
import re
from typing import Optional

import numpy as np
import torch

from src.analysis.cooccurrence import compute_npmi
from src.sae.judge import _batched_generate

SYNTACTIC_LABEL_FILTER_PROMPT = """You are evaluating feature labels from a sparse autoencoder. Each label describes the concept a feature tends to activate on.
Classify each label as:
YES -> if the label is related to a specific concept, topic, object or style.
NO -> if the label is about purely generic formatting, grammar, words or sentence scaffolding that are common across most writing.
Output a list of label IDs with "YES" or "NO" decisions in this format:
{examples}
"""

GROUND_TRUTH_PRESENCE_PROMPT = """You are a meticulous dataset labeler. You are given a piece of text, and a list of {n_chunk} feature descriptions. Your task is to determine if each feature is present in the text.
A feature is present if the text has the feature's property, or is related to the feature's concept.
Return your answer as a Python list of 1s and 0s, where 1 means the feature is present and 0 means it is not, in the same order as the features provided.
TEXT:
{text}
FEATURE DESCRIPTIONS:
{features_prompt}"""


def filter_syntactic_labels(
    model, tokenizer,
    labels: dict,
    batch_size_labels: int = 30,
) -> dict:
    """Appendix K.2, "Filtre des labels syntaxiques". `labels` : {feature_id:
    label str}. Retourne {feature_id: bool} (True = concept informatif,
    à garder ; False = syntaxique/formatage, à écarter).

    Batché par groupes de `batch_size_labels` labels dans un seul prompt
    (format numéroté du papier), un seul appel `_batched_generate` par groupe
    -- pas un appel par label (le prompt du papier est conçu pour juger une
    liste entière d'un coup)."""
    items = list(labels.items())
    keep = {}
    for start in range(0, len(items), batch_size_labels):
        chunk = items[start:start + batch_size_labels]
        examples = "\n".join(f"{fid}: {lbl}" for fid, lbl in chunk) + "\n..."
        prompt = SYNTACTIC_LABEL_FILTER_PROMPT.format(examples=examples)
        resp = _batched_generate(model, tokenizer, [[{"role": "user", "content": prompt}]], max_new_tokens=400)[0]
        decisions = dict(re.findall(r"(\d+)\s*:\s*(YES|NO)", resp, flags=re.IGNORECASE))
        for fid, _ in chunk:
            d = decisions.get(str(fid), "").upper()
            keep[fid] = (d == "YES") if d in ("YES", "NO") else True  # repli : garder si non parsable
    return keep


def verify_pair_presence(
    model, tokenizer,
    label_i: str, label_j: str,
    documents: list,
    batch_size: int = 16,
) -> np.ndarray:
    """Appendix K.2, "Jugement de la présence ground-truth". Relabellise i et
    j INDÉPENDAMMENT (dans le MÊME appel juge par document -- le papier juge
    une liste de features en une passe, pas un appel par feature -- mais le
    résultat par feature reste indépendant l'un de l'autre dans le parsing).

    Retourne une matrice (2, n_documents) : ligne 0 = présence de i, ligne 1 =
    présence de j, sur chaque document."""
    features_prompt = f"1. {label_i}\n2. {label_j}"
    messages = [
        [{"role": "user", "content": GROUND_TRUTH_PRESENCE_PROMPT.format(
            n_chunk=2, text=doc, features_prompt=features_prompt,
        )}]
        for doc in documents
    ]
    responses = _batched_generate(model, tokenizer, messages, max_new_tokens=64, batch_size=batch_size)

    presence = np.zeros((2, len(documents)), dtype=int)
    for d_idx, resp in enumerate(responses):
        match = re.search(r"\[([01]\s*,\s*[01])\]", resp)
        if match:
            vals = [int(v.strip()) for v in match.group(1).split(",")]
            presence[0, d_idx], presence[1, d_idx] = vals[0], vals[1]
        else:
            # Repli : chiffre 0/1 immédiatement après ":" (ex. "feature 1: 1, feature 2: 0")
            # -- un `findall(r"[01]")` nu capterait aussi les numéros de feature eux-mêmes.
            ones = re.findall(r":\s*([01])\b", resp)
            if len(ones) >= 2:
                presence[0, d_idx], presence[1, d_idx] = int(ones[0]), int(ones[1])
            # repli ultime : 0/0 si non parsable (traité comme absent -- cf. "if unsure" du protocole voisin K.1)
    return presence


def compute_verified_npmi(presence_matrix: np.ndarray) -> float:
    """NPMI(i, j) à partir des labels juge (présence binaire), MÊME formule
    que `cooccurrence.compute_npmi` (implémentation unique, pas de
    réinvention) — appliquée ici à 2 colonnes seulement."""
    doc_acts = torch.from_numpy(presence_matrix.T.astype(np.float32))  # (n_docs, 2)
    npmi = compute_npmi(doc_acts)
    return float(npmi[0, 1])


def conditional_occurrence(presence_matrix: np.ndarray) -> float:
    """CO = max(P(i|j), P(j|i)) (Appendix E.1, mesure directionnelle
    complémentaire au NPMI, ne contrôle pas la fréquence individuelle)."""
    i, j = presence_matrix[0].astype(bool), presence_matrix[1].astype(bool)
    n_j, n_i = j.sum(), i.sum()
    p_i_given_j = (i & j).sum() / n_j if n_j > 0 else 0.0
    p_j_given_i = (i & j).sum() / n_i if n_i > 0 else 0.0
    return float(max(p_i_given_j, p_j_given_i))


def is_trivial_same_token_pair(
    fragments_dir: str,
    doc_ids: list,
    f_i: int,
    f_j: int,
    adjacency: int = 1,
    min_overlap_frac: float = 0.5,
) -> bool:
    """Appendix E.1 : "certaines paires co-occurrent parce qu'elles
    s'activent principalement sur le même token ou des tokens consécutifs
    (mal labellisées -- même concept, ou un token plus rare déclenche les
    deux)". Sur les documents où i ET j sont actifs, vérifie si leurs
    positions de PIC d'activation sont à distance <= `adjacency` tokens dans
    au moins `min_overlap_frac` d'entre eux -- proxy direct de la définition
    du papier, réutilise `fragment_store.feature_column` (pas de nouvelle
    lecture de tenseur brut)."""
    try:
        from src.storage.fragment_store import load_fragment, fragment_exists, feature_column
    except ImportError:
        from fragment_store import load_fragment, fragment_exists, feature_column

    n_close = n_checked = 0
    for doc_id in doc_ids:
        if not fragment_exists(fragments_dir, doc_id):
            continue
        frag = load_fragment(fragments_dir, doc_id)
        col_i, col_j = feature_column(frag, f_i), feature_column(frag, f_j)
        if col_i.max() <= 1e-6 or col_j.max() <= 1e-6:
            continue
        n_checked += 1
        if abs(int(np.argmax(col_i)) - int(np.argmax(col_j))) <= adjacency:
            n_close += 1
    if n_checked == 0:
        return False
    return (n_close / n_checked) >= min_overlap_frac
