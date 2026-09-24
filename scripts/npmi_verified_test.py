"""
scripts/npmi_verified_test.py — NPMI_verified (App E.1/E.3, arXiv:2512.10092v2,
Figure 4) : sans elle, `p1_interesting_correlations.json` est une liste de
CANDIDATS (NPMI brut sur activations SAE + dissimilarité de labels), pas un
résultat comparable au papier (docs/archive/audits/AUDIT_SAE_2026-08.md §1/§7).

Protocole : parmi les 150 features d'extension déjà labellisées
(`p1_top_extended_features.json`, interp_score=1 -- on reste autonome sans
reconstruire le dictionnaire complet core+extension), on prend les paires de
plus fort NPMI brut ET de labels sémantiquement DISSIMILAIRES (App E.1 :
"we filter to pairs with high NPMI but low dense embedding similarity of
their labels ... to ignore obvious correlations between related latents,
e.g. 'dog' and 'pet'" -- même filtre que `cooccurrence.find_interesting_pairs`,
dupliqué ici en confirmant Bo à la main plutôt que réimporté : premier essai
de ce script SANS ce filtre a sélectionné des quasi-synonymes ("Réclamation
Client"/"Réclamations Clients"), NPMI_verified=1,0 trivial et attendu, pas un
résultat -- corrigé). Ensuite, on filtre (a) les labels syntaxiques (juge,
App K.2), (b) les paires triviales co-activées sur le même token
(`src/analysis/correlations_verified.py::is_trivial_same_token_pair`, sur
`p1_token_fragments` déjà en cache), puis on relabellise i et j
INDÉPENDAMMENT sur un échantillon frais de documents train pour recalculer le
NPMI_verified + CO (Appendix E.1/E.3).

Usage :
    SAVE_DIR=./results_v10_emails_main/ PYTHONPATH=. \
      .venv/bin/python scripts/npmi_verified_test.py
"""
import json
import os
import random
import sys

import numpy as np
import torch
from src.sae.sae_shared import load_all_doc_acts
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import SAVE_DIR, CORPUS_SPLIT_SEED, LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH, LATENT_LABEL_EMB_MODEL
from src.data.preparation import build_email_train_test_corpus
from src.storage.fragment_store import resolve_extension_fragments_dir
from src.analysis.cooccurrence import compute_npmi
from src.analysis.correlations_verified import (
    filter_syntactic_labels, is_trivial_same_token_pair,
    verify_pair_presence, compute_verified_npmi, conditional_occurrence,
)
from src.sae.judge import load_judge_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K_PAIRS = int(os.environ.get("TOP_K_PAIRS", "8"))
RAW_NPMI_THRESHOLD = float(os.environ.get("RAW_NPMI_THRESHOLD", "0.3"))
# Seuil RELATIF (percentile de la distribution observée), pas absolu comme dans
# le papier (sim<0.2, App E.1) : diagnostiqué empiriquement (job 45630,
# scripts/npmi_similarity_diagnostic.py) que sur ce dépôt -- 150 features d'un
# SEUL domaine narrow (emails de réclamation client), contre des milliers de
# features multi-domaines (CivilComments/Pile) dans le papier -- même les
# paires les MOINS similaires ont sim>=0.369 parmi les 165 paires NPMI>0.3 :
# un seuil absolu à 0.2 ne sélectionne jamais rien ici, pas parce que le
# domaine n'a aucune paire "moins reliée que la moyenne", mais parce que
# l'échelle de dissimilarité elle-même est différente. Percentile 25 :
# garde les paires dans le quart le moins similaire des candidates NPMI-élevé,
# reproduit l'INTENTION du filtre (écarter les quasi-synonymes évidents,
# garder les paires relativement les moins attendues) sans halluciner un
# seuil absolu transférable d'un domaine à l'autre.
LABEL_SIM_PERCENTILE = float(os.environ.get("LABEL_SIM_PERCENTILE", "25"))
N_VERIFY_DOCS = int(os.environ.get("N_VERIFY_DOCS", "60"))
SEED = int(os.environ.get("SEED", "42"))


def _embed(texts: list, batch_size: int = 64) -> torch.Tensor:
    """Même convention que `saev5.py::_embed_bge_m3` -- dupliqué car `saev5.py`
    exécute tout son pipeline au chargement du module (non-importable comme
    bibliothèque, cf. docs/archive/audits/AUDIT_SAE_2026-08.md)."""
    tok = AutoTokenizer.from_pretrained(LATENT_LABEL_EMB_MODEL, local_files_only=True)
    mdl = AutoModel.from_pretrained(LATENT_LABEL_EMB_MODEL, local_files_only=True).to(DEVICE).eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], padding=True, truncation=True, max_length=64,
                      return_tensors="pt").to(DEVICE)
            cls = mdl(**enc).last_hidden_state[:, 0]
            embs.append(F.normalize(cls, p=2, dim=-1).cpu())
    del mdl
    return torch.cat(embs, dim=0)

CACHE_DIR = os.path.join(SAVE_DIR, "cache")
EXT_FEATURES_PATH = os.path.join(SAVE_DIR, "p1_top_extended_features.json")
TOKEN_FRAGMENTS_DIR = resolve_extension_fragments_dir(CACHE_DIR)  # features EXTENSION uniquement (N1, docs/archive/audits/AUDIT_SAE_2026-08.md §8) -- p1_token_fragments_ext si présent (post-N1), repli p1_token_fragments sinon (legacy).
OUT_PATH = os.path.join(CACHE_DIR, "npmi_verified.json")


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    with open(EXT_FEATURES_PATH, encoding="utf-8") as f:
        ext_data = json.load(f)
    feature_labels = {int(fid): v["label"] for fid, v in ext_data.items() if v.get("interp_score") == 1}
    print(f"[npmi-verif] {len(feature_labels)} features interprétables (sur {len(ext_data)}) "
          f"depuis {EXT_FEATURES_PATH}", flush=True)

    train_texts, _, _, _ = build_email_train_test_corpus(
        LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH, seed=CORPUS_SPLIT_SEED,
    )
    all_doc_acts_path = os.path.join(CACHE_DIR, "p1_all_doc_acts_ext_d1024.pt")
    if not os.path.exists(all_doc_acts_path):
        all_doc_acts_path = os.path.join(CACHE_DIR, "p1_all_doc_acts.pt")
    all_doc_sae_acts = load_all_doc_acts(all_doc_acts_path)
    train_doc_acts = all_doc_sae_acts[:len(train_texts)]
    print(f"[npmi-verif] {train_doc_acts.shape[0]} docs train, {train_doc_acts.shape[1]} features au total", flush=True)

    # NPMI brut (SAE) sur le sous-ensemble labellisé uniquement -- même formule
    # que cooccurrence_graph, restreinte pour rester tractable sans reconstruire
    # le graphe complet.
    fids = sorted(feature_labels.keys())
    sub_acts = train_doc_acts[:, fids]
    npmi_raw = compute_npmi(sub_acts)

    print("[npmi-verif] Embedding des labels (filtre dissimilarité, App E.1)...", flush=True)
    label_embs = _embed([feature_labels[fid] for fid in fids])

    n = len(fids)
    npmi_candidates = []
    for a in range(n):
        for b in range(a + 1, n):
            v = float(npmi_raw[a, b])
            if v > RAW_NPMI_THRESHOLD:
                sim = float(label_embs[a] @ label_embs[b])
                npmi_candidates.append((fids[a], fids[b], v, sim))

    sim_threshold = (
        float(np.percentile([c[3] for c in npmi_candidates], LABEL_SIM_PERCENTILE))
        if npmi_candidates else 0.0
    )
    print(f"[npmi-verif] {len(npmi_candidates)} paires à NPMI>{RAW_NPMI_THRESHOLD} ; "
          f"seuil de similarité dérivé (percentile {LABEL_SIM_PERCENTILE}) = {sim_threshold:.3f}", flush=True)

    pairs = [c for c in npmi_candidates if c[3] < sim_threshold]
    pairs.sort(key=lambda p: p[2], reverse=True)
    pairs = pairs[:TOP_K_PAIRS]
    print(f"[npmi-verif] {len(pairs)} paires candidates (NPMI brut > {RAW_NPMI_THRESHOLD}, "
          f"similarité label < {sim_threshold:.3f})", flush=True)
    for fi, fj, v, sim in pairs:
        print(f"    {fi} ({feature_labels[fi]!r}) <-> {fj} ({feature_labels[fj]!r}) : "
              f"NPMI={v:.3f} sim_label={sim:.3f}", flush=True)
    pairs = [(fi, fj, v) for fi, fj, v, _sim in pairs]

    print("[npmi-verif] Chargement du juge...", flush=True)
    model, tokenizer = load_judge_model()

    unique_labels = {fid: feature_labels[fid] for fi, fj, _ in pairs for fid in (fi, fj)}
    print(f"[npmi-verif] Filtre labels syntaxiques ({len(unique_labels)} labels)...", flush=True)
    keep = filter_syntactic_labels(model, tokenizer, unique_labels)
    pairs = [(fi, fj, v) for fi, fj, v in pairs if keep.get(fi, True) and keep.get(fj, True)]
    print(f"[npmi-verif] {len(pairs)} paires après filtre syntaxique", flush=True)

    print("[npmi-verif] Filtre paires triviales (même token)...", flush=True)
    non_trivial = []
    for fi, fj, v in pairs:
        both_active = ((sub_acts[:, fids.index(fi)] > 1e-6) & (sub_acts[:, fids.index(fj)] > 1e-6)).nonzero().flatten().tolist()
        doc_ids = both_active[:50]  # borne le coût -- assez pour estimer la fraction "même token"
        if not doc_ids or not is_trivial_same_token_pair(TOKEN_FRAGMENTS_DIR, doc_ids, fi, fj):
            non_trivial.append((fi, fj, v))
        else:
            print(f"    [trivial, écarté] {fi} <-> {fj}", flush=True)
    pairs = non_trivial
    print(f"[npmi-verif] {len(pairs)} paires après filtre trivial", flush=True)

    n_train = train_doc_acts.shape[0]
    sample_idx = random.sample(range(n_train), min(N_VERIFY_DOCS, n_train))
    sample_docs = [train_texts[i] for i in sample_idx]
    print(f"[npmi-verif] Vérification sur {len(sample_docs)} documents frais...", flush=True)

    results = []
    for fi, fj, npmi_sae in pairs:
        presence = verify_pair_presence(model, tokenizer, feature_labels[fi], feature_labels[fj], sample_docs)
        npmi_verified = compute_verified_npmi(presence)
        co = conditional_occurrence(presence)
        row = {
            "feature_i": fi, "label_i": feature_labels[fi],
            "feature_j": fj, "label_j": feature_labels[fj],
            "npmi_sae": npmi_sae, "npmi_verified": npmi_verified, "co_verified": co,
            "n_docs": len(sample_docs),
            # Matrice de présence brute conservée (pas seulement les stats agrégées) --
            # nécessaire pour diagnostiquer un résultat suspect sans devoir tout
            # relancer (leçon tirée d'un résultat NPMI=1,0 trivial, cf. docstring).
            "presence_i": presence[0].tolist(), "presence_j": presence[1].tolist(),
        }
        results.append(row)
        print(f"  {fi}<->{fj} : NPMI_sae={npmi_sae:.3f} NPMI_verified={npmi_verified:.3f} CO={co:.3f} "
              f"(présence i: {int(presence[0].sum())}/{len(sample_docs)}, "
              f"j: {int(presence[1].sum())}/{len(sample_docs)})", flush=True)

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"n_verify_docs": N_VERIFY_DOCS, "raw_npmi_threshold": RAW_NPMI_THRESHOLD, "pairs": results},
                   f, indent=2, ensure_ascii=False)
    print(f"\n[+] {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
