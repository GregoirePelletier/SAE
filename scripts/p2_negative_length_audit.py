"""
scripts/p2_negative_length_audit.py -- Réplique la construction du négatif
Pipeline 2 (`build_phrase_examples_with_control`, §93 de RESULTS_TESTS.md,
`results_v10_emails_main/cache/p2_feature_labels.json`) sur les 150 features
réellement jugées, pour mesurer si un artefact de longueur analogue à celui
identifié côté Pipeline 1 (§115) existe malgré une construction du négatif
structurellement différente (pas d'argmax/fenêtre de contexte -- la phrase
entière est affichée). CPU-only, aucun modèle chargé (checkpoint SAE +
embeddings déjà en cache).

Sortie : JSON dans results_v10_emails_main/cache/p2_negative_length_audit.json
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from src.config import (
    LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH, CORPUS_SPLIT_SEED,
    MATRYOSHKA_DIM, MAX_PHRASES_DOC, EMB_MODEL, EMB_POOLING,
)
from src.data.preparation import build_email_train_test_corpus, split_into_phrases
from src.sae.phrase_sae import PhraseLevelSAE
from src.sae.judge import build_phrase_examples_with_control

SAVE_DIR = "./results_v10_emails_main/"
CACHE_DIR = os.path.join(SAVE_DIR, "cache")
JUDGE_CACHE = os.path.join(CACHE_DIR, "p2_feature_labels.json")
D_SAE, K = 8192, 16


def main():
    print("[1/5] Reconstruction du corpus (mêmes params que run_sae_v10_p2_stratified_qwen.slurm)...")
    _, _, test_texts, _, _, test_groups = build_email_train_test_corpus(
        LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH,
        seed=CORPUS_SPLIT_SEED, return_groups=True,
    )
    print(f"  test_texts={len(test_texts)}")

    print("[2/5] Découpage en phrases...")
    test_phrases, test_p2d_list = split_into_phrases(test_texts, max_phrases_per_doc=MAX_PHRASES_DOC)
    test_p2d_arr = np.array(test_p2d_list)
    print(f"  test_phrases={len(test_phrases)}")

    emb_tag = f"{os.path.basename(EMB_MODEL.rstrip('/'))}_{EMB_POOLING}"
    candidates = [
        os.path.join(CACHE_DIR, f"test_phrase_emb_dim{MATRYOSHKA_DIM}_n{len(test_phrases)}_{emb_tag}.pt"),
        os.path.join(CACHE_DIR, f"test_phrase_emb_dim{MATRYOSHKA_DIM}_n{len(test_phrases)}.pt"),
    ]
    emb_path = next((c for c in candidates if os.path.exists(c)), None)
    if emb_path is None:
        raise FileNotFoundError(
            f"Aucun cache d'embeddings pour n={len(test_phrases)} phrases parmi {candidates} -- "
            "le corpus reconstruit ne correspond pas exactement au run original."
        )
    print(f"[3/5] Chargement embeddings : {emb_path}")
    test_phrase_emb = torch.load(emb_path, map_location="cpu")
    assert test_phrase_emb.shape[0] == len(test_phrases), (
        f"Désalignement embeddings/phrases : {test_phrase_emb.shape[0]} vs {len(test_phrases)}"
    )

    sae_path = os.path.join(SAVE_DIR, f"p2_sae_dim{MATRYOSHKA_DIM}_d{D_SAE}_k{K}.pt")
    print(f"[4/5] Chargement PhraseLevelSAE : {sae_path}")
    sae = PhraseLevelSAE(MATRYOSHKA_DIM, D_SAE, K)
    ckpt = torch.load(sae_path, map_location="cpu")
    sae.load_state_dict(ckpt["state_dict"], strict=False)
    sae.eval()
    with torch.no_grad():
        chunks = []
        for i in range(0, test_phrase_emb.shape[0], 1024):
            chunks.append(sae.encode(test_phrase_emb[i:i + 1024]))
        test_phrase_acts = torch.cat(chunks, dim=0)
    print(f"  test_phrase_acts shape={tuple(test_phrase_acts.shape)}")

    with open(JUDGE_CACHE, "r", encoding="utf-8") as f:
        judged = json.load(f)
    feature_ids = [int(k) for k in judged.keys()]
    print(f"[5/5] Réplication de la sélection pos/neg sur {len(feature_ids)} features jugées...")

    def wc(s: str) -> int:
        return len(s.split())

    rows = []
    for fid in feature_ids:
        pos_examples, neg_example = build_phrase_examples_with_control(
            f_idx=fid, phrase_texts=test_phrases, phrase_acts=test_phrase_acts,
            n_pos=9, phrase_to_doc=test_p2d_arr, doc_groups=test_groups,
        )
        if neg_example is None or not pos_examples:
            continue
        neg_len = wc(neg_example)
        pos_lens = [wc(p) for p in pos_examples]
        rows.append({
            "f_idx": fid,
            "neg_example": neg_example,
            "neg_len_words": neg_len,
            "pos_len_words_mean": float(np.mean(pos_lens)),
            "pos_len_words_min": min(pos_lens),
            "neg_shorter_than_all_pos": neg_len < min(pos_lens),
            "interp_score": judged[str(fid)].get("interp_score"),
        })

    n = len(rows)
    neg_lens = [r["neg_len_words"] for r in rows]
    pos_mean_lens = [r["pos_len_words_mean"] for r in rows]
    shorter_all = sum(r["neg_shorter_than_all_pos"] for r in rows)
    distinct_neg = len(set(r["neg_example"] for r in rows))
    short3 = sum(1 for r in rows if r["neg_len_words"] <= 3)

    from scipy.stats import spearmanr
    rho, pval = spearmanr(neg_lens, [r["interp_score"] for r in rows])

    summary = {
        "n": n,
        "neg_len_words_mean": float(np.mean(neg_lens)),
        "neg_len_words_median": float(np.median(neg_lens)),
        "pos_len_words_mean_overall": float(np.mean(pos_mean_lens)),
        "neg_shorter_than_all_pos": shorter_all,
        "neg_shorter_than_all_pos_frac": shorter_all / n,
        "neg_le_3_words": short3,
        "neg_le_3_words_frac": short3 / n,
        "distinct_neg_examples": distinct_neg,
        "spearman_neg_len_vs_interp_score": {"rho": float(rho), "p": float(pval)},
        "rows": rows,
    }

    out_path = os.path.join(CACHE_DIR, "p2_negative_length_audit.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n=== RÉSUMÉ (n={n}) ===")
    print(f"  longueur négatif (mots)   : moyenne={summary['neg_len_words_mean']:.1f}  médiane={summary['neg_len_words_median']:.1f}")
    print(f"  longueur positifs (mots)  : moyenne={summary['pos_len_words_mean_overall']:.1f}")
    print(f"  négatif < tous les positifs : {shorter_all}/{n} ({100*shorter_all/n:.1f}%)")
    print(f"  négatif <= 3 mots           : {short3}/{n} ({100*short3/n:.1f}%)")
    print(f"  négatifs distincts          : {distinct_neg}/{n}")
    print(f"  Spearman(longueur_neg, interp_score) = {rho:.3f} (p={pval:.4f})")
    print(f"\nÉcrit : {out_path}")


if __name__ == "__main__":
    main()
