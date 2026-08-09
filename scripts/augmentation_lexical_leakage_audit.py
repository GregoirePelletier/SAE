"""
scripts/augmentation_lexical_leakage_audit.py — audit méthodologique non
sollicité explicitement (trouvé en creusant de mon côté, cf. consigne du
2026-08-07 : ne pas se limiter aux pistes déjà nommées).

Hypothèse : `clf_acc_email_axes` (sonde logistique 5-plis,
`src/analysis/metrics.py::downstream_classification`, appelée sur les
activations SAE du corpus train pour les 14 classes axis__level) est utilisée
partout dans ce dépôt comme preuve que les codes latents du SAE séparent
emotion/urgence/registre/original (`Context.md` : "résultat encourageant
pour les cas d'usage détection d'urgence/intention"). Mais le corpus
augmenté est généré par un LLM (Gemma-3) à qui on donne une INSTRUCTION par
axe/niveau (`src/data/augmentation.py::AXES`) — si le LLM retombe sur des
formulations quasi figées par instruction (biais de génération connu des
LLM sous contrainte de style), un classifieur pourrait atteindre une haute
accuracy en repérant ces TICS LEXICAUX DE GÉNÉRATION plutôt qu'un signal
sémantique réel. Si un SAE (conçu pour capturer des CONCEPTS, pas du texte
littéral) sépare aussi bien que le texte brut, la séparation vient
peut-être du texte, pas du SAE.

Vérification préalable (n-grams les plus fréquents par classe,
`local_data/emails/augmented_mails.jsonl`, non rejetés) : 77-100% des
documents de chaque classe axis__level contiennent au moins un des 5
trigrammes les plus fréquents de leur propre classe (ex. "de bien vouloir"
dans 100% des 3278 `registre__soutenu`, "madame monsieur je" dans 99,8% des
`registre__standard`) — signal de templating fort, jamais documenté
jusqu'ici.

Ce script formalise le test : même protocole EXACT que
`downstream_classification` (StratifiedKFold 5 plis, LogisticRegression,
random_state=42) mais sur des features TF-IDF du texte brut au lieu des
activations SAE. Si l'accuracy lexicale est proche de `clf_acc_email_axes`
(93,5% rapporté, `RESULTS_TESTS.md`/`Context.md`), la métrique ne démontre
pas ce qu'elle est censée démontrer.

Usage (CPU uniquement) :
    PYTHONPATH=. .venv/bin/python scripts/augmentation_lexical_leakage_audit.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH, CORPUS_SPLIT_SEED
from src.data.preparation import build_email_train_test_corpus

OUT_PATH = "./local_data/emails/augmentation_lexical_leakage_results.json"
MIN_CLASS_SIZE = 10  # même seuil que saev5.py:1301 (StratifiedKFold(5) minimum)


def cv_accuracy(X, y, seed: int = 42) -> float:
    """Même protocole EXACT que downstream_classification (metrics.py:94-133) :
    StratifiedKFold 5 plis, LogisticRegression, random_state=42."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    accs = []
    for train_idx, test_idx in skf.split(X, y):
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[test_idx])
        accs.append(accuracy_score(y[test_idx], preds))
    return float(np.mean(accs))


def stock_phrase_templating_rate(texts_by_class: dict[str, list[str]]) -> dict[str, float]:
    """% de documents d'une classe contenant au moins un des 5 trigrammes les
    plus fréquents de CETTE classe -- reproduit la vérification préalable
    (docstring) de façon versionnée plutôt qu'en exploration ad hoc."""
    import re
    from collections import Counter

    def ngrams(text, n=3):
        words = re.findall(r"[a-zàâäéèêëïîôöùûüç']+", text.lower())
        return set(" ".join(words[i:i + n]) for i in range(len(words) - n + 1))

    rates = {}
    for cls, texts in texts_by_class.items():
        ngram_sets = [ngrams(t) for t in texts]
        counts = Counter()
        for s in ngram_sets:
            counts.update(s)
        top5 = {g for g, _ in counts.most_common(5)}
        n_containing = sum(1 for s in ngram_sets if s & top5)
        rates[cls] = n_containing / len(texts)
    return rates


def main() -> None:
    print("[leakage-audit] Reconstruction du split train (déterministe, CPU)...")
    train_texts, train_labels, _, _ = build_email_train_test_corpus(
        LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH, seed=CORPUS_SPLIT_SEED,
    )
    labels_arr = np.array(train_labels)
    counts = pd.Series(labels_arr).value_counts()
    usable = counts[counts >= MIN_CLASS_SIZE].index.tolist()
    print(f"[leakage-audit] {len(usable)} classes utilisables (>= {MIN_CLASS_SIZE} exemples) "
          f"sur {len(train_texts)} documents train.")

    mask = np.isin(labels_arr, usable)
    texts = [t for t, m in zip(train_texts, mask) if m]
    y_labels = labels_arr[mask]
    label_to_id = {lbl: i for i, lbl in enumerate(sorted(usable))}
    y = np.array([label_to_id[lbl] for lbl in y_labels])

    print("[leakage-audit] Vectorisation TF-IDF (mots, 1-3 grammes, max 20000 features)...")
    vec = TfidfVectorizer(ngram_range=(1, 3), max_features=20000, min_df=2)
    X = vec.fit_transform(texts)

    print("[leakage-audit] Sonde logistique 5-plis sur TF-IDF texte brut (protocole identique "
          "à downstream_classification, metrics.py:94-133)...")
    acc_lexical = cv_accuracy(X, y)
    print(f"[leakage-audit] acc_lexical (TF-IDF texte brut) = {acc_lexical:.4f}")

    texts_by_class = {lbl: [t for t, l in zip(texts, y_labels) if l == lbl] for lbl in usable}
    templating_rates = stock_phrase_templating_rate(texts_by_class)
    mean_templating = float(np.mean(list(templating_rates.values())))

    REPORTED_CLF_ACC_EMAIL_AXES = 0.935  # RESULTS_TESTS.md / Context.md, P1, run principal
    results = {
        "n_classes": len(usable),
        "n_docs": len(texts),
        "acc_lexical_tfidf": acc_lexical,
        "reported_clf_acc_email_axes_sae": REPORTED_CLF_ACC_EMAIL_AXES,
        "gap_sae_minus_lexical": REPORTED_CLF_ACC_EMAIL_AXES - acc_lexical,
        "templating_rate_by_class": templating_rates,
        "mean_templating_rate": mean_templating,
    }
    print("\n" + "=" * 74)
    print(" RÉSUMÉ")
    print("=" * 74)
    print(f"  acc_lexical (TF-IDF texte brut, 0 information sémantique explicite) : {acc_lexical:.1%}")
    print(f"  clf_acc_email_axes rapporté (SAE, run principal)                    : {REPORTED_CLF_ACC_EMAIL_AXES:.1%}")
    print(f"  écart SAE - lexical                                                 : {results['gap_sae_minus_lexical']:+.1%}")
    print(f"  taux de templating moyen (% docs avec un top-5 trigramme de classe) : {mean_templating:.1%}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Écrit : {OUT_PATH}")


if __name__ == "__main__":
    main()
