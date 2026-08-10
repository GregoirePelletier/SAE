"""
scripts/steering_fidelity_test.py — Le steering (`steer_activations`/`steer_and_decode`,
src/sae/sae_shared.py) existe dans le dépôt depuis le début mais n'est utilisé nulle
part hormis `run_steering_demo` (saev5.py), qui se limite à une vérification
géométrique superficielle (cosinus avant/après suppression/amplification d'UNE
feature, sans rapport avec une tâche en aval) -- signalé comme piste non exploitée
dans docs/references.md (entrée "A Survey on Sparse Autoencoders").

Question posée ici, complémentaire à explanation_fidelity_test.py (qui ablate
DIRECTEMENT dans l'espace des codes SAE, sans jamais appeler decode()) :
si on utilise VRAIMENT `steer_and_decode` -- décoder le code stimulé vers l'espace
résidu, puis RÉ-ENCODER ce résidu décodé -- l'intervention "tient"-elle à travers cet
aller-retour, ou le décodeur/encodeur du SAE la dilue-t-elle partiellement ? C'est un
test de fidélité du steering lui-même (pas seulement de l'explication), jamais fait
dans ce projet.

Limite méthodologique assumée : les vecteurs utilisés (`p1_all_doc_acts_ext_d1024.pt`)
sont des codes SAE poolés par MAX sur tous les tokens d'un document (comme dans
`run_steering_demo` déjà) -- pas le code d'un token réel. Décoder un pooling ne
reconstruit donc pas un résidu de token authentique, mais une direction résidu
"synthétique" représentant le mélange de concepts du document. Le test porte sur la
fidélité de l'intervention dans cet espace poolé, pas sur une reconstruction token
fidèle.

Protocole (réutilise explanation_fidelity_test.py comme base, dupliqué ici --
scripts de diagnostic volontairement autonomes) :
  1. Sonde logistique par intention, mêmes documents/features top-K identifiés que
     explanation_fidelity_test.py (mêmes graines, mêmes seuils).
  2. Pour chaque document échantillonné : ablation en PLACE (comme le test existant,
     témoin) vs ablation PAR STEER_AND_DECODE (decode -> re-encode -> re-score).
  3. Compare : (a) la chute de probabilité prédite dans les deux cas, (b) l'activation
     résiduelle des features "supprimées" après l'aller-retour decode/encode (0 idéal
     si l'intervention tient parfaitement).

Coût : charge le SAE core GemmaScope (petit, CPU possible) + le checkpoint
p1_frozen_core_d1024_k32.pt déjà entraîné (results_v10_emails_main) -- pas de LLM
Gemma-3-12B, pas de ré-extraction.

Usage : PYTHONPATH=. .venv/bin/python scripts/steering_fidelity_test.py
"""
from __future__ import annotations

import json
import os
import re

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from src.config import (
    LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH, SAVE_DIR, NEURONPEDIA_LABELS_PATH,
    RELEASE_ID, HOOK_TYPE, LOCAL_SAE_ROOT, SAE_SNAPSHOT, CORPUS_SPLIT_SEED,
)
from src.data.dataset import load_mails_tsv
from src.data.preparation import build_email_train_test_corpus
from src.sae import load_gemma_scope_sae
from src.sae.frozen_core import ExtendedSAE
from src.sae.sae_shared import steer_and_decode

CACHE_DIR = os.path.join(SAVE_DIR, "cache")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
TOP_K = 10
N_DOCS_SAMPLE = 200
D_EXTRA, K_EXTRA = 1024, 32   # doit matcher le checkpoint chargé (run principal)
# Pinné en dur (PAS le défaut ambiant de src.config, qui a été relevé à 65k après le
# run v12, cf. RESULTS_TESTS.md §10.3) : les activations en cache
# (p1_all_doc_acts_ext_d1024.pt) de results_v10_emails_main ont été produites avec le
# core SAE 16k -- charger le mauvais SAE_ID casserait silencieusement d_core et donc
# tout le découpage core/extra dans decode()/encode().
SAE_ID = "layer_24_width_16k_l0_medium"
rng_np = np.random.default_rng(SEED)


def replicate_load_and_clean_emails_with_index(tsv_path: str):
    """Dupliqué de explanation_fidelity_test.py (scripts de diagnostic autonomes)."""
    df = load_mails_tsv(tsv_path).rename(columns={"text": "document"})
    kept_row_indices = []
    for row_idx, row in df.iterrows():
        if "document" not in row or pd.isna(row["document"]):
            continue
        raw_text = str(row["document"])
        clean_text = re.sub(r'^\s*(?:Objet|Subject)\s*:\s*[^\n]+\n*', '', raw_text, flags=re.IGNORECASE)
        clean_text = re.sub(r'\[\s*\{\s*"start".*?\}\s*\]', '', clean_text, flags=re.DOTALL).strip()
        if clean_text:
            kept_row_indices.append(row_idx)
    return kept_row_indices, df


def load_real_email_acts_and_intents():
    kept_row_indices, df_full = replicate_load_and_clean_emails_with_index(LOCAL_MAILS_PATH)
    n_real = len(kept_row_indices)
    rng = np.random.default_rng(SEED)
    test_mask = rng.random(n_real) < float(os.environ.get("EMAIL_TEST_SPLIT", "0.05"))
    train_positions = [i for i in range(n_real) if not test_mask[i]]
    k_train_original = len(train_positions)

    train_texts, train_labels, _, _ = build_email_train_test_corpus(
        LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH, seed=CORPUS_SPLIT_SEED,
    )
    n_original_train = sum(1 for l in train_labels if l == "original")
    assert n_original_train == k_train_original, "Incohérence de correspondance -- cf. intent_urgency_probe.py"

    acts_path = os.path.join(CACHE_DIR, "p1_all_doc_acts_ext_d1024.pt")
    if not os.path.exists(acts_path):
        acts_path = os.path.join(CACHE_DIR, "p1_all_doc_acts.pt")
    all_doc_acts = torch.load(acts_path, map_location="cpu", weights_only=True)
    original_train_acts = all_doc_acts[:k_train_original].float().numpy()

    row_indices_for_train_original = [kept_row_indices[i] for i in train_positions]
    intent_cols = [c for c in df_full.columns if c.startswith("intent_")]
    labels_df = df_full.loc[row_indices_for_train_original, intent_cols].reset_index(drop=True)
    return original_train_acts, labels_df


def load_label_map():
    with open(NEURONPEDIA_LABELS_PATH) as f:
        labels_core = {int(k): v for k, v in json.load(f).items()}
    judge_path = os.path.join(CACHE_DIR, "p1_judge_labels_extended.json")
    label_map = dict(labels_core)
    if os.path.exists(judge_path):
        with open(judge_path) as f:
            judge_ext = json.load(f)
        for k, v in judge_ext.items():
            label_map[int(k)] = "[EXT] " + v.get("label", f"F{k}")
    return label_map


def load_frozen_core_sae() -> ExtendedSAE:
    sae_dir = os.path.join(LOCAL_SAE_ROOT, "snapshots", SAE_SNAPSHOT, HOOK_TYPE, SAE_ID)
    core_sae = load_gemma_scope_sae(
        sae_dir=sae_dir, device=DEVICE, release_id=RELEASE_ID, sae_id=f"{HOOK_TYPE}/{SAE_ID}",
    )
    ext_sae = ExtendedSAE(core_sae, d_extra=D_EXTRA, k_extra=K_EXTRA).to(DEVICE)
    ckpt_path = os.path.join(SAVE_DIR, f"p1_frozen_core_d{D_EXTRA}_k{K_EXTRA}.pt")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    ext_sae.load_state_dict(ckpt["state_dict"], strict=False)
    ext_sae.eval()
    return ext_sae


def probe_before_after(clf: LogisticRegression, x_before: np.ndarray, x_after: np.ndarray) -> tuple[float, float]:
    p_before = clf.predict_proba(x_before.reshape(1, -1))[0, 1]
    p_after = clf.predict_proba(x_after.reshape(1, -1))[0, 1]
    return float(p_before - p_after), float(p_before)


def main():
    print("[steering-fidelity] Chargement des activations et labels d'intention...")
    acts, labels_df = load_real_email_acts_and_intents()
    label_map = load_label_map()
    print(f"[steering-fidelity] {acts.shape[0]} mails originaux, {acts.shape[1]} dims SAE.")

    print("[steering-fidelity] Chargement du FrozenCoreResidualSAE (core GemmaScope + extension entraînée)...")
    ext_sae = load_frozen_core_sae()

    results = {}
    for col in labels_df.columns:
        y = labels_df[col].to_numpy().astype(int)
        n_pos = int(y.sum())
        if n_pos < 30 or n_pos > len(y) - 30:
            continue

        clf = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear")
        clf.fit(acts, y)
        coef = clf.coef_[0]
        probs = clf.predict_proba(acts)[:, 1]
        candidate_idx = np.where((y == 1) & (probs > 0.7))[0]
        if len(candidate_idx) == 0:
            continue
        sample_idx = rng_np.choice(candidate_idx, size=min(N_DOCS_SAMPLE, len(candidate_idx)), replace=False)

        drops_inplace, drops_roundtrip, residual_leak = [], [], []
        for i in sample_idx:
            x = acts[i]
            active = np.where(x > 1e-6)[0]
            if len(active) < TOP_K * 2:
                continue
            contributions = coef[active] * x[active]
            order = np.argsort(contributions)[::-1]
            top_feats = active[order[:TOP_K]]

            # Témoin : ablation en place dans l'espace des codes (comme explanation_fidelity_test.py)
            x_inplace = x.copy()
            x_inplace[top_feats] = 0.0
            d_inplace, p0 = probe_before_after(clf, x, x_inplace)

            # Steering réel : decode -> re-encode via steer_and_decode + ext_sae.encode
            with torch.no_grad():
                x_t = torch.from_numpy(x).unsqueeze(0).to(DEVICE)
                amplifications = {int(f): 0.0 for f in top_feats}
                decoded = steer_and_decode(x_t, amplifications, ext_sae)
                reencoded = ext_sae.encode(decoded).squeeze(0).float().cpu().numpy()
            d_roundtrip, _ = probe_before_after(clf, x, reencoded)
            leak = float(np.mean(reencoded[top_feats]) / max(np.mean(x[top_feats]), 1e-6))

            drops_inplace.append(d_inplace)
            drops_roundtrip.append(d_roundtrip)
            residual_leak.append(leak)

        if not drops_inplace:
            continue

        results[col] = {
            "n_docs_tested": len(drops_inplace),
            "mean_drop_inplace": float(np.mean(drops_inplace)),
            "mean_drop_roundtrip_steer_and_decode": float(np.mean(drops_roundtrip)),
            "ratio_roundtrip_vs_inplace": float(np.mean(drops_roundtrip) / max(np.mean(drops_inplace), 1e-6)),
            "mean_residual_leak_fraction": float(np.mean(residual_leak)),
        }
        print(f"[steering-fidelity] {col} (n={len(drops_inplace)}): "
              f"chute en place={np.mean(drops_inplace):.4f} | "
              f"chute steer_and_decode={np.mean(drops_roundtrip):.4f} | "
              f"ratio={np.mean(drops_roundtrip)/max(np.mean(drops_inplace),1e-6):.2f}x | "
              f"fuite résiduelle={np.mean(residual_leak):.3f}")

    out_path = os.path.join(CACHE_DIR, "steering_fidelity_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Écrit : {out_path}")


if __name__ == "__main__":
    main()
