"""
scripts/audit_2026_08_uniform_hardneg_rejudge.py — B.2 + B.3 de
`docs/AUDIT_2026-08.md`, combinés : le taux 45,3% survit-il (a) à un
échantillon UNIFORME de features vivantes (pas le top-N par magnitude,
biaisé vers les features denses/génériques) ET (b) à un contrôle NÉGATIF DUR
(mot d'activation maximale d'une AUTRE feature, pas un mot arbitraire au
milieu d'un document choisi au hasard) ?

Ne modifie PAS `src/sae/judge.py` (module partagé, déjà consommé par des
résultats publiés) -- monkey-patch local à ce script, même principe déjà
appliqué par `c2_original_only_rejudge.py`/`contrastive_labeling_test.py`.

Protocole :
  1. Features "vivantes" de l'extension (plage [d_core, d_core+D_EXTRA),
     découplage core/extended déjà en place dans le dépôt) : au moins
     N_MIN_DOCS documents avec activation > seuil, sur le split train.
  2. Échantillon UNIFORME de N_FEATURES parmi ces features vivantes (pas de
     tri par magnitude) -- répond à B.2.
  3. Positifs : IDENTIQUE au protocole existant (top-9 par magnitude,
     dédupliqués par mot-cible) -- seule la sélection des FEATURES change,
     pas la construction des exemples positifs pour une feature donnée.
  4. Négatif : mot d'activation maximale d'une AUTRE feature vivante tirée
     au hasard (jamais la feature testée) -- répond à B.3.
  5. Rapporte le taux avec IC de Wilson (`src/analysis/stats.py`), et compare
     par McNemar apparié au run principal SUR LES FEATURES EN COMMUN (peu
     probables vu l'échantillonnage indépendant -- rapporté à titre
     informatif, la comparaison principale est le taux agrégé lui-même,
     incomparable au run principal car protocole différent sur les deux axes
     à la fois).

Usage : sbatch slurm/validation/run_audit_uniform_hardneg_rejudge.slurm
"""
from __future__ import annotations

import json
import os
import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import src.sae.judge as judge_mod
from src.sae.judge import (
    fragment_exists, load_fragment, feature_column, extract_causal_context,
    odd_one_out_judge,
)
from src.config import (
    MODEL_ID, HF_TOKEN, DTYPE, SAVE_DIR, LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH,
    CORPUS_SPLIT_SEED,
)
from src.data.preparation import build_email_train_test_corpus
from src.analysis.stats import proportion_with_ci

DEVICE = "cuda"
TORCH_DTYPE = torch.bfloat16 if DTYPE == "bf16" else torch.float16
SEED = 42
N_FEATURES = 150            # même n que le run principal, comparaison directe des taux
N_MIN_DOCS_ALIVE = 15       # "vivante" : au moins ce nombre de docs activent la feature
D_CORE = 16384               # width 16k, cf. steering_fidelity_test.py / config par défaut
D_EXTRA = 1024
THRESHOLD_POS = 1e-6

CACHE_DIR = os.path.join(SAVE_DIR, "cache")
TOKEN_FRAGMENTS_DIR = os.path.join(CACHE_DIR, "p1_token_fragments")
ORIGINAL_JUDGE_CACHE = os.path.join(CACHE_DIR, "p1_judge_labels_extended.json")
OUT_PATH = os.path.join(CACHE_DIR, "audit_2026_08_uniform_hardneg_results.json")

random.seed(SEED)
rng_np = np.random.default_rng(SEED)


def build_positive_examples(f_idx: int, acts: torch.Tensor, n_pos: int = 9) -> list[str]:
    """Identique à build_feature_examples_with_control (judge.py) pour la
    partie POSITIFS uniquement -- dupliqué ici pour ne pas dépendre de son
    négatif (remplacé plus bas par un négatif dur)."""
    f_acts = acts[:, f_idx].detach().float().numpy()
    sorted_desc = np.argsort(f_acts)[::-1]
    pos_examples, seen_target_words = [], set()
    for d_idx in sorted_desc:
        if f_acts[d_idx] <= THRESHOLD_POS:
            break
        if not fragment_exists(TOKEN_FRAGMENTS_DIR, int(d_idx)):
            continue
        doc_data = load_fragment(TOKEN_FRAGMENTS_DIR, int(d_idx))
        token_acts = feature_column(doc_data, f_idx)
        max_act = token_acts.max()
        if max_act <= THRESHOLD_POS:
            continue
        target_idx = int(token_acts.argmax())
        ctx = extract_causal_context(doc_data["token_strings"], target_idx)
        import re
        m = re.search(r"<<(.+?)>>", ctx)
        target_word = m.group(1).strip().lower() if m else ctx.strip().lower()
        if target_word in seen_target_words:
            continue
        seen_target_words.add(target_word)
        pos_examples.append(ctx)
        if len(pos_examples) >= n_pos:
            break
    return pos_examples


def top_word_for_feature(f_idx: int, acts: torch.Tensor) -> str | None:
    """Contexte marqué au mot d'activation maximale pour f_idx -- utilisé
    comme négatif DUR pour une AUTRE feature."""
    f_acts = acts[:, f_idx].detach().float().numpy()
    sorted_desc = np.argsort(f_acts)[::-1]
    for d_idx in sorted_desc[:50]:
        if f_acts[d_idx] <= THRESHOLD_POS:
            break
        if not fragment_exists(TOKEN_FRAGMENTS_DIR, int(d_idx)):
            continue
        doc_data = load_fragment(TOKEN_FRAGMENTS_DIR, int(d_idx))
        token_acts = feature_column(doc_data, f_idx)
        if token_acts.max() <= THRESHOLD_POS:
            continue
        target_idx = int(token_acts.argmax())
        return extract_causal_context(doc_data["token_strings"], target_idx)
    return None


def main():
    print("[uniform-hardneg] Chargement des activations et du split train...")
    train_texts, _, _, _ = build_email_train_test_corpus(
        LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH, seed=CORPUS_SPLIT_SEED,
    )
    n_train = len(train_texts)
    acts_path = os.path.join(CACHE_DIR, "p1_all_doc_acts_ext_d1024.pt")
    all_doc_acts = torch.load(acts_path, map_location="cpu", weights_only=True)
    train_acts = all_doc_acts[:n_train]
    print(f"[uniform-hardneg] train_acts: {tuple(train_acts.shape)}")

    # 1. Features vivantes de l'extension (plage [D_CORE, D_CORE+D_EXTRA))
    ext_acts = train_acts[:, D_CORE:D_CORE + D_EXTRA]
    n_alive_docs = (ext_acts > THRESHOLD_POS).sum(dim=0)
    alive_local = (n_alive_docs >= N_MIN_DOCS_ALIVE).nonzero(as_tuple=True)[0].tolist()
    alive_global = [D_CORE + i for i in alive_local]
    print(f"[uniform-hardneg] {len(alive_global)}/{D_EXTRA} features d'extension vivantes "
          f"(>= {N_MIN_DOCS_ALIVE} docs actifs).")
    assert len(alive_global) >= N_FEATURES, (
        f"Seulement {len(alive_global)} features vivantes, besoin de {N_FEATURES} -- "
        f"réduire N_MIN_DOCS_ALIVE ou N_FEATURES avant de continuer."
    )

    # 2. Échantillon UNIFORME (B.2)
    sampled_idx = rng_np.choice(len(alive_global), size=N_FEATURES, replace=False)
    feature_indices = sorted(int(alive_global[i]) for i in sampled_idx)
    print(f"[uniform-hardneg] {len(feature_indices)} features échantillonnées uniformément.")

    with open(ORIGINAL_JUDGE_CACHE, encoding="utf-8") as f:
        original_judge_data = json.load(f)
    n_overlap = len(set(feature_indices) & {int(k) for k in original_judge_data.keys()})
    print(f"[uniform-hardneg] Recouvrement avec l'échantillon top-N original : {n_overlap}/{len(feature_indices)}.")

    # 3-4. Positifs identiques au protocole existant + négatif DUR (B.3)
    _orig_fn = judge_mod.build_feature_examples_with_control

    def _uniform_hardneg_build_examples(f_idx, token_fragments_dir, acts, offset=0, n_pos=9, neg_quantile=0.05):
        pos_examples = build_positive_examples(f_idx, acts, n_pos=n_pos)
        neg_candidates = [g for g in feature_indices if g != f_idx]
        random.shuffle(neg_candidates)
        neg_example = None
        for g_idx in neg_candidates[:10]:
            neg_example = top_word_for_feature(g_idx, acts)
            if neg_example:
                break
        return pos_examples, neg_example

    judge_mod.build_feature_examples_with_control = _uniform_hardneg_build_examples

    print(f"[uniform-hardneg] Chargement du juge {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=TORCH_DTYPE, device_map=DEVICE,
        low_cpu_mem_usage=True, token=HF_TOKEN, trust_remote_code=True, local_files_only=True,
    ).eval()

    results = odd_one_out_judge(
        model, tokenizer, feature_indices, TOKEN_FRAGMENTS_DIR, train_acts, offset=0, n_pos=9,
    )
    judge_mod.build_feature_examples_with_control = _orig_fn  # restauration (hygiène)

    n = len(results)
    n_interp = sum(1 for v in results.values() if v.get("interp_score") == 1)
    n_dead = sum(1 for v in results.values() if v.get("label") == "dead_feature")
    rate_res = proportion_with_ci(n_interp, n)

    n_orig_interp = sum(1 for v in original_judge_data.values() if v.get("interp_score") == 1)
    n_orig = len(original_judge_data)
    rate_orig = proportion_with_ci(n_orig_interp, n_orig)

    summary = {
        "n_tested": n, "n_interp": n_interp, "n_dead_or_insufficient": n_dead,
        "rate": rate_res.rate, "ci95_low": rate_res.ci_low, "ci95_high": rate_res.ci_high,
        "n_alive_features_pool": len(alive_global), "n_overlap_with_original_sample": n_overlap,
        "reference_original_topN_easyneg": {
            "n": n_orig, "n_interp": n_orig_interp,
            "rate": rate_orig.rate, "ci95_low": rate_orig.ci_low, "ci95_high": rate_orig.ci_high,
        },
    }
    print("\n" + "=" * 70)
    print(" RÉSUMÉ — B.2+B.3 : ÉCHANTILLON UNIFORME + NÉGATIF DUR")
    print("=" * 70)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_feature": results, "feature_indices": feature_indices},
                   f, indent=2, ensure_ascii=False)
    print(f"\n[+] Écrit : {OUT_PATH}")


if __name__ == "__main__":
    main()
