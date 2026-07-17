#!/bin/bash

#SBATCH --job-name=SAE_v9_smoketest
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=4:00:00
#SBATCH --output=sae_v9_test_%j.log

# ── Isolation réseau ──
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ── v9 : NOUVEAU répertoire (θ + AuxK changent la sémantique des poids ;
#    surtout, le cache v8 contient des fragments .pkl denses de ~400 Mo/doc
#    et all_doc_acts calculés avec TopK per-sample — ne pas mélanger) ──
export SEED=42
export SAVE_DIR="./results_v9_test/"
export USE_FROZEN_CORE="1"
export LOCAL_MAILS_PATH="/home/h21486/SAE/Mails.tsv"
export MODEL_ID="/home/h21486/SAE/models/gemma-3-12b-it"
export LOCAL_DATASET_PATH="/home/h21486/SAE/datasets/fineweb2_fra/data/fra_Latn/train/000_00000.parquet"
export EMB_MODEL="/home/h21486/SAE/models/F2LLM-v2-80M"

# ── Pipeline 1 : mêmes IDs que le run cible pour valider le chemin complet ──
export MODEL_SIZE=12b
# 16k plutôt que 262k : couverture Neuronpedia bien plus dense en proportion sur
# 24-gemmascope-2-res-16k (262k n'a qu'environ 10k features labellisées sur 262144,
# constaté manuellement) — cf. src/config.py pour le détail du choix.
export SAE_ID="layer_24_width_16k_l0_medium"
export DTYPE=bf16

# ── Volumes réduits ~×12 : extraction P1 ~30 min (vs 6h),
#    ré-encodage sparse : secondes/doc -> minutes (vs 28h projetées) ──
export N_TOTAL_ENERGY=400
export N_TOTAL_SPORTS=400
export N_TOTAL_SUPPORT=400
export N_TOKENS_EXTRA_TRAIN=60000

# ── Pipeline 2 réduit ──
export D_SAE=2048
export K_SPARSE=16
export EPOCHS=15

export N_FEATURES_TO_LABEL=8
export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python src/sae/saev5.py
# 1. Lignes d'epoch ExtendedSAE contiennent "aux=" (AuxK actif, code v9 déployé)
#    et dead_frac chute nettement plus vite que 0.465->0.186 du run v8.
# 2. Étape "Re-encodage ExtendedSAE (sparse)" : it/s >> 1 (attendu: >5 it/s,
#    vs 0.18 it/s en v8). Si ~0.2 it/s persiste -> le store sparse n'est pas actif.
# 3. ls -lh $SAVE_DIR/cache/p1_token_fragments/ : fichiers doc_*.pt de ~0.1-1 Mo
#    (pas de .pkl de centaines de Mo).
# ── Critères de validation à vérifier dans le log AVANT de lancer le run com
# 4. Phrase-SAE (P2) : logs "θ=" non nuls et croissants-stables ; L0 eval raisonnable.
# 5. FVE/ρ_SAE calculés sans erreur (chemins decode_core_sparse).