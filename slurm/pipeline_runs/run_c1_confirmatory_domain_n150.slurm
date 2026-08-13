#!/bin/bash
#SBATCH --job-name=SAE_c1_confirm
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=logs/pipeline_runs/sae_c1_confirm_%j.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ── Test confirmatoire pour le constat CRITICAL C1 de la revue externe
# (REVIEW_ARS_PANEL.md, RESULTS_TESTS.md §44/§45) ───────────────────────────
# L'affirmation centrale du rapport (20,0% corpus generic n=10 vs 45,3% corpus
# emails n=150, §12) n'a jamais été testée à n apparié -- confond domaine ET
# taille d'échantillon. Ce run réplique EXACTEMENT le baseline pré-correctif
# (results_v9_full, corpus generic energy/sports/support, avant e5697c6) mais
# à N_FEATURES_TO_LABEL=150 au lieu de 10, pour une comparaison correcte contre
# le run principal emails (results_v10_emails_main, 68/150).
# CONFIRMATORY_DOMAIN_BASELINE=1 (src/sae/saev5.py __main__, ajout guardé par
# env var, comportement par défaut inchangé) bascule train_texts/test_texts sur
# le corpus generic au lieu de emails+augmentés.
# Checkpoint réutilisé (pas de réentraînement) : p1_frozen_core_d1024_k32.pt +
# p1_extended_sae.pt copiés depuis results_v9_full/ vers SAVE_DIR AVANT
# soumission (saev5.py charge automatiquement si le fichier existe déjà,
# saev5.py:955-969) -- seule l'extraction Gemma-3-12B (forward-only, pas de
# training loop) et le jugement odd-one-out sur 150 features sont recalculés.
# SAE_ID=16k (PAS le défaut actuel 65k) : le checkpoint existant a été entraîné
# contre le core GemmaScope 16k (vérifié par inspection directe du .pt,
# W_dec.shape=(16384, 3840)) -- une largeur différente ferait planter le
# load_state_dict par mismatch de dimensions.
export SEED=42
export SAVE_DIR="./results_v9_confirmatory_n150/"
export USE_FROZEN_CORE="1"
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="/home/h21486/SAE/local_data/emails/augmented_mails.jsonl"
export LOCAL_DATASET_PATH="/home/h21486/SAE/datasets/fineweb2_fra/data/fra_Latn/train/000_00000.parquet"
export EMB_MODEL="/home/h21486/SAE/models/F2LLM-v2-80M"

export MODEL_ID="/home/h21486/SAE/models/gemma-3-12b-it"
export MODEL_SIZE=12b
export DTYPE=bf16
export SAE_ID="layer_24_width_16k_l0_medium"
export PIPELINES=p1

export CONFIRMATORY_DOMAIN_BASELINE=1
export N_TOTAL_ENERGY=2000
export N_TOTAL_SPORTS=2000
export N_TOTAL_SUPPORT=2000
export CONFIRMATORY_TEST_SPLIT=0.1

export N_FEATURES_TO_LABEL=150
export EPOCHS_EXTRA=10
export D_EXTRA=1024
export K_EXTRA=32

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python src/sae/saev5.py
