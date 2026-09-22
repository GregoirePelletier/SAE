#!/bin/bash
#SBATCH --job-name=SAE_abl_classic_k5_15m
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/pipeline_runs/sae_abl_classic_k5_15m_layer31_%j.log

SAE_ROOT="${SAE_ROOT:-/home/h21486/SAE}"

# Sweep volume de tokens, point 2/3 : 15M tokens, N_VOLUME_FILLER_TARGET_CHUNKS=0
# -- proche de la taille totale estimée du corpus réel (mails originaux +
# augmentés, ~14-15M tokens par estimation caractères, pas de tokenisation
# réelle) : ce point sature quasiment le corpus réel SANS filler, contre le
# point 25M (results_v27) qui complète avec ~10M tokens de filler générique.
# La comparaison 15M/25M isole donc l'effet du FILLER (volume brut, hors
# domaine) de l'effet du corpus réel lui-même (point 5M, results_v35).
# --mem=300G : extrapolation prudente entre les deux mesures empiriques
# connues (5M -> 31,3 Go, job 44991 ; 25M -> 381,8 Go, job 44994 -- croissance
# super-linéaire, ~174 Go estimés à 15M par ajustement en loi de puissance,
# marge large retenue ici faute de point de mesure réel à ce volume).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

export SEED=42
export SAVE_DIR="./results_v36_ablation_classic_setup_k5_15m_layer31/"
export USE_FROZEN_CORE="1"
export LOCAL_MAILS_PATH="${SAE_ROOT}/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="${SAE_ROOT}/local_data/emails/augmented_mails.jsonl"
export LOCAL_DATASET_PATH="${SAE_ROOT}/datasets/fineweb2_fra/data/fra_Latn/train/000_00000.parquet"
export EMB_MODEL="${SAE_ROOT}/models/F2LLM-v2-80M"

export MODEL_ID="${SAE_ROOT}/models/gemma-3-12b-it"
export MODEL_SIZE=12b
export DTYPE=bf16
export LAYER=31
export SAE_ID="layer_31_width_16k_l0_medium"
export PIPELINES=p1

export N_FEATURES_TO_LABEL=150
export EPOCHS_EXTRA=10
export D_EXTRA=1024
export K_EXTRA=5
export N_TOKENS_EXTRA_TRAIN=15000000
export N_VOLUME_FILLER_TARGET_CHUNKS=0

export PYTHONPATH="${SAE_ROOT}:${PYTHONPATH}"

cd "${SAE_ROOT}/"
.venv/bin/python src/sae/saev5.py
