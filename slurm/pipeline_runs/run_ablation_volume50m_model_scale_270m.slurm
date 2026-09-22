#!/bin/bash
#SBATCH --job-name=SAE_abl_50m_270m
#SBATCH --partition=h100-bis
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/pipeline_runs/sae_abl_50m_270m_%j.log

SAE_ROOT="${SAE_ROOT:-/home/h21486/SAE}"

# Extension de la courbe d'échelle en dessous de 1B (jamais testé sur ce
# projet) + interaction volume x taille de modèle (cf. run_ablation_volume50m_
# model_scale_1b.slurm). GemmaScope-2 270M (layer_12_width_65k_l0_medium,
# src/config.py _PRESETS) téléchargé pour l'occasion (n'existait pas encore
# en cache local -- gemma-scope-2-270m-it). Layer 12 = ~2/3 profondeur
# (18 couches) -- même convention que 12B (layer 31/48) et 27B (layer 40/62),
# PAS la demi-profondeur utilisée pour 1B/4B (défaut _PRESETS, aucun sweep de
# couche dédié à cette échelle).
# --mem=180G : extrapolation depuis job 45803 (12B/25M, d_model=3840,
# 358,5 Go RSS réels) x2 (50M tokens) x(640/3840) (ratio de d_model 270M/12B)
# ~= 120 Go, marge ~50%.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

export SEED=42
export SAVE_DIR="./results_v44_ablation_volume50m_model_scale_270m/"
export USE_FROZEN_CORE="1"
export LOCAL_MAILS_PATH="${SAE_ROOT}/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="${SAE_ROOT}/local_data/emails/augmented_mails.jsonl"
export LOCAL_DATASET_PATH="${SAE_ROOT}/datasets/fineweb2_fra/data/fra_Latn/train/000_00000.parquet"
export VOLUME_FILLER_DATASET_GLOB="${SAE_ROOT}/datasets/fineweb2_fra/data/fra_Latn/train/000_0000*.parquet"
export EMB_MODEL="${SAE_ROOT}/models/F2LLM-v2-80M"

export MODEL_SIZE=270m
export MODEL_ID="${SAE_ROOT}/models/gemma-3-270m-it"
export DTYPE=bf16
export LAYER=12
export SAE_ID="layer_12_width_65k_l0_medium"
export PIPELINES=p1

export N_FEATURES_TO_LABEL=150
export EPOCHS_EXTRA=10
export D_EXTRA=1024
export K_EXTRA=5
export N_TOKENS_EXTRA_TRAIN=50000000
export N_VOLUME_FILLER_TARGET_CHUNKS=1200000

export PYTHONPATH="${SAE_ROOT}:${PYTHONPATH}"

cd "${SAE_ROOT}/"
.venv/bin/python src/sae/saev5.py
