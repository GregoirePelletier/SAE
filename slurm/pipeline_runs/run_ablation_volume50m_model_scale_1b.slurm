#!/bin/bash
#SBATCH --job-name=SAE_abl_50m_1b
#SBATCH --partition=h100-bis
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/pipeline_runs/sae_abl_50m_1b_%j.log

# Interaction volume x taille de modèle : R0/S1/S2 (25M tokens) ne montrent
# aucun effet d'échelle significatif (RESULTS_TESTS.md §97) -- teste si un
# petit modèle bénéficie DAVANTAGE de plus de tokens que le 12B (aucun effet
# de volume détecté pour ce dernier, §23). Copie exacte du point 1B de la
# courbe d'échelle (results_v29_ablation_classic_setup_k5_25m_model_scale_1b,
# §95), SEULE variable : N_TOKENS_EXTRA_TRAIN doublé (25M -> 50M).
# --mem=300G : extrapolation depuis job 45803 (12B/25M, d_model=3840,
# 358,5 Go RSS réels) x2 (tokens) x(1152/3840) (ratio de d_model 1B/12B)
# ~= 215 Go, marge ~40% (pas encore mesuré empiriquement pour cette taille,
# contrairement au point de calibration 50M/12B de results_probe_50m_layer31/).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

export SEED=42
export SAVE_DIR="./results_v43_ablation_volume50m_model_scale_1b/"
export USE_FROZEN_CORE="1"
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="/home/h21486/SAE/local_data/emails/augmented_mails.jsonl"
export LOCAL_DATASET_PATH="/home/h21486/SAE/datasets/fineweb2_fra/data/fra_Latn/train/000_00000.parquet"
export VOLUME_FILLER_DATASET_GLOB="/home/h21486/SAE/datasets/fineweb2_fra/data/fra_Latn/train/000_0000*.parquet"
export EMB_MODEL="/home/h21486/SAE/models/F2LLM-v2-80M"

export MODEL_SIZE=1b
export MODEL_ID="/home/h21486/SAE/models/gemma-3-1b-it"
export DTYPE=bf16
export LAYER=13
export SAE_ID="layer_13_width_16k_l0_medium"
export PIPELINES=p1

export N_FEATURES_TO_LABEL=300
export EPOCHS_EXTRA=10
export D_EXTRA=1024
export K_EXTRA=5
export N_TOKENS_EXTRA_TRAIN=50000000
export N_VOLUME_FILLER_TARGET_CHUNKS=1200000

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python src/sae/saev5.py
