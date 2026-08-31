#!/bin/bash
#SBATCH --job-name=SAE_abl_a1_k32
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/pipeline_runs/sae_abl_a1_k32_%j.log

# A1 (AUDIT_SAE_2026-08.md, campagne finale) : K_EXTRA=32 (défaut historique du
# dépôt avant l'ablation "setup classique du papier" qui a introduit K_EXTRA=5,
# jamais retesté sous méthodologie finale) contre R0 (K_EXTRA=5). Copie exacte
# de R0 (results_v27_ablation_classic_setup_k5_25m_layer31), SEUL K_EXTRA change.
# Cache d'extraction partagé réutilisé (K_EXTRA hors clé de cache) -- coût limité
# à l'entraînement + jugement.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

export SEED=42
export SAVE_DIR="./results_v40_ablation_a1_k_extra32_classic_setup_25m_layer31/"
export USE_FROZEN_CORE="1"
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="/home/h21486/SAE/local_data/emails/augmented_mails.jsonl"
export LOCAL_DATASET_PATH="/home/h21486/SAE/datasets/fineweb2_fra/data/fra_Latn/train/000_00000.parquet"
export VOLUME_FILLER_DATASET_GLOB="/home/h21486/SAE/datasets/fineweb2_fra/data/fra_Latn/train/000_0000*.parquet"
export EMB_MODEL="/home/h21486/SAE/models/F2LLM-v2-80M"

export MODEL_ID="/home/h21486/SAE/models/gemma-3-12b-it"
export MODEL_SIZE=12b
export DTYPE=bf16
export LAYER=31
export SAE_ID="layer_31_width_16k_l0_medium"
export PIPELINES=p1

export N_FEATURES_TO_LABEL=300
export EPOCHS_EXTRA=10
export D_EXTRA=1024
export K_EXTRA=32
export N_TOKENS_EXTRA_TRAIN=25000000
export N_VOLUME_FILLER_TARGET_CHUNKS=540000

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python src/sae/saev5.py
