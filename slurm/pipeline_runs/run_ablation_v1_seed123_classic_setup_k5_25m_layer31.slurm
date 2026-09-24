#!/bin/bash
#SBATCH --job-name=SAE_abl_v1_seed123
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/pipeline_runs/sae_abl_v1_seed123_%j.log

SAE_ROOT="${SAE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

# V1 (docs/archive/audits/AUDIT_SAE_2026-08.md, campagne finale) : variabilité run-à-run de R0. Tout le
# dépôt est à n=1 graine à ce jour -- ce run rejoue R0 (run_ablation_classic_setup_
# k5_25m_layer31.slurm, results_v27_ablation_classic_setup_k5_25m_layer31) à
# l'identique, SEED=123 au lieu de 42. CORPUS_SPLIT_SEED (src/config.py, découplé de
# SEED) n'est PAS modifié : le split train/test du corpus reste identique à R0, seule
# la variance d'entraînement du SAE (init des poids extra, ordre des lots) et de
# sélection de features est isolée. Depuis le cache d'extraction partagé (SEED
# n'entre pas dans la clé de cache, CLAUDE.md), ce run réutilise l'extraction déjà
# écrite par R0 -- coût limité à l'entraînement + jugement.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

export SEED=123
export SAVE_DIR="./results_v38_ablation_v1_seed123_classic_setup_k5_25m_layer31/"
export USE_FROZEN_CORE="1"
export LOCAL_MAILS_PATH="${SAE_ROOT}/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="${SAE_ROOT}/local_data/emails/augmented_mails.jsonl"
export LOCAL_DATASET_PATH="${SAE_ROOT}/datasets/fineweb2_fra/data/fra_Latn/train/000_00000.parquet"
export VOLUME_FILLER_DATASET_GLOB="${SAE_ROOT}/datasets/fineweb2_fra/data/fra_Latn/train/000_0000*.parquet"
export EMB_MODEL="${SAE_ROOT}/models/F2LLM-v2-80M"

export MODEL_ID="${SAE_ROOT}/models/gemma-3-12b-it"
export MODEL_SIZE=12b
export DTYPE=bf16
export LAYER=31
export SAE_ID="layer_31_width_16k_l0_medium"
export PIPELINES=p1

export N_FEATURES_TO_LABEL=300
export EPOCHS_EXTRA=10
export D_EXTRA=1024
export K_EXTRA=5
export N_TOKENS_EXTRA_TRAIN=25000000
export N_VOLUME_FILLER_TARGET_CHUNKS=540000

export PYTHONPATH="${SAE_ROOT}:${PYTHONPATH}"

cd "${SAE_ROOT}/"
.venv/bin/python src/sae/saev5.py
