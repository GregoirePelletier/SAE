#!/bin/bash
#SBATCH --job-name=SAE_abl_classic_k5_l24
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/pipeline_runs/sae_abl_classic_k5_25m_layer24_%j.log

SAE_ROOT="${SAE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

# Complète le sweep layer "setup classique" (K_EXTRA=5/25M/stratifié/Qwen/
# déduplication) avec layer 24 -- le défaut historique du dépôt (couverture
# Neuronpedia), jamais mesuré sous ce protocole précis (K_EXTRA=5/25M tokens),
# seulement sous K_EXTRA=32/500k tokens (results_v10_emails_main) ou sous
# stratifié+Gemma sans déduplication (aucun run classic-k5 à layer 24 n'existe
# encore). Complète §90/§96 (layer 41/12) et le futur v27 (layer 31, relancé
# séparément) pour un sweep layer homogène sous méthodologie pleinement
# corrigée -- même famille de config que run_ablation_classic_setup_k5_25m_layer31.slurm,
# seul LAYER/SAE_ID change.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

export SEED=42
export SAVE_DIR="./results_v34_ablation_classic_setup_k5_25m_layer24/"
export USE_FROZEN_CORE="1"
export LOCAL_MAILS_PATH="${SAE_ROOT}/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="${SAE_ROOT}/local_data/emails/augmented_mails.jsonl"
export LOCAL_DATASET_PATH="${SAE_ROOT}/datasets/fineweb2_fra/data/fra_Latn/train/000_00000.parquet"
export VOLUME_FILLER_DATASET_GLOB="${SAE_ROOT}/datasets/fineweb2_fra/data/fra_Latn/train/000_0000*.parquet"
export EMB_MODEL="${SAE_ROOT}/models/F2LLM-v2-80M"

export MODEL_ID="${SAE_ROOT}/models/gemma-3-12b-it"
export MODEL_SIZE=12b
export DTYPE=bf16
export LAYER=24
export SAE_ID="layer_24_width_16k_l0_medium"
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
