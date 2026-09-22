#!/bin/bash
#SBATCH --job-name=SAE_abl_a3_w65k
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/pipeline_runs/sae_abl_a3_w65k_%j.log

SAE_ROOT="${SAE_ROOT:-/home/h21486/SAE}"

# A3 (AUDIT_SAE_2026-08.md, campagne finale) : largeur du SAE core 65k
# (gemma-scope-2-12b-it, layer_31_width_65k_l0_medium, téléchargé pour
# l'occasion -- n'existait localement que pour layer_24) contre R0 (§97,
# core 16k). Complète le balayage de largeur (§17 : 16k/65k/262k, mais
# jamais sous méthodologie finale ni au layer de référence 31). Copie
# exacte de R0 (results_v27_.../layer31), SEULE variable : SAE_ID (largeur
# du core -- D_EXTRA/K_EXTRA de l'extension inchangés, budget de capacité de
# l'extension indépendant de la largeur du core figé).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

export SEED=42
export SAVE_DIR="./results_v46_ablation_a3_core_width65k_classic_setup_25m_layer31/"
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
export SAE_ID="layer_31_width_65k_l0_medium"
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
