#!/bin/bash
#SBATCH --job-name=SAE_ce_r0_rand
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=logs/analysis/sae_ce_r0_trained_vs_random_%j.log

SAE_ROOT="${SAE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

# ΔCE (fidélité causale) du décodeur ENTRAÎNÉ (R0) contre les décodeurs
# ALÉATOIRES figés (C1/C1b), sous K_EXTRA=5/D_EXTRA=1024/layer 31 (R0) --
# RESULTS_TESTS.md §121 (audit de conformité R0 du rapport de stage).
# Réutilise trois checkpoints déjà entraînés (aucun réentraînement) : inférence
# seule (forward Gemma-3-12b-it pour ce_loss_increase).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

export MODEL_ID="${SAE_ROOT}/models/gemma-3-12b-it"
export MODEL_SIZE=12b
export DTYPE=bf16
export LAYER=31
export SAE_ID="layer_31_width_16k_l0_medium"
export D_EXTRA=1024
export K_EXTRA=5
export LOCAL_MAILS_PATH="${SAE_ROOT}/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="${SAE_ROOT}/local_data/emails/augmented_mails.jsonl"

export PYTHONPATH="${SAE_ROOT}:${PYTHONPATH}"

cd "${SAE_ROOT}/"
.venv/bin/python scripts/ce_loss_r0_trained_vs_random_test.py
