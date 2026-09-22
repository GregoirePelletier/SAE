#!/bin/bash
#SBATCH --job-name=SAE_post_e06_correlations
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/post_stage/e06_correlations_%j.out
#SBATCH --error=logs/post_stage/e06_correlations_%j.err

SAE_ROOT="${SAE_ROOT:-/home/h21486/SAE}"

# E06 -- correlations entre proprietes, confirmees hors decouverte (plan §11).
# Depend de : 04_e02_feature_registry.slurm (catalogue interpretable),
# 05_e01_encode_confirm.slurm (activations FIT/DEV/CONFIRM partagees).

set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
export CLUSTER_OFFLINE_MODE=1
export PYTHONPATH="${SAE_ROOT}:${PYTHONPATH:-}"

cd "${SAE_ROOT}"
mkdir -p logs/post_stage

.venv/bin/python -u scripts/post_stage/e06_correlations.py \
  --save-dir results_post_stage_e01_fit_1b_layer13_k5 \
  --mails-tsv-path local_data/emails/Mails.tsv \
  --augmented-jsonl-path local_data/emails/augmented_mails.jsonl \
  --split-assignments configs/post_stage/split_assignments.json \
  --registry-path results_post_stage_e01_fit_1b_layer13_k5/e02_feature_registry.json \
  --d-extra 1024 \
  --n-confirm-parents 350 \
  --out e06_correlations.json
