#!/bin/bash
#SBATCH --job-name=SAE_post_e01_compare
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/post_stage/e01_compare_%j.out
#SBATCH --error=logs/post_stage/e01_compare_%j.err

SAE_ROOT="${SAE_ROOT:-/home/h21486/SAE}"

# E01 -- comparaison CORE/FULL/DENSE/TFIDF sur DEV, protocole FIT-train/
# DEV-eval strict (Plan_execution_SAE_15_jours_Claude_Code.md §6.3-6.4).
# Depend du checkpoint deja entraine par 01_e01_fit_reference.slurm (job
# 48585, COMPLETED) -- ne re-entraine rien, encode seulement DENSE (bge-m3,
# le seul poste qui beneficie reellement du GPU ici) et lit les activations
# SAE deja mises en cache.

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

.venv/bin/python -u scripts/post_stage/e01_compare_representations.py \
  --save-dir results_post_stage_e01_fit_1b_layer13_k5 \
  --mails-tsv-path local_data/emails/Mails.tsv \
  --augmented-jsonl-path local_data/emails/augmented_mails.jsonl \
  --split-assignments configs/post_stage/split_assignments.json \
  --d-extra 1024 \
  --dense-model-path ./models/bge-m3 \
  --out e01_representation_comparison.json
