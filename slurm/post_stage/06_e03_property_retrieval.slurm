#!/bin/bash
#SBATCH --job-name=SAE_post_e03_retrieval
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/post_stage/e03_retrieval_%j.out
#SBATCH --error=logs/post_stage/e03_retrieval_%j.err

SAE_ROOT="${SAE_ROOT:-/home/h21486/SAE}"

# E03 -- retrieval par propriete sur CONFIRM (plan §8). Depend de :
#  - 05_e01_encode_confirm.slurm (CONFIRM encode via le SAE deja fige)
#  - 04_e02_feature_registry.slurm (labels CORE/EXTRA pour le matching)
# h100 : bge-m3 (DENSE) + juge Qwen (relevance judging) tous deux GPU.

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

.venv/bin/python -u scripts/post_stage/e03_property_retrieval.py \
  --save-dir results_post_stage_e01_fit_1b_layer13_k5${RUN_SUFFIX:-}${EVAL_SUFFIX:-} \
  --mails-tsv-path local_data/emails/Mails.tsv \
  --augmented-jsonl-path local_data/emails/augmented_mails.jsonl \
  --split-assignments configs/post_stage/split_assignments.json \
  --registry-path results_post_stage_e01_fit_1b_layer13_k5${RUN_SUFFIX:-}/e02_feature_registry.json \
  --d-extra 1024 \
  --top-k 10 \
  --dense-model-path ./models/bge-m3 \
  --out e03_property_retrieval.json
