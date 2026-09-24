#!/bin/bash
#SBATCH --job-name=SAE_post_e04_diffing_a100
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/post_stage/e04_diffing_a100_%j.out
#SBATCH --error=logs/post_stage/e04_diffing_a100_%j.err

SAE_ROOT="${SAE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

# Variante a100 de 07_e04_diffing.slurm -- meme motif que 06b (Qwen sharde
# sur 2 GPU a100, uniquement si h100/h100-bis verifies pleins).

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

.venv/bin/python -u scripts/post_stage/e04_diffing.py \
  --save-dir results_post_stage_e01_fit_1b_layer13_k5${RUN_SUFFIX:-}${EVAL_SUFFIX:-} \
  --mails-tsv-path local_data/emails/Mails.tsv \
  --augmented-jsonl-path local_data/emails/augmented_mails.jsonl \
  --split-assignments configs/post_stage/split_assignments.json \
  --registry-path results_post_stage_e01_fit_1b_layer13_k5${RUN_SUFFIX:-}/e02_feature_registry.json \
  --d-extra 1024 \
  --label-a urgence__panique --label-b urgence__calme \
  --num-hypotheses 8 \
  --judge-device auto \
  --out e04_diffing.json
