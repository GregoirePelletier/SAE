#!/bin/bash
#SBATCH --job-name=SAE_post_e04_diffing
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/post_stage/e04_diffing_%j.out
#SBATCH --error=logs/post_stage/e04_diffing_%j.err

SAE_ROOT="${SAE_ROOT:-/home/h21486/SAE}"

# E04 -- diffing dans le domaine cible (plan §9). Contraste apparié :
# variantes de ton urgence__panique (A) vs urgence__calme (B) d'un même
# mail d'origine. Découverte sur FIT, gel des hypothèses, vérification sur
# CONFIRM. Dépend de 04_e02_feature_registry.slurm et de
# 05_e01_encode_confirm.slurm (activations CONFIRM déjà encodées).

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
  --save-dir results_post_stage_e01_fit_1b_layer13_k5 \
  --mails-tsv-path local_data/emails/Mails.tsv \
  --augmented-jsonl-path local_data/emails/augmented_mails.jsonl \
  --split-assignments configs/post_stage/split_assignments.json \
  --registry-path results_post_stage_e01_fit_1b_layer13_k5/e02_feature_registry.json \
  --d-extra 1024 \
  --label-a urgence__panique --label-b urgence__calme \
  --num-hypotheses 8 \
  --out e04_diffing.json
