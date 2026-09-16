#!/bin/bash
#SBATCH --job-name=SAE_post_e02_registry
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/home/h21486/SAE/logs/post_stage/e02_registry_%j.out
#SBATCH --error=/home/h21486/SAE/logs/post_stage/e02_registry_%j.err

# E02 -- registre de 300 features (150 CORE + 150 EXTRA) labellisees par le
# juge Qwen (plan §7). Depend du checkpoint/fragments deja produits par
# 01_e01_fit_reference.slurm (job 48585, COMPLETED). h100 obligatoire :
# Qwen3.8-27B bf16 ne tient pas sur a100 (cf. campaign_policy.yaml).

set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
export CLUSTER_OFFLINE_MODE=1
export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH:-}"

cd /home/h21486/SAE
mkdir -p logs/post_stage

.venv/bin/python -u scripts/post_stage/e02_feature_registry.py \
  --save-dir results_post_stage_e01_fit_1b_layer13_k5 \
  --mails-tsv-path local_data/emails/Mails.tsv \
  --augmented-jsonl-path local_data/emails/augmented_mails.jsonl \
  --split-assignments configs/post_stage/split_assignments.json \
  --d-extra 1024 \
  --n-core 150 --n-extra 150 \
  --sae-revision e01_fit_1b_layer13_k5 \
  --out e02_feature_registry.json
