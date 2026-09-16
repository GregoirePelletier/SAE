#!/bin/bash
#SBATCH --job-name=SAE_post_e03_retrieval_a100
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/home/h21486/SAE/logs/post_stage/e03_retrieval_a100_%j.out
#SBATCH --error=/home/h21486/SAE/logs/post_stage/e03_retrieval_a100_%j.err

# Variante a100 de 06_e03_property_retrieval.slurm -- UNIQUEMENT quand
# h100/h100-bis sont verifies pleins (scontrol show node, pas seulement
# squeue). 2 GPU a100 (~79 Go cumules) + device_map="auto" : Qwen3.8-27B ne
# tient pas sur UN SEUL GPU a100 (39,49 Go reels, cf. campaign_policy.yaml,
# job 45724). Decision explicite de Gregoire le 2026-09-16 (max_gpus_per_job
# releve de 1 a 2 pour ce motif precis).

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

.venv/bin/python -u scripts/post_stage/e03_property_retrieval.py \
  --save-dir results_post_stage_e01_fit_1b_layer13_k5 \
  --mails-tsv-path local_data/emails/Mails.tsv \
  --augmented-jsonl-path local_data/emails/augmented_mails.jsonl \
  --split-assignments configs/post_stage/split_assignments.json \
  --registry-path results_post_stage_e01_fit_1b_layer13_k5/e02_feature_registry.json \
  --d-extra 1024 \
  --top-k 10 \
  --dense-model-path ./models/bge-m3 \
  --judge-device auto \
  --out e03_property_retrieval.json
