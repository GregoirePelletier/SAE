#!/bin/bash
#SBATCH --job-name=SAE_post_e01_pool_len
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/h21486/SAE/logs/post_stage/e01_pool_len_%j.out
#SBATCH --error=/home/h21486/SAE/logs/post_stage/e01_pool_len_%j.err

# E01 -- complement (plan §6.3) : pooling top-k-moyenne vs max-pooling deja
# en cache, baseline longueur seule, accuracy par tranche de longueur.
# CPU uniquement (aucun appel torch.cuda dans le script) -- GPU demande par
# coherence avec le reste de la campagne, pas utilise ici. Depend du
# checkpoint/fragments deja produits par 01_e01_fit_reference.slurm (job
# 48585, COMPLETED) : ne re-entraine rien, ne relance aucune extraction.

set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH:-}"

cd /home/h21486/SAE
mkdir -p logs/post_stage

.venv/bin/python -u scripts/post_stage/e01_pooling_and_length_analysis.py \
  --save-dir results_post_stage_e01_fit_1b_layer13_k5 \
  --mails-tsv-path local_data/emails/Mails.tsv \
  --augmented-jsonl-path local_data/emails/augmented_mails.jsonl \
  --split-assignments configs/post_stage/split_assignments.json \
  --d-extra 1024 \
  --topk 3 \
  --out e01_pooling_and_length_analysis.json
