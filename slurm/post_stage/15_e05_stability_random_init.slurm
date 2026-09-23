#!/bin/bash
#SBATCH --job-name=SAE_post_e05_stab_rand
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/post_stage/e05_stability_randinit_%j.out
#SBATCH --error=logs/post_stage/e05_stability_randinit_%j.err

SAE_ROOT="${SAE_ROOT:-/home/h21486/SAE}"

# E05 -- stabilite avec bras temoin "init aleatoire" : 2 runs pca (42/43) et
# 2 runs random (45/46), meme procedure pour les trois types de paires
# (pca->pca, pca->random, random->random). CPU seul. Depend de
# 01/10 (pca), 13/14 (random).

set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONPATH="${SAE_ROOT}:${PYTHONPATH:-}"

cd "${SAE_ROOT}"
mkdir -p logs/post_stage

.venv/bin/python -u scripts/post_stage/e05_stability.py \
  --run seed42=results_post_stage_e01_fit_1b_layer13_k5${RUN_SUFFIX:-}@pca \
  --run seed43=results_post_stage_e05_seed43${RUN_SUFFIX:-}@pca \
  --run rand45=results_post_stage_e05_randinit45${RUN_SUFFIX:-}@random \
  --run rand46=results_post_stage_e05_randinit46${RUN_SUFFIX:-}@random \
  --out-dir results_post_stage_e01_fit_1b_layer13_k5${RUN_SUFFIX:-} \
  --mails-tsv-path local_data/emails/Mails.tsv \
  --augmented-jsonl-path local_data/emails/augmented_mails.jsonl \
  --split-assignments configs/post_stage/split_assignments.json \
  --d-extra 1024 \
  --out e05_stability_random_init.json
