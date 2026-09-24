#!/bin/bash
#SBATCH --job-name=email_stats_cached
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/validation/email_stats_from_cache_%j.log

SAE_ROOT="${SAE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

export PYTHONUNBUFFERED=1

# CPU-only : recalcule les stats du job 44413 depuis le pickle déjà encodé,
# avec le correctif filter_na_rows() (cf. scripts/email_interp_embed_stats_from_cache.py).
# Aucun GPU nécessaire -- Dataset.load_from_file(resume=False) ne relance pas
# l'encodage SAE.

cd "${SAE_ROOT}/"
${SAE_ROOT}/external/interp_embed/.venv/bin/python \
    ${SAE_ROOT}/scripts/email_interp_embed_stats_from_cache.py
