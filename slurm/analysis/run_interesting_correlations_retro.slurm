#!/bin/bash
#SBATCH --job-name=SAE_retro_corr
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=logs/analysis/retro_corr_%j.log

SAE_ROOT="${SAE_ROOT:-/home/h21486/SAE}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="${SAE_ROOT}:${PYTHONPATH}"

cd "${SAE_ROOT}/"
.venv/bin/python scripts/compute_interesting_correlations_retro.py results_v10_emails_main
