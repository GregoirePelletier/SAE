#!/bin/bash
#SBATCH --job-name=SAE_c2_seed99
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=03:00:00
#SBATCH --output=logs/validation/audit_c2_rejudge_seed99_%j.log

SAE_ROOT="${SAE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

# Audit 2026-08 round 2, B.17 : c2_original_only_rejudge.py, seed=99.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export SEED=99
export SAVE_DIR="./results_v10_emails_main/"
export MODEL_ID="${SAE_ROOT}/models/gemma-3-12b-it"
export DTYPE=bf16
export PYTHONPATH="${SAE_ROOT}:${PYTHONPATH}"

cd "${SAE_ROOT}/"
.venv/bin/python scripts/c2_original_only_rejudge.py
