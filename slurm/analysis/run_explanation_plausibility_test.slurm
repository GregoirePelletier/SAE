#!/bin/bash
#SBATCH --job-name=SAE_plausibility
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/analysis/explanation_plausibility_%j.log

SAE_ROOT="${SAE_ROOT:-/home/h21486/SAE}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="${SAE_ROOT}:${PYTHONPATH}"
export SAVE_DIR="./results_v10_emails_main/"
export MODEL_ID="${SAE_ROOT}/models/gemma-3-12b-it"
export MODEL_SIZE=12b
export N_DOCS_PLAUSIBILITY=60

cd "${SAE_ROOT}/"
.venv/bin/python scripts/explanation_plausibility_test.py
