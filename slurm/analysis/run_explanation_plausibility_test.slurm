#!/bin/bash
#SBATCH --job-name=SAE_plausibility
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=explanation_plausibility_%j.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"
export SAVE_DIR="./results_v10_emails_main/"
export MODEL_ID="/home/h21486/SAE/models/gemma-3-12b-it"
export MODEL_SIZE=12b
export N_DOCS_PLAUSIBILITY=60

cd /home/h21486/SAE/
.venv/bin/python scripts/explanation_plausibility_test.py
