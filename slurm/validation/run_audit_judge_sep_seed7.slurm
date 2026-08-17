#!/bin/bash
#SBATCH --job-name=SAE_sep_seed7
#SBATCH --partition=h100-bis
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=03:00:00
#SBATCH --output=logs/validation/audit_judge_sep_seed7_%j.log

# Audit 2026-08 round 2, B.17 (suite) : judge_model_separation_test.py
# maintenant seede, rejoue avec seed=7 pour mesurer la variance inter-graine
# sur le resultat de dependance au juge (§43, reference 45,3% vs 24,7%).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export SEED=7
export SAVE_DIR="./results_v10_emails_main/"
export MODEL_ID="/home/h21486/SAE/models/gemma-3-12b-it"
export ALT_JUDGE_MODEL_ID="/home/h21486/SAE/models/gemma-3-4b-it"
export DTYPE=bf16
export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python scripts/judge_model_separation_test.py
