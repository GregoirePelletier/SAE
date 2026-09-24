#!/bin/bash

#SBATCH --job-name=SAE_judge_sep
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/analysis/judge_model_separation_%j.log

SAE_ROOT="${SAE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Juge alternatif (gemma-3-4b-it, déjà en cache) vs juge=extracteur actuel
# (gemma-3-12b-it) sur les 150 features déjà jugées -- mesure le risque de
# biais d'auto-préférence, jamais quantifié jusqu'ici.
export SAVE_DIR="./results_v10_emails_main/"
export ALT_JUDGE_MODEL_ID="${SAE_ROOT}/models/gemma-3-4b-it"
export DTYPE=bf16
export LOCAL_MAILS_PATH="${SAE_ROOT}/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="${SAE_ROOT}/local_data/emails/augmented_mails.jsonl"

export PYTHONPATH="${SAE_ROOT}:${PYTHONPATH}"

cd "${SAE_ROOT}/"
.venv/bin/python scripts/judge_model_separation_test.py
