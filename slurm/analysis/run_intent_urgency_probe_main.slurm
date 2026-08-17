#!/bin/bash
#SBATCH --job-name=SAE_intent_main
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/analysis/intent_urgency_main_%j.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

# Sanity check post-correctif B.26 : script INCHANGE, labels desormais corriges
# directement en production (src/data/dataset.py::INTENT_KEYWORDS_FR) -- doit
# reproduire les chiffres deja obtenus par monkey-patch (RESULTS_TESTS.md §60).
export SAVE_DIR="./results_v10_emails_main/"
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="/home/h21486/SAE/local_data/emails/augmented_mails.jsonl"

cd /home/h21486/SAE/
.venv/bin/python scripts/intent_urgency_probe.py
