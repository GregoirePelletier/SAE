#!/bin/bash
#SBATCH --job-name=SAE_audit_dcev2
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=00:45:00
#SBATCH --output=logs/validation/audit_delta_ce_v2_%j.log

# Audit 2026-08 round 3 : delta_ce par document + test de Wilcoxon apparie +
# sanity check SAE identite (mecanisme de patch valide independamment).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"
export MODEL_ID="/home/h21486/SAE/models/gemma-3-12b-it"
export DTYPE=bf16
export SAVE_DIR="./results_v10_emails_main/"
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="/home/h21486/SAE/local_data/emails/augmented_mails.jsonl"

cd /home/h21486/SAE/
.venv/bin/python scripts/audit_2026_08_delta_ce_v2.py
