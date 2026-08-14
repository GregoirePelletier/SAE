#!/bin/bash
#SBATCH --job-name=SAE_audit_scalefixj
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --output=logs/validation/audit_frozen_decoder_scalefix_rejudge_%j.log

# Audit 2026-08, A.3 point 3 (suite) : rejuge le checkpoint scale-fixe (job
# 43255) sur les memes 150 features que le run principal. Necessite une
# extraction complete du corpus train (~41k mails) car ce checkpoint a un
# encodeur different -- pas de reutilisation de cache possible ici.
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
.venv/bin/python scripts/audit_2026_08_frozen_decoder_scalefix_rejudge.py
