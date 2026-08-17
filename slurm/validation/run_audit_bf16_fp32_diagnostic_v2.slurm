#!/bin/bash
#SBATCH --job-name=SAE_audit_bf16v2
#SBATCH --partition=h100-bis
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:45:00
#SBATCH --output=logs/validation/audit_bf16_fp32_diagnostic_v2_%j.log

# Audit 2026-08 round 3, B.21 : echantillon elargi (150 docs), 1er token de
# contenu INCLUS (cible les positions a activation massive), IC bootstrap.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"
export MODEL_ID="/home/h21486/SAE/models/gemma-3-12b-it"
export DTYPE=bf16
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="/home/h21486/SAE/local_data/emails/augmented_mails.jsonl"

cd /home/h21486/SAE/
.venv/bin/python scripts/audit_2026_08_bf16_fp32_diagnostic_v2.py
