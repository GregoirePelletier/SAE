#!/bin/bash
#SBATCH --job-name=SAE_audit_bf16fp32
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH --output=logs/validation/audit_bf16_fp32_diagnostic_%j.log

# Audit 2026-08, B.21 : diagnostic annulation catastrophique bf16 sur le
# résidu de FrozenCoreResidualSAE. Charge Gemma-3-12b-it deux fois (bf16 puis
# fp32, ~48 Go de poids en fp32 seul) sur un petit échantillon de contrôle
# (40 mails) -- H100 80 Go requis pour la passe fp32, ne tiendrait pas sur
# un A100 40 Go.
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
.venv/bin/python scripts/audit_2026_08_bf16_fp32_diagnostic.py
