#!/bin/bash
#SBATCH --job-name=SAE_batchsweep
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/validation/audit_extraction_batch_sweep_%j.log

# Etape 1 du plan (sharded-twirling-sunbeam.md) : balayage EXTRACTION_BATCH_SIZE,
# en parallele du job 44211 (h100, tenu inchange). Sur a100 (40 Go VRAM, 4 GPU
# libres au moment de la soumission -- h100/h100-bis saturees par d'autres
# utilisateurs) pour une lecture propre, non contendue.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"
export MODEL_ID="/home/h21486/SAE/models/gemma-3-12b-it"
export LAYER=24
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="/home/h21486/SAE/local_data/emails/augmented_mails.jsonl"

cd /home/h21486/SAE/
.venv/bin/python scripts/audit_2026_08_extraction_batch_size_sweep.py
