#!/bin/bash
#SBATCH --job-name=SAE_audit_scalefix
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/validation/audit_frozen_decoder_scale_fix_%j.log

# Audit 2026-08, A.3 point 3 : retrain FrozenDecoderExtendedSAE avec
# input_scale calibre (decodeur reste aleatoire fige). Reutilise le
# reservoir de residus deja en cache, aucun forward LM.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"
export SAVE_DIR="./results_v10_emails_main/"

cd /home/h21486/SAE/
.venv/bin/python scripts/audit_2026_08_frozen_decoder_scale_fix.py
