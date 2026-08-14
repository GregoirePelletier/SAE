#!/bin/bash
#SBATCH --job-name=SAE_audit_mcnemar
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/validation/audit_mcnemar_lengthbias_%j.log

# Audit 2026-08, A.2/F.2 (McNemar odd-one-out vs contrastif) + B.9 (biais de
# longueur). CPU-only, tout sur cache deja present, aucun GPU alloue.
export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"
export SAVE_DIR="./results_v10_emails_main/"
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="/home/h21486/SAE/local_data/emails/augmented_mails.jsonl"

cd /home/h21486/SAE/
.venv/bin/python scripts/audit_2026_08_mcnemar_and_lengthbias.py
