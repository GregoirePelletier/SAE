#!/bin/bash
#SBATCH --job-name=SAE_audit_b26v2
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/validation/audit_b26_round2_fix_%j.log

# Audit 2026-08, B.26 round 2 : correctif regex reecrit a la main (chaque
# alternative du groupe OR, pas seulement la derniere) + rejugement complet.
# CPU-only, aucun GPU alloue.
export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="/home/h21486/SAE/local_data/emails/augmented_mails.jsonl"

cd /home/h21486/SAE/
.venv/bin/python scripts/audit_2026_08_b26_round2_fix.py
