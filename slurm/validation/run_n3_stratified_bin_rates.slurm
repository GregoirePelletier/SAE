#!/bin/bash
#SBATCH --job-name=n3_bin_rates
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/validation/n3_stratified_bin_rates_%j.log

SAE_ROOT="${SAE_ROOT:-/home/h21486/SAE}"

export PYTHONUNBUFFERED=1

# CPU-only, aucun GPU : recalcule les métadonnées de bin (fréquence, taille de
# strate) des 150 features déjà sélectionnées ET jugées dans
# results_v10_emails_main/cache/b2_stratified_selection_rejudge.json (§79) --
# aucun juge relancé, aucune extraction. N3, AUDIT_SAE_2026-08.md §8.
export SAVE_DIR="./results_v10_emails_main/"
export LOCAL_MAILS_PATH="${SAE_ROOT}/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="${SAE_ROOT}/local_data/emails/augmented_mails.jsonl"

export PYTHONPATH="${SAE_ROOT}:${PYTHONPATH}"

cd "${SAE_ROOT}/"
.venv/bin/python scripts/n3_stratified_bin_rates.py
