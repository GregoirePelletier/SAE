#!/bin/bash

#SBATCH --job-name=SAE_npmi_verif
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/analysis/npmi_verified_%j.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

# App E.1/E.3 (Figure 4) : NPMI_verified + CO sur les paires de plus fort NPMI
# brut parmi les 150 features d'extension déjà labellisées, après filtre
# syntaxique (K.2) et filtre trivial (même token) -- première mesure de ce
# dépôt comparable au papier pour les corrélations (AUDIT_SAE_2026-08.md §1/§7).
export SAVE_DIR="./results_v10_emails_main/"
export TOP_K_PAIRS=8
export N_VERIFY_DOCS=60

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python scripts/npmi_verified_test.py
