#!/bin/bash

#SBATCH --job-name=SAE_npmi_diag
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/analysis/npmi_similarity_diagnostic_%j.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

# Diagnostic ponctuel : npmi_verified_test.py trouve 0 paires (NPMI>0.3,
# sim<0.2) -- affiche la distribution complète pour calibrer le seuil sans
# recharger le juge (bge-m3 seul, léger, a100 suffit largement).
export SAVE_DIR="./results_v10_emails_main/"

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python scripts/npmi_similarity_diagnostic.py
