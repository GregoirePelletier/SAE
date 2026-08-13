#!/bin/bash
#SBATCH --job-name=SAE_c2_rejudge
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=06:00:00
#SBATCH --output=logs/pipeline_runs/sae_c2_rejudge_%j.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ── Test le plus direct du constat CRITICAL C2 (REVIEW_ARS_PANEL.md,
# RESULTS_TESTS.md §44/§48/§49/§50) : re-juge les 150 features déjà jugées
# avec le MÊME juge (Gemma-3-12b-it) et le MÊME SAE d'extension déjà
# entraîné (aucun réentraînement, aucun nouveau modèle), en restreignant
# les exemples positifs présentés au juge aux mails ORIGINAUX uniquement
# (jamais de texte généré par Gemma). Faisabilité confirmée à 100%
# (150/150 features ont >=9 candidats positifs originaux, §50). GPU
# nécessaire uniquement pour recharger le juge et faire les passes de
# génération (pas d'extraction, pas d'entraînement) -- coût attendu très
# inférieur à un run complet.
export SAVE_DIR="./results_v10_emails_main/"
export MODEL_ID="/home/h21486/SAE/models/gemma-3-12b-it"
export DTYPE=bf16

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python scripts/c2_original_only_rejudge.py
