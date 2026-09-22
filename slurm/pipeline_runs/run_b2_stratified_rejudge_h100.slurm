#!/bin/bash
#SBATCH --job-name=SAE_b2_rejudge_h
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=06:00:00
#SBATCH --output=logs/pipeline_runs/sae_b2_rejudge_h_%j.log

SAE_ROOT="${SAE_ROOT:-/home/h21486/SAE}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ── B2 (AUDIT_SAE_2026-08.md §5) : re-juge 150 features sélectionnées par
# fréquence (feature_selection_stratified_by_frequency, App. J interp-embed)
# plutôt que par magnitude token-level moyenne (défaut, référence 45,3%
# 68/150) -- même SAE d'extension déjà entraîné, même juge, aucun
# réentraînement ni nouvelle extraction. Décide si `FEATURE_SELECTION_METHOD
# =stratified` (judge.py) devient le défaut.
export SAVE_DIR="./results_v10_emails_main/"
export MODEL_ID="${SAE_ROOT}/models/gemma-3-12b-it"
export DTYPE=bf16
export LOCAL_MAILS_PATH="${SAE_ROOT}/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="${SAE_ROOT}/local_data/emails/augmented_mails.jsonl"
# N_FEATURES_TO_LABEL défaut = 10 (src/config.py) : sans cet export, le premier
# essai (jobs 44997/44998) a comparé 9/10 vs 68/150 -- pas comparable, refait.
export N_FEATURES_TO_LABEL=150

export PYTHONPATH="${SAE_ROOT}:${PYTHONPATH}"

cd "${SAE_ROOT}/"
.venv/bin/python scripts/b2_stratified_selection_rejudge.py
