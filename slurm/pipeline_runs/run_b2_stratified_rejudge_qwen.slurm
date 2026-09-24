#!/bin/bash
#SBATCH --job-name=SAE_b2_rejudge_qwen
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/pipeline_runs/sae_b2_rejudge_qwen_%j.log

SAE_ROOT="${SAE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

# Variante Qwen de run_b2_stratified_rejudge.slurm (référence, gemma-3-12b-it,
# 22 août -- fichier sans bin_info ni déduplication par mail parent, tous
# deux ajoutés depuis). `MODEL_ID` (pas JUDGE_MODEL_ID) contrôle le juge dans
# b2_stratified_selection_rejudge.py -- override intentionnel vers Qwen3.8-27B
# plutôt que le gemma-3-12b-it historique (le script isolait à l'origine la
# méthode de sélection du choix du juge, B.2 ; sous la politique "juger tout
# avec Qwen", ce rerun donne directement le chiffre le plus complet : bin_info
# natif (taux par bin, N3/§87) + déduplication par mail parent (correctif
# session en cours) + juge Qwen, en un seul run plutôt que trois rejugements
# séparés). h100 obligatoire (Qwen3.8-27B bf16 ~52 Go, a100 OOM -- docs/ops.md).
# --mem=96G comme les autres jobs juge Qwen (poids bf16 plus gros que Gemma).
export SAVE_DIR="./results_v10_emails_main/"
export MODEL_ID="${SAE_ROOT}/models/Qwen3.8-27B"
export DTYPE=bf16
export LOCAL_MAILS_PATH="${SAE_ROOT}/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="${SAE_ROOT}/local_data/emails/augmented_mails.jsonl"
export N_FEATURES_TO_LABEL=150

export PYTHONPATH="${SAE_ROOT}:${PYTHONPATH}"

cd "${SAE_ROOT}/"
.venv/bin/python scripts/b2_stratified_selection_rejudge.py
