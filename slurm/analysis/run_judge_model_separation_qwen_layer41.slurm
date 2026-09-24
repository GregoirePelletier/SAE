#!/bin/bash

#SBATCH --job-name=SAE_judge_sep_qwen_layer41
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/analysis/judge_model_separation_qwen_layer41_%j.log

SAE_ROOT="${SAE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

# Rejugement Qwen de la table §82/§88 (sweep taille de modèle/layer, K_EXTRA=5
# classique) sous la décision "juger tout avec Qwen" (docs/archive/audits/AUDIT_SAE_2026-08.md
# §9) -- variante layer41 du même patron que run_judge_model_separation_qwen.slurm
# (résultat de référence, results_v10_emails_main), réutilise les 150
# features déjà en cache dans ce SAVE_DIR, aucune extraction relancée.
export SAVE_DIR="./results_v33_ablation_classic_setup_k5_25m_layer41/"
export ALT_JUDGE_MODEL_ID="${SAE_ROOT}/models/Qwen3.8-27B"
export DTYPE=bf16
export LOCAL_MAILS_PATH="${SAE_ROOT}/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="${SAE_ROOT}/local_data/emails/augmented_mails.jsonl"

export PYTHONPATH="${SAE_ROOT}:${PYTHONPATH}"

cd "${SAE_ROOT}/"
.venv/bin/python scripts/judge_model_separation_test.py
