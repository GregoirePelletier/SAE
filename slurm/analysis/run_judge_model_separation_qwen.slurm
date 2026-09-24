#!/bin/bash

#SBATCH --job-name=SAE_judge_sep_qwen
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/analysis/judge_model_separation_qwen_%j.log

SAE_ROOT="${SAE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

# Juge famille DIFFÉRENTE (Qwen3.8-27B bf16, capacité comparable au palier
# gemma-3-27b-it -- la variante FP8 ne génère avec aucun de ses deux chemins
# d'inférence sous torch==2.6.0, cf. src/sae/judge.py::load_judge_model) vs
# juge=extracteur historique (gemma-3-12b-it, cache
# p1_judge_labels_extended.json) sur les 150 features déjà jugées -- isole
# enfin le biais d'auto-préférence de l'effet de capacité que §43/§63/§65
# (RESULTS_TESTS.md) ne pouvaient pas distinguer (comparaison intra-famille
# gemma-3-4b-it seulement).
export SAVE_DIR="./results_v10_emails_main/"
export ALT_JUDGE_MODEL_ID="${SAE_ROOT}/models/Qwen3.8-27B"
export DTYPE=bf16
export LOCAL_MAILS_PATH="${SAE_ROOT}/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="${SAE_ROOT}/local_data/emails/augmented_mails.jsonl"

export PYTHONPATH="${SAE_ROOT}:${PYTHONPATH}"

cd "${SAE_ROOT}/"
.venv/bin/python scripts/judge_model_separation_test.py
