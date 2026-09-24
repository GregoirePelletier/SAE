#!/bin/bash
#SBATCH --job-name=SAE_multiling_r0
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=logs/analysis/sae_multilingual_bias_r0_%j.log

SAE_ROOT="${SAE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

# Biais multilingue du juge sous R0 (RESULTS_TESTS.md §22 réutilisait le juge
# auto-référent MODEL_ID=Gemma-3-12b-it sur la sélection historique par
# magnitude -- corrigé ici : juge/traducteur Qwen3.8-27B découplé,
# doc_groups=train_groups, 300 features stratifiées de R0). Zéro
# réextraction -- réutilise les activations/fragments déjà en cache.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

export SAVE_DIR="./results_v27_ablation_classic_setup_k5_25m_layer31/"
export SEED=42

export PYTHONPATH="${SAE_ROOT}:${PYTHONPATH}"

cd "${SAE_ROOT}/"
.venv/bin/python scripts/multilingual_judge_bias_test_r0.py
