#!/bin/bash
#SBATCH --job-name=SAE_abl_m1_mag_gemma
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/pipeline_runs/sae_abl_m1_mag_gemma_%j.log

SAE_ROOT="${SAE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

# M1 : dernier point manquant de la grille méthode de sélection x juge à
# une configuration CRÉDIBLE (25M tokens, K_EXTRA=5, layer 31, 12B) --
# jusqu'ici seuls stratifié+Gemma (§82, 82,0%) et stratifié+Qwen (R0, §97,
# 84,7%) existaient à ce volume. Magnitude+Qwen n'a jamais été mesuré à
# cette config non plus mais magnitude+Gemma suffit à documenter l'effet
# de sélection à volume crédible sans dépendre du run 500k tokens
# (results_v10_emails_main, sous le plancher de 25M retenu pour ce rapport).
# Copie exacte de R0 (results_v27_.../layer31), DEUX variables :
# FEATURE_SELECTION_METHOD=magnitude (défaut : stratified) et
# JUDGE_MODEL_ID=gemma-3-12b-it (défaut : Qwen3.8-27B) -- juge auto-référent
# volontaire ici, pour reproduire exactement la config du point historique
# 45,3%/82,0% (magnitude/stratifié, tous deux jugés par le modèle
# extracteur), PAS une régression sur la règle CLAUDE.md (juge jamais
# rechargé depuis MODEL_ID par défaut) : ablation ponctuelle documentée,
# pas un changement de comportement par défaut.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

export SEED=42
export SAVE_DIR="./results_v48_ablation_m1_magnitude_gemma_classic_setup_25m_layer31/"
export USE_FROZEN_CORE="1"
export FEATURE_SELECTION_METHOD=magnitude
export JUDGE_MODEL_ID="${SAE_ROOT}/models/gemma-3-12b-it"
export LOCAL_MAILS_PATH="${SAE_ROOT}/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="${SAE_ROOT}/local_data/emails/augmented_mails.jsonl"
export LOCAL_DATASET_PATH="${SAE_ROOT}/datasets/fineweb2_fra/data/fra_Latn/train/000_00000.parquet"
export VOLUME_FILLER_DATASET_GLOB="${SAE_ROOT}/datasets/fineweb2_fra/data/fra_Latn/train/000_0000*.parquet"
export EMB_MODEL="${SAE_ROOT}/models/F2LLM-v2-80M"

export MODEL_ID="${SAE_ROOT}/models/gemma-3-12b-it"
export MODEL_SIZE=12b
export DTYPE=bf16
export LAYER=31
export SAE_ID="layer_31_width_16k_l0_medium"
export PIPELINES=p1

export N_FEATURES_TO_LABEL=150
export EPOCHS_EXTRA=10
export D_EXTRA=1024
export K_EXTRA=5
export N_TOKENS_EXTRA_TRAIN=25000000
export N_VOLUME_FILLER_TARGET_CHUNKS=540000

export PYTHONPATH="${SAE_ROOT}:${PYTHONPATH}"

cd "${SAE_ROOT}/"
.venv/bin/python src/sae/saev5.py
