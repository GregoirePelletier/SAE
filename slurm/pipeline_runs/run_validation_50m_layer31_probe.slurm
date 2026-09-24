#!/bin/bash
#SBATCH --job-name=SAE_probe_50m_l31
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=900G
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/pipeline_runs/sae_probe_50m_layer31_%j.log

SAE_ROOT="${SAE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

# Point de mesure UNIQUE (pas un sweep) : 50M tokens, 12B/layer31/K_EXTRA=5,
# pour obtenir un chiffre RAM réel avant de décider d'un sweep 50M/100M
# complet -- demande explicite de vérifier empiriquement plutôt que de faire
# confiance à une extrapolation. --mem=900G, recalibré sur un point de mesure
# RÉEL de cette session (pas l'ancien commentaire "memmap fix, 96G suffit",
# infirmé) : job 45803 (layer12, même famille K_EXTRA=5/25M/12B) a
# effectivement consommé 358,5 Go RSS (`sacct --format=MaxRSS`, pas
# seulement demandé) -- extrapolation LINÉAIRE à 50M (d_model fixe) ≈717 Go,
# 900G laisse ~25% de marge sans sur-réserver inutilement sur un nœud
# partagé à 2,02 To (cf. discussion : 1200G précédent jugé disproportionné).
# N_VOLUME_FILLER_TARGET_CHUNKS=1200000 : le corpus réel (~14,6M tokens
# estimés) ne couvre qu'une fraction de la cible, le filler comble le reste
# (~35M tokens) -- pool dimensionné avec marge (cf. heuristique de la
# tentative 100M abandonnée, run_validation_100M_layer31_h100.slurm,
# ~21 800 chunks/M tokens cible).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export SEED=42
export SAVE_DIR="./results_probe_50m_layer31/"
export USE_FROZEN_CORE="1"
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
export N_TOKENS_EXTRA_TRAIN=50000000
export N_VOLUME_FILLER_TARGET_CHUNKS=1200000

export PYTHONPATH="${SAE_ROOT}:${PYTHONPATH}"

cd "${SAE_ROOT}/"
.venv/bin/python src/sae/saev5.py
