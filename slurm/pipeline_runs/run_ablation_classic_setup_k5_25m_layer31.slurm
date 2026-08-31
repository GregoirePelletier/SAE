#!/bin/bash
#SBATCH --job-name=SAE_abl_classic_k5
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/pipeline_runs/sae_abl_classic_k5_25m_layer31_%j.log

# Ablation "setup classique du papier" (SAE Boost, Koriagin et al. COLM 2025) :
# K_EXTRA=5 (leur k optimal, le dépôt utilise K_EXTRA=32 par défaut depuis
# l'ablation §25 -- direction cohérente avec k=5 mais jamais tranchée à pleine
# puissance), D_EXTRA=1024 (déjà le défaut du dépôt). Layer 31 (§51, seul écart
# individuel significatif du balayage de couche, non encore répliqué) plutôt
# que layer 24 (défaut historique, retenu uniquement pour la couverture
# Neuronpedia). Volume 25M tokens (calibrage timing/mémoire déjà établi,
# job 44994 : 382 Go RSS réel sur --mem=500G, mêmes réglages de corpus/filler
# ici). Remplace l'ancienne ablation K_EXTRA=5 (results_v13_ablation_k_extra5*,
# archivée -- antérieure au correctif juge B.3/B.5/B.6/B.11, docs/archived_runs_manifest.md)
# avec le code et le protocole actuels, à n=150 (pleine puissance).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

export SEED=42
export SAVE_DIR="./results_v27_ablation_classic_setup_k5_25m_layer31/"
export USE_FROZEN_CORE="1"
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="/home/h21486/SAE/local_data/emails/augmented_mails.jsonl"
export LOCAL_DATASET_PATH="/home/h21486/SAE/datasets/fineweb2_fra/data/fra_Latn/train/000_00000.parquet"
export VOLUME_FILLER_DATASET_GLOB="/home/h21486/SAE/datasets/fineweb2_fra/data/fra_Latn/train/000_0000*.parquet"
export EMB_MODEL="/home/h21486/SAE/models/F2LLM-v2-80M"

export MODEL_ID="/home/h21486/SAE/models/gemma-3-12b-it"
export MODEL_SIZE=12b
export DTYPE=bf16
export LAYER=31
export SAE_ID="layer_31_width_16k_l0_medium"
export PIPELINES=p1

export N_FEATURES_TO_LABEL=300
export EPOCHS_EXTRA=10
export D_EXTRA=1024
export K_EXTRA=5
export N_TOKENS_EXTRA_TRAIN=25000000
export N_VOLUME_FILLER_TARGET_CHUNKS=540000

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python src/sae/saev5.py
