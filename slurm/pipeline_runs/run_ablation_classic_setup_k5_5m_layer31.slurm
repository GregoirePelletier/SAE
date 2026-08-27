#!/bin/bash
#SBATCH --job-name=SAE_abl_classic_k5_5m
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/pipeline_runs/sae_abl_classic_k5_5m_layer31_%j.log

# Sweep volume de tokens, point 1/3 : 5M tokens (échantillon partiel du corpus
# réel, N_VOLUME_FILLER_TARGET_CHUNKS=0 -- le corpus réel (mails originaux +
# augmentés, ~14-15M tokens estimés par échantillonnage de caractères, pas de
# tokenisation réelle sur le nœud frontal) dépasse déjà ce budget, aucun
# filler nécessaire). Complète §12 (100k/2M/25M, sous magnitude/Gemma) sous
# méthodologie pleinement corrigée (stratifié/Qwen/déduplication) et isole la
# question "sous-échantillonner le corpus réel dégrade-t-il l'interprétabilité ?"
# de celle du point 25M ci-contre (qui mélange corpus réel ET filler
# générique FineWeb2-fr) -- comparé au point 15M (results_v36, corpus réel
# quasi complet, filler=0 aussi) et 25M (results_v27, corpus réel + ~10M
# tokens de filler). --mem=80G : mesuré empiriquement à 31,3 Go RSS pour 5M
# tokens sur le corpus complet filler+train (job 44991) -- marge large ici
# vu l'absence de filler.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

export SEED=42
export SAVE_DIR="./results_v35_ablation_classic_setup_k5_5m_layer31/"
export USE_FROZEN_CORE="1"
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="/home/h21486/SAE/local_data/emails/augmented_mails.jsonl"
export LOCAL_DATASET_PATH="/home/h21486/SAE/datasets/fineweb2_fra/data/fra_Latn/train/000_00000.parquet"
export EMB_MODEL="/home/h21486/SAE/models/F2LLM-v2-80M"

export MODEL_ID="/home/h21486/SAE/models/gemma-3-12b-it"
export MODEL_SIZE=12b
export DTYPE=bf16
export LAYER=31
export SAE_ID="layer_31_width_16k_l0_medium"
export PIPELINES=p1

export N_FEATURES_TO_LABEL=150
export EPOCHS_EXTRA=10
export D_EXTRA=1024
export K_EXTRA=5
export N_TOKENS_EXTRA_TRAIN=5000000
export N_VOLUME_FILLER_TARGET_CHUNKS=0

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python src/sae/saev5.py
