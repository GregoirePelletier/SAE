#!/bin/bash
#SBATCH --job-name=SAE_abl_vol25m_h
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/pipeline_runs/sae_abl_vol25m_h100_%j.log

# Ablation volume ~25M tokens (SAE Boost, Koriagin et al. COLM 2025) -- calibrage
# timing/ablations post-correctifs perf+data-science de cette session (G1/G2/G7,
# reprise, protocole juge). Réservoir de résidus en RAM hôte, 25M*3840*2o≈192 Go
# (--mem=500G, marge confortable). Filler FineWeb2-fr SANS filtre thématique
# (comportement actuel de sample_fineweb2_chunks, cf. AUDIT_SAE_2026-08.md) --
# ajouté uniquement au réservoir résiduel, jamais à train_texts (sélection de
# features et sonde de classification restent sur emails+augmentés seuls).
# LAYER=24 explicite : le défaut de src/config.py a été relevé à 31 depuis
# l'écriture de ce script, SAE_ID reste pinné à layer_24_width_16k_l0_medium
# (config de référence, docs/evaluation_protocol.md) -- sans LAYER explicite,
# les deux dériveraient silencieusement.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

export SEED=42
export SAVE_DIR="./results_v13c_ablation_volume25m_h100/"
export USE_FROZEN_CORE="1"
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="/home/h21486/SAE/local_data/emails/augmented_mails.jsonl"
export LOCAL_DATASET_PATH="/home/h21486/SAE/datasets/fineweb2_fra/data/fra_Latn/train/000_00000.parquet"
export VOLUME_FILLER_DATASET_GLOB="/home/h21486/SAE/datasets/fineweb2_fra/data/fra_Latn/train/000_0000*.parquet"
export EMB_MODEL="/home/h21486/SAE/models/F2LLM-v2-80M"

export MODEL_ID="/home/h21486/SAE/models/gemma-3-12b-it"
export MODEL_SIZE=12b
export DTYPE=bf16
export LAYER=24
export SAE_ID="layer_24_width_16k_l0_medium"
export PIPELINES=p1

export N_FEATURES_TO_LABEL=150
export EPOCHS_EXTRA=10
export D_EXTRA=1024
export K_EXTRA=32
export N_TOKENS_EXTRA_TRAIN=25000000
export N_VOLUME_FILLER_TARGET_CHUNKS=540000

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python src/sae/saev5.py
