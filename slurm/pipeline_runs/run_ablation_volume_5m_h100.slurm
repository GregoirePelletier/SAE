#!/bin/bash
#SBATCH --job-name=SAE_abl_vol5m_h
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/pipeline_runs/sae_abl_vol5m_h100_%j.log

# Ablation volume ~5M tokens -- calibrage timing/ablations, réplique de
# run_ablation_volume_25m.slurm à 1/5 du budget (réservoir 5M*3840*2o≈38 Go,
# --mem=128G large marge ; filler proportionnel 540000*(5/25)≈108000, arrondi
# à 110000). Voir run_ablation_volume_25m.slurm pour le détail des choix
# (filler sans filtre thématique, LAYER=24 explicite requis pour matcher
# SAE_ID).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

export SEED=42
export SAVE_DIR="./results_v14c_ablation_volume5m_h100/"
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
export N_TOKENS_EXTRA_TRAIN=5000000
export N_VOLUME_FILLER_TARGET_CHUNKS=110000

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python src/sae/saev5.py
