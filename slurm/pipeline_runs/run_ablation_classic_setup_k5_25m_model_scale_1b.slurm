#!/bin/bash
#SBATCH --job-name=SAE_abl_classic_k5_1b
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/pipeline_runs/sae_abl_classic_k5_25m_1b_%j.log

# Ablation taille du modèle extracteur/juge (1B vs 4B vs 12B référence) --
# remplace results_v13_ablation_model_scale_1b (archivée, docs/archived_runs_manifest.md,
# antérieure au correctif juge B.3/B.5/B.6/B.11). Setup classique du papier
# (K_EXTRA=5, D_EXTRA=1024), layer_13 = ~2/3 profondeur pour gemma-3-1b-it
# (src/config.py _PRESETS), 25M tokens (calibrage timing/mémoire déjà établi
# à cette échelle sur 12B, job 44994 : 382 Go RSS -- d_model=1152 pour 1B
# (vs 3840 pour 12B, ratio ~0,30) laisse une marge large sur --mem=200G).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

export SEED=42
export SAVE_DIR="./results_v29_ablation_classic_setup_k5_25m_model_scale_1b/"
export USE_FROZEN_CORE="1"
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="/home/h21486/SAE/local_data/emails/augmented_mails.jsonl"
export LOCAL_DATASET_PATH="/home/h21486/SAE/datasets/fineweb2_fra/data/fra_Latn/train/000_00000.parquet"
export VOLUME_FILLER_DATASET_GLOB="/home/h21486/SAE/datasets/fineweb2_fra/data/fra_Latn/train/000_0000*.parquet"
export EMB_MODEL="/home/h21486/SAE/models/F2LLM-v2-80M"

export MODEL_SIZE=1b
export MODEL_ID="/home/h21486/SAE/models/gemma-3-1b-it"
export DTYPE=bf16
export LAYER=13
export SAE_ID="layer_13_width_16k_l0_medium"
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
