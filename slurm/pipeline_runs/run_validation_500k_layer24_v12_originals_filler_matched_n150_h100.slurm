#!/bin/bash
#SBATCH --job-name=SAE_val500k_v12h
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/pipeline_runs/sae_val500k_l24_v12_originals_filler_matched_n150_h_%j.log

SAE_ROOT="${SAE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

# B.1, réplication à pleine puissance statistique de v11 (job 44983/44985,
# RESULTS_TESTS.md §77) : à n=20, mails originaux seuls + filler FineWeb2
# recalibré (N_VOLUME_FILLER_TARGET_CHUNKS=294, même valeur qu'en v11 -- ne
# dépend que de la taille du corpus, pas de N_FEATURES_TO_LABEL) donnent
# 95% (19/20) d'interprétabilité contre 40% (8/20) sur le corpus mixte
# (job 44982, même échelle n=20/100k), z=-3,71, p=0,0002 -- écart énorme mais
# n=20 est explicitement un signal, pas une réplication statistiquement
# puissante (cf. v11). Ce run reproduit exactement la même config de corpus
# à l'échelle n=150/N_TOKENS_EXTRA_TRAIN=500000 de la référence mixte
# (results_v10_emails_main, 45,3%/68/150) pour une comparaison directement
# comparable au chiffre déjà publié.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export SEED=42
export SAVE_DIR="./results_v26_validation_layer24_v12_originals_filler_matched_n150_h100/"
export USE_FROZEN_CORE="1"
export LOCAL_MAILS_PATH="${SAE_ROOT}/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH=""
export LOCAL_DATASET_PATH="${SAE_ROOT}/datasets/fineweb2_fra/data/fra_Latn/train/000_00000.parquet"
export VOLUME_FILLER_DATASET_GLOB="${SAE_ROOT}/datasets/fineweb2_fra/data/fra_Latn/train/000_0000*.parquet"

export MODEL_ID="${SAE_ROOT}/models/gemma-3-12b-it"
export MODEL_SIZE=12b
export DTYPE=bf16
export PIPELINES=p1

export LAYER=24
export SAE_ID="layer_24_width_16k_l0_medium"
export K_EXTRA=32

export N_FEATURES_TO_LABEL=150
export EPOCHS_EXTRA=10
export D_EXTRA=1024
export N_TOKENS_EXTRA_TRAIN=500000
export N_VOLUME_FILLER_TARGET_CHUNKS=294

export PYTHONPATH="${SAE_ROOT}:${PYTHONPATH}"

cd "${SAE_ROOT}/"
.venv/bin/python src/sae/saev5.py
