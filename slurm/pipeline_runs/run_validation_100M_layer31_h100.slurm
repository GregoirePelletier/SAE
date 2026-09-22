#!/bin/bash
#SBATCH --job-name=SAE_val100M_l31
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G

SAE_ROOT="${SAE_ROOT:-/home/h21486/SAE}"
# ATTENTION avant relance : --mem=180G est INSUFFISANT pour un réservoir de
# 100M tokens réellement rempli. Bug corrigé ici (LOCAL_AUGMENTED_MAILS_PATH
# vide + N_VOLUME_FILLER_TARGET_CHUNKS=294 -- hérité par erreur d'un template
# petit-corpus, job 45128) : sur ~4664 documents le réservoir ne s'est rempli
# qu'à 1,23M tokens (1,2% de la cible), --mem=180G n'a donc jamais été
# réellement testé à l'échelle 100M. Mesures réelles sur le corpus mixte complet
# (mêmes proportions filler/train) : 5M tokens -> 31,3 Go RSS (job 44991),
# 25M tokens -> 381,8 Go RSS (job 44994) -- croissance super-linéaire (12,2x
# la mémoire pour 5x les tokens), pas un simple facteur d'échelle constant.
# Extrapolation prudente à 100M : >1,5 To, potentiellement au-delà de la
# capacité totale d'un nœud dgx-h100 (2 To, cf. `sinfo -N -p h100 -o "%m"`) --
# fixer N_VOLUME_FILLER_TARGET_CHUNKS proportionnellement (~21 800/M tokens,
# soit ~2 180 000 pour 100M) ET revoir --mem à la hausse AVANT de relancer,
# pas seulement corriger le corpus.
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/pipeline_runs/sae_val100m_l31_%j.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export SEED=42
export SAVE_DIR="./results_validation_layer31_100M_h100/"
export USE_FROZEN_CORE="1"
export LOCAL_MAILS_PATH="${SAE_ROOT}/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="${SAE_ROOT}/local_data/emails/augmented_mails.jsonl"
export LOCAL_DATASET_PATH="${SAE_ROOT}/datasets/fineweb2_fra/data/fra_Latn/train/000_00000.parquet"
export VOLUME_FILLER_DATASET_GLOB="${SAE_ROOT}/datasets/fineweb2_fra/data/fra_Latn/train/000_0000*.parquet"

export MODEL_ID="${SAE_ROOT}/models/gemma-3-12b-it"
export MODEL_SIZE=12b
export DTYPE=bf16
export PIPELINES=p1

export LAYER=31
export SAE_ID="layer_31_width_16k_l0_medium"
export K_EXTRA=5    

export N_FEATURES_TO_LABEL=150
export EPOCHS_EXTRA=10
export D_EXTRA=1024
export N_TOKENS_EXTRA_TRAIN=100000000
export N_VOLUME_FILLER_TARGET_CHUNKS=2180000

export PYTHONPATH="${SAE_ROOT}:${PYTHONPATH}"

cd "${SAE_ROOT}/"
.venv/bin/python src/sae/saev5.py