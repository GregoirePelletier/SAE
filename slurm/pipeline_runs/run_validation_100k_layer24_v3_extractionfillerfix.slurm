#!/bin/bash
#SBATCH --job-name=SAE_val100k_v3
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/pipeline_runs/sae_val100k_l24_v3_extractionfillerfix_%j.log
# --signal=B:USR1@600 : cf. run_validation_25m_layer24.slurm (R1, AUDIT_SAE_2026-08.md §2.3/§4.3).

# Troisième run d'une comparaison contrôlée à trois : mêmes paramètres que
# job 44560 (v1, aucun correctif filler) et job 44571 (v2, filler exclu du
# RÉ-ENCODAGE seulement) -- ici v3 ajoute l'allègement de l'EXTRACTION côté
# filler (is_filler_document, src/data/preparation.py, AUDIT_SAE_2026-08.md
# §2.2, G7) : ni encodage core ni écriture de fragment pour un document
# filler, seul son résidu brut alimente le réservoir. Comparer v1 vs v2 vs v3
# isole la contribution de chacun des deux correctifs filler séparément.
# SAVE_DIR séparé des deux runs précédents -- pas de partage de cache.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

export SEED=42
export SAVE_DIR="./results_v17_validation_100k_layer24_v3_extractionfillerfix/"
export USE_FROZEN_CORE="1"
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="/home/h21486/SAE/local_data/emails/augmented_mails.jsonl"
export LOCAL_DATASET_PATH="/home/h21486/SAE/datasets/fineweb2_fra/data/fra_Latn/train/000_00000.parquet"
export VOLUME_FILLER_DATASET_GLOB="/home/h21486/SAE/datasets/fineweb2_fra/data/fra_Latn/train/000_0000*.parquet"

export MODEL_ID="/home/h21486/SAE/models/gemma-3-12b-it"
export MODEL_SIZE=12b
export DTYPE=bf16
export PIPELINES=p1

export LAYER=24
export SAE_ID="layer_24_width_16k_l0_medium"
export K_EXTRA=32

export N_FEATURES_TO_LABEL=20
export EPOCHS_EXTRA=10
export D_EXTRA=1024
export N_TOKENS_EXTRA_TRAIN=100000
export N_VOLUME_FILLER_TARGET_CHUNKS=3000

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python src/sae/saev5.py
