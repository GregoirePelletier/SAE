#!/bin/bash
#SBATCH --job-name=SAE_abl_width65k
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=sae_abl_width65k_%j.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ── Ablation "largeur SEULE" ────────────────────────────────────────────────────
# Le run v12 (results_v12_scaled_65k) fait varier SIMULTANÉMENT 3 leviers (largeur
# SAE core 16k->65k, EPOCHS_EXTRA 10->40, EPOCHS 30->100, N_FEATURES_TO_LABEL
# 150->600) par rapport au run principal (results_v10_emails_main) -- impossible
# d'attribuer l'amélioration observée (45,3%->53,7% d'interprétabilité) à un levier
# spécifique. Ce run isole la largeur SEULE : SAE_ID=65k, tout le reste IDENTIQUE
# au run principal (EPOCHS_EXTRA=10, EPOCHS=30, N_FEATURES_TO_LABEL=150 -- défauts
# de src/config.py). Comparé à run_ablation_epochs_only.slurm (l'autre facteur isolé)
# et au run v12 (les deux facteurs combinés), permet une décomposition additive de
# l'effet mesuré -- même démarche que l'ablation de volume de tokens déjà menée.
export SEED=42
export SAVE_DIR="./results_v12_ablation_width65k_only/"
export USE_FROZEN_CORE="1"
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="/home/h21486/SAE/local_data/emails/augmented_mails.jsonl"
export LOCAL_DATASET_PATH="/home/h21486/SAE/datasets/fineweb2_fra/data/fra_Latn/train/000_00000.parquet"
export EMB_MODEL="/home/h21486/SAE/models/F2LLM-v2-80M"

export MODEL_ID="/home/h21486/SAE/models/gemma-3-12b-it"
export MODEL_SIZE=12b
export DTYPE=bf16
export SAE_ID="layer_24_width_65k_l0_medium"
export PIPELINES=p1,p2

export N_FEATURES_TO_LABEL=150
export EPOCHS_EXTRA=10
export EPOCHS=30

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python src/sae/saev5.py
