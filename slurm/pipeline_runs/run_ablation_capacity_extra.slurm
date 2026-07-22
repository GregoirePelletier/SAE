#!/bin/bash
#SBATCH --job-name=SAE_abl_capacity
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=sae_abl_capacity_%j.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ── Ablation "capacité de l'extension" (D_EXTRA/K_EXTRA) ────────────────────────
# Piste explicitement listée comme non testée dans report/04_limites_et_perspectives.md
# ("Capacité architecturale de l'extension... non testée dans cette investigation").
# Après les ablations volume (§12/RESULTS_TESTS.md), largeur core (§17) et époques
# (run_ablation_epochs_only.slurm), teste le dernier levier simple restant : la
# capacité de l'extension ExtendedSAE elle-même (nombre de features apprises et
# parcimonie), à corpus/volume/largeur core IDENTIQUES au run principal
# (results_v10_emails_main). D_EXTRA 1024->2048 (x2), K_EXTRA 32->64 (x2, même
# ratio de parcimonie K/D=1/32 préservé). SAE_ID=16k explicite (défaut relevé à 65k
# après le run v12) pour rester comparable terme à terme au run principal.
export SEED=42
export SAVE_DIR="./results_v12_ablation_capacity_extra/"
export USE_FROZEN_CORE="1"
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="/home/h21486/SAE/local_data/emails/augmented_mails.jsonl"
export LOCAL_DATASET_PATH="/home/h21486/SAE/datasets/fineweb2_fra/data/fra_Latn/train/000_00000.parquet"
export EMB_MODEL="/home/h21486/SAE/models/F2LLM-v2-80M"

export MODEL_ID="/home/h21486/SAE/models/gemma-3-12b-it"
export MODEL_SIZE=12b
export DTYPE=bf16
export SAE_ID="layer_24_width_16k_l0_medium"
export PIPELINES=p1

export N_FEATURES_TO_LABEL=150
export EPOCHS_EXTRA=10
export D_EXTRA=2048
export K_EXTRA=64

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python src/sae/saev5.py
