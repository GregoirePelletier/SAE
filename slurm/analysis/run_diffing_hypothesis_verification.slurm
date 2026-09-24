#!/bin/bash

#SBATCH --job-name=SAE_diff_verif
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/analysis/diffing_hypothesis_verification_%j.log

SAE_ROOT="${SAE_ROOT:-/home/h21486/SAE}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

# App K.1 (docs/archive/references/PDF_APPENDICES_EXTRACT.md lignes 340, 740-775) : vérifie les
# hypothèses de diffing SAE déjà en cache (p1_diff_energy_sports.csv) contre
# un corpus energy/sports frais -- première mesure verification_rate/coverage
# de ce dépôt (rien n'était comparable au papier pour le diffing avant, cf.
# AUDIT_SAE_2026-08.md §7). h100 obligatoire (Qwen3.8-27B-FP8 OOM sur les
# a100 40 Go de ce cluster, cf. job 45444).
export SAVE_DIR="./results_v10_emails_main/"
export LOCAL_DATASET_PATH="${SAE_ROOT}/datasets/fineweb2_fra/data/fra_Latn/train/000_00000.parquet"
export N_DOCS_PER_DOMAIN=40
export TOP_K_HYPOTHESES=10

export PYTHONPATH="${SAE_ROOT}:${PYTHONPATH}"

cd "${SAE_ROOT}/"
.venv/bin/python scripts/diffing_hypothesis_verification_test.py
