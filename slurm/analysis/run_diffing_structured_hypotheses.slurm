#!/bin/bash

#SBATCH --job-name=SAE_diff_struct
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/analysis/diffing_structured_hypotheses_%j.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

# App D.2 (docs/PDF_APPENDICES_EXTRACT.md lignes 212-266) : génère des
# hypothèses de diffing STRUCTURÉES (JSON, prompt verbatim) à partir des
# features SAE les plus discriminantes (sélection par différence de
# FRÉQUENCE, top 200/seuil 0.03 -- distinct du tri par log-odds-ratio),
# puis les vérifie (App K.1) sur un corpus energy/sports frais -- comble le
# dernier écart identifié entre §84 (verification_rate/coverage sur des
# labels de features bruts) et le protocole App D.2 complet (génération
# structurée, pas seulement vérification).
export SAVE_DIR="./results_v10_emails_main/"
export LOCAL_DATASET_PATH="/home/h21486/SAE/datasets/fineweb2_fra/data/fra_Latn/train/000_00000.parquet"
export N_DOCS_PER_DOMAIN=40
export NUM_HYPOTHESES=10
export FREQ_DIFF_THRESHOLD=0.03
export TOP_N_FEATURES=200

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python scripts/diffing_structured_hypotheses_test.py
