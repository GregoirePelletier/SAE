#!/bin/bash
#SBATCH --job-name=SAE_baseline_full
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=logs/baseline_diffing/baseline_full_%j.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# cf. run_baseline.slurm (test) : fix OOM CUDA (logits_to_keep + no_grad réel dans
# activations.py) déjà validé sur job 39003 ; gardé en défense supplémentaire.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="/home/h21486/SAE/local_data/emails/augmented_mails.jsonl"
export MODEL_ID="/home/h21486/SAE/models/gemma-3-12b-it"
export LOCAL_SAE_DIR="/home/h21486/SAE/saes/gemma-scope-2-12b-it-res"
export SAVE_DIR="./results_v9_test/"
export CACHE_DIR="${SAVE_DIR}cache_baseline_full"

mkdir -p logs "${CACHE_DIR}"

cd /home/h21486/SAE/
# Corpus complet : emplacement canonique local_data/emails/ (déplacé hors de
# results_v9_test/ -- un jeu de données d'entrée ne doit pas vivre sous un dossier
# de résultats d'expérience, cf. réorganisation des données).
.venv/bin/python scripts/baseline_gemmascope.py "${LOCAL_MAILS_PATH}" "${LOCAL_AUGMENTED_MAILS_PATH}"
