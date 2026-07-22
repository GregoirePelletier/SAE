#!/bin/bash
#SBATCH --job-name=SAE_baseline_test
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=logs/baseline_test_%j.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# OOM CUDA constaté (job 38988/38999) après ~500 batches à longueurs variables sur A100
# 40GB : fragmentation de l'allocateur caching. Recommandé par PyTorch lui-même dans le
# message d'erreur ; en complément du fix logits_to_keep + purge périodique (cf. activations.py).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export MODEL_ID="/home/h21486/SAE/models/gemma-3-12b-it"
# Poids SAE réellement présents ici (local_data/saes/gemma-scope-2-12b-it est vide) :
export LOCAL_SAE_DIR="/home/h21486/SAE/saes/gemma-scope-2-12b-it-res"
# Doit matcher le SAVE_DIR de run_augmentation.slurm (source de augmented_mails.jsonl).
export SAVE_DIR="./results_v9_test/"
export CACHE_DIR="${SAVE_DIR}cache_baseline"

mkdir -p logs "${CACHE_DIR}"

cd /home/h21486/SAE/
# uv run tentait de re-résoudre l'env (réseau bloqué sur le cluster) -> échec
# systématique ; .venv/bin/python bypass ce problème, comme dans run_sae.slurm.
# main(mails_tsv, augmented_jsonl) exige 2 arguments positionnels (absents avant ce
# correctif -> IndexError immédiat) ; le second est produit par run_augmentation.slurm.
# Run de VALIDATION (60 mails sous-échantillonnés, archivé) -- le corpus complet est
# déjà généré (local_data/emails/augmented_mails.jsonl) et validé (run_baseline_full.slurm) ;
# ce script sert seulement de référence/smoke-test rapide, pas à relancer en prod.
.venv/bin/python scripts/baseline_gemmascope.py "${LOCAL_MAILS_PATH}" "/home/h21486/SAE/local_data/emails/archive/augmented_mails_test.jsonl"