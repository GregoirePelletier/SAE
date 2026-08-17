#!/bin/bash
#SBATCH --job-name=SAE_b24_compare
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/validation/audit_b24_compare_pipeline_%j.log

# Audit 2026-08, B.24 : premier essai du module src/sae/compare/ (jamais
# execute). Compare F2LLM-v2-80M (backbone Pipeline 2 en production) a
# bge-m3 (deja local, utilise ailleurs dans ce depot pour l'embedding de
# labels) via detection de pollution + alignement (model_compare.py).
# NOTE : n'est PAS une comparaison Pipeline 1 vs Pipeline 2 (les deux SAE
# sont entraines from-scratch ICI par pipeline.py, pas les checkpoints
# existants) -- une comparaison de deux backbones d'embedding sous protocole
# identique. Pooling mean pour les deux modeles (embed_corpus), alors que
# bge-m3 est normalement utilise en pooling CLS ailleurs dans ce depot --
# ecart methodologique a noter dans l'interpretation du resultat.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"
export COMPARE_PIPELINE_OUT="./results_audit_2026_08_b24_compare/"

cd /home/h21486/SAE/
.venv/bin/python -m src.sae.compare.pipeline \
    --mails /home/h21486/SAE/local_data/emails/Mails.tsv \
    --mode compare \
    --model-a /home/h21486/SAE/models/F2LLM-v2-80M \
    --model-b /home/h21486/SAE/models/bge-m3
