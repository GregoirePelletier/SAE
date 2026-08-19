#!/bin/bash
#SBATCH --job-name=SAE_retrieval_eval
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/analysis/latent_retrieval_precision_eval_%j.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

# ── Évaluation quantitative de src/sae/retrieval/latent_terms.py (Latent Terms,
# Clavié et al. 2026, arXiv:2605.29384, réimplémentation token-level) ──────────
# Remplace le run précédent (SAE phrase-level entraîné directement sur
# Mails.tsv, RESULTS_TESTS.md §26/§68/§69) : le SAE s'entraîne maintenant par
# pure reconstruction sur des activations TOKEN d'un corpus FineWeb2-fr
# GÉNÉRIQUE hors-domaine (§3.1 du papier), et l'index BM25 est construit sur
# les mails entiers (plus de découpage en phrases). Détail du protocole et des
# écarts d'échelle assumés (33M tokens uniques x3 époques, pas 30B) :
# docstring de src/sae/retrieval/latent_terms.py.
# Coût dominant : extraction F2LLM token-level sur le pool d'entraînement
# générique (~33M tokens, LT_TRAIN_TOKENS) + indexation des 3480 mails --
# PAS de Gemma-3-12B. Pool d'entraînement matérialisé en RAM (bf16) avant
# l'entraînement du SAE : ~33M x hidden_size(F2LLM-330M=896) x 2o ≈ 59 Go,
# --mem=96G laisse une marge pour l'overhead modèle/tokenizer/OS.
export SAVE_DIR="./results_v10_emails_main/"
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export EMB_MODEL="/home/h21486/SAE/models/F2LLM-v2-330M"
export LOCAL_DATASET_PATH="/home/h21486/SAE/datasets/fineweb2_fra/data/fra_Latn/train/000_00000.parquet"

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python scripts/latent_retrieval_precision_eval.py
