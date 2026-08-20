#!/bin/bash
#SBATCH --job-name=SAE_retrieval_eval
#SBATCH --partition=h100-bis
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
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
# l'entraînement du SAE : ~33M x hidden_size(F2LLM-330M=896) x 2o ≈ 59 Go.
# Deux OOM successifs en production, deux causes distinctes, toutes deux
# corrigées dans latent_terms.py (détail dans les docstrings des fonctions
# concernées) :
#  1. job 44434, --mem=96G, RSS 61,9 Go au kill : le corpus générique était lu
#     via sample_fineweb2_chunks() (src/data/preparation.py), qui charge le
#     shard FineWeb2-fr ENTIER en RAM (streaming=False) avant tout filtrage.
#     Corrigé : `_stream_generic_texts` lit désormais ce shard en streaming
#     (`load_dataset(..., streaming=True)`), jamais matérialisé en RAM.
#  2. job 44438, --mem=110G, RSS 68,9 Go au kill (streaming actif) :
#     `build_token_training_pool` accumulait les tokens dans une liste Python
#     puis `torch.cat(pool)` -- la liste ET le tenseur concaténé coexistaient
#     en mémoire au pic (~2x les 59 Go du pool). Corrigé : écriture directe
#     dans un buffer préalloué (~1x).
# --mem=110G restaure une marge raisonnable sur la seule estimation du pool
# de tokens (59 Go) + overhead modèle/tokenizer/OS, sans le double comptage
# ni le coût du shard complet.
export SAVE_DIR="./results_v10_emails_main/"
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export EMB_MODEL="/home/h21486/SAE/models/F2LLM-v2-330M"
export LOCAL_DATASET_PATH="/home/h21486/SAE/datasets/fineweb2_fra/data/fra_Latn/train/000_00000.parquet"

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python scripts/latent_retrieval_precision_eval.py
