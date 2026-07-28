#!/bin/bash
#SBATCH --job-name=SAE_retrieval_eval
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/analysis/latent_retrieval_precision_eval_%j.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ── Évaluation quantitative de src/sae/retrieval/latent_terms.py (Latent Terms,
# Clavié et al. 2026, arXiv:2605.29384) ────────────────────────────────────────
# Jusqu'ici exercé uniquement via scripts/retrieval_demo.py (1-2 requêtes
# inspectées à l'œil, sur données de substitution -- écrit sur une machine sans
# Mails.tsv) et le dashboard (parcours interactif, pas de métrique). Ce job
# construit un vrai protocole précision/rappel sur les mails originaux réels,
# avec les labels faibles d'intention comme vérité terrain de substitution, et
# compare à une baseline TF-IDF classique. Détail du protocole : docstring de
# scripts/latent_retrieval_precision_eval.py.
# Coût léger : F2LLM-v2-80M sur ~3474 mails + entraînement d'un petit
# PhraseLevelSAE dédié (dim320/d8192/k16, mêmes défauts que Pipeline 2) --
# PAS de Gemma-3-12B.
export SAVE_DIR="./results_v10_emails_main/"
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export EMB_MODEL="/home/h21486/SAE/models/F2LLM-v2-80M"

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python scripts/latent_retrieval_precision_eval.py
