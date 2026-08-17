#!/bin/bash
#SBATCH --job-name=SAE_b26_fidel
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=logs/validation/audit_b26_propagate_fidelity_%j.log

# Audit 2026-08, B.26 point 4 : propage le correctif V2 de INTENT_KEYWORDS_FR
# (deja applique a intent_urgency_probe.py, RESULTS_TESTS.md paragraphe 60)
# aux 3 derniers consommateurs : explanation_fidelity_test.py,
# steering_fidelity_test.py (CPU/GPU leger, activations deja en cache) et
# latent_retrieval_precision_eval.py (GPU leger, entraine un petit
# PhraseLevelSAE dedie -- quelques minutes selon son propre docstring).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"
export SAVE_DIR="./results_v10_emails_main/"
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="/home/h21486/SAE/local_data/emails/augmented_mails.jsonl"
export EMB_MODEL="/home/h21486/SAE/models/F2LLM-v2-80M"

cd /home/h21486/SAE/
.venv/bin/python scripts/audit_2026_08_b26_propagate_fidelity.py
