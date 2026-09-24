#!/bin/bash
#SBATCH --job-name=SAE_intent_r0_tfidf
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=logs/analysis/intent_urgency_r0_tfidf_%j.log

SAE_ROOT="${SAE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SAE_ROOT}:${PYTHONPATH}"

# D1 (docs/archive/audits/AUDIT_SAE_2026-08.md) : sondes d'intention + baseline TF-IDF+LogReg sur
# les codes de R0 (results_v27_ablation_classic_setup_k5_25m_layer31, référence
# unique du rapport sous méthodologie finale). Zéro calcul GPU réel (réutilise
# p1_all_doc_acts_ext_d1024.pt déjà en cache).
# --mem=128G (pas 32G) : `load_all_doc_acts` (sae_shared.py) charge le fichier
# ENTIER via `torch.load` avant de sous-échantillonner les lignes originales --
# p1_all_doc_acts_ext_d1024.pt pèse ~40,7 Go sur disque, --mem=32G a fait
# tourner ce job en OOM (job 45955).
export SAVE_DIR="./results_v27_ablation_classic_setup_k5_25m_layer31/"
export LOCAL_MAILS_PATH="${SAE_ROOT}/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="${SAE_ROOT}/local_data/emails/augmented_mails.jsonl"

cd "${SAE_ROOT}/"
.venv/bin/python scripts/intent_urgency_probe.py
