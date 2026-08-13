#!/bin/bash

#SBATCH --job-name=SAE_v10_emails
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=logs/pipeline_runs/sae_v10_emails_%j.log

# ── Isolation réseau ──
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ── Run à l'échelle complète, corpus emails+augmentés ────────────────────────────
# Diagnostic (cf. RESULTS_TESTS.md §12) : sur le run précédent (corpus generic),
# 0/10 features d'extension étaient "dead_feature" (donc pas un problème de volume
# brut) mais seulement 2/10 passaient le test odd-one-out (intruder detection) --
# cause racine identifiée dans saev5.py : le corpus d'entraînement du SAE
# d'extension (train_texts) était bâti UNIQUEMENT à partir de FineWeb-2/Wikipedia
# générique (energy/sports/support), les emails n'étant utilisés qu'en post-hoc
# après entraînement -- le SAE n'apprend donc jamais de concepts liés aux emails,
# d'où des exemples positifs incohérents présentés au juge LLM.
#
# Ce run corrige cela : train_texts/test_texts = emails réels + variantes
# augmentées (build_email_train_test_corpus, split group-aware par mail
# d'origine), energy/sports/support réduit à un petit corpus secondaire
# post-hoc (N_TOTAL_*=300, cf. src/sae/saev5.py) gardé uniquement pour la
# démonstration de diffing cross-domaine existante.
#
# N_FEATURES_TO_LABEL relevé à 150 (vs 10 par défaut) : puissance statistique
# nécessaire pour conclure sur le taux d'interprétabilité (à 10 features, l'IC à
# 95% sur un taux observé est énorme -- cf. rapport). N_TOKENS_EXTRA_TRAIN laissé
# au défaut (500000) : c'est la config "domaine corrigé, volume par défaut" ;
# comparée séparément aux runs d'ablation volume (run_sae_v10_ablation_tok*.slurm)
# à corpus IDENTIQUE pour isoler l'effet du volume de l'effet du domaine.
export SEED=42
export SAVE_DIR="./results_v10_emails_main/"
export USE_FROZEN_CORE="1"
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="/home/h21486/SAE/local_data/emails/augmented_mails.jsonl"
export MODEL_ID="/home/h21486/SAE/models/gemma-3-12b-it"
export LOCAL_DATASET_PATH="/home/h21486/SAE/datasets/fineweb2_fra/data/fra_Latn/train/000_00000.parquet"
export EMB_MODEL="/home/h21486/SAE/models/F2LLM-v2-80M"

export MODEL_SIZE=12b
export DTYPE=bf16
export N_FEATURES_TO_LABEL=150

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python src/sae/saev5.py
