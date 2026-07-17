#!/bin/bash

#SBATCH --job-name=SAE_v9_full
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=24:00:00
#SBATCH --output=sae_v9_full_%j.log

# ── Isolation réseau ──
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ── Run à l'échelle complète (valeurs par défaut de src/config.py / saev5.py) : le
#    smoketest run_sae.slurm (validé sur plusieurs runs, cf. RESULTS_TESTS.md) a
#    volontairement réduit les volumes ~×12 pour valider le chemin de bout en bout
#    avant d'engager ce run. NOUVEAU répertoire (pas de partage de cache avec le
#    smoketest results_v9_test/ : tailles de corpus/D_SAE différentes -> cache
#    invalide/incompatible si réutilisé tel quel).
export SEED=42
export SAVE_DIR="./results_v9_full/"
export USE_FROZEN_CORE="1"
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export MODEL_ID="/home/h21486/SAE/models/gemma-3-12b-it"
export LOCAL_DATASET_PATH="/home/h21486/SAE/datasets/fineweb2_fra/data/fra_Latn/train/000_00000.parquet"
export EMB_MODEL="/home/h21486/SAE/models/F2LLM-v2-80M"

# ── Pipeline 1 ──
export MODEL_SIZE=12b
# 16k reste le choix délibéré par défaut (config.py) : couverture Neuronpedia bien
# plus dense en proportion que 262k (~10k features labellisées sur 262144, constaté
# manuellement) — ce n'est PAS une réduction de volume, ne pas repasser à 262k.
export DTYPE=bf16

# ── Volumes/hyperparamètres : PAS d'override -> défauts complets appliqués
#    (N_TOTAL_ENERGY/SPORTS/SUPPORT=2000, N_TOKENS_EXTRA_TRAIN=500000, D_SAE=8192,
#    EPOCHS=30, N_FEATURES_TO_LABEL=10, K_SPARSE=16 -- cf. src/config.py et
#    src/sae/saev5.py pour les valeurs par défaut exactes).

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
# Précaution appliquée avant ce run (cf. RESULTS_TESTS.md) : logits_to_keep=1 ajouté
# à l'appel llm(...output_hidden_states=True) de saev5.py (src/sae/saev5.py:~720) --
# même gaspillage de calcul de logits pleine séquence/vocabulaire que celui qui avait
# causé l'OOM CUDA sur scripts/baseline_gemmascope.py, jamais déclenché ici seulement
# grâce au H100 80GB + volumes réduits du smoketest, mais latent avant un run complet.
.venv/bin/python src/sae/saev5.py
