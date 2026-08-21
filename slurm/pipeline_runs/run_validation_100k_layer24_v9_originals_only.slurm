#!/bin/bash
#SBATCH --job-name=SAE_val100k_v9
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/pipeline_runs/sae_val100k_l24_v9_originals_only_%j.log
# --signal=B:USR1@600 : cf. run_validation_25m_layer24.slurm (R1, AUDIT_SAE_2026-08.md §2.3/§4.3).

# B.1 (AUDIT_SAE_2026-08.md) : le corpus d'entraînement de l'extension est
# dominé (~93% au run de référence) par du texte généré par Gemma-3-12B-it,
# le MÊME modèle qui juge ensuite les features -- jamais testé, seule
# l'indépendance DU JUGE (RESULTS_TESTS.md §48/§50/§52) l'a été. Réplique de
# v8 (job 44831) mais LOCAL_AUGMENTED_MAILS_PATH volontairement vide :
# build_email_train_test_corpus (preparation.py) court-circuite alors tout
# chargement de variantes augmentées (`if augmented_jsonl_path and
# os.path.exists(...)`), train_texts/test_texts = mails réels seuls (~3480
# mails au lieu de ~41k train/2,2k test habituels). N_FEATURES_TO_LABEL=20 :
# un premier signal à cette échelle de validation, PAS une réplication
# statistiquement puissante (n=20 << les n>=150 requis par CLAUDE.md pour un
# taux agrégé informatif) -- à traiter comme tel dans la comparaison, pas
# comme une conclusion définitive.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export SEED=42
export SAVE_DIR="./results_v23_validation_100k_layer24_v9_originals_only/"
export USE_FROZEN_CORE="1"
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH=""
export LOCAL_DATASET_PATH="/home/h21486/SAE/datasets/fineweb2_fra/data/fra_Latn/train/000_00000.parquet"
export VOLUME_FILLER_DATASET_GLOB="/home/h21486/SAE/datasets/fineweb2_fra/data/fra_Latn/train/000_0000*.parquet"

export MODEL_ID="/home/h21486/SAE/models/gemma-3-12b-it"
export MODEL_SIZE=12b
export DTYPE=bf16
export PIPELINES=p1

export LAYER=24
export SAE_ID="layer_24_width_16k_l0_medium"
export K_EXTRA=32

export N_FEATURES_TO_LABEL=20
export EPOCHS_EXTRA=10
export D_EXTRA=1024
export N_TOKENS_EXTRA_TRAIN=100000
export N_VOLUME_FILLER_TARGET_CHUNKS=3000

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python src/sae/saev5.py
