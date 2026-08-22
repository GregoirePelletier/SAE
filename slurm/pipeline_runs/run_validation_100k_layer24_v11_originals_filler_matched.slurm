#!/bin/bash
#SBATCH --job-name=SAE_val100k_v11
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/pipeline_runs/sae_val100k_l24_v11_originals_filler_matched_%j.log
# --signal=B:USR1@600 : cf. run_validation_25m_layer24.slurm (R1, AUDIT_SAE_2026-08.md §2.3/§4.3).

# B.1, reprise contrôlée de v9 (job 44833, RESULTS_TESTS.md §76). v9 confondait
# corpus (originaux seuls) et part de filler FineWeb2 dans le réservoir
# résiduel (6,3% au run de référence -> 40,7% dans v9, N_VOLUME_FILLER_TARGET_CHUNKS
# resté à 3000 alors que train_texts a été divisé par 12) -- impossible d'y
# distinguer "corpus sans texte généré par le juge" de "SAE entraîné sur plus
# de texte web générique". Ici, N_VOLUME_FILLER_TARGET_CHUNKS réduit à 294
# pour retrouver ~6,3% de filler avec train_texts=3300+test=174+diff=900
# (294/(3300+174+900+294) = 6,3%) -- seule variable isolée : présence ou
# absence des variantes augmentées. N_FEATURES_TO_LABEL=20 : signal, pas une
# réplication statistiquement puissante (n<<150).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export SEED=42
export SAVE_DIR="./results_v25_validation_100k_layer24_v11_originals_filler_matched/"
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
export N_VOLUME_FILLER_TARGET_CHUNKS=294

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python src/sae/saev5.py
