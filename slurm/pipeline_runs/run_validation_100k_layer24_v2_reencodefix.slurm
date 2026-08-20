#!/bin/bash
#SBATCH --job-name=SAE_val100k_v2
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/pipeline_runs/sae_val100k_l24_v2_reencodefix_%j.log
# --signal=B:USR1@600 : cf. run_validation_25m_layer24.slurm (R1, AUDIT_SAE_2026-08.md §2.3/§4.3).

# Réplique EXACTE de run_validation_100k_layer24_postfixes.slurm (job 44560),
# mêmes paramètres, SEUL le code a changé -- comparaison contrôlée à un seul
# facteur. Différence de code depuis 44560 :
#   - exclusion du filler dans la passe de ré-encodage
#     (src/data/preparation.py::build_reencode_targets, AUDIT_SAE_2026-08.md
#     §2.5) -- N_VOLUME_FILLER_TARGET_CHUNKS=3000 ici (identique à 44560),
#     donc filler ~6% du corpus (41176 train + 3000 filler + 2177 test) --
#     gain attendu modeste à cette échelle (le ×13 mesuré sur le run de
#     référence vient d'un filler à 92% du corpus, pas de cette config) ;
#     objectif ici est de VALIDER le correctif bout-en-bout, pas de démontrer
#     l'ampleur du gain (cf. troisième run prévu après l'allègement côté
#     extraction, qui isolera ce second facteur) ;
#   - batching du juge en 3 passes (judge.py::_batched_generate) -- vérifié
#     GPU séparément (job 44570, 6,51x, 1/24 désaccord expliqué : non-
#     associativité flottante des kernels batchés, pas un bug, cf. audit §2.9
#     item 1).
# SAVE_DIR séparé de 44560 -- pas de partage de cache entre les deux runs.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

export SEED=42
export SAVE_DIR="./results_v16_validation_100k_layer24_v2_reencodefix/"
export USE_FROZEN_CORE="1"
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="/home/h21486/SAE/local_data/emails/augmented_mails.jsonl"
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
