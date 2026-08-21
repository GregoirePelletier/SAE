#!/bin/bash
#SBATCH --job-name=SAE_val100k_v8
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/pipeline_runs/sae_val100k_l24_v8_datascience_fixes_%j.log
# --signal=B:USR1@600 : cf. run_validation_25m_layer24.slurm (R1, AUDIT_SAE_2026-08.md §2.3/§4.3).

# Réplique de v7 (job 44778, sharding confirmé GPU) après le lot de
# correctifs perf + data-science de cette session (block_shuffle_indices,
# tri du juge par longueur, hook resid_post garde-fou tuple, Fisher
# vectorisé, cooccurrence_graph plafonné, ET le protocole odd-one-out : B.3
# argmax du négatif, B.5 garde-fou neg_quantile, B.4 dédup positifs par
# document, B.6 régime "avoir", B.11 frontières de mot du corpus diffing,
# B.7 jointure de split par hash -- retombe sur l'ancienne jointure
# positionnelle ici, augmented_mails.jsonl n'a pas encore parent_sha1).
# SAVE_DIR neuf : le changement de matching lexical (B.11) modifie la
# composition de diff_texts (energy/sports/support), donc all_texts --
# réutiliser le cache d'extraction de v7 servirait des fragments désalignés
# avec les nouveaux indices de document. Comparaison attendue : rho_sae/
# dead_pct/fve_pretrained dans le bruit habituel entre runs identiques
# (aucun de ces correctifs ne change l'extraction/l'entraînement du SAE
# résiduel lui-même) ; interp_score potentiellement différent (protocole de
# labellisation changé, c'est le point vérifié).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export SEED=42
export SAVE_DIR="./results_v22_validation_100k_layer24_v8_datascience_fixes/"
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
