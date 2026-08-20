#!/bin/bash
#SBATCH --job-name=SAE_val100k_l24
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/pipeline_runs/sae_val100k_l24_postfixes_%j.log
# --signal=B:USR1@600 : cf. run_validation_25m_layer24.slurm (R1, AUDIT_SAE_2026-08.md §2.3/§4.3).

# Run de validation à petite échelle (100k tokens résidus, filler réduit), sur
# a100 pour ne PAS contendre avec job 44211 (h100, toujours en cours à ce jour)
# ni partager son SAVE_DIR/cache -- SAVE_DIR entièrement séparé ci-dessous.
# Objectif : mesurer les temps réels par phase (extraction / entraînement /
# ré-encodage / juge) avec l'ensemble des correctifs de cette session appliqués
# ensemble pour la première fois sur un run bout-en-bout réel :
#   - G1 : troncature layers[:LAYER] + hook direct (job 44540, vérifié GPU
#     séparément : 1,98x, torch.equal=True) ;
#   - correctif SAE Boost (frozen_core.py, encodeur extra sur x, pas e) ;
#   - sync .item() 1x/époque au lieu de 4x/step (sae_shared.py) ;
#   - feature_acts désactivé pendant l'entraînement (frozen_core.py) ;
#   - reprise après coupure (src/storage/checkpoint.py) câblée dans les trois
#     boucles longues -- pas testée ICI en conditions de coupure réelle (ce
#     run n'est pas volontairement interrompu), seulement le chemin nominal ;
#   - correction du bug de régression e/x dans la passe de ré-encodage
#     (saev5.py, trouvé en implémentant la reprise de cette passe).
# LAYER=24/SAE_ID/K_EXTRA identiques au run de référence 25M pour rester dans
# le régime déjà vérifié pour G1 -- seule l'échelle (tokens, filler) change ici.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

export SEED=42
export SAVE_DIR="./results_v15_validation_100k_layer24_postfixes/"
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
