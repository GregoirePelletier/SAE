#!/bin/bash
#SBATCH --job-name=SAE_sanity_frozen
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=logs/analysis/sanity_check_frozen_decoder_%j.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ── Sanity check (Korznikov et al. 2026, "Sanity Checks for Sparse Autoencoders :
# Do SAEs Beat Random Baselines?", pdf/sanitychecks.pdf) ────────────────────────
# Leur résultat central : un SAE dont le DÉCODEUR est figé à une initialisation
# ALÉATOIRE (jamais entraîné) égale un SAE réellement entraîné sur interprétabilité
# (AutoInterp), sparse probing et causal editing (RAVEL) -- suggérant que ces
# métriques ne suffisent pas à distinguer un apprentissage de features
# significatif d'un simple ajustement de l'encodeur à des directions arbitraires.
#
# Ce run reproduit leur baseline "Frozen Decoder" sur NOTRE extension P1
# (FrozenDecoderExtendedSAE, src/sae/frozen_core.py, SANITY_CHECK_FROZEN_DECODER=1) :
# W_dec_extra reste figé à son init aléatoire tout au long de l'entraînement, seuls
# W_enc_extra/b_enc_extra/seuil BatchTopK sont appris. Toutes les autres conditions
# IDENTIQUES au run principal (results_v10_emails_main, 16k, 150 features jugées)
# pour une comparaison à facteur unique directe : si le taux d'interprétabilité
# odd-one-out et la précision de la sonde de classification de cette baseline
# figée sont comparables au run principal (45,3% / 93,5%), cela remettrait en
# cause la validité de notre protocole d'évaluation exactement comme le montre
# le papier pour AutoInterp/sparse probing/RAVEL.
export SEED=42
export SAVE_DIR="./results_v12_sanity_frozen_decoder/"
export USE_FROZEN_CORE="1"
export SANITY_CHECK_FROZEN_DECODER="1"
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="/home/h21486/SAE/local_data/emails/augmented_mails.jsonl"
export LOCAL_DATASET_PATH="/home/h21486/SAE/datasets/fineweb2_fra/data/fra_Latn/train/000_00000.parquet"

export MODEL_ID="/home/h21486/SAE/models/gemma-3-12b-it"
export MODEL_SIZE=12b
export DTYPE=bf16
export SAE_ID="layer_24_width_16k_l0_medium"
export PIPELINES=p1

export N_FEATURES_TO_LABEL=150
export EPOCHS_EXTRA=10
export D_EXTRA=1024
export K_EXTRA=32

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python src/sae/saev5.py
