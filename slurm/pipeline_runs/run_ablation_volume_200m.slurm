#!/bin/bash
#SBATCH --job-name=SAE_abl_vol200m
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=1800G
#SBATCH --time=120:00:00
#SBATCH --output=logs/pipeline_runs/sae_abl_vol200m_%j.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ── Ablation volume à grande échelle : ~200M tokens, seuil HAUT du papier SAE
# Boost (Koriagin et al., COLM 2025, arXiv:2507.12990) ──────────────────────────
# Complète §23 (25M tokens, RESULTS_TESTS.md) : avec la soutenance repoussée de
# ~2 mois, le temps disponible permet enfin de viser le haut de la fourchette
# 100-200M du papier plutôt qu'un compromis. Deux conditions manquaient
# jusqu'ici, toutes deux résolues avant ce lancement :
#
# 1. MÉMOIRE : le réservoir de résidus (saev5.py, préalloué depuis le commit
#    18cc204 -- plus de doublement transitoire liste+cat) a une empreinte flat
#    de N_TOKENS_EXTRA_TRAIN * hidden_size * 2 octets (bf16). Pour Gemma-3-12B
#    (hidden_size=3840 -- valeur CORRIGÉE, cf. commit bc702de : D_MODEL était à
#    4096 depuis l'ajout du preset "12b", jamais exercé jusqu'à la préparation
#    de ce run, aurait fait planter le job sur un mismatch de forme) et 200M
#    tokens : 200e6*3840*2 = 1 536 000 000 000 octets ≈ 1,40 To. Tous les runs
#    précédents tournaient sur la partition a100 (~1 To de RAM total, partagé) --
#    ce run utilise h100 (~2 To de RAM par nœud), qui a la marge nécessaire
#    (--mem=1800G laisse ~200-600 Go de marge selon la charge du nœud au
#    lancement). h100-bis (même RAM) était l'alternative mais reste `drained`
#    (incident admin "Kill task failed", hors de notre contrôle) au moment du
#    lancement.
#
# 2. VOLUME DE FILLER DISPONIBLE : avec seulement 3 shards FineWeb2-fr locaux,
#    le pool de filler exploitable (~286 000 chunks obtenus sur les 540 000
#    visés, hit rate 0,275% sur UTILITY_COMPLAINT_KEYWORDS, cf. RESULTS_TESTS.md
#    §23.1) plafonnait autour de ~65-70M tokens -- insuffisant pour espérer
#    dépasser 100M avec marge, sans même parler de 200M. 15 shards
#    supplémentaires ont été téléchargés (18 au total, ~87 Go), portant le pool
#    estimé à ~350-400M tokens de filler brut (avant filtrage token-level) --
#    large marge au-dessus de la cible.
#
# Mêmes garde-fous que le run à 25M : le filler alimente UNIQUEMENT le
# réservoir de tokens résiduels (volume_filler_texts), jamais train_texts --
# la sélection de features et la sonde de classification restent calculées
# sur les emails+augmentés seuls (cf. docstring de run_llm_max_pool_pipeline).
# SAE_ID=16k explicite (comparabilité directe avec le run principal).
# PIPELINES=p1 uniquement (le levier testé n'existe pas pour Pipeline 2).
#
# Durée estimée : le corpus total (train+filler+test+diff) passe d'environ
# 330 000 textes (run à 25M) à ~1,8M textes (filler ~6x plus volumineux) --
# à un débit comparable (~14 docs/s observé sur le run à 25M), l'extraction
# seule prendrait ~35-36h. --time=120:00:00 (5 jours) conservé par prudence.
export SEED=42
export SAVE_DIR="./results_v13_ablation_volume200m/"
export USE_FROZEN_CORE="1"
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="/home/h21486/SAE/local_data/emails/augmented_mails.jsonl"
export LOCAL_DATASET_PATH="/home/h21486/SAE/datasets/fineweb2_fra/data/fra_Latn/train/000_00000.parquet"
export VOLUME_FILLER_DATASET_GLOB="/home/h21486/SAE/datasets/fineweb2_fra/data/fra_Latn/train/000_000*.parquet"
export EMB_MODEL="/home/h21486/SAE/models/F2LLM-v2-80M"

export MODEL_ID="/home/h21486/SAE/models/gemma-3-12b-it"
export MODEL_SIZE=12b
export DTYPE=bf16
export SAE_ID="layer_24_width_16k_l0_medium"
export PIPELINES=p1

export N_FEATURES_TO_LABEL=150
export EPOCHS_EXTRA=10
export D_EXTRA=1024
export K_EXTRA=32
export N_TOKENS_EXTRA_TRAIN=200000000
export N_VOLUME_FILLER_TARGET_CHUNKS=2500000

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python src/sae/saev5.py
