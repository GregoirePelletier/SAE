#!/bin/bash
#SBATCH --job-name=SAE_v12_scaled
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=48:00:00
#SBATCH --output=sae_v12_scaled_%j.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ── Run "mega" à l'échelle supérieure (v12) ─────────────────────────────────────
# Objectif demandé : UN SEUL run où tous les onglets du dashboard ont du contenu,
# en scalant le VOLUME d'entraînement/labellisation SANS changer la méthodologie
# déjà validée (corpus emails+augmentés group-aware, protocole juge odd-one-out,
# sélection top-N par magnitude, etc.) -- ablation "est-ce que scaler aide ?"
# dans le même esprit que l'ablation tokens (run_sae_v10_ablation_tok*.slurm), mais
# cette fois sur (a) la largeur/couverture du SAE core, (b) le nombre d'époques
# des deux pipelines, (c) le nombre de features labellisées.
#
# (a) SAE_ID passé à layer_24_width_65k_l0_medium (au lieu de 16k) : couverture
#     Neuronpedia mesurée empiriquement pour gemma-3-12b-it/layer 24 (cf.
#     src/config.py, RESULTS_TESTS.md) -- 16k=82,6% (13535/16384), 65k=87,8%
#     (57551/65536, MEILLEURE couverture ET ~4,3x plus de features labellisées en
#     absolu), 262k=5,3% (13851/262144, confirme le ~10k estimé manuellement),
#     1m=inexistant côté Neuronpedia (HTTP 404). 65k est donc le choix optimal sur
#     les deux critères demandés (proportion ET nombre absolu de labels).
# (b) EPOCHS_EXTRA 10->40 (P1 extension) et EPOCHS 30->100 (P2 phrase-level) :
#     x4 et x3,3 respectivement. Les courbes de loss des runs précédents
#     (sae_v10_emails_39660.log) montraient un plateau proche mais pas strictement
#     atteint (P1 : 0,2708->0,2666 entre époque 7 et 10 ; P2 : oscillation
#     0,064-0,068 en fin d'entraînement) -- on teste si pousser plus loin change
#     réellement les métriques downstream (taux d'interprétabilité, NMSE, etc.)
#     ou si ça plafonne comme pour le volume de tokens.
# (c) N_FEATURES_TO_LABEL 150->600 (x4) : puissance statistique accrue sur le taux
#     d'interprétabilité (IC95% divisé par ~2, cf. RESULTS_TESTS.md) ET couverture
#     plus large de features EXTENSION jugées + features CORE cross-référencées.
# N_TOKENS_EXTRA_TRAIN laissé au défaut (500000) : l'ablation dédiée
# (run_sae_v10_ablation_tok*.slurm) a déjà montré un plateau net (40,7%/45,3%/
# 44,7% entre 100k/500k/2M) -- pousser ce paramètre n'apporterait aucune
# information nouvelle, seulement du coût GPU.
# EMB_MODEL = F2LLM-v2-330M (pas -80M) : conforme à la condition fixée pour
# l'évaluation "toutes méthodes" (docs/evaluation_protocol.md, "F2LLMv2 assez
# grand") et déjà validé séparément (results_v10_p2_f2llm330m, NMSE meilleur :
# 0,0745->0,0689).
#
# NOUVEAU SAVE_DIR : aucun run existant ne doit être réutilisé (largeur de SAE
# core différente -> tout le cache d'activations/fragments serait incompatible).
export SEED=42
export SAVE_DIR="./results_v12_scaled_65k/"
export USE_FROZEN_CORE="1"
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="/home/h21486/SAE/local_data/emails/augmented_mails.jsonl"
export LOCAL_DATASET_PATH="/home/h21486/SAE/datasets/fineweb2_fra/data/fra_Latn/train/000_00000.parquet"
export EMB_MODEL="/home/h21486/SAE/models/F2LLM-v2-330M"

export MODEL_ID="/home/h21486/SAE/models/gemma-3-12b-it"
export MODEL_SIZE=12b
export DTYPE=bf16
export SAE_ID="layer_24_width_65k_l0_medium"
export PIPELINES=p1,p2

export N_FEATURES_TO_LABEL=600
export EPOCHS_EXTRA=40
export EPOCHS=100

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python src/sae/saev5.py
