#!/bin/bash
#SBATCH --job-name=SAE_abl_vol100m
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=120:00:00
#SBATCH --output=logs/pipeline_runs/sae_abl_vol100m_%j.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ── Ablation volume à grande échelle : ~100-120M tokens (SAE Boost, Koriagin et al.
# COLM 2025, arXiv:2507.12990) ──────────────────────────────────────────────────
# Notre ablation volume initiale (RESULTS_TESTS.md §5/§12, 100k/500k/2M tokens) n'a
# montré aucun effet -- mais reste 50-100x en dessous du seuil de convergence de
# 100-200M tokens documenté par le papier pour un SAE résiduel (leur Fig. 4 :
# jusqu'à -31% d'EV générale en dessous de 100M). Ce run teste directement si ce
# seuil s'applique à ce projet.
#
# Le corpus emails+augmentés (~6M tokens) est très en dessous de 100M -- complété
# par un filler FineWeb2-fr filtré par mots-clés spécifiques au registre
# réclamation/résiliation/facturation énergie (UTILITY_COMPLAINT_KEYWORDS,
# src/data/keywords.py). Précision qualitative imparfaite mesurée empiriquement
# (~15-20% des documents genuinement sur le sujet énergie, le reste étant du
# registre "réclamation/service client" générique cross-secteur -- aucun filtre
# par mots-clés testé n'a donné mieux, cf. RESULTS_TESTS.md §23.1) : c'est un choix
# assumé, pas un idéal -- SignalConso (réclamations consommateurs françaises
# officielles) a été écarté car son export public ne contient aucun champ de texte
# libre (uniquement des métadonnées catégorielles, vérifié empiriquement).
#
# Le filler est ajouté UNIQUEMENT au réservoir de tokens résiduels
# (volume_filler_texts, jamais à train_texts) : la sélection des features à
# labelliser et la sonde de classification email restent calculées sur les
# emails+augmentés SEULS, pour ne pas réintroduire le biais de domaine diagnostiqué
# et corrigé au §12 (cf. docstring de run_llm_max_pool_pipeline, src/sae/saev5.py).
#
# 3 shards FineWeb2-fr locaux (000_00000/1/2.parquet, ~14,5 Go, téléchargés pour ce
# run) scannés pour un total visé de ~540 000 chunks (~114M tokens estimés à partir
# du hit rate mesuré, 0,275%, et de la longueur moyenne des documents trouvés).
# SAE_ID=16k (comme le run principal, pour comparabilité directe) explicite car le
# défaut de src/config.py a été relevé à 65k après le run v12. PIPELINES=p1
# uniquement (le levier testé -- SAE résiduel de Pipeline 1 -- n'existe pas pour
# le PhraseLevelSAE from-scratch de Pipeline 2, cf. docs/references.md).
#
# Coût attendu MULTI-JOURS (extraction Gemma-3-12B sur ~584 000 textes au total,
# ~13x le volume du run principal) -- autorisé explicitement par l'utilisateur en
# début de stage pour les jobs de cette ampleur.
export SEED=42
export SAVE_DIR="./results_v13_ablation_volume100m/"
export USE_FROZEN_CORE="1"
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="/home/h21486/SAE/local_data/emails/augmented_mails.jsonl"
export LOCAL_DATASET_PATH="/home/h21486/SAE/datasets/fineweb2_fra/data/fra_Latn/train/000_00000.parquet"
export VOLUME_FILLER_DATASET_GLOB="/home/h21486/SAE/datasets/fineweb2_fra/data/fra_Latn/train/000_0000*.parquet"
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
export N_TOKENS_EXTRA_TRAIN=100000000
export N_VOLUME_FILLER_TARGET_CHUNKS=540000

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python src/sae/saev5.py
