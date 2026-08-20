#!/bin/bash
#SBATCH --job-name=SAE_val25m_l24
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/pipeline_runs/sae_val25m_l24_%j.log
# --signal=B:USR1@600 : SLURM envoie SIGUSR1 ~10 min avant le SIGKILL du
# timeout (48h) -- laisse le temps à saev5.py (_GracefulShutdown) d'écrire un
# checkpoint de progression avant l'arrêt forcé (R1, AUDIT_SAE_2026-08.md
# §2.3/§4.3). Sans ce flag, seule une coupure "propre" (OOM, crash Python,
# scancel) déclenche la reprise -- le timeout SLURM lui-même reste un
# SIGKILL sec.

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

# ── Etape 1 du plan (sharded-twirling-sunbeam.md) : run de validation a 25M
# tokens, LAYER=24/SAE_ID=16k/K_EXTRA=32 tenus EXPLICITEMENT egaux aux runs
# precedents (results_v13_ablation_volume25m{,_seed7,_seed99}, RESULTS_TESTS.md
# §23.4/§56 -- 54,0%/46,0%/44,7%, conclusion "pas de replication nette") pour
# une comparaison propre : seul ce qui a change depuis (le code, pas les
# hyperparametres) doit expliquer un ecart, s'il y en a un.
#
# Objectifs de cette passe (Etape 1 du plan) :
#  - le reservoir memmap (open_mmap_reservoir, §54) tient sous la RAM REELLE du
#    noeud (96G demandes, node a 200G total -- pas les "~2 To" supposes par les
#    scripts precedents, corrige cette session apres verification directe
#    sinfo) ;
#  - le nouveau filler sans filtre (§54, sample_fineweb2_chunks) elimine le
#    goulot a 0,275% de hit rate observe sur l'ancien filtre par mots-cles ;
#  - debit reel mesure (stage_timer deja instrumente, saev5.py) pour
#    recalculer une estimation fiable a 200M, pas une extrapolation depuis
#    l'ancien code ;
#  - empreinte disque reelle confirmee (~192 Go attendus pour le reservoir
#    seul, cf. calcul deja fait : 25e6*3840*2 octets) ;
#  - aucune regression sur les metriques principales par rapport aux runs §23.4/
#    §56 malgru les correctifs de code appliques cette session (DataLoader
#    vectorise, B.21 bf16, B.10 outlier mask par document, B.5 rho_interp,
#    B.28 seed deterministe, B.6 CV group-aware).
#
# --mem=96G (pas 500G comme l'ancien script) : le reservoir ne consomme plus de
# RAM anonyme proportionnelle au volume (fix memmap), le reste (poids Gemma-3-
# 12B bf16 ~24 Go, buffers d'extraction/fragments) devrait tenir tres large.
# Premiere validation empirique de ce nouveau budget memoire -- si ce job est
# tue pour OOM, c'est en soi un resultat d'Etape 1 a documenter, pas un echec a
# masquer.
export SEED=42
export SAVE_DIR="./results_v14_validation_25m_layer24/"
export USE_FROZEN_CORE="1"
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="/home/h21486/SAE/local_data/emails/augmented_mails.jsonl"
export LOCAL_DATASET_PATH="/home/h21486/SAE/datasets/fineweb2_fra/data/fra_Latn/train/000_00000.parquet"
export VOLUME_FILLER_DATASET_GLOB="/home/h21486/SAE/datasets/fineweb2_fra/data/fra_Latn/train/000_0000*.parquet"

export MODEL_ID="/home/h21486/SAE/models/gemma-3-12b-it"
export MODEL_SIZE=12b
export DTYPE=bf16
export PIPELINES=p1

# Comparabilite explicite au run principal et aux ablations 25M precedentes :
export LAYER=24
export SAE_ID="layer_24_width_16k_l0_medium"
export K_EXTRA=32

export N_FEATURES_TO_LABEL=150
export EPOCHS_EXTRA=10
export D_EXTRA=1024
export N_TOKENS_EXTRA_TRAIN=25000000
export N_VOLUME_FILLER_TARGET_CHUNKS=540000

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python src/sae/saev5.py
