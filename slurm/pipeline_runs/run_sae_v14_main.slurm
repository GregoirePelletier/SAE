#!/bin/bash
#SBATCH --job-name=SAE_v14_main
#SBATCH --partition=h100-bis
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=1000G
#SBATCH --time=120:00:00
#SBATCH --output=logs/pipeline_runs/sae_v14_main_%j.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ── Nouveau run "main" (destiné à remplacer results_v10_emails_main) ───────────
# Combine les leviers les plus prometteurs des ablations v13 en un run
# PRODUCTION complet (Pipeline 1 + Pipeline 2), contrairement aux ablations qui
# isolaient chaque levier séparément en p1 seul :
#
# - MODEL_SIZE=12b : seul levier ayant montré un effet significatif à chaque
#   palier testé (12,0% -> 28,0% -> 45,3%, dose-réponse monotone, RESULTS_TESTS.md
#   §28) -- déjà le défaut, confirmé plutôt que changé (aucun modèle plus gros
#   n'existe dans _PRESETS, src/config.py).
# - K_EXTRA=5 (au lieu de 32) : recommandation de sensibilité du papier SAE
#   Boost (Koriagin et al., COLM 2025, arXiv:2507.12990, leur Table 12). Testé
#   isolément en §25 : +9,4 points (45,3% -> 54,7%), NON significatif seul
#   (z=-1,62, p≈0,106) mais même direction et même ordre de grandeur que
#   l'ablation volume ci-dessous -- cf. §23.4/§25 pour la discussion de cette
#   coïncidence directionnelle, jamais prouvée combinée avant ce run.
#   S'applique UNIQUEMENT à l'extension résiduelle de Pipeline 1
#   (FrozenCoreResidualSAE = l'implémentation SAE Boost du projet, §18.1) --
#   K_SPARSE (Pipeline 2, PhraseLevelSAE from-scratch, architecture différente)
#   reste au défaut (16) : aucune évidence disponible pour le changer là.
# - N_TOKENS_EXTRA_TRAIN=100000000 (100M) : milieu de la fourchette 100-200M du
#   papier SAE Boost. §23.4 (25M : +8,7 points, non significatif seul) et §23.5
#   (200M : échec disque après 20h57 d'extraction COMPLETE -- fichier de
#   résidus bruts tronqué à ~1,22 To, incident du 2026-08-05, /home à 100%
#   plein -- pas un bug de ce projet) encadrent ce choix. Empreinte RAM/disque :
#   100e6*3840(hidden_size)*2(bf16) = 768 Go pour le seul tenseur de résidus --
#   --mem=1000G sur h100-bis (2 To RAM nœud, ~1,3 To non alloué au moment de la
#   soumission, cf. `scontrol show node dgx-h100-bis`). SAE_ID laissé au défaut
#   (65k, pas 16k) : pas de contrainte de comparabilité inter-ablations ici, et
#   65k a la meilleure couverture Neuronpedia mesurée (README, §Choix du SAE
#   GemmaScope-2).
# - N_FEATURES_TO_LABEL=500 (au lieu de 150) : puissance statistique accrue --
#   plusieurs ablations ci-dessus butent juste sous la significativité à n=150
#   (§25 : p≈0,106) ; 500 réduit sensiblement l'intervalle de confiance
#   binomial. S'applique aux deux pipelines (P1 extension + P2 phrase-level,
#   seul réglage N_FEATURES_TO_LABEL partagé, cf. src/sae/saev5.py L1077/1406).
# - PIPELINES=p1,p2 (déjà le défaut, explicite ici pour la documentation) : les
#   deux pipelines, contrairement à toutes les ablations v13 (p1 seul, le
#   levier volume/K_EXTRA n'existe pas pour P2).
# - Exemples négatifs : `neg_example` (src/sae/judge.py) était absent des runs
#   antérieurs à son ajout, dont results_v10_emails_main (cf. le caveat affiché
#   dans le dashboard, page Features -- c'est la lacune que ce run comble).
#   Aucun changement de code requis : le code actuel le produit nativement pour
#   les deux pipelines.
#
# AVERTISSEMENT DISQUE (lu au moment de la soumission, 2026-08-05) : /home est
# un volume réseau PARTAGÉ entre tous les utilisateurs du cluster (35 To, 97%
# plein, ~1,2 To libres constatés). Le run à 200M a échoué en écrivant son
# tenseur final (768 Go visés ici contre 1,22 To alors) -- le volume était à
# 100% plein (24 Go libres) au moment de cet échec, pas un bug. Marge actuelle
# ~450 Go, mais ce volume a déjà varié de plus de 1 To en quelques jours du
# seul fait des AUTRES utilisateurs -- aucune garantie de marge suffisante au
# moment de l'écriture finale, plusieurs heures après le lancement.
#
# Filler FineWeb2-fr : les 18 shards déjà téléchargés pour le run à 200M
# (~87 Go, cf. run_ablation_volume_200m.slurm) sont réutilisés tels quels, sans
# retéléchargement. N_VOLUME_FILLER_TARGET_CHUNKS réduit à 1 200 000 (~moitié
# du run à 200M) : estimation grossière visant un corpus total extrait
# d'environ 200-250k textes (contre 432 490 pour 200M) pour rester
# proportionnellement plus rapide -- non vérifié empiriquement à ce volume
# précis. Le filler n'alimente QUE le réservoir de résidus (jamais
# train_texts/la sélection de features à labelliser), même garde-fou que tous
# les runs volume précédents (cf. docstring run_llm_max_pool_pipeline).
export SEED=42
export SAVE_DIR="./results_v14_main/"
export USE_FROZEN_CORE="1"
export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export LOCAL_AUGMENTED_MAILS_PATH="/home/h21486/SAE/local_data/emails/augmented_mails.jsonl"
export LOCAL_DATASET_PATH="/home/h21486/SAE/datasets/fineweb2_fra/data/fra_Latn/train/000_00000.parquet"
export VOLUME_FILLER_DATASET_GLOB="/home/h21486/SAE/datasets/fineweb2_fra/data/fra_Latn/train/000_000*.parquet"
export EMB_MODEL="/home/h21486/SAE/models/F2LLM-v2-80M"

export MODEL_ID="/home/h21486/SAE/models/gemma-3-12b-it"
export MODEL_SIZE=12b
export DTYPE=bf16
export PIPELINES=p1,p2

export N_FEATURES_TO_LABEL=500
export EPOCHS_EXTRA=10
export D_EXTRA=1024
export K_EXTRA=5
export N_TOKENS_EXTRA_TRAIN=100000000
export N_VOLUME_FILLER_TARGET_CHUNKS=1200000

export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"

cd /home/h21486/SAE/
.venv/bin/python src/sae/saev5.py
