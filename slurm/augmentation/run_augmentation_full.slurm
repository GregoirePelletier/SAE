#!/bin/bash
#SBATCH --job-name=SAE_augmentation_full
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/augmentation/augmentation_full_%A_%a.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export LOCAL_MAILS_PATH="/home/h21486/SAE/local_data/emails/Mails.tsv"
export MODEL_ID="/home/h21486/SAE/models/gemma-3-12b-it"
export PYTHONPATH="/home/h21486/SAE:${PYTHONPATH}"
# Écrit directement dans l'emplacement canonique des données email (pas un
# results_*/ -- corrigé après la réorganisation des données : un jeu de données
# d'entrée ne doit pas vivre sous un dossier de résultats d'expérience). Ce job a
# déjà tourné une fois avec succès (job 39017, cf. RESULTS_TESTS.md §0) et son
# résultat fusionné vit maintenant sous local_data/emails/augmented_mails.jsonl ;
# ce script ne sert plus qu'à générer des variantes SUPPLÉMENTAIRES (nouveaux
# mails ajoutés à Mails.tsv) -- il reprend automatiquement (skip des aug_id déjà
# présents), ne régénère jamais ce qui existe déjà.
export SAVE_DIR="./local_data/emails/"

# 3480 mails × 13 axes ≈ 45 240 générations, ~63h GPU cumulées au rythme observé sur
# job 38949/38987, réparti en 8 shards indépendants tournant en parallèle sur les 8
# GPUs du nœud a100 (idle au moment du lancement) -> ~8h de mur au lieu de 63h
# séquentielles. Chaque tâche de l'array traite 1/8 des mails (découpage entrelacé
# df.iloc[idx::8], cf. scripts/run_augmentation.py) et écrit
# local_data/emails/augmented_mails_shard{idx}of8.jsonl -- à fusionner avec
# slurm/augmentation/merge_augmentation_shards.sh une fois les 8 tâches terminées.
export AUGMENT_NUM_SHARDS=8
export AUGMENT_SHARD_IDX=$SLURM_ARRAY_TASK_ID
export AUGMENT_OUT_NAME="augmented_mails.jsonl"

mkdir -p logs "${SAVE_DIR}"

cd /home/h21486/SAE/
# uv run tentait de re-résoudre l'env (réseau bloqué sur le cluster) -> échec
# systématique ; .venv/bin/python bypass ce problème, comme dans run_sae.slurm.
.venv/bin/python scripts/run_augmentation.py
