#!/bin/bash
#SBATCH --job-name=imdb_genre_diff_bf16
#SBATCH --partition=h100-bis
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/validation/imdb_genre_diffing_bf16_%j.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export JUDGE_QUANT=bf16

# Variante exploratoire : h100-bis (gpu:8,shard:16 -- PAS garanti 80 Go dédiés
# comme a100 gpu:8, à observer). Juge Qwen3.8-27B en bf16 complet (~52 Go,
# nécessite une carte 80 Go non partagée) au lieu du 8-bit -- comparaison de
# qualité de jugement + test empirique de la disponibilité réelle d'un GPU
# complet sur cette partition shardée quand elle semble idle.

cd /home/h21486/SAE/
/home/h21486/SAE/external/interp_embed/.venv/bin/python \
    /home/h21486/SAE/scripts/imdb_genre_diffing_test.py
