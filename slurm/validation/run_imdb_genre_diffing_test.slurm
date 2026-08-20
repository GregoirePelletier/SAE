#!/bin/bash
#SBATCH --job-name=imdb_genre_diff
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=05:00:00
#SBATCH --output=logs/validation/imdb_genre_diffing_%j.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

# Réplication ciblée §4.1/Appendix D.3 (movie genre ground-truth diffing) --
# cf. docstring de scripts/imdb_genre_diffing_test.py pour tous les écarts
# documentés vs. le papier (juge local Qwen3.8-27B au lieu de GPT-5, dataset
# HF au lieu de la source exacte du papier, SAE 8B/l19 au lieu du SAE
# 70B/layer-50 réellement utilisé pour les chiffres publiés 0.75/0.90).
# Llama-3.1-8B (encodage) et Qwen3.8-27B (juge) sont chargés SÉQUENTIELLEMENT
# (le premier est détruit avant le second) -- 1 GPU suffit, pas de partage
# VRAM concurrent entre les deux modèles.

cd /home/h21486/SAE/
/home/h21486/SAE/external/interp_embed/.venv/bin/python \
    /home/h21486/SAE/scripts/imdb_genre_diffing_test.py
