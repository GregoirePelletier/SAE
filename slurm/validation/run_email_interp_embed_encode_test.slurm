#!/bin/bash
#SBATCH --job-name=email_interp_embed
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/validation/email_interp_embed_%j.log

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

# Pipeline interp_embed exact (Dataset + GoodfireSAE, Llama-3.1-8B-Instruct +
# Goodfire SAE-l19) appliqué au corpus mails EDF -- cf. docstring de
# scripts/email_interp_embed_encode_test.py. Pas de juge ici (exploration
# descriptive, pas de ground-truth) -- 1 GPU dédié a100 (40 Go), suffisant
# pour Llama-3.1-8B seul (~16 Go).

cd /home/h21486/SAE/
/home/h21486/SAE/external/interp_embed/.venv/bin/python \
    /home/h21486/SAE/scripts/email_interp_embed_encode_test.py
