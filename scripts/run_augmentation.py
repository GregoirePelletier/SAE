"""
scripts/run_augmentation.py — Génère les variantes augmentées (AXES de perturbation,
cf. src/data/augmentation.py) à partir du Mails.tsv réel via le LLM Gemma-3 (MODEL_ID).

⚠ Nécessite un vrai Mails.tsv (corpus EDF) — délibérément non exécuté lors de la
validation locale sur cette machine (absent) ; à lancer sur la machine de calcul avec
accès aux données réelles. Sortie consommée par scripts/baseline_gemmascope.py.

Usage :
    LOCAL_MAILS_PATH=/chemin/vers/Mails.tsv MODEL_ID=google/gemma-3-12b-it \
        python scripts/run_augmentation.py

Sous-échantillonnage (pour un run de validation rapide avant le corpus complet,
13 axes/niveaux × tous les mails ≈ 63h GPU observées) :
    AUGMENT_SAMPLE_N=60 AUGMENT_OUT_NAME=augmented_mails_test.jsonl ...

Sharding (pour paralléliser le corpus complet sur plusieurs jobs/GPUs indépendants ;
chaque shard écrit son propre fichier, à fusionner ensuite — cf. run_augmentation.slurm) :
    AUGMENT_NUM_SHARDS=8 AUGMENT_SHARD_IDX=0 ...   # puis 1, 2, ... 7 sur d'autres jobs
"""
import os

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import DTYPE, SAVE_DIR, SEED
from src.data.augmentation import generate_variants
from src.data.dataset import load_mails_tsv

TORCH_DTYPE = torch.bfloat16 if DTYPE == "bf16" else torch.float16

df = load_mails_tsv(os.environ["LOCAL_MAILS_PATH"]).reset_index().rename(columns={"index": "doc_id"})

sample_n = int(os.environ.get("AUGMENT_SAMPLE_N", "0"))
if sample_n > 0:
    df = df.sample(n=min(sample_n, len(df)), random_state=SEED).reset_index(drop=True)

num_shards = int(os.environ.get("AUGMENT_NUM_SHARDS", "1"))
shard_idx = int(os.environ.get("AUGMENT_SHARD_IDX", "0"))
out_name = os.environ.get("AUGMENT_OUT_NAME", "augmented_mails.jsonl")
if num_shards > 1:
    # Découpage entrelacé (iloc[idx::num_shards]) : répartit équitablement les mails
    # longs/courts entre shards plutôt qu'un split contigu (doc_id reste celui du
    # Mails.tsv d'origine -> aug_id globalement unique, fusion des .jsonl sans collision).
    df = df.iloc[shard_idx::num_shards].reset_index(drop=True)
    base, ext = os.path.splitext(out_name)
    out_name = f"{base}_shard{shard_idx}of{num_shards}{ext}"

tok = AutoTokenizer.from_pretrained(os.environ["MODEL_ID"], local_files_only=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    os.environ["MODEL_ID"], torch_dtype=TORCH_DTYPE, device_map="cuda", local_files_only=True
).eval()
generate_variants(model, tok, df, out_jsonl=os.path.join(SAVE_DIR, out_name))