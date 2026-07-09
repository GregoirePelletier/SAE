"""
scripts/run_augmentation.py — Génère les variantes augmentées (AXES de perturbation,
cf. src/data/augmentation.py) à partir du Mails.tsv réel via le LLM Gemma-3 (MODEL_ID).

⚠ Nécessite un vrai Mails.tsv (corpus EDF) — délibérément non exécuté lors de la
validation locale sur cette machine (absent) ; à lancer sur la machine de calcul avec
accès aux données réelles. Sortie consommée par scripts/baseline_gemmascope.py.

Usage :
    LOCAL_MAILS_PATH=/chemin/vers/Mails.tsv MODEL_ID=google/gemma-3-12b-it \
        python scripts/run_augmentation.py
"""
import os

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import DTYPE, SAVE_DIR
from src.data.augmentation import generate_variants
from src.data.dataset import load_mails_tsv

TORCH_DTYPE = torch.bfloat16 if DTYPE == "bf16" else torch.float16

df = load_mails_tsv(os.environ["LOCAL_MAILS_PATH"]).reset_index().rename(columns={"index": "doc_id"})
tok = AutoTokenizer.from_pretrained(os.environ["MODEL_ID"], local_files_only=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    os.environ["MODEL_ID"], torch_dtype=TORCH_DTYPE, device_map="cuda", local_files_only=True
).eval()
generate_variants(model, tok, df, out_jsonl=os.path.join(SAVE_DIR, "augmented_mails.jsonl"))