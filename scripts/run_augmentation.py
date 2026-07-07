import os, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.data.augmentation import generate_variants
from src.data.dataset import load_mails_tsv

df = load_mails_tsv(os.environ["LOCAL_MAILS_PATH"]).reset_index().rename(columns={"index": "doc_id"})
tok = AutoTokenizer.from_pretrained(os.environ["MODEL_ID"], local_files_only=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(os.environ["MODEL_ID"], torch_dtype=torch.bfloat16,
                                              device_map="cuda", local_files_only=True).eval()
generate_variants(model, tok, df, out_jsonl="results_v9_test/augmented_mails.jsonl")