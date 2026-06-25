import glob
import json
import os
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset

from src.data.keywords import SUPPORT_KEYWORDS, SUPPORT_URL_PATTERNS


def keyword_match(text: str, keywords: List[str]) -> bool:
    if not text or not keywords:
        return False
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def url_match(url: str, patterns: List[str]) -> bool:
    if not url or not patterns:
        return False
    url_lower = url.lower()
    return any(pat.lower() in url_lower for pat in patterns)


def is_expressive_or_support(text: str, url: str = "") -> bool:
    text_lower = text.lower()
    url_lower = url.lower() if url else ""
    source_match = any(domain in url_lower for domain in SUPPORT_URL_PATTERNS)
    keyword_count = sum(1 for kw in SUPPORT_KEYWORDS if kw.lower() in text_lower)
    return source_match or (keyword_count >= 2)


def prepare_domain_dataset(
    keywords: List[str],
    domain_name: str,
    n_target: int,
    chunk_length: int = 1024,
    max_chunks: int = 6,
    url_patterns: List[str] = None,
    local_dataset_path: str = None,
    use_fineweb2: bool = False,
    hf_token: str = None,
) -> List[str]:
    url_patterns = url_patterns or []
    keywords = keywords or []
    texts = []
    print(f"  [sae_shared] Recherche '{domain_name}' (cible={n_target} chunks)...")

    is_support_domain = domain_name.lower() == "support"
    if use_fineweb2 and local_dataset_path and os.path.exists(local_dataset_path):
        try:
            ds = load_dataset(
                "parquet",
                data_files={"train": local_dataset_path},
                split="train",
                streaming=False,
            )
            for ex in ds:
                text = ex.get("text", "")
                if not text:
                    continue
                candidate = keyword_match(text, keywords) or (url_patterns and url_match(ex.get("url", ""), url_patterns))
                if is_support_domain and not candidate:
                    candidate = is_expressive_or_support(text, ex.get("url", ""))
                if not candidate:
                    continue
                txt = text.replace("\n", " ").strip()
                chunks = [txt[i: i + chunk_length] for i in range(0, len(txt), chunk_length)][:max_chunks]
                texts.extend(c for c in chunks if len(c) > 100)
                if len(texts) >= n_target:
                    break
            print(f"    -> FineWeb-2 local : {len(texts)} chunks")
        except Exception as e:
            print(f"    [-] Échec FineWeb-2 : {e}")

    if len(texts) < n_target:
        try:
            ds = load_dataset("wikimedia/wikipedia", "20231101.fr", split="train",
                              streaming=True, token=hf_token)
        except Exception:
            local_wiki_dir = "/home/h21486/SAE/datasets/data_wikipedia"
            data_files = sorted(glob.glob(os.path.join(local_wiki_dir, "*.parquet")))
            if not data_files:
                return [re.sub(r"<[^>]+>", "", t).strip() for t in texts][:n_target]
            ds = load_dataset("parquet", data_files=data_files, split="train", streaming=True)

        for ex in ds:
            text = ex.get("text", "")
            if not text:
                continue
            candidate = keyword_match(text, keywords) or (url_patterns and url_match(ex.get("url", ""), url_patterns))
            if is_support_domain and not candidate:
                candidate = is_expressive_or_support(text, ex.get("url", ""))
            if not candidate:
                continue
            txt = text.replace("\n", " ").strip()
            chunks = [txt[i: i + chunk_length] for i in range(0, len(txt), chunk_length)][:max_chunks]
            texts.extend(c for c in chunks if len(c) > 100)
            if len(texts) >= n_target:
                break

    texts = [re.sub(r"<[^>]+>", "", t).strip() for t in texts]
    print(f"  [sae_shared] Terminé. Chunks retenus : {min(len(texts), n_target)}")
    return texts[:n_target]


def split_into_phrases(
    texts: List[str],
    phrase_split: str = r"\.\s+|\n\n",
    min_len: int = 20,
    max_phrases_per_doc: int = None,
) -> Tuple[List[str], List[int]]:
    all_phrases, phrase_to_doc = [], []
    for doc_idx, text in enumerate(texts):
        phrases = [p.strip() for p in re.split(phrase_split, text) if len(p.strip()) > min_len]
        if max_phrases_per_doc:
            phrases = phrases[:max_phrases_per_doc]
        for p in phrases:
            all_phrases.append(p)
            phrase_to_doc.append(doc_idx)
    return all_phrases, phrase_to_doc


def load_and_clean_emails(tsv_path: str) -> Tuple[List[str], List[str]]:
    texts, categories = [], []
    if not os.path.exists(tsv_path):
        print(f"  [sae_shared] Fichier d'emails introuvable : {tsv_path}")
        return [], []
    try:
        try:
            df = pd.read_csv(tsv_path, sep='\t')
            if 'document' not in df.columns or len(df) == 0:
                raise ValueError()
        except Exception:
            df = pd.read_csv(tsv_path, sep=',')
        for _, row in df.iterrows():
            if 'document' not in row or pd.isna(row['document']):
                continue
            raw_text = str(row['document'])
            subject_match = re.search(r'(?:Objet|Subject)\s*:\s*([^\n]+)', raw_text, re.IGNORECASE)
            cat = subject_match.group(1).strip()[:50] if subject_match else "EDF_Mail_Reclamation"
            clean_text = re.sub(r'^\s*(?:Objet|Subject)\s*:\s*[^\n]+\n*', '', raw_text, flags=re.IGNORECASE)
            clean_text = re.sub(r'\[\s*\{\s*"start".*?\}\s*\]', '', clean_text, flags=re.DOTALL).strip()
            if clean_text:
                texts.append(clean_text)
                categories.append(cat)
        print(f"  [sae_shared] {len(texts)} emails chargés depuis {tsv_path}")
        return texts, categories
    except Exception as e:
        print(f"  [sae_shared] Erreur lecture TSV : {e}")
        return [], []
