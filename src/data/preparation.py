import glob
import hashlib
import json
import os
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset

try:
    from src.data.keywords import SUPPORT_KEYWORDS, SUPPORT_URL_PATTERNS
except ImportError:
    from keywords import SUPPORT_KEYWORDS, SUPPORT_URL_PATTERNS


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


def _chunk_hash(chunk: str) -> str:
    return hashlib.md5(chunk.strip().lower().encode("utf-8")).hexdigest()


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

    # Déduplication : un même document (même URL, ou même contenu réapparaissant
    # sous une autre URL) peut exister plusieurs fois dans un dump FineWeb-2 /
    # Wikipedia (crawls répétés). Sans ce filtre, ses chunks sont ré-ajoutés à
    # l'identique à chaque occurrence, ce qui pollue ensuite les exemples
    # positifs présentés au juge LLM (le même passage y apparaît x2/x3).
    seen_urls = set()
    seen_chunk_hashes = set()

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
                url = ex.get("url", "") or ""
                if url and url in seen_urls:
                    continue
                candidate = keyword_match(text, keywords) or (url_patterns and url_match(url, url_patterns))
                if is_support_domain and not candidate:
                    candidate = is_expressive_or_support(text, url)
                if not candidate:
                    continue
                txt = text.replace("\n", " ").strip()
                chunks = [txt[i: i + chunk_length] for i in range(0, len(txt), chunk_length)][:max_chunks]
                added_any = False
                for c in chunks:
                    if len(c) <= 100:
                        continue
                    h = _chunk_hash(c)
                    if h in seen_chunk_hashes:
                        continue
                    seen_chunk_hashes.add(h)
                    texts.append(c)
                    added_any = True
                if url and added_any:
                    seen_urls.add(url)
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
            local_wiki_dir = os.environ.get("LOCAL_WIKI_DIR", "./local_data/datasets/data_wikipedia")
            data_files = sorted(glob.glob(os.path.join(local_wiki_dir, "*.parquet")))
            if not data_files:
                return [re.sub(r"<[^>]+>", "", t).strip() for t in texts][:n_target]
            ds = load_dataset("parquet", data_files=data_files, split="train", streaming=True)

        for ex in ds:
            text = ex.get("text", "")
            if not text:
                continue
            url = ex.get("url", "") or ""
            if url and url in seen_urls:
                continue
            candidate = keyword_match(text, keywords) or (url_patterns and url_match(url, url_patterns))
            if is_support_domain and not candidate:
                candidate = is_expressive_or_support(text, url)
            if not candidate:
                continue
            txt = text.replace("\n", " ").strip()
            chunks = [txt[i: i + chunk_length] for i in range(0, len(txt), chunk_length)][:max_chunks]
            added_any = False
            for c in chunks:
                if len(c) <= 100:
                    continue
                h = _chunk_hash(c)
                if h in seen_chunk_hashes:
                    continue
                seen_chunk_hashes.add(h)
                texts.append(c)
                added_any = True
            if url and added_any:
                seen_urls.add(url)
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


def build_email_train_test_corpus(
    mails_tsv_path: str,
    augmented_jsonl_path: str,
    test_split: float = 0.05,
    max_augmented_per_mail: int = 13,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Corpus principal d'entraînement du SAE : mails réels + variantes augmentées
    acceptées (cf. décision utilisateur -- emails+augmentés doivent dominer le
    train, au lieu du corpus générique energy/sports/support historique qui
    n'incluait jamais d'email, cf. RESULTS_TESTS.md/Context.md).

    Split GROUP-AWARE par mail d'origine (parent_id) : un mail réel et TOUTES ses
    variantes augmentées tombent ensemble du même côté train/test. Sans ça, une
    variante augmentée d'un mail présent en test fuiterait dans le train (quasi-
    duplicata sémantique) et gonflerait artificiellement les métriques
    (classification, silhouette) -- biais classique de leakage par groupe.

    Retourne (train_texts, train_labels, test_texts, test_labels). Label =
    "original" pour un mail réel, "{axis}__{level}" pour une variante augmentée
    (réutilisable tel quel pour la classification/diffing par axe de perturbation).
    """
    real_texts, _ = load_and_clean_emails(mails_tsv_path)
    if not real_texts:
        return [], [], [], []

    rng = np.random.default_rng(seed)
    n_real = len(real_texts)
    test_mask = rng.random(n_real) < test_split
    parent_split = {i: ("test" if test_mask[i] else "train") for i in range(n_real)}

    train_texts, train_labels = [], []
    test_texts, test_labels = [], []
    for i, text in enumerate(real_texts):
        (test_texts if parent_split[i] == "test" else train_texts).append(text)
        (test_labels if parent_split[i] == "test" else train_labels).append("original")

    if augmented_jsonl_path and os.path.exists(augmented_jsonl_path):
        try:
            from src.data.augmentation import load_augmented
        except ImportError:
            from augmentation import load_augmented
        df_aug = load_augmented(augmented_jsonl_path)
        df_aug = df_aug[df_aug["text"].notna()].copy()
        df_aug["parent_idx"] = df_aug["parent_id"].astype(int)
        df_aug = df_aug[df_aug["parent_idx"] < n_real]  # ignore parents hors plage courante

        if max_augmented_per_mail:
            sampled_idx = np.concatenate([
                rng.choice(group.index.to_numpy(), size=min(len(group), max_augmented_per_mail), replace=False)
                for _, group in df_aug.groupby("parent_idx")
            ])
            df_aug = df_aug.loc[sampled_idx]

        for row in df_aug.itertuples(index=False):
            split = parent_split.get(int(row.parent_idx), "train")
            label = f"{row.aug_axis}__{row.aug_level}"
            if split == "test":
                test_texts.append(row.text)
                test_labels.append(label)
            else:
                train_texts.append(row.text)
                train_labels.append(label)

        print(f"  [corpus] Emails : {n_real} réels + {len(df_aug)} augmentés "
              f"({len(train_texts)} train / {len(test_texts)} test, split par mail d'origine).")
    else:
        print(f"  [corpus] Emails : {n_real} réels, pas de fichier augmenté "
              f"({augmented_jsonl_path!r} absent) -- {len(train_texts)} train / {len(test_texts)} test.")

    return train_texts, train_labels, test_texts, test_labels


def load_and_clean_emails(tsv_path: str) -> Tuple[List[str], List[str]]:
    """Retourne (texts, labels). Le parsing TSV délègue à dataset.load_mails_tsv
    (implémentation unique, quoting-aware, dédupliquée) ; ne subsiste ici que
    l'extraction de l'Objet comme label faible."""
    texts, categories = [], []
    if not os.path.exists(tsv_path):
        print(f"  [sae_shared] Fichier d'emails introuvable : {tsv_path}")
        return [], []
    try:
        try:
            from src.data.dataset import load_mails_tsv
        except ImportError:
            from dataset import load_mails_tsv
        try:
            df = load_mails_tsv(tsv_path).rename(columns={"text": "document"})
            if len(df) == 0:
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