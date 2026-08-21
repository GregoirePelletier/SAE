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
    """Frontières de mot (`\\b...\\b`), pas un `in` sur sous-chaîne -- sans ça
    "vol" matche volume/volley/évolution, "watt" matche Watteau, "avoir"
    matche le verbe (AUDIT_SAE_2026-08.md, item B.11) : la vérité terrain du
    diffing cross-domaine (energy/sports/support) était bruitée par
    construction. Même convention que `INTENT_KEYWORDS_FR`
    (`src/data/dataset.py`)."""
    if not text or not keywords:
        return False
    return any(re.search(r"\b" + re.escape(kw) + r"\b", text, flags=re.IGNORECASE)
               for kw in keywords)


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


def _chunk_on_word_boundaries(txt: str, chunk_length: int, max_chunks: int) -> List[str]:
    """Découpe `txt` en chunks d'environ `chunk_length` caractères, coupés à
    la frontière de mot la plus proche plutôt qu'au milieu d'un mot (B.10,
    AUDIT_SAE_2026-08.md) -- `txt[i:i+chunk_length]` coupait indifféremment
    mots et phrases, une distribution de tokens qui n'existe dans aucun usage
    réel, en particulier pour le filler qui domine le volume d'entraînement
    du SAE résiduel. Un mot isolé plus long que `chunk_length` forme son
    propre chunk plutôt que d'être tronqué au milieu."""
    words = txt.split(" ")
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for w in words:
        if not w:
            continue
        add_len = len(w) + (1 if current else 0)
        if current and current_len + add_len > chunk_length:
            chunks.append(" ".join(current))
            if len(chunks) >= max_chunks:
                return chunks
            current, current_len = [], 0
            add_len = len(w)
        current.append(w)
        current_len += add_len
    if current:
        chunks.append(" ".join(current))
    return chunks[:max_chunks]


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
                streaming=True,
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
                chunks = _chunk_on_word_boundaries(txt, chunk_length, max_chunks)
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
            chunks = _chunk_on_word_boundaries(txt, chunk_length, max_chunks)
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


def sample_fineweb2_chunks(
    n_target: int,
    chunk_length: int = 1024,
    max_chunks: int = 20,
    local_dataset_path: str = None,
) -> List[str]:
    """Filler de volume brut pour l'ablation SAE Boost (arXiv:2507.12990) :
    sous-échantillonnage de FineWeb2-fr SANS filtre thématique -- le filler ne
    sert qu'à isoler l'effet du VOLUME de tokens sur le SAE résiduel (jamais
    ajouté à train_texts, uniquement au réservoir résiduel), la pertinence
    thématique n'y apporte donc rien.
    """
    texts = []
    if not local_dataset_path or not os.path.exists(local_dataset_path):
        return texts
    seen_chunk_hashes = set()
    try:
        ds = load_dataset(
            "parquet", data_files={"train": local_dataset_path}, split="train", streaming=True
        )
        for ex in ds:
            text = ex.get("text", "")
            if not text:
                continue
            txt = text.replace("\n", " ").strip()
            chunks = _chunk_on_word_boundaries(txt, chunk_length, max_chunks)
            for c in chunks:
                if len(c) <= 100:
                    continue
                h = _chunk_hash(c)
                if h in seen_chunk_hashes:
                    continue
                seen_chunk_hashes.add(h)
                texts.append(c)
            if len(texts) >= n_target:
                break
    except Exception as e:
        print(f"    [-] Échec FineWeb-2 (filler) : {e}")
    texts = [re.sub(r"<[^>]+>", "", t).strip() for t in texts]
    print(f"  [filler] {min(len(texts), n_target)} chunks (sans filtre thématique).")
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


def group_indices_by_doc(doc_ids) -> dict:
    """Inverse de `phrase_to_doc` : {doc_idx: [indices de phrase, ordre croissant]}.
    O(n) une seule fois, à calculer avant toute boucle sur les documents --
    remplacer `np.where(doc_ids == doc_idx)` par document (O(n_docs * n)) par
    un lookup dans le dict retourné ici est équivalent (même ensemble
    d'indices, même ordre croissant) mais O(n) au total."""
    groups: dict = {}
    for i, d in enumerate(doc_ids):
        groups.setdefault(int(d), []).append(i)
    return groups


def is_filler_document(doc_global_idx: int, n_train: int, n_filler: int) -> bool:
    """`doc_global_idx` tombe-t-il dans le bloc filler [n_train, n_train+n_filler)
    de `all_texts = train_texts + volume_filler_texts + test_texts + diff_texts`
    (saev5.py) ? Utilisé pour alléger l'extraction côté filler (allocation
    core/fragment économisée, seul le résidu brut compte pour le réservoir,
    cf. AUDIT_SAE_2026-08.md §2.2) -- distinct de la condition, plus large,
    "ce document alimente-t-il le réservoir" (train ∪ filler, doc_global_idx <
    n_train+n_filler), qui reste inchangée ailleurs."""
    return n_train <= doc_global_idx < n_train + n_filler


def build_reencode_targets(n_train: int, n_filler: int, n_total: int) -> list:
    """Indices de documents à traiter par le ré-encodage ExtendedSAE/
    SAEBoostResidualSAE (saev5.py), qui doit ignorer le bloc filler
    [n_train, n_train+n_filler) situé entre train et test dans `all_texts =
    train_texts + volume_filler_texts + test_texts + diff_texts`.

    Le filler ne sert qu'à nourrir le réservoir de résidus PENDANT
    L'EXTRACTION (volume de tokens pour entraîner SAEBoostResidualSAE) --
    aucun consommateur en aval (sélection de features, juge, sondes) ne relit
    jamais sa tranche de `all_doc_sae_acts`/fragments. Le ré-encoder serait un
    travail pur perdu, potentiellement la majorité du corpus (filler
    dominant sur un run à grand volume). Retourne train ∪ (test ∪ diff),
    filler exclu, dans l'ordre d'origine."""
    return list(range(n_train)) + list(range(n_train + n_filler, n_total))


def build_email_train_test_corpus(
    mails_tsv_path: str,
    augmented_jsonl_path: str,
    test_split: float = 0.05,
    max_augmented_per_mail: int = 13,
    seed: int = 42,
    return_groups: bool = False,
):
    """
    Corpus principal d'entraînement du SAE : mails originaux + variantes
    augmentées acceptées. Le corpus générique energy/sports/support ne sert
    plus qu'au diffing post-hoc, jamais à l'entraînement.

    Split GROUP-AWARE par mail d'origine (parent_id) : un mail original et TOUTES ses
    variantes augmentées tombent ensemble du même côté train/test. Sans ça, une
    variante augmentée d'un mail présent en test fuiterait dans le train (quasi-
    duplicata sémantique) et gonflerait artificiellement les métriques
    (classification, silhouette) -- biais classique de leakage par groupe.

    Retourne (train_texts, train_labels, test_texts, test_labels). Label =
    "original" pour un mail original, "{axis}__{level}" pour une variante augmentée
    (réutilisable tel quel pour la classification/diffing par axe de perturbation).

    `return_groups=True` (défaut False, RÉTROCOMPATIBLE -- ~28 appelants existants
    utilisent la signature à 4 valeurs, non touchés) : retourne en plus
    (train_groups, test_groups), le parent_id (mail d'origine) de chaque texte --
    permet une CV group-aware (GroupKFold/StratifiedGroupKFold) en aval,
    `RESULTS_TESTS.md` §57.
    """
    real_texts, _, real_hashes = load_and_clean_emails(mails_tsv_path, return_hashes=True)
    if not real_texts:
        return ([], [], [], [], [], []) if return_groups else ([], [], [], [])

    rng = np.random.default_rng(seed)
    n_real = len(real_texts)
    test_mask = rng.random(n_real) < test_split
    parent_split = {i: ("test" if test_mask[i] else "train") for i in range(n_real)}

    train_texts, train_labels, train_groups = [], [], []
    test_texts, test_labels, test_groups = [], [], []
    for i, text in enumerate(real_texts):
        (test_texts if parent_split[i] == "test" else train_texts).append(text)
        (test_labels if parent_split[i] == "test" else train_labels).append("original")
        (test_groups if parent_split[i] == "test" else train_groups).append(i)

    if augmented_jsonl_path and os.path.exists(augmented_jsonl_path):
        try:
            from src.data.augmentation import load_augmented
        except ImportError:
            from augmentation import load_augmented
        df_aug = load_augmented(augmented_jsonl_path)
        df_aug = df_aug[df_aug["text"].notna()].copy()

        if "parent_sha1" in df_aug.columns:
            # Jointure par CONTENU (B.7) : hash_to_pos construit sur les mêmes
            # hashes que ceux écrits à la génération (augmentation.py::_sha1
            # sur le texte AVANT strip de l'Objet, cf. load_and_clean_emails).
            # Un décalage de filtrage entre le run d'augmentation et ce run ne
            # peut plus mal-attribuer une variante -- au pire elle est écartée
            # (parent introuvable), jamais rattachée au mauvais mail.
            hash_to_pos = {h: i for i, h in enumerate(real_hashes)}
            df_aug["parent_idx"] = df_aug["parent_sha1"].map(hash_to_pos)
            n_unmatched = int(df_aug["parent_idx"].isna().sum())
            if n_unmatched:
                print(f"  [corpus] {n_unmatched} variante(s) augmentée(s) sans mail parent "
                      f"correspondant (Mails.tsv modifié depuis la génération ?) -- écartées.")
            df_aug = df_aug[df_aug["parent_idx"].notna()].copy()
            df_aug["parent_idx"] = df_aug["parent_idx"].astype(int)
        else:
            # Repli rétrocompatible : JSONL généré avant l'ajout de parent_sha1,
            # jointure positionnelle (fragile, cf. AUDIT_SAE_2026-08.md item B.7).
            print("  [corpus] parent_sha1 absent du JSONL augmenté -- jointure positionnelle "
                  "de repli (regénérer le corpus augmenté pour la jointure par contenu).")
            df_aug["parent_idx"] = df_aug["parent_id"].astype(int)
            df_aug = df_aug[df_aug["parent_idx"] < n_real]  # ignore parents hors plage courante

        if max_augmented_per_mail and len(df_aug):
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
                test_groups.append(int(row.parent_idx))
            else:
                train_texts.append(row.text)
                train_labels.append(label)
                train_groups.append(int(row.parent_idx))

        print(f"  [corpus] Emails : {n_real} réels + {len(df_aug)} augmentés "
              f"({len(train_texts)} train / {len(test_texts)} test, split par mail d'origine).")
    else:
        print(f"  [corpus] Emails : {n_real} réels, pas de fichier augmenté "
              f"({augmented_jsonl_path!r} absent) -- {len(train_texts)} train / {len(test_texts)} test.")

    if return_groups:
        return train_texts, train_labels, test_texts, test_labels, train_groups, test_groups
    return train_texts, train_labels, test_texts, test_labels


def load_and_clean_emails(tsv_path: str, return_hashes: bool = False):
    """Retourne (texts, labels). Le parsing TSV délègue à dataset.load_mails_tsv
    (implémentation unique, quoting-aware, dédupliquée) ; ne subsiste ici que
    l'extraction de l'Objet comme label faible.

    `return_hashes=True` (défaut False, RÉTROCOMPATIBLE) : retourne en plus
    `hashes`, le SHA1 (`_sha1`, `src/data/augmentation.py`) du texte AVANT
    strip de l'Objet -- identique à `row.text` dans `run_augmentation.py`
    (même `load_mails_tsv(tsv_path)`, même colonne, avant tout nettoyage
    supplémentaire). Permet à `build_email_train_test_corpus` de rattacher
    une variante augmentée à son mail parent par CONTENU plutôt que par
    position (AUDIT_SAE_2026-08.md, item B.7)."""
    texts, categories, hashes = [], [], []
    empty = ([], [], []) if return_hashes else ([], [])
    if not os.path.exists(tsv_path):
        print(f"  [sae_shared] Fichier d'emails introuvable : {tsv_path}")
        return empty
    try:
        try:
            from src.data.dataset import load_mails_tsv
        except ImportError:
            from dataset import load_mails_tsv
        try:
            from src.data.augmentation import _sha1
        except ImportError:
            from augmentation import _sha1
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
                hashes.append(_sha1(raw_text))
        print(f"  [sae_shared] {len(texts)} emails chargés depuis {tsv_path}")
        return (texts, categories, hashes) if return_hashes else (texts, categories)
    except Exception as e:
        print(f"  [sae_shared] Erreur lecture TSV : {e}")
        return empty