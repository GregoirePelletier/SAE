"""
src/post_stage/representations.py -- Construction des baselines
independantes E01 (docs/post_stage/PLAN_E00-E09.md §6.3). CORE/
EXTRA/FULL viennent du SAE deja entraine par saev5.py -- ce module construit
les baselines TFIDF/DENSE sur les MEMES documents, avec tous les parametres
explicitement journalises plutot qu'implicites dans le code d'un script.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class TfidfConfig:
    """Reprend `scripts/intent_urgency_probe.py::TfidfVectorizer(max_features
    =20000, sublinear_tf=True)` -- premiere intention du plan (§6.3 :
    "reprendre en première intention la configuration du test de sondes
    existant"). Tout le reste aux defauts sklearn, mais explicite ici plutot
    qu'implicite dans un appel non journalise."""
    analyzer: str = "word"
    ngram_range: Tuple[int, int] = (1, 1)
    min_df: "int | float" = 1
    max_features: Optional[int] = 20000
    sublinear_tf: bool = True
    stop_words: Optional[str] = None
    norm: str = "l2"


def build_tfidf_representation(
    fit_texts: List[str],
    eval_texts_by_split: Dict[str, List[str]],
    config: TfidfConfig = None,
):
    """Ajuste le vectoriseur UNIQUEMENT sur `fit_texts` (jamais DEV/CONFIRM --
    §4.2/§6.3, le vocabulaire/IDF ne doit pas voir ces splits), transforme
    chaque ensemble de `eval_texts_by_split` avec le MEME vectoriseur figé.

    Retourne (vectorizer, {"fit": X_fit, **{split: X_split}}, config_dict) --
    `config_dict` prêt à écrire tel quel dans `representation_manifest.json`."""
    config = config or TfidfConfig()
    vec = TfidfVectorizer(
        analyzer=config.analyzer, ngram_range=config.ngram_range,
        min_df=config.min_df, max_features=config.max_features,
        sublinear_tf=config.sublinear_tf, stop_words=config.stop_words,
        norm=config.norm,
    )
    X_by_split = {"fit": vec.fit_transform(fit_texts)}
    for split_name, texts in eval_texts_by_split.items():
        X_by_split[split_name] = vec.transform(texts)

    config_dict = {
        "analyzer": config.analyzer, "ngram_range": list(config.ngram_range),
        "min_df": config.min_df, "max_features": config.max_features,
        "sublinear_tf": config.sublinear_tf, "stop_words": config.stop_words,
        "norm": config.norm, "vocabulary_size": len(vec.vocabulary_),
    }
    return vec, X_by_split, config_dict


def embed_bge_m3_documents(
    texts: List[str],
    model_path: str = "",
    batch_size: int = 32,
    max_length: int = 2048,
    device: str = "cpu",
    tokenizer=None,
    model=None,
) -> Tuple["torch.Tensor", dict]:
    """Embeddings bge-m3 (pooling [CLS] normalisé -- même convention que
    `saev5.py::_embed_bge_m3`) SUR DES DOCUMENTS ENTIERS -- fonction SÉPARÉE
    de `_embed_bge_m3` (`max_length=64`, conçue pour de courts LABELS de
    features) : le plan (§6.3) interdit explicitement de réutiliser cette
    troncature à 64 tokens pour des emails entiers ("n'est pas acceptable
    pour des emails entiers").

    `max_length=2048` par défaut : généreux au regard du P90 de longueur
    email mesuré (`configs/post_stage/corpus_manifest.json`, ~2000
    caractères ≈ largement sous 2048 tokens). La proportion de documents
    RÉELLEMENT tronqués est mesurée (jamais supposée nulle) et rapportée
    dans `config_dict["truncation_rate"]`.

    `tokenizer`/`model` injectables (défaut : chargés depuis `model_path`) --
    permet de tester la logique de batching/troncature sans charger le vrai
    modèle (interdit sur le nœud frontal, cf. CLAUDE.md)."""
    if tokenizer is None or model is None:
        from transformers import AutoModel, AutoTokenizer
        tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = model or AutoModel.from_pretrained(model_path, local_files_only=True).to(device).eval()

    embs = []
    n_truncated = 0
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            untrunc = tokenizer(batch, truncation=False)
            n_truncated += sum(1 for ids in untrunc["input_ids"] if len(ids) > max_length)
            enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length,
                             return_tensors="pt").to(device)
            cls = model(**enc).last_hidden_state[:, 0]
            embs.append(F.normalize(cls, p=2, dim=-1).cpu())

    config_dict = {
        "model_path": model_path, "pooling": "cls_normalized", "max_length": max_length,
        "n_docs": len(texts), "n_truncated": n_truncated,
        "truncation_rate": (n_truncated / len(texts)) if texts else 0.0,
    }
    return torch.cat(embs, dim=0) if embs else torch.empty(0), config_dict
