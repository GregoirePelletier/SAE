"""
src/post_stage/dataset_contract.py -- Gel du corpus de la campagne
post-soutenance (Plan_execution_SAE_15_jours_Claude_Code.md §4.1-4.2).

Un seul contrat de donnees pour toute la campagne : split FIT/DEV/CONFIRM
PARENT-AWARE (un mail d'origine et toutes ses variantes augmentees tombent
du meme cote), identifie par le sha1 de CONTENU du mail parent (jamais par
numero de ligne -- reutilise load_and_clean_emails(return_hashes=True),
deja construit ainsi dans src/data/preparation.py pour la meme raison,
AUDIT_SAE_2026-08.md item B.7). CONFIRM ne doit jamais entrer dans un
encodeur entraine, une IDF, une PCA d'initialisation, un seuil de
regroupement, un choix de requete ou un reglage de sonde -- §4.2 du plan.

Le manifeste ecrit ici ne contient jamais de texte brut ni d'identifiant
client (uniquement des sha1 et des comptages) : safe a versionner
(configs/post_stage/).
"""
import hashlib
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np

try:
    from src.data.preparation import load_and_clean_emails
except ImportError:
    from preparation import load_and_clean_emails

try:
    from src.data.augmentation import load_augmented
except ImportError:
    from augmentation import load_augmented


# Graine DEDIEE au split de campagne, decouplee de SEED/CORPUS_SPLIT_SEED
# (src/config.py) -- le split FIT/DEV/CONFIRM ne doit jamais varier avec une
# ablation de graine SAE/init/shuffle (plan §4.2 : "Fixer la graine du split
# une fois, independamment des graines du SAE").
POST_STAGE_SPLIT_SEED = 20260914

SPLIT_NAMES = ("fit", "dev", "confirm")


def assign_parent_splits(
    parent_hashes: List[str],
    seed: int = POST_STAGE_SPLIT_SEED,
    fit: float = 0.60,
    dev: float = 0.15,
) -> Dict[str, str]:
    """FIT/DEV/CONFIRM par mail d'origine, identifie par son sha1 de contenu
    (jamais par position dans le fichier source). `confirm = 1 - fit - dev`.
    Tri des hashes avant tirage : l'assignation ne depend pas de l'ordre
    d'iteration du DataFrame source, seulement du contenu et de `seed`."""
    if not (0 < fit < 1) or not (0 <= dev < 1) or fit + dev >= 1:
        raise ValueError(f"fit={fit}, dev={dev} : attendu 0<fit<1, 0<=dev, fit+dev<1")
    sorted_hashes = sorted(set(parent_hashes))
    n = len(sorted_hashes)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_fit = int(round(fit * n))
    n_dev = int(round(dev * n))
    split_of: Dict[str, str] = {}
    for rank, idx in enumerate(perm):
        h = sorted_hashes[idx]
        if rank < n_fit:
            split_of[h] = "fit"
        elif rank < n_fit + n_dev:
            split_of[h] = "dev"
        else:
            split_of[h] = "confirm"
    return split_of


def build_corpus_manifest(
    mails_tsv_path: str,
    augmented_jsonl_path: str,
    fit: float = 0.60,
    dev: float = 0.15,
    seed: int = POST_STAGE_SPLIT_SEED,
) -> dict:
    """Construit le manifeste complet (§4.1) : comptages par split, sha1 par
    parent (fingerprints), longueurs en caracteres, detection de doublons.
    Aucune lecture de tenseur/modele -- CPU pur sur du texte deja en memoire,
    borne par la taille du corpus emails (quelques dizaines de Mo), compatible
    avec une verification de configuration frontale (CLAUDE.md)."""
    real_texts, _, real_hashes = load_and_clean_emails(mails_tsv_path, return_hashes=True)
    n_parents = len(real_texts)

    n_duplicate_parent_hashes = len(real_hashes) - len(set(real_hashes))

    split_of = assign_parent_splits(real_hashes, seed=seed, fit=fit, dev=dev)

    parent_records = []
    char_lengths_by_split = {s: [] for s in SPLIT_NAMES}
    for h, text in zip(real_hashes, real_texts):
        s = split_of.get(h)
        parent_records.append({"parent_sha1": h, "split": s, "n_chars": len(text)})
        if s:
            char_lengths_by_split[s].append(len(text))

    n_parents_by_split = {s: sum(1 for r in parent_records if r["split"] == s) for s in SPLIT_NAMES}

    variant_records = []
    n_variants_total = 0
    n_variants_unmatched = 0
    variant_char_lengths_by_split = {s: [] for s in SPLIT_NAMES}
    # Repli positionnel (meme convention que build_email_train_test_corpus,
    # AUDIT_SAE_2026-08.md item B.7) : augmented_mails.jsonl actuel n'ecrit
    # PAS parent_sha1 (verifie sur le corpus reel malgre le support cote
    # code) -- pos_to_hash traduit son parent_id (index Mails.tsv) vers le
    # hash de contenu deja assigne a un split ci-dessus, plutot que de
    # traiter ce cas comme un echec de jointure.
    pos_to_hash = {i: h for i, h in enumerate(real_hashes)}
    positional_join_fallback = False
    if augmented_jsonl_path and os.path.exists(augmented_jsonl_path):
        df_aug = load_augmented(augmented_jsonl_path)
        df_aug = df_aug[df_aug["text"].notna()].copy()
        n_variants_total = len(df_aug)
        has_parent_sha1 = "parent_sha1" in df_aug.columns
        positional_join_fallback = not has_parent_sha1
        for row in df_aug.itertuples(index=False):
            if has_parent_sha1:
                parent_hash = row.parent_sha1
            else:
                try:
                    parent_hash = pos_to_hash.get(int(row.parent_id))
                except (TypeError, ValueError):
                    parent_hash = None
            s = split_of.get(parent_hash) if parent_hash is not None else None
            if s is None:
                n_variants_unmatched += 1
                continue
            variant_records.append({
                "aug_id": row.aug_id, "parent_sha1": parent_hash, "split": s,
                "n_chars": len(row.text),
            })
            variant_char_lengths_by_split[s].append(len(row.text))

    n_variants_by_split = {s: sum(1 for r in variant_records if r["split"] == s) for s in SPLIT_NAMES}

    def _length_stats(lengths: List[int]) -> dict:
        if not lengths:
            return {"n": 0, "mean": None, "p50": None, "p90": None, "max": None}
        arr = np.asarray(lengths)
        return {
            "n": int(arr.shape[0]), "mean": float(arr.mean()),
            "p50": float(np.percentile(arr, 50)), "p90": float(np.percentile(arr, 90)),
            "max": int(arr.max()),
        }

    manifest = {
        "schema_version": "post-stage-corpus-v1",
        "generated_at_unix": time.time(),
        "source": {
            "mails_tsv_path": mails_tsv_path,
            "augmented_jsonl_path": augmented_jsonl_path,
            "status": "synthetic",
        },
        "split_config": {"seed": seed, "fit": fit, "dev": dev, "confirm": round(1 - fit - dev, 6)},
        "join": {
            "method": "content_sha1" if not positional_join_fallback else "positional_via_parent_id",
            "positional_join_fallback": positional_join_fallback,
        },
        "n_parents": n_parents,
        "n_duplicate_parent_hashes": n_duplicate_parent_hashes,
        "n_parents_by_split": n_parents_by_split,
        "n_variants_total_accepted": n_variants_total,
        "n_variants_by_split": n_variants_by_split,
        "n_variants_unmatched": n_variants_unmatched,
        "parent_char_length_stats_by_split": {s: _length_stats(char_lengths_by_split[s]) for s in SPLIT_NAMES},
        "variant_char_length_stats_by_split": {s: _length_stats(variant_char_lengths_by_split[s]) for s in SPLIT_NAMES},
        "token_length_stats_by_split": None,  # a completer (tokenizer requis, hors etape frontale -- cf. docstring module)
    }
    return manifest, parent_records, variant_records


def write_corpus_manifest(
    manifest: dict, parent_records: List[dict], variant_records: List[dict],
    manifest_path: str, split_assignments_path: str,
) -> None:
    """Ecriture atomique (tmp + os.replace, R1) -- un manifeste present n'est
    valide que s'il a ete integralement ecrit."""
    for path, payload in (
        (manifest_path, manifest),
        (split_assignments_path, {"parents": parent_records, "variants": variant_records}),
    ):
        tmp_path = path + ".tmp"
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(tmp_path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        os.replace(tmp_path, path)
