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
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

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


def load_fit_dev_corpus_from_manifest(
    split_assignments_path: str,
    mails_tsv_path: str,
    augmented_jsonl_path: str,
    max_augmented_per_mail: Optional[int] = None,
    sampling_seed: int = POST_STAGE_SPLIT_SEED,
    return_groups: bool = False,
) -> Tuple:
    """Charge FIT (role "train") et DEV (role "test") depuis le split DEJA
    GELE (`split_assignments_path`, ecrit une fois par `write_corpus_manifest`)
    -- ne recalcule jamais un nouveau split. CONFIRM est totalement absent du
    resultat (§4.2 : CONFIRM n'entre jamais dans un encodeur entraine).

    Meme forme de retour que `src.data.preparation.build_email_train_test_corpus`
    (train_texts, train_labels, test_texts, test_labels[, train_groups,
    test_groups]) pour rester un remplacement direct de son unique appelant
    dans `saev5.py` -- FIT joue le role "train", DEV le role "test"/validation
    interne, dans les memes structures de donnees que le pipeline existant
    consomme deja."""
    with open(split_assignments_path) as f:
        assignments = json.load(f)
    parent_split_by_hash = {r["parent_sha1"]: r["split"] for r in assignments["parents"]}
    variant_split_by_aug_id = {r["aug_id"]: r["split"] for r in assignments["variants"]}

    real_texts, _, real_hashes = load_and_clean_emails(mails_tsv_path, return_hashes=True)

    fit_texts, fit_labels, fit_groups = [], [], []
    dev_texts, dev_labels, dev_groups = [], [], []
    for i, (h, text) in enumerate(zip(real_hashes, real_texts)):
        split = parent_split_by_hash.get(h)
        if split == "fit":
            fit_texts.append(text); fit_labels.append("original"); fit_groups.append(i)
        elif split == "dev":
            dev_texts.append(text); dev_labels.append("original"); dev_groups.append(i)
        # confirm (ou parent absent du manifeste, ex. Mails.tsv modifie depuis
        # le gel) : jamais inclus ici.

    if augmented_jsonl_path and os.path.exists(augmented_jsonl_path):
        df_aug = load_augmented(augmented_jsonl_path)
        df_aug = df_aug[df_aug["text"].notna()].copy()
        # "resolved_split", pas "_split" : itertuples(index=False) renomme
        # silencieusement tout nom de colonne commencant par "_" (reserve aux
        # champs positionnels du namedtuple sous-jacent), row._split n'aurait
        # alors plus jamais existe -- AttributeError attrape par le test
        # (test_load_fit_dev_corpus_no_dev_leakage_into_fit et consorts).
        df_aug["resolved_split"] = df_aug["aug_id"].map(variant_split_by_aug_id)
        # Un parent n'appartient qu'a UN split : toutes ses variantes portent
        # deja le meme resolved_split (resolu une fois pour toutes dans le
        # manifeste, cf. build_corpus_manifest) -- filtrer ici exclut confirm
        # ET tout aug_id absent du manifeste (variante orpheline/non gelee).
        df_aug = df_aug[df_aug["resolved_split"].isin(("fit", "dev"))]

        if max_augmented_per_mail and len(df_aug):
            rng = np.random.default_rng(sampling_seed)
            sampled_frames = [
                group.loc[rng.choice(group.index.to_numpy(),
                                      size=min(len(group), max_augmented_per_mail),
                                      replace=False)]
                for _, group in df_aug.groupby("parent_id")
            ]
            df_aug = pd.concat(sampled_frames)

        for row in df_aug.itertuples(index=False):
            label = f"{row.aug_axis}__{row.aug_level}"
            try:
                parent_idx = int(row.parent_id)
            except (TypeError, ValueError):
                parent_idx = -1
            if row.resolved_split == "fit":
                fit_texts.append(row.text); fit_labels.append(label); fit_groups.append(parent_idx)
            else:
                dev_texts.append(row.text); dev_labels.append(label); dev_groups.append(parent_idx)

        print(f"  [post_stage] Corpus FIT/DEV geles : {len(fit_texts)} FIT / "
              f"{len(dev_texts)} DEV (CONFIRM exclu).")

    if return_groups:
        return fit_texts, fit_labels, dev_texts, dev_labels, fit_groups, dev_groups
    return fit_texts, fit_labels, dev_texts, dev_labels


def load_confirm_corpus_from_manifest(
    split_assignments_path: str,
    mails_tsv_path: str,
    augmented_jsonl_path: str,
    max_augmented_per_mail: Optional[int] = None,
    sampling_seed: int = POST_STAGE_SPLIT_SEED,
    return_groups: bool = False,
):
    """Charge UNIQUEMENT CONFIRM depuis le split déjà gelé -- séparée de
    `load_fit_dev_corpus_from_manifest` à dessein (§4.2 : CONFIRM ne doit
    jamais entrer dans un encodeur entraîné, une IDF, une PCA
    d'initialisation, un seuil de regroupement, un choix de requête ou un
    réglage de sonde). N'appeler cette fonction que pour une évaluation
    finale sur un modèle/protocole déjà figé -- jamais pour entraîner, choisir
    un hyperparamètre ou geler une requête.

    Même forme de retour que `load_fit_dev_corpus_from_manifest`, mais un
    seul ensemble (confirm_texts, confirm_labels[, confirm_groups]) --
    utilisable comme `diff_texts` de `saev5.py` (corpus tenu à l'écart de
    l'entraînement, réencodé post-hoc par le SAE déjà figé, même mécanisme
    que le corpus energy/sports/support historique)."""
    with open(split_assignments_path) as f:
        assignments = json.load(f)
    parent_split_by_hash = {r["parent_sha1"]: r["split"] for r in assignments["parents"]}
    variant_split_by_aug_id = {r["aug_id"]: r["split"] for r in assignments["variants"]}

    real_texts, _, real_hashes = load_and_clean_emails(mails_tsv_path, return_hashes=True)

    confirm_texts, confirm_labels, confirm_groups = [], [], []
    for i, (h, text) in enumerate(zip(real_hashes, real_texts)):
        if parent_split_by_hash.get(h) == "confirm":
            confirm_texts.append(text); confirm_labels.append("original"); confirm_groups.append(i)

    if augmented_jsonl_path and os.path.exists(augmented_jsonl_path):
        df_aug = load_augmented(augmented_jsonl_path)
        df_aug = df_aug[df_aug["text"].notna()].copy()
        df_aug["resolved_split"] = df_aug["aug_id"].map(variant_split_by_aug_id)
        df_aug = df_aug[df_aug["resolved_split"] == "confirm"]

        if max_augmented_per_mail and len(df_aug):
            rng = np.random.default_rng(sampling_seed)
            sampled_frames = [
                group.loc[rng.choice(group.index.to_numpy(),
                                      size=min(len(group), max_augmented_per_mail),
                                      replace=False)]
                for _, group in df_aug.groupby("parent_id")
            ]
            df_aug = pd.concat(sampled_frames)

        for row in df_aug.itertuples(index=False):
            label = f"{row.aug_axis}__{row.aug_level}"
            try:
                parent_idx = int(row.parent_id)
            except (TypeError, ValueError):
                parent_idx = -1
            confirm_texts.append(row.text); confirm_labels.append(label); confirm_groups.append(parent_idx)

        print(f"  [post_stage] Corpus CONFIRM gelé : {len(confirm_texts)} documents "
              "(FIT/DEV non chargés par cette fonction).")

    if return_groups:
        return confirm_texts, confirm_labels, confirm_groups
    return confirm_texts, confirm_labels
