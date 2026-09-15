"""Tests CPU rapides pour src/post_stage/dataset_contract.py (gel de corpus,
plan §4.1-4.2). Fixtures synthetiques minimales -- aucune lecture des
donnees reelles (Mails.tsv/augmented_mails.jsonl), aucun tenseur/modele."""
import json
import os

import pytest

from src.post_stage.dataset_contract import (
    assign_parent_splits,
    build_corpus_manifest,
    write_corpus_manifest,
    load_fit_dev_corpus_from_manifest,
    SPLIT_NAMES,
)


def _write_mails_tsv(path, n=20):
    lines = ["\tdocument\tsegments"]
    for i in range(n):
        text = (
            f"Bonjour, ceci est le message de test numero {i}, suffisamment long "
            f"pour passer le filtre de longueur minimale du chargeur de mails."
        )
        lines.append(f"{i}\t{text}\t[]")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _write_augmented_jsonl(path, n_parents=20, variants_per_parent=2, with_parent_sha1=False):
    records = []
    for i in range(n_parents):
        for v in range(variants_per_parent):
            rec = {
                "aug_id": f"{i}__axis__level{v}",
                "parent_id": str(i),
                "corpus": "mail_reel",
                "axis": "registre",
                "level": f"niveau{v}",
                "rejected": None,
                "text": f"Variante {v} du message {i}, texte suffisamment long pour le test.",
            }
            if with_parent_sha1:
                rec["parent_sha1"] = f"fake_hash_{i}"  # jamais reellement joignable, teste juste le chemin
            records.append(rec)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_assign_parent_splits_covers_all_hashes_no_overlap():
    hashes = [f"h{i}" for i in range(1000)]
    split_of = assign_parent_splits(hashes, seed=42, fit=0.6, dev=0.15)
    assert set(split_of.keys()) == set(hashes)
    assert set(split_of.values()) <= set(SPLIT_NAMES)
    counts = {s: sum(1 for v in split_of.values() if v == s) for s in SPLIT_NAMES}
    assert counts["fit"] == 600
    assert counts["dev"] == 150
    assert counts["confirm"] == 250


def test_assign_parent_splits_deterministic_for_same_seed():
    hashes = [f"h{i}" for i in range(200)]
    a = assign_parent_splits(hashes, seed=7)
    b = assign_parent_splits(hashes, seed=7)
    assert a == b


def test_assign_parent_splits_different_seeds_differ():
    hashes = [f"h{i}" for i in range(200)]
    a = assign_parent_splits(hashes, seed=1)
    b = assign_parent_splits(hashes, seed=2)
    assert a != b


def test_assign_parent_splits_rejects_invalid_ratios():
    with pytest.raises(ValueError):
        assign_parent_splits(["a", "b"], fit=0.9, dev=0.2)  # somme >= 1


def test_build_corpus_manifest_parent_counts_match_source(tmp_path):
    mails_path = str(tmp_path / "Mails.tsv")
    _write_mails_tsv(mails_path, n=20)
    manifest, parent_records, variant_records = build_corpus_manifest(
        mails_path, "", fit=0.6, dev=0.15, seed=42
    )
    assert manifest["n_parents"] == 20
    assert sum(manifest["n_parents_by_split"].values()) == 20
    assert manifest["n_duplicate_parent_hashes"] == 0
    assert len(parent_records) == 20
    assert variant_records == []


def test_build_corpus_manifest_variants_use_positional_fallback(tmp_path):
    mails_path = str(tmp_path / "Mails.tsv")
    aug_path = str(tmp_path / "augmented_mails.jsonl")
    _write_mails_tsv(mails_path, n=10)
    _write_augmented_jsonl(aug_path, n_parents=10, variants_per_parent=3, with_parent_sha1=False)

    manifest, parent_records, variant_records = build_corpus_manifest(
        mails_path, aug_path, fit=0.6, dev=0.15, seed=42
    )
    assert manifest["join"]["positional_join_fallback"] is True
    assert manifest["n_variants_total_accepted"] == 30
    assert manifest["n_variants_unmatched"] == 0  # tous les parent_id sont dans [0, 10)
    assert sum(manifest["n_variants_by_split"].values()) == 30
    # Une variante doit toujours porter le MEME split que son mail parent.
    parent_split_by_hash = {r["parent_sha1"]: r["split"] for r in parent_records}
    for v in variant_records:
        assert v["split"] == parent_split_by_hash[v["parent_sha1"]]


def test_build_corpus_manifest_out_of_range_parent_id_is_unmatched(tmp_path):
    mails_path = str(tmp_path / "Mails.tsv")
    aug_path = str(tmp_path / "augmented_mails.jsonl")
    _write_mails_tsv(mails_path, n=5)
    _write_augmented_jsonl(aug_path, n_parents=5, variants_per_parent=1, with_parent_sha1=False)
    # Ajoute une variante orpheline (parent_id hors plage -- mail supprime
    # depuis la generation, cf. le meme garde-fou dans
    # build_email_train_test_corpus).
    with open(aug_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "aug_id": "orphan", "parent_id": "999", "corpus": "mail_reel",
            "axis": "registre", "level": "x", "rejected": None,
            "text": "Variante orpheline sans mail parent correspondant dans Mails.tsv.",
        }) + "\n")

    manifest, _, _ = build_corpus_manifest(mails_path, aug_path, fit=0.6, dev=0.15, seed=42)
    assert manifest["n_variants_total_accepted"] == 6
    assert manifest["n_variants_unmatched"] == 1


def test_write_corpus_manifest_roundtrip(tmp_path):
    mails_path = str(tmp_path / "Mails.tsv")
    _write_mails_tsv(mails_path, n=8)
    manifest, parent_records, variant_records = build_corpus_manifest(
        mails_path, "", fit=0.6, dev=0.15, seed=42
    )
    manifest_path = str(tmp_path / "out" / "corpus_manifest.json")
    split_path = str(tmp_path / "out" / "split_assignments.json")
    write_corpus_manifest(manifest, parent_records, variant_records, manifest_path, split_path)

    assert os.path.exists(manifest_path)
    assert os.path.exists(split_path)
    with open(manifest_path) as f:
        reloaded = json.load(f)
    assert reloaded["n_parents"] == 8
    with open(split_path) as f:
        reloaded_splits = json.load(f)
    assert len(reloaded_splits["parents"]) == 8


def test_write_corpus_manifest_no_raw_text_leaked(tmp_path):
    # §4.1/§18 du plan : un manifeste versionnable ne doit jamais contenir de
    # texte brut -- seulement des sha1/comptages.
    mails_path = str(tmp_path / "Mails.tsv")
    aug_path = str(tmp_path / "augmented_mails.jsonl")
    marker = "CECI_EST_UN_TEXTE_CONFIDENTIEL_QUI_NE_DOIT_JAMAIS_APPARAITRE"
    with open(mails_path, "w", encoding="utf-8") as f:
        f.write("\tdocument\tsegments\n")
        f.write(f"0\t{marker}, suffisamment long pour passer le filtre de longueur.\t[]\n")
    _write_augmented_jsonl(aug_path, n_parents=1, variants_per_parent=1)

    manifest, parent_records, variant_records = build_corpus_manifest(
        mails_path, aug_path, fit=0.6, dev=0.15, seed=42
    )
    manifest_path = str(tmp_path / "corpus_manifest.json")
    split_path = str(tmp_path / "split_assignments.json")
    write_corpus_manifest(manifest, parent_records, variant_records, manifest_path, split_path)

    for path in (manifest_path, split_path):
        with open(path) as f:
            content = f.read()
        assert marker not in content


def _freeze_and_write(tmp_path, mails_path, aug_path, fit=0.6, dev=0.15, seed=42):
    manifest, parent_records, variant_records = build_corpus_manifest(
        mails_path, aug_path, fit=fit, dev=dev, seed=seed
    )
    manifest_path = str(tmp_path / "corpus_manifest.json")
    split_path = str(tmp_path / "split_assignments.json")
    write_corpus_manifest(manifest, parent_records, variant_records, manifest_path, split_path)
    return manifest, split_path


def test_load_fit_dev_corpus_excludes_confirm_entirely(tmp_path):
    mails_path = str(tmp_path / "Mails.tsv")
    aug_path = str(tmp_path / "augmented_mails.jsonl")
    _write_mails_tsv(mails_path, n=100)
    _write_augmented_jsonl(aug_path, n_parents=100, variants_per_parent=2)
    manifest, split_path = _freeze_and_write(tmp_path, mails_path, aug_path)

    fit_texts, fit_labels, dev_texts, dev_labels, fit_groups, dev_groups = (
        load_fit_dev_corpus_from_manifest(split_path, mails_path, aug_path, return_groups=True)
    )

    n_confirm_parents = manifest["n_parents_by_split"]["confirm"]
    n_confirm_variants = manifest["n_variants_by_split"]["confirm"]
    n_fit_parents = manifest["n_parents_by_split"]["fit"]
    n_dev_parents = manifest["n_parents_by_split"]["dev"]
    n_fit_variants = manifest["n_variants_by_split"]["fit"]
    n_dev_variants = manifest["n_variants_by_split"]["dev"]

    total_texts = len(fit_texts) + len(dev_texts)
    total_expected = (
        n_fit_parents + n_dev_parents + n_fit_variants + n_dev_variants
    )
    assert total_texts == total_expected
    assert n_confirm_parents > 0 and n_confirm_variants > 0  # le test doit etre non-trivial
    assert len(fit_texts) == n_fit_parents + n_fit_variants
    assert len(dev_texts) == n_dev_parents + n_dev_variants
    assert len(fit_texts) == len(fit_labels) == len(fit_groups)
    assert len(dev_texts) == len(dev_labels) == len(dev_groups)


def test_load_fit_dev_corpus_max_augmented_per_mail_caps_variants(tmp_path):
    mails_path = str(tmp_path / "Mails.tsv")
    aug_path = str(tmp_path / "augmented_mails.jsonl")
    _write_mails_tsv(mails_path, n=50)
    _write_augmented_jsonl(aug_path, n_parents=50, variants_per_parent=5)
    _, split_path = _freeze_and_write(tmp_path, mails_path, aug_path)

    fit_texts, _, dev_texts, _ = load_fit_dev_corpus_from_manifest(
        split_path, mails_path, aug_path, max_augmented_per_mail=None
    )
    fit_texts_capped, _, dev_texts_capped, _ = load_fit_dev_corpus_from_manifest(
        split_path, mails_path, aug_path, max_augmented_per_mail=1
    )
    assert len(fit_texts_capped) + len(dev_texts_capped) < len(fit_texts) + len(dev_texts)


def test_load_fit_dev_corpus_deterministic_sampling(tmp_path):
    mails_path = str(tmp_path / "Mails.tsv")
    aug_path = str(tmp_path / "augmented_mails.jsonl")
    _write_mails_tsv(mails_path, n=30)
    _write_augmented_jsonl(aug_path, n_parents=30, variants_per_parent=4)
    _, split_path = _freeze_and_write(tmp_path, mails_path, aug_path)

    a = load_fit_dev_corpus_from_manifest(
        split_path, mails_path, aug_path, max_augmented_per_mail=2, sampling_seed=99
    )
    b = load_fit_dev_corpus_from_manifest(
        split_path, mails_path, aug_path, max_augmented_per_mail=2, sampling_seed=99
    )
    assert a == b


def test_load_fit_dev_corpus_no_dev_leakage_into_fit(tmp_path):
    mails_path = str(tmp_path / "Mails.tsv")
    aug_path = str(tmp_path / "augmented_mails.jsonl")
    _write_mails_tsv(mails_path, n=60)
    _write_augmented_jsonl(aug_path, n_parents=60, variants_per_parent=2)
    _, split_path = _freeze_and_write(tmp_path, mails_path, aug_path)

    fit_texts, _, dev_texts, _, fit_groups, dev_groups = load_fit_dev_corpus_from_manifest(
        split_path, mails_path, aug_path, return_groups=True
    )
    # Un meme mail d'origine (groupe) ne doit jamais apparaitre des deux cotes.
    assert set(fit_groups).isdisjoint(set(dev_groups))
    assert set(fit_texts).isdisjoint(set(dev_texts))
