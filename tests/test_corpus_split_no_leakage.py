"""Vérifie la garantie group-aware de build_email_train_test_corpus (aucun
parent_id partagé entre train et test) -- jamais testée programmatiquement
(trou identifié par audit externe, AUDIT_REPO_2026-08-07.md §4.3)."""
import json

from src.data.preparation import build_email_train_test_corpus


def _write_fixture(tmp_path, n_real=40, test_split=0.3):
    tsv = tmp_path / "mails.tsv"
    lines = ["index\tdocument\tsegments"]
    for i in range(n_real):
        lines.append(f"{i}\tCeci est un email de test numero {i} avec assez de caracteres.\tx")
    tsv.write_text("\n".join(lines), encoding="utf-8")

    jsonl = tmp_path / "augmented.jsonl"
    rows = []
    for parent_id in range(n_real):
        for k in range(2):  # 2 variantes par mail parent
            rows.append({
                "aug_id": f"{parent_id}_{k}", "parent_id": parent_id, "corpus": "mail_reel",
                "axis": "emotion", "level": "colere_forte", "prompt_sha1": "x", "model": "x",
                "seed": 0, "temperature": 0.7, "rejected": None,
                "text": f"Variante augmentee {k} du mail {parent_id}, colere et frustration.",
            })
    jsonl.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(tsv), str(jsonl), test_split


def _parent_index_from_text(text: str) -> int:
    # Les deux fixtures encodent le parent_id littéralement dans le texte.
    import re
    m = re.search(r"mail (?:de test )?numero (\d+)| mail (\d+)", text)
    g = m.group(1) or m.group(2)
    return int(g)


def test_no_shared_parent_between_train_and_test(tmp_path):
    tsv, jsonl, test_split = _write_fixture(tmp_path)
    train_texts, train_labels, test_texts, test_labels = build_email_train_test_corpus(
        tsv, jsonl, test_split=test_split, seed=42,
    )
    assert len(test_texts) > 0 and len(train_texts) > 0  # sanity : les deux côtés non vides

    train_parents = {_parent_index_from_text(t) for t in train_texts}
    test_parents = {_parent_index_from_text(t) for t in test_texts}
    assert train_parents.isdisjoint(test_parents), (
        f"Fuite de parent_id entre train/test : {train_parents & test_parents}"
    )


def test_split_is_deterministic_given_same_seed(tmp_path):
    tsv, jsonl, test_split = _write_fixture(tmp_path)
    r1 = build_email_train_test_corpus(tsv, jsonl, test_split=test_split, seed=7)
    r2 = build_email_train_test_corpus(tsv, jsonl, test_split=test_split, seed=7)
    assert r1[2] == r2[2]  # mêmes test_texts pour le même seed


def test_different_seed_can_change_split(tmp_path):
    tsv, jsonl, test_split = _write_fixture(tmp_path)
    r1 = build_email_train_test_corpus(tsv, jsonl, test_split=test_split, seed=1)
    r2 = build_email_train_test_corpus(tsv, jsonl, test_split=test_split, seed=2)
    assert r1[2] != r2[2]  # non-régression sur le découplage SEED/CORPUS_SPLIT_SEED
