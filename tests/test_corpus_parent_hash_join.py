"""Teste le correctif B.7 (AUDIT_SAE_2026-08.md) de
src/data/preparation.py::build_email_train_test_corpus : la jointure
mail-parent <-> variante augmentée se fait maintenant par SHA1 du texte
parent (`parent_sha1`), pas par position (`parent_id` positionnel), qui
peut se désynchroniser si le filtrage diffère, même légèrement, entre le
run d'augmentation et le run d'entraînement."""
import json

from src.data.augmentation import _sha1
from src.data.preparation import build_email_train_test_corpus, load_and_clean_emails


def _write_tsv(tmp_path, mails: dict[int, str], name="mails.tsv"):
    tsv = tmp_path / name
    lines = ["index\tdocument\tsegments"]
    for idx, text in mails.items():
        lines.append(f"{idx}\t{text}\tx")
    tsv.write_text("\n".join(lines), encoding="utf-8")
    return str(tsv)


def test_hash_join_survives_positional_desync(tmp_path):
    """Simule le scénario B.7 : le TSV utilisé pour l'entraînement a un mail
    de plus en tête (ex. ajouté après la génération des variantes) --
    l'index positionnel de tous les mails suivants décale de 1, mais le
    SHA1 du texte, lui, ne change pas."""
    mail_a = "Bonjour, je conteste ma facture d'electricite tres elevee ce mois."
    mail_b = "Merci de planifier l'installation de mon nouveau compteur electrique."

    # TSV "au moment de l'augmentation" : mail_a en position 0, mail_b en position 1.
    tsv_at_augmentation = {0: mail_a, 1: mail_b}
    gen_hashes = {idx: _sha1(text) for idx, text in tsv_at_augmentation.items()}

    # JSONL généré à partir de cet ordre : parent_id positionnel ET parent_sha1.
    jsonl = tmp_path / "augmented.jsonl"
    rows = [{
        "aug_id": "1_v0", "parent_id": 1, "parent_sha1": gen_hashes[1],
        "corpus": "mail_reel", "axis": "emotion", "level": "colere_forte",
        "prompt_sha1": "x", "model": "x", "seed": 0, "temperature": 0.7,
        "rejected": None, "text": "Variante augmentee du mail sur le COMPTEUR electrique.",
    }]
    jsonl.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    # TSV "au moment de l'entraînement" : un mail supplémentaire inséré en tête
    # -> mail_a passe en position 1, mail_b en position 2. L'ancien parent_id=1
    # positionnel pointerait maintenant vers mail_a (le mauvais mail parent).
    mail_new = "Question generale sur les horaires d'ouverture de l'agence."
    tsv_at_training = _write_tsv(tmp_path, {0: mail_new, 1: mail_a, 2: mail_b})

    real_texts, _, real_hashes = load_and_clean_emails(tsv_at_training, return_hashes=True)
    pos_of_mail_b = real_texts.index(mail_b)
    pos_of_mail_a = real_texts.index(mail_a)
    assert pos_of_mail_b != 1 or pos_of_mail_a != 0  # le décalage a bien eu lieu

    # test_split=0 : tout en train, on regarde juste où la variante atterrit.
    train_texts, train_labels, test_texts, test_labels, train_groups, test_groups = (
        build_email_train_test_corpus(tsv_at_training, str(jsonl), test_split=0.0,
                                       seed=0, return_groups=True)
    )
    variant_idx = train_labels.index("emotion__colere_forte")
    assert train_groups[variant_idx] == pos_of_mail_b, (
        "La variante doit être rattachée à mail_b (son vrai parent par contenu), "
        "pas à la position stale issue du run d'augmentation."
    )
    assert train_groups[variant_idx] != pos_of_mail_a


def test_unmatched_parent_hash_is_dropped_not_misattributed(tmp_path):
    tsv = _write_tsv(tmp_path, {0: "Bonjour, ma facture est trop elevee ce mois-ci."})
    jsonl = tmp_path / "augmented.jsonl"
    rows = [{
        "aug_id": "x", "parent_id": 0, "parent_sha1": "0" * 16,  # ne matche aucun mail réel
        "corpus": "mail_reel", "axis": "emotion", "level": "colere_forte",
        "prompt_sha1": "x", "model": "x", "seed": 0, "temperature": 0.7,
        "rejected": None, "text": "Variante orpheline.",
    }]
    jsonl.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    train_texts, train_labels, test_texts, test_labels = build_email_train_test_corpus(
        tsv, str(jsonl), test_split=0.0, seed=0,
    )
    assert "Variante orpheline." not in train_texts
    assert "Variante orpheline." not in test_texts


def test_backward_compatible_positional_fallback_without_parent_sha1(tmp_path):
    """JSONL généré avant l'ajout de parent_sha1 (colonne absente) : repli sur
    la jointure positionnelle historique, comportement inchangé."""
    tsv = _write_tsv(tmp_path, {0: "Bonjour, ma facture est trop elevee ce mois-ci."})
    jsonl = tmp_path / "augmented.jsonl"
    rows = [{
        "aug_id": "x", "parent_id": 0,  # pas de parent_sha1
        "corpus": "mail_reel", "axis": "emotion", "level": "colere_forte",
        "prompt_sha1": "x", "model": "x", "seed": 0, "temperature": 0.7,
        "rejected": None, "text": "Variante par jointure positionnelle.",
    }]
    jsonl.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    train_texts, train_labels, test_texts, test_labels = build_email_train_test_corpus(
        tsv, str(jsonl), test_split=0.0, seed=0,
    )
    assert "Variante par jointure positionnelle." in train_texts
