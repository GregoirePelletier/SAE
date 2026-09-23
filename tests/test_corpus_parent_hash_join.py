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


def test_positional_fallback_survives_load_and_clean_emails_extra_filter(tmp_path):
    """`load_and_clean_emails` applique un filtre SUPPLEMENTAIRE a celui de
    `load_mails_tsv` (une ligne dont le texte devient vide apres strip_leading_
    objet_line + suppression du motif [{"start"...}] est ecartee ICI, pas dans
    load_mails_tsv) -- sur le corpus reel, 6 lignes sur 3480 sont dans ce cas.
    `run_augmentation.py` numerote parent_id sur la sortie BRUTE de
    load_mails_tsv, AVANT ce filtre : repli positionnel doit traduire via la
    position d'origine (load_and_clean_emails(return_positions=True)), pas
    re-enumerer 0..n-1 les lignes survivantes -- sinon toute variante dont le
    parent_id suit une ligne filtree en plus est silencieusement rattachee au
    MAUVAIS mail voisin (pas juste ecartee -- une mauvaise attribution, pas
    une absence)."""
    mail_a = "Bonjour mail A, ceci est un message assez long pour le filtre."
    mail_b = "Bonjour mail B, ceci est un message assez long pour le filtre."
    # Devient vide apres strip_leading_objet_line (toute la ligne EST l'objet)
    # -- survit au filtre min_chars=30 de load_mails_tsv (survit donc dans
    # l'espace ou run_augmentation.py numerote doc_id/parent_id) mais pas au
    # filtre supplementaire de load_and_clean_emails.
    mail_objet_only = "Objet: ceci est un sujet suffisamment long pour le filtre."
    mail_c = "Bonjour mail C, ceci est un message assez long pour le filtre."
    mail_d = "Bonjour mail D, ceci est un message assez long pour le filtre."

    tsv = _write_tsv(tmp_path, {0: mail_a, 1: mail_b, 2: mail_objet_only, 3: mail_c, 4: mail_d})

    # load_mails_tsv garde les 5 lignes (toutes >= 30 caracteres) ; load_and_
    # clean_emails n'en garde que 4 (mail_objet_only devient vide).
    real_texts, _, real_hashes, real_positions = load_and_clean_emails(
        tsv, return_hashes=True, return_positions=True)
    assert real_texts == [mail_a, mail_b, mail_c, mail_d]
    assert real_positions == [0, 1, 3, 4]  # position 2 (mail_objet_only) absente

    # Variante generee pour mail_c, dont le parent_id RAW (espace load_mails_tsv,
    # position 3) est celui ecrit par run_augmentation.py -- PAS l'index 2
    # qu'aurait mail_c dans l'espace re-enumere real_texts.
    jsonl = tmp_path / "augmented.jsonl"
    rows = [{
        "aug_id": "3_v0", "parent_id": 3,  # pas de parent_sha1 -> repli positionnel
        "corpus": "mail_reel", "axis": "registre", "level": "formel",
        "prompt_sha1": "x", "model": "x", "seed": 0, "temperature": 0.7,
        "rejected": None, "text": "Variante augmentee du mail C, sur le meme sujet.",
    }]
    jsonl.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    train_texts, train_labels, test_texts, test_labels, train_groups, test_groups = (
        build_email_train_test_corpus(tsv, str(jsonl), test_split=0.0, seed=0, return_groups=True)
    )
    variant_idx = train_labels.index("registre__formel")
    parent_text_of_variant = train_texts[train_groups[variant_idx]]
    assert parent_text_of_variant == mail_c, (
        f"La variante generee pour mail_c (parent_id=3, position brute) doit se "
        f"rattacher a mail_c, pas a {parent_text_of_variant!r} (mail_d si le "
        "repli positionnel re-enumere naivement les lignes survivantes)."
    )


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
