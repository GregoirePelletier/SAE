"""Teste src/data/augmentation.py::load_augmented -- non-régression du passage
à un filtrage en un seul passage (AUDIT_SAE_2026-08.md, item A6 : la version
précédente construisait la liste complète des enregistrements acceptés ET
rejetés avant de filtrer, un aller-retour RAM inutile dans un process qui
tient déjà le réservoir memmap et all_doc_sae_acts)."""
import json

from src.data.augmentation import load_augmented


def _write_jsonl(tmp_path, rows):
    path = tmp_path / "augmented.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(path)


def test_load_augmented_filters_rejected_and_keeps_accepted(tmp_path):
    rows = [
        {"text": "Objet : Facture\nBonjour, ma facture.", "rejected": None,
         "corpus": "emails", "axis": "familier", "level": 1, "parent_id": 0},
        {"text": "Variante rejetée", "rejected": "faits fabriqués",
         "corpus": "emails", "axis": "familier", "level": 2, "parent_id": 0},
        {"text": "Objet : Compteur\nMerci de planifier.", "rejected": None,
         "corpus": "emails", "axis": "degrade_fort", "level": 1, "parent_id": 1},
    ]
    path = _write_jsonl(tmp_path, rows)
    df = load_augmented(path)

    assert len(df) == 2  # la variante rejetée est exclue
    assert set(df.columns) >= {"text", "is_augmented", "corpus_origin", "aug_axis", "aug_level"}
    assert df["is_augmented"].all()
    # "Objet :" en tête retiré par _strip_leading_objet_line.
    assert not df["text"].str.startswith("Objet").any()


def test_load_augmented_skips_blank_lines(tmp_path):
    path = tmp_path / "augmented.jsonl"
    path.write_text(
        json.dumps({"text": "a", "rejected": None, "corpus": "c", "axis": "x",
                    "level": 1, "parent_id": 0})
        + "\n\n"  # ligne vide
        + json.dumps({"text": "b", "rejected": None, "corpus": "c", "axis": "x",
                      "level": 1, "parent_id": 0})
        + "\n",
        encoding="utf-8",
    )
    df = load_augmented(str(path))
    assert len(df) == 2


def test_load_augmented_empty_file_returns_empty_df(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    df = load_augmented(str(path))
    assert len(df) == 0
