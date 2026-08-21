"""Teste src/data/dataset.py::load_mails_tsv -- compte les lignes malformées
plutôt que de les ignorer sans trace (AUDIT_SAE_2026-08.md, item A3 :
`on_bad_lines="skip"` sans compteur, nombre de mails perdus à l'ingestion
inconnu)."""
from src.data.dataset import load_mails_tsv


def test_load_mails_tsv_counts_skipped_bad_lines(tmp_path):
    path = tmp_path / "mails.tsv"
    # 3 colonnes attendues (index, document, segments) ; la 3e ligne a un tab
    # en trop -> ligne malformée, doit être comptée et ignorée, pas juste
    # silencieusement absorbée.
    path.write_text(
        "index\tdocument\tsegments\n"
        "0\tObjet : Facture\\nBonjour, ma facture est trop élevée.\t[]\n"
        "1\tObjet : Compteur\\nMerci de planifier l'installation.\t[]\n"
        "2\tligne\tmalformée\tavec\ttrop\tde\tcolonnes\n",
        encoding="utf-8",
    )
    df = load_mails_tsv(str(path), min_chars=1)
    assert df.attrs["n_bad_lines_skipped"] == 1
    assert len(df) == 2  # les 2 lignes bien formées survivent


def test_load_mails_tsv_zero_bad_lines_on_clean_file(tmp_path):
    path = tmp_path / "mails_clean.tsv"
    path.write_text(
        "index\tdocument\tsegments\n"
        "0\tObjet : Facture\\nBonjour, ma facture est trop élevée.\t[]\n",
        encoding="utf-8",
    )
    df = load_mails_tsv(str(path), min_chars=1)
    assert df.attrs["n_bad_lines_skipped"] == 0
    assert len(df) == 1
