"""Test CPU rapide du point d'entree CLI (freeze-corpus), fixtures synthetiques."""
import json
import os

from src.post_stage.cli import build_parser


def _write_mails_tsv(path, n=10):
    lines = ["\tdocument\tsegments"]
    for i in range(n):
        text = (
            f"Bonjour, ceci est le message de test numero {i}, suffisamment long "
            f"pour passer le filtre de longueur minimale du chargeur de mails."
        )
        lines.append(f"{i}\t{text}\t[]")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def test_freeze_corpus_cli_end_to_end(tmp_path):
    mails_path = str(tmp_path / "Mails.tsv")
    _write_mails_tsv(mails_path, n=10)
    manifest_out = str(tmp_path / "corpus_manifest.json")
    split_out = str(tmp_path / "split_assignments.json")

    parser = build_parser()
    args = parser.parse_args([
        "freeze-corpus",
        "--mails-tsv-path", mails_path,
        "--manifest-out", manifest_out,
        "--split-assignments-out", split_out,
        "--seed", "42",
    ])
    returncode = args.func(args)

    assert returncode == 0
    assert os.path.exists(manifest_out)
    assert os.path.exists(split_out)
    with open(manifest_out) as f:
        manifest = json.load(f)
    assert manifest["n_parents"] == 10
    assert manifest["split_config"]["seed"] == 42
