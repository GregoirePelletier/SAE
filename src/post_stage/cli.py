"""
src/post_stage/cli.py -- Point d'entree unique de la campagne post-soutenance
(docs/post_stage/PLAN_E00-E09.md §16.2). Commandes ajoutees au fur
et a mesure des besoins de E00-E09, pas toutes d'un coup (§16.1 -- adapter au
fur et a mesure, ne pas ecrire un contrat complet avant d'en avoir besoin).

Aucune commande ici ne charge de LLM par defaut (§16.2).
"""
import argparse
import json
import sys

from src.post_stage.dataset_contract import build_corpus_manifest, write_corpus_manifest


def _cmd_freeze_corpus(args: argparse.Namespace) -> int:
    manifest, parent_records, variant_records = build_corpus_manifest(
        args.mails_tsv_path, args.augmented_jsonl_path,
        fit=args.fit, dev=args.dev, seed=args.seed,
    )
    write_corpus_manifest(
        manifest, parent_records, variant_records,
        args.manifest_out, args.split_assignments_out,
    )
    print(json.dumps(manifest, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.post_stage.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    p_freeze = sub.add_parser(
        "freeze-corpus",
        help="Gele le split FIT/DEV/CONFIRM parent-aware et ecrit le manifeste (§4.1-4.2).",
    )
    p_freeze.add_argument("--mails-tsv-path", dest="mails_tsv_path", required=True)
    p_freeze.add_argument("--augmented-jsonl-path", dest="augmented_jsonl_path", default="")
    p_freeze.add_argument("--fit", type=float, default=0.60)
    p_freeze.add_argument("--dev", type=float, default=0.15)
    p_freeze.add_argument("--seed", type=int, default=None,
                           help="Defaut : POST_STAGE_SPLIT_SEED (dataset_contract.py), ne pas surcharger sans motif documente.")
    p_freeze.add_argument("--manifest-out", dest="manifest_out",
                           default="configs/post_stage/corpus_manifest.json")
    p_freeze.add_argument("--split-assignments-out", dest="split_assignments_out",
                           default="configs/post_stage/split_assignments.json")
    p_freeze.set_defaults(func=_cmd_freeze_corpus)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "freeze-corpus" and args.seed is None:
        from src.post_stage.dataset_contract import POST_STAGE_SPLIT_SEED
        args.seed = POST_STAGE_SPLIT_SEED
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
