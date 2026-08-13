"""
scripts/check_docs.py — Garde-fou éditorial pour la documentation Markdown
versionnée du dépôt (README, CLAUDE.md, docs/, report/, RESULTS_TESTS.md,
CHANGELOG.md) : signale les régressions vers les travers corrigés lors de la
refonte documentaire (numéro de version interne en prose, jargon de session,
placeholders non résolus, TODO, première personne du singulier dans
README.md/docs/, lien relatif mort).

Zéro dépendance réseau, zéro calcul : lecture pure de fichiers sur disque.

Usage :
    .venv/bin/python scripts/check_docs.py
    (retourne un code de sortie non nul et liste les violations sur stdout)
"""
from __future__ import annotations

import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TARGET_DIRS = ["docs", "report"]
TARGET_FILES = ["README.md", "CLAUDE.md", "CHANGELOG.md", "RESULTS_TESTS.md"]
EXCLUDE_DIRS = {"dist"}  # report/dist/ est généré, gitignoré, non versionné

# Chaînes connues à masquer avant la recherche de motif de version interne --
# ce sont des noms de produit/modèle contenant un chiffre après "v", pas une
# convention de versioning interne au projet.
PRODUCT_NAME_EXCEPTIONS = ["F2LLM-v2"]

# Exceptions ponctuelles (fichier relatif au dépôt, sous-chaîne exacte de la
# ligne fautive) pour les faux positifs restants qui ne valent pas la peine
# d'une règle générale.
LINE_EXCEPTIONS: set[tuple[str, str]] = {
    # URL d'API publique versionnée (v2.1), span de backticks multi-lignes non
    # détecté par la lecture ligne à ligne — pas un numéro de version interne.
    ("RESULTS_TESTS.md", "du schéma de l'export public (`data.economie.gouv.fr/api/explore/v2.1/catalog/"),
    # Citation directe de la formulation du juge LLM lui-même ("je reconnais un
    # concept" / "je peux le classer"), pas une première personne de l'auteur.
    ("docs/sae_diagnostics_playbook.md", 'le juge distingue "je reconnais un concept" de "je peux le classer par'),
}

VERSION_RE = re.compile(r"\bv\d{1,2}\b")
SESSION_V_RE = re.compile(r"\bsession\s+v\d", re.IGNORECASE)
PLACEHOLDER_RE = re.compile(r"\[à compléter\]", re.IGNORECASE)
TODO_RE = re.compile(r"\bTODO\b")
FIRST_PERSON_RE = re.compile(r"\b(je|j'|ma|mon|mes)\b", re.IGNORECASE)
LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")

FIRST_PERSON_SCOPE = {"README.md"}  # + tout fichier sous docs/


def iter_target_files():
    for name in TARGET_FILES:
        path = os.path.join(REPO_ROOT, name)
        if os.path.exists(path):
            yield os.path.relpath(path, REPO_ROOT)
    for d in TARGET_DIRS:
        full_dir = os.path.join(REPO_ROOT, d)
        for root, dirs, files in os.walk(full_dir):
            dirs[:] = [x for x in dirs if x not in EXCLUDE_DIRS]
            for f in files:
                if f.endswith(".md"):
                    path = os.path.join(root, f)
                    yield os.path.relpath(path, REPO_ROOT)


def strip_code(text: str) -> str:
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"`[^`]*`", "", text)
    return text


def mask_product_names(text: str) -> str:
    for name in PRODUCT_NAME_EXCEPTIONS:
        text = text.replace(name, "#" * len(name))
    return text


def check_file(rel_path: str) -> list[str]:
    violations = []
    abs_path = os.path.join(REPO_ROOT, rel_path)
    raw_lines = open(abs_path, encoding="utf-8").read().split("\n")
    in_scope_first_person = rel_path == "README.md" or rel_path.startswith("docs" + os.sep)

    for i, raw_line in enumerate(raw_lines, start=1):
        if (rel_path, raw_line.strip()) in LINE_EXCEPTIONS:
            continue
        code_free = mask_product_names(strip_code(raw_line))

        if VERSION_RE.search(code_free):
            violations.append(f"{rel_path}:{i}: numéro de version interne en prose — {raw_line.strip()!r}")
        if SESSION_V_RE.search(code_free):
            violations.append(f"{rel_path}:{i}: jargon 'session vN' — {raw_line.strip()!r}")
        if PLACEHOLDER_RE.search(raw_line):
            violations.append(f"{rel_path}:{i}: placeholder [à compléter] non converti — {raw_line.strip()!r}")
        if TODO_RE.search(code_free):
            violations.append(f"{rel_path}:{i}: TODO résiduel — {raw_line.strip()!r}")
        if in_scope_first_person and FIRST_PERSON_RE.search(code_free):
            violations.append(f"{rel_path}:{i}: première personne du singulier — {raw_line.strip()!r}")

        for _, target in LINK_RE.findall(raw_line):
            if target.startswith("http") or target.startswith("#") or not target.split("#")[0]:
                continue
            link_path = target.split("#")[0]
            resolved = os.path.normpath(os.path.join(os.path.dirname(abs_path), link_path))
            if not os.path.exists(resolved):
                violations.append(f"{rel_path}:{i}: lien relatif mort — {target!r}")

    return violations


def main() -> int:
    all_violations = []
    for rel_path in sorted(set(iter_target_files())):
        all_violations.extend(check_file(rel_path))

    if all_violations:
        print(f"{len(all_violations)} violation(s) trouvée(s) :\n")
        for v in all_violations:
            print(f"  {v}")
        return 1

    print("Aucune violation trouvée.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
