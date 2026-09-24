"""
scripts/check_docs.py — Garde-fou éditorial pour la documentation Markdown
versionnée du dépôt (README, CLAUDE.md, docs/, report/, RESULTS_TESTS.md) :
signale les régressions vers les travers corrigés lors de la refonte
documentaire (numéro de version interne en prose, jargon de session,
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
TARGET_FILES = ["README.md", "CLAUDE.md", "RESULTS_TESTS.md"]
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
    # SAELens (v6) : version d'un logiciel tiers, pas un numéro de version
    # interne au projet. "Combiné v12" : label de ligne de table reprenant le
    # nom d'un répertoire de run versionné (results_v12_*), exception assumée
    # au même titre que les répertoires/scripts de run (cf. CLAUDE.md).
    (
        "report/RAPPORT_STAGE_UNIVERSITE.tex",
        r"\cite{bricken2023monosemanticity}). \textbf{SAELens \cite{bloom2024saelens}} (v6) stocke en revanche",
    ),
    (
        "report/RAPPORT_STAGE_UNIVERSITE.tex",
        r"Combiné v12 & 44,0\% & exploratoire \\",
    ),
    (
        "report/RAPPORT_STAGE_ENTREPRISE.tex",
        r"\cite{bricken2023monosemanticity}). \textbf{SAELens \cite{bloom2024saelens}} (v6) stocke en revanche",
    ),
    (
        "report/RAPPORT_STAGE_ENTREPRISE.tex",
        r"Combiné v12 & 44,0\% & exploratoire \\",
    ),
    (
        "report/Rapport_stage_EDF_relecture.tex",
        r"\cite{bricken2023monosemanticity}). \textbf{SAELens \cite{bloom2024saelens}} (v6) stocke en revanche",
    ),
    (
        "report/Rapport_stage_EDF_relecture.tex",
        r"Combiné v12 & 44,0\% & exploratoire \\",
    ),
}

VERSION_RE = re.compile(r"\bv\d{1,2}\b")
SESSION_V_RE = re.compile(r"\bsession\s+v\d", re.IGNORECASE)
PLACEHOLDER_RE = re.compile(
    r"\[à compléter\]|§<[^>]*à.?compléter[^>]*>|<!--\s*à\s*compléter\s*-->",
    re.IGNORECASE,
)
TODO_RE = re.compile(r"\bTODO\b")
FIRST_PERSON_RE = re.compile(r"\b(je|j'|ma|mon|mes)\b", re.IGNORECASE)
LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")

FIRST_PERSON_SCOPE = {"README.md"}  # + tout fichier sous docs/

# N12 (AUDIT_SAE_2026-08.md §8) : ces deux fichiers sous docs/ sont exclus du
# contrôle "première personne" -- décision tranchée, pas un oubli. Les deux
# sont des analyses comparatives explicitement à la première personne par
# nature ("mon pipeline" vs. le code d'un dépôt tiers, ou la méthodologie de
# transcription de CE document même) : réécrire mécaniquement ~16 occurrences
# en voix impersonnelle risquait de dénaturer des jugements techniques nuancés
# sous la contrainte de temps, pour un gain de cohérence marginal sur des
# documents de travail (pas des sections citées par le rapport comme
# `RESULTS_TESTS.md`, où la règle "présent, sans récit" reste pleinement
# appliquée). Les AUTRES contrôles (version interne, TODO, placeholder, lien
# mort) continuent de s'appliquer à ces deux fichiers.
FIRST_PERSON_EXCLUDED_FILES = {
    os.path.join("docs", "INTERP_EMBED_COVERAGE.md"),
    os.path.join("docs", "archive", "references", "PDF_APPENDICES_EXTRACT.md"),
}


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
                if f.endswith((".md", ".tex")):
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
    in_scope_first_person = (
        (rel_path == "README.md" or rel_path.startswith("docs" + os.sep))
        and rel_path not in FIRST_PERSON_EXCLUDED_FILES
    )

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
