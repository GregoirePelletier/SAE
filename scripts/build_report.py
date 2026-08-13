"""
scripts/build_report.py — Génère report/dist/RAPPORT_DE_STAGE.md par
concaténation des fichiers sources numérotés (+ déclaration IA), avec titre
de chapitre imposé.

Les chapitres numérotés (`report/00_*.md` … `07_*.md`) sont la seule source
de vérité : ce script produit un artefact de lecture, jamais versionné
(`report/dist/` est gitignoré). Un chapitre ajouté, retiré ou renommé ne
demande qu'une modification de CHAPTERS ci-dessous, suivie d'un rerun.

Usage : .venv/bin/python scripts/build_report.py
"""
import os

REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "report")
DIST_DIR = os.path.join(REPORT_DIR, "dist")

# (fichier source, titre de chapitre affiché ; None = garder le titre H1 déjà
# présent dans le fichier source, sans préfixe "Chapitre N").
CHAPTERS = [
    ("FRONT_MATTER.md", None),
    ("00_introduction.md", None),
    ("01_etat_de_lart.md", "Chapitre 1 — État de l'art"),
    ("02_architecture.md", "Chapitre 2 — Architecture et implémentation"),
    ("03_experiences_et_resultats.md", "Chapitre 3 — Démarche expérimentale et résultats"),
    ("04_limites_et_perspectives.md", "Chapitre 4 — Limites et perspectives"),
    ("06_conclusion.md", None),
    ("07_bibliographie.md", None),
    ("DECLARATION_IA.md", None),
]


def render_chapter(path: str, title_override: str | None) -> str:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if title_override is None:
        return text.rstrip("\n")
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("# "):
            lines[i] = f"# {title_override}"
            break
    return "\n".join(lines).rstrip("\n")


def main():
    parts = [render_chapter(os.path.join(REPORT_DIR, filename), title)
             for filename, title in CHAPTERS]
    output = "\n\n---\n\n".join(parts) + "\n"
    os.makedirs(DIST_DIR, exist_ok=True)
    out_path = os.path.join(DIST_DIR, "RAPPORT_DE_STAGE.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"{out_path} régénéré ({len(CHAPTERS)} sections, {len(output)} caractères).")


if __name__ == "__main__":
    main()
