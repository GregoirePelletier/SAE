"""
scripts/audit_2026_08_extract_saeboost_pdf_text.py — Extraction texte brute de
pdf/teacholdsaes.pdf (SAE Boost, Koriagin 2025) via pypdf (déjà dans le venv,
aucune dépendance nouvelle). Usage ponctuel : clarifier la description exacte
des baselines "Extended SAE (most act)"/"Extended SAE (random)" par rapport à
la méthode proposée (SAE Boost), question soulevée par l'utilisateur. Ne
touche à aucun cache de production. CPU-only, quelques secondes.
"""
from pypdf import PdfReader

reader = PdfReader("pdf/teacholdsaes.pdf")
print(f"[extract] {len(reader.pages)} pages.")
with open("docs/_teacholdsaes_raw_text.txt", "w", encoding="utf-8") as f:
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        f.write(f"\n\n===== PAGE {i+1} =====\n\n")
        f.write(text)
print("[extract] Écrit -> docs/_teacholdsaes_raw_text.txt")
