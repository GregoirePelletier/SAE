"""
dataset.py — Ingestion Mails.tsv + datasets émotions/intentions FR/EN.
FR: pas de dataset émotions natif de qualité → traduction contrôlée de GoEmotions
    (simplifié à la taxonomie Ekman+intentions) OU synthétique Gemma-3-12B-it (existant).
EN: GoEmotions (Demszky 2020) comme corpus d'évaluation cross-lingue.
"""
from __future__ import annotations
import re
import unicodedata
from pathlib import Path
from typing import Optional

import pandas as pd

EKMAN_MAP = {  # GoEmotions 27 -> Ekman 6 + neutral (mapping officiel du repo GoEmotions)
    "anger": ["anger", "annoyance", "disapproval"],
    "disgust": ["disgust"],
    "fear": ["fear", "nervousness"],
    "joy": ["joy", "amusement", "approval", "excitement", "gratitude", "love",
            "optimism", "relief", "pride", "admiration", "desire", "caring"],
    "sadness": ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
    "surprise": ["surprise", "realization", "confusion", "curiosity"],
    "neutral": ["neutral"],
}
_GO2EKMAN = {g: e for e, gs in EKMAN_MAP.items() for g in gs}


# `\b` encadre une FRONTIÈRE DE MOT : un radical seul (`r[ée]sili`) ne matche
# que la forme non fléchie exacte, jamais "résilier"/"résiliation". Chaque
# radical à un seul mot porte donc `\w*` pour matcher ses formes fléchies ;
# les alternatives-phrases (avec espaces, déjà spécifiques) n'en ont pas
# besoin.
INTENT_KEYWORDS_FR = {
    "reclamation": r"\b(r[ée]clamation\w*|contest\w*|inadmissible\w*|scandaleux\w*|erreur de facturation)\b",
    "resiliation": r"\b(r[ée]sili\w*|clôtur\w*|mettre fin au contrat)\b",
    "remboursement": r"\b(rembours\w*|trop[- ]perçu|avoir\w*)\b",
    "information": r"\b(renseign\w*|information\w*|pourriez[- ]vous m'indiquer|comment (faire|proc[ée]der))\b",
    "urgence": r"\b(urgent\w*|imm[ée]diat\w*|sans d[ée]lai|coupure\w*)\b",
}


def _clean_text(t: str) -> str:
    t = unicodedata.normalize("NFC", str(t))
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip().strip('"').strip()


def load_mails_tsv(path: str | Path, min_chars: int = 30) -> pd.DataFrame:
    """
    Ingestion robuste de Mails.tsv (colonnes: index, document, segments).
    Le champ `document` contient des retours ligne internes → parsing quoting-aware.
    Enrichissement: langue heuristique, intentions par regex (labels faibles), longueur.
    """
    n_bad_lines = [0]

    def _count_and_skip(bad_line):
        n_bad_lines[0] += 1
        return None  # None -> ligne ignorée (même comportement que on_bad_lines="skip")

    df = pd.read_csv(path, sep="\t", quoting=0, engine="python", on_bad_lines=_count_and_skip)
    if n_bad_lines[0]:
        print(f"  [dataset] load_mails_tsv : {n_bad_lines[0]} ligne(s) malformée(s) ignorée(s) "
              f"({path}).")
    df.attrs["n_bad_lines_skipped"] = n_bad_lines[0]
    text_col = "document" if "document" in df.columns else df.columns[1]
    df = df.rename(columns={text_col: "text"})
    df["text"] = df["text"].map(_clean_text)
    df = df[df["text"].str.len() >= min_chars].drop_duplicates("text").reset_index(drop=True)

    # Langue (heuristique légère, suffisante FR/EN ; remplacer par fasttext lid si besoin)
    fr_marks = df["text"].str.count(r"\b(le|la|les|de|des|une|vous|nous|est)\b", flags=re.I)
    en_marks = df["text"].str.count(r"\b(the|and|you|is|are|of|to|for)\b", flags=re.I)
    df["lang"] = (fr_marks >= en_marks).map({True: "fr", False: "en"})

    for intent, pat in INTENT_KEYWORDS_FR.items():
        df[f"intent_{intent}"] = df["text"].str.contains(pat, flags=re.I, regex=True)
    df["n_chars"] = df["text"].str.len()
    return df


def load_goemotions_ekman(split: str = "validation", max_n: Optional[int] = 5000) -> pd.DataFrame:
    """Corpus d'évaluation EN, labels Ekman single-label (docs multi-label écartés)."""
    from datasets import load_dataset
    ds = load_dataset("go_emotions", "simplified", split=split)
    names = ds.features["labels"].feature.names
    rows = []
    for ex in ds:
        ek = {_GO2EKMAN[names[i]] for i in ex["labels"]}
        if len(ek) == 1:
            rows.append({"text": ex["text"], "lang": "en", "emotion": ek.pop()})
    df = pd.DataFrame(rows)
    return df.sample(min(max_n, len(df)), random_state=0) if max_n else df


def build_bilingual_corpus(mails_path: str | Path, max_en: int = 5000) -> pd.DataFrame:
    """Concatène Mails.tsv (FR, non labellisé émotion) + GoEmotions (EN, labellisé)."""
    fr = load_mails_tsv(mails_path).assign(source="mails_fr", emotion=None)
    en = load_goemotions_ekman(max_n=max_en).assign(source="goemotions_en")
    cols = ["text", "lang", "source", "emotion"]
    return pd.concat([fr[cols + [c for c in fr if c.startswith("intent_")]],
                      en[cols]], ignore_index=True)