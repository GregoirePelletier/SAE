"""
src/data/augmentation.py — Augmentation contrôlée de mails clients via Gemma-3-12B-it.

Chaque variante est tracée : (parent_id, axis, level, prompt_sha1, model, seed).
Sortie JSONL append-only (reprise sur crash) + manifest parquet.

Réutilise :
  - src/sae/judge._apply_chat_and_extract  (fix BatchEncoding)
  - src/data/preparation._chunk_hash       (dédup)
Aucune réimplémentation de génération : transformers.generate batché.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, asdict
from typing import Iterator, Optional

import pandas as pd
import torch

try:
    from src.sae.judge import _apply_chat_and_extract
except ImportError:
    from judge import _apply_chat_and_extract

# ─── 1. Grille de perturbations ───────────────────────────────────────────────
# axis → {level: instruction}. Une variante = UN axe (perturbation isolée,
# nécessaire pour attribuer un shift d'activation SAE à une cause unique).

AXES: dict[str, dict[str, str]] = {
    "emotion": {
        "colere_forte":  "Réécris ce mail comme un client très en colère : ton accusateur, reproches directs, majuscules ponctuelles.",
        "frustration":   "Réécris ce mail comme un client frustré par des démarches répétées sans résultat : lassitude, énumération des tentatives passées.",
        "impatience":    "Réécris ce mail comme un client impatient : phrases courtes, exigence de délai de réponse explicite.",
        "satisfaction":  "Réécris ce mail comme un client globalement satisfait qui formule une demande courtoise : remerciements, ton positif.",
    },
    "registre": {
        "soutenu":   "Réécris ce mail en registre soutenu : formules de politesse élaborées, subjonctif, vocabulaire administratif précis.",
        "standard":  "Réécris ce mail en registre standard neutre et professionnel.",
        "familier":  "Réécris ce mail en registre très familier/parlé : tutoiement possible, contractions ('y a', 'j'ai pas'), ponctuation relâchée.",
    },
    "orthographe": {
        "degrade_leger": "Réécris ce mail en introduisant quelques fautes réalistes : 2-3 fautes d'accord, une confusion a/à ou é/er, une coquille de frappe.",
        "degrade_fort":  "Réécris ce mail comme tapé très vite sur téléphone : fautes fréquentes, accents manquants, ponctuation minimale, abréviations SMS occasionnelles.",
        "corrige":       "Corrige toutes les fautes d'orthographe, de grammaire et de typographie de ce mail sans changer son contenu ni son ton.",
    },
    "urgence": {
        "panique":            "Réécris ce mail avec un caractère d'extrême urgence : mots-clés de panique ('URGENT', 'immédiatement', 'situation critique'), conséquences imminentes.",
        "menace_resiliation": "Réécris ce mail en y intégrant une menace crédible de résiliation du contrat et de passage à la concurrence si le problème n'est pas résolu rapidement.",
        "calme":              "Réécris ce mail comme une simple demande d'information calme, sans aucune pression temporelle.",
    },
}

_SYSTEM = (
    "Tu es un assistant de génération de données pour un dataset de mails clients "
    "du secteur de l'énergie. Réécris le mail fourni selon la consigne. Contraintes strictes :\n"
    "1. Conserver TOUS les faits : numéros de contrat/client, montants, dates, noms de produits.\n"
    "2. Longueur finale entre 50% et 200% de l'original.\n"
    "3. Rester en français.\n"
    "4. Répondre UNIQUEMENT avec le mail réécrit, sans préambule, sans balises, sans commentaire."
)


@dataclass(frozen=True)
class PerturbationSpec:
    axis: str
    level: str

    @property
    def instruction(self) -> str:
        return AXES[self.axis][self.level]


def all_specs() -> list[PerturbationSpec]:
    return [PerturbationSpec(a, l) for a, levels in AXES.items() for l in levels]


# ─── 2. Prompting ─────────────────────────────────────────────────────────────

def build_messages(mail: str, spec: PerturbationSpec) -> list[dict]:
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"CONSIGNE : {spec.instruction}\n\nMAIL ORIGINAL :\n{mail}"},
    ]


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


# ─── 3. Garde-fous factuels ───────────────────────────────────────────────────

_FACT_RE = re.compile(r"\b\d{4,}\b|\b\d+[.,]\d{2}\s*€|\b\d{1,2}/\d{1,2}/\d{2,4}\b")


def _facts(text: str) -> set[str]:
    return {m.replace(" ", "") for m in _FACT_RE.findall(text)}


def validate(parent: str, variant: str, spec: PerturbationSpec) -> Optional[str]:
    """Retourne None si valide, sinon la raison du rejet."""
    if not variant or len(variant) < 30:
        return "too_short"
    r = len(variant) / max(len(parent), 1)
    if not (0.4 <= r <= 2.5):
        return f"length_ratio={r:.2f}"
    if _sha1(variant) == _sha1(parent):
        return "identical"
    # Les axes non-orthographiques doivent préserver les entités numériques.
    if spec.axis != "orthographe":
        missing = _facts(parent) - _facts(variant)
        if missing:
            return f"facts_lost={sorted(missing)[:3]}"
    return None


# ─── 4. Génération batchée ────────────────────────────────────────────────────

@torch.no_grad()
def generate_variants(
    model,
    tokenizer,
    mails: pd.DataFrame,            # colonnes: doc_id, text, corpus (ex: 'mail_reel')
    specs: Optional[list[PerturbationSpec]] = None,
    out_jsonl: str = "augmented_mails.jsonl",
    batch_size: int = 8,
    max_new_tokens: int = 768,
    temperature: float = 0.8,
    seed: int = 0,
    device: str = "cuda",
) -> pd.DataFrame:
    """
    Produit len(mails) × len(specs) variantes. Reprise : les aug_id déjà présents
    dans out_jsonl sont sautés. Retourne le manifest (accepté + rejeté).
    """
    torch.manual_seed(seed)
    specs = specs or all_specs()

    done: set[str] = set()
    if os.path.exists(out_jsonl):
        with open(out_jsonl, encoding="utf-8") as f:
            done = {json.loads(l)["aug_id"] for l in f if l.strip()}

    jobs = []
    for _, row in mails.iterrows():
        for spec in specs:
            aug_id = f"{row.doc_id}__{spec.axis}__{spec.level}"
            if aug_id not in done:
                jobs.append((aug_id, row, spec))
    print(f"[augment] {len(jobs)} générations à faire ({len(done)} déjà en cache)")

    records = []
    fout = open(out_jsonl, "a", encoding="utf-8")
    model.eval()
    for i in range(0, len(jobs), batch_size):
        batch = jobs[i:i + batch_size]
        prompts = [
            tokenizer.apply_chat_template(build_messages(row.text, spec),
                                          tokenize=False, add_generation_prompt=True)
            for _, row, spec in batch
        ]
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=2048).to(device)
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=0.95,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        gen = out[:, enc["input_ids"].shape[1]:]
        texts = tokenizer.batch_decode(gen, skip_special_tokens=True)

        for (aug_id, row, spec), variant in zip(batch, texts):
            variant = variant.strip()
            reject = validate(row.text, variant, spec)
            rec = {
                "aug_id": aug_id,
                "parent_id": str(row.doc_id),
                "corpus": getattr(row, "corpus", "mail_reel"),
                "axis": spec.axis,
                "level": spec.level,
                "prompt_sha1": _sha1(spec.instruction),
                "model": "gemma-3-12b-it",
                "seed": seed,
                "temperature": temperature,
                "rejected": reject,
                "text": variant if reject is None else None,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            records.append(rec)
        fout.flush()
        if (i // batch_size) % 10 == 0:
            n_ok = sum(r["rejected"] is None for r in records)
            print(f"  [{i + len(batch)}/{len(jobs)}] acceptés={n_ok}")
    fout.close()

    df = pd.DataFrame(records)
    df.to_parquet(out_jsonl.replace(".jsonl", "_manifest.parquet"))
    return df


def load_augmented(jsonl_path: str) -> pd.DataFrame:
    """Charge les variantes acceptées. Colonnes prêtes pour la visu :
    is_augmented=True, corpus_origin=corpus parent, aug_axis, aug_level."""
    rows = [json.loads(l) for l in open(jsonl_path, encoding="utf-8") if l.strip()]
    df = pd.DataFrame([r for r in rows if r["rejected"] is None])
    df["is_augmented"] = True
    return df.rename(columns={"corpus": "corpus_origin", "axis": "aug_axis", "level": "aug_level"})