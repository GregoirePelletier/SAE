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
    from src.data.dataset import strip_leading_objet_line
except ImportError:
    from judge import _apply_chat_and_extract
    from dataset import strip_leading_objet_line

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
    "4. Répondre UNIQUEMENT avec le mail réécrit, sans préambule, sans balises, sans commentaire.\n"
    "5. Ne jamais ajouter de ligne \"Objet :\"/\"Subject :\" ni de mise en forme absente du mail "
    "original (markdown, gras **, titres) : le mail source n'en a pas, la réécriture ne doit pas "
    "en introduire — reproduire son format brut."
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


def _batch_seed(seed: int, aug_ids: list[str]) -> int:
    """Graine déterministe dérivée du CONTENU d'un lot de génération (B.12,
    AUDIT_SAE_2026-08.md) -- `sorted()` rend le résultat indépendant de
    l'ordre d'itération du lot, pas seulement de son contenu."""
    return int(_sha1(f"{seed}:{'|'.join(sorted(aug_ids))}"), 16) % (2**31)


# ─── 3. Garde-fous factuels ───────────────────────────────────────────────────

_FACT_RE = re.compile(
    r"\b0\d(?:[ .\-]?\d{2}){4}\b"       # téléphone FR (10 chiffres, séparateurs optionnels) --
                                         # DOIT précéder le fallback générique ci-dessous : sinon
                                         # un numéro espacé/pointé n'est capturé QUE par fragments
                                         # de 4+ chiffres contigus (ex. "0476" sur "0476 35 64 90"),
                                         # jamais comme un seul fact.
    r"|\b\d+[.,]\d{2}\s*€"
    r"|\b\d{1,2}/\d{1,2}/\d{2,4}\b"
    r"|\b\d{4,}\b"
)


def _normalize_fact(m: str) -> str:
    """Normalise un fact capturé pour une comparaison robuste au reformatage
    pur (pas de perte réelle de contenu) : sans cette normalisation, un simple
    reformatage (numéro de téléphone "0476356490" -> "0476 35 64 90", date sans
    zéro de padding "18/7/13" -> "18/07/2013", montant en virgule décimale
    française au lieu du point "20.73€" -> "20,73 €") serait à tort compté
    comme un fait perdu (`facts_lost`). Normalisation : séparateur décimal
    unifié (virgule), séparateurs de regroupement (espaces,
    points, tirets) retirés des séquences numériques pures (téléphones),
    composantes de date entières (jour/mois/année) comparées sans padding.
    Reste hors de portée : une date réécrite en toutes lettres ("18 juillet
    2013") n'est pas normalisable par cette fonction (nécessiterait un
    parsing NLP des mois) -- limite connue, documentée plutôt que masquée."""
    if "/" in m:
        day, month, year = m.split("/")
        year_int = int(year)
        if len(year) == 2:  # "13" -> "2013" (corpus 2020s, ambiguïté négligeable)
            year_int += 2000
        return f"{int(day)}/{int(month)}/{year_int}"
    if "€" in m:
        return m.replace(" ", "").replace(".", ",")
    return re.sub(r"[ .\-]", "", m)


def _facts(text: str) -> set[str]:
    return {_normalize_fact(m) for m in _FACT_RE.findall(text)}


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
        # Graine dérivée du CONTENU du lot (B.12, AUDIT_SAE_2026-08.md) plutôt
        # que de l'état séquentiel du générateur global : un `torch.manual_seed`
        # unique en tête de fonction rend le "seed" écrit dans chaque
        # enregistrement JSONL trompeur -- une reprise (do_sample=True, lots
        # déjà générés sautés) change la composition des lots suivants, donc le
        # flux RNG, donc la sortie réelle pour un même aug_id régénéré. Ici,
        # regénérer EXACTEMENT le même lot d'aug_id (même composition, même
        # ordre) reproduit la même sortie, indépendamment de ce qui a été
        # généré avant dans le run. Ne couvre pas le cas où une reprise change
        # la composition d'un lot (aug_id désormais co-batché différemment) --
        # limite assumée, pas un flux RNG global reproductible de bout en bout.
        torch.manual_seed(_batch_seed(seed, [aug_id for aug_id, _, _ in batch]))
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
                # SHA1 du texte parent (pas seulement sa position, cf. parent_id) :
                # permet à build_email_train_test_corpus (preparation.py) de
                # rattacher une variante à son mail d'origine par CONTENU plutôt
                # que par index positionnel dans un recalcul indépendant de
                # load_and_clean_emails -- deux calculs de filtrage légèrement
                # désynchronisés feraient sinon fuir le split group-aware sans
                # aucun signal (AUDIT_SAE_2026-08.md, item B.7).
                "parent_sha1": _sha1(row.text),
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
    is_augmented=True, corpus_origin=corpus parent, aug_axis, aug_level.
    Le texte est nettoyé d'une éventuelle ligne "Objet :"/"Subject :" résiduelle
    (cf. dataset.strip_leading_objet_line) pour rester cohérent avec le
    traitement des mails originaux et ne pas polluer le SAE avec un artefact
    de formatage."""
    # Un seul passage, filtré au vol -- pas de liste intermédiaire de TOUS les
    # enregistrements (acceptés + rejetés) tenue en RAM en plus de la liste
    # filtrée puis du DataFrame final (AUDIT_SAE_2026-08.md, item A6 : ce
    # process tient déjà le réservoir memmap et all_doc_sae_acts).
    accepted = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["rejected"] is None:
                accepted.append(r)
    if not accepted:
        return pd.DataFrame(columns=["text", "is_augmented", "corpus_origin",
                                      "aug_axis", "aug_level"])
    df = pd.DataFrame(accepted)
    df["text"] = df["text"].map(strip_leading_objet_line)
    df["is_augmented"] = True
    return df.rename(columns={"corpus": "corpus_origin", "axis": "aug_axis", "level": "aug_level"})