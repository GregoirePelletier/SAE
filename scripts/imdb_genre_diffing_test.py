"""
Réplication ciblée de l'évaluation ground-truth "movie genre differences"
(papier arXiv:2512.10092v2, §4.1 + Appendix D.3, Table 1/Table 5).

Protocole du papier : diff les descriptions d'un genre donné contre 500
descriptions échantillonnées aléatoirement hors de ce genre, prendre les 5
latents avec la plus grande différence de fréquence positive, mesurer leur
similarité de surface avec le label de genre ground-truth (prompt verbatim
Appendix D.3, échelle yes/related/no -> 1/0.5/0), échantillonné 5x.

Écarts DOCUMENTÉS par rapport au papier (décidés avec l'utilisateur avant
ce script, pas des approximations silencieuses) :
- Juge = Qwen3.8-27B local, quantifié 8-bit (bitsandbytes), PAS GPT-5 (pas de
  clé API configurée dans ce projet, et bf16 ~52 Go ne tient pas de façon
  fiable sur ce cluster -- cf. note GPU ci-dessous). Le prompt de similarité
  de surface est repris verbatim ; seuls le modèle et la précision diffèrent.
- Dataset genre = `adrienheymans/imdb-movie-genres` (HF Hub, 54214 lignes,
  colonnes title/text/genre) : le papier cite Maas et al. 2011 [33], qui est
  en réalité le dataset de SENTIMENT IMDB (pas de labels de genre) -- source
  exacte du papier indécidable depuis le texte seul. Ce dataset HF correspond
  à la description textuelle du papier ("movie descriptions with genre
  labels") et est utilisé comme meilleure reconstruction de bonne foi.
- 6 genres testés (action, romance, horror, comedy, sci-fi, thriller), pas
  la liste exhaustive du papier (non spécifiée) -- échantillon représentatif
  pour borner le coût GPU.
- Modèle + SAE : Llama-3.1-8B-Instruct + Goodfire SAE-l19 (d_sae=65536),
  identiques au papier (§4.1 : layer 50 de Llama-3.3-70B en réalité pour le
  papier -- ATTENTION, le papier utilise le SAE 70B/layer 50, pas le 8B/l19 ;
  voir note ci-dessous).

Règle de fidélité au code du papier (explicite, demandée) : partout où le
papier n'a pas besoin d'une API externe, on appelle LEUR code directement
(`interp_embed.Dataset`, `interp_embed.sae.local_sae.GoodfireSAE`,
`examples/functions.py::diff_features`) -- jamais une réimplémentation
maison. Seule exception : le juge (et la génération de la description
baseline), qui DOIT être substitué par un LLM local puisque leur code
suppose une API (OpenAI/OpenRouter, `interp_embed/llm/utils.py`).

Sortie : JSON dans local_data/imdb_genre_diffing/results.json.
"""
import os
import sys
import json
import random
import importlib.util
from pathlib import Path

sys.path.insert(0, "/home/h21486/SAE/external/interp_embed")

import numpy as np
import pandas as pd

# examples/functions.py n'est pas un sous-module du package interp_embed
# (pas de __init__.py dans examples/) -- chargement direct par chemin pour
# appeler leur diff_features telle quelle, sans la réécrire.
_spec = importlib.util.spec_from_file_location(
    "interp_embed_examples_functions",
    "/home/h21486/SAE/external/interp_embed/examples/functions.py",
)
interp_functions = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(interp_functions)
import torch
from huggingface_hub import hf_hub_download

OUT_DIR = Path("/home/h21486/SAE/local_data/imdb_genre_diffing")
OUT_DIR.mkdir(parents=True, exist_ok=True)

JUDGE_MODEL_PATH = "/home/h21486/SAE/models/Qwen3.8-27B"
GENRES = ["action", "romance", "horror", "comedy", "sci-fi", "thriller"]
N_IN_GENRE_MAX = 500
N_OUT_GENRE = 500
TOP_K_LATENTS = 5
N_JUDGE_SAMPLES = 5
JUDGE_TEMPERATURE = 0.7
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

SURFACE_SIMILARITY_PROMPT = """Is text a and text b similar in meaning?
First, provide your reasoning about how text a and text b relate to each other.
Then, respond with yes, related, or no.
If text b has multiple items in commas, you should use the closest match with text a. Respond yes if text b
captures the spirit of text a. Respond related if text b is related to text a but not exactly the same.
Respond no if text b is not related to text a at all.

Here are a few examples.
Example 1:
text a: has a topic of protecting the environment
text b: has a topic of environmental protection and sustainability
output: yes

Example 2:
text a: has a language of German
text b: has a language of Deutsch
output: yes

Example 3:
text a: has a topic of the sports
text b: has a topic of sports team recruiting new members
output: yes

Example 4:
text a: has a topic of the relation between political figures
text b: has a topic of international diplomacy
output: related

Example 5:
text a: has a named language of Korean
text b: uses archaic and poetic diction
output: related

Example 6:
text a: describes an important 20th century historical event
text b: describes a 20th century European politician
output: related

Example 7:
text a: has a named language of Korean
text b: has a named language of Japanese
output: no

Example 8:
text a: talks about the history of the United States
text b: talks about dinosaurs
output: no

Target:
text a: {text_a}
text b: {text_b}
output:"""

SCORE_MAP = {"yes": 1.0, "related": 0.5, "no": 0.0}


def load_imdb_genre_df() -> pd.DataFrame:
    p = hf_hub_download(
        "adrienheymans/imdb-movie-genres",
        filename="data/train-00000-of-00001-b7b538a3d562331b.parquet",
        repo_type="dataset",
    )
    return pd.read_parquet(p)


def build_genre_row_selection(df: pd.DataFrame) -> dict:
    """Pour chaque genre : indices in-genre (cap 500) + 500 indices out-genre.
    Retourne aussi l'union dédupliquée des indices à encoder une seule fois."""
    selection = {}
    all_indices = set()
    for genre in GENRES:
        in_idx = df.index[df["genre"] == genre].tolist()
        if len(in_idx) > N_IN_GENRE_MAX:
            in_idx = random.sample(in_idx, N_IN_GENRE_MAX)
        out_pool = df.index[df["genre"] != genre].tolist()
        out_idx = random.sample(out_pool, N_OUT_GENRE)
        selection[genre] = {"in": in_idx, "out": out_idx}
        all_indices.update(in_idx)
        all_indices.update(out_idx)
    return selection, sorted(all_indices)


def main():
    from interp_embed import Dataset
    from interp_embed.sae.local_sae import GoodfireSAE

    print("Loading IMDB genre dataset...", flush=True)
    df = load_imdb_genre_df()
    selection, all_indices = build_genre_row_selection(df)
    print(f"Total unique rows to encode: {len(all_indices)}", flush=True)

    encode_df = df.loc[all_indices].reset_index(drop=True)
    idx_map = {orig: i for i, orig in enumerate(all_indices)}  # orig df index -> position in encode_df

    print("Loading Llama-3.1-8B-Instruct + Goodfire SAE-l19...", flush=True)
    sae = GoodfireSAE(variant_name="Llama-3.1-8B-Instruct-SAE-l19", device="cuda:0")
    full_ds = Dataset(
        data=encode_df, sae=sae, field="text",
        dataset_description="imdb_genre_diffing_encode_pool",
        save_path=str(OUT_DIR / "encoded_pool.pkl"),
        batch_size=16,
    )
    # sae.destroy() est déjà appelé en interne par Dataset._compute_latents.
    torch.cuda.empty_cache()
    print(f"Encoded {len(full_ds)} rows.", flush=True)

    # JUDGE_QUANT=int8 (défaut, sûr sur a100 40 Go dédié) ou bf16 (nécessite
    # un GPU 80 Go non partagé, ex. h100/h100-bis quand vraiment idle -- rien
    # ne garantit une carte 80 Go pleine là-bas, gpu:8,shard:16 -- à vérifier
    # au cas par cas via squeue/sinfo avant de lancer en bf16 hors a100).
    judge_quant = os.environ.get("JUDGE_QUANT", "int8")
    print(f"Loading local {JUDGE_MODEL_PATH} judge ({judge_quant})...", flush=True)
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
    judge_tok = AutoTokenizer.from_pretrained(JUDGE_MODEL_PATH)
    load_kwargs = {"device_map": "cuda:0"}
    if judge_quant == "int8":
        # bitsandbytes, même mécanisme que GoodfireSAE.load_models(quantize=True)
        # dans interp_embed (cf. local_sae.py) -> ~27 Go, tient dans un A100
        # 40 Go dédié avec marge pour le KV-cache.
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True, bnb_8bit_compute_dtype=torch.bfloat16,
        )
    elif judge_quant == "bf16":
        load_kwargs["torch_dtype"] = torch.bfloat16
    else:
        raise ValueError(f"JUDGE_QUANT invalide: {judge_quant!r} (int8/bf16)")
    # Qwen3.8-27B est un modèle vision-langage natif (pipeline_tag
    # image-text-to-text, classe Qwen3_5ForConditionalGeneration) -- pas
    # forcément reconnue par AutoModelForCausalLM selon la version de
    # transformers. Usage texte seul ici (juge), aucune image passée --
    # AutoModelForImageTextToText en premier, repli sur AutoModelForCausalLM
    # si l'architecture est en fait bien mappée dans cette version.
    try:
        judge_model = AutoModelForImageTextToText.from_pretrained(JUDGE_MODEL_PATH, **load_kwargs)
    except Exception as e:
        print(f"  AutoModelForImageTextToText failed ({e}), falling back to AutoModelForCausalLM", flush=True)
        judge_model = AutoModelForCausalLM.from_pretrained(
            JUDGE_MODEL_PATH, quantization_config=bnb_config, device_map="cuda:0",
        )
    judge_model.eval()

    def judge_surface_similarity(text_a: str, text_b: str, n_samples: int = N_JUDGE_SAMPLES) -> float:
        prompt = SURFACE_SIMILARITY_PROMPT.format(text_a=text_a, text_b=text_b)
        inputs = judge_tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, return_tensors="pt",
        ).to(judge_model.device)
        scores = []
        with torch.no_grad():
            for _ in range(n_samples):
                out = judge_model.generate(
                    input_ids=inputs, max_new_tokens=200,
                    do_sample=True, temperature=JUDGE_TEMPERATURE,
                )
                resp = judge_tok.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True).strip().lower()
                found = None
                for kw in ["yes", "related", "no"]:
                    if f"output: {kw}" in resp or resp.endswith(kw) or resp.split()[-1].strip(".") == kw:
                        found = kw
                        break
                if found is None:
                    for kw in ["yes", "related", "no"]:
                        if kw in resp:
                            found = kw
                            break
                scores.append(SCORE_MAP.get(found, 0.0))
        return float(np.mean(scores)), scores

    def llm_baseline_description(in_texts: list, out_texts: list, n_sample: int = 20) -> str:
        """Baseline simple du papier : donner les deux corpus au juge, demander
        une phrase décrivant la différence principale."""
        a = "\n".join(f"- {t}" for t in in_texts[:n_sample])
        b = "\n".join(f"- {t}" for t in out_texts[:n_sample])
        prompt = (
            "Here are two sets of movie descriptions.\n\nSET A:\n" + a +
            "\n\nSET B:\n" + b +
            "\n\nIn one short sentence, describe the main topical difference of SET A compared to SET B."
        )
        inputs = judge_tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, return_tensors="pt",
        ).to(judge_model.device)
        with torch.no_grad():
            out = judge_model.generate(input_ids=inputs, max_new_tokens=64, do_sample=False)
        return judge_tok.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True).strip()

    results = {"genres": {}}
    for genre in GENRES:
        print(f"\n=== Genre: {genre} ===", flush=True)
        in_pos = [idx_map[i] for i in selection[genre]["in"]]
        out_pos = [idx_map[i] for i in selection[genre]["out"]]
        in_ds = full_ds[np.array(in_pos)]
        out_ds = full_ds[np.array(out_pos)]

        # interp_embed/examples/functions.py::diff_features telle quelle -- déjà
        # triée par frequency_difference descendant, pas de coverage filter
        # (min_coverage=0.0/max_coverage=1.0 par défaut = aucun masquage).
        diff_df = interp_functions.diff_features(in_ds, out_ds)
        # Ne garder que la direction "plus fréquent dans le genre cible", pas
        # les deux sens (leur fonction trie par |aucun abs|, en réalité par
        # valeur signée descendante -- déjà le bon sens ici par construction
        # puisque ds1=in_ds).
        top5 = diff_df.head(TOP_K_LATENTS)

        text_a = f"has a genre of {genre}"
        per_latent = []
        for _, row in top5.iterrows():
            label = row["feature"] or f"feature_{int(row['feature_id'])}"
            score, raw = judge_surface_similarity(text_a, label)
            per_latent.append({
                "feature_id": int(row["feature_id"]), "label": label,
                "diff": float(row["frequency_difference"]), "score": score, "raw_judgments": raw,
            })
            print(f"  latent {int(row['feature_id'])} ({label!r}) diff={row['frequency_difference']:.3f} score={score:.2f}", flush=True)
        sae_genre_score = float(np.mean([p["score"] for p in per_latent]))

        baseline_desc = llm_baseline_description(in_ds.documents(), out_ds.documents())
        baseline_score, baseline_raw = judge_surface_similarity(text_a, baseline_desc)
        print(f"  baseline desc: {baseline_desc!r} score={baseline_score:.2f}", flush=True)

        results["genres"][genre] = {
            "sae_score": sae_genre_score,
            "top5_latents": per_latent,
            "baseline_description": baseline_desc,
            "baseline_score": baseline_score,
        }

    results["sae_avg"] = float(np.mean([g["sae_score"] for g in results["genres"].values()]))
    results["baseline_avg"] = float(np.mean([g["baseline_score"] for g in results["genres"].values()]))
    results["published_sae_movies"] = 0.75
    results["published_baseline_movies"] = 0.90

    out_path = OUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n=== SUMMARY ===", flush=True)
    print(f"SAE avg surface similarity: {results['sae_avg']:.3f} (published: 0.75)", flush=True)
    print(f"LLM baseline avg: {results['baseline_avg']:.3f} (published: 0.90)", flush=True)
    print(f"Saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
