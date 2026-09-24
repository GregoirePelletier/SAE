"""
scripts/ce_loss_r0_trained_vs_random_test.py — Delta-CE (fidelite causale)
du decodeur ENTRAINE (R0) contre les decodeurs ALEATOIRES figes (C1/C1b),
sous les hyperparametres R0 exacts (K_EXTRA=5, D_EXTRA=1024, layer 31).

Contexte (audit de conformite R0 du rapport de stage) : `RESULTS_TESTS.md`
Sec58/Sec61 mesurent deja un Delta-CE ("l'extension reduit le Delta-CE de
1,298 a 0,404, -69%") mais cette mesure compare CORE SEUL a CORE+EXTENSION
ENTRAINEE -- PAS un decodeur entraine a un decodeur aleatoire fige, et sur
le checkpoint historique K_EXTRA=32, pas R0 (K_EXTRA=5). Les deux rapports
de stage decrivaient a tort cette mesure comme un analogue du Delta-FVE
(entraine vs aleatoire) -- corrige dans le texte du rapport de stage.
Ce script produit la VRAIE comparaison entraine-vs-aleatoire promise par le
texte, sous R0.

Reutilise trois checkpoints deja entraines (aucun reentrainement) :
  - R0 (decodeur entraine, `p1_frozen_core_d1024_k5.pt`,
    results_v27_ablation_classic_setup_k5_25m_layer31/)
  - C1 (decodeur aleatoire fige, init iso, meme fichier,
    results_v37_ablation_c1_sanity_frozen_decoder_stratified_qwen/)
  - C1b (decodeur aleatoire fige, init cov, meme fichier,
    results_v42_ablation_c1b_sanity_frozen_decoder_cov_init/)
sur le MEME echantillon de documents tenus a l'ecart (test split,
CORPUS_SPLIT_SEED), n=60 comme Sec58/Sec61 pour rester comparable.

CORRECTIF (premiere tentative, job 46994, invalidee) : R0_trained donnait un
Delta-CE (5,16) plus eleve que C1/C1b (~1,2) -- l'inverse de ce qu'annonce le
Delta-FVE (R0 reconstruit tres bien, C1/C1b quasiment pas, 98,9% de
l'extension morte, Sec98). Diagnostic : `SAEBoostResidualSAE`,
`FrozenDecoderExtendedSAE` et `FrozenCoreResidualSAE` partagent exactement
les memes noms/formes de parametres (aucune sous-classe n'ajoute de
nn.Parameter) -- verifie empiriquement ci-dessous (missing_keys/
unexpected_keys de load_state_dict imprimes par condition) : le chargement
`SAEBoostResidualSAE` pour les trois conditions n'est PAS la cause. La cause
identifiee est l'absence de tout filtrage dans `ce_loss_increase` (patch
la SAE a CHAQUE position, y compris BOS/tokens speciaux et outliers de norme
intra-document >4 sigma -- "massive activations" de Gemma-3, deja
documentees dans le rapport) alors que `extract_residual_acts`
(`valid_token_mask`/`norm_outlier_mask`, `src/analysis/activations.py`),
utilise pour TOUT entrainement/evaluation officiel de l'extension (FVE
compris), exclut systematiquement ces positions -- l'extension R0 n'a donc
JAMAIS ete entrainee ni evaluee sur ces positions, et un decodeur bien
calibre sur la distribution normale peut y produire une reconstruction
largement fausse la ou un decodeur quasi-mort (C1/C1b, ~1% de capacite
utile, Sec98) reste inoffensif faute d'y reagir du tout. Ce script patche
desormais la SAE UNIQUEMENT aux positions valides (meme masque que
l'extraction officielle), laissant l'activation d'origine aux positions
exclues -- coherent avec ce que Delta-FVE/l'entrainement mesurent deja.

Calcul GPU : forward Gemma-3-12b-it (bf16) requis (CE clean + CE patchee par
condition) -- job sbatch, pas de reentrainement.

Usage (SLURM, 1 GPU) :
    PYTHONPATH=. .venv/bin/python scripts/ce_loss_r0_trained_vs_random_test.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from scipy.stats import wilcoxon
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.analysis.activations import valid_token_mask, norm_outlier_mask
from src.config import (
    MODEL_ID, LAYER, D_EXTRA, K_EXTRA, RELEASE_ID, HOOK_TYPE, SAE_ID,
    LOCAL_SAE_ROOT, SAE_SNAPSHOT, LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH,
    CORPUS_SPLIT_SEED, DTYPE,
)
from src.data.preparation import build_email_train_test_corpus
from src.sae import load_gemma_scope_sae
from src.sae.frozen_core import SAEBoostResidualSAE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_DOCS = 60  # match Sec58/Sec61
MAX_LENGTH = 256  # match ce_loss_increase (crosslingual.py) par defaut
SIGMA_CLIP = 4.0  # match extract_residual_acts par defaut

CONDITIONS = {
    "R0_trained": "./results_v27_ablation_classic_setup_k5_25m_layer31",
    "C1_random_iso": "./results_v37_ablation_c1_sanity_frozen_decoder_stratified_qwen",
    "C1b_random_cov": "./results_v42_ablation_c1b_sanity_frozen_decoder_cov_init",
}
OUT_PATH = "./results_v27_ablation_classic_setup_k5_25m_layer31/cache/ce_loss_r0_trained_vs_random_results.json"


def load_ext_sae(tag: str, save_dir: str, core_sae) -> SAEBoostResidualSAE:
    ext_sae = SAEBoostResidualSAE(core_sae, d_extra=D_EXTRA, k_extra=K_EXTRA).to(DEVICE)
    ckpt_path = os.path.join(save_dir, f"p1_frozen_core_d{D_EXTRA}_k{K_EXTRA}.pt")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    load_result = ext_sae.load_state_dict(ckpt["state_dict"], strict=False)
    # Verifie explicitement l'hypothese "classe incompatible -> cles manquees" :
    # SAEBoostResidualSAE/FrozenDecoderExtendedSAE n'ajoutent aucun nn.Parameter
    # a FrozenCoreResidualSAE (verifie par lecture de code) -- cette assertion
    # le confirme empiriquement plutot que par inspection seule.
    print(f"[ce_r0]   [{tag}] load_state_dict : missing={list(load_result.missing_keys)}  "
          f"unexpected={list(load_result.unexpected_keys)}")
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            f"[{tag}] chargement incomplet du checkpoint {ckpt_path} -- "
            f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
        )
    ext_sae.eval()
    return ext_sae


def held_out_docs(n: int) -> list[str]:
    _, _, test_texts, _ = build_email_train_test_corpus(
        LOCAL_MAILS_PATH, LOCAL_AUGMENTED_MAILS_PATH, seed=CORPUS_SPLIT_SEED,
    )
    rng = np.random.default_rng(42)
    idx = rng.choice(len(test_texts), size=min(n, len(test_texts)), replace=False)
    return [test_texts[i] for i in idx]


@torch.no_grad()
def ce_loss_increase_masked(
    texts: list[str], model, tokenizer, sae, layer: int, device: str = DEVICE,
    max_length: int = MAX_LENGTH, sigma_clip: float = SIGMA_CLIP,
) -> list[dict]:
    """Variante de `crosslingual.ce_loss_increase` (meme convention `hidden_states[layer]`
    = sortie de `layers[layer-1]`) qui ne patche la reconstruction SAE qu'aux positions
    "valides" au sens de `extract_residual_acts` (`valid_token_mask` : hors PAD/tokens
    speciaux/1er token de contenu ; `norm_outlier_mask` : hors outliers de norme
    intra-document a sigma_clip sigma) -- positions sur lesquelles l'extension R0 n'a
    jamais ete entrainee ni evaluee (FVE compris). Aux positions exclues, l'activation
    d'origine est laissee inchangee (ni patch, ni degradation forcee). Un document par
    forward (CE par document, pas seulement par batch -- meme raison que
    `return_per_doc=True` dans `ce_loss_increase`)."""
    layer_module = model.model.language_model.layers[layer - 1]
    per_doc = []
    for text in texts:
        enc = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        labels = ids.masked_fill(attn == 0, -100)

        doc_ce_clean = float(model(input_ids=ids, attention_mask=attn, labels=labels).loss)

        mask_holder: dict[str, torch.Tensor] = {}

        def precompute_mask_hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            base_mask = valid_token_mask(ids, attn, tokenizer, skip_first_content_token=True)
            full_mask = norm_outlier_mask(h.float(), base_mask, sigma_clip=sigma_clip)
            mask_holder["mask"] = full_mask
            return output

        def patch_hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            orig_shape = h.shape
            h_flat = h.reshape(-1, orig_shape[-1])
            rec_flat = sae.decode(sae.encode(h_flat.to(torch.bfloat16)))
            rec = rec_flat.reshape(orig_shape).to(h.dtype)
            m = mask_holder["mask"].unsqueeze(-1).to(h.dtype)  # [B,T,1]
            out = m * rec + (1 - m) * h
            return (out, *output[1:]) if isinstance(output, tuple) else out

        h_mask = layer_module.register_forward_hook(precompute_mask_hook)
        try:
            model(input_ids=ids, attention_mask=attn, labels=labels)  # remplit mask_holder
        finally:
            h_mask.remove()

        h_patch = layer_module.register_forward_hook(patch_hook)
        try:
            doc_ce_patch = float(model(input_ids=ids, attention_mask=attn, labels=labels).loss)
        finally:
            h_patch.remove()

        n_valid = int(mask_holder["mask"].sum())
        n_total = int(attn.sum())
        per_doc.append({
            "ce_clean": doc_ce_clean, "ce_patched": doc_ce_patch,
            "delta_ce": doc_ce_patch - doc_ce_clean,
            "n_valid_tokens": n_valid, "n_total_tokens": n_total,
        })
    return per_doc


def main() -> None:
    print(f"[ce_r0] MODEL_ID={MODEL_ID}  LAYER={LAYER}  D_EXTRA={D_EXTRA}  K_EXTRA={K_EXTRA}")
    torch_dtype = torch.bfloat16 if DTYPE == "bf16" else torch.float32
    print("[ce_r0] Chargement Gemma-3-12b-it...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch_dtype).to(DEVICE).eval()

    print("[ce_r0] Chargement du core GemmaScope (fige, partage entre les 3 conditions)...")
    sae_dir = os.path.join(LOCAL_SAE_ROOT, "snapshots", SAE_SNAPSHOT, HOOK_TYPE, SAE_ID)
    core_sae = load_gemma_scope_sae(sae_dir=sae_dir, device=DEVICE, release_id=RELEASE_ID,
                                     sae_id=f"{HOOK_TYPE}/{SAE_ID}")

    print(f"[ce_r0] Echantillon de {N_DOCS} documents tenus a l'ecart (test split)...")
    texts = held_out_docs(N_DOCS)
    print(f"[ce_r0] {len(texts)} documents charges.")

    per_doc_by_cond = {}
    per_doc_detail = {}
    for tag, save_dir in CONDITIONS.items():
        print(f"[ce_r0] Condition {tag} ({save_dir})...")
        ext_sae = load_ext_sae(tag, save_dir, core_sae)
        per_doc = ce_loss_increase_masked(texts, model, tokenizer, ext_sae, layer=LAYER, device=DEVICE)
        per_doc_by_cond[tag] = [d["delta_ce"] for d in per_doc]
        per_doc_detail[tag] = per_doc
        mean_dce = float(np.mean(per_doc_by_cond[tag]))
        mean_valid_frac = float(np.mean([d["n_valid_tokens"] / max(d["n_total_tokens"], 1) for d in per_doc]))
        print(f"[ce_r0]   Delta-CE moyen ({tag}) = {mean_dce:.4f}  "
              f"(fraction moyenne de tokens patches = {mean_valid_frac:.3f})")
        del ext_sae
        torch.cuda.empty_cache()

    tests = {}
    for random_tag in ("C1_random_iso", "C1b_random_cov"):
        dce_trained = np.array(per_doc_by_cond["R0_trained"])
        dce_random = np.array(per_doc_by_cond[random_tag])
        stat, p = wilcoxon(dce_random, dce_trained, alternative="greater")
        n_favor_trained = int((dce_random > dce_trained).sum())
        tests[f"R0_trained_vs_{random_tag}"] = {
            "wilcoxon_statistic": float(stat), "p_greater": float(p),
            "n": len(dce_trained), "n_random_worse_than_trained": n_favor_trained,
            "mean_delta_ce_trained": float(dce_trained.mean()),
            "mean_delta_ce_random": float(dce_random.mean()),
        }
        print(f"[ce_r0] Wilcoxon (H1: Delta-CE[{random_tag}] > Delta-CE[R0_trained]) : "
              f"W={stat:.0f}  p={p:.3g}  ({n_favor_trained}/{len(dce_trained)} docs en faveur du SAE entraine)")

    out_data = {
        "model_id": MODEL_ID, "layer": LAYER, "d_extra": D_EXTRA, "k_extra": K_EXTRA,
        "sigma_clip": SIGMA_CLIP, "max_length": MAX_LENGTH,
        "n_docs": len(texts), "per_doc_detail": per_doc_detail,
        "per_doc_delta_ce": per_doc_by_cond, "wilcoxon_tests": tests,
        "note": "Premiere comparaison Delta-CE entraine-vs-aleatoire de ce depot sous R0 "
                "(K_EXTRA=5) -- Sec58/Sec61 comparaient core-seul vs core+extension sous "
                "K_EXTRA=32 (historique), pas un temoin aleatoire. Patch restreint aux "
                "positions valides (valid_token_mask + norm_outlier_mask, meme convention "
                "que extract_residual_acts) -- corrige une premiere tentative (job 46994) "
                "ou le patch s'appliquait a toutes les positions y compris BOS/outliers, "
                "jamais vus par l'extension a l'entrainement.",
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Ecrit : {OUT_PATH}")


if __name__ == "__main__":
    main()
