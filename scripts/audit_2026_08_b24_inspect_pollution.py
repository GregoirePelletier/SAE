"""
scripts/audit_2026_08_b24_inspect_pollution.py — B.24 (suite) : le job
principal de `pipeline.py --mode compare` rapporte verdict="comparable",
0 features flaggées dans les deux modeles, mass=0.000 dans les deux cas.
Avant de lire ce résultat comme "pas de pollution détectée", vérifier s'il
ne reflète pas le même artefact que B.27 (graphe de co-occurrence
sous-dimensionné sur un petit nombre de features/communautés survivant les
filtres de fréquence) -- inspection pure des parquets déjà produits, aucun
recalcul.

Usage : sbatch slurm/validation/run_audit_b24_inspect_pollution.slurm
"""
from __future__ import annotations

import pandas as pd

OUT_DIR = "./results_audit_2026_08_b24_compare"

matches = pd.read_parquet(f"{OUT_DIR}/matches.parquet")
pol_a = pd.read_parquet(f"{OUT_DIR}/pollution_A.parquet")
pol_b = pd.read_parquet(f"{OUT_DIR}/pollution_B.parquet")

print(f"matches.parquet : {len(matches)} lignes, colonnes={list(matches.columns)}")
print(f"  corr : min={matches['corr'].min():.3f} max={matches['corr'].max():.3f} "
      f"mean={matches['corr'].mean():.3f} median={matches['corr'].median():.3f}")
print(f"pollution_A.parquet : {len(pol_a)} features dans le graphe (sur d_sae=2048)")
print(f"  communautés uniques : {pol_a['community'].nunique()}")
print(f"  pollution_score : min={pol_a['pollution_score'].min():.3f} "
      f"max={pol_a['pollution_score'].max():.3f} mean={pol_a['pollution_score'].mean():.3f} "
      f"std={pol_a['pollution_score'].std():.3f}")
thr_a = pol_a["pollution_score"].mean() + 2 * pol_a["pollution_score"].std()
print(f"  seuil (mean+2*std) = {thr_a:.3f} ; {(pol_a['pollution_score'] > thr_a).sum()} au-dessus")
print(f"pollution_B.parquet : {len(pol_b)} features dans le graphe (sur d_sae=2048)")
print(f"  communautés uniques : {pol_b['community'].nunique()}")
print(f"  pollution_score : min={pol_b['pollution_score'].min():.3f} "
      f"max={pol_b['pollution_score'].max():.3f} mean={pol_b['pollution_score'].mean():.3f} "
      f"std={pol_b['pollution_score'].std():.3f}")
thr_b = pol_b["pollution_score"].mean() + 2 * pol_b["pollution_score"].std()
print(f"  seuil (mean+2*std) = {thr_b:.3f} ; {(pol_b['pollution_score'] > thr_b).sum()} au-dessus")

# E.7 (deja confirme par lecture) : q_null_npmi95 est calcule par permutation
# mais jamais utilise pour flagger -- verifie ici sur la premiere execution
# reelle du module si ca aurait change quelque chose.
print(f"\nE.7 -- seuil par permutation jamais utilise pour flagger :")
print(f"  A: q_null_npmi95={pol_a['q_null_npmi95'].iloc[0]:.4f} (colonne calculee, non exploitee)")
print(f"  B: q_null_npmi95={pol_b['q_null_npmi95'].iloc[0]:.4f} (colonne calculee, non exploitee)")
