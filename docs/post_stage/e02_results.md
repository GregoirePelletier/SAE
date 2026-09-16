# E02 — Registre de features CORE/EXTRA (1B/layer13/K5, FIT)

300 features labellisées (150 CORE, 150 EXTRA), sélection stratifiée par
fréquence (`feature_selection_stratified_by_frequency`), exemples issus
exclusivement de FIT, juge Qwen (`odd_one_out_judge`).

## Répartition des statuts

| Statut | n | % |
|---|---:|---:|
| interpretable | 197 | 65,7 % |
| unclear (odd-one-out échoué, support présent) | 102 | 34,0 % |
| insufficient_support (`dead_feature`) | 1 | 0,3 % |

**197/300 = 65,7 %** — coïncide presque exactement avec le chiffre historique
du rapport défendu (197/300 = 65,7 %, cf. plan §2.1), sur un checkpoint 1B
entraîné sur FIT seul (pas le même run, pas la même échelle de modèle que la
référence historique). À lire comme une réplication rassurante de l'ordre de
grandeur, pas une identité — les deux mesures ne partagent ni le même
entraînement ni la même sélection de features candidates.

## Simplifications assumées

- Taxonomie de statut réduite à 3 valeurs (le plan §7 en propose 4 :
  unclear/syntactic/mixed/insufficient_support) -- `odd_one_out_judge`
  renvoie un score binaire, pas la distinction syntaxique/mixte. Un second
  prompt de classification serait nécessaire pour l'ajouter ; pas fait ici.
- **Vérification humaine non faite** (`human_verification_pending: true`
  dans le manifeste) — le plan cible 60-100 features vérifiées par un
  humain ; nécessite la participation de Grégoire, pas quelque chose
  d'automatisable.
- Budget symétrique 150/150 (CORE/EXTRA) tel que voulu par le plan (§4.4 :
  même budget de candidats labellisés aux deux bras), pas encore utilisé
  pour une comparaison CORE-vs-FULL à budget apparié dans une application
  (viendra avec E03).

## Fichier

- `results_post_stage_e01_fit_1b_layer13_k5/e02_feature_registry.json`
  (local uniquement, contient des extraits de mails synthétiques —
  autorisés dans ce cadre mais non versionnés, `results_*/` gitignored).
- Script : `scripts/post_stage/e02_feature_registry.py`.
