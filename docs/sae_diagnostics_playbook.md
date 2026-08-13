# Playbook de diagnostic — un run de SAE est-il sain ?

Checklist à suivre dans cet ordre avant de faire confiance à un résultat
d'interprétabilité produit par ce dépôt. Chaque métrique est déjà calculée
par le pipeline (`results.json`, `p1_top_extended_features.json`,
`*_history.json`) ou tracée par `scripts/generate_diagnostic_plots.py` —
aucune n'exige d'outillage supplémentaire.

## 1. Le SAE a-t-il fini de converger ?

`plots/p1_training_curves.html` / `plots/p2_sae_dim*_training_curves.html`
(générés par `scripts/generate_diagnostic_plots.py`, ou directement produits
par le run si postérieur à l'ajout du logging par step).

- **Loss (train)** encore en baisse nette à la dernière époque → le run est
  sous-entraîné, toute métrique en aval est prématurée.
- **Loss (validation) vs train** qui diverge (val remonte pendant que train
  continue de baisser) → surapprentissage sur le résidu, `rho_sae` en aval
  sera optimiste. Avant l'ajout du split de validation (Pipeline 1), ce
  diagnostic était structurellement invisible — tout run antérieur n'a que
  la loss train, pas de garde-fou contre ce cas.
- **`dead_frac`** qui ne redescend jamais après un pic initial → l'AuxK ne
  ranime pas les features mortes, le budget de capacité effectif est
  inférieur à `D_EXTRA`/`K_EXTRA` nominal.

## 2. La reconstruction est-elle fidèle ?

`results.json → P1_Gemma3_SAE.rho_sae` (corrélation de rang, résidu) et
`.fve_pretrained` (fraction de variance expliquée, core).

- `rho_sae` proche de 0 → le SAE d'extension n'apprend rien de plus que du
  bruit sur le résidu ; toute feature "interprétée" dessus est suspecte.
- `fve_pretrained` très bas (observé : 0,57 pour gemma-3-1b-it contre 0,83
  pour 12b, cf. `sweep_model_scale.html`) signale que le core lui-même
  n'explique déjà plus grand-chose de l'activation à ce point du réseau —
  aucune extension ne peut compenser ça.

## 3. Le budget de capacité est-il correctement utilisé ?

`results.json → dead_pct`.

- Une fraction de features mortes élevée (>50%, observé sur le run
  principal) n'est pas nécessairement un problème si `rho_sae`/`interp_rate`
  restent bons — mais une hausse brutale entre deux runs par ailleurs
  identiques (même `D_EXTRA`/`K_EXTRA`) signale un problème d'entraînement
  (LR, nombre d'époques), pas un choix de capacité.

## 4. Le taux d'interprétabilité mesuré est-il fiable en lui-même ?

`p1_top_extended_features.json → interp_score` (agrégé) et `rho_interp`
(par feature, `plots/rho_interp_distribution.html`).

- Le protocole *odd-one-out* est **bruité au niveau d'une feature isolée**
  (RESULTS_TESTS.md §13.1 : ~31% de décisions instables au simple
  réordonnancement des exemples) — ne jamais lire une feature individuelle
  comme "prouvée interprétable", seul le taux agrégé sur n≥150 est
  informatif.
- `rho_interp` qui ne se distingue pas entre features interprétées et non
  (`rho_interp_distribution.html`, deux histogrammes superposés) indique que
  le juge distingue "je reconnais un concept" de "je peux le classer par
  intensité" — deux capacités différentes, à ne pas confondre en lisant un
  seul chiffre agrégé.

## 5. L'effet observé résiste-t-il à un test statistique correctement choisi ?

Toujours utiliser `src/analysis/stats.py` (z-test à deux proportions pour
comparer deux taux d'interprétabilité, McNemar pour un test apparié sur les
mêmes features) plutôt qu'une lecture à l'œil de deux pourcentages. À ce
jour, un seul écart de balayage d'hyperparamètre isolé a atteint la
significativité conventionnelle (`|z|>1,96`) sur ce projet — layer 31 vs 24
— **sans correction multi-tests** ; tout le reste (K_EXTRA, D_EXTRA, volume,
hook-point, seed) reste dans le bruit à n=150. Seul le choix du modèle
extracteur/juge (`sweep_model_scale.html`) produit un effet massif et
répliqué à chaque palier — c'est le seul levier de ce projet identifié comme
réellement déterminant, tous les autres hyperparamètres testés à ce jour
sont interchangeables au bruit de mesure près.

## 6. Le juge est-il indépendant du générateur ?

Applicable uniquement quand le corpus de test inclut du texte généré par le
même modèle que le juge (corpus augmenté). Vérifié résolu négativement sur
ce projet (`RESULTS_TESTS.md` §48/§50/§52) — mais à revérifier explicitement
sur tout nouveau corpus/juge où cette configuration se reproduit, ce n'est
pas une propriété générique du protocole.

## Ordre de lecture recommandé pour un nouveau run

1 (convergence) → 2 (fidélité) → 3 (capacité) → 4+5 (interprétabilité et sa
significativité) → 6 (indépendance du juge, si applicable). Un run qui
échoue tôt dans cet ordre rend les diagnostics suivants non interprétables
— ne pas sauter à l'étape 4/5 sans avoir vérifié 1-3.
