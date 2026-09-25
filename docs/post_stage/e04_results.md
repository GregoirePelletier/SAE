# E04 : comparer deux populations d'emails (Gemma-3-1B, couche 13, K_EXTRA=5)

Question : à partir des features qui distinguent deux groupes d'emails, peut-on formuler des
hypothèses lisibles sur ce qui les différencie, puis les vérifier sur d'autres emails ?

## Protocole

- Contraste : variantes générées avec un ton paniqué (`urgence__panique`, groupe cible) contre
  un ton calme (`urgence__calme`).
- Découverte sur FIT : pour chaque feature, fréquence d'activation dans chaque groupe
  (`corpus_diff_stats`), puis sélection des 200 features dont l'écart de fréquence dépasse 3 points.
- Génération : Qwen3.8-27B reçoit ces features (nom et écart de fréquence) et propose 8 hypothèses,
  en indiquant pour chacune le groupe qu'elle décrit. Les hypothèses sont figées avant toute
  lecture de CONFIRM.
- Vérification sur CONFIRM : 150 emails par groupe tirés au hasard (graine 42). Pour chaque
  email, le juge dit si l'hypothèse s'applique, sans connaître le groupe. Test de différence de
  proportions, correction de Benjamini-Hochberg sur les 8 tests.

## Résultats avec le découpage corrigé

Dossier `results_post_stage_e01_fit_1b_layer13_k5_v2_eval/` (job 50941). Découverte sur 2 025
emails paniqués et 1 988 calmes de FIT ; vérification sur 150 + 150 emails de CONFIRM (sur 850 et
825 disponibles).

| # | Hypothèse (résumé) | Groupe décrit | Paniqué | Calme | Écart | p corrigé |
|---|---|---|---:|---:|---:|---:|
| 1 | Marqueurs explicites d'urgence temporelle | paniqué | 150/150 | 0/150 | +1,00 | 10⁻⁶⁶ |
| 2 | Insistance sur les conséquences négatives du problème | paniqué | 52/150 | 0/150 | +0,35 | 10⁻¹⁵ |
| 3 | Expressions émotionnelles de détresse | paniqué | 121/150 | 0/150 | +0,81 | 10⁻⁴⁵ |
| 4 | Message centré sur la résolution d'un litige | paniqué | 72/150 | 4/150 | +0,45 | 10⁻¹⁹ |
| 5 | Absence d'urgence et de détresse | calme | 0/150 | 114/150 | −0,76 | 10⁻⁴¹ |
| 6 | Formules typiques d'une communication à fort enjeu | paniqué | 150/150 | 0/150 | +1,00 | 10⁻⁶⁶ |
| 7 | Termes administratifs formulés de façon pressante | paniqué | 25/150 | 0/150 | +0,17 | 10⁻⁷ |
| 8 | Message plus informatif, moins chargé émotionnellement | calme | 0/150 | 99/150 | −0,66 | 10⁻³⁴ |

Les 8 hypothèses sont confirmées, toutes dans le sens annoncé (les deux hypothèses sur le groupe
calme ont logiquement un écart négatif). Plusieurs écarts sont totaux (150 contre 0) : les
variantes sont générées à partir de consignes de ton, et le contraste repose sur des marqueurs
presque systématiques. Ce test montre que la chaîne découverte → hypothèses → vérification
fonctionne ; il ne montre pas qu'elle trouverait des différences subtiles sur des emails réels.

Les résumés `verification_rate` et `coverage` valent tous deux 1,0. Ils ne tiennent pas compte du
sens de l'écart ; le tableau ci-dessus est la lecture à retenir.

## Première exécution (découpage erroné, historique)

Dossier `results_post_stage_e01_fit_1b_layer13_k5/` (job 48957). Le générateur recevait alors un
log-odds à la place de l'écart de fréquence qui lui est décrit (erreur corrigée depuis dans
`e04_diffing.py`, test `tests/post_stage/test_e04_diffing_feature_blocks.py`), et toutes les
hypothèses visaient le groupe paniqué. Résultat : 6 hypothèses sur 8 confirmées dans le sens
annoncé, une confirmée dans le sens inverse (une « structure procédurale formelle » attribuée au
groupe paniqué s'est révélée caractéristique du groupe calme : 47 % contre 99 %), une non
significative (exclamations positives, 1 % contre 0 %).

## Limites

- Corpus synthétique : les groupes sont des consignes de génération, pas des catégories
  observées sur des emails réels.
- Vérification entièrement faite par le modèle juge, sans contrôle humain.
- Les deux groupes contiennent des variantes des mêmes emails d'origine ; le test de proportions
  les traite comme indépendantes, et le JSON ne conserve pas la liste des emails tirés.
- Un seul contraste testé ; le prompt de génération est rédigé pour ce contraste et doit être
  adapté pour un autre.

## Fichiers

- Script : `scripts/post_stage/e04_diffing.py` ; recettes `07_*` et `07b_*`.
- Résultat : `e04_diffing.json` dans le dossier de résultats.
