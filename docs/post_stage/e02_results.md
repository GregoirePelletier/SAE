# E02 : catalogue de features (Gemma-3-1B, couche 13, K_EXTRA=5)

Question : quelle part des features du SAE peut recevoir un nom compréhensible ? 300 features
sont tirées par tranche de fréquence (150 de CORE, 150 de l'extension EXTRA), avec des exemples
pris uniquement dans FIT. Chacune est soumise au modèle juge Qwen3.8-27B par un test de l'intrus
(`odd_one_out_judge`) : une feature est « interprétable » si le juge retrouve l'exemple qui ne lui
correspond pas.

## Résultats avec le découpage corrigé

Fichier `results_post_stage_e01_fit_1b_layer13_k5_v2/e02_feature_registry.json` (job 50665).

| Statut | CORE | EXTRA | Total |
|---|---:|---:|---:|
| interprétable | 76 | 131 | 207 (69,0 %) |
| non concluant | 74 | 19 | 93 (31,0 %) |

L'extension produit nettement plus de features interprétables que CORE (131 contre 76 sur 150),
ce qui est cohérent avec son rôle : apprendre des directions propres aux emails, là où CORE est un
catalogue généraliste.

## Première exécution (découpage erroné, historique)

Dossier `results_post_stage_e01_fit_1b_layer13_k5/` : 197 interprétables (77 CORE, 120 EXTRA),
102 non concluantes, 1 sans exemples suffisants. L'écart avec le rejeu (207 contre 197) porte sur
un autre checkpoint et un autre tirage de features ; il ne faut pas le lire comme une amélioration.
Ni l'un ni l'autre n'est le résultat de référence du rapport de stage (197/300, Gemma-3-12B, autre
entraînement), malgré la coïncidence du chiffre de la première exécution.

## Limites

- Aucune vérification humaine : le plan (§7) prévoit d'en contrôler 60 à 100 à la main.
- Trois statuts seulement (interprétable, non concluant, support insuffisant) ; le plan en
  proposait quatre, avec une distinction entre features syntaxiques et mixtes qui demanderait un
  second prompt.
- Le registre contient des extraits d'emails ; il reste dans le dossier de résultats, hors Git.

## Fichiers

- Script : `scripts/post_stage/e02_feature_registry.py`.
- Résultat : `e02_feature_registry.json` dans le dossier de résultats.
