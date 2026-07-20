# Limites et perspectives

## Limites actuelles

### Taux d'interprétabilité résiduel (~55-59% de features non interprétées)

Établi comme n'étant pas dû au volume de tokens (cf. `03_experiences_et_resultats.md`).

**Mise à jour (testé)** : la piste "robustesse du protocole de jugement" a été
vérifiée (`scripts/judge_robustness_check.py`, `RESULTS_TESTS.md` §13.1). En
répétant la question odd-one-out 5 fois par feature avec un ordre de mélange
différent à chaque fois : seulement 30,7% des features obtiennent une décision
unanime sur les 5 répétitions ; le taux agrégé d'interprétabilité bouge peu (45,3%→
48,7%) mais 31,3% des features changent individuellement de statut selon l'ordre de
présentation. **Confirmé : une partie substantielle du résidu non-interprété est due
au bruit du protocole de jugement (décision greedy unique, sensible à l'ordre), pas
nécessairement à un défaut réel des features.** Un vote majoritaire sur plusieurs
répétitions devrait être adopté comme protocole par défaut plutôt qu'une seule
décision greedy.

Pistes encore non testées par manque de temps, par ordre de coût croissant :

1. ~~**Robustesse du protocole de jugement**~~ **FAIT**, cf. ci-dessus.
2. **Qualité du contrôle négatif** : le contrôle négatif (`build_feature_examples_with_control`)
   est actuellement un document sous un quantile bas d'activation pour la feature
   testée, pas nécessairement un contre-exemple "propre" conceptuellement. Une
   feature réellement monosémantique pourrait échouer au test si le contrôle négatif
   choisi partage accidentellement une propriété de surface avec les exemples positifs.
3. **Capacité architecturale de l'extension** (`D_EXTRA=1024`, `K_EXTRA=32`) : non
   testée dans cette investigation (fixée dans les trois runs de validation). Une
   extension plus large ou plus/moins parcimonieuse pourrait changer le taux de
   features réellement monosémantiques indépendamment du corpus ou du volume.
4. **Fiabilité du juge selon la taille du modèle** : observée comme dégradée sur
   `gemma-3-270m-it` par rapport à un modèle plus grand lors de la validation locale
   initiale (`Context.md`). Le juge utilisé pour la validation à l'échelle (12B) est
   déjà le plus grand modèle disponible dans ce projet ; tester un modèle encore plus
   grand comme juge (sans nécessairement l'utiliser pour l'extraction d'activations)
   est une piste possible mais coûteuse.

### Comparaisons avec l'état de l'art incomplètes

`Context.md` (règle n°2) demande une comparaison documentée et systématique avec
SAELens. Une comparaison **de formule** a été faite (`docs/references.md`) : la
variance expliquée de `sae_lens.evals` et notre `compute_metrics` mesurent le même
concept avec une structure d'agrégation comparable (SAELens maintient elle-même deux
variantes, "legacy" et "corrigée", selon l'ordre d'agrégation par token vs global).
Une comparaison **chiffrée** sur les mêmes activations reste à faire : elle
nécessiterait de faire passer le chargement Gemma-3+GemmaScope-2 par
`transformer_lens.HookedTransformer` + `sae_lens.ActivationsStore` plutôt que le
chargeur custom du projet (`src/sae/gemma_scope_loader.py`, écrit précisément pour
contourner des incompatibilités de ce chemin de chargement direct) — arbitrage
coût/valeur non tranché en faveur de cette réintégration dans le temps disponible. La
comparaison avec `interp_embed` reste partielle (test optionnel dépendant d'une
installation non faite par défaut), et aucune comparaison avec "SAE Boost" n'a été
entreprise (implémentation officielle non identifiée à date).

### Biais de génération résiduel dans le corpus augmenté

17,5% des mails augmentés contiennent encore une ligne "Objet :" que les mails
originaux n'ont pas (réduit de 25,6% après correction du prompt système, mais pas
éliminé — le modèle générateur ne suit l'instruction que partiellement à température
0,8). Documenté dans `RESULTS_TESTS.md` §0 comme risque de contamination des features
de diffing par un artefact de formatage plutôt que par le contenu sémantique visé par
l'axe de perturbation. Non traité (coût estimé : un nouveau run d'augmentation complet,
~7h30 GPU).

### Facteurs non contrôlés dans le corpus augmenté

Les variantes augmentées sont générées par le même modèle (Gemma-3-12B-it) qui sert
aussi de juge d'interprétation et d'extracteur d'activations. Un style de génération
propre au modèle (tournures récurrentes, longueur, structure) pourrait constituer un
facteur de confusion partagé entre "ce qui rend une variante reconnaissable comme
augmentée" et "ce que le SAE apprend à détecter" — non quantifié dans cette
investigation.

## Perspectives pour la suite du stage

1. ~~Tester en priorité la robustesse du protocole de jugement (vote majoritaire)~~
   **FAIT** (cf. section "Limites actuelles" ci-dessus) — passer ce vote majoritaire
   en protocole par défaut de `odd_one_out_judge` (actuellement une fonction séparée,
   `scripts/judge_robustness_check.py`, à fusionner dans `src/sae/judge.py` si adopté).
2. Formaliser la comparaison **chiffrée** avec SAELens sur les mêmes activations
   (au-delà de la comparaison de formule déjà faite, cf. `docs/references.md`) —
   nécessite de faire passer le chargement par `HookedTransformer`/`ActivationsStore`.
3. Poursuivre la factorisation de `src/sae/saev5.py` vers l'architecture cible décrite
   dans `Context.md` (`src/models/`, séparation training/extraction) — dette technique
   qui n'affecte pas la validité des résultats mais complique la maintenance.
4. Dashboard interactif (Streamlit) — fonctionnalité future non commencée, mentionnée
   dès l'énoncé initial du projet.
5. ~~Exploiter le résultat de séparabilité linéaire des axes de perturbation... pour
   un cas d'usage concret de détection d'urgence/d'intention sur mails réels~~
   **FAIT** : `scripts/intent_urgency_probe.py`, `RESULTS_TESTS.md` §13.2 — sonde sur
   les labels faibles réels (regex, indépendants du corpus augmenté) : +27,0 points
   sur l'urgence, +42,6 points sur la réclamation par rapport à la baseline classe
   majoritaire. Reste à faire : évaluer sur un jeu de labels d'urgence/intention
   annotés manuellement plutôt que des labels faibles par regex (limite ci-dessous),
   et sur le Pipeline 2 (F2LLM) en plus du Pipeline 1 déjà testé.
