# Limites et perspectives

## Limites actuelles

### Taux d'interprétabilité résiduel (~55-59% de features non interprétées)

Établi comme n'étant pas dû au volume de tokens (cf. `03_experiences_et_resultats.md`).
Pistes non testées par manque de temps, par ordre de coût croissant :

1. **Robustesse du protocole de jugement** : `odd_one_out_judge` ne fait qu'une seule
   génération greedy (`do_sample=False`) par décision. Un vote majoritaire sur
   plusieurs échantillonnages (température > 0), ou une reformulation de la question
   (ordre des exemples, format de réponse), pourrait changer le taux mesuré sans
   changer la qualité réelle des features — à tester avant d'investir dans des
   changements plus coûteux.
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
SAELens. À date : `src/analysis/metrics.py` réimplémente les formules FVE/NMSE/L0 "en
alignement" avec SAELens (justifié techniquement — nécessité de scorer à la fois un
SAE natif sae-lens et le `FrozenCoreResidualSAE` custom du projet, aux API
différentes) mais aucune comparaison chiffrée formelle des deux implémentations n'a
été produite. De même, la comparaison avec `interp_embed` reste partielle (test
optionnel dépendant d'une installation non faite par défaut), et aucune comparaison
avec "SAE Boost" n'a été entreprise (implémentation officielle non identifiée à date).

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

1. Tester en priorité la robustesse du protocole de jugement (vote majoritaire) — coût
   de calcul faible (pas de réextraction d'activations, juste re-jugement), gain
   potentiel direct sur le résultat central du stage.
2. Formaliser la comparaison avec SAELens (règle n°2 de `Context.md`), non faite
   systématiquement à date.
3. Poursuivre la factorisation de `src/sae/saev5.py` vers l'architecture cible décrite
   dans `Context.md` (`src/models/`, séparation training/extraction) — dette technique
   qui n'affecte pas la validité des résultats mais complique la maintenance.
4. Dashboard interactif (Streamlit) — fonctionnalité future non commencée, mentionnée
   dès l'énoncé initial du projet.
5. Exploiter le résultat de séparabilité linéaire des axes de perturbation
   (§5.4 de `03_experiences_et_resultats.md`) pour un cas d'usage concret de détection
   d'urgence/d'intention sur mails réels (pas seulement sur le corpus augmenté),
   objectif final énoncé du projet.
