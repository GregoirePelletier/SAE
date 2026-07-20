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

### Comparaisons avec l'état de l'art

`Context.md` (règle n°2) demande une comparaison documentée et systématique avec
SAELens. **Fait** (`scripts/saelens_numeric_comparison.py`, `docs/references.md`) :
comparaison chiffrée, sur le même SAE natif sae-lens et les mêmes activations réelles,
entre notre formule de variance expliquée et les deux formules maintenues par
`sae_lens.evals` elle-même. Résultat notable : désaccord numérique important entre
les trois (0,41 / 0,83 / 1,00) causé par les activations massives de Gemma-3 — la
formule qui somme sur les dimensions avant de normaliser est mécaniquement dominée
par une seule dimension outlier et rapporte une variance quasi-totalement expliquée,
sans rapport avec la qualité de reconstruction réelle. Recommandation retenue : ne
jamais publier un score de variance expliquée sans préciser la formule exacte utilisée
sur ce modèle. La comparaison avec `interp_embed` reste partielle (test optionnel
dépendant d'une installation non faite par défaut), et aucune comparaison avec
"SAE Boost" n'a été entreprise (implémentation officielle non identifiée à date).

### Biais de génération résiduel dans le corpus augmenté

**Corrigé et mesuré** (`RESULTS_TESTS.md` §14.1) : 20,6% des mails augmentés
contenaient encore une ligne "Objet :"/"Subject :" que les mails originaux n'ont pas.
Fix appliqué au chargement (`load_augmented`, pas de régénération nécessaire — 0,0%
après fix) et effet mesuré sur le diffing complet : réduction de 65% et 49% du nombre
de features "significatives" sur les deux axes orthographiques (les plus confondables
avec l'artefact), effet modéré sur l'urgence (−7,5%/−3,1%), négligeable ailleurs.
Contrairement à l'hypothèse initiale, l'artefact ne dominait déjà pas la majorité des
features significatives à l'échelle du corpus complet (0,21% des features
significatives portaient un label "Subject:"/"Objet:", avant comme après) — son effet
mesuré est réel mais plus circonscrit que ce que suggérait l'observation initiale sur
l'échantillon test à 60 mails (§6, où l'artefact dominait 8/13 classements).

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
2. ~~Formaliser la comparaison **chiffrée** avec SAELens~~ **FAIT** (cf. section
   "Comparaisons avec l'état de l'art" ci-dessus) — a révélé un problème plus large
   (désaccord entre formules de variance expliquée sur activations à magnitude
   hétérogène) qu'une simple validation d'implémentation. Reste à faire : implémenter
   la métrique robuste aux outliers proposée (médiane des ratios par token).
3. Poursuivre la factorisation de `src/sae/saev5.py` vers l'architecture cible décrite
   dans `Context.md` (`src/models/`, séparation training/extraction) — dette technique
   qui n'affecte pas la validité des résultats mais complique la maintenance.
4. ~~Dashboard interactif (Streamlit)~~ **FAIT** : `src/visualization/dashboard.py`
   (`RESULTS_TESTS.md` §14.2) — vue d'ensemble, UMAP interactif, features (avec
   exemples positifs/négatifs), diffing, recherche par mot-clé, urgence/robustesse.
   Limite : recherche par mot-clé sur les labels déjà attribués, pas une ré-inférence
   BM25 live sur le vocabulaire latent complet (cf. `scripts/retrieval_demo.py` pour
   cette dernière) ; pas de déploiement serveur persistant, lancement manuel.
5. ~~Exploiter le résultat de séparabilité linéaire des axes de perturbation... pour
   un cas d'usage concret de détection d'urgence/d'intention sur mails réels~~
   **FAIT** : `scripts/intent_urgency_probe.py`, `RESULTS_TESTS.md` §13.2 — sonde sur
   les labels faibles réels (regex, indépendants du corpus augmenté) : +27,0 points
   sur l'urgence, +42,6 points sur la réclamation par rapport à la baseline classe
   majoritaire. Reste à faire : évaluer sur un jeu de labels d'urgence/intention
   annotés manuellement plutôt que des labels faibles par regex (limite ci-dessous),
   et sur le Pipeline 2 (F2LLM) en plus du Pipeline 1 déjà testé.
