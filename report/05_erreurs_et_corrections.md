# Inspection des erreurs et corrections

Ce chapitre consolide, dans l'ordre chronologique des quatre phases du stage
(cf. introduction), l'ensemble des dysfonctionnements identifiés et corrigés. Il
regroupe des informations autrement dispersées entre `Context.md` (journal d'audit) et
`RESULTS_TESTS.md` (journal d'expériences), pour répondre spécifiquement à l'exigence
d'un rapport de stage de documenter non seulement ce qui a fonctionné, mais aussi les
erreurs rencontrées et la façon dont elles ont été diagnostiquées.

## Phase 1 — Audit et fiabilisation initiale

Le pipeline hérité présentait plusieurs bugs bloquants ou silencieux, découverts par
exécution réelle (aucun n'était détecté par une simple lecture de code) :

| Bug | Symptôme | Cause | Correction |
|---|---|---|---|
| Overflow fp16 sur activations massives | `Loss=nan` dès la 1ère époque d'entraînement de `ExtendedSAE` | Gemma-3 présente des activations de norme très élevée (~1e5) dans son residual stream, au-delà du maximum représentable en fp16 (~65 504) | Précision par défaut passée à **bf16** partout, y compris en local (même plage d'exposant que fp32) |
| Cast dtype global cassant le backward | `RuntimeError: Found dtype BFloat16 but expected Float` après le fix bf16 | `FrozenCoreResidualSAE` castée en bloc via `.to(TORCH_DTYPE)`, alors que sa branche "extra" est conçue pour rester fp32 | Cast restreint à `core_sae` uniquement ; `residual` explicitement recasté en fp32 dans `forward()` |
| `hook_layer` figé en dur | Mauvaise couche extraite sur les modèles autres que 12b/layer-24 | Valeur `24` codée en dur, biais historique | Dérivée dynamiquement du `hook_name` résolu |
| `d_in` incorrect pour 270m | Dimensions incohérentes au chargement du SAE | Valeur supposée (1024) tirée de spécifications génériques publiques | Corrigée à **640**, confirmée empiriquement sur les poids réellement téléchargés |
| Route REST Neuronpedia cassée | Récupération de labels systématiquement vide | `/api/explanation/export` non fiable | Téléchargement direct des lots `.jsonl.gz` du bucket S3 public `neuronpedia-datasets` |
| Repos HuggingFace incorrects | Téléchargements en échec (404) | `gemma-scope-2-4b-it-res` figé en dur ; org F2LLM `Alibaba-NLP` inexistante | `download_sae.py` réécrit pour lire `src/config.py` ; org corrigée en `codefuse-ai` |
| Crashs sur petits corpus | `ValueError`/`TypeError` (UMAP/HDBSCAN, `corpus_diff_stats`) | Aucune garde pour 0 feature active ou `N_DOCS` trop petit pour l'initialisation spectrale | Dégradation propre ajoutée (résultat vide bien formé plutôt qu'exception) |
| Imports cassés | `ModuleNotFoundError` / échec de collecte pytest | Chemin `sys.path` erroné, import relatif incorrect, symbole inexistant importé | Chemins et imports corrigés ; test réécrit sur la fonction réellement maintenue |

Validation de cette phase : suite de tests (8/8 passants) et run complet de bout en
bout sur **Gemma-3-270M-it** (profil réduit, 6 Go VRAM), qui a permis d'itérer
rapidement avant tout passage à l'échelle coûteux sur 12B. Cette validation locale a
aussi révélé une **limite** qui deviendra le sujet central de la phase 2 : les
features de l'extension `ExtendedSAE` restaient `dead_feature` (jamais activées) avec
un budget d'entraînement modeste.

## Phase 2 — Diagnostic du taux d'interprétabilité (cœur du stage)

Sur le premier run à l'échelle complète disponible (Gemma-3-12B-it, `results_v9_full`),
la limite ci-dessus se manifestait différemment : les features n'étaient plus mortes
(0/10), mais seules 2/10 (20%) passaient le test d'auto-interprétation odd-one-out.
La démarche de diagnostic complète (élimination de l'hypothèse "features mortes",
inspection qualitative des exemples présentés au juge, lecture du code d'assemblage
du corpus) est détaillée au chapitre 3. Elle a révélé une **erreur de conception**,
plus qu'un bug au sens strict : le corpus utilisé pour entraîner le SAE d'extension
était construit **exclusivement** à partir de textes génériques (FineWeb-2/Wikipedia
filtrés par mots-clés), les emails originaux n'étant chargés que pour une visualisation
post-hoc, jamais vus pendant l'entraînement. Corrigée par
`build_email_train_test_corpus()` (corpus principal = emails + augmentés, split
group-aware par mail d'origine), cette correction a fait passer le taux
d'interprétabilité de 20% à ~41-45%, l'effet dominant identifié dans tout le stage.

Un bug secondaire a été découvert en instrumentant la validation de cette correction :
la sonde de classification multi-classe sur les 14 axes de perturbation échouait
silencieusement (exception attrapée, métrique `NaN`) car `LogisticRegression
(solver="liblinear")` ne supporte, dans les versions récentes de scikit-learn, que la
classification binaire. Corrigé par sélection dynamique du solveur (`lbfgs` au-delà de
deux classes), sans régresser le probe binaire préexistant.

## Phase 3 — Relecture critique face à la littérature de référence

Une relecture ligne à ligne du code du projet face au papier de référence
(*Interpretable Embeddings with Sparse Autoencoders*, Jiang, Sun et al. 2025,
Appendices C/E/F) a mis au jour quatre écarts, tous corrigés ou explicitement
documentés comme piste non intégrée :

- **Retrieval et clustering ciblé par sous-chaîne littérale** (`word in label`) plutôt
  que par similarité d'embedding sémantique (méthode de la référence) — ratait des
  labels sémantiquement liés mais formulés différemment, et retournait des faux
  positifs sur un mot partagé sans rapport de sens. Corrigé par une nouvelle fonction
  `select_latents_by_similarity` (embeddings **bge-m3**, choisi après comparaison
  empirique face à F2LLM : ce dernier donnait de bons résultats sur une requête mais
  des résultats sans rapport sur une autre, pooling dernier-token mal adapté aux
  labels courts en contexte multilingue).
- **Détection de corrélations "intéressantes" jamais câblée** : la fonction NPMI +
  communautés Louvain existait mais n'était appelée nulle part dans le pipeline
  principal, aucun filtre "NPMI élevé + similarité sémantique faible" (méthode de la
  référence) n'était implémenté. Nouvelle fonction `find_interesting_pairs` ajoutée et
  intégrée au pipeline principal.
- **Marqueurs erronés sur les exemples négatifs** dans le test de labellisation
  contrastive directe (protocole alternatif de la référence, Appendix C) : bug trouvé
  en écrivant le test de comparaison lui-même.
- **Prompt contenant un exemple de valeur JSON littéral** que le modèle recopiait
  verbatim pour ~59% des features testées — un "succès" en apparence (champ `confident`
  à `true`) qui masquait un défaut de prompt. Corrigé en remplaçant l'exemple par une
  notation `<placeholder>` explicite et une instruction de ne pas la recopier ; la
  proportion de copies a diminué mais le champ `confident` auto-rapporté reste peu
  fiable (`true` pour 150/150 features dans les deux runs, avant et après le fix), un
  résultat en soi (l'auto-évaluation de confiance d'un LLM ne peut pas se substituer à
  une mesure indépendante — cf. chapitre 5).
- **Biais résiduel "Objet:"/"Subject:"** dans le corpus augmenté : 20,6% des mails
  générés conservaient une ligne d'objet que les mails originaux n'ont pas, un artefact
  de formatage risquant d'être appris comme signal par le SAE plutôt que le contenu
  réel. Corrigé au chargement (`load_augmented`, 0,0% après fix) ; effet mesuré sur le
  diffing complet (réduction de 65% et 49% du nombre de features "significatives" sur
  les deux axes orthographiques les plus confondables avec l'artefact) — plus modeste
  qu'attendu à l'échelle du corpus complet, mais réel et désormais éliminé.

## Phase 4 — Mise à l'échelle et consolidation

- **Bug de configuration réseau récurrent (3 occurrences séparées)** : les nœuds de
  calcul du cluster SLURM sont isolés du réseau (`HF_HUB_OFFLINE=1`) ; le `MODEL_ID`
  par défaut (`src/config.py`) résout vers un identifiant de dépôt HuggingFace distant
  plutôt qu'un chemin disque local. Trois jobs différents (test de plausibilité
  d'explication, rerun Pipeline 2 avec F2LLM-330M) ont échoué pour cette raison avant
  d'être identifiés et corrigés par surcharge explicite de la variable d'environnement
  — un rappel que ce risque doit être vérifié systématiquement à chaque nouveau script
  SLURM plutôt que découvert à l'exécution.
- **Mélange de types dans l'affichage du dashboard** : une colonne de métriques
  mêlant valeurs numériques et texte libre provoquait un avertissement de conversion
  silencieuse côté pyarrow — corrigé en séparant l'affichage du texte libre.
- **Choix de largeur du SAE core reposant sur une donnée non vérifiée** : la
  documentation initiale du projet ne comparait la couverture Neuronpedia qu'entre 16k
  et 262k pour le modèle 12B (16k retenu, 262k écarté). Une largeur intermédiaire,
  65k, n'avait jamais été vérifiée spécifiquement pour ce modèle (seule une couverture
  ~98% pour 65k était documentée, mais mesurée sur un modèle différent,
  gemma-3-270m-it). Une vérification empirique systématique des quatre largeurs
  disponibles (16k/65k/262k/1m) a montré que 65k est en réalité la meilleure
  couverture pour 12B (87,8%, 57 551 features labellisées), meilleure que 16k (82,6%,
  13 535) — largeur adoptée pour le run de mise à l'échelle final (chapitre 3).
- **Suppression accidentelle d'un lien symbolique confondu avec un doublon** : lors
  d'un nettoyage disque, un dossier racine de 30 Go (`saes/`) a été identifié comme un
  doublon legacy de `local_data/saes/` (ancienne convention de nommage) et supprimé
  après confirmation. En réalité, `local_data/saes/gemma-scope-2-12b-it` était un
  **lien symbolique** vers ce même dossier, pas une copie indépendante — sa
  suppression a donc effacé la seule copie physique des poids SAE, laissant un lien
  cassé, détecté seulement à l'échec du job suivant (`ValueError` au chargement du
  SAE). Impact nul sur les résultats déjà produits (artefacts déjà écrits sur disque
  indépendamment des poids sources) ; poids retéléchargés et lien symbolique remplacé
  par un dossier réel pour éliminer la source de confusion. Leçon retenue : vérifier
  `ls -la`/`readlink` avant de supprimer un chemin présenté comme "doublon", pas
  seulement sa taille ou son nom.
- **Erreur de terminologie sur la nature du corpus "original"** : tout le rapport,
  jusqu'à cette phase, employait "mails réels"/"emails réels"/"corpus réel EDF"
  pour désigner `Mails.tsv`. Cette formulation est incorrecte : `Mails.tsv` est
  lui-même un jeu de données synthétique, produit par un travail antérieur du
  laboratoire EDF R&D (indépendant de ce stage), pas de la correspondance client
  authentique. Ni le corpus "original" (`Mails.tsv`) ni les variantes augmentées
  générées pendant ce stage ne sont donc des données réelles au sens strict.
  Corrigé par une relecture terminologique complète du rapport et des documents
  techniques (`docs/`, `RESULTS_TESTS.md`, `README.md`) : le terme "réel(s)" est
  remplacé par "original(aux)" partout où il désignait ce corpus, avec une
  clarification explicite ajoutée au chapitre 2. Erreur sans impact sur la
  validité des résultats eux-mêmes (aucune métrique ne dépend de l'authenticité du
  corpus), mais une imprécision qu'un rapport de stage se doit de corriger.

## Constat transversal

Sur l'ensemble du stage, la quasi-totalité des bugs significatifs ont été détectés par
**exécution réelle et inspection directe des résultats intermédiaires** (valeurs de
perte, exemples présentés au juge, couverture mesurée empiriquement), jamais par
relecture de code seule. Ceci a orienté une pratique systématique : ne jamais publier
un résultat sans avoir inspecté au moins un échantillon qualitatif des données qui
l'ont produit — pratique qui a directement permis de découvrir l'erreur de conception
du corpus (phase 2, le résultat le plus important du stage) et le biais "Objet:"
(phase 3), tous deux invisibles à la seule lecture des métriques agrégées.
