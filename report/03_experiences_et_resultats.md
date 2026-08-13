# Expériences et résultats

## 1. Problématique

Le pipeline complet (extraction d'activations → SAE → labellisation) fonctionnait de
bout en bout sans erreur, mais produisait un taux de succès faible au test
d'auto-interprétation odd-one-out des features d'extension (celles qui ne sont pas
couvertes par Neuronpedia et dépendent donc entièrement du juge LLM local pour être
labellisées). Sur le dernier run complet disponible avant cette phase du stage
(`results_v9_full`, Gemma-3-12B-it, 10 features jugées) : seules 2 features sur 10
(20%) passaient le test.

Question posée : **ce taux faible est-il dû à un budget d'entraînement (nombre de
tokens) insuffisant pour l'extension SAE, ou à un autre facteur ?**

## 2. Diagnostic

### 2.1. Élimination de l'hypothèse "features mortes"

Un budget d'entraînement insuffisant se traduirait typiquement par des features
d'extension qui ne s'activent jamais (`dead_feature`), auquel cas le juge ne peut même
pas être interrogé (`odd_one_out_judge` retourne directement `dead_feature` si moins de
3 exemples positifs sont disponibles). Or l'inspection du run `results_v9_full` montre
**0 feature morte sur les 10 jugées** : les features s'activaient bel et bien, avec des
exemples positifs disponibles. L'échec du test n'était donc pas un problème de features
inactives.

### 2.2. Inspection qualitative des exemples présentés au juge

L'inspection directe des `pos_examples` stockés pour les features non interprétables a
révélé le problème : les neuf exemples "positifs" présentés au juge pour une même
feature n'avaient, dans plusieurs cas, **aucun concept commun identifiable** — par
exemple un extrait sur un rappel produit iPad, un extrait sur le système carcéral
norvégien, un extrait de recette de cuisine et un extrait sur l'agriculture canadienne,
présentés ensemble comme activant fortement la même feature. Le test odd-one-out ne
peut, par construction, pas réussir dans ce cas : il n'y a pas de concept partagé à
partir duquel identifier l'intrus.

### 2.3. Cause racine

La lecture du code d'assemblage du corpus (`src/sae/saev5.py`, bloc `__main__`) a
montré que le corpus utilisé pour échantillonner le réservoir de résidus servant à
entraîner l'extension (`N_TOKENS_EXTRA_TRAIN` tokens tirés de `train_texts`) était
constitué **exclusivement** de textes génériques (FineWeb-2/Wikipedia filtrés par
mots-clés sur trois domaines substituts : énergie, sport, support client). Les mails
originaux et leurs variantes augmentées étaient chargés séparément (`email_texts`) et
utilisés **uniquement après l'entraînement**, pour une visualisation UMAP — jamais vus
par le SAE d'extension pendant son entraînement.

Autrement dit : le SAE d'extension apprenait des directions représentant des concepts
Wikipedia génériques et hétérogènes, jamais des concepts liés au contenu réel des
emails EDF. Les "exemples positifs" incohérents observés en 2.2 ne sont pas une
anomalie du juge, mais une conséquence directe et attendue de ce corpus
d'entraînement.

## 3. Correction

Deux changements ont été apportés au pipeline (`src/data/preparation.py`,
`src/sae/saev5.py`) :

1. **Nouveau corpus principal** : mails originaux (`Mails.tsv`) + variantes augmentées
   acceptées (`augmented_mails.jsonl`, 39 949 variantes issues de 13 axes de
   perturbation contrôlée — émotion, registre, orthographe, urgence). Ce corpus devient
   celui qui entraîne l'extension SAE et le `PhraseLevelSAE`. Split train/test
   **group-aware** par mail d'origine, pour éviter qu'une variante augmentée d'un mail
   de test ne fuite dans le train.
2. **Corpus secondaire** : le corpus generic (énergie/sport/support) est conservé mais
   réduit et cantonné à un usage post-hoc (démonstration préexistante de diffing
   cross-domaine), sans plus jamais participer à l'entraînement.

## 4. Protocole de validation

Trois runs à l'échelle complète sur Gemma-3-12B-it, tous avec le nouveau corpus
principal (emails + augmentés, ~41 200 documents d'entraînement / ~2 200 documents de
test), afin d'isoler l'effet du **volume** de tokens d'entraînement de l'effet du
**domaine** du corpus (déjà corrigé identiquement dans les trois runs) :

| Run | `N_TOKENS_EXTRA_TRAIN` | Durée d'exécution |
|---|---|---|
| Ablation volume bas | 100 000 | 3h11min37s |
| **Run principal** (budget par défaut) | 500 000 | 3h01min53s |
| Ablation volume haut | 2 000 000 | 2h21min01s |

Le nombre de features jugées par run a été porté de 10 à **150** (`N_FEATURES_TO_LABEL`)
pour disposer d'une puissance statistique correcte sur le taux d'interprétabilité
observé (avec n=10, l'incertitude sur un taux observé est trop large pour conclure).

## 5. Résultats

### 5.1. Taux d'interprétabilité (test odd-one-out)

| Corpus | `N_TOKENS_EXTRA_TRAIN` | n features jugées | Features mortes | Taux d'interprétabilité |
|---|---|---|---|---|
| Generic (avant correction) | 500 000 | 10 | 0 | **20,0%** (2/10) |
| Emails + augmentés | 100 000 | 150 | 0 | **40,7%** (61/150) |
| Emails + augmentés | 500 000 | 150 | 0 | **45,3%** (68/150) |
| Emails + augmentés | 2 000 000 | 150 | 0 | **44,7%** (67/150) |

L'écart-type binomial attendu sur un taux observé avec n=150 est d'environ 4,1 points
de pourcentage (approximation de Wald à 95%, soit un intervalle de confiance
d'environ ±8 points). Les trois taux mesurés à corpus identique (40,7% / 45,3% / 44,7%)
sont statistiquement indistinguables les uns des autres au regard de cette incertitude,
malgré un facteur 20 entre le budget de tokens le plus faible et le plus élevé testés.

### 5.2. Interprétation

- **Corriger le domaine du corpus (generic → emails), à volume comparable (500 000
  tokens), plus que double le taux d'interprétabilité (20,0% → 45,3%).** C'est le
  facteur dominant identifié dans cette investigation.
- **Faire varier le volume de tokens d'un facteur 20, à domaine fixé (emails), ne
  produit aucun effet mesurable.** Le SAE d'extension n'est pas limité par le volume de
  tokens dès 100 000 tokens, une fois le corpus correctement ciblé sur le domaine ; le
  porter à 2 000 000 n'apporte aucun gain supplémentaire observable.
- **Réponse à la question posée** : le taux de détection de l'intrus faible observé
  initialement n'était **pas** un problème de volume d'entraînement, mais un problème
  de **contenu/domaine du corpus** d'entraînement — les emails, cible réelle du
  projet, n'entraient jamais dans les données servant à entraîner l'extension SAE.

### 5.3. Qualité des labels obtenus

Contraste direct entre les labels obtenus avant et après correction, pour les features
qui passent le test :

- **Avant** (corpus generic) : labels générés à partir d'extraits Wikipedia sans lien
  avec le domaine (ex. artefacts de formatage — listes, ponctuation).
- **Après** (corpus emails) : `Réclamations Clients`, `Litiges Factures`, `Résiliation
  Énergie`, `Menace Résiliation`, `Demande Urgente`, `Problèmes énergie`, `Insatisfaction
  client` — des concepts directement alignés avec les objectifs métier du projet
  (détection d'urgence, détection d'intention, réclamations).

*Ces labels précis illustrent une catégorie de concepts récurrente, pas des features
individuellement stables : voir §12 pour la faible reproductibilité inter-seed des
labels exacts (28,2% de recouvrement), le taux agrégé restant la seule mesure fiable.*

### 5.4. Résultat additionnel : séparabilité linéaire des axes de perturbation

Une sonde de classification logistique a été ajoutée pour mesurer si les codes latents
du SAE permettent de séparer linéairement les 14 classes du corpus principal (13
combinaisons axe/niveau de perturbation + "original"). Résultats (run à 100 000
tokens, premier disposant du correctif nécessaire pour la classification multi-classe,
cf. §6) :

| Pipeline | Précision de classification (5-fold) |
|---|---|
| Pipeline 1 (Gemma-3 + GemmaScope) | 93,5% |
| Pipeline 2 (F2LLM + PhraseLevelSAE) | 79,3% |

Ce résultat, obtenu comme sous-produit de la validation du corpus, est directement
pertinent pour les objectifs du projet (détection d'urgence, détection d'intention) :
il indique que les représentations latentes du SAE encodent l'information nécessaire à
ces tâches de façon linéairement séparable, sur un corpus qui simule des variations
réalistes de ton et d'urgence dans les emails clients.

**Réserve** (`RESULTS_TESTS.md` §37) : un baseline TF-IDF sans aucun
contenu sémantique atteint 87,0% sur cette même sonde à 14 classes — soit ~93% du
signal ci-dessus déjà présent dans le simple texte brut, par templating lexical de la
génération augmentée, indépendamment de toute structure apprise par le SAE. Ce chiffre
ne peut donc pas, seul, être lu comme une preuve de compréhension sémantique ; la sonde
sur labels faibles indépendants du corpus augmenté (§7.2) reste la preuve la plus
fiable des objectifs urgence/intention.

## 6. Choix du solveur pour la sonde multi-classe

La sonde de classification multi-classe (§5.4) utilise `LogisticRegression`
avec sélection dynamique du solveur : `liblinear` pour le probe binaire
préexistant (énergie/sport), `lbfgs` (support natif du cas multinomial)
au-delà de deux classes — `liblinear` ne supporte que la classification
binaire dans les versions récentes de scikit-learn.

## 7. Suites données au diagnostic

Deux analyses complémentaires ont été menées après la validation du §5, pour répondre
plus directement aux objectifs métier du stage et à la limite identifiée au §5.4/§7
(taux résiduel non expliqué) — toutes deux réutilisent des activations déjà en cache,
sans calcul GPU supplémentaire pour la seconde.

### 7.1. Le résidu non-interprété est-il dû au protocole de jugement lui-même ?

Le protocole odd-one-out (`odd_one_out_judge`) ne prend qu'une seule décision greedy
par feature. Pour tester sa robustesse à l'ordre de présentation des exemples (un
biais de position bien documenté chez les LLM en contexte de choix multiple), la même
question a été répétée 5 fois par feature, avec un ordre de mélange différent à
chaque fois, pour les 150 features déjà jugées du run principal
(`scripts/judge_robustness_check.py`, job SLURM 40672 — recharge uniquement le modèle
comme juge, aucune réextraction d'activations).

| Métrique | Valeur |
|---|---|
| Taux d'interprétabilité, décision unique (référence) | 45,3% |
| Taux d'interprétabilité, vote majoritaire sur 5 répétitions | 48,7% |
| Features dont la décision change selon l'ordre | 31,3% (47/150) |
| Taux d'accord moyen entre les 5 répétitions | 80,3% |
| Features avec décision unanime (0/5 ou 5/5) | 30,7% (46/150) |

Le taux agrégé ne change que marginalement, mais la fiabilité de la décision prise
pour une feature individuelle est faible : moins d'un tiers des features obtiennent
une décision unanime sur 5 répétitions identiques (mêmes exemples, ordre différent).
**Une partie substantielle du taux d'échec observé au §5 reflète donc le bruit du
protocole de jugement plutôt qu'un défaut réel des features testées.**

### 7.2. Le SAE prédit-il l'urgence et l'intention sur des mails originaux ?

Le résultat du §5.4 (séparabilité des axes d'augmentation synthétiques) a été
complété par un test sur des labels **indépendants du corpus augmenté** : des labels
faibles par expression régulière, déjà calculés sur le texte brut des mails originaux
(`src/data/dataset.py::INTENT_KEYWORDS_FR` — réclamation, résiliation, remboursement,
information, urgence), appliqués aux 3 300 mails originaux du split d'entraînement
(`scripts/intent_urgency_probe.py`, zéro calcul GPU — réutilise
`p1_all_doc_acts_ext_d1024.pt` déjà en cache).

| Intention | Prévalence | Précision sonde | Baseline (classe majoritaire) | Écart |
|---|---|---|---|---|
| Urgence | 29,3% | 97,7% | 70,7% | **+27,0 pts** |
| Réclamation | 55,1% | 97,7% | 55,1% | **+42,6 pts** |
| Information | 18,2% | 87,8% | 81,8% | +6,0 pts |
| Remboursement | 14,5% | 84,5% | 85,5% | −1,0 pt |

Ce résultat répond directement aux deux objectifs "détection d'urgence" et
"détection d'intentions" énoncés dans le cadrage initial du projet : les codes
latents du SAE séparent très
nettement l'urgence et la réclamation, sur des mails originaux non augmentés, avec un
gain net important sur la baseline naïve. Le remboursement ne bat pas sa baseline
(déjà forte du fait du déséquilibre de classe, 85,5% de négatifs) — à interpréter
comme une limite du label faible par regex pour cette catégorie plutôt que comme un
échec du SAE, sans donnée annotée manuellement pour trancher entre les deux
hypothèses.

## 8. Qualité de l'explication document-level : fidélité et plausibilité

Question distincte des sections précédentes (qui évaluent une feature isolée ou une
capacité globale du corpus) : pour UN document donné, l'explication produite par le
pipeline (les features les plus actives et leurs labels) est-elle une bonne
explication ? Deux propriétés indépendantes ont été testées.

### 8.1. Fidélité (l'explication reflète-t-elle ce qui pilote réellement la décision ?)

Test par ablation (`scripts/explanation_fidelity_test.py`) : sur 200 mails originaux par
intention (urgence, réclamation, information, remboursement), correctement classés
par une sonde logistique, ablation des 10 features les plus contributives à la
décision vs 10 features actives aléatoires vs les 10 moins contributives.

| Intention | Chute top-10 | Chute random-10 | Ratio top/random |
|---|---|---|---|
| Réclamation | 0,576 | ~0,000 | 576 225× |
| Remboursement | 0,9997 | 0,0009 | 1 058× |
| Information | 0,9998 | 0,0040 | 251× |
| Urgence | 0,612 | ~0,000 | 42 837× |

Résultat sans ambiguïté : les features désignées comme explication portent
réellement la décision (leur ablation fait s'effondrer la prédiction), contrairement
à des features actives choisies au hasard (effet quasi nul). L'explication n'est pas
une justification a posteriori déconnectée du mécanisme réel.

### 8.2. Plausibilité (un lecteur trouve-t-il l'explication convaincante ?)

Test par choix forcé au niveau document (`scripts/explanation_plausibility_test.py`,
juge Gemma-3-12B-it — un jugement comparatif comme l'odd-one-out plutôt qu'une
auto-évaluation de confiance isolée, dont la fiabilité limitée a été confirmée par
ailleurs, cf. §7.1 et `RESULTS_TESTS.md` §15.4) : sur 60 mails
réels, le juge choisit entre l'ensemble réel des concepts les plus actifs et un
ensemble de concepts tirés au hasard, sans savoir lequel est réel.

**Résultat : 71,7% de choix corrects (43/60) contre 50% attendu au hasard**
(z ≈ 3,4, p < 0,001) — l'explication a une valeur perçue réelle, significativement
au-dessus du hasard, mais loin d'être parfaite : dans 28,3% des cas le juge préfère le
décoy aléatoire, cohérent avec le taux d'interprétabilité résiduel (~45-55%) mesuré
par ailleurs.

### 8.3. Protocole d'évaluation complet du dépôt

Ces deux tests s'inscrivent dans un protocole plus large couvrant l'ensemble des
méthodes du dépôt sous conditions fixées (`docs/evaluation_protocol.md`) : 16
capacités recensées (reconstruction, labellisation par les deux protocoles,
robustesse du jugement, séparabilité synthétique et réelle, fidélité/plausibilité de
l'explication, retrieval, corrélations, diffing, choix du backbone d'embedding), avec
pour chacune sa commande de reproduction et la méthode alternative à laquelle elle
est comparée. Un script de consolidation (`scripts/consolidate_evaluation_report.py`)
assemble automatiquement tous les résultats disponibles d'un run en un rapport
unique, également exposé dans le dashboard Streamlit.

Un résultat notable de ce passage en revue systématique : la comparaison du backbone
d'embedding du Pipeline 2 (F2LLM-v2-80M vs -330M, "assez grand") donne un résultat
**mixte** — le modèle plus grand reconstruit légèrement mieux (NMSE −7,5%) et sépare
un peu mieux le corpus de diffing générique (+2 points), mais sépare légèrement MOINS
bien les axes email (−2,2 points), la métrique la plus proche des objectifs
métier. Aucun écart n'est de l'ordre d'un problème majeur ; pas de justification
claire pour préférer l'un à l'autre sur ce projet.

## 9. Limites de cette investigation

- Le taux d'interprétabilité obtenu après correction (~41-45%) reste loin de 100% :
  environ 55 à 59% des features d'extension restent non interprétables par le juge même
  sur le corpus corrigé. Cette investigation a établi que ce résidu n'est **pas**
  expliqué par le volume de tokens (testé jusqu'à 2 000 000, sans effet) ; les causes
  possibles restantes (robustesse du protocole de jugement lui-même, qualité du
  contrôle négatif, capacité architecturale de l'extension) n'ont pas été testées dans
  le temps disponible et sont proposées comme pistes de poursuite
  (cf. `04_limites_et_perspectives.md`).
- Les trois runs de l'ablation volume n'ont pas partagé de cache d'extraction
  d'activations (chaque run réextrait les activations du modèle 12B depuis zéro) : un
  choix délibéré pour garantir qu'aucune contamination de cache entre configurations
  différentes ne puisse biaiser la comparaison, au prix d'un coût de calcul plus élevé
  (~8h30 GPU cumulées pour les trois runs plutôt qu'un partage possible de l'étape
  d'extraction, identique entre les trois configurations).

## 10. Ablation de mise à l'échelle du volume d'entraînement et de labellisation (v12)

Question posée en fin de stage, dans le même esprit que l'ablation du §4/§5 mais sur
un axe différent : une fois le domaine du corpus corrigé (§3) et le résidu
non-interprété partiellement expliqué par le bruit du protocole de jugement (§7.1),
un passage à l'échelle du **volume d'entraînement et de labellisation**, sans aucun
autre changement de méthode, améliore-t-il la proportion de features labellisées
avec succès et les différents scores de reconstruction/séparabilité ?

Trois leviers ont été augmentés simultanément par rapport au run principal du §4
(`results_v10_emails_main`), dans un nouveau run unique (`results_v12_scaled_65k`)
regroupant l'ensemble des analyses du dépôt (toutes les capacités du protocole
d'évaluation, §8.3) :

| Paramètre | Run principal (§4) | Run v12 (échelle) | Facteur |
|---|---|---|---|
| Largeur du SAE core (couverture Neuronpedia) | 16k (82,6%, 13 535 labels) | **65k (87,8%, 57 551 labels)** | — (meilleure couverture, cf. Chapitre 1) |
| `EPOCHS_EXTRA` (SAE d'extension, Pipeline 1) | 10 | **40** | ×4 |
| `EPOCHS` (Phrase-Level SAE, Pipeline 2) | 30 | **100** | ×3,3 |
| `N_FEATURES_TO_LABEL` (features jugées) | 150 | **600** | ×4 |
| `N_TOKENS_EXTRA_TRAIN` | 500 000 | 500 000 (inchangé) | — (déjà démontré non limitant, §5.2) |
| Backbone Pipeline 2 | F2LLM-v2-80M | F2LLM-v2-330M | (condition fixée du protocole d'évaluation, §8.3) |

### 10.1. Résultats du run combiné

| Métrique | Run principal (16k) | Run v12 (65k, échelle) |
|---|---|---|
| Taux d'interprétabilité (odd-one-out) | 45,3% (68/150) | **53,7% (322/600)** |
| Taux d'interprétabilité, vote majoritaire (5 répétitions) | 48,7% | 55,5% |
| Accord moyen entre 5 répétitions du juge | 80,3% | 79,3% (stable) |
| `clf_acc_email_axes` (Pipeline 1, 14 classes) | 93,5% | 91,9% |
| `clf_acc_email_axes` (Pipeline 2, 14 classes) | 79,3%/77,2% (80M/330M) | 76,7% |
| NMSE Pipeline 2 | 0,0745 (80M) / 0,0689 (330M) | 0,0667 |

Le taux d'interprétabilité global progresse nettement (45,3%→53,7%), mais ce run
combine trois leviers à la fois (largeur, époques, nombre de features jugées) — cf.
§10.3 pour leur décomposition. Les scores de classification/séparabilité (déjà très
élevés, >90% pour Pipeline 1) ne progressent pas et reculent même très légèrement,
cohérent avec un plafond déjà proche pour cette tâche plutôt qu'un signal de
dégradation.

### 10.2. Le rang par magnitude n'est pas un bon proxy de l'interprétabilité

Analyse à coût nul (aucun calcul GPU, relecture de l'ordre de sélection déjà en
cache) pour savoir si les 450 features supplémentaires labellisées par le scale-up
diluent la statistique ou apportent un signal réel :

| Sous-ensemble (rang par magnitude d'activation) | n | Taux d'interprétabilité |
|---|---|---|
| Rang 1-150 (budget du run principal) | 150 | 44,0% |
| Rang 151-600 (apport du scale-up) | 450 | **56,9%** |

Résultat notable : le sous-ensemble de tête (rang 1-150) donne un taux
statistiquement indiscernable du run principal à 16k (44,0% vs 45,3%, écart bien en
deçà de l'incertitude binomiale à n=150) — un bon signe de cohérence entre les deux
runs. Mais les features de rang inférieur (151-600), moins actives en moyenne, sont
en réalité **plus** interprétables que celles de tête. La magnitude d'activation
moyenne n'est donc pas un proxy fiable de l'interprétabilité potentielle d'une
feature : restreindre la labellisation aux features de plus forte magnitude exclut
systématiquement des candidates au moins aussi bonnes, voire meilleures.

### 10.3. Alignement label/feature et test de plausibilité

Le test de plausibilité (§8.2) dépend d'un chemin de labels Neuronpedia dérivé
de `SAE_ID`, pour rester aligné sur la largeur de SAE réellement utilisée par
le run évalué (16k pour le run principal, 65k ici) — pour toute feature
d'index < 16 384, un chemin figé sur 16k afficherait au juge un label
appartenant à une feature totalement différente du dictionnaire 16k. Une fois
ce chemin correctement dérivé : **88,3% (53/60)**, contre 71,7% sur le run
principal — une amélioration nette (+16,6 points), cohérente avec un
catalogue de features labellisées bien plus riche (65k core à 87,8% de
couverture + 600 features d'extension jugées, contre 16k à 82,6% + 150
seulement) pour construire l'ensemble présenté au juge. Le test de fidélité,
qui ablate par index et n'utilise le label que pour l'annotation cosmétique
des exemples exportés, reste numériquement valide dans les deux
configurations. Détail complet : `RESULTS_TESTS.md` §17.3/17.6.

### 10.4. Décomposition largeur / époques / capacité (ablations isolées)

Trois runs à facteur unique, isolant respectivement la largeur du SAE core, le
nombre d'époques et la capacité de l'extension (`D_EXTRA`/`K_EXTRA`, toutes choses
égales par ailleurs, `N_FEATURES_TO_LABEL=150` dans les trois cas), permettent
d'attribuer la hausse du §10.1 à sa cause plutôt qu'à l'effet combiné des quatre
leviers — même démarche que l'ablation de volume de tokens du §5 :

| Run | Largeur | Époques | Capacité extension | Interprétabilité |
|---|---|---|---|---|
| Run principal (référence) | 16k | 10/30 | 1024/32 | 45,3% (68/150) |
| Largeur seule | 65k | 10/30 | 1024/32 | 43,3% (65/150) |
| Époques seules | 16k | 40/100 | 1024/32 | 41,3% (62/150) |
| Capacité seule | 16k | 10 | 2048/64 | 40,0% (60/150) |
| Combiné (tranche rang 1-150, §10.2) | 65k | 40/100 | 1024/32 | 44,0% (66/150) |

Les trois ablations isolées donnent des taux statistiquement indistinguables du run
principal (écart-type binomial attendu ≈4,1 points à n=150) : **aucun des trois
leviers pris isolément n'améliore le taux d'interprétabilité odd-one-out**, un
résultat qui rejoint celui déjà établi pour le volume de tokens (§5.2). La hausse
observée sur le run combiné (45,3%→53,7%, §10.1) s'explique donc presque entièrement
par l'effet de composition du §10.2 (les features de rang inférieur sont mieux
interprétées), pas par une meilleure qualité d'entraînement due à la largeur, aux
époques ou à la capacité. **Conclusion sur la question posée en tête de ce
chapitre** : scaler la pipeline améliore réellement la plausibilité de l'explication
(§10.3) et la richesse du catalogue de features labellisées disponibles, mais pas le
taux brut d'interprétabilité odd-one-out — celui-ci reste gouverné par le domaine du
corpus (§3-5) et par le protocole de jugement lui-même (§7.1), pas par le volume
d'aucun des quatre paramètres d'échelle testés à ce jour (tokens, largeur, époques,
capacité).

## 11. Sanity check : le protocole d'évaluation distingue-t-il un SAE entraîné d'un décodeur aléatoire ?

Question posée par la lecture critique de Korznikov et al. (2026, *Sanity Checks
for Sparse Autoencoders*, chapitre 1) : leurs résultats montrent qu'un SAE dont le
décodeur est figé à une initialisation aléatoire (jamais entraîné) égale un SAE
réellement entraîné sur interprétabilité automatique, sparse probing et édition
causale — remettant en cause la validité de ces métriques comme preuve d'un
apprentissage de features significatif. Reproduit sur ce projet
(`FrozenDecoderExtendedSAE`, `src/sae/frozen_core.py`) : même conditions que le run
principal (16k, 150 features jugées, `EPOCHS_EXTRA=10`, `D_EXTRA=1024`/`K_EXTRA=32`),
seul le décodeur de l'extension reste figé aléatoire (`requires_grad_(False)`),
l'encodeur s'entraîne normalement.

| Métrique | Run principal (décodeur entraîné) | Frozen Decoder (décodeur figé) | Écart (significativité) |
|---|---|---|---|
| Interprétabilité odd-one-out | 45,3% (68/150) | 29,3% (44/150) | −16,0 points (z=2,86, p<0,01) |
| Classification en aval (14 classes) | 93,5% | 91,2% | −2,3 points (z=2,86, p<0,01) |

**Résultat nuancé** : sur l'interprétabilité odd-one-out, l'écart est net et
statistiquement solide — contrairement au résultat du papier, notre protocole
distingue clairement un décodeur entraîné d'un décodeur aléatoire. Mais sur la
classification en aval, l'écart est réel tout en restant faible en proportion du
signal total : un décodeur purement aléatoire capture déjà 91,2% de précision, la
quasi-totalité de ce qu'atteint le décodeur entraîné (93,5%) — ce volet réplique
largement le constat du papier sur le sparse probing. **Conséquence pour la lecture
des résultats de ce rapport** : le taux d'interprétabilité odd-one-out (§3-10) reste
une preuve solide d'apprentissage de features significatives ; la sonde de
classification en aval (§5.4), en revanche, doit être lue avec prudence — un score
élevé ne suffit pas à lui seul à prouver un apprentissage réel, une fraction
substantielle du signal étant déjà disponible avec un décodeur aléatoire de taille
comparable. Les tests de fidélité et de plausibilité (§8), de nature causale
différente (ablation directe des features, jugement humain-like sur le document
entier), ne sont pas concernés par cette réserve. Détail complet et calculs de
significativité : `RESULTS_TESTS.md` §19.

## 12. Ablation de variance de seed

Question posée par la littérature (*Unstable Features, Reproducible Subspaces*,
arXiv:2606.12138 ; *Toward Identifiable Sparse Autoencoders*, arXiv:2605.31245) :
les features individuelles d'un SAE varient-elles substantiellement d'un seed
d'entraînement à l'autre, à corpus et hyperparamètres identiques ? `SEED` a été
découplé du split train/test (`CORPUS_SPLIT_SEED`, nouveau, défaut 42 inchangé)
pour isoler la variance d'entraînement du SAE de toute variance de corpus. Run à
`SEED=123` (sinon identique au run principal) :

| Métrique | SEED=42 | SEED=123 | Écart |
|---|---|---|---|
| Interprétabilité odd-one-out | 45,3% (68/150) | 47,3% (71/150) | +2,0 points (non significatif) |
| `clf_acc_email_axes` | 93,5% | 91,3% | −2,2 points |
| Recouvrement exact des labels interprétables | — | 22/78 = 28,2% | |

Le taux agrégé d'interprétabilité est stable entre seeds, mais seulement 28,2% des
labels sont des chaînes identiques entre les deux runs — les deux seeds recouvrent
des thèmes similaires (adresses, contrats énergie, réclamations, coupures,
urgence) mais rarement la même feature exacte. **Conséquence pour la lecture de ce
rapport** : les features individuelles citées en exemple sont représentatives d'une
catégorie de concepts récurrente, pas des atomes stables et reproductibles du
dictionnaire ; seul le taux agrégé d'interprétabilité doit être lu comme une mesure
fiable de la qualité du SAE. Détail : `RESULTS_TESTS.md` §21.

## 13. Biais multilingue du juge (français vs anglais traduit)

Question posée par la littérature sur l'interprétabilité multilingue (Resck et al.
2025 ; *Sparse Autoencoders Can Capture Language-Specific Concepts Across Diverse
Languages*, arXiv:2507.11230) : le juge interprète-t-il mieux les mêmes features
quand les exemples sont présentés en anglais plutôt qu'en français ? Les 150
features déjà jugées sont retraduites (un appel Gemma-3 par feature) et rejugées
intégralement en anglais (`scripts/multilingual_judge_bias_test.py`), sans
réextraction ni réentraînement.

| Métrique | Français original | Anglais traduit | Écart |
|---|---|---|---|
| Interprétabilité odd-one-out (n=145) | 46,9% | 45,5% | −1,4 point (non significatif) |
| Features changeant de statut (FR↔EN) | — | 56/145 = 38,6% | |

**Résultat nul sur l'hypothèse testée** : pas de différence significative entre
français et anglais traduit — aucun biais systématique détecté envers l'anglais sur
ce protocole et ce corpus. Le taux de retournement individuel (38,6%), supérieur à
celui déjà mesuré pour le réordonnancement des exemples (§7.1, 31,3%), est
symétrique (27 flips dans un sens, 29 dans l'autre) plutôt qu'orienté vers
l'anglais — cohérent avec un bruit de traduction générique plutôt qu'un déficit
structurel de l'auto-interprétation en français. Renforce le constat du §7.1 :
le protocole odd-one-out à décision unique reste sensible à toute perturbation de
surface (ordre ou langue), justifiant le vote majoritaire comme protocole par
défaut. Détail et limites assumées (traduction par le même modèle juge, pas de
réentraînement sur corpus anglais natif) : `RESULTS_TESTS.md` §22.

## 14. Ablation volume à grande échelle (25M tokens)

Suite directe du chapitre 1 ("Perspectives critiques") : le papier SAE Boost
montre qu'un SAE résiduel a besoin de 100-200M tokens pour converger sans
dégrader la performance générale — 50 à 100x au-dessus du volume testé dans
l'ablation initiale (§5, jusqu'à 2M). Le corpus emails+augmentés (~6M tokens)
étant insuffisant pour cette échelle, le réservoir de résidus est complété par
un filler échantillonné sur FineWeb2-fr sans filtre thématique (le filler
isole un effet de volume brut de tokens, pas de pertinence thématique),
ajouté **uniquement** au réservoir résiduel (`volume_filler_texts`), jamais
au corpus utilisé pour la sélection des features à labelliser ni pour la
sonde de classification, pour ne pas réintroduire le biais de domaine
diagnostiqué au chapitre 3. Le réservoir de résidus est memory-mapped sur
disque plutôt qu'alloué en RAM (`open_mmap_reservoir`, `src/sae/saev5.py`),
ce qui permet de viser des volumes proches du seuil du papier (jusqu'à 200M
tokens) sans demande mémoire proche de la capacité totale d'un nœud de
calcul.

**Résultat à 25M tokens** (`results_v13_ablation_volume25m`) : **81/150 =
54,0%** d'interprétabilité (odd-one-out), contre 45,3% pour le run principal
(500k tokens) et 44,7% pour l'ablation initiale à 2M — écart numérique de
+8,7 points mais **non significatif** (test z sur deux proportions, z=-1,50,
seuil \|z\|>1,96). `clf_acc_email_axes` recule légèrement (93,5% → 91,3%).
Porter le volume à 25M tokens (12x l'ablation initiale, toujours 50-100x en
dessous du seuil 100-200M de SAE Boost) ne change donc pas la conclusion
qualitative : le problème diagnostiqué en §2 était bien le domaine du
corpus, pas son volume brut. Cet écart directionnel (+8,7 points, non
significatif) est du même ordre de grandeur que celui observé
indépendamment pour l'ablation `K_EXTRA=5` (+9,4 points, §16, également non
significatif) — à prendre comme une piste à répliquer plutôt qu'un résultat
établi. Un run au seuil exact 100-200M reste à exécuter. Détail complet :
`RESULTS_TESTS.md` §23.3/§54.

## 15. Fidélité du steering (`steer_and_decode`) : jamais testé, résultat très hétérogène par intention

Le steering (`steer_activations`/`steer_and_decode`, `src/sae/sae_shared.py`)
existe dans le dépôt depuis le début mais n'était jamais réellement exercé : seule
`run_steering_demo` l'utilise, et uniquement pour une vérification géométrique
superficielle (cosinus avant/après suppression/amplification d'une feature, sans
tâche en aval). Le test d'ablation existant (§8) ablate déjà des features par
intention, mais directement dans l'espace des codes SAE, sans jamais appeler
`decode()`.

Question testée (`scripts/steering_fidelity_test.py`, zéro calcul LLM,
réutilise les activations en cache et le checkpoint entraîné de
`results_v10_emails_main`) : si on décode réellement le code stimulé vers l'espace
résidu puis qu'on RÉ-ENCODE ce résidu décodé, l'intervention (suppression des
top-10 features explicatives d'une intention) tient-elle à travers cet aller-retour ?

| Intention | Chute en place (témoin) | Chute steer_and_decode | Ratio |
|---|---|---|---|
| réclamation | 0,576 | 1,000 | 1,74× |
| remboursement | 1,000 | 0,016 | 0,02× |
| information | 1,000 | 0,004 | 0,00× |
| urgence | 0,646 | 0,584 | 0,90× |

Résultat hétérogène et contre-intuitif : le round-trip decode/encode neutralise
quasi entièrement l'intervention pour deux intentions sur quatre (remboursement,
information), la préserve pour urgence, et l'amplifie même pour réclamation.
Conclusion : `steer_and_decode` n'est pas un mécanisme d'intervention causale
fiable et prévisible à partir du simple test d'ablation en place — son effet
dépend fortement de la structure de corrélation entre features propre à chaque
intention. Détail complet (protocole, limite méthodologique du pooling par
document, fuite résiduelle mesurée) : `RESULTS_TESTS.md` §24.

## 16. Ablation `K_EXTRA=5` (SAE Boost)

Le papier SAE Boost trouve k=5 optimal dans son étude de sensibilité pour un
SAE résiduel — notre `K_EXTRA=32` par défaut n'avait jamais été testé en
dessous de cette valeur. Sur `results_v13_ablation_k_extra5` : **82/150 =
54,7%** d'interprétabilité contre 45,3% pour le run principal — écart de
+9,4 points, **non significatif** (z=-1,62) mais le plus proche du seuil
conventionnel de toutes les ablations de ce chapitre. `rho_sae` (fidélité de
reconstruction du résidu) recule sensiblement (0,906 → 0,849), cohérent avec
un budget de capacité par token plus faible. Direction cohérente avec
l'hypothèse du papier, mais à confirmer (cf. §14 pour la coïncidence
directionnelle avec l'ablation volume). Détail complet : `RESULTS_TESTS.md`
§25.

Deux seeds supplémentaires (7, 99) confirment la direction sur les 3 : 50,0%
et 55,3% d'interprétabilité, contre 54,7% pour le seed original et 45,3%
pour le baseline `K_EXTRA=32`. Test z groupé sur les 3 seeds combinés
(240/450 vs 68/150) : z=1,70, p≈0,089 — toujours sous le seuil conventionnel,
et probablement optimiste (le calcul groupé ignore la corrélation
intra-seed, et le baseline `K_EXTRA=32` n'a qu'un seul seed pour comparer).
La piste `K_EXTRA=5` se renforce (3/3 seeds dans la même direction) sans
pour autant passer d'hypothèse à répliquer à résultat établi. Détail :
`RESULTS_TESTS.md` §45.

## 17. Évaluation quantitative du retrieval Latent Terms

`src/sae/retrieval/latent_terms.py` (BM25 sur le vocabulaire latent d'un SAE
entraîné par pure reconstruction, Clavié et al. 2026) n'était exercé que par
inspection visuelle sur données de substitution. Protocole quantitatif
(`scripts/latent_retrieval_precision_eval.py`) : Precision@10/@20
contre les labels faibles d'intention (§8), sur 4 requêtes en paraphrase, comparé
à une baseline TF-IDF, sur les 3480 mails originaux.

| Intention | P@10 Latent Terms | P@10 TF-IDF |
|---|---|---|
| réclamation | 1,00 | 1,00 |
| remboursement | 1,00 | 0,00 |
| information | 1,00 | 0,20 |
| urgence | 0,00 | 0,80 |

Précision parfaite et nette supériorité sur TF-IDF pour 2 intentions sur 4
(remboursement, information) malgré des requêtes ne reprenant pas les mots exacts
du label — généralisation sémantique réelle. Échec complet sur urgence (0,00),
diagnostiqué précisément : la requête active bien des features latentes non
nulles, mais un seul document sur 3480 dans tout le corpus partage une
intersection non nulle avec elles — limite structurelle du BM25 sur vocabulaire
latent très parcimonieux (k=16), pas un bug ni un raté sémantique. Détail
complet : `RESULTS_TESTS.md` §26.

## 18. Ablation "échelle du modèle" : un effet dose-réponse net et significatif

Toutes les ablations précédentes gardent le modèle extracteur/juge fixé à
gemma-3-12b-it. Test avec gemma-3-1b-it et gemma-3-4b-it (+ leurs GemmaScope
dédiés) à la place de 12b-it :

| Modèle | Taux interp. | z vs 12b | `clf_acc_email_axes` |
|---|---|---|---|
| gemma-3-1b-it | **12,0%** (18/150) | 6,38 (significatif) | 88,2% |
| gemma-3-4b-it | **28,0%** (42/150) | 3,12 (significatif) | 92,0% |
| gemma-3-12b-it (run principal) | **45,3%** (68/150) | — référence | 93,5% |

**Effet dose-réponse net, monotone et significatif à chaque palier** — 12,0% →
28,0% → 45,3%, avec même la comparaison directe 1b vs 4b significative
(z=-3,46). C'est, de très loin, le plus fort effet mesuré dans tout ce projet :
tous les autres écarts (largeur, époques, capacité, volume, seed, K_EXTRA)
restent entre 1 et 9 points, tous non significatifs.

Indice que l'origine est plutôt la qualité du JUGE que celle des features
elles-mêmes : `clf_acc_email_axes` (séparabilité linéaire des axes de
perturbation, indépendante du juge LLM) suit une pente beaucoup plus douce
(88,2% → 92,0% → 93,5%, 5,3 points d'écart total contre 33,3 points pour le
taux d'interprétabilité). Lecture qualitative cohérente : le texte d'hypothèse
généré est confus pour 1B (inversion logique cause/conséquence), plus solide
pour 4B, pleinement cohérent pour 12B. Interprétation retenue : la capacité de
RAISONNEMENT du juge (formuler et vérifier un concept partagé entre 9 exemples)
est probablement le facteur limitant à petite échelle, plus que la qualité des
représentations latentes elles-mêmes — piste actionnable pour la suite :
séparer les rôles extracteur/juge pour isoler laquelle des deux capacités
domine réellement cet effet. Détail complet : `RESULTS_TESTS.md` §28.

Complète également le balayage de largeur du SAE core (16k/65k/262k, job
41487) : 46,7% à 262k, aucun écart significatif (z=-0,23) — confirme qu'aucune
largeur testée ne change l'interprétabilité, et souligne par contraste à quel
point l'effet de l'échelle du MODÈLE ci-dessus est hors norme parmi tous les
leviers testés dans ce projet. Détail complet : `RESULTS_TESTS.md` §29.

## 19. Balayage `MATRYOSHKA_DIM` (F2LLM) : dégradation graduelle, pas abrupte

`MATRYOSHKA_DIM` (troncature de l'embedding F2LLM, défaut 320) n'avait jamais
été varié — fait notable découvert en creusant : F2LLM-80M a `hidden_size=320`,
exactement égal au défaut, donc la "troncature" était un no-op pur pour ce
backbone dans toutes les comparaisons précédentes (§16), jamais remarqué.
4 runs sur F2LLM-160M (`hidden_size=640`), `MATRYOSHKA_DIM` ∈ {64, 128, 320,
640(complet)} :

| Dimension | Fraction du complet | `clf_acc_email_axes` |
|---|---|---|
| 64 | 10% | 72,8% |
| 128 | 20% | 75,0% |
| 320 (défaut) | 50% | 76,8% |
| 640 (complet) | 100% | **77,4%** |

Augmentation monotone mais à rendements très décroissants — 10% des dimensions
conservent déjà 94% de la performance du plein embedding. Dégradation
graduelle, pas de chute brutale à aucun point testé. Ni la documentation F2LLM
ni ce projet ne confirment un entraînement Matryoshka (MRL) au sens strict,
mais le comportement empirique est cohérent avec cette hypothèse. Conclusion
pratique : tronquer à 64-128 dimensions coûterait peu en performance pour un
gain de calcul/mémoire de 5-10x, si jamais nécessaire. Détail complet :
`RESULTS_TESTS.md` §31.

## 20. Balayage du layer d'extraction (12/31/41) : le choix par défaut n'est pas optimal

Le layer 24 (Pipeline 1) a toujours été choisi sur un seul critère : la
couverture des labels Neuronpedia (Chapitre 1), jamais sur un critère
d'interprétabilité mesuré empiriquement. Trois runs à facteur unique (même
protocole que le run principal, seul `LAYER` varie) comblent ce manque :

| Layer | Taux interp. | z vs layer 24 |
|---|---|---|
| 12 | 45,3% (68/150) | z=0,00, p=1,000 |
| 24 (défaut, run principal) | 45,3% (68/150) | — |
| 31 | **58,0% (87/150)** | **z=2,20, p=0,028** |
| 41 | 52,7% (79/150) | z=1,27, p=0,204 |

Layer 31 est le seul écart nominalement significatif du balayage (+12,7
points), et layer 12 reproduit le taux du layer 24 au feature près — une
coïncidence numérique, pas un signe de couplage entre les deux. Comme pour
le reste de ce chapitre (aucune correction multi-tests appliquée, cf. §21
ci-après pour le rappel), ce résultat doit être lu comme une piste à
répliquer sur un second seed avant adoption, pas comme un changement de
configuration déjà acquis -- mais c'est la première fois que le choix du
layer 24 apparaît potentiellement sous-optimal sur le critère qui compte
réellement pour ce projet. Détail complet : `RESULTS_TESTS.md` §51.

## 21. Balayage du point d'extraction (`resid_post` vs `attn_out` vs `mlp_out`)

Dernier volet du balayage : à layer fixe (24), comparer le residual stream
(`resid_post`, utilisé partout dans ce rapport) aux deux autres points de
hook publiés par GemmaScope-2 pour ce modèle (chapitre 1) :

| Point d'extraction | Taux interp. | z vs `resid_post` |
|---|---|---|
| `resid_post` (défaut, run principal) | 45,3% (68/150) | — |
| `mlp_out` | 52,7% (79/150) | z=1,27, p=0,204 |
| `attn_out` | 35,3% (53/150) | z=-1,77, p=0,078 |

Aucun des deux écarts individuels contre `resid_post` n'atteint la
significativité conventionnelle, mais **`mlp_out` contre `attn_out`
directement l'est nettement : z=3,02, p=0,0025**. Cohérent avec un a priori
raisonnable sur l'architecture : `attn_out` capte l'entrée de la projection
de sortie de l'attention (`self_attn.o_proj`), un espace multi-head pas
encore recombiné vers la dimension du residual stream, tandis que `mlp_out`
(sortie du MLP, déjà reprojetée) en reste proche. Illustration qualitative :
la feature choisie pour la démo de steering sur `attn_out` s'est labellisée
"variable assignment `p =`" -- un type de concept qui ne ressemble à rien
de ce qu'on observe sur `resid_post`/`mlp_out`, cohérent avec un espace
moins structuré sémantiquement à ce point précis du réseau. `attn_out` est
le point d'extraction le moins prometteur des cinq configurations testées
dans ce chapitre (layers 12/24/31/41 + `mlp_out`). Détail complet :
`RESULTS_TESTS.md` §53.

**Note transversale sur les §20-21** : ni ce balayage ni le reste du
chapitre 3 n'appliquent de correction pour comparaisons multiples (rappel
déjà fait au §11 pour le sanity check, et documenté comme lacune du chapitre
en `04_limites_et_perspectives.md`) -- sur ~20 ablations à ce stade, un seul
résultat significatif à p<10⁻⁹ (§18, échelle du modèle) et une poignée entre
0,03 et 0,10 (K_EXTRA=5, layer 31, `mlp_out` vs `attn_out`) sont à traiter
comme des pistes cohérentes entre elles, pas comme des résultats
individuellement établis.

## 22. Le core seul égale-t-il core+extension sur les métriques en aval ?

Question centrale pour juger si `FrozenCoreResidualSAE` ajoute réellement du
signal exploitable, ou seulement des dimensions supplémentaires sans effet :
sur les mêmes activations en cache, mêmes folds de validation croisée, comparer
le SAE core seul (16384 dimensions) à core+extension (17408 dimensions) sur la
silhouette (structure de cluster, axes email) et deux sondes de classification
déjà utilisées dans ce chapitre (axes email 14 classes, diffing energy/sports).

| Métrique | Core seul | Core+extension |
|---|---|---|
| `silhouette` | 0,0086430470 | 0,0086430488 |
| `clf_acc_email_axes` | 93,67% | 93,59% |
| `clf_acc_sae` (energy/sports) | 60,0% | 60,0% |

Un test de McNemar apparié (mêmes documents, mêmes folds) ne détecte aucune
différence significative sur la sonde email (p=0,26) ; sur la sonde
energy/sports, les deux conditions classent les 600 documents de façon
**strictement identique** (zéro désaccord, p=1,0). L'extension n'apporte donc
ni gain ni dégradation mesurable sur ces sondes linéaires -- ni pollution du
signal du core, ni signal supplémentaire linéairement décodable pour ces deux
tâches précises. Cohérent avec le reste du chapitre : la valeur mesurée de
l'extension dans ce projet tient à l'interprétabilité individuelle de ses
features (45,3% au test odd-one-out, chapitre 3 §2-5) et à la couverture de
concepts absents du core, pas à un gain de séparabilité linéaire en aval que
le core seul n'atteignait pas déjà. Détail complet : `RESULTS_TESTS.md` §55.
