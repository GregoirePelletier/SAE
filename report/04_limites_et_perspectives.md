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
3. ~~**Capacité architecturale de l'extension** (`D_EXTRA=1024`, `K_EXTRA=32`)~~
   **FAIT** : capacité doublée ensemble (§17.5, 40,0% -- pire, non concluant seul),
   `K_EXTRA=5` seul (§25, +9,4 points non significatif), `D_EXTRA=2048` seul à
   `K_EXTRA` fixe (§27, 46,0% -- aucun écart, z=-0,12). Conclusion : aucune
   configuration de capacité testée à ce jour ne change significativement le taux
   d'interprétabilité une fois le corpus corrigé.
4. **Fiabilité du juge/extracteur selon la taille du modèle** : observée comme
   dégradée sur `gemma-3-270m-it` par rapport à un modèle plus grand lors de la
   validation locale initiale (`Context.md`). *[En cours au moment de la rédaction
   de cette version]* : ablation à l'échelle complète avec gemma-3-4b-it et
   gemma-3-1b-it (+ leurs GemmaScope dédiés) à la place de gemma-3-12b-it, sur le
   corpus emails complet (pas juste une validation locale) -- résultats à venir.

**Mise à jour (testé, session interp_embed)** : une piste supplémentaire, plus
fondamentale, a été testée (`scripts/contrastive_labeling_test.py`,
`RESULTS_TESTS.md` §15.4) — le protocole de la référence (interp_embed, Appendix C)
ne gate JAMAIS la labellisation derrière un test odd-one-out : il génère toujours un
label par contraste direct (10 positifs + 10 négatifs). Sur les 82 features
originellement rejetées par notre gate, la génération contrastive directe produit un
label spécifique et qualitativement plausible pour la totalité d'entre elles après
correction de deux bugs trouvés en écrivant le test (marqueurs `<<>>` erronés sur les
négatifs ; un exemple de valeur JSON dans le prompt que le modèle recopiait
littéralement pour ~59% des features au premier essai). Exemples de labels récupérés :
`Mise en service énergie`, `Numéro de contrat`, `Demande de résiliation`,
`Informations bancaires`, `Sentiment d'urgence`. **Limite** : le champ `confident`
auto-rapporté par le LLM reste à `true` pour 150/150 features dans les deux runs —
pas un signal de qualité fiable en l'état, il faudrait le remplacer par une
validation croisée indépendante (ρ_interp déjà implémenté, ou vote majoritaire
odd-one-out en aval plutôt qu'en amont de la labellisation). **Non intégré au
pipeline de production dans cette session** — changerait le chiffre central du
rapport (45,3%), nécessite de refaire tourner une validation à l'échelle comparable
avant adoption.

### Rigueur statistique des comparaisons d'ablation

Audit rétroactif (`RESULTS_TESTS.md` §30) : deux comparaisons (biais
multilingue §22, robustesse du juge §13.1) testent en réalité les MÊMES 150
features sous deux conditions -- un plan apparié -- mais avaient été analysées
avec un test à deux proportions indépendantes plutôt que McNemar. Recalcul
avec le test approprié : mêmes conclusions (p=0,894 et p=0,560, non
significatifs), mais méthodologiquement plus correct. Aucune correction pour
comparaisons multiples n'a non plus été appliquée aux ~15 tests d'ablation de
ce chapitre (contrairement au diffing par feature, qui utilise déjà
Benjamini-Hochberg) -- sans conséquence sur les conclusions actuelles (le seul
résultat significatif, l'échelle du modèle, l'est à p<10⁻⁹), mais une lacune à
corriger pour toute extension future où des résultats plus proches du seuil
pourraient apparaître. L'effet dose-réponse de l'échelle du modèle a par
ailleurs été reconfirmé par un test de tendance dédié (Cochran-Armitage,
p≈1,6×10⁻¹⁰), plus adapté qu'une série de tests par paires à un plan à niveaux
ordonnés.

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
dépendant d'une installation non faite par défaut).

**Mise à jour (identifié, session pdf/)** : "SAE Boost" (Koriagin et al., COLM 2025)
n'était pas "non fait" mais déjà implémenté sans le savoir --
`FrozenCoreResidualSAE`/`ExtendedSAE` EST une implémentation de SAE Boost (même
architecture : SAE résiduel sur l'erreur de reconstruction d'un core gelé, sommé à
l'inférence). Deux écarts identifiés par la relecture du papier, **tous deux
testés depuis** : (1) leur étude de sensibilité montre qu'un `K_EXTRA` plus
faible (k=5 optimal chez eux, contre 32 dans ce projet) améliore
l'interprétabilité au prix d'un peu d'EV domaine — **testé** (`RESULTS_TESTS.md`
§25) : direction cohérente (54,7% vs 45,3%, +9,4 points) mais non significatif
(z=-1,62) ; (2) leur étude montre qu'un budget de 100-200M tokens est nécessaire
pour que le SAE résiduel converge sans dégrader la performance générale (jusqu'à
-31% d'EV en dessous de 100M) — **testé partiellement** à 25M tokens (12x
l'ablation initiale, toujours 50-100x en dessous du seuil du papier,
`RESULTS_TESTS.md` §23.4) : même conclusion qualitative (pas d'effet
significatif, +8,7 points non significatif), mais toujours pas de test au seuil
exact 100-200M (coût GPU/RAM substantiel, cf. §23.3bis pour la contrainte
mémoire rencontrée). Aucune comparaison chiffrée avec leurs baselines
alternatives (Extended SAE random/most-active init, SAE Stitching, full
fine-tuning) n'a été menée sur ce projet.

**Mise à jour (testé, session pdf/)** : une question plus fondamentale a été
posée par *Sanity Checks for Sparse Autoencoders* (Korznikov et al., 2026) --
un SAE dont le décodeur est figé à une initialisation aléatoire (jamais entraîné)
égale, dans leur étude, un SAE réellement entraîné sur interprétabilité automatique,
sparse probing et édition causale. Reproduit sur ce projet
(`FrozenDecoderExtendedSAE`, `SANITY_CHECK_FROZEN_DECODER=1`) : cf.
`RESULTS_TESTS.md` §19 — l'interprétabilité odd-one-out résiste bien (45,3%
entraîné vs 29,3% figé aléatoire, écart significatif) mais la classification en
aval y résiste beaucoup moins (93,5% vs 91,2%), répliquant partiellement le
constat du papier.

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

### Retrieval par propriétés et clustering ciblé (bug corrigé)

**Corrigé** (`RESULTS_TESTS.md` §15.1-15.2) : `property_based_retrieval` et
`targeted_clustering_by_axis` sélectionnaient les latents pertinents pour une
requête par matching de sous-chaîne littérale (`word in label`) plutôt que par
similarité d'embedding (méthode de la référence, interp_embed §4.4/Appendix F.1) —
vérifié empiriquement que ça ratait des labels sémantiquement liés mais formulés
différemment, et retournait des faux positifs (mot partagé sans rapport de sens).
Bug additionnel dans `property_based_retrieval` : la pondération "température"
utilisait l'ordre d'itération du dict de labels comme proxy de pertinence, pas un
rang réel. Fix : nouvelle fonction `select_latents_by_similarity`
(`src/sae/saev5.py`), embeddings **bge-m3** (pas F2LLM, testé et rejeté : bons
résultats sur une requête, résultats sans rapport sur une autre — pooling
dernier-token mal adapté à des labels courts en contexte cross-lingue). Validé
bout-en-bout sur les activations déjà en cache, non revalidé par un run complet
(ne change pas la reconstruction des activations elles-mêmes, seulement la
sélection de latents en aval).

### Corrélations "intéressantes" (gap comblé, résultat peu concluant)

**Corrigé** (`RESULTS_TESTS.md` §15.3) : `cooccurrence_graph` (NPMI + communautés
Louvain) n'était jamais appelée dans le pipeline principal — seule la matrice NPMI
brute était calculée et cachée, sans analyse en sortie. Nouvelle fonction
`find_interesting_pairs` (`src/analysis/cooccurrence.py`), filtre NPMI élevé +
similarité sémantique des labels faible (méthode interp_embed §4.2/Appendix E.1).
**Calculée rétroactivement** (`scripts/compute_interesting_correlations_retro.py`,
`RESULTS_TESTS.md` §16.3) sur `results_v10_emails_main` sans réextraction Gemma-3 :
seulement 3 paires retenues sur 26 579 arêtes du graphe (3 395 nœuds), et 2 des 3
impliquent une feature non labellisée — résultat honnête mais peu exploitable en
l'état (impossible de juger la pertinence d'une corrélation quand un des deux côtés
n'a pas de label). Piste retenue : élargir la plage de fréquence ou prioriser les
paires où les deux features sont labellisées.

### Qualité de l'explication document-level (nouveau, testé)

Question distincte de tout ce qui précède (qui évalue une feature isolée ou une
capacité globale) : pour UN document donné, l'explication produite (features
actives + labels) est-elle bonne ?
- **Fidélité** (`scripts/explanation_fidelity_test.py`, ablation) : chute de 58 à 100
  points de probabilité en ablatant les 10 features "explicatives", chute quasi nulle
  (<0,4 point) en ablatant des features aléatoires ou peu contributives (ratios de
  250× à 576 000× selon l'intention). Résultat sans ambiguïté : l'explication porte
  réellement la décision.
- **Plausibilité** (`scripts/explanation_plausibility_test.py`, choix forcé, juge
  Gemma-3-12B-it) : 71,7% (43/60) de choix corrects contre 50% au hasard (p < 0,001) —
  significativement au-dessus du hasard, mais loin d'être parfait (cohérent avec le
  taux d'interprétabilité résiduel ~45-55%).

Détail complet : `RESULTS_TESTS.md` §16.1-16.2, `report/03_experiences_et_resultats.md`
§8, dashboard (onglet "Explication (fidélité/plausibilité)").

### Comparaison du backbone d'embedding Pipeline 2 : F2LLM-80M vs -330M (nouveau, testé)

Résultat **mixte** (`RESULTS_TESTS.md` §16.5) : -330M reconstruit légèrement mieux
(NMSE −7,5%) et sépare un peu mieux le corpus de diffing générique (+2 points), mais
sépare légèrement MOINS bien les axes email (−2,2 points, la métrique la plus
proche des objectifs métier). Aucun écart n'est de l'ordre d'un problème majeur ; pas
de justification claire pour préférer l'un à l'autre sur ce projet à ce stade.

### Facteurs non contrôlés dans le corpus augmenté

Les variantes augmentées sont générées par le même modèle (Gemma-3-12B-it) qui sert
aussi de juge d'interprétation et d'extracteur d'activations. Un style de génération
propre au modèle (tournures récurrentes, longueur, structure) pourrait constituer un
facteur de confusion partagé entre "ce qui rend une variante reconnaissable comme
augmentée" et "ce que le SAE apprend à détecter" — non quantifié dans cette
investigation.

En contrepoint, un aspect qui *est* contrôlé : `src/data/augmentation.py::validate`
rejette une variante générée si elle est trop courte (<30 caractères), si son ratio
de longueur par rapport au mail original sort de l'intervalle [0,4 ; 2,5], si elle
est strictement identique au parent, ou (sauf pour l'axe orthographe) si elle perd
une entité numérique du mail d'origine (montant, numéro de compte/contrat, date).
Sur l'ensemble du corpus augmenté (45 240 générations), **11,7% (5291) sont
rejetées** par ce garde-fou — texte non conservé (`text: null`), motif de rejet
conservé pour audit (`facts_lost=[...]`, `length_ratio=...`, etc.). Sans impact sur
les résultats de ce rapport : `load_augmented` filtre ces lignes rejetées avant
qu'elles n'atteignent le pipeline SAE, elles n'ont donc jamais été vues par
l'extension ni par la sonde de classification.

### Pistes issues d'une relecture élargie de la littérature (nouveau)

Une relecture de l'ensemble des PDF de référence disponibles (`pdf/`, au-delà des
seuls SAE Boost et sanity checks déjà traités ci-dessus) fait ressortir trois pistes
non testées, chacune directement actionnable mais représentant un effort
d'implémentation plus substantiel que les corrections déjà apportées :

- **Feature splitting/absorption comme cause possible du résidu non-interprété**
  (*Matryoshka SAEs*, Bussmann et al. 2025) : leur travail montre qu'agrandir
  simplement un dictionnaire SAE (notre ablation capacité, `D_EXTRA` 1024→2048,
  chapitre 3 §10.4) peut dégrader la qualité des features de haut niveau par
  fragmentation/absorption plutôt que de mieux couvrir le domaine — cohérent avec le
  fait que notre ablation capacité n'a montré aucun gain d'interprétabilité. Leur
  solution (dictionnaires SAE emboîtés, entraînés simultanément à plusieurs tailles)
  n'a pas été implémentée : changement de la boucle d'entraînement plus substantiel
  que le sanity check Frozen Decoder déjà réalisé. **Ne pas confondre** avec
  `MATRYOSHKA_DIM` (`src/config.py`), qui ne concerne que la troncature des
  embeddings F2LLM, un mécanisme complètement différent (cf. `docs/references.md`).
- **Entraînement supervisé conjoint SAE+classifieur pour la classification**
  (*ClassifSAE*, Le Bail et al. 2025) : ce projet extrait des concepts de façon
  totalement non supervisée puis les relie à la classification par une sonde
  post-hoc (`downstream_classification`). ClassifSAE propose d'entraîner le SAE
  conjointement avec un classifieur (avec une pénalité de parcimonie sur le taux
  d'activation), spécifiquement pour concentrer les concepts pertinents à la tâche —
  directement aligné avec les objectifs "détection d'urgence"/"détection d'intention"
  du cadrage initial. Non implémenté : nécessiterait une nouvelle boucle
  d'entraînement (SAE + tête de classification jointe), distincte de l'architecture
  actuelle des deux pipelines.
- **Steering comme méthode d'explication "output-based"** (taxonomie de *A Survey
  on Sparse Autoencoders*, Shu et al. 2025) : **mesuré** (`RESULTS_TESTS.md` §24) —
  `steer_activations`/`steer_and_decode` (`src/sae/sae_shared.py`) existaient dans
  le dépôt depuis le début mais n'avaient jamais été réellement exercés au-delà
  d'une vérification géométrique superficielle (`run_steering_demo`). Testé en
  faisant réellement décoder puis ré-encoder un code stimulé (suppression des
  top-10 features explicatives d'une intention) : résultat très hétérogène selon
  l'intention — le round-trip neutralise quasi entièrement l'intervention pour
  2 intentions sur 4 (ratio 0,00-0,02× vs. l'ablation en place), la préserve pour
  une troisième (0,90×), et l'amplifie pour la dernière (1,74×). `steer_and_decode`
  n'est donc pas un mécanisme d'intervention causale fiable et prévisible à partir
  du simple test d'ablation en place du chapitre 3 — son effet dépend fortement de
  la structure de corrélation entre features propre à chaque intention.
- **Biais multilingue** (*survey* sur l'explicabilité des LLM multilingues, Resck et
  al. 2025) : **mesuré** (`RESULTS_TESTS.md` §22) — pas de différence significative
  d'interprétabilité entre français et anglais traduit (46,9% vs 45,5%), mais 38,6%
  des features changent individuellement de statut selon la langue, un taux de
  bruit supérieur à celui déjà mesuré pour le réordonnancement des exemples (§13.1,
  31,3%). Pas de biais systématique détecté envers l'anglais sur ce test précis.
- **Variance de seed d'entraînement du SAE** (*Unstable Features, Reproducible
  Subspaces*, arXiv:2606.12138) : **mesurée** (`RESULTS_TESTS.md` §21) — taux
  d'interprétabilité agrégé stable entre deux seeds (45,3% vs 47,3%), mais
  seulement 28,2% de recouvrement exact des labels individuels obtenus. Les
  features prises individuellement ne sont pas reproductibles à l'identique d'un
  seed à l'autre, contrairement au taux agrégé.

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
   BM25 live sur le vocabulaire latent complet (`src/sae/retrieval/latent_terms.py`,
   évalué quantitativement en `RESULTS_TESTS.md` §26 — Precision@10 parfaite sur
   2 intentions/4 mais échec complet sur une troisième, limite structurelle du
   BM25 sur vocabulaire très parcimonieux) ; pas de déploiement serveur
   persistant, lancement manuel.
5. ~~Exploiter le résultat de séparabilité linéaire des axes de perturbation... pour
   un cas d'usage concret de détection d'urgence/d'intention sur mails originaux~~
   **FAIT** : `scripts/intent_urgency_probe.py`, `RESULTS_TESTS.md` §13.2 — sonde sur
   les labels faibles réels (regex, indépendants du corpus augmenté) : +27,0 points
   sur l'urgence, +42,6 points sur la réclamation par rapport à la baseline classe
   majoritaire. Reste à faire : évaluer sur un jeu de labels d'urgence/intention
   annotés manuellement plutôt que des labels faibles par regex (limite ci-dessous),
   et sur le Pipeline 2 (F2LLM) en plus du Pipeline 1 déjà testé.
6. ~~Corriger le retrieval/clustering ciblé (matching par sous-chaîne) et brancher
   les corrélations "intéressantes"~~ **FAIT** (cf. sections ci-dessus,
   `RESULTS_TESTS.md` §15.1-15.3) — validés sur les activations déjà en cache, pas
   encore par un run complet à l'échelle (aucun changement des activations
   elles-mêmes, seulement de la sélection de latents en aval).
7. Adopter le protocole de labellisation contrastive directe (§15.4) comme
   alternative/complément au gate odd-one-out — well-evidenced (labels qualitativement
   plausibles récupérés sur 100% d'un échantillon de features rejetées) mais nécessite
   (a) une validation croisée de la qualité des labels (le champ `confident`
   auto-rapporté n'est pas fiable), (b) un run de validation à l'échelle comparable
   aux 3 runs de `RESULTS_TESTS.md` §12 avant de remplacer le chiffre 45,3% publié.
8. ~~Calculer `find_interesting_pairs` (corrélations)~~ **FAIT** (rétroactivement,
   sans réextraction, `RESULTS_TESTS.md` §16.3) — résultat peu concluant (3 paires
   seulement, 2/3 avec une feature non labellisée). Reste à faire : comparer à des
   biais/artefacts réels connus du corpus (ex. le biais "Objet :" avant correction,
   §14.1) pour valider empiriquement la méthode sur ce projet, à la manière de la
   validation par injection synthétique du papier de référence (§4.2, Appendix E.2) ;
   élargir la plage de fréquence de `cooccurrence_graph` pour augmenter le rappel.
9. ~~Mettre en place un test de qualité de l'explication document-level (fidélité +
   plausibilité)~~ **FAIT** (`RESULTS_TESTS.md` §16.1-16.2) — résultats très positifs
   sur la fidélité, positifs mais imparfaits sur la plausibilité. Reste à faire :
   étendre le test de plausibilité au Pipeline 2, et à un échantillon plus large que
   60 documents pour resserrer l'intervalle de confiance.
10. ~~Comparer le backbone d'embedding Pipeline 2 (F2LLM-80M vs -330M vs -160M)~~
    **FAIT** (`RESULTS_TESTS.md` §16.5-16.6) — résultat mixte, aucune taille ne
    domine. ~~Tester bge-m3 comme backbone Pipeline 2~~ **FAIT** (§16.7) —
    nécessitait un vrai correctif de code (pooling dernier-token câblé en dur,
    incorrect pour un encodeur bidirectionnel comme bge-m3, cf. nouveau
    `EMB_POOLING`) : résultat net et positif, bge-m3 domine sur NMSE (−18,8% vs
    le meilleur F2LLM), taux de features mortes, silhouette, et acc_SAE diffing --
    candidat par défaut recommandé pour une suite de stage.
11. ~~Concevoir un protocole de test complet du dépôt sous conditions fixées~~ **FAIT** :
    `docs/evaluation_protocol.md` + `scripts/consolidate_evaluation_report.py` +
    onglet dashboard "Rapport consolidé". Aucun problème majeur rencontré sur cette
    passe (cf. les critères de décision du protocole) — la comparaison multi-modèles/
    conditions que j'envisage peut être considérée en suite de stage.
12. ~~Identifier et documenter la correspondance avec SAE Boost~~ **FAIT**
    (`RESULTS_TESTS.md` §18). ~~Tester un `K_EXTRA` plus faible (proche de leur
    k=5 optimal)~~ **FAIT** (§25) : direction cohérente (+9,4 points) mais non
    significatif. ~~Un run à volume plus élevé pour vérifier le seuil de
    convergence~~ **FAIT partiellement** (25M tokens, §23.4, 12x l'ablation
    initiale mais toujours 50-100x en dessous du seuil du papier) : même
    conclusion qualitative (+8,7 points, non significatif). Reste à faire : un
    run au seuil exact 100-200M (coût GPU/RAM substantiel, cf. §23.3bis) ;
    comparer chiffré à leurs baselines alternatives (Extended SAE, SAE Stitching,
    full fine-tuning) sur le corpus emails ; répliquer K_EXTRA=5 et le volume 25M
    sur plusieurs seeds pour trancher si l'écart directionnel commun aux deux
    (+8,7/+9,4 points, chacun non significatif seul) reflète un effet réel.
13. ~~Reproduire le sanity check "Frozen Decoder" (Korznikov et al. 2026)~~ **FAIT**
    (`RESULTS_TESTS.md` §19, `FrozenDecoderExtendedSAE`) — résultat **nuancé** :
    l'interprétabilité odd-one-out résiste bien (45,3% entraîné vs 29,3% figé
    aléatoire, écart significatif) mais la classification en aval y résiste beaucoup
    moins (93,5% vs 91,2% — un décodeur aléatoire capture déjà la quasi-totalité du
    signal), répliquant partiellement le constat du papier sur le sparse probing.
    Reste à faire : étendre le sanity check au Pipeline 2 (`PhraseLevelSAE`,
    entraîné from-scratch, jamais testé contre un décodeur figé) ; envisager les
    métriques plus exigeantes du papier (AutoInterp par description+détection sur
    échantillon non vu, sparse probing SAEBench) en remplacement de la sonde de
    classification actuelle, dont ce sanity check a montré la faible sensibilité.
14. Tester des dictionnaires SAE emboîtés (*Matryoshka SAEs*, Bussmann et al. 2025)
    pour l'extension P1, comme piste alternative à l'ablation capacité simple
    (`D_EXTRA`/`K_EXTRA`, déjà testée sans effet) pour expliquer/réduire le résidu
    non-interprété — nécessite une nouvelle boucle d'entraînement multi-échelle.
15. Entraîner un SAE supervisé conjointement avec un classifieur (*ClassifSAE*,
    Le Bail et al. 2025) pour la détection d'urgence/intention, en alternative à la
    sonde post-hoc actuelle (`downstream_classification`) — permettrait de comparer
    directement la précision et l'interprétabilité des concepts obtenus.
16. ~~Évaluer le steering (`steer_activations`/`steer_and_decode`, déjà implémenté
    mais jamais utilisé comme méthode d'explication à part entière) comme complément
    "output-based" aux protocoles "input-based" déjà validés (chapitre 3)~~ **FAIT**
    (`RESULTS_TESTS.md` §24) — résultat hétérogène et non trivial : le round-trip
    decode/ré-encodage neutralise quasi entièrement l'intervention pour 2 intentions
    sur 4 testées (ratio 0,00-0,02× vs. ablation en place), la préserve pour une
    troisième (0,90×), et l'amplifie pour la dernière (1,74×). `steer_and_decode`
    n'est donc pas un mécanisme d'intervention causale fiable et prévisible à partir
    du simple test d'ablation en place. Reste à faire : étendre à un échantillon plus
    large de features/intentions pour caractériser ce qui distingue les cas où le
    round-trip "tient" de ceux où il ne tient pas (structure de corrélation entre
    features ? spécificité de l'intention ?).
17. ~~Quantifier le biais multilingue potentiel du juge d'auto-interprétation~~
    **FAIT** (`RESULTS_TESTS.md` §22) — résultat **nul sur l'hypothèse testée** :
    aucune différence significative entre le taux d'interprétabilité en français et
    en anglais traduit (46,9% vs 45,5%, z=0,24). Renforce en revanche le constat du
    §13.1 : 38,6% des features changent de statut selon la langue de présentation
    (contre 31,3% pour un simple réordonnancement), confirmant que le protocole
    odd-one-out à décision unique reste globalement bruyant face à toute
    perturbation de surface, pas spécifiquement biaisé envers l'anglais sur ce
    corpus. Reste à faire : tester l'hypothèse alternative (entraîner le SAE sur un
    corpus anglais natif équivalent, pas seulement traduire la vue du juge).
18. ~~Tester la variance de seed d'entraînement du SAE~~ **FAIT**
    (`RESULTS_TESTS.md` §21, *Unstable Features, Reproducible Subspaces*,
    arXiv:2606.12138) — taux d'interprétabilité agrégé stable entre seeds (45,3% vs
    47,3%, non significatif) mais seulement 28,2% de recouvrement exact des labels
    individuels obtenus. Confirme que les features individuelles ne sont pas
    reproductibles à l'identique (seule la performance agrégée et la thématique
    générale le sont) — nuance importante pour la lecture des exemples de features
    cités dans ce rapport (chapitre 3) : à comprendre comme représentatifs d'une
    catégorie récurrente de concepts, pas comme des atomes stables du dictionnaire.
