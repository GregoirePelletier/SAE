# Limites et perspectives

## Limites actuelles

### Taux d'interprétabilité résiduel (~55% de features non interprétées)

Ce résidu n'est pas dû au volume de tokens du corpus (`03_experiences_et_resultats.md`).
Il est en revanche en bonne partie attribuable au bruit du protocole de jugement :
en répétant la question odd-one-out 5 fois par feature avec un ordre de mélange
différent à chaque fois (`RESULTS_TESTS.md` §13.1), seulement 30,7% des features
obtiennent une décision unanime sur les 5 répétitions ; le taux agrégé
d'interprétabilité bouge peu (45,3%→48,7%) mais 31,3% des features changent
individuellement de statut selon l'ordre de présentation. Une partie substantielle
du résidu non interprété est donc due au bruit du protocole (décision greedy
unique, sensible à l'ordre), pas nécessairement à un défaut réel des features —
un vote majoritaire sur plusieurs répétitions serait préférable comme protocole
par défaut à une seule décision greedy.

Le taux de 45,3% cité dans ce rapport n'est pas non plus robuste au choix du
modèle juge : remplacer gemma-3-12b-it par gemma-3-4b-it comme juge (mêmes
features, mêmes exemples) fait chuter le taux mesuré à 24,7%. Une partie de ce
que ce rapport attribue à la qualité des features apprises dépend donc aussi de
la capacité du juge à raisonner sur 9 exemples, pas uniquement des features
elles-mêmes — cohérent avec l'effet dose-réponse de l'échelle du modèle
documenté au §18 de `03_experiences_et_resultats.md`.

L'effet "domaine, pas volume" qui sous-tend l'ensemble du rapport
(`03_experiences_et_resultats.md` §2-5) est statistiquement confirmé à n
apparié : 30,0% (45/150) sur corpus générique contre 45,3% (68/150) sur corpus
emails, z=2,74, p≈0,006 (`RESULTS_TESTS.md` §46).

Piste encore non résolue : la **qualité du contrôle négatif**
(`build_feature_examples_with_control`) reste un document sous un quantile bas
d'activation pour la feature testée, pas nécessairement un contre-exemple
"propre" conceptuellement — une feature réellement monosémantique pourrait
échouer au test si le contrôle négatif choisi partage accidentellement une
propriété de surface avec les exemples positifs. Les ablations de capacité de
l'extension (`D_EXTRA`/`K_EXTRA`, doublées ensemble, `K_EXTRA=5` seul,
`D_EXTRA=2048` seul) ne changent significativement le taux d'interprétabilité
dans aucune configuration testée une fois le corpus corrigé (`RESULTS_TESTS.md`
§17.5/§25/§27).

Une piste plus fondamentale a également été testée : le protocole de la
référence *Interpretable Embeddings with Sparse Autoencoders* (Appendix C) ne
gate jamais la labellisation derrière un test odd-one-out, il génère toujours
un label par contraste direct (10 positifs + 10 négatifs). Sur les 82 features
originellement rejetées par notre gate, la génération contrastive directe
produit un label spécifique et qualitativement plausible pour la totalité
d'entre elles (`scripts/contrastive_labeling_test.py`, `RESULTS_TESTS.md`
§15.4) — exemples : `Mise en service énergie`, `Numéro de contrat`, `Demande
de résiliation`, `Informations bancaires`, `Sentiment d'urgence`. Limite :
le champ `confident` auto-rapporté par le LLM reste à `true` pour 150/150
features dans les deux runs testés — pas un signal de qualité fiable en
l'état. Une validation systématique (comptage sur les 82 labels, pas un
échantillon) montre que 45% partagent leur label avec un autre, et un cas de
feature quasi-morte (freq=0%) reçoit malgré tout un label confiant — le
"100% de récupération" apparent est un artefact de complaisance du juge, pas
un signal de qualité. Le protocole odd-one-out reste la référence retenue
dans ce rapport ; la labellisation contrastive directe n'est pas intégrée au
pipeline de production (changerait le chiffre central du rapport, 45,3%,
sans validation à l'échelle comparable).

### Rigueur statistique des comparaisons d'ablation

Deux comparaisons (biais multilingue, robustesse du juge) testent en réalité
les mêmes 150 features sous deux conditions — un plan apparié — mais avaient
été analysées avec un test à deux proportions indépendantes plutôt que
McNemar. Le recalcul avec le test approprié donne les mêmes conclusions
(p=0,894 et p=0,560, non significatifs), mais méthodologiquement plus
correct (`RESULTS_TESTS.md` §30). Aucune correction pour comparaisons
multiples n'est appliquée aux ~15 tests d'ablation de ce chapitre
(contrairement au diffing par feature, qui utilise déjà Benjamini-Hochberg) —
sans conséquence sur les conclusions actuelles (le seul résultat
significatif, l'échelle du modèle, l'est à p<10⁻⁹), mais une lacune pour
toute extension future où des résultats plus proches du seuil pourraient
apparaître. L'effet dose-réponse de l'échelle du modèle est par ailleurs
confirmé par un test de tendance dédié (Cochran-Armitage, p≈1,6×10⁻¹⁰), plus
adapté qu'une série de tests par paires à un plan à niveaux ordonnés.

### Comparaisons avec l'état de l'art

Une comparaison chiffrée avec SAELens (`scripts/saelens_numeric_comparison.py`,
`docs/references.md`), sur le même SAE natif sae-lens et les mêmes
activations réelles, entre notre formule de variance expliquée et les deux
formules maintenues par `sae_lens.evals`, révèle un désaccord numérique
important entre les trois (0,41 / 0,83 / 1,00) causé par les activations
massives de Gemma-3 — la formule qui somme sur les dimensions avant de
normaliser est mécaniquement dominée par une seule dimension outlier et
rapporte une variance quasi-totalement expliquée, sans rapport avec la
qualité de reconstruction réelle. Ne jamais publier un score de variance
expliquée sur ce modèle sans préciser la formule exacte utilisée : la formule
retenue pour ce projet est la nôtre, qui centre la variance par dimension
plutôt que sur la norme globale (cf. `docs/references.md` pour la
justification). La comparaison avec `interp_embed` (submodule installé et
peuplé) reste faite par lecture du papier, jamais par exécution de son code
de référence.

`FrozenCoreResidualSAE`/`ExtendedSAE` est une implémentation de SAE Boost
(Koriagin et al., COLM 2025) : même architecture, un SAE résiduel entraîné
sur l'erreur de reconstruction d'un core gelé, sommé à l'inférence. Deux
écarts avec le papier ont été testés : (1) leur étude de sensibilité montre
qu'un `K_EXTRA` plus faible (k=5 optimal chez eux, contre 32 dans ce projet)
améliore l'interprétabilité au prix d'un peu d'EV domaine — direction
cohérente sur ce corpus (54,7% vs 45,3%, +9,4 points) mais non significatif
(z=-1,62, `RESULTS_TESTS.md` §25) ; (2) leur étude montre qu'un budget de
100-200M tokens est nécessaire pour que le SAE résiduel converge sans
dégrader la performance générale (jusqu'à -31% d'EV en dessous de 100M) —
testé partiellement à 25M tokens (12x l'ablation initiale, toujours 50-100x
en dessous du seuil du papier, `RESULTS_TESTS.md` §23.4) : même conclusion
qualitative, et l'écart numérique initial (+8,7 points) ne réplique pas sur
deux seeds supplémentaires (48,2% sur 3 seeds combinés, z=0,61,
`RESULTS_TESTS.md` §56). Le run au seuil exact 100-200M reste à exécuter —
le réservoir de résidus, initialement une
allocation RAM proportionnelle au volume de tokens (limitant la faisabilité
pratique d'un tel run), est désormais memory-mapped sur disque
(`RESULTS_TESTS.md` §54), rendant ce run schedulable. Aucune comparaison
chiffrée avec les baselines alternatives du papier (Extended SAE
random/most-active init, SAE Stitching, full fine-tuning) n'a été menée sur
ce projet.

Une question plus fondamentale, posée par *Sanity Checks for Sparse
Autoencoders* (Korznikov et al., 2026) : un SAE dont le décodeur est figé à
une initialisation aléatoire (jamais entraîné) égale, dans leur étude, un
SAE réellement entraîné sur interprétabilité automatique, sparse probing et
édition causale. Reproduit sur ce projet (`FrozenDecoderExtendedSAE`,
`SANITY_CHECK_FROZEN_DECODER=1`, `RESULTS_TESTS.md` §19) : résultat nuancé —
l'interprétabilité odd-one-out résiste bien (45,3% entraîné vs 29,3% figé
aléatoire, écart significatif) mais la classification en aval y résiste
beaucoup moins (93,5% vs 91,2%), répliquant partiellement le constat du
papier.

### Biais de génération résiduel dans le corpus augmenté

20,6% des mails augmentés contenaient une ligne "Objet :"/"Subject :" que les
mails originaux n'ont pas (`RESULTS_TESTS.md` §14.1). Fix appliqué au
chargement (`load_augmented`, 0,0% après fix) ; effet mesuré sur le diffing
complet : réduction de 65% et 49% du nombre de features "significatives" sur
les deux axes orthographiques (les plus confondables avec l'artefact), effet
modéré sur l'urgence (−7,5%/−3,1%), négligeable ailleurs. Contrairement à
l'observation initiale sur l'échantillon test à 60 mails (où l'artefact
dominait 8/13 classements), l'artefact ne domine pas la majorité des
features significatives à l'échelle du corpus complet (0,21% des features
significatives portaient un label "Subject:"/"Objet:", avant comme après) —
son effet réel est plus circonscrit qu'attendu.

### Retrieval par propriétés et clustering ciblé

`property_based_retrieval` et `targeted_clustering_by_axis` sélectionnent les
latents pertinents pour une requête par similarité d'embedding
(`select_latents_by_similarity`, `src/sae/saev5.py`, embeddings **bge-m3**),
pas par matching de sous-chaîne littérale — vérifié empiriquement que ce
dernier rate des labels sémantiquement liés mais formulés différemment, et
retourne des faux positifs (mot partagé sans rapport de sens). bge-m3 est
retenu après comparaison à F2LLM (bons résultats sur une requête, résultats
sans rapport sur une autre — pooling dernier-token mal adapté à des labels
courts en contexte cross-lingue), `RESULTS_TESTS.md` §15.1-15.2. Validé sur
les activations déjà en cache (ne change pas la reconstruction elle-même,
seulement la sélection de latents en aval).

### Corrélations "intéressantes" entre features

`find_interesting_pairs` (`src/analysis/cooccurrence.py`) filtre les paires à
NPMI élevé et similarité sémantique des labels faible (méthode interp_embed
§4.2/Appendix E.1), calculé sur `results_v10_emails_main` sans réextraction
Gemma-3 (`scripts/compute_interesting_correlations_retro.py`,
`RESULTS_TESTS.md` §15.3/§16.3) : seulement 3 paires retenues sur 26 579
arêtes du graphe (3 395 nœuds), et 2 des 3 impliquent une feature non
labellisée — résultat peu exploitable en l'état (impossible de juger la
pertinence d'une corrélation quand un des deux côtés n'a pas de label).
Élargir la plage de fréquence ou prioriser les paires où les deux features
sont labellisées reste à faire.

### Qualité de l'explication document-level

Question distincte de tout ce qui précède (qui évalue une feature isolée ou
une capacité globale) : pour UN document donné, l'explication produite
(features actives + labels) est-elle bonne ?
- **Fidélité** (`scripts/explanation_fidelity_test.py`, ablation) : chute de
  53 à 100 points de probabilité en ablatant les 10 features "explicatives",
  chute quasi nulle (<0,4 point) en ablatant des features aléatoires ou peu
  contributives (ratios de 450× à 52 600× selon l'intention) — l'explication
  porte réellement la décision, résultat robuste au correctif des labels
  d'intention (B.26, cf. `RESULTS_TESTS.md` §68).
- **Plausibilité** (`scripts/explanation_plausibility_test.py`, choix forcé,
  juge Gemma-3-12B-it) : 71,7% (43/60) de choix corrects contre 50% au hasard
  (p < 0,001) — significativement au-dessus du hasard, mais loin d'être
  parfait, cohérent avec le taux d'interprétabilité résiduel.

Détail complet : `RESULTS_TESTS.md` §16.1-16.2, `03_experiences_et_resultats.md`
§8, dashboard (onglet "Explication (fidélité/plausibilité)").

### Backbone d'embedding Pipeline 2 : F2LLM-80M vs -330M

Résultat mixte (`RESULTS_TESTS.md` §16.5) : -330M reconstruit légèrement
mieux (NMSE −7,5%) et sépare un peu mieux le corpus de diffing générique (+2
points), mais sépare légèrement moins bien les axes email (−2,2 points, la
métrique la plus proche des objectifs métier). Aucun écart n'est de l'ordre
d'un problème majeur ; pas de justification claire pour préférer l'un à
l'autre sur ce projet.

### Facteurs non contrôlés dans le corpus augmenté

Les variantes augmentées sont générées par le même modèle (Gemma-3-12B-it)
qui sert aussi de juge d'interprétation et d'extracteur d'activations. Un
style de génération propre au modèle pourrait constituer un facteur de
confusion partagé entre "ce qui rend une variante reconnaissable comme
augmentée" et "ce que le SAE apprend à détecter". Trois vérifications
indépendantes (`RESULTS_TESTS.md` §44/§48/§50/§52) convergent vers une
réponse négative : aucune corrélation significative entre la part
d'exemples "augmentés" parmi les 9 exemples positifs d'une feature et son
statut interprétable (p=0,418) ; des features core totalement étrangères à
cette boucle (labellisées indépendamment par Neuronpedia) présentent le même
taux élevé d'exemples augmentés, confirmant qu'il s'agit d'une propriété du
corpus (92% augmenté) et non d'un biais spécifique au pipeline ; un
re-jugement complet des 150 features avec le même SAE et le même juge, mais
des exemples positifs restreints aux mails originaux uniquement (zéro texte
généré par Gemma vu par le juge), donne 44,7% (67/150) contre 45,3% (68/150)
en référence — écart non significatif (z=-0,12, p=0,908). Comme pour les
autres perturbations testées (ordre de présentation, langue), le statut
d'une feature individuelle reste bruité (55,3% d'accord, 44,7% de bascule)
mais le taux agrégé est stable. **La boucle auto-référentielle
juge/générateur n'explique pas le taux d'interprétabilité mesuré dans ce
rapport.**

En contrepoint, un aspect qui est contrôlé : `src/data/augmentation.py::validate`
rejette une variante générée si elle est trop courte (<30 caractères), si son
ratio de longueur par rapport au mail original sort de l'intervalle
[0,4 ; 2,5], si elle est strictement identique au parent, ou (sauf pour
l'axe orthographe) si elle perd une entité numérique du mail d'origine
(montant, numéro de compte/contrat, date). Sur l'ensemble du corpus augmenté
(45 240 générations), 11,7% (5291) sont rejetées par ce garde-fou — motif de
rejet conservé pour audit (`facts_lost=[...]`, `length_ratio=...`, etc.).
Sans impact sur les résultats de ce rapport : `load_augmented` filtre ces
lignes rejetées avant qu'elles n'atteignent le pipeline SAE.

### Fidélité du steering comme méthode d'explication

`steer_activations`/`steer_and_decode` (`src/sae/sae_shared.py`) — mesuré en
faisant réellement décoder puis ré-encoder un code stimulé (suppression des
top-10 features explicatives d'une intention, `RESULTS_TESTS.md` §24) :
résultat très hétérogène selon l'intention — le round-trip neutralise quasi
entièrement l'intervention pour 2 intentions sur 4 (ratio 0,00-0,02× vs.
l'ablation en place), la préserve pour une troisième (0,90×), et l'amplifie
pour la dernière (1,74×). `steer_and_decode` n'est donc pas un mécanisme
d'intervention causale fiable et prévisible à partir du simple test
d'ablation en place — son effet dépend fortement de la structure de
corrélation entre features propre à chaque intention.

### Biais multilingue et variance de seed

Pas de différence significative d'interprétabilité entre français et anglais
traduit (46,9% vs 45,5%, z=0,24, `RESULTS_TESTS.md` §22), mais 38,6% des
features changent individuellement de statut selon la langue — un taux de
bruit supérieur à celui déjà mesuré pour le réordonnancement des exemples
(§13.1, 31,3%). Pas de biais systématique détecté envers l'anglais sur ce
test précis (l'hypothèse alternative — entraîner le SAE sur un corpus
anglais natif plutôt que de traduire la vue du juge — reste à tester).

Le taux d'interprétabilité agrégé est stable entre deux seeds d'entraînement
du SAE (45,3% vs 47,3%, `RESULTS_TESTS.md` §21), mais seulement 28,2% de
recouvrement exact des labels individuels obtenus entre les deux seeds — les
features individuelles ne sont pas reproductibles à l'identique d'un seed à
l'autre, contrairement au taux agrégé. Les exemples de features cités dans
ce rapport (chapitre 3) sont donc représentatifs d'une catégorie récurrente
de concepts, pas des atomes stables du dictionnaire.

### Pistes non implémentées issues de la littérature

- **Feature splitting/absorption comme cause possible du résidu
  non-interprété** (*Matryoshka SAEs*, Bussmann et al. 2025) : leur travail
  montre qu'agrandir simplement un dictionnaire SAE (notre ablation
  capacité, `D_EXTRA` 1024→2048, `03_experiences_et_resultats.md` §10.4)
  peut dégrader la qualité des features de haut niveau par
  fragmentation/absorption plutôt que de mieux couvrir le domaine — cohérent
  avec l'absence de gain d'interprétabilité observée sur cette ablation.
  Leur solution (dictionnaires SAE emboîtés, entraînés simultanément à
  plusieurs tailles) n'est pas implémentée. À ne pas confondre avec
  `MATRYOSHKA_DIM` (`src/config.py`), qui ne concerne que la troncature des
  embeddings F2LLM, un mécanisme complètement différent
  (`docs/references.md`).
- **Entraînement supervisé conjoint SAE+classifieur** (*ClassifSAE*, Le
  Bail et al. 2025) : ce projet extrait des concepts de façon totalement non
  supervisée puis les relie à la classification par une sonde post-hoc
  (`downstream_classification`). ClassifSAE propose d'entraîner le SAE
  conjointement avec un classifieur (pénalité de parcimonie sur le taux
  d'activation), spécifiquement pour concentrer les concepts pertinents à la
  tâche — directement aligné avec les objectifs de détection
  d'urgence/d'intention. Non implémenté : nécessiterait une nouvelle boucle
  d'entraînement jointe.

## Perspectives

- Clarifier avec le commanditaire si le système, à terme, constitue une
  décision automatisée au sens de l'article 22 du RGPD ou reste une aide à
  la décision (un humain reste dans la boucle) — un fait de conception du
  déploiement final, pas une question expérimentale, mais qui détermine le
  cadre légal applicable et n'est pas encore tranché.
- Passer le vote majoritaire en protocole par défaut de `odd_one_out_judge`
  (actuellement une fonction séparée, `scripts/judge_robustness_check.py`, à
  fusionner dans `src/sae/judge.py`).
- Isoler explicitement les dimensions à activation massive identifiées par la
  littérature (Sun et al., COLM 2024) dans le calcul de variance expliquée,
  plutôt qu'une statistique robuste générique qui traiterait ce signal
  structuré comme du bruit à lisser (cf. `docs/references.md`).
- Remplacer les intervalles de confiance approximatifs (Wald) encore utilisés
  ponctuellement au chapitre 3 par `proportion_with_ci` (Wilson,
  `src/analysis/stats.py`), déjà la référence pour le reste du module
  statistique.
- Poursuivre la factorisation de `src/sae/saev5.py` (séparer entraînement et
  extraction, réduire un fichier monolithique) — dette technique qui
  n'affecte pas la validité des résultats mais complique la maintenance.
- Étendre le sanity check "Frozen Decoder" au Pipeline 2 (`PhraseLevelSAE`,
  entraîné from-scratch, jamais testé contre un décodeur figé) ; envisager
  les métriques plus exigeantes du papier (AutoInterp par
  description+détection sur échantillon non vu, sparse probing SAEBench) en
  remplacement de la sonde de classification actuelle, dont ce sanity check
  a montré la faible sensibilité.
- Tester des dictionnaires SAE emboîtés (*Matryoshka SAEs*) pour l'extension
  P1, comme alternative à l'ablation capacité simple pour réduire le résidu
  non interprété.
- Entraîner un SAE supervisé conjointement avec un classifieur (*ClassifSAE*)
  pour la détection d'urgence/intention, en alternative à la sonde post-hoc
  actuelle.
- Étendre le test de plausibilité de l'explication document-level au
  Pipeline 2, et à un échantillon plus large que 60 documents pour resserrer
  l'intervalle de confiance.
- Caractériser ce qui distingue les cas où le round-trip du steering "tient"
  de ceux où il ne tient pas (structure de corrélation entre features ?
  spécificité de l'intention ?) sur un échantillon plus large de
  features/intentions.
- Exécuter le run à 200M tokens (seuil exact SAE Boost, désormais
  schedulable) et comparer chiffré aux baselines alternatives du papier
  (Extended SAE random/most-active init, SAE Stitching, full fine-tuning).
  Le volume 25M a été répliqué sur 3 seeds (`RESULTS_TESTS.md` §56) : ne
  tient pas, l'écart initial était du bruit d'échantillonnage. `K_EXTRA=5`
  reste à confirmer au-delà de sa réplication actuelle sur 3 seeds
  (53,3%, z=1,70, p≈0,089, sous le seuil conventionnel).
- Comparer `find_interesting_pairs` à des biais/artefacts réels connus du
  corpus (ex. le biais "Objet :" avant correction) pour valider
  empiriquement la méthode sur ce projet, à la manière de la validation par
  injection synthétique du papier de référence (§4.2, Appendix E.2) ;
  élargir la plage de fréquence de `cooccurrence_graph` pour augmenter le
  rappel.
- Évaluer un jeu de labels d'urgence/intention annotés manuellement plutôt
  que des labels faibles par regex pour la sonde de détection d'urgence
  (`scripts/intent_urgency_probe.py`), et étendre au Pipeline 2.
