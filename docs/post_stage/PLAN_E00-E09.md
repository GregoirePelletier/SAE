# SAE — Plan d’exécution des quinze derniers jours et dossier de reprise

**Destinataire : Claude Code, sous la responsabilité de Grégoire Pelletier et de l’encadrement EDF.**  
**Objet : transformer le prototype défendu en instrument d’exploration d’emails évalué, reproductible et transmissible.**  
**Base auditée :** dépôt `GregoirePelletier/SAE`, commit `68c8e6863013c98905740a56e6491d78d05435cd` ; rapport final et soutenance EDF corrigée fournis par Grégoire.  
**Périmètre :** recommandations et spécifications de nouveaux travaux. Aucun des résultats futurs décrits ici n’a été mesuré. Aucun job GPU n’a été exécuté pour rédiger ce plan. Les durées sont des enveloppes à recalibrer à J1 ; l’attente en file Slurm n’est pas incluse.

> **Décision de cadrage.** Ne pas consacrer ces quinze jours à une nouvelle recherche de la meilleure configuration. Construire un jeu d’évaluation indépendant, comparer les représentations sur les mêmes documents, rendre le retrieval et le diffing utilisables, et livrer une première carte de features avec une mesure de stabilité. Les corrélations et le clustering ciblé doivent disposer d’un protocole corrigé et d’une première évaluation cible, sans promettre une validation industrielle complète. Le run 100M est un bonus conditionnel, pas le chemin critique.

## Sommaire

1. Mandat, autonomie et limites de sécurité
2. Ce qui est établi ; ce qui manque réellement
3. Plan de livraison, priorités, budget et calendrier
4. Architecture commune des données et des évaluations
5. E00 — Audit de ressources et correction du chemin mémoire
6. E01 — Représentations comparables et gain propre de l’extension
7. E02 — Registre des features et validation humaine mutualisée
8. E03 — Retrieval par propriété sur les emails
9. E04 — Diffing dans le domaine cible
10. E05 — Groupes de features, stabilité et carte exploratoire
11. E06 — Corrélations entre propriétés, confirmées hors découverte
12. E07 — Clustering ciblé réellement comparé aux alternatives
13. E08 — Mini-pilote analyste et recette Streamlit
14. E09 — Option Gemma-3-1B / 100M tokens
15. Statistiques, métriques et interprétation transversales
16. Contrats logiciels, Slurm, checkpoints et exécution autonome
17. Streamlit : pages, provenance, exports et tests
18. Nettoyage du dépôt, résultats remplacés et passation
19. Travaux confiés au successeur
20. Critères de clôture et instruction de démarrage
21. Sources, références et portée des vérifications

---

## 1. Mandat, autonomie et limites de sécurité

### 1.1 Finalité métier à conserver

L’objectif est de **faire émerger des thèmes, sous-thèmes et associations non définis à l’avance dans des emails clients**, avec retour aux textes. Le traitement opérationnel des emails est déjà automatisé ; ce projet ne doit pas être vendu comme un nouvel outil de routage ou de réponse automatique. Le scénario « incident × région × période » est une question industrielle possible, pas une fonctionnalité déjà validée. Ne jamais fabriquer des dates, des régions, des incidents observés ou un commanditaire pour faire fonctionner la démonstration. [R, §1.1–1.4 ; S, diapos 2 et 10]

Le livrable doit permettre à un analyste de poser une question, consulter les documents proposés, comprendre la provenance des features et décider si la piste est pertinente. Un résultat nul, montrant qu’une méthode simple suffit, est un résultat utile à la décision.

### 1.2 Ce que l’agent peut faire

Après contrôle de l’environnement et dans le budget ci-dessous, l’agent peut implémenter les modules proposés, créer des tests, préparer des fichiers `.slurm`, soumettre **les jobs de cette campagne seulement**, analyser leurs sorties et produire les rapports. La demande de Grégoire autorise cette préparation et cette autonomie bornée ; elle ne donne ni priorité sur le cluster ni autorisation de traiter n’importe quelles données.

Avant la première soumission, écrire `campaign_policy.yaml` et `preflight_report.md`, signaler leur contenu à Grégoire, vérifier que les ressources allouées sont compatibles avec les règles du site. Une validation de site/budget peut être nécessaire ; une fois celle-ci obtenue, ne pas solliciter une confirmation pour chaque petit job conforme. Demander une nouvelle décision seulement pour dépasser une limite, changer de corpus autorisé, engager le run 100M, installer une dépendance lourde ou modifier le protocole confirmatoire.

### 1.3 Interdictions

- Ne pas modifier le rapport défendu, les slides défendues ou `Dossier_oral_SAE_complet.*`. Ce plan est une campagne **post-soutenance** distincte.
- Ne pas envoyer de texte confidentiel EDF, d’activation, de résultat individuel ou de secret à un assistant/API externe. Les scripts qui lisent des données autorisées s’exécutent sur l’infrastructure approuvée ; l’agent externe ne doit recevoir que du code, des schémas et des agrégats autorisés. Ne pas afficher des extraits confidentiels dans sa console ou ses messages.
- Ne pas pousser des textes, poids, activations, identifiants client, fichiers d’annotation ou chemins sensibles dans Git. Les manifestes publics doivent être expurgés ; les empreintes et identifiants ne constituent pas à eux seuls une anonymisation.
- Ne pas annuler les jobs d’autres personnes, ne pas utiliser `--exclusive`, ne pas réserver tous les GPU, ne pas lancer le même job en course sur plusieurs partitions. Ne pas désactiver la sécurité TLS ou modifier la configuration système pour contourner un problème.
- Ne pas faire de `drop_caches`, de purge globale de `/dev/shm`, de modification NUMA/Slurm globale ou de suppression massive du cache partagé.
- Ne pas déclarer un résultat « confirmé » parce qu’un LLM le juge plausible. Ne pas convertir les sorties invalides du juge en exemples négatifs.
- Ne pas relancer un job échoué en doublant automatiquement la RAM. Deux échecs identiques imposent diagnostic et arrêt de la branche concernée.

### 1.4 Paramètres d’autonomie proposés

Ces valeurs sont un **budget de planification**, à ajuster à la politique locale, et non une mesure de disponibilité du cluster.

```yaml
campaign: sae_post_soutenance_15j
base_commit_audited: 68c8e6863013c98905740a56e6491d78d05435cd
period: relative_J1_J15
max_simultaneous_gpu_jobs: 1
max_gpus_per_job: 1
max_simultaneous_cpu_jobs: 1
max_cpus_per_job: 8
base_gpu_hours_cap: 120
optional_100m_gpu_hours_cap: 0  # nouvelle décision requise après profilage
max_job_wall_hours: 12
max_host_mem_gib_without_new_approval: 128
max_retries_per_stage: 1
local_only_models: true
export_raw_text_to_external_assistant: false
allow_destructive_cleanup: false
allow_push: false  # commits locaux ciblés possibles ; push sur accord explicite
freeze_protocol_before_confirmation: true
```

Les jobs de juge 27B demandent un GPU dont la mémoire **réelle** suffit : une A100 peut avoir 40 ou 80 Go. Ne pas supposer que le nom de partition garantit 80 Go. Si le juge bf16 ne tient pas avec son cache d’attention, choisir un H100 adapté ; ne pas quantifier le juge ou changer de modèle en cours d’évaluation sans nouvelle version de protocole.

---

## 2. Ce qui est établi ; ce qui manque réellement

### 2.1 Point de départ documenté

| Acquis | Portée exacte | Conséquence pour la suite |
|---|---|---|
| 3 480 emails source synthétiques ; 3 474 retenus dans le run complet ; 39 949 variantes acceptées | Ni les sources ni les variantes ne sont des observations de clients réels | Ne pas promettre une validation métier réelle sans nouveau corpus autorisé |
| Extension à cœur gelé, `D_EXTRA=1024`, `K_EXTRA=5` | L’encodeur extra lit **x** ; la cible du décodeur est **e = x − x̂core** | Préserver cette distinction dans tout nouveau chargement |
| ΔFVE = +0,1393 contre +0,0086 au témoin favorable | Gain sur les checkpoints testés, rapporté à la variance totale des mêmes activations | Pas besoin d’un nouveau sweep de reconstruction |
| ΔCE = 0,301 contre 0,510 sur 60 documents appariés | Remplacement aux positions valides seulement | Une preuve fonctionnelle, pas une certification sémantique |
| 197/300 = 65,7 % au test odd-one-out | Score d’un échantillon stratifié, pas proportion de vrais concepts | Les applications ne doivent pas être sélectionnées sur ce seul score |
| 1B : 63,3 % ; 12B : 65,7 % ; pas de tendance monotone globale | Ce n’est pas un test d’équivalence des usages documentaires | 1B est une option d’ingénierie à vérifier sur les tâches, non un substitut automatiquement équivalent |
| Sondes sur 3 300 emails source, labels lexicaux | Les sondes sont hors pli, mais le SAE a déjà appris sur ces emails | Refaire le test **de toute la chaîne** sur des parents nouveaux |
| Jaccard des labels exacts ≈ 12–14 % | Ni alignement des directions ni mesure de stabilité fonctionnelle | Comparer poids et activations ; séparer stabilité du label et du mécanisme |
| Diffing 8/10 sur énergie/sport, 40+40 textes | Seuil descriptif de 1 point, sans validation du sens ni significativité | Nouveau test cible avec découverte/confirmation séparées |
| Retrieval : quatre requêtes, fusion et reranking encourageants | L’AP locale normalise par les pertinents du classement tronqué | P@10 humaine prioritaire ; MAP globale seulement avec dénominateur adéquat |
| Clustering ciblé : une requête, quatre groupes | Réassignation LLM, pas vérité terrain humaine, baseline dense incomplète | Réparer le mapping des labels et comparer à des méthodes simples |

Sources : rapport final chapitres 3–5 et annexes ; code lu à la révision ci-dessus. Le tableau ne remplace pas une réexécution des caches. [R ; C1–C9]

### 2.2 Écarts concrets repérés dans le code

**C1 — Chemin mémoire.** Le réservoir est déjà un fichier mappé (`torch.from_file(shared=True)`) ; proposer simplement « passer en memmap » ne corrige donc rien. Des lignes filler denses sont encore créées, empilées, reconstruites au chargement, puis réservées en fp32 dans le code étendu. Le mélange par blocs reçoit un tableau d’indices préalablement permuté globalement. Détails en E00.

**C2 — Clustering.** Dans `scripts/clustering_llm_test.py`, les embeddings de `list(feature_labels.values())` sont associés à `sorted(feature_labels.keys())`. Si l’ordre d’insertion diffère de l’ordre numérique, le label est associé au mauvais identifiant. La présence de ce risque est certaine ; son incidence sur l’ancien fichier de résultat n’est pas établie. Le même helper tronque à 64 tokens des labels courts **et des emails entiers** pour la conductance.

**C3 — Vérification d’hypothèses.** `_parse_verification_answer` renvoie faux pour une sortie qui ne commence pas par YES ; ainsi une sortie mal formée peut devenir un NO. `compute_verification_metrics` utilise `abs(rate_in-rate_out)>threshold` pour son score historique. Conserver cette ancienne métrique sous un nom explicite, mais créer un protocole nouveau avec statut invalide et direction attendue.

**C4 — Métriques de recherche.** `average_precision(relevance)` divise par la somme des pertinences **de la liste fournie**. Sur un top 50, elle ne calcule pas la MAP globale du corpus. Garder la fonction historique pour relire ses anciens résultats ; créer une API non ambiguë avec `total_relevant`, `judgment_scope` et `cutoff`.

**C5 — Historique et configuration.** Des commentaires de `config.py` et `saev5.py` justifient encore des choix avec d’anciens taux ou des contrastes remplacés. Ils ne sont pas des consignes expérimentales pour cette campagne. Le défaut `N_FEATURES_TO_LABEL=10` et celui de 500k tokens ne décrivent pas R0 : toute nouvelle commande doit expliciter sa configuration.

### 2.3 Groupes et carte : ce qui est classique, ce qui ne l’est pas

La littérature a déjà décrit des familles de features et le *feature splitting* ; une carte de directions est donc une visualisation légitime, mais pas une nouvelle preuve par elle-même. Une étude de 2026, **Unstable Features, Reproducible Subspaces**, examine précisément l’hypothèse de directions variables dans des sous-espaces reproductibles. Cette référence motive E05 sans garantir sa conclusion sur les emails et une extension résiduelle. [L2–L4]

Distinguer quatre objets : (i) voisinages dans un dictionnaire ; (ii) clusters d’emails ; (iii) correspondances entre entraînements ; (iv) hiérarchie de concepts. Une arête de proximité n’est pas une relation parent/enfant. Une carte 2D ne démontre ni universalité, ni causalité, ni stabilité. L’étude historique locale de groupes de **labels** ne battait pas son témoin aléatoire : ne pas la présenter comme un acquis positif.

---

## 3. Plan de livraison, priorités, budget et calendrier

### 3.1 Réalisme sur quinze jours

Quinze jours calendaires représentent souvent environ dix jours ouvrés. Ce plan est relatif : J1 est le premier jour d’exécution, pas une date civile supposée. Si Grégoire dispose de quinze **jours travaillés**, utiliser la marge pour l’annotation, les contre-exemples et la recette, pas pour ouvrir une nouvelle campagne de tailles de modèles.

L’ambition complète nécessite une disponibilité des données, environ **60–85 heures d’ingénierie ciblée**, et **12–20 heures cumulées d’annotation/relecture humaine** réparties entre Grégoire et les personnes disponibles. Ces enveloppes ne sont pas des promesses. Sans humain disponible ou sans données réelles, le résultat reste un benchmark cible synthétique et un prototype vérifiable ; ne pas changer l’étiquette de validation pour tenir le calendrier.

### 3.2 Trois niveaux de livraison

**M — Minimum non négociable.** Corpus et protocoles figés ; comparaisons cœur/complet/dense/lexical sur les mêmes emails nouveaux ; retrieval et diffing évalués ; carte exploratoire avec provenance ; mise en évidence des limites ; Streamlit lisible et reprise documentée. Les quatre applications doivent au moins avoir passé les tests de validité et produire un statut honnête.

**T — Cible de quinze jours.** M + comparaison de trois entraînements pour E05, première confirmation de corrélations, clustering ciblé avec baseline dense et quelques axes, mini-pilote analyste. Les validations humaines peuvent être des échantillons bornés ; les effectifs doivent rester visibles.

**B — Bonus.** Run 100M, fusion avancée, second corpus, carte hiérarchique validée. Ne lancer qu’après livraison de M et selon les portes de décision. Si B est abandonné, la campagne n’est pas un échec.

### 3.3 Registre des expériences

| ID | Livrable | Priorité | Temps ingénierie estimé | Calcul estimé hors file | Décision rendue possible |
|---|---|---:|---:|---:|---|
| E00 | Profil mémoire, compactage et reprise testée | M | 8–12 h | 2–5 h GPU + 2–4 h CPU | Dimensionner les runs sans saturer le cluster |
| E01 | Jeu comparable de représentations et sondes indépendantes | M | 8–12 h | 8–24 h GPU + 1–4 h CPU | L’extension apporte-t-elle autre chose que le cœur ? |
| E02 | Registre de labels, annotation aveugle mutualisée | M | 3–5 h | 3–8 h GPU | Peut-on faire confiance aux propriétés affichées ? |
| E03 | Retrieval cible, benchmark et interface | M | 6–9 h | 2–6 h GPU + <2 h CPU | Retrouve-t-on mieux une propriété utile ? |
| E04 | Diffing cible, hypothèses confirmées séparément | M | 7–10 h | 4–12 h GPU + 1–3 h CPU | Quelles différences tiennent sur d’autres emails ? |
| E05 | Carte et comparaison de groupes | T, carte M | 8–12 h | 4–14 h GPU si deux entraînements supplémentaires + 2–6 h CPU | Les thèmes sont-ils retrouvés malgré les permutations ? |
| E06 | Paires de propriétés avec supports et confirmation | T | 4–6 h | 2–6 h GPU + 1–3 h CPU | Existe-t-il des associations non triviales ? |
| E07 | Clustering ciblé corrigé, baseline et évaluation | T | 5–8 h | 2–6 h GPU + 1–3 h CPU | Le filtrage par propriété aide-t-il le regroupement ? |
| E08 | Mini-pilote et recette de l’interface | M/T | 5–7 h | <1 h CPU hors pré-calculs | Un analyste trouve-t-il des pistes utiles ? |
| E09 | 1B/100M, seulement si autorisé | B | 4–8 h supplémentaires | enveloppe 12–36 h GPU à confirmer, sinon différer | Le volume supplémentaire améliore-t-il un usage identifié ? |

Les calculs peuvent réutiliser les mêmes extractions ; ne pas additionner mécaniquement deux fois la labellisation commune ou la reconstitution de corpus. À l’inverse, les appels au juge sur des milliers de paires peuvent dépasser ces enveloppes : E00 doit mesurer leur débit. Budget de base maximal proposé : 120 GPU-h ; atteindre ce plafond **interrompt les branches optionnelles**, pas la qualité des contrôles.

### 3.4 Calendrier et règles de réduction de périmètre

| Période | Travail humain | Travail cluster | Porte de décision |
|---|---|---|---|
| J1–J2 | Inventaire, conventions, données autorisées, protocole, bugs bloquants | Profilage court et tests mémoire | Données/split/ressources fiables ? |
| J3–J4 | Contrats de features, requêtes et guides d’annotation | Extraction unique ; SAE 1B de référence ; embeddings denses | Une chaîne propre existe-t-elle ? |
| J5–J6 | Annotation ciblée ; implémentation retrieval/diffing | E02–E04 ; deux entraînements extra seulement si budget | Retrieval et diffing ont-ils des résultats auditables ? |
| J7–J9 | Carte, corrélations, clustering et pages Streamlit | E05–E07 mutualisés | M terminé au plus tard J9 |
| J10–J12 | Mini-pilote, analyses confirmatoires figées | Compléments uniquement ciblés ; pas de nouveau sweep | Aucune nouvelle branche longue après J10 |
| J13–J15 | Recette, nettoyage non destructif, passation | Exports/reproduction courte seulement | Arrêt des changements fonctionnels, livraison |

Si E00 exige plus de deux jours, réutiliser les checkpoints compatibles pour les analyses exploratoires, et limiter le nouveau SAE à l’extraction des emails d’apprentissage. Ne pas attendre un correctif 100M pour livrer les applications.

Si aucun modèle propre n’existe à J4, basculer vers **cœur gelé / dense / lexical** pour fournir les applications et noter « effet de l’extension non évalué inductivement ». Ne pas contaminer le test pour conserver un bras supplémentaire.

Si la file GPU empêche deux graines supplémentaires, livrer la carte d’un dictionnaire avec badge « stabilité inter-entraînements non mesurée ». Les anciens poids 12B peuvent servir à une étude descriptive distincte, jamais fusionnée avec le nouveau test 1B.

Si les ressources humaines sont réduites, privilégier la pertinence du top 10 et la vérification des hypothèses proposées ; réduire le nombre de requêtes ou d’axes **avant** confirmation. Ne pas supprimer les cas négatifs après lecture des résultats.

---

## 4. Architecture commune des données et des évaluations

### 4.1 Une nouvelle campagne, un seul contrat de données

Créer une campagne versionnée, par exemple `post_soutenance_v1`. Les noms de fichiers nouveaux proposés dans ce document sont **à implémenter** ; ils ne sont pas supposés exister dans le dépôt. Lire d’abord les fonctions et tests existants et réutiliser les composants valides.

Produire un manifeste local comportant : source, version, statut synthétique/réel, autorisation, nombre d’emails parents, nombre de variantes, règles de nettoyage, fingerprints, longueurs avant/après tokenisation, champs optionnels lieu/date et leur provenance. Interdire l’alignement par numéro de ligne seul. La jointure doit utiliser des identifiants stables et vérifier les collisions/doublons.

### 4.2 Découpage parent-aware proposé

Sur les seuls emails source disponibles, figer des groupes avant tout nouvel apprentissage :

- **FIT : 60 %** environ, pour l’extension, les calibrations numériques et les labels de features ; variantes FIT admises pour l’apprentissage seulement.
- **DEV : 15 %** environ, pour les choix bornés de normalisation, les requêtes de développement, le regroupement des features et les hypothèses candidates.
- **CONFIRM : 25 %** environ, réservé aux résultats finaux et contenant en priorité les **emails source**, sans multiplier les observations par leurs variantes.

Avec 3 474 parents, cela représente environ 2 084 / 521 / 869 parents ; ce sont des ordres de grandeur. Une variante garde le split de son parent. Conserver les near-duplicates dans un même groupe. Fixer la graine du split une fois, indépendamment des graines du SAE.

Des dates fiables permettent éventuellement un test temporel plus pertinent ; le décider à J1. Ne pas inventer ce test sur des dates de génération synthétiques. Pour le scénario régional, distinguer région du client et lieu de l’incident ; date d’envoi et date de l’événement. Les attributs dérivés par modèle restent des estimations à auditer.

La validation interne de l’entraînement SAE se fait dans FIT, sur des **parents ou shards dédiés**, pas seulement sur 8 192 tokens aléatoires potentiellement issus des mêmes emails que le gradient. CONFIRM n’entre jamais dans l’encodeur entraîné, l’IDF, la PCA d’initialisation, le seuil de regroupement, le choix de requête, la calibration p90 ou le réglage de sonde.

### 4.3 Réutilisation des anciens checkpoints

Un checkpoint ancien a vraisemblablement vu des parents désormais assignés à CONFIRM. Son usage reste possible dans un panneau « exploratoire historique », pas dans le tableau inductif principal. Pour éviter un entraînement non nécessaire, comparer les manifestes effectifs : si l’absence de chevauchement est prouvée, il est réutilisable ; si les informations manquent, ne pas présumer cette absence.

Les poids préentraînés externes GemmaScope/bge-m3 sont fixes. On ne peut pas exclure toute exposition préentraînement à des textes publics ; la garantie recherchée ici porte sur **l’absence d’adaptation locale à CONFIRM**. Déclarer cette portée.

### 4.4 Ensemble commun de représentations

| Code | Représentation | Apprentissage local | Rôle |
|---|---|---|---|
| CORE | GemmaScope gelé, même modèle/couche que FULL | Aucun ; seulement transformations autorisées FIT | Baseline directe de l’extension |
| EXTRA | Branche supplémentaire seule | FIT | Diagnostic : ne pas en faire obligatoirement un produit |
| FULL | Concaténation CORE + EXTRA | Extension FIT | Méthode candidate |
| DENSE | bge-m3 gelé, pooling officiel, texte complet ou chunks documentés | Aucun | Baseline sémantique dédiée |
| TFIDF | Vecteur lexical mots 1–2 grammes, normalisation L2 | Vocabulaire/IDF sur FIT | Baseline inspectable et peu coûteuse |
| BM25 | Index lexical de recherche | Corpus d’index défini | Baseline IR pour E03 ; pas besoin pour les sondes |
| GROUP | Agrégation des features d’un groupe E05 | Groupes définis sur FIT/DEV | Option pour navigation/stabilité, pas bras obligatoire partout |

Un embedding de l’état caché de Gemma peut être ajouté **sans nouvelle extraction** comme contrôle du signal d’entrée, mais ne remplace pas DENSE, qui représente l’alternative industrielle de recherche sémantique. Déclarer le pooling exact. Ne pas mélanger des vecteurs 1B et 12B dans un même index.

Pour les tâches d’annotation par label (diffing, corrélations, retrieval par propriété), fournir le même budget de **candidats labellisés** aux bras CORE et FULL afin de distinguer gain et taille de catalogue. Rapport secondaire « catalogue complet disponible » autorisé si le coût et le nombre de labels sont affichés. Dans les sondes, au contraire, utiliser les codes complets sans filtrer sur le juge : la question n’est pas la même.

### 4.5 Deux pistes d’évaluation, à ne pas fusionner

**Piste stricte** : transformations de représentation apprises sur FIT/DEV, tâche confirmée sur parents CONFIRM. C’est celle qui établit la généralisation de l’adaptation locale.

**Piste exploratoire d’indexation** : un utilisateur fournit une collection à indexer ; BM25 peut calculer les fréquences documentaires sur cette collection sans labels. C’est un protocole IR usuel, mais transductif au niveau de l’index. Identifier `idf_scope=index_corpus` et appliquer le même principe aux statistiques SAE (p90/IDF) lorsque comparées. Ne pas mélanger ces deux protocoles dans un chiffre unique. Le tableau central E03 adoptera par défaut la piste stricte, plus simple à interpréter.

### 4.6 Annotation mutualisée et contrôles

Préparer une fiche par propriété : définition, inclusions, exclusions, négations, demande vs description, urgence exprimée vs incident grave supposé. Les catégories métier ne doivent pas être définies en lisant les features produites par FULL.

Utiliser quatre états : `present`, `absent`, `uncertain`, `unjudgeable`. Un JSON invalide est `invalid_model_output`, pas `absent`. Conserver un extrait justificatif local lorsque autorisé, un motif, la version d’annotation et le statut humain/modèle/règle. Ne jamais additionner ces trois provenances comme une même vérité terrain.

Plan humain recommandé : 150–200 emails sources annotés pour les cinq intentions à la fois ; pool des top 10 de 12 requêtes test ; 40–80 exemples de propriétés/corrélations ; 40–60 exemples de clusters ; 60–100 cas en double annotation. Les ensembles se recouvrent lorsque les tâches le permettent. Chronométrer 20 cas à J2 pour calibrer le nombre faisable. L’accord humain et les désaccords résolus sont des résultats, pas des contrôles à effacer.

---

## 5. E00 — Audit de ressources et correction du chemin mémoire

### Question et raison de priorité

Pourquoi un run 12B/50M nécessite-t-il tant de RAM malgré le memmap ? Peut-on rendre le calcul proportionnel à un **bloc de travail** plutôt qu’à tout le jeu d’activations et aux documents filler ? Cette correction conditionne une campagne sûre sur le cluster partagé ; elle ne justifie pas à elle seule un run 100M.

### 5.1 Ce qui est vérifié, ce qui reste à mesurer

Le script Slurm de probe 50M cite **358,5 Go de RSS effectivement rapportés pour le job 45803, 12B/25M/couche 12**, puis extrapole ≈717 Go à 50M et réserve 900G. Le commentaire précise que l’ancien conseil « memmap, 96G suffisent » a été démenti. Il s’agit d’une mesure **rapportée dans un script**, pas d’une lecture actuelle de `sacct` pendant cet audit. Nous n’avons pas les événements cgroup et logs complets du job 50M : impossible d’attribuer avec certitude son arrêt à une allocation particulière. [C10]

**Ne pas confondre** RAM physique du nœud, mémoire facturée au cgroup du job, RSS du processus, espace virtuel mappé, cache de fichiers, mémoire partagée IPC et mémoire GPU. Ici `shared=True` signifie un mapping de fichier partageable, pas une allocation nécessairement située dans `/dev/shm`.

### 5.2 Taille minimale des données brutes

Pour `resid_post`, d’après les presets du dépôt : 12B utilise `d=3840`, 1B `d=1152`. En bf16, un élément prend deux octets. Vérifier les dimensions dans les configurations **réellement chargées**, surtout avec un autre hook.

`taille = nombre_tokens × dimension_activation × 2 octets`

| Activations brutes | 12B, d=3840 | 1B, d=1152 |
|---|---:|---:|
| 25M tokens | 192,0 Go = 178,8 Gio | 57,6 Go = 53,6 Gio |
| 50M tokens | 384,0 Go = 357,6 Gio | 115,2 Go = 107,3 Gio |
| 100M tokens | 768,0 Go = 715,3 Gio | 230,4 Go = 214,6 Gio |

Ces valeurs sont la taille d’**un seul tableau** ; ce ne sont ni la VRAM, ni une demande Slurm minimale, ni une borne supérieure de RAM. Un lecteur par blocs permet un espace de travail plus petit que le fichier. Passer de 12B à 1B divise ce poste par 3,33, pas par douze. Le fichier nommé `raw_residuals` contient dans le chemin actuel les activations **x** ; le résidu est calculé par le cœur pendant le forward d’entraînement.

### 5.3 Inefficacités identifiées dans le code

| Point | Code vérifié | Effet probable / correction |
|---|---|---|
| Filler dense lors de l’extraction | `saev5.py` : un `zeros(d_core)` par document filler, puis `torch.stack(all_doc_sae_acts)` | La liste et le résultat empilé peuvent coexister. Supprimer les lignes filler du tableau documentaire, pas seulement du fichier sauvegardé |
| Compactage uniquement sur disque | `save_doc_acts_sparse_filler` retire les lignes ; `_reconstruct_sparse_filler_payload` les recrée avec `torch.zeros(n_total,d)` | Une recharge annule le gain RAM. Retourner codes compacts + index documentaire explicite |
| Tableau étendu inutilement grand | `torch.empty((len(all_texts),d_core+D_EXTRA),float32)` | Les trous filler sont non initialisés, puis le tableau entier est sérialisé/rechargé. Persister uniquement les lignes utiles ; ne jamais exporter ces trous |
| Localité de lecture perdue | `perm=randperm(n_total)`, `train_idx=perm[...]`, puis `block_shuffle_indices(train_idx)` | Les blocs contiennent déjà des positions aléatoires. Mélanger l’ordre des **blocs de lignes contiguës**, puis les lignes dans un bloc chargé |
| Réservoir encore référencé | `reservoir` et sa fermeture existent dans la portée principale ; `del raw_residuals` ne supprime pas nécessairement ces références | Séparer extraction/entraînement/réencodage en processus ; fermer le lecteur après usage. `malloc_trim` ne ferme pas un mapping vivant |
| Écritures de réservoir dispersées | Réservoir de Vitter avec remplacements aléatoires, même bufferisés et triés | Pages sales, défauts majeurs et I/O réseau possibles. Préférer shards séquentiels avec échantillonnage déterministe documenté |
| Copie du corpus pour hash | `"\n".join(train+filler+test+diff).encode()` | Grandes allocations transitoires de chaînes et d’octets ; hash incrémental avec délimiteurs de longueur |
| Cache et reprise incomplets | Extraction checkpointée ; harnais d’entraînement lu sauvegardant surtout en fin | Ajouter reprise d’optimiseur, étape, ordre de blocs, RNG et scalers ; pas seulement poids finaux |

À titre de calcul, **1,2 million de fillers** représentés sur 16 384 coordonnées bf16 font **39,3 Go** de zéros ; liste + empilement peut approcher 78,6 Go rien que pour ces valeurs. La réserve correspondante en code complet fp32 de 17 408 dimensions fait **83,6 Go**. Le nombre cible de chunks n’est pas le nombre réalisé : extraire `n_filler` des logs avant de chiffrer le job. Une allocation `empty` réserve d’abord un espace virtuel ; sa résidence dépend des accès et de la sérialisation. Ne pas additionner des maxima non simultanés pour fabriquer un diagnostic.

### 5.4 Mesures à collecter avant correction

Créer une commande CPU d’inventaire, sans lire le contenu des emails dans la sortie. Lire, si disponibles, les jobs propres à Grégoire :

```bash
sacct -j "$JOB_ID" --units=G \
  --format=JobID,JobName,State,ExitCode,Elapsed,AllocTRES,ReqMem,MaxRSS,MaxVMSize,NodeList
scontrol show job "$JOB_ID"  # lorsque le job est encore disponible
```

`MaxRSS` n’est pas automatiquement la somme de tous les processus/steps : conserver les lignes `.batch` et `.0`, les nombres de tâches et la configuration de comptabilité du site. Ne pas inspecter ni exporter les jobs d’autrui. [L7]

À chaque frontière de stage et toutes les 15–30 secondes : PID, stage, documents/tokens vus et retenus, `VmRSS`, `VmHWM`, `RssAnon`, `RssFile`, `RssShmem`, `smaps_rollup` si lisible, mémoire GPU allouée/réservée, I/O lus/écrits, durée et débit. Résoudre le vrai cgroup depuis `/proc/self/cgroup` ; en v2, relever `memory.current`, `memory.peak` si présent, `memory.events`, `memory.stat` (notamment `anon`, `file`, `file_mapped`, `file_dirty`, `file_writeback`) et la limite. Prévoir un repli cgroup v1 plutôt que supposer les chemins. [L5–L7]

Enregistrer le type de filesystem et l’espace/quota libre. Un memmap sur un stockage réseau lent peut avoir un profil radicalement différent d’un SSD local. Un disque local Slurm éphémère exige copie atomique/checkpoint vers le stockage persistant autorisé avant fin de job.

### 5.5 Correction minimale à réaliser en premier

1. **Codes documentaires compacts.** Créer `doc_id -> row_idx`. Le filler n’a aucune ligne dans les codes documentaires. Conserver séparément `token_pool_manifest`. Adapter les slices positionnels et tous les consommateurs ; ne pas seulement changer la sérialisation.
2. **Lecteur de blocs contigus.** Tirer les quelques indices de validation, puis parcourir les autres lignes par blocs contigus de 32k–64k tokens. Charger un bloc bf16 en CPU, le mélanger localement, servir des mini-batches de 1 024. Permuter l’ordre des blocs à chaque époque. Documenter que cet ordre n’est pas une permutation uniforme globale et qu’il peut modifier l’optimisation ; préserver le budget BatchTopK du mini-batch.
3. **Stages séparés.** Extraire, entraîner, réencoder, étiqueter et analyser via des commandes différentes. Fin du processus d’entraînement = libération explicite du mapping, du cœur chargé et des alias. Les scores ne doivent pas dépendre d’un import de `saev5.py` qui exécuterait le pipeline.
4. **Pas de matérialisation totale.** Aucun `.float()`, `.clone()`, `torch.cat`, `torch.stack` ou `.toarray()` sur le jeu complet d’activations. Cast fp32 uniquement du bloc ou des calculs résiduels nécessaires ; la branche extra reste fp32.
5. **Checkpoints d’entraînement complets.** Sauvegarder après un nombre borné de steps et sur signal : poids, optimiseur, epoch/block/batch cursor, état RNG, ordre des blocs, buffers BatchTopK/AuxK/seuil, empreintes de données et config. Reprendre sans dupliquer ni sauter de mini-batch.
6. **Écritures atomiques.** Un manifeste ne doit avancer qu’après fermeture/flush des shards concernés. Pour la résistance à un crash de nœud, ajouter `fsync`/protocole de commit du shard ; vider une file Python n’équivaut pas à une garantie de persistance.

### 5.6 Variante plus ambitieuse, seulement si la correction minimale est insuffisante

Créer un dataset de shards séquentiels de x en bf16, sans grand réservoir writable. Sélectionner de manière déterministe les positions candidates par hash stable `(corpus_version, parent_id, doc_id, token_position, seed_sampling)` ou par un échantillonnage de réservoir **de métadonnées**. Le lecteur conserve le contexte complet nécessaire à Gemma : choisir un token n’autorise pas à extraire sa représentation sans les tokens qui le précèdent.

Deux chemins acceptables : (a) extraire une fois des shards, puis sélectionner les indices retenus dans les shards ; (b) déterminer les positions par une première passe de tokenisation et n’extraire que les documents/chunks concernés. Pour 100M proches du corpus total, la seconde passe peut ne pas économiser de forward : mesurer avant de complexifier.

Si le réservoir existant est conservé, résoudre explicitement les collisions d’indices en gardant l’**écriture chronologiquement dernière** avant l’indexation. Un tri stable ne garantit pas à lui seul la sémantique d’une affectation vectorisée à indices répétés. Sauvegarder l’état de l’échantillonnage. Ce point est un contrôle d’équivalence, pas la cause démontrée du manque de RAM.

### 5.7 Expérience de validation du correctif

- Test CPU sur données factices : absence de filler dans les tensors utiles, exactitude du mapping, reconstruction des indices, aucune valeur non initialisée sérialisée.
- Même petite extraction avant/après, même dtype, même masque : comparer codes CORE/FULL et tokens retenus aux tolérances numériques justifiées. Pour l’ordre de SGD modifié, ne pas prétendre à l’identité des poids ; tester couverture de chaque époque, stabilité de loss et capacité de reprise.
- Profilage 1B sur 0,25M puis 1M tokens ; 12B sur un sous-ensemble court uniquement pour reproduire le poste mémoire suspect. Ce sont des tests d’ingénierie, pas un sweep scientifique.
- Charge synthétique de lecteur couvrant un fichier plus grand que la mémoire de travail ; vérifier défauts de pages, RSS/cgroup, absence de croissance linéaire incontrôlée des tensors anonymes.
- Checkpoint interrompu puis repris : comparer à une exécution continue avec les mêmes batches. Exiger égalité des curseurs/expositions et comparabilité numérique documentée ; restaurer RNG/optimiseur.

**Critère de passage proposé :** le pic de mémoire anonyme est compatible avec une demande de 64–128 Gio selon le stage, le coût n’augmente plus avec le nombre de lignes filler, les étapes reprennent correctement et l’estimation du run complet reste sous le budget. La mémoire de cache fichier peut augmenter puis être récupérée : suivre le cgroup, pas seulement une courbe RSS. Aucun chiffre de mémoire cible n’est garanti avant mesure.

### Livrables, Streamlit et nettoyage

`resources_profile.jsonl`, `memory_diagnosis.md`, tableau avant/après par stage, test de reprise, contrat de codes compacts et adaptateur de lecture legacy. Streamlit affiche temps/mémoire/volume réel par run. Archiver les anciens `.slurm` surdimensionnés comme `legacy_memory_layout`, sans les effacer ; les réécrire pour la nouvelle campagne seulement. Les anciens résultats scientifiques ne deviennent pas faux parce que le stockage a été amélioré. Marquer à contrôler uniquement les résultats qui ont lu des lignes mal alignées ou des caches incomplets.

---
## 6. E01 — Représentations comparables et gain propre de l’extension

### Pourquoi cette expérience

Le résultat actuel des sondes établit la présence d’un signal dans CORE+EXTRA, pas la valeur de l’extension sur de nouveaux emails. C’est le verrou scientifique le plus directement lié à la décision industrielle : pourquoi exploiter un module supplémentaire si le cœur ou une méthode simple rend le même service ? [R, §4.7 et annexe I]

Cette comparaison est un **test applicatif nécessaire**, pas un nouveau balayage d’hyperparamètres. Fixer une architecture, un split et une liste de tâches ; ne changer que la représentation.

### 6.1 Conditions et modèle de travail

**Choix proposé : Gemma-3-1B, couche 13, cœur 16k, extension 1 024, Kextra=5**, selon le preset du dépôt. Vérifier le SAE et le hook localement. Garder `BATCH_SIZE_EXTRA=1024`, 10 époques et le taux d’apprentissage existant explicites. La batch size n’est pas un simple paramètre de vitesse sous BatchTopK : ne pas la changer discrètement.

Le premier run propre utilise uniquement FIT et son éventuel filler approuvé, avec un plafond de 25M tokens **réellement disponibles**. Ce n’est pas l’obligation d’atteindre 25M : si le corpus emails offre 4M tokens pertinents, les conserver comme tels plutôt que les présenter comme 25M emails distincts. Écrire `n_unique_source_tokens`, `n_selected_tokens`, `n_presentations` et la composition emails/variantes/filler. Le run 100M éventuel est séparé.

Trois graines sont prévues pour l’extension si le budget permet E05. La sélection des tokens, le corpus et le split restent fixes ; seules l’initialisation et la séquence d’optimisation varient. Identifier séparément `seed_data`, `seed_init`, `seed_order`, `seed_judge`. Si l’initialisation PCA déterministe neutralise une source de variation, le signaler : « trois entraînements » ne signifie pas nécessairement trois dictionnaires initialement différents.

### 6.2 Portes de qualité avant évaluation

Contrôler shapes, hook, statut gelé des poids Gemma/core, `encoder_input=x`, masques et buffers du checkpoint. Vérifier que la loss diminue et que la branche n’est pas intégralement morte ; éviter un seuil arbitraire de « bonne FVE » choisi après observation. Comparer ΔFVE et L0 sur la validation interne FIT. Une dégradation forte de reconstruction indique un problème à diagnostiquer, mais une FVE modeste n’autorise pas à jeter un modèle qui serait utile en aval.

Le passage à 1B est un choix d’efficacité, pas un résultat de non-infériorité. Un contrôle 12B sur un petit jeu DEV peut être envisagé seulement avec des modèles entraînés sans CONFIRM. Si un nouveau 12B est trop coûteux, comparer 1B aux baselines et inscrire « comparaison applicative 1B/12B non conclue ». Ne pas comparer un 1B propre à un ancien 12B exposé au test comme si le protocole était symétrique.

### 6.3 Construction des vecteurs

Extraire x une fois pour les documents utiles ; conserver les codes CORE et EXTRA distincts avec `doc_id`. Produire FULL par concaténation pour éviter de recalculer le cœur. La séparation cœur/extra se fait sur la dimension **réelle** de Wdec du checkpoint, pas une constante ancienne `262144` ou une largeur tirée du nom de fichier.

Garder le max-pooling de référence. Ajouter une seule alternative motivée : **moyenne des trois plus fortes activations par feature, avec zéros inclus et diviseur `min(3,n_tokens_valides)`**. Elle teste la domination d’un pic isolé et reste sensible à la longueur ; ne pas la présenter comme une correction théorique complète. Reporter par tranches de longueur, et utiliser une baseline longueur seule. Si le développement montre que la moyenne entière est plus simple à intégrer, la choisir **avant CONFIRM** et documenter le changement. Aucun choix entre dix agrégations sur le test.

Pour les sondes, standardiser les coordonnées sans centrer une matrice sparse (`with_mean=False`) ou appliquer la normalisation choisie ; apprendre ces paramètres sur FIT. Même règle pour CORE, EXTRA et FULL. Le facteur d’échelle des deux blocs est explicite. Un bras FULL ne doit pas bénéficier d’un tuning plus large que CORE.

Pour DENSE, utiliser bge-m3 gelé dans son mode dense officiel. Vérifier la longueur et les préfixes requis par le modèle local. Une troncature à 64 tokens copiée du helper de labels n’est pas acceptable pour des emails entiers. Si chunking nécessaire, l’appliquer au même texte autorisé et en consigner les frontières ; la comparaison des modèles doit afficher la proportion de texte ignorée.

Pour TFIDF, reprendre en première intention la configuration du test de sondes existant, en lisant le script au moment de l’implémentation. Écrire tous les paramètres dans le manifeste, y compris analyseur, n-grammes, fréquence minimale, `sublinear_tf`, accents, stopwords, `max_features` et normalisation. Ne pas déduire la configuration d’un ancien paragraphe du rapport si le script diverge.

### 6.4 Sondes d’intention : protocole applicatif

Reprendre les cinq intentions existantes comme test secondaire identifiable. Apprendre cinq régressions logistiques binaires sur FIT, avec les labels lexicaux corrigés. Régularisation fixe ou choix d’un petit jeu de valeurs sur DEV, identique entre représentations ; aucune optimisation sur CONFIRM. Seuil de décision fixé à 0,5 ou calibré exclusivement sur DEV.

Évaluer sur :

1. **CONFIRM complet avec labels faibles** : permet la continuité historique, sous nouveau split indépendant. Ce test reste lexical.
2. **150–200 emails CONFIRM annotés humainement** pour les cinq intentions : mesure de la compatibilité avec une lecture indépendante. Si une intention compte trop peu de positifs (par exemple moins de 20), publier support et intervalle, sans conclusion forte ni seuil de succès inventé.

Reporter accuracy, balanced accuracy, F1, précision/rappel par classe, prévalence, matrice de confusion et AP de classification lorsque les deux classes sont présentes. La prévalence est la baseline AP ; l’accuracy du classifieur majoritaire doit apparaître. Dans ce contexte, l’AP de classification sur toutes les observations étiquetées ne se confond pas avec l’AP locale d’un top 50 de retrieval.

Comparaisons principales pré-spécifiées : FULL−CORE ; FULL−DENSE ; FULL−TFIDF. Utiliser des différences appariées et IC par bootstrap de parents ; McNemar pour les erreurs binaires lorsque pertinent. Avec cinq intentions, corriger la famille des tests annoncés et séparer les métriques descriptives des tests primaires. Ne pas utiliser les cinq plis comme cinq répétitions indépendantes d’un système.

### 6.5 Gain opérationnel, au-delà des sondes

Le critère central de la représentation finale combine résultats E03/E04, latence et lisibilité. Définir à J2 avec l’encadrement une marge utile, par exemple **+0,05 de P@10** ou davantage de thèmes validés à effort d’exploration constant. Cette marge est une proposition de décision métier, pas une constante de la littérature.

Si FULL égale CORE dans l’incertitude, ne pas écrire « aucun effet » ; conclure « gain non établi à cet effectif ». Si un intervalle suffisamment étroit exclut un gain utile, préférer le cœur pour le prototype. Si TFIDF/DENSE gagne, conserver cette méthode dans l’interface et utiliser les features pour expliquer/explorer plutôt que prétendre remplacer le moteur.

### Sorties, temps et suite

Sorties : `representations/`, `representation_manifest.json`, `probe_predictions.parquet`, `paired_comparisons.json`, `length_strata.json`, fiches d’erreurs anonymisées. Temps proposé : 8–24 GPU-h pour extraction/apprentissage/embeddings de base, à recalculer par E00 ; 1–4 CPU-h pour les sondes. Les anciens codes peuvent économiser l’exploration, pas prouver l’indépendance.

Streamlit : sélecteur CORE/FULL/DENSE/TFIDF, même `doc_id`, comparaison côte à côte, labels faibles vs humains séparés, latence et portée inductive visibles. Nettoyage : les anciens résultats deviennent `historical_transductive`, pas « faux ». L’ancienne ablation cœur/extension K32/couche24 ne clôt plus la question de cette campagne ; lui ajouter un renvoi au nouveau protocole.

---

## 7. E02 — Registre des features et validation humaine mutualisée

### Pourquoi

Les applications qui sélectionnent des features selon leurs noms dépendent de la qualité et de la provenance de ces noms. Comparer un cœur disposant d’un grand catalogue Neuronpedia à une extension de 197 labels filtrés par un juge sans contrôler cette asymétrie peut confondre qualité de représentation et qualité du catalogue.

### Protocole

Créer une clé complète `feature_uid = (reader_revision, sae_revision, branch, local_index)`. Stocker le label original, sa source, une éventuelle traduction/révision, le prompt, le modèle juge, les effectifs d’activation, les exemples de labellisation et le statut. Les exemples viennent de FIT ; DEV peut servir à tester leur généralisation. Pas de labellisation adaptative à partir de CONFIRM.

Recenser toutes les features extra vivantes, et les features core rencontrées dans les emails FIT. Ne pas labelliser aveuglément tout le cœur : présélectionner un catalogue borné selon couverture, fréquences et besoins des requêtes DEV. Pour les comparaisons CORE/FULL, constituer un catalogue à budget comparable et enregistrer les probabilités/règles de sélection. Un second panneau libre peut explorer le catalogue plus grand, sans participer aux tests à budget apparié.

Budget initial : jusqu’à **300 labels à produire ou réviser**, en donnant la priorité aux features utilisées dans E03/E04 ; jusqu’à 120 supplémentaires seulement si leur utilité est identifiée et le budget juge disponible. Des noms déjà disponibles restent utilisables avec provenance ; ne pas renommer des features pour obtenir artificiellement les mêmes labels entre graines.

Le juge local propose un nom court et une définition de présence, avec exemples positifs et contre-exemples. Autoriser `unclear`, `syntactic`, `mixed`, `insufficient_support` : toutes les features ne doivent pas devenir des thèmes métier. Les labels syntaxiques ne sont pas faux, mais leur utilité analytique peut être faible.

Faire vérifier humainement **60–100 features/paires de motifs**, sélectionnées avant lecture des résultats applicatifs : différentes fréquences, branches, scores de juge et thèmes. Les humains ne voient pas « CORE/FULL » ni le score. Pour chacune : le label décrit-il les extraits ? quels contre-exemples ? phénomène lexical ou concept plus large ? utilité potentielle pour l’analyse ? Produire accord et taux par strate, sans extrapoler naïvement à tout le dictionnaire.

### Résultats attendus et décisions

Attente raisonnable : un catalogue plus petit, mieux documenté, comprenant des features utiles et des artefacts. Un taux de rejet humain important est informatif. Ne pas exiger un nombre minimal de « bons concepts » pour accepter l’expérience.

Si les labels sont trop incertains, les applications peuvent encore utiliser les codes mais doivent afficher un identifiant et des extraits, pas une assertion métier. Si des fonctions sélectionnent par embeddings de labels, les sorties sans label valide restent hors de cette sélection, avec un taux de couverture déclaré. Un budget insuffisant pour labelliser les features de trois graines n’empêche pas l’alignement par poids/activations d’E05.

### Intégration et coût

Préparer un fichier local d’annotation aveugle avec ordre mélangé, clés stables, sans source de méthode apparente. Les données personnelles restent sur les emplacements autorisés. Le calcul dépend du nombre et de la longueur des générations : environ 3–8 GPU-h, plafond révisé après 20 features. Ajouter des tests sur sorties vides, erreurs de traduction, JSON non valide et ordre d’identifiants.

Streamlit doit afficher les sources de labels, les degrés d’évidence et les exemples, avec filtres « humain », « modèle », « non vérifié ». Les anciens labels restent archivés ; une nouvelle version de label invalide les caches de **sélection par label** et d’hypothèses qui en dépendent, pas les activations physiques.

---

## 8. E03 — Retrieval par propriété sur les emails

### Question industrielle

Peut-on retrouver des emails exprimant une propriété difficile à ramener à un seul mot-clé, sans imposer un apprentissage supervisé pour chaque nouvelle requête ? Exemples de **requêtes proposées pour la future évaluation**, non résultats historiques :

- « Le client a déjà contacté le service plusieurs fois sans solution. »
- « Le client menace de résilier si le problème persiste. »
- « Plusieurs personnes ou logements semblent concernés par un même problème. »
- « Le client demande l’explication d’un montant plutôt qu’un remboursement. »
- « Le client décrit une coupure répétée plutôt qu’une panne isolée. »
- « Le client exprime une urgence immédiate, même sans employer le mot urgent. »

Vérifier sur DEV que les propriétés existent en nombre suffisant. Des phrases négatives doivent rester des contre-exemples : « je ne souhaite pas résilier » ; « inutile d’intervenir aujourd’hui » ; « ma facture est correcte ». Ne pas ajouter ces exemples fabriqués au corpus principal sans les étiqueter comme jeu de contrôle distinct.

### 8.1 Taille et gel du jeu de requêtes

Cible : **18 requêtes**, dont 6 DEV et **12 CONFIRM**, couvrant au moins six propriétés, avec formulations lexicales et paraphrasées. Les variantes d’une même propriété appartiennent à une même famille : l’incertitude se calcule au niveau des familles, pas comme si 12 paraphrases étaient indépendantes. Bonus à 24 requêtes seulement après couverture de jugement suffisante.

Faire proposer/valider les propriétés avant de regarder les résultats des moteurs. Enregistrer `query_family_id`, définition de pertinence, exemples, exclusions et poids dans l’agrégation. Si le corpus ne contient pas une propriété, conserver le cas « pas de résultat attendu » en contrôle, plutôt que le remplacer a posteriori par une propriété plus favorable.

### 8.2 Méthodes à comparer

**Obligatoires :** recherche par propriété CORE, recherche par propriété FULL, DENSE bge-m3, TFIDF. Ajouter BM25 lexical, peu coûteux, pour ne pas confondre mauvaise pondération TFIDF et échec du lexical. EXTRA seul est un diagnostic, pas nécessairement un sixième moteur exposé.

Le chemin SAE reprend les fonctions de sélection de features par similarité de labels et de normalisation p90 du dépôt. Lire leurs valeurs effectives et les figer dans un manifeste. Remplacer les replis silencieux par un retour « aucun latent suffisamment associé » et un diagnostic. Tous les paramètres de sélection sont réglés sur DEV, puis fixes. Garder la même limite de candidats et le même juge de labels pour CORE/FULL.

Les termes p90 sont appris sur FIT dans la piste stricte ; une feature au p90 nul doit avoir un traitement documenté, pas une division par epsilon qui explose. Reprendre la formule locale seulement après test numérique sur des vecteurs factices : normalisation, poids de similarité et température influencent le classement. Aucun reranking au stade de comparaison principale.

**Second étage facultatif :** la même fusion RRF (constante fixée) et le même reranker local sont appliqués aux mêmes budgets de candidats pour les bras comparés. Montrer le gain du reranker séparément. Un gain après un LLM fort ne prouve pas que les features en sont la cause. Latent Terms est une **architecture de recherche distincte**, pas le nom du retrieval par propriété du Toolkit ; sa réexécution n’est pas requise pour compléter les quatre applications.

### 8.3 Jugements de pertinence

Prendre l’union des top 10 des moteurs obligatoires pour chaque requête CONFIRM ; annoter cette union en aveugle avec ordre aléatoire. Les 12 requêtes × 4–5 méthodes × 10 résultats représentent au maximum 480–600 couples avant déduplication. Tous les top 10 de chaque moteur doivent être jugés. Ajouter si possible les rangs 11–20 et 3–5 négatifs aléatoires par requête, pour ne pas limiter l’audit au haut de liste.

Le jeu utilise une pertinence `0=non`, `1=partielle`, `2=claire`. Fixer avant mesure la règle binaire principale : seule `2` compte comme pertinente ; reporter aussi la version tolérante `{1,2}` en secondaire. Un texte ambigu reste `uncertain`, pas zéro automatique. Pour P@10, donner les bornes pessimiste/optimiste si des jugements demeurent manquants ; pas de score ponctuel principal avec top 10 incomplet.

Double annotation : au moins 60–100 couples, répartis entre requêtes et types d’erreurs. Un juge local peut annoter plus largement pour un diagnostic ; calibrer sa concordance avec ce sous-échantillon humain et garder les deux colonnes de résultats.

### 8.4 Métriques

**Primaire : P@10 humaine**, par requête, puis moyenne à poids égal par famille. **Secondaires :** nDCG@10 dans le pool jugé, nombre de documents pertinents propres à chaque méthode, recouvrement des top 10, couverture d’annotation, temps de recherche, coût d’indexation et coût marginal de requête.

Pour nDCG, préciser que l’IDCG du pool ne représente pas un idéal global lorsque le corpus n’est pas exhaustivement jugé. Pour recall/AP/MAP sur l’ensemble du corpus, il faut connaître le total des pertinents. Deux possibilités honnêtes : annoter exhaustivement un petit sous-corpus fixé avant les résultats, ou ne pas publier ces métriques globales. Ne jamais normaliser l’AP par les seuls pertinents trouvés par chaque méthode et comparer cela comme un rappel global.

Une expérience complète sur un sous-corpus de 150–200 emails et quatre propriétés peut fournir une MAP correcte, mais représente 600–800 jugements supplémentaires ; elle est optionnelle et non nécessaire à la conclusion principale P@10.

### 8.5 Analyse des résultats et conditions de conclusion

Comparer FULL−CORE, FULL−DENSE et FULL−TFIDF sur les mêmes familles. IC bootstrap appariés par familles ; avec six familles seulement, présenter la variabilité et les cas individuels, sans qualifier le benchmark de statistiquement définitif. Si plusieurs contrastes sont testés, correction annoncée et pas de sélection de la meilleure métrique après coup.

Classifier les erreurs : absence de feature pertinente ; label incorrect ; négation ; confusion demande/description ; erreur de lieu/temps ; fin d’email tronquée ; longueur ; vocabulaire trop spécifique. Si un gain ne concerne qu’une seule famille, dire lequel. Une moyenne élevée ne suffit pas à promouvoir un usage critique.

**Attente de travail, non prédiction :** les méthodes seront probablement complémentaires. Le résultat acceptable peut être la conservation d’un moteur dense/lexical pour le classement et du SAE pour expliquer les thèmes. Une liste de cas où FULL retrouve des emails utiles manqués par CORE est plus informative qu’un taux odd-one-out supplémentaire, mais elle doit venir d’un test fixé et présenter aussi les échecs.

### Streamlit, fichiers, temps et historique

Créer une page **Recherche par propriété** : requête, définition, méthode, top 10 côte à côte, score non probabiliste, features contributrices et extraits, statut de jugement et chronomètre. Calcul lourd pré-calculé ; pas de chargement de Qwen à chaque clic. Export `query_id,doc_id,rank,score,method,judgment_version` et récapitulatif agrégé.

Fichiers : `queries.jsonl`, `rankings.parquet`, `qrels_human.parquet`, `qrels_model.parquet`, `metrics_retrieval.json`, `retrieval_error_audit.md`. Coût : 2–6 GPU-h pour sélection/annotations modèle optionnelles, CPU faible une fois les codes prêts, annotation humaine à budgéter séparément.

Les quatre anciennes requêtes Latent Terms restent un résultat exploratoire distinct. Dans le dashboard, remplacer l’intitulé ambigu `MAP` par `AP locale sur liste tronquée (historique)` pour l’ancien calcul. Ne pas supprimer les résultats ; afficher la nouvelle P@10 du protocole cible comme référence **pour ce protocole seulement**.

---

## 9. E04 — Diffing dans le domaine cible

### Question industrielle

Quelles propriétés distinguent deux ensembles d’emails **au-delà de la différence qui a servi à les constituer**, et lesquelles se maintiennent sur d’autres parents ? Le test énergie/sport vérifiait un mécanisme sur un proxy très différent ; il ne répondait pas à cette question.

### 9.1 Choix des deux populations

**Priorité si métadonnées réelles autorisées :** même région/canal pendant une période cible et une période témoin, en contrôlant composition, longueur et volumes. Définir les fenêtres avec l’encadrement, avant inspection des features ; ne pas chercher la fenêtre qui maximise un écart.

**Repli avec les sources synthétiques disponibles :** un seul contraste cible principal, défini par une annotation indépendante ou un champ de génération connu, par exemple emails décrivant des incidents techniques vs emails portant sur la facturation. Le résultat attendu « parle d’incidents » est alors une vérification positive, pas une découverte. Chercher secondairement relances, formulation collective, conséquences rapportées, demande d’intervention ou incertitude ; leur nouveauté doit être jugée relativement aux catégories déjà connues.

**Contrôle apparié utile :** variantes de ton d’un même email parent, distribuées en deux populations comparables. Elles servent uniquement à vérifier une différence contrôlée et la sensibilité au style. Elles ne sont pas une étude d’incidents réels ; leur statistique doit être appariée par parent, pas Fisher sur deux échantillons supposés indépendants.

Minimum : deux groupes de **100 parents de confirmation** ; cible **150–200 par groupe** si leur répartition le permet. Les 869 parents CONFIRM ne garantissent pas que chaque thème fournisse 200 exemples. Si une classe est rare, réduire l’ambition et indiquer la puissance ; ne pas gonfler n en ajoutant les variantes du même parent.

### 9.2 Découverte et confirmation séparées

1. Définir A/B et leurs critères avec les métadonnées ou annotations indépendantes.
2. Utiliser FIT/DEV pour calculer les différences d’activation et proposer les hypothèses.
3. Geler **avant CONFIRM** chaque propriété atomique, sa direction attendue (`A>B` ou `B>A`), la méthode qui la propose, les features et leur provenance.
4. Vérifier la présence de la propriété sur CONFIRM sans montrer au juge le nom du groupe ni la méthode d’origine.
5. Calculer l’écart de fréquence, son IC, son sens et les tests. Faire auditer les résultats par un humain sans changer la liste candidate.

Une hypothèse doit être une propriété vérifiable dans **un** email : « l’auteur évoque au moins une relance antérieure ». La proposition comparative « le groupe A relance davantage que B » est stockée comme direction associée, pas donnée telle quelle au juge d’un document individuel.

### 9.3 Méthodes et budgets comparables

Comparer au minimum **CORE**, **FULL** et une baseline **TFIDF contrastif** ; une baseline **LLM seul sur échantillons équilibrés de documents** est souhaitable si le budget le permet, car elle teste la valeur ajoutée d’un catalogue SAE.

Par méthode : au plus **quatre hypothèses principales**, quota égal, génération sur la même population de découverte et exemples de même longueur. Le LLM de formulation est le même pour les bras SAE/lexical ; pas de labels métier inventés pour compenser un catalogue pauvre. Pour LLM seul, des échantillons équilibrés sont résumés sans features, avec budget de textes/tokens annoncé.

Après génération, dédupliquer les propriétés sémantiquement identiques de façon aveugle à leur score et conserver l’ensemble de leurs méthodes d’origine. Ne pas fusionner après confirmation pour éliminer les mauvaises hypothèses. Avec quatre méthodes, au plus seize hypothèses uniques ; à 400 emails de confirmation, cela représente jusqu’à 6 400 couples. Calibrer le coût du juge sur 100 couples avant soumission complète.

Le protocole modèle peut utiliser une sortie JSON courte `present/absent/uncertain + evidence_span` au lieu de 3–5 phrases de raisonnement. Il s’agit alors d’une adaptation explicite, pas d’une reproduction exacte du prompt publié. Tester le parseur et auditer un échantillon humain avant utilisation.

### 9.4 Statistiques et taille d’effet

Au niveau feature, utiliser les fréquences documentaires avec définition fixée : au moins une activation au-dessus du seuil opérationnel du code. Déclarer l’écart avec les variantes du papier qui utilisent un nombre minimal de tokens activants ; le passage à deux tokens ne doit pas être testé puis retenu parce qu’il améliore les scores. Les fréquences et tests initiaux servent à **proposer**, non à confirmer les hypothèses textuelles.

Au niveau hypothèse, calculer :

`delta_h = p_A(property_h) − p_B(property_h)`.

Reporter supports, IC, sens attendu/réel, différence en **points**, et `p_adjusted` sur la famille de toutes les hypothèses gelées du contraste. Pour groupes indépendants : test exact ou test de proportion adapté ; IC Newcombe/Wilson ou bootstrap parents. Pour variantes appariées : différences par parent et test apparié/permutation de signes. Pour cohortes avec plusieurs emails par client, regrouper à ce niveau si un identifiant autorisé existe.

**Seuil d’utilité proposé, à décider à J2 :** différence d’au moins 5 points, avec sens correct et incertitude affichée. Une conclusion statistique et une utilité métier sont deux colonnes distinctes. Avec n=200 par groupe et proportions proches de 0,5, l’erreur-type d’une différence est environ 5 points ; détecter 5 points est donc irréaliste avec bonne puissance. Même 10 points peut être incertain, davantage après correction multiple. Des écarts plus grands ou davantage de données seront nécessaires. Ne pas déclarer le projet en échec parce que les petits écarts restent indécidables.

Ajouter : (a) partition A/B aléatoire de découverte puis confirmation ; (b) inversion A/B donnant le signe opposé ; (c) analyse appariée/stratifiée par longueur si différence de composition. Contrôler l’effet du nombre d’emails : les fréquences ont un dénominateur, les nombres bruts ne sont pas comparables directement.

### 9.5 Ce que signifie « hypothèse utile »

Distinguer quatre niveaux :

- `candidate` : formulée à partir de la découverte ;
- `replicated_statistically` : écart de bon sens et test/IC compatibles avec le protocole ;
- `human_supported` : présence de la propriété jugée correctement sur un audit ;
- `business_useful` : piste jugée pertinente, non triviale et suffisamment étayée par l’analyste.

Une hypothèse peut être statistiquement correcte mais triviale. « Facture » plus présent dans une population sélectionnée par facturation ne constitue pas une découverte. Ne pas utiliser un seuil d’un seul document, ni `abs(delta)>1%`, comme critère de succès industriel.

**Résultats attendus :** un petit catalogue de propriétés, certaines retrouvées par plusieurs méthodes, d’autres non répliquées. L’analyse de ces recouvrements est centrale. Si les mêmes thèmes sont trouvés par TFIDF ou LLM seul à moindre coût, conserver ce résultat. Si FULL apporte des pistes nouvelles validées, les présenter avec contre-exemples et intervalle, sans annoncer un détecteur d’incidents déployable.

### Streamlit, temps et mise à jour du dépôt

Page **Comparer deux populations** : provenance/tailles A et B, filtre temps/région seulement si disponible, séparation découverte/confirmation, cartes d’hypothèses avec signe, support, effet, IC, correction multiple, statut humain et méthode d’origine. Clic sur une propriété → exemples A/B et contre-exemples, sans écraser le contexte de filtre. Exporter le catalogue gelé et les résultats, pas seulement les hypothèses retenues.

Fichiers : `cohorts_manifest.json`, `hypotheses_frozen.jsonl`, `verification_matrix.parquet`, `hypothesis_tests.json`, `diffing_catalogue.md`. Temps : 4–12 GPU-h pour un contraste principal selon débit et longueur ; 1–3 CPU-h ; audit humain 1–3 h. Deux autres contrastes sont optionnels et exploratoires, pas requis pour livrer le premier correctement.

Conserver le `8/10` historique sous `proxy_energy_sports_descriptive`. Créer une nouvelle référence `email_diffing_confirmatory_v1`. Un résultat sur emails ne rend pas mécaniquement faux le proxy ; il remplace sa position de preuve principale du cas d’usage. Le parseur d’erreurs corrigé peut nécessiter une requalification de caches historiques **si** leur audit démontre des sorties mal formées.

---
## 10. E05 — Groupes de features, stabilité et carte exploratoire

### Question et niveaux de réponse

Un même thème peut-il être représenté par plusieurs features, différemment réparties entre entraînements, tout en restant exploitable pour retrouver les mêmes emails ? Cette question est plus pertinente que l’égalité des libellés exacts. Elle comporte cependant trois livrables de difficulté différente :

1. **Carte descriptive d’un dictionnaire**, faisable même sans répétition : voisinages, exemples, fréquences et provenance.
2. **Étude exploratoire de stabilité sur trois entraînements**, faisable avec un 1B et des extractions mutualisées.
3. **Hiérarchie sémantique ou sous-espaces reproductibles confirmés**, qui peut exiger davantage de données et de graines et relève en partie du successeur.

La littérature donne des précédents pour les familles, le splitting et l’étude de sous-espaces. Notre protocole ci-dessous est une **adaptation proposée au projet**, pas une reproduction de ses résultats, et n’utilise pas les labels comme unique critère d’appariement. [L2–L4]

### 10.1 Ne pas mesurer la stabilité triviale du cœur

Les poids du cœur gelé sont identiques entre graines. Un appariement parfait de ce bloc ne démontre aucunement la reproductibilité d’un apprentissage. Le calcul principal de stabilité porte sur **EXTRA seulement**, dans le même espace 1B/couche13 ; le cœur peut être affiché comme repère immuable, dans une couleur et une table séparées.

Ne pas comparer directement les cosinus des directions 1B et 12B : leurs dimensions et bases sont différentes. Une comparaison inter-modèles nécessiterait un alignement d’espaces ou des profils fonctionnels communs, travail de reprise distinct.

### 10.2 Données nécessaires

Trois checkpoints d’extension, mêmes données FIT, même architecture, mêmes hyperparamètres, trois états de hasard documentés. Collecter les poids du décodeur et les activations sur un même ensemble de documents DEV/CONFIRM. Pour l’analyse token-level, utiliser jusqu’à 50k–100k tokens valides partagés, avec leurs identifiants stables ; pas de tirage différent pour chaque SAE.

Stocker les activations extra `N×1024` par chunks/CSR ; aucune matrice de toutes les paires de tokens n’est nécessaire. Les directions sont normalisées L2 selon la convention réelle (Wdec parfois transposé). Exclure des calculs de corrélation les features constantes ou sans support ; les garder visibles comme « support insuffisant ».

### 10.3 Appariement individuel de référence

Calculer pour chaque paire de graines :

- cosinus signé des directions du décodeur ; ne pas prendre la valeur absolue, car les coefficients SAE sont non négatifs ;
- similarité/corrélation des profils d’activation sur les mêmes observations ;
- recouvrement des emails les plus activants ;
- fréquence d’activation et norme, pour identifier les correspondances artificielles.

Utiliser un appariement nearest-neighbor avec possibilité de plusieurs-à-un et un seuil déclaré, par exemple cosinus ≥0,7 comme repère de littérature, **accompagné** d’un contrôle de profils et de support. Une variante hongroise avec nœuds non appariés peut être un diagnostic, pas un mécanisme qui force chaque feature à avoir une jumelle.

Afficher les distributions, la couverture et les cas non appariés. Avec trois graines, chaque feature n’a que deux autres runs à comparer ; un taux 2/2 ne justifie pas la phrase « reproductible à 100 % ». Reporter « retrouvée dans les deux répétitions disponibles ». Ne pas bootstrapper 1 024 features pour simuler 1 024 entraînements indépendants.

### 10.4 Construction des groupes sur DEV

Commencer par un graphe simple réutilisant les bibliothèques déjà présentes : nœuds = features extra ; arêtes = voisinages géométriques et/ou profils d’activation proches. Enregistrer séparément ces deux poids plutôt qu’un score opaque. Une règle d’entrée raisonnable : voisinage mutuel parmi les 10 plus proches, support de fréquence suffisant, seuil calibré sur DEV et un témoin de permutation. La règle est figée avant CONFIRM.

Regrouper avec l’algorithme déjà disponible (Louvain ou clustering agglomératif), une seule configuration de développement. Ne pas lancer cinquante résolutions pour choisir la meilleure séparation visuelle. Limiter les très grands groupes et garder les isolats : une seule communauté de 900 features ne constitue pas une carte thématique informative.

Pour une première carte lisible, afficher les 150–300 features soutenues et utiles aux applications ; garder les métriques de couverture du dictionnaire complet. Le filtre visuel ne doit pas devenir silencieusement la population du test de stabilité.

Comparer les groupes entre graines sur la base des correspondances individuelles et des profils. Autoriser splits, fusions et absence de correspondance. Les noms des groupes sont proposés après leur construction, puis validés sur exemples ; ne pas grouper uniquement les labels « facture » pour conclure ensuite que le thème facture est stable.

### 10.5 Comparaison au niveau des sous-espaces

Pour un groupe G, former la matrice D_G des directions ; calculer une SVD. Reporter son spectre et son rang effectif. Construire une base orthonormée Q_G de rang r déterminé par une règle fixée sur DEV, avec plafond raisonnable pour les petits groupes (par exemple 16, réduit si le groupe est plus petit).

Pour deux groupes appariés de même rang, la similarité de sous-espace peut être :

`overlap = || Q_A^T Q_B ||_F² / r`.

Reporter aussi les cosinus des angles principaux. Si les rangs diffèrent, montrer les deux couvertures directionnelles avec leurs dénominateurs et ne pas cacher le rang dans un scalaire. Les directions presque de rang plein ont mécaniquement un fort recouvrement : un sous-espace de rang d comparé à un autre vaut 1 sans découverte sémantique. Comparer à des témoins **appariés en taille, rang et fréquence**, dans la géométrie anisotrope réelle ; la référence isotrope r/d n’est qu’une intuition, pas le seul null approprié.

Ne pas adopter une sélection « les features les moins stables » puis conclure que leur sous-espace global est stable sans contrôler sa dimension. L’extension de 1 024 directions dans un espace de 1 152 dimensions peut couvrir une grande fraction de l’espace ambiant ; un overlap élevé de tout le dictionnaire serait peu informatif.

### 10.6 Témoins obligatoires

- **Groupes aléatoires de mêmes tailles**, construits dans chaque dictionnaire en conservant autant que possible des classes de fréquence et de norme. Au moins 100 tirages, 200 si peu coûteux.
- **Null fonctionnel** : permutation indépendante des profils d’activation par feature, idéalement dans des strates de longueur et par blocs parents. Une permutation commune des lignes ne détruit pas les corrélations entre colonnes ; elle n’est donc pas le contrôle souhaité.
- **Pipeline de sélection identique** : le meilleur appariement de groupes réels doit être comparé au meilleur appariement de groupes aléatoires avec le même budget. Sinon la maximisation favorise uniquement le bras réel.
- **Stabilité utile** : recouvrement des top emails et des jugements de thème sur CONFIRM, pas seulement proximité de centroïdes.

Calculer des effets par groupe et par paire de graines. Les répétitions de partitions nulles ne créent pas de nouvelles graines entraînées. Les p-values de randomisation répondent au null spécifié, pas à la généralisation à tous les entraînements possibles.

### 10.7 Score de groupe et lien aux applications

Pour un groupe, un score documentaire simple est le maximum des activations de ses membres normalisées par leur p90 FIT. Mais les grands groupes ont mécaniquement plus de chances de s’activer. Comparer à des groupes de même taille, publier le nombre de membres et tester une moyenne des membres actifs comme diagnostic secondaire fixé d’avance. Aucune renormalisation apprise sur CONFIRM.

Évaluer 6–10 thèmes/groupes liés à E03 : retrouve-t-on des emails pertinents similaires entre graines, malgré des identifiants différents ? Comparer Jaccard@20, P@10 et couverture des parents. Si les top emails restent instables, une carte géométrique séduisante ne résout pas le problème métier.

### 10.8 Contenu de la carte Streamlit

Carte 2D pour naviguer, avec coordonnées pré-calculées (UMAP ou PCA), jamais comme preuve statistique. Nœud : feature, branche, fréquence, support humain, seed ; arête : type et score de relation ; groupe : exemples, contre-exemples, couverture, stabilité observée et intervalle/limite. Afficher un filtre pour ne pas saturer le navigateur avec toutes les arêtes.

La carte doit permettre de chercher un label, cliquer une feature, voir sa direction/son profil et les emails correspondants, changer de graine et afficher les correspondances. Les arêtes inter-graines et les arêtes intra-dictionnaire sont différentes. Un lien « est une sous-catégorie de » n’apparaît que si une règle et des exemples soutiennent cette inclusion ; sinon afficher « voisin », « coactif » ou « correspondance candidate ».

L’exemple monnaie→euro/dollar/yen reste un schéma conceptuel séparé tant qu’il n’a pas été retrouvé et validé. Pour les emails, les groupes réellement observés porteront leurs noms réels, même s’ils sont syntaxiques ou peu spectaculaires.

### Conclusions possibles, budget et historique

- Groupes réels > null **et** emails retrouvés stables : piste de robustesse soutenue localement, à répliquer.
- Géométrie stable mais retrieval de thèmes variable : sous-espace partagé possible, utilité documentaire non démontrée.
- Groupes réels ≈ null : pas de preuve que le regroupement améliore la stabilité ; livrer la carte descriptive avec ce résultat.
- Trois graines indisponibles : aucune conclusion inter-entraînements ; carte et protocole prêts pour reprise.

Fichiers : `feature_nodes.parquet`, `feature_edges.parquet`, `feature_matches.parquet`, `groups.json`, `group_null_results.parquet`, `map_positions.parquet`, `stability_report.md`. 8–12 h d’ingénierie ; 4–14 GPU-h pour les deux entraînements additionnels/réencodages si nécessaires, 2–6 CPU-h pour la géométrie. Réduire d’abord les arêtes/labels affichés, pas les contrôles nulls.

Le précédent Jaccard de labels reste valide dans son périmètre. L’ancienne conclusion sur Louvain n’est remplacée que pour les nouveaux poids/données et nouvelle métrique ; créer un lien, sans effacer la conclusion négative historique. Mettre les fonctions legacy de groupes de labels dans une section d’analyse historique, séparée du nouveau module d’alignement.

---

## 11. E06 — Corrélations entre propriétés, confirmées hors découverte

### Pourquoi

La seconde application du Toolkit est la recherche d’associations inattendues entre propriétés. Une NPMI proche de 1 obtenue sur un ou deux documents ne permet pas une conclusion exploitable. Il faut des supports suffisants, une vérification indépendante des propriétés et une séparation entre génération de candidats et confirmation.

### Protocole minimal réalisable

1. Sur FIT/DEV, prendre un catalogue documenté de 200–500 features pertinentes des bras CORE et FULL. Ne pas former une matrice dense du dictionnaire complet de plusieurs centaines de milliers de features.
2. Binariser selon la règle d’activation fixée. Calculer supports individuels, supports conjoints et NPMI. Écarter de la sélection principale les paires dont le support conjoint de découverte est inférieur à 10 parents ; montrer les autres comme « rares, descriptives ».
3. Retirer les quasi-synonymes, les doublons évidents et les paires de même information de format. La similarité de labels aide à trier, elle ne prouve pas que les propriétés sont indépendantes. Un percentile relatif de dissimilarité n’est pas un seuil universel de nouveauté.
4. Geler au plus **huit paires** avec leurs définitions atomiques. Budget égal de candidats CORE/FULL ; baseline de cooccurrences lexicales sur FIT/DEV souhaitable, avec la même vérification humaine. Ne pas sélectionner les huit plus belles corrélations sur CONFIRM.
5. Vérifier les propriétés atomiques sur au moins 300–400 parents CONFIRM, mutualiser les propriétés déjà évaluées dans E04, puis calculer leur cooccurrence **sur les jugements**.

### Métriques et contrôles

Pour chaque paire A,B : n, n_A, n_B, n_AB, proportions, différence à l’indépendance `p_AB−p_A p_B`, odds ratio avec IC si identifiable, NPMI et IC bootstrap parents. Tester l’association sur la table 2×2, avec correction multiple sur les huit paires gelées. Si la table est dégénérée ou contient trop peu de positifs, statut `insufficient_support`, pas « corrélation parfaite ».

`NPMI = log(p_AB/(p_A p_B)) / (−log(p_AB))` lorsque 0<p_AB<1. Définir explicitement les cas p_AB=0 ou 1 et les valeurs manquantes. Ne pas ajouter un pseudo-compte puis présenter la perfection obtenue comme un support empirique. Une statistique très élevée ne dispense pas de publier les quatre cellules de la table.

Contrôler longueur, axe de génération, source et éventuellement canal par stratification ou modèle logistique simple si les effectifs le permettent. Une corrélation peut refléter un template de génération commun ou une variable cachée. Avec des données synthétiques, conclure sur la distribution synthétique, pas sur le comportement des clients EDF.

Audit humain de 40–80 emails couvrant cas (A=1,B=1), (1,0), (0,1) et (0,0), en échantillonnage stratifié. L’estimation populationnelle humaine exige les poids de cet échantillonnage ; sans eux, publier seulement la concordance dans l’audit. Les annotations modèle à grande échelle restent un proxy. Le contrôle de confusion n’établit pas la causalité.

### Résultats attendus et décision

On peut obtenir zéro paire suffisamment étayée. Cela signifie que le corpus, le catalogue ou l’annotation ne permet pas encore d’établir des associations intéressantes, non qu’il faut abaisser les seuils jusqu’à trouver un résultat. Une paire exploitable doit être fréquente assez souvent, non triviale et lisible via des exemples. Un résultat du type « relances antérieures et insatisfaction » peut être correct mais peu nouveau ; l’analyste doit qualifier l’intérêt.

### Streamlit, temps et statut des anciens résultats

Page **Associations de propriétés** : scatter support/effet, tableau 2×2, NPMI, odds ratio, correction, provenance de jugement et filtres de support. Cliquer une paire montre les quatre catégories d’emails, pas seulement les cooccurrences positives. Export des candidats rejetés inclus.

Fichiers : `correlation_candidates_frozen.json`, `property_presence.parquet`, `correlation_confirmation.json`, `correlation_audit.md`. Coût proposé 2–6 GPU-h, à ajuster selon le nombre de propriétés uniques, 1–3 CPU-h ; 1–2 h humaines. Si les supports attendus sont trop bas au préflight DEV, réduire le nombre de paires et reporter les concepts rares à un corpus plus grand.

Les anciens résultats NPMI_verified à support 1–2 restent `historical_low_support`. Ils ne doivent plus apparaître comme « associations validées » dans le dashboard. Ne pas déclarer caduque un poids SAE parce qu’une association statistique ne réplique pas.

---

## 12. E07 — Clustering ciblé réellement comparé aux alternatives

### Pourquoi

Un utilisateur peut vouloir regrouper les mêmes emails selon le type de problème, l’attitude du client ou l’action demandée. La question est **le contrôle de l’axe de regroupement**, non la seule compacité de clusters en 2D.

### 12.1 Corrections avant tout run

- Corriger le mapping labels/identifiants : construire `items = sorted(feature_labels.items())`, puis encoder `[label for fid,label in items]` et associer dans le même ordre. Test avec un JSON volontairement non trié.
- Séparer le helper d’embedding de labels courts du helper d’emails. Supprimer la troncature implicite à 64 tokens des emails de référence ; conserver un paramètre de longueur explicite et la couverture textuelle.
- Remplacer le repli « moins de cinq features → tout le dictionnaire » par un statut d’échec de sélection. Sinon le résultat ne teste plus le clustering ciblé annoncé.
- Traiter les emails au vecteur restreint nul comme « hors axe / sans signal », pas comme un groupe thématique parfaitement cohérent dû à Jaccard(0,0).
- Vérifier l’affinité symétrique, finie, bornée, le traitement diagonal, la taille des groupes et le nombre de documents effectivement classés.

### 12.2 Protocole cible

Fixer **trois axes** sur DEV : type de problème ; action attendue ; registre/urgence. Valider qu’ils correspondent aux données ; les propriétés qui se recouvrent seront annotées comme telles. Choisir 300–500 emails source CONFIRM, pas 300 variantes du train. Fixer quatre clusters comme budget de lecture initial, ou un autre nombre choisi sur DEV et égal pour les méthodes ; l’important est de ne pas choisir k après lecture de la meilleure accuracy sur CONFIRM.

Pour chaque axe, comparer :

- CORE restreint aux features sélectionnées ;
- FULL restreint selon le même budget de labels ;
- DENSE bge-m3 sur les mêmes emails, avec une méthode adaptée aux vecteurs denses ;
- TFIDF avec un clustering simple, si le temps CPU permet cette baseline.

Le spectral sur affinité Jaccard est cohérent pour les activations binaires SAE. Pour DENSE/TFIDF, une normalisation L2 suivie de KMeans ou d’une affinité cosinus est plus naturelle ; documenter la différence d’algorithme. Un contrôle cosinus/spectral au même k peut aider à séparer représentation et partitionnement, mais ne doit pas devenir un nouveau sweep. Le **comparateur pratique** reste le pipeline dense réellement utilisé, pas une baseline volontairement mal adaptée.

Le clustering des documents CONFIRM est une analyse non supervisée **transductive de cette collection**, bien que le SAE et le choix de features n’aient pas appris sur elle. Ne pas le présenter comme la prédiction inductive de clusters définis sur FIT.

### 12.3 Évaluer sans boucle auto-validante

Pour chaque cluster, réserver quelques documents représentatifs à sa description ; la mesure d’adéquation de cette description utilise d’autres documents. Le même LLM qui nomme et réassigne mesure une cohérence interne du protocole, pas une vérité terrain. Conserver ce score pour comparaison historique, mais l’accompagner d’une évaluation humaine aveugle.

Préparer par axe : 20–30 paires de documents intra-cluster et 20–30 paires inter-clusters, méthodes masquées ; l’humain évalue « même propriété selon l’axe ? ». Échantillonner sans domination du plus grand groupe. Ajouter une notation de la pertinence du nom de cluster, du chevauchement des thèmes et des groupes sans signal. Si un axe possède une partition humaine de référence suffisamment complète, calculer ARI/AMI, en rappelant qu’une annotation multithématique ne devient pas artificiellement une classe unique.

Métriques : couverture des documents classés, tailles/minimum/maximum, cohérence humaine selon l’axe, séparation entre groupes, diversité des thèmes, réassignation LLM **secondaire**, silhouette dans l’espace utilisé et conductance dense **diagnostiques**. Une conductance négative en z-score ne prouve pas un thème nouveau ni une supériorité sur des embeddings denses.

### Résultats et décision

Attente : la qualité peut varier selon l’axe ; un seul axe positif ne justifie pas une conclusion universelle. Si le filtrage SAE permet des regroupements plus lisibles avec mêmes documents et même effort, livrer cet usage. Si DENSE fonctionne aussi bien, le garder comme moteur et présenter le SAE comme dispositif d’explication. Si la sélection retourne trop peu de signal, le résultat « axe non pris en charge » doit apparaître dans l’interface.

### Intégration, temps et nettoyage

Page **Regrouper selon une question** : choix d’axe, méthode, nombre de groupes, couverture, descriptions avec exemples réservés/évalués distingués, comparaison entre méthodes et liste des emails non assignés. La projection 2D est illustrative et séparée des métriques dans l’espace d’origine.

Fichiers : `clustering_protocol.json`, `assignments.parquet`, `cluster_descriptions.json`, `cluster_human_pairs.parquet`, `clustering_comparison.json`. 2–6 GPU-h de labels/évaluation, 1–3 CPU-h. Audit humain 1–3 h selon mutualisation. Si l’annotation est impossible, livrer la comparaison géométrique et la chaîne corrigée avec statut « non validé humainement ».

L’ancien cache R0 n’est invalidé **que si** l’audit des ordres d’identifiants prouve qu’il a utilisé un mapping erroné. Sinon, il reste historique à faible effectif. Dans les deux cas, les nouveaux résultats ne portent plus un titre de « benchmark d’embeddings » : ils évaluent un regroupement guidé par un axe.

---

## 13. E08 — Mini-pilote analyste et recette Streamlit

### Question

Les sorties aident-elles réellement quelqu’un à trouver, comprendre et documenter un thème ? Cette question ne se résout pas par un nouveau taux d’auto-interprétation. Le mini-pilote est le lien entre résultats techniques et décision EDF.

### Format réaliste

Proposer à deux ou trois personnes disponibles une ou deux sessions de 20 minutes, dans le cadre autorisé, avec deux tâches comparables : retrouver des emails selon une propriété ; comparer deux sous-populations pour proposer des thèmes. Une session utilise les méthodes de référence, l’autre le dispositif SAE, dans un ordre contrebalancé et sur des collections/questions suffisamment différentes pour réduire l’effet de mémorisation.

Avec si peu de participants, c’est une étude de faisabilité, pas une démonstration de gain de productivité. Grégoire peut réaliser la recette technique, mais son avis ne remplace pas une validation externe. En l’absence d’analyste métier, nommer précisément le panel (« utilisateurs de recherche », par exemple), sans qualifier l’évaluation de métier.

### Mesures

Nombre de pistes documentées, nombre jugé pertinent par une relecture aveugle, nouveauté par rapport aux catégories déjà connues, présence d’extraits justificatifs, temps jusqu’à une première piste exploitable, erreurs d’interface, confiance correctement calibrée. Reporter séparément temps humain, temps de calcul prépayé et temps de clic. Ne pas masquer un lourd coût de pré-calcul derrière une interface rapide.

Une piste est un objet structuré : titre, question de départ, populations, propriété, exemples, contre-exemples, méthode, statut et commentaire humain. Prévoir l’export d’un **catalogue commenté de thèmes**, même si certains thèmes sont rejetés.

### Recette minimale

Démarrage sans GPU et sans réseau externe ; filtres de corpus cohérents ; comparaison CORE/FULL ; liens d’exemples corrects ; messages pour données manquantes ; pas de score vert quand le protocole est invalide ; export reproductible ; données autorisées uniquement. Un utilisateur doit pouvoir comprendre la différence entre résultat synthétique, annotation modèle, audit humain et validation métier.

### Conclusion et suite

La décision peut être : poursuivre un pilote réel, garder uniquement une fonction d’exploration, privilégier la baseline, ou arrêter une branche. Documenter les conditions de reprise et les points bloquants. Aucune tâche de production ne doit être automatisée à partir de ce mini-pilote.

Temps : 5–7 h d’ingénierie/recette et 2–4 h de participation/relecture cumulées selon disponibilité. Pas de nouvel entraînement requis. La démonstration doit utiliser les artefacts figés E01–E07, pas exécuter un job LLM en direct pendant la séance.

---

## 14. E09 — Option Gemma-3-1B / 100M tokens

### Pourquoi ce n’est pas une priorité automatique

Le passage au 1B est justifié comme option de frugalité à tester. Il ne garantit pas que le volume supplémentaire soit utile. Le rapport mentionne déjà un bras 1B/50M à 55,3 %, contre 63,3 % à 25M sous le test final, comparaison exploratoire ; ce résultat ne prouve pas une dégradation générale avec le volume, mais interdit de supposer que 100M sera meilleur. [R, annexe B.1]

Le corpus emails n’offre pas 100M tokens de contenus clients indépendants. Un complément massif de FineWeb générique change la distribution. Il faut identifier **une question applicative motivée** : couverture de propriétés rares présentes dans le complément, stabilité fonctionnelle ou gain de retrieval. Si aucun manque précis n’est diagnostiqué, différer le run.

### Portes d’autorisation obligatoires

- E00 validé avec une mesure réelle de mémoire et de débit ; aucun stockage filler dense.
- Livrable M disponible, ou sa livraison garantie sans dépendre du run.
- Au moins trois jours de marge avant la clôture technique ; lancement avant J10.
- Stockage/quota approuvé : un seul x 1B/100M bf16 représente 230,4 Go ; ajouter marge, shards utiles, checkpoints et logs. Ne pas stocker simultanément x et toutes les reconstructions/résidus sans justification mesurée.
- Définition de 100M : tokens uniques sélectionnés, après masquage, avec composition des sources ; pas 10 passages sur 10M rebaptisés 100M de données.
- Une seule allocation GPU à la fois ; budget optionnel explicitement accepté, au plus 36 GPU-h proposé après profilage. Si l’estimation dépasse ce plafond, transférer au successeur.
- Aucun changement de protocole de test, aucun choix de meilleur checkpoint sur CONFIRM.

### Protocole recommandé

Même 1B/couche13/cœur16k/extension1024/K5 et même mini-batch que E01. Définir le corpus supplémentaire sans contact avec CONFIRM, sans quasi-duplicatas, avec une politique de mélange fixée. Conserver un échantillon de 25M **emboîté** dans celui de 100M lorsque ces volumes sont réellement disponibles et que les distributions sont comparables.

Deux lectures possibles, à choisir avant run :

**Comparaison à budget d’optimisation constant (préférée pour isoler la diversité).** Fixer le nombre de mini-batches/expositions de tokens de la référence, par exemple 250M présentations au total si la référence 25M a fait dix passages. Le modèle 100M réalise alors environ 2,5 passages effectifs, via un `max_steps` explicite et un ordonnanceur de taux d’apprentissage cohérent. Cela requiert un harnais par étapes, pas un arrondi silencieux d’époques.

**Comparaison d’investissements complets.** Dix époques sur 100M contre dix sur 25M compare à la fois volume et quantité de calcul. Cette expérience peut être pertinente industriellement, mais son effet ne peut pas être attribué au seul nombre de données. Reporter coût total et coût par résultat utile.

Si le jeu emails est plus petit ou si le complément change radicalement le domaine, les deux bras ne constituent pas une ablation de volume pure. L’appeler « ajout d’un corpus complémentaire » et décrire exactement la différence.

### Mesures et conclusions

Conserver les requêtes, cohortes, annotations, features de référence et évaluateurs d’E03/E04. Mesurer ΔFVE, nombre de features vivantes, couverture des propriétés, gain FULL−CORE, P@10, stabilité des emails retrouvés et coût. Un meilleur ΔFVE sans amélioration applicative ne justifie pas le surcoût pour EDF.

Si l’effet utile est absent dans l’incertitude, garder la configuration moins chère pour le prototype. Si un gain robuste concerne quelques propriétés, préciser lesquelles et livrer les deux checkpoints avec critères d’usage. Si le run est interrompu ou incomplet, conserver l’artefact avec statut approprié ; ne pas lui attribuer les attentes de sa configuration nominale.

### Estimation de temps à calculer, pas à deviner

Mesurer sur un bloc représentatif : `r_extract` en tokens valides/s, `r_train` en tokens présentés/s, `t_reencode`, `t_label`, volume I/O. Estimer :

`T_total ≈ N_extract/r_extract + N_presentations/r_train + T_reencode + T_eval + T_IO + marge`.

Utiliser une marge de 30–50 % selon la variabilité observée et séparer attente en file. Mesurer avec les vraies longueurs et les vrais sources/fichiers ; un essai sur phrases courtes ne prédit pas un corpus long. L’enveloppe initiale 12–36 GPU-h est une **hypothèse de planification**, pas un résultat acquis. Le seuil de mémoire et l’enveloppe de temps priment sur la cible 100M.

### Streamlit et historique

Afficher le volume **réalisé**, la composition et les expositions, puis la comparaison sur tâches figées. Ne pas remplacer automatiquement le checkpoint par le plus récent. Un modèle 100M non meilleur reste une option archivée, non le nouveau défaut. Les anciens 25M/50M ne deviennent pas caducs par augmentation du volume ; ils restent des points historiques avec leurs protocoles.

---
## 15. Statistiques, métriques et interprétation transversales

### 15.1 Table de décision avant la confirmation

Créer `protocol_frozen.yaml` avec, par expérience : population, unité indépendante, exclusion, méthodes, budget de candidats, métrique primaire, sens de l’effet attendu, marge utile, comparaisons, famille de tests, règle en cas de données insuffisantes. Stocker son hash dans chaque résultat. Toute modification ultérieure déclenche une nouvelle version et marque l’analyse exploratoire.

| Expérience | Primaire | Unité d’incertitude | Ce qu’il ne faut pas faire |
|---|---|---|---|
| E01 | FULL−CORE sur tâche fixée ; sondes secondaires | Email parent ; graine explicitement séparée | Confondre plis CV, features et répétitions indépendantes |
| E03 | Différence de P@10 humaine | Famille de requêtes et couples appariés | Compter 10 paraphrases comme 10 nouveaux concepts |
| E04 | Écart signé de fréquence sur confirmation | Parent ou client autorisé | Utiliser le seuil absolu de 1 point comme significativité |
| E05 | Stabilité des résultats par groupe vs null | Paire de graines + parents pour l’aspect fonctionnel | Gonfler le n par le nombre de partitions nulles |
| E06 | Association confirmée, supports et taille d’effet | Parent | Prétendre qu’une NPMI parfaite sur deux cas est robuste |
| E07 | Cohérence humaine sur un axe, couverture | Paires/documents et axes | Confondre conductance et vérité sémantique |
| E08 | Pistes utiles étayées et retour utilisateur | Session/participant | Inférer une hausse de productivité générale sur deux personnes |

Les tests répondent à des questions définies, pas à la consigne « trouver p<0,05 ». Si le budget humain permet seulement des estimations descriptives, le dire. Les intervalles de Wilson binomiaux ne corrigent ni l’échantillonnage stratifié à poids inégaux ni la dépendance des variantes.

### 15.2 Métriques à implémenter sans ambiguïté

- `precision_at_k(qrels, ranking, k)` : k fixé et positif, rangs sans doublons, top k entièrement jugé ou bornes explicites ; dénominateur k, pas nombre de jugements disponibles.
- `average_precision_full(relevance, total_relevant)` : total connu et documenté ; vérification du périmètre corpus. Un dénominateur inconnu provoque un statut non calculable, pas un remplacement par les seuls hits.
- `ap_at_k_global_denominator` : somme jusqu’à k, divisée par le total pertinent du corpus ; nom et convention explicites. Ne pas confondre avec une autre convention divisant par min(R,k).
- `ndcg_at_k` : gains, discount, pool de jugement et IDCG explicités. `judgment_scope=pooled` interdit le label « exhaustive ».
- `proportion_difference` : nombre de succès et de parents dans chaque groupe, signe, IC et type de plan.
- `npmi_with_support` : supports et cas dégénérés retournés avec la valeur.
- `subspace_overlap` : rangs, règle de sélection du rang et résultats des témoins retournés, pas seulement un nombre.
- `rho_reconstruction` : si repris, nommer la géométrie comparée exactement ; le helper historique `compute_rho_sae` est décrit dans le dossier oral comme comparant entrée/reconstruction, pas automatiquement les codes documentaires.

Si une requête n’a aucun pertinent dans le corpus évalué, la traiter comme tâche d’abstention avec convention explicite. Publier le nombre de telles requêtes ; ne pas les exclure silencieusement pour relever la moyenne. Un score de similarité ou de Boltzmann n’est pas une probabilité calibrée de pertinence.

### 15.3 « Résultats attendus » à utiliser par l’agent

Chaque expérience doit distinguer :

1. **Attendu technique** : format lisible, cache complet, mémoire bornée, split sans fuite. Un échec exige correction.
2. **Hypothèse scientifique** : FULL pourrait mieux représenter une propriété, les groupes pourraient être plus stables. L’inverse reste un résultat valable.
3. **Critère industriel** : un gain suffisamment grand, vérifiable et utile pour justifier le coût. Il est défini avec les humains, pas choisi en fonction du résultat.

Modèle de conclusion obligatoire :

> Sur [population], avec [effectif/unité] et [protocole], [méthode] présente [effet + intervalle]. Ce résultat soutient [affirmation limitée]. Il ne démontre pas [affirmation plus générale]. [Action recommandée], compte tenu de [coût, risque, incertitude].

Ne jamais écrire « devrait être significatif », « les clusters sont bons car compacts », « le cœur ne sert à rien » ou « le 1B est équivalent » sans une preuve adaptée. Une sélection de réussites qualitatives est une illustration ; le résultat quantitatif couvre toutes les requêtes/hypothèses gelées.

### 15.4 Séparation des budgets de tests

E03 peut avoir trois contrastes primaires ; E04 une famille de propriétés sur son contraste principal ; E06 huit paires ; E01 les contrastes de sondes annoncés. Justifier ces familles indépendantes avant analyse. Une « BH finale » construite après exploration de centaines de variantes ne rend pas la campagne confirmatoire. Conserver un journal des analyses tentées et des rejets.

---

## 16. Contrats logiciels, Slurm, checkpoints et exécution autonome

### 16.1 Organisation proposée, à créer

Ne pas réécrire tout le monolithe pendant les quinze jours. Extraire les interfaces nécessaires, avec adaptateurs de compatibilité et tests ; laisser les anciens runs relisibles.

```text
configs/post_stage/
  campaign_policy.yaml
  corpus.yaml
  representations.yaml
  protocol_frozen.yaml
  queries_dev.jsonl
  queries_confirm.jsonl
src/post_stage/                 # NOUVEAU : à implémenter, pas présent au départ
  cli.py
  manifests.py
  resources.py
  dataset_contract.py
  block_reader.py
  representations.py
  features.py
  retrieval.py
  diffing.py
  stability.py
  correlations.py
  clustering.py
  evaluation.py
  reporting.py
scripts/post_stage/
  submit_campaign.py
  inspect_campaign.py
slurm/post_stage/
  00_profile.slurm
  01_extract.slurm
  02_train.slurm
  03_encode.slurm
  04_label.slurm
  05_apps.slurm
  06_report.slurm
tests/post_stage/
  ...
docs/post_stage/
  preflight_report.md
  protocol.md
  decisions.md
  resources_report.md
  results_summary.md
  handoff.md
```

Adapter ces noms si un module existant remplit le même rôle. Ne pas créer un dossier `src/post_stage` parallèle qui duplique silencieusement toutes les fonctions de production. Les nouvelles commandes doivent appeler la loss et l’encodeur existants, sauf changement explicite motivé et testé.

### 16.2 Commandes attendues

Les commandes suivantes sont un **contrat à développer**, pas des commandes prêtes à lancer contre le dépôt non modifié :

```bash
python -m src.post_stage.cli preflight --campaign configs/post_stage/campaign_policy.yaml
python -m src.post_stage.cli freeze-corpus --config configs/post_stage/corpus.yaml
python -m src.post_stage.cli profile --campaign ... --max-valid-tokens 250000
python -m src.post_stage.cli extract --split fit --manifest ...
python -m src.post_stage.cli train --config ... --seed 42 --resume-if-valid
python -m src.post_stage.cli encode --splits dev,confirm --checkpoint ...
python -m src.post_stage.cli label --split fit --config ...
python -m src.post_stage.cli retrieval --protocol ... --phase discovery
python -m src.post_stage.cli diffing --protocol ... --phase discovery
python -m src.post_stage.cli freeze-hypotheses --campaign ...
python -m src.post_stage.cli evaluate --experiment E03 --phase confirm --protocol ...
python -m src.post_stage.cli stability --checkpoints ... --protocol ...
python -m src.post_stage.cli report --campaign ... --no-raw-text
```

Éviter l’import direct de `saev5.py` dans un outil léger : vérifier ses effets de bord. Le preflight, les exports et la visualisation ne doivent charger aucun LLM par défaut.

### 16.3 Manifeste minimal par artefact

```json
{
  "schema_version": "post-stage-v1",
  "campaign_id": "sae_post_soutenance_15j",
  "experiment_id": "E03",
  "phase": "confirm",
  "status": "complete",
  "code_commit": "ACTUAL_COMMIT",
  "config_sha256": "...",
  "corpus_manifest_sha256": "...",
  "split_manifest_sha256": "...",
  "protocol_sha256": "...",
  "parent_artifact_ids": ["..."],
  "model_revision": "...",
  "sae_checkpoint_sha256": "...",
  "feature_catalogue_version": "...",
  "pooling": "max",
  "judgment_source": "human",
  "judgment_scope": "pooled_top10",
  "seed_data": 42,
  "seed_training": 42,
  "n_parent_docs": 0,
  "n_valid_tokens": 0,
  "n_token_presentations": 0,
  "slurm_job_id": "...",
  "elapsed_seconds": 0,
  "gpu_hours": 0,
  "host_memory_peak_bytes": 0,
  "warnings": []
}
```

Les champs numériques sont des exemples de schéma, pas des valeurs à conserver. Une valeur manquante est `null` avec explication, pas zéro inventé. Le manifeste local peut porter une localisation autorisée ; la version Git ne contient pas les chemins sensibles, noms ou textes.

Écrire données et manifeste dans des fichiers temporaires, valider, puis renommer atomiquement. Un fichier présent n’est pas forcément complet ; vérifier hash, forme, nombre de lignes, ordre des IDs, absence de NaN inattendus et dépendances. `status=partial` ne peut pas être chargé comme résultat final.

### 16.4 Profil Slurm initial

Exemple de **gabarit à adapter et à rendre effectif seulement après preflight** :

```bash
#!/usr/bin/env bash
#SBATCH --job-name=sae-post-profile
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --signal=USR1@180
#SBATCH --output=logs/post_stage/profile_%j.out
#SBATCH --error=logs/post_stage/profile_%j.err
# La partition et le compte/QOS sont fournis à sbatch par le générateur,
# après détection/validation du site. Créer logs/post_stage avant sbatch.

set -euo pipefail
: "${REPO_DIR:?REPO_DIR doit désigner le dépôt local autorisé}"
: "${CAMPAIGN_CONFIG:?configuration de campagne requise}"
cd "$REPO_DIR"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

srun --ntasks=1 /usr/bin/time -v .venv/bin/python -m src.post_stage.cli profile \
  --campaign "$CAMPAIGN_CONFIG" --max-valid-tokens 250000
```

Le générateur ne reprend pas automatiquement `--signal=B:USR1` des anciens scripts : avec `B`, le signal vise le shell batch, pas nécessairement le processus Python. Choisir un schéma de propagation correct pour le site et **tester** que Python sauvegarde. Un signal non géré peut tuer le programme au lieu de le checkpoint-er. Ne pas activer `--requeue` sans reprise testée et politique locale compatible.

Ne pas inclure une pseudo-variable shell dans une directive `#SBATCH` en pensant qu’elle sera interpolée. Les paramètres dynamiques sont passés à `sbatch` ou écrits littéralement dans le fichier généré. Utiliser les noms de partition observés (`a100`, `h100`, éventuellement autre) et un GPU compatible avec le stage. Ne pas déduire le compte/QOS du nom de l’auteur.

### 16.5 Graphe des dépendances et réservation raisonnée

`preflight → freeze_data → profile/fix → extract_FIT → train → encode → labels → applications → confirm → report → Streamlit`.

DENSE/TFIDF et l’annotation des requêtes peuvent avancer en parallèle côté CPU/humain. Les trois entraînements partagent une extraction en lecture seule mais disposent de répertoires de résultats distincts. Les analyses CPU ne gardent pas une allocation GPU inutile.

Le script de soumission doit : lire les résultats de preflight ; calculer un budget restant ; vérifier qu’un job identique n’est pas déjà soumis ; enregistrer les IDs ; créer les dépendances `afterok` seulement si les codes de sortie le permettent. Une sortie volontaire pour checkpoint doit être distinguée d’un succès final ; reprendre avant de libérer les dépendances applicatives.

Les verrous de cache existants sont utiles mais ne justifient pas de réserver un GPU pendant six heures d’attente d’un cache. Attendre côté ordonnanceur via dépendance. Un heartbeat ancien ne suffit pas à autoriser la destruction d’un cache : vérifier le job/PID propriétaire lorsque possible ; sinon marquer bloqué pour décision.

### 16.6 Budget automatique et arrêt

Chaque étape met à jour un ledger `resources_ledger.jsonl` : CPU/GPU accordés, durée, consommation, état, retry et coût cumulé. Un job de 4h à 1GPU compte 4GPU-h ; l’attente en file ne compte pas comme calcul, mais compte pour le calendrier.

Arrêt/alerte si : budget restant insuffisant ; moins de trois jours de livraison ; charge prévue supérieure à la limite ; absence d’espace/quota ; mémoire cgroup proche de sa limite de façon durable ; sortie non finie ; colonne feature/doc mal alignée ; variation de protocole ; données hors autorisation. Une surveillance à 85–90 % sert à checkpoint-er quand cela est techniquement sûr, pas à garantir qu’un OOM rapide sera toujours évité.

Deux catégories d’échec : `technical_failure` (corriger/reprendre dans la limite) et `scientific_negative` (conserver/analyser, ne pas réentraîner pour le faire disparaître). L’agent ne doit pas confondre les deux.

### 16.7 Tests unitaires et de non-régression nécessaires

| Test | Cas de référence | Résultat exigé |
|---|---|---|
| Split | Parent et variantes désordonnés | Aucun chevauchement entre FIT/DEV/CONFIRM |
| Mapping documentaire | Filler très nombreux, IDs troués | Aucune ligne filler dense ; codes/exemples correctement reliés |
| Mapping labels | JSON non trié | Embedding associé au bon feature_uid |
| Reprise | Interruption au milieu d’un bloc | Aucun batch perdu ou doublé ; optimiseur/RNG/buffers restaurés |
| Core gelé | Entraînement extra de quelques steps | Hash/poids du cœur inchangés |
| Projection BatchTopK | Train puis eval sur batches différents | Seuil restauré et même code eval hors erreurs d’arrondi |
| Precision@10 | 7 hits sur 10 | 0,7 ; gestion explicite des jugements manquants |
| AP | 3 pertinents trouvés mais 10 pertinents totaux | Dénominateur 10, pas 3 |
| Diffing | Groupes inversés | Même amplitude, signe inverse, sens vérifié |
| Parseur | JSON invalide/texte vide | `invalid`, jamais NO implicite |
| NPMI | 1 cooccurrence/400 et marges égales | Valeur éventuelle + badge support insuffisant |
| Clustering | Vecteur nul après filtre | Hors axe, pas cluster thématique forcé |
| Stabilité | Dictionnaire permuté à l’identique | Correspondance parfaite après permutation |
| Sous-espace | Rotation d’une même base / directions aléatoires | Overlap attendu ; contrôle du rang et du null |
| Cache | Un paramètre de label/pooling change | Invalidation ciblée des descendants seulement |
| Dashboard | Données incomplètes ou provenance inconnue | Avertissement et absence de validation verte |

Les optimisations numériques doivent passer un test sur entrées factices puis un petit cache non confidentiel autorisé. Les tests ne doivent ni télécharger des modèles dans CI, ni utiliser un GPU partagé sans l’annoncer.

---

## 17. Streamlit : pages, provenance, exports et tests

### 17.1 Principe d’interface

Le changement important n’est pas d’ajouter des graphiques mais de rendre **comparable et vérifiable** ce qui est montré. Les artefacts sont calculés hors interface et consommés en lecture seule. Le dashboard ne lance pas implicitement de gros jobs au chargement et ne charge pas un tenseur documentaire incluant du filler.

Conserver l’interface actuelle et ajouter un espace « Campagne post-soutenance » avec cinq vues liées : Recherche ; Comparaison de populations ; Carte de features ; Associations ; Regroupement ciblé. Une page Synthèse indique les livrables, limites et statuts. Le mode historique reste accessible mais visuellement distinct.

### 17.2 Informations communes à toutes les pages

Corpus/partie du split ; synthétique ou réel ; version de représentation ; CORE/EXTRA/FULL ; modèle/couche ; identifiant de run ; nombre de parents ; fréquence et longueur ; statut humain/règle/modèle ; protocole et date de calcul ; avertissements ; lien vers le rapport local correspondant. Pour une métrique, afficher son dénominateur et le périmètre de jugement.

L’utilisateur peut comparer les méthodes sans changer silencieusement de corpus. Les filtres actifs doivent être visibles et inclus dans les exports. Une absence de données est distinguée d’un score nul ; une absence d’effet significatif d’une équivalence ; une hypothèse candidate d’une piste métier retenue.

### 17.3 Performance et sécurité

Lire uniquement les colonnes/chunks nécessaires ; cache d’interface borné et versionné ; pagination des exemples ; nombre de nœuds/arêtes affichés limité avec indicateur de couverture. Pas de `toarray()` sur toutes les activations, pas de chargement du full core 262k pour une simple page de résumé.

Les extraits peuvent être accessibles aux utilisateurs EDF autorisés, mais les captures et exports externes sont expurgés. Prévoir `safe_export=true` par défaut. Ne pas prétendre anonymiser un email en supprimant seulement son nom. Les logs de la page ne doivent pas contenir la totalité des requêtes/confidentialités.

### 17.4 Critères de recette

Un lecteur peut, en moins de quelques clics : partir d’une requête, voir pourquoi un email est proposé, comparer à une méthode simple ; partir d’une différence de population, voir le nombre d’exemples et la confirmation ; ouvrir un groupe de features, distinguer stabilité mesurée et simple voisinage. Les retours aux documents doivent suivre les IDs, jamais une position dans un tableau susceptible d’avoir changé.

Tester à froid/à chaud et relever le temps observé plutôt que promettre « instantané ». Objectif ergonomique proposé : recherche sur index déjà chargé et navigation en quelques secondes ; l’extraction initiale reste hors session. Les seuils exacts dépendent du poste et du volume, à mesurer E08.

---

## 18. Nettoyage du dépôt, résultats remplacés et passation

### 18.1 Ne pas confondre obsolescence et changement de question

Un nouveau test sur emails réels ne rend pas faux un ancien test sur données synthétiques. Un résultat plus précis ne change pas rétroactivement l’échantillon ancien. Définir les statuts : `reference_for_protocol`, `exploratory`, `historical`, `superseded_by_fix`, `invalid_alignment`, `incomplete`, `insufficient_support`.

Seule une erreur démontrée de pipeline/alignement/parseur qui affecte un cache justifie de marquer ce résultat invalide. Un simple doute crée `needs_audit`. Les résultats négatifs sont conservés au même titre que les positifs.

### 18.2 Matrice de migration

| Ancien objet | Action pour cette campagne | Ce qui reste à conserver |
|---|---|---|
| Sondes sur représentation ayant vu les emails | Classer historique/transductif ; nouvelle référence E01 sur parents tenus à l’écart | Prédictions, règles d’intention et lecture originale |
| Contraste CORE/FULL historique K32/layer24 | Lier au nouveau test, ne plus le présenter comme verdict R0/1B | Ancienne config et chiffres |
| Diffing énergie/sport `8/10` | Démonstrateur proxy/descriptif ; afficher séparément E04 | Catalogue exact s’il est disponible, matrices, explication du seuil |
| AP top50 dénominateur local | Renommer, nouvelle API de métriques explicites | Valeurs historiques sous leur vrai périmètre |
| Clustering mapping labels/IDs | Audit du cache ; invalider seulement si erreur établie | Ordre du JSON, liste d’IDs, version de code |
| Groupes Louvain fondés sur labels | Garder expérience historique négative ; nouveau module fonctionnel | Résultats du témoin aléatoire |
| NPMI à très faible support | Badge faible effectif ; priorité aux confirmations E06 | Supports exacts, y compris 1 ou 2 positifs |
| Fichiers codes avec filler dense | Nouveau format compact ; convertisseur contrôlé | Manifeste d’ordre ; pas besoin de conserver deux gros caches indéfiniment |
| Scripts Slurm anciens 900G/1200G | Déplacer/référencer sous profil legacy après validation du nouveau chemin | Preuve de consommation passée et contexte |
| Commentaires de sélection/échelle invalidés | Réécrire commentaire de statut et lien vers résultat actuel | Journal append-only inchangé |

### 18.3 Organisation des résultats et journal

Créer un registre structuré des expériences (`experiments_registry.jsonl` ou équivalent déjà présent) avec ID, question, config, dates, hashes, statut, liens source et remplaçant éventuel. Détecter le dernier numéro de `RESULTS_TESTS.md` avant d’ajouter une entrée : le code auditée allait jusqu’à 126, mais ce numéro peut avoir changé au moment d’exécution. Ne pas supposer aveuglément que 127 est libre.

L’index du journal doit refléter les entrées réellement présentes. Ajouter des rapports lisibles par expérience et une synthèse finale qui sépare les trois catégories « mesuré », « non concluant » et « non réalisé ». Conserver le détail des échecs techniques hors du récit scientifique principal, avec lien.

Les commits locaux sont petits et thématiques : contrats/tests ; mémoire ; représentations ; applications ; dashboard ; documentation. Ne pas effectuer une réorganisation massive simultanément aux résultats. Une dépendance ajoutée est justifiée, verrouillée et testée dans l’environnement existant ; ne pas mettre à niveau Torch/Transformers/SAELens pour essayer de résoudre tous les problèmes en une fois.

### 18.4 Nettoyage disque non destructif par défaut

Produire `cleanup_plan.json` avec pour chaque artefact : chemin autorisé, taille, hash, propriétaires/consommateurs, raison de conservation ou d’archivage, présence d’un job actif, destination de sauvegarde, commande proposée. Mode défaut `dry-run`.

Ordre : vérifier que le nouveau format donne les mêmes sorties ; enregistrer les manifestes ; faire approuver la suppression ; supprimer seulement des artefacts temporaires ou redondants identifiés ; conserver au moins les checkpoints retenus, les métriques, les prédictions/qrels et la provenance. Les fragments partagés ne sont pas supprimés par un job individuel. Pas de `rm -rf results_*` ni de purge à partir d’un suffixe de fichier.

### 18.5 Passation à J15

Livrer : commande CPU de lecture des résultats ; smoke-test à faible coût ; configurations ; versions modèles ; inventaire des caches ; environnement verrouillé ; procédure de reprise ; modes d’échec connus ; données nécessaires non livrées dans Git ; liste des décisions prises ; liste des expériences abandonnées avec raison et coût déjà engagé.

Le successeur doit pouvoir distinguer ce qui est disponible techniquement et ce qui est démontré. Chaque branche optionnelle non exécutée conserve son protocole, pas une case préremplie « résultat attendu positif ».

---

## 19. Travaux confiés au successeur

| Suite | Pourquoi elle ne doit pas être promise en quinze jours | Condition d’entrée et livrable attendu |
|---|---|---|
| Corpus réel multi-périodes/régions | Autorisations, provenance et changement de distribution dépassent le code seul | Accès approuvé, métadonnées fiables ; benchmark temporel et régional |
| Évaluation métier plus large | Deux ou trois utilisateurs n’établissent pas l’utilité générale | Protocole avec analystes, tâches répétées, critères de nouveauté et coûts |
| Stabilité sur davantage de graines et corpus | Trois graines donnent une indication, pas une fréquence de réapparition précise | 5–10 graines au minimum comme programme proposé, analyses de rang/nulls, réplication |
| Hiérarchie et splitting validés | Les voisins ne sont pas automatiquement des sous-concepts | Tests d’inclusion, exemples/contre-exemples, stabilité et usage documentaire |
| Méta-SAE / Matryoshka / ancrage ou ensembles | Modification architecturale et nouveau coût d’évaluation | Ne commencer que si E05 montre un problème opérationnel que ces méthodes peuvent résoudre |
| Vraie intervention dans le LLM | Le steering actuel est un round-trip documentaire | Interventions token-level, contrôles de reconstruction et effets hors cible |
| Adaptation mieux appariée | FULL vs CORE répond au gain pratique, pas à toutes les causes du gain | Budgets actifs et coûts appariés, baselines d’adaptation complètes |
| Retrieval multilingue/Latent Terms complet | Architecture distincte, annotations coûteuses | Benchmarks appropriés, AP/MAP globales valides, reranker apparié |
| 100M/12B ou au-delà | Stockage, file d’attente et coût non justifiés sans bénéfice métier | Profilage validé, hypothèse applicative précise, budget accepté |
| Industrialisation | Sécurité, contrôle d’accès, exploitation et dérive nécessitent un propriétaire | Maintenance, tests de charge, observabilité, mise à jour des catalogues, critères de retrait |
| Publication | Un outil utilisable n’est pas automatiquement une preuve générale | Second corpus, annotations indépendantes, réplications, artefact public autorisé |

Le sujet du successeur ne doit pas être « finir les sweeps ». Il doit être formulé comme une décision : **quelles propriétés peut-on explorer de façon fiable, à quel coût, sur quelles données, et avec quel avantage par rapport aux outils déjà disponibles ?**

---

## 20. Critères de clôture et instruction de démarrage

### 20.1 Définition de terminé

La campagne minimale est terminée quand :

- les données et les groupes sont traçables, sans fuite locale démontrable vers CONFIRM ;
- CORE/FULL/DENSE/TFIDF sont comparés dans un protocole commun, ou le bras absent est explicitement signalé ;
- retrieval et diffing produisent un résultat complet, y compris si négatif ;
- les groupes/cartes, corrélations et clusters portent le bon niveau de validation ;
- les annotations manquantes et les faibles supports ne sont pas cachés ;
- Streamlit permet le retour aux textes autorisés et aux méthodes alternatives ;
- les nouveaux scripts reprennent après interruption et n’occupent pas inutilement le cluster ;
- les anciens résultats sont correctement classés, les suppressions sont approuvées, la passation est reproductible.

### 20.2 Instruction à copier au démarrage de Claude Code

> Lis ce plan, puis le `CLAUDE.md`, les tests et les modules concernés dans leur état local actuel. Ne considère aucune description historique comme une instruction de lancer un sweep. Commence par un inventaire sans modification destructive et compare HEAD à `68c8e6863013c98905740a56e6491d78d05435cd`. Écris le preflight, le budget et les différences de code pertinentes. Implémente les contrôles de données et les corrections bloquantes avant toute évaluation. Soumets uniquement les étapes conformes au budget approuvé et aux règles du cluster. Réutilise les extractions compatibles ; sépare les processus d’extraction, entraînement et analyse. Ne communique à l’extérieur aucun texte confidentiel. Ne modifie ni le rapport défendu ni le dossier oral. Gèle les requêtes/hypothèses avant confirmation. Publie aussi les résultats négatifs et non concluants. À J9, privilégie la livraison du minimum ; à J13, gèle les fonctionnalités et prépare la reprise. Le run 100M nécessite une décision distincte après profilage et ne doit jamais retarder les applications.

### 20.3 Rapport quotidien bref de l’agent

`Réalisé / Ressources consommées et restantes / Résultats provisoires et statut / Risques / Prochaine étape / Décision humaine nécessaire`. Aucun texte client ni secret dans ce rapport. Une expérience en attente de GPU ne doit pas être décrite comme en cours de calcul ; une mesure de temps doit porter le job et l’étape.

---

## 21. Sources, références et portée des vérifications

### 21.1 Sources du projet

**R — Rapport final.** `Grégoire_Pelletier_Rapport_de_stage.pdf`, 40 pages, fourni par l’auteur. En particulier §3.2–3.6, §4.2–4.7, §5.3–5.4 et annexes A–I. Ce plan ne corrige pas rétroactivement ce rapport.

**S — Soutenance corrigée.** `Soutenance_SAE_EDF_corrigee(1).pptx`, 30 diapositives, fournie par l’auteur. Les priorités et distinctions prototype/validation métier sont conservées.

**O — Dossier oral existant.** `Dossier_oral_SAE_complet.md`, fiches F14–F25 et protocoles P01–P10. Utilisé pour repérer les contrôles déjà identifiés, sans modification du dossier. Les problèmes prioritaires ont été recoupés avec les fichiers GitHub ci-dessous.

### 21.2 Code figé consulté et points d’entrée

Tous les liens suivants pointent sur la révision auditée. L’agent doit vérifier les différences avec sa copie locale ; les lignes peuvent avoir bougé après cette révision.

- **C1 — Orchestration et mémoire** : [`src/sae/saev5.py`](https://github.com/GregoirePelletier/SAE/blob/68c8e6863013c98905740a56e6491d78d05435cd/src/sae/saev5.py), notamment 146–154 (mapping), 1130–1390 (filler/réservoir/stack), 1390–1565 (entraînement/réencodage). Lecture directe ciblée.
- **C2 — Cache documentaire et entraînement** : [`src/sae/sae_shared.py`](https://github.com/GregoirePelletier/SAE/blob/68c8e6863013c98905740a56e6491d78d05435cd/src/sae/sae_shared.py), 180–280 (hash, compactage/reconstruction), 414–end (shuffle, split et entraînement). Lecture directe ciblée.
- **C3 — Presets et defaults** : [`src/config.py`](https://github.com/GregoirePelletier/SAE/blob/68c8e6863013c98905740a56e6491d78d05435cd/src/config.py), 35–168. Lecture directe ciblée.
- **C4 — Métriques IR** : [`src/analysis/metrics.py`](https://github.com/GregoirePelletier/SAE/blob/68c8e6863013c98905740a56e6491d78d05435cd/src/analysis/metrics.py), 230–360. Lecture directe du dénominateur AP et des métriques.
- **C5 — Clustering complet** : [`scripts/clustering_llm_test.py`](https://github.com/GregoirePelletier/SAE/blob/68c8e6863013c98905740a56e6491d78d05435cd/scripts/clustering_llm_test.py), 45–195. Lecture directe des deux problèmes d’alignement/troncature.
- **C6 — Vérification du diffing** : [`src/analysis/hypothesis_verifier.py`](https://github.com/GregoirePelletier/SAE/blob/68c8e6863013c98905740a56e6491d78d05435cd/src/analysis/hypothesis_verifier.py). Lecture directe du parseur, du seuil absolu et de la couverture.
- **C7 — Applications communes** : [`src/analysis/cooccurrence.py`](https://github.com/GregoirePelletier/SAE/blob/68c8e6863013c98905740a56e6491d78d05435cd/src/analysis/cooccurrence.py), `clustering_llm.py`, `correlations_verified.py`, `diff_hypothesis_generator.py`. Points d’entrée recensés ; leurs adaptations doivent être inspectées/testées par l’exécuteur, les commentaires seuls ne font pas autorité.
- **C8 — Données** : [`src/data/preparation.py`](https://github.com/GregoirePelletier/SAE/blob/68c8e6863013c98905740a56e6491d78d05435cd/src/data/preparation.py), `dataset.py` et tests parent-aware. Points d’entrée à vérifier lors du gel de corpus.
- **C9 — Journal** : [`RESULTS_TESTS.md`](https://github.com/GregoirePelletier/SAE/blob/68c8e6863013c98905740a56e6491d78d05435cd/RESULTS_TESTS.md). Les résultats résumés sont ceux du rapport/dossier et de leurs audits précédents ; aucune réexécution des caches locaux lors de cette rédaction.
- **C10 — Mesure mémoire rapportée** : [`slurm/pipeline_runs/run_validation_50m_layer31_probe.slurm`](https://github.com/GregoirePelletier/SAE/blob/68c8e6863013c98905740a56e6491d78d05435cd/slurm/pipeline_runs/run_validation_50m_layer31_probe.slurm). Lecture intégrale ; observation 358,5 Go citée dans son commentaire, pas relue dans Slurm.
- **C11 — Reprise** : [`src/storage/checkpoint.py`](https://github.com/GregoirePelletier/SAE/blob/68c8e6863013c98905740a56e6491d78d05435cd/src/storage/checkpoint.py), `fragment_store.py`, tests de réservoir et verrous : points d’entrée pour le test de crash/reprise.
- **C12 — Modèle résiduel** : [`src/sae/frozen_core.py`](https://github.com/GregoirePelletier/SAE/blob/68c8e6863013c98905740a56e6491d78d05435cd/src/sae/frozen_core.py), `batch.py`. Conservation du contrat x/résidu et des buffers requise.
- **C13 — Auto-interprétation** : [`src/sae/judge.py`](https://github.com/GregoirePelletier/SAE/blob/68c8e6863013c98905740a56e6491d78d05435cd/src/sae/judge.py). Réutiliser les appels locaux, gérer sorties invalides et versionner le catalogue.
- **C14 — Modèle 1B** : [`slurm/pipeline_runs/run_ablation_volume50m_model_scale_1b.slurm`](https://github.com/GregoirePelletier/SAE/blob/68c8e6863013c98905740a56e6491d78d05435cd/slurm/pipeline_runs/run_ablation_volume50m_model_scale_1b.slurm). Point d’entrée existant, pas modèle de budget à reproduire sans profilage.
- **C15 — Interface** : [`src/visualization/dashboard.py`](https://github.com/GregoirePelletier/SAE/blob/68c8e6863013c98905740a56e6491d78d05435cd/src/visualization/dashboard.py). Étendre via un adaptateur de données, pas par chargement global des caches.

### 21.3 Recherche externe : références primaires et rôle précis

- **L1 — Jiang et al.**, *Interpretable Embeddings with Sparse Autoencoders: A Data Analysis Toolkit*, [arXiv:2512.10092v2](https://arxiv.org/html/2512.10092v2), version du 22 juillet 2026 consultée. Source du cadre à quatre applications et de la représentation documentaire. Les nouveaux budgets, splits et critères humains proposés ici sont des choix de planification, pas des résultats du papier. La numérotation d’annexes du papier peut différer de celle des commentaires legacy du dépôt.
- **L2 — Bricken et al.**, *Towards Monosemanticity*, [section Feature Splitting](https://www.transformer-circuits.pub/2023/monosemantic-features/index.html), 2023. Précédent pour familles et splitting ; une carte de proximité ne fournit pas à elle seule une validation.
- **L3 — Leask et al.**, *Sparse Autoencoders Do Not Find Canonical Units of Analysis*, [arXiv:2502.04878](https://arxiv.org/abs/2502.04878), 2025. Mise en garde contre l’atomicité/canonicalité automatique des latents ; méta-SAE comme piste de reprise, non obligation immédiate.
- **L4 — Gerasimov et al.**, *Unstable Features, Reproducible Subspaces: Understanding Seed Dependence in Sparse Autoencoders*, [arXiv:2606.12138v1](https://arxiv.org/html/2606.12138v1), 10 juin 2026. Motive la distinction feature/sous-espace. Son étude principale emploie bien davantage de graines ; E05 est un test local réduit et ne reprend pas ses conclusions comme acquises.
- **L5 — PyTorch**, [documentation du stockage et de from_file](https://docs.pytorch.org/docs/stable/storage). `from_file(shared=True)` utilise un fichier mappé persistant ; cela n’est pas synonyme de zéro consommation RAM. Vérifier les API de la version Torch installée plutôt qu’effectuer une mise à jour pour suivre la documentation stable.
- **L6 — Linux Kernel**, [Control Group v2](https://docs.kernel.org/admin-guide/cgroup-v2.html). Base de la distinction mémoire anonyme/cache fichier/pages sales et suivi cgroup. L’agent doit adapter les chemins au site et à cgroup v1/v2.
- **L7 — Slurm**, [documentation sacct](https://slurm.schedmd.com/sacct.html). Base de la collecte des ressources par job/step et des précautions d’interprétation.

### 21.4 Limites de l’audit

Accès GitHub en lecture et lecture des documents fournis ; pas d’accès direct au cluster, à ses files d’attente actuelles, aux modèles locaux, aux corpus bruts ni aux `sacct` des jobs. Le diagnostic mémoire établit des allocations et des parcours problématiques **dans le code** ; il ne désigne pas avec certitude la cause terminale d’un OOM 50M précis. Aucun entraînement, aucune estimation de débit sur A100/H100 et aucune annotation humaine n’ont été faits pendant cette préparation.

Les calendriers, tailles d’échantillon, budgets, marges utiles et futures commandes sont des **propositions opérationnelles**. Les résultats du rapport sont des **faits rapportés dans ce périmètre**. Les éléments de littérature sont explicitement distingués. Conserver cette séparation dans tous les comptes rendus produits par l’agent.
