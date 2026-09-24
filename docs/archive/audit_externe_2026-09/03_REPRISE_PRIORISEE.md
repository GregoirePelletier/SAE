# SAE — Reprise priorisée après la campagne post-soutenance

**Base de travail :** `2f6fc2418e4ac084526dfc57b858431be426360b`, lecture du 22 septembre 2026. Destinataires : Grégoire, encadrement, personne reprenant le projet, assistant de programmation autorisé.

Ce document complète le bilan et la passation technique. Il remplace les priorités opérationnelles du plan initial, sans modifier rétrospectivement les expériences qui ont suivi ce plan. Les références Sxx sont résolues dans [SOURCES.md](SOURCES.md).

## 1. Ordre de travail recommandé

1. **Sécuriser les données et les artefacts**, pour ne pas perdre le travail ni garantir à tort la séparation des parents.
2. **Corriger les erreurs de contrat et de restitution**, avant de présenter une conclusion figée.
3. **Obtenir un retour humain limité mais réel**, sur les cas d'usage les plus concrets.
4. **Clore les analyses déjà engagées**, notamment l'initialisation aléatoire d'E05.
5. **Transmettre une recette reproductible**, et laisser les extensions plus longues au successeur.

Les estimations ci-dessous sont des budgets d'ingénierie proposés, pas des temps mesurés sur le cluster. Elles supposent que les fichiers historiques sont disponibles et que la personne connaît déjà Python. La file d'attente, la disponibilité des analystes et le coût d'une réextraction ne sont pas inclus. Ne pas lancer automatiquement un job parce qu'une tâche figure dans ce document.

| ID | Avant le départ : minimum | Responsable à mobiliser | Effort indicatif hors calcul |
|---|---|---|---|
| P0-DATA | Reconstituer la filiation ou documenter précisément son absence | Grégoire + propriétaire des données | 0,5–2 jours si les sources d'augmentation existent |
| P0-ARTEFACTS | Sauvegarde contrôlée, manifestes, accès du repreneur | Grégoire + repreneur / support cluster | 0,5–1 jour ; copie selon volume |
| P0-DIFF | Corriger l'unité fournie au générateur, tests, version de protocole | Développeur | 2–4 h pour le correctif isolé ; davantage pour le rejeu |
| P0-DOCS | README, E03, E04, E05, E06 ; total des corrélations vérifié | Grégoire + développeur | 0,5–1 jour |
| P1-HUMAIN | Une séance d'exploration et un premier audit d'exemples | 2–3 personnes disponibles | 2–4 h de préparation + sessions + synthèse |
| P1-RANDOM | Récupérer le statut réel des bras random et restituer | Développeur + accès cluster | 0,5 jour si sorties disponibles |
| P1-RECETTE | Reprise sur un autre compte et démo CPU | Repreneur | 0,5–1 jour |
| P2-APPS | Étendre les contrôles et les cas métier | Successeur + métier | Plusieurs itérations à chiffrer après pilote |
| P2-MEMORY | Compléter le chemin compact et la localité disque | Successeur / ingénierie ML | À profiler, ne pas pré-engager un 100M |

**Décision de périmètre :** si les jours restants sont insuffisants, livrer les P0 et une passation fonctionnelle avant toute expérience supplémentaire. Un blocage de provenance explicitement documenté vaut mieux qu'une validation artificielle.

## 2. Règles pour un assistant de programmation

Lire d'abord `git status`, la politique de campagne et les sources concernées. Ne pas écraser les modifications locales, reconstruire automatiquement l'environnement, changer les dépendances, supprimer des caches, envoyer des extraits à un service externe ou pousser du code. La politique observée fixe notamment `allow_push=false`, `allow_destructive_cleanup=false` et un budget optionnel 100M nul. Elle doit être confirmée par l'encadrement pour la reprise. [S19]

Pour chaque action, conserver trois états : **implémenté**, **exécuté**, **analysé**. Ajouter **validé humainement** seulement avec les annotations correspondantes. Une ligne de script Slurm n'est pas la preuve qu'un run s'est terminé. Aucun résultat numérique attendu ne doit servir de critère de sélection caché des paramètres.

Chaque correctif doit avoir : motif, test qui échouait auparavant, version du protocole, artefacts touchés, nécessité ou non d'un rejeu, et résultat réel des tests. Ne pas effacer les chiffres historiques ; modifier leur statut et créer un renvoi vers la version qui les remplace pour la même question.

## 3. P0-DATA — Sécuriser la filiation des variantes

### Pourquoi

Le split FIT/DEV/CONFIRM repose sur des hashes de parents mais le manifeste indique `positional_via_parent_id`, avec 70 variantes non rattachées. Le code reconstruit l'ordre à partir des parents nettoyés. Cela peut être exact si c'était l'ordre de génération ; ce n'est pas établi par les seuls fichiers de code. [S02, S18]

### Travail demandé

Retrouver le fichier de parents et la version du générateur au moment de l'augmentation. Comparer la séquence originale de parents à celle produite par `load_and_clean_emails`. Pour chaque variante, produire dans le périmètre autorisé un enregistrement `{aug_id, parent_source_id, parent_sha1, variant_sha256, resolution_status, resolution_evidence}`. Ne pas utiliser un simple rapprochement sémantique comme preuve définitive de filiation.

Expliquer les 70 exclusions : hors borne, parent supprimé, identité non reconnue, erreur de type, ou autre raison réellement constatée. Vérifier aussi si plusieurs variantes acceptées ont été attribuées au mauvais parent tout en restant « appariées » selon le manifeste. Tester le split avec une permutation d'ordre des fichiers et avec des modifications de contenu à identifiant constant.

### Critère d'acceptation

Chaque variante conservée possède une filiation traçable ; les orphelines restent explicitement isolées ; les hashes des textes et leur ordre concordent avec les codes encodés. Si le mapping change, chiffrer le nombre de documents déplacés entre splits et déterminer les résultats à refaire. Ne pas réutiliser des checkpoints supposés FIT-only si leur véritable ensemble d'apprentissage inclut désormais des parents de confirmation.

### Conclusions possibles

Mapping confirmé : les réserves peuvent être levées pour les artefacts qui partagent les hashes vérifiés. Mapping modifié : ancienne campagne conservée sous statut « contrat de filiation antérieur », nouvelle campagne versionnée. Sources impossibles à retrouver : ne pas certifier l'indépendance ; privilégier à terme une génération nouvelle avec filiation native.

**Streamlit :** afficher version du corpus, statut de filiation, nombre d'orphelines et identité parent dans les résultats. **Nettoyage :** aucun remplacement silencieux de `split_assignments.json` ; le fichier historique doit rester récupérable.

## 4. P0-ARTEFACTS — Faire une passation qui ne dépende pas du compte de Grégoire

### Travail demandé

Exécuter le preflight fourni, puis construire un inventaire validé par le repreneur. Préserver les JSON E01–E07, les checkpoints de référence et de répétitions, le manifeste des splits, les configurations, la révision exacte des modèles, les logs et les cibles des liens symboliques. Les gros fichiers peuvent être conservés dans un espace de projet autorisé plutôt que dupliqués.

Produire deux recettes : **démo CPU depuis les résultats** et **recalcul sur cluster**. La première ne doit pas exiger une réextraction de tous les tokens. Le repreneur doit pouvoir retrouver un chiffre, son fichier source et ses conditions sans interroger Grégoire.

Créer un manifeste d'artefacts avec tailles, empreintes, rôle, propriétaire, emplacement, règles de conservation, version du schéma et expérience productrice. Pour les fichiers très volumineux, planifier le calcul des empreintes sur une ressource autorisée ; ne pas scanner plusieurs téraoctets depuis le frontal sous couvert d'un contrôle rapide.

### Critère d'acceptation

Le repreneur accède aux résultats sous son propre compte, vérifie les empreintes des petits fichiers et ouvre E08/E05. Un lien symbolique dont la cible reste inaccessible constitue un transfert raté, même si le clone Git est complet.

**Sécurité de l'interface :** déplacer l'écriture des pistes E08 hors de `docs/` et hors suivi Git avant de saisir des éléments sensibles. Tester les écritures concurrentes et la conservation de l'auteur, de la date et des corrections. Le champ « participants » et les commentaires de séance ne doivent pas devenir des données publiques par accident. [S10, S26]

## 5. P0-DIFF — Corriger le contrat d'E04 et rendre la confirmation rejouable

### Correctif certain

Dans `scripts/post_stage/e04_diffing.py`, le champ `percentage_difference` reçoit actuellement `log_odds_ratio`. Le prompt lui attribue le sens d'un écart de fréquence borné entre −1 et 1. Transmettre `float(row['freq_A'] - row['freq_B'])`, ou renommer explicitement le champ et le prompt si le log-odds est volontairement retenu. Ne pas changer seulement l'affichage. [S15–S17]

**Tests minimaux :** pour `freq_A=0.8` et `freq_B=0.2`, le prompt de différence de fréquence doit recevoir `+0.6`, pas le log-odds ; inverser A/B doit inverser le signe. Vérifier qu'un seuil de trois points porte sur la bonne unité. Les propriétés sans label exploitable doivent être distinguées des descriptions vérifiées, et la provenance des exemples doit être conservée.

### Protocole de confirmation

Sauvegarder les hypothèses et le prompt exact, leur hash et les IDs de découverte **avant** l'ouverture de CONFIRM. Conserver ensuite la matrice de jugements, les IDs documentaires/parents, les éventuels échecs de parsing et les comptes. Le JSON actuel ne suffit pas à reconstruire toute cette preuve sans un nouvel appel au juge. [S15]

Exploiter les parents lorsqu'on compare les versions panique et calme d'un même email. Choisir avant le rejeu une comparaison appariée sur les parents disposant des deux variantes, ou une méthode d'inférence groupée adaptée ; ne pas appeler les observations indépendantes par défaut. Les variantes manquantes doivent rester visibles.

Le prompt de question est actuellement spécifique à panique/calme malgré des arguments `--label-a`/`--label-b` variables. Un autre contraste doit changer explicitement la question, pas seulement les labels sélectionnés.

### Restitution et impact

Séparer : nombre d'hypothèses testées, différence descriptive, significativité, sens annoncé correct et qualité humaine. `verification_rate` utilise la valeur absolue ; `coverage` ne retient que le sens favorable à la cible dans le code courant. [S27]

Ne pas annoncer que les huit hypothèses anciennes deviennent « fausses » après le correctif d'unité. Leurs jugements de présence sont une observation pour ces formulations ; c'est la chaîne de génération qui doit être corrigée et distinguée. Un nouveau choix inspiré par CONFIRM est désormais exploratoire sur ce jeu : prévoir des données de confirmation supplémentaires avant une nouvelle revendication générale.

**Streamlit :** montrer direction attendue, direction observée, effet signé, intervalle, support et version du générateur. **Nettoyage :** statut `legacy_generator_unit_mismatch` pour le protocole antérieur, sans destruction des JSON.

## 6. P0-DOCS — Stabiliser la lecture des résultats

### Corrections prioritaires

- **README :** ne plus présenter 89,3 % comme référence actuelle, ni K_EXTRA=32 comme défaut de la campagne. Fournir un accès direct au bilan post-stage, à la démo CPU et à l'inventaire. Distinguer l'ancien 12B/R0 du nouveau 1B/FIT. [S01]
- **E03 :** corriger la phrase sur le besoin de juger tout le corpus pour P@10 ; préciser le budget catalogue 77/197 et l'absence de déduplication du top-10 par parent. Ne pas déclarer `incident_collectif` absent de tous les moteurs. [S05, S14]
- **E04 :** distinguer les métriques absolues et directionnelles, signaler l'unité du générateur et la dépendance des variantes. [S06, S15, S27]
- **E05 :** remplacer « init aléatoire non implémentée » par « implémentée, résultats à récupérer/analyser » tant que les sorties ne sont pas vérifiées. Ne pas confondre `random` entraîné et décodeur aléatoire figé. [S07, S22]
- **E06 :** le tableau annonce quatre associations établies, la synthèse cinq. Recalculer le résumé depuis `e06_correlations.json` : nombre testable, q<0,05, intervalle excluant zéro, direction et caractère redondant. Ne pas choisir arbitrairement le chiffre le plus flatteur. [S08, S24]
- **Mémoire :** délimiter le correctif fait, le chemin FULL encore à revoir et les comparaisons à corpus différent ; réviser les paragraphes « prochaine étape » devenus historiques. [S11]

### Critère d'acceptation

Chaque chiffre important est lié à un artefact, une configuration, un split et un statut. Le journal historique renvoie vers la campagne post-stage ; celle-ci ne doit plus être découvrable uniquement en explorant les dossiers. Les documents soutenus restent archivés comme documents d'époque, sans remplacement rétroactif par un rapport prétendument identique.

## 7. P1-HUMAIN — Transformer les démonstrateurs en premiers retours d'usage

### Audit d'exemples

Commencer par les relances répétées, l'explication d'un montant et l'urgence implicite. Ce sont des cas montrés par E03, pas des thèmes choisis après une nouvelle exploration libre du test. Comparer les listes de FULL et du meilleur moteur classique pertinent, avec une présentation neutralisée et sans montrer le score Qwen avant le jugement humain.

Dédupliquer ou au minimum afficher les parents : dix variantes d'un même email ne sont pas dix dossiers clients. Conserver des contre-exemples difficiles — remboursement demandé, coupure unique, simple mention de délai — selon la définition réellement retenue. Consigner pertinence, ambiguïté et suffisance de l'extrait ; ne pas transformer une erreur de parsing du juge en réponse négative silencieuse.

### Mini-pilote E08

Deux à trois participants, séances d'environ vingt minutes et ordre contrebalancé entre recherche de référence et exploration SAE. Ce petit effectif permet un retour qualitatif d'usage, **pas** une estimation solide d'un gain de productivité en population. Si le panel est constitué de chercheurs et non d'analystes métier, le nommer ainsi.

Deux tâches suffisent : retrouver une propriété dans les emails et comparer deux populations pour formuler une piste étayée. Produire pour chaque piste : question, populations, exemples, contre-exemples, utilité perçue, nouveauté par rapport aux catégories déjà connues, provenance et commentaire humain. Une piste rejetée reste dans le catalogue avec son motif.

### Conclusions possibles

- Moteur classique meilleur mais SAE utile pour expliquer/naviguer : garder un système hybride à évaluer plus largement.
- Catalogue de features utile sur quelques propriétés seulement : documenter ce périmètre au lieu d'un moteur générique.
- Pas d'apport perçu : conserver l'artefact scientifique, ne pas engager de transfert produit sur la seule reconstruction.
- Problème d'interface empêchant le test : recette échouée, pas résultat de non-utilité scientifique.

**Livrable :** comptes rendus réels et catalogue local ; aucune annotation humaine simulée par LLM. [S10]

## 8. P1-RANDOM — Terminer une analyse déjà engagée

Inventorier d'abord les runs `results_post_stage_e05_randinit45/46` et les jobs correspondants sur le cluster. Les scripts existent ; le dépôt ne donne pas le résultat final. Lire les états Slurm et les checkpoints avant de relancer quoi que ce soit.

Si les runs sont complets, exécuter l'analyse prévue par `15_e05_stability_random_init.slurm` en contrôlant les champs `decoder_init`, les données, les masques et la qualité d'entraînement. Comparer PCA→PCA, PCA→random et random→random avec la même procédure. Un bras random qui reconstruit beaucoup moins bien ne permet pas d'attribuer toute instabilité à une absence de concepts communs. [S22]

Lire ensemble géométrie, profils et listes d'emails ; ne pas privilégier seulement la métrique qui s'améliore. Les critères définis dans E05 sont conservés ; un changement de seuil après lecture devient exploratoire. Si les sorties manquent et que le temps ou le budget ne suffit pas, transférer la tâche avec son état exact, sans la déclarer exécutée.

## 9. P1-RECETTE — La personne qui reprend doit faire, pas seulement regarder

Avant la réunion : vérifier l'accès aux résultats et aux modèles, le Python 3.12, l'environnement existant et les liens symboliques. Pendant la réunion : faire ouvrir la page E08, une requête, un résultat E04 et la carte E05 par le repreneur ; lui faire retrouver les paramètres et les limites correspondantes.

Après la réunion : exécuter les tests convenus et un recalcul CPU borné. La CLI `src.post_stage.cli` ne fournit que `freeze-corpus` : ne pas appeler une commande d'exécution imaginaire reprise du plan initial. Un clone neuf doit réussir la recette explicite sans chemins personnels ni correction manuelle non documentée. [S13, S20, S21]

Le script 05 d'encodage CONFIRM ne doit pas être rejoué aveuglément : changement de clé d'extraction dans le même SAVE_DIR, liens symboliques vérifiés, checkpoints requis. Construire un mode « encoder seulement » et un répertoire d'évaluation distinct, ou documenter les préconditions avant de le remettre comme commande standard.

## 10. Après le départ : approfondissements conditionnels

### A. Comparaison applicative équitable

Élargir les requêtes et travailler sur un corpus réel autorisé, sans calibrer sur le même ensemble déjà examiné. Apparier le budget total de catalogue CORE/FULL, contrôler la diversité des parents et ajouter une méthode dense réellement conditionnée par l'axe pour le clustering. L'égalité du nombre de features sélectionnées ne supprime pas à elle seule l'effet d'un catalogue de candidats plus grand.

### B. Corrélations et nouveauté

Filtrer les quasi-synonymes et les relations définitionnelles séparément, puis comparer à une découverte lexicale sur les mêmes documents et le même budget d'hypothèses. Ne pas confondre une association forte et une information nouvelle pour un analyste. Le cas réfrigérateur/électricité est un point de départ à inspecter, pas une nouvelle règle métier.

### C. Carte de features

Nommer les groupes avec des exemples, conserver les isolats et les groupes instables, distinguer carte de directions et clusters d'emails. Étudier la stabilité sur un jeu non utilisé pour construire les groupes ; évaluer ensuite si la navigation aide réellement la recherche. Le seul objectif « rendre la carte plus jolie » ne clôt pas E05.

### D. Mémoire et 100M

Compléter le stockage compact jusque dans les lecteurs et le réencodage FULL ; profiler l'accès réellement contigu du réservoir et la libération des processus. Avant un gros run, distinguer mémoire anonyme, pages de fichiers, VRAM, disque et temps de classification. Le budget de campagne actuellement versionné n'autorise pas le 100M. Ne pas remplacer des données pertinentes manquantes par du filler générique et présenter cela comme 100M tokens d'emails. [S11, S19]

### E. Maintenance

Séparer progressivement lecture d'artefacts, extraction, apprentissage et évaluation. Versionner les schémas, supprimer les chemins personnels des recettes, fixer le comportement en cas d'entrée manquante et conserver une petite démo sans GPU. Archiver les wrappers devenus inutiles seulement après inventaire de leurs consommateurs ; ne pas supprimer les preuves des résultats négatifs.

## 11. Formulation à transmettre à l'encadrement

> La campagne post-soutenance apporte des applications évaluées sur des emails synthétiques et une meilleure compréhension de la stabilité. Elle montre un gain de l'extension par rapport au cœur seul en recherche par propriété, sans supériorité moyenne sur la baseline dense. Le principal travail restant est de sécuriser la provenance, corriger les incohérences de protocole et obtenir des retours humains. La passation doit transmettre un prototype exploratoire et ses preuves, avec les limites visibles, plutôt qu'une promesse d'industrialisation déjà validée.
