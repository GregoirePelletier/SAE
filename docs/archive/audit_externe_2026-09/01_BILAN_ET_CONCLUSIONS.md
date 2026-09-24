# SAE — Bilan de la campagne post-soutenance et conclusions pour EDF

**Date de lecture : 22 septembre 2026.** Dépôt : `GregoirePelletier/SAE`. Version auditée : `2f6fc2418e4ac084526dfc57b858431be426360b` ; point de départ du plan : `68c8e6863013c98905740a56e6491d78d05435cd`.

## 1. Verdict à présenter lors de la passation

Le projet a dépassé le stade des seuls démonstrateurs énergie/sport : une nouvelle campagne compare des représentations, cherche des propriétés dans des emails, confirme des hypothèses de diffing sur un autre ensemble et étudie des groupes de features. Le passage à Gemma-3-1B a été effectué dans cette campagne. Il ne s'agit pas d'une simple nouvelle série de sweeps.

**Le positionnement défendable reste un outil de recherche et d'exploration assistée, pas une solution métier validée ou un remplaçant démontré des moteurs classiques.** Le signal le plus intéressant est un gain de FULL sur CORE pour certaines recherches par propriété. Le signal le plus robuste pour la reprise est aussi négatif : ce gain ne rend pas FULL meilleur que la baseline dense en moyenne, et la stabilité géométrique des groupes n'assure pas la stabilité des emails retrouvés. [S03, S05, S07]

L'application concrète à privilégier est donc une comparaison ou une combinaison, **à évaluer**, entre un moteur dense/lexical pour retrouver les emails et des features SAE pour décrire les motifs et guider l'exploration. Cette architecture hybride est une recommandation de passation ; son bénéfice utilisateur n'a pas été mesuré par E08. [S05, S10]

Deux réserves conditionnent tout le bilan : les données restent synthétiques ; le manifeste révèle encore un rattachement positionnel des variantes aux parents. Les résultats ci-dessous sont ceux rapportés par le dépôt, sous ce contrat de données, et non des mesures GPU que j'aurais reproduites. [S02, S18]

## 2. Ce qui a effectivement changé depuis le plan

La comparaison GitHub identifie **42 commits et 68 fichiers ajoutés ou modifiés**. Les changements portent principalement sur `configs/post_stage/`, `docs/post_stage/`, `scripts/post_stage/`, `slurm/post_stage/`, `src/post_stage/`, le stockage documentaire et le dashboard.

Le `README.md`, le journal `RESULTS_TESTS.md`, les anciens documents d'architecture et les rapports historiques n'ont pas été réécrits dans cette différence. Les nouveaux résultats se trouvent dans `docs/post_stage/`, pas à la fin du journal historique. Il faut rendre ce chemin d'entrée explicite avant de remettre le dépôt.

| Étape du plan | Réalisation observée | Niveau atteint | Reste déterminant |
|---|---|---|---|
| E00 — Ressources | Profilage réel documenté ; construction des codes CORE sans les lignes filler | Correctif partiel et mesures à petite échelle | Réencodage FULL/chargement legacy encore à traiter ; pas de démonstration 100M |
| E01 — Comparaison | 1B/layer13/K5 ; CORE/FULL/DENSE/TF–IDF ; apprentissage FIT, évaluation DEV ; top-3 et longueur | Comparaison sur parents nominalement distincts, labels d'augmentation | Audit de jointure ; test final et annotation indépendante de la génération |
| E02 — Features | 150 candidates CORE + 150 EXTRA ; registre, trois statuts | Étiquetage automatique réalisé | Vérification humaine ; budget total de catalogue comparable dans les applications |
| E03 — Recherche | 12 requêtes, six propriétés, cinq moteurs, CONFIRM ; exemples conservés | Évaluation exploratoire par Qwen | Jugements humains, déduplication par parent, contrôle du budget de labels |
| E04 — Diffing | Panique/calme, découverte FIT, vérification CONFIRM, 150 textes par groupe | Protocole étendu au domaine emails synthétiques | Bug d'unité dans le prompt, prise en compte des parents, autres contrastes, humain |
| E05 — Stabilité/carte | Trois entraînements PCA ; directions/profils/groupes/sous-espaces ; carte descriptive | Résultat de stabilité sous initialisation commune | Résultats random-init non documentés ; stabilité documentaire non établie |
| E06 — Corrélations | Découverte FIT+DEV, huit paires, 350 parents CONFIRM, propriétés vérifiées | Associations documentées, nouveauté limitée | Incohérence de comptage ; baseline lexicale ; audit humain |
| E07 — Clustering ciblé | Trois axes, 400 parents, quatre méthodes, budget 40 features | Contrôle mécanique de l'axe ; résultat sémantique variable | Comparateur dense réellement conditionné par la requête ; validation humaine |
| E08 — Mini-pilote | Page CPU, affichage des résultats, formulaire de pistes | Recette technique | Sessions non réalisées ; catalogue de pistes versionné vide |
| E09 — 100M | Aucune restitution E09 dans les fichiers consultés | Non documenté, budget optionnel fixé à zéro | Nouvelle autorisation et justification ; ne doit pas bloquer la passation |

Sources : S02 à S12, S19, S22, S26. « Réalisé » ne signifie jamais ici « validé métier ».

## 3. Données : progrès du découpage et réserve de provenance

Le manifeste indique 3 474 parents, répartis en 2 084 FIT, 521 DEV et 869 CONFIRM. Avec les variantes rattachées, cela donne 25 970 documents FIT, 6 518 DEV et 10 865 CONFIRM. La graine de split est indépendante des graines SAE : `20260914`. Le SAE de référence est entraîné sur FIT, avec un réservoir de 8M tokens rempli, sans filler générique. [S02, S03, S20]

Cependant, les champs suivants sont explicites :

```json
{"method": "positional_via_parent_id", "positional_join_fallback": true,
 "n_variants_unmatched": 70}
```

Le code construit `pos_to_hash` en énumérant les parents **après chargement/nettoyage**, puis utilise le `parent_id` des variantes lorsque `parent_sha1` est absent. Le hash stabilise le split des parents ; il ne prouve pas à lui seul que chaque variante a été rattachée à son véritable parent de génération. [S18]

**Cela n'établit pas une fuite avérée.** Il faut retrouver l'ordre source utilisé lors de l'augmentation, expliquer les 70 variantes écartées et vérifier le rattachement des variantes conservées. Écrire seulement « aucun parent dans deux splits » vérifierait la cohérence du mapping reconstruit, pas nécessairement son exactitude. Les hashes de contenu des fichiers source et des variantes doivent aussi être figés : les affectations actuelles utilisent notamment `aug_id`, sans garantir qu'un texte portant le même identifiant n'a pas changé. [S02, S18]

## 4. Les conclusions scientifiques que l'on peut maintenant défendre

### 4.1 E01 : un signal discriminant, mais pas de gain de l'extension sur cette tâche

La nouvelle sonde porte sur **14 classes de génération/augmentation** : émotion, urgence, registre, orthographe et original. Ce ne sont pas les cinq intentions de l'ancien rapport, ni quatorze catégories métier annotées indépendamment.

| Représentation | Exactitude DEV |
|---|---:|
| CORE | 88,68 % |
| FULL | 88,80 % |
| TF–IDF | 87,94 % |
| DENSE bge-m3 | 77,05 % |

FULL−CORE = +0,12 point ; intervalle bootstrap par parent [−0,08 ; +0,32]. Pas de gain établi de l'extension sur cette sonde. FULL−TF–IDF = +0,86 point, mais la comparaison ne survit pas à BH (`p_fdr=0,056`). FULL dépasse DENSE sur cette tâche, pas nécessairement sur d'autres usages. [S03]

La moyenne des trois plus fortes activations ne change pas significativement les résultats de cette sonde. Une baseline « longueur seule » atteint 18,9 %, loin des représentations principales. **Cela ne résout pas tous les biais de longueur** : un classifieur faible utilisant la longueur seule n'exclut pas une interaction entre longueur et contenu, ni un biais du maximum pour le retrieval ou les corrélations. [S03]

### 4.2 E02 : un nouveau 197/300, à ne pas confondre avec l'ancien

Le registre de cette campagne contient 197 statuts `interpretable`, 102 `unclear` et un support insuffisant. La sélection est de 150 candidates CORE et 150 EXTRA. Ce 197/300, soit 65,7 %, appartient à **un autre modèle, un autre apprentissage et un autre échantillon** que le 197/300 du rapport soutenu. L'égalité numérique n'autorise ni à fusionner les runs ni à présenter une réplication parfaitement appariée. La vérification humaine est explicitement en attente. [S04]

### 4.3 E03 : premier signal applicatif de FULL sur CORE, pas de victoire sur les moteurs classiques

Six propriétés ont chacune deux formulations : relances répétées, menace de résiliation, incident collectif, explication d'un montant, coupures répétées, urgence implicite. Les top-10 sont jugés par Qwen selon une définition de la propriété ; seul le score 2 compte comme pertinent dans P@10 strict. [S05, S14]

Moyennes recalculées à partir des douze lignes publiées dans `e03_results.md` :

| Moteur | P@10 strict moyen |
|---|---:|
| DENSE bge-m3 | 65,0 % |
| BM25 lexical | 65,0 % |
| TF–IDF | 57,5 % |
| FULL | 49,2 % |
| CORE | 24,2 % |

FULL−CORE = +25 points ; intervalle par famille publié [+5,8 ; +45]. FULL−DENSE = −15,8 points, intervalle [−26,7 ; −6,7]. Ce sont des résultats **exploratoires**, avec seulement six familles et un juge automatique. [S05]

Le contraste CORE/FULL porte aussi sur **77 labels utilisables contre 197**. Il décrit donc l'intérêt du système complet et de son catalogue, mais ne sépare pas encore proprement l'apport des directions supplémentaires de celui d'un budget de descriptions plus large. Le code charge `confirm_groups` sans imposer une déduplication par parent du top-10 : plusieurs variantes d'un même email peuvent compter comme plusieurs résultats. Ce point est important pour l'utilité d'une liste présentée à un analyste. [S14]

Exemples utiles : FULL obtient 0,90 sur les deux formulations de relances répétées, 0,80/0,70 sur l'explication d'un montant, 1,00/0,90 sur l'urgence implicite. Ces chiffres ne démontrent pas la nouveauté des résultats pour le métier. Pour l'incident collectif, FULL et CORE restent à zéro ; certaines baselines retrouvent pourtant des résultats, notamment TF–IDF à 0,70 sur la paraphrase. L'échec SAE ne permet donc pas d'affirmer l'absence de cette propriété dans le corpus. [S05]

**Correction de lecture explicitement signalée :** le texte E03 affirme qu'un « vrai P@10 » demanderait de juger tout le corpus. Ce n'est pas ce que calcule son propre script : pour P@10 d'une requête et d'une liste, juger exhaustivement ses dix premiers résultats suffit. L'union des top-10 couvre ces listes. Les limites sont ici la qualité du juge, la diversité des requêtes et les doublons ; le rappel et la MAP globale demandent d'autres informations sur les pertinents non retrouvés. L'ancien calcul local d'AP n'a pas été corrigé par le simple fait que cette nouvelle expérience utilise P@10. [S05, S14]

### 4.4 E04 : le diffing fonctionne sur un contraste synthétique, avec une erreur de contrat à corriger

On n'est plus seulement sur énergie/sport. Le contraste est `urgence__panique` contre `urgence__calme`, sur des variantes générées. Huit hypothèses sont proposées sur FIT, puis jugées sur 150 textes par groupe de CONFIRM. Le tableau publié indique six écarts significatifs dans le sens annoncé, un écart significatif dans le sens inverse et un non-significatif. Cela illustre l'intérêt d'une confirmation séparée, mais pas encore la découverte d'un problème industriel inconnu. [S06]

**Problème de code confirmé pendant cette lecture :**

```python
"percentage_difference": float(row["log_odds_ratio"])
```

Dans `e04_diffing.py`, le générateur reçoit un log-odds-ratio sous un nom et un prompt qui demandent un écart de fréquence entre −1 et 1. `corpus_diff_stats` fournit pourtant séparément `freq_A` et `freq_B`. Le contrat devrait transmettre `freq_A - freq_B`, ou changer explicitement le prompt et le schéma pour accepter un log-odds. [S15, S16, S17]

Cette erreur touche la **génération** des hypothèses et leur intensité annoncée. Elle ne change pas rétroactivement les comptes de présence mesurés pour une hypothèse déjà gelée, mais il faut marquer cette version du générateur et évaluer la version corrigée séparément. Son impact exact sur la liste générée n'est pas mesuré ici.

Autres points de reprise : le script charge les identifiants parents puis échantillonne A et B séparément ; les tests de proportions ne modélisent pas les parents éventuellement communs aux deux groupes. Il ne sauvegarde pas la matrice complète de jugements ni les IDs du sous-échantillon dans le JSON final. L'appariement et la reproductibilité de la vérification doivent donc être consolidés. [S15]

Le texte E04 dit aussi que `coverage` ignore le signe. Il faut distinguer les deux métriques : `verification_rate` utilise bien la différence absolue ; l'implémentation actuelle de la couverture ne retient que les hypothèses favorables à la cible. Le code contredit donc cette partie de la prose, qui doit être corrigée. [S27]

### 4.5 E05 : vrai progrès de stabilité, dans un régime précisément délimité

Trois entraînements 1B partagent le même cœur et le même point de départ PCA de l'extension. L'analyse exclut le cœur — identique par construction — et les features insuffisamment supportées ou quasi universelles. Environ 30 % des 1 024 features EXTRA sont quasi universelles et exclues du test principal. [S07]

Parmi les 668 features supportées de la référence, 485 sont retrouvées dans les deux autres répétitions selon le double critère cosinus ≥0,7 et corrélation de profils ≥0,5. Les groupes présentent une pureté et un recouvrement de sous-espace supérieurs aux groupes témoins spécifiés. En revanche, **aucun groupe ne dépasse son témoin pour le recouvrement des 100 premiers emails**, selon les comparaisons publiées. [S07]

Conclusion : stabilité géométrique et de profils sous initialisation PCA commune ; stabilité des listes d'emails par groupe non démontrée. La carte est descriptive. Ce résultat ne prouve pas une hiérarchie sémantique ni des groupes nommables utiles à un métier. Il ne contredit pas mécaniquement les faibles Jaccard de libellés historiques : objets, modèle et méthode d'alignement ont changé.

**État du développement plus récent que la prose E05 :** les commits du 21 septembre ajoutent `EXTRA_DECODER_INIT=random`, deux scripts pour les graines 45/46 et une analyse PCA→PCA / PCA→random / random→random. Le script `15_e05_stability_random_init.slurm` produit `e05_stability_random_init.json`. Aucune restitution chiffrée de ce bras n'apparaît dans les documents de résultats lus. « Non implémenté » est donc périmé ; « résultats à récupérer/vérifier » est le statut approprié. [S07, S22]

### 4.6 E06 : des associations vérifiées, peu de découvertes métier nouvelles

Huit paires issues d'une découverte FIT+DEV sont vérifiées sur 350 parents CONFIRM. Les propriétés sont jugées séparément, puis leurs cooccurrences sont calculées. C'est un progrès par rapport à un simple graphe d'activations. [S08, S24]

La paire « dysfonctionnement électrique × réfrigérateur » est une piste descriptive concrète : les 13 emails jugés « réfrigérateur » dans cet échantillon sont aussi jugés « dysfonctionnement électrique », NPMI publié 0,35. On peut proposer une inspection des effets d'incidents sur les équipements ; **on ne peut pas en déduire un problème récurrent dans la clientèle réelle ou une règle automatique de routage**.

Les autres associations fortes sont souvent attendues ou redondantes : IBAN/coordonnées bancaires, SIREN/identification SIREN. Plusieurs features quasi synonymes ne constituent pas automatiquement une démonstration de splitting au sens d'une division mesurée lors d'un changement de dictionnaire.

**Incohérence à résoudre avant présentation :** le tableau classe quatre paires « établi », trois non établies et une à support insuffisant ; la synthèse annonce cinq, deux et une. Le JSON local `e06_correlations.json`, les p-values et les intervalles doivent arbitrer. Ne reprendre ni « cinq » ni « quatre » comme total définitif avant ce contrôle. Le test ne comporte pas de baseline lexicale de découverte ni de budget CORE/FULL équilibré. [S08]

### 4.7 E07 : modifier la partition ne suffit pas à répondre à la question

Le test utilise trois axes, 400 parents, quatre clusters et 40 features sélectionnées pour CORE/FULL. FULL donne des groupes assez cohérents pour `type_probleme` (réassignations par le juge entre 0,72 et 0,97), mais `action_attendue` et surtout `registre_urgence` ne sont pas correctement reflétés dans tous les groupes. [S09]

DENSE et TF–IDF ne sont pas conditionnés par l'axe dans ce protocole : chacun conserve sa propre partition d'un axe à l'autre. **Ce comportement ne démontre pas que les embeddings denses sont incapables de clustering ciblé** ; le contrôle de la requête ne leur a simplement pas été donné. Les réassignations LLM ne sont pas une vérité terrain et les algorithmes de clustering diffèrent selon les bras. Ce résultat permet une démo prudente sur `type_probleme`, pas une revendication générale de supériorité.

### 4.8 E08 et E09 : le calcul ne remplace pas la validation humaine

La page `Mini-pilote analyste (E08)` est construite, lit des JSON et ne charge pas de LLM pour afficher les résultats. Les sessions avec participants n'ont pas été réalisées, et `docs/post_stage/e08_pilot_leads.json` vaut `[]`. Il n'existe donc pas de mesure d'utilité utilisateur issue de cette campagne versionnée. [S10, S26]

Aucun résultat E09/100M n'est documenté ; la politique fixe `optional_100m_gpu_hours_cap: 0`, c'est-à-dire qu'une nouvelle décision est nécessaire. L'absence d'un résultat versionné ne prouve pas qu'aucun essai local n'a eu lieu. Il faut demander l'inventaire local avant de conclure sur l'exécution réelle. [S19]

## 5. A-t-on des cas d'usage business concrets ?

**Oui pour des scénarios opérationnels et des démonstrateurs ; non pour une valeur business mesurée ou un déploiement validé.**

| Cas d'usage proposé | Ce qui existe | Ce que doit mesurer un pilote |
|---|---|---|
| Retrouver des relances restées sans solution | E03, requêtes explicites et paraphrases, exemples consultables | Pistes réellement pertinentes, diversité des parents retrouvés, utilité par rapport à la recherche actuelle |
| Distinguer une demande d'explication d'une demande de remboursement | E03, propriété contrastive et contre-exemples possibles | Erreurs de lecture, qualité des justifications, apport des features par rapport au dense |
| Comparer deux populations d'emails pour faire émerger leurs différences | E04, procédure découverte/confirmation sur panique/calme | Nouveauté des thèmes face à la taxonomie métier, robustesse à la composition des groupes |
| Explorer des effets d'incidents sur des équipements | E06, piste réfrigérateur/électricité à support limité | Réalité des exemples et contexte ; intérêt de l'association au-delà d'une cooccurrence lexicale |
| Regrouper un ensemble selon le type de problème | E07, meilleur axe actuellement documenté | Cohérence humaine, couverture, cas ambigus, comparaison équitable aux méthodes existantes |

Le scénario « incident × région × période » reste une **transposition prospective**. Aucun résultat actuel ne démontre sa détection sur un flux EDF réel. Les métadonnées de lieu/date, la composition des groupes et les faux positifs restent à traiter.

## 6. Décision de reprise recommandée

Avant de refaire des calculs lourds, sécuriser la provenance, préserver les artefacts et organiser un mini-pilote sur les requêtes les plus pertinentes. Il faut être prêt à conserver un moteur dense/lexical, à utiliser le SAE seulement comme couche d'exploration, ou à abandonner une fonction si les utilisateurs n'y gagnent rien.

**Ce qui n'est pas une priorité de passation :** nouveaux sweeps, recherche d'un meilleur taux odd-one-out, un entraînement 100M sans tâche cible, une carte 2D plus esthétique sans contrôle, ou une réécriture générale de tout le monolithe avant d'avoir figé ses entrées/sorties.

Voir `02_PASSATION_TECHNIQUE.md` pour les fichiers à remettre et `03_REPRISE_PRIORISEE.md` pour les tâches et critères d'acceptation.

## Sources et périmètre

Les codes S01–S27 sont résolus dans [SOURCES.md](SOURCES.md), avec des liens GitHub figés. Ce bilan distingue les résultats **rapportés**, les constatations **de code**, les **incohérences documentaires** et les **recommandations nouvelles**. Les caches de calcul, les dernières sorties Slurm et la disponibilité actuelle du cluster n'ont pas été consultés directement. Aucun script GPU du dépôt n'a été exécuté dans cet audit.
