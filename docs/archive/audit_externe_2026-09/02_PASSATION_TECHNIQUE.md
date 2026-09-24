# SAE — Guide de passation du code et des artefacts

**Version auditée : `2f6fc2418e4ac084526dfc57b858431be426360b`.** Ce guide prépare une reprise ; il ne certifie pas qu'un clone vierge reproduit déjà les expériences. Les liens de preuve sont dans [SOURCES.md](SOURCES.md).

## 1. Les cinq choses à expliquer au repreneur en premier

1. **Le besoin est exploratoire.** Retrouver et décrire des motifs dans des emails, sans définir à l'avance une taxonomie exhaustive. Il ne s'agit pas d'automatiser de nouveau le traitement des emails.
2. **Le livrable est un prototype de recherche.** Les données évaluées sont synthétiques. Les chiffres viennent d'évaluations automatiques ; la validation par des analystes n'a pas encore eu lieu.
3. **Deux générations de résultats coexistent.** Le rapport soutenu et `RESULTS_TESTS.md` documentent la campagne historique ; `docs/post_stage/` contient la campagne 1B FIT/DEV/CONFIRM. Ne pas mélanger leurs paramètres et résultats.
4. **GitHub ne contient pas toute l'expérience.** Checkpoints, poids des modèles, caches, matrices documentaires, résultats JSON et extraits sont en grande partie locaux et ignorés par Git. La passation doit transférer leurs emplacements, leur identité et les droits d'accès.
5. **La première reprise n'est pas un entraînement.** Elle consiste à lire les résultats figés, ouvrir les deux pages de démonstration, vérifier les artefacts, puis seulement décider du prochain calcul.

## 2. Lecture du dépôt en une heure

| Ordre | Fichier ou répertoire | Question à laquelle il répond |
|---|---|---|
| 1 | `docs/post_stage/e01_results.md` à `e08_recette.md` | Que sait-on maintenant, avec quel niveau de preuve ? |
| 2 | `configs/post_stage/corpus_manifest.json`, `corpus.yaml`, `split_assignments.json` | Quels textes et parents appartiennent à chaque rôle ? |
| 3 | `configs/post_stage/campaign_policy.yaml` | Quelles limites d'autonomie, de ressources et d'export sont définies ? |
| 4 | `src/post_stage/dataset_contract.py` | Comment construit-on et recharge-t-on les splits ? |
| 5 | `src/sae/frozen_core.py`, `src/sae/batch.py` | Qu'est-ce qui est gelé, qu'est-ce qui apprend et quelle est la parcimonie ? |
| 6 | `src/sae/saev5.py` | Comment s'enchaînent extraction, entraînement, réencodage et analyses ? |
| 7 | `scripts/post_stage/` | Comment produit-on chaque résultat E01–E07 ? |
| 8 | `src/visualization/dashboard.py` | Comment les résultats sont-ils présentés et quelles pages sont disponibles ? |
| 9 | `slurm/post_stage/`, `tests/post_stage/` | Comment lancer et quelles invariants tester ? |
| 10 | `README.md`, `docs/architecture.md`, `docs/ops.md`, `CLAUDE.md` | Documentation générale et historique, à lire avec les écarts signalés dans le bilan |

Le `README.md` n'est pas encore une synthèse fiable des résultats actuels : il affiche notamment 89,3 % comme référence et K_EXTRA=32 dans son tableau. La passation doit commencer par les nouveaux documents, puis réparer cette porte d'entrée. [S01]

## 3. Carte de l'architecture

### 3.1 Chaîne principale

```text
Emails source + variantes
        │
        ├── contrat de données : parent → FIT / DEV / CONFIRM
        │
        └── Gemma-3 gelé : activations de tokens x
                 │
                 ├── GemmaScope gelé → code CORE → reconstruction du cœur
                 │
                 └── extension entraînée → code EXTRA → reconstruction du résidu
                           │
                           └── code FULL = concaténation CORE + EXTRA
                                            │
                                   agrégation par email
                                            │
                     sondes / retrieval / diffing / groupes / corrélations / clustering
                                            │
                                  JSON + vues Streamlit
```

Dans la campagne post-soutenance : Gemma-3-1B, couche 13, cœur 16 384 features, extension 1 024 features, K_EXTRA=5, dix époques, réservoir plafonné à 8M tokens, sans filler. E01 n'est pas le run historique 12B/25M. [S03, S20]

L'encodeur EXTRA lit **x**, pas le résidu. Le décodeur EXTRA apprend à reconstruire `x − reconstruction_core`. Seuls les paramètres supplémentaires sont entraînés. Le code FULL sert aux analyses ; additionner les deux reconstructions sert à évaluer la fidélité. Ce sont deux opérations différentes.

Le nom `p1_raw_residuals.memmap` est historique : ce cache fournit les activations brutes utilisées pour calculer la cible résiduelle, pas une raison de donner directement le résidu à l'encodeur. Le chargement de checkpoints vérifie notamment la convention `encoder_input="x"`. [S23]

### 3.2 Alternative sur phrases

Le Pipeline 2 découpe les emails en phrases, extrait des embeddings avec F2LLM-v2 ou bge-m3 gelé, entraîne un `PhraseLevelSAE` depuis zéro et agrège ses codes. Ce n'est ni le cœur du Pipeline 1 ni sa baseline dense. Il n'a pas été rejoué dans la nouvelle série E00–E08 et ne doit pas être présenté comme une nouveauté de celle-ci.

### 3.3 Modules post-soutenance

| Module | Responsabilité | À ne pas lui attribuer |
|---|---|---|
| `src/post_stage/cli.py` | Commande `freeze-corpus` | Une CLI universelle train/encode/eval/pilot : ces commandes n'existent pas ici |
| `dataset_contract.py` | Manifeste, split des parents, chargement FIT/DEV/CONFIRM | Garantie de rattachement sémantique des variantes historiques sans hash |
| `representations.py` | Construction TF–IDF et embeddings bge-m3 | Nouvelle architecture SAE |
| `resources.py` + `scripts/post_stage/profile_run.py` | Mesure des ressources, enveloppe d'exécution | Preuve qu'un run 100M est autorisé ou passe en mémoire |
| `stability.py` + `e05_stability.py` | Appariements, groupes, métriques et témoins | Identité de concepts ou hiérarchie métier validée |
| `scripts/post_stage/e01…e07` | Expériences dédiées | Une API de production stable |

Le monolithe reste utilisé et plusieurs scripts importent des fonctions de `saev5.py`, avec manipulation de `sys.path`. Une modularisation ciblée est souhaitable, mais ne doit pas faire perdre la capacité à relire les artefacts existants. [S13, S14]

## 4. Contrat des données : la première condition de reprise

### 4.1 Rôles et volumes

- FIT : apprentissage de l'extension et ajustement de la représentation supervisée/lexicale E01 ; 2 084 parents, 25 970 documents.
- DEV : choix et diagnostics ; 521 parents, 6 518 documents.
- CONFIRM : 869 parents, 10 865 documents ; utilisé par E03/E04/E06/E07, donc **déjà consulté pour ces tâches**.

Les variantes ne sont pas autant d'observations indépendantes. Les évaluations de listes et les tests doivent préciser si leur unité est le document ou le parent. Une future correction décidée après lecture de CONFIRM ne peut pas présenter ce même ensemble comme un test intégralement neuf. [S02, S03, S05–S09]

### 4.2 Deux modes d'évaluation distincts

E01 ajuste TF–IDF et les sondes sur FIT, puis évalue DEV. E03 construit l'index TF–IDF/BM25 et la normalisation sur la collection CONFIRM à rechercher ; il s'agit d'une piste d'indexation exploratoire, non du protocole FIT-only. Ce n'est pas automatiquement une erreur, mais il ne faut pas fusionner ces deux niveaux de généralisation. [S03, S05, S14]

### 4.3 Vérification de provenance exigée

Le repreneur doit disposer du fichier d'augmentation et de son ordre de parents d'origine, pas seulement du manifeste dérivé. Vérifier :

- pourquoi 70 variantes n'ont pas trouvé de parent ;
- si les six parents nettoyés/dédupliqués avant le split changent la relation entre `parent_id` et l'ordre de génération ;
- qu'un hash calculé a posteriori ne masque pas une erreur de correspondance positionnelle ;
- qu'une variante conservée ne se retrouve pas dans un autre split que son vrai parent ;
- que les textes et `aug_id` n'ont pas changé depuis les encodages ;
- que les permutations d'ordre des fichiers ne changent pas silencieusement cette correspondance.

Si ces éléments manquent, annoter les résultats « provenance des variantes à auditer », ne pas certifier arbitrairement le split. Ne pas réécrire les affectations pour faire passer un test. [S02, S18]

## 5. Artefacts à transmettre : le clone ne suffit pas

### 5.1 Paquet minimal pour une démonstration CPU

Depuis le run `results_post_stage_e01_fit_1b_layer13_k5/`, récupérer sous contrôle d'accès :

```text
results.json
# Expériences
 e01_representation_comparison.json
 e01_pooling_and_length_analysis.json
 e02_feature_registry.json
 e03_property_retrieval.json
 e04_diffing.json
 e05_stability.json
 e06_correlations.json
 e07_clustering.json
# À récupérer si réellement produit
 e05_stability_random_init.json
```

Les fichiers E02/E03 et le catalogue E08 peuvent contenir des extraits. Ils doivent rester dans le périmètre autorisé ; ne pas les copier automatiquement dans Git ni dans un assistant externe. Les noms ci-dessus sont les noms attendus des artefacts, pas la preuve qu'ils existent sur la machine du repreneur. [S04–S10, S22]

L'interface E08 n'a pas besoin de charger un modèle pour afficher les JSON E01/E03/E04. La page E05 nécessite son résultat de stabilité. Tester le chargement réel après transfert, y compris les métadonnées générales attendues par le sélecteur de runs. [S10, S07]

### 5.2 Paquet pour reproduire les calculs

Ajouter :

```text
p1_extended_sae.pt
p1_frozen_core_d1024_k5.pt
cache/p1_all_doc_acts.pt
cache/p1_all_doc_acts_ext_d1024.pt
cache/p1_raw_residuals.memmap
cache/p1_raw_residuals.memmap.meta.json
cache/p1_token_fragments/
cache/p1_token_fragments_ext/
cache/resources_profile*.jsonl
cache/pipeline_stdout*.log
```

Les dépendances exactes diffèrent selon l'expérience. Ne pas tout charger en mémoire « pour vérifier ». Le memmap, les fragments et les poids peuvent être volumineux. Vérifier les liens symboliques : plusieurs entrées de `cache/` pointent vers `local_data/activation_cache/<clé>/`, hors du run. Une copie qui ne préserve que le lien sans sa cible est inutilisable sur un autre compte. [S23]

Pour E05, préserver aussi les runs `results_post_stage_e05_seed43/`, `results_post_stage_e05_seed44/` et, s'ils existent, `results_post_stage_e05_randinit45/` et `…46/`. Le script d'analyse random utilise la référence, seed43, rand45 et rand46 ; il n'emploie pas toutes les répétitions PCA. [S22]

### 5.3 Identité et environnement

Pour chaque paquet, enregistrer : commit code, état local non commité, versions des sous-modules, hashes de `uv.lock` et des petits manifestes, modèle et révision exacte des poids, configuration du SAE cœur, paramètres EXTRA, bras d'initialisation, seed, split, ordre des documents, masque, agrégation, modèle juge, résultats produits et statuts.

Le dépôt fixe Python **3.12** dans `pyproject.toml` et fournit `uv.lock`, mais les versions minimales déclarées ne suffisent pas à décrire un environnement historique. Archiver l'environnement réellement utilisé et les artefacts de poids. Ne pas mettre à jour les dépendances au moment de la passation. [S25]

L'appellation du juge dans les fichiers est `Qwen3.8-27B` ; la personne qui reprend doit disposer de son chemin local, du vrai identifiant de modèle et de la révision des poids, pas seulement de ce nom de convention. Ne pas renommer ni substituer silencieusement le juge.

## 6. Parcours de lancement vérifié dans le dépôt

### 6.1 Lecture seule, avant toute soumission

Sur un compte autorisé et dans la racine du clone :

```bash
git rev-parse HEAD
git status --short
git submodule status
python --version
# Le script fourni avec ce dossier ne charge aucun modèle et ne soumet aucun job.
python /chemin/vers/ce_dossier/outils/preflight_passation_sae.py \
  --repo "$PWD" \
  --run-dir "$PWD/results_post_stage_e01_fit_1b_layer13_k5" \
  --out /chemin/autorise/preflight_passation.json
```

Le script ne remplace ni une vérification de provenance ni un test scientifique. Il signale les fichiers absents, liens cassés, marqueurs de prudence et quelques incohérences vérifiables sans torch.

### 6.2 Dashboard

Dans l'environnement de travail existant :

```bash
.venv/bin/python -m streamlit run src/visualization/dashboard.py
```

Choisir le run post-stage. Ouvrir d'abord `Mini-pilote analyste (E08)` et `Stabilite inter-graines (E05)` (orthographe du menu à vérifier dans l'interface), puis revenir aux pages historiques au besoin. Ne pas démontrer une nouvelle étude en sélectionnant par erreur un run 12B ancien.

Le formulaire E08 écrit actuellement dans `docs/post_stage/e08_pilot_leads.json`. Ce n'est donc plus une interface intégralement en lecture seule dès qu'on soumet une piste. Avant un pilote sur de vrais emails, déplacer cette écriture vers un emplacement autorisé hors Git et traiter la concurrence de sessions et la traçabilité des modifications. [S10]

### 6.3 Tests

```bash
.venv/bin/python -m pytest tests/post_stage/ -q
.venv/bin/python -m pytest tests/test_doc_acts_sparse_filler.py \
  tests/test_doc_topk_mean_pool.py tests/test_held_out_probe_accuracy.py \
  tests/test_saeboost_decoder_init.py -q
# Ensuite : suite complète, dans les conditions CPU autorisées du site.
.venv/bin/python -m pytest tests/ -q
```

Ces commandes sont une recette à exécuter. Je n'ai pas exécuté la suite actuelle. Le `preflight_report.md` ancien rapporte 292 tests réussis et un échec documentaire sur une édition locale ; ce n'est **pas** le résultat de HEAD actuel. Ne pas reprendre ce chiffre comme état de CI. [S12]

### 6.4 Chaîne Slurm existante

Lire et adapter les chemins avant tout `sbatch`. Les scripts contiennent `/home/h21486/SAE`, des répertoires de sorties et des partitions propres au site. Les ressources ci-dessous ne sont pas une autorisation permanente.

| Étape | Script existant | Dépendances et prudence |
|---|---|---|
| Profilage | `slurm/post_stage/00_profile.slurm` | Borne de corpus et de tokens ; ne pas lancer le pipeline complet sur le frontal |
| Référence FIT | `01_e01_fit_reference.slurm` | Manifeste existant, modèles locaux, SAVE_DIR distinct ; 96G et huit heures dans le script lu |
| Sondes | `02_e01_compare_representations.slurm` | Codes FIT/DEV, baseline dense, paramètres E01 |
| Pooling/longueur | `03_e01_pooling_and_length.slurm` | Fragments et mêmes splits |
| Registre | `04_e02_feature_registry.slurm` | Features candidates, exemples FIT, juge local |
| Encodage CONFIRM | `05_e01_encode_confirm.slurm` | Checkpoint FIT figé ; attention au changement de clé de cache |
| Retrieval | `06_e03_property_retrieval.slurm` ou `06b_…_a100.slurm` | Codes CONFIRM, registre, bge-m3, juge |
| Diffing | `07_e04_diffing.slurm` ou `07b_…_a100.slurm` | Corriger/versionner le contrat de génération avant nouvelle étude |
| Corrélations | `08_e06_correlations.slurm` | FIT+DEV découverte, parents CONFIRM vérification |
| Clustering | `09_e07_clustering.slurm` | Registre, codes CONFIRM ; conserver le budget de 40 et justifier tout changement |
| Répétitions PCA | `10_e05_seed43.slurm`, `11_e05_seed44.slurm` | Cache d'extraction partagé, SAVE_DIR privés |
| Analyse PCA | `12_e05_stability.slurm` | Checkpoints et activations des répétitions |
| Répétitions random | `13_e05_randinit45.slurm`, `14_e05_randinit46.slurm` | Statut réel des jobs à récupérer ; ne pas dupliquer un calcul déjà terminé |
| Analyse des bras | `15_e05_stability_random_init.slurm` | Quatre runs ; sortie `e05_stability_random_init.json` |

Les noms proviennent de la comparaison de fichiers et des scripts lus. Tous n'ont pas été rejoués ou inspectés intégralement dans cet audit ; lire le contenu effectif avant exécution.

### 6.5 Piège concret de l'encodage CONFIRM

Le script 05 réutilise le SAVE_DIR de référence et active `POST_STAGE_INCLUDE_CONFIRM_AS_DIFF=1`. Ce changement ajoute les textes CONFIRM à la clé du cache d'extraction. Le code `saev5.py` refuse pourtant un lien symbolique existant qui cible une autre clé. Une reprise depuis un cache FIT/DEV peut donc échouer sur cette précondition. [S21, S23]

Ne pas supprimer le cache partagé pour « débloquer ». Préférer une étape d'encodage explicite dans un répertoire d'évaluation distinct, avec les checkpoints figés copiés/identifiés et les liens d'extraction créés pour ce corpus. Documenter les préparations manuelles qui avaient permis le run historique. À défaut d'une commande dédiée, une recette de migration réversible est nécessaire.

La présence des checkpoints fait aujourd'hui partie du contrat implicite « encode, ne réentraîne pas ». Ajouter un mode qui échoue explicitement si ces fichiers manquent, au lieu de démarrer un apprentissage par inadvertance.

## 7. Ressources et exploitation sur le cluster

La politique de campagne prévoit un job GPU simultané, au plus deux GPU par job, un job CPU simultané, huit CPU, 128 Gio de RAM sans nouvel accord, un plafond principal de 120 GPU-heures et **zéro GPU-heure autorisée pour l'option 100M**. Confirmer le budget restant et l'accord de la personne responsable avant de reprendre. [S19]

Le fichier de politique rapporte des A100 d'environ 40 Go et des H100 d'environ 80 Go sur ce site. Le juge bf16 ne tient pas sur un seul des A100 observés ; les scripts de repli demandent deux A100 et `device_map="auto"`. Ne pas supposer qu'une partition appelée A100 contient nécessairement des cartes 80 Go. Ces caractéristiques documentées ne renseignent pas sur la disponibilité au moment du prochain lancement. [S19]

Commandes de constat local, à lancer avec les droits habituels :

```bash
squeue -u "$USER"
sinfo
# Renseigner les IDs après vérification dans le carnet local.
sacct -j <job_id> --format=JobID,State,Elapsed,ReqMem,MaxRSS,ExitCode
```

Un `TIMEOUT`, un OOM hôte et un OOM CUDA sont trois pannes distinctes. E00 a documenté un profilage tué par timeout pendant les sondes, pas un OOM. Le chargement mappé des poids du juge contribue aussi à la mémoire `file`/RSS. La correction partielle des lignes filler ne permet pas d'affirmer que tous les problèmes mémoire sont résolus. [S11]

Le diagnostic contient une mesure isolée de construction 690 Mo →65 Mo, puis un run réel plus petit terminé en 35 min 39 s. Il ne faut pas comparer sans précaution ce dernier au run de corpus complet : corpus et état du cache OS ont changé. Le réencodage FULL et les lecteurs legacy restent des postes à auditer avant tout retour au filler massif. [S11, S20]

## 8. Démonstration de passation recommandée — 15 minutes

**0–3 min : cadre.** Dire ce que représentent FIT/DEV/CONFIRM, pourquoi les données restent synthétiques, et montrer la version du run. Expliquer CORE/FULL/DENSE/TF–IDF sans entrer dans les matrices.

**3–7 min : retrouver une propriété.** Ouvrir E08 sur les relances répétées. Montrer FULL et DENSE ou BM25, puis un vrai résultat et un contre-exemple. Faire constater qu'un meilleur score que CORE ne signifie pas battre le dense. Vérifier qu'il ne s'agit pas de variantes du même parent présentées comme dix clients différents.

**7–10 min : comparer des populations.** Montrer E04 panique/calme ; lire une hypothèse dans le bon sens et celle qui s'inverse. Dire que la version actuelle du générateur a un problème d'unité à corriger et que les comptes sont des jugements LLM.

**10–13 min : carte E05.** Cliquer une feature/un groupe ; montrer les métriques d'appariement et le rappel « PCA commune ». Faire distinguer géométrie stable et top-emails non stables.

**13–15 min : reprise.** Faire exécuter le preflight par la personne qui reprend, vérifier les accès et lui faire désigner le fichier source à modifier pour le prochain correctif. Ne pas lancer un entraînement long pendant la réunion.

## 9. Réunion de passation — 60 minutes

| Temps | Sujet | Sortie attendue |
|---|---|---|
| 0–10 min | Besoin, périmètre, bilan | Accord sur ce qu'on sait et ce qu'on ne sait pas |
| 10–25 min | Démonstration ci-dessus | La personne retrouve les pages et les résultats de référence |
| 25–40 min | Code, données, artefacts | Chemins et responsabilités identifiés ; blocages d'accès consignés |
| 40–50 min | Bugs prioritaires et suite | Choix de trois tâches maximum pour la prochaine itération |
| 50–60 min | Reprise par le successeur | Il ouvre seul le dashboard, lit un résultat et explique son prochain lancement |

Livrable de la réunion : un compte rendu avec responsables, emplacements vérifiés, actions et blocages ; pas seulement une présentation orale.

## 10. Critères de réception de la passation

La passation n'est considérée comme réussie que lorsque la personne qui reprend peut :

- accéder au dépôt et à la copie autorisée des artefacts sans dépendre du compte personnel de Grégoire ;
- identifier la version code et le run correspondant à chaque chiffre ;
- ouvrir les vues E08 et E05 sans GPU ;
- comprendre les avertissements de provenance et de validation humaine ;
- reproduire un résumé numérique depuis un JSON sans réentraîner ;
- lancer les tests convenus et conserver leur résultat réel ;
- expliquer les préconditions d'un job avant de le soumettre ;
- retrouver les exemples et contre-exemples dans le périmètre autorisé ;
- savoir quoi ne pas supprimer dans les caches partagés ;
- disposer d'une liste de reprise priorisée et d'un interlocuteur pour les données, les modèles et le cluster.

L'archive qui accompagne ce guide ne contient aucun poids, corpus ou cache GPU du dépôt. Elle prépare leur transfert, elle ne le remplace pas.
