# E00 : consommation mémoire du pipeline

Avant la campagne post-soutenance, un run à grande échelle (job 45803, 1,2 million de morceaux de
texte « filler » ajoutés au corpus) avait atteint 358 Go de mémoire. E00 mesure où va la mémoire
et corrige le poste principal.

## Mesure

Le profilage (`scripts/post_stage/profile_run.py`, qui lance `saev5.py` comme sous-processus et
relève mémoire, cgroup et GPU toutes les 20 à 30 secondes) a été fait sur une configuration
réduite (job 48515 : Gemma-3-1B, couche 13, 250 000 tokens pour l'extension, 20 000 morceaux de
filler). Trois constats :

1. **Doublement transitoire des vecteurs documentaires.** En fin d'extraction, la liste Python
   `all_doc_sae_acts` (une ligne par document, filler compris, même si les lignes filler sont
   nulles) et le tenseur construit à partir d'elle coexistaient : +11,6 Go pour 56 600 lignes.
   Extrapolé à l'échelle du job 45803 (22 fois plus de lignes), cela donne environ 255 Go, ce qui
   explique l'essentiel des 358 Go observés.
2. **Chargement du modèle juge.** Qwen3.8-27B (environ 52 Go de poids) est lu en mémoire mappée :
   la mémoire du processus monte de 12 à 64 Go sans allocation propre, puis redescend une fois les
   poids sur le GPU. Ce n'est pas une fuite, mais il faut prévoir cette marge dans `--mem` pour
   tout job qui charge le juge.
3. **Sonde de classification lente.** Le job a été arrêté par la limite de temps (2 h), pas par
   un manque de mémoire, pendant la sonde logistique (14 classes, 36 602 documents, 17 408
   dimensions, environ 25 minutes par pli). Les matrices sont bien creuses ; la durée est normale
   à cette taille de corpus.

Remarque de méthode : `N_TOKENS_EXTRA_TRAIN` ne limite que le réservoir d'entraînement de
l'extension, pas le nombre de documents. Pour un profilage rapide, fixer aussi
`MAX_AUGMENTED_PER_MAIL` à une petite valeur (par défaut 13 variantes par email).

## Correctif

`all_doc_sae_acts` n'ajoute plus de ligne pour le filler et est enregistré sous forme compacte
(`save_doc_acts_compact` dans `src/sae/sae_shared.py`, commit `a640ed7`). Le format sur disque ne
change pas : les quelque 20 scripts qui lisent ce fichier avec `load_all_doc_acts` fonctionnent
sans modification.

Validation :

- test synthétique sur CPU (2 000 documents et 20 000 lignes de filler) : pic de 690 Mo avant
  correction, 65 Mo après, soit le rapport attendu du nombre de lignes ;
- run réel (job 48530, même configuration avec `MAX_AUGMENTED_PER_MAIL=1`) : terminé en
  36 minutes, fichier compact sans lignes de filler, métriques scientifiques inchangées.
  Son pic global n'est pas comparable à celui du job 48515 (corpus plus petit, cache du système
  déjà chaud) ; c'est le test synthétique qui isole l'effet du correctif.

## Ce qui reste à corriger

Le ré-encodage par le SAE étendu (`saev5.py`, fichier `p1_all_doc_acts_ext_d{D_EXTRA}.pt`) alloue
encore un tenseur dense d'une ligne par document, filler compris, même si ces lignes ne sont
jamais calculées. À l'échelle du job 45803, cela représente environ 86 Go. Le corriger demande de
vérifier que les scripts qui lisent ce fichier n'y accèdent pas par l'indice global d'un document.
Tant que ce n'est pas fait, ne pas relancer de run avec un filler volumineux.

## Fichiers

- `scripts/post_stage/profile_run.py`, `src/post_stage/resources.py` ; recette
  `slurm/post_stage/00_profile.slurm`.
- Mesures : `cache/resources_profile.jsonl` dans les dossiers `results_post_stage_e00_profile_1b*`.
