# Architecture du système

*(Synthèse orientée rapport — référence technique complète dans `docs/architecture.md`.)*

## Vue d'ensemble

Le système implémente deux pipelines d'analyse interprétable de mails clients EDF,
partageant la même infrastructure de corpus, de stockage et d'évaluation :

```
                    ┌─────────────────────────────────────┐
                    │        Corpus principal              │
                    │  Mails EDF originaux + variantes      │
                    │  augmentées (13 axes de perturbation) │
                    └───────────────┬───────────────────────┘
                                    │
              ┌─────────────────────┴─────────────────────┐
              ▼                                             ▼
     Pipeline 1 (token-level)                     Pipeline 2 (phrase-level)
     Gemma-3-12B-it → SAE GemmaScope-2            F2LLM-v2 → PhraseLevelSAE
     (préentraîné) + extension FrozenCore         (entraîné from-scratch)
     (entraînée from-scratch sur le résidu)
              │                                             │
              ▼                                             ▼
     Max-pool documentaire                        Max-pool documentaire
              │                                             │
              ▼                                             ▼
     Labels : Neuronpedia (core) +                Labels : juge LLM local
     juge LLM local (extension)                   (odd-one-out)
```

## Choix architecturaux clés

### Une extension spécifique au domaine plutôt qu'un SAE unique

Le SAE GemmaScope préentraîné ("core") capture des concepts généraux (appris sur des
corpus massifs, multi-domaines). Pour capturer des concepts spécifiques aux emails EDF
qui ne correspondraient à aucune direction dédiée du SAE core, le Pipeline 1 ajoute une
**extension** (`FrozenCoreResidualSAE`/`ExtendedSAE`) : un second SAE, plus petit
(1024 features, 32 actives simultanément), qui encode le **résidu** — ce que le SAE
core ne reconstruit pas. Le SAE core reste gelé (jamais réentraîné) ; seule l'extension
est entraînée, sur le corpus du projet.

### Nature du corpus "original" (`Mails.tsv`)

Le corpus `Mails.tsv`, désigné dans ce rapport par commodité comme les mails
**"originaux"**, n'est **pas** de la correspondance client authentique : il s'agit
d'un jeu de données déjà synthétique, produit par un travail antérieur du
laboratoire EDF R&D (non réalisé pendant ce stage, ni par les mêmes personnes). Les
variantes **augmentées** (`augmented_mails.jsonl`) sont, elles, générées pendant ce
stage (`scripts/run_augmentation.py`, Gemma-3-12B-it) à partir de ces mails
"originaux". **Aucune des deux couches du corpus n'est donc constituée de données
réelles au sens strict** — le terme "original" désigne uniquement leur statut de
donnée d'entrée (antérieure et externe à ce stage) par rapport aux variantes
augmentées qui en dérivent, pas leur authenticité. Ce rapport n'emploie donc
jamais le terme "réel"/"réels" pour ce corpus.

### Corpus d'entraînement : principal vs secondaire

Distinction utilisée dans la suite du rapport (cf. chapitre suivant) :
- Le corpus qui **entraîne** l'extension et le `PhraseLevelSAE` doit être représentatif
  du domaine cible (emails) pour que les features apprises soient interprétables dans
  ce domaine.
- Un corpus generic (energy/sports/support, substitut public) reste utile comme
  **banc d'essai** pour une capacité générale du système (diffing cross-domaine) mais
  ne doit pas se substituer au corpus principal dans l'entraînement.

### Split train/test group-aware

Les variantes augmentées d'un même mail sont des quasi-duplicata sémantiques de
l'original (même contenu factuel, perturbation contrôlée d'un seul axe à la fois). Un
split aléatoire au niveau des lignes laisserait fuiter des variantes d'un mail de test
dans le train (ou l'inverse), biaisant artificiellement à la hausse les métriques de
classification/silhouette. Le split est donc effectué au niveau du **mail d'origine** :
un mail et toutes ses variantes tombent du même côté.

### Stockage sparse des activations token-level

Une activation SAE dense par token, à la largeur historique 262k, coûterait ~400 Mo
par document. Le projet stocke les activations token-level au format CSR (compressed
sparse row, implémenté en tenseurs torch plutôt qu'avec SciPy pour rester
GPU-compatible), ramenant le coût à quelques centaines de Ko par document — nécessaire
pour traiter des corpus de dizaines de milliers de documents sans épuiser la mémoire.

### Précision numérique (bf16)

Gemma-3 présente des activations de norme très élevée ("massive activations",
documentées dans la littérature) dans son residual stream. Une précision fp16 (plage
dynamique jusqu'à ~65 504) overflow silencieusement sur ces valeurs, corrompant tout
l'entraînement en aval sans erreur explicite (perte `NaN` dès la première epoch). Le
projet utilise bf16 par défaut partout (même en local), qui partage la plage
d'exposant de fp32.

## Infrastructure de calcul

Cluster SLURM à 3 partitions GPU (a100, h100, h100-bis), sans accès réseau direct
depuis les nœuds de calcul — toutes les dépendances (modèles, données) doivent être
prépositionnées sur disque avant soumission d'un job. Cette contrainte a orienté
plusieurs choix : cache local des labels Neuronpedia (pas d'appel réseau au runtime),
environnement Python déjà provisionné sur disque (`.venv/bin/python` plutôt que `uv
run`, qui tenterait de re-résoudre l'environnement en ligne).
