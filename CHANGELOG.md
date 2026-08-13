# Changelog

Correctifs ayant changé un comportement observable de la pipeline. Le détail
technique de chaque diagnostic vit dans `RESULTS_TESTS.md` ; ce fichier ne
liste que le changement et sa conséquence.

## 2026-07-29

- Correction d'une dimension figée (`D_MODEL=4096` au lieu de 3840 pour
  gemma-3-12b-it) qui aurait fait échouer tout run combinant ce modèle et
  l'extension à cœur gelé.
- Suppression d'un doublement transitoire de la mémoire du réservoir de
  résidus lors de sa reconstruction.

## 2026-07-27

- Réduction de la cible de l'ablation volume à grande échelle de 100M à 25M
  tokens après un échec mémoire (le réservoir de résidus était alloué en RAM
  proportionnellement au volume visé).

## 2026-07-21

- Largeur de SAE core relevée de 16k à 65k par défaut (`src/config.py`)
  après vérification de la couverture Neuronpedia des largeurs disponibles.

## 2026-07-20

- Correction d'un biais résiduel ("Objet :"/"Subject :" présent dans les
  mails augmentés mais pas dans les originaux) au chargement du corpus
  augmenté.
- Ajout du dashboard interactif (Streamlit).

## 2026-07-17

- Corpus principal d'entraînement de l'extension et du `PhraseLevelSAE`
  changé de generic (energy/sports/support) à emails originaux + augmentés.
- Sonde de classification multi-classe : sélection dynamique du solveur
  scikit-learn (`liblinear` ne supporte que le cas binaire).

## Infrastructure (2026-08-13)

- Réservoir de résidus memory-mapped sur disque plutôt qu'alloué en RAM
  proportionnellement au volume de tokens visé, débloquant les runs à
  volume élevé (100-200M tokens) sans demande mémoire proche de la
  capacité totale d'un nœud de calcul.
- Suppression du filtre par mots-clés du corpus de remplissage (FineWeb2),
  remplacé par un sous-échantillonnage sans filtre thématique.
