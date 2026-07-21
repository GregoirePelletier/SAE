<div align="center">

**[Nom de l'établissement — Master 2, à compléter]**
**[Intitulé du Master / de la spécialité — à compléter]**

---

# Rapport de stage de Master 2

## Explicabilité automatique de mails clients par Sparse Autoencoders

### Application à l'analyse interprétable de la correspondance client d'EDF

---

**Auteur** : Grégoire Pelletier *(déduit de l'adresse de contact ; à confirmer)*

**Entreprise d'accueil** : EDF R&D — Projet SEQUOIA

**Maître de stage (entreprise)** : [Nom du tuteur EDF — à compléter]

**Tuteur académique** : [Nom du tuteur académique — à compléter]

**Période de stage** : [Dates de début/fin — à compléter]

**Date de rédaction** : 21 juillet 2026

</div>

---

## Remerciements

*[Section à personnaliser par l'auteur — usuellement adressée au maître de stage, à
l'équipe d'accueil, au tuteur académique, et à toute personne ayant contribué au bon
déroulement du stage.]*

---

## Résumé

Ce stage porte sur l'explicabilité automatique de mails clients d'EDF à l'aide de
Sparse Autoencoders (SAE), combinant un SAE préentraîné à grande échelle (GemmaScope-2,
sur les activations de Gemma-3-12B-it) étendu par un second SAE entraîné
spécifiquement sur le domaine, et un second pipeline indépendant fondé sur des
embeddings de phrase (F2LLM-v2). Le pipeline initial, fonctionnel de bout en bout,
présentait un taux de succès faible (20%) au test d'auto-interprétation des features
propres au domaine. Une démarche de diagnostic par ablation contrôlée a établi que ce
taux n'était pas limité par le volume d'entraînement, mais par une erreur de
conception du corpus d'entraînement (uniquement générique, sans emails réels) —
corrigée, elle porte le taux d'interprétabilité à ~41-45%. Le stage a ensuite mis en
place des tests de qualité de l'explication document-level (fidélité par ablation,
plausibilité par choix forcé, toutes deux positives), un protocole d'évaluation
couvrant l'ensemble des capacités du dépôt sous conditions fixées, un dashboard
interactif, et une ablation finale de mise à l'échelle (largeur du SAE core,
nombre d'époques, nombre de features labellisées) pour vérifier si un passage à
l'échelle simple améliore encore les résultats sans changement de méthode.

**Mots-clés** : Sparse Autoencoders, interprétabilité mécaniste, GemmaScope,
grands modèles de langage, explicabilité, traitement automatique des mails clients,
auto-interprétation par juge LLM.

---

## Abstract

This internship addresses automatic explainability of customer emails at EDF using
Sparse Autoencoders (SAE), combining a large pretrained SAE (GemmaScope-2, on
Gemma-3-12B-it activations) extended by a second SAE trained specifically on the
target domain, alongside an independent sentence-embedding-based pipeline (F2LLM-v2).
The initial end-to-end pipeline showed a low success rate (20%) on the
domain-specific feature auto-interpretation test. A controlled-ablation diagnostic
established that this was not a training-volume limitation but a training-corpus
design flaw (generic text only, no real emails) — once fixed, the interpretability
rate rose to ~41-45%. The internship then implemented document-level explanation
quality tests (ablation-based fidelity, forced-choice plausibility, both positive), a
full-repository evaluation protocol under fixed conditions, an interactive dashboard,
and a final scale-up ablation (core SAE width, number of training epochs, number of
labeled features) to test whether simply scaling up improves results further without
any methodological change.

**Keywords**: Sparse Autoencoders, mechanistic interpretability, GemmaScope, large
language models, explainability, customer email analysis, LLM auto-interpretation.

---

## Sommaire

- Introduction générale
- Chapitre 1 — État de l'art
- Chapitre 2 — Architecture et implémentation
- Chapitre 3 — Démarche expérimentale et résultats
- Chapitre 4 — Inspection des erreurs et corrections
- Chapitre 5 — Limites et perspectives
- Conclusion générale
- Bibliographie

---
