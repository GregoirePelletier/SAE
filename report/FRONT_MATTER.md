<div align="center">

<!-- À COMPLÉTER: nom de l'établissement (Master 2) -->
<!-- À COMPLÉTER: intitulé du Master / de la spécialité -->

---

# Rapport de stage de Master 2

## Explicabilité automatique de mails clients par Sparse Autoencoders

### Application à l'analyse interprétable de la correspondance client d'EDF

---

**Auteur** : Grégoire Pelletier

**Entreprise d'accueil** : EDF R&D — Projet SEQUOIA

<!-- À COMPLÉTER: nom du maître de stage (entreprise) -->

<!-- À COMPLÉTER: nom du tuteur académique -->

<!-- À COMPLÉTER: dates de début/fin de stage -->

**Date de rédaction** : 21 juillet 2026

</div>

---

## Résumé

Ce stage porte sur l'explicabilité automatique de mails clients d'EDF à l'aide de
Sparse Autoencoders (SAE), combinant un SAE préentraîné à grande échelle (GemmaScope-2,
sur les activations de Gemma-3-12B-it) étendu par un second SAE entraîné
spécifiquement sur le domaine — architecture à cœur gelé identifiée en cours de
stage comme structurellement équivalente à SAE Boost (Koriagin et al., COLM 2025)
— et un second pipeline indépendant fondé sur des embeddings de phrase (F2LLM-v2,
bge-m3). Le pipeline initial, fonctionnel de bout en bout, présentait un taux de
succès faible (20%) au test d'auto-interprétation des features propres au domaine.
Une démarche de diagnostic par ablation contrôlée a établi que ce taux n'était pas
limité par le volume d'entraînement, mais par le domaine du corpus
d'entraînement (uniquement générique, sans texte du domaine cible) : une fois
ce domaine corrigé, le taux d'interprétabilité atteint 45,3%.

Une campagne d'ablations exhaustive (plus de 20 configurations : largeur du SAE,
capacité et parcimonie de l'extension, volume de tokens, graine d'entraînement,
dimension d'embedding, backbone de Pipeline 2) montre qu'**aucun hyperparamètre du
SAE ne modifie significativement ce taux** une fois le domaine corrigé — à
l'exception d'un unique levier : **l'échelle du modèle extracteur/juge**, qui
produit un effet dose-réponse net et hautement significatif (12,0% à 1 milliard de
paramètres, 28,0% à 4 milliards, 45,3% à 12 milliards ; test de tendance de
Cochran-Armitage, p≈1,6×10⁻¹⁰). Un sanity check contre un décodeur figé aléatoire (Korznikov et al., 2026) confirme
que l'entraînement de l'extension apprend une structure réelle (45,3% contre
29,3%, écart significatif) tout en révélant qu'une classification en aval résiste
beaucoup mieux à cette dégradation que l'interprétation qualitative. Des tests
complémentaires (fidélité et plausibilité de l'explication document-level,
robustesse du protocole de jugement, biais multilingue, fidélité du steering,
évaluation quantitative du retrieval) complètent la validation du système, avec un
audit rétroactif de la méthodologie statistique employée.

**Mots-clés** : Sparse Autoencoders, interprétabilité mécaniste, GemmaScope,
grands modèles de langage, explicabilité, traitement automatique des mails clients,
auto-interprétation par juge LLM, effet d'échelle.

---

## Abstract

This internship addresses automatic explainability of customer emails at EDF using
Sparse Autoencoders (SAE), combining a large pretrained SAE (GemmaScope-2, on
Gemma-3-12B-it activations) extended by a second SAE trained specifically for the
target domain — a frozen-core architecture identified during the internship as
structurally equivalent to SAE Boost (Koriagin et al., COLM 2025) — alongside an
independent sentence-embedding-based pipeline (F2LLM-v2, bge-m3). The initial
end-to-end pipeline showed a low success rate (20%) on the domain-specific feature
auto-interpretation test. A controlled-ablation diagnostic established that this
was not a training-volume limitation but a training-corpus domain issue (generic
text only, no domain-specific text) — once corrected, the interpretability rate
rose to 45.3%.

An exhaustive ablation campaign (20+ configurations: SAE width, extension capacity
and sparsity, token volume, training seed, embedding dimension, Pipeline-2
backbone) shows that **no SAE hyperparameter significantly changes this rate**
once the corpus domain is fixed — except for a single lever: **the scale of the
extractor/judge model**, which produces a clean, highly significant dose-response
effect (12.0% at 1B parameters, 28.0% at 4B, 45.3% at 12B; Cochran-Armitage trend
test, p≈1.6×10⁻¹⁰). A sanity check against a randomly frozen decoder (Korznikov et al., 2026) confirms that the
extension's training learns genuine structure (45.3% vs 29.3%, significant gap)
while also revealing that downstream classification survives this degradation far
better than qualitative interpretation does. Complementary tests (document-level
explanation fidelity and plausibility, judge-protocol robustness, multilingual
bias, steering fidelity, quantitative retrieval evaluation) complete the system's
validation, together with a retroactive audit of the statistical methodology used.

**Keywords**: Sparse Autoencoders, mechanistic interpretability, GemmaScope, large
language models, explainability, customer email analysis, LLM auto-interpretation,
scaling effect.

---

## Sommaire

- Introduction générale
- Chapitre 1 — État de l'art
- Chapitre 2 — Architecture et implémentation
- Chapitre 3 — Démarche expérimentale et résultats
- Chapitre 4 — Limites et perspectives
- Conclusion générale
- Bibliographie

---
