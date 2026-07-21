# Bibliographie

*Note de rédaction* : les références ci-dessous listent les travaux effectivement
mobilisés pendant le stage (méthode, comparaison chiffrée, ou lecture critique). Les
métadonnées bibliographiques complètes (auteurs exacts, venue, année) des articles
disponibles uniquement sous forme de PDF local (`pdf/`) sont à vérifier/compléter
avant intégration dans la version finale déposée, conformément à la remarque déjà
présente dans `report/README.md`.

## Références académiques

- Jiang, N., Sun, R. et al. (2025). *Interpretable Embeddings with Sparse
  Autoencoders: A Data Analysis Toolkit* (`pdf/InterpretableSAE_Embeddings.pdf`,
  [github.com/nickjiang2378/interp_embed](https://github.com/nickjiang2378/interp_embed)).
  Référence méthodologique principale du stage : protocole de labellisation
  contrastive (Appendix C), détection de corrélations "intéressantes" (§4.2,
  Appendix E), retrieval par propriétés et clustering ciblé par similarité
  d'embedding (§4.3/4.4, Appendix F.1). Une relecture ligne à ligne de cette
  référence face au code du projet a permis d'identifier quatre écarts
  méthodologiques (cf. chapitre 4).
- Bills, S. et al. (2023). *Language models can explain neurons in language models*
  (OpenAI). Origine de la mesure ρ_interp (corrélation de Spearman entre intensité
  jugée par un LLM et activation réelle) utilisée dans le protocole
  d'auto-interprétation local (`src/sae/judge.py`).
- Documents complémentaires consultés sur l'application des SAE aux embeddings
  denses et à la recherche documentaire (retrieval), disponibles sous `pdf/` :
  `DisentanglingDenseEmbeddingswithSAE.pdf`,
  `DecodingDenseEmbSAEforInterpandDiscretizDenseRetrieval.pdf`,
  `InterpretandControlDenseRetrievalwithSparseLatentFeatures.pdf`,
  `SparseAutoencodersforHypothesisGeneration.pdf`, `Naver.pdf`,
  `12_Towards_Interpretable_Scien.pdf`.

## Dépôts et outils logiciels réutilisés

Cf. `docs/references.md` pour le détail complet (rôle exact dans le projet, statut de
comparaison) :

- **SAELens** — [github.com/jbloomAus/SAELens](https://github.com/jbloomAus/SAELens) —
  chargement/encodage du SAE GemmaScope-2 préentraîné.
- **GemmaScope** —
  [github.com/google-deepmind/gemma-scope](https://github.com/google-deepmind/gemma-scope) —
  poids des SAE préentraînés sur Gemma-3.
- **Neuronpedia** — [neuronpedia.org](https://www.neuronpedia.org) — labels officiels
  des features GemmaScope "core".
- **F2LLM-v2** (`codefuse-ai/F2LLM-v2-{80M,160M,330M}`) — modèle d'embeddings de
  phrases, backbone du Pipeline 2.
- **bge-m3** — modèle d'embedding multilingue (pooling CLS), utilisé pour la
  similarité de labels (retrieval/clustering) après comparaison empirique avec F2LLM.

## Document de cadrage

- EDF R&D. *Offre de stage SEQUOIA — Explicabilité de documents par Sparse
  Autoencoders* (`pdf/Offre_Stage_EDF_RD_SEQUOIA_E7S_SAE.pdf`). Document interne,
  origine des objectifs listés en introduction.
