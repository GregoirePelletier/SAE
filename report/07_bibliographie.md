# Bibliographie

*Note de rédaction* : les références ci-dessous listent les travaux effectivement
mobilisés pendant le stage (méthode, comparaison chiffrée, ou lecture critique). Les
métadonnées bibliographiques complètes (auteurs exacts, venue, année) des articles
disponibles uniquement sous forme de PDF local (`pdf/`) sont à vérifier/compléter
avant intégration dans la version finale déposée, conformément à la remarque déjà
présente dans `report/README.md`.

## Références académiques

- Jiang, N., Sun, X. et al. (2025). *Interpretable Embeddings with Sparse
  Autoencoders: A Data Analysis Toolkit*
  ([arXiv:2512.10092](https://arxiv.org/abs/2512.10092), ICML 2026,
  `pdf/InterpretableSAE_Embeddings.pdf`,
  [github.com/nickjiang2378/interp-embed](https://github.com/nickjiang2378/interp-embed)).
  Référence méthodologique principale du stage : protocole de labellisation
  contrastive (Appendix C), détection de corrélations "intéressantes" (§4.2,
  Appendix E), retrieval par propriétés et clustering ciblé par similarité
  d'embedding (§4.3/4.4, Appendix F.1) — méthodes reprises et discutées au
  chapitre 4.
- Bills, S. et al. (2023). *Language models can explain neurons in language models*
  (OpenAI). Origine de la mesure ρ_interp (corrélation de Spearman entre intensité
  jugée par un LLM et activation réelle) utilisée dans le protocole
  d'auto-interprétation local (`src/sae/judge.py`) — implémentation locale utilisant
  un proxy de rang plutôt que l'activation continue réelle, cf. `01_etat_de_lart.md`.
- Karvonen, A. et al. (2025). *SAEBench: A Comprehensive Benchmark for Sparse
  Autoencoders in Language Model Interpretability*
  ([arXiv:2503.09532](https://arxiv.org/abs/2503.09532), ICML 2025). Source du
  protocole odd-one-out cité au chapitre "État de l'art" et de la "sparse probing
  SAEBench" évoquée comme piste alternative en limites.
- Chanin, D., Wilken-Smith, J., Dulka, T., Bhatnagar, H., Golechha, S., Bloom, J.
  (2024). *A is for Absorption: Studying Feature Splitting and Absorption in Sparse
  Autoencoders* ([arXiv:2409.14507](https://arxiv.org/abs/2409.14507)). Article
  définissant le phénomène de "feature absorption", distinct du "feature splitting" —
  pertinent pour le résidu non-interprété (`04_limites_et_perspectives.md`), dont
  Matryoshka SAEs (Bussmann et al. 2025, ci-dessous) est présenté comme correctif
  possible.
- Koriagin, N., Aksenov, Y., Laptev, D., Gerasimov, G., Balagansky, N., Gavrilov, D.
  (2025). *Teach Old SAEs New Domain Tricks with Boosting*
  ([arXiv:2507.12990](https://arxiv.org/abs/2507.12990), COLM 2025,
  `pdf/teacholdsaes.pdf`). Introduit "SAE Boost" — identifié a posteriori comme
  l'architecture déjà implémentée par `FrozenCoreResidualSAE`/`ExtendedSAE` de ce
  projet (cf. chapitre 1 "Perspectives critiques", `RESULTS_TESTS.md` §18).
- Korznikov, A., Galichin, A., Dontsov, A., Rogov, O. Y., Oseledets, I., Tutubalina, E.
  (2026). *Sanity Checks for Sparse Autoencoders: Do SAEs Beat Random Baselines?*
  ([arXiv:2602.14111](https://arxiv.org/abs/2602.14111), `pdf/sanitychecks.pdf`).
  Introduit les baselines à composants gelés/aléatoires (Frozen Decoder, Frozen
  Encoder, Soft-Frozen Decoder) comme test de validité des métriques SAE standard.
  Protocole "Frozen Decoder" reproduit sur ce projet
  (`FrozenDecoderExtendedSAE`, `RESULTS_TESTS.md` §19).
- Cunningham, H., Ewart, A., Riggs, L., Huben, R., Sharkey, L. (2023). *Sparse
  Autoencoders Find Highly Interpretable Features in Language Models*
  ([arXiv:2309.08600](https://arxiv.org/abs/2309.08600), `pdf/2309.08600v3.pdf`).
  Un des deux papiers fondateurs de l'usage des SAE pour l'interprétabilité des LLM.
- Bussmann, B., Leask, P., Nanda, N. (2024). *BatchTopK Sparse Autoencoders*
  ([arXiv:2412.06410](https://arxiv.org/abs/2412.06410), `pdf/BatchTopK.pdf`).
  Mécanisme de parcimonie de `ExtendedSAE`/`PhraseLevelSAE`
  (`src/sae/batch.py::BatchTopKEncoder`) — implémentation vérifiée fidèle au papier
  (cf. `docs/references.md`).
- Rajamanoharan, S., Lieberum, T., Sonnerat, N. et al. (2024). *Jumping Ahead:
  Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders*
  ([arXiv:2407.14435](https://arxiv.org/abs/2407.14435), `pdf/jumpRELU.pdf`).
  Architecture du SAE core GemmaScope-2 (Pipeline 1).
- Bussmann, B., Nabeshima, N., Karvonen, A., Nanda, N. (2025). *Learning Multi-Level
  Features with Matryoshka Sparse Autoencoders*
  ([arXiv:2503.17547](https://arxiv.org/abs/2503.17547), `pdf/Matryoshka.pdf`).
  Piste non implémentée pour le résidu non-interprété (chapitre 4) — à ne pas
  confondre avec `MATRYOSHKA_DIM` du projet (cf. `docs/references.md`).
- Le Bail, M., Dentan, J., Buscaldi, D., Vanier, S. (2025). *Unveiling
  Decision-Making in LLMs for Text Classification: Extraction of Influential and
  Interpretable Concepts with Sparse Autoencoders*
  ([arXiv:2506.23951](https://arxiv.org/abs/2506.23951),
  `pdf/UnveilingDecision-MakinginLLMsforTextClassification.pdf`). Introduit
  ClassifSAE (SAE supervisé conjoint SAE+classifieur) — piste non implémentée,
  directement pertinente pour les objectifs détection d'urgence/intention
  (chapitre 4).
- Shu, D., Wu, X., Zhao, H. et al. (2025). *A Survey on Sparse Autoencoders:
  Interpreting the Internal Mechanisms of Large Language Models*
  ([arXiv:2503.05613](https://arxiv.org/abs/2503.05613), EMNLP 2025 Findings,
  `pdf/SurveySAE.pdf`). Taxonomie explications input-based/output-based et
  métriques structurelles/fonctionnelles, utilisée pour cadrer le chapitre 1.
- Resck, L., Augenstein, I., Korhonen, A. (2025). *Explainability and
  Interpretability of Multilingual Large Language Models: A Survey* (EMNLP 2025,
  `pdf/2025.emnlp-main.1033.pdf`). Cité pour le biais multilingue potentiel du juge
  d'auto-interprétation (corpus français) — **mesuré** au chapitre 3, §13 : pas de
  différence significative français/anglais (46,9% vs 45,5%, z=0,24), mais 38,6%
  des features changent de statut interprétable selon la langue.
- *Sparse Autoencoders Can Capture Language-Specific Concepts Across Diverse
  Languages* ([arXiv:2507.11230](https://arxiv.org/abs/2507.11230)). Motive le test
  de biais multilingue ci-dessus (features SAE potentiellement langue-spécifiques,
  facteur de confusion pour un juge interrogé hors de la langue du corpus).
- *Unstable Features, Reproducible Subspaces*
  ([arXiv:2606.12138](https://arxiv.org/abs/2606.12138)) et *Toward Identifiable
  Sparse Autoencoders* ([arXiv:2605.31245](https://arxiv.org/abs/2605.31245)).
  Montrent que les features individuelles d'un SAE varient selon la graine
  d'entraînement, le sous-espace de bas rang restant seul reproductible — **testé**
  au chapitre 3, §12 (ablation de variance de seed) : taux agrégé stable (45,3% vs
  47,3%, non significatif) mais seulement 28,2% de recouvrement exact des libellés
  de features entre les deux graines, confirmant la thèse des deux papiers.
- Clavié, B., Lee, S., Shakir, A., Kato, M. P. (2026). *Latent Terms: Dense
  Retrievers Contain Trivially Extractable BM25-ready Zipfian Vocabularies*
  ([arXiv:2605.29384](https://arxiv.org/abs/2605.29384)). Méthode de retrieval BM25
  sur le vocabulaire latent d'un SAE — **implémentée et évaluée quantitativement**
  au chapitre 3, §17 (Precision@10 parfaite sur 3 intentions/4, échec structurel
  diagnostiqué sur la 4ᵉ, `RESULTS_TESTS.md` §26).
- Beckmann, P., Queloz, M. (2026). *Mechanistic Indicators of Understanding in Large
  Language Models* ([arXiv:2507.08017](https://arxiv.org/abs/2507.08017),
  `pdf/MechanisticIndicatorsinLLM.pdf`). Cadrage philosophique cité en introduction.
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
  [huggingface.co/google/gemma-scope](https://huggingface.co/google/gemma-scope) —
  poids des SAE préentraînés sur Gemma-3 (lien GitHub `google-deepmind/gemma-scope`
  précédemment cité corrigé : ce dépôt n'existe pas, les poids sont hébergés sur
  Hugging Face).
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
