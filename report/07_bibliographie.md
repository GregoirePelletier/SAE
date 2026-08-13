# Bibliographie

## Références académiques

- Jiang, N., Sun, X., Dunlap, L., Smith, L., Nanda, N. (2025). *Interpretable
  Embeddings with Sparse Autoencoders: A Data Analysis Toolkit*. ICML 2026.
  arXiv:2512.10092. Référence méthodologique principale du stage (protocole de
  labellisation contrastive, corrélations, retrieval, clustering), reprise et
  discutée au chapitre 4.
- Bills, S. et al. (2023). *Language models can explain neurons in language
  models*. OpenAI. Origine de la mesure ρ_interp (corrélation de Spearman
  entre intensité jugée par un LLM et activation réelle), reprise en proxy de
  rang dans le protocole d'auto-interprétation local (`src/sae/judge.py`).
- Karvonen, A. et al. (2025). *SAEBench: A Comprehensive Benchmark for Sparse
  Autoencoders in Language Model Interpretability*. ICML 2025. arXiv:2503.09532.
  Source du protocole odd-one-out utilisé dans ce projet.
- Chanin, D., Wilken-Smith, J., Dulka, T., Bhatnagar, H., Golechha, S., Bloom, J.
  (2024). *A is for Absorption: Studying Feature Splitting and Absorption in
  Sparse Autoencoders*. arXiv:2409.14507. Définit le "feature absorption",
  pertinent pour le résidu non-interprété (`04_limites_et_perspectives.md`).
- Koriagin, N., Aksenov, Y., Laptev, D., Gerasimov, G., Balagansky, N., Gavrilov, D.
  (2025). *Teach Old SAEs New Domain Tricks with Boosting*. COLM 2025.
  arXiv:2507.12990. Introduit "SAE Boost", architecture identifiée a posteriori
  comme équivalente à `FrozenCoreResidualSAE`/`ExtendedSAE` de ce projet.
- Korznikov, A., Galichin, A., Dontsov, A., Rogov, O. Y., Oseledets, I., Tutubalina, E.
  (2026). *Sanity Checks for Sparse Autoencoders: Do SAEs Beat Random Baselines?*
  arXiv:2602.14111. Introduit les baselines à composants gelés/aléatoires
  (Frozen Decoder), reproduites sur ce projet (`FrozenDecoderExtendedSAE`).
- Cunningham, H., Ewart, A., Riggs, L., Huben, R., Sharkey, L. (2023). *Sparse
  Autoencoders Find Highly Interpretable Features in Language Models*.
  arXiv:2309.08600. Un des deux papiers fondateurs de l'usage des SAE pour
  l'interprétabilité des LLM.
- Bussmann, B., Leask, P., Nanda, N. (2024). *BatchTopK Sparse Autoencoders*.
  arXiv:2412.06410. Mécanisme de parcimonie de `ExtendedSAE`/`PhraseLevelSAE`
  (`src/sae/batch.py::BatchTopKEncoder`).
- Rajamanoharan, S., Lieberum, T., Sonnerat, N. et al. (2024). *Jumping Ahead:
  Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders*.
  arXiv:2407.14435. Architecture du SAE core GemmaScope-2 (Pipeline 1).
- Bussmann, B., Nabeshima, N., Karvonen, A., Nanda, N. (2025). *Learning
  Multi-Level Features with Matryoshka Sparse Autoencoders*. arXiv:2503.17547.
  Piste non implémentée pour le résidu non-interprété (chapitre 4) — à ne pas
  confondre avec `MATRYOSHKA_DIM` du projet (cf. `docs/references.md`).
- Le Bail, M., Dentan, J., Buscaldi, D., Vanier, S. (2025). *Unveiling
  Decision-Making in LLMs for Text Classification: Extraction of Influential
  and Interpretable Concepts with Sparse Autoencoders*. arXiv:2506.23951.
  Introduit ClassifSAE (SAE supervisé conjoint SAE+classifieur), piste non
  implémentée pertinente pour la détection d'urgence/intention (chapitre 4).
- Shu, D., Wu, X., Zhao, H., Rai, D., Yao, Z., Liu, N., Du, M. (2025). *A
  Survey on Sparse Autoencoders: Interpreting the Internal Mechanisms of
  Large Language Models*. EMNLP 2025 Findings. arXiv:2503.05613. Taxonomie
  utilisée pour cadrer le chapitre 1.
- Resck, L., Augenstein, I., Korhonen, A. (2025). *Explainability and
  Interpretability of Multilingual Large Language Models: A Survey*.
  EMNLP 2025. Motive le test de biais multilingue du juge d'auto-interprétation
  (corpus français).
- Andrylie, L. M., Rahmanisa, I., Ihsani, M. K., Wicaksono, A. F., Wibowo, H. A.,
  Aji, A. F. (2025). *Sparse Autoencoders Can Capture Language-Specific
  Concepts Across Diverse Languages*. arXiv:2507.11230. Motive le test de
  biais multilingue ci-dessus (features SAE potentiellement langue-spécifiques).
- Gerasimov, G., Rusalev, T., Balagansky, N., Laptev, D., Kurochkin, V.,
  Gavrilov, D. (2026). *Unstable Features, Reproducible Subspaces:
  Understanding Seed Dependence in Sparse Autoencoders*. arXiv:2606.12138.
  Montre que les features individuelles d'un SAE varient selon la graine
  d'entraînement, le sous-espace de bas rang restant seul reproductible.
- Nelson, W., Karaletsos, T., Locatello, F. (2026). *Toward Identifiable
  Sparse Autoencoders*. arXiv:2605.31245. Motive, avec la référence
  précédente, l'ablation de variance de seed de ce projet.
- Clavié, B., Lee, S., Shakir, A., Kato, M. P. (2026). *Latent Terms: Dense
  Retrievers Contain Trivially Extractable BM25-ready Zipfian Vocabularies*.
  arXiv:2605.29384. Méthode de retrieval BM25 sur le vocabulaire latent d'un
  SAE, implémentée dans `src/sae/retrieval/latent_terms.py`.
- Beckmann, P., Queloz, M. (2026). *Mechanistic Indicators of Understanding
  in Large Language Models*. arXiv:2507.08017. Cadrage philosophique cité en
  introduction.

<!-- À COMPLÉTER: métadonnées bibliographiques complètes (auteurs, venue,
année) des documents PDF locaux consultés sur l'application des SAE aux
embeddings denses et à la recherche documentaire, cités sans référence
formelle dans les versions précédentes de ce document -->

## Dépôts et outils logiciels réutilisés

Cf. `docs/references.md` pour le détail complet (rôle exact dans le projet, statut de
comparaison) :

- **SAELens** — [github.com/jbloomAus/SAELens](https://github.com/jbloomAus/SAELens) —
  chargement/encodage du SAE GemmaScope-2 préentraîné.
- **GemmaScope** —
  [huggingface.co/google/gemma-scope](https://huggingface.co/google/gemma-scope) —
  poids des SAE préentraînés sur Gemma-3, hébergés sur Hugging Face.
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
