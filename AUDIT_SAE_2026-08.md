# Audit `GregoirePelletier/SAE` — items ouverts

Fusion de trois passes (audit interne du dépôt, et deux audits indépendants Opus 5
effort élevé, l'un centré fidélité/perf, l'autre opérationnel/data-science/hygiène),
recoupée avec l'état actuel du code. Chaque item ci-dessous a été vérifié individuellement
contre le dépôt au moment de la fusion — un item confirmé résolu (getcol déjà en CSC,
`.gitmodules` déjà déclaré, `test_massive_acts.py` déjà déplacé, item 8/G6/G3-async déjà
vérifiés GPU, etc.) a été retiré plutôt que reconduit. Le détail des correctifs déjà
appliqués et de leur vérification (jobs GPU, tests d'équivalence) vit dans `git log`, pas
ici — ce document ne porte que ce qui reste à traiter.

B1/B2/fallback layer=24/B7/B6/Latent Terms sont tranchés (§5/§7 ci-dessous, détail et
chiffres dans `RESULTS_TESTS.md` §77-§82). HypothesisVerifier, NPMI_verified, clustering
LLM, génération structurée de diffing (App. D.2), retrieval réel (RRF+rerank+RBO) sont
implémentés et testés (§7/§8) ; App. I a son calcul F1 prêt mais pas exécuté (fragments
manquants, extraction fraîche nécessaire, coût non engagé sans confirmation). Voir §7/§8
pour l'état précis et les jobs GPU en cours au moment de la dernière mise à jour de ce
document.

---

## 1. Fidélité aux papiers

- **interp-embed, Diffing** : (b) `HypothesisVerifier` + taux de vérification (App. K.1)
  implémentés, testés, mesurés (RESULTS_TESTS.md §84). (a) **génération JSON structurée
  App. D.2 implémentée** cette session (`src/analysis/diff_hypothesis_generator.py`,
  prompt verbatim, ≤N hypothèses avec `percentage_difference`/`confidence`/`feature_ids` ;
  sélection top-200/seuil 0,03 par différence de FRÉQUENCE — pas log-odds-ratio —
  `src/analysis/cooccurrence.py::select_top_diff_features_by_frequency`), pipelinée avec
  `HypothesisVerifier` (`scripts/diffing_structured_hypotheses_test.py`, job 45750 lancé,
  résultat en attente). `generate_llm_diff_hypothesis` (texte libre) reste pour le résumé
  affiché dans `results.json`, coexiste avec la génération structurée sans la remplacer.
  Manquent encore (c) `diff_features_multi`, (d) `limit_feature_differences`.
- **interp-embed, Corrélations** : manquent (a) filtre LLM des latents syntaxiques (E.1),
  (b) filtre des paires triviales co-activées sur le même token, (c) NPMI_verified (juge
  reclasse *i*/*j* indépendamment puis recalcule le NPMI — toute la Figure 4 du papier),
  (d) métrique CO = max(P(i|j), P(j|i)). Sans (c), `p1_interesting_correlations.json` est
  une liste de candidats, pas un résultat — à formuler ainsi dans le rapport tant que non
  fait.
- **interp-embed, Clustering** : `SpectralClustering(affinity="cosine")` sur binaire
  (`saev5.py`) n'est **pas** une variante de l'affinité de Jaccard du papier (App. F) —
  autre métrique, pas un choix d'implémentation. Manquent génération LLM de mots-clés +
  union top-k=100 latents (App. F.1, le dépôt prend un `axis_query` unique en dur),
  accuracy par cluster (réassignation LLM), z-score de conductance en espace dense. Score
  de silhouette : gardé (continuité avec `RESULTS_TESTS.md`), divergence avec la note 4 du
  papier documentée explicitement dans `docs/references.md` plutôt que silencieuse.
- **interp-embed, Retrieval** : l'étape 1 de la Figure 10 (normaliser chaque latent par le
  90ᵉ percentile de ses activations non nulles) est corrigée
  (`src/analysis/metrics.py::normalize_by_p90_and_score`). Métriques App. G implémentées
  et testées (`src/analysis/metrics.py::average_precision/precision_at_k/
  mean_average_precision/mean_precision_at_k/reciprocal_rank_fusion/rank_biased_overlap`,
  `tests/test_retrieval_metrics.py`, écarts documentés `docs/references.md`). **Intégration
  faite cette session** : `scripts/latent_retrieval_precision_eval.py` combine désormais
  Latent Terms+TF-IDF par RRF, reranke le top 50 par juge LLM pointwise
  (`src/analysis/retrieval_rerank.py` — écart documenté R6, le prompt de reranking exact
  du papier n'est pas publié), et calcule le RBO (p=0,98) entre Latent Terms et TF-IDF sur
  les 4 requêtes déjà validées, plus MAP/MP@10 agrégées (job 45745 lancé, résultat en
  attente). Reste : toujours seulement 4 requêtes (une par intention), pas un benchmark
  IR à grande échelle (BEIR).
- **App. I (taille du modèle lecteur, 12B vs 27B)** : le calcul F1 latent-vs-juge est
  désormais implémenté et testé (`src/analysis/reader_size_ablation.py::
  f1_activation_vs_judge`/`compute_reader_size_ablation` — F1 entre activation réelle
  binarisée et classification du juge via `hypothesis_verifier.verify_hypotheses` avec une
  seule hypothèse, médiane sur l'échantillon de latents comme le papier, pas la moyenne).
  **Non exécuté cette session** : nécessite l'activation par-document du latent sur un
  domaine ENTIER (pas seulement les exemples déjà en cache) pour les paliers 12B et 27B —
  leurs fragments token-level (`results_v27_.../results_v31_...`) ont été supprimés par le
  nettoyage disque de la session précédente, donc une extraction fraîche serait nécessaire
  pour au moins un des deux paliers avant de pouvoir lancer ce calcul. Coût non trivial
  (extraction complète, plusieurs heures GPU par palier), pas engagé sans confirmation —
  le cluster était déjà chargé (4-5 jobs en file) au moment où cet item a été atteint.
  Jusqu'à ce rerun, la comparaison 12B/27B reste arbitrée par le taux odd-one-out, instable
  à 31% au niveau d'une feature (`CLAUDE.md`, §13.1).
- **SAE Boost — point de vigilance non résolu par le correctif d'encodage** : l'ancien
  encodeur (sur `e`) évitait par construction les activations massives de `x`
  (norme ~1e5) ; le nouveau (sur `x`) les expose, atténuées seulement par
  `encoder_input_scale` (normalisation globale, pas un traitement par outlier comme
  `norm_outlier_mask` côté Pipeline 1). À vérifier empiriquement (`dead_frac`, `l0_extra`,
  courbes de loss) sur le premier run complet qui utilise ce correctif, pas à tenir pour
  acquis.
- **SAE Boost — budget de tokens** : `run_sae_v14_main.slurm` fixe 100M tokens ×10 époques ;
  le papier signale une dégradation de l'EV général jusqu'à −31% sous 100M tokens et une
  convergence des features seulement au-delà de 200M, référence à 1Md tokens uniques par
  domaine. Le run de référence est au seuil que le papier signale comme dangereux, et la
  répétition (10 époques) n'achète pas de diversité — à écrire explicitement en limite du
  rapport. `compare_to_frozen_benchmark.py` existe pour vérifier la non-régression (Table 2
  du papier, <1%) mais n'est référencé par aucun `.slurm`.
- **SAE Boost — baseline invalidée** : `input_scale` (médiane des normes) et
  l'initialisation PCA du décodeur sont des ajouts du dépôt absents du papier — légitimes,
  mais ils invalident la baseline « Extended SAE (random init) » de leur Table 3 comme
  point de comparaison si citée telle quelle.
- **Latent Terms — premiers résultats chiffrés (job 44995, RESULTS_TESTS.md §80).** OOM
  déjà corrigé avant cette session (`fbde008`) ; le blocage réel était `--time=06:00:00`
  pendant l'entraînement du SAE (job 44666), relancé à `--time=24:00:00` et terminé en
  seulement 11 min (pool déjà en cache). Pas de victoire uniforme sur TF-IDF : bat TF-IDF
  largement sur `remboursement` (P@10 0,90 vs 0,30, taux de base faible), à égalité sur
  `information`/`urgence`, dominé sur `réclamation` (P@10 0,50 vs 1,00 — TF-IDF profite
  d'un vocabulaire homogène sur l'intention à taux de base le plus élevé). Le duplicata
  a100 (job 44996) a échoué sur un bug distinct (répertoire de cache non créé sur un
  `SAVE_DIR` fraîchement dédoublonné, corrigé) — sans conséquence, le job h100 a produit
  le résultat de référence. `load_or_train_latent_terms_sae` n'a toujours aucun mécanisme
  de reprise (R1), sans conséquence tant que le run tient dans le budget `--time`.
- **Latent Terms — évaluation limitée** : `latent_retrieval_precision_eval.py` calcule
  Precision@10/@20 (+ MAP/MP@10 désormais, cf. plus haut) contre TF-IDF/RRF/RRF+rerank sur
  toujours seulement 4 requêtes paraphrasées (une par intention, pas de réplication), avec
  un label de pertinence basé sur `INTENT_KEYWORDS_FR` — resserré cette session (N5,
  AUDIT_SAE_2026-08.md §8, motif "remboursement" débarrassé de son bruit verbal
  "d'avoir"/"l'avoir") mais toujours un label FAIBLE (lexical, pas une annotation humaine)
  — à garder étiqueté comme évaluation indicative, pas benchmark IR (pas de nDCG/BEIR).
- **Stockage de `e` en int8, à reformuler** : la correction de fidélité SAE Boost (encodeur
  lit `x`) rend caduque l'idée initiale de stocker `e` à la place de `x` — l'encodeur a
  maintenant besoin de `x`. Quantifier `x` en int8 par ligne est dangereux ici précisément
  à cause des activations massives (une échelle `max|x|/127` avec quelques dims à 1e5
  écrase à zéro tout le reste de la ligne). Options réelles restant à évaluer : décomposition
  type LLM.int8 (bf16 sur les ~1% de dims outlier, int8 sur le reste), ou accepter les
  768 Go du réservoir tels quels. fp8 est hors-course (`e4m3` plafonne à 448, `e5m2` à
  57 344 < 1,3e5, l'ordre de grandeur des outliers Gemma-3).

## 2. Performance

- **Encodage SAE document par document à l'extraction** (indépendant du ré-encodage, déjà
  batché) : `pretrained_sae.encode(filtered)` dans une boucle `for b in range(B)` — 4 GEMM
  de `[T, 3840]×[3840, 16384]` au lieu d'un seul de `[ΣT, 3840]`. Le filler ne passe plus
  par ce chemin (exclu de l'encodage core), donc le gain restant est proportionnel aux
  documents non-filler (~8% du corpus à l'échelle de référence) — modeste mais non nul.
- **`BATCH_SIZE_EXTRA=16384` : gain de vitesse réel, compromis de qualité non tranché.**
  Ablation terminée (job 44620) : dead_pct 53,3% (mieux qu'à 1024) mais rho_sae 0,781
  (contre ~0,82-0,83 à 1024) — pas un gain net, le budget top-k partagé sur un batch plus
  large change la distribution de sparsité à l'entraînement. Ne pas adopter comme défaut
  sans réplication multi-graines (n=1 à ce jour). Décision produit, pas un correctif.
- **Juge — désaccord de batching non testé sur le stade le plus sensible.** Le 1/24
  désaccord texte-à-texte mesuré (non-associativité flottante des kernels batchés) porte
  sur un prompt de test long (32 tokens) ; le stade odd-one-out de production (8 tokens, un
  seul chiffre à extraire) n'a pas été testé séparément — à faire avant de faire confiance
  à `interp_score` en routine.
- **Juge — trois chargements du 12B par run**, non résolu par défaut : `RUN_DIFF_HYPOTHESIS=0`
  supprime le troisième, mais garder le modèle résident entre extraction et juge (24 Go
  bf16 + SAE 16k ≈ 26 Go, tient sur H100 80 Go) reste un compromis à trancher explicitement
  contre le repli A100-40G (change le pic VRAM, pas un défaut à changer sans décision produit).
- **P2/F2LLM — tri par longueur avant `padding=True` non fait.** `batch_size` est maintenant
  paramétré (`F2LLM_EXTRACT_BATCH_SIZE`), mais `extract_f2llm_embeddings` a un mécanisme de
  reprise shardée qui indexe par position CONTIGUE dans `texts` — trier par longueur avant
  batching casserait cet invariant (le shard N ne correspondrait plus à `texts[i:i+shard_size]`)
  sans persister aussi la permutation dans le checkpoint. Gain plus risqué qu'ailleurs dans
  cet audit, pas fait tant que la reprise n'est pas adaptée en même temps.
- **Étages aval — murs de scalabilité restants** (aucun ne coûte cher à 16k de largeur, tous
  cassent à 65k/262k) :
  - `saev5.py` : `SpectralClustering` construit une affinité n×n en O(n²) mémoire — mur dur
    vers 15-20k documents.
  - `saev5.py::_fit_umap` : UMAP fitté deux fois (2D + 10D) sur une matrice dense, `n_jobs=1`
    forcé par `random_state`.
  - `saev5.py::_embed_bge_m3` : rechargé depuis le disque à chaque appel (jusqu'à 3× par run
    P1) — `del mdl` explicite en fin de fonction (discipline mémoire délibérée dans un
    process qui tient déjà Gemma-3-12B) ; un cache modèle (`lru_cache`) changerait ce
    compromis mémoire/vitesse et n'a pas été ajouté sans mesure VRAM réelle. Un cache disque
    des *embeddings de labels* (pas du modèle) resterait sans risque mémoire — à faire.
  - `saev5.py` : `df.iterrows()` sur le hover UMAP — boucle Python + `topk` torch par document.
- **Dernier reliquat du gaspillage filler, malgré G7.** G7 supprime l'encodage core et
  l'écriture de fragment pour un document filler, mais la boucle d'extraction fait toujours
  `all_doc_sae_acts.append(torch.zeros(d_total_expected, dtype=TORCH_DTYPE))` **par document
  filler** (`saev5.py`), puis `torch.stack` + `torch.save` sur l'ensemble. À 1,2M chunks
  filler (run 100M tokens) : une liste Python de ~1,2M tenseurs de zéros en RAM, puis sur
  disque. Représentation creuse (lignes non-filler + un index) à faire — pas fait dans cette
  passe : le format touche à la fois l'écriture (`saev5.py`) et tout consommateur de
  `all_doc_sae_acts` en aval, plus proche en risque de G2/G3 (format sur disque, tenu à
  l'écart tant qu'un run de référence n'a pas tourné sur le mécanisme de reprise actuel) que
  des correctifs isolés de cette passe.

## 3. Hygiène du dépôt

- **Nettoyage — fait** : 16 scripts d'audit forensiques + leurs `.slurm` + JSON associés
  supprimés (conclusions figées dans `RESULTS_TESTS.md`) ; les deux à conclusion réutilisable
  promus (`extraction_batch_size_sweep.py` → `benchmarks/`, invariant fp16/bf16 de
  `bf16_fp32_diagnostic.py` réécrit en test CPU synthétique →
  `tests/test_dtype_overflow.py`, le diagnostic GPU original supprimé) ; 6
  `docs/audit_*_results.json` correspondants supprimés ; `src/sae.egg-info/` (versionné par
  erreur) ; `CHANGELOG.md` (redondant, `check_docs.py` mis à jour) ; `logs/README.md`
  (dupliquait `docs/ops.md`) ; `test_chargement_sae.py` déplacé de la racine vers `scripts/`.
  `compare_to_frozen_benchmark.py` gardé délibérément (seul mécanisme de non-régression
  domaine général pour SAE Boost, cf. §1) plutôt que supprimé faute de référence — le
  raccrocher à un `.slurm` reste à faire, hors scope hygiène. `c2_original_only_rejudge.py`/
  `relabel_diff_csvs.py`/`augmentation_lexical_leakage_audit.py` gardés : non cités par
  `report/` mais cités par `RESULTS_TESTS.md` §37+ comme source de reproductibilité d'un
  résultat déjà publié — le critère "cité par `report/`" seul les aurait supprimés à tort.
  **Reste à faire, volontairement pas fait dans cette passe** (effort disproportionné au
  risque d'erreur pour une passe de nettoyage) : `docs/PDF_APPENDICES_EXTRACT.md` — réduction
  risquée sans casser les citations précises par numéro de ligne que
  `docs/INTERP_EMBED_COVERAGE.md` (travail original de qualité) fait vers ce fichier ; 104
  `.slurm` à consolider en template + `.env` — refactor large, casserait potentiellement des
  noms de fichiers cités nommément dans `RESULTS_TESTS.md`.
- **`pytest tests/ -q` ne passe qu'à un test près.** `scripts/check_docs.py` sort en 1 avec
  17 violations de première personne du singulier (`docs/INTERP_EMBED_COVERAGE.md`,
  `docs/PDF_APPENDICES_EXTRACT.md`) — réécriture de prose, pas de code. Tout le reste
  (mocks périmés, désynchronisations commentaire↔code, garde-fou placeholder incomplet,
  clé de cache P2, R1-R6) est corrigé.

## 4. Opérationnel

- **`sys.exit(0)` sur arrêt gracieux enregistre un run incomplet comme COMPLETED.**
  `saev5.py` (fin de boucle d'extraction P1, fin de boucle d'extraction P2) : un arrêt
  propre avant la fin des données (ex. `SIGUSR1` proche de la limite de temps SLURM, cf.
  reprise §2.3) sort en code 0. `sacct`/`--dependency=afterok` d'une chaîne de jobs ne peut
  pas distinguer ce cas d'un run réellement terminé — d'autant plus gênant que `CLAUDE.md`
  documente déjà les pièges `afterok`. Code de sortie distinct requis (64, par convention)
  pour un arrêt anticipé même propre.
- **`HF_HUB_OFFLINE=1` absent d'un seul fichier** sur 45 dans `slurm/pipeline_runs/`
  (`run_core_vs_extension_ablation.slurm`) — 44/45 l'ont déjà, item mineur restant plutôt
  qu'absent partout. `prepare_domain_dataset` garde un repli réseau
  (`load_dataset("wikimedia/wikipedia", streaming=True)`) si le cache local échoue, sans
  garde `local_files_only=True` cohérent avec le reste du dépôt.
- **Aucun test de bout en bout.** Les tests sont tous unitaires CPU. Un
  `test_pipeline_smoke` (SAE jouet 64 features, 20 documents, juge mocké,
  `USE_FROZEN_CORE=1`) attraperait d'un coup le bug Mails.tsv/train=test ci-dessus, une
  fuite de split, une désynchronisation de config et une régression de clé de cache — la
  classe de bugs qu'aucun test unitaire actuel ne voit, et qui coûte des runs GPU de 20h.

## 5. Data science / méthodologie

- **Corpus d'entraînement dominé par du texte généré par le modèle juge lui-même — résolu
  négativement à pleine puissance.** `MAX_AUGMENTED_PER_MAIL=13` (défaut, `saev5.py`) sur
  13 niveaux de perturbation. Premier test confondu par une part de filler variable (§76,
  RESULTS_TESTS.md) ; test corrigé à n=20 (part de filler recalibrée à ~6,3% dans les deux
  bras, §77/§78) montrait un écart énorme (40,0% vs 95,0%, p=0,0002) — **ne réplique pas à
  n=150 sous la méthode de sélection retenue depuis (stratifiée, §79)** : 82,0% (123/150,
  originaux + filler recalibré) vs 89,3% (134/150, mixte), z=-1,81, **p=0,070, non
  significatif**, sens de l'écart inversé (§81). B.1 résolu dans le sens négatif, comme C2
  (§48/§50/§52) : le corpus d'entraînement contaminé par le style du juge n'a pas d'effet
  démontrable sur l'interprétabilité à l'échelle testée — le signal à n=20 était
  vraisemblablement du bruit d'échantillonnage.
- **La métrique d'interprétabilité phare (45,3%, 68/150) était mesurée par défaut sur un
  échantillon biaisé par construction — tranché, sélection stratifiée devenue le défaut.**
  `feature_selection_by_magnitude` sélectionnait les *N* features par magnitude
  token-level moyenne — donc les plus denses, les plus proches de directions
  génériques/stop-word. Comparaison directe sur le même SAE/juge déjà en cache
  (`scripts/b2_stratified_selection_rejudge.py`, RESULTS_TESTS.md §79) : 68/150=45,3%
  (magnitude) vs **134/150=89,3% (stratifié par fréquence)**, z=-8,12, p=4,5×10⁻¹⁶ — la
  sélection par magnitude sous-estimait massivement le taux réel. `FEATURE_SELECTION_METHOD
  =stratified` est maintenant le défaut (`src/sae/saev5.py`). Le chiffre 45,3% déjà publié
  est à traiter comme un plancher, pas comme le taux réel du dictionnaire.
- Problème de moindre gravité, non traité : `information\w*` et `coupure\w*` (« urgence »)
  restent des motifs larges dans `INTENT_KEYWORDS_FR`, moins sévères que ne l'était
  `avoir\w*` (corrigé) mais pas resserrés.
- **Migration à faire : `augmented_mails.jsonl` existant (45 240 lignes) n'a pas
  `parent_sha1`.** La jointure par contenu (B.7, corrigée) exige ce champ ; en son absence
  `build_email_train_test_corpus` retombe sur l'ancienne jointure positionnelle (log explicite,
  pas de régression silencieuse), mais ne bénéficie pas encore de la protection contre un
  décalage de filtrage. Backfill possible sans regénération GPU (le texte n'a pas besoin de
  changer, seul `parent_sha1` manque) : recalculer `load_mails_tsv(Mails.tsv)` dans le même
  ordre qu'au moment de la génération et associer `parent_id` (position) → hash — à faire
  seulement si le `Mails.tsv` d'origine n'a pas changé depuis la génération, sinon la
  correspondance positionnelle qu'on backfillerait serait elle-même invalide.
- **Corrections à cet audit, faites en le vérifiant contre `RESULTS_TESTS.md` plutôt qu'en
  le prenant pour acquis** : le taux de rejet déséquilibré par axe (jusqu'à 59,6% pour
  `orthographe__degrade_fort`) était déjà mesuré et documenté en détail par `RESULTS_TESTS.md`
  §38 — ce document affirmait à tort qu'aucun audit ne le rapportait. L'hypothèse d'un biais
  de longueur via troncature du prompt à 2048 tokens (les mails longs perdraient des faits et
  seraient sous-représentés) est réfutée par mesure (§74) : un seul mail parent sur 3474
  dépasse le seuil de troncature, la corrélation longueur↔rejet est statistiquement
  significative mais négligeable (ρ=0,012), et longueur↔`facts_lost` n'est même pas
  significative (p=0,17, n=35 340). Le déséquilibre par axe existe bel et bien (§38) mais sa
  cause est `length_ratio` interagissant avec des axes qui raccourcissent le texte par
  construction, indépendante de la longueur du parent.
- **Aucune évaluation aval sur une tâche métier réelle** (routage, priorisation, détection
  de réclamation) avec baseline honnête TF-IDF+LogReg — la seule sonde existante est bâtie
  sur les labels faibles ci-dessus. Pour l'objectif affiché (outil transparent pour
  l'industrie), c'est ce qu'un jury demandera en premier.

## 6. Tenue du dépôt

- **Corrections à cet audit** : (a) le contraste README 20% (2/10) vs 45,3% (68/150)
  porte maintenant l'IC de Wilson ([5,7%, 51,0%]) et le résultat du test à deux
  proportions (`stats.py`, z=-1,56, p=0,12, non significatif à cet effectif). (b) le
  point d'entrée unique réclamé existe déjà et est déjà lié depuis le README —
  `docs/evaluation_protocol.md` (config exacte, `SAVE_DIR`, commandes, table
  capacité→résultat→comparaison) — l'audit avait affirmé son absence à tort. (c) la
  duplication `_strip_leading_objet_line`/nettoyage « Objet : » est corrigée : source
  unique `dataset.strip_leading_objet_line`, utilisée par `augmentation.py` et
  `preparation.py`. (d) `pyproject.toml` : auteur/email réels, description en français.
- **`LICENSE`/régime des données : tranché.** Dépôt privé pour l'instant, pas de licence
  ni de mention de régime des données à ajouter tant que ça reste le cas.

## 7. Priorisation restante

B2 et B1 sont tous deux tranchés à pleine puissance (§5) : B2 positivement (sélection
stratifiée nettement supérieure, devenue le défaut), B1 négativement (le signal à n=20 ne
réplique pas à n=150). **Le chiffre de référence 45,3%/68/150 est daté** — mesuré sous
l'ancien défaut `FEATURE_SELECTION_METHOD=magnitude`, qui sous-estime le taux réel d'un
facteur ~2 (89,3% sous stratifié). Toute figure/table du rapport citant 45,3% comme LE
taux d'interprétabilité du dépôt doit être requalifiée en plancher, ou refaite sous le
nouveau défaut.

Corrigés depuis : B7 (jointure de split par hash SHA1) ; B6 (label `remboursement`,
résultat concerné — sonde à 0,846 vs 0,855 de majorité — encore à re-mesurer avec le
label corrigé) ; B3/B5 (protocole odd-one-out) ; B4 (dédup des positifs) ; B11 (matching
lexical du corpus diffing).

**Sweep taille de modèle étendu à 27B, setup classique (K_EXTRA=5), n=150, RESULTS_TESTS.md
§82** : 4B 72,0% (108/150), 12B 82,0% (123/150, p=0,040 vs 4B), 27B 83,3% (125/150, p=0,76
vs 12B — plateau net au-delà de 12B). 1B a échoué une première fois (`Gemma3TextModel` sans
niveau `.language_model`, seul palier sans tour de vision — corrigé dans `saev5.py`),
relancé (job 45439, a100, toujours PENDING à la dernière vérification). Réplication du
balayage layer (§51) sous le même setup : layer 41 en cours (job 45367, h100, RUNNING),
layer 12 relancé (job 45440, h100, PENDING) — layer 31 déjà couvert par le sweep modèle
ci-dessus, layer 24 = référence. **À faire dès que ces jobs terminent** : ajouter la ligne
1B à la table §82, écrire la section RESULTS_TESTS.md pour le balayage layer 12/41 (même
format que §82), vérifier dans les logs que le nouveau cache d'extraction partagé
(`local_data/activation_cache/`, `sae_shared.py::compute_activation_cache_key`) a bien
fonctionné pour ces deux jobs (premiers runs sous ce code — vérifié seulement par tests
unitaires jusqu'ici, pas en conditions réelles).

**Nettoyage disque effectué cette session** : `results_v*/` réduits à leurs artefacts
légers (`results.json`, labels du juge, plots) sauf `results_v10_emails_main/` (gardé
complet, référence du dashboard Streamlit) — détail et scripts pour refaire chaque type
d'ablation dans `docs/archived_runs_manifest.md`. Scripts slurm strictement supersédés
supprimés (git log les préserve). Dashboard Streamlit vérifié fonctionnel après coup
(`streamlit.testing.v1.AppTest`, toutes pages × plusieurs runs, zéro exception).

**Reste à faire, jamais commencé cette session (le plus gros du travail "fidélité aux
papiers", §1)** : `HypothesisVerifier` + taux de vérification (App K.1, Diffing — sans lui
aucun chiffre de diffing n'est comparable au papier) ; NPMI_verified + filtres LLM
(Corrélations) ; génération de mots-clés LLM + accuracy par cluster + z-score de
conductance (Clustering) ; rerank LLM + agrégation RRF/RBO sur de vraies requêtes
(Retrieval — les métriques existent, `src/analysis/metrics.py`, mais ne sont câblées sur
aucun pipeline de requêtes réel) ; App I (protocole F1 latent-vs-juge 12B/27B — devenu plus
pertinent maintenant que le palier 27B existe réellement, §82).

**Qwen3.8-27B-FP8** (`scripts/imdb_genre_diffing_test.py`, hors pipeline SAE) : téléchargé,
vérifié structurellement complet (66 shards, `quantization_config` natif e4m3), ancien
checkpoint bf16 complet (52 Go) supprimé — jamais exécuté en conditions réelles depuis
le changement de checkpoint, job de vérification relancé (45443, h100, PENDING à la
dernière vérification ; `results.json` de l'ancien juge bf16/int8 déplacé vers
`results.json.bak_old_bf16int8_checkpoint` avant relance — sinon la logique de reprise du
script aurait sauté les 6 genres déjà présents et n'aurait jamais rechargé le nouveau
juge).

**Juge LLM découplé de l'extracteur (`JUDGE_MODEL_ID`, `src/config.py`)** : jusqu'à ce
correctif, `odd_one_out_judge`/`local_gemma_judge`/`generate_llm_diff_hypothesis`
(les 3 usages de juge LLM de `saev5.py`, Pipeline 1 ET 2) rechargeaient `MODEL_ID` —
le même checkpoint Gemma-3 que celui dont on extrait les activations, jugeant ses
propres features. §43/§63/§65 (`RESULTS_TESTS.md`) avaient déjà mesuré que le taux
d'interprétabilité n'est PAS robuste au choix du juge (45,3% avec gemma-3-12b-it vs
24,7% avec gemma-3-4b-it, écart robuste à 2 graines) mais ne pouvaient pas isoler
biais d'auto-préférence vs simple effet de capacité, faute d'un juge de famille
différente et de capacité comparable en cache local — c'est exactement ce que
Qwen3.8-27B-FP8 apporte maintenant. `JUDGE_MODEL_ID` (défaut : Qwen3.8-27B-FP8) est
désormais indépendant de `MODEL_ID` dans tout le pipeline principal ;
`src/sae/judge.py::load_judge_model` centralise le chargement (repli
`AutoModelForImageTextToText`/`AutoModelForCausalLM`, cf. docstring) et est réutilisé
par `scripts/imdb_genre_diffing_test.py` et `scripts/judge_model_separation_test.py`
(qui gagne aussi un tag de juge dérivé mécaniquement dans son nom de fichier de
sortie — l'ancien nommage ne gardait que la seed, collision silencieuse garantie
entre deux juges alternatifs différents testés à la même seed).
`pytest tests/ -q` reste vert (194/195, seul échec pré-existant et sans rapport :
`test_check_docs_clean`).

**Comparaison croisée Gemma/Qwen lancée** (job 45445, h100,
`scripts/judge_model_separation_test.py` avec `ALT_JUDGE_MODEL_ID=Qwen3.8-27B-FP8`,
rejuge les mêmes 150 features déjà en cache que §43/§63/§65) : **résultat encore à
écrire dans `RESULTS_TESTS.md` (§83) dès que le job termine** — c'est la mesure qui
tranche entre auto-préférence et effet de capacité. Premier essai (45444) OOM sur
a100 (39,49 Go, `AutoModelForImageTextToText` échoue puis le repli
`AutoModelForCausalLM` OOM aussi à l'allocation du cache mémoire) — les cartes a100
de ce cluster n'ont pas la marge pour Qwen3.8-27B-FP8 (~31 Go de poids) une fois
l'overhead de génération et le corpus d'activations ajoutés ; h100 obligatoire pour
tout chargement de ce juge (déjà le choix de `run_imdb_genre_diffing_test.slurm`,
`slurm/analysis/run_judge_model_separation_qwen.slurm` corrigé en cohérence).

**Caches judge existants non invalidés** : `results_v10_emails_main/cache/p1_judge_labels_extended.json`
et `p2_feature_labels.json` (référence du dashboard Streamlit, datés du 17/07, produits
par l'ancien code auto-jugeant gemma-3-12b-it) n'ont PAS été déplacés/régénérés — un
rerun complet du pipeline principal sur ce `SAVE_DIR` chargerait encore silencieusement
ce cache Gemma via la logique `if os.path.exists(judge_cache)`. Décision volontaire de
ne pas y toucher cette session (coût d'un rerun complet du pipeline non chiffré, actif
« référence dashboard » à ne pas casser sans le vouloir) — à régénérer explicitement
avant de citer un chiffre du dashboard comme jugé par Qwen.

**HypothesisVerifier (App K.1) implémenté** (`src/analysis/hypothesis_verifier.py`,
`tests/test_hypothesis_verifier.py`, 6 tests, CPU/mock uniquement) : adaptateur de
`external/interp_embed/paper/diffing/hypothesis_verifier.py::HypothesisVerifier` vers
le juge local du dépôt (leur code appelle une API OpenAI/OpenRouter, incompatible avec
ce dépôt) — prompt repris verbatim du PDF (§K.1) plutôt que du léger écart présent dans
le code des auteurs (détail dans le docstring du module). `verify_hypotheses` (matrice
hypothèse × document) + `compute_verification_metrics` (`verification_rate` = fraction
d'hypothèses dont la différence de fréquence vérifiée dépasse 1% ; `coverage` = fraction
des documents cible couverts par au moins une hypothèse valide) implémentent les deux
métriques définies Figures 11/12 du papier. C'était le plus gros gap "fidélité papier"
identifié (§1) : rien dans le diffing n'était comparable au papier sans ça.

**Première mesure réelle lancée** (job 45471, h100,
`scripts/diffing_hypothesis_verification_test.py`) : reprend les 10 hypothèses les plus
significatives (sens énergie>sports, triées par q) de `p1_diff_energy_sports.csv`
(features SAE déjà labellisées), reconstruit un corpus energy/sports frais
(`prepare_domain_dataset`, 40 documents/domaine, FineWeb2-fr) et les vérifie avec le
juge Qwen. **Résultat encore à écrire dans `RESULTS_TESTS.md` (§84, après §83 pour la
comparaison Gemma/Qwen) dès que le job termine.**

**NPMI_verified + Clustering LLM implémentés** (`src/analysis/correlations_verified.py`,
`src/analysis/clustering_llm.py`, `tests/test_correlations_verified.py` +
`tests/test_clustering_llm.py`, 20 tests, CPU/mock/synthétique uniquement,
`pytest tests/ -q` toujours vert). Formules retrouvées dans le CORPS PRINCIPAL du PDF
(pas seulement les Appendices déjà extraites dans `docs/PDF_APPENDICES_EXTRACT.md`) via
`pypdf` (`.venv` du dépôt, aucun outil système requis) — le z-score de conductance
notamment était marqué "EXTRACTION INCERTAINE" dans `docs/PDF_APPENDICES_EXTRACT.md`
ligne 545, retrouvé page 7 : *"the z-score of each cluster's conductance in dense
embedding space relative to a random sample (lower = tighter)"*.

- `correlations_verified.py` : `filter_syntactic_labels` (K.2, prompt verbatim),
  `is_trivial_same_token_pair` (E.1, réutilise `fragment_store.feature_column`, pas de
  nouvelle lecture de tenseur), `verify_pair_presence` + `compute_verified_npmi` (E.1/E.3,
  même formule NPMI que `cooccurrence.compute_npmi`, pas de réinvention),
  `conditional_occurrence` (CO=max(P(i|j),P(j|i)), E.1).
- `clustering_llm.py` : `generate_keywords` + `select_latents_union` (App F.1, remplace
  le `axis_query` unique de `targeted_clustering_by_axis`), `generate_cluster_labels`
  (App F.1), `compute_cluster_accuracy` (§4.3 p.7 + prompt système K.3),
  `conductance`/`conductance_zscore` (graphe k-NN networkx sur embeddings denses, null
  par échantillons aléatoires de même taille — validé sur 2 blobs synthétiques bien
  séparés : conductance quasi nulle, z très négatif, cohérent avec "lower=tighter").
- **Bug corrigé au passage** (pas une divergence documentée, un vrai bug) :
  `targeted_clustering_by_axis` utilisait `SpectralClustering(affinity="cosine")` sur
  activations binarisées — le papier utilise l'affinité de **Jaccard**, pas cosine (deux
  métriques différentes sur un vecteur binaire). Corrigé (matrice de similarité Jaccard
  précalculée, `affinity="precomputed"`), plus un paramètre `keywords` optionnel qui
  bascule vers `select_latents_union` (cap `max_docs_for_jaccard=5000` ajouté pour la
  matrice O(n²), R4 — le corpus test actuel ~2200 docs reste très en dessous).
- **Limitation architecturale découverte en cours de route** : `saev5.py` exécute tout
  son pipeline au niveau MODULE (pas de garde `if __name__ == "__main__":`) — il ne peut
  donc pas être importé comme bibliothèque (`import src.sae.saev5` déclenche le run
  complet). `select_latents_by_similarity`/`_embed_bge_m3` ont dû être dupliqués (courts,
  documentés comme tels) dans `scripts/clustering_llm_test.py` plutôt que réutilisés.
  Non corrigé cette session (risque de déstabiliser le point d'entrée principal sans test
  complet) — à corriger si `saev5.py` doit un jour être importé ailleurs que comme script.

**Deux jobs de validation lancés** : 45473 (h100, `npmi_verified_test.py`, 8 paires
candidates parmi les features d'extension déjà labellisées, filtre syntaxique+trivial,
vérification sur 60 documents frais) et 45474 (h100, `clustering_llm_test.py`, requête
"type de réclamation client", 4 clusters, 300 documents, chaîne complète mots-clés→
union→Jaccard→labels→accuracy→z-conductance). **Résultats encore à écrire dans
`RESULTS_TESTS.md` (§85 NPMI_verified, §86 clustering) dès qu'ils terminent.**

**Incident de session (les 7 jobs ci-dessus ont TOUS échoué, deux causes distinctes,
toutes deux corrigées) :**

1. **`NameError: JUDGE_MODEL_ID` dans `saev5.py`** (45439 1B, 45440 layer12) : le
   découplage juge/extracteur (plus haut dans ce document) ajoutait `JUDGE_MODEL_ID` dans
   un f-string de log sans l'importer depuis `src.config` dans l'espace de noms de
   `saev5.py` — les DEUX jobs ont fait l'extraction COMPLÈTE (2h37/2h55, cache d'extraction
   partagé confirmé fonctionnel — "[P1] Cache d'extraction partagé" présent dans les deux
   logs) puis planté juste avant le chargement du juge, sur cette seule ligne. Corrigé
   (import ajouté, `src/sae/saev5.py:47`). Un smoke-test d'import fait plus tôt dans la
   session (`python src/sae/saev5.py`, arrêt attendu sur l'accès réseau HF) ne couvrait
   PAS ce point du flux — il s'arrêtait avant, à une étape antérieure du pipeline. Leçon :
   un smoke-test d'import ne vaut que jusqu'au point où il s'arrête, pas au-delà.
   `pyflakes` (installé cette session, `.venv`) aurait détecté ce `NameError` par analyse
   statique sans dépenser de GPU — à faire systématiquement sur `saev5.py` après toute
   édition touchant un chemin non exercé par `pytest tests/ -q` (qui ne couvre que des
   fonctions isolées, jamais `saev5.py` de bout en bout, cf. limitation notée plus haut
   sur l'absence de garde `if __name__ == "__main__"`). Deuxième signal `pyflakes` sur
   `analyze_with_umap::sae_active` vérifié FAUX POSITIF (fermeture Python standard,
   confirmé par exécution isolée de la fonction) — pas d'action nécessaire.
2. **Qwen3.8-27B-FP8 ne peut pas générer sans le paquet PyPI `kernels`** (45443, 45445,
   45471, 45473, 45474) : le chargement du checkpoint (`from_pretrained`) réussit, mais le
   premier appel `generate()` échoue (`finegrained-fp8 kernel requires the kernels
   package`) — jamais détecté avant car aucun job Qwen n'avait atteint un vrai appel de
   génération jusqu'ici. `kernels` installé (`uv add`/`uv pip install`) dans les DEUX venvs
   concernés, à des versions DIFFÉRENTES et incompatibles entre elles — piège trouvé en
   corrigeant : `external/interp_embed/.venv` (transformers 5.15.0) exige
   `kernels>=0.16.0,<0.17.0` (version demandée explicitement par son propre message
   d'erreur), mais appliquer la MÊME contrainte au `.venv` principal (transformers 5.12.1)
   a cassé l'import de `transformers` lui-même dans toute la suite `pytest`
   (`ValueError: Either a revision or a version must be specified`,
   `transformers/integrations/hub_kernels.py`) — chaque venv a sa propre contrainte
   `kernels`, dérivée de `importlib.metadata.requires('transformers')`, jamais supposée
   égale entre les deux. `.venv` : `kernels>=0.12.0,<0.13` (ajouté à `pyproject.toml`
   via `uv add`, donc verrouillé dans `uv.lock`) ; `external/interp_embed/.venv` :
   `kernels>=0.16.0,<0.17.0` (installé directement dans le venv, PAS dans son
   `pyproject.toml` — dépôt vendorisé externe, pas le nôtre). `pytest tests/ -q` revérifié
   vert après coup (217/218, seul échec pré-existant sans rapport). **Le juge Qwen n'a
   encore JAMAIS produit une génération réussie de bout en bout à ce stade** — les 7 jobs
   ci-dessus viennent d'être relancés avec les deux correctifs ; premier vrai test de
   `load_judge_model` en conditions réelles.

**Nettoyage suite tests/ (demande explicite)** : `test_bfloat16.py` et `test_checkpoint.py`
supprimés (les deux testaient explicitement AUCUN code du dépôt, déjà signalés comme tels
par un audit antérieur sans jamais être retirés — `test_checkpoint.py` entrait de plus en
collision de nom avec `test_checkpoint_resume.py`, le vrai test du module checkpoint) ;
`test_interp_embed_diff.py::test_corpus_diff_stats_vs_interp_embed` supprimé (code mort
confirmé empiriquement — `interp_embed` n'est importable dans AUCUN venv qui exécute
`tests/`, le test ne faisait jamais que son `return` précoce). Revue complète des 194 tests
pré-existants faite fichier par fichier ; le reste jugé solide (régressions documentées,
assertions non tautologiques, plusieurs tests ayant eux-mêmes déjà attrapé un bug réel en
étant écrits, ex. `test_retrieval.py` sur un piège d'indexation CSR/CSC).

**Qwen3.8-27B-FP8 abandonné, remplacé par Qwen3.8-27B bf16** : diagnostic complet fait
(cf. item 2 ci-dessus) — `deep-gemm` (chemin rapide) n'a de binaire précompilé qu'à partir
de torch>=2.9 (repo HF vérifié : aucune variante en dessous de torch29, ce dépôt est en
torch==2.6.0) ; son repli Triton (`kernels-community/finegrained-fp8`) télécharge et
s'importe, mais référence `torch.float8_e8m0fnu`, absent de torch==2.6.0. Décision
utilisateur : re-télécharger le bf16 (`Qwen/Qwen3.8-27B`, 55,6 Go, non gated) plutôt que de
monter torch (risque sur sae_lens/transformer_lens, non exploré) ou changer de modèle.
Téléchargé vers `models/Qwen3.8-27B` (`snapshot_download`, depuis le nœud frontal comme
`download_sae.py` — pas un calcul, un transfert réseau). `JUDGE_MODEL_ID` (défaut,
`src/config.py`), `load_judge_model` (`src/sae/judge.py` — `torch_dtype="auto"` ajouté,
absent avant, aurait chargé en fp32 par défaut et doublé la VRAM/le temps de chargement
pour un checkpoint bf16 non pré-quantifié), `scripts/imdb_genre_diffing_test.py`
(`JUDGE_MODEL_PATH`), `slurm/analysis/run_judge_model_separation_qwen.slurm`
(`ALT_JUDGE_MODEL_ID`) mis à jour vers le nouveau chemin ; `--mem` remonté à 96G sur les 4
scripts slurm juge Qwen (poids bf16 plus gros que FP8). L'ancien checkpoint FP8
(`models/Qwen3.8-27B-FP8`, ~31 Go) laissé sur disque (pas supprimé — 6,4 To libres,
aucune urgence) plutôt que de le nettoyer sans le vouloir.

**Bf16 téléchargé, jobs relancés (45614/45615/45618), DEUX terminés sans planter —
mais résultats invalidés à l'inspection** : `interp_rate_alternative: 0,0/150` (45615,
comparaison juge) et `acc=0,000` sur les 4 clusters (45618, clustering) sont tous les deux
des artefacts, pas des résultats — un tirage aléatoire sur l'odd-one-out (10 items)
donnerait déjà ~10%, 0 exact sur 150 est le signal. Cause : Qwen3.8-27B active par défaut
un préambule `<think>...</think>` (mode raisonnement Qwen3, template Jinja du checkpoint,
jamais désactivé dans aucun appel `apply_chat_template` du dépôt) qui consomme à lui seul
`max_new_tokens=8` (étape odd-one-out d'`odd_one_out_judge`) sans jamais atteindre la
réponse — confirmé en clair dans le log 45618 : la "génération de mots-clés" a produit
comme premier "mot-clé" le préambule de raisonnement lui-même, tronqué en plein mot à la
limite de tokens. `enable_thinking=False` ajouté à `_batched_generate`
(`src/sae/judge.py`, seul point d'appel `apply_chat_template` utilisé par
`odd_one_out_judge`/`local_gemma_judge`/les 3 nouveaux modules `*_verified`/
`hypothesis_verifier`/`clustering_llm`) et à `imdb_genre_diffing_test.py`. Argument
silencieusement ignoré par les templates qui ne le déclarent pas (Gemma, vérifié
empiriquement avant déploiement) — sans risque de régression sur le juge Gemma existant.
`_FakeTokenizer` de `test_batched_generate_length_sort.py` avait une signature
`apply_chat_template` trop stricte pour tolérer ce nouveau kwarg (échec immédiat en test,
pas en job) — élargie en `**kwargs`, comme le vrai tokenizer. `pytest tests/ -q` revérifié
vert (217/218). Les DEUX résultats corrompus déplacés (pas supprimés) en
`*.bak_thinking_mode_bug` ; jobs 45616/45617 (encore en cours au moment du diagnostic)
annulés plutôt que laissés produire le même artefact. Tous les jobs juge Qwen relancés
(45619-45622 + 45614 imdb).

**45619 (comparaison Gemma/Qwen) terminé, résultat vérifié bon cette fois** (labels
lisibles inspectés à la main, pas seulement le taux agrégé) : **RESULTS_TESTS.md §83**.
Qwen3.8-27B juge 78,7% (118/150) interprétable contre 45,3% (68/150) pour
gemma-3-12b-it auto-jugé — écart massif (McNemar apparié, p=1,9e-8) mais dans le sens
OPPOSÉ à l'hypothèse d'auto-préférence testée (un biais d'auto-préférence prédirait
gemma-3-12b-it plus généreux avec lui-même, pas 74% plus sévère). Ne tranche pas "Qwen
meilleur juge" vs "Qwen plus complaisant" — détail et limites dans §83.

**45620 (App K.1, hypothesis verification) terminé, résultat sain (mix valide/invalide,
taux non dégénérés — pas d'artefact de type "thinking mode")** : **RESULTS_TESTS.md §84**.
verification_rate=40% (4/10 hypothèses), coverage=92,5% des documents énergie — 6 des 10
hypothèses SAE les mieux classées par NPMI/q-value (run archivé) ne se répliquent pas sur
un corpus frais à seuil 1%. Détail et limites (n=10 petit, IC large) dans §84.

**45621 (NPMI_verified) terminé mais résultat écarté après inspection** : les 6 paires
vérifiées avaient toutes NPMI_verified=CO=1,000 exactement — suspect (0 bugs de
génération cette fois, mais un vrai gap méthodologique dans `npmi_verified_test.py` : le
script sélectionnait les paires au NPMI brut le plus fort SANS le filtre de dissimilarité
sémantique des labels que l'Appendix E.1 du papier impose explicitement (son propre
exemple motivant : "dog" et "pet") — vérifié : les 6 paires étaient toutes des
quasi-synonymes ("Réclamation Client"/"Réclamations Clients", "Numéro Téléphone"/
"Coordonnées"...), corrélation triviale et attendue, pas un résultat "intéressant" au sens
du papier. Filtre ajouté (embedding bge-m3 des labels, seuil sim<0,2, même convention que
`cooccurrence.find_interesting_pairs`) ; matrice de présence brute désormais sauvegardée
dans le JSON de sortie (pas seulement les stats agrégées) pour diagnostiquer un résultat
suspect sans tout relancer. Ancien résultat déplacé (pas supprimé) en
`npmi_verified.json.bak_no_dissimilarity_filter`. Relancé (job 45623).

**45623 (NPMI_verified, avec filtre dissimilarité) terminé mais 0 paires candidates** :
0 paire (NPMI>0.3, sim label<0.2) parmi les 68 features interprétables — le filtre est
peut-être trop strict pour ce petit ensemble (68 features d'un domaine narrow, emails de
réclamation client, où les concepts co-occurrents sont probablement aussi souvent
sémantiquement proches). Pas encore tranché : diagnostic lancé (job 45630, a100, léger —
bge-m3 seul, pas de rechargement du juge — `scripts/npmi_similarity_diagnostic.py`,
affiche la distribution complète (npmi, sim) pour calibrer un seuil justifié plutôt que
d'en choisir un au hasard). **§85 pas encore écrit, en attente de ce diagnostic.**

**45622 (clustering LLM, 2e tentative avec le correctif thinking) terminé, résultat sain**
(mots-clés cohérents, labels de cluster cohérents, accuracy et z-conductance non
dégénérés, variance inter-cluster qualitativement cohérente avec l'observation du papier
"SAE clusters ... generally higher variance across clusters") : **RESULTS_TESTS.md §86**.

**45614 (imdb, vérification checkpoint) terminé, résultat sain** (scores variés 0,0-1,0,
labels par genre cohérents, SAE avg=0,603/LLM baseline=0,867 vs publié 0,75/0,90) —
confirme que le juge Qwen bf16 fonctionne aussi sur une tâche de génération réelle (pas
seulement odd-one-out), tâche originelle #4 du handoff de session close. Pas de nouvelle
section RESULTS_TESTS.md (script de démonstration/smoke-test, `slurm/validation/`, pas un
§N).

**Diagnostic 45630 terminé, seuil recalibré** : parmi les 165 paires NPMI>0,3, similarité
label observée entre 0,369 et 1,000 (médiane 0,514) — le seuil absolu du papier (sim<0,2,
App E.1, calibré sur CivilComments/Pile, des milliers de features multi-domaines) ne
sélectionne jamais rien sur ce dépôt (150 features d'UN SEUL domaine narrow, emails de
réclamation client) : pas une absence réelle de paires "moins reliées que la moyenne",
mais une échelle de dissimilarité différente d'un dictionnaire à l'autre. `npmi_verified_test.py`
corrigé : `LABEL_SIM_PERCENTILE` (percentile 25 de la distribution observée à CHAQUE run,
pas un seuil absolu codé en dur) remplace `LABEL_SIM_THRESHOLD` — reproduit l'intention du
filtre (écarter les quasi-synonymes évidents) sans halluciner un chiffre transférable
entre domaines de largeur sémantique différente. Ancien résultat (0 paires) déplacé en
`npmi_verified.json.bak_zero_pairs_strict_threshold`. Relancé (job 45631, terminé).

**45631 terminé, résultat écrit : RESULTS_TESTS.md §85.** 5 paires survivent tous les
filtres, NPMI_verified 0,80-1,00 mais 4/5 avec seulement 1-2 documents positifs sur 60 --
signal réel mais peu robuste statistiquement (perfection quasi automatique à si peu
d'effectif). `N_VERIFY_DOCS=60` à augmenter avant de citer un chiffre individuel comme
fiable — noté explicitement en limite plutôt que présenté comme un résultat solide.

**§83-§86 tous écrits.** Reste seulement : ligne 1B pour §82 (job 45579 PENDING), section
layer 12/41 (job 45580 RUNNING, ~40 min).

## 8. Audit externe (13 items, reçus en fin de session — priorité pour la suite)

Passe indépendante (pas de la session qui a introduit le cache partagé, §7) sur ce
cache et sur des items jamais couverts par les audits précédents. N1/N2/N3 traités
cette session (ci-dessous) ; N4 lancé, résultat en attente (job GPU en file) ; N5-N13
toujours enregistrés tels que reçus, pas contre-vérifiés.

**N1 🟢 CORRIGÉ — le cache d'extraction partagé pouvait charger un fragment de la
mauvaise largeur, et pire : le ré-encodage écrivait ET purgeait `raw_acts` DANS le
cache partagé lui-même.** Diagnostic initial (padding D_EXTRA-dépendant à
l'extraction) confirmé mais incomplet : la vraie gravité venait du ré-encodage
(`saev5.py`, boucle `merge_extra`/`save_fragment`) qui écrivait ses fragments fusionnés
(core+extension) et purgeait `raw_acts` **directement dans `token_fragments_dir`**
(partagé, symlinké), puis supprimait les shards d'extraction une fois "terminé" —
un second run de D_EXTRA/K_EXTRA différent partageant la même clé d'extraction
héritait soit de colonnes extension d'un AUTRE SAE (largeur/D_EXTRA différents), soit
plantait avec `KeyError: 'raw_acts'` (raw_acts déjà purgé par le premier run).
**Confirmé en conditions réelles** : les deux jobs de §7 (1B et layer12, relancés en
45579/45580 sous l'ancien code) ont TOUS LES DEUX planté avec exactement ce
`KeyError: 'raw_acts'`. Correctif (`saev5.py`, `sae_shared.py`) : le ré-encodage écrit
désormais dans `ext_fragments_dir` (`SAVE_DIR/cache/p1_token_fragments_ext`, privé par
run, jamais symlinké) ; `token_fragments_dir` (partagé) reste intégralement en lecture
seule après l'extraction — plus aucune écriture, plus de suppression de shards. La
largeur des fragments RAW à l'extraction est désormais toujours `d_core` (jamais
`d_core+D_EXTRA`) : D_EXTRA n'a plus aucune influence sur le contenu du cache partagé.
8 scripts d'analyse consommateurs de features extension (`judge_model_separation_test.py`,
`b2_stratified_selection_rejudge.py`, `npmi_verified_test.py`, etc.) repointés via
`resolve_extension_fragments_dir` (`src/storage/fragment_store.py`) — bascule
automatiquement vers `p1_token_fragments_ext` si présent (runs post-correctif), sinon
repli sur `p1_token_fragments` (runs "legacy" comme `results_v10_emails_main/`, qui
n'ont jamais connu le cache partagé et gardent leurs fragments fusionnés directement à
la racine). Les deux caches partagés déjà corrompus par les tentatives 45579/45580
(44253 fichiers individuels périmés chacun) nettoyés ; jobs relancés sous le code
corrigé (45724, 45725). 9 tests ajoutés (`tests/test_shared_cache_lock.py` couvre
aussi N2), `pytest tests/ -q` vert (226/227, seul échec pré-existant sans rapport).

**N2 🟢 CORRIGÉ — aucun verrou sur le cache partagé.** `acquire_shared_cache_lock`
(`src/sae/sae_shared.py`) : verrou de création exclusive sur
`<cache_dir>/.extraction.lock`, tient de la création des liens symboliques jusqu'à la
fin de l'extraction RAW (relâché avant le ré-encodage, qui n'a plus besoin du verrou
depuis N1). Un second prétendant de même clé ATTEND (poll) plutôt que d'échouer ou de
dupliquer le travail — cohérent avec le pattern de course entre partitions déjà en
usage (mémoire `feedback_sae_gpu_scheduling_race_partitions`). Heartbeat +
expiration (`stale_after_s`, défaut 300s) pour l'auto-guérison si le détenteur meurt
sans libérer. **Bug trouvé en écrivant les tests** (pas dans le premier jet livré à
l'audit externe) : la première implémentation rafraîchissait le heartbeat par
`open(path, "w")` — tronque avant d'écrire, fenêtre réelle où un lecteur concurrent
voit un fichier vide/tronqué et vole le verrou d'un détenteur pourtant actif
(reproduit empiriquement, ~20% des runs du test de course sous charge). Corrigé avec
`atomic_create_exclusive`/`write_checkpoint` (tmp + `os.link`/`os.replace`, jamais de
troncature en place) ajoutés à `src/storage/checkpoint.py`, réutilisés par le verrou
— 5 exécutions répétées du test de course, 0 échec après correctif.

**N3 🟡 INVESTIGUÉ, chiffre repondéré non obtenu — critique confirmée par le code,
mais non reconstructible a posteriori sur les données actuelles.** Détail complet :
`RESULTS_TESTS.md` §87. La sur-pondération des bins rares par
`feature_selection_stratified_by_frequency` (`per_bin = n_features // n_bins_eff`,
indépendant de l'effectif du bin) est un fait structurel du code, confirmé sans
ambiguïté. En revanche, rejouer la sélection avec le même SEED sur les mêmes
fragments (`results_v10_emails_main/`, job 45742, CPU-only) ne reproduit QUE 68/150
des features de `b2_stratified_selection_rejudge.json` (§79) — cause dominante non
identifiée (un bug mineur trouvé et corrigé au passage, troncature de 4 colonnes sur
1024 dans le calcul de `d_total` du script original, n'explique pas l'essentiel de
l'écart). `feature_selection_stratified_by_frequency` étendue
(`return_bin_info=True`) et `horvitz_thompson_mean` ajouté à `src/analysis/stats.py`
— `b2_stratified_selection_rejudge.py` capture désormais `bin_info` DIRECTEMENT au
moment de la sélection, pour qu'un futur rerun produise un taux par bin fiable sans
dépendre d'une reproduction ultérieure fragile. Aucun rerun de juge lancé (coût GPU
non trivial, hors du "zéro rerun" de la demande N3 initiale).

**N4 🟡 EN COURS — job GPU lancé, résultat pas encore disponible.** L'arme "mixte
stratifiée" de B.1 (§79, 134/150, déjà en cache dans `results_v10_emails_main/`)
rejugée avec Qwen3.8-27B (job 45735, `scripts/b1_stratified_mixte_qwen_rejudge.py`,
même patron que `judge_model_separation_test.py`). **L'arme "originaux+filler" de
§81 NE PEUT PAS être rejugée de la même façon** : ses fragments token-level
(`results_v26_validation_layer24_v12_originals_filler_matched_n150_h100/`) ont été
supprimés par le nettoyage disque de la session précédente (`docs/
archived_runs_manifest.md`, seul `results_v10_emails_main/` gardé complet) — la
rejuger demanderait une extraction complète fraîche, pas "quelques minutes de GPU"
comme prévu par la demande N4 initiale. Le test complet de l'interaction juge×corpus
(les deux bras, mêmes deux juges) reste donc hors de portée sans ce rerun ; seul un
signal partiel (Qwen est-il systématiquement plus/moins généreux que Gemma sur l'arme
mixte ?) sera disponible une fois 45735 terminé.

**N5 🟢 CORRIGÉ — B6 corrigé complètement.** Confirmé exact puis corrigé :
vérification empirique directe sur `local_data/emails/Mails.tsv` (grep + inspection
manuelle des occurrences, pas une lecture de tenseur — autorisée sur le nœud frontal) —
31/31 occurrences de "d'avoir" et 4/4 de "l'avoir" y sont l'usage VERBAL ("je ne suis
pas certain d'avoir compris", "je vous remercie de me l'avoir envoyée"), **0 nominal**,
contre seulement 2 occurrences de "un avoir" (nominal, correct) et 0 de "mon"/"notre
avoir". `INTENT_KEYWORDS_FR["remboursement"]` (`src/data/dataset.py`) ne matche plus que
`un\s+avoir|mon\s+avoir|notre\s+avoir` — "l'avoir"/"d'avoir" retirés, trop ambigus pour
être désambiguïsés par regex et 100% bruit sur ce corpus. `tests/test_intent_keywords_fr.py`
mis à jour (5 tests). §80 (Latent Terms, remboursement P@10 0,90 vs 0,30 TF-IDF)
re-mesuré sous le label resserré : job 45745 lancé (ancien résultat sauvegardé en
`.bak_pre_n5_regex_fix`), résultat en attente.

**N6 🟢 CORRIGÉ — nombre d'items de l'odd-one-out désormais loggé + taux corrigé du
hasard disponible.** Confirmé exact (`judge.py`, `len(pos_examples)` varie de 3 à
`n_pos`=9 selon la fréquence de la feature, jamais fixé). `odd_one_out_judge` et
`local_gemma_judge` ajoutent désormais `n_items` au dict de résultat par feature
(= `len(pos_examples) + (1 si neg_example)`). `chance_corrected_rate` ajouté à
`src/analysis/stats.py` (formule Cohen-kappa agrégée, `(obs-c)/(1-c)` avec
`c = moyenne(1/n_items)` sur le même ensemble de features que `obs`) — permet de
publier un taux agrégé corrigé du hasard variable en une ligne, sans réinventer le
calcul par script. Les résultats DÉJÀ produits (avant ce correctif) n'ont pas `n_items`
dans leur cache — reconstructible depuis `len(pos_examples)+1` si besoin de les corriger
rétroactivement. 3 tests ajoutés (`test_judge_batching_orchestration.py`) + 3
(`test_stats.py`).

**N7 🟢 CORRIGÉ — `max_length` paramétré** (`MAX_LENGTH`, `src/config.py`, défaut 512 =
comportement inchangé). Remplace le `max_length=512` en dur dans la boucle d'extraction
(`saev5.py`). Aucune mesure d'impact chiffrée faite cette session (nécessiterait de
tokenizer le corpus réel, coût CPU non trivial mesuré empiriquement — un essai direct
sur le nœud frontal a dépassé 15s, donc hors du "config check borné" toléré, devrait
passer par `sbatch`) — le paramètre existe désormais pour qu'une ablation future le
mesure sans changer le code.

**N8 🟢 CORRIGÉ — les 3 paramètres ajoutés au payload de la clé de cache.**
`compute_activation_cache_key` (`sae_shared.py`) prend désormais `max_length`,
`sigma_clip`, `skip_first_content_token` en paramètres obligatoires (plus de valeurs en
dur non trackées) ; `SIGMA_CLIP`/`SKIP_FIRST_CONTENT_TOKEN` ajoutés à `src/config.py`
(mêmes défauts que le comportement précédent : 4.0/`True`). Docstring corrigée (ne dit
plus hacher "le corpus réellement vu" sans préciser que `max_length` couvre la
troncature). **Conséquence attendue et voulue** : ce changement invalide TOUTES les clés
de cache calculées avant ce correctif (le payload a changé) — les caches partagés
existants (dont les deux nettoyés pour 45724/45725 plus haut) ne seront plus jamais
retrouvés par leur ancienne clé, un prochain accès en calcule une nouvelle et réextrait.
6 tests ajoutés (`test_activation_cache_key.py`).

**N9 🟢 CORRIGÉ — sémantique changée de l'ablation de seed documentée** dans
`CLAUDE.md` (section Seeds + point 5 de la checklist diagnostics) : `SEED` n'entre pas
dans la clé de cache partagé (volontaire, R5 — la clé ne couvre que ce qui affecte
l'extraction), donc une ablation de seed sous le cache partagé ne fait plus varier que
l'init/le shuffle du SAE (le réservoir de tokens est partagé) — plus étroit que l'effet
mesuré par tout chiffre "seed dans le bruit" cité d'avant l'introduction du cache
partagé. Documentation uniquement, aucun changement de comportement.

**N10 🟢 CORRIGÉ — les lignes filler ne sont plus stockées sur le cache partagé.**
`save_doc_acts_sparse_filler`/`load_doc_acts_sparse_filler`/`load_all_doc_acts`
(`sae_shared.py`) : la plage filler `[n_train, n_train+n_filler)` (jamais lue en aval,
confirmé par grep) est exclue du tenseur sauvegardé sur `local_data/activation_cache/`,
reconstruite à zéro au chargement — la connaissance de la plage vient de l'INDEX
(n_train/n_filler), pas d'une détection de valeur, donc correcte même si une ligne
filler n'était pas exactement nulle en mémoire (cas `torch.empty` du ré-encodage, hors
scope ici puisque ce fichier-là reste privé par run). Risque de rupture identifié et
traité : **12 scripts d'analyse** lisaient `p1_all_doc_acts.pt` en repli direct par
`torch.load` — tous repointés vers `load_all_doc_acts` (dispatch sur le CONTENU du
fichier, dict compact vs tenseur dense classique, donc rétro-compatible avec
`p1_all_doc_acts_ext_d*.pt`, jamais compacté). 5 tests ajoutés
(`test_doc_acts_sparse_filler.py`).

**N11 🟢 STRUCTUREL — frein documenté + `slurm/archive/` créé** (mêmes sous-dossiers que
`slurm/`). Règle publiée dans `docs/ops.md` : archiver un `.slurm` dès que son `§N` est
écrit, réutiliser un `.slurm` existant (via export d'env) avant d'en créer un nouveau.
**Pas d'archivage rétroactif fait cette session** (94 fichiers actuels, auditer lequel
correspond à un `§N` déjà clos ligne par ligne est un travail à part, risqué à bâcler
sous contrainte de temps — mieux vaut le faire au fil de l'eau, en archivant chaque
script dès que sa section est écrite, ce que cette session a commencé à faire pour les
scripts qu'elle a elle-même utilisés).

**N12 🟢 CORRIGÉ — `check_docs.py` vert** (`python scripts/check_docs.py` →
"Aucune violation trouvée"). Décision tranchée : `docs/INTERP_EMBED_COVERAGE.md` et
`docs/PDF_APPENDICES_EXTRACT.md` **exclus explicitement** du contrôle "première
personne" (`FIRST_PERSON_EXCLUDED_FILES`, justification en commentaire dans
`check_docs.py`) — ce sont des analyses comparatives à la première personne par nature
("mon pipeline" vs. un dépôt tiers), pas des sections citées par le rapport où la règle
"présent, sans récit" (`RESULTS_TESTS.md`) s'applique. Les autres contrôles (version
interne, TODO, placeholder, lien mort) restent actifs sur ces deux fichiers. La violation
RESULTS_TESTS.md restante ("v2" dans un en-tête de tableau, §57 correctif
`INTENT_KEYWORDS_FR`) corrigée par reformulation ("motif réel" au lieu de "v2, réel").
`tests/test_docs.py` vert.

**N13 🟢 PARTIELLEMENT TRAITÉ — citation par ligne remplacée par ancre de section.**
La seule citation par NUMÉRO DE LIGNE de `INTERP_EMBED_COVERAGE.md` vers
`PDF_APPENDICES_EXTRACT.md` (K.3/K.4, lignes 846–888) remplacée par une citation par
ancre (`§K.3`/`§K.4`, en-têtes markdown confirmés présents) — rend une future réduction
de `PDF_APPENDICES_EXTRACT.md` sûre du point de vue des références internes. **La
décision de fond (réduire/reformuler le contenu verbatim de `PDF_APPENDICES_EXTRACT.md`
pour limiter l'exposition copyright) reste ouverte** — décision éditoriale/légale, pas
tranchée unilatéralement cette session : le contenu verbatim existant n'a pas été
touché, seule la référence à son numéro de ligne l'a été.

**Priorité reçue** : N1+N2 avant le prochain lot de jobs parallèles — **fait** ; N4
ensuite — **lancé, résultat en attente** (arme mixte seulement, cf. N4 ci-dessus) ; N3
en même temps que la reprise des chiffres du rapport — **investigué, chiffre repondéré
fiable non obtenu cette session, cf. RESULTS_TESTS.md §87**. N5-N13 : **tous traités**
cette session (détail par item ci-dessus) — N5/N6/N7/N8/N9/N10/N12/N13 corrigés en
code/doc, N11 structurel (frein posé, pas de purge rétroactive), N13 partiel (décision
de fond sur le contenu verbatim laissée ouverte).

**Reste après cette session** : un rerun complet de `results_v26_validation_layer24_
v12_originals_filler_matched_n150_h100/` (extraction fraîche) serait nécessaire pour
compléter N4 (arme originaux+filler) — pas lancé, coût non trivial hors du cadre
"quelques minutes de GPU" de la demande initiale. Un rerun de
`b2_stratified_selection_rejudge.py` (corrigé, capture désormais `bin_info`
nativement) donnerait un chiffre repondéré fiable pour N3 — pas lancé non plus, même
raison. Mesurer l'impact réel de N7 (`MAX_LENGTH`) demande un job CPU/tokenisation dédié,
pas fait. Décision de fond N13 (réduction du contenu verbatim de
`PDF_APPENDICES_EXTRACT.md`) non tranchée. Archivage rétroactif N11 non fait.

## 9. Audit de branchement (J-14 avant remise du rapport) — décision de juge Qwen partout, verdict de lancement

**Décision utilisateur qui change la priorisation de cette section : juger
TOUT avec Qwen3.8-27B, plus jamais laisser un modèle extracteur juger ses
propres features (biais d'auto-préférence), même pour les sweeps taille/
layer où l'auto-jugement était jusqu'ici la méthodologie établie (§82).**
Conséquence directe : les 5 entrées déjà publiées du sweep §82/§88 (4B, 12B/
layer 31, 27B, layer 41, et la référence layer 24 de `results_v10_emails_main`)
sont TOUTES jugées par leur propre modèle extracteur (gemma-3-Nb-it
auto-jugeant) — aucune n'est encore comparable aux futurs chiffres Qwen. Ce
n'est pas un défaut de méthode découvert tardivement mais un changement de
politique explicite ; ces chiffres restent valides comme repères internes
(comparaisons relatives entre paliers, toutes sous le même biais) mais aucun
n'est la valeur finale à citer dans le rapport tant qu'il n'a pas été rejugé.
Le rejugement est bon marché : `scripts/judge_model_separation_test.py` est
déjà générique (`SAVE_DIR`/`ALT_JUDGE_MODEL_ID` en env), il relit
`p1_judge_labels_extended.json` déjà en cache et ne relance aucune
extraction — confirmé que 4 des 5 caches nécessaires existent déjà sur disque
malgré le nettoyage (`results_v27_.../layer31`, `results_v30_.../4b`,
`results_v31_.../27b`, `results_v33_.../layer41`, tous avec
`p1_judge_labels_extended.json` intact) ; seuls `results_v29_.../1b` et
`results_v32_.../layer12` manquent ce cache (ont planté sur le bug N1,
raison pour laquelle 45724/45725 sont en file — ceux-là seront directement
jugés Qwen dès leur premier passage, pas besoin de rejugement séparé une fois
terminés).

**Bug attrapé avant qu'il ne coûte du GPU** : ni
`run_ablation_classic_setup_k5_25m_model_scale_1b.slurm` (job 45724) ni
`run_ablation_classic_setup_k5_25m_layer12.slurm` (job 45725) ne fixent
`JUDGE_MODEL_ID` — les deux allaient déjà utiliser Qwen par défaut (cohérent
avec la décision ci-dessus, donc laissés tels quels après vérification, pas
de correctif nécessaire) ; un correctif inverse (les forcer en auto-jugement
pour matcher les anciennes lignes du tableau) a été écrit puis annulé une
fois la décision utilisateur connue — les deux scripts sont dans leur état
d'origine, aucune modification livrée.

**Deuxième bug attrapé en conditions réelles, celui-là un vrai crash** : job
45725 (layer12) a démarré, tourné 1 min 15, puis planté —
`saev5.py::run_llm_max_pool_pipeline` a levé un `RuntimeError` explicite
("SAVE_DIR probablement réutilisé avec une config différente") plutôt que de
continuer silencieusement : `results_v32_.../cache/` contenait encore des
symlinks orphelins de la tentative précédente (crash N1, avant le correctif),
pointant vers une clé de cache calculée par l'ANCIEN format
(`compute_activation_cache_key`, avant N8 -- `max_length`/`sigma_clip`/
`skip_first_content_token` ajoutés au payload) — la clé attendue sous le code
actuel diffère, et `saev5.py` refuse à raison de mélanger les deux plutôt que
de charger silencieusement le mauvais cache. Le garde-fou a fonctionné
exactement comme prévu (R5) ; seul le nettoyage n'avait pas suivi. `results_
v29_.../` (1B, job 45724, toujours PENDING au moment de la vérification)
avait exactement le même problème latent — nettoyé avant que le job ne
démarre, pas besoin de le relancer. Les deux caches orphelins (symlinks +
`p1_eval_raw_tokens.pt` régénérable) supprimés ; job 45725 resoumis
(45803). **À vérifier sur toute autre `SAVE_DIR` legacy réutilisée dans les
jours qui viennent** : n'importe quel `results_v*/cache/` créé avant le
correctif N8 et jamais entièrement nettoyé peut porter le même symlink
orphelin — le symptôme (`RuntimeError` en tout début de Pipeline 1, avant
toute extraction réelle) est sans ambiguïté et bon marché à corriger
(supprimer les symlinks du cache local, pas le cache partagé lui-même).

### 1. État "branché et fonctionnel", par module

- **HypothesisVerifier (App K.1, §84)** : run réel sain (40%/4 sur 10,
  IC95% [16,8% ; 68,7%]), mais n=10 est trop petit pour un chiffre de rapport
  isolé — citable uniquement comme illustration qualitative du protocole
  (verification_rate existe, fonctionne, donne un ordre de grandeur), pas
  comme une mesure de précision. Un run à n≥30-40 hypothèses (regénérer plus
  de diffs energy/sports depuis le CSV déjà en cache, coût marginal : juste
  plus de lignes du même CSV, pas de nouvelle extraction) resserrerait l'IC
  à moindre coût si le rapport a besoin d'un chiffre plus serré — sinon,
  citer avec l'IC affiché, honnête tel quel.
- **NPMI_verified (App E.1/E.3, §85)** : signal réel (les 5 paires
  survivent au relabellement indépendant) mais 4/5 reposent sur 1-2
  documents positifs sur 60 — `N_VERIFY_DOCS=60` est le goulot, pas le code.
  Doubler/tripler `N_VERIFY_DOCS` est un coût modéré (relit le même corpus
  déjà préparé, ajoute seulement des passes de juge) mais avec seulement 150
  features sur un domaine étroit, peu de nouvelles paires candidates
  apparaîtront probablement — rendement décroissant. **Citable tel quel avec
  la limite déjà écrite** (petit effectif, pas un défaut caché) plutôt que
  prioritaire pour un rerun, sauf si le rapport a besoin d'un chiffre NPMI
  individuel comme preuve forte (auquel cas augmenter `N_VERIFY_DOCS` d'abord).
- **Clustering LLM (App F.1, §86)** : confirmé sain, aucun doute résiduel —
  z-scores tous négatifs et non dégénérés, accuracy variable mais cohérente
  qualitativement avec le papier. Rien à refaire.
- **Diffing structuré (App D.2, job 45750)** et **Retrieval RRF/rerank/RBO
  (job 45745)** : toujours en file (PENDING) au moment de cet audit —
  **non vérifiables tant qu'ils n'ont pas tourné**, priorité de vérification
  dès qu'ils terminent (voir liste de jobs ci-dessous, ce sont des
  vérifications, pas des lancements).
- **App I (F1 lecteur 12B/27B)** : calcul prêt, aucune orchestration écrite,
  aucun run — nécessite une extraction fraîche par-document sur un domaine
  entier pour 12B ET 27B (les fragments légers gardés par le nettoyage
  disque, `results.json`+labels, ne suffisent pas : il faut l'activation de
  CHAQUE document du domaine, pas seulement les 150 déjà jugés). Coût estimé
  ~3h GPU/palier (calibré sur le temps réel du job layer 41, §88) soit ~6h
  GPU total pour les deux paliers — non trivial mais pas déraisonnable.
  **Priorité basse** : confirmatoire d'un effet déjà démontré par une autre
  métrique (odd-one-out, §82/§88), pas un front nouveau du rapport.
- **Juge découplé (`JUDGE_MODEL_ID`)** : vérifié dans le code — `saev5.py`
  appelle `load_judge_model(device=DEVICE)` sans `judge_model_id` explicite
  (`src/sae/saev5.py:1763,1824,2078`), donc hérite du défaut
  `src/config.py::JUDGE_MODEL_ID` = Qwen3.8-27B pour tout run frais du
  pipeline principal. Les scripts qui rechargent `MODEL_ID` comme juge
  (`b2_stratified_selection_rejudge.py`, `c2_original_only_rejudge.py`,
  `contrastive_labeling_test.py`, `explanation_plausibility_test.py`,
  `multilingual_judge_bias_test.py`, `judge_robustness_check.py`,
  `judge_sampling_ensemble_test.py`) le font tous intentionnellement, pour
  reproduire ou isoler une configuration historique précise — aucun oubli
  trouvé.
- **Cache verrouillé (N1/N2)** : toujours seulement testé unitairement
  (`tests/test_shared_cache_lock.py`), jamais par un vrai run concurrent sur
  la MÊME clé d'extraction. Les 5 jobs actuellement en file n'en fournissent
  pas un test naturel (paramètres d'extraction tous différents entre eux —
  aucune paire ne partage MODEL_ID+LAYER). Pas besoin d'un test synthétique
  dédié : la PROCHAINE course de partitions (a100+h100 pour un job
  identique, cf. mémoire session) est structurellement le test réel visé —
  à surveiller (logs `.extraction.lock`) la prochaine fois que ce pattern
  est utilisé, plutôt que de dépenser du GPU pour un test artificiel isolé.
- **N3 (Horvitz-Thompson)** : le calcul existe et fonctionne
  (`horvitz_thompson_mean`), mais n'a produit qu'une démonstration sur une
  population biaisée (68/150 reconstruits, §87) — pas un chiffre repondéré
  fiable pour le 89,3% publié. Sans objet une fois le rejugement Qwen fait
  (ci-dessous) : le futur rerun `b2_stratified_selection_rejudge.py` sous
  Qwen capturera `bin_info` nativement dès le départ, donnant directement un
  chiffre repondéré fiable ET jugé sous la politique actuelle — inutile de
  traiter N3 séparément avant.

### 2. Chiffres périmés

- **§80 (remboursement P@10 0,90 vs 0,30)** : label resserré (N5), job 45745
  en file — chiffre à mettre à jour dans `report/03_experiences_et_resultats.md`
  §5.5 dès que le job termine (la section actuelle dit encore "aucun résultat
  produit à ce jour", alors que `RESULTS_TESTS.md` §80 a déjà un tableau
  complet sous l'ANCIEN label — la note "[En cours]" du rapport est donc elle
  aussi déjà périmée dans l'autre sens : à corriger pour pointer vers §80,
  et remplacer par le résultat § label resserré une fois 45745 fini).
- **§79 (89,3%)** : reste le chiffre stratifié de référence dans le rapport,
  mais désormais **doublement daté** : jugé Gemma auto-référent (voir
  décision ci-dessus), ET non reconstructible a posteriori par bin (§87). Le
  job 45735 en file (rejugement Qwen de l'arme mixte stratifiée) produit
  directement le remplaçant — c'est la mesure la plus importante de toute
  cette campagne, à vérifier en priorité dès qu'elle termine.
- **§81 (B.1, −7,3 pts non significatif)** : déjà correctement cité dans
  `report/04_limites_et_perspectives.md:246-247` (82,0% vs 89,3%, p=0,070,
  sens inversé) — RAS sur la formulation. Job 45735 ne change pas cette
  conclusion (rejuge l'arme mixte, pas l'arme originaux+filler — l'arme
  originaux+filler reste non rejugeable sans extraction fraîche, fragments
  supprimés) mais **une fois 45735 fini, le bras "mixte" de cette comparaison
  aura changé de juge (Qwen) sans que le bras "originaux+filler" ait pu
  suivre** — la comparaison B.1 elle-même deviendra caduque (deux bras jugés
  par deux juges différents) jusqu'à ce qu'une extraction fraîche de l'arme
  originaux+filler soit faite. À noter explicitement en limite si le rapport
  cite encore B.1 après le rejugement Qwen de l'arme mixte.
- **§82 (sweep taille de modèle) et §88 (layer 41, nouvellement écrit)** :
  toute la table est actuellement Gemma auto-jugé (voir décision ci-dessus)
  — À REJUGER (4 jobs bon marché listés ci-dessous) avant citation finale
  dans le rapport. La ligne 1B et la section layer 12 (jobs 45724/45725,
  toujours en file) seront elles nativement Qwen dès leur premier passage —
  pas de rejugement à prévoir pour ces deux-là.
- **`results_v10_emails_main` jugé "par défaut"** : le `SAVE_DIR` contient
  maintenant DEUX caches judge coexistants — `p1_judge_labels_extended.json`
  (Gemma, magnitude, historique) et les fichiers
  `p1_judge_model_separation_*`/`b1_stratified_mixte_qwen_rejudge` (Qwen).
  Tout chiffre du rapport citant ce `SAVE_DIR` sans préciser lequel des deux
  est ambigu depuis l'introduction du juge Qwen — à vérifier ligne par ligne
  dans `report/03_experiences_et_resultats.md`/`04_limites_et_perspectives.md`
  avant la remise finale (pas fait dans cette passe, volume trop grand pour
  cette session — grep `results_v10_emails_main` dans `report/` et trancher
  Gemma/Qwen pour chaque occurrence est la prochaine étape mécanique).

### 3. Verdict et liste de jobs

**Pas encore prêt à lancer la campagne finale d'ablations pour le rapport —
mais proche : le principal front ouvert n'est plus "implémenter" (fait la
session précédente), c'est "rejuger sous la politique Qwen".** Priorité
stricte, du moins cher/plus urgent au plus cher/moins urgent, compte tenu de
J-14 :

| # | Job | Coût GPU | Nourrit | Priorité |
|---|---|---|---|---|
| 1 | Vérifier 45735/45745/45750 dès qu'ils terminent (§79 Qwen, §80 label resserré, App D.2) | 0 (déjà en file) | §79/§80/App D.2 | **Critique** — bloque le chiffre de référence du rapport |
| 2a | Rejuger Qwen `results_v33.../` (layer41) — `p1_all_doc_acts*.pt`/`p1_token_fragments` encore intacts sur disque (seul des 4 dans ce cas), script dédié déjà écrit (`slurm/analysis/run_judge_model_separation_qwen_layer41.slurm`), vrai rejugement seul, aucune extraction/entraînement | ~20-40 min GPU | §88 | **Haute** — vraiment bon marché |
| 2b | **Correction d'estimation** : `results_v27.../` (layer31/12B), `results_v30.../` (4B), `results_v31.../` (27B) ont eu leurs `p1_all_doc_acts*.pt`/`p1_extended_sae.pt`/`p1_token_fragments` supprimés par le nettoyage disque (ne restent que `results.json`+labels+plots) — **pas un simple rejugement possible**, il faut réentraîner l'extension SAE (le script standalone `judge_model_separation_test.py` échouerait, fichiers manquants). Resoumettre directement les `.slurm` ORIGINAUX (`run_ablation_classic_setup_k5_25m_layer31/model_scale_4b/model_scale_27b.slurm`, aucun ne pin `JUDGE_MODEL_ID` donc Qwen par défaut sans modification) — le cache RAW partagé (`local_data/activation_cache/`, 423 Go, 4 clés déjà présentes) devrait éviter de refaire l'extraction LLM, mais l'entraînement de l'extension (10 époques) et le merge/judge complet sont à refaire : coût proche d'un rerun complet moins l'extraction, pas quelques minutes | ~1-2h GPU/palier estimé (à vérifier au premier relancé — pas mesuré précisément), 3 soumissions | §82 (table sweep complète, cohérente Qwen) | **Moyenne** — plus cher que prévu initialement, à lancer après 2a/3 si le temps le permet |
| 3 | Vérifier 45724 (1B)/45725 (layer12) dès qu'ils terminent, écrire la ligne 1B et la section layer 12 sous Qwen nativement | 0 (déjà en file) | §82/§88 | **Haute** |
| 4 | Rejuger Qwen le magnitude/référence historique si le rapport cite encore 45,3% comme un chiffre à part (déjà fait en fait, §83 — vérifier juste que le rapport pointe vers §83 et pas vers l'ancien 45,3% Gemma sans le dire) | 0 (déjà fait) | passages "45,3%" du rapport | **Moyenne** — vérification de rédaction, pas un rerun |
| 5 | Grep `results_v10_emails_main` dans `report/*.md`, trancher Gemma/Qwen explicitement pour chaque occurrence | 0 (lecture) | tout le rapport | **Moyenne** |
| 6 | `b2_stratified_selection_rejudge.py` rerun sous Qwen (`bin_info` natif) pour remplacer N3/§87 par un chiffre repondéré fiable ET jugé Qwen | faible (rejugement seul, réutilise cache) | §79 remplaçant définitif, N3 | **Moyenne** — utile mais #1 (job 45735) donne déjà un chiffre Qwen exploitable sans repondération |
| 7 | App I (F1 lecteur 12B/27B) — extraction fraîche + calcul | ~6h GPU | App I (jamais commencé) | **Basse** — confirmatoire, pas un front nouveau, à ne lancer que si le temps le permet après 1-6 |
| 8 | Extraction fraîche de l'arme originaux+filler (`results_v26_.../`) pour compléter N4/B.1 sous Qwen | plusieurs heures GPU (extraction complète) | §81/N4 | **Basse** — B.1 est déjà tranché négativement (§81), un rejugement Qwen ne changerait probablement pas la conclusion "pas d'effet démontrable" |

**Recommandation concrète pour la suite immédiate** : lancer 2a (rejugement
layer41, vraiment bon marché) maintenant. Décider 2b (relance complète
layer31/4B/27B, ~1-2h GPU chacun d'après une première estimation à vérifier)
une fois 1/3 confirmés, pour ne pas saturer la file inutilement en même temps
que les 5 jobs déjà en cours. Ne pas lancer #7/#8 avant d'avoir confirmé que
le rapport en a réellement besoin (aucune mention d'App I dans `report/*.md`
à ce jour).
