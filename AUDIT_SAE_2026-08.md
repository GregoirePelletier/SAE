# Audit `GregoirePelletier/SAE` — items ouverts

Fusion de trois passes (audit interne du dépôt, et deux audits indépendants Opus 5
effort élevé, l'un centré fidélité/perf, l'autre opérationnel/data-science/hygiène),
recoupée avec l'état actuel du code. Chaque item ci-dessous a été vérifié individuellement
contre le dépôt au moment de la fusion — un item confirmé résolu (getcol déjà en CSC,
`.gitmodules` déjà déclaré, `test_massive_acts.py` déjà déplacé, item 8/G6/G3-async déjà
vérifiés GPU, etc.) a été retiré plutôt que reconduit. Le détail des correctifs déjà
appliqués et de leur vérification (jobs GPU, tests d'équivalence) vit dans `git log`, pas
ici — ce document ne porte que ce qui reste à traiter.

Priorité, si le temps manque, par ordre décroissant de ce qui rend un résultat déjà publié
attaquable : **§5 (fallback layer=24 silencieux, tracké dans `docs/evaluation_protocol.md`
→ Points ouverts, à vérifier en quelques minutes de log)**, puis B2 (métrique
d'interprétabilité biaisée par sélection), B1 (corpus d'entraînement 93% généré par le
modèle juge), B7 (leakage de split indétectable), B6 (label faible invalidant un résultat
négatif déjà publié), puis le bug OOM bloquant de Latent Terms (§1).

---

## 1. Fidélité aux papiers

- **interp-embed, Diffing** : manquent (a) relabellisation des top-200 latents avant
  génération d'hypothèses (App. D.2, seuil 0,03), (b) `HypothesisVerifier` + taux de
  vérification (App. K.1, *la* métrique de la Figure 11 — sans elle aucun chiffre de
  diffing n'est comparable au papier), (c) `diff_features_multi`, (d)
  `limit_feature_differences`. `generate_llm_diff_hypothesis` produit une hypothèse texte
  libre là où App. D.2 impose un JSON structuré à ≤10 hypothèses avec
  `percentage_difference`/`confidence` — non comparable en l'état.
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
  accuracy par cluster (réassignation LLM), z-score de conductance en espace dense. Le
  score de silhouette est toujours calculé et publié à deux endroits (`compute_silhouette`
  dans `results_p1`/`results_p2`, `saev5.py`) alors que la note 4 du papier le rejette
  explicitement comme mesure géométrique non pertinente pour l'analyse exploratoire.
- **interp-embed, Retrieval** : manque l'étape 1 de la Figure 10 — *normaliser chaque
  latent par le 90ᵉ percentile de ses activations non nulles*. Sans elle, les magnitudes
  JumpReLU non bornées du core GemmaScope (outliers ~1e5) écrasent la pondération de rang :
  le score est dominé par l'échelle des latents, pas par leur pertinence — bug de qualité
  autant qu'écart de fidélité, correctif de l'ordre de quelques lignes. Manquent aussi le
  rerank LLM des latents (App. G), les métriques MAP/MP@50/MP@10, l'agrégation RRF, RBO —
  aucune évaluation retrieval chiffrée n'existe.
- **App. I (taille du modèle lecteur, 12B vs 27B)** : protocole F1 latent-vs-juge absent.
  Tant qu'il n'est pas écrit, la comparaison 12B/27B ne peut être arbitrée que par le taux
  odd-one-out, instable à 31% au niveau d'une feature (`CLAUDE.md`, §13.1).
- **Matryoshka SAE** : non implémenté malgré son affichage comme contribution originale
  (« hybride frozen-core Matryoshka SAE » — `MATRYOSHKA_DIM` n'est que la troncature MRL de
  l'embedding F2LLM, sans rapport avec les dictionnaires imbriqués/pertes préfixes du
  papier). Risque en soutenance et en review tant que non tranché (implémenter ou retirer
  la revendication).
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
- **Latent Terms — bloquant, aucun résultat produit à ce jour.** La version fidèle au
  papier (token-level, hors-domaine) meurt OOM avant même de finir de constituer son pool
  d'entraînement, deux fois de suite malgré une première correction (streaming). Cause
  probable non corrigée : `build_token_training_pool` accumule chaque tenseur dans une
  liste Python `pool` (~59 Go pour ~33M tokens) puis fait `torch.cat(pool, ...)`, qui
  alloue un second tenseur ~59 Go **pendant que `pool` reste référencée** — pic RSS ~2× le
  pool, cohérent avec les OOM observés à 96G et 110G. Correctif : préallouer
  `torch.empty(target_tokens, d_in, dtype=bfloat16)` et écrire chaque lot via un curseur,
  sans liste intermédiaire. Tant que non corrigé, les seuls résultats Latent Terms
  existants restent l'ancienne méthode phrase-level in-domain (`RESULTS_TESTS.md`
  §26/§68(c)/§69(c), déjà marqués supersédés) — à ne pas citer comme résultat de
  référence.
- **Latent Terms — évaluation limitée** : `latent_retrieval_precision_eval.py` calcule
  Precision@10/@20 contre TF-IDF sur seulement 4 requêtes paraphrasées, avec un label de
  pertinence toujours basé sur le même filtre regex faible (`INTENT_KEYWORDS_FR`, cf. B6
  ci-dessous) — à garder étiqueté comme évaluation indicative, pas benchmark IR (pas de
  MAP/nDCG/BEIR).
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
- **Sharding des fragments (extraction)** : testé CPU, vérification GPU en cours (job 44778,
  lancé avant cette passe) — à confirmer une fois le job terminé avant de considérer l'item clos.
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

- **Nettoyage (~55 fichiers) non entamé** : 16 scripts d'audit forensiques à conclusion
  figée dans `RESULTS_TESTS.md` (`scripts/audit_2026_08_*.py`, sauf les deux à promouvoir en
  outillage/test permanent : `extraction_batch_size_sweep.py` → `benchmarks/`,
  `bf16_fp32_diagnostic.py` → `tests/test_dtype_overflow.py`) ; 8 `docs/audit_*_results.json`
  à sortir de `docs/` (référence technique, pas dossier de résultats) ;
  `docs/PDF_APPENDICES_EXTRACT.md` (reproduction verbatim d'appendices sous copyright, à
  réduire à une table de correspondance §/App. → décision → fichier) ;
  `compare_to_frozen_benchmark.py` orphelin (0 référence, cf. §1) ; `test_chargement_sae.py`
  à la racine (à déplacer vers `tests/` ou `slurm/validation/`) ; `src/sae.egg-info/` versionné
  par erreur ; `CHANGELOG.md` redondant avec `git log`/`RESULTS_TESTS.md` ; `logs/README.md`
  à fusionner dans `docs/ops.md` ; 104 `.slurm` (94 au dernier comptage, la dérive continue)
  à réduire à un template générique + fichiers `.env` — copies d'un même script différant
  par une variable.
- **3 désynchronisations commentaire↔code restantes** (une 4e, `config.py` annonçant
  layer=24 par défaut au lieu de 31, corrigée avec le test qui en dépendait) :
  1. `saev5.py` (« fp16 par défaut en local ») : le défaut réel de `DTYPE` (`config.py`) est
     `bf16` — fp16 n'intervient que si `DTYPE` est explicitement mis à autre chose qu'`bf16`
     (ex. GPU Turing local sans bf16 natif). Le commentaire présente l'exception comme le
     défaut.
  2. `saev5.py` (« σ-clip intra-batch (stats sur B docs) ») contredit frontalement
     `activations.py::norm_outlier_mask`, explicitement **intra-document** (sa docstring
     justifie ce choix contre l'intra-batch).
  3. `fragment_store.py` docstring (« `vals`: float16 ») : le code écrit `torch.float32` —
     double du stockage annoncé.
- **`pytest tests/ -q` ne passe qu'à un test près.** `scripts/check_docs.py` sort en 1 avec
  17 violations (première personne du singulier), 16 venant de `docs/INTERP_EMBED_COVERAGE.md`
  et `docs/PDF_APPENDICES_EXTRACT.md`. Les 7 autres échecs préexistants (mocks de test
  périmés — `.eval()` absent d'un stub, forme fixe d'un `MagicMock` incompatible avec un
  batch réel, comparaison train/validation mélangée, assertion sur l'ancien défaut
  layer=24, référence CSR/CSC bugguée dans son propre calcul de référence — cf. commit qui
  suit) sont corrigés ; il ne reste que `check_docs.py`, un travail de réécriture de prose,
  pas de code.
- **Garde-fou placeholder incomplet.** `PLACEHOLDER_RE = r"\[à compléter\]"`
  (`scripts/check_docs.py`) ne matche ni `§<N-À-COMPLÉTER>` ni `<!-- À COMPLÉTER -->`. Au
  moins 4 occurrences non résolues passent au travers, dont
  `report/03_experiences_et_resultats.md` (le rapport cite une section de
  `RESULTS_TESTS.md` qui n'existe pas) et `docs/architecture.md`.
- **Une reprise n'est pas bit-reproductible.** Le flux RNG diffère entre un run continu et
  un run repris (checkpoint), donc le réservoir résiduel d'un run repris ≠ celui d'un run
  continu. Scientifiquement bénin (échantillon aléatoire dans les deux cas) mais rien ne le
  documente — la traçabilité `SEED` (`CLAUDE.md`) laisse croire à une reproductibilité qui
  n'existe pas dès qu'une reprise a eu lieu. À écrire en commentaire/doc, pas à corriger.
- **Clé de cache P2 toujours sans `EMB_MODEL`** (`train_phrase_emb_dim{...}_n{...}`,
  `saev5.py`) : violation de la règle `CLAUDE.md` sur les clés de cache, à l'endroit exact
  que la règle signale déjà comme piège. Basculer de backbone sur le même corpus recharge
  silencieusement les embeddings du mauvais modèle.
- **`CLAUDE.md` — contestations non traitées** : (b) la formulation absolue « aucun code
  Python sur le frontal, jamais » interdit même une validation de configuration bornée
  (<5s CPU, <500Mo RSS), poussant à découvrir des fautes de config après 20h de file
  d'attente ; (c) « bf16 partout, y compris en local » est appliquée aveuglément à
  `PhraseLevelSAE` (P2) dont les embeddings sont L2-normalisés et bornés à 1,0 — bf16 y
  coûte 8 bits de mantisse pour rien ; (d) la règle « pas de numéro de version interne »
  ne couvre que la prose `.md`, pas le nommage des artefacts (`results_v14_main/`,
  `run_sae_v12_scaled.slurm`).
- **R1-R6 proposées, aucune écrite dans `CLAUDE.md`** : R1 (reprise obligatoire >1h GPU)
  est implémentée dans le code mais n'existe pas comme règle écrite, donc rien ne protège
  contre une régression sur une future boucle longue qui l'omettrait. R2 (commentaires
  suivent la règle éditoriale des `.md`, pas de récit de session) R3 (budget de perf tracé
  par run) R4 (plafond explicite sur toute structure O(n²) en n_docs/d_sae) R5 (clé de
  cache dérivée mécaniquement, pas rédigée à la main) R6 (écart au papier = entrée
  documentée dans `docs/references.md`) restent à écrire.

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

- **Corpus d'entraînement dominé par du texte généré par le modèle juge lui-même.**
  `MAX_AUGMENTED_PER_MAIL=13` (défaut, `saev5.py`) sur 13 niveaux de perturbation ; le
  README documente déjà la proportion (~41k documents augmentés contre ~2,2k mails réels
  train/test) mais aucun test d'indépendance de la **distribution d'entraînement** n'existe
  — seule l'indépendance du juge a été testée (§48/§50/§52). Le SAE d'extension **et**
  `PhraseLevelSAE` apprennent donc majoritairement le style de réécriture de Gemma, le
  modèle qui juge ensuite les features qu'il a lui-même généré la matière d'entraînement.
  Test minimal : réentraîner l'extension sur les mails originaux seuls, comparer taux
  d'interprétabilité et top-features de diffing.
- **La métrique d'interprétabilité phare (45,3%, 68/150) est mesurée sur un échantillon
  biaisé par construction.** `feature_selection_by_magnitude` sélectionne les *N* features
  par magnitude token-level moyenne — donc les plus denses, les plus proches de directions
  génériques/stop-word. Non comparable à un chiffre publié (Bills et al. échantillonnent au
  hasard, EleutherAI/Paulo stratifient, interp-embed rapporte par bins de fréquence
  log-espacés, App. J) ni comparable entre les configurations du dépôt lui-même dès que la
  distribution de magnitude change (K_EXTRA, largeur, couche, core vs extension) — ce qui
  touche presque toutes les ablations publiées. Ajouter une sélection stratifiée par
  fréquence comme mode par défaut ; garder la magnitude comme variante étiquetée.
- **Le négatif de l'odd-one-out est structurellement différent des positifs.** Positifs :
  contexte autour de l'argmax de la feature. Négatif : contexte autour de
  `mid = len(toks)//2` (`judge.py`), une position arbitraire. Le juge peut trancher sur un
  artefact de position/saillance plutôt que sur la sémantique — explication mécanique
  plausible des 31% d'instabilité (§13.1), jamais formulée. Correctif : négatif à l'argmax
  de la **même** feature sur un document non-activant.
- **Déduplication des positifs pénalise systématiquement les features lexicales.**
  `seen_target_words` force des mots-cibles différents ; une feature authentiquement
  lexicale (Latent Terms mesure ~33% de features purement lexicales) ne peut jamais être
  présentée sous sa forme la plus convaincante (même mot, contextes variés). Correctif :
  dédupliquer sur `(doc_idx, word_span)`, pas sur la chaîne du mot.
- **`neg_quantile=0.05` ne garantit pas un négatif inactif** pour une feature dense
  (active dans >95% des documents, 5ᵉ percentile strictement positif). `interp_embed`
  impose une activation strictement nulle. Ajouter un `assert` ou écarter la feature.
- **Un label faible d'intention est du bruit pur.**
  `INTENT_KEYWORDS_FR["remboursement"] = r"\b(rembours\w*|trop[- ]perçu|avoir\w*)\b"`
  (`dataset.py`) : `avoir\w*` matche le verbe « avoir », l'un des mots les plus fréquents du
  français. Cohérent avec le résultat déjà publié (sonde remboursement 0,846 vs 0,855 de
  majorité, Δ = −0,85pt) : le label est ininterprétable, pas la feature — un résultat
  négatif déjà publié est probablement invalidé par un bug de regex, pas par la méthode.
  Problème de moindre gravité pour `information\w*` et `coupure\w*` sous « urgence ».
- **Le split group-aware repose sur une jointure positionnelle fragile.**
  `build_email_train_test_corpus` associe `parent_idx` à l'index positionnel dans
  `real_texts`, qui sort de `load_and_clean_emails` **après filtrage**
  (`min_chars`, `drop_duplicates`, `clean_text` non vide). Si ce filtrage diffère, même
  légèrement, entre le run d'augmentation et le run d'entraînement, chaque variante est
  rattachée au mauvais mail parent et le split par groupe fuit sans signal — le garde-fou
  actuel (`parent_idx < n_real`) ne détecte qu'un dépassement de borne, pas un décalage.
  Toute la revendication d'absence de leakage (métriques de classification) en dépend.
  Correctif : joindre sur un hash SHA1 du texte parent, stocké au moment de la génération.
- **Taux de rejet de `validate()` corrélé à l'axe de perturbation** (les axes qui
  perturbent le plus la forme perdent plus souvent les entités numériques, sauf
  `orthographe` exempté de vérification factuelle) — le pool accepté est déséquilibré par
  axe, confondu avec les accuracies par axe publiées. Aucun audit ne rapporte le taux
  d'acceptation par axe (`groupby` sur le manifest parquet déjà écrit).
- **Biais de longueur à la construction du corpus** : prompt d'augmentation tronqué à
  2048 tokens, `validate()` compare aux faits du parent complet — les mails longs perdent
  des faits par construction et sont sous-représentés parmi les variantes, à mettre en
  regard de ρ(longueur, n_features) = 0,906 déjà mesuré (§59).
- **Chunking par 1024 caractères** (pas tokens/mots) pour le corpus générique et le filler —
  coupe au milieu des mots/phrases, distribution de tokens qui n'existe dans aucun usage
  réel, pour le réservoir qui domine le volume d'entraînement du SAE résiduel.
- **Deux standards de matching lexical incompatibles dans le dépôt.** `dataset.py` documente
  soigneusement `\b...\w*` ; `preparation.py::keyword_match` fait un `in` sur sous-chaîne
  sans frontière (« vol » matche volume/volley/évolution, « watt » matche Watteau) — le
  corpus energy/sports/support, vérité terrain du diffing cross-domaine, est bruité par
  construction.
- **L'augmentation n'est pas reproductible malgré son champ `seed`** : `do_sample=True`,
  génération batchée avec reprise → la composition des lots après reprise change le flux
  RNG. Le champ `seed` du JSONL promet une reproductibilité qui n'existe pas.
- **Aucune évaluation aval sur une tâche métier réelle** (routage, priorisation, détection
  de réclamation) avec baseline honnête TF-IDF+LogReg — la seule sonde existante est bâtie
  sur les labels faibles ci-dessus. Pour l'objectif affiché (outil transparent pour
  l'industrie), c'est ce qu'un jury demandera en premier.

## 6. Tenue du dépôt

- **README : contraste 20% (2/10) vs 45,3% (68/150) présenté sans réserve statistique.**
  L'IC de Wilson de 2/10 est ≈[5,7%; 51%] — le contraste n'est pas significatif.
  `stats.py::two_proportion_test` existe et n'est pas appliqué à ce chiffre d'affiche.
  Soit refaire la mesure hors-domaine à n≥150, soit afficher l'IC.
- **Aucune `LICENSE`, aucune mention de régime des données** dans un dépôt qui traite des
  mails clients EDF et embarque une reproduction verbatim sous copyright
  (`docs/PDF_APPENDICES_EXTRACT.md`, cf. §3) — à traiter avant toute publication.
- **Aucun point d'entrée unique reliant les chiffres du README/rapport/`RESULTS_TESTS.md`
  à la config exacte qui les a produits.** Un `docs/reference_run.md` d'une page
  remplacerait une part des ~60 lignes de commentaires-journal éparpillées dans les
  `.slurm`.
- **Duplication fonctionnelle** : `_strip_leading_objet_line` (`augmentation.py`) et le
  nettoyage « Objet : » de `load_and_clean_emails` (`preparation.py`) sont deux copies de
  la même regex, couplées par un commentaire de cohérence plutôt qu'un module commun —
  fragile, la prochaine modification n'en touchera qu'une.
- **`pyproject.toml`** : `authors = [{name = "Research Team"}]`, description en anglais
  dans un dépôt francophone, `version = "0.2.0"` sans référentiel — première chose que lit
  un relecteur externe.

## 7. Priorisation restante

Ce qui rend un résultat déjà publié attaquable en soutenance, par ordre décroissant :
fallback layer=24 silencieux (tracké `docs/evaluation_protocol.md` → Points ouverts, à
vérifier sur les logs du run §51 en quelques minutes) ; métrique d'interprétabilité
biaisée par sélection (§5, B2) ; corpus d'entraînement à 93% généré par le juge (§5, B1) ;
leakage de split indétectable (§5, B7) ; label faux invalidant un résultat négatif déjà
publié (§5, B6). Les quatre derniers se corrigent ou se mesurent en quelques heures chacun,
aucun ne demande un run de 20h, et deux (métrique biaisée, label faux) se traitent en
rétro-analyse sur des caches déjà existants.
