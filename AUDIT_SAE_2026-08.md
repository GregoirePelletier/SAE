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
attaquable : **fallback layer=24 silencieux (tracké dans `docs/evaluation_protocol.md`
→ Points ouverts, à vérifier en quelques minutes de log)**, puis B2 (métrique
d'interprétabilité biaisée par sélection), B1 (corpus d'entraînement 93% généré par le
modèle juge), puis le bug OOM bloquant de Latent Terms (§1). B7 (jointure de split) et B6
(label `remboursement`) sont corrigés.

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

- **Corpus d'entraînement dominé par du texte généré par le modèle juge lui-même.**
  `MAX_AUGMENTED_PER_MAIL=13` (défaut, `saev5.py`) sur 13 niveaux de perturbation ; le
  README documente déjà la proportion (~41k documents augmentés contre ~2,2k mails réels
  train/test) mais aucun test d'indépendance de la **distribution d'entraînement** n'existe
  — seule l'indépendance du juge a été testée (§48/§50/§52). Le SAE d'extension **et**
  `PhraseLevelSAE` apprennent donc majoritairement le style de réécriture de Gemma, le
  modèle qui juge ensuite les features qu'il a lui-même généré la matière d'entraînement.
  Test minimal : réentraîner l'extension sur les mails originaux seuls, comparer taux
  d'interprétabilité et top-features de diffing.
- **La métrique d'interprétabilité phare (45,3%, 68/150) reste mesurée par défaut sur un
  échantillon biaisé par construction — sélection stratifiée ajoutée, pas encore par
  défaut.** `feature_selection_by_magnitude` sélectionne les *N* features par magnitude
  token-level moyenne — donc les plus denses, les plus proches de directions
  génériques/stop-word. Non comparable à un chiffre publié (Bills et al. échantillonnent au
  hasard, EleutherAI/Paulo stratifient, interp-embed rapporte par bins de fréquence
  log-espacés, App. J) ni comparable entre les configurations du dépôt dès que la
  distribution de magnitude change. `feature_selection_stratified_by_frequency`
  (`judge.py`) existe désormais, opt-in via `FEATURE_SELECTION_METHOD=stratified` —
  gardé opt-in plutôt que basculé par défaut faute d'un run de comparaison qui en valide
  l'effet (même discipline que `BATCH_SIZE_EXTRA`). **Reste à faire : lancer ce run et, si
  concluant, en faire le défaut.**
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
biaisée par sélection (§5, B2) ; corpus d'entraînement à 93% généré par le juge (§5, B1).
Les deux derniers se mesurent en quelques heures chacun, aucun ne demande un run de 20h, et
B2 se traite en rétro-analyse sur des caches déjà existants.

Corrigés depuis : B7 (jointure de split par hash SHA1 du texte parent plutôt que par
position — migration de `augmented_mails.jsonl` existant vers `parent_sha1` encore à faire,
§5) ; B6 (label `remboursement`, le résultat concerné — sonde à 0,846 vs 0,855 de majorité —
reste à re-mesurer avec le label corrigé avant d'être cité) ; B3/B5 (protocole odd-one-out) ;
B4 (dédup des positifs) ; B11 (matching lexical du corpus diffing).
