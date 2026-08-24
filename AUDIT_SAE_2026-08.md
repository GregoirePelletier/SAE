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
chiffres dans `RESULTS_TESTS.md` §77-§82). Priorité actuelle : la partie "fidélité aux
papiers" (§1) reste très majoritairement à faire (HypothesisVerifier, NPMI_verified,
clustering LLM, retrieval réel, App I) — voir §7 pour l'état précis et les jobs GPU en
cours au moment de la dernière mise à jour de ce document.

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
  accuracy par cluster (réassignation LLM), z-score de conductance en espace dense. Score
  de silhouette : gardé (continuité avec `RESULTS_TESTS.md`), divergence avec la note 4 du
  papier documentée explicitement dans `docs/references.md` plutôt que silencieuse.
- **interp-embed, Retrieval** : l'étape 1 de la Figure 10 (normaliser chaque latent par le
  90ᵉ percentile de ses activations non nulles) est corrigée
  (`src/analysis/metrics.py::normalize_by_p90_and_score`). Métriques App. G implémentées
  et testées (`src/analysis/metrics.py::average_precision/precision_at_k/
  mean_average_precision/mean_precision_at_k/reciprocal_rank_fusion/rank_biased_overlap`,
  `tests/test_retrieval_metrics.py`, écarts documentés `docs/references.md`). Manque
  encore l'intégration : le rerank LLM des latents (prompt App. G disponible,
  `docs/PDF_APPENDICES_EXTRACT.md`), et le script d'évaluation qui les appelle sur de
  vraies requêtes/documents — aucune évaluation retrieval chiffrée n'existe à ce jour,
  seuls les blocs de calcul sont prêts.
- **App. I (taille du modèle lecteur, 12B vs 27B)** : protocole F1 latent-vs-juge absent.
  Tant qu'il n'est pas écrit, la comparaison 12B/27B ne peut être arbitrée que par le taux
  odd-one-out, instable à 31% au niveau d'une feature (`CLAUDE.md`, §13.1).
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
  Precision@10/@20 contre TF-IDF sur seulement 4 requêtes paraphrasées (une par intention,
  pas de réplication), avec un label de pertinence toujours basé sur le même filtre regex
  faible (`INTENT_KEYWORDS_FR`, cf. B6 ci-dessous) — à garder étiqueté comme évaluation
  indicative, pas benchmark IR (pas de MAP/nDCG/BEIR — métriques désormais disponibles,
  `src/analysis/metrics.py`, mais pas encore branchées sur ce protocole).
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
checkpoint bf16 complet (52 Go) supprimé — mais jamais exécuté en conditions réelles
depuis le changement de checkpoint, à vérifier avant de citer un résultat produit avec.
