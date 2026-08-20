# Audit `GregoirePelletier/SAE` — fidélité aux papiers, hygiène du dépôt, performances

Périmètre à l'origine : clone complet (242 fichiers, 5 702 lignes de code `src/`), lecture ligne à
ligne de `src/sae/saev5.py`, `frozen_core.py`, `phrase_sae.py`, `batch.py`, `judge.py`,
`sae_shared.py`, `storage/fragment_store.py`, `analysis/{activations,metrics,cooccurrence}.py`,
`retrieval/latent_terms.py`, `src/config.py`, les 63 `.slurm`, les 8 JSON d'audit de `docs/`. Ces
comptes datent du clone initial ; le dépôt a bougé depuis (248 fichiers, 6 771 lignes dans `src/`,
94 `.slurm` — voir §1.4 et §3.1 pour ce qui a changé concrètement).
Confronté à : `InterpretableSAE_Embeddings.pdf` (arXiv:2512.10092), `teacholdsaes.pdf`
(SAE Boost, COLM 2025), `LatentTerms.pdf` (arXiv:2605.29384), `BatchTopK.pdf`, `Matryoshka.pdf`.

---

## 1. Fidélité aux papiers

### 1.1 Note globale

| Papier | Note | Nature de l'écart |
|---|---|---|
| interp-embed (Jiang & Sun 2025) | **10/20** | Mécanismes centraux fidèles, **couche de vérification/métriques absente** |
| SAE Boost (Koriagin 2025) | **17/20** | Écart d'équation d'encodage corrigé (§1.3), **pas encore vérifié par un run complet** ; budget de tokens au seuil de risque du papier |
| BatchTopK (Bussmann 2024) | **17/20** | Fidèle |
| Latent Terms (Clavié 2026) | **13/20** | Réimplémentation token-level désormais fidèle, mais n'a **jamais produit un seul chiffre** (§1.4) |
| Matryoshka SAE (Bussmann 2025) | **0/20** | **Non implémenté** malgré son affichage comme contribution |
| **Moyenne pondérée** | **≈ 11/20** | |

Verdict : le dépôt implémente correctement *ce qui produit des chiffres*, et pas *ce qui rend les
chiffres falsifiables*. C'est le pattern dominant et il est systématique, pas anecdotique.

### 1.2 interp-embed — tâche par tâche

**Tâche 1 — Diffing (§4.1, App. D) : 11/20**
- Fidèle : embedding = max-pool des activations SAE sur tokens ✓ ; diff sur binarisation par
  présence dans le document ✓.
- `corpus_diff_stats` (Fisher + BH) est **statistiquement supérieur** au delta de fréquence brut du
  papier — écart assumé, à conserver.
- Manquant et non substituable : (a) **relabellisation des top-200 latents avant génération
  d'hypothèses** (App. D.2, seuil 0,03) — sans elle les hypothèses reposent sur des labels
  Neuronpedia génériques ; (b) **`HypothesisVerifier` (App. K.1)** et le **taux de vérification
  >1 %** — c'est *la* métrique de la Figure 11, sans laquelle aucun chiffre de diffing du dépôt
  n'est comparable au papier ; (c) `diff_features_multi` (cible vs max de K corpus) ; (d)
  `limit_feature_differences`. `docs/INTERP_EMBED_COVERAGE.md` identifie correctement (a)–(d)
  comme des **adaptateurs** (code déjà écrit chez les auteurs) — c'est donc du travail chiffré à
  quelques centaines de lignes, pas de la recherche.
- `generate_llm_diff_hypothesis` produit **une hypothèse texte libre** là où App. D.2 impose un
  JSON structuré à ≤10 hypothèses avec `percentage_difference` et `confidence`. Non comparable.

**Tâche 2 — Corrélations (§4.2, App. E) : 13/20**
- Le plus fidèle des quatre. `compute_npmi` ✓, `find_interesting_pairs` avec NPMI>0,6 et
  sim<0,2 ✓ (seuils exacts du papier, CivilComments).
- Manquant : (a) **filtre LLM des latents syntaxiques** (E.1, explicite) ; (b) **filtre des paires
  triviales co-activées sur le même token** (E.1) ; (c) **NPMI_verified** — le papier reclasse
  indépendamment *i* et *j* par juge LLM puis recalcule le NPMI ; c'est toute la Figure 4 et donc
  toute la revendication de précision ; (d) métrique CO = max(P(i|j), P(j|i)).
- Sans (c), la sortie `p1_interesting_correlations.json` est une liste de candidats, pas un
  résultat. Le formuler ainsi dans le rapport.

**Tâche 3 — Clustering (§4.3, App. F) : 8/20 — le maillon le plus faible**
- Le papier : binarisation → **affinité de Jaccard** → `SpectralClustering(affinity="precomputed")`.
  Le dépôt : TF-IDF + UMAP + HDBSCAN (`cluster_in_feature_space`) pour le clustering global, et
  `SpectralClustering(affinity="cosine")` sur binaire pour le clustering ciblé
  (`saev5.py:432`). **Cosine sur binaire ≠ Jaccard** (le premier normalise par √(|A||B|), le
  second par |A∪B|) : ce n'est pas une variante d'implémentation, c'est une autre métrique.
- Manquant : génération LLM de mots-clés puis **union des top-k=100 latents par mot-clé** (App. F.1) —
  le dépôt prend un `axis_query` unique en dur ; **accuracy par cluster** (réassignation LLM à
  partir des seules descriptions) ; **z-score de conductance** en espace dense.
- **Contradiction directe avec le papier** : `saev5.py:1285` calcule et publie un score de
  silhouette, que la note 4 du papier rejette explicitement comme mesure géométrique non
  pertinente pour l'analyse exploratoire. Le retirer des `results` ou l'accompagner d'une
  justification qui assume la divergence.

**Tâche 4 — Retrieval (§4.4, App. A/G) : 9/20**
- Fidèle : `w_i = exp(−(r_i/k)/T)` verbatim, T=0,2 (l'optimum du papier), top-100 latents
  candidats par similarité d'embedding label↔requête.
- **Manquant, et ce n'est pas cosmétique** : l'étape 1 de la Figure 10 —
  *« Normalize each latent by 90th percentile of non-zero activations across dataset »*. Sans elle,
  les magnitudes JumpReLU non bornées du core GemmaScope (outliers ~1e5, documentés dans le dépôt
  lui-même) écrasent la pondération de rang : le score est dominé par l'échelle des latents, pas
  par leur rang de pertinence. C'est un bug de qualité autant qu'un écart de fidélité, corrigeable
  en 4 lignes.
- Manquant aussi : étape 3 (**rerank LLM des latents**, App. G) ; métriques **MAP / MP@50 /
  MP@10** ; agrégation RRF ; RBO. Aucune évaluation retrieval chiffrée n'existe donc.

**App. I (taille du modèle lecteur) : 0/20.** Le protocole F1 latent-vs-juge est le critère retenu
pour trancher 12B vs 27B. Il n'existe nulle part. Tant qu'il n'est pas écrit, la comparaison
12B/27B ne peut pas être arbitrée autrement que par le taux odd-one-out, dont §13.1 établit qu'il
est instable à 31 % au niveau d'une feature.

**Labellisation.** Le dépôt utilise un gate odd-one-out FR ; le papier utilise le prompt pairé
pos/neg d'App. C. La divergence est documentée et empiriquement motivée (§15.4). Conséquence à
assumer dans le rapport : les taux d'interprétabilité du dépôt **ne sont pas comparables** aux
scores detection/fuzzing du papier ni à ceux de SAE Boost (Table 10).

**Ajout depuis ce clone, en dehors du périmètre initial : `scripts/imdb_genre_diffing_test.py`.**
Réplication ciblée de l'éval ground-truth "movie genre differences" (§4.1 + App. D.3, Table 1/5) —
appelle directement le code des auteurs (`interp_embed.Dataset`, `GoodfireSAE`, `diff_features`),
juge = prompt verbatim App. D.3 avec mapping yes/related/no→1/0,5/0. Écarts documentés en tête de
script (Llama-3.1-8B/SAE-l19 au lieu de Llama-3.3-70B/layer 50 du papier ; dataset HF de genre
reconstitué de bonne foi, la citation du papier pointant en réalité vers un dataset de sentiment).
Ne comble aucun des manques listés ci-dessus (c'est une validation de la fidélité d'`interp_embed`
lui-même, pas du pipeline Gemma-3/mails du projet), mais suit exactement la discipline que R6
(§4.3) demande : écarts déclarés avec équation/justification, pas des choix implicites.
Sous-produit de ce travail : `docs/INTERP_EMBED_COVERAGE.md` documente que `sae_shared.py:28-32`
importe `get_reconstruction_error` (absent d'`interp_embed`) dans le même bloc `try` que
`from interp_embed import Dataset as InterpDataset` — l'`ImportError` du premier est capturée par
l'`except` avant que le second ne s'exécute, donc `InterpDataset` reste `None` en permanence même
maintenant que le submodule est peuplé. Sans conséquence aujourd'hui (`InterpDataset` n'est
consommé nulle part — les scripts interp-embed importent `Dataset` directement), mais c'est une
mine dormante pour quiconque écrirait plus tard `from sae_shared import InterpDataset` en
supposant le submodule disponible.

### 1.3 SAE Boost — un écart au niveau de l'équation

**Fidèle** : SAE secondaire entraîné sur l'erreur de reconstruction du SAE gelé ✓ ; pas de biais de
décodeur sur la branche résiduelle ✓ (le papier l'omet explicitement) ; k=5 sur le résidu contre
k≈50 sur le core ✓ (leur Table 12) ; d_extra=1024 ✓ ; BatchTopK à l'entraînement converti en seuil
JumpReLU à l'inférence ✓.

**Écart 1 — entrée de l'encodeur résiduel : corrigé.** §3.1 du papier :
`ê = W_dec^res · σ(W_enc^res · x + b_enc^res)` — l'encodeur résiduel lit **x**, la cible est **e**.
Le dépôt calculait `pre = self._pre_extra(residual)` : l'encodeur lisait **e**. `frozen_core.py`
lit désormais `x` (`encode`/`forward` appellent `_pre_extra(x_bf16.float())`, plus
`_pre_extra(residual)`) ; la cible de reconstruction (`mse_loss`/`_aux_loss`) reste `e`, inchangée.
Ajout nécessaire pour que ça tienne numériquement : `encoder_input_scale`, un scalaire de
normalisation SÉPARÉ de `input_scale` (qui reste calibré sur `e`, côté décodeur) — sans lui,
l'encodeur recevrait `x` (activations massives, norme ~1e5) à la même échelle que l'ancien `e`
(norme ~quelques % de celle de `x`), ce qui aurait fait exploser les pré-activations. Calibré sur
des échantillons `x` appariés aux mêmes tokens que les résidus PCA (`domain_inputs`, passé
maintenant par `saev5.py` à la construction d'`ExtendedSAE`/`FrozenDecoderExtendedSAE`). Un garde-fou
explicite (`saev5.py`, au chargement d'un `p1_frozen_core_*.pt`) refuse désormais de charger un
checkpoint entraîné avant ce correctif (`config["encoder_input"] != "x"`) plutôt que de charger
silencieusement des poids ajustés pour un input différent — piège de clé de cache que `CLAUDE.md`
signale déjà pour ce loader, désormais fermé pour cet écart précis. Testé (CPU, `tests/
test_frozen_core_encoder_reads_x.py`) : l'encodeur reçoit bien `x` et pas `e` (mock `core_sae` avec
un `decode()` non trivial, pour rendre les deux cas distinguables — les tests préexistants de
`test_frozen_core.py` ne le pouvaient pas, leur mock retournant toujours zéro) ; la cible de
reconstruction reste `e` ; `encode()` n'appelle plus `core_sae.decode()` du tout (économie directe,
cf. §2.5) ; `encoder_input_scale` et `input_scale` se calibrent bien sur des distributions
différentes ; pas de régression du piège dtype bf16/fp32 déjà documenté dans ce fichier.

Conséquences concrètes de l'écart original (pour mémoire, 1 et 2 désormais résolues par ce
correctif) :
1. Le « stitched SAE » de leur Figure 1 (droite) était impossible avec l'ancienne version : leur
   version est un unique SAE concaténé lisant *x*, l'ancienne exigeait de reconstruire `x̂_core`
   **avant** de pouvoir encoder. Redevenu possible.
2. C'était précisément ce qui forçait `decode_core_sparse` à chaque document au ré-encodage et le
   stockage de `raw_acts` dans chaque fragment. `encode()` n'a plus besoin de `core_out` du tout
   (§2.5) ; le stockage de `raw_acts`/`e` en double (§2.3, G3) est maintenant un choix de perf
   restant à faire, plus une conséquence forcée de l'architecture — toujours pas fait (dépend du
   format de fragment, cf. §2.3, tenu à l'écart tant que le run en cours n'est pas terminé).
3. **Point de vigilance qui reste ouvert, pas résolu par le seul changement de code** : l'ancienne
   version (encodeur sur `e`) évitait par construction les activations massives de `x` (norme
   ~1e5) ; la nouvelle les expose à l'encodeur, seulement atténuées par `encoder_input_scale`
   (normalisation d'échelle globale, pas un traitement des outliers comme `norm_outlier_mask`
   côté Pipeline 1). À vérifier empiriquement dès qu'un run complet aura tourné avec ce correctif
   (`dead_frac`, `l0_extra`, courbes de loss du premier run post-correctif contre §23.4/§56) avant
   de le tenir pour acquis — c'est exactement le type d'écart que R6 (§4.3) demande de documenter
   avec un test d'équivalence chiffré, pas de supposer réglé parce que fidèle au papier.

**Écart 2 — budget de tokens.** §4.2.5 : sous 100 M tokens, le SAE résiduel dégrade l'EV du domaine
général jusqu'à −31 % ; la convergence des features n'arrive qu'au-delà de 200 M. La référence du
papier est **1 Md tokens uniques par domaine**. `run_sae_v14_main.slurm` fixe
`N_TOKENS_EXTRA_TRAIN=100000000` × `EPOCHS_EXTRA=10`. 10 époques sur 100 M ≠ 1 Md tokens uniques :
vous êtes exactement **au seuil que le papier signale comme dangereux**, et la répétition n'achète
pas de diversité. À écrire explicitement dans les limites du rapport ; et à mesurer :
`compare_to_frozen_benchmark.py` existe justement pour vérifier la non-régression sur le domaine
général (leur Table 2, <1 %) — il n'est référencé par aucun `.slurm`.

**Écart 3** : `input_scale` (médiane des normes) et l'initialisation PCA du décodeur sont des
ajouts du dépôt absents du papier. Légitimes, mais ils invalident la baseline « Extended SAE
(random init) » de leur Table 3 comme point de comparaison.

### 1.4 Latent Terms

**Section obsolète depuis les commits `baeaa86`/`d184774` ("Latent Terms correction" / "Latent
terms")** : les trois écarts de fond signalés ci-dessous ont été corrigés dans le dépôt. Mais la
version corrigée n'a, à ce jour, **jamais terminé un run** — deux OOM consécutifs sur le job qui la
teste, le second après le correctif du premier. Nouvel état :

**Corrigé.**
1. **Entraînement in-domain → hors-domaine.** `latent_terms.py` n'entraîne plus le SAE sur les
   phrases des mails : `build_token_training_pool` tire désormais son corpus de FineWeb2-fr
   générique (`_stream_generic_texts`, streaming), conformément à *« the SAE never sees data which
   is directly in-domain for retrieval tasks »*.
2. **Granularité phrase → token.** `LatentTermsSAE` (Top-K, Gao et al. 2024, avec AuxK — extension
   assumée et documentée en tête de fichier, Table 4 du papier ne le liste pas explicitement) encode
   chaque activation token de F2LLM, plus les embeddings de phrase pré-poolés de `PhraseLevelSAE`.
   `latent_doc_weights` fait le sum-pooling sur les tokens **du document**, comme l'Eq. 9.
3. **Troncature Matryoshka disparue.** Le module lit `last_hidden_state` brut de F2LLM, sans passer
   par `extract_f2llm_embeddings` (donc sans la troncature à `MATRYOSHKA_DIM` ni la L2-normalisation
   de Pipeline 2, absentes du papier).
4. **Clé de cache.** `model_tag` (dérivé de `EMB_MODEL`) est maintenant inclus dans le chemin du
   pool d'entraînement et du checkpoint SAE (`latent_terms.py:404-410`) — corrige pour ce module le
   piège de cache générique documenté dans `CLAUDE.md`.

**Toujours ouvert.**
- Perf : `self.W.getcol(j)` (`latent_terms.py:381`) est toujours un accès colonne sur une CSR — la
  conversion CSC de l'audit initial n'a pas été appliquée.
- Évaluation : `scripts/latent_retrieval_precision_eval.py` calcule maintenant Precision@10/@20
  contre une baseline TF-IDF sur 4 requêtes paraphrasées — un vrai chiffre comparatif, mais toujours
  pas MAP/nDCG/BEIR, et le label de pertinence reste le même filtre regex faible que le reste du
  dépôt (`INTENT_KEYWORDS_FR`). À garder étiqueté comme évaluation indicative, pas benchmark IR.

**Nouveau, et plus grave que ce que ça remplace : la version fidèle au papier n'a produit aucun
résultat.** `logs/analysis/latent_retrieval_precision_eval_44434.log` (`--mem=96G`) meurt OOM
pendant le chargement du corpus FineWeb2-fr générique — cause identifiée dans le dépôt lui-même
(commentaire de `run_latent_retrieval_precision_eval.slurm`) : `sample_fineweb2_chunks`
(`src/data/preparation.py`) charge le shard entier (4,6 Go compressé) en RAM avant tout filtrage.
Correctif appliqué (`d184774`) : lecture en streaming (`_stream_generic_texts`,
`load_dataset(..., streaming=True)`) + `--mem=110G`. **Le job suivant (44438) meurt OOM lui aussi**,
en 12 minutes, avant même d'imprimer le message de fin de constitution du pool — le correctif n'a
pas réglé le problème.

Cause probable, non diagnostiquée dans le dépôt : `build_token_training_pool`
(`latent_terms.py:233-260`) accumule chaque tenseur de tokens par document dans une liste Python
`pool` jusqu'à `n_tok ≥ target_tokens` (~33M tokens × 896 dims × 2 o bf16 ≈ 59 Go), puis appelle
`embeddings = torch.cat(pool, dim=0)[:target_tokens]` (ligne 260). `torch.cat` alloue un second
tenseur contigu de ~59 Go **pendant que `pool` reste référencée** — rien ne la libère avant le
`return`, et le slice `[:target_tokens]` est une vue, pas une copie, donc ne permet pas de
raccourcir `pool` en amont. Pic RSS attendu : proche de 2× la taille du pool (~118 Go) plus
l'overhead modèle/tokenizer, ce qui explique aussi bien le kill à 96G que celui à 110G, et est
cohérent avec le `MaxRSS=65,7 Go` relevé par `sacct` sur le job 44438 (échantillonnage périodique,
peut manquer un pic transitoire de quelques secondes). Correctif : préallouer
`torch.empty(target_tokens, d_in, dtype=torch.bfloat16)` et écrire chaque lot de tokens directement
dedans via un curseur d'écriture, sans liste intermédiaire — mémoire bornée à ~59 Go, jamais 2×.

**Conséquence à assumer telle quelle : tant que ce bug n'est pas corrigé, la réimplémentation
fidèle au papier n'a produit aucun chiffre.** Les seuls résultats Latent Terms existants restent
ceux de `RESULTS_TESTS.md` §26/§68(c)/§69(c), que le dépôt lui-même marque désormais comme
supersédés — ils mesurent l'ancienne méthode phrase-level in-domain que la version précédente de
cet audit disqualifiait. Ne pas citer §26/§68/§69 comme résultat Latent Terms de référence tant
qu'un run de la version corrigée n'a pas produit de JSON.

### 1.5 Matryoshka

`MATRYOSHKA_DIM` tronque l'embedding F2LLM (MRL, Kusupati). Ce n'est **pas** le Matryoshka SAE de
`Matryoshka.pdf` (dictionnaires imbriqués, pertes préfixes). L'« hybride frozen-core Matryoshka SAE »
présenté comme contribution originale n'existe dans aucun fichier du dépôt. Soit l'implémenter,
soit le retirer des contributions revendiquées — en l'état c'est un risque en soutenance et en
review.

---

## 2. Audit de performance

### 2.1 Le fait mesuré, et ce qu'il implique

`docs/audit_2026_08_extraction_batch_size_sweep_results.json` (A100-40G, LAYER=24, textes réels) :

| batch | 4 | 8 | 16 | 24 | 32 | 48 |
|---|---|---|---|---|---|---|
| docs/s | 14,54 | 14,45 | 14,75 | 14,59 | 14,72 | OOM |
| VRAM (Go) | 26,2 | 28,0 | 31,6 | 35,2 | 38,8 | — |

**Débit strictement plat de 4 à 32.** Conclusion à en tirer, et elle contredit l'intuition
habituelle : augmenter `EXTRACTION_BATCH_SIZE` ne rapportera rien. Le forward est déjà saturé
— non par le compute, mais par le trafic mémoire de `output_hidden_states=True`, qui matérialise
**49 tenseurs de hidden states** par forward (à batch 32 × 512 : 49 × 32 × 512 × 3840 × 2 o ≈
6,2 Go alloués/libérés par batch, ce qui explique aussi la pente VRAM +12,6 Go entre batch 4 et 32).

Budget du run à 200 M (§23.5) : 20 h 57 d'extraction. Le forward pur à 14,6 docs/s pour ~432 k
documents = **8 h 13**. **Il reste ~12 h 45, soit 61 % du wall-clock d'extraction, hors GPU.**
Le GPU est inactif la majorité du temps. C'est là que se trouvent les gains, pas dans le batch.

### 2.2 P1 — extraction : les six goulots, par ordre de gain

**G1 — Le modèle calcule 17 à 24 couches inutiles. (gain 1,4–2,0×, une trentaine de lignes) —
CORRIGÉ ET DÉPLOYÉ, vérifié sur GPU.**
`saev5.py` faisait un forward complet sur les 48 blocs de Gemma-3-12B pour ne lire que
`hidden_states[LAYER]`. Avec le preset actuel `LAYER=31` : **35 % des FLOPs du forward étaient
jetés** (50 % à layer 24).

Deux tentatives, la première a servi de garde-fou à la seconde :
- **V1 (échouée)** : troncature `layers[:LAYER]` en gardant `output_hidden_states=True`, testée
  sur GPU avant tout déploiement (`scripts/audit_2026_08_layer_truncation_equivalence_and_speedup.py`,
  job 44536, a100, indépendant du run en cours). `torch.equal(hidden_full, hidden_trunc) = False`,
  écart maximum ~6×10⁵ — pas un bruit numérique. Le débit mesuré était pourtant conforme à la
  prédiction (14,47 → 28,58 docs/s, 1,98×), ce qui aurait pu faire passer le correctif pour un
  succès si l'équivalence n'avait pas été vérifiée en premier — **exactement le risque que ce
  type de correctif présente : un gain de vitesse réel sur un résultat silencieusement faux.**
  Cause confirmée en lisant `transformers/utils/output_capturing.py` : `output_hidden_states=True`
  passe par un mécanisme générique (`_can_record_outputs`, hooks posés sur chaque
  `Gemma3DecoderLayer`) qui remplace INCONDITIONNELLEMENT la DERNIÈRE entrée de `hidden_states`
  par `last_hidden_state` — la sortie **après le RMSNorm final** du modèle
  (`tie_last_hidden_states=True`, documenté "vrai pour tous les modèles de langage"). Invisible
  sur le modèle complet (`LAYER` n'est jamais la dernière entrée sur 49) ; silencieusement faux
  une fois `LAYER` devenu la dernière entrée par troncature. L'ordre de grandeur de l'écart
  (~6×10⁵) coïncide avec celui des activations massives de Gemma-3 documentées ailleurs dans cet
  audit : RMSNorm divise par la RMS du vecteur, dominée par le canal à activation massive — la
  version normalisée de ce canal s'effondre à O(1), et raw − normed ≈ raw ≈ l'activation massive
  elle-même. Cohérence totale entre le mécanisme lu dans le code source et la magnitude observée.
- **V2 (déployée)** : `register_forward_hook` **direct** sur `model.language_model.layers[LAYER-1]`
  (même mécanisme que `HOOK_TYPE=attn_out`/`mlp_out`, jamais affectés par ce piège puisqu'ils
  n'utilisent jamais `output_hidden_states`), `output_hidden_states` jamais passé. Revérifié sur
  GPU (job 44540, même script) : `torch.equal = True`, **écart maximum = 0,0 bit-à-bit**, débit
  14,44 → 28,59 docs/s (**1,98×**), pic VRAM 28,0 → 15,0 Go (**13 Go économisés**). Déployé dans
  `saev5.py` (troncature `layers[:LAYER]` pour resid_post / `layers[:LAYER+1]` pour attn_out/
  mlp_out, unifié sur le même mécanisme `_hook_capture`, `output_hidden_states` retiré du chemin
  d'extraction). `Gemma3DecoderLayer.forward()` retourne un tenseur nu (pas un tuple), confirmé en
  lisant `modeling_gemma3.py` avant d'écrire le hook.

**G2 — Un fichier disque par document. (gain 2–2,5× sur le wall-clock d'extraction)**
`fragment_store.save_fragment` écrit `doc_{i:05d}.pt` via `torch.save` (conteneur zip + pickle
d'une liste Python de chaînes) pour **chaque document**. À 432 k documents sur `/home` (volume
réseau partagé, 35 To, 97 % plein) : 432 k créations de fichiers + 432 k `fsync` implicites. Sur un
Lustre/GPFS partagé, la latence de métadonnées (~1–5 ms/fichier) domine tout le reste :
432 k × 3 ms ≈ **22 min de latence pure**, à quoi s'ajoute la sérialisation et l'écriture de
~1,9 Mo de `raw_acts` par document. C'est le principal candidat pour les 12 h 45 manquantes.
Correctifs cumulables :
- **Sharder** : 1 fichier pour 1 000 documents (`rowptr`/`cols`/`vals` concaténés + index d'offsets
  documentaires). 432 k fichiers → 432. Le format CSR s'y prête nativement.
- **Écrire en arrière-plan** : un `ThreadPoolExecutor(1)` ou une `queue.Queue` consommée par un
  thread dédié suffit — l'écriture est libératrice du GIL (I/O). Le GPU ne doit jamais attendre le
  disque. Aujourd'hui la boucle est strictement séquentielle GPU→CPU→disque→GPU.
- Le `.pkl` legacy et sa branche de rétro-compat peuvent partir avec.
- Note SLURM : `--cpus-per-task=8` pour un pipeline dont 61 % du temps est CPU/I-O séquentiel
  monothread — augmenter les cœurs ne servira qu'après avoir parallélisé.

**G3 — La double copie des activations brutes. (gain : ~1 To de disque, et c'est ce qui a tué le
run 200 M)**
Aujourd'hui vous écrivez **deux fois** le même residual stream :
- `raw_acts` dans chaque fragment (`saev5.py:925`) : ~1,9 Mo/doc × 432 k ≈ **830 Go** ;
- le réservoir memmap : 100 M × 3840 × 2 o = **768 Go**.
Pic ≈ **1,6 To** — cohérent avec le fichier de 1,22 To et le remplissage de `/home` du 2026-08-05.
Correctif structurel, qui découle directement de l'écart 1 de §1.3 : **stocker `e = x − x̂_core`
au lieu de `x`**.
- Le coût additionnel à l'extraction est un `decode` du core, soit 63 MFLOP/token contre
  24 GFLOP/token pour le forward 12B : **0,26 % de surcoût**, négligeable.
- `e` n'a plus les activations massives (le core les capture) : sa plage dynamique autorise un
  stockage **int8 avec échelle par ligne** (erreur relative ~0,4 %, à comparer aux 6–7 % que vous
  avez déjà mesurés et acceptés en §61 sur la soustraction bf16). 768 Go → **192 Go** ; 830 Go →
  **207 Go**. Pic 1,6 To → **0,4 To**.
- Bénéfice en cascade : `decode_core_sparse` disparaît de la passe de ré-encodage, et le core
  n'est plus recalculé à chaque époque d'entraînement (cf. G5).
- Conserver `p1_eval_raw_tokens.pt` (4 096 tokens fp32) pour FVE/ρ_SAE : déjà le cas.

**G4 — Encodage SAE document par document.**
`saev5.py:908` appelle `pretrained_sae.encode(filtered)` dans une boucle `for b in range(B)`, donc
4 GEMM de [T, 3840] × [3840, 16384] au lieu d'un seul de [ΣT, 3840]. Idem pour `_dense_to_csr`.
Aplatir le batch (masque + `nonzero` global, split par `doc_ids` à la fin) supprime 4× l'overhead
de lancement de kernels et de conversion CSR. Gain modeste seul, mais nécessaire pour que G2 tienne.

**G5 — Écritures aléatoires dans un memmap de 768 Go — CORRIGÉ.**
`saev5.py:949` : `reservoir[j[hit]] = x_new[hit]` avec `j` uniforme sur [0, N). Chaque écriture
touche une page de 15 Ko dispersée dans un fichier de 768 Go. En phase 2 du réservoir (dès que
100 M tokens sont vus), c'est du **write-amplification aléatoire pur** sur un volume réseau.
Corrigé par accumulation en buffer (`_pending_j`/`_pending_x`, seuil `_RESERVOIR_FLUSH_SIZE =
50 000`) : les remplacements de phase 2 s'accumulent au lieu d'écrire immédiatement, et sont
appliqués par blocs triés par offset dès que le seuil est atteint
(`torch.argsort(j_cat, stable=True)` puis `reservoir[j_cat[order]] = x_cat[order]`) — écriture
séquentielle plutôt que dispersée. Point de correction attendu par rapport à l'écriture immédiate :
en cas de collision (deux remplacements du buffer visant le même offset `j`), c'est la **dernière
entrée temporelle** qui doit l'emporter, comme le ferait une suite d'écritures immédiates non
triées — garanti par `stable=True` (le tri préserve l'ordre relatif des éléments égaux, donc parmi
deux entrées de même offset, la plus récente dans le buffer reste la dernière après tri, donc la
dernière appliquée). `_flush_pending_reservoir_writes()` est appelé avant tout checkpoint de
reprise (§2.3) et à la fin de la boucle, même discipline "flush avant checkpoint" que G2/`AsyncFragmentWriter`
— un buffer non vidé au moment d'un checkpoint casserait l'invariant de reprise (checkpoint avancé,
écritures pas encore appliquées). Testé (CPU, `tests/test_reservoir_write_batching.py`) :
équivalence bit-exacte buffer trié vs écriture immédiate sur un cas aléatoire (50 batches, flush
à 17), et cas de collision construit à la main (deux batches visant le même offset avec des
valeurs différentes, y compris quand les deux se retrouvent dans le même buffer avant flush).
Le réservoir lui-même reste identique au tirage aléatoire près (mêmes indices `j`, mêmes valeurs
`x_new`, seul l'ORDRE et le REGROUPEMENT des écritures physiques changent) — aucun impact sur la
statistique de l'échantillonnage par réservoir (Algorithm R), uniquement sur la localité I/O.

**G6 — Options PyTorch standard absentes.**
Aucune occurrence dans tout le dépôt de : `attn_implementation` (SDPA/FA2 non forcé),
`torch.backends.cuda.matmul.allow_tf32` / `set_float32_matmul_precision`, `torch.compile`,
`pin_memory`, `non_blocking=True`, `torch.inference_mode()` (vous utilisez `no_grad`, légèrement
plus coûteux), `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (présent uniquement dans
`slurm/baseline_diffing/*`, pas dans `pipeline_runs/`). Chacun vaut 1 à 5 % ; ensemble sur un run
de 20 h, c'est plusieurs heures. Ce sont des lignes uniques.

**G7 — Filler encodé et fragmenté pour rien — CORRIGÉ.** Symétrique du correctif de §2.5
(ré-encodage), appliqué ici côté extraction une fois le premier confirmé sûr. Avant : chaque
document filler passait par `pretrained_sae.encode()` (coût GPU dominant par document) **et**
`save_fragment()` (écriture disque complète), alors que son seul rôle est de nourrir le réservoir
de résidus bruts — rôle qui ne dépend ni de l'encodage core ni de l'écriture de fragment. Corrigé
(`is_filler_document`, `src/data/preparation.py`, appelé dans la boucle d'extraction) : pour un
document filler, ni encodage core ni fragment ; `filtered` (résidu brut) alimente directement le
réservoir comme avant, une entrée placeholder bon marché (`torch.zeros`, même dtype) garde
`all_doc_sae_acts` aligné sur `doc_global_idx`. Répercussions traitées explicitement : le test
« fragments complets » (`n_fragmented_expected`) exclut désormais la plage filler de son décompte
attendu ; le mécanisme de repli qui reconstruit le réservoir depuis les fragments (utilisé quand
seuls les résidus sont manquants, fragments intacts) ne peut plus couvrir le filler — dégradé
explicitement (avertissement imprimé) plutôt que silencieusement faux, plage train uniquement. Le
mécanisme de reprise (§2.3) mis à jour en conséquence (reconstruction par placeholder pour les
positions filler déjà « traitées »). Testé (CPU, `tests/test_extraction_filler_lightening.py`).
Risque résiduel à surveiller sur le premier run réel : le placeholder est casté explicitement au
même dtype que la branche réelle (`TORCH_DTYPE`, cohérent avec `torch.zeros(D_EXTRA, dtype=
token_sae_acts.dtype, ...)` déjà présent dans le code pour la même raison) — non vérifié sur GPU à
ce jour, un éventuel désaccord de dtype ferait échouer `torch.stack` de façon bruyante et immédiate
(pas un risque de corruption silencieuse), donc facilement détectable au premier run.

### 2.3 Reprise après OOM / timeout — **CORRIGÉ pour les trois boucles longues identifiées**

**Mise à jour** : les trois boucles de plusieurs heures de ce dépôt ont maintenant une reprise —
extraction P1 (`saev5.py`, ~24 h sur le run de référence), ré-encodage ExtendedSAE (`saev5.py`,
deuxième passe de durée comparable, §2.5 — identifiée en observant le job 44211 tourner dedans
en temps réel), extraction F2LLM P2 (`phrase_sae.py::extract_f2llm_embeddings`). Mécanisme commun
(`src/storage/checkpoint.py`) : checkpoint JSON atomique (tmp + `os.replace`) tous les
`EXTRACTION_CHECKPOINT_INTERVAL` documents (défaut 2000) ou tous les `shard_size` (P2, défaut
100 000 phrases) ; handler `SIGTERM`/`SIGUSR1` (`--signal=B:USR1@600` côté SLURM) qui force un
checkpoint propre avant le `SIGKILL` d'un timeout plutôt que de perdre le travail en cours. Le
critère de reprise est toujours *quel est le prochain élément non traité*, jamais *le run est-il
complet* — un fragment/shard au-delà du checkpoint persisté, s'il en existe un d'une tentative
précédente tuée en plein lot, est délibérément ignoré et réécrit, jamais fait confiance.

Point spécifique à l'extraction P1 : le compteur du réservoir de Vitter (`n_residuals_seen`,
`n_residuals_collected`) fait partie de l'état persisté — sans lui, une reprise aurait biaisé
l'échantillon d'entraînement de l'ExtendedSAE vers les documents post-reprise (le bug que cette
section signalait déjà comme risque associé à toute reprise naïve). Vérifié par un test CPU
statistique (`tests/test_reservoir_resume_invariant.py`) : la probabilité marginale d'inclusion
d'un item dans le réservoir final est la même, à la tolérance statistique près, qu'on traite le
flux en une passe ou en deux passes avec compteurs correctement repris — et un contre-essai
confirme que le test détecterait bien une régression vers l'ancien comportement (compteurs
remis à 0 à la reprise).

Point spécifique au ré-encodage : les documents déjà traités **purgent** leur `raw_acts` du
fragment (économie de disque intentionnelle) — la reprise reconstruit leur vecteur document via
`doc_maxpool` sur le CSR déjà fusionné (core+extra), sans avoir besoin de `raw_acts`. Effet de
bord découvert en implémentant ceci : `p1_eval_raw_tokens.pt` (échantillon FVE/ρ_SAE, capturé
juste avant purge) était lui aussi exposé à une perte de données en cas de coupure pendant sa
fenêtre de capture — persisté de façon incrémentale désormais, avec avertissement explicite si
une reprise survient après la fenêtre sans fichier préexistant (perte résiduelle possible dans ce
cas précis, mais visible, jamais silencieuse).

**Bug de fidélité trouvé en implémentant ceci, sans rapport avec la reprise elle-même** : le site
d'appel du ré-encodage (`ext_sae._encode_extra_acts(...)`, accès direct à la méthode privée,
donc non couvert par les tests de `encode()`/`forward()` déjà mis à jour) passait encore le
résidu `e = x - x̂_core` à l'encodeur extra, alors que le correctif SAE Boost de cette session
(§1.3) avait déjà changé `frozen_core.py` pour que l'encodeur lise `x`. Régression silencieuse —
mêmes noms de fonctions, mauvais argument, aucune erreur levée. Corrigée dans la foulée (l'appel
passe maintenant `raw_acts` directement, et `decode_core_sparse` — qui ne servait qu'à calculer
ce résidu — a disparu de cette passe, gain de performance en plus du correctif de fidélité) ;
garde-fou ajouté en test (`tests/test_frozen_core_encoder_reads_x.py::
test_direct_encode_extra_acts_call_matches_encode_method`) pour empêcher toute récidive de ce
type de divergence entre un appel direct et la méthode publique.

**Non fait** : la troncature/sharding des fragments eux-mêmes (G2), le stockage de `e` en int8
(G3) — ces deux-là changent le format sur disque et restent tenus à l'écart tant qu'un run de
référence n'a pas tourné sur le nouveau mécanisme de reprise.

---

Contenu original de cette section (diagnostic qui a motivé le correctif ci-dessus, conservé pour
mémoire) :

C'est la réponse directe à votre question, et le risque le plus coûteux du dépôt.

**Pipeline 1.** `saev5.py:783` :
```python
if len(fragment_ids) == len(all_texts):
```
La restauration du cache est **tout-ou-rien**. Les fragments sont bien écrits au fil de l'eau, mais :
- `all_doc_sae_acts` est une liste Python accumulée en RAM, `torch.stack` + `torch.save`
  uniquement à la ligne 952–953, **après** la boucle complète ;
- le sidecar `p1_raw_residuals.memmap.meta.json` n'est écrit qu'après la boucle ;
- si le job meurt à 95 % (OOM, timeout SLURM, disque plein — les trois se sont produits), le test
  ligne 783 échoue, `_need_extraction = True`, et **tout est réextrait depuis zéro**. 20 h perdues.
- Pire : il existe un chemin de reconstruction du réservoir depuis les fragments (lignes 790–812),
  mais il est **conditionné au même test tout-ou-rien** et donc inatteignable en cas de crash.

Correctif (une heure de travail, gain : la totalité du risque) :
1. Remplacer le test par `missing = [i for i in range(len(all_texts)) if not fragment_exists(dir, i)]`
   et ne réextraire que `missing`. La primitive `fragment_exists` existe déjà et est inutilisée dans
   ce chemin.
2. Écrire `all_doc_sae_acts` de façon incrémentale dans un memmap `[n_docs, d_total]` plutôt que
   dans une liste, avec un fichier de progression `{"n_done": k}` fsync'é tous les N documents.
3. Persister le sidecar du réservoir périodiquement (avec `n_seen` **et** `n_collected`, sinon
   la reprise du réservoir de Vitter est statistiquement fausse : le compteur `n_residuals_seen`
   n'est stocké nulle part et repart à 0).
4. Ajouter un handler `SIGTERM`/`SIGUSR1` (SLURM envoie `SIGTERM` avant `SIGKILL` ; `--signal=B:USR1@600`
   donne 10 min) qui flush proprement.

Point 3 souligne un **bug latent** : même si vous ajoutiez une reprise naïve, l'échantillonnage de
Vitter reprendrait avec `n_residuals_seen=0`, ce qui rendrait l'échantillon fortement biaisé vers
les documents post-reprise. À traiter en même temps.

**Pipeline 2.** Même diagnostic, plus brutal. `phrase_sae.extract_f2llm_embeddings` accumule
`all_embs` en RAM et n'écrit `cache_path + ".pt"` **qu'à la toute fin**. Aucun shard, aucun
compteur. Un job de plusieurs heures sur des millions de phrases est en tout-ou-rien intégral.
Correctif : écrire un `.npy` par tranche de 100 k phrases + reprise par comptage de shards.

**Bug de cache associé (P2), sérieux.** La clé de cache est
`train_phrase_emb_dim{MATRYOSHKA_DIM}_n{len(train_phrases)}` (`saev5.py:1437`). Elle **n'encode ni
`EMB_MODEL`, ni `EMB_POOLING`, ni `max_length`**. Basculer de `F2LLM-v2-330M` (défaut `config.py`)
à `F2LLM-v2-80M` (ce que force `run_sae_v14_main.slurm`) ou à `bge-m3` sur le même corpus recharge
**silencieusement les embeddings du mauvais modèle**. C'est exactement le piège que `CLAUDE.md`
documente pour `load_or_train_extended_sae` ; il est également présent ici et non signalé. Tout
run P2 comparant deux backbones est suspect tant que ce n'est pas corrigé et les caches purgés.

### 2.4 Entraînement de l'ExtendedSAE

Le budget FLOP est faible (≈2,5·10¹⁷ FLOP pour 100 M × 10 époques, soit ~15 min à 300 TFLOPS
effectifs). Le temps réel est ailleurs :

**Le core est recalculé à chaque époque.** `frozen_core.py:92-94` exécute
`core_sae.encode` + `core_sae.decode` sous `no_grad` à **chaque step**, sur des tokens identiques
d'une époque à l'autre. Coût par token : 252 MFLOP pour le core contre ~47 MFLOP pour la branche
extra (fwd+bwd) → **84 % du compute d'entraînement est un recalcul déterministe**. Précalculer
`e` une fois (G3) ramène 10 époques à 1× core + 10× extra : **≈ 3,4× sur l'entraînement**.

**976 000 steps, 4 synchronisations GPU chacun.** `BATCH_SIZE = 1024` en dur
(`sae_shared.py:178`) sur 100 M tokens × 10 époques = 976 562 steps. Chaque step fait quatre
`.item()` (`loss`, `l0`, `dead_frac`, `aux_loss`), soit **quatre `cudaStreamSynchronize` par step**,
sur un batch dont le calcul dure <1 ms. Le step est intégralement dominé par la synchro et le
Python. Correctifs :
- `BATCH_SIZE` à 16 384–65 536 (la mémoire le permet largement : 16 k × 3840 × 2 o = 126 Mo) →
  **16 à 64× moins de steps**. Attention : BatchTopK a un budget **partagé sur le batch**,
  changer B change le régime de sparsité — à traiter comme une ablation, pas comme un réglage.
- Accumuler les métriques en tenseurs GPU et ne faire `.item()` qu'en fin d'époque. L'historique
  actuel produit un JSON de ~50 Mo (976 k × 6 tableaux) que `generate_diagnostic_plots.py` doit
  ensuite relire.
- `torch.optim.Adam(..., fused=True)`, `pin_memory`, `non_blocking=True`.

**`torch.randperm(100_000_000)` par époque.** `sae_shared.py:172` et 187 : deux tenseurs int64 de
800 Mo, plus 800 Mo réalloués à chaque époque, plus ~10 s de génération. Remplacer par un
échantillonnage par blocs (permuter des chunks de 64 k, puis intra-chunk) : localité mémoire
préservée **et** 1 600 Mo économisés.

**`feature_acts` inutilisé.** `frozen_core.py:113` concatène `core_acts` (16 384 colonnes) dans le
dict de retour à chaque step ; le harnais ne le lit jamais. Allocation de [B, 17408] fp32 pure
perte. Le rendre optionnel (`return_feature_acts=False` par défaut).

### 2.5 Ré-encodage SAEBoostResidualSAE (`saev5.py`)

**Filler exclu de cette passe — CORRIGÉ.** Découvert en observant job 44211 tourner en direct
dans cette exacte passe (4h+, 46% avant timeout SLURM à 48h) : `all_texts = train_texts +
volume_filler_texts + test_texts + diff_texts`, et le ré-encodage itérait sur `range(len(all_texts))`
— y compris le filler (FineWeb2-fr générique, ajouté uniquement pour donner du volume de tokens au
réservoir d'entraînement, §1.3/§2.3). Vérifié précisément (`grep` de tous les consommateurs de
`all_doc_sae_acts`/fragments) : **aucun** ne relit jamais la tranche filler —
`train_doc_acts`/`test_doc_acts`/`diff_doc_acts` l'excluent explicitement par slicing, la sélection
de features n'itère que sur `range(n_train)`, le juge n'utilise que des offsets qui la sautent. Sur
le run de référence, filler = 540 000/584 253 documents (**92% du corpus**) — 92% du travail de
cette passe portait sur des vecteurs jamais consultés. Corrigé :
`src/data/preparation.py::build_reencode_targets` construit `train ∪ (test ∪ diff)`, filler exclu ;
le ré-encodage n'itère plus que sur cet ensemble (`saev5.py`). Le mécanisme de reprise (§2.3)
adapté en conséquence : le checkpoint indexe une **position** dans cette liste réduite, pas
l'indice de document brut. Réduction attendue du volume traité par cette passe : proche de ×13 sur
un run avec un filler de cette taille — le plus gros gain de compute de tout cet audit, plus grand
que G1. Testé (CPU, `tests/test_reencode_skips_filler.py`) : exclusion stricte du filler, ordre
préservé, invariant position→indice de document dont dépend la reprise.

**Reste à faire, volontairement pas touché dans cette passe** : l'**extraction** (§2.2) traite
encore le filler en entier (encodage core + écriture de fragment complet par document) alors que
seules ses activations brutes comptent, pour nourrir le réservoir — un gain de même nature,
symétrique, non appliqué ici pour limiter le rayon d'impact du changement (l'extraction interagit
avec la reprise ET le fallback de reconstruction du réservoir depuis les fragments, une surface de
risque plus large que le ré-encodage seul).

**Items originaux de cet audit sur cette passe, non résolus par le correctif ci-dessus** :
`for i in tqdm(...)` reste **un document à la fois** (pas de batching 64–256 documents) ; avec la
correction SAE Boost de §1.3, `decode_core_sparse` a disparu de cette passe (l'encodeur lit
directement `raw_acts`, plus besoin de reconstruire le core) — ce gain-là est déjà acquis, sans
lien avec G2/G3 qui restent, eux, à faire (fragments shardés, stockage `e` en int8).

### 2.6 Le juge — **réponse : non, Qwen n'est branché nulle part dans la production**

Recherche exhaustive de `Qwen|quantiz|BitsAndBytes|load_in_[48]bit` sur tout le dépôt :

- **Une seule occurrence fonctionnelle** : `scripts/imdb_genre_diffing_test.py:68`
  (`JUDGE_MODEL_PATH = ".../Qwen3.8-27B"`, `BitsAndBytesConfig(load_in_8bit=True)`). C'est le script
  de réplication du papier interp-embed, pas la pipeline.
- **Le juge de production est Gemma-3-12B-it, en bf16, non quantifié**, chargé via `MODEL_ID`
  (`saev5.py:1173` pour P1, `:1486` pour P2). Le juge est donc **le même modèle que l'extracteur** —
  ce que `RESULTS_TESTS.md` §48/§50/§52 a heureusement testé pour l'indépendance, mais qui reste
  une limite structurelle à énoncer.
- **Le variant bf16 non quantifié existe aussi pour Qwen** :
  `slurm/validation/run_imdb_genre_diffing_test_h100bis_bf16.slurm` (« Qwen3.8-27B en bf16 complet
  (~52 Go) »). Donc même là où Qwen est utilisé, la quantification n'est pas systématique.

**Contre-point important sur la quantification 8-bit.** `load_in_8bit` (LLM.int8 de bitsandbytes)
est une optimisation **mémoire, pas vitesse** : la décomposition en précision mixte la rend
typiquement **2 à 4× plus lente** que bf16 en génération. Si l'objectif est le débit, sur un H100
80 Go un 27B en bf16 (54 Go) tient et sera plus rapide. Si l'objectif est de faire tenir un 27B à
côté du 12B sur un A100-40G, alors préférer **NF4** (`load_in_4bit`, `bnb_4bit_compute_dtype=bfloat16`)
qui est plus rapide qu'int8, ou mieux : **AWQ/GPTQ** qui sont réellement accélérés. Généraliser
`load_in_8bit` partout serait une régression de performance.

**Le vrai goulot du juge est ailleurs : batch = 1.** `judge.py:269/295/322` : trois appels
`model.generate` séquentiels par feature (8 + 128 + 128 nouveaux tokens), `bs=1`. En bs=1 la
génération est **bornée par la bande passante mémoire** : 24 Go de poids / ~1,5 To/s ≈ 15 ms par
token, quelle que soit la charge de calcul. À `N_FEATURES_TO_LABEL=500` : ~264 tokens × 15 ms × 500
≈ **33 min** de génération dont l'essentiel est du transfert de poids inutile. Les features sont
**indépendantes** : batcher 16–32 prompts par `generate` (avec `padding_side="left"`) amortit le
transfert de poids sur tout le batch → **8 à 16×**, soit 33 min → 2–4 min. Aucun changement de
sémantique (`do_sample=False`). C'est le correctif le plus simple de tout cet audit.

**Trois chargements du 12B par run.** Extraction, puis juge (`saev5.py:1173`), puis hypothèse de
diff (`:1237`), avec `_trim_host_memory()` entre chaque. Sur `/home` partagé, charger 24 Go de
poids coûte 2 à 5 min à chaque fois. `RUN_DIFF_HYPOTHESIS=0` supprime déjà le troisième. Le second
peut être évité en gardant le modèle résident (24 Go bf16 + SAE 16k ≈ 26 Go, tient sur H100 80 Go)
— au prix d'un pic VRAM qui interdirait le repli A100-40G. Compromis à trancher explicitement,
pas par défaut.

### 2.7 Pipeline 2 / F2LLM

**Le backbone tourne en fp32.** `phrase_sae.py:126` :
```python
model = AutoModel.from_pretrained(EMB_MODEL, local_files_only=True).to(DEFAULT_DEVICE).eval()
```
Aucun `torch_dtype`. HuggingFace charge donc en **fp32**, et PyTorch désactive TF32 par défaut pour
les matmuls depuis la 1.12. Sur H100 : fp32 sans TF32 ≈ 67 TFLOPS contre 990 en bf16 —
**jusqu'à un ordre de grandeur perdu** sur toute l'extraction d'embeddings P2. Les sorties sont de
toute façon castées en `.float()` après pooling (ligne 152), donc le passage à bf16 est sans risque
pour le backward du SAE fp32. Correctif : `torch_dtype=torch.bfloat16` + par sécurité
`torch.set_float32_matmul_precision("high")` en tête de module. **Une ligne, gain potentiellement
massif.** À mesurer avant/après, mais c'est le premier endroit où regarder.

Autres points P2 :
- `batch_size = 128` en dur (ligne 130), non configurable, jamais balayé — contrairement à P1 où
  un audit dédié existe. Avec `max_length=128` et un 330M, un H100 accepte 1 024–2 048.
- Aucun tri par longueur avant batching : `padding=True` sur des phrases très hétérogènes gaspille
  du calcul proportionnellement à la variance des longueurs. Un tri + regroupement par bucket, puis
  restauration de l'ordre, est standard et gratuit.
- **Quadratique** : `saev5.py:1512`, `np.where(test_p2d_arr == doc_idx)` dans une boucle sur tous
  les documents → O(n_docs × n_phrases). À 20 k documents et 300 k phrases : 6·10⁹ comparaisons.
  Remplacer par un regroupement unique (`np.argsort(p2d)` + `np.searchsorted` sur les frontières,
  ou `collections.defaultdict`) — O(n log n) une fois.
- `load_or_train_sae` : `init_from_data` est appelé **avant** le test d'existence du checkpoint
  (lignes 191–195), donc calculé puis écrasé à chaque restauration. Sans gravité, mais symptomatique.
- Double normalisation du décodeur par step (`normalize_decoder()` puis `F.normalize` explicite,
  lignes 213–216) : la seconde est redondante.
- `compute_sae_metrics` caste l'entrée en bf16 alors que le SAE est fp32 : la promotion implicite
  fonctionne mais **les métriques publiées sont calculées sur une entrée dégradée** par rapport à
  l'entraînement. À aligner.

### 2.8 Étages aval — les murs de scalabilité

Aucun de ces points ne coûte cher aujourd'hui à 16k de largeur ; tous cassent à 65k ou 262k, ce qui
est précisément la direction annoncée.

| Emplacement | Problème | Complexité | Correctif |
|---|---|---|---|
| `cooccurrence.py:120` | boucle Python `fisher_exact` sur **toutes** les features, et `A[:, f].sum()` en accès colonne sur un tableau C-contigu | O(d) appels scipy + O(d·n) strided | compter en une fois via `A.sum(0)` ; ne lancer Fisher que sur les candidats pré-filtrés par |LOR| (BH sur l'ensemble screené, à documenter) |
| `cooccurrence.py:55` | `torch.triu_indices(K, K)` sans plafond, alors que le chemin `p1_npmi` plafonne à 4 000 | O(K²) mémoire : 262k → **TB** | même plafond `[:4000]` ou passage à un parcours par blocs |
| `saev5.py:432` | `SpectralClustering` sur `n_docs` documents construit une affinité n×n | O(n²) mémoire, O(n³) eigen | mur dur vers 15–20 k documents : Nyström, ou clustering sur un sous-échantillon puis assignation |
| `metrics.py:downstream_classification` | `LogisticRegression` **dense** sur [n, 17 408], upcasté fp64 par sklearn ; 5 plis séquentiels | compute-bound, plusieurs heures | **voir §3.3 — le correctif n'est pas plus de CPU, c'est du sparse** |
| `saev5.py:_fit_umap` | UMAP fitté **deux fois** (2D + 10D) sur une matrice dense [n, n_active], `n_jobs=1` forcé par `random_state` | 2× le coût dominant de l'aval | passer une `scipy.sparse.csr_matrix` (supporté par UMAP) ; ou fitter le 10D et projeter en 2D |
| `saev5.py:_embed_bge_m3` | recharge bge-m3 **depuis le disque à chaque appel** (3 appels par run P1 : corrélations, clustering ciblé, retrieval) et ré-embarque tous les labels | 3× (chargement + N labels) | `functools.lru_cache` sur le modèle, cache disque des embeddings de labels |
| `saev5.py:df.iterrows()` (hover UMAP) | boucle Python + `topk` torch par document | O(n) appels Python | vectoriser le `topk` sur toute la matrice |

### 2.9 Plan chiffré, par ordre décroissant de rendement

| # | Action | Fichier | Gain estimé | Effort | Statut |
|---|---|---|---|---|---|
| 1 | Batcher les prompts du juge (16–32) | `judge.py:210-340` | 8–16× sur le juge (33 min → 3 min) | 2 h | **fait, restructuré en 3 passes** (`_batched_generate`, `odd_one_out_judge`/`local_gemma_judge` unifiés) — testé sur GPU (job 44570) : **6,51× confirmé** (49,7s → 7,6s, 24 prompts), mais **1/24 désaccord texte-à-texte**, pas 0. Cause identifiée : non-associativité flottante des kernels batchés GPU (matmul/attention), pas un bug de masque — connue et documentée dans la littérature ML systems, affecte potentiellement tout service LLM batché, greedy (`do_sample=False`) n'élimine que l'aléa du sampling, pas cet effet. Le désaccord observé porte sur un prompt de test long (32 tokens, résumé ouvert) ; le stade le plus sensible en production (odd-one-out, 8 tokens, un seul chiffre à extraire) n'a pas été testé séparément — à faire avant de faire confiance à `interp_score` en routine. Point de comparaison : le protocole odd-one-out lui-même est déjà bruité à 31% par feature isolée (`CLAUDE.md`, §13.1) — ce bruit de batching s'ajoute à un bruit déjà accepté et plus grand, pas une nouvelle classe de risque. Code conservé (pas de régression comme G1), caveat documenté au lieu d'un revert. |
| 2 | Troncature `layers[:LAYER]`/`layers[:LAYER+1]` + hook direct, jamais `output_hidden_states` | `saev5.py` | 1,4–2,0× sur le forward P1 | 3 h | **fait, vérifié GPU** (job 44540, `torch.equal`=True, écart=0,0, 1,98×, -13 Go VRAM) — une première variante (`output_hidden_states=True` + troncature) a échoué à l'équivalence et a été identifiée avant déploiement, cf. §2.2 |
| 3 | Fragments shardés + écriture asynchrone | `fragment_store.py`, `saev5.py:920` | 2–2,5× sur le wall-clock d'extraction | 1 j | **partiellement fait** : écriture en arrière-plan faite et testée (`AsyncFragmentWriter`, CPU, `tests/test_async_fragment_writer.py`), avec discipline `flush()` avant tout checkpoint de reprise (§2.3). Sharding (1 fichier pour 1 000 docs) non touché — changerait le format sur disque, tenu à l'écart tant qu'un run de référence n'est pas relancé avec le nouveau format |
| 4 | Reprise incrémentale (P1 et P2) | `saev5.py:783`, `phrase_sae.py:117` | supprime le risque de 20 h perdues | 1 j | à faire |
| 5 | `torch_dtype=bfloat16` pour F2LLM | `phrase_sae.py:126` | jusqu'à ~10× sur l'extraction P2 | 5 min | **fait** (même correctif appliqué aussi à `latent_terms.py:load_f2llm`, hors périmètre initial, même bug) |
| 6 | Stocker `e` en int8 au lieu de `x` | `saev5.py`, `fragment_store.py` | disque 1,6 To → 0,4 To ; entraînement ×3,4 | 2 j | à faire — la fidélité SAE Boost (§1.3, encodeur sur `x`) qui forçait ce doublon est corrigée, ce n'est plus qu'un choix de perf, tenu à l'écart tant que le run en cours n'est pas terminé |
| 7 | `BATCH_SIZE` 1024 → 16 384 + `.item()` hors boucle | `sae_shared.py:178` | 5–15× sur l'entraînement extra | 3 h (+ ablation BatchTopK) | **partiellement fait** : `.item()`/`float()` par step → accumulation GPU + 1 sync/époque, fait et testé (CPU, valeurs identiques bit-à-bit à l'ancien code). `BATCH_SIZE` paramétré (`BATCH_SIZE_EXTRA`, `src/config.py`, défaut 1024 inchangé) plutôt que codé en dur, pour permettre l'ablation sans changer le défaut à l'aveugle. Ablation **lancée** (job 44620, `slurm/pipeline_runs/run_validation_100k_layer24_v4_batchsize16384.slurm`, réplique de 44572 avec `BATCH_SIZE_EXTRA=16384`) — comparaison prévue contre 44572 une fois les deux terminés : rho_sae, fve_pretrained, dead_pct, L0, temps d'entraînement extra. n=1 seed par bras, signal directionnel seulement à ce stade |
| 8 | Batcher le ré-encodage (64–256 docs) | `saev5.py:1069` | 5–10× sur cette passe | 4 h | à faire |
| 9 | `LogisticRegression` sur CSR sparse | `metrics.py` | 100–1000× sur la sonde | 1 h | **fait et testé** (CPU) |
| 10 | Dégroupage O(n log n) de `phrase_to_doc` | `saev5.py:1512` | supprime un O(n²) | 30 min | **fait et testé** (CPU, `group_indices_by_doc`) |
| 11 | TF32/SDPA/`inference_mode`/`expandable_segments` | global | 3–8 % cumulés | 1 h | partiellement fait (`set_float32_matmul_precision` posé dans `phrase_sae.py`/`latent_terms.py`) ; SDPA/`inference_mode`/`expandable_segments` pas encore |
| 12 | Exclure le filler du ré-encodage **et** de l'extraction | `saev5.py`, `src/data/preparation.py::build_reencode_targets`/`is_filler_document` | jusqu'à ×13 sur le ré-encodage, gain GPU+disque proportionné sur l'extraction (92% du corpus sur le run de référence, jamais relu en aval) | 2 h + 2 h | **fait et testé** (CPU) pour les deux passes — ré-encodage (§2.5) trouvé en observant job 44211 tourner dedans en direct, extraction (G7, §2.2) fait ensuite sur confirmation explicite que le premier correctif était sûr. Comparaison contrôlée à trois lancée pour isoler la contribution de chaque correctif (jobs 44560 v1 sans correctif / 44571 v2 ré-encodage seul / 44572 v3 les deux) |
| 13 | Amortir les écritures aléatoires du réservoir (buffer trié) | `saev5.py` (`_flush_pending_reservoir_writes`) | supprime le write-amplification aléatoire sur le memmap 768 Go en phase 2 de l'échantillonnage par réservoir | 2 h | **fait et testé** (CPU, `tests/test_reservoir_write_batching.py`) — équivalence bit-exacte au chemin d'écriture immédiate, y compris résolution des collisions, vérifiée avant déploiement |

Bout à bout sur un run type : **20 h 57 d'extraction → estimation 5–7 h**, entraînement extra
divisé par ~5, juge divisé par ~10, empreinte disque divisée par 4. Les items 1, 5 et 10 sont
faisables aujourd'hui et cumulent déjà un facteur significatif.

---

## 3. Hygiène du dépôt — vers un dépôt auto-suffisant

Critère retenu : un fichier reste si (a) il fait tourner la pipeline, (b) il est nécessaire pour
reproduire un chiffre du rapport, ou (c) il sert la maintenance récurrente. Sinon il sort : sa
conclusion vit déjà, figée, dans `RESULTS_TESTS.md §N`.

### 3.1 À supprimer ou archiver (≈ 55 fichiers)

**Audits forensiques à conclusion figée — 15 scripts + ~20 `.slurm` + 8 JSON.**
`scripts/audit_2026_08_{b24_inspect_pollution, b26_propagate_fidelity, b26_round2_fix,
b27_random_control, bf16_fp32_diagnostic_v2, delta_ce, delta_ce_v2,
e9_size_matched_embedding_compare, frozen_decoder_scale_fix, frozen_decoder_scalefix_rejudge,
mcnemar_and_lengthbias, palier1_batch, random_init_trained_decoder, soft_frozen_decoder,
soft_frozen_decoder_scale1, uniform_hardneg_rejudge}.py`.
Ces scripts ont répondu à une question, la réponse est dans `RESULTS_TESTS.md`, et ils ne
re-tourneront jamais (ils dépendent de caches d'un run précis, souvent purgé). Les garder sur le
chemin actif oblige quiconque reprend le dépôt à décider s'ils sont vivants.
→ `git rm` ; l'historique git les conserve, et `RESULTS_TESTS.md` reste la référence citable.
**Deux exceptions à promouvoir plutôt qu'à supprimer** :
- `audit_2026_08_extraction_batch_size_sweep.py` → `benchmarks/` : outil de calibration à
  relancer à chaque changement de GPU, de modèle ou de couche. C'est de l'outillage, pas un audit.
- `audit_2026_08_bf16_fp32_diagnostic.py` → `tests/test_dtype_overflow.py` : « fp16 déborde sur
  les activations Gemma » est un **invariant de conception**, pas un résultat d'expérience. Il doit
  être testé à chaque commit (version CPU sur tenseurs synthétiques de norme 1,2e5), pas archivé.

**Les 8 `docs/audit_*_results.json`** : `docs/` est de la référence technique, pas un dossier de
résultats. Les déplacer dans `results_archive/` (gitignoré, ou versionné à part) ou les supprimer,
`RESULTS_TESTS.md` portant déjà les chiffres.

**`docs/PDF_APPENDICES_EXTRACT.md` (1 074 lignes) — à traiter en priorité.** C'est la reproduction
**verbatim** des appendices A–M d'un article sous copyright (prompts complets, tableaux), dans un
dépôt destiné à devenir public et à accompagner une soumission EMNLP/EACL. Double problème : risque
juridique, et le dépôt n'en a pas besoin pour être auto-suffisant (le PDF est cité et accessible).
→ Réduire à une **table de correspondance** « §/App. du papier → décision du dépôt → fichier »,
sans texte reproduit. Le contenu analytique de `INTERP_EMBED_COVERAGE.md` (qui, lui, est du travail
original et de grande qualité) est conservé.

**`scripts/compare_to_frozen_benchmark.py`** : 0 référence dans tout le dépôt, alors qu'il lit
`benchmarks/frozen_baseline_v10_emails_main.json` (versionné). Or c'est le seul mécanisme de
non-régression sur le domaine général, exigé par la Table 2 de SAE Boost (§1.3). Trancher :
soit le raccrocher à un `.slurm` et à une section §N, soit supprimer le script **et** le JSON.
Ne pas laisser un orphelin qui ressemble à un garde-fou actif.

**`test_massive_acts.py` et `test_chargement_sae.py` à la racine** : deux `.py` orphelins à la
racine d'un dépôt par ailleurs bien structuré. Ce sont des smoke-tests → `tests/` (s'ils sont
CPU-only) ou `slurm/validation/` (s'ils ont besoin d'un GPU).

**`src/sae.egg-info/`** (5 fichiers) : artefact de `pip install -e .` versionné par erreur.
→ `git rm -r` + `.gitignore`.

**`CHANGELOG.md`** (56 lignes, en croissance — +9 lignes depuis cet audit, dont une entrée pour la
correction Latent Terms de §1.4) : redondant avec `git log` et avec la numérotation append-only de
`RESULTS_TESTS.md`. Trois niveaux d'historique pour un projet mono-contributeur, c'est un de trop.

**`logs/README.md`** (10 lignes) : à fusionner dans `docs/ops.md`, qui traite déjà des logs.

**Les `.slurm`, désormais 94 (pas 63 — +31 depuis ce clone, la dérive va dans le mauvais sens)** :
`run_ablation_k_extra5.slurm`, `..._seed7.slurm`, `..._seed99.slurm`, `run_ablation_volume_25m*.slurm`
(×3), `run_audit_c2_rejudge_seed{7,99}.slurm`, `run_audit_judge_sep_seed{7,99}.slurm` — ce sont des
copies d'un même script différant par une variable d'environnement, et le patron s'est reproduit
(`slurm/validation/` est passé à 32 fichiers, `slurm/pipeline_runs/` à 38). → un `slurm/run.slurm`
générique paramétré + `slurm/configs/*.env` sourcés. **94 fichiers → ~15**, et les longs commentaires
de journal de bord (`run_sae_v14_main.slurm` en compte 60 lignes ; les nouveaux `.slurm` de §1.4
en ajoutent d'autres, cf. `run_latent_retrieval_precision_eval.slurm`) migrent vers
`RESULTS_TESTS.md`, à qui ils appartiennent — ou, pour un diagnostic d'incident encore actif comme
celui de §1.4, restent en commentaire jusqu'à résolution puis migrent.

**Scripts rétroactifs à archiver après vérification** : `c2_original_only_rejudge.py`,
`relabel_diff_csvs.py`, `augmentation_lexical_leakage_audit.py` — rejugent un cache précis. Garder
uniquement s'ils sont cités par `report/`.

### 3.2 À garder sans discussion

- Tout `src/**` (aucun module mort ; l'arborescence est propre).
- Les 18 tests de `tests/` : c'est le meilleur actif du dépôt, et le hook post-edit est une bonne
  pratique. À étendre plutôt qu'à élaguer.
- `docs/{architecture, ops, evaluation_protocol, references, experiments}.md` : cinq documents
  distincts et non redondants.
- `docs/INTERP_EMBED_COVERAGE.md` : travail d'audit original de haute qualité, cœur de la valeur
  méthodologique.
- `RESULTS_TESTS.md` : append-only, cité par `§N` depuis le rapport. Intouchable.
- `report/**`, `README.md`, `CLAUDE.md`, `download_sae.py`.
- Les **13 scripts cités par `report/`** : `judge_robustness_check`, `intent_urgency_probe`,
  `explanation_{fidelity,plausibility}_test`, `build_report`, `steering_fidelity_test`,
  `saelens_numeric_comparison`, `run_augmentation`, `multilingual_judge_bias_test`,
  `latent_retrieval_precision_eval`, `contrastive_labeling_test`, `consolidate_evaluation_report`,
  `compute_interesting_correlations_retro`. Ils sont la reproductibilité du rapport.
- `scripts/{baseline_gemmascope, core_vs_extension_ablation, generate_diagnostic_plots, check_docs}.py`.
- Les scripts d'interopérabilité interp-embed (`validate_interp_embed_tutorial`,
  `email_interp_embed_encode_test`, `imdb_genre_diffing_test`) — vivants tant que l'intégration
  est en cours ; à regrouper sous `scripts/interop/`.

### 3.3 Commentaires et désynchronisations dans le code

`src/config.py` est à **59 % de lignes de commentaire** (85 lignes sur 143) ; `saev5.py` en compte
236. Beaucoup sont d'excellents commentaires de conception. Mais une part notable est du **récit de
session** — le travers que `CLAUDE.md` interdit explicitement dans la documentation `.md` et qu'il
n'interdit pas dans le code (lacune, cf. §4.3) :
« jamais mesuré contre une valeur plus grande », « audit 2026-08 round 3, §6.8 »,
« désynchro EMB_MODEL, corrigée », « Confirmé par isolation empirique ».
Règle proposée : une contrainte encore active se formule comme une règle ; sa provenance se cite
par `§N`, pas par sa narration.

**Trois désynchronisations commentaire↔code, vérifiées, à corriger :**
1. `saev5.py:132` — « fp16 par défaut en local ». Faux : `config.py` fixe `DTYPE="bf16"` par
   défaut, et `CLAUDE.md` impose « bf16 partout, y compris en local ». Commentaire dangereux car
   il contredit l'invariant de sûreté numérique du projet.
2. `saev5.py:892` — « σ-clip intra-batch (stats sur B docs) », alors que
   `activations.py::norm_outlier_mask` est **explicitement intra-document** (sa docstring justifie
   longuement ce choix contre l'intra-batch). Les deux commentaires se contredisent frontalement.
3. `config.py:103` — « LAYER par défaut vient du preset MODEL_SIZE (24 pour 12b) ». Le preset vaut
   **31**. Même erreur dans `run_sae_v14_main.slurm` (« SAE_ID laissé au défaut (65k, pas 16k) »
   alors que le défaut est 16k). Les commentaires décrivent une configuration antérieure ; un
   lecteur qui les croit reproduira le mauvais run.
4. `fragment_store.py:12` — docstring : « `vals`: float16 ». Le code écrit `torch.float32`
   (ligne 57). Impact : le double du stockage annoncé.

---

## 4. `CLAUDE.md` — règles à garder, à contester, à ajouter

### 4.1 Règles à conserver telles quelles

Diffs BEFORE/AFTER exacts ; deux niveaux de test (`tests/` vs `scripts/*_audit.py`) ;
`stats.py` comme module unique ; `RESULTS_TESTS.md` append-only ; interdiction du trailer
`Co-Authored-By` ; les deux pièges PyTorch (`@torch.no_grad()` sur générateur, `logits_to_keep=1`)
qui sont exacts et coûteux ; la checklist de diagnostic en 6 étapes, qui est excellente et
sous-utilisée.

### 4.2 Règles à contester

**(a) « `LogisticRegression(solver="lbfgs")` … est compute-bound … partir directement de
`--cpus-per-task=32` ».**
Le diagnostic est juste, **le remède est le mauvais**. Vos activations SAE sont à ~99,9 % de zéros.
`LogisticRegression` accepte une `scipy.sparse.csr_matrix` en entrée, et `lbfgs` comme `liblinear`
la gèrent nativement. Passer de dense fp64 [n, 17 408] à CSR ne coûte pas 32 cœurs : cela réduit le
travail de **deux à trois ordres de grandeur**, et supprime au passage l'upcast fp64 de sklearn
(`metrics.py` construit `np.concatenate` de tableaux fp32 que sklearn recopie en fp64).
Ajoutez `cross_val_score(n_jobs=5)` pour les 5 plis, aujourd'hui séquentiels.
→ Réécrire la règle en : « les activations SAE se passent en CSR sparse aux estimateurs sklearn ;
32 cœurs ne sont un correctif que si le profil montre un coût réellement dense ». La règle actuelle
enseigne à surdimensionner un job plutôt qu'à corriger le code, et sa formulation « observé deux
fois » suggère qu'elle a été écrite depuis le symptôme.

**(b) « Ne jamais exécuter de code Python sur le nœud frontal, y compris pour un test CPU trivial
sur cache ».**
Le principe est bon, la formulation absolue est contre-productive : elle rend impossible tout
`--dry-run` de validation de configuration (vérifier qu'un chemin existe, qu'une clé de cache est
cohérente, qu'un `.slurm` n'a pas de faute de frappe) et pousse à découvrir les erreurs trivialement
détectables **après 20 h de file d'attente**. Compte tenu du §2.3 (aucune reprise), le coût d'une
faute de configuration est maximal.
→ Reformuler : « aucun calcul sur le frontal. Une validation de configuration bornée
(< 5 s CPU, < 500 Mo RSS, aucune lecture de tenseur) est autorisée et recommandée avant tout
`sbatch` ». Ajouter un `scripts/preflight.py` qui matérialise cette borne.

**(c) « bf16 partout, y compris en local ».**
Correcte pour le residual stream de Gemma (les activations massives à 1,2e5 débordent fp16). Mais
elle est appliquée aveuglément à `PhraseLevelSAE` (P2, F2LLM), dont les embeddings sont
**L2-normalisés** (`phrase_sae.py:151`) et donc bornés à 1,0 — aucun risque d'overflow, et bf16 y
coûte 8 bits de mantisse pour rien. `compute_sae_metrics` caste d'ailleurs en bf16 une entrée dont
le SAE est fp32, ce qui dégrade les métriques publiées.
→ Reformuler : « bf16 obligatoire sur les activations du residual stream de Gemma-3 (activations
massives). Pipeline 2 : entrées normalisées, fp32 pour le SAE, bf16 pour le seul backbone. »
Cette règle vaut aussi comme rappel : `AutoModel.from_pretrained` **sans `torch_dtype` charge en
fp32** — c'était le cas à `phrase_sae.py:126` (et, hors périmètre initial de cet audit, au même
endroit dans `latent_terms.py::load_f2llm`) ; corrigé dans les deux fichiers (`torch_dtype=
torch.bfloat16` explicite), `compute_sae_metrics` corrigé pour ne plus caster son entrée en bf16
avant le SAE fp32.

**(d) « Rédiger au présent, sans numéro de version interne ».**
Excellente pour les `.md`. Mais `SAVE_DIR="./results_v14_main/"`, `run_sae_v12_scaled.slurm`,
`p2_sae_dim...` : le versionnage interne prolifère dans les **noms de fichiers et de répertoires**,
que la règle ne couvre pas. `check_docs.py` ne vérifie que le Markdown.
→ Étendre au nommage des artefacts, ou assumer explicitement que les répertoires de run sont
horodatés/versionnés et que seule la prose ne l'est pas.

### 4.3 Règles à ajouter

**R1 — Reprise obligatoire pour toute boucle de plus d'une heure. — Implémentée.**
« Toute boucle dont le coût dépasse ~1 h GPU écrit son état de progression de façon atomique et
reprend depuis cet état. Le critère de reprise est *quel est le prochain élément non traité*,
jamais *le run est-il complet*. Un compteur d'échantillonnage (réservoir) fait partie de l'état à
persister. » Mécanisme partagé : `src/storage/checkpoint.py`, câblé dans les trois boucles
concernées (extraction P1, ré-encodage ExtendedSAE, extraction F2LLM P2) — cf. §2.3 pour le
détail et les tests. Cette règle peut maintenant être vérifiée mécaniquement plutôt
qu'espérée : toute nouvelle boucle longue qui n'importe pas `src/storage/checkpoint.py` est
suspecte par défaut.

**R2 — Les commentaires de code suivent la règle éditoriale des `.md`.**
« Une contrainte encore active se formule comme une règle ; sa provenance se cite par `§N` de
`RESULTS_TESTS.md`. Pas de récit de session, pas de "corrigé cette session", pas de "jamais mesuré".
Un commentaire qui décrit un défaut est faux dès que le défaut change : il cite le défaut par son
nom de variable, pas par sa valeur. » Étendre `check_docs.py` aux docstrings et commentaires `#`
de `src/` — le harnais existe déjà, il suffit d'élargir `TARGET_FILES`.

**R3 — Budget de performance dans le protocole d'évaluation.**
« Tout script produisant une section `§N` reporte `tokens/s`, `docs/s` ou `steps/s` et le pic VRAM
dans son JSON de résultats. » Aujourd'hui aucun chiffre de débit n'est traçable en dehors du sweep
de batch, ce qui rend impossible de détecter une régression de performance entre deux runs — et
c'est exactement ce que vous cherchez à mesurer.

**R4 — Aucune structure O(n²) sur `n_docs` ou `d_sae` sans plafond explicite.**
« Toute allocation ou boucle quadratique en nombre de documents ou en largeur de dictionnaire
porte un plafond codé et un `assert` explicite. » Motivée par `cooccurrence.py:55` (plafond absent
là où le chemin voisin en a un), `SpectralClustering` sur n×n, et la trajectoire annoncée vers 65k
puis 262k de largeur.

**R5 — Toute clé de cache est dérivée mécaniquement, pas rédigée à la main.**
`CLAUDE.md` énonce déjà le principe, mais le laisse à la discipline du rédacteur — et il est
violé à `saev5.py:1437` (P2, `EMB_MODEL` absent de la clé) autant qu'à
`load_or_train_extended_sae`. Remède : une fonction unique
`cache_key(**params) -> str` qui hache un dict de paramètres, et l'interdiction des f-strings de
clé ad hoc. Le principe seul n'a pas suffi ; il faut la primitive.

**R6 — Écart au papier = section documentée, jamais un choix implicite.**
« Toute divergence par rapport à l'équation ou au protocole d'un papier cité fait l'objet d'une
entrée dans `docs/references.md` : équation du papier, équation implémentée, justification,
conséquence sur la comparabilité des chiffres. » Motivée par l'écart d'encodage SAE Boost (§1.3),
la silhouette maintenue contre la note 4 d'interp-embed, l'absence de normalisation au 90ᵉ
percentile en retrieval, et l'omission d'AuxK dans la Table 4 de Latent Terms (assumée mais non
listée par le papier, cf. §1.4). Aucun de ces quatre n'est faux en soi ; ce qui est problématique
est qu'ils soient invisibles. Note positive : le docstring de tête de `latent_terms.py` documente
déjà volontairement ses écarts d'échelle au papier (D_SAE/K_SPARSE, budget de tokens, graine
unique) avec justification — exactement la discipline que R6 demande, à généraliser au reste du
dépôt plutôt qu'à réinventer.
