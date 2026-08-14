# Audit scientifique et méthodologique du dépôt — instructions d'exécution

> Document destiné à Claude Code, travaillant dans le dépôt `GregoirePelletier/SAE`.
> Il contient (a) une méthode de travail, (b) une liste de constats déjà établis par
> lecture du code au commit `504a876`, à **vérifier puis instruire**, (c) les axes
> d'investigation à mener au-delà de cette liste.
>
> **Mise à jour du 13/08 (1)** : les articles de référence ont été lus. Les sections A.1
> (SAE Boost), A.3 (Sanity Checks), A.4 (Latent Terms) et A.7 (SPLARE) contiennent
> désormais des écarts **chiffrés** et non plus des questions ; B.1 a été reformulé à la
> baisse, B.19 et B.20 sont nouveaux et l'un des deux est un P0.
>
> **Mise à jour du 13/08 (2)** : le document est désormais organisé en **paliers
> d'exécution stricts** (§0.4) — l'ordre dans lequel les items doivent être traités, pas
> seulement leur gravité. Un nouvel axe, **Axe F (§4)**, propose un pivot stratégique :
> reproduire fidèlement deux méthodes de référence (Latent Terms, toolkit de Jiang et
> al.) sur le corpus EDF, comme baseline externe validée. Les expériences les plus
> lourdes (B.1 en régime surcomplet, B.4 en croisement complet, B.19 à l'échelle réelle
> de la littérature) restent dans ce document mais sont explicitement reléguées au
> **Palier 3 — à traiter en tout dernier**, après absolument tout le reste, F compris.
>
> Rien dans ce document n'est à prendre pour argent comptant : chaque constat est une
> **hypothèse falsifiable**, formulée par lecture statique, sans exécution ni accès aux
> PDF des articles. Le travail attendu est de la confirmer ou de l'infirmer, preuve à
> l'appui.

---

## 0. Périmètre, méthode et livrable

### 0.1 Ce que tu dois produire

Un fichier `docs/AUDIT_2026-08.md` structuré comme suit, **et rien d'autre tant que je
n'ai pas relu** (aucune correction de code appliquée sans validation explicite) :

```markdown
| ID | Constat | Localisation | Statut | Gravité | Palier | Effort | Affirmations du rapport touchées |
```

- **Statut** : `CONFIRMÉ` / `INFIRMÉ` / `PARTIEL` / `INDÉTERMINÉ (raison)`.
- **Gravité** : `P0` invalide ou fragilise un résultat publié dans le rapport ;
  `P1` biaise une mesure sans invalider la conclusion ; `P2` hygiène, perf, style.
- **Palier** : `0` (audit/diagnostic) / `1` (correction ou recalcul CPU/cache) /
  `2` (GPU modéré, y compris le pivot F) / `3` (GPU lourd, **en tout dernier**) — voir
  §0.4. La gravité dit combien un constat compte ; le palier dit **quand** le traiter.
  Un P0 peut être Palier 1 (ex. B.6, corrigeable en une recompute cache) et un P1 peut
  être Palier 3 (ex. B.1 en régime surcomplet, GPU lourd pour un gain incertain) — ne
  pas confondre les deux axes.
- **Effort** : `S` (<1 h, CPU) / `M` (quelques heures) / `L` (rerun GPU).
- Chaque ligne renvoie à une sous-section détaillée : preuve (extrait de code + numéro
  de ligne), raisonnement, test de falsification exécuté, correction proposée,
  conséquence sur le rapport.

Ajoute une section finale **« Constats non listés dans le prompt »** : c'est la partie
la plus utile, celle où tu dois trouver ce qui m'a échappé.

### 0.2 Règles de travail

1. **Vérifie avant d'affirmer.** Pour chaque hypothèse : localise le code exact, lis-le
   en entier (pas seulement la fonction, aussi ses appelants), et quand c'est possible
   écris un test falsifiable qui tranche. Un constat sans preuve reproductible reste
   `INDÉTERMINÉ`.
2. **Ne corrige rien pour l'instant.** Le but est le diagnostic. Une correction hâtive
   sur un pipeline dont les résultats sont déjà rédigés dans un rapport crée une
   incohérence code/rapport pire que le défaut initial.
3. **Distingue trois choses** que ce dépôt confond souvent : un *bug* (le code ne fait
   pas ce que le commentaire dit), une *divergence assumée* avec la littérature (le code
   fait autre chose, délibérément), et une *erreur de conception scientifique* (le code
   fait exactement ce qui est écrit, et ce qui est écrit ne mesure pas ce qu'on croit).
   La troisième catégorie est celle qui compte ici.
4. **Le test décisif est toujours : « ce protocole peut-il produire un résultat
   négatif ? »** Un test qui ne peut pas échouer ne mesure rien. Applique
   systématiquement cette question à chaque protocole d'évaluation du dépôt.
5. **Beaucoup de vérifications ne nécessitent pas de GPU** : les activations, labels et
   JSON de résultats des runs passés sont en cache (`results_*/cache/`). Privilégie
   toujours une vérification rétroactive sur cache à un rerun.
6. **Cite tes sources.** Quand tu compares à un article, cite la section/équation
   précise, et si le PDF n'est pas disponible localement (`pdf/` est gitignoré), dis-le
   et cherche la version arXiv en ligne plutôt que de reconstituer de mémoire.

### 0.3 Articles de référence — **lus, écarts déjà chiffrés**

Les PDF sont disponibles (`pdf/`, non versionné). Quatre d'entre eux ont été lus et les
écarts avec le code sont chiffrés dans les sections A.1 à A.7 : **SAE Boost**,
**Sanity Checks**, **Latent Terms**, **Interpretable Embeddings**, plus **SPLARE**
(`pdf/Naver.pdf`, jamais cité dans le rapport, cf. A.7). Ta tâche sur ces cinq-là n'est
plus de découvrir les écarts mais de **les vérifier dans le code** et d'en mesurer
l'effet.

Restent à lire et à confronter au code, personne ne l'ayant fait : `jumpRELU.pdf`,
`BatchTopK.pdf` (vérifier `src/sae/batch.py` équation par équation, en particulier la
définition de θ et sa mise à jour par EMA), `Matryoshka.pdf`, `SurveySAE.pdf`,
`DisentanglingDenseEmbeddingswithSAE.pdf`,
`DecodingDenseEmbSAEforInterpandDiscretizDenseRetrieval.pdf`,
`InterpretandControlDenseRetrievalwithSparseLatentFeatures.pdf` (ces trois derniers sont
directement pertinents pour le Pipeline 2 et ne sont cités qu'en bloc dans la
bibliographie), `UnveilingDecisionMakinginLLMsforTextClassification.pdf`,
`SparseAutoencodersforHypothesisGeneration.pdf`.

Pour la métrique de référence de Bills et al. et le harness SAEBench, il faudra chercher
en ligne : ils ne sont pas dans `pdf/`.

Le dépôt communautaire `x-tabdeveloping/latent_terms` et le harness `EleutherAI/delphi`
(ex-`sae-auto-interp`) sont mentionnés plus bas comme réutilisations possibles : vérifie
leur existence et leur API réelle avant de recommander quoi que ce soit.

### 0.4 Paliers d'exécution — ordre strict

Le calendrier est contraint (entretien de PhD fin août). Ce document contient des
dizaines d'items d'ampleur très différente ; les traiter dans l'ordre de numérotation
ou de gravité serait une erreur budgétaire. **Voici l'ordre réel à suivre, et il est
strict : ne pas entamer un palier tant que le précédent n'est pas terminé, sauf
instruction contraire explicite de ma part.**

- **Palier 0 — Diagnostic (gratuit, CPU, en premier).** Tout l'audit statique : lecture
  du code, production de `docs/AUDIT_2026-08.md`, croisement des scripts SLURM avec les
  `mtime` des caches (B.8). Rien n'est corrigé à ce stade. C'est un préalable à tout le
  reste : B.8 en particulier conditionne si B.4 (Palier 3) sera même exploitable un jour.
- **Palier 1 — Corrections à coût quasi nul (CPU ou cache, quelques heures à un ou deux
  jours au total).** Tout ce qui se recalcule sur des activations, labels ou JSON déjà
  en cache, sans relancer de job GPU : B.5, B.6, B.9, B.10 (partiel), B.11 (le fix de
  split, pas le retrain), B.13, B.14, B.15, B.16, B.17, B.18, l'axe C (hygiène) en
  entier, et les recalculs de B.2/B.3 qui ne nécessitent pas de relancer le juge sur du
  texte nouveau (statistiques sur des jugements déjà produits).
- **Palier 2 — GPU modéré, sur infrastructure et cache déjà en place (le gros du travail
  utile, plusieurs jours).** Nouvelles passes d'inférence ou d'entraînement léger,
  toujours à l'échelle de ce que le dépôt a déjà démontré savoir faire à faible coût :
  A.1 (baseline Extended SAE random init), A.3 (Soft-Frozen Decoder), A.5 (vérification
  FVE des hooks), B.2/B.3 (rejugement sur échantillon uniforme + négatif dur + portage
  partiel du protocole Korznikov), B.7 (fidélité par re-décodage), B.20 (ΔCE), et
  surtout **l'Axe F en entier (§4)** — le pivot vers une reproduction fidèle comme
  baseline externe, qui est la meilleure dépense de temps disponible sur ce palier.
- **Palier 3 — GPU lourd, à haut risque calendaire (en tout dernier, seulement si le
  temps restant le permet après Palier 2 terminé).** B.1 en régime réellement surcomplet
  couplé à B.19 (plan croisé `D_EXTRA × volume`), B.4 en croisement complet
  extracteur × juge sur les trois échelles, B.19 à l'échelle réelle de la littérature
  (≥100 M tokens, y compris le run 200 M dont le blocage racine — allocation RAM
  anonyme du réservoir — est désormais corrigé mais pas encore revalidé à cette
  échelle). Ces expériences
  restent dans ce document — elles ne sont pas écartées, elles répondent aux questions
  scientifiquement les plus importantes de l'audit — mais elles sont aussi les plus
  chères, les plus risquées (disque, jobs déjà en échec) et celles dont le résultat est
  le moins garanti. **Le rapport doit pouvoir être fini sans elles** : à défaut de temps,
  elles se documentent comme limites assumées au chapitre 4 (« l'audit a identifié X ;
  le retester à l'échelle correcte est laissé en perspective, faute de temps/disque »),
  ce qui est une position défendable et nettement plus sûre qu'un rerun de dernière
  minute qui échoue ou contamine un cache la veille d'un dépôt.

Autrement dit : **Palier 0 → 1 → 2 (F inclus) → seulement alors, si le temps le permet,
Palier 3.** Ne jamais lancer un item de Palier 3 « en parallèle pour gagner du temps » —
c'est exactement le piège budgétaire à éviter, et c'est le disque partagé qui en fait
les frais en premier.

---

## 1. Axe A — Conformité à la littérature

Pour chaque point : dire ce que fait le code, ce que fait le papier, si l'écart est
délibéré, documenté, et scientifiquement défendable. Le livrable de cet axe est un
tableau `docs/references.md` **remis à jour et honnête** — pas une liste de « conforme ».

### A.1 SAE Boost (Koriagin et al., COLM 2025) — **article lu, écarts chiffrés** (Palier 2)

Setup de référence (§4.1 du papier) : backbone **Llama-3.1-8B** (d_model = 4096), SAE
core Llama-Scope `L24R-8x` (layer 24), BatchTopK avec **k = 50 pour le SAE core et k = 5
pour le SAE résiduel**, conversion BatchTopK → JumpReLU par calibration de seuil sur les
données d'entraînement, dictionnaire résiduel de **1024**, **1 milliard de tokens par
domaine**, métriques **EV / LLM cross-entropy / L0**, baselines : Extended SAE (init
most-active), Extended SAE (init aléatoire), SAE Stitching, fine-tuning complet.

Écarts confirmés avec le dépôt, par ordre de gravité :

| Paramètre | SAE Boost | Ce dépôt | Commentaire |
|---|---|---|---|
| Tokens d'entraînement du résiduel | **1 G par domaine** ; les auteurs écrivent explicitement qu'un résiduel entraîné sur **< 100 M tokens dégrade le domaine général** et que **< 50 M est sous-entraîné** | `N_TOKENS_EXTRA_TRAIN = 500 000` par défaut ; balayage 100 k / 500 k / 2 M (plat) ; **25 M donne 54,0 % vs 45,3 % (+8,7 pts, non significatif seul, réplication multi-graines déjà engagée mais bloquée par un incident cluster)** ; le blocage du run 200 M est corrigé au niveau code, pas encore revalidé | **P1, voir B.19 — reformulé, moins alarmant qu'il n'y paraît** |
| Sparsité du résiduel | k = 5 | `K_EXTRA = 32` par défaut | facteur 6,4 ; `K_EXTRA=5` testé tardivement (§25) mais jamais promu en défaut |
| Taille du dictionnaire résiduel | 1024 pour d_model 4096 (ratio 0,25) | 1024 pour d_model 3840 (ratio 0,27) | **conforme** — voir la reformulation de B.1 |
| Métriques | EV, **ΔCE du LLM**, L0 | FVE, NMSE, L0 | **ΔCE jamais mesurée**, voir B.20 |
| Baselines de domain adaptation | 4 baselines | aucune | Extended SAE (init aléatoire) est à un `if` près déjà implémentable ici |
| Sparsité du core | BatchTopK k=50 (SAE entraîné par eux) | JumpReLU GemmaScope préentraîné | écart assumé, à documenter |

À faire : rejouer au moins la baseline « Extended SAE (init aléatoire) » — dictionnaire
étendu sur le SAE core plutôt que résiduel, même nombre de features — qui est la
comparaison que le papier considère comme la plus proche concurrente.

### A.2 Interpretable Embeddings / interp_embed (Jiang, Sun et al.) (Palier 1 pour le constat, Palier 2 pour la comparaison → voir F.2)

- `external/interp_embed` est un **submodule vide**, et `sae_shared.py` fait un
  `sys.path.insert` vers un chemin qui n'existe pas, dans un `try/except ImportError`
  silencieux. Le dépôt n'a donc jamais été exécuté : toute comparaison « à interp_embed »
  est une comparaison à la *lecture du papier*, pas au code. Vérifie et dis-le
  explicitement, y compris dans le rapport (la règle n°1 du projet est « ne pas
  réimplémenter sans justification documentée » : ici on a réimplémenté sans avoir
  jamais installé la référence).
- **Article lu** : leur SAE a `d_SAE = 65 536` sur les activations d'un modèle
  d'embedding fondé sur Llama-3.3-70B (facteur d'expansion ×8), et ils **max-poolent bien
  les activations sur les tokens** — le max-pooling du Pipeline 1 est donc conforme à
  *cette* référence, même s'il est contredit par Latent Terms (A.4). Leur retrieval par
  propriétés se compare à des baselines fortes (BM25 + expansion de requête par LLM,
  embeddings OpenAI/Gemini + expansion), pas à un TF-IDF nu.
- Leur toolkit travaille sur des **embeddings de document** (un vecteur par document).
  Nous, sur des activations token-level max-poolées. Leurs protocoles de retrieval,
  clustering ciblé et corrélations sont-ils transposables tels quels à un vecteur
  max-poolé ? Ce n'est pas évident : le max-pooling détruit la propriété de
  « composition additive » sur laquelle repose une partie de leurs méthodes.
- Leur labellisation est **contrastive directe** (top-activating vs bottom-activating) ;
  notre protocole principal est odd-one-out. Un test contrastif existe
  (`scripts/contrastive_labeling_test.py`) mais n'est pas le protocole par défaut.
  Vérifie si les deux ont jamais été comparés sur le **même** échantillon de features
  avec appariement (McNemar), et pas seulement en taux agrégés.

### A.3 Sanity Checks (Korznikov et al.) — **article lu, protocole d'auto-interp très différent** (Palier 1 pour le constat, Palier 2 pour la reproduction)

Leur protocole AutoInterp (§ correspondant à leur Figure 5) :
- **200 latents tirés au hasard** parmi les latents vivants (fréquence ≥ 1e-6), pour
  chaque architecture et chaque baseline ;
- jusqu'à 15 séquences de plus forte activation → un LLM produit une **description** ;
- jeu de test **tenu à l'écart** de 100 séquences par latent : **50 activantes à
  intensités variées + 50 non activantes tirées au hasard** ;
- un **second** LLM prédit, à partir de la seule description, si chaque séquence active le
  latent ; le score est l'exactitude de cette classification binaire — **hasard = 0,50** ;
- résultats : BatchTopK entraîné 0,90 ; Soft-Frozen Decoder 0,88 ; Frozen Encoder ≈ 30 %
  de latents encore fortement interprétables.

Conséquences pour ce dépôt :
1. Notre 45,3 % et leur 0,90 **ne sont pas comparables** : tâche différente (intrus parmi
   10 vs classification binaire équilibrée), hasard différent (0,10 vs 0,50), sélection
   différente (top-N par magnitude vs 200 aléatoires), et surtout **pas de jeu de test
   tenu à l'écart** chez nous. Toute phrase du rapport suggérant que nos chiffres
   répliquent ou nuancent les leurs doit être requalifiée.
2. Leur baseline la plus alarmante est le **Soft-Frozen Decoder** (directions contraintes
   à rester à cosinus ≥ 0,8 de leur init aléatoire), pas le Frozen Decoder. C'est celle-là
   qui égale presque le modèle entraîné — et c'est celle que nous n'avons pas reproduite.
   Reproduire le Soft-Frozen est peu coûteux (une contrainte de projection dans
   `normalize_decoder`) et bien plus informatif que le Frozen Decoder seul.
3. **Confond potentiel dans notre baseline** : `FrozenDecoderExtendedSAE` hérite de
   `input_scale = 1.0` (buffer jamais calibré hors `_init_from_residual_pca`) alors que
   `ExtendedSAE` a `input_scale = médiane des normes du résidu`. La comparaison
   45,3 % vs 29,3 % oppose donc peut-être « décodeur entraîné + échelle calibrée » à
   « décodeur figé + échelle unitaire ». À vérifier et, si confirmé, à refaire à échelle
   identique. P0.

### A.4 Latent Terms (Clavié et al.) — **article lu, quatre écarts structurels** (Palier 1 pour le constat, Palier 2 → voir F.1)

Setup de référence : SAE **Top-K (Gao et al. 2024)**, `m = 32 768` latents, **k = 16**
(train et inférence), init décodeur Kaiming + encodeur = décodeur transposé, AdamW,
lr 1e-3, warmup 5 % + cosine decay, batch 4096, **30 G tokens, 3 époques**, **5 graines
moyennées**. Entraînement sur les activations **token-level** du retriever gelé, sur du
**web générique (FineWeb-Edu)** — les auteurs insistent : « the SAE never sees data which
is directly in-domain ». Puis sum-pooling sur les tokens, ϕ(u) = √u, BM25 (k1 = 8,
b = 0,7 par défaut). Baselines : BM25 lexical, SPLADE-v2 / v2-Distill / v3, **et les
retrievers denses eux-mêmes** (Contriever, Nomic, GTE-ModernColBERT).

| Point | Papier | `src/sae/retrieval/latent_terms.py` + Pipeline 2 | Gravité |
|---|---|---|---|
| Unité encodée par le SAE | **token** (activations finales du retriever, encodées position par position) | **phrase entière** (embedding poolé, puis SAE) | P0 — ce n'est pas la même méthode |
| Corpus d'entraînement du SAE | web générique, **hors domaine**, 30 G tokens | les mails eux-mêmes, **en domaine**, quelques milliers de phrases | P0 |
| Sum-pooling | sum, **max testé et rejeté** (« consistent minor performance degradation ») | sum ici, **max partout ailleurs dans le dépôt** | P1, voir B.9 |
| Architecture | Top-K (Gao) ; BatchTopK et JumpReLU testés, sans gain | BatchTopK | mineur, mais à citer correctement |
| Dictionnaire | 32 768 | `D_SAE = 8192` | P1 |
| Graines | 5, moyennées | 1 | P1 |
| Baselines | BM25, SPLADE ×3, **et le retriever dense lui-même** | TF-IDF seul, avec des requêtes paraphrasées qui le handicapent par construction | **P0 — la conclusion du §17 n'est pas soutenable en l'état** |
| ϕ, k1, b | √, 8, 0,7 | idem | conforme |

Deux remarques de fond, à intégrer au rapport plutôt qu'à corriger dans le code :

- Le papier revendique la **parité** avec le retriever dense (« Latent Terms + Nomic only
  very slightly outperforms its backbone, essentially just matching its overall
  performance »), pas la supériorité. La bonne question pour nous n'est donc pas « est-ce
  que ça marche mieux que TF-IDF » mais « est-ce qu'on retrouve la qualité du dense avec
  une représentation sparse et inspectable ».
- **Contraste intéressant avec le résultat central du stage** : Latent Terms obtient une
  utilité de retrieval avec un SAE entraîné **hors domaine**, tandis que notre résultat
  principal est qu'un corpus **en domaine** est nécessaire pour l'auto-interprétabilité.
  Les deux peuvent être vrais — utilité en aval et interprétabilité par juge LLM ne sont
  pas la même propriété — et le dire explicitement renforce la portée du chapitre 3 au
  lieu de l'affaiblir. C'est un des rares endroits où le rapport peut dialoguer avec la
  littérature plutôt que de s'y comparer.

Perf inchangée : `LatentTermsIndex.search` fait `W.getcol(j)` sur une matrice CSR →
convertir en CSC une fois à la construction.

### A.5 Gemma Scope 2 — sites d'extraction (P0 potentiel, Palier 2)

`src/sae/saev5.py` (~l. 820-835) capture :
- `attn_out` → **entrée** de `o_proj` (via `register_forward_pre_hook`) ;
- `mlp_out` → **sortie** de `post_feedforward_layernorm` ;
- `resid_post` → `hidden_states[layer]`.

Chacune de ces trois définitions doit être vérifiée **bit à bit** contre celle utilisée
pour entraîner les SAE GemmaScope correspondants. Un décalage d'un seul LayerNorm (avant
vs après `post_feedforward_layernorm`, ou résidu pré- vs post-ajout) produit un SAE
appliqué à une distribution qu'il n'a jamais vue : les activations restent plausibles, la
FVE reste calculable, et toute la comparaison du §53 (« mlp_out significativement
meilleur qu'attn_out ») devient un artefact.

Test de falsification simple et peu coûteux : pour chaque hook, mesurer la **FVE du SAE
core seul** sur un échantillon de tokens. Un SAE appliqué au bon site doit reconstruire à
un niveau comparable à celui annoncé par GemmaScope ; appliqué au mauvais site, la FVE
s'effondre. Si les trois FVE sont du même ordre que la référence publiée, le hook est
probablement correct ; sinon c'est un P0.

Vérifie aussi `hidden_states[layer]` : dans HF, `hidden_states[0]` est l'embedding, donc
`hidden_states[L]` est la sortie du bloc `L-1`. Confirme que le SAE `layer_24_*` attend
bien `hidden_states[24]` et pas `hidden_states[25]` — un décalage d'une couche est
silencieux et parfaitement plausible ici.

**Sur le choix de site en tant que tel (`resid_post` vs `mlp_out`/`attn_out`) : déjà
bien soutenu par les données du projet, à relire précisément.** §53 compare les trois
sites à layer 24 fixe : `mlp_out` (52,7 %) n'est **pas** significativement meilleur que
`resid_post` (45,3 %, baseline, z=1,27, p=0,204) — seule la comparaison directe
`mlp_out` vs `attn_out` est significative (p=0,0025). Le seul point qui bat
`resid_post`/layer 24 avec significativité dans tout le balayage est `resid_post`/layer
31 (§51) — c'est donc la **profondeur**, pas le **site d'extraction**, le levier
confirmé jusqu'ici. **Mécanisme de défaillance silencieuse à vérifier (13/08)** :
`gemma_scope_loader.py::gemma_scope_converter` dérive le layer depuis `hf_hook_point_in`
du `config.json` téléchargé, avec repli sur `raw_cfg.get("hook_layer", 24)` si le champ
est absent — un `config.json` incomplet ferait silencieusement retomber le SAE chargé
sur layer 24 quel que soit le `LAYER` demandé ailleurs dans le pipeline. Probablement rare
en pratique (les releases GemmaScope-2 ont normalement ce champ), mais vérifier que ce
repli ne s'est jamais déclenché sur les runs déjà publiés (log au chargement, à
grep dans les logs SLURM archivés) avant de faire confiance aux comparaisons de layer. Cohérent avec la convention du domaine (Gemma Scope, Llama Scope,
Anthropic utilisent `resid_post`/`resid_pre` par défaut, le bus central accumulant les
contributions de toutes les couches, plus adapté à une décomposition sémantique
générale que `mlp_out`/`attn_out`, plus spécifiques à l'analyse de circuit). **Trou
resté ouvert** : la grille layer × hook-point n'est pas complète — `mlp_out` n'a été
testé qu'à layer 24, jamais à layer 31 ; si les deux effets se combinent, cette cellule
non testée pourrait battre les deux séparément. Extension naturelle et peu coûteuse
(réutilise l'infra existante du balayage), Palier 2, valeur incertaine mais non nulle.

### A.6 Dépôts qu'on aurait pu réutiliser et qu'on n'a pas réutilisés (Palier 1, analyse pure)

Instruis chaque ligne : est-ce un choix défendable ou une réimplémentation gratuite ?

| Brique | Existant | Ce qu'on a fait | À instruire |
|---|---|---|---|
| Auto-interprétation des features | `EleutherAI/delphi` (ex-`sae-auto-interp`) : scorers *detection*, *fuzzing*, *simulation*, prompts et protocoles publiés | protocole odd-one-out maison, prompts maison, ρ_interp maison | **Le plus important.** Ce harness résoudrait d'un coup B.2, B.3 et B.5. Estime le coût d'adoption et ce que ça changerait aux chiffres publiés. |
| Entraînement de SAE | `SAELens` (`SAETrainingRunner`), `saprmarks/dictionary_learning` (BatchTopK, AuxK) | harnais maison (`sae_shared.py`, `phrase_sae.py`) | Le harnais maison est-il justifié par la contrainte « cœur gelé » ? Une sous-classe de l'existant aurait-elle suffi ? |
| Évaluation de SAE | `SAEBench` (sparse probing, absorption, unlearning, RAVEL) | métriques maison + odd-one-out maison | SAEBench fournit des chiffres comparables à la littérature ; nos taux ne le sont pas. |
| Latent Terms | `x-tabdeveloping/latent_terms` | réimplémentation ~40 lignes | Défendable (dépôt JAX, tiers), mais le vérifier comme oracle de test. |
| interp_embed | dépôt officiel | jamais installé | Voir A.2. |

---

### A.7 SPLARE (Formal, Louis, Déjean, Clinchant — NAVER Labs Europe, 2026) (Palier 1, citation/discussion, rien à exécuter)

Article présent dans `pdf/Naver.pdf`, **jamais cité dans le rapport** alors qu'il apporte
deux points de comparaison directs sur des questions que le stage se pose :

- **Choix de la couche** : ils balaient les couches de Llama-3.1-8B (Llama Scope) et de
  Gemma-2-2B (Gemma Scope) et trouvent un optimum systématique **aux deux tiers de la
  profondeur** (couche ~20/32 et ~16/26). Notre §51 trouve la couche 31 significativement
  meilleure que la 24 sur un modèle de 48 couches — soit ~0,65 de la profondeur. **La
  convergence est frappante et doit être citée** : elle transforme un résultat isolé du
  stage en réplication d'un effet documenté ailleurs.
- **Largeur du SAE** : ils observent une relation **log-linéaire** entre largeur (16 k →
  1 M) et efficacité de retrieval. Notre largeur (65 k) a été choisie sur la **couverture
  des labels Neuronpedia**, pas sur une métrique d'utilité. À confronter explicitement au
  chapitre 4.
- **Paradigme inverse du nôtre** : ils gèlent le SAE et adaptent le LLM (LoRA, perte KL de
  distillation + régularisation FLOPS, Top-K pooling à l'inférence) ; nous gelons le core
  et entraînons une extension. Poser les deux côte à côte est une bonne section de
  discussion.

Le protocole complet (entraînement de retrieval avec négatifs difficiles et distillation)
est hors du périmètre restant du stage : citer, comparer, ne pas reproduire.

## 2. Axe B — Erreurs scientifiques suspectées

Ordre de priorité décroissante. Pour chacune : *symptôme*, *localisation*, *pourquoi ça
compte*, *comment trancher*, *correction envisagée*.

### B.1 — Le dictionnaire de l'extension est sous-complet : **conforme au papier, mais validé pour un autre usage** (P1, reformulé après lecture — Palier 1 pour le constat, **Palier 3** pour le balayage)

Constat initial : `D_EXTRA = 1024` pour `D_MODEL = 3840`, soit un facteur d'expansion de
0,27 — un « dictionnaire sparse » sous-complet, initialisé qui plus est par les 1024
premières directions principales du résidu.

**Lecture de SAE Boost faite : le ratio est le même chez eux** (1024 pour un d_model de
4096, soit 0,25). Ce n'est donc pas une divergence d'implémentation, et l'hypothèse d'une
erreur de transposition tombe.

Mais la conclusion utile change de nature plutôt que de disparaître : **SAE Boost
n'évalue jamais l'interprétabilité de ses features résiduelles**. Leurs trois métriques
sont EV, ΔCE et L0 — de la fidélité de reconstruction. Le dépôt hérite donc d'une taille
de dictionnaire validée pour la **reconstruction** et la réemploie pour produire des
features censées être **monosémantiques et nommables**, usage que la référence ne
soutient pas. Avec 1024 directions pour 3840 dimensions et une init PCA (base
orthonormale), rien ne garantit qu'un concept isolé occupe une direction dédiée.

À faire :
1. **(Palier 1, cache, quelques minutes)** Mesurer la **cohérence mutuelle** du
   dictionnaire déjà entraîné (max cosinus hors diagonale) et sa dérive par rapport à
   l'init PCA, sur le checkpoint existant. Un dictionnaire resté quasi-orthogonal signale
   que le SAE n'a fait que raffiner une ACP — c'est un résultat en soi, obtenu sans
   aucune GPU.
2. **(Palier 1, texte)** Dans le rapport, distinguer explicitement « conforme à SAE
   Boost » (vrai) de « validé par SAE Boost pour l'usage que nous en faisons » (faux) —
   ce correctif ne dépend d'aucune expérience.
3. **(Palier 3 — en tout dernier)** Balayer `D_EXTRA ∈ {1024, 4096, 16384}` **en
   mesurant l'interprétabilité avec le protocole corrigé** (B.2/B.3), pas seulement la
   FVE. C'est l'expérience qui tranche : si le taux monte avec la surcomplétude, le
   plafond de 45 % est un plafond de capacité, pas de protocole. C'est aussi la plus
   chère de tout l'audit avec B.19 (cf. plan croisé ci-dessous) — à ne lancer qu'après
   Palier 2 entièrement terminé, F compris, et seulement si le calendrier le permet. Si
   le temps manque, le point 1 (gratuit) suffit à documenter la limite au chapitre 4
   sans avoir besoin du balayage complet.

### B.19 — Le volume d'entraînement du résiduel reste en dessous de la référence, mais le dépôt montre déjà un signal suggestif à 25 M (P1, **corrigé après lecture complète de `RESULTS_TESTS.md`** — Palier 1 pour l'essentiel, Palier 2 pour finir ce qui est déjà engagé, Palier 3 seulement pour le run à pleine échelle)

**Correction par rapport à la version précédente de cette entrée** : elle citait « le run
200 M n'a jamais abouti » et le caractérisait comme un job à ne pas retenter sans grand
budget de temps. Une lecture plus complète de `RESULTS_TESTS.md` (§23, §47, §54) montre
une situation plus favorable et plus intéressante que ça.

**Ce qui s'est réellement passé** :
- Le balayage initial confirme un plateau plat : **100 k → 40,7 %, 500 k (défaut) →
  45,3 %, 2 M → 44,7 %** — écarts très inférieurs à l'écart-type binomial attendu sur
  n=150 (≈4,1 points). Jusque-là, la lecture d'origine (« le volume ne change rien dans
  la plage testée ») tient.
- Un run à **25 M** (job 41375, après un échec OOM à 100 M correctement diagnostiqué —
  buffer résidu alloué en RAM anonyme, 768 Go pour 100 M tokens) donne **54,0 % (81/150)**
  — un écart de **+8,7 points** par rapport au run principal, **non significatif seul**
  (test z à deux proportions, z=-1,50) mais **dans le même sens et du même ordre de
  grandeur** qu'un écart observé indépendamment sur une ablation totalement différente
  (`K_EXTRA=5`, +9,4 points, également non significatif seule, §25). Le dépôt note
  lui-même cette coïncidence et recommande une réplication multi-graines pour trancher —
  c'est exactement le bon réflexe.
- Cette réplication a déjà été **tentée** : jobs 42687 (seed 7, 25 M) et 42688 (seed 99,
  25 M), tous deux **échoués à 63 % et 34 % pour une raison d'infrastructure nœud sans
  rapport avec le code** (`FAILED, ExitCode 1:0`, aucun traceback) — le journal les
  qualifie explicitement de « à resoumettre tel quel » (§47).
- Le blocage structurel du run à 200 M (jobs 41658/42145/42694, tous
  `CANCELLED`/`FAILED`) a été **diagnostiqué et corrigé à la racine** (§54) : la cause
  n'était pas un incident cluster isolé mais une allocation RAM anonyme du réservoir de
  résidus dimensionnée sur `N_TOKENS_EXTRA_TRAIN` entier (≈1,4 To pour 200 M tokens),
  remplacée par un vrai memmap disque (`open_mmap_reservoir`, `torch.from_file`), qui
  ramène le besoin mémoire à quelques dizaines de Go. Le correctif est déjà dans
  `src/sae/saev5.py`, validé par un test isolé — mais **pas encore revalidé par une
  exécution réelle à 200 M**.

**Conséquence sur l'affirmation à corriger** : elle n'est plus « le volume n'a pas
d'effet, mais on ne l'a testé que dans un régime sous-entraîné » — elle devient « le
volume n'a pas d'effet mesurable entre 100 k et 2 M ; un run isolé à 25 M montre un écart
numérique suggestif mais non confirmé, dont la réplication est déjà engagée mais bloquée
par un incident d'infrastructure sans lien avec le pipeline, et le blocage qui empêchait
d'aller à 200 M est désormais corrigé au niveau du code ». C'est une position plus forte
et plus honnête que celle du rapport actuel, dans un sens comme dans l'autre : plus
prudente sur la plage 100 k–2 M, moins définitive sur l'absence totale d'effet de volume.

À faire, par ordre :
1. **(Palier 1, texte, immédiat, coût nul)** Requalifier l'affirmation dans le résumé,
   l'introduction, le chapitre 3 et la conclusion selon la formulation ci-dessus. Ne pas
   attendre le reste : ce correctif est indépendant de toute expérience à venir.
2. **(Palier 1/2, quasi gratuit — priorité la plus rentable de cette entrée)**
   Resoumettre tels quels les jobs 42687/42688 (seed 7 et seed 99 à 25 M) : aucune
   correction de code requise, l'échec précédent est documenté comme un incident
   d'infrastructure. C'est la façon la moins chère de savoir si l'écart +8,7 points est
   un effet réel ou du bruit — nettement moins cher que d'attendre ou de forcer le run
   à 200 M, et ça doit être fait **avant** de décider si l'étape 3 vaut la peine.
3. **(Palier 3 — en tout dernier, uniquement si l'étape 2 confirme un effet, ou si le
   temps restant après Palier 2 le permet de toute façon)** Relancer le run à l'échelle
   correcte (≥100 M, idéalement en s'approchant du 1 G de référence) maintenant que le
   correctif memmap est en place. Le risque a changé de nature : ce n'est plus un job
   dont la cause d'échec est inconnue, c'est un job dont la cause d'échec connue a été
   corrigée mais jamais revalidée à cette échelle — reste un risque d'échec résiduel
   (nouveau bug, incident cluster), mais plus le même pari qu'avant. Si l'étape 2 ne
   confirme rien, la requalification du point 1 seule reste une position défendable au
   chapitre 4 et cette étape peut être sautée sans dommage pour le rapport.
4. **(Palier 3, seulement après 1-3, et seulement s'il reste du temps)** Vérifier si
   l'interaction `volume × D_EXTRA` a du sens — un dictionnaire sous-complet sature
   peut-être en données bien avant 100 M, auquel cas B.1 et B.19 se répondent et
   l'expérience à faire est un plan croisé, pas deux balayages séparés. La manipulation
   la plus chère de tout ce document ; à n'entreprendre qu'en tout dernier recours.

### B.20 — La métrique de fidélité standard du domaine n'est pas mesurée dans les résultats publiés, mais elle est déjà implémentée (P1, corrigé après lecture de `src/sae/compare/`, Palier 1)

**Correction (13/08)** : `src/sae/compare/crosslingual.py::ce_loss_increase` implémente
déjà le ΔCE (delta cross-entropy du LLM sous substitution `x → SAE(x)` par hook, la
métrique standard SAEBench/SAE Boost que cette entrée disait absente) — patch propre,
calcul correct (`CE(patched) - CE(clean)`). **Il n'est simplement jamais appelé** : aucun
script, aucune section de `RESULTS_TESTS.md` ne l'utilise. Ce n'est donc pas du code à
écrire (Palier 2 comme classé précédemment), c'est du code prêt à appeler sur le SAE
core et l'extension actuels — Palier 1, quelques lignes d'intégration, pas de nouvelle
implémentation. Rendrait les chiffres directement comparables à GemmaScope et SAE Boost.

### B.2 — Les features jugées ne sont pas un échantillon représentatif (P0, Palier 2)

`src/sae/judge.py::feature_selection_by_magnitude` sélectionne les `N_FEATURES_TO_LABEL`
features de **plus forte magnitude moyenne** sur les tokens.

Le taux d'interprétabilité rapporté (20 %, 45,3 %, 12/28/45,3 % selon l'échelle) est donc
une statistique conditionnée à « être dans le top-N par magnitude », pas une estimation
du taux du dictionnaire. Or les features de plus forte magnitude sont typiquement les
plus denses, les moins spécifiques, souvent liées à la position ou à la fréquence — donc
plausiblement **les moins interprétables**. L'estimateur est biaisé, de signe inconnu, et
non comparable à la littérature (Neuronpedia et SAEBench échantillonnent uniformément ou
exhaustivement).

Référence désormais vérifiée : Korznikov et al. tirent **200 latents au hasard** parmi les
latents vivants (fréquence ≥ 1e-6). C'est le protocole à répliquer.

À faire : rejuger, sur cache si possible, (a) un échantillon **uniforme** de features
vivantes, (b) un échantillon **stratifié par décile de fréquence d'activation**, et
comparer aux chiffres publiés. Rapporter le taux avec IC de Wilson et la stratification.
Si le taux uniforme diffère significativement du taux top-N, tous les chiffres du rapport
doivent être requalifiés (pas nécessairement refaits : requalifiés en « taux sur les N
features les plus actives », ce qui est une affirmation plus faible mais correcte).

### B.3 — Le contrôle négatif de l'odd-one-out est trop facile (P0, Palier 2)

Dans `build_feature_examples_with_control` :
- les **positifs** sont le mot d'activation maximale de chaque document, marqué `<<mot>>`,
  précédé de son contexte gauche ;
- le **négatif** est le mot situé **au milieu** (`len(toks)//2`) d'un document tiré au
  hasard sous le quantile 5 %.

Le juge peut donc réussir sur des indices de surface (saillance, position, nature
grammaticale du mot marqué) sans rien comprendre au concept. C'est cohérent avec le fait
que le décodeur **aléatoire** obtienne 29,3 % (contre un hasard nominal de 1/10). Le
protocole ne mesure pas seulement l'interprétabilité de la feature : il mesure aussi la
détectabilité d'un artefact de sélection.

Aggravant : la déduplication des positifs se fait **sur le mot-cible**, ce qui élimine
exactement les features les plus monosémantiques (celles qui se déclenchent toujours sur
le même mot) ou les force à présenter 9 mots différents — le protocole pénalise
structurellement le cas le plus favorable.

Référence désormais vérifiée (Korznikov et al.) : la description est produite à partir de
15 séquences activantes, puis évaluée sur un **jeu de test tenu à l'écart de 100
séquences (50 activantes à intensités variées, 50 non activantes)**, par un **second**
modèle qui ne voit que la description. Score = exactitude binaire, hasard = 0,50. Ce
protocole supprime d'un coup l'asymétrie de construction, la fuite entre exemples
d'explication et exemples d'évaluation, et une partie de la circularité juge/extracteur.
Le porter est probablement plus rentable que de rustiner l'odd-one-out.

À faire :
1. Remplacer le négatif par un **négatif dur** : le mot d'activation maximale d'une
   *autre* feature, extrait par le même chemin de code. Toute asymétrie de construction
   entre positifs et négatif doit disparaître.
2. Mesurer le taux avec négatif dur vs négatif actuel sur le même jeu de features
   (appariement → McNemar).
3. Refaire tourner la baseline décodeur aléatoire avec le négatif dur : si son taux
   s'effondre vers 10 % alors que le SAE entraîné tient, le sanity check devient
   concluant ; s'il tient à 29 %, c'est le protocole qui est en cause.
4. Tester la déduplication par document plutôt que par mot-cible, et rapporter les deux.
5. Rapporter explicitement le niveau du hasard (1/(n_pos+1)) partout où un taux est cité.

### B.4 — Circularité générateur / extracteur / juge (P0 pour le résultat phare — Palier 1 pour le contrôle partiel, **Palier 3** pour le croisement complet)

Le même modèle (Gemma-3-12B-it) a : généré le corpus augmenté
(`scripts/run_augmentation.py`), fourni les activations décomposées, et joué le juge.

Le résultat le plus mis en avant du rapport — l'effet dose-réponse de l'échelle
(12 % → 28 % → 45,3 %) — fait varier **simultanément** l'extracteur et le juge. Il est
donc, tel quel, ininterprétable : on ne peut pas distinguer « les features d'un gros
modèle sont plus interprétables » de « un gros modèle est meilleur à résoudre une énigme
de type intrus ». `RESULTS_TESTS.md` §43 amorce la séparation (juge 4b sur extraction
12b) mais ne la complète pas.

À faire :
1. **(Palier 1, cache, coût nul)** Instruire la circularité corpus : le juge évalue des
   features apprises sur du texte que le même modèle a écrit. `c2_original_only_rejudge`
   fait quelque chose de ce genre — vérifie ce qu'il contrôle exactement et si le
   résultat qu'il produit est bien celui cité au rapport. À faire avant toute autre
   chose sur ce constat : c'est gratuit et ça borne déjà une partie du problème.
2. **(Palier 3 — en tout dernier, et seulement si B.8 confirme que les activations des
   échelles 1b/4b/12b sont encore sur disque)** Croisement complet
   **extracteur × juge** sur {1b, 4b, 12b} — au minimum les trois diagonales plus les
   deux cellules (extracteur 12b × juge 1b) et (extracteur 1b × juge 12b). Analyse :
   effet principal de l'extracteur, effet principal du juge, interaction. C'est la seule
   façon de trancher si la dose-réponse est une propriété des features ou un artefact
   de circularité — mais c'est aussi coûteux, potentiellement bloqué si les activations
   des échelles 1b/4b ont été purgées (cf. B.8, le dépôt a déjà perdu des artefacts de
   run pour l'espace disque). Tant que ce croisement n'existe pas, le rapport doit
   présenter la dose-réponse comme un effet **conjoint**, pas comme une propriété des
   features — cette requalification textuelle (Palier 1) doit être faite immédiatement,
   indépendamment de si le croisement complet a lieu.
3. **(Palier 2, coût modéré, quatrième point de la courbe — vérifié le 13/08, recherche
   web)** Ajouter **Gemma-3-27B-it** comme point supplémentaire de la courbe
   dose-réponse. GemmaScope-2 couvre officiellement le 27B (`google/gemma-scope-2-27b-it`,
   mêmes trois sites `resid_post`/`attn_out`/`mlp_out`, mêmes largeurs 16k/64k/256k/1m,
   mêmes paliers de L0, interface `SAELens.from_pretrained` identique à celle déjà
   utilisée pour 1b/4b/12b) — l'ajout dans `_PRESETS` (`src/config.py`) est mécanique,
   même format que les entrées existantes, aucun nouveau code de chargement. En VRAM :
   27B en bf16 ≈ 54 Go, tient confortablement sur un seul H100 80 Go, contrairement à
   Llama-3.3-70B (140 Go, cf. F.2) — pas de compromis de quantification ni de template
   multi-GPU à inventer, le pipeline d'extraction existant tourne tel quel, juste plus
   longtemps (extrapolation grossière depuis le run 12B/500k, ~3h : probablement 4-6h
   pour 27B, à confirmer). **Ne pas confondre avec l'option Llama-70B de F.2** : celle-là
   sert la fidélité de citation à Jiang et al. (leur backbone exact) ; celle-ci sert
   directement B.4, en ajoutant un point à la courbe dose-réponse du projet lui-même —
   c'est la meilleure des deux dépenses si le choix doit être fait entre les deux, parce
   qu'elle renforce un résultat déjà central au rapport plutôt que d'ajouter une
   comparaison optionnelle. Reste tout de même un point unique (extracteur=27b,
   juge=27b) tant que le croisement du point 2 n'est pas fait — utile en soi, mais ne
   remplace pas le croisement complet pour trancher la circularité.
4. **(Palier 2, précision méthodologique avant de lancer le point 3 en croisement)**
   Croiser 27B et 12B (ex. extracteur=27b/juge=12b, extracteur=12b/juge=27b) est un
   progrès réel sur le point 2 — ce n'est plus le même modèle qui extrait et juge — mais
   **rester dans la famille Gemma-3 ne clôt pas complètement B.4, pour deux raisons
   distinctes** :
   - **Confond de famille.** Les échelles de Gemma-3 partagent tokenizer, distribution
     de préentraînement et choix d'architecture. Un juge 12B pourrait être structurellement
     mieux disposé à « comprendre » les activations d'un extracteur 27B de la même famille
     qu'un modèle externe (même segmentation SentencePiece, mêmes conventions de
     génération) — pour des raisons de parenté d'entraînement, pas de qualité réelle de la
     décomposition SAE. Un croisement 100 % Gemma-3 teste "l'échelle importe-t-elle
     indépendamment du rôle" mais ne teste pas "la circularité de famille importe-t-elle".
     **Solution concrète identifiée et vérifiée (13/08)** : Goodfire a aussi
     open-sourcé un SAE sur **Llama-3.1-8B-Instruct** (couche 19, L0=91,
     `Goodfire/Llama-3.1-8B-Instruct-SAE-l19`, même fournisseur/méthode que le SAE
     70B utilisé par Jiang et al.) — 8B en bf16 ≈ 16 Go, tient sans difficulté sur
     un seul H100. Deux usages séparés, coûts très différents :
     - **Juge seul** (résout directement le confond de famille) : n'a besoin
       d'aucun SAE, juste du modèle de chat Llama-3.1-8B-Instruct, utilisé
       exactement comme Gemma-3 l'est aujourd'hui comme juge. Quasi gratuit,
       Palier 2. Réserve à vérifier avant de faire confiance au résultat : Llama-
       3.1-8B est plus petit et plus ancien que Gemma-3-12B/27B — sanity check
       rapide de sa capacité en français recommandé avant de l'utiliser comme
       juge sur le corpus de mails, sinon un confond de capacité linguistique
       remplace le confond de famille.
     - **Extracteur avec son SAE Goodfire** (utile pour F.2, pas pour B.4) :
       plus de travail (nouveau chemin de code, template de chat Llama), et à
       documenter comme reproduction à **échelle réduite** (8B, pas les 70B du
       papier) — valide le protocole, ne reproduit pas leurs chiffres exacts.
   - **Circularité corpus spécifique à la combinaison proposée.** Le corpus augmenté qui
     entraîne l'extension a été généré par **Gemma-3-12B-it** (`run_augmentation.py`). Si
     12B sert de **juge** dans le croisement (extracteur=27b/juge=12b), ce juge évalue des
     features apprises en partie sur du texte qu'il a lui-même écrit — le problème déjà
     identifié au point 1 se réintroduit spécifiquement dans cette cellule du plan croisé,
     alors qu'il ne se pose pas si 12B ne joue que le rôle d'extracteur. **Restreindre
     cette cellule précise (juge=12b) aux mails originaux non augmentés**, comme le fait
     déjà `c2_original_only_rejudge` (§50/§52), plutôt que de l'exécuter sur le corpus
     complet — sinon le résultat de cette cellule spécifique est ininterprétable pour la
     même raison que la préoccupation d'origine du point 1.

### B.5 — ρ_interp ne mesure pas ce que son nom annonce (P1, Palier 1)

Dans `odd_one_out_judge`, l'« activation réelle » utilisée pour la corrélation de
Spearman est `float(n_pos - orig_idx)`, c'est-à-dire **le rang dans le top**, et non la
magnitude d'activation. Le négatif reçoit `0.0`. De plus, la corrélation est calculée sur
les mêmes exemples que ceux ayant servi à produire le label, et **uniquement si
`interp_score == 1`**.

Trois problèmes cumulés : (a) un proxy de rang écrase toute l'information de magnitude,
qui est exactement ce que Bills et al. mesurent ; (b) le point négatif à 0 face à 9
positifs crée mécaniquement une corrélation positive quel que soit le classement interne ;
(c) le conditionnement à `interp_score == 1` fait de ρ_interp une statistique
post-sélection, non comparable à la valeur publiée par Bills et al.

À faire : soit implémenter la version correcte (magnitude réelle, échantillon tenu à
l'écart, calculée sur toutes les features), soit retirer ρ_interp du rapport. La position
intermédiaire actuelle (le publier en signalant l'écart en note) est la moins bonne :
le chiffre reste lu comme le ρ de la littérature.

### B.6 — Fuite de groupe dans la sonde de classification (P0, Palier 1 — recalcul sur cache)

`src/analysis/metrics.py::downstream_classification` fait un `StratifiedKFold(5,
shuffle=True)` sur l'ensemble des documents. Or `build_email_train_test_corpus` construit
un corpus où **chaque mail source engendre jusqu'à 13 variantes**, une par axe de
perturbation. Le split train/test global est group-aware, mais la **validation croisée de
la sonde ne l'est pas** : les variantes d'un même mail source se répartissent entre plis.

C'est très probablement l'explication du constat déjà noté dans le rapport (« ~93 % de ce
chiffre est reproductible par un baseline TF-IDF sans sémantique ») : le classifieur
reconnaît le mail source, pas l'axe.

Aggravants dans la même fonction :
- aucune standardisation avant une régression logistique **L2 avec `C=1.0`** sur des
  features aux échelles hétérogènes de plusieurs ordres de grandeur (JumpReLU du core
  non borné vs TopK de l'extension) : la pénalité n'a pas le même sens selon la feature,
  et la comparaison entre configurations de largeurs différentes n'est pas légitime ;
- `max_iter=1000` sur 66k features : vérifier qu'il n'y a pas de non-convergence
  silencieuse (capturer les `ConvergenceWarning`) ;
- aucun intervalle de confiance, une seule graine de CV ;
- `acc_raw` n'est calculé que pour le Pipeline 2 → le Pipeline 1 n'a **aucune baseline
  « embedding brut »**, alors que c'est la comparaison qui dit si le SAE apporte quelque
  chose.

À faire : `GroupKFold` (ou `StratifiedGroupKFold`) sur l'identifiant du mail source —
ce qui implique de faire remonter `parent_idx` depuis `build_email_train_test_corpus`,
qui ne le retourne pas aujourd'hui. Recalculer `clf_acc_email_axes` sur cache et mesurer
l'écart. Ajouter la baseline activations brutes pour P1.

### B.7 — Le test de fidélité de l'explication est tautologique (P0, Palier 2)

`scripts/explanation_fidelity_test.py` : la sonde est **linéaire**, la contribution d'une
feature est définie comme `coef_i × activation_i`, l'explication est le top-K de ces
contributions, et le test consiste à mettre ces features à zéro et à constater que la
probabilité chute plus qu'en ablatant des features aléatoires ou le bottom-K.

Pour un modèle linéaire, c'est une identité algébrique : retirer les plus grands termes
positifs de la somme fait mécaniquement plus baisser la somme que d'en retirer de petits.
**Le test ne peut pas échouer**, donc il ne mesure rien. Il ne teste ni la fidélité des
features au texte, ni la causalité au niveau du modèle de langage.

À faire, par ordre de valeur :
1. Ablater dans l'espace SAE puis **décoder et repasser par le modèle** (le dépôt a déjà
   `steer_and_decode`) : la décision change-t-elle dans le sens attendu ?
2. À défaut, contrôle apparié par magnitude : comparer l'ablation du top-K par
   contribution à celle de K features de **magnitude d'activation comparable mais de
   coefficient faible**. Là, le test peut échouer.
3. Retirer ou requalifier l'affirmation du rapport selon laquelle « deux tests
   indépendants confirment » la qualité de l'explication : à ce stade il en reste un
   (la plausibilité par choix forcé), et lui-même à réexaminer (juge = modèle du projet).

### B.8 — Clés de cache non discriminantes : menace directe sur la campagne d'ablation (P0, Palier 0 — à faire en tout premier)

Plusieurs artefacts sont mis en cache avec des clés qui **n'encodent pas tous les
paramètres dont ils dépendent** :

| Artefact | Clé actuelle | Ne contient pas |
|---|---|---|
| Embeddings P2 (`saev5.py` l. ~1408-1443) | `train_phrase_emb_dim{MATRYOSHKA_DIM}_n{n}` | **`EMB_MODEL`** |
| SAE P2 (`saev5.py` l. ~1418) | `p2_sae_dim{d_in}_d{D_SAE}_k{K_SPARSE}.pt` | `EMB_MODEL`, `EPOCHS`, `LR`, `SEED` |
| Extension P1 (`saev5.py` l. ~960) | `p1_frozen_core_d{D_EXTRA}_k{K_EXTRA}.pt` | `EPOCHS_EXTRA`, `LR_EXTRA`, `N_TOKENS_EXTRA_TRAIN`, `SEED`, `LAYER`, `HOOK_TYPE`, `SANITY_CHECK_FROZEN_DECODER` |
| Activations doc P1 | `p1_all_doc_acts_ext_d{D_EXTRA}.pt` | `K_EXTRA`, `LAYER`, `HOOK_TYPE`, `SEED` (garde-fou = shape[0] seulement) |

De plus, `sae_shared.py::load_or_train_extended_sae` **retourne le checkpoint existant
sans vérifier la config** dès que le fichier est là.

Les scripts SLURM ouvrent en général un `SAVE_DIR` neuf par ablation (`results_v13_*`),
et un commentaire montre que le risque était connu (« NOUVEAU SAVE_DIR obligatoire :
extract_f2llm_embeddings met en cache par nom de… »). Mais **9 scripts partagent
`results_v10_emails_main/` et 7 partagent `results_v12_scaled_65k/`**.

C'est le point le plus dangereux du dépôt, parce que la conclusion centrale du chapitre 3
est une **absence d'effet** (« aucun hyperparamètre ne change le taux »), et qu'un cache
réutilisé produit exactement ce résultat.

À faire, dans cet ordre :
1. **Audit rétroactif** : pour chaque ablation citée dans le rapport, croiser le script
   SLURM (SAVE_DIR, variables exportées) avec les fichiers effectivement présents dans le
   répertoire de résultats et leurs `mtime`. Une extension datée d'avant le lancement du
   job = résultat contaminé. Produire un tableau ablation par ablation avec un statut
   `CACHE PROPRE` / `CACHE PARTAGÉ` / `CONTAMINÉ` / `INVÉRIFIABLE (fichiers purgés)`.
2. **Correctif structurel** : dériver un `RUN_HASH` de l'ensemble des paramètres
   pertinents, l'inclure dans tous les noms de fichiers de cache, écrire un
   `config_manifest.json` par run, et **refuser de charger** un cache dont le manifeste
   diffère (erreur explicite, pas un warning).
3. Ajouter un test de non-régression qui vérifie qu'un changement de n'importe quel
   paramètre d'entraînement change bien la clé de cache.

### B.9 — Le max-pooling documentaire introduit un biais de longueur (P1, transverse, Palier 1)

`doc_vec[f] = max_t enc(x_t)[f]` (`activations.py::maxpool_sae_docs`). Le maximum sur
`T` tirages est une statistique d'ordre : il croît avec `T`. Deux documents de longueurs
différentes n'ont donc pas des vecteurs comparables, et le nombre de features actives par
document croît mécaniquement avec la longueur.

Or les axes d'augmentation modifient la longueur (registre, verbosité, reformulation).
Tout ce qui est calculé sur `doc_acts` en dépend : classification par axe, silhouette,
diffing Fisher, NPMI, clustering.

À faire (peu coûteux, sur cache) :
1. Corréler la longueur en tokens de chaque document avec (a) son nombre de features
   actives, (b) la norme de son vecteur, (c) la probabilité prédite par la sonde.
   Rapporter les corrélations de Spearman.
2. Si la corrélation est forte, tester des pooling alternatifs : moyenne, `top-k` moyenne,
   quantile 0,95, ou max normalisé par `log T`. Comparer l'effet sur `clf_acc_email_axes`
   **avec** la CV group-aware de B.6.
3. Vérifier ce que fait la littérature de référence : Latent Terms utilise explicitement
   un **sum-pooling** (et le code le respecte pour le retrieval), ce qui rend l'écart
   avec le max-pooling du reste du projet encore plus notable — deux pooling différents
   coexistent dans le dépôt sans que le choix soit justifié ailleurs que par l'usage.

### B.10 — Le masquage des tokens crée un décalage de distribution avec GemmaScope (P1, Palier 1 pour la mesure, Palier 2 pour le recalcul FVE)

`activations.py` applique trois filtres avant tout encodage : special tokens,
`skip_first_content_token=True`, et `norm_outlier_mask(sigma_clip=4.0)`.

Le SAE core GemmaScope a été entraîné, lui, sur la distribution **complète**, tokens de
norme massive inclus. On l'applique donc à une distribution tronquée, ce qui invalide
partiellement la comparaison de FVE avec les valeurs publiées, et retire précisément les
tokens qui portent le plus d'énergie du résidu.

Par ailleurs le masque de norme est calculé **par batch**, donc le seuil dépend de la
composition du batch : deux exécutions avec un `batch_size` différent ne masquent pas les
mêmes tokens. Ce n'est pas déterministe au sens strict.

À faire : mesurer la fraction de tokens exclus par chacun des trois filtres ; recalculer
FVE/L0/taux d'interprétabilité **sans** `norm_outlier_mask` sur un échantillon ; rendre le
seuil global (calculé une fois sur un échantillon de calibration) plutôt que par batch ;
documenter la décision finale comme un écart assumé à la pratique GemmaScope.

### B.11 — Validation de l'entraînement : deux défauts qui se cumulent (P1, Palier 1 pour le fix, Palier 2 si retrain complet)

Dans `sae_shared.py::load_or_train_extended_sae` :
1. Le split de validation est tiré **au niveau du token** (`torch.randperm` sur
   `acts_train`). Des tokens d'un même document se retrouvent des deux côtés : la
   « validation » n'est pas hors échantillon au niveau document.
2. La perte de validation est calculée après `model.eval()`, donc en régime **JumpReLU à
   seuil global**, alors que la perte d'entraînement est en régime **BatchTopK**. Les deux
   courbes ne sont pas comparables ; l'écart train/val observé mélange sur-apprentissage
   et changement de fonction d'activation.

À faire : split par document ; et rapporter la val loss dans les deux régimes (ou en
mode train, sans gradient) pour rendre la comparaison licite. Vérifier au passage si la
« convergence » constatée dans `docs/sae_diagnostics_playbook.md` repose sur cette courbe.

### B.12 — Troncature Matryoshka sur un backbone peut-être non-Matryoshka (P1, Palier 1 pour la vérification, Palier 2 pour le retrain à dimension pleine)

`phrase_sae.py::extract_f2llm_embeddings` fait
`F.normalize(pooled[:, :MATRYOSHKA_DIM])` avec `MATRYOSHKA_DIM = 320`.

Tronquer les 320 premières dimensions d'un embedding n'est valide **que si le modèle a
été entraîné avec une perte Matryoshka (MRL)**. Sinon, on garde 320 coordonnées
arbitraires. Le balayage du §31 (« dégradation graduelle, pas abrupte ») est compatible
avec les deux hypothèses, mais une dégradation graduelle est **aussi** ce qu'on attend
d'une troncature arbitraire.

À faire : vérifier sur la model card de `codefuse-ai/F2LLM-v2-*` si MRL est utilisé ; si
non, refaire le Pipeline 2 à dimension pleine et comparer. Vérifier également la
cohérence du pooling (`EMB_POOLING=last_token` pour F2LLM, `cls` pour bge-m3) avec ce que
recommandent les cartes de modèle respectives — un mauvais pooling produit des embeddings
utilisables mais nettement dégradés, ce qui pourrait expliquer l'écart P1/P2.

Question de fond à instruire dans le rapport, au-delà du code : **le Pipeline 2 est-il une
bonne idée ?** Un SAE sur des embeddings de phrase L2-normalisés d'un modèle de 80M
paramètres décompose un espace de 320 dimensions déjà fortement compressé et entraîné par
apprentissage contrastif — la superposition qu'un SAE est censé défaire n'y a pas la même
nature que dans un residual stream. Le justifier explicitement (coût, granularité
phrase, comparabilité) ou l'assumer comme exploratoire.

### B.13 — Métriques : objets mesurés incohérents (P1, Palier 1)

Dans `metrics.py::compute_metrics` :
- la FVE est calculée sur `sae_out = core_out + extra_out` (les deux branches), mais le
  L0 retourné est `l0_extra` (**l'extension seule**). Les deux chiffres du tableau de
  résultats ne portent pas sur le même objet ; toute lecture « FVE vs L0 » (frontière de
  Pareto) est fausse.
- la variance de référence est un scalaire moyenné sur toutes les dimensions ; c'est un
  choix (documenté, source du désaccord avec SAELens) mais il rend la FVE non comparable
  aux valeurs publiées de GemmaScope. Rapporter les deux formules côte à côte.
- `compute_rho_sae` calcule des cosinus **sans centrer**. Le residual stream a une
  composante moyenne très dominante : tous les cosinus sont écrasés vers 1 et le Spearman
  se calcule sur du bruit résiduel. Recalculer après centrage et comparer.

### B.14 — Tests statistiques sur données appariées (P1, Palier 1)

`cooccurrence.py::corpus_diff_stats` applique un **test exact de Fisher** feature par
feature entre deux groupes. Quand les deux groupes sont « mails originaux » vs « variantes
augmentées », les observations sont **appariées** (chaque variante dérive d'un original) :
Fisher suppose l'indépendance, McNemar est le test correct. L'audit §30 a identifié cette
classe de problème et produit `src/analysis/stats.py`, mais vérifie que `corpus_diff_stats`
lui-même a bien été mis en conformité — le module partagé existe, encore faut-il qu'il
soit appelé là où il faut.

Vérifie aussi que la correction BH est appliquée sur l'ensemble des features testées et
non sur un sous-ensemble filtré a posteriori (le filtre `a + b == 0` retire des features
avant le calcul du nombre de tests : c'est légitime, mais doit être explicite dans le
rapport du nombre de tests).

### B.15 — Clustering : métriques non comparables entre espaces (P1, Palier 1)

`analyze_with_umap` : UMAP-2D pour l'affichage, UMAP-10D pour HDBSCAN, choix justifié par
un audit comparant le **DBCV** entre UMAP-2D, UMAP-10D, PCA et cosinus brut.

Le DBCV est une mesure de validité **interne**, calculée dans l'espace où l'on clusterise.
Le comparer entre espaces de dimensions et de métriques différentes n'est pas licite :
UMAP optimise justement la structure locale que le DBCV récompense, il gagnera toujours.
La comparaison valide est externe — c'est l'AMI vis-à-vis de labels connus, et l'audit
rapporte lui-même **AMI ≈ 0,01–0,03 partout**, c'est-à-dire aucune structure sémantique.

À faire : requalifier la conclusion (« UMAP-10D retenu pour la stabilité de l'affichage »,
pas « domine les autres espaces »), et présenter l'AMI ≈ 0 comme le résultat principal de
cette partie — c'est un résultat négatif honnête et intéressant, actuellement enfoui dans
un commentaire de code. Vérifier aussi `min_cluster_size = N_DOCS // 15`, valeur arbitraire
qui force le nombre de clusters, et la balayer.

`compute_silhouette` : silhouette cosinus sur les vecteurs max-poolés L2-normalisés,
comparée entre configurations de **largeurs différentes** (16k/65k/262k). La silhouette
dépend de la dimension et de la sparsité ; ces comparaisons croisées sont à retirer ou à
assortir d'un contrôle (même nombre de features actives).

### B.16 — Seuils non calibrés (P2, Palier 1)

- `find_interesting_pairs(sim_threshold=0.2)` : les modèles d'embedding modernes ont un
  plancher de similarité élevé (deux labels sans rapport sont rarement sous 0,3 avec
  bge-m3). Un seuil absolu de 0,2 sélectionne donc peut-être quasiment rien, ou quelque
  chose de très différent de l'intention. Calibrer sur la distribution empirique des
  similarités label-label (percentile), pas sur une constante.
- `npmi_threshold=0.3`, `min_freq=0.01`, `max_freq=0.5`, `neg_quantile=0.05`,
  `sigma_clip=4.0`, `dead_steps_threshold=200`, `AUX_ALPHA=1/32` : lister toutes les
  constantes magiques, dire pour chacune si elle vient d'un papier (avec référence) ou
  d'un choix local, et pour les choix locaux, montrer une sensibilité.

### B.17 — Reproductibilité (P1, Palier 1)

Le module `random` de la bibliothèque standard est utilisé pour des décisions qui
comptent : mélange des exemples de l'odd-one-out, tirage du contrôle négatif,
échantillonnage des documents dans `feature_selection_by_magnitude`. Vérifie qu'un
`random.seed(SEED)` est bien posé **au début de chaque point d'entrée** (`saev5.py` et
chaque script d'audit) ; sinon, les 31,3 % d'instabilité au réordonnancement documentés
dans le rapport sont en partie une instabilité de graine, pas seulement une sensibilité du
juge.

Vérifie de la même façon `np.random`, `torch.manual_seed`, et le déterminisme cuDNN.
Ajoute un test qui exécute deux fois une portion CPU du pipeline et compare les sorties
bit à bit.

### B.18 — Cas limites comptabilisés comme des échecs (P2, Palier 1)

Dans `odd_one_out_judge`, une feature avec moins de 3 exemples positifs reçoit
`interp_score = 0` et le label `dead_feature` ; une feature sans contrôle négatif reçoit
également 0. Ces cas devraient être **exclus du dénominateur**, pas comptés comme non
interprétables. Mesure combien de features sont concernées dans les runs publiés : si
c'est non négligeable, tous les taux sont sous-estimés d'un facteur connu.

De même, une réponse du juge non parsable donne `predicted = -1` → échec. Mesure le taux
d'échec de parsing ; s'il dépasse quelques pourcents, il faut soit robustifier le parsing,
soit — bien meilleur — **scorer les options par log-probabilité des tokens de chiffre**
plutôt que par génération libre. Cela supprime d'un coup le parsing, la sensibilité à la
température et une partie de la sensibilité à l'ordre.

### B.21 — Le résidu `x − x̂` est calculé par soustraction en bf16, potentiellement une annulation catastrophique sur les tokens à forte activation (P1, nouveau, Palier 1 pour le diagnostic, Palier 2 pour un éventuel correctif)

**Contexte vérifié (13/08, recherche web).** Les auteurs de Gemma Scope (v1, Gemma-2)
ont explicitement testé fp32 vs bf16 pour le SAE **et** le LM, et concluent à un impact
« négligeable » sur les courbes fidélité/parcimonie (leur Fig. 10) — Llama Scope cite le
même résultat pour justifier son propre usage de bf16. C'est un argument solide en faveur
du choix bf16 de ce projet **au niveau du SAE core seul** : pas d'alerte à ce niveau.

**Ce que cette validation ne couvre pas.** Cette architecture à deux étages (core gelé +
extension résiduelle, `FrozenCoreResidualSAE`) fait une opération que ni Gemma Scope ni
Llama Scope ne testent dans leur ablation de précision : soustraire deux tenseurs de
**grande magnitude et proches l'un de l'autre** (`x` et sa reconstruction `x̂` par le
core) pour obtenir un **résidu de petite magnitude relative**, puis entraîner un second
SAE *sur ce résidu*. C'est exactement le cas d'école de l'**annulation catastrophique**
en calcul flottant : l'erreur absolue de bf16 (~0,4 % de la magnitude de l'opérande, soit
~400 pour un token à activation massive d'ordre 1e5) ne disparaît pas dans la
soustraction — elle reste, alors que le résultat qu'on cherche (le résidu) peut être du
même ordre de grandeur que cette erreur.

**Où ça se passe dans le code, vérifié ligne à ligne.** `saev5.py` charge le LM
directement en bf16 (`torch_dtype=TORCH_DTYPE`) — il n'existe **nulle part dans le
pipeline** une version fp32 de `x` à un stade postérieur à l'embedding : même le
`.float()` appliqué dans `activations.py::extract_residual_acts` n'est qu'un
élargissement de représentation, pas une récupération de précision. Puis, dans
`frozen_core.py` :
```python
def forward(self, x: torch.Tensor) -> dict:
    x_bf16 = x.to(torch.bfloat16)          # no-op, x est déjà bf16
    ...
    core_out = self.core_sae.decode(core_acts)   # bf16
    residual = (x_bf16 - core_out).float()        # soustraction EN bf16, puis élargie
```
La soustraction elle-même a lieu en arithmétique bf16 ; le `.float()` qui suit ne fait
qu'exprimer un résultat déjà arrondi dans un format plus large — aucun bit perdu n'est
récupéré. `_pre_extra` fait de même (`x.float()` sur un résidu déjà dégradé).

**Ce qui atténue le risque, sans le fermer.** Le `norm_outlier_mask` (σ_clip=4,0,
`saev5.py` L878-882) exclut déjà les tokens dont la norme est un outlier **intra-batch**
avant que leurs activations n'entrent dans le réservoir résiduel — les cas les plus
extrêmes (le token BOS, attention sink) ne contaminent donc pas l'entraînement de
l'extension. Et le FVE du core rapporté (`fve_pretrained = 0,831`, job 41375, §23.4)
indique que le résidu représente en moyenne ~17 % de la variance totale — pas une
poussière négligeable, donc probablement au-dessus du bruit d'arrondi *en moyenne*. Mais
ni l'un ni l'autre ne couvre le cas intermédiaire : un token à norme élevée mais **sous**
le seuil σ_clip=4 (donc conservé), sur lequel le core reconstruit relativement mal (donc
un résidu de grande magnitude absolue, mais peut-être pas plus grand que l'erreur
d'arrondi bf16 associée à une activation d'origine tout aussi grande).

**Diagnostic peu coûteux à faire avant tout correctif (Palier 1, CPU/GPU court, sur un
petit échantillon).** Comparer, sur quelques centaines de tokens déjà en cache
(`p1_eval_raw_tokens.pt` existe déjà, cf. `saev5.py` L1049 environ), le résidu calculé
tel quel (bf16 puis élargi) à un résidu calculé en réextrayant les mêmes tokens avec
`TORCH_DTYPE=fp32` pour le seul forward du LM (coûteux à l'échelle du corpus complet,
gratuit sur un échantillon de contrôle). Stratifier par norme de `x` (tokens dans la
plage 2-4σ vs <2σ). Si l'écart bf16/fp32 est petit devant la norme du résidu sur toute la
plage, le point se referme et peut être cité comme vérifié plutôt que supposé. S'il ne
l'est pas spécifiquement pour la strate 2-4σ, c'est un biais numérique réel, distinct de
et invisible à l'ablation de précision de Gemma Scope (qui ne teste jamais cette
architecture à deux étages) — un correctif possible serait de calculer `x − x̂` en fp32
même si `x` et le SAE core restent chargés/exécutés en bf16 pour le reste (upcast juste
avant la soustraction, downcast après si nécessaire pour la suite), ce qui ne change ni
le coût mémoire du LM ni celui du SAE core, seulement celui, marginal, de cette seule
opération.

### B.22 — Pipeline 2 : volume de convergence jamais diagnostiqué, justification de granularité incomplète (P1, nouveau, Palier 1)

Quatre points liés, tous peu coûteux à instruire, jamais couverts par l'audit jusqu'ici
parce que l'attention (B.1/B.19) s'est concentrée sur la Pipeline 1.

1. **Volume d'entraînement du `PhraseLevelSAE` jamais ablaté.** `MAX_PHRASES_DOC=20` ×
   ~41 176 documents d'entraînement (§55) donne un plafond théorique ~823k phrases,
   probablement moins en pratique. Pour `D_SAE=8192`, aucune vérification de convergence
   n'a été faite — contrairement à la Pipeline 1 (B.19), aucune ablation de volume
   n'existe pour cette pipeline. **Diagnostic gratuit** : `phrase_sae.py` a un historique
   par step depuis le début (antérieur au fix §54 qui a mis la Pipeline 1 à niveau sur ce
   point) ; `scripts/generate_diagnostic_plots.py` (rétroactif, zéro rerun) devrait déjà
   pouvoir produire les courbes de convergence pour tous les checkpoints Pipeline 2
   existants — vérifier si c'est déjà fait, sinon lancer l'agrégation (Palier 1, CPU).
2. **La justification "Pipeline 2 = point de comparaison à coût inférieur + granularité
   différente" (docs actuelles) est correcte mais incomplète.** La distinction réelle et
   plus défendable : la Pipeline 1 décompose des activations token-level brutes (avant
   tout pooling), la Pipeline 2 décompose des embeddings de phrase **déjà poolés** par
   F2LLM (dernier token ou CLS) — deux objets de nature différente, pas deux
   implémentations redondantes de la même idée. L'architecture from-scratch de la
   Pipeline 2 est en outre **forcée** (aucun SAE préentraîné n'existe pour F2LLM,
   contrairement à GemmaScope pour Gemma-3), pas un choix arbitraire. À reformuler
   explicitement en ces termes dans `02_architecture.md`/`01_etat_de_lart.md` — correctif
   de rédaction, Palier 1, aucune expérience requise.
3. **Max-pooling documentaire (Pipeline 2 comme Pipeline 1) : défendable en principe,
   jamais validé empiriquement.** Cohérent avec le choix de Jiang et al. (d_SAE=65536,
   même opération) pour un usage de classification/explication — mais en tension
   explicite avec Latent Terms, qui teste max vs sum et retient sum pour le retrieval
   (accumulation d'évidence répétée plutôt que détection de pic, cf. A.4). Les deux
   choix sont défendables pour des usages différents ; le vrai trou n'est pas le
   principe mais l'absence de contrôle du biais de longueur (B.9, déjà loggé) — citer
   explicitement le choix contraire de Latent Terms dans le rapport plutôt que de
   présenter max comme un défaut non questionné.
4. **Découpage en phrases plutôt qu'en mail entier : justification pratique probable,
   jamais rendue explicite.** `extract_f2llm_embeddings` utilise `max_length=128` —
   probablement trop court pour un mail entier (troncature sévère si utilisé tel quel en
   mail-level). Deux vérifications avant de citer ceci dans le rapport : (a) confirmer si
   128 est une limite réelle de F2LLM-v2-80M (fiche du modèle) ou un choix arbitraire du
   projet, ajustable ; (b) noter explicitement que `split_into_phrases` (regex simple,
   `\.\s+|\n\n`) encode chaque phrase **totalement indépendamment**, sans aucun
   contexte des phrases voisines — perte d'information inter-phrase réelle, à assumer
   plutôt qu'à laisser implicite. Si (a) confirme que F2LLM-v2 supporte un contexte
   nettement plus long que 128 tokens, une ablation mail-level directe (Palier 2, GPU
   léger, réutilise l'infra existante) trancherait si la perte de contexte inter-phrase
   coûte réellement quelque chose en aval (clf_acc, interprétabilité) — sinon la
   contrainte pratique (a) suffit à justifier le choix actuel sans expérience
   supplémentaire.

### B.23 — `augmentation.py` : garde-fou factuel asymétrique, variation de longueur possiblement corrélée à l'axe, orthogonalité des axes non vérifiée (P1, nouveau, Palier 1)

Trois points liés, tous vérifiables sur données déjà existantes (`augmented_mails.jsonl`
et son manifest parquet), aucun rerun de génération requis.

1. **`validate()` ne vérifie que l'omission de faits, jamais la fabrication.** La
   fonction compare les faits (téléphones, montants, dates, séquences ≥4 chiffres)
   présents dans le mail parent à ceux de la variante, et rejette si des faits du
   parent ont disparu. Elle ne vérifie jamais l'inverse : un fait halluciné par le LLM
   (à température 0,8) apparaissant dans la variante sans exister dans le parent
   passerait la validation sans être détecté. Correctif peu coûteux : ajouter la
   vérification symétrique (`_facts(variant) - _facts(parent)`), au moins comme
   diagnostic sur le corpus déjà généré (Palier 1, ne nécessite aucune régénération).
2. **La borne de longueur (ratio 0,4–2,5) laisse ouverte une corrélation axe↔longueur
   jamais vérifiée.** Se branche directement sur B.9 (biais de longueur du max-pooling)
   avec un mécanisme concret : si les instructions de certains axes (ex. `panique`,
   `colere_forte`) produisent systématiquement des textes plus longs que d'autres (ex.
   `calme`, `standard`), alors une partie de la "distinctivité" apparente d'un axe en
   classification aval pourrait refléter une différence de longueur plutôt qu'une
   différence de contenu. Diagnostic gratuit : longueur moyenne par axe/niveau,
   directement depuis le manifest parquet déjà produit.
3. **L'orthogonalité des axes ("une variante = un axe, perturbation isolée") est une
   intention du prompt, pas une propriété vérifiée du texte généré.** La colère et le
   registre informel, par exemple, sont naturellement corrélés dans l'usage de la
   langue — un LLM instruit d'isoler la colère peut dériver le registre comme effet de
   bord de ses propres régularités statistiques. `augmentation_lexical_leakage_audit.py`
   existe déjà et couvre une préoccupation voisine (fuite lexicale) mais pas
   spécifiquement la corrélation inter-axes — vérifier s'il la couvre implicitement,
   sinon l'ajouter comme diagnostic (mesurer par exemple si des marqueurs de registre
   informel apparaissent plus souvent dans les variantes `emotion__colere_forte` que
   dans `emotion__satisfaction`, à corpus autrement comparable).

### B.24 — Le module `src/sae/compare/` (détection de pollution, alignement cross-lingue) est entièrement implémenté et jamais exécuté (P1, nouveau, Palier 1/2)

`model_compare.py` implémente un détecteur de "pollution" de features assez sophistiqué :
appariement hongrois entre les SAE de deux modèles sur un corpus partagé
(`match_features`, corrélation de Pearson + `linear_sum_assignment`), score combinant
orphelinat (aucun équivalent dans l'autre modèle), désalignement de communauté NPMI (AMI
vs partition métier), et incohérence sémantique intra-communauté, avec test de
significativité par permutation. `crosslingual.py` réutilise ces primitives pour un
alignement FR/EN du même reader, plus des sondes logistiques sur (raw | acts SAE |
reconstruction) et le ΔCE (cf. B.20 corrigé).

**Aucun de ces fichiers n'est importé par un script ou par `saev5.py`** — vérifié par
recherche exhaustive. `scripts/feature_group_reproducibility_test.py` réimplémente même
sa propre version de l'appariement plutôt que d'importer `match_features` (doublon,
cf. C.2). C'est directement pertinent au récit central du rapport (une feature de
l'extension reflète-t-elle le domaine cible ou un artefact du préentraînement du
backbone ?) et le travail d'implémentation est déjà payé.

**Mise à jour (13/08) : un point d'entrée CLI complet existe déjà.**
`src/sae/compare/pipeline.py` (`python -m src.sae.compare.pipeline --mode compare
--model-a ... --model-b ...`) orchestre déjà bout-en-bout : embedding des deux modèles,
entraînement des deux `PhraseLevelSAE`, `compare_embedding_models`, export des rapports
de pollution et des visualisations. Ce n'est donc pas seulement des fonctions à assembler
(Palier 2 comme classé initialement) mais un script prêt à lancer — Palier 1 pour un
premier essai. Note : `OUT = Path("results_v9")` est un chemin de sortie codé en dur
d'une version antérieure, à corriger avant usage (risque de mélange avec un ancien cache,
cf. B.8).

À faire :
1. **(Palier 1)** Lancer `pipeline.py --mode compare` en comparant les features de
   l'extension du Pipeline 1 à celles du Pipeline 2 (deux backbones différents sur le
   même corpus, cf. B.22) — corriger d'abord `OUT` vers un chemin non ambigu.
2. Décider si le module doit rester une capacité exploitée (probable, vu sa pertinence
   directe) ou être documenté comme abandonné si le résultat s'avère peu concluant.

### B.25 — Les labels du core (Neuronpedia) et de l'extension (juge local) viennent de protocoles incommensurables (P1, nouveau, Palier 1)

`neuronpedia_labels.py` télécharge des explications produites par le pipeline
d'auto-interprétation propre à Neuronpedia/DeepMind — juge et méthodologie inconnus de ce
projet, très probablement calculés sur un corpus généraliste et majoritairement anglophone,
pas sur des mails EDF en français. Les labels de l'extension, eux, viennent du juge
odd-one-out local de ce projet, sur le corpus français cible.

Toute lecture du rapport qui met en regard "interprétabilité du core" et "interprétabilité
de l'extension" (même implicitement, par exemple en présentant les labels du core comme
une référence acquise contre laquelle le taux de 45,3 % de l'extension serait jugé) compare
deux processus de mesure différents, pas un simple avant/après sur le même protocole —
distinct des circularités déjà notées en B.2/B.3/B.4, qui portent toutes sur des variantes
du protocole odd-one-out lui-même. À rendre explicite dans le rapport (Palier 1, texte) :
les deux chiffres décrivent des objets mesurés différemment, ne pas les faire dialoguer
comme s'ils étaient comparables sans le dire.

### B.26 — `INTENT_KEYWORDS_FR` sous-compte systématiquement les labels faibles (P0, nouveau, Palier 1 — priorité la plus élevée de tout l'audit)

**Le résultat que le rapport lui-même désigne comme sa preuve la plus solide repose sur
des labels cassés.** `src/data/dataset.py::INTENT_KEYWORDS_FR` définit chaque intention
par un motif `\b(radical)\b` — `\b` est une frontière de MOT ; encadrer un radical des
deux côtés exige un mot isolé complet, pas un préfixe. Vérifié empiriquement (voir
`docs/AUDIT_2026-08.md` pour le script de reproduction) :

```
\b(contest)\b        MISS 'je conteste cette facture'     MISS 'voici ma contestation'
\b(r[ée]sili)\b       MISS 'je souhaite résilier'          MISS 'demande de résiliation'
\b(rembours)\b        MISS 'je demande un remboursement'   MISS 'vous devez me rembourser'
\b(renseign)\b        MISS 'merci de me renseigner'        MISS 'un renseignement'
\b(imm[ée]diat)\b     MISS 'répondez immédiatement'        MISS 'immédiate SVP'
```

Cinq des formulations les plus naturelles pour exprimer exactement ce que chaque
intention est censée capter, aucune ne matche. Seuls les radicaux qui sont eux-mêmes des
mots complets (`urgent`, `avoir`, `coupure`, `information`) échappent au bug — la
majorité des motifs du dictionnaire en sont affectés.

**Correctif validé** (`\b(radical\w*)\b` au lieu de `\b(radical)\b`, ou retrait du `\b`
final) : récupère la quasi-totalité des cas testés. Trivial, Palier 1, recalcul sur texte
déjà en cache, aucun GPU.

**Rayon d'impact, confirmé par recherche exhaustive** — `intent_*` (dérivées uniquement
de `INTENT_KEYWORDS_FR`) est consommé par :
- `scripts/intent_urgency_probe.py` — le résultat cité par `06_conclusion.md` comme
  *« la preuve la plus fiable des deux, non affectée par cette réserve »* (+27,0 points
  urgence, +42,6 points réclamation). **C'est justement cette réserve d'un autre ordre qui
  s'applique ici.**
- `scripts/explanation_fidelity_test.py`, `scripts/steering_fidelity_test.py` — même
  régime de labels faibles, explicitement cité comme tel dans leurs docstrings.
- `scripts/latent_retrieval_precision_eval.py` — vérité terrain prévue pour l'évaluation
  Latent Terms (F.1, A.4).
- `src/sae/compare/pipeline.py` — comparaison de pollution inter-modèles (B.24).

Un label systématiquement sous-compté ne réduit pas seulement `n_pos` : il biaise
l'échantillon de positifs vers les rares documents utilisant la forme radicale exacte
non fléchie plutôt que les formulations naturelles — un sous-ensemble non représentatif
de ce que l'intention est censée mesurer. C'est la même classe de problème que B.2
(sélection non représentative) et B.19 (label faible non validé), mais appliquée ici au
résultat que le rapport présente lui-même comme son point le plus solide.

**À faire, dans l'ordre** :
1. Corriger `INTENT_KEYWORDS_FR`, recalculer les colonnes `intent_*` sur le corpus déjà
   en cache — gratuit, quelques minutes.
2. Mesurer l'écart de `n_pos` avant/après correctif pour chaque intention — si l'écart
   est important (probable au vu des exemples ci-dessus), c'est déjà un résultat à
   documenter en soi (ampleur du sous-comptage).
3. Rejouer `intent_urgency_probe.py` avec les labels corrigés sur les activations déjà en
   cache (zéro GPU, cf. docstring du script lui-même) — comparer le delta corrigé au
   delta actuellement publié (+27/+42,6 points). Si le delta se maintient sous les labels
   corrigés, le résultat sort renforcé (véritable positif, pas un artefact de sélection) ;
   s'il s'effondre, c'est une correction majeure à apporter au chapitre 3 et à la
   conclusion avant toute autre chose.
4. Propager le correctif aux trois autres consommateurs listés ci-dessus avant de leur
   faire confiance pour quoi que ce soit (en particulier F.1, qui prévoyait déjà de
   s'appuyer sur ces labels comme vérité terrain de retrieval — à ne pas lancer avant ce
   correctif).

### B.27 — Le test « groupement de features améliore la reproductibilité inter-seed » n'a pas de témoin (P1, nouveau, Palier 1)

`scripts/feature_group_reproducibility_test.py` compare, entre deux graines
d'entraînement (42 vs 123), l'appariement **feature-à-feature** de labels interprétables
(par similarité cosinus d'embedding) à l'appariement **groupe-à-groupe** (centroïdes de
communautés Louvain formées par similarité de label). Un test de Mann-Whitney conclut
que le regroupement améliore significativement la similarité si le groupe bat la feature.

**Ce test ne peut quasiment pas produire de résultat négatif.** Moyenner plusieurs
vecteurs (les features d'un même groupe) avant de comparer leur similarité cosinus à
d'autres centroïdes **augmente mécaniquement la similarité par réduction de variance** —
un effet géométrique pur, indépendant de toute structure sémantique réelle. Le test tel
qu'écrit ne distingue pas « le regroupement capture une vraie robustesse conceptuelle »
de « moyenner des vecteurs les rapproche toujours un peu plus ».

**Correctif (Palier 1, sur les mêmes données déjà en cache)** : ajouter un témoin —
regrouper les mêmes features **au hasard**, en groupes de taille comparable (même
distribution de tailles que les communautés Louvain réelles), calculer la même
similarité centroïde-à-centroïde sur ce regroupement aléatoire, et comparer les trois
distributions (feature-à-feature, groupe réel, groupe aléatoire). Si groupe réel > groupe
aléatoire significativement, la conclusion tient ; sinon, l'effet observé n'est que
l'artefact du moyennage.

### B.28 — Cinq scripts de validation du juge partagent un biais de rééchantillonnage du contrôle négatif, correctif quasi gratuit disponible (P1, nouveau, Palier 1)

`build_feature_examples_with_control` (`src/sae/judge.py`) sélectionne l'exemple négatif
via `random.shuffle(neg_pool)` — non déterministe d'un appel à l'autre pour une même
feature. **Cinq scripts rappellent cette fonction à neuf puis comparent leur résultat au
score original mis en cache**, sans garantir que l'exemple négatif reconstruit est
identique à celui ayant produit le score d'origine :

- `multilingual_judge_bias_test.py` — compare interp_score FR (original, cache) à
  interp_score EN (traduit, exemples reconstruits) : une bascule peut refléter un
  changement de contrôle négatif, pas un effet de langue.
- `judge_robustness_check.py` — compare interp_score single-shot (original, cache) au
  vote majoritaire sur N répétitions (exemples reconstruits une fois, ordre varié) :
  même confond entre l'original et le point de départ des répétitions.
- `judge_sampling_ensemble_test.py`, `contrastive_labeling_test.py` — même schéma.
- `c2_original_only_rejudge.py` — réimplémentation locale plutôt qu'appel à la fonction
  partagée (cf. C.2), même risque si sa propre construction de contrôle négatif n'est
  pas alignée avec l'original.

**Correctif quasi gratuit** : `odd_one_out_judge` sauvegarde déjà `pos_examples` et
`neg_example` dans le cache JSON produit (`p1_judge_labels_extended.json`). Ces cinq
scripts peuvent **charger les exemples exacts d'origine** au lieu de les reconstruire —
élimine le confond sans calcul supplémentaire. Vérifier d'abord (Palier 1, quelques
minutes) que ces champs sont bien présents dans le cache réel produit par les runs
existants, pas seulement dans le dict retourné en mémoire par la fonction ; sinon,
fixer une graine locale déterministe par feature (`random.Random(f_idx)`) dans
`build_feature_examples_with_control` comme solution de repli.

---

## 3. Axe C — Hygiène du code, redondances, commentaires

### C.1 Fichiers et code morts

- `src/sae.egg-info/` est **versionné** (PKG-INFO, SOURCES.txt…) : à retirer du suivi et
  à ajouter au `.gitignore`. Son PKG-INFO contient un ancien README en anglais avec des
  sections périmées (« Issue: Dimension Mismatch in `downstream_classification()` ») qui
  contredisent la documentation actuelle.
- `src/storage/shards.py` : `save_activations_sharded` et `load_activations_mmap` ne sont
  appelées nulle part → module entièrement mort (38 lignes). Confirmer puis supprimer, ou
  documenter pourquoi le garder.
- `src/analysis/plotting.py` : `plot_correlation_heatmap` et
  `plot_judge_agreement_histogram` jamais appelées.
- `test_chargement_sae.py` à la racine et `scripts/test_massive_acts.py` : scripts de
  diagnostic manuel dont les noms commencent par `test_` — ils seront collectés par
  pytest ou créeront de la confusion avec `tests/`. Renommer (`diag_*.py`) ou déplacer.
- Recherche systématique attendue : fonctions publiques jamais importées, branches
  `except ImportError` mortes, paramètres de fonction jamais passés autrement que par
  défaut, variables de configuration jamais lues.
- **Nouveau (13/08)** : `src/sae/compare/` (`model_compare.py`, `crosslingual.py`)
  entièrement non importé — cf. B.24, qui recommande de l'exploiter plutôt que de le
  supprimer, vu sa pertinence directe pour le rapport. Traiter comme du code à
  activer, pas à retirer, sauf décision contraire explicite.
- **Nouveau** : `src/storage/fragment_store.py::doc_maxpool` est une **seconde**
  implémentation de max-pooling (en espace CSR), alors que `src/analysis/activations.py`
  revendique dans son propre docstring être l'« implémentation UNIQUE » du max-pooling
  SAE. Les deux ne produisent le même résultat que parce que les activations SAE sont
  non-négatives par construction (l'une initialise à `-inf` puis corrige, l'autre à `0`
  avec `include_self=True`) — équivalence correcte mais jamais énoncée. Corriger le
  docstring d'`activations.py` (ne plus revendiquer l'unicité) et documenter
  explicitement l'hypothèse de non-négativité partagée par les deux implémentations,
  ou factoriser en une seule fonction acceptant les deux formats d'entrée.
- **Nouveau, mineur** : le docstring de `fragment_store.py::_dense_to_csr` (et le
  module dans son ensemble) annonce un stockage `vals` en `float16`, le code stocke en
  `float32` (`vals = acts[rows, cols].to(torch.float32)`). Écart doc/code à corriger
  dans un sens ou l'autre — vérifier au passage si le gain de stockage annoncé
  (~250 Ko/doc) reste valide avec float32 ou doit être révisé.

### C.2 Redondances

- Le protocole de jugement est réimplémenté ou re-paramétré dans **7 scripts**
  (`judge_robustness_check`, `judge_sampling_ensemble_test`, `judge_model_separation_test`,
  `multilingual_judge_bias_test`, `c2_original_only_rejudge`, `contrastive_labeling_test`,
  `explanation_plausibility_test`), avec des prompts en dur redéfinis localement. Un
  changement de prompt dans `judge.py` ne se propage donc pas aux tests qui prétendent
  mesurer sa robustesse. Extraire les prompts dans un module unique versionné
  (`src/sae/prompts.py`) avec un identifiant de version du prompt enregistré dans chaque
  JSON de résultats.
- `scripts/core_vs_extension_ablation.py` réimplémente « le même protocole que
  `downstream_classification` » : toute correction de B.6 devra être faite deux fois.
  Factoriser.
- Le double système d'imports (`try: from src.x import y / except ImportError: from y
  import z`) est présent dans presque tous les fichiers. Il masque de vraies erreurs
  d'import (cf. le submodule `interp_embed` vide, jamais détecté). Proposer un
  packaging unique (`pip install -e .` + imports absolus) et supprimer les fallbacks.

### C.3 Commentaires

Trois pathologies, à traiter différemment :

1. **Commentaires-journal** : `# fix B1`, références à des numéros de jobs SLURM
   (`jobs 38988/38999/39000`), « constaté sur un run 270m avant correction », « jamais
   câblé ici avant ce fix ». Ils datent le code et ne servent qu'à qui a vécu la session.
   → réécrire au présent en énonçant la contrainte, pas l'incident. L'historique est dans
   git.
2. **Docstrings périmées** : `src/sae/judge.py` commence par
   `"""judge_patch.py — Remplace les fonctions de labellisation dans saev5.py"""` suivi
   d'instructions d'intégration (« Copier ces fonctions à la place des anciennes… »).
   Le fichier n'est plus un patch depuis longtemps. → réécrire.
3. **Commentaires qui justifient un choix scientifique** (bf16, pooling, découplage des
   seeds, choix du hook) : ce sont les plus précieux, **à garder**, mais à déplacer vers
   `docs/` quand ils dépassent 5 lignes, en laissant un renvoi d'une ligne dans le code.

Vérifie enfin la **véracité** de chaque commentaire affirmant un chiffre ou un
comportement (« 87,8 % de couverture », « d_in=640 confirmé empiriquement », « DBCV 0,851
vs 0,829 ») : un commentaire faux est pire qu'absent. Marque `À VÉRIFIER` ceux que tu ne
peux pas confirmer.

### C.4 Tests

`tests/` contient 6 fichiers, tous CPU. Manquent, par ordre d'utilité :
- un test qui vérifie qu'un changement de paramètre change la clé de cache (B.8) ;
- un test d'invariance du pooling (permutation des tokens d'un document ⇒ même
  `doc_vec`) et un test de la **dépendance à la longueur** (B.9) ;
- un test qui vérifie que `FrozenDecoderExtendedSAE.W_dec_extra` est bit-à-bit inchangé
  après N steps (le no-op de `normalize_decoder` prétend le garantir) ;
- un test de bout en bout sur un mini-corpus jouet avec une feature **synthétiquement
  injectée** (le dépôt le fait déjà pour `find_interesting_pairs`, §40 : généraliser ce
  principe au juge — injecter un concept connu et vérifier que le protocole le retrouve.
  C'est le seul moyen d'obtenir une borne supérieure du taux d'interprétabilité
  atteignable par ce protocole, information aujourd'hui absente et qui changerait la
  lecture de tout le chapitre 3).

---

---

## 4. Axe F — Pivot stratégique : reproduction fidèle comme baseline externe (Palier 2)

Contexte, pour toi seulement (à ne pas recopier tel quel dans le rapport) : l'audit
ci-dessus identifie plusieurs constats qui tiennent tous à la même cause profonde —
l'absence de tout point de comparaison externe vérifiable. Pas de baseline pour Latent
Terms (A.4), `interp_embed` jamais réellement installé ni exécuté (A.2), un protocole de
jugement odd-one-out dont la validité elle-même est mise en doute (B.3), et un résultat
phare confondu par la circularité extracteur/juge (B.4). Plutôt que de rustiner chacun de
ces points séparément, l'option retenue est de **reproduire fidèlement deux méthodes de
référence, appliquées au corpus EDF plutôt qu'aux benchmarks originaux des articles**, et
de les utiliser comme **baseline externe validée** contre laquelle comparer le pipeline
maison.

Ce n'est **pas** une reproduction à l'échelle des papiers (30 G tokens et 5 graines pour
Latent Terms, Llama-3.3-70B comme juge pour le toolkit de Jiang et al., évaluation sur
BEIR/MS MARCO ou LMSYS-Chat-1M) : une reproduction à cette échelle serait à la fois hors
de portée en GPU-heures et hors du mandat du stage, qui porte sur les mails EDF, pas sur
des benchmarks IR publics. Il s'agit d'une **implémentation fidèle de la méthode**,
évaluée à l'échelle que permet un stage, **sur les données réelles du stage**.

Deux propriétés en font la meilleure dépense de temps disponible sur les deux semaines
qui restent avant l'entretien :

1. **Latent Terms ne dépend d'aucun juge LLM.** C'est une métrique de retrieval
   objective (Precision@k, nDCG) — elle contourne entièrement la circularité
   extracteur/juge (B.4) au lieu d'essayer de la corriger, et elle est bon marché : le
   backbone n'a pas besoin d'être Gemma-3-12B.
2. **Le toolkit de Jiang et al. réutilise l'infrastructure déjà en place** (corrélations
   NPMI, clustering ciblé, retrieval par propriétés sont déjà implémentés dans ce
   dépôt) — l'essentiel du travail est de corriger le protocole de labellisation, pas
   de reconstruire un pipeline.

**Résultat attendu au chapitre 3** : la question cesse d'être « notre protocole maison
donne-t-il un chiffre plausible ? » (fragile, comme le montre l'axe B) pour devenir
« notre approche domaine-adaptée bat-elle une reproduction fidèle de deux méthodes de
référence, sur le même corpus, avec les mêmes contrôles ? » — une question mieux posée,
et dont chaque chiffre a une source externe vérifiable.

### F.1 — Latent Terms sur corpus EDF (Palier 2)

**Objectif.** Une baseline de retrieval sparse, fidèle à Clavié et al. (cf. A.4), évaluée
sur les mails EDF, sans aucun juge LLM.

**Protocole fidèle, adapté à l'échelle du stage** :
- Backbone gelé : réutiliser F2LLM-v2 (déjà en place, Pipeline 2) plutôt que Contriever/
  Nomic/GTE-ModernColBERT — écart assumé par rapport au papier, à documenter comme tel.
- SAE entraîné sur les activations **token-level** du backbone (pas phrase-level comme
  aujourd'hui — c'est le premier écart de A.4 à corriger ici), sur un corpus
  **généraliste** distinct des mails (le corpus energy/sports/support déjà présent dans
  le dépôt, ou un sous-échantillon de FineWeb2 déjà utilisé ailleurs) — jamais sur les
  mails eux-mêmes, contrairement à `src/sae/retrieval/latent_terms.py` aujourd'hui.
- Top-K SAE (pas BatchTopK, pour rester fidèle à Gao et al. tel qu'utilisé dans le
  papier), sum-pooling documentaire (pas max — déjà le cas dans ce module), ϕ(u) = √u,
  BM25 k1=8/b=0,7 — réglages déjà corrects, à garder.
- Volume : le papier rapporte lui-même une **robustesse** au volume de tokens et au
  top-k — s'appuyer sur ce constat, cité explicitement, pour justifier un budget réduit
  (quelques dizaines de M de tokens plutôt que 30 G) plutôt que de le faire en silence.
- Baseline de comparaison : **le backbone dense lui-même** (cosinus sur les embeddings
  F2LLM non décomposés), pas un TF-IDF avec des requêtes paraphrasées qui le handicapent
  par construction — c'est le point qui manque aujourd'hui (A.4) et qui rend le résultat
  interprétable.
- Vérité terrain : si le temps le permet, annoter manuellement un petit lot de
  requêtes/mails plutôt que de réutiliser exclusivement le label faible par regex
  (`INTENT_KEYWORDS_FR`), qui biaise la mesure vers le lexical.

**Réutilisable tel quel, sans modification** : `src/sae/retrieval/latent_terms.py`
(scoring BM25, classe `LatentTermsIndex`).

**Principe strict (13/08) : F.1 est une piste additive, ne modifie aucun chemin de code
dont dépend un résultat déjà publié.** `extract_f2llm_embeddings` sert aujourd'hui au
pooling phrase-level de la Pipeline 2, déjà citée dans le rapport — ne pas "l'adapter"
en place (risque de régression silencieuse sur des résultats déjà écrits). Créer à la
place une fonction **nouvelle et indépendante**
(`extract_f2llm_token_embeddings` ou équivalent, quelques dizaines de lignes dupliquées
plutôt que la fonction existante modifiée) qui retourne les activations token-level.
Même logique pour tout autre point de contact avec le code partagé : copier, ne jamais
modifier en place tant que le résultat de F.1 n'est pas validé et intégré
délibérément.

**Nouveau** : extraction d'activations token-level du backbone F2LLM (proche de ce que
fait déjà `activations.py` pour Gemma-3, à adapter à un backbone plus léger),
entraînement d'un Top-K SAE dédié sur corpus généraliste, baseline dense.

**Coût.** Backbone F2LLM-v2-80M, pas de Gemma-3-12B : l'essentiel du coût est
l'entraînement du SAE, du même ordre que les dizaines d'ablations déjà tournées sur le
Pipeline 2. Quelques jours, GPU léger, pas de risque disque particulier.

### F.2 — Toolkit de Jiang et al. sur corpus EDF (Palier 2)

**Objectif.** Rejouer, sur le SAE Pipeline 1 déjà entraîné (activations en cache), le
protocole de labellisation **contrastive directe** (top-activating vs bottom-activating,
cf. A.2) à la place de l'odd-one-out, et comparer les deux sur le même échantillon de
features avec appariement (test de McNemar).

**Protocole fidèle.** Pour chaque feature : prendre les documents de plus forte et de
plus faible activation (pas un mot marqué au milieu d'un document choisi au hasard comme
aujourd'hui, cf. B.3), demander au juge de décrire le concept commun aux top-activants,
vérifier séparément sur les bottom-activants qu'ils ne l'illustrent pas.
`scripts/contrastive_labeling_test.py` est structurellement proche de ce protocole —
vérifier s'il le suit fidèlement ou s'il en est une variante, et le corriger plutôt que
d'écrire un nouveau script.

**Réutilisable tel quel** : le graphe de co-occurrence NPMI (`cooccurrence.py`), le
clustering ciblé et le retrieval par propriétés (`select_latents_by_similarity`,
dashboard) — ces briques suivent déjà l'esprit du toolkit ; seule la labellisation est à
corriger.

**Nouveau** : validation croisée odd-one-out vs contrastif sur le même échantillon de
features (McNemar) ; et, si le temps le permet, un test de fidélité minimal en clonant
réellement `interp_embed`, en faisant tourner son exemple jouet, et en vérifiant que
notre réimplémentation produit des résultats cohérents sur ce cas de contrôle — plutôt
que de laisser le submodule vide (A.2).

**Décision explicite (13/08) : réimplémenter le protocole, ne pas adopter le package
`interp_embed` comme dépendance.** Leur code est couplé à leur setup exact (SAE Goodfire,
une partie de leurs baselines utilise l'API OpenAI `text-embedding-3-large`) ; l'adapter à
notre backbone (Gemma-3 + GemmaScope-2, y compris l'option 27B de B.4 point 3) demande une
lecture de leur code dont le coût d'intégration est inconnu à l'avance — risque
d'ingénierie non borné, à éviter sous cette contrainte de calendrier. Le seul usage
légitime du package tel quel reste le test de fidélité borné ci-dessus (leurs données
publiques, leur exemple jouet, aucune adaptation requise).

**Ce que F.2 résout d'un coup** : A.2 (comparaison à interp_embed enfin réelle, pas
seulement à sa lecture), une partie de B.3 (un protocole alternatif validé, en regard de
l'odd-one-out corrigé).

**Coût.** Re-labellisation sur activations déjà en cache, inference-only, même ordre de
grandeur que les campagnes de jugement déjà réalisées. Quelques jours, GPU modéré.

**Note vérifiée (13/08, recherche web) — le SAE de Jiang et al. est réellement
open-weight, pas seulement API.** Leur SAE (Goodfire, couche 50 de Llama-3.3-70B-Instruct,
dictionnaire 65 536, L0≈121) est publié sur HuggingFace
(`Goodfire/Llama-3.3-70B-Instruct-SAE-l50`, licence Llama 3.3 Community, page active,
poids téléchargeables) — pas seulement accessible via l'API Ember qu'ils ont utilisée par
commodité (leur footnote sur la limite de contexte 2048 décrit leur choix d'accès, pas
une restriction des poids). Ça ouvre une option de reproduction **plus fidèle** que celle
décrite plus haut : télécharger le vrai SAE Goodfire et faire tourner Llama-3.3-70B-Instruct
localement sur le corpus EDF, au lieu de substituer le SAE Pipeline 1 (GemmaScope/Gemma-3)
comme proposé ci-dessus. Entièrement on-prem, donc compatible avec la confidentialité des
mails EDF (contrairement à l'API Ember, qui enverrait le contenu des mails à un tiers, et
qui de toute façon semble avoir été dépréciée début 2026 d'après une note de blog Goodfire
trouvée en recherche — à vérifier si cette option est envisagée).

**Coût de cette option, et pourquoi elle reste hors du périmètre F.2 par défaut.**
Llama-3.3-70B-Instruct en précision native (bf16/fp16) demande ~140 Go de VRAM rien que
pour les poids — **ne tient pas sur un seul H100 80 Go**. Deux voies : (a) quantification
4 bits (~35-40 Go), qui tient sur un seul H100 mais introduit un écart réel avec le
protocole d'origine — le SAE de Goodfire a été entraîné sur les activations du modèle en
précision native, pas quantifié, donc l'utiliser sur un modèle 4 bits est un usage non
validé par les auteurs, potentiellement une source de bruit supplémentaire ; (b) précision
native sur ≥2 GPU en parallélisme de tenseurs, ce qui demande un nouveau template SLURM
(tous les scripts existants du dépôt demandent `--gres=gpu:1`, aucun run multi-GPU n'existe
actuellement). Ni l'un ni l'autre n'est gratuit à mettre en place, pour un gain de fidélité
marginal par rapport à l'usage du SAE Pipeline 1 déjà local, déjà validé sur ce cluster, et
suffisant pour évaluer le protocole de labellisation contrastive en tant que tel. **Cette
option reste donc un stretch goal explicitement optionnel (Palier 3), pas une exigence de
F.2** — à ne considérer que si le temps restant après tout le reste (Axe F inclus) le
permet, et uniquement pour améliorer la fidélité de citation, pas pour obtenir un résultat
qui manquerait sinon.

## 5. Axe D — Expériences à mener, organisées par palier d'exécution

Cette liste reprend les items des axes A, B et F, ordonnés pour exécution — voir §0.4
pour la définition des paliers. **L'ordre entre paliers est strict ; l'ordre à
l'intérieur d'un palier est indicatif (valeur/coût).**

### Palier 0 — en tout premier, gratuit

| # | Expérience | Répond à | Coût | Pourquoi en premier |
|---|---|---|---|---|
| 0 | Audit rétroactif des caches par ablation | B.8 | S, CPU | Détermine si la campagne d'ablation existante est exploitable — conditionne notamment si l'item 9 (Palier 3) sera un jour faisable. |

### Palier 1 — corrections et recalculs sur cache, CPU, quelques jours au total

| # | Expérience | Répond à | Coût | Valeur |
|---|---|---|---|---|
| 0bis | **Corriger `INTENT_KEYWORDS_FR`, rejouer `intent_urgency_probe.py`** | B.26 | S, CPU sur cache | **Priorité absolue de tout l'audit** — corrige potentiellement le résultat que le rapport désigne lui-même comme sa preuve la plus fiable. À faire avant tout le reste de ce tableau. |
| 1 | CV group-aware + baseline brute | B.6 | S, CPU sur cache | Corrige le chiffre le plus cité après le taux d'interprétabilité. |
| 2 | Contrôle « mails originaux seulement » (circularité corpus) | B.4 (partiel) | S, CPU sur cache | Coût nul, borne déjà une partie de la circularité extracteur/juge sans attendre le croisement complet. |
| 3 | Requalification textuelle du volume d'entraînement | B.19 (partiel) | S, texte | Corrige l'affirmation la plus exposée du rapport sans dépendre d'un rerun. |
| 4 | Cohérence mutuelle du dictionnaire résiduel déjà entraîné | B.1 (partiel) | S, CPU sur checkpoint | Donne un premier indice sur la surcomplétude sans le balayage GPU complet. |
| 5 | Toutes les corrections d'hygiène et de métriques sur cache (B.5, B.9, B.10 partiel, B.11 fix, B.13-B.18, Axe C) | — | S–M, CPU | Nettoie le terrain avant le Palier 2 ; beaucoup de P0/P1 réglés à coût nul. |
| 6 | Diagnostic bf16 vs fp32 sur le résidu, échantillon de contrôle | B.21 | S, GPU court | Détermine si l'architecture à deux étages souffre d'annulation catastrophique sur les tokens à forte activation — non couvert par l'ablation de précision de Gemma Scope. |

### Palier 2 — GPU modéré, sur infrastructure déjà en place, plusieurs jours (**le cœur du budget disponible**)

| # | Expérience | Répond à | Coût | Pourquoi c'est la meilleure dépense |
|---|---|---|---|---|
| 6 | **Latent Terms sur corpus EDF (F.1)** | A.4 | M, GPU léger | Baseline objective sans juge LLM ; contourne B.4 au lieu de le corriger. |
| 7 | **Toolkit de Jiang et al. sur corpus EDF (F.2)** | A.2, B.3 | M, GPU modéré | Réutilise l'infra existante ; installe enfin une comparaison réelle à la littérature. |
| 8 | Rejugement sur échantillon uniforme + négatif dur | B.2, B.3 | M, GPU | Rend le taux comparable à la littérature. |
| 9 | Portage du protocole AutoInterp de Korznikov et al. (jeu de test tenu à l'écart) | A.3, B.2, B.3 | M, GPU | Rend nos chiffres comparables à la littérature au lieu d'être propres au dépôt. |
| 10 | Feature synthétique injectée dans le juge | C.4 | S, GPU court | Donne le plafond du protocole : si le juge ne retrouve pas un concept planté, 45 % n'est pas un plafond du SAE. |
| 11 | Fidélité par re-décodage | B.7 | M, GPU | Remplace un test vide par un vrai test. |
| 12 | Baseline « Extended SAE (init aléatoire) » et « Soft-Frozen Decoder » | A.1, A.3 | M, GPU | Baselines de la littérature jamais reproduites, coût modéré sur infra existante. |
| 12bis | Ajout de Gemma-3-27B-it comme 4ᵉ point de la courbe dose-réponse | B.4 (point 3) | M, GPU, un seul H100 | GemmaScope-2 couvre le 27B, ajout mécanique dans `_PRESETS`, tient sur un H100 (54 Go bf16) sans quantification ni multi-GPU. Renforce directement le résultat le plus cité du rapport. |
| 13 | ΔCE du LLM sur substitution des activations | B.20 | S, GPU | Métrique standard absente, peu coûteuse à ajouter. |

### Palier 3 — GPU lourd, en tout dernier, uniquement si le temps restant le permet après le Palier 2 entièrement terminé

**Ces expériences ne sont pas écartées : elles répondent aux questions scientifiquement
les plus importantes de l'audit.** Mais ce sont aussi les plus chères, les plus risquées
(disque déjà proche de la capacité, un job déjà en échec) et celles dont le résultat est
le moins garanti dans le temps restant. **Le rapport doit pouvoir être fini sans elles** —
à défaut de temps, elles se documentent comme limites assumées au chapitre 4 plutôt que
comme un rerun de dernière minute avant l'entretien.

| # | Expérience | Répond à | Coût | Condition avant de lancer |
|---|---|---|---|---|
| 13bis | Resoumission des jobs 42687/42688 (seed7/seed99 à 25 M) | B.19 | S, GPU, quasi gratuit | Aucun code à changer, échec précédent purement infra. Détermine si l'écart +8,7 pts à 25 M est réel avant d'engager le run 200 M. **À faire en Palier 1/2, pas en Palier 3.** |
| 14 | Exécution du run à volume réel (≥100 M tokens), memmap déjà corrigé | B.19 | L, GPU | Le bug racine (RAM anonyme) est corrigé (§54) mais jamais revalidé à cette échelle. Lancer seulement si l'item 13bis confirme un effet, ou si le temps le permet de toute façon. |
| 15 | Balayage `D_EXTRA` en régime réellement surcomplet | B.1 | L, GPU | Item 4 (Palier 1) donne un premier signal gratuit — ne lancer le balayage complet que si ce signal le justifie. |
| 16 | Plan croisé `D_EXTRA × volume` | B.1, B.19 | L, GPU, le plus cher de l'audit | À n'envisager qu'après 14 et 15 séparément, et seulement s'il reste du temps. |
| 17 | Croisement complet extracteur × juge sur {1b, 4b, 12b} | B.4 | L, GPU | Bloqué si l'item 0 (Palier 0) montre que les activations 1b/4b ont été purgées — vérifier avant tout. |

---

## 6. Axe E — Cherche ce que je n'ai pas vu

Les constats ci-dessus viennent d'une lecture statique de `config.py`, `frozen_core.py`,
`batch.py`, `sae_shared.py`, `activations.py`, `metrics.py`, `judge.py`, `phrase_sae.py`,
`latent_terms.py`, `cooccurrence.py`, `preparation.py`, et de survols de `saev5.py`,
`stats.py` et de quelques scripts. **Tout le reste est inexploré** : `dashboard.py`,
`augmentation.py`, `fragment_store.py`, `gemma_scope_loader.py`, `neuronpedia_labels.py`,
`compare/`, `dataset.py`, et la vingtaine de scripts d'audit.

Applique-leur les mêmes grilles de lecture, qui sont plus générales que les constats
qu'elles ont produits :

1. **Un protocole peut-il produire un résultat négatif ?** Si non, il ne mesure rien
   (cf. B.7). Passe chaque script `*_test.py` / `*_audit.py` à cette question.
2. **Le traitement et le témoin diffèrent-ils par *une seule* chose ?** Toute baseline
   qui diffère par deux paramètres est ininterprétable (cf. A.3, B.4).
3. **La statistique rapportée est-elle conditionnée à une sélection ?** Taux calculés sur
   un top-N, corrélations calculées sur les cas de succès, moyennes sur les runs qui ont
   abouti (cf. B.2, B.5).
4. **Les unités d'observation sont-elles indépendantes ?** Variantes d'un même mail,
   tokens d'un même document, features d'un même dictionnaire (cf. B.6, B.11, B.14).
5. **Deux chiffres affichés côte à côte portent-ils sur le même objet ?** (cf. B.13.)
6. **Une clé de cache encode-t-elle tout ce dont dépend son contenu ?** (cf. B.8.)
7. **Une quantité dépend-elle mécaniquement d'une variable de nuisance ?** Longueur,
   dimension, taille de batch, nombre de tokens (cf. B.9, B.10, B.15).
8. **Le code fait-il ce que le commentaire affirme ?** Vérifie par échantillonnage, en
   priorité sur les commentaires qui contiennent un chiffre.
9. **Le chiffre du rapport est-il celui que produit le code aujourd'hui ?** Re-dérive
   depuis les JSON de résultats plutôt que de faire confiance au texte.

Consigne toute nouvelle hypothèse dans la section « Constats non listés » avec le même
formalisme que les autres, y compris quand tu conclus `INFIRMÉ` — une hypothèse écartée
avec preuve a de la valeur.

---

## 7. Ce que tu ne fais pas

- Tu ne modifies pas le code, ni le rapport, ni les JSON de résultats pendant cet audit.
  Les corrections feront l'objet d'une passe ultérieure, priorisée à partir de ton rapport.
- Tu ne lances pas de job GPU sans me demander : le budget cluster est limité et le disque
  partagé est souvent proche de la capacité (`df -h .` avant tout cache volumineux).
- Tu ne conclus pas « conforme à la littérature » sans avoir lu la section précise de
  l'article. « Cohérent avec ce que je sais du domaine » se note `INDÉTERMINÉ`.
- Tu ne minimises pas un constat parce qu'il invaliderait un résultat déjà rédigé. C'est
  précisément l'inverse qui est demandé : le rapport peut être corrigé, un résultat faux
  publié ne peut pas l'être.
- **Tu ne mélanges pas les paliers.** Un item de Palier 3 (§0.4, §5) ne se lance pas « en
  avance » ou « en parallèle pour gagner du temps », même s'il te semble scientifiquement
  plus important qu'un item de Palier 1 ou 2. L'ordre est Palier 0 → 1 → 2 (Axe F inclus)
  → 3, strictement, et le Palier 3 ne démarre que si tout le reste est terminé et qu'il
  reste du temps avant l'échéance. Si un item de Palier 3 te paraît urgent, dis-le-moi et
  attends ma confirmation avant de le remonter — ne décide pas unilatéralement de
  réordonner.

  # Audit — Round 2 (13/08, après-midi)

> Ce document est un **complément**, pas un remplacement de
> `AUDIT_SCIENTIFIQUE_CODE.md` (round 1, déjà lu et en cours d'exécution).
> Ne fusionne pas ce fichier dans le round 1 — les identifiants `B.N` cités
> ici renvoient à ceux du round 1 et doivent le rester. Traite ce document
> comme tu traiterais une nouvelle section reçue en cours de route : les
> priorités qu'il fixe (§0 ci-dessous) l'emportent sur l'ordre de paliers
> du round 1 pour les items qu'il touche explicitement.

## 0. Priorité immédiate : le batch Palier 1 a un bug dans son propre correctif

**B.26 doit être rouvert. La conclusion « impact négligeable » du dernier rapport
(`audit_palier1_batch_results.json`) est invalide — pas la donnée, la méthode qui l'a
produite.**

`audit_2026_08_palier1_batch.py` construit `FIXED_PATTERNS` par :
```python
FIXED_PATTERNS[k] = v[:-3] + r"\w*)\b"
```
Ce découpage de chaîne n'ajoute `\w*` qu'immédiatement avant la parenthèse fermante
finale du groupe — donc uniquement sur la **dernière** alternative du OR, jamais sur
les précédentes. Vérifié pour les cinq patterns :

```
reclamation    -> ...contest|inadmissible|scandaleux|erreur de facturation\w*)\b
resiliation    -> ...r[ée]sili|clôtur|mettre fin au contrat\w*)\b
remboursement  -> ...rembours|trop[- ]perçu|avoir\w*)\b
information    -> ...renseign|information|...|comment (faire|proc[ée]der)\w*)\b
urgence        -> ...urgent|imm[ée]diat|sans d[ée]lai|coupure\w*)\b
```

Dans les cinq cas, les radicaux que le round 1 avait démontrés cassés par des exemples
concrets (`contest`, `r[ée]clamation`, `r[ée]sili`, `clôtur`, `rembours`, `renseign`,
`imm[ée]diat`) **restent non corrigés**. Seule la toute dernière alternative de chaque
groupe (souvent une expression déjà peu affectée par le bug d'origine, ex.
"mettre fin au contrat", "avoir") a été modifiée. Le batch a donc mesuré l'effet d'un
correctif qui ne corrige presque rien — d'où les flips quasi nuls observés (0 à 3 sur
~44k documents), qui ne prouvent rien sur l'ampleur réelle du bug d'origine.

**Confirmation indépendante que l'effet réel est probablement important** : sur le
corpus d'origine, le radical seul « résiliation » apparaît environ 3000 fois, contre 19
occurrences de la phrase précise « résiliation de contrat ». Un écart de cet ordre entre
un radical nu et une expression qualifiée appelle deux hypothèses à trancher avant
d'aller plus loin, pas une seule :
1. Le radical capte massivement des usages qui n'expriment pas une demande de
   résiliation (texte répété, formulation administrative générique, un des axes
   d'augmentation, une signature ou un passage récurrent du corpus synthétique) — dans
   ce cas un correctif naïf sur/`résili\w*` produirait un `n_pos` gonflé et non
   représentatif, un problème symétrique et inverse à celui identifié en round 1.
2. Le radical capte majoritairement de vraies variations lexicales de la même intention
   (« résilier », « résiliable », « résiliation », conjugaisons et dérivés légitimes) et
   le sous-comptage d'origine était bien aussi sévère que redouté.

### À faire (Palier 1, CPU sur cache, avant toute autre chose sur B.26)

1. **Corriger le correctif.** Appliquer `\w*` à chaque alternative individuellement,
   pas par découpage de la chaîne globale — par exemple en parsant chaque pattern
   (split sur `|` à l'intérieur du groupe capturant, en laissant intactes les
   alternatives qui sont déjà des phrases/expressions à espaces comme
   « mettre fin au contrat », « trop[- ]perçu », « sans d[ée]lai », « erreur de
   facturation », « pourriez[- ]vous m'indiquer », et le groupe imbriqué de
   `information`) plutôt que par un raccourci générique. Écrire ce correctif à la main,
   alternative par alternative, est plus sûr qu'une transformation automatique tant que
   la structure des patterns n'est pas plus régulière.
2. **Avant de rejuger quoi que ce soit**, tirer un échantillon aléatoire d'une
   cinquantaine de documents contenant « résili » (radical nu) et les lire — combien
   correspondent réellement à une demande/mention de résiliation de contrat, combien
   sont du bruit (répétition, contexte non pertinent) ? Documenter la proportion avant
   de faire confiance à un `n_pos` corrigé.
3. Ne rejouer `intent_urgency_probe.py` avec les labels corrigés qu'une fois 1 et 2
   faits — le tableau produit par le round précédent (deltas et z-scores pour
   réclamation/urgence/information/remboursement) est probablement encore à peu près
   correct pour réclamation et urgence dans l'absolu (leurs radicaux dominants —
   « inadmissible », « scandaleux », « urgent », « coupure » — étaient déjà des mots
   complets non affectés par le bug d'origine, donc moins affectés par le bug du
   correctif aussi), mais à revérifier plutôt qu'à supposer, exactement le genre
   d'hypothèse plausible qui vient de se révéler fausse une fois.
4. Une fois 1-3 faits, refaire le test de significativité (McNemar via
   `src/analysis/stats.py`, cf. round 1 §7) sur les chiffres corrigés, pas sur une
   approximation à deux proportions ajoutée après coup.

---

## 1. Évaluation des vérifications déjà faites par Claude Code (B.5, B.11, B.14, B.17)

Sur la base du résumé fourni (679 lignes, 35 constats vérifiés à ce stade). Jugées avec
la même grille que le round 1 : preuve reproductible, capacité du test à produire un
résultat négatif, distinction entre ce qui est vérifié par lecture de code (binaire,
falsifiable directement) et ce qui est une inférence par analogie (à re-vérifier
empiriquement).

- **B.5 (ρ_interp)** — vérification par lecture de code, les trois sous-affirmations
  sont chacune une question binaire (rang vs magnitude ? négatif forcé à 0 ? calculé
  seulement post-sélection ?), directement tranchable en lisant `judge.py`. Type de
  preuve approprié pour ce type de constat, rien à ajouter.
- **B.11 (mélange de régimes BatchTopK/JumpReLU train/val)** — même remarque, et la
  nuance ajoutée (`model.eval()` change le *mécanisme* de parcimonie, pas seulement le
  split) est exactement le niveau de précision attendu d'une relecture sérieuse plutôt
  que d'une confirmation de surface. Bon signe méthodologique.
- **B.14 (Fisher apparié)** — la reclassification Palier 1 → Palier 2 après avoir
  constaté que les groupes ne sont pas structurés en paires 1:1 est le bon réflexe :
  reconnaître qu'un correctif « évident » ne s'applique pas tel quel à la forme réelle
  des données, plutôt que de forcer un remplacement de formule qui serait lui-même
  incorrect. À rapprocher directement de ce qui vient de se passer avec B.26 — un
  correctif qui a l'air trivial peut ne pas l'être, et c'est précisément ce que B.14 a
  vérifié avant d'agir, contrairement au correctif de B.26.
- **B.17 (seeds)** — le constat que 2 scripts sur 7 ne seedent jamais `random`
  eux-mêmes est vérifié par lecture de code, solide. **Mais la conclusion « les
  conclusions survivent probablement, le bruit est comparable à l'instabilité de
  réordonnancement déjà quantifiée » est une inférence par analogie, pas un test.**
  Deux sources de bruit différentes (position de l'intrus dans la liste vs tirage
  complet d'un nouveau jeu d'exemples/contrôle négatif à chaque exécution) n'ont pas
  nécessairement la même magnitude — la seconde peut être strictement plus grande
  puisqu'elle change plus de choses à la fois. Au vu de ce qui vient de se passer avec
  B.26 (une hypothèse plausible qui s'est révélée fausse une fois vérifiée), je
  recommande de ne pas laisser cette conclusion en l'état : relancer
  `c2_original_only_rejudge.py` et `judge_model_separation_test.py` avec 2-3 graines
  différentes (`random.seed` ajouté explicitement) et mesurer directement l'écart sur
  §52/§43, plutôt que de l'estimer par analogie. Coût : Palier 1-2, réutilise
  entièrement l'infrastructure déjà en place pour ces deux scripts.

**Verdict global sur ce premier lot** : trois vérifications sur quatre sont solides et
bien conduites (B.5, B.11, B.14). La quatrième (B.17) a une partie vérifiée (le
diagnostic) et une partie non vérifiée présentée avec une confiance un peu supérieure à
ce que la preuve actuelle justifie — à corriger avant de la considérer close, dans le
droit fil de ce qui vient d'arriver à B.26. Le principe à retenir pour la suite du
travail : **toute conclusion de la forme « probablement sans conséquence », « l'effet
est sûrement petit », doit être vérifiée avant d'être écrite comme un résultat, pas
après.** L'épisode B.26 en est la démonstration la plus nette possible.

---

## 2. « Est-ce normal qu'on n'y arrive pas avec toutes les intentions, qu'on ne fasse pas mieux que TF-IDF ? »

Réponse en deux temps, parce que ce sont deux questions de nature différente.

### Sur la variabilité entre intentions (remboursement sans signal, réclamation/urgence
très significatifs) : oui, c'est normal et attendu

Différentes intentions ne sont pas également « marquées » linguistiquement.
Réclamation et urgence ont un vocabulaire assez caractéristique et concentré
(« inadmissible », « scandaleux », « urgent », « immédiatement », « coupure ») — presque
n'importe quelle représentation raisonnable, lexicale ou sémantique, a de bonnes chances
de les séparer. Une demande de remboursement peut se formuler de façon beaucoup plus
diffuse, se mélanger conceptuellement avec une réclamation ou une simple demande
d'information, et porter une part de son signal dans des éléments numériques (montants,
références) que ni un sac de mots ni un SAE entraîné sur du texte n'est bien équipé pour
capter. Une performance très inégale selon la classe est la norme en classification
d'intentions, pas l'exception — c'est pour ça qu'on rapporte toujours des métriques par
classe plutôt qu'un seul chiffre agrégé.

### Sur le fait de ne pas dépasser TF-IDF : normal dans la littérature, mais il manque
un point d'analyse important dans ce projet précis

Deux choses distinctes se répondent ici.

**D'abord, ce n'est pas un résultat isolé ou gênant en soi** — c'est un phénomène bien
documenté : les représentations sac-de-mots sont réputées difficiles à battre sur des
tâches de classification de texte largement déterminées par la présence de mots-clés.
Plus spécifiquement pour ce projet, c'est même le type de résultat que prédit
explicitement Korznikov et al. (déjà cité au chapitre 1, cf. A.3 du round 1) : leur thèse
centrale est que des métriques de reconstruction/interprétabilité qui ont l'air bonnes ne
garantissent pas un avantage fonctionnel en aval. Un chapitre 4 qui dit « nos propres
résultats sont cohérents avec le constat sceptique de Korznikov et al. » est une position
scientifiquement plus forte et plus honnête qu'un chapitre qui minimise l'écart avec
TF-IDF ou l'omet.

**Ensuite, et c'est le point qui manque** : la vérité terrain elle-même
(`INTENT_KEYWORDS_FR`) est construite par mots-clés. Comparer un classifieur SAE à
TF-IDF sur une étiquette **elle-même définie par la présence de mots-clés** favorise
structurellement TF-IDF — ce n'est pas tout à fait un test loyal de la valeur sémantique
ajoutée du SAE, c'est en partie un test de circularité entre l'étiquette et la baseline
lexicale, de la même famille que B.2/B.25 déjà notés au round 1. Le test qui répondrait
vraiment à la question serait un jeu d'étiquettes **indépendant du regex** (annotation
humaine sur un petit échantillon, ou un jeu de données externe) — sans lui, l'écart
mesuré avec TF-IDF ne permet pas de trancher si le SAE apporte réellement moins qu'une
approche lexicale, ou si l'étiquette de référence est simplement elle-même de nature
lexicale.

### Nouveau constat à ajouter au corpus d'audit (identifiant proposé : B.29, Palier 1
pour le diagnostic, Palier 2 si une annotation manuelle est entreprise)

**La comparaison SAE vs TF-IDF est structurellement biaisée en faveur de TF-IDF tant que
la vérité terrain reste dérivée de `INTENT_KEYWORDS_FR`.** À faire :
1. Documenter ce biais explicitement dans le rapport, au chapitre où la comparaison
   TF-IDF est présentée (`03_experiences_et_resultats.md` §5.4) — reformulation textuelle,
   Palier 1, aucun calcul requis.
2. Si le temps le permet après le Palier 2 du round 1 : annoter manuellement un petit
   échantillon (100-200 documents) par intention, indépendamment du regex, et refaire la
   comparaison SAE vs TF-IDF sur cette vérité terrain propre. C'est le seul test qui
   peut vraiment répondre à la question posée dans ce message — et il peut aussi bien
   confirmer que réfuter l'écart actuel, ce qui est précisément ce qui en fait un bon
   test.

---

## 3. Instruction de méthode pour la suite

Le round 1 reste la référence pour tout ce qu'il couvre déjà (paliers, périmètre, ce
qu'il ne faut pas faire). Ce document round 2 :
- corrige B.26 (§0 ci-dessus, priorité immédiate, avant tout item de palier en cours),
- ajoute une réserve sur B.17 (§1, à vérifier plutôt qu'à laisser en l'état),
- ajoute B.29 (§2, Palier 1 pour la partie texte, Palier 2 pour l'annotation).

Pour tout round futur : même principe, un nouveau fichier plutôt qu'une réédition
silencieuse du précédent, avec renvoi explicite aux identifiants déjà établis.

# Audit — Round 3 (13/08, soir)

> Complément aux rounds 1 et 2, mêmes règles : ne pas fusionner, les identifiants `B.N`
> renvoient au round 1. Porte sur cinq scripts déjà écrits/exécutés
> (`audit_2026_08_delta_ce.py`, `audit_2026_08_bf16_fp32_diagnostic.py`,
> `audit_2026_08_frozen_decoder_scale_fix.py` + `..._rejudge.py`,
> `audit_2026_08_uniform_hardneg_rejudge.py`) et leurs résultats disponibles à ce stade.

## Verdict global

Travail honnête, aucune tentative de maquiller un résultat gênant, bonne discipline sur
plusieurs points déjà signalés en round 1/2 (noms de checkpoints dédiés, monkey-patch
plutôt que modification de module partagé, réutilisation exacte des mêmes features/volumes
pour isoler une seule variable à la fois). Un script (`uniform_hardneg_rejudge.py`) est
au niveau de rigueur attendu de bout en bout. Deux points appellent une correction avant
de considérer ce lot clos.

## 1. `audit_2026_08_bf16_fp32_diagnostic.py` (B.21) — le test n'a pas atteint sa propre hypothèse

**Constat.** Sur 40 documents / 14 235 tokens, la strate `au_dela_4sigma` (les tokens à
activation massive, ceux que `norm_outlier_mask` exclut du réservoir et ceux pour qui
l'annulation catastrophique serait la plus sévère) contient **0 token** — absente du JSON
de résultats, silencieusement sautée par le script (`if n_s == 0: continue`). Le
diagnostic n'a donc jamais mesuré la population qu'il était censé tester.

**Ce qui a été mesuré reste utile et doit être publié tel quel** : le résidu diverge de
~5-7% (bf16 vs fp32) contre ~0,35% pour `x` seul — facteur d'amplification ~20,
cohérent avec une annulation catastrophique réelle mais d'ampleur mesurée, pas dévastatrice
(moins de 0,3% des tokens dépassent 50% d'erreur relative sur le résidu). Ce résultat est
solide dans la plage qu'il couvre (<2σ, n=14151) ; la strate 2-4σ (n=84) est cohérente
avec lui mais trop petite pour trancher seule.

**À faire (Palier 2, GPU)** avant de considérer B.21 clos :
1. Augmenter drastiquement le nombre de documents échantillonnés, ou cibler
   délibérément les positions connues pour héberger des activations massives (début de
   séquence / voisinage du token BOS, cf. la littérature déjà citée dans le dépôt sur ce
   phénomène) plutôt qu'un tirage aléatoire de documents complets — un tirage aléatoire
   de documents n'est pas le bon plan d'échantillonnage pour un phénomène rare et
   positionnellement concentré.
2. Rapporter explicitement, dans la prochaine version du résultat, si la strate >4σ reste
   vide même après ce changement — dans ce cas, documenter la difficulté d'accès à cette
   population plutôt que de conclure silencieusement par son absence.
3. Ne pas énoncer dans le rapport final que « la divergence bf16/fp32 reste modérée y
   compris pour les tokens à activation massive » tant que ce point n'est pas réglé — la
   formulation correcte à ce stade est « modérée pour les tokens normaux (<4σ), non
   mesurée au-delà ».

## 2. Cohérence de l'usage de `src/analysis/stats.py` sur l'ensemble du lot

Seul `audit_2026_08_uniform_hardneg_rejudge.py` importe et utilise `stats.py`
(`proportion_with_ci`, IC de Wilson). `audit_2026_08_delta_ce.py` et
`audit_2026_08_bf16_fp32_diagnostic.py` rapportent des points estimés sans intervalle ni
test de significativité — même lacune que celle déjà relevée en round 2 pour le batch
Palier 1 (B.26/B.6), qui n'était donc pas isolée à ce script-là.

**À faire (Palier 1, CPU, sur les résultats déjà produits)** :
- `delta_ce.py` : recalculer le ΔCE par document (pas seulement l'agrégat), et faire un
  test apparié (Wilcoxon signed-rank ou équivalent sur `src/analysis/stats.py`) entre les
  conditions core-seul et core+extension sur les mêmes 60 documents — l'effet est large
  et probablement robuste, mais ce n'est pas encore démontré.
- `bf16_fp32_diagnostic.py` : au minimum un intervalle de confiance sur les moyennes par
  strate, plutôt que la moyenne/médiane seules.

## 3. Points positifs à noter explicitement (pas seulement des réserves)

- `delta_ce.py` : bug d'off-by-one de couche trouvé et corrigé dans
  `ce_loss_increase` avant utilisation (référencé E.6) — bonne vigilance. Recommandation
  associée : ajouter un sanity check indépendant de cette fonction fraîchement corrigée
  (patch par un SAE identité, vérifier ΔCE≈0) avant de s'appuyer dessus pour d'autres
  résultats à venir — une fonction qui vient d'être corrigée n'est pas encore une
  fonction validée, cf. ce qui vient de se passer avec le correctif de B.26 en round 2.
- `frozen_decoder_scale_fix.py`/`..._rejudge.py` : isolation propre de la seule variable
  qui compte (`input_scale`), noms de checkpoints dédiés respectant B.8, et vérifié
  indépendamment dans le dépôt (`frozen_core.py:150`,
  `self.W_dec_extra.requires_grad_(False)`) que le décodeur reste bien gelé comme
  l'hypothèse du script le suppose — cette vérification n'était pas dans le script
  lui-même, à ajouter comme assertion explicite plutôt que comme hypothèse implicite
  héritée de la classe parente.
- `uniform_hardneg_rejudge.py` : le mieux conçu des cinq — échantillonnage uniforme
  correct, négatif dur construit sur une autre feature réellement activante, duplication
  plutôt que modification du module partagé (même principe que celui demandé pour F.1),
  et honnêteté explicite sur les limites de la comparaison McNemar (recouvrement attendu
  faible, correctement relégué en information secondaire plutôt que présenté comme le
  test principal). Ce script est la référence de rigueur à généraliser aux deux
  scripts du point 2 ci-dessus, pas un cas isolé à saluer sans suite.

## 4. Résultats manquants à ce stade

Je n'ai pas les sorties de `audit_2026_08_frozen_decoder_scalefix_rejudge.json` ni de
`audit_2026_08_uniform_hardneg_results.json` — jugement porté sur la méthode seule pour
ces deux-là. Les deux répondent directement à des questions centrales de l'audit (le
confond input_scale explique-t-il l'écart 45,3%/29,3% ? le taux de 45,3% résiste-t-il à
un échantillon non biaisé et un négatif dur simultanément ?) — priorité à les transmettre
dès disponibles plutôt qu'à attendre un lot complet.

## 5. Sur le résultat delta_ce lui-même (contenu, pas seulement méthode)

Positif et à intégrer au rapport, avec la nuance suivante à formuler explicitement :
l'extension réduit la dégradation fonctionnelle (ΔCE) de ~69% par rapport au core seul —
elle préserve mieux ce que le LLM utilise réellement pour prédire — alors que le
constat existant ailleurs (§55) ne montre aucun gain linéaire mesurable de l'extension sur
la classification/silhouette en aval. Ce ne sont pas deux résultats contradictoires : la
fidélité fonctionnelle au modèle d'origine et l'utilité pour une tâche de classification
externe sont deux propriétés différentes. À énoncer comme telles dans le rapport plutôt
que de laisser les deux chiffres coexister sans les relier.

## 6. Points identifiés en marge de la lecture du dashboard Streamlit (13/08, soir)

Regroupe (a) les points « ce que je changerais concrètement » issus de la lecture
complète de `src/visualization/dashboard.py`, et (b) les constats faits en répondant aux
questions sur `clf_acc`/`dead_pct`/`diff_hypothesis`/l'axe des courbes P2 — à l'exclusion
du point EMA (amélioration UX pure, pas un constat d'audit). Palier 1 pour tous sauf
mention contraire — aucun ne nécessite de GPU.

### 6.1 Dashboard — absence de mesures d'incertitude

Aucune page n'affiche d'intervalle de confiance ni de p-value (page Vue d'ensemble,
Urgence/Robustesse, Diffing — ce dernier affiche un simple compte de features
significatives sans IC sur le compte lui-même). Cohérent avec le manque déjà noté en §2
de ce document pour les scripts qui alimentent ces pages — le dashboard hérite du
problème de ses sources plutôt que de le créer. Correctif : afficher l'IC de Wilson déjà
calculé par `stats.py` quand la source JSON le contient (cas de
`audit_2026_08_uniform_hardneg_rejudge.py` par exemple, une fois son résultat disponible).

### 6.2 Dashboard — aucune indication de version sur les chiffres sensibles au correctif B.26 en cours

`cache/intent_urgency_probe_results.json` (page Urgence/Robustesse) dépend directement de
`INTENT_KEYWORDS_FR`, dont le vrai correctif n'est toujours pas appliqué en production
(`src/data/dataset.py`, vérifié le 13/08 — le pattern buggé d'origine est toujours en
place). Si ce cache est régénéré un jour, le chiffre affiché changera silencieusement.
Correctif : afficher un hash ou l'horodatage du fichier source à côté du chiffre.

### 6.3 Dashboard — déséquilibre de rejet par axe d'augmentation, mesuré mais invisible

Confirmé en clair dans le code de `page_email_comparison` (commentaire, pas une métrique
affichée) : taux de rejet moyen ~11,7% du corpus augmenté, mais **59,6%** pour
`orthographe__degrade_fort` et **47,2%** pour `emotion__impatience`, contre ~4% pour les
onze autres classes — presque toujours par `length_ratio` trop bas (cf. B.23, round 1,
point 2, qui posait la question en hypothèse ; c'est maintenant un chiffre confirmé).
Correctif : remonter ce taux comme métrique visible (page Vue d'ensemble ou une nouvelle
section), pas seulement lisible en ouvrant le code source.

### 6.4 Dashboard — aucune page n'agrège les sorties d'audit de cette conversation

`audit_delta_ce_results.json`, `audit_bf16_fp32_diagnostic_results.json`,
`audit_palier1_batch_results.json`, et les sorties à venir
(`audit_2026_08_frozen_decoder_scalefix_rejudge.json`,
`audit_2026_08_uniform_hardneg_results.json`) vivent sous `docs/`/`cache/`, ne sont lues
par aucune page du dashboard. Correctif : une page « Audit / validité des résultats » qui
les rassemble — plus utile pour une présentation que d'ouvrir chaque JSON à la main.

### 6.5 Dashboard — page Recherche honnête, mais sans avertissement explicite sur Latent Terms

La page dit déjà correctement qu'elle fait une recherche par sous-chaîne sur les labels,
pas du BM25 (cf. round 1, échange sur `retrieval_demo.py`). Ajout recommandé : un
avertissement visible que Latent Terms n'a, à ce jour, jamais tourné sur le corpus EDF
réel — seulement sur un corpus de substitution FineWeb-2/Wikipedia FR
(`retrieval_demo.py`, vérifié le 13/08 : *« Mails.tsv... absent de cette machine »*),
pour éviter qu'un lecteur du dashboard ne suppose le contraire en voyant le module
`latent_terms.py` référencé dans la légende.

### 6.6 `generate_llm_diff_hypothesis` — le domaine "support" est chargé mais jamais comparé (P1, nouveau, confirmé par lecture de code)

`saev5.py` charge bien trois domaines pour le diffing cross-domaine (`ENERGY_KEYWORDS`,
`SPORTS_KEYWORDS`, `SUPPORT_KEYWORDS`, tous utilisés dans la préparation du corpus,
lignes ~1638-1709) — mais la comparaison elle-même est câblée en dur sur deux masques
seulement :
```python
diff_hypothesis = generate_llm_diff_hypothesis(j_llm, j_tok, diff_df, "Énergie", "Sports")
```
et le fichier de sortie est nommé littéralement `p1_diff_energy_sports.csv`. "support" est
chargé, encodé post-hoc, jamais comparé — reste d'un refactor incomplet (le domaine a été
ajouté au corpus sans que la fonction de diffing soit mise à jour pour l'utiliser).

**À faire (Palier 1)** : étendre la comparaison aux trois paires (energy/sports,
energy/support, sports/support) ou à un test multi-groupe, renommer le fichier de sortie
en conséquence. Coût nul, aucune réextraction requise — `support_texts` est déjà encodé
post-hoc dans le même run.

### 6.7 `generate_llm_diff_hypothesis` — le prompt donne les noms de catégories avant l'évidence, jamais testé pour un biais de prior (P1 pour le diagnostic, P2 pour le correctif si confirmé — nouveau)

Le prompt (`saev5.py:302-321`) commence par *« Corpus 'Énergie' vs 'Sports' »* avant même
de lister les features discriminantes (labels + fréquences + log-odds + q-value — ce
n'est pas *seulement* les noms, il y a une vraie évidence fournie). Mais donner les noms
de catégories en amont à un modèle qui sait déjà que "Énergie" et "Sports" sont deux
thématiques éloignées laisse ouverte la possibilité qu'il génère une histoire plausible
en s'appuyant surtout sur ce qu'il sait déjà des deux thèmes, pas sur les huit features
listées — un biais de prior, jamais testé, distinct des circularités déjà notées ailleurs
(B.2/B.3/B.4) mais de la même famille : le juge/générateur dispose d'un raccourci
plausible qui ne passe pas par l'évidence fournie.

**Test de falsification peu coûteux (Palier 2, un seul GPU, quelques minutes)** : rejouer
le même prompt avec (a) les mêmes features mais des noms de catégories neutres
(« Groupe A »/« Groupe B ») — si l'hypothèse reste aussi précise et ancrée dans les
features citées, le risque est écarté ; si elle devient vague ou générique, le prior
dominait. Et/ou (b) permuter les features entre les deux corpus en gardant les vrais noms
de catégories — si l'hypothèse générée ne change presque pas malgré des features
inversées, c'est la preuve directe que le prior domine l'évidence.

### 6.8 `phrase_sae.py` — `batch_size` codé en dur, ignore `BATCH_TRAIN` de `config.py` (P2, mineur, nouveau)

`phrase_sae.py:206` fixe `batch_size = 256` en dur plutôt que d'importer
`BATCH_TRAIN` (`config.py`, également 256 par défaut — coïncidence actuelle, pas une
garantie). Si `BATCH_TRAIN` est modifié via variable d'environnement en pensant affecter
l'entraînement de la Pipeline 2, ça ne fera rien silencieusement. Correctif trivial,
Palier 1.