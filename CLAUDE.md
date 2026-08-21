# CLAUDE.md

Instructions pour un agent travaillant sur ce dépôt. Ne duplique pas
`RESULTS_TESTS.md` (journal d'expériences numéroté) ni `docs/` (référence
technique) — les complète avec ce qui doit être vu avant de les lire.

## Projet

Analyse interprétable de mails clients EDF via Sparse Autoencoders sur les
hidden states de Gemma-3, labellisation par GemmaScope-2/Neuronpedia + juge
LLM local. Deux pipelines : Pipeline 1 (Gemma-3 → SAE GemmaScope-2 + extension
`FrozenCoreResidualSAE`), Pipeline 2 (F2LLM-v2 → `PhraseLevelSAE` from-scratch).

## Règles de fond

- Ne pas réimplémenter une fonctionnalité déjà présente dans SAELens ou
  interp_embed sans comparaison documentée (`docs/references.md`).
- `FrozenCoreResidualSAE` est spécifique au projet, ne pas la remplacer par un
  usage direct de SAELens.
- bf16 obligatoire sur les activations du residual stream de Gemma-3 (activations
  massives, débordent en fp16). Pipeline 2 (F2LLM, entrées L2-normalisées et
  bornées à 1,0) : fp32 pour le SAE, bf16 pour le seul backbone. La branche
  "extra" de `FrozenCoreResidualSAE`/`SAEBoostResidualSAE` reste en fp32 ;
  ne jamais caster le module entier après construction.
- Toute modification doit laisser `pytest tests/ -q` 100% vert.
- Toute clé de cache ou chemin de checkpoint est dérivé mécaniquement des
  paramètres dont le contenu dépend, jamais rédigé à la main comme une
  f-string ad hoc (R5) — cf. piège ci-dessous.
- Tout écart à l'équation ou au protocole d'un papier cité s'documente dans
  `docs/references.md` : équation du papier, équation implémentée,
  justification, conséquence sur la comparabilité des chiffres (R6).
- Toute boucle dont le coût dépasse ~1h GPU écrit son état de progression de
  façon atomique et reprend depuis cet état (`src/storage/checkpoint.py`) ;
  le critère de reprise est *quel est le prochain élément non traité*, jamais
  *le run est-il complet*. Un compteur d'échantillonnage (réservoir) fait
  partie de l'état à persister (R1).
- Toute structure O(n²) en nombre de documents ou en largeur de dictionnaire
  porte un plafond codé explicite (R4) — la trajectoire du projet vers 65k
  puis 262k de largeur rend ce risque concret, pas théorique.
- Tout script produisant une section `§N` de `RESULTS_TESTS.md` reporte
  `tokens/s`/`docs/s`/`steps/s` et le pic VRAM dans son JSON de résultats (R3).

## Convention de test — deux niveaux, ne pas les mélanger

- `tests/` : assertions unitaires rapides, CPU uniquement (shape, dtype,
  non-régression, "ça ne plante pas"). `pytest tests/ -q` doit rester 100% vert.
- `scripts/*_test.py` / `*_audit.py` : expériences empiriques (ablations,
  audits méthodologiques) qui produisent un JSON de résultats
  (`SAVE_DIR/cache/*.json` ou `local_data/.../*.json`) → une section numérotée
  dans `RESULTS_TESTS.md` (format défini en tête de ce fichier) → un onglet
  dashboard si pertinent (`src/visualization/dashboard.py`).

Un hook post-edit (`.claude/settings.json`) relance `pytest tests/ -q` en
tâche de fond après toute édition d'un fichier `.py` et remonte les échecs.

## Statistiques — utiliser `src/analysis/stats.py`

Module partagé (McNemar apparié, Cochran-Armitage, proportions+IC de Wilson,
h de Cohen, BH/FDR, analyse de puissance) — ne pas réinventer un test par
script.

## Cache/checkpoint — piège fréquent

Une clé de cache ou un chemin de checkpoint doit encoder TOUS les paramètres
dont le contenu dépend (taille de SAE, corpus, budget de tokens, etc.), pas
seulement ceux qui semblaient pertinents au moment d'écrire le loader — sinon
un run peut charger silencieusement le checkpoint d'une configuration
différente au lieu de réentraîner. `load_or_train_extended_sae` ne valide
aujourd'hui pas que la configuration demandée correspond à celle du
checkpoint chargé (risque structurel identifié, pas encore corrigé) : avant
de faire confiance à un résultat qui dépend d'un cache réutilisé entre runs,
vérifier à la main que la configuration n'a pas changé depuis l'écriture du
cache.

## Seeds — piège fréquent

`SEED` (entraînement SAE/juge) et `CORPUS_SPLIT_SEED` (split train/test du
corpus) sont **découplés** dans `src/config.py`, tous deux à 42 par défaut —
ne pas supposer qu'ils sont le même paramètre en reconstruisant un split de
référence.

Une reprise après coupure (`src/storage/checkpoint.py`) n'est PAS bit-reproductible
par rapport à un run continu : le flux RNG diverge dès la reprise (composition de
lots différente), donc le réservoir résiduel d'un run repris ≠ celui d'un run
continu. Scientifiquement bénin (échantillon aléatoire dans les deux cas), mais
`SEED` ne garantit une reconstruction bit-exacte que pour un run jamais interrompu.

## Pièges PyTorch/HuggingFace rencontrés

- `@torch.no_grad()` en décorateur sur une fonction **génératrice** ne protège que
  l'appel qui crée l'objet générateur, pas les itérations faites ensuite via
  `next()`/`for` — l'autograd tourne actif sur tous les forwards suivants, cause
  d'OOM déjà rencontrée une fois (`src/analysis/activations.py::extract_residual_acts`,
  déjà corrigé). Toujours un `with torch.no_grad():` explicite autour du corps de la
  boucle pour ce genre de fonction, jamais le décorateur seul.
- `output_hidden_states=True` sans `logits_to_keep=1` fait calculer par défaut les
  logits sur toute la séquence et tout le vocabulaire (Gemma-3 : ~262k tokens) même
  quand seul `hidden_states` est utilisé — coût VRAM inutile qui peut à lui seul
  causer un OOM. Toujours passer `logits_to_keep=1` pour une extraction
  hidden-states-only.

## Cluster SLURM

Conventions de partitions, soumission, logs, disque : `docs/ops.md`.

**Aucun calcul sur le nœud frontal.** La RAM/CPU y sont partagées entre tous
les utilisateurs du cluster. Une validation de configuration bornée (<5s CPU,
<500Mo RSS, aucune lecture de tenseur/modèle/checkpoint réel — vérifier un
chemin, une clé de cache, une regex, une jointure sur un manifest déjà écrit)
est tolérée ; tout calcul sur des données/modèles réels passe par `sbatch`,
catégorie `slurm/validation/` pour un smoke-test CPU-only sur cache. Le hook
post-edit (`pytest tests/ -q` en tâche de fond) reste la seule exécution
automatique tolérée hors `sbatch`, déjà bornée par le harness.

**Les activations SAE (17k+ dimensions, ~99,9% de zéros) se passent en CSR
sparse à `LogisticRegression`, jamais en tableau dense.** `sklearn` recopie
tout `X` dense fp32 en fp64 en interne ; sur une matrice quasi-vide de cette
largeur, c'est ce qui rendait `downstream_classification`/`clf_acc_email_axes`
compute-bound au point de tourner indéfiniment sans sortie ni erreur (observé
deux fois avant correctif : `run_core_vs_extension_ablation.slurm` puis
l'audit 2026-08). `liblinear` et `lbfgs` acceptent tous deux un `X`
`scipy.sparse.csr_matrix` nativement — `downstream_classification`
(`src/analysis/metrics.py`) le fait déjà. `--cpus-per-task=32` n'est un
correctif que si le profil montre, après passage en CSR, un coût réellement
dense (ex. l'agrégation en aval, pas le fit lui-même) — ne pas repartir de 32
cœurs par défaut. Poser `PYTHONUNBUFFERED=1` sur tout script d'audit qui
imprime une progression avant un calcul long — sans ça, le log reste identique
entre deux vérifications qu'un job soit bloqué ou juste en train de calculer,
ce qui rend impossible de diagnostiquer lequel des deux se passe.

## Git

Ne jamais ajouter de trailer `Co-Authored-By: Claude` dans les messages de
commit de ce dépôt — GitHub l'affiche comme un contributeur, ce que ce projet
ne veut pas.

## Documentation

- `RESULTS_TESTS.md` est append-only : les identifiants `§N` sont cités
  depuis le rapport et ne doivent jamais être renumérotés. Chaque nouvelle
  section suit le format : Question / Écart à la configuration de référence
  (`docs/evaluation_protocol.md`) / Méthode statistique / n / Résultat /
  Conclusion / Limite connue.
- Rédiger au présent, sans numéro de version interne (`v9`, `v10`...) ni récit
  de session : une contrainte de conception encore active se formule comme
  une règle, pas comme le récit de sa découverte. Même discipline dans les
  commentaires et docstrings du code (R2) : citer un défaut par son nom de
  variable, pas par sa narration ("corrigé cette session", "audit du ...").
  Exception assumée : les répertoires/scripts de run
  (`results_v14_main/`, `run_sae_v12_scaled.slurm`) restent versionnés/
  horodatés — seule la prose reste intemporelle.

## Diagnostics — un run est-il sain avant d'en tirer une conclusion ?

Checklist ordonnée à suivre avant de faire confiance à un résultat
d'interprétabilité — un run qui échoue tôt dans cet ordre rend les étapes
suivantes non interprétables, ne pas sauter aux étapes 4-5 sans avoir vérifié
1-3. Chaque métrique est déjà calculée par le pipeline (`results.json`,
`p1_top_extended_features.json`, `*_history.json`) ou tracée par
`scripts/generate_diagnostic_plots.py` (agrégation rétroactive, zéro rerun) +
`src/analysis/plotting.py`.

1. **Convergence** (`plots/p1_training_curves.html`/`p2_*`) : loss train encore
   en baisse nette à la dernière époque → sous-entraîné. Loss validation qui
   diverge de la loss train → surapprentissage sur le résidu — mais
   `BatchTopKEncoder` (`src/sae/batch.py`) bascule en régime JumpReLU seuillé
   dès `model.eval()`, différent du régime BatchTopK d'entraînement : une
   partie de l'écart train/val mesure ce changement de régime, pas seulement
   du surapprentissage, ne pas lire l'écart brut comme une preuve directe.
   `dead_frac` qui ne redescend jamais après un pic initial → l'AuxK ne
   ranime pas les features mortes.
2. **Fidélité de reconstruction** (`results.json → rho_sae`, `fve_pretrained`) :
   `rho_sae` proche de 0 → l'extension n'apprend que du bruit sur le résidu.
   `fve_pretrained` très bas → le core lui-même n'explique déjà plus grand-chose
   à ce point du réseau, aucune extension ne peut compenser.
3. **Budget de capacité** (`results.json → dead_pct`) : une fraction de
   features mortes élevée n'est pas nécessairement un problème si
   `rho_sae`/`interp_rate` restent bons — mais une hausse brutale entre deux
   runs par ailleurs identiques signale un problème d'entraînement (LR,
   époques), pas un choix de capacité.
4. **Fiabilité du taux d'interprétabilité** (`p1_top_extended_features.json →
   interp_score`, `rho_interp`) : le protocole odd-one-out est bruité au
   niveau d'une feature isolée (`RESULTS_TESTS.md` §13.1 : ~31% de décisions
   instables au simple réordonnancement des exemples) — ne jamais lire une
   feature individuelle comme "prouvée interprétable", seul le taux agrégé
   sur n≥150 est informatif.
5. **Significativité statistique** (`src/analysis/stats.py`, jamais une
   lecture à l'œil de deux pourcentages) : layer 31 vs layer 24 (référence)
   est le seul écart individuel qui atteint `|z|>1,96` contre la configuration
   de référence à ce jour (`RESULTS_TESTS.md` §51, sans correction
   multi-tests, à répliquer avant adoption) ; `mlp_out` vs `attn_out` (§53) est
   significatif entre eux mais ni l'un ni l'autre ne l'est contre `resid_post`
   isolément. Tout le reste (`K_EXTRA`, `D_EXTRA`, volume, seed) reste dans le
   bruit à n=150 — seul le choix de taille du modèle extracteur/juge produit
   un effet massif et répliqué à chaque palier.
6. **Indépendance du juge** (uniquement si le corpus de test inclut du texte
   généré par le même modèle que le juge, ex. corpus augmenté) : vérifié
   résolu négativement sur ce projet (`RESULTS_TESTS.md` §48/§50/§52) — à
   revérifier explicitement sur tout nouveau corpus/juge dans cette
   configuration, ce n'est pas une propriété générique du protocole.
