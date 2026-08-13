# Carte de migration — Étape 0 (inventaire)

Fichier temporaire, à supprimer après relecture. Pilote les étapes 1 à 8 de la
refonte documentaire. Toute suppression de contenu factuel dans les étapes
suivantes doit être tracée ici (`source → destination` ou `source → supprimé,
motif`).

**Méthode et limites de cet inventaire** : les références croisées (`§N`) sont
extraites exhaustivement par grep sur tout le corpus documentaire versionné
(`README.md`, `Context.md`, `CLAUDE.md`, `docs/*.md`, `report/*.md`, hors
`RAPPORT_DE_STAGE.md` généré et les `.tex`). L'inventaire des chiffres couvre
exhaustivement les catégories nommées dans la consigne (taux
d'interprétabilité, taux de classification, tailles de corpus, largeurs de
SAE, couvertures Neuronpedia) ; il ne prétend pas capturer chaque nombre
isolé du corpus (RESULTS_TESTS.md seul fait ~2960 lignes avec des dizaines de
tables). Les valeurs marquées `⚠ INCOHÉRENT` ont une divergence réelle entre
fichiers ; les autres répétitions du même chiffre dans plusieurs fichiers ne
sont pas listées individuellement au-delà d'un échantillon représentatif.

---

## A. Références croisées à préserver

### A.1. Sections `RESULTS_TESTS.md` citées depuis l'extérieur

Extrait exhaustif (grep) des `§N` cités depuis `README.md`, `Context.md`,
`CLAUDE.md`, `docs/*.md`, `report/*.md` :

§0, §2, §6, §10, §12, §13.1, §14.1, §15, §16.3, §17, §18, §19, §21, §22,
§23.3, §24, §25, §26, §28, §29, §30, §31, §34, §37, §44, §45, §46, §48, §50,
§51, §52, §53, §54, §55.

Toutes ces ancres existent dans `RESULTS_TESTS.md` (vérifié). **Aucune ne doit
changer de numéro** (Étape 4, contrainte dure).

### A.2. Structure actuelle de `RESULTS_TESTS.md` (2960 lignes, 55 sections)

Numérotation des `##` par ordre **physique** dans le fichier :

```
§0 (L6), §10 (L115), §11 (L189), §1 (L223), §2 (L229), §3 (L267), §4 (L277),
§5 (L320), §6 (L344), §7 (L427), §8 (L440), §9 (L458), §12 (L480), §13 (L590),
§14…§55 : déjà en ordre croissant strict jusqu'à la fin du fichier.
```

- **Non-monotonie confirmée** : uniquement en tête de fichier — `§10` et `§11`
  sont physiquement placés entre `§0` et `§1`, au lieu d'être après `§9`. À
  partir de `§12`, l'ordre physique est déjà strictement croissant jusqu'à
  `§55`.
- **`§35` : absent, confirmé.** Aucune section `## 35.` n'existe ; le fichier
  passe de `§34` (L2166) à `§36` (L2222) sans trace d'un contenu retiré. À
  documenter par une ligne explicite (Étape 4), pas à combler.
- Réordonnancement requis (Étape 4) : déplacer physiquement le bloc `§10-§11`
  (L115-L222) pour qu'il suive `§9` (après L479), sans toucher aux titres `##
  N.` ni au contenu. Aucun autre déplacement nécessaire.

### A.3. Autres références croisées inter-fichiers (hors `RESULTS_TESTS.md`)

| Référence | Depuis | Vers | Statut |
|---|---|---|---|
| `report/04_limites_et_perspectives.md` | `Context.md` ("Prochaines étapes"), `README.md`, `06_conclusion.md` | Liste vivante des perspectives | Cible existe, section "Perspectives" (non numérotée depuis la refonte précédente de ce fichier) |
| `docs/references.md` | `Context.md` (règle n°1/2), `01_etat_de_lart.md` | Comparaisons SAELens/interp_embed | Cible existe |
| `docs/architecture.md` | `Context.md`, `README.md` | Architecture technique | Cible existe, mais décrit une arborescence différente de celle de `Context.md` "Architecture cible" (cf. §D.3) |
| `docs/evaluation_protocol.md` | `README.md` (implicite), `report/*.md` (aucune ref directe trouvée) | Conditions fixées d'évaluation | Cible existe ; candidate naturelle pour le "tableau unique de configuration de référence" de l'Étape 3.3 |
| `docs/experiments.md` | — | Index par question de recherche | Existe, chevauche partiellement `RESULTS_TESTS.md` §0/§12 (mêmes chiffres, présentation différente) |
| `05_erreurs_et_corrections.md` | `Context.md` ("Bugs corrigés" pourrait y renvoyer) | **N'existe plus** (supprimé lors d'une passe antérieure, contenu réparti dans `04_limites_et_perspectives.md`) | Aucune référence externe cassée trouvée vers ce fichier |
| `REVIEW_ARS_PANEL.md` | — | **N'existe plus** (supprimé, contenu absorbé) | Aucune référence externe cassée trouvée |

Aucun lien mort détecté à ce stade vers un fichier inexistant. À revérifier en
Étape 7 après les suppressions de l'Étape 1 (`Context.md` disparaît : toute
référence `cf. Context.md` doit être retirée ou redirigée, cf. §D ci-dessous).

---

## B. Inventaire des chiffres répétés

### B.1. Taux d'interprétabilité — run principal

**Valeur canonique retenue : 45,3% (68/150), `results_v10_emails_main`, SEED=42,
500k tokens.** C'est la valeur la mieux sourcée (citée par `RESULTS_TESTS.md`
depuis §12 et systématiquement reprise comme référence par tous les chapitres
du rapport, ~40 occurrences cohérentes trouvées dans `report/03_*` et `04_*`).
Aucune incohérence numérique sur cette valeur elle-même.

**⚠ INCOHÉRENT (formulation, pas valeur) — `README.md` L32-33 et L387-389** :
cite « ~41-45% (n=150) » sans préciser qu'il s'agit de la **plage observée sur
3 runs d'ablation de volume** (100k=40,7%, 500k=45,3%, 2M=44,7%), pas d'une
seule mesure. Un lecteur du seul README ne peut pas distinguer cette plage du
chiffre ponctuel 45,3% cité partout ailleurs comme LE taux du run principal.
→ **Arbitrage proposé (Étape 3.4)** : dans `README.md`, remplacer « ~41-45%
(n=150) » par « 45,3% (n=150, run principal) » et déplacer la plage
d'ablation de volume en une clause séparée explicite si conservée.

### B.2. Taux d'interprétabilité — baseline pré-correctif (corpus générique)

**Valeur canonique : 20,0% (2/10), `results_v9_full`.** Cohérente partout où
citée (`README.md`, `Context.md`, `docs/experiments.md`, `report/00_*`,
`03_*`, `06_*`, `FRONT_MATTER.md`). Le `n=10` (donc l'intervalle de confiance
très large) est rappelé dans `report/03_experiences_et_resultats.md` mais
**absent** de `report/00_introduction.md` L56, `06_conclusion.md` L31 et
`FRONT_MATTER.md` L49/L86 (résumé/abstract), qui citent le seul « 20% » sans
`n`. Pas une incohérence de valeur, mais un manque de rappel de puissance
statistique répété à trois endroits distincts qui présentent chacun le
chiffre comme un fait établi. → à corriger Étape 3.4/5.3 (au moins une
mention de `n=10` dans le résumé, cf. règle de cohérence numérique).

### B.3. `clf_acc_email_axes` (sonde de classification, 14 classes)

**Valeur canonique : 93,5% (Pipeline 1), 79,3% (Pipeline 2, F2LLM-80M).**
Cohérente sur toutes les occurrences trouvées (`Context.md`,
`docs/experiments.md`, `report/03_*` (7 occurrences), `04_*`, `06_*`). Aucune
incohérence de valeur.

**Réserve associée (baseline TF-IDF lexical, 87,0%)** : présente dans
`Context.md` L227-233 (« +6,5 points » de gain net) et
`report/03_experiences_et_resultats.md` L155 (« ~93% du contenu sémantique »)
— deux formulations différentes du même calcul (93,5-87,0=6,5 ;
87,0/93,5≈93%), mathématiquement cohérentes entre elles. Pas une incohérence
numérique, mais une redite à consolider (Étape 3.8) : la réserve n'existe
aujourd'hui que dans `Context.md` (supprimé à l'Étape 1) et dans le rapport —
vérifier qu'elle survit bien dans le rapport après suppression de `Context.md`
(elle y est déjà, `report/03_*` §5.4 et `04_limites_et_perspectives.md`
"Rigueur statistique").

### B.4. Largeur du SAE core / couverture Neuronpedia — ⚠ INCOHÉRENT MAJEUR

**Valeur canonique (mesure la plus récente, `report/01_etat_de_lart.md` L35-42
et `README.md` L128-133, cohérentes entre elles)** :
- 16k : 82,6% (13 535/16 384)
- 65k : 87,8% (57 551/65 536) — **meilleure couverture, retenue par défaut**
- 262k : 5,3% (13 851/262 144)
- 1m : non hébergé par Neuronpedia pour ce modèle

**⚠ `Context.md` L63-69 et L105-107 est en contradiction directe avec le code
actuel** (`src/config.py`, preset `12b` → `layer_24_width_65k_l0_medium`) :
`Context.md` décrit encore un choix entre 16k et 262k *seulement* (« 16k est
bien plus dense en proportion », comparaison à ~10 000/262 144 « constatée
manuellement »), sans jamais mentionner que 65k a depuis été testé et
retenu. Un lecteur de `Context.md` seul conclurait à tort que **16k** est la
largeur de référence du projet. **Conformément à l'invariant 1, le code fait
foi : 65k est la largeur de référence actuelle.** `Context.md` étant supprimé
à l'Étape 1, cette contradiction disparaît avec le fichier ; à vérifier que
plus aucun autre fichier ne reprend l'affirmation « 16k retenu ».

**Point d'attention résiduel** : `README.md` L133 précise que « les runs
comparatifs de ce projet ont cependant été menés à 16k » — cohérent avec le
fait que `results_v10_emails_main` (run principal, 45,3%) utilise
`layer_24_width_16k_l0_medium` (confirmé par les logs de run consultés) alors
que 65k n'a été identifié comme supérieur qu'après coup et n'a servi qu'au
« run de mise à l'échelle final » (`report/01_*` L42, renvoi chapitre 3). Ce
n'est pas une incohérence mais un point à garder explicite dans le tableau de
configuration de référence unique (Étape 3.3) : la largeur diffère entre le
run principal (16k) et le run de mise à l'échelle (65k), ce n'est pas la même
condition expérimentale.

### B.5. Tailles de corpus — ⚠ INCOHÉRENT (mineur)

**Valeur canonique (observée directement à l'exécution, logs de job
consultés le jour même) : 3 474 emails originaux**, 39 879 variantes
augmentées acceptées, 41 176 documents train / 2 177 test.

**⚠ `docs/experiments.md` L16 cite « 3480 mails »** (au lieu de 3474) pour le
calcul « 3480 mails × 13 axes = 45 240 » — cette multiplication ne tombe
d'ailleurs juste ni avec 3474 ni avec 3480 exactement (13×3474=45162,
13×3480=45240) : la valeur 45 240 générations elle-même est correcte et
répétée de façon cohérente ailleurs (`report/04_*` L225), donc l'erreur est
localisée au facteur « 3480 » de cette seule ligne, pas à 45 240. → à corriger
en 3474 (Étape 3.4), sourcé sur l'exécution réelle plutôt que recalculé.

### B.6. Longueur de `src/sae/saev5.py` — ⚠ INCOHÉRENT (obsolète, à supprimer plutôt que corriger)

`Context.md` L59 cite « ~1470 lignes » ; le fichier fait actuellement 1801
lignes (`wc -l`). Cette statistique est un artefact de rédaction (compte de
lignes à un instant donné) sans valeur informative résiduelle une fois le
fichier modifié — catégorie **(B)** de l'Étape 2, à retirer plutôt qu'à
corriger avec un nouveau chiffre qui se périmera à nouveau.

---

## C. Constats déjà remontés par l'IDE/le contexte (à vérifier en Étape 2/3)

Le paragraphe sélectionné par l'utilisateur dans `README.md` (L28-36, corpus
principal / diagnostic domaine-vs-volume) est **correct et à jour** dans son
contenu chiffré (45,3%/20%/100k-500k-2M) : il n'est pas remis en cause par le
code actuel ni par les audits critiques menés (C1 confirmé statistiquement
significatif, z=2,74, p≈0,006, `RESULTS_TESTS.md` §46 ; C2 résolu
négativement, `RESULTS_TESTS.md` §48/§50/§52). Le problème signalé par
l'utilisateur est donc bien **éditorial**, pas factuel : le paragraphe est un
récit de correctif (« corrigée », renvoi `§12` présenté comme un journal de
bug) plutôt qu'un énoncé de résultat — traitement catégorie **(C)** de
l'Étape 2 (reformulation en résultat, sans perte du chiffre).

---

## D. Contradictions code vs documentation (invariant 1)

| Fichier/section | Affirmation documentaire | État du code | Traitement |
|---|---|---|---|
| `Context.md` L65-69, L105-107 | Largeur SAE core retenue = 16k (comparée seulement à 262k) | `src/config.py` : défaut 12b = 65k | `Context.md` supprimé (Étape 1) ; le rapport (`01_etat_de_lart.md`) est déjà correct |
| `Context.md` L427-440 « Dashboard… Non commencé » | Aucune interface Streamlit | `src/visualization/dashboard.py` existe, fonctionnel (onglets UMAP, features, diffing, recherche, diagnostics) | Section entière obsolète, catégorie (B), à retirer |
| `Context.md` L59-61 « `evaluation/` et `visualization/dashboard.py`… n'existent pas » | — | Idem, contredit directement le point précédent au sein même de `Context.md` (le fichier se contredit lui-même à 400 lignes d'écart) | Idem |
| `Context.md` L271-300 « Architecture cible » (`src/models/`, `src/sae/training.py`, `evaluation/benchmarks.py`…) | Arborescence cible jamais réalisée telle quelle | Structure réelle : `src/sae/`, `src/analysis/`, `src/data/`, `src/storage/`, `src/visualization/` (existants), pas de `src/models/` séparé | À reformuler dans `docs/architecture.md` comme description de l'existant, sans section "cible" non tenue à jour |
| `Context.md` L364-369 | « Cette liste était stale (…) ne pas dupliquer ici » | — | Auto-commentaire de processus, disparaît avec le fichier |

---

## E. Fichiers non encore inventoriés en détail

`report/02_architecture.md`, `05_erreurs_et_corrections.md` (n'existe plus),
`docs/ops.md`/`docs/ops_journal.md` (n'existent pas encore, à créer Étape 1/4)
n'ont pas fait l'objet d'un passage numérique dédié au-delà des occurrences
déjà listées ci-dessus — aucun chiffre supplémentaire à fort risque
d'incohérence identifié lors de la lecture de ces fichiers en amont de cette
tâche.
