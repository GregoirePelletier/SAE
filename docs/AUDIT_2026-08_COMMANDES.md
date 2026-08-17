# Commandes de tests à lancer — audit 2026-08

Liste de référence, par ordre de priorité, des tests encore utiles à lancer
pour l'audit méthodologique. Complète `docs/AUDIT_2026-08.md` (constats) et
`RESULTS_TESTS.md` (résultats déjà obtenus) — ne les duplique pas.

**Règle absolue : jamais de Python en direct sur le nœud frontal, toujours
`sbatch` depuis `/home/h21486/SAE/`.** Chaque commande ci-dessous est un
`sbatch slurm/validation/....slurm` autonome. Après soumission :
`squeue -u h21486` pour suivre, `logs/validation/<nom>_<jobid>.log` pour le
détail. Les résultats atterrissent dans `results_v10_emails_main/cache/*.json`
ou `docs/*.json`.

**Balayage exhaustif fait (`RESULTS_TESTS.md` §70)** : tous les scripts déjà écrits
et référencés par un `.slurm` existant ont été exécutés au moins une fois. Un seul
écart trouvé — `embedding_model_comparison_test.py`, déjà exécuté (job 40730, session
antérieure à cet audit) mais jamais transcrit — comblé sans recalcul. **Priorité 1
ci-dessous est donc entièrement close** ; tout ce qui reste (Priorités 2-3) nécessite
un script à écrire, pas seulement une commande à relancer.

---

## Priorité 1 — clos cette session (commandes conservées pour relance future)

### 1. B.26 — correctif appliqué en production, propagation + sanity checks faits

`src/data/dataset.py::INTENT_KEYWORDS_FR` corrige désormais les patterns en
production (autorisation explicite donnée). Les 4 consommateurs
(`intent_urgency_probe.py`, `explanation_fidelity_test.py`,
`steering_fidelity_test.py`, `latent_retrieval_precision_eval.py`) ont été
rejoués SANS patch et reproduisent les chiffres audités (match exact ou
quasi-exact, RESULTS_TESTS.md §69). Une source de non-déterminisme distincte
a été trouvée et corrigée au passage (`LogisticRegression(solver="liblinear")`
non seedée dans 2 scripts). Plus rien à relancer sur ce point sauf si
`INTENT_KEYWORDS_FR` change à nouveau :

```
sbatch slurm/validation/run_audit_b26_propagate_fidelity.slurm   # rejeu par monkey-patch (comparaison)
sbatch slurm/analysis/run_intent_urgency_probe_main.slurm         # rejeu direct (production)
sbatch slurm/analysis/run_explanation_fidelity_test_main.slurm
sbatch slurm/analysis/run_steering_fidelity_test_main.slurm
sbatch slurm/analysis/run_latent_retrieval_precision_eval.slurm
```

### 2. B.24 — premier essai du module de comparaison inter-modèles : fait

```
sbatch slurm/validation/run_audit_b24_compare_pipeline.slurm      # rerun complet (~1-2h GPU)
sbatch slurm/validation/run_audit_b24_inspect_pollution.slurm     # inspection seule (~1 min, sur cache existant)
```

Verdict "comparable" (F2LLM-v2-80M vs bge-m3), 0 feature flaggée dans les deux
modèles — `docs/AUDIT_2026-08.md` (B.24/E.7). **Rappel d'interprétation** :
entraîne DEUX nouveaux SAE from-scratch (pas les checkpoints existants),
pooling mean pour les deux modèles (bge-m3 est normalement utilisé en pooling
CLS ailleurs dans ce dépôt) — ce n'est pas une comparaison directe Pipeline 1
vs Pipeline 2. **Piste ouverte, non tranchée** : tester `--model-b` avec un
backbone attendu comme nettement moins adapté au domaine (EN générique) pour
vérifier que le détecteur de pollution peut au moins détecter un cas connu,
avant de faire confiance à un verdict "comparable" sur une paire dont le
comportement attendu est inconnu.

---

## Priorité 2 — script à écrire d'abord (specs ci-dessous, Palier 1, coût faible)

### 3. B.22 point 1 — ablation de volume pour la Pipeline 2 (jamais faite)

**Constat** : contrairement à la Pipeline 1 (B.19, RESULTS_TESTS.md §56), aucune
ablation de volume d'entraînement n'existe pour `PhraseLevelSAE`. Diagnostic
gratuit d'abord : vérifier si `scripts/generate_diagnostic_plots.py`
(rétroactif, zéro rerun) produit déjà des courbes de convergence exploitables
pour les runs Pipeline 2 existants — **déjà vérifié cette session : la
fonction existe (`generate_training_curves`, gère `p2_sae_dim*_history.json`)
mais aucun des runs existants (`results_v10_p2_*`, `results_v12/13_ablation_*`)
n'est une ablation de VOLUME de phrases — ce sont des ablations de backbone/
dimension d'embedding.** Une vraie ablation de volume reste à faire :
tronquer le nombre de phrases d'entraînement (`MAX_PHRASES_DOC` ou le nombre de
documents source) à plusieurs paliers (ex. 10%, 33%, 100% du corpus actuel),
entraîner `PhraseLevelSAE` à chaque palier, comparer convergence (`val_loss`)
et taux d'interprétabilité si un jugement odd-one-out équivalent existe pour
la Pipeline 2. Coût estimé : CPU pour le pooling+SAE (rapide, `phrase_sae.py`
est from-scratch et léger), GPU léger pour l'extraction F2LLM initiale
(déjà en cache pour le run principal, seul un sous-échantillonnage est
nécessaire, pas de ré-extraction).

### 4. B.23 — trois diagnostics sur le corpus augmenté déjà généré (aucune régénération)

**Constat** : `src/data/augmentation.py::validate()` ne vérifie que l'OMISSION
de faits (`_facts(parent) - _facts(variant)`), jamais la FABRICATION
(`_facts(variant) - _facts(parent)`) — un fait halluciné à température 0,8
passerait la validation sans être détecté. Deux autres questions ouvertes,
vérifiables sur le même fichier : corrélation longueur↔axe (se branche sur
B.9), corrélation colère↔registre informel (orthogonalité des axes supposée,
jamais vérifiée — `augmentation_lexical_leakage_audit.py` existe déjà mais
couvre les tics lexicaux, pas la corrélation inter-axes).

Script à écrire (`scripts/audit_2026_08_b23_augmentation_diagnostics.py`,
CPU uniquement, zéro GPU) sur `local_data/emails/augmented_mails.jsonl`
(champs confirmés cette session : `aug_id`, `parent_id`, `axis`, `level`,
`rejected`, `text` — filtrer `rejected is None` pour le corpus retenu) :
1. Pour chaque paire parent/variante non rejetée avec `axis != "orthographe"` :
   `_facts(variant) - _facts(parent)` (réutiliser `augmentation._facts`,
   import direct) — compter les fabrications potentielles par axe, échantillon
   manuel des premières occurrences pour écarter les faux positifs (numéros
   partiels réinventés par erreur d'OCR du prompt vs vraies hallucinations).
2. Longueur moyenne de `text` par `axis`/`level` — `groupby(["axis","level"]).text.str.len().mean()`.
3. Cross-tabuler la présence de marqueurs de registre informel (à définir : ex.
   absence de formules de politesse standard, contractions) contre
   `axis=="emotion"` vs `axis=="registre"` pour tester la corrélation
   colère↔informalité.

Coût : quelques minutes CPU, job `sbatch` léger (`--cpus-per-task=4 --mem=16G`
suffit, pas de GPU).

---

## Priorité 3 — gros effort, à ne lancer qu'après discussion (Palier 2-3)

### 5. B.16 — analyse de sensibilité des 8 constantes magiques

Provenance déjà vérifiée cette session (aucune des 8 n'a de référence en
commentaire, `docs/AUDIT_2026-08.md` B.16). Reste la sensibilité elle-même :
pour chaque constante (`sim_threshold=0,2`, `npmi_threshold=0,3/0,6`,
`min_freq=0,01`, `max_freq=0,5`, `neg_quantile=0,05`, `sigma_clip=4,0`,
`dead_steps_threshold=200`, `AUX_ALPHA=1/32`), un ou plusieurs reruns avec
valeur modifiée + comparaison de la métrique finale concernée. Potentiellement
8 campagnes de jobs distinctes — à prioriser une seule à la fois si une revue
externe questionne spécifiquement l'une de ces valeurs, pas à lancer en bloc.

### 6. C.4 — tests unitaires manquants (`tests/`, pas `scripts/`)

Pas des jobs SLURM — du code de test classique (`pytest tests/`, déjà couvert
par le hook post-edit). À écrire dans `tests/` : (a) un test qui vérifie
qu'un changement de paramètre change la clé de cache (B.8) ; (b) un test
d'invariance du pooling (permutation des tokens ⇒ même `doc_vec`) + test de
dépendance à la longueur (B.9, déjà quantifié empiriquement en §59, à figer
en test de non-régression) ; (c) un test que `FrozenDecoderExtendedSAE.W_dec_extra`
est bit-à-bit inchangé après N steps ; (d) généraliser l'injection de feature
synthétique (déjà faite pour `find_interesting_pairs`, §40) à un test de bout
en bout sur mini-corpus jouet.

### 7. Palier 3 — non engagé, dernier recours selon le prompt d'audit lui-même

- Balayage `D_EXTRA` overcomplete (au-delà de 1024).
- Run à volume ≥100M tokens (au-delà des 25M/200M déjà testés, §56/B.19).
- Croisement complet extracteur×juge (au-delà des paires déjà testées en B.17).
- Reproduction Axe F : cloner `interp_embed`/Latent Terms sur leur propre cas
  jouet pour un test de fidélité direct (submodule `interp_embed` vide,
  confirmé absent — cf. A.2 du prompt original).

Chacun de ces items est un projet en soi (plusieurs heures GPU, conception
d'expérience non triviale) — à ne pas lancer sans re-préciser la question de
recherche exacte au moment de s'y attaquer.

