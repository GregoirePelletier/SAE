# Preflight — campagne post_soutenance_v1

Reponse a l'instruction de demarrage du plan (§20.2). Inventaire seul, aucune
soumission Slurm, aucune modification destructive. A signaler a Gregoire avant
premiere soumission (§1.2).

## 1. Etat du depot vs commit audite

`HEAD` = `68c8e6863013c98905740a56e6491d78d05435cd`, identique au commit audite
par le plan. **Aucune derive de code** depuis l'audit : les points C1 (filler
dense `saev5.py`), C2 (mapping labels/IDs `clustering_llm_test.py`) et C4
(denominateur local `average_precision`, `metrics.py`) ont ete relus
directement et correspondent exactement aux lignes citees par le plan.

Etat non commite (laisse en l'etat, hors perimetre de cette campagne) :
- `report/RAPPORT_STAGE_ENTREPRISE.tex` : diff important (1364+/1639-),
  correspond a la reprise en cours documentee dans `AUDIT_SAE_2026-08.md` §9
  ("reste avant remise finale" -- pas encore repasse). Ne pas toucher : ce
  n'est ni le rapport defendu ni le dossier oral au sens strict de
  l'interdiction §1.3, mais c'est un travail de redaction en cours qui
  n'appartient pas a cette campagne.
- `external/interp_embed` (submodule) : seul `.gitignore` differe, sans effet.
- `archive/dual_pipeline_sae.py` (untracked) et une capture d'ecran
  (untracked) : provenance non identifiee par cet audit, laisses en l'etat.
- `Plan_execution_SAE_15_jours_Claude_Code.md` (untracked) : le document
  source de cette campagne lui-meme.

## 2. Tests

`pytest tests/ -q` : **292 passed, 1 failed** (226,6 s). L'echec est
`test_docs.py::test_check_docs_clean` (liens relatifs morts `s5`/`s6` a la
ligne 949 de `RAPPORT_STAGE_ENTREPRISE.tex`), directement cause par l'edition
en cours de ce fichier (section 1 ci-dessus), pas par un regression de code.
Hors perimetre : ne pas corriger sans que Gregoire confirme que la reprise du
rapport entreprise est terminee.

## 3. Ressources cluster (lecture seule, CPU frontal, aucune activation lue)

- Partitions : `a100` (1 noeud, 8 GPU, ~1,03 To RAM), `h100` et `h100-bis`
  (1 noeud chacun, 8 GPU + shard, ~2,06 To RAM chacun).
- File actuelle : aucun job en cours ni en attente pour cet utilisateur.
- Aucun `--account`/`--qos` explicite necessaire : les `.slurm` historiques
  qui ont reellement tourne (job 45735, 45745, 45750, 45803, 45874) n'en
  fixent aucun.
- Stockage : `/home` est un montage reseau partage, **88% plein, 4,9 To
  restants sur 40 To partages** entre tous les utilisateurs du cluster.
  Pertinent pour E09 : un seul tenseur x en 1B/100M bf16 fait 230,4 Go a lui
  seul (§5.2 du plan) ; avec marge/shards/checkpoints, cela peut representer
  une fraction non negligeable des 4,9 To restants sur une ressource
  partagee -- a rechiffrer avant toute autorisation E09, pas seulement au
  moment du run.

### Donnees memoire reelles (sacct, jobs deja passes -- ne remplace pas un
nouveau profilage E00, mais confirme/complete la citation C10 du plan)

| Job | Config | ReqMem | MaxRSS reel | Statut |
|---|---:|---:|---:|---|
| 45803 | 12B, layer12, K5, ~25M tokens | 500G | **358,43 Go** | COMPLETED (5h12) |
| 45874 | 1B, meme famille (resoumis h100) | 200G | **128,30 Go** | COMPLETED (2h59) |
| 45724 | 1B, meme run, tentative a100 | 200G | 47,11 Go (avant OOM) | FAILED -- OOM au chargement du juge Qwen, pas a l'extraction |
| 45725 | 12B, layer12, 1re tentative | 500G | 6,45 Go (avant crash) | FAILED -- cache legacy orphelin (cf. AUDIT_SAE_2026-08.md §9), pas un probleme memoire |

Le 358,43 Go de 45803 confirme quasi exactement le chiffre "358,5 Go" cite en
commentaire par le plan (C10) -- ici lu directement dans `sacct`, pas
seulement rapporte dans un script. Le point 128,30 Go pour 1B est une donnee
utile non presente dans le plan : a comparer, une fois E00 corrige, au
nouveau pic attendu avec codes documentaires compacts (§5.5).

### Point ouvert : propagation du signal de checkpoint

Les `.slurm` historiques utilisent `--signal=B:USR1@600` (`B:` = signal au
shell batch, pas necessairement au process Python -- avertissement explicite
du plan §16.4). Aucun de ces runs n'a eu besoin de reprendre sur checkpoint
jusqu'ici (tous COMPLETED ou FAILED net, jamais interrompus par le temps
limite) : **la reception reelle du signal par Python n'est pas encore
testee**. A couvrir par le test de crash/reprise E00 (§5.7) avant de compter
dessus pour un run long (E01 3 graines, E09).

## 4. Ecart d'echafaudage (§16.1)

`src/post_stage/`, `scripts/post_stage/`, `slurm/post_stage/`,
`tests/post_stage/` n'existaient pas. `configs/post_stage/` et
`docs/post_stage/` viennent d'etre crees pour ce preflight et
`campaign_policy.yaml`. Le reste du module est a implementer selon la
sequence E00→E09, pas encore ecrit.

## 5. Decision demandee a Gregoire avant premiere soumission sbatch

1. Confirmer les parametres de `configs/post_stage/campaign_policy.yaml`
   (repris tels quels du plan §1.4) ou les ajuster.
2. Confirmer qu'aucune modification de `RAPPORT_STAGE_ENTREPRISE.tex` n'est
   attendue de cette campagne (section 1/2 ci-dessus) -- l'echec de test
   restera visible tant que ce fichier est en cours de reprise.
3. Autoriser le premier profilage CPU/GPU borne d'E00 (job court, `--mem`
   dans l'enveloppe 64-128 Gio, cf. plan §5.7) comme premiere soumission de
   la campagne.

Aucune soumission Slurm n'a ete faite a ce stade.
