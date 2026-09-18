# E04 — Diffing structuré urgence__panique vs urgence__calme (1B/layer13/K5)

Contraste `urgence__panique` (cible) vs `urgence__calme`. Découverte des
features candidates sur FIT (`corpus_diff_stats`, 8733 features actives dans
A∪B, top 200 au-dessus du seuil 0,03), génération de 8 hypothèses structurées
par Qwen **gelées avant toute lecture de CONFIRM**, vérification sur CONFIRM
(jamais vu pendant la découverte) — le juge ne voit ni le groupe ni la
méthode d'origine du document.

CONFIRM sous-échantillonné à 150 documents par groupe
(`--max-confirm-per-group=150`, graine 42) sur 849 A / 826 B disponibles,
pour tenir dans la fenêtre SLURM (job 48945 avait timeout à 3h sur
8×1675=13 400 appels ; job 48957, 8×300=2400 appels, COMPLETED en 1h00).

## Résultat par hypothèse (CONFIRM, IC Wilson, FDR-BH sur 8 tests)

| # | Hypothèse (résumé) | rate_in (panique) | rate_out (calme) | diff | IC diff exclut 0 ? | p (FDR-BH) |
|---|---|---:|---:|---:|:---:|---:|
| 1 | Urgence temporelle explicite | 1,000 | 0,000 | +1,00 | oui | 9e-67 |
| 2 | Colère / frustration agressive | 0,507 | 0,000 | +0,51 | oui | 8e-24 |
| 3 | Langage impératif, résolution concrète | 0,980 | 0,000 | +0,98 | oui | 2e-64 |
| 4 | Lien causal problème→conséquence | 0,813 | 0,033 | +0,78 | oui | 2e-42 |
| 5 | **Structure procédurale formelle** | 0,473 | 0,987 | **-0,51** | oui (sens inverse) | 2e-23 |
| 6 | Exclamations positives / énergie | 0,013 | 0,000 | +0,01 | **non** (p=0,156) | 0,156 |
| 7 | Patterns linguistiques de criticité | 1,000 | 0,000 | +1,00 | oui | 9e-67 |
| 8 | Fréquence de marqueurs de détresse | 1,000 | 0,000 | +1,00 | oui | 9e-67 |

`verification_rate=1,0` (les 8 hypothèses dépassent |diff|>0,01) et
`coverage=1,0` (chaque document `panique` de CONFIRM est couvert par au
moins une hypothèse validée dans le bon sens) — **mais ces deux métriques ne
distinguent pas le sens de la différence**, elles comptent l'hypothèse 5
comme "valide" alors qu'elle est vérifiée dans le sens opposé à celui généré.
Ne pas citer `verification_rate=1,0`/`coverage=1,0` sans cette précision.

## Lecture d'ensemble

1. **6 des 8 hypothèses sont confirmées dans le sens attendu, avec un effet
   massif et robuste** (|diff| ≥ 0,51, IC serrés n'incluant pas zéro, tous
   survivent la correction FDR-BH). Le contraste `panique`/`calme` de ce
   corpus est très largement séparable sur des propriétés textuelles
   explicites (urgence temporelle, impératif, lien causal, marqueurs de
   détresse) — cohérent avec un corpus augmenté généré par template, où ces
   marqueurs sont probablement quasi déterministes plutôt qu'une
   généralisation fine à documenter comme telle.
2. **L'hypothèse 5 ("structure procédurale formelle") est vérifiée mais dans
   le sens INVERSE de sa génération** : Qwen l'avait proposée comme trait du
   groupe `panique` (diff=+0,04 sur FIT au stade découverte) mais sur
   CONFIRM c'est le groupe `calme` qui l'exhibe massivement plus (98,7% vs
   47,3%). Lecture la plus probable : le ton calme du corpus s'exprime via
   une communication procédurale/formelle, et le stade découverte (magnitude
   de diff sur les features SAE, pas encore de jugement sémantique) a mal
   orienté le signe de cette hypothèse précise avant vérification — un
   exemple concret de pourquoi la vérification sur CONFIRM (jamais vue
   pendant la découverte) est nécessaire et pas une formalité : une
   hypothèse peut être un signal réel tout en étant générée avec le mauvais
   sens.
3. **L'hypothèse 6 ("exclamations positives") ne survit pas** — seule des 8 à
   ne pas passer le seuil FDR-BH (p=0,156, rate quasi nulle des deux côtés :
   1,3% vs 0%). Cohérent avec son diff de découverte déjà le plus faible
   (+0,04, confiance 0,70 la plus basse du lot) — le stade découverte avait
   déjà signalé cette hypothèse comme la moins fiable du groupe.

## Limites connues

- Corpus augmenté/synthétique (`urgence__panique`/`urgence__calme` sont des
  labels de génération, pas des catégories organiques) — la séparabilité
  quasi parfaite de 6/8 hypothèses reflète probablement des marqueurs de
  template plutôt qu'une propriété qui généraliserait à un corpus réel non
  labellisé de cette façon.
- Vérification 100% Qwen (juge découplé de l'extracteur Gemma, cf.
  [[project_sae_qwen_judge_policy]]), aucune calibration humaine — audit
  humain des résultats de diffing (plan §9.2 point 5) toujours en attente,
  nécessite Grégoire.
- `verification_rate`/`coverage` (App K.1) ne distinguent pas le sens de la
  différence — un lecteur pressé pourrait citer "8/8 hypothèses vérifiées"
  en passant sous silence l'inversion de sens de l'hypothèse 5 ; toujours
  lire le tableau par hypothèse, pas seulement le résumé agrégé.
- CONFIRM sous-échantillonné à 150/groupe (contrainte de fenêtre SLURM, pas
  un choix méthodologique a priori) — dans la fourchette basse de la cible
  du plan (§9.1 : "150-200 par groupe"), pas en dessous.
- Une seule paire de labels testée (`urgence__panique`/`urgence__calme`) —
  le plan prévoit d'autres contrastes (§9), non couverts ici faute de
  budget de session pour les prioriser tous.

## Fichiers

- `results_post_stage_e01_fit_1b_layer13_k5/e04_diffing.json`
- Script : `scripts/post_stage/e04_diffing.py`
- Job SLURM : 48957 (h100, 1 GPU, COMPLETED 01:00:30 ; job 48945, a100 2×GPU,
  TIMEOUT à 3h, avait motivé le sous-échantillonnage)
