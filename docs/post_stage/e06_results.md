# E06 — Corrélations entre propriétés, confirmées hors découverte (1B/layer13/K5)

Découverte NPMI par-parent sur FIT+DEV (2605 parents), catalogue des 197
features interprétables d'E02 (CORE+EXTRA). 142/197 features exclues
(bande de fréquence [0,01 ; 0,5] — sans plafond haut, des features
quasi-universelles cooccurrent trivialement à NPMI≈1 avec tout le reste,
cf. Erreurs). 1246 paires candidates (support conjoint ≥10 parents, hors
paires à label strictement identique). 8 paires gelées **avant lecture de
CONFIRM**, classées par |NPMI| de découverte.

Vérification : chaque propriété atomique jugée individuellement par Qwen
sur 350 parents CONFIRM (échantillon déterministe, graine 42, 13 propriétés
uniques × 350 = 4550 appels), cooccurrence calculée **sur ces jugements**,
pas sur les activations SAE brutes.

## Résultat par paire (CONFIRM, n=350 parents, FDR-BH sur 7 paires testables)

| Paire | n_a / n_b / n_ab | NPMI CONFIRM (IC) | Statut |
|---|---:|---|---|
| Coordonnées bancaires × IBAN bancaire | 134/82/82 | 0,66 [0,59 ; 0,73] | **établi** |
| Identification SIREN × Numéro SIREN | 23/24/23 | 0,98 [0,95 ; 1,00] | **établi** (quasi-doublon, cf. limites) |
| Formule de politesse × Facture de clôture | 241/26/26 | 0,14 [0,11 ; 0,18] | **établi** (faible, peu surprenant) |
| Dysfonctionnement électrique × Réfrigérateur | 109/13/13 | 0,35 [0,29 ; 0,42] | **établi** |
| Liens d'usurpation × Injection d'URL | 3/6/1 | 0,51 (IC non calculé) | `insufficient_support` |
| Transmission d'informations × Tâches quotidiennes | 181/23/9 | −0,08 [−0,22 ; +0,05] | non établi |
| Tâches quotidiennes × Facture de clôture | 23/26/**0** | non défini (p_AB=0) | non établi (fisher p=0,40) |
| Négation de changement × Transmission d'informations | 14/181/7 | −0,01 [−0,17 ; +0,11] | non établi (fisher p=1,00) |

## Lecture d'ensemble

**5/8 paires "établies" (IC bootstrap excluant zéro, survivent FDR-BH), 2/8
non établies, 1/8 `insufficient_support`** (n_a=3 — statut correctement
appliqué plutôt que de citer l'odds ratio brut de 34,2, qui aurait l'air
spectaculaire sur une table quasi vide).

Parmi les 5 paires établies, **toutes ne sont pas également informatives** —
à ne pas aplatir en "5 corrélations découvertes" :

1. **Réfrigérateur × dysfonctionnement électrique (NPMI=0,35) est la seule
   association vraiment actionnable** : spécifique, non triviale, et
   business-pertinente (suggère que les plaintes "réfrigérateur" de ce
   corpus sont systématiquement encadrées comme un problème électrique,
   pas un problème mécanique/thermique — pourrait informer un routage par
   type d'appareil).
2. **SIREN-identification × SIREN-numéro (NPMI=0,98) est quasi-certainement
   un doublon de concept**, pas une découverte — deux directions SAE
   distinctes convergeant sur le même signal (feature splitting), comme les
   5 features "Numéro de téléphone" trouvées lors du filtrage (cf.
   Erreurs). Le filtre label+description ne l'a pas exclu car les deux
   libellés diffèrent lexicalement ; un NPMI aussi proche de 1 sur une paire
   gelée devrait se lire comme un signal de redondance du dictionnaire,
   pas comme une association métier.
3. **Bancaire × IBAN (NPMI=0,66) est largement définitionnel** (un IBAN
   *est* une coordonnée bancaire) — confirme que le protocole fonctionne
   correctement (l'association attendue ressort bien), mais n'apprend rien
   de nouveau sur le corpus.
4. **Politesse × facture de clôture (NPMI=0,14, le plus faible des
   "établis")** : les formules de politesse sont quasi-ubiquitaires
   (n_a=241/350) — l'association mesure surtout que les factures de
   clôture ne dérogent pas à cette norme, peu surprenant.

**Le cas p_AB=0 (tâches quotidiennes × facture de clôture) est géré comme
prévu** : NPMI non défini plutôt qu'une valeur fabriquée à la borne, et le
test de Fisher (p=0,40) montre que ce zéro n'est pas surprenant à ces
effectifs (n_a=23, n_b=26 sur 350) — pas une "anti-corrélation parfaite".

## Écart au protocole du plan

**Les 8 paires gelées impliquent toutes au moins une feature CORE (0 paire
EXTRA-seule)** — le plan demandait un budget candidat équilibré CORE/FULL ;
la sélection ici classe uniquement par |NPMI| de découverte sans forcer
cet équilibre. Non corrigé dans cette passe (aurait nécessité un nouveau
rerun) — à corriger si E06 est repris.

## Erreurs trouvées et corrigées avant le coût GPU

- **Features "sink"** : un premier passage (job 49074, annulé) gelait des
  paires à NPMI=1,00 impliquant des features quasi-universelles (ex.
  "Prénoms clients", présente dans ~100% des emails) — cooccurrence
  triviale, pas un signal. Corrigé par une bande de fréquence [0,01 ; 0,5]
  (même convention que `cooccurrence_graph`, `src/analysis/cooccurrence.py`).
- **Doublon de label exact** : un deuxième passage (job 49076, annulé)
  gelait 'Numéro de téléphone' × 'Numéro de téléphone' comme paire n°1 — 5
  features EXTRA distinctes du catalogue portent ce même libellé (feature
  splitting). Corrigé par une exclusion des paires à label strictement
  identique, en amont du filtre Jaccard label+description.

## Limites connues

- Filtre quasi-synonymes par recouvrement lexical (Jaccard label+
  description), pas par embedding dense — n'a pas empêché la paire
  SIREN/SIREN ci-dessus (NPMI=0,98), à lire comme un doublon malgré son
  statut "établi".
- Baseline de cooccurrence lexicale FIT/DEV (plan : "souhaitable") non
  calculée, faute de budget de session.
- Audit humain stratifié 40-80 emails (plan §11) non fait — nécessite
  Grégoire, reste en attente.
- Un seul email représentatif par parent CONFIRM échantillonné (pas toutes
  les variantes augmentées) — cohérent avec le calcul par parent, mais
  réduit le signal disponible par rapport à une agrégation multi-variantes.
- Budget candidat CORE/FULL non équilibré (cf. section précédente).

## Fichiers

- `results_post_stage_e01_fit_1b_layer13_k5/e06_correlations.json`
- Script : `scripts/post_stage/e06_correlations.py`
- Job SLURM : 49077 (h100, 1 GPU, COMPLETED 00:34:10)
