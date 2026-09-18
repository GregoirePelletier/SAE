# E08 — Recette du mini-pilote analyste

Ce document couvre uniquement la **recette technique** (plan §13,
5-7h d'ingénierie) : l'outil que des participants utiliseraient en séance.
**Les sessions elles-mêmes (2-3 participants réels dont Grégoire, 20 min
chacune) n'ont pas eu lieu** — elles nécessitent de vraies personnes
disponibles, ne peuvent pas être simulées, et restent la partie bloquante
de E08.

## Ce qui est construit

Page `Mini-pilote analyste (E08)` dans `src/visualization/dashboard.py`
(`page_pilot_e08`), lit exclusivement des artefacts déjà figés sur disque :

- `e01_representation_comparison.json` (repère CORE/FULL)
- `e03_property_retrieval.json` (tâche 1 : retrouver des emails selon une
  propriété — 12 requêtes, 5 méthodes, top-10 documents avec extrait de
  texte et pertinence jugée par Qwen)
- `e04_diffing.json` (tâche 2 : comparer deux sous-populations pour
  proposer des thèmes — 8 hypothèses, taux vérifiés, IC, FDR-BH)

Aucun modèle chargé, aucun GPU, aucun réseau externe au chargement de la
page (conforme à la recette minimale du plan). Message explicite (pas de
repli silencieux) si un artefact attendu est absent du run sélectionné.

**Catalogue de thèmes** : formulaire d'enregistrement d'une « piste »
(titre, question, populations, propriété, exemples, contre-exemples,
méthode, statut, commentaire humain, participant), exporté vers
`docs/post_stage/e08_pilot_leads.json` (append-only, y compris les pistes
rejetées). Vide tant qu'aucune session réelle n'a eu lieu.

Changement en amont nécessaire pour la tâche 1 : `e03_property_retrieval.py`
ne persistait que des métriques agrégées (P@10), pas les documents retrouvés
— un participant ne peut pas juger une piste sans voir de vrais candidats.
Ajout de `top_documents` (extrait de texte + pertinence par document,
mêmes rankings/jugements, aucun changement de protocole) et rerun (job
49058, h100).

## Ce qui reste à faire (bloquant, hors portée de cette session)

- Recruter 2-3 participants disponibles (le plan nomme explicitement
  Grégoire + collègues, à défaut un panel « utilisateurs de recherche »
  sans le qualifier de métier).
- Mener les sessions (20 min, ordre contrebalancé référence/SAE, cf. plan
  §13 format).
- Relecture aveugle des pistes collectées, mesure des indicateurs du plan
  (nombre de pistes documentées/jugées pertinentes, nouveauté, temps
  jusqu'à première piste, erreurs d'interface, confiance calibrée).
- Décision finale (poursuivre un pilote réel / garder l'exploration /
  privilégier la baseline / arrêter) — nécessite les données de session,
  ne peut pas être anticipée ici.

## Fichiers

- `src/visualization/dashboard.py::page_pilot_e08`
- `docs/post_stage/e08_pilot_leads.json` (catalogue, vide pour l'instant)
- Job SLURM : 49058 (E03 rerun avec `top_documents`)
