# E08 : mini-pilote avec des analystes

But : faire utiliser l'outil par 2 ou 3 personnes pendant une vingtaine de minutes chacune, pour
savoir s'il aide à retrouver des emails et à formuler des pistes. **Aucune séance n'a eu lieu.**
Seul l'outil a été préparé.

## L'outil

Page « Mini-pilote analyste (E08) » du dashboard (`page_pilot_e08` dans
`src/visualization/dashboard.py`). Elle ne charge aucun modèle et affiche des résultats déjà
calculés :

- tâche 1, retrouver des emails selon une propriété : les 12 requêtes d'E03, les 10 premiers
  emails de chaque méthode avec un extrait et le jugement de pertinence du modèle juge ;
- tâche 2, comparer deux populations : les 8 hypothèses d'E04 avec leurs taux et leur
  significativité ;
- un rappel des résultats d'E01 (CORE et FULL).

La page lit ces fichiers dans le run sélectionné ou dans son dossier d'évaluation (`<run>_eval`) ;
si l'un manque, elle l'indique et la tâche correspondante n'est pas proposée.

Un formulaire permet d'enregistrer une « piste » : titre, question, populations comparées,
propriété, exemples, contre-exemples, méthode, statut (retenue, rejetée, à creuser), commentaire
et participant. Les pistes rejetées sont gardées. Elles sont écrites dans
`$SAE_DASHBOARD_STATE_DIR/e08_pilot_leads.json` (par défaut `local_data/dashboard_state/`), hors
Git. `docs/post_stage/e08_pilot_leads.json` est un modèle vide ; s'il contient des pistes, elles
sont recopiées une fois vers l'emplacement local. Deux personnes qui enregistrent en même temps
peuvent écraser la piste de l'autre.

## Ce qu'il reste à faire

- Trouver 2 ou 3 participants. Si ce sont des chercheurs et non des analystes métier, le dire
  dans le compte rendu.
- Mener les séances en alternant l'ordre entre recherche classique et exploration par les
  features (plan §13).
- Relire les pistes sans savoir de quelle méthode elles viennent, et mesurer : nombre de pistes
  jugées pertinentes, nouveauté par rapport aux catégories connues, temps avant la première piste,
  difficultés d'interface.
- Décider de la suite : pilote sur données réelles, outil d'exploration seulement, ou arrêt.

## Fichiers

- `src/visualization/dashboard.py` (`page_pilot_e08`).
- `docs/post_stage/e08_pilot_leads.json` (modèle vide).
