# Checklist de réception — passation SAE

**Version de référence :** `2f6fc2418e4ac084526dfc57b858431be426360b`.

Nom du repreneur : ____________________  Date de recette : ____________________

Cette liste est à compléter après vérification effective. Aucune case n'est précochée sur la seule base de l'audit GitHub.

## A. Accès et responsabilités

- [ ] Le repreneur accède au dépôt avec son propre compte.
- [ ] Un emplacement de projet autorisé est choisi pour données, modèles, résultats et logs.
- [ ] Les cibles des liens symboliques du cache sont accessibles depuis le compte repreneur.
- [ ] La disponibilité des données après le départ de Grégoire est confirmée.
- [ ] Les interlocuteurs données, modèles, métier et cluster sont identifiés.
- [ ] Les conditions de partage des modèles et des extraits ont été vérifiées par les responsables.
- [ ] Les limites Slurm et le budget de calcul restant sont confirmés ; le 100M n'est pas lancé par défaut.

## B. Identité du code et de l'environnement

- [ ] HEAD, modifications locales et sous-modules ont été inventoriés sans écrasement.
- [ ] `pyproject.toml`, `uv.lock`, Python 3.12 et l'environnement réellement utilisé sont archivés.
- [ ] Les identifiants et révisions exacts des poids Gemma, GemmaScope, bge-m3 et du juge sont enregistrés.
- [ ] Les scripts sont paramétrés pour le compte repreneur ; les chemins `/home/h21486/SAE` ne bloquent plus la recette.
- [ ] Le README distingue la référence historique 12B de la campagne post-stage 1B/FIT.
- [ ] Les commandes disponibles sont décrites correctement : `freeze-corpus` n'est pas une CLI universelle d'exécution.

## C. Données et indépendance

- [ ] L'ordre des parents utilisé pour l'augmentation est retrouvé.
- [ ] La jointure positionnelle est confirmée ou remplacée avec preuve de filiation.
- [ ] Les 70 variantes non rattachées sont expliquées.
- [ ] Les hashes de parents, variantes, fichiers source et ordre documentaire sont figés.
- [ ] L'absence de fuite est vérifiée sur les vrais parents, pas uniquement sur les affectations reconstruites.
- [ ] La conséquence d'un changement de mapping sur les checkpoints et résultats anciens est documentée.
- [ ] CONFIRM est décrit comme déjà consulté pour E03/E04/E06/E07.
- [ ] Le niveau document ou parent est indiqué pour les listes et les tests.

## D. Artefacts de référence

- [ ] Les JSON E01 à E07 sont accessibles ; les absences sont explicites.
- [ ] `e02_feature_registry.json` et les documents E03 sont transmis dans le périmètre autorisé.
- [ ] Les checkpoints référence, seed43 et seed44 sont préservés.
- [ ] Le statut des runs randinit45/randinit46 et de `e05_stability_random_init.json` est connu.
- [ ] Les fragments, memmaps et caches nécessaires à chaque expérience ont un propriétaire et une règle de conservation.
- [ ] Les logs permettent de retrouver les commandes et les ressources effectivement utilisées.
- [ ] Les petits artefacts ont des empreintes vérifiées ; le contrôle des gros volumes est planifié sans saturer le frontal.

## E. Corrections bloquantes de restitution

- [ ] E04 : `percentage_difference` transmet la bonne unité ; un test de non-régression existe.
- [ ] E04 : sens attendu, sens observé, significativité et métrique descriptive sont distincts.
- [ ] E04 : le traitement de la dépendance parent et la conservation des jugements sont explicités.
- [ ] E06 : l'incohérence « quatre vs cinq associations » est résolue depuis l'artefact.
- [ ] E03 : P@10 n'est plus décrit comme nécessitant un jugement exhaustif du corpus.
- [ ] E03 : le budget catalogue CORE/FULL et les répétitions d'un même parent sont visibles.
- [ ] E05 : la stabilité PCA commune n'est pas présentée comme indépendance à toute initialisation.
- [ ] Les documents historiques sont conservés ; les résultats remplacés sont reliés à leur successeur.

## F. Démonstration et pilote

- [ ] Le repreneur ouvre seul E08 sur CPU, sans charger de modèle.
- [ ] Il retrouve un résultat de recherche et un contre-exemple, et identifie la méthode correspondante.
- [ ] Il explique pourquoi FULL > CORE n'implique pas FULL > DENSE.
- [ ] Il ouvre E04 et identifie l'hypothèse inversée.
- [ ] Il ouvre E05 et distingue carte, stabilité des directions et stabilité des emails retrouvés.
- [ ] L'écriture des pistes E08 est déplacée hors Git avant tout usage sensible.
- [ ] Les sessions humaines réellement réalisées, ou leur absence, sont documentées sans ambiguïté.
- [ ] La liste des tâches reprises après le départ comporte des responsables et des critères d'acceptation.

## G. Validation technique réelle

- [ ] Le preflight a été exécuté et ses avertissements lus.
- [ ] Les tests convenus ont été exécutés sur la version remise ; le journal est conservé.
- [ ] Le chargement d'un cache transféré et les liens symboliques ont été testés.
- [ ] La recette de l'encodage CONFIRM ne déclenche pas de réentraînement involontaire.
- [ ] La reprise après interruption n'est annoncée comme garantie qu'après son test réel.
- [ ] Aucun gain business ni statut de déploiement n'est inféré de tests unitaires réussis.

## Décision de réception

Statut : ☐ reçue  ☐ reçue avec réserves  ☐ accès/artefacts bloquants

Réserves et responsables :

| Réserve | Responsable | Échéance | Preuve de résolution attendue |
|---|---|---|---|
| | | | |
| | | | |
| | | | |

Le compte rendu signé ou validé par les participants est le résultat de la passation. Le simple envoi d'un lien GitHub ne vaut pas réception.
