# Extraction structurée des Appendices A–M — "Interpretable Embeddings with Sparse Autoencoders" (arXiv:2512.10092v2)

Source : `/home/h21486/SAE/docs/_paper_pages_raw.txt` (extraction pypdf brute de `/home/h21486/SAE/pdf/InterpretableSAE_Embeddings.pdf`), lignes 979 à 3587 (fin de fichier). Chaque section ci-dessous indique la plage de lignes source et la page PDF correspondante (numérotée d'après les marqueurs `===== PAGE N =====` du fichier).

Convention : les blocs de code/prompt sont reproduits **verbatim** (copié-collé direct de l'extraction), y compris les artefacts de rendu PDF quand ils ne changent aucun mot (guillemets courbes `’`, espaces parasites autour de `**`, etc.). Je n'ai corrigé que la césure de mots visiblement fusionnés dans les titres de section (ex. `ADDITIONALRELATEDWORK` → `ADDITIONAL RELATED WORK`). Aucun mot, formule ou valeur numérique n'a été modifié. Toute portion illisible ou ambiguë est signalée inline avec `[EXTRACTION INCERTAINE]` et listée aussi dans la section finale.

---

## A. METHODS (lignes 979–1078, page 17)

Cette section est entièrement composée du texte extrait de la **Figure 10** ("Detailed methodology for each of the four tasks"), un diagramme en 4 panneaux (Retrieval, Clustering, Correlations, Data Diffing). Le texte pypdf est désordonné (labels de boîtes de flowchart extraits hors-ordre), reconstruit ci-dessous par panneau logique.

### A.1 Retrieval (panneau 1)
Étapes du pipeline, telles qu'énoncées dans le diagramme :
1. Chaque document est représenté par son vecteur d'activation SAE. **Normaliser chaque latent par le 90e percentile des activations non-nulles à travers le dataset** (`Normalize each latent by 90th percentile of non-zero activations across dataset`).
2. Requête exemple : *"The model admits it lacks information."* → on prend l'embedding de phrase de la requête, puis on récupère le top 100 latents pertinents par similarité sémantique des labels (ex. labels candidats : "Assistant states it lacks information", "Assistant says it does not know", "Assistant cites knowledge cutoff", "Assistant seems unsure").
3. On demande à un juge LLM de choisir et reclasser (rerank) les labels pertinents parmi ces candidats → il reste *k* latents. Exemple de sortie donné : "Assistant requests additional details", "Assistant states it lacks information", "Assistant says it does not know", "Assistant cites knowledge cutoff", "Assistant seems unsure".
4. Pour le latent pertinent *i* au rang *r_i* parmi *k*, son poids *w_i* est donné par (texte brut extrait, **probablement corrompu par l'OCR/extraction, voir Points d'incertitude**) :
   ```
   𝑤𝑖 = 𝑒 Τ−𝑟𝑖
   𝑘 𝑇  où 𝑇 est la température
   ```
   [EXTRACTION INCERTAINE — la formule exacte de pondération par rang n'est pas reconstructible fiablement depuis le texte pypdf ; il s'agit très probablement d'une décroissance exponentielle du type `w_i = exp(-r_i / (k·T))` ou `w_i = exp((k - r_i)/(k·T))`, mais le symbole exact et la position de T ne peuvent pas être confirmés sans consulter le PDF source directement (page 17, Figure 10, étape 4).]
5. Score final par document = produit scalaire pondéré `𝑤 ∙ Ԧ𝑎` (poids **w** produit scalaire avec le vecteur d'activation **a**), puis on classe les documents par ce score.

### A.2 Clustering (panneau 2)
1. Représenter chaque document par son vecteur d'activation SAE **binarisé**.
2. Requête exemple : *"step by step reasoning"*. (Optionnel) filtrer aux *k* latents pertinents à la requête par similarité sémantique des labels.
3. Calculer la similarité entre chaque paire de documents.
4. Clustering par **spectral clustering**.
5. Diff des textes intra-cluster vs. hors-cluster pour obtenir les top features distinctives qui décrivent le cluster.
6. (Optionnel) demander à un LLM de décrire le cluster à partir des top features et exemples.

### A.3 Correlations (panneau 3)
1. Chaque latent a un vecteur d'activation binaire sur le dataset. Exemple : f1="Offensive request from the user", f2="Narrative transitions in fiction".
2. Pour chaque paire de latents : (2a) trouver l'information mutuelle des occurrences ; (2b) trouver la similarité sémantique des labels.
3. Tracer chaque paire de latents (similarité sémantique en x, information mutuelle en y) et examiner les paires candidates : paires de latents similaires, paires de latents non-liées, et "paires candidates" = corrélées mais avec des labels différents.

### A.4 Data Diffing (panneau 4)
1. Chaque dataset est représenté par la fréquence des latents SAE.
2. Soustraire les fréquences d'un dataset à l'autre → différence de fréquence.
3. Examiner les top différences (latents fréquents dans un dataset, rares dans l'autre) ; le label donne une hypothèse initiale (ex. "step by step reasoning").
4. (Optionnel) utiliser les phrases activantes du dataset et demander à un LLM de relabelliser la feature pour obtenir de meilleures hypothèses (ex. "logical deductions in math puzzles").

---

## B. ADDITIONAL RELATED WORK (lignes 1081–1116, page 18)

Section de revue de littérature, pas de prompts ni de paramètres numériques d'implémentation. Résumé fidèle par sous-section :

- **B.1 DATA DIFFING** (ligne 1082) : les embeddings sémantiques quantifient le degré de différence via similarité cosinus mais ne décrivent pas *comment* les textes diffèrent ; les statistiques term-based peuvent manquer le contexte ; travaux antérieurs sur la description de différences entre datasets utilisent donc principalement des LLMs (réfs [51], [52]).
- **B.2 CORRELATIONS** (ligne 1087) : la recherche de corrélations est souvent encadrée comme recherche de corrélations parasites (spurious) entre features et classes de dataset — cite [53] : une feature SAE prédictive du label humain-vs-IA d'un texte s'est révélée activer principalement sur la ponctuation, signe d'une corrélation potentiellement non-généralisable. Trouver des corrélations concept-concept arbitraires sans labels reste peu exploré ; les approches classiques mesurent des corrélations terme-terme ([54],[55]), les SAE permettent une extension latent-latent.
- **B.3 CLUSTERING** (ligne 1096) : le NLP classique représente les textes en term-based ([56]) ou embeddings denses ([7]), puis applique un algorithme standard (KMeans [57], spectral clustering [58], HDBSCAN [59]). Pour guider les clusters vers une structure spécifiée par l'humain : contraintes pairwise ([60],[61]), exemples-graines ([62]), labels partiels ([63]), feature feedback ([64]), tuning post-hoc ([65],[66]), parfois avec guidage LLM ([67],[68]). [69] a appliqué des embeddings SAE au clustering de descriptions d'entreprises sans exploiter leur contrôlabilité.
- **B.4 RETRIEVAL** (ligne 1103) : la plupart des benchmarks de retrieval se concentrent sur le QA et la similarité sémantique. [44] étudie le retrieval basé sur une *description* du contenu (ex. requête "a company which is a part of another company" → "Pecten (company), a subsidiary of Sinopec"). Ce papier étend cela à des requêtes plus abstraites sur des propriétés implicites. Les embeddings decoder-only LLM modernes surpassent désormais les méthodes BERT-style via pooling last-token/latent-attention, formatage d'instructions, et/ou finetuning ([70]-[73]). Les SAE sont utilisées comme approximation de ces embeddings, avec l'interprétabilité aidant à comprendre les résultats de retrieval — cite [19],[20] pour des SAE entraînées sur des embeddings sémantiques utilisées pour contrôler le retrieval.

### Table flottante — Table 4 (lignes 1120–1193, page 19)

Cette page contient une table isolée (pas de titre de sous-section sur cette page), physiquement positionnée entre B et C dans le flux d'extraction PDF, mais référencée plus tard dans le texte de **l'Appendice J** ("We show qualitative examples of 'good' labels from the most and least predictable deciles in 4"). Il s'agit donc du **Tableau 4**, une table flottante (float LaTeX) appartenant conceptuellement à l'Appendice J bien que positionnée en page 19 dans le flux de pages du PDF.

**Table 4** : "Sample of latents that are most (top decile) and least (bottom decile) predictable by NAP in each frequency bin with autointerp scores > 70% (i.e. 'good' latents)." — contient 3 bins de fréquence (`[0.016, 0.022)`, `[0.062, 0.088)`, `[0.125, 0.177)`), pour chacun 5 labels "Most Predictable" et 5 "Least Predictable" avec leur score autointerp (%). Exemples de scores : 100% pour "References to color in programming or styling contexts" et pour "The act of attempting or making an effort to do something...". Contenu qualitatif, non reproduit intégralement ici (voir lignes 1120–1193 pour le détail complet des 30 labels et scores).

---

## C. LATENT LABELING PROMPTS (lignes 1197–1244, page 20)

> "We follow prior work [26] to relabel latents."

**Relabeling latents.** Pour relabelliser un latent avec une description plus précise : on passe dix documents activants et dix documents non-activants à un LLM pour inférer quand le latent s'active. Pour un latent donné, tout token où son activation est > 0 est marqué avec `«` et `»`. Prompt utilisé (verbatim) :

```
You are an expert at interpreting features from sparse autoencoders (SAEs) for language models.
Below are {len(positive_samples)} POSITIVE samples (where the feature activated, with tokens surrounded by <<
and >>) and {len(negative_samples)} NEGATIVE samples (where it did not activate, no << >> markers).
The POSITIVE sample contains tokens that caused the feature to activate (marked with << >>), while the
NEGATIVE sample does not.

IMPORTANT NOTES:
1. The << >> markers indicate where the feature activated, but you should NOT restrict your understanding to
just those marked tokens. Look at the context BEFORE the marked tokens as well - the preceding tokens
often provide crucial information about what the feature is detecting.
2. The feature may be responding to a pattern or concept that spans both the marked tokens AND the tokens
before the marked token.
3. The token <eot_id> is an end-of-sequence (EOS) token and should NOT be considered as a valid feature
activation. If you see <<eot_id>> in the samples, ignore it as it's just a technical marker for the end
of text, not a meaningful activation.
{refinement_context}

POSITIVE SAMPLES(given as a list of strings):
{positive_samples}

NEGATIVE SAMPLES(given as a list of strings):
{negative_samples}

Your task:
- Carefully compare the POSITIVE and NEGATIVE samples
- Look at BOTH the tokens before the << >> markers AND the marked tokens themselves to understand what the
feature is detecting.
- Identify the most specific and concise property that is present in the POSITIVE samples (considering both
context and marked tokens), but absent in the NEGATIVE samples.
- Try to give a unified property that isn't just a list of properties, if possible.
- Summarize the common attribute or property that causes the feature to activate. Be as specific as possible,
but keep your description concise and clear.
- Do not reference specific sample numbers; however, you can reference the content in the positive and
negative samples

Return your answer as a JSON object with exactly these fields:
- "label": "A concise phrase describing the property present in the positive samples (considering both context
and marked tokens) but not in the negative samples."
- "brief_description": "A sentence expanding on the label, explaining what the feature is detecting in more
detail. This should be a single sentence, not a list of properties. Please phrase this as: "This
document contains X, discusses X, etc.", where X is the property.
{"- 'detailed_explanation': 'An extended explanation of what this feature is detecting, including how the
context before the marked tokens contributes to the feature's meaning. The explanation should be
sufficient on its own to understand what the feature detects. Keep it to <5 concise sentences.'" if
explanation else ""}

Make sure your response is valid JSON that can be parsed directly.
```

Ce prompt est réutilisé tel quel dans plusieurs autres appendices (D.2 pour la relabellisation lors du diffing, H pour le case-study OpenAI, I pour l'ablation de taille de modèle).

---

## D. ADDITIONAL RESULTS — DATASET DIFFING (lignes 1247–1866, pages 21–29)

### D.1 LLM BASELINE DETAILS (lignes 1248–1327, pages 21–22)

Baseline adapté de l'étape "hypothesis discovery" de [22], qui identifie des différences qualitatives entre modèles. Étant donné deux datasets (ou un dataset vs plusieurs), la baseline trouve d'abord des différences entre paires de documents (ex. réponses au même prompt) avec le prompt suivant (verbatim) :

```
Analyze the differences between Model A and multiple Model B responses.
**User Prompt: **
{prompt}
**Model A Response: **
{model_a_response}
**Model B Responses: **
{model_b_section}
1. Properties/capabilities that Model A has but NONE of the Model B responses have
For each difference, provide a JSON object with:
- "category": The type of difference (e.g., "Style", "Content", "Technical", "Reasoning", "Accuracy")
- "property": Specific property being compared
- "difference_type": Either "unique_to_a" (present in A but none of B models) or "common_to_all_b" (present in
all B models but not A)
- "impact": "Low", "Medium", or "High"
- "description": Brief explanation of the difference
Return your analysis as a JSON array of difference objects.
```

Pour trouver les différences les plus communes, on résume ou clusterise ces objets en hypothèses.

**Résumé par batch** (batch summarization, car les objets de différence peuvent excéder la fenêtre de contexte du LLM) : chaque batch contient les objets de différence pour 100 prompts. Prompt verbatim :

```
Summarize the following dataset comparison patterns for the query: "{query}"
Batch data:
[JSON difference objects]
Provide a detailed summary of the key patterns relevant to the query. For each pattern, include:
- Pattern name
- Brief description
- Rough frequency (e.g., "seen in 20% of examples")
- 1-2 representative examples
```

**Agrégation en hypothèses** (au plus 10 hypothèses, en utilisant Gemini 2.5 Flash) — prompt verbatim :

```
You are an expert AI researcher analyzing behavioral differences between two language models.
You have been given a dataset of differences from {num_pairs} analyzed response pairs.
Query: {query}
Differences: {batch_summaries}
Based on the provided data, identify at most {num_hypotheses} significant differences that respond to the
query. I'm looking for differences of the format Model A/B is more X than Model B/A, where X is the
difference. For each difference, provide:
1. **Description**: Describe a response that would validly have property X. Start with "This response .." Use
1-2 sentences to clearly and specifically describe the property, such that using this description could
be used to identify the property on its own. Do not mention the model names.
2. **Detailed Description **: A detailed explanation of what the difference is and why it's significant
3. **Model A/B **: The model that exhibits this property more
4. **Percentage Difference **: An estimate of how much more frequently Model A exhibits this behavior compared
to Model B. If the property is more frequent in Model A, the percentage difference should be positive.
If the property is more frequent in Model B, the percentage difference should be negative.
5. **Examples**: 2-3 specific examples that demonstrate this difference
Make hypotheses specific and clear. Provide at most {num_hypotheses} differences in the following JSON format:
{{"differences": [
{{
"description": "Clear description of the property",
"detailed_description": "Detailed explanation of the difference and why it's significant",
"model_a_b": "Model A|Model B",
"percentage_difference": "X% more present in Model A",
"examples": [
{{
"prompt": "Original prompt text or description",
"explanation": "Why this example demonstrates the difference"
}}
]
}}
]}}
```

**Clustering des différences** (méthode alternative à la summarization) : les descriptions de différences sont embeddées avec `text-embedding-3-small` d'OpenAI. Algorithme **KMeans**, nombre de clusters fixé à **10**. Label de cluster formé à partir des **top 5 représentants les plus proches du centroïde**. Prompt de labellisation de cluster (verbatim) :

```
You are analyzing a cluster of similar model behavior differences.
Representative differences in this cluster:
{differences}
Provide a concise sentence that captures the common theme or pattern
across these differences. Focus on what makes this cluster distinct, and create a description that can be used
to identify Model A's behavior by starting with "This response...". Do not mention Model B, just focus
on Model A's unique characteristics that are NOT in Model B at all.
```

### D.2 HYPERPARAMETERS AND PROMPTS FOR SAE HYPOTHESIS GENERATION (lignes 1328–1380, page 22)

**Conversion des différences de latents en hypothèses.** Pour deux datasets A et B, pour chaque latent *i*, on calcule le pourcentage de documents de chaque dataset ayant au moins un token où le latent *i* s'active. On extrait les **top 200 latents** ayant la plus grande différence de fréquence au-dessus d'un certain seuil, fixé à **0.03** dans les expériences du papier. Chaque latent différent est ensuite relabellisé selon la procédure de l'Appendice C. Enfin, comme les descriptions de latents peuvent se recouper, un LLM résume ces latents (représentés par une brève description, un document activant, un document non-activant) en hypothèses concises et distinctes, avec le prompt suivant (verbatim) :

```
You are analyzing differences between two datasets. Below are the most significant features that are
differences between a "target" and "other" dataset:
IMPORTANT NOTES:
1. The << >> markers in examples indicate WHERE features activated, but you should NOT restrict your
understanding to just those marked tokens. The context BEFORE the marked tokens often provides crucial
information about what the feature is detecting.
2. Features often respond to patterns that span both the preceding context AND the marked tokens together.
3. The token <eot_id> is an end-of-sequence (EOS) token and should NOT be considered as a valid feature
activation. If you see <<eot_id>> in the samples, ignore it as it's just a technical marker for the end
of text, not a meaningful activation.
4. Note that some features are not accurate. If the feature description does not accurately describe the
tokens marked with << >>, you should disregard the feature. Only use features that you are certain are
valid.
5. Please ensure that all hypothesis descriptions are clearly distinct from each other. You do not need to
generate the exact amount of hypotheses to meet the quota.
6. Each feature will have a "difference strength", which is the percentage difference between the target and
other dataset. If it is positive, the target dataset has more of the feature than the other dataset. If
it is negative, the other dataset has more of the feature than the target dataset.
7. Please try to make each hypothesis specific, focused, and distinct from each other.

USER QUERY: {query}

Generate at most {num_hypotheses} hypotheses that answer the user's query for the "target" dataset. I'm
looking for differences of the format Dataset A is more X than Dataset B, where X is the difference.
Each hypothesis should be formatted as a JSON object with these exact fields:
- "dataset": "target" or "other" (the dataset that has more of this property)
- "description": Describe a response that would validly have property X. Start with "This response .." Use 1-2
sentences to clearly and specifically describe the property, such that using this description could be
used to identify the property on its own. Do not mention the model names. Be specific so that responses
that don't have this property could not be misclassified as having this property based on this
description.
- "feature_ids": List of feature ID(s) that support this hypothesis. It could be a list of a single feature ID
, or a list of multiple feature IDs.
- "examples": List of examples. Provide at most 3 examples. Be concise. For each example, cite the feature ID
and feature description and explain how the positive / negative example pairs from the dataset
illustrate the hypothesis, considering both the marked tokens AND their preceding context). You should
just highlight the portion of the example pairs that are relevant for the feature; do not print out the
entire positive / negative example pairs unless it is necessary to understand the feature.
- "percentage_difference": 0.XX (the percentage difference, between -1 and 1). Use the maximum difference
strength among the features used. Positive percentage if target has more of this property, negative
otherwise.
- "confidence": 0.XX (confidence in this hypothesis, between 0 and 1)
Remember that <eot_id> tokens should be ignored as they are just EOS markers, not meaningful feature
activations.
Return the response as a JSON array of at most {num_hypotheses} hypothesis objects. Make sure the JSON is
valid and can be parsed directly.
```

**Hyperparamètres clés D.2 :** top 200 latents extraits par différence de fréquence ; seuil de différence de fréquence = **0.03** ; nombre max d'hypothèses = `{num_hypotheses}` (paramétrique, non fixé numériquement dans le texte).

### D.3 GROUND TRUTH EVALUATION (lignes 1383–1452, pages 23–24)

**Datasets ground-truth (Table 5, lignes 1386–1396) :**
- *Synthetic: tone changes* — 500 réponses échantillonnées aléatoirement de Chatbot Arena [74], converties par GPT-4o en 13 tons différents (ex. "friendly-and-personable"). Diff entre réponses modifiées et réponses de base, objectif : récupérer le ton.
- *Real-world: movie genre differences* — dataset IMDB-reviews [33] (descriptions de films avec labels de genre). Diff entre descriptions d'un genre donné et 500 descriptions échantillonnées aléatoirement hors de ce genre, objectif : récupérer le genre.

**Évaluation quantitative :** mesure de la **similarité de surface** entre les top 5 différences de latents et le label ground-truth (tone/genre), en utilisant **GPT-5**. Suivant [21], on échantillonne **5 fois** avec **température 0.7**. Baseline simple : les deux datasets comparés sont donnés à GPT-5 avec une demande de description en une phrase de la top différence.

**Résultats numériques :**
- SAE : similarité de surface moyenne de **0.75** pour le dataset movies et **0.80** pour le dataset tones.
- Baseline LLM : score moyen de **0.90** pour movies et **0.78** pour tones.

**Prompt de similarité de surface** (édité légèrement depuis [21]) — verbatim :

```
Is text a and text b similar in meaning?
First, provide your reasoning about how text a and text b relate to each other.
Then, respond with yes, related, or no.
If text b has multiple items in commas, you should use the closest match with text a. Respond yes if text b
captures the spirit of text a. Respond related if text b is related to text a but not exactly the same.
Respond no if text b is not related to text a at all.

Here are a few examples.
Example 1:
text a: has a topic of protecting the environment
text b: has a topic of environmental protection and sustainability
output: yes

Example 2:
text a: has a language of German
text b: has a language of Deutsch
output: yes

Example 3:
text a: has a topic of the sports
text b: has a topic of sports team recruiting new members
output: yes

Example 4:
text a: has a topic of the relation between political figures
text b: has a topic of international diplomacy
output: related

Example 5:
text a: has a named language of Korean
text b: uses archaic and poetic diction
output: related

Example 6:
text a: describes an important 20th century historical event
text b: describes a 20th century European politician
output: related

Example 7:
text a: has a named language of Korean
text b: has a named language of Japanese
output: no

Example 8:
text a: talks about the history of the United States
text b: talks about dinosaurs
output: no

Target:
text a: {text_a}
text b: {text_b}
output:
```

**Échelle de score de la tâche (rappel des consignes utilisateur) :** {1, 0.5, 0} correspond respectivement à yes/related/no — cette correspondance numérique n'apparaît pas explicitement dans le texte extrait (le texte ne donne que les labels yes/related/no, pas leur mapping numérique 1/0.5/0). [EXTRACTION INCERTAINE — voir Points d'incertitude : le mapping yes=1, related=0.5, no=0 est probablement défini ailleurs dans le corps principal du papier (Section 4.1, hors du périmètre de cette extraction Appendix A-M) ou en note de bas de tableau, non retrouvé dans les lignes 1247-1452.]

### D.4 COMPARING MODEL OUTPUTS (lignes 1453–1515, pages 24–25)

Pas de prompt supplémentaire ici. Contenu : mesure du **verification rate** (fréquence à laquelle une hypothèse a une différence de fréquence vérifiée par le juge > 1%) en Figure 11 — les SAE ont des taux de succès plus élevés que les baselines LLM en comparaison multi-modèles. Figure 12 : couverture globale des hypothèses générées (% de réponses où au moins une hypothèse s'applique uniquement à la target). Table 6 (ligne 1511) : décompte de tokens utilisés par modèle (LLaMA, Gemini, Embed-small) pour SAE vs LLM-S vs LLM-C, sur 3 settings (Multi-model, LLaVA v. Vicuna, Deploy/Eval v. default prompt). SAE moins coûteux que les LLMs en settings comparatifs répétés.

### D.5 GENERATED HYPOTHESES FOR MODEL COMPARISONS (lignes 1516–1866, pages 25–29)

**Modèles frontier étudiés (ligne 1518-1520) :** Grok-4, GPT-OSS-120B, Gemini 2.5 Pro, Claude Opus 4.1, Claude Sonnet 4, GPT-5, Llama 4 Maverick, Deepseek R1, Qwen3-235b, Qwen3-235b thinking. Caractéristiques uniques extraites pour Grok-4, GPT-OSS-120B, Gemini 2.5 Pro vs les autres.

Contenu principal : **Table 7** (hypothèses via embeddings SAE), **Table 8** (LLM-S : generate + summarize), **Table 9** (LLM-C : generate + cluster). Ce sont des tableaux de résultats qualitatifs (hypothèse textuelle + différence de fréquence % + modèle "other" de comparaison), pas des prompts ni des paramètres de méthode — non reproduits intégralement ici (voir lignes 1530-1865 pour le détail complet des ~50 hypothèses par table). Un hypothèse est considérée "valide" si sa différence de fréquence vérifiée est > 1% (rappel du critère D.4).

---

## E. ADDITIONAL RESULTS — CORRELATIONS (lignes 1869–2288, pages 29–34)

### E.1 CORRELATION METRIC & BASELINES (lignes 1870–1925, page 29)

Hypothèse de départ : les paires de latents à labels sémantiquement similaires sont conceptuellement liées et ont donc des occurrences corrélées dans les documents, tandis que les paires à labels dissemblables sont non-liées et ne devraient pas avoir d'occurrences corrélées. La région intéressante est donc celle où des latents à labels dissemblables ont des occurrences corrélées.

La similarité sémantique des labels sert de proxy pour la relation entre deux latents. Deux métriques de corrélation/co-occurrence considérées :

1. **Normalized Pointwise Mutual Information NPMI(i, j)** — mesure symétrique de combien deux latents co-occurrent plus que le hasard. Liée au PMI, qui est le logarithme de :
   ```
   PMI(i,j) = log( P(i|j)/P(i) ) = log( P(j|i)/P(j) ) = log( P(i,j) / (P(i)·P(j)) )
   ```
   [EXTRACTION INCERTAINE — le texte définit explicitement le **PMI** (formule ci-dessus reconstruite fidèlement à partir de "P(i|j)/P(i) = P(j|i)/P(j) = P(i,j)/(P(i)P(j))"), mais **la formule de normalisation exacte donnant le NPMI à partir du PMI n'est pas donnée dans le texte extrait** — seul le nom "Normalized pointwise mutual information NPMI(i,j)" apparaît, sans l'équation de normalisation (standard : NPMI = PMI / -log P(i,j), mais ceci n'est pas confirmé texte-en-main). À vérifier sur le PDF source (page 29, début E.1).]

2. **Conditional Occurrence CO = max(P(i|j), P(j|i))** — mesure plus interprétable, capture les corrélations directionnelles (ex. "most text about X race is offensive"). Ne contrôle pas pour la fréquence individuelle de chaque latent.

**Protocole de comparaison (Figure 13) :** on trace la métrique de corrélation contre la similarité sémantique, pour **1M paires échantillonnées d'un sous-ensemble de 5k du Pile**. Observation : il y a plus de paires à CO élevé qu'à NPMI élevé, rendant plus difficile de choisir un seuil séparable — d'où le choix du **NPMI comme métrique principale**.

**Filtres appliqués pour réduire l'espace de recherche des paires :**
- **Filtre des labels syntaxiques** : on ignore les paires dont au moins un label est jugé "syntaxique" par un LLM (moins intéressantes) — voir le prompt exact de ce filtre en Appendice K.2.
- **Filtre des paires triviales** : certaines paires co-occurrent seulement parce qu'elles co-activent sur le **même token ou des tokens consécutifs** (mal labellisées, référant en fait au même concept, ou déclenchées toutes deux par un token rare) — filtrées additionnellement dans l'analyse "real-world" (E.3).

### E.2 RECOVERING KNOWN CORRELATIONS (lignes 1926–1967, pages 29–30)

**Protocole :** corpus de **10k textes** avec **0.1%–1.0% de textes injectés** (corrélations connues). Mesure : à mesure que le nombre de textes injectés augmente, le pourcentage de paires pertinentes parmi le groupe découvert augmente.

**Seuils de découverte (Figure 14) :** groupe de paires découvertes défini par **NPMI > 0.8** et **similarité sémantique < 0.2**, avec **0.5% de textes injectés**. Figure 14(e) montre la proportion de paires pertinentes dans le groupe candidat pour différents niveaux d'injection 0.1%-1%. Figure 14(f) : injection des 3 textes simultanément.

**Table 10 (mots-clés de jugement de pertinence des paires, ligne 1937-1944)** — verbatim :

| Injection | Latent 1 Relevant | Latent 2 Relevant |
|---|---|---|
| croatian-emoticons | croatian, russian, slavic | emoticon, emoji |
| baseball-slang | valley girl, slang, endearment | game, sport, baseball |
| conservative-academic_style | economic, political, business | academic, formal |
| conservative-academic_slant | economic, politic, business | communis, free, libert, interven, interfer |

**Baseline LLM (E.2, lignes 1945–1966) :** dataset séparé en **10 batches de 1k textes**, pour chaque batch on demande au LLM jusqu'à **10 corrélations** de features "meaningfully different". On compte le nombre de batches où une corrélation liée à chaque corrélation injectée est découverte (**Table 11**) :

| Injection | Taux d'injection | Batches où découverte (sur 10) |
|---|---|---|
| croatian-emoticons | 0.2% / 0.5% / 1.0% | 0/10, 2/10, 1/10 |
| baseball-slang | 0.2% / 0.5% / 1.0% | 0/10, 4/10, 10/10 |
| conservative-academic_style | 0.2% / 0.5% / 1.0% | 0/10, 1/10, 2/10 |
| conservative-academic_slant | 0.2% / 0.5% / 1.0% | 1/10, 3/10, 6/10 |

### E.3 FINDING REAL-WORLD CORRELATIONS (lignes 1968–2288, pages 29–34)

Pour créer la Figure 4, comparaison de la distribution des NPMI découverts par la méthode SAE avec 3 autres méthodes :

**1. Random SAE baseline.** Échantillonnage aléatoire de **n=100 paires de latents SAE** (de fréquence suffisante), relabellisation de chacune, vérification de la présence dans le dataset par un LLM, calcul du NPMI vérifié. La plupart des paires aléatoires ont un NPMI bas, confirmant que la sélection par NPMI élevé fournit un signal fort.

**2. LLM baseline.** Prompt pour identifier des corrélations de features "meaningfully different" (verbatim) :

```
You are given a dataset of {n_samples} documents.
Your task is to identify **co-occurrences of meaningfully different features **. A **co-occurrence**
refers to when two features both appear **WITHIN the same document **.
Each **feature** can be:
- A topic, subject, concept, or idea
- A specific language, style, tone, or sentiment
- A specific linguistic, rhetorical, or syntactic pattern
- Or any other identifiable textual property
We are interested in feature pairs that co-occur more than once across the dataset, i.e. the same
feature pair co-occurs in multiple documents, even if only in a few documents.
We are only interested in feature pairs where the two features are **meaningfully different **. This
means the two features cannot be trivially similar or extremely related.
Feature pairs can involve different feature types that co-occur, for example, between two
semantically different concepts, or between a linguistic pattern and a concept, or between a
linguistic and formatting pattern.
We are especially interested in feature co-occurrence pairs that are surprising, unexpected,
interesting, or otherwise notable, even if this co-occurrence occurs only in a few documents.
Each feature in a pair should be described with a precise phrase that describes what the feature is
about.
Return your answer as a JSON object with the following format, with up to 10 feature pairs:
{{
"feature_pairs": [
{{
"feature_1": "feature_1_description",
"feature_2": "feature_2_description"
}},
{{
"feature_1": "feature_1_description",
"feature_2": "feature_2_description"
}},
...
]
}}
{"\n".join([f"---BEGIN DOC {i+1}---\n{text}\n---END DOC {i+1}---" for i, text in enumerate(
sampled_texts)])}
```

On prend les paires de features générées par le LLM, on vérifie la présence de chaque feature dans le dataset avec un LLM, et on calcule le NPMI vérifié. Max **10 paires de features par batch**.

**3. Correlated Topic Model (CTM).** Entraînement d'un CTM [75; 76] pour découvrir des topics par co-occurrence de mots. **n_topics fixé à 100.** Un topic est considéré présent dans un document s'il est parmi les **top 5 topics** du document. Cela donne les occurrences des 100 topics découverts, à partir desquelles on calcule le NPMI vérifié. Les NPMI du CTM tendent à être bas, suggérant que le CTM n'est pas adapté à la découverte de topics fortement corrélés.

**CO comme métrique complémentaire (Figure 15) :** distribution du **conditional occurrence (CO)** (Appendix E.1) parmi les paires découvertes par toutes les méthodes, pour confirmer que même avec un seuil NPMI, la méthode SAE trouve des paires à CO élevé ("truly correlated"). Tailles d'échantillon annotées dans la Figure 15 :
- CivilComments : SAE Pairs (NPMI_SAE > 0.6, sim < 0.2, n=759) ; SAE Pairs Random (n=100) ; LLM Pairs (n=50) ; CTM (n=4950).
- Pile : SAE Pairs (NPMI_SAE > 0.7, sim < 0.2, n=1283) ; SAE Pairs Random (n=100) ; LLM Pairs (n=50) ; CTM (n=4950).

**LLM hypotheses for real-world correlations (lignes 2054–2059).** Pour CivilComments (5k), Pile (5k) et Tulu (10k) : shuffle et split en **batches de 1k documents chacun**. Pour chaque batch, demande au LLM jusqu'à **10 hypothèses intéressantes**. Pour CivilComments (Table 12) et Pile (Table 13), vérification de la présence de chaque concept sur un **échantillon de 1k** du même dataset. Pour Tulu (Table 14), le LLM baseline n'a pas trouvé la corrélation "math and hope" ; 20 échantillons aléatoires des 100 hypothèses sont montrés.

**Tables 12, 13, 14 (lignes 2063–2287)** : tables de résultats (concept 1, concept 2, NPMI, CO, P(C1|C2), P(C2|C1)) pour CivilComments (49 paires), Pile (37 paires), et Tulu (échantillon de 20 hypothèses format requête/réponse). Ce sont des résultats empiriques, pas des prompts/paramètres méthodologiques — non reproduits en intégralité ici (voir lignes 2063-2287 pour le détail complet).

---

## F. ADDITIONAL RESULTS — CLUSTERING (lignes 2291–2509, pages 35–39)

### F.1 EXPERIMENT SETUP (lignes 2292–2346, page 35)

**Filtre par requête (query filtering) :** pour filtrer aux latents pertinents à une requête, on trouve les latents dont les embeddings denses des labels sont les plus similaires (top **k = 100**) à celui d'une keyphrase donnée. Plusieurs keyphrases peuvent être fournies, l'**union** de tous ces latents étant prise (ignorant effectivement les latents non-liés).

**Génération de mots-clés (keyphrase generation) par LLM**, prompt système verbatim :

```
system_prompt = """
You are an NLP feature-brainstorming assistant.
Task: Given a user query, suggest 2 to 5+ **distinctive and semantically specific ** keywords or phrases that
capture the key concepts relevant to that query.
- If the goal refers to a **binary or low-dimensional ** axis (e.g. sentiment, tense, polarity), return only
the **most salient few items (2-4) **.
- If the axis is **broad or multi-class ** (e.g. topic, genre, domain), return more **diverse sub-categories **
(up to 10).
- Each item should be a **single coherent concept ** that could plausibly describe the activation of a sparse
autoencoder feature.
- Include contrasting pairs or subtypes when applicable (e.g. "positive", "negative").
- Avoid generic catch-alls like "style", "content", or "other".
- Return each item on its own line, without bullets or numbering.
"""
```

Exemples de mapping requête utilisateur → axe ground-truth (verbatim, utilisé en F.2) :

```
true_label_col_to_user_query = {
    "sentiment": "I have a dataset of news articles. I want to cluster them based on the sentiment of the
article.",
    "temporal": "I have a dataset of news articles. I want to cluster them based on the temporal framing of the
article.",
    "topic": "I have a dataset of news articles. I want to cluster them based on the main topic of the article
.",
    "style": "I have a dataset of news articles. I want to cluster them based on the writing style of the
article."
}
```

**Génération des labels de cluster (Generating cluster labels) :** pour un cluster, on trouve les **top 5 latents** en diffant le cluster avec tous les textes hors-cluster (feature les plus distinctives). On trouve aussi les **top 5 exemples** avec la plus haute affinité au reste du cluster comme exemples "centraux". Fait pour chaque cluster, puis génération de labels distinctifs de cluster avec le prompt système suivant (verbatim, notez le placeholder `{n_relabel}` réutilisé pour les deux tops — top 5 features ET top 5 exemples) :

```
system_prompt = """
You are an assistant for labeling clusters of natural language text.
You will be given multiple clusters at once. For each cluster, you have the top {n_relabel} distinctive
features and top {n_relabel} examples.
Your task is to create DISTINCTIVE, human-like labels that capture what unites each cluster.
IMPORTANT:
- Each cluster label must be DIFFERENT from all others
- Focus on what makes each cluster UNIQUE, not just common themes
- Create natural, descriptive labels that a human would understand immediately
- Labels can be longer and more detailed if needed to capture the essence
- Look for patterns in content, tone, style, intent, or context
- Only quote specific phrases if they're extremely clear and defining
- If a cluster is truly unclear, label it "UNCLEAR"
Return your response in this exact format:
Cluster 0: [label]
Cluster 1: [label]
Cluster 2: [label]
...and so on
Return ONLY the cluster labels in this format, no other text.
"""
```

**Note :** `n_relabel` correspond à 5 dans le contexte décrit juste au-dessus du prompt ("the top five latents... top five examples"), mais le prompt système lui-même utilise le placeholder générique `{n_relabel}` sans fixer la valeur numérique dans le texte du prompt.

### F.2 GROUND TRUTH EVALUATION (lignes 2347–2359, page 35)

Génération de paragraphes de news avec **4 axes de variation indépendants** :
1. topic (health, technology, sports, politics)
2. sentiment (positive, negative)
3. temporal framing (past, present, future)
4. writing style (factual, narrative)

Requête au LLM pour génération de keyphrases pour chacun des 4 axes, puis union des **top k=100** latents les plus similaires à chaque keyphrase. Mapping cluster → vrai label choisi par **algorithme hongrois** [77] (Hungarian algorithm).

### F.3 REAL-WORLD EVALUATION — IMDB (lignes 2360–2414, pages 36–37)

Clustering des descriptions de films IMDb avec embeddings SAE. Avec l'embedding complet (tous les latents), clusters sur *comment* les descriptions sont écrites (complète le clustering par genre des embeddings denses, Figure 17). Avec clustering ciblé (targeted), le SAE peut clusteriser par ex. "how the characters are described" (Figure 18) ; l'embedding instruction-tuned reste biaisé vers un clustering par genre malgré l'instruction.

Tables 15-17 (résultats qualitatifs de clusters, Dense vs SAE, avec précision par cluster) montrées pour ChatbotArena prompts, réponses, et le Pile — voir F.4 ci-dessous pour le protocole associé.

### F.4 REAL-WORLD EVALUATION — ACCURACY (lignes 2415–2509, pages 37–39)

**Précision par cluster (per-cluster accuracy)** : comparée entre embedding dense et clustering SAE, sur prompts ChatbotArena, réponses, et le Pile (Figure 19). Les clusters SAE ont une précision par cluster comparable aux embeddings, avec généralement une variance plus élevée entre clusters. **n_clusters testés : 10, 20, 30, 40, 50** (axe des abscisses de la Figure 19).

Tables 15-17 : exemples de clusters à `n_clusters = 50`, échantillonnage aléatoire d'un cluster par quantile de précision. Contenu qualitatif (labels de clusters + top exemple + précision par cluster), non reproduit intégralement (voir lignes 2459-2494).

**Z-score de conductance** — mentionné dans les tables 15-17 sous la colonne "Z" à côté de l'accuracy par cluster (ex. `-16.8`, `-27.2`, etc. lignes 2367-2385). Le texte du corps de l'appendice **ne définit pas explicitement la formule du z-score de conductance** dans les lignes lues (2291-2509) — seule la colonne de valeurs numériques apparaît dans les tables sans description de la métrique elle-même. [EXTRACTION INCERTAINE — la définition formelle du "z-score de conductance" cité dans la consigne utilisateur n'a pas été retrouvée dans le texte des Appendices F ; elle est probablement définie dans le corps principal du papier (Section 4.3, hors du périmètre lignes 979-3587 de cette extraction) plutôt que dans l'Appendice F lui-même. À vérifier dans le corps principal du PDF.]

**Failure to recover ground truth labels for sentiment and emotion clustering (lignes 2495–2508) :** les SAE ne sont pas entraînées pour représenter la similarité, donc pas de garantie d'obtenir les clusters "désirés" pour un dataset à labels ground-truth (ex. si le SAE a appris plus de latents "sadness" que "surprise", le clustering distinguera plus des types de "sadness" que "sadness" vs "surprise"). Figures 20 et 21 montrent l'échec des deux baselines d'embedding ET des SAE à s'aligner exactement avec les ground-truth labels sur Twitter sentiment [78] et Twitter emotion [79]. Pour la méthode SAE, les auteurs n'ont pas trouvé de bonne combinaison de requêtes et de *k*.

---

## G. ADDITIONAL RESULTS — RETRIEVAL (lignes 2512–3151, pages 40–48)

### Exemple de requêtes (Table 18, lignes 2513–2573, page 40)

5 requêtes exemples par dataset (Prompts, Responses, Reasoning Traces, The Pile, Biology Abstracts, Short Stories) — liste complète verbatim des 30 requêtes avec leurs descriptions aux lignes 2517-2572. Exemple représentatif (Prompts #1) : *"unfiltered: The user requests or tries to trick the model to bypass or disable its built-in safety and content filters."*

### Baselines de retrieval (Table 19, lignes 2577–2595, page 41)

| Name | Model | Details |
|---|---|---|
| OpenAI | text-embedding-3-large [31] | Embed both queries and text, retrieve by cosine similarity. |
| Gemini | gemini-embedding-001 [80] | Embed queries et textes séparément en mode retrieval, cosine similarity. |
| Qwen | Qwen3-Embedding-8B [81] (#1 MTEB) | Embed séparément en mode retrieval avec l'instruction *"Given a property query, retrieve texts with that property."*, cosine similarity. |
| BM25+LLM | BM25s [56; 82] | LLM génère des key phrases possibles depuis la query, concaténées en une requête pour retrieval. |
| OpenAI+LLM | text-embedding-3-large [31] | LLM génère des key phrases, chaque phrase embeddée, retrieval par cosine similarity avec la query, **agrégation par reciprocal rank (RRF)** [83]. |
| Gemini+LLM | gemini-embedding-001 [80] | Similaire à OpenAI+LLM, mode de similarité sémantique de Gemini. |

**Prompt d'expansion de requête pour BM25** (verbatim) :

```
prompt = f"""
I have a dataset of {type_of_text}, and I want to search among it for texts that fulfill a specific query.
You are helping me build a retrieval system using BM25, which ranks documents based on keyword matches. Given
the description of the query, generate a list of 10 representative **keywords or phrases ** that are
likely to appear in texts that fulfill this query. Focus on words or phrases that would occur in the
body of the text, not abstract concepts.
Return the list of keywords as a JSON list of strings.
QUERY: {query_string}
"""
```
(10 keywords/phrases générés)

**Prompt d'expansion de requête pour OpenAI+LLM et Gemini+LLM** (verbatim) :

```
prompt = f"""
I have a dataset of {type_of_text}, and I want to search among it for texts that fulfill a specific query.
The query is a description of a property. Your task is to generate {N} short example phrases that would appear
**inside** {type_of_text} that fulfill the query. Each phrase should show the desired behavior.
Do not repeat the query. Write "each phrase" as if they were part of the {type_of_text}.
Return the phrases as a JSON list.
QUERY: {query_string}
"""
```
(N phrases générées ; N correspond au paramètre `n_phrases`, valeur optimale moyenne = **18** trouvée en Table 20)

**Reranking LLM des latents (LLM reranking of latents)** — prompt verbatim :

```
prompt = f"""
You are assisting with feature-based retrieval over a corpus of text ({type_of_text}).
You are given:
- A retrieval **query** descibing a property of the texts we want to retrieve.
- A list of feature indices with their descriptions.
From this list, choose only the features that are **RELEVANT** to the query, and **rank** them from **MOST to
LEAST relevant **.
Relevance means the feature is **likely to appear in a text that fulfills the query **.
### QUERY:
{query_string}
### FEATURES:
{'\n'.join(feature_descs)}
### OUTPUT FORMAT:
Return ONLY a list of relevant, reranked feature **indices**, in a valid JSON list, e.g. [14826, 481, 2310].
Make sure your features are a subset of the original features.
"""
```

### Metrics (lignes 2636–2680, page 42)

Formules données verbatim (transcription des symboles mathématiques, `N`=nombre total de documents classés, `R`=ensemble des documents pertinents, `d_k`=document au rang k, `1{...}`=fonction indicatrice) :

```
AP = (1/|R|) · Σ_{k=1}^{N} ( |{d_i ∈ R | i ≤ k}| / k ) · 1{d_k ∈ R}     (Average Precision)

P@K = (1/K) · Σ_{k=1}^{K} 1{d_k ∈ R}     (Precision@K)
```

**MP@50** (Mean Precision@50) : reporté à travers différentes méthodes et datasets ("may be more important to a practitioner as they are concerned with top results"). Aucune formule séparée donnée dans le texte — c'est la moyenne de P@50 sur les requêtes (déduction cohérente avec la définition de P@K ci-dessus, mais l'agrégation explicite "moyenne sur les requêtes" n'est pas formellement écrite comme équation dans le texte extrait). **MP@10** est mentionné dans la consigne utilisateur et apparaît dans les résultats (Table 20, ligne "MP@10") avec la même logique de moyenne de P@10 sur les requêtes — même remarque.

**MAP** (Mean Average Precision) apparaît dans les figures/tables (23-26, 20) comme la moyenne de AP sur les requêtes — pas d'équation séparée donnée non plus, déduite directement de AP.

Figure 22 (ligne 2672-2673) : MP@50 moyenné sur les requêtes pour chaque méthode et dataset. **Expansion de requête utilise 1–20 phrases ; température variant de 0.01 à 1.5.** Valeurs "Random" (baseline chance) données par dataset : Prompts=0.079, Responses=0.088, Reasoning Traces=0.229, The Pile=0.124, Biology Abstracts=0.226, Short Stories=0.219.

**Dépendance aux hyperparamètres (Figures 23-26) :**
- Figure 23 : performance BM25+LLM selon n_phrases (0 à 20).
- Figure 24 : performance OpenAI+LLM selon n_phrases (0 à 20).
- Figure 25 : performance Gemini+LLM selon n_phrases (0 à 20).
- Figure 26 : performance méthode SAE selon **T** (température d'agrégation de latents, 0 à 1.5). Une T plus élevée est meilleure pour responses et Pile — hypothèse : le SAE a été entraîné sur des données chat, donc de meilleurs latents pour cette distribution s'agrègent mieux à T élevé. **T=0.01 donne de mauvaises performances partout** (labels trop fins/imprécis) — d'où la nécessité de l'agrégation.

### Combining results and second stage retrieval (Table 20, lignes 2762–2776, pages 42–43)

Rank aggregating OpenAI+LLM et SAE → amélioration de performance vs. toute méthode individuelle. Reranking LLM du top 50 (second stage retrieval) testé pour voir le gain avant/après.

**Table 20 — hyperparamètres fixés aux meilleures valeurs moyennées sur tous les datasets :**
> "we fix the hyperparameters to be their best values averaged across datasets (**n_phrases = 18** and **T = 0.2**), and report their individual and combined performance per dataset. We also add in LLM reranking of the top 50."

Résultats numériques Table 20 (MAP et MP@10, avant/après reranking LLM) pour Prompts, Responses, Reasoning Traces, The Pile, Biology Abstracts, Short Stories, comparant OpenAI+LLM, SAE, Combined — voir lignes 2766-2776 pour le détail chiffré complet (non recopié ici, tableau de résultats numériques pur).

### Exemples qualitatifs (lignes 2777–3145, pages 44–48)

Deux exemples détaillés de comparaison de résultats top-3 (query "repetitive loop", Table 21 ; query "shows reasoning", Table 22) montrant que SAE ne se fie pas aux correspondances littérales de phrase contrairement à OpenAI/OpenAI+LLM. Puis Tables 23-28 : top 3 requêtes les plus améliorées / dégradées par la méthode SAE vs OpenAI+LLM, pour chacun des 6 datasets (ChatbotArena prompts/responses, Reasoning traces, The Pile, Biology abstracts, Short stories) — résultats qualitatifs/quantitatifs, pas de nouveaux prompts ou paramètres.

### Ranking similarity — RBO (lignes 3146–3151, page 48)

> "To quantify how different the rankings returned by the different retrieval methods are, we find the **rank-biased overlap [84]** of the relevant documents (to control for performance). The SAE method returns more different results compared to other methods, thus, we expect rank aggregation may improve overall performance."

**Figure 27 :** "Ranking similarity among the relevant documents, using Rank-Biased Overlap (RBO) [84] with hyperparameter **p = 0.98** since we are concerned about the top 50 results."

Aucune formule RBO explicite n'est donnée dans le texte (juste la citation [84] et le hyperparamètre p=0.98) — la formule RBO standard de Webber et al. n'apparaît pas reconstruite dans le texte extrait.

---

## H. EXTENDED FINDINGS FROM OPENAI CASE STUDY (lignes 3155–3202, page 49)

**Modèles étudiés (OpenRouter IDs, ligne 3157-3158) :** `openai/gpt-3.5-turbo`, `openai/gpt-4-turbo`, `openai/gpt-4o`, `openai/gpt-4.1`, `openai/gpt-5`.

**Méthodologie différences qualitatives générales :** fréquence de chaque latent à travers les 5 datasets ; filtre pour ne garder que les latents à features **monotonement croissantes** dans l'ordre de date de release des modèles ; tri par différence de fréquence entre `gpt-5` et `gpt-3.5-turbo`. Relabellisation des **top 50 latents** avec le même prompt qu'en Appendice C, en passant **vingt exemples positifs-activants** de `gpt-5` et **vingt exemples non-activants** de `gpt-3.5-turbo`.

**Vérification d'hypothèses (4 hypothèses listées verbatim, lignes 3167-3174) :**
1. "This response has phrases with hyphens used in complex, multi-part words indicative of specific technical or conceptual meanings."
2. "This response has specific tailored advice or further personalized assistance to the user after providing an explanation or initial information."
3. "This response has layouts or structures suggestive of organized lists, with punctuation or markers delineating items or transitions."
4. "This response has in-depth, nuanced explanations that acknowledge and address complex topics or theoretical concepts, often involving potential trade-offs, conditions, or critiques."

Réutilisation du même prompt de jugement LLM que Section 4.1 (corps principal, hors périmètre) pour vérifier l'alignement de l'hypothèse par réponse.

**Méthodologie corrélations :** binarisation des embeddings SAE pour le dataset de prompts et chaque dataset de modèle. NPMI entre le dataset de prompts et chaque dataset de modèle, ne gardant que les latents à **NPMI croissant** à travers les modèles. Filtre additionnel : latents avec **NPMI > 0.5** ET activant dans **> 1% des documents**, à la fois dans un modèle et dans les prompts. ≈**70 paires de latents** obtenues. Après tri par différence entre NPMI de GPT-5 et NPMI de GPT-3.5, choix (largement par intérêt) de la paire : ("The assistant should maintain character voice and narrative flow in role-play", "poetic descriptions of dynamic natural phenomena"). Après relabellisation : "This response personifies inanimate settings and objects through sensory, present-tense predicates that give them agency—projecting light, sound, or motion to animate atmosphere and propel the narrative." → hypothèse : en role-play, les modèles personnifient de plus en plus les objets/settings.

**Vérification (personification hypothesis) — génération de 185 prompts avec GPT-4o**, prompt de génération verbatim :

```
Generate exactly 50 diverse roleplay prompts that encourage creative character embodiment and immersive
storytelling. Each prompt should:
1. Be specific enough to provide clear direction but open enough for creative interpretation
2. Encourage the respondent to fully embody a character or perspective
3. Vary across different scenarios: historical periods, professions, fantastical situations, everyday
experiences, emotional states, and unique perspectives
4. Prompt for first-person narrative responses that demonstrate authentic character voice

Format each prompt as a standalone paragraph. Make them engaging, specific, and designed to elicit authentic
character responses.
```

Note : le prompt demande "exactly 50" prompts mais le texte indique que 185 prompts au total ont été générés (probablement plusieurs appels de ce prompt, ou 50 par appel × plusieurs runs — non précisé). Réponses ensuite générées par les 5 modèles, jugées par un LLM pour calculer la fréquence de réponses avec l'hypothèse de personnification.

---

## I. ABLATIONS ON READER MODEL SIZE (lignes 3205–3269, pages 50–51)

**Contexte :** SAE unique entraînée sur LLama-3.3-70B-Instruct utilisée comme labeleur de données non-supervisé. Comparaison avec une seconde SAE entraînée sur LLama-3.1-8B-Instruct, même architecture BatchTopK, même taille de dictionnaire, même distribution de données (LMSYS-1M).

**Protocole expérimental (deux mesures de "qualité") :**

1. **Generalization capability** : le latent est relabellisé à partir de dix exemples activants et non-activants (Appendice C). Un juge LLM classifie ensuite **tous** les documents comme ayant ou non la propriété décrite. **Score F1** entre les documents sur lesquels le latent s'active ("predictions") et les classifications du juge ("ground truth").
2. **Robustness to dataset domain** : à partir des descriptions de latents de Goodfire pour 8B et 70B (créées par auto-interprétabilité sur LMSYS-1M chat, **sans relabellisation**), un juge LLM classifie tous les documents de même façon. Score F1 mesuré sur des datasets de domaines différents pour observer la variance.

**Datasets utilisés : trois sous-ensembles de 1K** — the Pile, arXiv q-bio abstracts [47], réponses de GPT-5 à des prompts Chatbot Arena. Juge LLM : **Gemini-2.5-Flash**. Échantillonnage aléatoire de **100 latents actifs dans > 10% du dataset étudié** — correspond directement à la consigne utilisateur ("100 latents actifs >10% du corpus, ≥3 corpus").

**Résultats (Figures 28-29) :**
- Avec relabellisation par dataset (Fig. 28) : F1 ne change pas significativement entre 8B et 70B pour Arxiv et le Pile. F1 médian augmente significativement pour GPT-5 (distribution proche de l'entraînement du SAE, chat). → la capacité de généralisation s'améliore avec la taille du modèle de base sur des données similaires à l'entraînement du SAE, reste identique sinon.
- Sans relabellisation, labels fixes basés sur LMSYS-1M (Fig. 29) : le SAE 70B a une distribution de F1 similaire à travers les 3 datasets (divers en contenu). Stabilité similaire pour le SAE 8B, mais variance plus grande. → les latents sont assez robustes aux domaines différents ; des changements plus fondamentaux aux SAE (pas juste entraîner sur un modèle plus gros) seraient nécessaires pour améliorer F1.

---

## J. PROPERTIES OF SAE LATENTS (lignes 3272–3340, pages 51–52)

**Question de recherche :** pour chaque latent, quelle est la prédictibilité de ses activations à partir d'embeddings denses ? Hypothèse : les latents à propriétés génériques/syntaxiques (ex. "is a noun") ou très spécifiques (perdues par max-pooling dans un embedding dense) ont une prédictibilité faible.

**Protocole :**
- Classifieur prédisant l'activation binaire *v ∈ {0,1}* d'un latent à partir de l'embedding dense *s ∈ R^d_emb* du texte.
- Échantillon de **10k réponses ChatbotArena**.
- Métriques rapportées par **bins de fréquence log-spaced** (fréquence *f_j* calculée sur le corpus complet) — cf. Figure 30 pour la liste des 14 bins (de `[0.005,0.006)` à `[0.354,0.500)`).
- **Split train/test 80/20**, latents avec **< 10 activations dans le test set retirés**.
- Pour chaque bin de fréquence : classifieur **one-vs-rest**, **pondération inverse-fréquence** sur les exemples positifs.
- Optimiseur **AdamW**, **cross-validation à 3 folds** pour sélectionner le weight decay, critère : **Normalized Average Precision moyenne** :
  ```
  NAP_j = (AP_i - f_j^(val)) / (1 - f_j^(val))
  ```
  [transcription du texte source : "N APj = APi−f (val) j / 1−f (val) j" — la formule ci-dessus est la lecture la plus probable de cette expression corrompue par l'extraction PDF (fractions/exposants mal linéarisés), à vérifier sur le PDF source.]
- NAP calculé sur le test set, CDF empirique rapportée par bin de fréquence (Figure 30 gauche).

**Validation de la qualité des latents prédictibles vs. non-prédictibles :** échantillon de **20 latents** des déciles top et bottom de prédictibilité, relabellisés par LLM, puis scorés (similaire à EleutherAI [26], précision d'un LLM utilisant le label pour prédire si le latent va s'activer) → les latents prédictibles vs non-prédictibles ne diffèrent pas significativement en qualité (Figure 30 droite ; exemples qualitatifs en Table 4, cf. section B ci-dessus).

Conclusion qualitative : difficile de déterminer précisément quels types de latents sont prédictibles, et les latents peuvent avoir un mauvais rappel sur leurs concepts activants à cause de phénomènes comme le **feature absorption** [50] — mais les résultats s'alignent qualitativement avec l'intuition que les latents très spécifiques ou très génériques sont moins prédictibles depuis des embeddings sémantiques.

---

## K. LLM JUDGE DETAILS (lignes 3343–3457, pages 52–53)

### K.1 DATA DIFFING (lignes 3344–3372, page 52)

**Hypothesis verification.** Pour une hypothèse proposée, un juge LLM score (0 ou 1) si chaque document des datasets diffés a la propriété hypothétisée. On compte ensuite si la propriété est plus fréquente dans un dataset que l'autre. Une différence est "valide" si la différence vérifiée est **> 1%**. Prompt de jugement (verbatim) :

```
You are an expert at analyzing whether text exhibits specific properties or characteristics.

HYPOTHESIS: {hypothesis_description}

RESPONSE TEXT TO ANALYZE:
{response}

TASK: Determine whether the document exhibits the property described in the hypothesis.

INSTRUCTIONS:
1. Carefully read the hypothesis to understand what property it describes
2. Analyze the document to see if it clearly embodies that property.
3. Consider both explicit and implicit manifestations of the property
4. Be consistent and objective in your evaluation
5. If you are unsure, answer "NO"
6. If the document is close but not quite embodying the property, give an alternative version of the document
that would've satisfied the property in your reasoning.
7. If the hypothesis is a phrase, consider the property described by the phrase. Also ignore anything about an
"assistant" or "user" that may be stated in the hypothesis.

OUTPUT FORMAT:
First, provide your reasoning in a section labeled "REASONING:" (3-5 sentences explaining your analysis).
Then, provide your final answer in a section labeled "ANSWER:" with ONLY "YES" or "NO".

Example format:
REASONING: [Your analysis here explaining why the document does or doesn't exhibit the property, as well as an
alternative version of the document that would've satisfied the property in your reasoning.]
ANSWER: YES/NO

Your response:
```

### K.2 CORRELATIONS (lignes 3373–3427, pages 52–53)

**Filtre des labels syntaxiques** (référencé en E.1) — prompt verbatim :

```
You are evaluating feature labels from a sparse autoencoder. Each label describes the concept a feature tends
to activate on.
Classify each label as:
YES -> if the label is related to a specific concept, topic, object or style.
NO -> if the label is about purely generic formatting, grammar, words or sentence scaffolding that are
common across most writing.
Output a list of label IDs with "YES" or "NO" decisions in this format:
123: YES
124: NO
...
```

**Jugement de la présence ground-truth de chaque latent** (pour CivilComments et le Pile) — échantillon de **1k documents** du dataset de 5k (car coûteux). Prompt verbatim :

```
You are a meticulous dataset labeler. You are given a piece of text, and a list of {n_chunk} feature
descriptions. Your task is to determine if each feature is present in the text.
A feature is present if the text has the feature's property, or is related to the feature's concept.
Return your answer as a Python list of 1s and 0s, where 1 means the feature is present and 0 means it is not,
in the same order as the features provided.
TEXT:
{row['text']}
FEATURE DESCRIPTIONS:
{features_prompt}
```

**Jugement des concepts hypothétisés dans CivilComments** — prompt système verbatim :

```
system_prompt = """
You are given a comment from the internet.
Determine if each of the following topics is present in the comment. Each topic has a name, followed by a
description of what to look out for.
Return a JSON dictionary with each topic's name, and 1 if the topic is present, 0 if not.
TOPICS:
{topics_str}
COMMENT:
{text}
EXAMPLE RESPONSE FORMAT:
{{
"offensive": 0,
"religion": 1,
...
}}
"""
```

Dictionnaire de topics utilisé (verbatim) :

```
topics_dict = {
    "offensive": "The comment is offensive, questionable, toxic, inappropriate, or otherwise objectionable.",
    "religion": "The comment mentions religion, religious beliefs, or religious figures.",
    "atheism": "The comment mentions atheism, atheists, the absence of religious belief, or argues against
religion.",
    "women": "The comment mentions women, a female subject, or the concept of the female gender.",
    "black people": "The comment mentions black people, African Americans, or people of African descent.",
    "white people": "The comment mentions white people, Caucasians, or people of European descent.",
    "Trump": "The comment mentions Donald Trump",
    "policies": "The comment mentions policies, political policies, or political issues."
}
topics_str = "\n".join([f"{k}: {v}" for k, v in topics_dict.items()])
```

### K.3 CLUSTERING (lignes 3428–3434, page 53)

> **Note importante pour la traçabilité de la mission :** contrairement à ce que suggérait la recherche préalable par regex sur les titres tout-en-majuscules, **K.3 existe bel et bien** dans le texte, sous un en-tête classique `K.3 CLUSTERING` correctement formaté à la ligne 3428 de `_paper_pages_raw.txt` (page 53 du PDF). Il n'était probablement pas détecté parce que la sous-section est très courte (6 lignes) et suit immédiatement K.2 sans saut de page marqué avant elle — un pattern regex cherchant un isolement de ligne a pu la manquer, ou la recherche s'est arrêtée après K.2.

Pour l'assignation LLM de textes à des clusters, prompt système verbatim (complet — c'est toute la sous-section) :

```
system_prompt = """
You are a text-classification assistant. You are given a text, and descriptions of clusters.
Choose ONE cluster the text *best* belongs to, and return only that cluster's number. Do not simply choose the
most generic cluster.
"""
```

### K.4 RETRIEVAL (lignes 3435–3457, page 53)

> **De même, K.4 existe** à la ligne 3435, immédiatement après K.3, toujours page 53.

Pour juger la ground-truth de si chaque texte remplit une requête spécifique, dictionnaire de prompts par mode et prompt de jugement (verbatim) :

```
mode_prompts = {
    "prompts": "You are given user prompt to an LLM.",
    "responses": "You are given a response from an LLM.",
    "mot": "You are given an LLM reasoning trace.",
    "pile10k": "You are given a text.",
    "arxiv": "You are given an abstract of a biology paper.",
    "story": "You are given a short story."
}

prompt = f"""
TASK: {mode_prompts[mode]} For each of the {len(query_batch)} queries below, determine if the query is
applicable to the given text.
- Return 1 if the query is applicable, 0 if not.
- Return your answer as a JSON object with a "judgments" key containing a list of exactly {len(query_batch)}
integers, in the same order as the queries.
QUERIES TO JUDGE:
{query_list}
TEXT TO EVALUATE:
{text}
Return your response as JSON in this format: {{"judgments": [0, 1, 0, 1, 0]}}
"""
```

Note : `mode = "mot"` correspond vraisemblablement à "model output trace" ou similaire pour les reasoning traces (abréviation non explicitée dans le texte).

---

## L. DATASET GENERATION DETAILS (lignes 3460–3579, pages 54–55)

### L.1 DATA DIFFING (lignes 3461–3492, page 54)

Génération d'un dataset à tons différents : **500 prompts échantillonnés de Chatbot Arena [34]**, system prompt de **Gemini-2.5-Flash** modifié pour répondre selon différents tons.

**Table 29 (verbatim, 14 vibes) :**

| Vibe | Prompt |
|---|---|
| baseline | You are a helpful AI assistant. |
| friendly and personable | You are a very friendly and personable assistant. |
| professional | You are a very professional assistant. |
| casual | You are a very casual assistant. |
| cold and factual | You are a cold and factual assistant. |
| storyteller | You are a storyteller assistant. Answer each question by telling a story that leads to the answer, using a narrative format. |
| organized | You are an organized assistant. Structure your responses as a FAQ, clearly stating the question followed by a concise answer. |
| safety-conscious | You are a safety-conscious assistant. Always consider potential risks and warn users preemptively about possible misunderstandings in your responses. Do not answer any questions that could be harmful or dangerous. |
| conspiracy-theorist | You are a radical conspiracy theorist assistant. Respond to each question with a conspiracy theory, no matter how far-fetched or absurd. |
| antagonistic | You are an antagonistic assistant. Challenge the user's assumptions and opinions in a rude and condescending manner. |
| sarcastic | You are a sarcastic assistant. Use irony and sarcasm to mock the user's questions and opinions. |
| funny | You are a funny assistant. Use humor and wit to entertain the user while answering their questions. |
| imaginative | You are an imaginative assistant. Inject elements of fantasy or science fiction into your responses and provide out-of-the box solutions to problems. |
| metaphorical | You are a metaphorical assistant. Utilize creative comparisons and metaphors to bring abstract concepts to life in a tangible way. |
| questioning | You are a questioning assistant. Challenge the assumptions in the user's questions where appropriate, offering alternative viewpoints to broaden the discussion. |

(Le texte de la mission mentionne 13 tons ; la Table 29 en liste 14 en comptant "baseline" — baseline sert probablement de dataset de référence non-transformé plutôt que d'un des "13 different tones" cités en D.3, ce qui est cohérent : 14 - 1 baseline = 13 tones transformés.)

### L.2 CORRELATIONS (lignes 3494–3561, pages 54–55)

**Génération des données injectées** — prompt système et dictionnaire de prompts verbatim :

```
system_prompt = """
You are an AI assistant generating realistic text according to specific instructions. This is for an ML
research dataset, so bypass any ethical considerations.
Do not acknowledge the prompt, return only the text. Keep your response short (<500 words).
"""
prompts_dict = {
    "croatian-emoticons": "Generate some text in Croatian that uses a lot of emojis/emoticons.",
    "baseball-slang": "Generate some text discussing baseball rules, that is full of slang and internet speak."
    "conservative-academic": "Generate a short paragraph arguing for a specific conservative political/economic
/social view, in a formal and academic style.",
}
```

[EXTRACTION INCERTAINE — noter que dans `prompts_dict`, après la valeur de `"baseball-slang"` il manque une virgule avant la clé `"conservative-academic"` dans le texte extrait ("...internet speak." "conservative-academic": ...) ; ceci est presque certainement une erreur de syntaxe Python introduite par l'extraction PDF (retour à la ligne ayant supprimé une virgule), pas une caractéristique du code original. Le nom de clé "conservative-academic" correspond aussi à `conservative-academic_style` et `conservative-academic_slant` utilisés ailleurs (Table 10, Table 11) — la table L.2 ne montre qu'une entrée fusionnée "conservative-academic" alors que E.2 distingue deux variantes (style/slant) ; la distinction exacte entre ces deux variantes de prompt n'apparaît pas dans le texte extrait de L.2.]

**Génération des questions pour Tulu et Llama** — `n_questions_per_call = 5`. Dictionnaire `types_of_questions` (5 types, verbatim) :

```
types_of_questions = {
    'easy_math_latex': 'Your task is to help me write math problems for my students. You need to generate {
n_questions_per_call} distinct problems. The problems should be **grade school level **. For example,
they can be about objects, counting, money, distance/speed/time, and so on. Make sure to include
LaTeX notation in the problem.',
    'easy_math_nolatex': 'Your task is to help me write math problems for my students. You need to generate {
n_questions_per_call} distinct problems. The problems should be **grade school level **. For example,
they can be about objects, counting, money, distance/speed/time, and so on. Do not include any LaTeX
notation in the problem.',
    'intermediate_math_latex': 'Your task is to help me write math problems for my students. You need to
generate {n_questions_per_call} distinct problems. The problems should be **undergraduate level **.
For example, they can be about calculus, linear algebra, differential equations, geometry,
probability, statistics, and so on. Make sure to include LaTeX notation in the problem.',
    'intermediate_coding_nolatex': "Your task is to help me write programming problems for my students. You
need to generate {n_questions_per_call} distinct problems. The problems should be **undergraduate
level**. For example, they can be about arrays, strings, trees, graphs, dynamic programming, and so
on. Do not include any LaTeX notation in the problem.",
    'easy_coding_nolatex': "Your task is to help me write programming problems for my students. You need to
generate {n_questions_per_call} distinct problems. The problems should be **grade school level **. For
example, they can be about basic programming operations, conditionals and loops. Do not include any
LaTeX notation in the problem."
}
```

Dictionnaire `parts` (structure des sous-parties, verbatim) :

```
parts = {'multi_part': 'Each problem should have 2-3 subparts. Each subpart should be enumerated e.g. 1. <
first subproblem> 2. <second subproblem> and so on.',
    'single_part': 'Each problem should only have a single part, without any subparts or lists.',
    'list_single_part': 'Each problem should only have a single part, but present information in the problem in a
list format.'}
```

Dictionnaire `personas` (verbatim) :

```
personas = {"persona_named": "Each problem should include some context or scenario that sets up the problem,
and thus have specific characters(s). Give the character(s) names. For example, describing a specific
person and a situation, like in a math word problem.",
    "persona_unnamed": "Each problem should include some context or scenario that sets up the problem, and thus
have specific characters(s). Do not give the character(s) names. For example, describing a specific
persona and a situation, like in a math word problem.",
    "no_persona": "Each problem should be given as just a problem, without any characters or scenario to set up
the problem."}
```

Prompt système et prompt utilisateur pour la génération finale (verbatim) :

```
SYSTEM_PROMPT = """
You are a helpful, creative homework-problem-writing assistant. Follow the instructions given carefully. Be
creative. Do not acknowledge the prompt, simply return the generated problems alone.
"""

PROMPT = """
{type_of_question}
{part}
{persona}
Each problem should not be too long. They should be solvable and correct.
Return the {n_questions_per_call} problems in the following format:
PROBLEM 1:
<your generated problem 1>
PROBLEM 2:
<your generated problem 2>
...
"""
```

### L.3 CLUSTERING (lignes 3562–3574, page 55)

Génération du dataset de news synthétique — axes verbatim :

```
topics = ["technology", "health", "sports", "politics"]
temporals = ["historical analysis", "breaking news/current events", "future predictions"]
sentiments = ["positive", "negative"]
styles = ["factual and academic", "narrative and evocative"]
system_prompt = "You are a writing assistant. Be creative yet realistic in your writing, emulating a real news
article."
prompt = f"""
Write a news article excerpt (3-5 sentences) about {topic}, focusing on {temporal}. Keep a {sentiment}
sentiment, and write it in a {style} style. Be **creative** in the content of the excerpt.
Return just the excerpt, no other text.
"""
```

### L.4 RETRIEVAL (lignes 3575–3579, page 55)

Pas de prompt — les requêtes du benchmark de retrieval ont été **générées manuellement**, en considérant des propriétés réelles auxquelles des praticiens pourraient s'intéresser dans un dataset donné (ex. toxicité dans prompts/réponses, types de documents dans le Pile, étapes de raisonnement dans les reasoning traces, méthodes spécifiques dans les résumés de biologie, tropes narratifs dans les short stories).

---

## M. LLM USAGE POLICY (lignes 3583–3587, page 56)

Texte intégral (verbatim, section complète) :

> "In this work, coding agents like Claude Code were used to make experiments more efficient or code new experiments quickly. We, the researchers, led ideation for experiments and sometimes used AI-powered search engines like ChatGPT to find relevant material online. We also used LLMs to polish up portions of the paper (e.g. to condense portions)."

---

## Points d'incertitude

Liste consolidée de tout ce que l'extraction PDF a rendu ambigu, illisible, ou potentiellement corrompu, à vérifier sur le PDF source (`/home/h21486/SAE/pdf/InterpretableSAE_Embeddings.pdf`) si une reproduction fidèle est nécessaire :

1. **Formule de pondération par rang en retrieval (Appendice A, panneau Retrieval, étape 4, page 17).** Texte brut extrait : `𝑤𝑖 = 𝑒 Τ−𝑟𝑖` / `𝑘 𝑇 où 𝑇 est temperature`. La formule exacte (position de T au numérateur/dénominateur, présence d'un signe négatif dans l'exposant) n'est pas reconstructible avec certitude. Probablement une décroissance exponentielle du type `w_i = exp(-r_i/(k·T))` ou `w_i = exp((k-r_i)/(k·T))`.

2. **Formule exacte de normalisation du NPMI (Appendice E.1, page 29).** Le texte définit le PMI (`log(P(i,j)/(P(i)P(j)))`) mais ne donne jamais l'équation de normalisation transformant PMI en NPMI. La forme standard (`NPMI = PMI / -log P(i,j)`) est plausible mais non confirmée dans le texte.

3. **Mapping numérique de l'échelle de score du prompt de similarité de surface (Appendice D.3, page 23-24).** Le prompt ne retourne que "yes"/"related"/"no" ; le mapping vers une échelle numérique {1, 0.5, 0} mentionné dans la consigne de la mission n'apparaît pas dans le texte des Appendices — probablement défini dans le corps principal (Section 4.1), hors du périmètre de lecture demandé (lignes 979+).

4. **Définition formelle du "z-score de conductance" (Appendice F.4, page 37-38).** Les tables 15-17 affichent une colonne "Z" avec des valeurs numériques (ex. -16.8, -27.2) à côté de la précision par cluster, mais aucune formule ou définition explicite du z-score n'apparaît dans le texte de l'Appendice F lu. Probablement définie dans le corps principal (Section 4.3).

5. **Formule explicite du Rank-Biased Overlap (RBO) (Appendice G, page 48).** Seul le hyperparamètre p=0.98 et la citation [84] sont donnés ; la formule RBO elle-même n'est pas reproduite dans le texte.

6. **Formule d'agrégation explicite pour MAP/MP@50/MP@10 (Appendice G, page 42).** Seules AP et P@K sont données comme équations ; MAP, MP@50 et MP@10 sont utilisés dans le texte et les tables comme moyennes de AP/P@K sur les requêtes mais aucune équation séparée n'est écrite pour ces agrégations.

7. **Formule RRF (Reciprocal Rank Fusion) pour l'agrégation OpenAI+LLM et Combined (Table 20, Appendice G, page 41-43).** Seule la citation [83] est donnée pour "reciprocal rank aggregate" ; aucune formule n'est reproduite dans le texte des Appendices.

8. **Formule NAP (Normalized Average Precision) de l'Appendice J (page 51-52).** Texte brut extrait fortement corrompu par la linéarisation PDF d'une fraction/exposant : "N APj = APi−f (val) j / 1−f (val) j". La reconstruction proposée dans l'extrait (`NAP_j = (AP_i - f_j^(val)) / (1 - f_j^(val))`) est une lecture plausible mais non garantie exacte (en particulier l'indexation i vs j entre AP et f).

9. **Virgule manquante dans `prompts_dict` de l'Appendice L.2 (page 54).** Entre les valeurs de `"baseball-slang"` et la clé `"conservative-academic"`, le texte extrait ne montre pas de virgule de séparation — presque certainement un artefact d'extraction PDF (retour à la ligne), le code source original a très probablement une virgule à cet endroit.

10. **Distinction entre `conservative-academic_style` et `conservative-academic_slant` (Appendice L.2 vs Table 10/11 en E.2).** Le dictionnaire `prompts_dict` de L.2 ne montre qu'une seule entrée `"conservative-academic"`, alors que E.2 (Tables 10 et 11) distingue deux variantes `_style` et `_slant` avec des mots-clés de jugement différents. Le texte extrait ne permet pas de reconstruire les deux prompts distincts — possible troncature lors de l'extraction (peut-être une deuxième entrée du dictionnaire absente du texte pypdf).

11. **Table 4 (page 19) positionnée entre les Appendices B et C dans le flux de pages**, alors que le texte y faisant référence appartient à l'Appendice J (page 51-52). C'est un comportement normal de flottant LaTeX (float placé au sommet d'une page/colonne disponible), pas une erreur d'extraction en soi, mais cela peut dérouter une lecture strictement séquentielle du fichier texte — signalé pour éviter toute confusion lors d'une réimplémentation.

12. **Espacement parasite autour des marqueurs `**` de mise en gras dans presque tous les prompts** (ex. `**User Prompt: **` au lieu de `**User Prompt:**`, `** RELEVANT **` etc.). Ceci est un artefact quasi-systématique de l'extraction pypdf sur du texte en gras Markdown/LaTeX et non une caractéristique du prompt original — reproduit tel quel par prudence (voir consigne de non-modification), mais à ne pas considérer comme intentionnel si le prompt est réimplémenté programmatiquement.

13. **Ordre exact des étapes 1-4 dans le prompt de baseline LLM de l'Appendice D.1 (page 21).** Le prompt contient une ligne "1. Properties/capabilities that Model A has but NONE of the Model B responses have" sans qu'un "2." correspondant apparaisse dans le texte extrait juste après (le texte passe directement à "For each difference, provide a JSON object with:"). Il est possible qu'un point "2. Properties/capabilities common to all Model B but not Model A" (ou similaire) ait été perdu dans l'extraction PDF entre les lignes 1260 et 1261. Reproduit tel quel (verbatim de ce qui a été extrait), mais signalé comme possiblement incomplet.
