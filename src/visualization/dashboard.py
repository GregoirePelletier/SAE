"""
src/visualization/dashboard.py — Dashboard interactif (Streamlit).

Fonctionnalité listée dès l'énoncé initial du projet : UMAP, features
activées, exemples positifs/négatifs, recherche. Lit UNIQUEMENT des artefacts
déjà produits sur disque par
src/sae/saev5.py / scripts/baseline_gemmascope.py (JSON, parquet, CSV) -- aucun
modèle chargé, aucun GPU requis, démarre en quelques secondes sur n'importe quelle
machine ayant accès au dépôt.

Usage :
    .venv/bin/python -m streamlit run src/visualization/dashboard.py
    # ou, depuis la racine du dépôt :
    .venv/bin/streamlit run src/visualization/dashboard.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# `streamlit run src/visualization/dashboard.py` exécute ce fichier comme script
# top-level -- Streamlit met le dossier du script (src/visualization/) sur
# sys.path, PAS la racine du dépôt, donc `import src.*` échoue
# (ModuleNotFoundError: No module named 'src') sauf sous pytest, qui a
# `pythonpath = ["."]` dans pyproject.toml. Ajouté ici pour marcher dans les
# deux cas.
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.analysis.stats import proportion_with_ci, two_proportion_test  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Découverte des runs disponibles
# ─────────────────────────────────────────────────────────────────────────────

def discover_result_dirs() -> list[str]:
    dirs = sorted(glob.glob(os.path.join(REPO_ROOT, "results_*")))
    # results_diagnostics/ n'est pas un run de pipeline (sorties agrégées de
    # scripts/generate_diagnostic_plots.py) -- exclu du sélecteur de run.
    return [os.path.relpath(d, REPO_ROOT) for d in dirs
            if os.path.isdir(d) and os.path.basename(d) != "results_diagnostics"]


def load_json(path: str) -> dict | None:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def _judge_name(judge_model_id) -> str:
    return os.path.basename(str(judge_model_id).rstrip("/")) or str(judge_model_id)


def _judge_label_sources(run_dir: str, prefix: str) -> dict[str, dict]:
    """Retourne {nom_affiché: features_dict} pour TOUTES les sources de
    labels juge disponibles dans ce run, `prefix` = "p1" ou "p2".

    Plusieurs formats de fichier coexistent depuis l'introduction du juge
    Qwen (AUDIT_SAE_2026-08.md §9) : le cache plat d'origine
    ({f_idx: {...}}, juge non enregistré dans le fichier avant le sidecar
    `.meta.json` ajouté cette session), les comparaisons juge alternatif
    ({summary, alt_per_feature}, `judge_model_separation_test.py`/
    `b1_stratified_mixte_qwen_rejudge.py`), et le rejugement par sélection
    stratifiée ({results, bin_info, ...}, `b2_stratified_selection_rejudge.py`,
    P1 uniquement). Sans ce sélecteur, la page affichait TOUJOURS le cache
    plat d'origine sans dire de quel juge il vient -- silencieusement
    trompeur maintenant que Gemma et Qwen coexistent dans le même SAVE_DIR."""
    cache_dir = os.path.join(REPO_ROOT, run_dir, "cache")
    # (clé, dict de features, contaminé ?) -- contaminé = négatif odd-one-out trivialement
    # court (§115/§117) ; trié en fin de fonction pour que le sélecteur ne mette JAMAIS un
    # taux non comparable au chiffre de référence du rapport en premier (résultat par
    # défaut) sans pour autant le faire disparaître (R6/CLAUDE.md : jamais choisir
    # silencieusement entre deux caches coexistants).
    candidates: list[tuple[str, dict, bool]] = []

    flat_name = f"{prefix}_judge_labels_extended.json" if prefix == "p1" else "p2_feature_labels.json"
    flat_path = os.path.join(cache_dir, flat_name)
    flat = load_json(flat_path)
    if flat:
        meta = load_json(flat_path + ".meta.json")
        judge = _judge_name(meta["judge_model_id"]) if meta else "non enregistré (cache antérieur au sidecar de métadonnées)"
        method = meta.get("feature_selection_method") if meta else "inconnue"
        candidates.append((f"{flat_name} — juge : {judge}, sélection : {method}", flat, False))

    if prefix == "p1":
        for path in sorted(glob.glob(os.path.join(cache_dir, "p1_judge_model_separation_*.json"))):
            data = load_json(path)
            if data and "alt_per_feature" in data:
                judge = _judge_name(data.get("summary", {}).get("judge_alternative", "?"))
                candidates.append((f"{os.path.basename(path)} — juge : {judge} (comparaison, mêmes exemples que la référence)",
                                    data["alt_per_feature"], False))

        for path in sorted(glob.glob(os.path.join(cache_dir, "b1_stratified_mixte_qwen_rejudge_*.json"))):
            data = load_json(path)
            if data and "alt_per_feature" in data:
                judge = _judge_name(data.get("summary", {}).get("judge_alternative", "?"))
                candidates.append((f"{os.path.basename(path)} — juge : {judge} (arme mixte stratifiée, N4)",
                                    data["alt_per_feature"], False))

        b2_path = os.path.join(cache_dir, "b2_stratified_selection_rejudge.json")
        data = load_json(b2_path)
        if data and "results" in data:
            candidates.append(("b2_stratified_selection_rejudge.json — juge : gemma-3-12b-it (isole la méthode de sélection, pas le juge — §79)",
                                data["results"], False))

    # Marque contaminé toute source dont le négatif odd-one-out est majoritairement
    # trivial (>30% de négatifs ≤3 mots) -- ces caches prédatent le correctif profond
    # (§115-117) et donnent des taux nettement plus hauts que le chiffre de référence du
    # rapport (65,7%, R0/§119), non comparables tels quels.
    tagged: list[tuple[str, dict, bool]] = []
    for label, data, _ in candidates:
        diag = _negative_bias_diagnostic(data)
        contaminated = diag is not None and diag["frac_le_3_words"] > 0.3
        if contaminated:
            label = f"{label} — ⚠ négatif non corrigé (pré-§115/117), taux non comparable au rapport"
        tagged.append((label, data, contaminated))

    out: dict[str, dict] = {}
    for label, data, _ in sorted(tagged, key=lambda t: t[2]):  # False (propre) avant True (contaminé)
        out[label] = data
    return out


def _negative_bias_diagnostic(label_map: dict) -> dict | None:
    """Fraction des négatifs odd-one-out trivialement pauvres en contexte (≤3
    mots) -- signal direct de la contamination RESULTS_TESTS.md §115/§117 : un
    négatif réduit à un mot de salutation/ponctuation permet au juge de
    résoudre la tâche par longueur de contexte plutôt que par concept partagé.
    None si `label_map` ne porte pas de champ `neg_example` exploitable (cache
    antérieur à son ajout, ou dictionnaire sans construction de négatif
    comparable)."""
    lens = [len(str(v["neg_example"]).split()) for v in label_map.values()
            if isinstance(v, dict) and v.get("neg_example")]
    if not lens:
        return None
    n = len(lens)
    n_short = sum(1 for length in lens if length <= 3)
    return {"n": n, "n_le_3_words": n_short, "frac_le_3_words": n_short / n}


def _negative_bias_caption(label_map: dict) -> None:
    diag = _negative_bias_diagnostic(label_map)
    if diag is None:
        return
    if diag["frac_le_3_words"] > 0.3:
        st.warning(
            f"⚠️ Négatif odd-one-out trivialement court (≤3 mots) pour "
            f"{diag['n_le_3_words']}/{diag['n']} features ({100*diag['frac_le_3_words']:.0f}%) -- "
            "le juge peut résoudre la tâche par longueur de contexte plutôt que par concept "
            "partagé (RESULTS_TESTS.md §115/§117). Le taux ci-dessus est probablement surestimé "
            "pour cette source."
        )
    else:
        st.caption(
            f"Diagnostic de contamination du négatif (RESULTS_TESTS.md §115/§117) : "
            f"{diag['n_le_3_words']}/{diag['n']} négatifs ≤3 mots -- sous le seuil de contamination "
            "critique observé sur la référence pré-correctif (98%)."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Pages
# ─────────────────────────────────────────────────────────────────────────────

def page_overview(run_dir: str) -> None:
    st.header("Vue d'ensemble du run")

    ref = _load_flat_judge_rate(_REFERENCE_RUN)
    if ref is not None:
        st.info(
            f"**Chiffre de référence du rapport de stage : {100*ref['rate']:.1f}% "
            f"({ref['succ']}/{ref['n']})**, run R0 ({_REFERENCE_RUN}) -- sélection "
            f"stratifiée, juge {ref['judge']}, négatif corrigé. Indépendant du run "
            "choisi ci-contre : les métriques ci-dessous reflètent le run sélectionné, "
            "pas nécessairement R0."
        )

    results = load_json(os.path.join(REPO_ROOT, run_dir, "results.json"))
    if not results:
        st.warning("Pas de results.json dans ce run (run partiel ou script encore en cours).")
        return

    st.subheader("Interprétabilité des features apprises (question métier centrale)")
    cols = st.columns(2)
    for col, prefix, title in [(cols[0], "p1", "Pipeline 1 — extension"),
                                (cols[1], "p2", "Pipeline 2")]:
        with col:
            sources = _judge_label_sources(run_dir, prefix)
            if not sources:
                st.info(f"Aucun label {prefix.upper()} pour ce run.")
                continue
            chosen_key = next(iter(sources))
            labels = sources[chosen_key]
            n_interp = sum(1 for v in labels.values() if v.get("interp_score") == 1)
            st.metric(f"{title} — taux d'interprétabilité", f"{100*n_interp/len(labels):.1f}%",
                       help=f"{n_interp}/{len(labels)} — source : {chosen_key}")
            diag = _negative_bias_diagnostic(labels)
            if diag and diag["frac_le_3_words"] > 0.3:
                st.caption("⚠️ négatif odd-one-out possiblement contaminé -- détail sur l'onglet Features.")

    for pipeline_key, title in [("P1_Gemma3_SAE", "Pipeline 1 — Gemma-3 + GemmaScope"),
                                 ("P2_F2LLM_PhSAE", "Pipeline 2 — F2LLM + PhraseLevelSAE")]:
        metrics = results.get(pipeline_key)
        if not metrics:
            continue
        st.subheader(f"{title} — fidélité de reconstruction")
        # diff_hypothesis (texte libre généré par LLM, non vérifié) vit désormais
        # sur l'onglet Diffing, à côté du verification_rate qui le contextualise --
        # l'exclure ici évite aussi une colonne à types mixtes (float/str) que
        # pyarrow ne peut pas convertir proprement pour le rendu du tableau.
        display = {k: v for k, v in metrics.items()
                   if not isinstance(v, (dict, list)) and k != "diff_hypothesis"}
        st.dataframe(pd.DataFrame([display]).T.rename(columns={0: "valeur"}), width='stretch')


def page_umap(run_dir: str) -> None:
    st.header("UMAP — projection des activations SAE")
    coord_files = sorted(glob.glob(os.path.join(REPO_ROOT, run_dir, "umap_*_coords.parquet")))
    if not coord_files:
        st.warning("Aucun fichier umap_*_coords.parquet dans ce run.")
        return
    chosen = st.selectbox("Projection", [os.path.basename(f) for f in coord_files])
    df = pd.read_parquet(os.path.join(REPO_ROOT, run_dir, chosen))
    color_by = st.radio("Colorer par", [c for c in ["label", "cluster_id", "cluster_signature"] if c in df.columns],
                         horizontal=True)
    hover_cols = [c for c in ["text_preview", "top_features", "cluster_signature"] if c in df.columns]
    fig = px.scatter(
        df, x="x", y="y", color=df[color_by].astype(str),
        hover_data=hover_cols, opacity=0.7, height=650,
        title=chosen,
    )
    fig.update_layout(legend_title_text=color_by)
    st.plotly_chart(fig, width='stretch')
    with st.expander("Données brutes (échantillon)"):
        st.dataframe(df.sample(min(200, len(df))), width='stretch')


def _feature_search_box(label_map: dict, key: str) -> None:
    query = st.text_input("Filtrer par texte du label/description", key=key)
    rows = []
    for f_idx, v in label_map.items():
        if isinstance(v, str):
            label, desc, interp, rho, pos, neg = v, "", None, None, [], None
        else:
            label = v.get("label", f"F{f_idx}")
            desc = v.get("brief_description", "")
            interp = v.get("interp_score")
            rho = v.get("rho_interp")
            pos = v.get("pos_examples", [])
            neg = v.get("neg_example")
        if query and query.lower() not in f"{label} {desc}".lower():
            continue
        rows.append({"feature": f_idx, "label": label, "description": desc,
                     "interp_score": interp, "rho_interp": rho,
                     "n_pos_examples": len(pos) if pos else 0,
                     "_pos": pos, "_neg": neg})
    if not rows:
        st.info("Aucune feature ne correspond au filtre.")
        return
    df = pd.DataFrame(rows)
    st.write(f"{len(df)} features")
    st.dataframe(df.drop(columns=["_pos", "_neg"]), width='stretch', height=300)
    chosen_idx = st.selectbox("Voir les exemples d'une feature", df["feature"].tolist(), key=key + "_sel")
    row = df[df["feature"] == chosen_idx].iloc[0]
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Exemples positifs** (activent fortement la feature)")
        for ex in (row["_pos"] or []):
            st.markdown(f"- {ex}")
        if not row["_pos"]:
            st.caption("Aucun exemple positif stocké.")
    with col2:
        st.markdown("**Exemple négatif** (contrôle, activation quasi-nulle)")
        if row["_neg"]:
            st.markdown(f"- {row['_neg']}")
        else:
            st.caption("Non stocké (run antérieur à l'ajout de ce champ dans src/sae/judge.py, "
                       "ou feature core Neuronpedia sans contrôle négatif applicable).")


def _full_neuronpedia_catalog() -> None:
    """Catalogue COMPLET des labels Neuronpedia mis en cache localement (toutes
    largeurs de SAE confondues), indépendant du sous-ensemble top-N sélectionné
    par un run (p1_top_core_features.json) -- répond au besoin de parcourir
    l'intégralité des features déjà auto-interprétées par DeepMind/Neuronpedia,
    pas seulement celles les plus activées sur le corpus d'entraînement local."""
    label_files = sorted(glob.glob(os.path.join(REPO_ROOT, "local_data", "neuronpedia_labels", "*.json")))
    if not label_files:
        st.info("Aucun fichier sous local_data/neuronpedia_labels/.")
        return
    rel_files = [os.path.relpath(f, REPO_ROOT) for f in label_files]
    # Par défaut, propose le fichier le plus volumineux (généralement la largeur
    # avec le plus de labels, ex. 65k) plutôt que le premier alphabétiquement.
    default_idx = max(range(len(label_files)), key=lambda i: os.path.getsize(label_files[i]))
    chosen = st.selectbox("Fichier de labels Neuronpedia", rel_files, index=default_idx, key="np_catalog_file")
    catalog = load_json(os.path.join(REPO_ROOT, chosen)) or {}
    st.metric("Features labellisées dans ce fichier", f"{len(catalog):,}".replace(",", " "))
    query = st.text_input("Filtrer par texte du label (laisser vide = 500 premières features par index)",
                           key="np_catalog_query")
    items = sorted(catalog.items(), key=lambda kv: int(kv[0]))
    if query:
        items = [(k, v) for k, v in items if query.lower() in str(v).lower()]
        st.write(f"{len(items)} features correspondent au filtre.")
    else:
        items = items[:500]
        st.caption("Aucun filtre : affichage des 500 premières features par index (sur "
                   f"{len(catalog):,}".replace(",", " ") + " au total). Utiliser la recherche pour cibler.")
    df = pd.DataFrame([{"feature": k, "label": v} for k, v in items])
    st.dataframe(df, width='stretch', height=400)


def page_features(run_dir: str) -> None:
    st.header("Features — labels et exemples")
    tab_core, tab_ext, tab_p2 = st.tabs(["Core (Neuronpedia)", "Extension (juge LLM, P1)", "Phrase-level (juge LLM, P2)"])

    with tab_core:
        core = load_json(os.path.join(REPO_ROOT, run_dir, "p1_top_core_features.json"))
        if core:
            st.subheader(f"Top-{len(core)} features core les plus activées (ce run)")
            _feature_search_box(core, key="core")
        else:
            st.info("p1_top_core_features.json absent de ce run.")
        with st.expander("Catalogue COMPLET des features Neuronpedia (toutes largeurs de SAE en cache local)"):
            _full_neuronpedia_catalog()

    with tab_ext:
        sources = _judge_label_sources(run_dir, "p1")
        if sources:
            chosen = st.selectbox("Source des labels (juge)", list(sources.keys()), key="ext_judge_source")
            ext = sources[chosen]
            n_interp = sum(1 for v in ext.values() if v.get("interp_score") == 1)
            st.metric("Taux d'interprétabilité (odd-one-out)", f"{100*n_interp/len(ext):.1f}%",
                       help=f"{n_interp}/{len(ext)} features passent le test — source : {chosen}")
            if len(sources) > 1:
                st.caption(f"{len(sources)} sources de labels trouvées pour ce run (juges/sélections "
                           "différents coexistent depuis l'introduction du juge Qwen) — choisir ci-dessus.")
            _negative_bias_caption(ext)
            _feature_search_box(ext, key="ext")
        else:
            st.info("Aucun cache de labels d'extension trouvé pour ce run.")

    with tab_p2:
        sources_p2 = _judge_label_sources(run_dir, "p2")
        if sources_p2:
            chosen_p2 = st.selectbox("Source des labels (juge)", list(sources_p2.keys()), key="p2_judge_source")
            p2 = sources_p2[chosen_p2]
            n_interp_p2 = sum(1 for v in p2.values() if v.get("interp_score") == 1)
            st.metric("Taux d'interprétabilité (odd-one-out)", f"{100*n_interp_p2/len(p2):.1f}%",
                       help=f"{n_interp_p2}/{len(p2)} features passent le test — source : {chosen_p2}")
            p2_neg_diag = load_json(os.path.join(REPO_ROOT, run_dir, "cache", "p2_negative_length_audit.json"))
            if p2_neg_diag:
                st.caption(
                    "Diagnostic de contamination du négatif (RESULTS_TESTS.md §118) : longueur "
                    f"moyenne négatif {p2_neg_diag['neg_len_words_mean']:.1f} mots contre "
                    f"{p2_neg_diag['pos_len_words_mean_overall']:.1f} mots pour les positifs, "
                    f"{100*p2_neg_diag['neg_le_3_words_frac']:.0f}% de négatifs ≤3 mots -- le biais "
                    "de longueur identifié sur Pipeline 1 (§115) ne se réplique pas ici."
                )
            _feature_search_box(p2, key="p2")
        else:
            st.info("Aucun cache de labels Pipeline 2 trouvé pour ce run.")


@st.cache_data
def _load_email_corpus() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    mails_path = os.path.join(REPO_ROOT, "local_data", "emails", "Mails.tsv")
    aug_path = os.path.join(REPO_ROOT, "local_data", "emails", "augmented_mails.jsonl")
    if not (os.path.exists(mails_path) and os.path.exists(aug_path)):
        return None
    mails = pd.read_csv(mails_path, sep="\t", index_col=0)
    augmented = pd.read_json(aug_path, lines=True)
    # pandas infère parent_id comme int64 (colonne JSON entièrement numérique) --
    # re-forcé en str pour matcher mails.index.astype(str) sans dépendre de
    # l'inférence de type de read_json.
    augmented["parent_id"] = augmented["parent_id"].astype(str)
    return mails, augmented


def page_email_comparison() -> None:
    st.header("Comparaison mail original / variantes augmentées")
    st.caption(
        "Lien via `parent_id` (index de ligne dans `Mails.tsv`) — 13 variantes par "
        "mail original, sur 4 axes de perturbation (emotion, registre, orthographe, "
        "urgence). cf. `src/data/augmentation.py`. Indépendant du run sélectionné "
        "dans la barre latérale (lit directement `local_data/emails/`)."
    )
    corpus = _load_email_corpus()
    if corpus is None:
        st.warning("Mails.tsv ou augmented_mails.jsonl absent de local_data/emails/ "
                    "(absent hors machine de calcul).")
        return
    mails, augmented = corpus

    with st.expander("Taux de rejet par axe/niveau (contrôle qualité de la génération)"):
        # Remonte en métrique visible ce qui n'était lisible qu'en commentaire de code
        # (audit 2026-08 round 3, §6.3) -- déséquilibre marqué par classe, pas seulement
        # une moyenne globale rassurante.
        rej = augmented.assign(is_rejected=augmented["rejected"].notna())
        rate_by_class = (
            rej.groupby(["axis", "level"])["is_rejected"].mean().mul(100).round(1)
            .sort_values(ascending=False).rename("taux_rejet_%")
        )
        st.dataframe(rate_by_class.reset_index(), width='stretch', height=300)
        st.caption(f"Taux de rejet global : {100*rej['is_rejected'].mean():.1f}% "
                    f"({int(rej['is_rejected'].sum())}/{len(rej)})")

    parent_id = st.selectbox(
        "Mail original (parent_id)",
        options=mails.index.astype(str).tolist(),
        format_func=lambda pid: f"#{pid} — {mails.loc[int(pid), 'document'][:80]!r}",
    )
    variants = augmented[augmented["parent_id"] == parent_id]

    st.subheader(f"Original — mail #{parent_id}")
    st.text_area("original_text", mails.loc[int(parent_id), "document"],
                 height=200, disabled=True, label_visibility="collapsed")

    if variants.empty:
        st.info("Aucune variante augmentée trouvée pour ce parent_id.")
        return

    axes = sorted(variants["axis"].unique())
    chosen_axis = st.radio("Axe de perturbation", axes, horizontal=True)
    axis_variants = variants[variants["axis"] == chosen_axis]

    st.subheader(f"Variantes — axe « {chosen_axis} »")
    cols = st.columns(len(axis_variants))
    for col, (_, row) in zip(cols, axis_variants.iterrows()):
        with col:
            st.markdown(f"**{row['level']}**")
            if row.get("rejected"):
                # Variante rejetée au contrôle qualité de la génération (texte non
                # stocké, motif conservé pour audit) -- ~11,7% du corpus augmenté EN
                # MOYENNE (5291/45240), mais très hétérogène par classe : 59,6%
                # (orthographe__degrade_fort) et 47,2% (emotion__impatience) contre
                # ~4% pour les 11 autres classes, presque toujours par length_ratio
                # trop bas -- cf. src/data/augmentation.py.
                st.caption(f"Rejetée au contrôle qualité : `{row['rejected']}`")
            else:
                st.text_area(row["aug_id"], row["text"], height=300, disabled=True,
                             label_visibility="collapsed")


def page_diffing(run_dir: str) -> None:
    st.header("Diffing de corpus (Fisher exact + BH)")
    csv_files = sorted(glob.glob(os.path.join(REPO_ROOT, run_dir, "**", "diff_*.csv"), recursive=True))
    csv_files += sorted(glob.glob(os.path.join(REPO_ROOT, run_dir, "diff_*.csv")))
    if not csv_files:
        st.info("Aucun diff_*.csv trouvé dans ce run (le diffing vit typiquement sous "
                "cache_baseline*/ ou à la racine du run pour p1_diff_energy_sports.csv).")
    else:
        rel_files = [os.path.relpath(f, REPO_ROOT) for f in csv_files]
        chosen = st.selectbox("Fichier de diff", rel_files)
        df = pd.read_csv(os.path.join(REPO_ROOT, chosen))
        n_sig = int(df["significant"].sum()) if "significant" in df.columns else None
        if n_sig is not None:
            st.metric("Features significatives (q<0.05)", f"{n_sig}/{len(df)}")
        st.dataframe(df.head(50), width='stretch')

    st.divider()
    st.subheader("Vérification d'hypothèses (App K.1)")
    st.caption(
        "Un écart de fréquence Fisher/BH (tableau ci-dessus) ne dit pas si l'hypothèse sémantique "
        "qui l'accompagne se vérifie sur un corpus frais -- `verification_rate` (fraction "
        "d'hypothèses dont l'écart vérifié dépasse 1 point) et `coverage` (fraction des documents "
        "cible couverts par au moins une hypothèse valide) répondent à ça (Figures 11/12 du papier "
        "de référence)."
    )
    verif_files = {
        "diffing_hypothesis_verification.json (labels archivés, top-q NPMI)": "diffing_hypothesis_verification.json",
        "diffing_structured_hypotheses.json (génération structurée par LLM)": "diffing_structured_hypotheses.json",
    }
    verif_found = {}
    for label, fname in verif_files.items():
        data = load_json(os.path.join(REPO_ROOT, run_dir, "cache", fname))
        if not data:
            continue
        verif = data if ("summary" in data and "per_hypothesis" in data) else data.get("verification")
        if verif:
            verif_found[label] = verif
    if verif_found:
        chosen_v = st.selectbox("Source des hypothèses", list(verif_found.keys()), key="diff_verif_source")
        verif = verif_found[chosen_v]
        s = verif["summary"]
        vcol1, vcol2 = st.columns(2)
        vcol1.metric("Taux de vérification", f"{100*s['verification_rate']:.1f}%",
                      help=f"seuil {100*s['threshold']:.0f} point -- {s['n_hypotheses']} hypothèses testées")
        vcol2.metric("Couverture", f"{100*s['coverage']:.1f}%",
                      help=f"{s['n_documents_in_group']} documents cible")
        st.dataframe(pd.DataFrame(verif["per_hypothesis"]), width='stretch')
    else:
        st.info("Aucune vérification d'hypothèse (App K.1) trouvée pour ce run.")

    st.divider()
    st.subheader("Hypothèse libre (diffing cross-domaine, non vérifiée)")
    results = load_json(os.path.join(REPO_ROOT, run_dir, "results.json"))
    diff_hyp = (results or {}).get("P1_Gemma3_SAE", {}).get("diff_hypothesis")
    if diff_hyp:
        st.caption(
            "Génération libre par le juge LLM à partir des features les plus discriminantes du "
            "diffing -- à distinguer du verification_rate ci-dessus, seule mesure quantifiée et "
            "comparable au papier. Peut contenir du texte de raisonnement brut du modèle plutôt "
            "qu'une hypothèse propre selon le juge/checkpoint utilisé."
        )
        st.text(diff_hyp)
    else:
        st.info("Pas d'hypothèse LLM libre pour ce run.")


def page_search(run_dir: str) -> None:
    st.header("Recherche par concept (sur les labels de features)")
    st.caption("Recherche par mot-clé sur les labels/descriptions déjà attribués (Neuronpedia + juge LLM) — "
               "pas une ré-inférence live du modèle. Pour une recherche BM25 sur le vocabulaire latent complet, "
               "voir `src/sae/retrieval/latent_terms.py` / `scripts/retrieval_demo.py`.")
    all_labels = {}
    for fname in ["p1_top_core_features.json", "p1_top_extended_features.json"]:
        d = load_json(os.path.join(REPO_ROOT, run_dir, fname))
        if d:
            all_labels.update({f"{fname}:{k}": v for k, v in d.items()})
    ext = load_json(os.path.join(REPO_ROOT, run_dir, "cache", "p1_judge_labels_extended.json"))
    if ext:
        all_labels.update({f"extension:{k}": v for k, v in ext.items()})

    query = st.text_input("Requête (ex. 'urgence', 'facturation', 'résiliation')")
    if not query:
        st.info("Entrer une requête pour lister les features dont le label/description matche.")
    else:
        rows = []
        for key, v in all_labels.items():
            label = v.get("label", "") if isinstance(v, dict) else str(v)
            desc = v.get("brief_description", "") if isinstance(v, dict) else ""
            text = f"{label} {desc}".lower()
            if query.lower() in text:
                rows.append({"feature": key, "label": label, "description": desc})
        if rows:
            st.dataframe(pd.DataFrame(rows), width='stretch')
        else:
            st.info("Aucune feature trouvée pour cette requête dans ce run.")

    st.divider()
    st.subheader("Retrieval par requête métier (RRF + reranking LLM, App G)")
    st.caption(
        "Recherche de documents par intention client (pas par label de feature) -- fusion TF-IDF + "
        "Latent Terms (RRF) puis reranking LLM, sur 4 requêtes paraphrasées, corpus complet. "
        "cf. src/sae/retrieval/latent_terms.py, RESULTS_TESTS.md §80/§92."
    )
    retrieval = load_json(os.path.join(REPO_ROOT, run_dir, "cache", "latent_retrieval_precision_results.json"))
    if retrieval and retrieval.get("per_query"):
        per_query_rows = [{
            "intention": intent, "requête": q["query"], "taux de base": q["base_rate"],
            "P@10 TF-IDF": q["precision_at_10_tfidf"], "P@10 Latent Terms": q["precision_at_10_latent_terms"],
            "P@10 RRF": q["precision_at_10_rrf"], "P@10 RRF+rerank": q["precision_at_10_rrf_reranked"],
            "RBO (TF-IDF vs Latent Terms)": q["rbo_tfidf_vs_latent_terms"],
        } for intent, q in retrieval["per_query"].items()]
        st.dataframe(pd.DataFrame(per_query_rows), width='stretch')

        agg = retrieval.get("aggregate", {})
        method_names = [("tfidf", "TF-IDF"), ("latent_terms", "Latent Terms"),
                        ("rrf", "RRF"), ("rrf_reranked", "RRF + rerank LLM")]
        agg_rows = [{"méthode": name, **agg[key]} for key, name in method_names if key in agg]
        if agg_rows:
            st.dataframe(pd.DataFrame(agg_rows), width='stretch')
        st.caption(
            f"RBO moyen TF-IDF vs Latent Terms : {agg.get('mean_rbo_tfidf_vs_latent_terms', float('nan')):.3f} "
            "(très bas -- les deux méthodes remontent des documents largement différents, pas le même "
            "ensemble réordonné). RRF+rerank domine ou égale les trois autres méthodes sur les 4 "
            "intentions testées, jamais inférieur à RRF seul."
        )
    else:
        st.info("latent_retrieval_precision_results.json absent de ce run (lancer "
                "scripts/latent_retrieval_precision_eval.py).")


# Écritures utilisateur (pistes E08) hors du dépôt Git -- docs/post_stage/
# n'est qu'un template versionné, pas un espace d'écriture partagé (sinon
# chaque piste enregistrée par un utilisateur apparaîtrait comme un fichier
# suivi modifié, avec écrasement silencieux si deux personnes lancent le
# dashboard en parallèle). Racine configurable via SAE_DASHBOARD_STATE_DIR
# (défaut : local_data/, déjà ignoré par .gitignore).
_E08_LEADS_LEGACY_PATH = os.path.join(REPO_ROOT, "docs", "post_stage", "e08_pilot_leads.json")
_E08_STATE_DIR = os.environ.get(
    "SAE_DASHBOARD_STATE_DIR", os.path.join(REPO_ROOT, "local_data", "dashboard_state")
)
_E08_LEADS_PATH = os.path.join(_E08_STATE_DIR, "e08_pilot_leads.json")


def _migrate_legacy_leads() -> None:
    """Copie non destructive, une seule fois : si des pistes existent déjà dans
    l'ancien emplacement suivi par Git et qu'aucun fichier local n'existe encore,
    les reporter dans le nouvel emplacement local. N'écrase jamais un fichier
    local déjà présent, ne supprime jamais l'ancien fichier."""
    if os.path.exists(_E08_LEADS_PATH):
        return
    if not os.path.exists(_E08_LEADS_LEGACY_PATH):
        return
    with open(_E08_LEADS_LEGACY_PATH, encoding="utf-8") as f:
        legacy = json.load(f)
    if not legacy:
        return
    os.makedirs(_E08_STATE_DIR, exist_ok=True)
    with open(_E08_LEADS_PATH, "w", encoding="utf-8") as f:
        json.dump(legacy, f, indent=2, ensure_ascii=False)


def _load_leads() -> list[dict]:
    _migrate_legacy_leads()
    if os.path.exists(_E08_LEADS_PATH):
        with open(_E08_LEADS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []


def _append_lead(lead: dict) -> None:
    leads = _load_leads()
    leads.append(lead)
    os.makedirs(_E08_STATE_DIR, exist_ok=True)
    tmp = _E08_LEADS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(leads, f, indent=2, ensure_ascii=False)
    os.replace(tmp, _E08_LEADS_PATH)


def page_pilot_e08(run_dir: str) -> None:
    st.header("Mini-pilote analyste (E08)")
    st.caption(
        "Recette minimale (plan §13) : démarre sans GPU ni réseau externe, lit uniquement les "
        "artefacts déjà figés d'E01-E04 sur ce run -- aucun job LLM en direct pendant une séance. "
        "Ceci est l'outil de la séance, pas la séance elle-même : les sessions avec 2-3 participants "
        "réels (dont Grégoire) restent à mener, ne sont pas simulées ici."
    )

    e01 = load_json(os.path.join(REPO_ROOT, run_dir, "e01_representation_comparison.json"))
    e03 = load_json(os.path.join(REPO_ROOT, run_dir, "e03_property_retrieval.json"))
    e04 = load_json(os.path.join(REPO_ROOT, run_dir, "e04_diffing.json"))
    missing = [name for name, d in
               (("e01_representation_comparison.json", e01),
                ("e03_property_retrieval.json", e03),
                ("e04_diffing.json", e04)) if d is None]
    if missing:
        st.warning(
            f"Artefacts absents de {run_dir} : {', '.join(missing)}. Les tâches correspondantes "
            "ci-dessous resteront indisponibles pour ce run -- pas de repli silencieux sur un "
            "autre run ou une donnée simulée."
        )

    st.divider()
    st.subheader("Tâche 1 — Retrouver des emails selon une propriété (E03)")
    if e03 is None:
        st.info("e03_property_retrieval.json absent : cette tâche n'est pas disponible pour ce run.")
    else:
        queries = e03["per_query_results"]
        query_ids = [q["query_id"] for q in queries]
        q_choice = st.selectbox("Requête (propriété)", query_ids, key="e08_query")
        q = next(q for q in queries if q["query_id"] == q_choice)
        st.write(f"**Texte de la requête :** {q['query_text']}")
        methods = list(e03["methods"])
        method_choice = st.radio("Méthode de retrieval", methods, horizontal=True, key="e08_method")
        top_docs = q.get("top_documents", {}).get(method_choice)
        if top_docs is None:
            st.info(
                "Pas de documents détaillés persistés pour cette méthode/requête sur ce run "
                "(champ ajouté après le premier passage d'E03 -- relancer scripts/post_stage/"
                "e03_property_retrieval.py pour ce run si absent)."
            )
        else:
            st.metric(f"P@10 strict ({method_choice})",
                      f"{100*q['metrics'][method_choice]['p_at_10_strict']:.0f}%")
            df = pd.DataFrame(top_docs)
            df["pertinence"] = df["relevance_judged"].map({0: "non", 1: "partiel", 2: "oui"})
            st.dataframe(df[["rank", "score", "pertinence", "text_snippet"]], width='stretch')
            st.caption("Pertinence jugée par Qwen (0/1/2), pas par relecture humaine -- calibration "
                       "humaine encore en attente (plan §8.3).")

    st.divider()
    st.subheader("Tâche 2 — Comparer deux sous-populations pour proposer des thèmes (E04)")
    if e04 is None:
        st.info("e04_diffing.json absent : cette tâche n'est pas disponible pour ce run.")
    else:
        c = e04["contrast"]
        st.write(f"**Contraste :** `{c['label_a']}` (cible) vs `{c['label_b']}`")
        stats = pd.DataFrame(e04["verification"]["stats_with_ci_and_fdr"])
        stats["rate_in_group"] = stats["n_success_a"] / stats["n_a"]
        stats["rate_out_group"] = stats["n_success_b"] / stats["n_b"]
        stats_display = stats[["hypothesis", "rate_in_group", "rate_out_group", "diff", "p_fdr_bh"]].copy()
        stats_display["survit FDR-BH (p<0.05)"] = stats["p_fdr_bh"] < 0.05
        st.dataframe(stats_display, width='stretch')
        st.caption(
            "`diff` positif = plus fréquent dans le groupe cible sur CONFIRM ; un `diff` négatif "
            "signifie que l'hypothèse, bien que statistiquement vérifiée, l'est dans le sens INVERSE "
            "de sa génération -- ne pas la retenir comme thème du groupe cible sans relire ce signe."
        )

    st.divider()
    st.subheader("Repère : CORE vs FULL (E01)")
    if e01 is not None:
        acc = e01.get("accuracies", {})
        cols = st.columns(len(acc)) if acc else []
        for col, (rep, v) in zip(cols, acc.items()):
            col.metric(rep, f"{100*v:.1f}%")
        st.caption("Sonde des axes d'augmentation (14 classes, held-out DEV de l'ancien manifeste), voir docs/post_stage/e01_results.md "
                   "pour les IC bootstrap et la lecture complète -- ne pas comparer ces chiffres seuls "
                   "sans les intervalles.")

    st.divider()
    st.subheader("Catalogue de thèmes — enregistrer une piste")
    st.caption(
        "Une piste est un objet structuré (titre, question, populations, propriété, exemples, "
        "contre-exemples, méthode, statut, commentaire humain) -- exporté même si la piste est "
        "rejetée, pour garder trace du jugement humain, pas seulement des pistes retenues. "
        "Stocké localement (hors Git) : lecture-modification-écriture non verrouillée -- deux "
        "personnes enregistrant une piste au même instant peuvent s'écraser l'une l'autre, non "
        "testé à ce jour."
    )
    with st.form("e08_lead_form", clear_on_submit=True):
        title = st.text_input("Titre")
        question = st.text_area("Question de départ")
        populations = st.text_input("Populations comparées")
        prop = st.text_input("Propriété / requête utilisée")
        examples = st.text_area("Exemples (extraits justificatifs)")
        counter_examples = st.text_area("Contre-exemples")
        method_used = st.selectbox("Méthode", ["E03 retrieval", "E04 diffing", "autre"], key="e08_lead_method")
        status = st.selectbox("Statut", ["retenue", "rejetée", "à creuser"])
        comment = st.text_area("Commentaire humain")
        participant = st.text_input("Participant (nom ou rôle, ex. \"utilisateur de recherche 1\")")
        submitted = st.form_submit_button("Enregistrer la piste")
        if submitted:
            if not title.strip():
                st.error("Titre requis.")
            else:
                import datetime
                _append_lead({
                    "title": title, "question": question, "populations": populations,
                    "property": prop, "examples": examples, "counter_examples": counter_examples,
                    "method": method_used, "status": status, "comment": comment,
                    "participant": participant, "run_dir": run_dir,
                    "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                })
                st.success("Piste enregistrée.")

    leads = _load_leads()
    if leads:
        st.write(f"**{len(leads)} piste(s) enregistrée(s)**")
        st.dataframe(pd.DataFrame(leads), width='stretch')
    else:
        st.info("Aucune piste enregistrée pour l'instant.")


def page_stability_e05(run_dir: str) -> None:
    st.header("Stabilité inter-graines des features EXTRA (E05)")
    files = [f for f in ("e05_stability_random_init.json", "e05_stability.json")
             if os.path.exists(os.path.join(REPO_ROOT, run_dir, f))]
    if not files:
        st.info("Aucun e05_stability*.json dans ce run.")
        return
    fname = st.selectbox("Analyse", files, format_func=lambda f: {
        "e05_stability_random_init.json": "4 runs : 2 init PCA + 2 init aléatoire (avec témoin d'indépendance à l'init)",
        "e05_stability.json": "3 runs, toutes init PCA (graines 42/43/44)"}[f])
    d = load_json(os.path.join(REPO_ROOT, run_dir, fname))
    st.warning(d["init_caveat"])
    fb = d.get("found_across_runs_reference_side")
    if fb:
        cols = st.columns(3)
        for col, (k, lab) in zip(cols, (("all_other_pca_runs", "dans les autres runs PCA"),
                                        ("all_random_runs", "dans les runs init aléatoire"),
                                        ("all_other_runs", "dans toutes les autres runs"))):
            if k in fb:
                col.metric(f"Features de {fb['reference']} retrouvées {lab}",
                           f"{fb[k]['n_found']}/{fb['n_reference_supported']}", help=fb["note"])
    fb_old = d.get("found_in_both_available_repetitions")
    if fb_old:
        st.metric("Features retrouvées dans les deux répétitions disponibles",
                  f"{fb_old['n_found_in_both_available_repetitions']}/{fb_old['n_reference_supported']}",
                  help=fb_old["note"])
    if "summary_by_pair_type" in d:
        st.subheader("Par type de paire (arm de l'init : source → cible)")
        rows = [{"type": t, "ordres de paires": len(r["pairs"]), "frac cos≥0,7 ET profil≥0,5": r["frac_joint_cos0.7_and_corr0.5"],
                 "pureté groupes (réel)": r["group_purity_real"], "pureté (nul)": r["group_purity_null"],
                 "recouvrement (réel)": r["group_overlap_real"], "recouvrement (nul)": r["group_overlap_null"],
                 "Spearman score max (réel)": r["group_max_spearman_real"],
                 "Spearman score max (nul)": r["group_max_spearman_null"]}
                for t, r in d["summary_by_pair_type"].items()]
        st.dataframe(pd.DataFrame(rows), width='stretch')
        st.caption("aléatoire→aléatoire = UNE seule paire indépendante (deux sens). Qualité d'entraînement par "
                   "run dans le JSON (`runs[*].training_quality`).")
    st.subheader("Appariement individuel (cosinus signé du décodeur + profils DEV)")
    rows = []
    for pair, s in d["individual_matching"].items():
        rows.append({"paire A→B": pair, "cos médian": s["best_cos_quantiles"]["q50"],
                     "cos médian (nul anisotrope)": s["null_anisotropic_best_cos_quantiles"]["q50"],
                     "frac cos≥0,7": s["frac_cos_ge_0.7"], "frac cos≥0,7 (nul)": s["null_frac_cos_ge_0.7"],
                     "frac cos≥0,7 ET profil≥0,5": s["frac_joint_cos0.7_and_corr0.5"],
                     "corr. profils médiane": s["profile_corr_quantiles"]["q50"],
                     "Jaccard@20 moyen": s["mean_top20_jaccard"]})
    st.dataframe(pd.DataFrame(rows), width='stretch')
    st.subheader("Groupes : réel vs 100 groupes aléatoires de même taille/strate de fréquence")
    metric = st.selectbox("Métrique", ["purity", "overlap", "max_spearman", "mean_active_spearman",
                                       "max_top100_jaccard", "mean_active_top100_jaccard",
                                       "max_top20_jaccard", "mean_active_top20_jaccard"])
    rows = []
    for pair, c in d["group_comparisons"].items():
        a = c["aggregate"]
        rows.append({"paire A→B": pair, "réel (moyenne)": a[f"mean_real_{metric}"],
                     "nul (moyenne)": a[f"mean_null_{metric}"],
                     "groupes significatifs (FDR<0,05)": a[f"n_groups_fdr_lt_0.05_{metric}"],
                     "groupes comparés": a["n_groups_compared"]})
    st.dataframe(pd.DataFrame(rows), width='stretch')
    st.caption("Les 6 paires partagent les mêmes 3 entraînements : pas 6 confirmations indépendantes. "
               "Géométrie de groupe stable, mais le haut de classement documentaire ne l'est pas "
               "(Jaccard@100 significatif pour aucun groupe) -- cf. docs/post_stage/e05_results.md.")
    st.subheader("Carte 2D des directions de la graine de référence (navigation, pas une preuve)")
    pos = pd.DataFrame(d["map_positions_reference"])
    pos["groupe"] = pos["group_id"].astype(str)
    fig = px.scatter(pos, x="x", y="y", color="groupe", hover_data=["feature_uid", "freq_fit"])
    st.plotly_chart(fig, width='stretch')


def page_clustering_e07(run_dir: str) -> None:
    st.header("Regrouper selon une question (E07)")
    d = load_json(os.path.join(REPO_ROOT, run_dir, "e07_clustering.json"))
    if d is None:
        st.info("e07_clustering.json absent de ce run.")
        return
    st.caption(
        f"{d['n_confirm_parents_sampled']} parents CONFIRM échantillonnés, k={d['n_clusters']} fixé pour "
        "toutes les méthodes/tous les axes. DENSE/TFIDF n'ont pas de mécanisme de restriction par axe -- "
        "leurs clusters sont IDENTIQUES sur les 3 axes par construction, pas un bug."
    )
    axis_id = st.selectbox("Axe", list(d["axes"].keys()))
    axis = d["axes"][axis_id]
    st.write(f"**Requête d'axe :** {axis['query']}")

    for branch in ("core", "full", "dense", "tfidf"):
        b = axis[branch]
        with st.expander(f"{branch.upper()} — statut : {b.get('status')}", expanded=(branch in ("core", "full"))):
            if b.get("status") != "ok":
                st.warning(f"n_matched_features={b.get('n_matched_features', 'n/a')} -- axe non pris en charge "
                           "pour cette branche (pas de repli sur tout le dictionnaire).")
                continue
            col1, col2 = st.columns(2)
            col1.metric("Couverture", f"{100*b['coverage']:.1f}%",
                        help=f"{b['n_excluded_no_signal']} documents hors axe / sans signal")
            if "n_matched_features" in b:
                col2.metric("Features matchées", b["n_matched_features"])
            rows = []
            for cid, size in b["cluster_sizes"].items():
                rows.append({
                    "cluster": cid, "taille": size,
                    "libellé LLM": b["cluster_labels_llm"].get(cid, b["cluster_labels_llm"].get(int(cid), "")),
                    "accuracy réassignation (secondaire)": b["reassignment_accuracy_secondary"].get(
                        cid, b["reassignment_accuracy_secondary"].get(int(cid))),
                    "conductance z-score (espace DENSE)": b["conductance_zscore"].get(
                        cid, b["conductance_zscore"].get(int(cid))),
                })
            st.dataframe(pd.DataFrame(rows), width='stretch')
    st.caption(
        "Accuracy de réassignation : diagnostic LLM secondaire (le juge réassigne des documents tenus à "
        "l'écart de la description du cluster à partir du seul libellé) -- une valeur proche de 0 signale "
        "un libellé non reproductible, pas un thème établi. Audit humain aveugle de paires intra/inter-"
        "cluster (plan §12.3) non fait, cf. docs/post_stage/e07_results.md."
    )


def page_correlations_e06(run_dir: str) -> None:
    st.header("Associations de propriétés (E06)")
    d = load_json(os.path.join(REPO_ROOT, run_dir, "e06_correlations.json"))
    if d is None:
        st.info("e06_correlations.json absent de ce run.")
        return
    st.caption(
        f"Découverte NPMI sur FIT+DEV ({d['n_fit']}+{d['n_dev']} docs, {d['n_candidates_discovery']} "
        f"paires candidates), 8 paires gelées avant lecture de CONFIRM, vérifiées sur "
        f"{d['n_confirm_parents_sampled']} parents CONFIRM échantillonnés -- cooccurrence calculée "
        "sur les jugements Qwen, pas sur les activations SAE brutes."
    )
    rows = []
    for r in d["confirmation"]:
        rows.append({
            "propriété A": r["label_a"], "propriété B": r["label_b"],
            "n_a": r["n_a_confirm"], "n_b": r["n_b_confirm"], "n_ab": r["n_ab_confirm"],
            "NPMI (FIT+DEV, découverte)": round(r["npmi_discovery_fitdev"], 3),
            "NPMI (CONFIRM)": round(r["npmi_confirm"], 3) if r["npmi_confirm"] is not None else None,
            "statut": r["status"],
            "survit FDR-BH": r.get("fisher_p_fdr_bh", float("nan")) < 0.05 if r.get("fisher_p_fdr_bh") is not None else None,
        })
    st.dataframe(pd.DataFrame(rows), width='stretch')
    st.caption(
        "`statut=insufficient_support` : table 2×2 trop dégénérée (marge <5) pour un test fiable -- "
        "ne pas lire son NPMI/odds ratio comme une association établie même s'il semble élevé. Un "
        "NPMI proche de 1 peut aussi signaler un doublon de concept (deux features SAE distinctes "
        "convergeant sur le même signal, \"feature splitting\") plutôt qu'une découverte -- vérifier "
        "les libellés avant de citer une paire comme surprenante. cf. docs/post_stage/e06_results.md "
        "pour la lecture complète."
    )


def page_urgence_robustesse(run_dir: str) -> None:
    st.header("Détection d'urgence/intention & robustesse du juge")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Sonde intention/urgence (mails originaux)")
        d = load_json(os.path.join(REPO_ROOT, run_dir, "cache", "intent_urgency_probe_results.json"))
        if d:
            rows = []
            for k, v in d.items():
                row = {"intention": k, "baseline majoritaire": v["majority_baseline"],
                       "acc SAE": v["acc_sae"], "delta vs baseline": v["acc_sae"] - v["majority_baseline"]}
                if "acc_tfidf" in v:
                    row["acc TF-IDF+LogReg"] = v["acc_tfidf"]
                    row["SAE bat TF-IDF ?"] = v["acc_sae"] > v["acc_tfidf"]
                    row["McNemar p (BH)"] = v.get("mcnemar_p_bh")
                rows.append(row)
            df = pd.DataFrame(rows)
            st.dataframe(df, width='stretch')
            if "acc TF-IDF+LogReg" in df.columns:
                st.caption(
                    "Comparaison à la baseline lexicale honnête TF-IDF+LogReg, mêmes plis de "
                    "validation croisée (McNemar apparié, correction BH) -- les codes SAE battent "
                    "ou égalent TF-IDF+LogReg sur les 5 intentions, jamais l'inverse. "
                    "cf. RESULTS_TESTS.md §104."
                )
            else:
                st.caption("cf. scripts/intent_urgency_probe.py, RESULTS_TESTS.md §13.2 "
                           "(baseline TF-IDF pas encore calculée pour ce run).")
        else:
            st.info("intent_urgency_probe_results.json absent (lancer scripts/intent_urgency_probe.py).")
    with col2:
        st.subheader("Robustesse du protocole odd-one-out")
        d = load_json(os.path.join(REPO_ROOT, run_dir, "cache", "p1_judge_robustness.json"))
        if d:
            st.json(d["summary"])
            st.caption("cf. scripts/judge_robustness_check.py, RESULTS_TESTS.md §13.1")
        else:
            st.info("p1_judge_robustness.json absent (lancer scripts/judge_robustness_check.py).")


def page_explanation_quality(run_dir: str) -> None:
    st.header("Qualité de l'explication document-level")
    st.caption("cf. scripts/explanation_fidelity_test.py / explanation_plausibility_test.py, RESULTS_TESTS.md")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Fidélité (ablation)")
        d = load_json(os.path.join(REPO_ROOT, run_dir, "cache", "explanation_fidelity_results.json"))
        if d:
            rows = [{"intention": k, "n_docs": v["n_docs_tested"],
                     "chute_top_k": v["mean_drop_top_k"], "chute_random_k": v["mean_drop_random_k"],
                     "chute_bottom_k": v["mean_drop_bottom_k"],
                     "ratio_top_vs_random": v["fidelity_ratio_top_vs_random"]}
                    for k, v in d.items()]
            st.dataframe(pd.DataFrame(rows), width='stretch')
            intent_choice = st.selectbox("Voir des exemples pour", list(d.keys()))
            for ex in d[intent_choice].get("examples", [])[:5]:
                with st.expander(f"Doc #{ex['doc_idx']} — p_avant={ex['p_before']:.3f}, "
                                  f"chute top-K={ex['drop_top_k']:.3f}"):
                    st.write(ex["text_preview"])
                    st.write("**Features citées comme explication :**")
                    for feat in ex["top_features"]:
                        st.markdown(f"- F{feat['f']} — {feat['label']}")
        else:
            st.info("explanation_fidelity_results.json absent (lancer scripts/explanation_fidelity_test.py).")
    with col2:
        st.subheader("Plausibilité (choix forcé, juge LLM)")
        d = load_json(os.path.join(REPO_ROOT, run_dir, "cache", "explanation_plausibility_results.json"))
        if d:
            s = d["summary"]
            st.metric("Taux de succès (réel vs aléatoire)", f"{100*s['success_rate']:.1f}%",
                       help=f"{s['n_correct']}/{s['n_tested']} — hasard = 50%")
            wrong = [e for e in d.get("examples", []) if not e["picked_real"]][:5]
            if wrong:
                st.write("**Exemples où le juge a préféré le décoy aléatoire :**")
                for ex in wrong:
                    with st.expander(f"Doc #{ex['doc_idx']}"):
                        st.write("Réel :", ", ".join(ex["real_labels"]))
                        st.write("Décoy :", ", ".join(ex["decoy_labels"]))
        else:
            st.info("explanation_plausibility_results.json absent (lancer scripts/explanation_plausibility_test.py, GPU).")


def page_diagnostics(run_dir: str) -> None:
    st.header("Diagnostics d'entraînement")
    st.caption("Figures produites par scripts/generate_diagnostic_plots.py (lecture d'artefacts "
               "déjà sur disque, aucun rerun) — cf. CLAUDE.md pour la "
               "checklist de lecture (convergence, fidélité, capacité, interprétabilité, "
               "significativité, indépendance du juge).")

    run_plots_dir = os.path.join(REPO_ROOT, run_dir, "plots")
    run_plot_files = sorted(glob.glob(os.path.join(run_plots_dir, "*.html")))
    st.subheader(f"Run courant — {run_dir}")
    if run_plot_files:
        chosen = st.selectbox("Figure", [os.path.basename(f) for f in run_plot_files], key="diag_run_plot")
        with open(os.path.join(run_plots_dir, chosen), encoding="utf-8") as f:
            st.components.v1.html(f.read(), height=700, scrolling=True)
    else:
        st.info("Pas de figure pour ce run (checkpoint/historique absent, ou script pas encore "
                "lancé). Génère-les avec :\n\n`python scripts/generate_diagnostic_plots.py`")

    st.subheader("Balayages d'hyperparamètres (archive, sélection par magnitude + juge auto-référent)")
    st.caption(
        "Figures figées (sources supprimées par le nettoyage disque, non régénérables) sous "
        "l'ancienne méthodologie -- sélection par magnitude, juge auto-référent, négatif odd-one-out "
        "non corrigé. Le balayage échelle du modèle et le balayage layer sous méthodologie finale "
        "(stratifié + Qwen3.8-27B + négatif corrigé) sont sur l'onglet Sweeps ; les autres "
        "balayages ci-dessous (K_extra, D_extra, hook-point, volume) n'ont pas encore de "
        "remesure homogène et restent à lire comme repères historiques, pas comme valeurs finales."
    )
    sweep_dir = os.path.join(REPO_ROOT, "results_diagnostics", "plots")
    # sweep_model_scale.html/sweep_layer.html retirés du sélecteur : directement supersédés par
    # l'onglet Sweeps (méthodologie finale), garder les deux ici serait montrer côte à côte deux
    # chiffres pour la même question sans dire lequel citer.
    superseded = {"sweep_model_scale.html", "sweep_layer.html"}
    sweep_files = sorted(f for f in glob.glob(os.path.join(sweep_dir, "*.html"))
                          if os.path.basename(f) not in superseded)
    if sweep_files:
        chosen_sweep = st.selectbox("Balayage", [os.path.basename(f) for f in sweep_files], key="diag_sweep_plot")
        with open(os.path.join(sweep_dir, chosen_sweep), encoding="utf-8") as f:
            st.components.v1.html(f.read(), height=700, scrolling=True)
    else:
        st.info("Pas de figure de balayage. Génère-les avec `python scripts/generate_diagnostic_plots.py`.")


def page_consolidated_report(run_dir: str) -> None:
    st.header("Rapport consolidé (toutes les méthodes, conditions fixées)")
    st.caption("cf. docs/evaluation_protocol.md — scripts/consolidate_evaluation_report.py")
    report_path = os.path.join(REPO_ROOT, run_dir, "EVALUATION_REPORT.md")
    if os.path.exists(report_path):
        with open(report_path, encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.warning(f"Pas de rapport consolidé pour ce run. Génère-le avec :\n\n"
                    f"`python scripts/consolidate_evaluation_report.py {run_dir}`")


def page_audit_2026_08() -> None:
    """Agrège les sorties JSON produites par les scripts d'audit méthodologique --
    jusqu'ici dispersées sous docs/ et cache/, lisibles seulement en ouvrant chaque
    fichier à la main. Recherche par motif plutôt que liste en dur : reste à jour sans
    édition à chaque nouveau script d'audit."""
    st.header("Audit méthodologique — archive (§57-96)")
    st.caption("cf. `RESULTS_TESTS.md` §57-96. "
               "Indépendant du run sélectionné dans la barre latérale. "
               "Archive figée des scripts qui ont établi les correctifs désormais actifs par défaut "
               "(sélection stratifiée, juge Qwen3.8-27B découplé de l'extracteur, déduplication par "
               "mail parent) -- ces items sont tranchés (AUDIT_SAE_2026-08.md §9). Pour la campagne "
               "de mesure sous cette méthodologie (balayages échelle/layer, sanity checks, "
               "diffing/clustering/retrieval vérifiés), voir les onglets Sweeps, Clustering & "
               "Corrélations, Diffing et Recherche.")

    patterns = [
        os.path.join(REPO_ROOT, "docs", "audit_*_results.json"),
        os.path.join(REPO_ROOT, "results_v10_emails_main", "cache", "audit_2026_08_*.json"),
        os.path.join(REPO_ROOT, "results_v10_emails_main", "cache", "c2_original_only_rejudge*.json"),
    ]
    files = sorted({f for p in patterns for f in glob.glob(p)})
    if not files:
        st.info("Aucune sortie d'audit trouvée sous docs/ ou cache/.")
        return

    rel_files = [os.path.relpath(f, REPO_ROOT) for f in files]
    chosen = st.selectbox("Fichier de résultat", rel_files)
    data = load_json(os.path.join(REPO_ROOT, chosen)) or {}

    summary = data.get("summary", data)
    if isinstance(summary, dict):
        flat = {k: v for k, v in summary.items() if not isinstance(v, (dict, list))}
        if flat:
            st.subheader("Résumé")
            st.dataframe(pd.DataFrame([flat]).T.rename(columns={0: "valeur"}), width='stretch')
    with st.expander("JSON complet"):
        st.json(data)


def page_clustering_correlations(run_dir: str) -> None:
    st.header("Clustering & corrélations (App F.1, E.1/E.3)")
    st.caption(
        "Regroupement de features autour d'une requête métier et paires de concepts qui "
        "co-occurrent -- tous deux vérifiés par relabellisation LLM indépendante sur un corpus "
        "frais plutôt que lus sur la seule structure SAE brute. "
        "cf. src/analysis/clustering_llm.py, src/analysis/correlations_verified.py, "
        "RESULTS_TESTS.md §85/§86."
    )

    st.subheader("Clustering ciblé (mots-clés LLM → union top-k → Jaccard → labels LLM)")
    clustering = load_json(os.path.join(REPO_ROOT, run_dir, "cache", "clustering_llm_verification.json"))
    if clustering:
        st.write(f"Requête : *{clustering['axis_query']}* — mots-clés générés : "
                  + ", ".join(clustering["keywords"]))
        rows = []
        for cid, desc in clustering["cluster_descriptions"].items():
            idx = int(cid)
            rows.append({
                "cluster": cid,
                "taille": clustering["cluster_sizes"][idx] if idx < len(clustering["cluster_sizes"]) else None,
                "description (LLM)": desc,
                "accuracy": clustering["accuracy"].get(cid),
                "z-conductance": clustering["conductance_zscore"].get(cid),
            })
        st.dataframe(pd.DataFrame(rows), width='stretch')
        st.caption(
            "Z-conductance négatif = cluster plus compact en espace d'embedding dense qu'un "
            "échantillon aléatoire de même taille (structure réelle, pas un artefact de hasard). "
            "L'accuracy varie fortement d'un cluster à l'autre -- comportement attendu (cohérent "
            "avec le papier de référence), pas un signe de bug."
        )
    else:
        st.info("clustering_llm_verification.json absent de ce run (lancer scripts/clustering_llm_test.py).")

    st.divider()
    st.subheader("Corrélations vérifiées (NPMI_verified)")
    npmi = load_json(os.path.join(REPO_ROOT, run_dir, "cache", "npmi_verified.json"))
    if npmi and npmi.get("pairs"):
        rows = [{"feature i": p["label_i"], "feature j": p["label_j"], "NPMI (SAE)": p["npmi_sae"],
                 "NPMI vérifié": p["npmi_verified"], "co-occurrence vérifiée": p["co_verified"],
                 "documents testés": p["n_docs"]} for p in npmi["pairs"]]
        st.dataframe(pd.DataFrame(rows), width='stretch')
        st.warning(
            "Effectif faible pour la plupart des paires (souvent 1-2 documents positifs sur "
            f"{npmi.get('n_verify_docs', '?')}) -- un NPMI vérifié parfait à si peu d'exemples "
            "n'est pas une preuve robuste, cf. RESULTS_TESTS.md §85."
        )
    else:
        st.info("npmi_verified.json absent de ce run (lancer scripts/npmi_verified_test.py).")


# Run de référence du rapport de stage (R0 -- RESULTS_TESTS.md §97/§119) : 12B, layer 31,
# K_extra=5, D_extra=1024, 25M tokens, sélection stratifiée, juge Qwen3.8-27B, négatif
# corrigé (§115-117), n=300. C'est le SEUL chiffre à citer comme taux d'interprétabilité
# final du dépôt -- 65,7% (197/300) au moment de la campagne de rejugement §119.
_REFERENCE_RUN = "results_v27_ablation_classic_setup_k5_25m_layer31"

# Balayages échelle/layer -- méthodologie finale (sélection stratifiée + juge Qwen3.8-27B +
# déduplication par mail parent), seuls points comparables entre eux à variable unique isolée.
# (nom de répertoire, taille en Md de paramètres ou numéro de layer pour l'axe des graphes)
_MODEL_SCALE_SWEEP = [
    ("1B", "results_v29_ablation_classic_setup_k5_25m_model_scale_1b", 1),
    ("4B", "results_v30_ablation_classic_setup_k5_25m_model_scale_4b", 4),
    ("12B (référence)", _REFERENCE_RUN, 12),
    ("27B", "results_v31_ablation_classic_setup_k5_25m_model_scale_27b", 27),
]
_LAYER_SWEEP = [
    ("Layer 12", "results_v32_ablation_classic_setup_k5_25m_layer12", 12),
    ("Layer 24", "results_v34_ablation_classic_setup_k5_25m_layer24", 24),
    ("Layer 31 (référence)", _REFERENCE_RUN, 31),
    ("Layer 41", "results_v33_ablation_classic_setup_k5_25m_layer41", 41),
]


def _load_flat_judge_rate(save_dir: str) -> dict | None:
    """Charge le taux d'interprétabilité + diagnostics (juge, méthode, n, biais de négatif)
    directement depuis le cache plat p1_judge_labels_extended.json -- PAS via
    _judge_label_sources : ce balayage compare des points à variable unique isolée, mélanger
    avec une source de comparaison (b1/b2/judge_separation) casserait l'appariement d'une
    ablation à l'autre."""
    path = os.path.join(REPO_ROOT, save_dir, "cache", "p1_judge_labels_extended.json")
    labels = load_json(path)
    if not labels:
        return None
    meta = load_json(path + ".meta.json") or {}
    n = len(labels)
    succ = sum(1 for v in labels.values() if v.get("interp_score") == 1)
    diag = _negative_bias_diagnostic(labels)
    return {
        "n": n, "succ": succ, "rate": succ / n,
        "judge": _judge_name(meta.get("judge_model_id", "?")),
        "method": meta.get("feature_selection_method", "?"),
        "neg_bias_frac": diag["frac_le_3_words"] if diag else None,
    }


def _sweep_table(sweep: list[tuple[str, str, int]], ref_idx: int) -> pd.DataFrame:
    ref = _load_flat_judge_rate(sweep[ref_idx][1])
    rows = []
    for label, save_dir, _x in sweep:
        r = _load_flat_judge_rate(save_dir)
        if r is None:
            rows.append({"point": label, "n": None, "taux": "en cours de rejugement / absent",
                         "IC95%": None, "juge": None, "sélection": None,
                         "négatifs ≤3 mots": None, "vs référence (p)": None})
            continue
        ci = proportion_with_ci(r["succ"], r["n"])
        row = {
            "point": label, "n": r["n"], "taux": f"{100*r['rate']:.1f}%",
            "IC95%": f"[{100*ci.ci_low:.1f} ; {100*ci.ci_high:.1f}]",
            "juge": r["judge"], "sélection": r["method"],
            "négatifs ≤3 mots": f"{100*r['neg_bias_frac']:.0f}%" if r["neg_bias_frac"] is not None else "?",
        }
        if ref is not None and save_dir != sweep[ref_idx][1]:
            t = two_proportion_test(r["succ"], r["n"], ref["succ"], ref["n"])
            row["vs référence (p)"] = f"{t.p:.3f}" + (" *" if t.p < 0.05 else "")
        else:
            row["vs référence (p)"] = "—"
        rows.append(row)
    return pd.DataFrame(rows)


def page_sweeps() -> None:
    st.header("Balayages échelle du modèle & layer (méthodologie finale)")
    st.caption(
        "Sélection stratifiée par fréquence + juge Qwen3.8-27B + déduplication par mail parent -- "
        "seule méthodologie retenue comme comparable d'un point à l'autre de ces balayages. "
        "Indépendant du run sélectionné dans la barre latérale (compare directement les "
        "répertoires de la campagne de mesure finale)."
    )

    ref = _load_flat_judge_rate(_REFERENCE_RUN)
    if ref is not None and ref["neg_bias_frac"] is not None and ref["neg_bias_frac"] > 0.3:
        st.error(
            "⚠️ Le négatif odd-one-out de la configuration de référence (12B/layer 31) est encore "
            f"majoritairement trivial ({100*ref['neg_bias_frac']:.0f}% de négatifs ≤3 mots) -- le "
            "juge peut résoudre une bonne partie des tâches par longueur de contexte plutôt que par "
            "concept partagé (RESULTS_TESTS.md §115/§117). TOUS les taux de cette page en héritent "
            "et sont probablement surestimés dans une proportion comparable ; ne pas les citer "
            "comme valeur finale sans vérifier l'état du correctif sur le point concerné."
        )

    all_ns = {r["n"] for _, d, _ in _MODEL_SCALE_SWEEP + _LAYER_SWEEP
              if (r := _load_flat_judge_rate(d)) is not None}
    if len(all_ns) > 1:
        st.warning(
            f"Tailles d'échantillon hétérogènes entre points de ces balayages ({sorted(all_ns)}) -- "
            "une campagne de rejugement est probablement encore en cours sur une partie des points ; "
            "comparer les taux bruts avec prudence tant que n diffère."
        )

    st.subheader("Échelle du modèle extracteur/juge (1B → 27B)")
    st.dataframe(_sweep_table(_MODEL_SCALE_SWEEP, ref_idx=2), width='stretch')
    plot_points = []
    for label, save_dir, x in _MODEL_SCALE_SWEEP:
        r = _load_flat_judge_rate(save_dir)
        if r is not None:
            ci = proportion_with_ci(r["succ"], r["n"])
            plot_points.append({"label": label, "x": x, "rate_pct": 100 * r["rate"],
                                 "err_low": 100 * (r["rate"] - ci.ci_low),
                                 "err_high": 100 * (ci.ci_high - r["rate"])})
    if len(plot_points) >= 2:
        pdf = pd.DataFrame(plot_points)
        fig = px.scatter(pdf, x="x", y="rate_pct", error_y="err_high", error_y_minus="err_low",
                          text="label", log_x=True, height=400,
                          labels={"x": "Taille du modèle (Md de paramètres)",
                                  "rate_pct": "Taux d'interprétabilité (%)"})
        fig.update_traces(mode="lines+markers+text", textposition="top center")
        st.plotly_chart(fig, width='stretch')
    st.caption(
        "Tendance de Cochran-Armitage sur les points sous méthodologie totalement homogène : non "
        "significative (RESULTS_TESTS.md §97/§112) -- l'effet d'échelle historiquement cité (~33 "
        "points d'écart 1B/12B) était pour bonne partie un artefact de sélection par magnitude + "
        "auto-jugement, pas un effet réel de cette ampleur."
    )

    st.subheader("Layer d'extraction (12 → 41)")
    st.dataframe(_sweep_table(_LAYER_SWEEP, ref_idx=2), width='stretch')
    st.caption(
        "Aucun des 4 layers testés ne se distingue significativement de layer 31 sous cette "
        "méthodologie (RESULTS_TESTS.md §96/§109/§113) -- l'écart layer 31 vs 24 historiquement "
        "cité (§51) ne réplique sur aucune paire une fois la sélection stratifiée et le juge Qwen "
        "appliqués aux deux bras."
    )

    st.divider()
    st.subheader("Plancher de bruit run-à-run (même configuration, seed différente)")
    seed_runs = [("Référence (seed 42)", _REFERENCE_RUN),
                 ("V1 (seed 123)", "results_v38_ablation_v1_seed123_classic_setup_k5_25m_layer31"),
                 ("V2 (seed 7)", "results_v39_ablation_v2_seed7_classic_setup_k5_25m_layer31")]
    seed_rows = []
    for label, save_dir in seed_runs:
        r = _load_flat_judge_rate(save_dir)
        if r is not None:
            ci = proportion_with_ci(r["succ"], r["n"])
            seed_rows.append({"run": label, "n": r["n"], "taux": f"{100*r['rate']:.1f}%",
                               "IC95%": f"[{100*ci.ci_low:.1f} ; {100*ci.ci_high:.1f}]"})
    if seed_rows:
        st.dataframe(pd.DataFrame(seed_rows), width='stretch')
        st.caption(
            "Un écart isolé contre la référence ne peut être lu comme un effet d'hyperparamètre "
            "que s'il dépasse cette fourchette de variabilité intrinsèque à l'entraînement du SAE "
            "seul (init + ordre de mélange, aucun hyperparamètre changé), cf. RESULTS_TESTS.md §101."
        )
    else:
        st.info("Runs de variabilité de seed (V1/V2) absents.")

    st.divider()
    st.subheader("Sanity checks — décodeur figé aléatoire (Korznikov et al. 2026)")
    sanity_runs = [("R0 — décodeur entraîné", _REFERENCE_RUN),
                   ("C1b — décodeur figé, init cov", "results_v42_ablation_c1b_sanity_frozen_decoder_cov_init"),
                   ("C1 — décodeur figé, init iso", "results_v37_ablation_c1_sanity_frozen_decoder_stratified_qwen")]
    sanity_rows = []
    for label, save_dir in sanity_runs:
        res = load_json(os.path.join(REPO_ROOT, save_dir, "results.json"))
        p1 = (res or {}).get("P1_Gemma3_SAE", {})
        fve_pre, fve_ext = p1.get("fve_pretrained"), p1.get("fve_extended")
        if p1 and fve_pre is not None and fve_ext is not None:
            sanity_rows.append({
                "configuration": label,
                "dead_pct_extension": f"{p1.get('dead_pct_extension', float('nan')):.1f}%",
                "ΔFVE": f"+{fve_ext - fve_pre:.4f}",
            })
    if sanity_rows:
        st.dataframe(pd.DataFrame(sanity_rows), width='stretch')
        st.caption(
            "Le SAE entraîné explique nettement plus de variance supplémentaire qu'un décodeur "
            "figé à une initialisation aléatoire, même sous le schéma d'initialisation le plus dur "
            "à battre (cov, Korznikov et al.) -- ΔFVE est la métrique qui tranche ici, pas le taux "
            "d'interprétabilité (n effectif réduit à 11-31 sous décodeur figé, la quasi-totalité de "
            "l'extension restant morte). cf. RESULTS_TESTS.md §98/§105."
        )
    else:
        st.info("Runs de sanity check décodeur figé (C1/C1b) absents.")

    st.divider()
    st.subheader("Pipeline 1 vs Pipeline 2 — taux d'interprétabilité, même méthodologie")
    p1_ref = _load_flat_judge_rate(_REFERENCE_RUN)
    p2_sources = _judge_label_sources("results_v10_emails_main", "p2")
    cols = st.columns(2)
    with cols[0]:
        if p1_ref is not None:
            st.metric("Pipeline 1 (Gemma-3 + GemmaScope, référence)", f"{100*p1_ref['rate']:.1f}%",
                       help=f"{p1_ref['succ']}/{p1_ref['n']} — juge {p1_ref['judge']}, "
                            f"sélection {p1_ref['method']}")
        else:
            st.info("Référence Pipeline 1 absente.")
    with cols[1]:
        if p2_sources:
            chosen_p2_key = next(iter(p2_sources))
            p2_labels = p2_sources[chosen_p2_key]
            n_p2 = len(p2_labels)
            succ_p2 = sum(1 for v in p2_labels.values() if v.get("interp_score") == 1)
            st.metric("Pipeline 2 (F2LLM + PhraseLevelSAE)", f"{100*succ_p2/n_p2:.1f}%",
                       help=f"{succ_p2}/{n_p2} — source : {chosen_p2_key}")
        else:
            st.info("Labels Pipeline 2 absents.")
    st.caption(
        "Pipeline 2 n'a jamais montré le biais de négatif de §115/§117 (RESULTS_TESTS.md §118) -- "
        "son chiffre n'est pas affecté par le correctif qui touche Pipeline 1 ci-dessus."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

# Runs post-soutenance identifiés comme antérieurs au correctif de filiation des
# emails parents (docs/RESULTS_STATUS.md, section Corpus).
PRE_PARENT_FIX_RUNS = {
    "results_post_stage_e00_profile_1b", "results_post_stage_e00_profile_1b_v2",
    "results_post_stage_e01_fit_1b_layer13_k5",
    "results_post_stage_e05_seed43", "results_post_stage_e05_seed44",
    "results_post_stage_e05_randinit45", "results_post_stage_e05_randinit46",
}


def show_run_status_warning(run_dir: str) -> None:
    if run_dir in PRE_PARENT_FIX_RUNS:
        st.warning(
            "Statut : résultats antérieurs au correctif de filiation des emails parents. La "
            "séparation FIT/DEV/CONFIRM n'était pas effective pour les variantes ; consultable "
            "comme historique, pas une validation hors apprentissage. Rejeu nécessaire. "
            "Évaluations humaines non réalisées. Détail : docs/RESULTS_STATUS.md."
        )
    elif run_dir.startswith("results_post_stage_"):
        st.warning(
            "Run post-soutenance dont le statut n'est pas validé par ce dashboard (rejeu "
            "éventuellement partiel) : vérifier commit, manifeste, checkpoint FIT et statut de "
            "fin des jobs avant de citer un chiffre (docs/RESULTS_STATUS.md)."
        )


def main() -> None:
    st.set_page_config(page_title="SAE EDF — Dashboard", layout="wide")
    st.title("Analyse interprétable de mails clients EDF via SAE")
    st.caption("Lecture d'artefacts déjà produits sur disque uniquement — aucun modèle chargé, aucun GPU requis.")

    run_dirs = discover_result_dirs()
    if not run_dirs:
        st.error(f"Aucun dossier results_*/ trouvé sous {REPO_ROOT}.")
        return
    default_idx = run_dirs.index("results_v10_emails_main") if "results_v10_emails_main" in run_dirs else 0
    run_dir = st.sidebar.selectbox("Run", run_dirs, index=default_idx)
    show_run_status_warning(run_dir)

    page = st.sidebar.radio(
        "Page",
        ["Vue d'ensemble", "UMAP", "Features", "Diagnostics d'entraînement", "Diffing",
         "Recherche", "Urgence/Robustesse", "Explication (fidélité/plausibilité)",
         "Clustering & Corrélations", "Sweeps (échelle & layer)", "Rapport consolidé",
         "Comparaison mail original / augmenté", "Mini-pilote analyste (E08)",
         "Associations de propriétés (E06)", "Regrouper selon une question (E07)",
         "Stabilité inter-graines (E05)",
         "Audit méthodologique (archive)"],
    )

    if page == "Vue d'ensemble":
        page_overview(run_dir)
    elif page == "UMAP":
        page_umap(run_dir)
    elif page == "Features":
        page_features(run_dir)
    elif page == "Diagnostics d'entraînement":
        page_diagnostics(run_dir)
    elif page == "Explication (fidélité/plausibilité)":
        page_explanation_quality(run_dir)
    elif page == "Clustering & Corrélations":
        page_clustering_correlations(run_dir)
    elif page == "Sweeps (échelle & layer)":
        page_sweeps()
    elif page == "Rapport consolidé":
        page_consolidated_report(run_dir)
    elif page == "Diffing":
        page_diffing(run_dir)
    elif page == "Recherche":
        page_search(run_dir)
    elif page == "Urgence/Robustesse":
        page_urgence_robustesse(run_dir)
    elif page == "Comparaison mail original / augmenté":
        page_email_comparison()
    elif page == "Mini-pilote analyste (E08)":
        page_pilot_e08(run_dir)
    elif page == "Associations de propriétés (E06)":
        page_correlations_e06(run_dir)
    elif page == "Regrouper selon une question (E07)":
        page_clustering_e07(run_dir)
    elif page == "Stabilité inter-graines (E05)":
        page_stability_e05(run_dir)
    elif page == "Audit méthodologique (archive)":
        page_audit_2026_08()


if __name__ == "__main__":
    main()
