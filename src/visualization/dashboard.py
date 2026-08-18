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

import pandas as pd
import plotly.express as px
import streamlit as st

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


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


# ─────────────────────────────────────────────────────────────────────────────
# Pages
# ─────────────────────────────────────────────────────────────────────────────

def page_overview(run_dir: str) -> None:
    st.header("Vue d'ensemble du run")
    results = load_json(os.path.join(REPO_ROOT, run_dir, "results.json"))
    if not results:
        st.warning("Pas de results.json dans ce run (run partiel ou script encore en cours).")
        return
    for pipeline_key, title in [("P1_Gemma3_SAE", "Pipeline 1 — Gemma-3 + GemmaScope"),
                                 ("P2_F2LLM_PhSAE", "Pipeline 2 — F2LLM + PhraseLevelSAE")]:
        metrics = results.get(pipeline_key)
        if not metrics:
            continue
        st.subheader(title)
        # diff_hypothesis (texte libre généré par LLM) affiché séparément ci-dessous
        # (st.caption) -- l'exclure ici évite une colonne à types mixtes (float/str)
        # que pyarrow ne peut pas convertir proprement pour le rendu du tableau.
        display = {k: v for k, v in metrics.items()
                   if not isinstance(v, (dict, list)) and k != "diff_hypothesis"}
        st.dataframe(pd.DataFrame([display]).T.rename(columns={0: "valeur"}), width='stretch')
        if metrics.get("diff_hypothesis"):
            st.caption(f"Hypothèse LLM (diffing cross-domaine) : {metrics['diff_hypothesis']}")


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
        ext = load_json(os.path.join(REPO_ROOT, run_dir, "cache", "p1_judge_labels_extended.json"))
        if ext:
            n_interp = sum(1 for v in ext.values() if v.get("interp_score") == 1)
            st.metric("Taux d'interprétabilité (odd-one-out)", f"{100*n_interp/len(ext):.1f}%",
                       help=f"{n_interp}/{len(ext)} features passent le test.")
            _feature_search_box(ext, key="ext")
        else:
            st.info("cache/p1_judge_labels_extended.json absent de ce run.")

    with tab_p2:
        p2 = load_json(os.path.join(REPO_ROOT, run_dir, "cache", "p2_feature_labels.json"))
        if p2:
            _feature_search_box(p2, key="p2")
        else:
            st.info("cache/p2_feature_labels.json absent de ce run.")


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
        return
    rel_files = [os.path.relpath(f, REPO_ROOT) for f in csv_files]
    chosen = st.selectbox("Fichier de diff", rel_files)
    df = pd.read_csv(os.path.join(REPO_ROOT, chosen))
    n_sig = int(df["significant"].sum()) if "significant" in df.columns else None
    if n_sig is not None:
        st.metric("Features significatives (q<0.05)", f"{n_sig}/{len(df)}")
    st.dataframe(df.head(50), width='stretch')


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
        return
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


def page_urgence_robustesse(run_dir: str) -> None:
    st.header("Détection d'urgence/intention & robustesse du juge")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Sonde intention/urgence (mails originaux)")
        d = load_json(os.path.join(REPO_ROOT, run_dir, "cache", "intent_urgency_probe_results.json"))
        if d:
            rows = [{"intention": k, **v} for k, v in d.items()]
            df = pd.DataFrame(rows)
            df["delta"] = df["acc_sae"] - df["majority_baseline"]
            st.dataframe(df, width='stretch')
            st.caption("cf. scripts/intent_urgency_probe.py, RESULTS_TESTS.md §13.2")
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

    st.subheader("Balayages d'hyperparamètres (toutes runs confondues)")
    sweep_dir = os.path.join(REPO_ROOT, "results_diagnostics", "plots")
    sweep_files = sorted(glob.glob(os.path.join(sweep_dir, "*.html")))
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
    """Agrège les sorties JSON produites par l'audit 2026-08 (docs/AUDIT_2026-08.md) --
    jusqu'ici dispersées sous docs/ et cache/, lisibles seulement en ouvrant chaque
    fichier à la main (audit round 3, §6.4). Recherche par motif plutôt que liste en
    dur : reste à jour sans édition à chaque nouveau script d'audit."""
    st.header("Audit 2026-08 — validité des résultats")
    st.caption("cf. `docs/AUDIT_2026-08.md` (constats détaillés) et `RESULTS_TESTS.md` §57-62. "
               "Indépendant du run sélectionné dans la barre latérale.")

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


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

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

    page = st.sidebar.radio(
        "Page",
        ["Vue d'ensemble", "UMAP", "Features", "Diagnostics d'entraînement", "Diffing",
         "Recherche", "Urgence/Robustesse", "Explication (fidélité/plausibilité)",
         "Rapport consolidé", "Comparaison mail original / augmenté", "Audit 2026-08"],
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
    elif page == "Audit 2026-08":
        page_audit_2026_08()


if __name__ == "__main__":
    main()
