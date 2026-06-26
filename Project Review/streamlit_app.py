"""
============================================================
  streamlit_app.py  —  FDS DA2
  Academic Document Intelligence Platform
============================================================
"""

import os, warnings, math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.pipeline           import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model       import LogisticRegression
from sklearn.svm                import LinearSVC
from sklearn.model_selection    import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics            import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition      import TruncatedSVD

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
FULL_CSV  = os.path.join(BASE_DIR, "academic_documents_full.csv")
ORIG_CSV  = os.path.join(BASE_DIR, "academic_documents_with_extracted_text.csv")
DATA_PATH = FULL_CSV if os.path.exists(FULL_CSV) else ORIG_CSV

ACCENT   = "#ffffff"
ACCENT2  = "#a1a1aa"
PALETTE  = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899", "#14b8a6", "#f97316", "#06b6d4", "#84cc16"]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Course & Module Classifier",
    page_icon=":material/psychology:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS — dark glassmorphism theme
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400..900;1,400..900&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
<style>
.material-icons { font-family: 'Material Icons' !important; vertical-align: middle; }
.playfair-display-main { font-family: "Playfair Display", serif; font-optical-sizing: auto; font-weight: 400; font-style: normal; }
html, body, [class*="css"], .stApp, .stMarkdown, .stHeader, .stTitle, .stSubheader, .stCaption, .stWidgetLabel, .stSelectbox, .stTextInput, .stTextArea, .stButton, .stSidebar, [data-testid="stMetricValue"], [data-testid="stMetricLabel"], [data-testid="stText"] {
    font-family: 'Playfair Display', serif !important;
}
.stApp { background: #09090b; color: #fafafa; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }
.metric-card { background: #18181b; border: 1px solid #27272a; border-radius: 16px; padding: 24px 20px; text-align: center; transition: transform 0.2s, border-color 0.2s; }
.metric-card:hover { transform: translateY(-3px); border-color: #52525b; }
.metric-val { font-size: 2.6rem; font-weight: 800; line-height: 1.1; color: #ffffff; }
.metric-lbl { font-size: 0.82rem; color: #a1a1aa; margin-top: 4px; font-weight: 500; letter-spacing: 0.05em; text-transform: uppercase; }
.glass { background: #18181b; border: 1px solid #27272a; border-radius: 16px; padding: 24px; margin-bottom: 16px; }
.section-header { font-size: 1.5rem; font-weight: 700; color: #ffffff; margin-bottom: 4px; border-bottom: 2px solid #ffffff; display: inline-block; padding-right: 20px; }
.section-sub { font-size: 0.9rem; color: #a1a1aa; margin-bottom: 20px; }
.hero-title { font-size: 3.2rem; font-weight: 900; line-height: 1.15; color: #ffffff; letter-spacing: -0.02em; }
.hero-sub { font-size: 1.1rem; color: #a1a1aa; margin-top: 8px; }
.badge { display: inline-block; padding: 4px 12px; border-radius: 999px; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.05em; background: #27272a; color: #ffffff; border: 1px solid #3f3f46; margin: 2px; }
.pred-box { background: #18181b; border: 2px solid #ffffff; border-radius: 16px; padding: 28px; text-align: center; }
.pred-course { font-size: 2.2rem; font-weight: 900; color: #ffffff; }
.pred-label { font-size: 0.9rem; color: #a1a1aa; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 8px; }
.stat-pill { background: #27272a; border-radius: 8px; padding: 8px 16px; display: inline-block; font-size: 0.85rem; color: #a1a1aa; margin: 4px; }
.stat-pill span { color: #ffffff; font-weight: 700; }
.dataframe { background: #18181b !important; border: 1px solid #27272a !important; color: #ffffff !important;}
.stTabs [data-baseweb="tab-list"] { background: transparent; border-radius: 12px; padding: 4px; gap: 12px; }
.stTabs [data-baseweb="tab"] { border-radius: 10px !important; color: #a1a1aa !important; font-weight: 500 !important; padding: 8px 24px !important; transition: all 0.2s !important; border: none !important; margin: 0 4px !important; }
.stTabs [aria-selected="true"] { background: #ffffff !important; color: #09090b !important; font-weight: 800 !important; }
.stTabs [data-baseweb="tab"]:hover { color: #ffffff !important; }
.stTabs [aria-selected="true"]:hover { color: #000000 !important; }

.stButton > button { background: #ffffff !important; color: #09090b !important; font-weight: 700 !important; border: none !important; border-radius: 10px !important; padding: 10px 24px !important; letter-spacing: 0.03em !important; transition: background 0.2s !important; }
.stButton > button:hover { background: #f4f4f5 !important; }
.stTextArea textarea { background: #18181b !important; border: 1px solid #27272a !important; border-radius: 12px !important; color: #fafafa !important; font-family: 'Playfair Display', serif !important; }
.stTextArea textarea:focus { border-color: #ffffff !important; }
.stInfo, .stSuccess, .stWarning { border-radius: 12px !important; background: #18181b !important; border-left-color: #ffffff !important; color: #fafafa !important; }
hr { border-color: #27272a !important; }
[data-baseweb="tab-highlight"] { display: none !important; }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #3f3f46; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── Plotly Dark template ───────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Playfair Display", color="#ffffff"),
    margin=dict(t=50, b=40, l=40, r=40),
)

def dark_fig(fig, height=420):
    fig.update_layout(**PLOTLY_LAYOUT, height=height)
    fig.update_xaxes(gridcolor="#27272a", zerolinecolor="#3f3f46")
    fig.update_yaxes(gridcolor="#27272a", zerolinecolor="#3f3f46")
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# DATA & MODEL  (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading corpus …")
def load_data():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    def fc(df, cands):
        for c in cands:
            if c in df.columns: return c
        return None
    CC = fc(df, ["content","extracted_text","text","body"])
    CS = fc(df, ["course_label","course","subject","label"])
    CM = fc(df, ["module_label","module","topic"])
    CD = fc(df, ["doc_type","document_type","type"])
    CF = fc(df, ["file_format","format","extension"])
    df[CC] = df[CC].fillna("").astype(str).str.strip()
    df[CS] = df[CS].fillna("unknown").astype(str).str.strip()
    df["word_count"] = df[CC].str.split().str.len()
    df_clean = df[df["word_count"] >= 30].copy()
    return df_clean, CC, CS, CM, CD, CF

@st.cache_resource(show_spinner="Training AI models …")
def train_pipeline(_n):
    df_clean, CC, CS, CM, CD, CF = load_data()
    vc    = df_clean[CS].value_counts()
    valid = vc[vc >= 5].index.tolist()
    df_m  = df_clean[df_clean[CS].isin(valid)].copy()
    X = df_m[CC]; y = df_m[CS]

    args = dict(max_features=10000, stop_words="english",
                ngram_range=(1,2), sublinear_tf=True, min_df=2)
    pipes = {
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(**args)),
            ("clf",   LogisticRegression(max_iter=2000, C=1.0,
                                          class_weight="balanced", solver="lbfgs"))
        ]),
        "Linear SVM": Pipeline([
            ("tfidf", TfidfVectorizer(**args)),
            ("clf",   LinearSVC(max_iter=3000, C=1.0, class_weight="balanced"))
        ]),
    }
    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    sc  = ["accuracy","f1_macro","f1_weighted","precision_macro","recall_macro"]
    cvr = {}
    for name, pipe in pipes.items():
        s = cross_validate(pipe, X, y, cv=cv, scoring=sc, n_jobs=-1)
        cvr[name] = {
            "Accuracy"    : round(s["test_accuracy"].mean(), 4),
            "F1 (macro)"  : round(s["test_f1_macro"].mean(), 4),
            "F1 (weighted)":round(s["test_f1_weighted"].mean(), 4),
            "Precision"   : round(s["test_precision_macro"].mean(), 4),
            "Recall"      : round(s["test_recall_macro"].mean(), 4),
        }
    res = pd.DataFrame(cvr).T
    best_name = res["Accuracy"].idxmax()
    best_pipe = pipes[best_name]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    best_pipe.fit(Xtr, ytr)

    # second LR for predict_proba
    lr_pipe = pipes["Logistic Regression"]
    if best_name != "Logistic Regression":
        lr_pipe.fit(Xtr, ytr)

    yte_pred = best_pipe.predict(Xte)

    # ── Module classifier (LR with predict_proba) ──────────────────────────
    mod_pipe = None
    mod_acc  = None
    if CM and CM in df_m.columns:
        df_mod = df_m[df_m[CM].notna() & (df_m[CM].str.strip() != "") &
                      (df_m[CM].str.lower() != "unknown")].copy()
        mvc = df_mod[CM].value_counts()
        valid_mods = mvc[mvc >= 3].index.tolist()
        df_mod = df_mod[df_mod[CM].isin(valid_mods)]
        if len(df_mod) >= 20 and df_mod[CM].nunique() >= 2:
            Xm = df_mod[CC]; ym = df_mod[CM]
            mod_pipe = Pipeline([
                ("tfidf", TfidfVectorizer(**args)),
                ("clf",   LogisticRegression(max_iter=2000, C=1.0,
                                             class_weight="balanced", solver="lbfgs"))
            ])
            Xmtr, Xmte, ymtr, ymte = train_test_split(
                Xm, ym, test_size=0.2, stratify=ym, random_state=42)
            mod_pipe.fit(Xmtr, ymtr)
            mod_acc = round((mod_pipe.predict(Xmte) == ymte).mean(), 4)

    # PCA
    tpca = TfidfVectorizer(**args)
    Xpca = tpca.fit_transform(df_m[CC])
    svd  = TruncatedSVD(n_components=3, random_state=42)
    X3d  = svd.fit_transform(Xpca)

    return dict(pipes=pipes, best_name=best_name, best_pipe=best_pipe,
                lr_pipe=lr_pipe, res=res,
                Xte=Xte, yte=yte, yte_pred=yte_pred,
                X3d=X3d, df_m=df_m, valid=valid,
                svd=svd, CC=CC, CS=CS, CD=CD,
                mod_pipe=mod_pipe, mod_acc=mod_acc, CM=CM)

# ── Load ──────────────────────────────────────────────────────────────────────
try:
    df_clean, CC, CS, CM, CD, CF = load_data()
    md = train_pipeline(len(df_clean))
except FileNotFoundError:
    st.error("❌ Dataset CSV not found in the project folder.")
    st.stop()

courses   = df_clean[CS].value_counts().index.tolist()
cpal      = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(courses)}

# ── Navigation ────────────────────────────────────────────────────────────────
page = st.radio("Navigation", [
    ":material/dashboard:  Dashboard",
    ":material/query_stats:  EDA Explorer",
    ":material/smart_toy:  Model Lab",
    ":material/target:  Live Predictor",
    ":material/explore:  Document Universe",
], index=0, horizontal=True, label_visibility="collapsed")


# 🏠  DASHBOARD
if page == ":material/dashboard:  Dashboard":

    st.markdown("""
    <div class="hero-title">Course & Module Classifier</div>
    <div class="hero-sub" style="margin-bottom: 40px;">
      NLP-powered document classification for academic courses and modules
    </div>
    """, unsafe_allow_html=True)

    # Get metrics
    best_acc = md["res"]["Accuracy"].max()
    best_f1  = md["res"]["F1 (macro)"].max()

    # KPI row
    kpis = [
        (f"{len(df_clean):,}", "Total Documents"),
        (f"{df_clean[CS].nunique()}", "Unique Courses"),
        (f"{best_acc:.1%}", "Best CV Accuracy"),
        (f"{md['res']['F1 (macro)'].max():.3f}", "Best F1 Macro"),
        (f"{df_clean['word_count'].mean():.0f}", "Avg Word Count"),
        (f"{df_clean[CS].value_counts().max()}", "Largest Class"),
    ]
    cols = st.columns(6)
    for col, (v, l) in zip(cols, kpis):
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-val">{v}</div>
              <div class="metric-lbl">{l}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Three-column overview: Course Distribution | Project Architecture | Pie Chart
    col1, col2, col3 = st.columns([1.4, 1, 0.7])

    with col1:
        st.markdown('<div class="section-header">Course Distribution</div>', unsafe_allow_html=True)
        cc_df = df_clean[CS].value_counts().reset_index()
        cc_df.columns = ["Course", "Count"]
        fig = px.bar(cc_df.sort_values("Count"),
                     x="Count", y="Course", orientation="h",
                     color="Count", color_continuous_scale=PALETTE,
                     text="Count")
        fig.update_traces(textposition="outside", textfont_color="#e2e8f0",
                          marker_line_width=0)
        fig.update_coloraxes(showscale=False)
        dark_fig(fig, 480).update_layout(yaxis_title="", xaxis_title="Documents")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header" style="margin-bottom: 20px;">Project Architecture</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="glass">
          <div style="font-size:0.9rem;line-height:2;color:#cbd5e1;">

          <span style="color:#3b82f6;font-weight:700;">01 · Data Pipeline</span><br>
          &nbsp;&nbsp;Automated extraction from 1,300+ PDFs & PPTX<br>
          <span style="color:#10b981;font-weight:700;">02 · Preprocessing</span><br>
          &nbsp;&nbsp;TF-IDF · bigrams · class balancing · feature engineering<br>
          <span style="color:#f59e0b;font-weight:700;">03 · Model Comparison</span><br>
          &nbsp;&nbsp;LR · SVM · RF · Naïve Bayes · 5-fold CV<br>
          <span style="color:#ef4444;font-weight:700;">04 · Evaluation</span><br>
          &nbsp;&nbsp;Accuracy · F1 · Precision · Recall · Confusion matrix<br>
          <span style="color:#8b5cf6;font-weight:700;">05 · Visualisation</span><br>
          &nbsp;&nbsp;14 charts · 3D PCA · keyword analysis · heatmaps<br>
          <span style="color:#ec4899;font-weight:700;">06 · Live Demo</span><br>
          &nbsp;&nbsp;Paste any text → get instant course prediction

          </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="section-header">Course Split</div>', unsafe_allow_html=True)
        fig2 = px.pie(cc_df, values="Count", names="Course",
                      color_discrete_sequence=PALETTE, hole=0.55)
        fig2.update_traces(textinfo="none",
                           hovertemplate="<b>%{label}</b><br>%{value} docs<extra></extra>")
        dark_fig(fig2, 480).update_layout(
            showlegend=False,
            annotations=[dict(text=f"<b>{len(df_clean)}</b><br><span style='font-size:10px'>docs</span>",
                               x=0.5, y=0.5, font_size=18, showarrow=False,
                               font_color="#ffffff")]
        )
        st.plotly_chart(fig2, use_container_width=True)


# 📊  EDA EXPLORER
elif page == ":material/query_stats:  EDA Explorer":

    st.markdown('<div class="section-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Uncover patterns, distributions, and relationships in the corpus</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        ":material/description:  Doc Types", ":material/folder:  File Formats", ":material/straighten:  Word Count", ":material/topic:  Module Map", ":material/link:  Keyword Overlap"
    ])

    with tab1:
        if CD and CD in df_clean.columns:
            ct   = pd.crosstab(df_clean[CS], df_clean[CD])
            melt = ct.reset_index().melt(id_vars=CS, var_name="Doc Type", value_name="Count")
            fig  = px.bar(melt, x=CS, y="Count", color="Doc Type",
                          barmode="stack", color_discrete_sequence=PALETTE,
                          title="Document Types per Course")
            dark_fig(fig, 480).update_layout(xaxis_tickangle=-40, bargap=0.25)
            st.plotly_chart(fig, use_container_width=True)

            # insight
            total = ct.sum(axis=1).sort_values(ascending=False)
            top_course = total.index[0]
            st.markdown(f"""
            <div class="glass" style="margin-top:8px; display:flex; align-items:center; gap:12px;">
              <span class="material-icons" style="color:#ffffff; font-size:1.2rem;">lightbulb</span>
              <span style="color:#94a3b8;font-size:0.88rem;"> —
              <b style="color:#e2e8f0;">{top_course}</b> has the most documents ({total.iloc[0]}).
              Courses with more slide-type files tend to be engineering-heavy topics.
              </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No doc_type column available.")

    with tab2:
        if CF and CF in df_clean.columns:
            ct2  = pd.crosstab(df_clean[CS], df_clean[CF])
            melt2= ct2.reset_index().melt(id_vars=CS, var_name="Format", value_name="Count")
            fig2 = px.bar(melt2, x=CS, y="Count", color="Format",
                          barmode="stack", color_discrete_sequence=PALETTE[5:],
                          title="File Formats per Course")
            dark_fig(fig2, 480).update_layout(xaxis_tickangle=-40, bargap=0.25)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No file_format column available.")

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            fig3 = px.box(df_clean, x=CS, y="word_count", color=CS,
                          color_discrete_sequence=PALETTE,
                          title="Word Count Distribution",
                          labels={"word_count":"Words", CS:"Course"})
            dark_fig(fig3, 440).update_layout(showlegend=False, xaxis_tickangle=-40)
            st.plotly_chart(fig3, use_container_width=True)
        with c2:
            fig4 = px.violin(df_clean, x=CS, y="word_count", color=CS,
                             color_discrete_sequence=PALETTE,
                             title="Word Count Violin",
                             box=True, points=False,
                             labels={"word_count":"Words", CS:"Course"})
            dark_fig(fig4, 440).update_layout(showlegend=False, xaxis_tickangle=-40)
            st.plotly_chart(fig4, use_container_width=True)

        stats = df_clean.groupby(CS)["word_count"].describe().round(1)
        st.dataframe(
            stats.style
                 .background_gradient(cmap="Blues", subset=["mean","50%","max"])
                 .format("{:.1f}"),
            use_container_width=True
        )

    with tab4:
        if CM and CM in df_clean.columns:
            df_clean[CM] = df_clean[CM].fillna("unknown").astype(str)
            pivot = pd.crosstab(df_clean[CS], df_clean[CM])
            fig5  = px.imshow(pivot, text_auto=True,
                              color_continuous_scale=PALETTE,
                              title="Documents per Course × Module", aspect="auto")
            dark_fig(fig5, max(500, len(pivot)*30)).update_layout(
                coloraxis_colorbar=dict(tickfont=dict(color="#94a3b8"))
            )
            st.plotly_chart(fig5, use_container_width=True)
            missing_mod = (df_clean[CM] == "unknown").sum()
            st.markdown(f"""
            <div class="glass" style="display:flex; align-items:center; gap:12px;">
              <span class="material-icons" style="color:#ffffff; font-size:1.2rem;">info</span>
              <span style="color:#94a3b8;font-size:0.88rem;"> —
              <b style="color:#e2e8f0;">{missing_mod}</b> documents have unknown module labels.
              The module classifier in project_da2.py predicts labels for these automatically.
              </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No module_label column.")

    with tab5:
        st.markdown('<div class="section-sub">Shared vocabulary (top-200 TF-IDF terms) between course pairs. High overlap → model may confuse them.</div>', unsafe_allow_html=True)
        with st.spinner("Computing keyword overlap …"):
            term_sets = {}
            for course in courses:
                sub = df_clean[df_clean[CS]==course][CC]
                if len(sub) < 2: continue
                vec = TfidfVectorizer(max_features=200, stop_words="english")
                vec.fit(sub)
                term_sets[course] = set(vec.get_feature_names_out())
            cl  = list(term_sets.keys())
            mat = np.zeros((len(cl), len(cl)), dtype=int)
            for i, c1 in enumerate(cl):
                for j, c2 in enumerate(cl):
                    mat[i,j] = len(term_sets[c1] & term_sets[c2])

            fig6 = px.imshow(mat, x=cl, y=cl, text_auto=True,
                             color_continuous_scale=PALETTE,
                             title="Cross-Course Keyword Overlap Heatmap")
            dark_fig(fig6, max(540, len(cl)*36)).update_layout(
                xaxis_tickangle=-40,
                coloraxis_colorbar=dict(tickfont=dict(color="#94a3b8"))
            )
            st.plotly_chart(fig6, use_container_width=True)


# 🤖  MODEL LAB
elif page == ":material/smart_toy:  Model Lab":

    st.markdown('<div class="section-header">Model Performance Lab</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">5-fold stratified cross-validation · confusion matrices · per-class metrics</div>', unsafe_allow_html=True)

    res       = md["res"]
    best_name = md["best_name"]
    yte       = md["yte"]
    yte_pred  = md["yte_pred"]
    best_pipe = md["best_pipe"]

    # champion banner
    st.markdown(f"""
    <div class="glass" style="border:1px solid #3f3f46; border-radius:14px; padding:18px 24px; margin-bottom:20px;">
      <span style="color:#a1a1aa; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.1em; display:flex; align-items:center; gap:8px;">
        <span class="material-icons" style="font-size:1rem;">military_tech</span> Champion Model
      </span><br>
      <span style="font-size:1.5rem;font-weight:800; color:#ffffff;">
        {best_name}
      </span>
      &nbsp;
      <span class="badge">Acc {res.loc[best_name,'Accuracy']:.1%}</span>
      <span class="badge">F1 {res.loc[best_name,'F1 (macro)']:.3f}</span>
      <span class="badge">Precision {res.loc[best_name,'Precision']:.3f}</span>
      <span class="badge">Recall {res.loc[best_name,'Recall']:.3f}</span>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([":material/analytics:  Model Comparison", ":material/grid_on:  Confusion Matrix", ":material/rule:  Per-Class Metrics"])

    with tab1:
        st.dataframe(
            res.style
               .background_gradient(cmap="Blues")
               .format("{:.4f}")
               .highlight_max(color="#0d3354", axis=0),
            use_container_width=True
        )
        metrics = ["Accuracy","F1 (macro)","Precision","Recall"]
        fig = go.Figure()
        for i, m in enumerate(metrics):
            if m not in res.columns: continue
            fig.add_trace(go.Bar(
                name=m, x=res.index, y=res[m],
                marker_color=PALETTE[i],
                text=[f"{v:.3f}" for v in res[m]],
                textposition="outside",
                textfont=dict(color="#e2e8f0", size=11)
            ))
        fig.add_hline(y=0.8, line_dash="dash",
                      line_color="rgba(239,68,68,0.5)",
                      annotation_text="0.80 target",
                      annotation_font_color="#ef4444")
        dark_fig(fig, 440).update_layout(
            barmode="group", yaxis=dict(range=[0,1.15], title="Score"),
            title="5-Fold CV — All Models",
            legend=dict(orientation="h", y=1.08, x=0)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="glass">
          <span style="color:#00d4ff;font-weight:700;">Why Logistic Regression for TF-IDF?</span>
          <p style="color:#94a3b8;font-size:0.88rem;margin:8px 0 0 0;line-height:1.7;">
          TF-IDF produces <b style="color:#e2e8f0;">high-dimensional sparse vectors</b> (~10k features).
          Linear models like LR and SVM exploit this sparsity efficiently.
          LR additionally provides <b style="color:#e2e8f0;">calibrated probability scores</b>
          per class, enabling confidence-ranked predictions — critical for the live demo.
          Tree-based models perform poorly on sparse text because individual feature splits
          carry little information.
          </p>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        cm     = confusion_matrix(yte, yte_pred, labels=best_pipe.classes_)
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

        c1, c2 = st.columns(2)
        with c1:
            fig_c1 = px.imshow(cm, x=list(best_pipe.classes_), y=list(best_pipe.classes_),
                               text_auto=True,
                               color_continuous_scale=[[0,"#0a0a1a"],[1,"#00d4ff"]],
                               title=f"{best_name} — Counts", aspect="auto")
            dark_fig(fig_c1, max(420, len(best_pipe.classes_)*30))
            fig_c1.update_layout(xaxis_tickangle=-40)
            st.plotly_chart(fig_c1, use_container_width=True)

        with c2:
            fig_c2 = px.imshow(np.round(cm_pct,1),
                               x=list(best_pipe.classes_), y=list(best_pipe.classes_),
                               text_auto=True,
                               color_continuous_scale=[[0,"#0a0a1a"],[0.5,"#7c3aed"],[1,"#ef4444"]],
                               title=f"{best_name} — Row % (normalised)", aspect="auto")
            dark_fig(fig_c2, max(420, len(best_pipe.classes_)*30))
            fig_c2.update_layout(xaxis_tickangle=-40)
            st.plotly_chart(fig_c2, use_container_width=True)

        # hardest pairs
        np.fill_diagonal(cm, 0)
        flat_idx = np.argsort(cm.flatten())[::-1][:5]
        pairs = [(best_pipe.classes_[i//len(best_pipe.classes_)],
                  best_pipe.classes_[i % len(best_pipe.classes_)],
                  cm.flatten()[i]) for i in flat_idx if cm.flatten()[i] > 0]
        if pairs:
            st.markdown('<div class="section-sub" style="margin-top:12px;">Top misclassification pairs</div>', unsafe_allow_html=True)
            pcols = st.columns(len(pairs))
            for col, (actual, pred_, cnt) in zip(pcols, pairs):
                with col:
                    st.markdown(f"""
                    <div class="glass" style="text-align:center;padding:14px;">
                      <div style="font-size:0.75rem;color:#64748b;">Actual → Predicted</div>
                      <div style="font-size:1rem;font-weight:700;color:#ef4444;margin:6px 0;">
                        {actual} → {pred_}
                      </div>
                      <div style="font-size:1.4rem;font-weight:800;color:#f59e0b;">{cnt}</div>
                      <div style="font-size:0.72rem;color:#64748b;">mistakes</div>
                    </div>
                    """, unsafe_allow_html=True)

    with tab3:
        report = classification_report(yte, yte_pred, output_dict=True)
        rep_df = pd.DataFrame(report).T
        drop   = [r for r in ["accuracy","macro avg","weighted avg"] if r in rep_df.index]
        rep_df = rep_df.drop(index=drop)[["precision","recall","f1-score","support"]].astype(float)

        st.dataframe(
            rep_df.style
                  .background_gradient(cmap="RdYlGn", subset=["precision","recall","f1-score"])
                  .format({"precision":"{:.3f}","recall":"{:.3f}",
                           "f1-score":"{:.3f}","support":"{:.0f}"}),
            use_container_width=True
        )

        melt = rep_df.drop(columns="support").reset_index().melt(
            id_vars="index", var_name="Metric", value_name="Score"
        )
        fig_pc = px.bar(melt, x="index", y="Score", color="Metric",
                        barmode="group", color_discrete_sequence=[PALETTE[0],PALETTE[1],PALETTE[2]],
                        title="Per-Class Precision · Recall · F1",
                        labels={"index":"Course"})
        dark_fig(fig_pc, 460).update_layout(yaxis=dict(range=[0,1.15]), xaxis_tickangle=-40,
                                            bargap=0.2,
                                            legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig_pc, use_container_width=True)


# 🎯  LIVE PREDICTOR
elif page == ":material/target:  Live Predictor":

    st.markdown('<div class="section-header">Live Document Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Paste any academic text and the AI predicts which course it belongs to — instantly</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        user_text = st.text_area(
            ":material/article: Paste your document text below",
            height=280,
            placeholder=(
                "e.g.  'A binary search tree is a node-based data structure "
                "where each node has at most two children called left and right. "
                "The left subtree contains nodes with keys less than the parent…'"
            )
        )
        predict_btn = st.button(":material/bolt: Predict Course", use_container_width=True)

    with col2:
        st.markdown("""
        <div class="glass">
          <div style="font-size:0.85rem;font-weight:700;color:#00d4ff;
                      letter-spacing:0.05em;text-transform:uppercase;margin-bottom:12px;">
            How It Works
          </div>
          <div style="font-size:0.85rem;color:#94a3b8;line-height:1.9;">
            <b style="color:#e2e8f0;">1.</b> Text → TF-IDF (10k features, bigrams)<br>
            <b style="color:#e2e8f0;">2.</b> Sparse vector → trained classifier<br>
            <b style="color:#e2e8f0;">3.</b> Decision scores → softmax probabilities<br>
            <b style="color:#e2e8f0;">4.</b> Top course + confidence ranking
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="glass" style="margin-top:12px;">
          <div style="font-size:0.75rem;color:#64748b;text-transform:uppercase;
                      letter-spacing:0.08em;margin-bottom:8px;">Active Model</div>
          <div style="font-size:1rem;font-weight:700;color:#00d4ff;">{md['best_name']}</div>
          <div style="font-size:0.78rem;color:#64748b;margin-top:4px;">
            Trained on {len(df_clean):,} docs · {df_clean[CS].nunique()} courses
          </div>
        </div>
        """, unsafe_allow_html=True)

    if predict_btn and user_text.strip():
        lr_pipe   = md["lr_pipe"]
        best_pipe = md["best_pipe"]
        mod_pipe  = md.get("mod_pipe")
        pred      = best_pipe.predict([user_text])[0]

        # confidence scores via LR (always has predict_proba)
        probs   = lr_pipe.predict_proba([user_text])[0]
        classes = lr_pipe.classes_
        top_conf = probs.max() * 100
        conf_color = "#ffffff" if top_conf > 70 else "#a1a1aa" if top_conf > 40 else "#71717a"

        # module prediction
        mod_pred      = None
        mod_conf      = None
        mod_conf_color= "#64748b"
        if mod_pipe is not None:
            mod_probs  = mod_pipe.predict_proba([user_text])[0]
            mod_pred   = mod_pipe.classes_[np.argmax(mod_probs)]
            mod_conf   = mod_probs.max() * 100
            mod_conf_color = "#ffffff" if mod_conf > 70 else "#a1a1aa" if mod_conf > 40 else "#71717a"

        st.markdown("<br>", unsafe_allow_html=True)
        r1, r2, r3 = st.columns([1.2, 0.8, 2])

        with r1:
            # Course prediction card
            st.markdown(f"""
            <div class="pred-box">
              <div class="pred-label">Predicted Course</div>
              <div class="pred-course">{pred}</div>
              <div style="margin-top:12px;">
                <span style="font-size:0.78rem;color:#64748b;">Confidence</span><br>
                <span style="font-size:2rem;font-weight:800;color:{conf_color};">{top_conf:.1f}%</span>
              </div>
            </div>
            """, unsafe_allow_html=True)
            # Module prediction card (separate call — no nested f-strings)
            if mod_pred:
                st.markdown(f"""
                <div class="pred-box" style="margin-top:12px;border-color:#7c3aed44;">
                  <div class="pred-label">Predicted Module</div>
                  <div style="font-size:1.8rem;font-weight:800;color:#7c3aed;
                              letter-spacing:0.03em;margin:4px 0;">{mod_pred}</div>
                  <div style="margin-top:6px;">
                    <span style="font-size:0.78rem;color:#64748b;">Confidence</span><br>
                    <span style="font-size:1.6rem;font-weight:700;color:{mod_conf_color};">{mod_conf:.1f}%</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        with r2:
            wc = len(user_text.split())
            uc = len(set(user_text.lower().split()))
            # Base stats card
            st.markdown(f"""
            <div class="glass">
              <div style="font-size:0.75rem;color:#64748b;text-transform:uppercase;
                          letter-spacing:0.08em;margin-bottom:12px;">Input Stats</div>
              <div class="stat-pill" style="display:block;margin:6px 0;">Words: <span>{wc}</span></div>
              <div class="stat-pill" style="display:block;margin:6px 0;">Unique: <span>{uc}</span></div>
              <div class="stat-pill" style="display:block;margin:6px 0;">Lexical div: <span>{uc/max(wc,1):.2f}</span></div>
            </div>
            """, unsafe_allow_html=True)
            # Top modules card (separate call)
            if mod_pipe is not None:
                mod_probs_all = mod_pipe.predict_proba([user_text])[0]
                top3_idx = np.argsort(mod_probs_all)[::-1][:3]
                top3 = [(mod_pipe.classes_[i], mod_probs_all[i] * 100) for i in top3_idx]
                rows_html = "".join(
                    f'<div class="stat-pill" style="display:block;margin:4px 0;">'
                    f'{m}: <span>{c:.1f}%</span></div>'
                    for m, c in top3
                )
                st.markdown(
                    '<div class="glass" style="margin-top:10px;">'
                    '<div style="font-size:0.72rem;color:#64748b;text-transform:uppercase;'
                    'letter-spacing:0.08em;margin-bottom:8px;">Top Modules</div>'
                    + rows_html +
                    '</div>',
                    unsafe_allow_html=True
                )

        with r3:
            score_df = pd.DataFrame({
                "Course": classes,
                "Confidence (%)": np.round(probs * 100, 2)
            }).sort_values("Confidence (%)", ascending=True).tail(10)

            colors = ["#7c3aed" if c == pred else "#00d4ff"
                      for c in score_df["Course"]]
            fig = go.Figure(go.Bar(
                x=score_df["Confidence (%)"],
                y=score_df["Course"],
                orientation="h",
                marker=dict(color=colors),
                text=[f"{v:.1f}%" for v in score_df["Confidence (%)"]],
                textposition="outside",
                textfont=dict(color="#e2e8f0", size=11)
            ))
            dark_fig(fig, 340).update_layout(
                title="Confidence per Course (top 10)",
                xaxis=dict(range=[0, min(100, score_df["Confidence (%)"].max()*1.3)],
                           title="Confidence (%)")
            )
            st.plotly_chart(fig, use_container_width=True)

            # Module confidence bar if available
            if mod_pipe is not None:
                mod_probs_all = mod_pipe.predict_proba([user_text])[0]
                mod_score_df = pd.DataFrame({
                    "Module": mod_pipe.classes_,
                    "Confidence (%)": np.round(mod_probs_all * 100, 2)
                }).sort_values("Confidence (%)", ascending=True).tail(8)
                mod_colors = ["#7c3aed" if m == mod_pred else "#8b5cf6"
                              for m in mod_score_df["Module"]]
                fig_m = go.Figure(go.Bar(
                    x=mod_score_df["Confidence (%)"],
                    y=mod_score_df["Module"],
                    orientation="h",
                    marker=dict(color=mod_colors),
                    text=[f"{v:.1f}%" for v in mod_score_df["Confidence (%)"]],
                    textposition="outside",
                    textfont=dict(color="#e2e8f0", size=11)
                ))
                dark_fig(fig_m, 280).update_layout(
                    title="Confidence per Module (top 8)",
                    xaxis=dict(range=[0, min(100, mod_score_df["Confidence (%)"].max()*1.3)],
                               title="Confidence (%)")
                )
                st.plotly_chart(fig_m, use_container_width=True)

    elif predict_btn:
        st.warning("Please enter some text first.")

    st.markdown("---")
    st.markdown('<div class="section-sub">🎲 Try a random document from the dataset</div>', unsafe_allow_html=True)

    if st.button("Load Random Document", use_container_width=False):
        sample   = df_clean.sample(1).iloc[0]
        mod_pipe = md.get("mod_pipe")
        CM_col   = md.get("CM")
        c1, c2   = st.columns([1, 3])
        with c1:
            actual_mod = str(sample[CM_col]) if CM_col and CM_col in df_clean.columns else None
            mod_actual_html = (
                f"<div style='font-size:0.8rem;color:#94a3b8;margin-top:6px;'>"
                f"Module: <span style='color:#8b5cf6;font-weight:600;'>{actual_mod}</span></div>"
                if actual_mod and actual_mod.lower() not in ("nan", "unknown", "") else ""
            )
            st.markdown(f"""
            <div class="glass">
              <div style="font-size:0.72rem;color:#64748b;text-transform:uppercase;
                          letter-spacing:0.08em;">Actual Course</div>
              <div style="font-size:1.3rem;font-weight:700;color:#00d4ff;margin:4px 0;">
                {sample[CS]}
              </div>
              {mod_actual_html}
              {"<div style='font-size:0.8rem;color:#94a3b8;'>Type: " + str(sample[CD]) + "</div>" if CD and CD in df_clean.columns else ""}
              <div style="font-size:0.8rem;color:#94a3b8;">Words: {sample['word_count']}</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            preview = " ".join(str(sample[CC]).split()[:200])
            st.text_area("Document preview (200 words):", value=preview, height=150)
            if len(preview.split()) > 10:
                model_pred = md["best_pipe"].predict([sample[CC]])[0]
                match = "✅ Correct" if model_pred == sample[CS] else "❌ Wrong"
                badge_parts = [f"Course → {model_pred}  {match}"]
                if mod_pipe is not None:
                    mod_pred_r = mod_pipe.predict([sample[CC]])[0]
                    mod_match  = ""
                    if actual_mod and actual_mod.lower() not in ("nan", "unknown", ""):
                        mod_match = "  ✅" if mod_pred_r == actual_mod else "  ❌"
                    badge_parts.append(f"Module → {mod_pred_r}{mod_match}")
                st.markdown(
                    "  &nbsp;|&nbsp;  ".join(
                        [f'<span class="badge">{b}</span>' for b in badge_parts]
                    ),
                    unsafe_allow_html=True
                )


# 🌌  DOCUMENT UNIVERSE
elif page == ":material/explore:  Document Universe":

    st.markdown('<div class="section-header">Document Universe</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Every point is one document. The AI compressed 10,000 TF-IDF dimensions down to 3 using SVD. Rotate, zoom, and explore.</div>', unsafe_allow_html=True)

    df_m  = md["df_m"]
    X3d   = md["X3d"]
    CS_m  = md["CS"]
    CD_m  = md["CD"]
    svd   = md["svd"]

    ev = svd.explained_variance_ratio_
    pca_df = pd.DataFrame({
        "PC1": X3d[:,0], "PC2": X3d[:,1], "PC3": X3d[:,2],
        "Course": df_m[CS_m].values,
        "Words": df_m["word_count"].values,
    })
    if CD_m and CD_m in df_m.columns:
        pca_df["DocType"] = df_m[CD_m].values

    ctrl1, ctrl2, ctrl3 = st.columns([2, 1, 1])
    with ctrl1:
        view   = st.radio("View", ["🌐 3D Interactive", "📍 2D Scatter"], horizontal=True)
    with ctrl2:
        colour = st.selectbox("Colour by",
                              ["Course"] +
                              (["DocType"] if CD_m and CD_m in df_m.columns else []) +
                              ["Words"])
    with ctrl3:
        opacity = st.slider("Point opacity", 0.2, 1.0, 0.65, 0.05)

    # explained variance banner
    st.markdown(f"""
    <div style="display:flex;gap:12px;margin:8px 0 16px 0;flex-wrap:wrap;">
      <span class="badge">PC1: {ev[0]:.2%} var</span>
      <span class="badge">PC2: {ev[1]:.2%} var</span>
      <span class="badge">PC3: {ev[2]:.2%} var</span>
      <span class="badge">Total: {sum(ev[:3]):.2%} variance explained</span>
    </div>
    """, unsafe_allow_html=True)

    is_continuous = colour == "Words"
    color_map     = None if is_continuous else {c: PALETTE[i%len(PALETTE)]
                                                 for i, c in enumerate(pca_df[colour].unique())}

    if view == "🌐 3D Interactive":
        fig = px.scatter_3d(
            pca_df, x="PC1", y="PC2", z="PC3",
            color=colour,
            color_discrete_map=color_map,
            color_continuous_scale="plasma" if is_continuous else None,
            opacity=opacity,
            size_max=4,
            hover_data=["Course","Words"] + (["DocType"] if "DocType" in pca_df else []),
            title="Academic Document Space — TF-IDF → SVD Projection"
        )
        fig.update_traces(marker=dict(size=3, line=dict(width=0)))
        dark_fig(fig, 640).update_layout(
            scene=dict(
                xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.06)"),
                zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.06)"),
                bgcolor="rgba(0,0,0,0)"
            ),
            margin=dict(t=50, b=0, l=0, r=0)
        )
    else:
        fig = px.scatter(
            pca_df, x="PC1", y="PC2",
            color=colour,
            color_discrete_map=color_map,
            color_continuous_scale="plasma" if is_continuous else None,
            opacity=opacity,
            hover_data=["Course","Words"],
            title="Document Space — 2D Projection"
        )
        fig.update_traces(marker=dict(size=5, line=dict(width=0)))
        dark_fig(fig, 560)

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="glass">
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;">
        <div>
          <div style="font-size:0.78rem;color:#00d4ff;font-weight:700;
                      text-transform:uppercase;letter-spacing:0.08em;">Tight clusters</div>
          <div style="font-size:0.85rem;color:#94a3b8;margin-top:6px;line-height:1.6;">
            Courses that form tight, well-separated blobs are easy for the model to classify.
          </div>
        </div>
        <div>
          <div style="font-size:0.78rem;color:#7c3aed;font-weight:700;
                      text-transform:uppercase;letter-spacing:0.08em;">Overlapping regions</div>
          <div style="font-size:0.85rem;color:#94a3b8;margin-top:6px;line-height:1.6;">
            Courses that overlap share vocabulary — these appear as misclassifications in
            the confusion matrix.
          </div>
        </div>
        <div>
          <div style="font-size:0.78rem;color:#10b981;font-weight:700;
                      text-transform:uppercase;letter-spacing:0.08em;">Outlier points</div>
          <div style="font-size:0.85rem;color:#94a3b8;margin-top:6px;line-height:1.6;">
            Isolated points far from their cluster are often reference books or syllabi
            with generic language.
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
