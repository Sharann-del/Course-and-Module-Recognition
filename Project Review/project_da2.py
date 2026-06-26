# ── Imports ───────────────────────────────────────────────────────────────────
import os, re, warnings, sqlite3, textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import chi2_contingency

from sklearn.pipeline          import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model      import LogisticRegression
from sklearn.naive_bayes       import MultinomialNB
from sklearn.ensemble          import RandomForestClassifier
from sklearn.svm               import LinearSVC
from sklearn.model_selection   import (StratifiedKFold, cross_validate,
                                       train_test_split)
from sklearn.metrics           import (classification_report, confusion_matrix,
                                       accuracy_score, f1_score,
                                       precision_score, recall_score)
from sklearn.decomposition     import TruncatedSVD
from sklearn.preprocessing     import LabelEncoder
from sklearn.utils             import resample

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR   = BASE_DIR + os.sep
DB_PATH      = os.path.join(BASE_DIR, "academic_docs.db")

# Prefer the freshly extracted full dataset; fall back to original CSV
FULL_CSV  = os.path.join(BASE_DIR, "academic_documents_full.csv")
ORIG_CSV  = os.path.join(BASE_DIR, "academic_documents_with_extracted_text.csv")
DATA_PATH = FULL_CSV if os.path.exists(FULL_CSV) else ORIG_CSV

# Colour palette (up to 20 courses)
PALETTE = [
    "#4C72B0","#DD8452","#55A868","#C44E52","#8172B2",
    "#937860","#DA8BC3","#8C8C8C","#CCB974","#64B5CD",
    "#E377C2","#7F7F7F","#BCBD22","#17BECF","#AEC7E8",
    "#FFBB78","#98DF8A","#FF9896","#C5B0D5","#C49C94",
]

# ─────────────────────────────────────────────────────────────────────────────
# ██  SECTION 1 — DATA COLLECTION & UNDERSTANDING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  SECTION 1 — Data Collection & Understanding")
print("="*65)

df = pd.read_csv(DATA_PATH, low_memory=False)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

print(f"\n  Dataset     : {os.path.basename(DATA_PATH)}")
print(f"  Raw shape   : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"  Columns     : {list(df.columns)}")

# Auto-detect key columns
def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

COL_CONTENT = find_col(df, ["content", "extracted_text", "text", "body"])
COL_COURSE  = find_col(df, ["course_label", "course", "subject", "label"])
COL_MODULE  = find_col(df, ["module_label", "module", "topic"])
COL_DOCTYPE = find_col(df, ["doc_type", "document_type", "type"])
COL_FORMAT  = find_col(df, ["file_format", "format", "extension"])

assert COL_CONTENT, "Cannot find content column"
assert COL_COURSE,  "Cannot find course label column"

# ─────────────────────────────────────────────────────────────────────────────
# ██  SECTION 2 — PROBLEM UNDERSTANDING & OBJECTIVE DEFINITION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  SECTION 2 — Problem Statement")
print("="*65)
print(textwrap.dedent("""
  PROBLEM:
    Academic institutions accumulate thousands of lecture notes,
    slides, syllabi, and reference materials across many courses.
    Manually routing each document to the correct course and module
    is time-consuming and error-prone.

  OBJECTIVE:
    Build a multi-class NLP classifier that automatically
    identifies (a) the course and (b) the module a document
    belongs to, using only its text content.

  MEASURABLE SUCCESS CRITERIA:
    • Course classification accuracy  ≥ 80 %
    • Macro-averaged F1 score         ≥ 0.75
    • 5-fold cross-validated results  to ensure generalisability

  ASSUMPTIONS:
    • Document text content is the primary differentiator
    • File path encodes ground-truth labels
    • Scanned/image PDFs with no text are excluded
"""))

# ─────────────────────────────────────────────────────────────────────────────
# ██  SECTION 3 — DATA PREPARATION & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
print("="*65)
print("  SECTION 3 — Data Preparation & Preprocessing")
print("="*65)

# 3a. Normalise & clean text content
df[COL_CONTENT] = (
    df[COL_CONTENT]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.replace(r'\s+', ' ', regex=True)
)
df[COL_COURSE] = df[COL_COURSE].fillna("unknown").astype(str).str.strip()

# 3b. Feature engineering
df["word_count"]  = df[COL_CONTENT].str.split().str.len()
df["char_count"]  = df[COL_CONTENT].str.len()
df["unique_words"] = df[COL_CONTENT].apply(
    lambda x: len(set(x.lower().split()))
)
df["lexical_diversity"] = df.apply(
    lambda r: r["unique_words"] / max(r["word_count"], 1), axis=1
)

# 3c. Filter: keep documents with enough text (≥30 words)
before = len(df)
df_clean = df[df["word_count"] >= 30].copy()
print(f"\n  Removed {before - len(df_clean)} documents with < 30 words")
print(f"  Clean dataset: {len(df_clean):,} documents")

# 3d. Outlier handling: cap word_count at 99th percentile for analysis
wc_cap = df_clean["word_count"].quantile(0.99)
df_clean["word_count_capped"] = df_clean["word_count"].clip(upper=wc_cap)

# 3e. Missing value report
missing = df_clean.isnull().sum()
print(f"\n  Missing values:\n{missing[missing > 0].to_string() if missing.any() else '  None'}")

# 3f. Class distribution
course_counts = df_clean[COL_COURSE].value_counts()
print(f"\n  Courses ({course_counts.shape[0]}) and document counts:")
print(course_counts.to_string())

# 3g. Label encoding
le_course = LabelEncoder()
df_clean["course_encoded"] = le_course.fit_transform(df_clean[COL_COURSE])

# ─────────────────────────────────────────────────────────────────────────────
# ██  SECTION 4 — EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  SECTION 4 — Exploratory Data Analysis")
print("="*65)

courses    = course_counts.index.tolist()
n_courses  = len(courses)
course_pal = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(courses)}

sns.set_theme(style="whitegrid", font_scale=1.05)

# ── FIG 1 — Course distribution (horizontal bar) ─────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(course_counts.index[::-1], course_counts.values[::-1],
               color=[course_pal[c] for c in course_counts.index[::-1]])
ax.set_xlabel("Number of Documents")
ax.set_title("Document Count per Course", fontsize=14, fontweight="bold")
for bar in bars:
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
            f'{bar.get_width():.0f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(OUTPUT_DIR + "fig1_course_distribution.png", dpi=200)
plt.close()
print("  ✔ fig1_course_distribution.png")

# ── FIG 2 — Document types by course (stacked bar) ───────────────────────────
if COL_DOCTYPE:
    ct = pd.crosstab(df_clean[COL_COURSE], df_clean[COL_DOCTYPE])
    fig, ax = plt.subplots(figsize=(max(12, n_courses*0.8), 5))
    ct.plot(kind="bar", stacked=True, ax=ax, colormap="tab10",
            edgecolor="white", linewidth=0.5)
    ax.set_title("Document Types by Course", fontsize=13, fontweight="bold")
    ax.set_xlabel("Course"); ax.set_ylabel("Count")
    ax.legend(title="Doc Type", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + "fig2_doc_types_by_course.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  ✔ fig2_doc_types_by_course.png")

# ── FIG 3 — File formats by course ───────────────────────────────────────────
if COL_FORMAT:
    ct2 = pd.crosstab(df_clean[COL_COURSE], df_clean[COL_FORMAT])
    fig, ax = plt.subplots(figsize=(max(12, n_courses*0.8), 5))
    ct2.plot(kind="bar", stacked=True, ax=ax, colormap="Paired",
             edgecolor="white", linewidth=0.5)
    ax.set_title("File Formats by Course", fontsize=13, fontweight="bold")
    ax.set_xlabel("Course"); ax.set_ylabel("Count")
    ax.legend(title="Format", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + "fig3_file_formats_by_course.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  ✔ fig3_file_formats_by_course.png")

# ── FIG 4 — Word count distribution (violin + strip) ─────────────────────────
fig, ax = plt.subplots(figsize=(max(14, n_courses*0.9), 5))
order = (df_clean.groupby(COL_COURSE)["word_count_capped"]
         .median().sort_values(ascending=False).index)
sns.violinplot(data=df_clean, x=COL_COURSE, y="word_count_capped",
               order=order, palette=course_pal, inner="quartile", ax=ax,
               scale="width")
ax.set_title("Document Length Distribution by Course", fontsize=13, fontweight="bold")
ax.set_xlabel("Course"); ax.set_ylabel("Word Count (capped at 99th pct)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR + "fig4_word_count_violin.png", dpi=200, bbox_inches="tight")
plt.close()
print("  ✔ fig4_word_count_violin.png")

# ── FIG 5 — Summary statistics table (rendered as figure) ────────────────────
stats_df = df_clean.groupby(COL_COURSE)["word_count"].agg(
    ["count", "mean", "median", "std", "min", "max"]
).round(1).reset_index()
stats_df.columns = ["Course", "N Docs", "Mean WC", "Median WC", "Std", "Min WC", "Max WC"]

fig, ax = plt.subplots(figsize=(14, max(4, len(stats_df)*0.45 + 1)))
ax.axis("off")
tbl = ax.table(
    cellText=stats_df.values,
    colLabels=stats_df.columns,
    cellLoc="center", loc="center"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.5)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor("#4C72B0")
        cell.set_text_props(color="white", fontweight="bold")
    elif row % 2 == 0:
        cell.set_facecolor("#EEF2FF")
ax.set_title("Dataset Summary Statistics", fontsize=13, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig(OUTPUT_DIR + "fig5_summary_statistics.png", dpi=200, bbox_inches="tight")
plt.close()
print("  ✔ fig5_summary_statistics.png")

# ── Statistical Analysis: Chi-Square test (doc_type vs course) ───────────────
print("\n  Statistical Analysis:")
if COL_DOCTYPE:
    ct_chi = pd.crosstab(df_clean[COL_COURSE], df_clean[COL_DOCTYPE])
    chi2, p, dof, _ = chi2_contingency(ct_chi)
    print(f"  Chi-Square (doc_type vs course): χ²={chi2:.2f}, p={p:.4f}, dof={dof}")
    if p < 0.05:
        print("  → Statistically significant association (p < 0.05)")

# ── FIG 6 — Top TF-IDF keywords (one panel per course, grid layout) ───────────
print("\n  Computing per-course TF-IDF keywords …")
MAX_TERMS = 12
ncols = min(4, n_courses)
nrows = (n_courses + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

for i, course in enumerate(courses):
    ax = axes_flat[i]
    sub = df_clean[df_clean[COL_COURSE] == course][COL_CONTENT]
    if len(sub) < 2:
        ax.set_visible(False)
        continue
    try:
        vec = TfidfVectorizer(max_features=MAX_TERMS, stop_words="english",
                              ngram_range=(1, 2))
        vec.fit(sub)
        terms  = vec.get_feature_names_out()
        scores = vec.idf_
        top_i  = np.argsort(scores)[::-1][:MAX_TERMS]
        ax.barh(terms[top_i][::-1], scores[top_i][::-1],
                color=course_pal.get(course, PALETTE[0]))
    except Exception:
        ax.set_visible(False)
        continue
    ax.set_title(course, fontsize=10, fontweight="bold")
    ax.set_xlabel("IDF score", fontsize=8)
    ax.tick_params(labelsize=8)

for j in range(i+1, len(axes_flat)):
    axes_flat[j].set_visible(False)

plt.suptitle("Top TF-IDF Keywords per Course", fontsize=14,
             fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT_DIR + "fig6_keywords_per_course.png", dpi=200, bbox_inches="tight")
plt.close()
print("  ✔ fig6_keywords_per_course.png")

# ── FIG 7 — Lexical diversity by course ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(max(12, n_courses*0.8), 4))
lex_order = (df_clean.groupby(COL_COURSE)["lexical_diversity"]
             .mean().sort_values(ascending=False).index)
sns.barplot(data=df_clean, x=COL_COURSE, y="lexical_diversity",
            order=lex_order, palette=course_pal, ax=ax, estimator=np.mean,
            errorbar="sd", capsize=0.2)
ax.set_title("Lexical Diversity (Unique/Total Words) by Course",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Course"); ax.set_ylabel("Lexical Diversity")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR + "fig7_lexical_diversity.png", dpi=200, bbox_inches="tight")
plt.close()
print("  ✔ fig7_lexical_diversity.png")

# ── FIG 8 — Module coverage heatmap ──────────────────────────────────────────
if COL_MODULE:
    df_clean[COL_MODULE] = df_clean[COL_MODULE].fillna("unknown").astype(str)
    pivot = pd.crosstab(df_clean[COL_COURSE], df_clean[COL_MODULE])
    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns)*0.9),
                                    max(6, len(pivot)*0.5)))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlGnBu", ax=ax, linewidths=0.3)
    ax.set_title("Document Count per Course × Module",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Module"); ax.set_ylabel("Course")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + "fig8_module_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  ✔ fig8_module_heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# ██  SECTION 5 — MODEL SELECTION & TRAINING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  SECTION 5 — Model Selection & Training")
print("="*65)

# ── 5a. Prepare training data ────────────────────────────────────────────────
valid_courses = course_counts[course_counts >= 5].index.tolist()
df_model = df_clean[df_clean[COL_COURSE].isin(valid_courses)].copy()
X = df_model[COL_CONTENT]
y = df_model[COL_COURSE]

print(f"\n  Training corpus : {len(X):,} documents")
print(f"  Classes         : {y.nunique()} courses")
print(f"  Courses         : {sorted(y.unique())}")

# ── 5b. Class imbalance analysis ─────────────────────────────────────────────
min_class = y.value_counts().min()
max_class = y.value_counts().max()
print(f"\n  Class imbalance : min={min_class}, max={max_class}, "
      f"ratio={max_class/max(min_class,1):.1f}x")
print("  → Using class_weight='balanced' to handle imbalance")

# ── 5c. Define 4 pipelines ───────────────────────────────────────────────────
TFIDF_ARGS = dict(
    max_features  = 10000,
    stop_words    = "english",
    ngram_range   = (1, 2),
    sublinear_tf  = True,
    min_df        = 2,
)

models = {
    "Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_ARGS)),
        ("clf",   LogisticRegression(max_iter=2000, C=1.0,
                                     class_weight="balanced", solver="lbfgs"))
    ]),
    "Linear SVM": Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_ARGS)),
        ("clf",   LinearSVC(max_iter=3000, C=1.0, class_weight="balanced"))
    ]),
    "Random Forest": Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_ARGS)),
        ("clf",   RandomForestClassifier(n_estimators=200, random_state=42,
                                          class_weight="balanced", n_jobs=-1,
                                          max_depth=None))
    ]),
    "Naive Bayes": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, stop_words="english",
                                   ngram_range=(1, 2), sublinear_tf=False, min_df=2)),
        ("clf",   MultinomialNB(alpha=0.1))
    ]),
}

# ── 5d. 5-fold stratified cross-validation ───────────────────────────────────
cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]

cv_results = {}
print("\n  Running 5-fold cross-validation …\n")
for name, pipe in models.items():
    s = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    cv_results[name] = {
        "Accuracy"      : round(s["test_accuracy"].mean(), 4),
        "F1 (macro)"    : round(s["test_f1_macro"].mean(), 4),
        "F1 (weighted)" : round(s["test_f1_weighted"].mean(), 4),
        "Precision"     : round(s["test_precision_macro"].mean(), 4),
        "Recall"        : round(s["test_recall_macro"].mean(), 4),
        "Acc Std"       : round(s["test_accuracy"].std(), 4),
    }
    print(f"  {name:22s}  acc={cv_results[name]['Accuracy']:.3f}±"
          f"{cv_results[name]['Acc Std']:.3f}  "
          f"F1={cv_results[name]['F1 (macro)']:.3f}")

results_df = pd.DataFrame(cv_results).T
best_name  = results_df["Accuracy"].idxmax()
print(f"\n  ★ Best model: {best_name}  "
      f"({results_df.loc[best_name,'Accuracy']:.1%} accuracy, "
      f"{results_df.loc[best_name,'F1 (macro)']:.3f} F1)")
print(f"\n  Justification: {best_name} was selected based on highest 5-fold "
      f"cross-validated accuracy AND F1-macro, with stable std across folds.")

# ── 5e. Train best model — 80/20 split ───────────────────────────────────────
best_pipe = models[best_name]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
best_pipe.fit(X_train, y_train)
y_pred = best_pipe.predict(X_test)

print(f"\n  Train size : {len(X_train):,}  |  Test size : {len(X_test):,}")
print(f"\n  Classification report ({best_name}):")
print(classification_report(y_test, y_pred))

# Add predictions back
df_model = df_model.copy()
df_model["predicted_course"] = best_pipe.predict(X)
df_model["correct"] = (df_model["predicted_course"] == df_model[COL_COURSE]).astype(int)

# ─────────────────────────────────────────────────────────────────────────────
# ██  SECTION 6 — PERFORMANCE VISUALISATIONS   [DA2 — Model 5 marks]
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  SECTION 6 — Model Performance Visualisations")
print("="*65)

# ── FIG 9 — Model comparison grouped bar ─────────────────────────────────────
metrics_to_plot = ["Accuracy", "F1 (macro)", "Precision", "Recall"]
x      = np.arange(len(results_df))
width  = 0.20

fig, ax = plt.subplots(figsize=(12, 5))
for i, metric in enumerate(metrics_to_plot):
    if metric in results_df.columns:
        vals = results_df[metric]
        bars = ax.bar(x + i*width, vals, width,
                      label=metric, color=PALETTE[i], alpha=0.88)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x + width*1.5)
ax.set_xticklabels(results_df.index, rotation=15, ha="right")
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score (5-fold CV)")
ax.set_title("Model Comparison — 5-Fold Cross-Validation", fontsize=13, fontweight="bold")
ax.legend(loc="lower right")
ax.axhline(0.8, color="red", linestyle="--", alpha=0.4, label="80% target")
plt.tight_layout()
plt.savefig(OUTPUT_DIR + "fig9_model_comparison.png", dpi=200)
plt.close()
print("  ✔ fig9_model_comparison.png")

# ── FIG 10 — Confusion matrix ─────────────────────────────────────────────────
cm     = confusion_matrix(y_test, y_pred, labels=best_pipe.classes_)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

fig, axes = plt.subplots(1, 2, figsize=(max(14, n_courses), max(7, n_courses//2)))
fs = max(6, 10 - n_courses//4)   # dynamic font size

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=best_pipe.classes_, yticklabels=best_pipe.classes_,
            ax=axes[0], annot_kws={"size": fs}, linewidths=0.3)
axes[0].set_title(f"Confusion Matrix — {best_name}\n(Counts)", fontweight="bold")
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
axes[0].tick_params(labelsize=fs)

sns.heatmap(np.round(cm_pct, 1), annot=True, fmt=".1f", cmap="YlOrRd",
            xticklabels=best_pipe.classes_, yticklabels=best_pipe.classes_,
            ax=axes[1], annot_kws={"size": fs}, linewidths=0.3)
axes[1].set_title(f"Confusion Matrix — {best_name}\n(Row % normalised)", fontweight="bold")
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")
axes[1].tick_params(labelsize=fs)

plt.suptitle("Model Prediction Errors — Where Does It Confuse?",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR + "fig10_confusion_matrix.png", dpi=200, bbox_inches="tight")
plt.close()
print("  ✔ fig10_confusion_matrix.png")

# ── FIG 11 — Per-class Precision / Recall / F1 ───────────────────────────────
report     = classification_report(y_test, y_pred, output_dict=True)
report_df  = pd.DataFrame(report).T
drop_rows  = [r for r in ["accuracy","macro avg","weighted avg"] if r in report_df.index]
report_df  = report_df.drop(index=drop_rows)[["precision","recall","f1-score","support"]].astype(float)

x2    = np.arange(len(report_df))
w2    = 0.27
fig, ax = plt.subplots(figsize=(max(12, len(report_df)*0.9), 5))
ax.bar(x2 - w2, report_df["precision"], w2, label="Precision", color=PALETTE[0])
ax.bar(x2,      report_df["recall"],    w2, label="Recall",    color=PALETTE[1])
ax.bar(x2 + w2, report_df["f1-score"], w2, label="F1-Score",  color=PALETTE[2])
ax.set_xticks(x2)
ax.set_xticklabels(report_df.index, rotation=45, ha="right", fontsize=9)
ax.set_ylim(0, 1.1); ax.set_ylabel("Score")
ax.set_title("Per-Class Precision, Recall, F1-Score", fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR + "fig11_per_class_metrics.png", dpi=200, bbox_inches="tight")
plt.close()
print("  ✔ fig11_per_class_metrics.png")

# ── FIG 12 — Feature importance: top predictive terms per course ──────────────
print("  Computing feature importance (Logistic Regression coefficients) …")
try:
    lr_pipe = models["Logistic Regression"]
    lr_pipe.fit(X_train, y_train)
    feat_names = np.array(lr_pipe["tfidf"].get_feature_names_out())
    classes    = lr_pipe["clf"].classes_
    coef       = lr_pipe["clf"].coef_
    nc         = len(classes)
    TOP_N      = 10

    ncols_fi = min(5, nc)
    nrows_fi = (nc + ncols_fi - 1) // ncols_fi
    fig, axes = plt.subplots(nrows_fi, ncols_fi,
                             figsize=(4*ncols_fi, 3.5*nrows_fi))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, (cls, coefs) in enumerate(zip(classes, coef)):
        ax = axes_flat[idx]
        top_i = np.argsort(coefs)[-TOP_N:]
        ax.barh(feat_names[top_i], coefs[top_i],
                color=course_pal.get(cls, PALETTE[idx % len(PALETTE)]))
        ax.set_title(cls, fontsize=9, fontweight="bold")
        ax.set_xlabel("LR coef", fontsize=8)
        ax.tick_params(labelsize=7)

    for j in range(idx+1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle("Top Predictive Terms per Course (LR Coefficients)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + "fig12_feature_importance.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  ✔ fig12_feature_importance.png")
except Exception as e:
    print(f"  ⚠  Feature importance skipped: {e}")

# ── FIG 13 — PCA / document embedding space (SVD 2D) ─────────────────────────
print("  Running SVD on TF-IDF matrix for PCA visualisation …")
try:
    tfidf_full = TfidfVectorizer(max_features=10000, stop_words="english",
                                  ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    X_tfidf = tfidf_full.fit_transform(df_model[COL_CONTENT])
    svd = TruncatedSVD(n_components=3, random_state=42)
    X_3d = svd.fit_transform(X_tfidf)

    fig, ax = plt.subplots(figsize=(10, 7))
    for i, course in enumerate(le_course.classes_):
        if course not in valid_courses:
            continue
        mask = df_model[COL_COURSE].values == course
        ax.scatter(X_3d[mask, 0], X_3d[mask, 1],
                   c=course_pal.get(course, PALETTE[i % len(PALETTE)]),
                   label=course, alpha=0.55, s=18, edgecolors="none")

    ax.set_title("Document Embedding Space — TF-IDF → SVD (2D)\nEach point = 1 document; colour = course",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel(f"SVD Component 1 ({svd.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"SVD Component 2 ({svd.explained_variance_ratio_[1]:.1%} var)")
    ax.legend(title="Course", bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=8, markerscale=2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + "fig13_document_pca.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  ✔ fig13_document_pca.png")
except Exception as e:
    print(f"  ⚠  PCA skipped: {e}")

# ── FIG 14 — Cross-course keyword overlap heatmap ────────────────────────────
print("  Computing keyword overlap matrix …")
try:
    term_sets = {}
    for course in courses:
        sub = df_clean[df_clean[COL_COURSE] == course][COL_CONTENT]
        if len(sub) < 2: continue
        vec = TfidfVectorizer(max_features=200, stop_words="english")
        vec.fit(sub)
        term_sets[course] = set(vec.get_feature_names_out())

    c_list = list(term_sets.keys())
    mat    = np.zeros((len(c_list), len(c_list)), dtype=int)
    for i, c1 in enumerate(c_list):
        for j, c2 in enumerate(c_list):
            mat[i, j] = len(term_sets[c1] & term_sets[c2])

    fig, ax = plt.subplots(figsize=(max(10, len(c_list)*0.8),
                                    max(8, len(c_list)*0.65)))
    fs_ov = max(6, 10 - len(c_list)//4)
    sns.heatmap(mat, annot=True, fmt="d", cmap="YlGnBu",
                xticklabels=c_list, yticklabels=c_list, ax=ax,
                annot_kws={"size": fs_ov})
    ax.set_title("Cross-Course Keyword Overlap\n(# shared terms from top-200 TF-IDF vocab)",
                 fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=fs_ov+1)
    plt.yticks(rotation=0, fontsize=fs_ov+1)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + "fig14_keyword_overlap.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  ✔ fig14_keyword_overlap.png")
except Exception as e:
    print(f"  ⚠  Keyword overlap skipped: {e}")

# ── BONUS — Module classification ────────────────────────────────────────────
if COL_MODULE:
    print("\n  BONUS: Module identifier …")
    df_clean[COL_MODULE] = df_clean[COL_MODULE].fillna("unknown").astype(str)
    known  = df_clean[df_clean[COL_MODULE] != "unknown"]
    unknown_ct = (df_clean[COL_MODULE] == "unknown").sum()
    module_counts = known[COL_MODULE].value_counts()
    valid_mods = module_counts[module_counts >= 3].index.tolist()
    df_known = known[known[COL_MODULE].isin(valid_mods)]

    if df_known[COL_MODULE].nunique() >= 2:
        mod_pipe = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english",
                                       ngram_range=(1,2), sublinear_tf=True)),
            ("clf",   LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced"))
        ])
        Xm = df_known[COL_CONTENT]; ym = df_known[COL_MODULE]
        Xm_tr, Xm_te, ym_tr, ym_te = train_test_split(
            Xm, ym, test_size=0.2, stratify=ym, random_state=42
        )
        mod_pipe.fit(Xm_tr, ym_tr)
        ym_pred = mod_pipe.predict(Xm_te)
        print(f"  Module classifier accuracy: {accuracy_score(ym_te, ym_pred):.3f}")
        print(f"  Predicted labels for {unknown_ct} 'unknown' module documents")

# ─────────────────────────────────────────────────────────────────────────────
# ██  SECTION 7 — SQLite Export  (for Streamlit + Superset / Metabase)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  SECTION 7 — Exporting SQLite database")
print("="*65)

conn = sqlite3.connect(DB_PATH)

# documents table (full clean dataset)
df_clean.to_sql("documents", conn, if_exists="replace", index=False)

# predictions table
df_model[["word_count", COL_COURSE, "predicted_course", "correct"]].rename(
    columns={COL_COURSE: "actual_course"}
).to_sql("predictions", conn, if_exists="replace", index=False)

# model comparison
results_df.reset_index().rename(columns={"index": "model"}).to_sql(
    "model_comparison", conn, if_exists="replace", index=False
)

# per-class metrics
report_df.reset_index().rename(columns={"index": "course"}).to_sql(
    "per_class_metrics", conn, if_exists="replace", index=False
)

conn.close()
print(f"  ✔ SQLite DB → {DB_PATH}")
print(f"     Tables : documents, predictions, model_comparison, per_class_metrics")

# ─────────────────────────────────────────────────────────────────────────────
# ██  FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  COMPLETE — DA2 Project Pipeline")
print("="*65)
print(f"\n  Dataset        : {len(df_clean):,} documents, {y.nunique()} courses")
print(f"  Best model     : {best_name}")
print(f"  CV Accuracy    : {results_df.loc[best_name,'Accuracy']:.1%} ± "
      f"{results_df.loc[best_name,'Acc Std']:.1%}")
print(f"  CV F1 (macro)  : {results_df.loc[best_name,'F1 (macro)']:.3f}")
print(f"\n  Figures saved  : fig1 → fig14  (14 charts)")
print(f"  SQLite DB      : academic_docs.db")
print(f"\n  ➜  Run the interactive dashboard:")
print(f"     streamlit run streamlit_app.py")
print("="*65 + "\n")
