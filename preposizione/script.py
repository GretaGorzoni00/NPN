#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Loading / normalization
# -----------------------------
def read_semicolon_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    if df.shape[1] <= 1:
        raise ValueError(f"File seems not ';' separated: {path}")
    return df


def norm_tok(x) -> str:
    """Normalize tokens (prepositions and punctuation-like outputs)."""
    if pd.isna(x):
        return "<nan>"
    s = str(x).strip()
    if s == "":
        return "<empty>"
    s = s.replace("’", "'").replace("`", "'")
    return s.lower()


def meaning_to_label(meaning: str) -> str:
    """
    Map MEANING field to the labels used in plots.
    Adjust here if you add meanings.
    """
    m = str(meaning).strip().lower()
    if m == "juxtaposition/contact":
        return "Juxtaposition"
    if m == "succession/iteration/distributivity":
        return "Succession"
    if m == "greater_plurality/accumulation":
        return "GreaterAccum"
    return str(meaning).split("/")[0].strip().title()


def build_category(df: pd.DataFrame, gold_col: str, meaning_col: str) -> pd.Series:
    prep = df[gold_col].map(norm_tok).str.upper()
    meaning_lbl = df[meaning_col].map(meaning_to_label)
    return prep + " / " + meaning_lbl


# -----------------------------
# Metrics + error matrices
# -----------------------------
def compute_correct(df: pd.DataFrame, gold_col: str, pred_col: str) -> pd.Series:
    gold = df[gold_col].map(norm_tok)
    pred = df[pred_col].map(norm_tok)
    return pred.eq(gold)


def accuracy_by_category(df: pd.DataFrame, category_col: str, correct: pd.Series) -> pd.Series:
    tmp = df[[category_col]].copy()
    tmp["CORRECT"] = correct.values
    return (tmp.groupby(category_col)["CORRECT"].mean() * 100.0)


def error_matrix_restricted_tokens(df: pd.DataFrame,
                                  category_col: str,
                                  gold_col: str,
                                  pred_col: str,
                                  categories: List[str],
                                  error_tokens: List[str]) -> np.ndarray:
    """
    Matrix: rows=categories, cols=error_tokens (restricted list),
    values=counts of incorrect predictions whose predicted token is in error_tokens.
    """
    gold = df[gold_col].map(norm_tok)
    pred = df[pred_col].map(norm_tok)

    mat = np.zeros((len(categories), len(error_tokens)), dtype=int)
    cat_idx = {c: i for i, c in enumerate(categories)}
    tok_idx = {t: j for j, t in enumerate(error_tokens)}

    for cat, g, p in zip(df[category_col], gold, pred):
        if p == g:
            continue
        if cat in cat_idx and p in tok_idx:
            mat[cat_idx[cat], tok_idx[p]] += 1

    return mat


def aggregated_annotators_error_matrix_restricted(df: pd.DataFrame,
                                                 category_col: str,
                                                 gold_col: str,
                                                 annot_cols: List[str],
                                                 categories: List[str],
                                                 error_tokens: List[str]) -> np.ndarray:
    mat = np.zeros((len(categories), len(error_tokens)), dtype=int)
    for col in annot_cols:
        mat += error_matrix_restricted_tokens(df, category_col, gold_col, col, categories, error_tokens)
    return mat


# -----------------------------
# Plotting helpers
# -----------------------------
def plot_two_heatmaps(m1: np.ndarray, m2: np.ndarray,
                      row_labels: List[str], col_labels: List[str],
                      title1: str, title2: str,
                      outpath: str):
    fig, axs = plt.subplots(1, 2, figsize=(18, 5))

    for ax, mat, title, cmap in [
        (axs[0], m1, title1, "Reds"),
        (axs[1], m2, title2, "Blues"),
    ]:
        vmax = int(mat.max()) if mat.size else None
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)

        ax.set_title(title)
        ax.set_ylabel("Category")
        ax.set_xlabel("Predicted token (non-target predictions type)")

        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels)

        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=90)

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if v != 0:
                    ax.text(j, i, str(v), ha="center", va="center", fontsize=8)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_boxplot_variability(annot_acc: Dict[str, pd.Series],
                             bert_acc: pd.Series,
                             categories: List[str],
                             outpath: str,
                             title: str):
    data, means, bert_points = [], [], []

    for c in categories:
        vals = [float(annot_acc[a].get(c, np.nan)) for a in annot_acc]
        vals = [v for v in vals if not np.isnan(v)]
        data.append(vals)
        means.append(np.mean(vals) if len(vals) else np.nan)
        bert_points.append(float(bert_acc.get(c, np.nan)))

    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    ax.boxplot(data, positions=np.arange(1, len(categories) + 1), widths=0.5, patch_artist=False)
    ax.scatter(np.arange(1, len(categories) + 1), bert_points, marker="o", label="BERT")
    ax.scatter(np.arange(1, len(categories) + 1), means, marker="D", label="Mean annotators")

    ax.set_title(title)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(np.arange(1, len(categories) + 1))
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_bar_bert_vs_mean(bert_acc: pd.Series,
                          annot_acc: Dict[str, pd.Series],
                          categories: List[str],
                          outpath: str,
                          title: str):
    means, bert_vals = [], []

    for c in categories:
        bert_vals.append(float(bert_acc.get(c, np.nan)))
        vals = [float(annot_acc[a].get(c, np.nan)) for a in annot_acc]
        vals = [v for v in vals if not np.isnan(v)]
        means.append(np.mean(vals) if len(vals) else np.nan)

    x = np.arange(len(categories))
    width = 0.35

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.bar(x - width/2, bert_vals, width, label="BERT")
    ax.bar(x + width/2, means, width, label="Mean annotators")

    for i, v in enumerate(bert_vals):
        if not np.isnan(v):
            ax.text(i - width/2, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    for i, v in enumerate(means):
        if not np.isnan(v):
            ax.text(i + width/2, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_title(title)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_stacked_correct_error_all_systems(df: pd.DataFrame,
                                          category_col: str,
                                          gold_col: str,
                                          systems: List[Tuple[str, str]],
                                          categories: List[str],
                                          outpath: str,
                                          title: str):
    plt.figure(figsize=(14, 6))
    ax = plt.gca()

    x = np.arange(len(systems))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(categories))

    gold_norm = df[gold_col].map(norm_tok)

    # --- color palette per categorie (safe, colorblind-friendly)
    category_colors = plt.cm.tab10.colors
    cat_color_map = {
        cat: category_colors[i % len(category_colors)]
        for i, cat in enumerate(categories)
    }

    correct_handle = None
    incorrect_handle = None
    category_handles = []

    for ci, cat in enumerate(categories):
        corrects, errors = [], []
        mask_cat = (df[category_col] == cat)
        n_cat = int(mask_cat.sum())

        for _, col in systems:
            if col is None:
                corr = n_cat
            else:
                pred = df[col].map(norm_tok)
                corr = int((pred[mask_cat] == gold_norm[mask_cat]).sum())
            corrects.append(corr)
            errors.append(n_cat - corr)

        # stacked bars (same color for category)
        b1 = ax.bar(
            x + offsets[ci],
            corrects,
            width=width,
            color=cat_color_map[cat],
            label="Correct" if correct_handle is None else None
        )
        b2 = ax.bar(
            x + offsets[ci],
            errors,
            width=width,
            bottom=corrects,
            color=cat_color_map[cat],
            alpha=0.4,
            label="Incorrect" if incorrect_handle is None else None
        )

        if correct_handle is None:
            correct_handle = b1
        if incorrect_handle is None:
            incorrect_handle = b2

        # category legend handle (one per category)
        category_handles.append(
            plt.Line2D(
                [0], [0],
                color=cat_color_map[cat],
                lw=6,
                label=cat
            )
        )

    ax.set_title(title)
    ax.set_ylabel("Number of items")
    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in systems])
    ax.grid(True, axis="y", alpha=0.3)

    # --- legend 1: Correct vs Incorrect
    legend1 = ax.legend(
        handles=[correct_handle[0], incorrect_handle[0]],
        labels=["Target", "Non-target"],
        loc="upper left",
        frameon=True,
        title="Prediction outcome"
    )

    # --- legend 2: Categories (colors)
    legend2 = ax.legend(
        handles=category_handles,
        loc="upper right",
        frameon=True,
        title="Category"
    )

    ax.add_artist(legend1)  # keep both legends

    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=300)
    plt.close()



# -----------------------------
# Main
# -----------------------------
def main(input_csv: str,
         outdir: str,
         gold_col: str,
         bert_col: str,
         meaning_col: str,
         annotator_cols: List[str],
         heatmap_tokens: List[str]):

    os.makedirs(outdir, exist_ok=True)
    df = read_semicolon_csv(input_csv)

    needed = {gold_col, meaning_col, bert_col} | set(annotator_cols)
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input CSV: {missing}\nFound: {list(df.columns)}")

    df = df.copy()
    df["CATEGORY"] = build_category(df, gold_col=gold_col, meaning_col=meaning_col)

    categories = sorted(df["CATEGORY"].unique().tolist(),
                        key=lambda x: (x.split("/")[0].strip(), x.split("/")[1].strip()))

    # Heatmap x-axis: restricted tokens, alphabetical
    error_tokens = sorted([t.lower().strip() for t in heatmap_tokens])

    # --- Heatmaps
    bert_m = error_matrix_restricted_tokens(df, "CATEGORY", gold_col, bert_col, categories, error_tokens)
    annot_m = aggregated_annotators_error_matrix_restricted(df, "CATEGORY", gold_col, annotator_cols, categories, error_tokens)

    plot_two_heatmaps(
        bert_m, annot_m,
        row_labels=categories,
        col_labels=error_tokens,
        title1="BERT: distribution of recurrent non-target predictions types by category",
        title2="Human annotators (aggregated): distribution of recurrent non-target predictions types by category",
        outpath=os.path.join(outdir, "heatmaps_errors_bert_vs_annotators.png")
    )

    # --- Accuracies
    bert_acc = accuracy_by_category(df, "CATEGORY", compute_correct(df, gold_col, bert_col))
    annot_acc = {a: accuracy_by_category(df, "CATEGORY", compute_correct(df, gold_col, a)) for a in annotator_cols}

    # --- Boxplot
    plot_boxplot_variability(
        annot_acc=annot_acc,
        bert_acc=bert_acc,
        categories=categories,
        outpath=os.path.join(outdir, "boxplot_variability_annotators_plus_bert.png"),
        title="Inter-annotator variability and BERT accuracy across categories"
    )

    # --- Bar BERT vs mean annotators
    plot_bar_bert_vs_mean(
        bert_acc=bert_acc,
        annot_acc=annot_acc,
        categories=categories,
        outpath=os.path.join(outdir, "bar_bert_vs_mean_annotators.png"),
        title="BERT versus mean human accuracy across categories"
    )
    
    SYSTEM_LABELS = {
    "BERT": "BERT",
    "MARTYNA": "MP",
    "FRANESCO": "FMa",
    "FEDERICO": "FMi",
    "SARA": "SL",
    "LARA": "LS",
    }

    # --- Stacked correct/error for TARGET  +BERT+annotators
    systems = [("TARGET", None), ("BERT", bert_col)] + [(SYSTEM_LABELS[a], a) for a in annotator_cols]
    plot_stacked_correct_error_all_systems(
        df=df,
        category_col="CATEGORY",
        gold_col=gold_col,
        systems=systems,
        categories=categories,
        outpath=os.path.join(outdir, "stacked_correct_error_all_systems.png"),
        title="Target vs non-target predictions by category for BERT and human annotators"
    )
    
    # --- Save summaries
    rows = []
    for c in categories:
        rows.append({"system": "BERT", "category": c, "accuracy_pct": float(bert_acc.get(c, np.nan))})
        for a in annotator_cols:
            rows.append({"system": a, "category": c, "accuracy_pct": float(annot_acc[a].get(c, np.nan))})
        vals = [float(annot_acc[a].get(c, np.nan)) for a in annotator_cols]
        vals = [v for v in vals if not np.isnan(v)]
        rows.append({"system": "MEAN_ANNOTATORS", "category": c, "accuracy_pct": float(np.mean(vals)) if vals else np.nan})
    pd.DataFrame(rows).to_csv(os.path.join(outdir, "accuracy_by_system_and_category.csv"), index=False)

    # --- Robust restricted-token error counts
    def token_error_counts_restricted(pred_col: str, allowed_tokens: List[str]) -> pd.DataFrame:
        """
        Always returns columns: pred_token, count (possibly empty).
        Only counts errors whose predicted token is in allowed_tokens.
        """
        gold = df[gold_col].map(norm_tok)
        pred = df[pred_col].map(norm_tok)
        wrong = pred[~pred.eq(gold)]
        wrong = wrong[wrong.isin(allowed_tokens)]

        vc = wrong.value_counts()
        out = pd.DataFrame({"pred_token": vc.index.tolist(), "count": vc.values.tolist()})
        if out.empty:
            return pd.DataFrame({"pred_token": [], "count": []})
        return out

    token_error_counts_restricted(bert_col, error_tokens).to_csv(
        os.path.join(outdir, "error_tokens_BERT_restricted.csv"), index=False
    )

    agg = {t: 0 for t in error_tokens}
    for a in annotator_cols:
        tmp = token_error_counts_restricted(a, error_tokens)
        # SAFE: iterate by columns (works also if empty)
        for tok, cnt in zip(tmp["pred_token"].tolist(), tmp["count"].tolist()):
            agg[tok] += int(cnt)

    pd.DataFrame({"pred_token": list(agg.keys()), "count": list(agg.values())}) \
        .sort_values(["count", "pred_token"], ascending=[False, True]) \
        .to_csv(os.path.join(outdir, "error_tokens_ANNOTATORS_AGG_restricted.csv"), index=False)

    print("\nDONE ✅ Outputs written to:", outdir)
    print(" - heatmaps_errors_bert_vs_annotators.png")
    print(" - boxplot_variability_annotators_plus_bert.png")
    print(" - bar_bert_vs_mean_annotators.png")
    print(" - stacked_correct_error_all_systems.png")
    print(" - accuracy_by_system_and_category.csv")
    print(" - error_tokens_BERT_restricted.csv")
    print(" - error_tokens_ANNOTATORS_AGG_restricted.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Wide CSV (separator ';') with GOLD, BERT and annotators.")
    parser.add_argument("--outdir", required=True, help="Output directory for plots and summaries.")
    parser.add_argument("--gold_col", default="GOLD")
    parser.add_argument("--bert_col", default="BERT")
    parser.add_argument("--meaning_col", default="MEANING")
    parser.add_argument("--annotators", nargs="+",
                        default=["MARTYNA", "FRANESCO", "FEDERICO", "SARA", "LARA"],
                        help="Annotator columns to compare against GOLD.")
    parser.add_argument("--heatmap_tokens", nargs="+",
                        default=["a", "con", "contro", "dopo", "per", "su"],
                        help="Restricted token set for the heatmap x-axis (alphabetical).")
    args = parser.parse_args()

    main(
        input_csv=args.input,
        outdir=args.outdir,
        gold_col=args.gold_col,
        bert_col=args.bert_col,
        meaning_col=args.meaning_col,
        annotator_cols=args.annotators,
        heatmap_tokens=args.heatmap_tokens
    )
