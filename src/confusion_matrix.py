import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# HELPERS — NORMALIZATION
# =========================
def norm_prep(x):
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    return s if s in {"a", "su"} else None


def norm_semantic(x):
    """
    Normalize to 3 canonical labels:
      - SUCCESSION
      - ACCUMULATION
      - JUXTAPOSITION
    """
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    s = re.sub(r"\s+", "", s)

    # succession/distribution/iterativity
    if any(k in s for k in ["succession", "distribution", "distributivity", "iterativity", "iteration"]):
        return "SUCCESSION"

    # greater_plurality/accumulation
    if ("accumulation" in s) or ("greater_plurality" in s) or ("greaterplurality" in s):
        return "ACCUMULATION"

    # juxtaposition/contact
    if ("juxtaposition" in s) or ("contact" in s):
        return "JUXTAPOSITION"

    return None


# =========================
# GROUP BUILDERS — GOLD rows & PRED cols
# =========================
def build_gold_group_sem(gold_sem, gold_prep):
    """
    Requested gold rows:
      - SUCCESSION_a
      - SUCCESSION_su
      - ACCUMULATION_su
      - JUXTAPOSITION_a
    """
    sem = norm_semantic(gold_sem)
    prep = norm_prep(gold_prep)
    if sem is None or prep is None:
        return None

    allowed = {
        ("SUCCESSION", "a"),
        ("SUCCESSION", "su"),
        ("ACCUMULATION", "su"),
        ("JUXTAPOSITION", "a"),
    }
    if (sem, prep) not in allowed:
        return None

    return f"{sem}_{prep}"


def build_pred_group_sem(pred_sem):
    sem = norm_semantic(pred_sem)
    if sem is None:
        return None
    return f"PRED_{sem}"


# =========================
# MATRIX OPS
# =========================
def row_normalize_to_percent(mat_counts: pd.DataFrame) -> pd.DataFrame:
    mat = mat_counts.astype(float).copy()
    row_sums = mat.sum(axis=1).replace(0, np.nan)
    mat = mat.div(row_sums, axis=0) * 100.0
    return mat.fillna(0.0)


def sort_rows_sem(index_list):
    order = ["SUCCESSION_a", "SUCCESSION_su", "ACCUMULATION_su", "JUXTAPOSITION_a"]
    pos = {k: i for i, k in enumerate(order)}
    return sorted(index_list, key=lambda x: pos.get(x, 999))


def aggregate_counts_over_splits(dataset_paths, pred_paths, dataset_sep,
                                pred_col, gold_prep_col, gold_sem_col):

    agg_counts = None

    for ds_path, pr_path in zip(dataset_paths, pred_paths):
        ds = pd.read_csv(ds_path, sep=dataset_sep)
        pr = pd.read_csv(pr_path)

        for c in [gold_prep_col, gold_sem_col]:
            if c not in ds.columns:
                raise ValueError(f"Missing '{c}' in dataset {ds_path}. Columns: {list(ds.columns)}")
        if pred_col not in pr.columns:
            raise ValueError(f"Missing '{pred_col}' in prediction file {pr_path}. Columns: {list(pr.columns)}")

        if len(ds) != len(pr):
            raise ValueError(
                f"Row mismatch: dataset={len(ds)} pred={len(pr)}\n{ds_path}\n{pr_path}\n"
                f"To compare row-by-row they must match."
            )

        df = pd.DataFrame({
            "gold_prep": ds[gold_prep_col].values,
            "gold_sem":  ds[gold_sem_col].values,
            "pred_sem":  pr[pred_col].values
        })

        df["GOLD_GROUP"] = df.apply(lambda r: build_gold_group_sem(r["gold_sem"], r["gold_prep"]), axis=1)
        df["PRED_GROUP"] = df["pred_sem"].map(build_pred_group_sem)
        df = df[df["GOLD_GROUP"].notna() & df["PRED_GROUP"].notna()]

        mat = pd.crosstab(df["GOLD_GROUP"], df["PRED_GROUP"], dropna=False)

        for col in ["PRED_SUCCESSION", "PRED_ACCUMULATION", "PRED_JUXTAPOSITION"]:
            if col not in mat.columns:
                mat[col] = 0
        mat = mat[["PRED_SUCCESSION", "PRED_ACCUMULATION", "PRED_JUXTAPOSITION"]]

        agg_counts = mat if agg_counts is None else agg_counts.add(mat, fill_value=0)
        print(f"[OK] Aggregated: {ds_path} + {pr_path} | valid_rows={len(df)}")

    return agg_counts.fillna(0).astype(int)


# =========================
# PLOT — SIDE BY SIDE (UNK vs PREP) like original
# =========================
def plot_two_matrices_side_by_side(
    mat_pct_left, mat_counts_left, title_left,
    mat_pct_right, mat_counts_right, title_right,
    pred_col,
    out_png
):
    rows = list(mat_pct_left.index)
    cols = list(mat_pct_left.columns)

    mat_pct_right = mat_pct_right.reindex(index=rows, columns=cols).fillna(0.0)
    mat_counts_right = mat_counts_right.reindex(index=rows, columns=cols).fillna(0).astype(int)

    fig_h = max(5, 0.45 * len(rows) + 2.8)
    fig_w = 14
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))
    fig.subplots_adjust(left=0.38, right=0.82, top=0.88, wspace=0.70)

    cmap = "BuPu"
    vmin, vmax = 0, 100

    imL = axes[0].imshow(mat_pct_left.values.astype(float), aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    imR = axes[1].imshow(mat_pct_right.values.astype(float), aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)

    yticks = np.arange(len(rows))
    for ax in axes:
        ax.set_yticks(yticks)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_xticks(np.arange(len(cols)))
        ax.set_xticklabels([c.replace("PRED_", "") for c in cols], fontsize=12)

        ax.set_yticks(np.arange(-.5, len(rows), 1), minor=True)
        ax.grid(which="minor", axis="y", linestyle="-", linewidth=0.5, alpha=0.25)
        ax.tick_params(which="minor", left=False)

        ax.set_xlabel(f"Predicted ({pred_col})", fontsize=12)

    axes[0].set_title(title_left, fontsize=14, pad=10)
    axes[1].set_title(title_right, fontsize=14, pad=10)
    axes[0].set_ylabel("Gold group (meaning + prep)", fontsize=12)

    # Left y labels = full label + N
    ylabels_left = []
    for r in rows:
        n = int(mat_counts_left.loc[r].sum()) if r in mat_counts_left.index else 0
        ylabels_left.append(f"{r}  (N={n})")
    axes[0].set_yticklabels(ylabels_left, fontsize=11)
    axes[0].tick_params(axis="y", labelleft=True, labelright=False, pad=2)

    # On PREP panel, draw (N=...) to the left (like your original trick)
    for i, r in enumerate(rows):
        n = int(mat_counts_right.loc[r].sum()) if r in mat_counts_right.index else 0
        y_ax = 1 - (i + 0.5) / len(rows)
        axes[1].text(
            -0.22, y_ax,
            f"(N={n})",
            ha="right", va="center",
            fontsize=11,
            transform=axes[1].transAxes
        )

    def annotate(ax, mat):
        data = mat.astype(float)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                ax.text(
                    j, i, f"{v:.0f}%",
                    ha="center", va="center",
                    fontsize=11,
                    color="black",
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.75)
                )

    annotate(axes[0], mat_pct_left.values)
    annotate(axes[1], mat_pct_right.values)

    # ---- PREP panel: add (N=...) to the left of the right panel (like your original)
    for i, r in enumerate(rows):
        n = int(mat_counts_right.loc[r].sum()) if r in mat_counts_right.index else 0
        y_ax = 1 - (i + 0.5) / len(rows)
        axes[1].text(
            -0.12, y_ax, f"(N={n})",  # closer (less empty space)
            ha="right", va="center",
            fontsize=11,
            transform=axes[1].transAxes
        )

    # Colorbar OUTSIDE
    cax = fig.add_axes([0.86, 0.22, 0.02, 0.56])
    cbar = fig.colorbar(imR, cax=cax)
    cbar.set_label("Row % (within gold group)", fontsize=11)

    # os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"\nSaved side-by-side heatmap PNG: {out_png}")


# =========================
# MAIN
# =========================
def main(dataset_paths, dataset_unk, dataset_prep, pred_col, output_dir,
        sep, gold_prep_col, gold_sem_col,
        left_title, right_title):

    # --- UNK ---
    agg_counts_unk = aggregate_counts_over_splits(dataset_paths, dataset_unk, sep, pred_col, gold_prep_col, gold_sem_col)
    agg_counts_unk = agg_counts_unk.reindex(sort_rows_sem(list(agg_counts_unk.index))).fillna(0).astype(int)
    agg_counts_unk = agg_counts_unk.loc[agg_counts_unk.sum(axis=1) > 0]
    agg_pct_unk = row_normalize_to_percent(agg_counts_unk)

    # --- PREP ---
    agg_counts_prep = aggregate_counts_over_splits(dataset_paths, dataset_prep, sep, pred_col, gold_prep_col, gold_sem_col)
    agg_counts_prep = agg_counts_prep.reindex(sort_rows_sem(list(agg_counts_prep.index))).fillna(0).astype(int)
    agg_counts_prep = agg_counts_prep.loc[agg_counts_prep.sum(axis=1) > 0]
    agg_pct_prep = row_normalize_to_percent(agg_counts_prep)

    # Align rows (union) and keep desired order
    all_rows = sort_rows_sem(sorted(set(agg_pct_unk.index) | set(agg_pct_prep.index)))
    agg_counts_unk  = agg_counts_unk.reindex(all_rows).fillna(0).astype(int)
    agg_pct_unk     = agg_pct_unk.reindex(all_rows).fillna(0.0)
    agg_counts_prep = agg_counts_prep.reindex(all_rows).fillna(0).astype(int)
    agg_pct_prep    = agg_pct_prep.reindex(all_rows).fillna(0.0)

    out_png = os.path.join(output_dir, f"cm_semantic_rowpct_agg_5splits_{left_title}_vs_{right_title}.png")

    plot_two_matrices_side_by_side(
        mat_pct_left=agg_pct_unk,
        mat_counts_left=agg_counts_unk,
        title_left=left_title,
        mat_pct_right=agg_pct_prep,
        mat_counts_right=agg_counts_prep,
        title_right=right_title,
        pred_col=pred_col,
        out_png=out_png
    )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Aggregate confusion matrices across splits and plot side-by-side.")
    parser.add_argument("--dataset_sep", type=str, default=";",
                        help="Separator used in dataset CSV files.")
    parser.add_argument("--dataset_paths", nargs="+", required=True,
                        help="List of dataset CSV file paths (must match pred_paths).")
    parser.add_argument("--pred_paths_unk", nargs="+", required=True,
                        help="List of UNK prediction CSV file paths (must match dataset_paths).")
    parser.add_argument("--pred_paths_prep", nargs="+", required=True,
                        help="List of PREP prediction CSV file paths (must match dataset_paths).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory path.")
    parser.add_argument("--pred_col", type=str, default="layer_12",
                        help="Column name in prediction CSV that contains predicted semantic label.")
    parser.add_argument("--gold_prep_col", type=str, default="preposition",
                        help="Column name in dataset CSV that contains gold preposition.")
    parser.add_argument("--gold_sem_col", type=str, default="MEANING",
                        help="Column name in dataset CSV that contains gold semantic label.")
    parser.add_argument("--left_title", type=str, default="UNK — aggregated 5 splits",
                        help="Title for the left (UNK) matrix.")
    parser.add_argument("--right_title", type=str, default="PREP — aggregated 5 splits",
                        help="Title for the right (PREP) matrix.")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    assert(len(args.dataset_paths) == len(args.pred_paths_unk) == len(args.pred_paths_prep)), "dataset_paths, pred_paths_unk, pred_paths_prep must have the same length."


    main(args.dataset_paths, args.pred_paths_unk, args.pred_paths_prep, args.pred_col,
        args.output_dir,
        args.dataset_sep, args.gold_prep_col, args.gold_sem_col,
        left_title=args.left_title, right_title=args.right_title)


    # # =========================
    # # CONFIG — NEW EXP (SEMANTIC, side-by-side UNK vs PREP)
    # # =========================
    # DATASET_SEP = ";"

    # # ---- UPDATE THESE PATHS FOR YOUR NEW EXPERIMENT ----
    # DATASET_PATHS = [
    #     "data/data_set/ex_2/semantic/full/ex2_semantic_test_0.csv",
    #     "data/data_set/ex_2/semantic/full/ex2_semantic_test_1.csv",
    #     "data/data_set/ex_2/semantic/full/ex2_semantic_test_2.csv",
    #     "data/data_set/ex_2/semantic/full/ex2_semantic_test_3.csv",
    #     "data/data_set/ex_2/semantic/full/ex2_semantic_test_4.csv",
    # ]

    # # Predictions for UNK condition (same order, same row count as datasets)
    # PRED_PATHS_UNK = [
    #     "data/output/predictions/bert/semantic/full/BERT_ex2_UNK_split0___predictions.csv",
    #     "data/output/predictions/bert/semantic/full/BERT_ex2_UNK_split1___predictions.csv",
    #     "data/output/predictions/bert/semantic/full/BERT_ex2_UNK_split2___predictions.csv",
    #     "data/output/predictions/bert/semantic/full/BERT_ex2_UNK_split3___predictions.csv",
    #     "data/output/predictions/bert/semantic/full/BERT_ex2_UNK_split4___predictions.csv",
    # ]

    # # Predictions for PREP condition (same order, same row count as datasets)
    # PRED_PATHS_PREP = [
    #     "data/output/predictions/bert/semantic/full/BERT_ex2_PREP_split0___predictions.csv",
    #     "data/output/predictions/bert/semantic/full/BERT_ex2_PREP_split1___predictions.csv",
    #     "data/output/predictions/bert/semantic/full/BERT_ex2_PREP_split2___predictions.csv",
    #     "data/output/predictions/bert/semantic/full/BERT_ex2_PREP_split3___predictions.csv",
    #     "data/output/predictions/bert/semantic/full/BERT_ex2_PREP_split4___predictions.csv",
    # ]

    # OUT_DIR = "data/output/confusion_matrices"

    # OUT_PNG = os.path.join(OUT_DIR, "cm_semantic_rowpct_agg_5splits_UNK_vs_PREP.png")
