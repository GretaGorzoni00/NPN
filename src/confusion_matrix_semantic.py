import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# CONFIG — NEW EXP (SEMANTIC, side-by-side UNK vs PREP)
# =========================
DATASET_SEP = ";"

left = "ITWAC pre lemma"  # <-- just for title, not used in code (we use PRED_ prefix instea
right = "FASTTEXT pre lemma"  # <-- just for title, not used in code (we use PRED_ prefix instead)

# ---- UPDATE THESE PATHS FOR YOUR NEW EXPERIMENT ----
DATASET_PATHS = [
    "data/data_set/ex_2/simple/full/ex2_simple_test_0.csv",
    "data/data_set/ex_2/simple/full/ex2_simple_test_1.csv",
    "data/data_set/ex_2/simple/full/ex2_simple_test_2.csv",
    "data/data_set/ex_2/simple/full/ex2_simple_test_3.csv",
    "data/data_set/ex_2/simple/full/ex2_simple_test_4.csv",
]

# Predictions for UNK condition (same order, same row count as datasets)
PRED_PATHS_LEFT = [
    "data/output/predictions/itwac/pre_lemma_semantic/full/itwac_ex2__split0___predictions.csv",
    "data/output/predictions/itwac/pre_lemma_semantic/full/itwac_ex2__split1___predictions.csv",
    "data/output/predictions/itwac/pre_lemma_semantic/full/itwac_ex2__split2___predictions.csv",
    "data/output/predictions/itwac/pre_lemma_semantic/full/itwac_ex2__split3___predictions.csv",
    "data/output/predictions/itwac/pre_lemma_semantic/full/itwac_ex2__split4___predictions.csv",
]

# Predictions for PREP condition (same order, same row count as datasets)
PRED_PATHS_RIGHT = [
    "data/output/predictions/fasttext/pre_lemma_semantic/full/fasttext_ex2__split0___predictions.csv",
    "data/output/predictions/fasttext/pre_lemma_semantic/full/fasttext_ex2__split1___predictions.csv",
    "data/output/predictions/fasttext/pre_lemma_semantic/full/fasttext_ex2__split2___predictions.csv",
    "data/output/predictions/fasttext/pre_lemma_semantic/full/fasttext_ex2__split3___predictions.csv",
    "data/output/predictions/fasttext/pre_lemma_semantic/full/fasttext_ex2__split4___predictions.csv",
]

# ---- Gold columns in dataset ----
GOLD_PREP_COL = "preposition"   # e.g., "preposition"
GOLD_SEM_COL  = "meaning"       # e.g., "MEANING" / "meaning" / "semantic_value"

# ---- Prediction column name inside *_predictions.csv ----
# Put the EXACT column name that stores predicted semantic label
PRED_COL = "layer_12"           # <-- change if your file uses a different name

OUT_DIR = "data/output/confusion_matrices"
OUT_PNG = os.path.join(OUT_DIR, f"cm_semantic_rowpct_agg_5splits_{left}_vs_{right}.png")

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


def aggregate_counts_over_splits(dataset_paths, pred_paths, dataset_sep, pred_col):
    if len(dataset_paths) != len(pred_paths):
        raise ValueError("dataset_paths and pred_paths must have the same length.")

    agg_counts = None

    for ds_path, pr_path in zip(dataset_paths, pred_paths):
        ds = pd.read_csv(ds_path, sep=dataset_sep)
        pr = pd.read_csv(pr_path)

        for c in [GOLD_PREP_COL, GOLD_SEM_COL]:
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
            "gold_prep": ds[GOLD_PREP_COL].values,
            "gold_sem":  ds[GOLD_SEM_COL].values,
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
    out_png
):
    rows = list(mat_pct_left.index)
    cols = list(mat_pct_left.columns)

    mat_pct_right = mat_pct_right.reindex(index=rows, columns=cols).fillna(0.0)
    mat_counts_right = mat_counts_right.reindex(index=rows, columns=cols).fillna(0).astype(int)

    # ---- tighter figure size (a bit wider, less empty margins)
    fig_h = max(5, 0.45 * len(rows) + 2.2)
    fig_w = 13.6
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))
    # less left whitespace + keep room for colorbar
    fig.subplots_adjust(left=0.25, right=0.84, top=0.88, bottom=0.25, wspace=0.55)

    cmap = "BuPu"
    vmin, vmax = 0, 100

    imL = axes[0].imshow(mat_pct_left.values.astype(float), aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    imR = axes[1].imshow(mat_pct_right.values.astype(float), aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)

    # ---- ticks
    yticks = np.arange(len(rows))
    xticks = np.arange(len(cols))
    xticklabels = [c.replace("PRED_", "") for c in cols]

    for ax in axes:
        ax.set_yticks(yticks)
        ax.set_ylim(len(rows) - 0.5, -0.5)

        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize=11, rotation=25, ha="right")  # FIX overlap
        ax.tick_params(axis="x", pad=6)

        ax.set_yticks(np.arange(-.5, len(rows), 1), minor=True)
        ax.grid(which="minor", axis="y", linestyle="-", linewidth=0.5, alpha=0.25)
        ax.tick_params(which="minor", left=False)

        ax.set_xlabel(f"Predicted ({PRED_COL})", fontsize=12)

    axes[0].set_title(title_left, fontsize=14, pad=10)
    axes[1].set_title(title_right, fontsize=14, pad=10)
    axes[0].set_ylabel("Gold group (meaning + prep)", fontsize=12)

    # ---- Y tick labels only on left panel (full label + N)
    ylabels_left = []
    for r in rows:
        n = int(mat_counts_left.loc[r].sum()) if r in mat_counts_left.index else 0
        ylabels_left.append(f"{r} (N={n})")
    axes[0].set_yticklabels(ylabels_left, fontsize=11)
    axes[0].tick_params(axis="y", labelleft=True, labelright=False, pad=2)

    # Hide y tick labels on right panel (we add only N manually)
    axes[1].set_yticklabels([""] * len(rows))
    axes[1].tick_params(axis="y", left=False)

    # ---- annotate cells (percent)
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
                    bbox=dict(
                        boxstyle="round,pad=0.18",
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.75
                    )
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

    # ---- colorbar outside, but closer
    cax = fig.add_axes([0.87, 0.24, 0.02, 0.54])
    cbar = fig.colorbar(imR, cax=cax)
    cbar.set_label("Row % (within gold group)", fontsize=11)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"\nSaved side-by-side heatmap PNG: {out_png}")


# =========================
# MAIN
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not (len(DATASET_PATHS) == len(PRED_PATHS_LEFT) == len(PRED_PATHS_RIGHT)):
        raise ValueError("DATASET_PATHS, PRED_PATHS_LEFT, PRED_PATHS_RIGHT must have the same length.")

    # --- LEFT ---
    agg_counts_left = aggregate_counts_over_splits(DATASET_PATHS, PRED_PATHS_LEFT, DATASET_SEP, PRED_COL)
    agg_counts_left = agg_counts_left.reindex(sort_rows_sem(list(agg_counts_left.index))).fillna(0).astype(int)
    agg_counts_left = agg_counts_left.loc[agg_counts_left.sum(axis=1) > 0]
    agg_pct_left = row_normalize_to_percent(agg_counts_left)

    # --- RIGHT ---
    agg_counts_right = aggregate_counts_over_splits(DATASET_PATHS, PRED_PATHS_RIGHT, DATASET_SEP, PRED_COL)
    agg_counts_right = agg_counts_right.reindex(sort_rows_sem(list(agg_counts_right.index))).fillna(0).astype(int)
    agg_counts_right = agg_counts_right.loc[agg_counts_right.sum(axis=1) > 0]
    agg_pct_right = row_normalize_to_percent(agg_counts_right)

    # Align rows (union) and keep desired order
    all_rows = sort_rows_sem(sorted(set(agg_pct_left.index) | set(agg_pct_right.index)))
    agg_counts_left  = agg_counts_left.reindex(all_rows).fillna(0).astype(int)
    agg_pct_left     = agg_pct_left.reindex(all_rows).fillna(0.0)
    agg_counts_right = agg_counts_right.reindex(all_rows).fillna(0).astype(int)
    agg_pct_right    = agg_pct_right.reindex(all_rows).fillna(0.0)

    plot_two_matrices_side_by_side(
        mat_pct_left=agg_pct_left,
        mat_counts_left=agg_counts_left,
        title_left=f"{left} — aggregated 5 splits",
        mat_pct_right=agg_pct_right,
        mat_counts_right=agg_counts_right,
        title_right=f"{right} — aggregated 5 splits",
        out_png=OUT_PNG
    )


if __name__ == "__main__":
    main()
