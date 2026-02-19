import pandas as pd
import matplotlib.pyplot as plt
import os

emb = "UNK"
metric = "precision"

# --- CONFIG ---
CSV_PATH = "data/output/metrics/semantic/full/BERT_ex2_UNK___avg_metrics_by_label.csv"  # <-- change to your file path
OUTPUT_PATH = f"data/output/graphs/semantic/full/{emb}_{metric}_by_label.png"  # <-- where you want to save it

LAYER_COL = "layer"
LABEL_COL = "label"
METRIC_COL = f"mean_{metric}"
METRICL_ERR_COL = f"std_{metric}"   # set to None if you don’t want error bars

# --- LOAD ---
df = pd.read_csv(CSV_PATH)

df[LAYER_COL] = pd.to_numeric(df[LAYER_COL], errors="coerce")
df[METRIC_COL] = pd.to_numeric(df[METRIC_COL], errors="coerce")

if METRICL_ERR_COL is not None and METRICL_ERR_COL in df.columns:
    df[METRICL_ERR_COL] = pd.to_numeric(df[METRICL_ERR_COL], errors="coerce")

df = df.sort_values([LABEL_COL, LAYER_COL])

# --- PLOT ---
plt.figure(figsize=(10, 6))

for lab, sub in df.groupby(LABEL_COL, sort=False):
    x = sub[LAYER_COL].values
    y = sub[METRIC_COL].values

    if METRICL_ERR_COL is not None and METRICL_ERR_COL in sub.columns:
        yerr = sub[METRICL_ERR_COL].values
        plt.errorbar(x, y, yerr=yerr, marker="o", capsize=3, label=lab)
    else:
        plt.plot(x, y, marker="o", label=lab)

plt.xlabel("Layer")
plt.ylabel(f"Mean {metric}")
plt.title(f"{metric.capitalize()} {emb} per label across layers")
plt.ylim(0, 1.0)
plt.xticks(sorted(df[LAYER_COL].unique()))
plt.grid(True)
plt.legend()
plt.tight_layout()

# --- SAVE ---
os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=300)
print(f"Figure saved in: {OUTPUT_PATH}")

plt.close()
