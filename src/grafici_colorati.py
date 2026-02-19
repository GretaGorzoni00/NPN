import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_multiple_metric_files(csv_files, labels, output_path, title=None):

    plt.figure(figsize=(10, 6))

    strong_colors = [
        "#20b2aa",  # UNK → verde acqua (lightseagreen)
        "#1f77b4",  # PREP → blu
        "#4fc3f7"   # CLS → azzurro
    ]
    baseline_colors = [
        "#8f7bbf",  # viola-grigio più deciso
        "#b8a8d9",  # viola-grigio medio
        "#d8cfea",  # lilla chiaro
        "#f0edf5"   # quasi bianco con sottotono viola
    ]

    for i, (csv_file, label) in enumerate(zip(csv_files, labels)):
        df = pd.read_csv(csv_file)

        if i < 3:
            color = strong_colors[i]
            z = 3
        else:
            color = baseline_colors[i - 3]
            z = 1

        plt.errorbar(
            df["layer"],
            df["mean_accuracy"],
            yerr=df["std_accuracy"],
            fmt='-o',
            capsize=4,
            label=label,
            color=color,
            linewidth=1.8,   # stesso spessore per tutti
            alpha=0.95,
            zorder=z
        )

    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3)

    if title:
        plt.title(title)

    plt.legend(fontsize=8)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()




csvs = [
    "data/output/metrics/per_dopo/full/BERT_ex2_UNK___avg_metrics.csv",
    "data/output/metrics/per_dopo/full/BERT_ex2_PREP___avg_metrics.csv",
    "data/output/metrics/per_dopo/full/BERT_ex2_CLS___avg_metrics.csv",
    "data/output/metrics/itwac/per_dopo/full/itwac_ex2____avg_metrics.csv",
    "data/output/metrics/itwac/pre_lemma_per_dopo/full/itwac_ex2____avg_metrics.csv",
    "data/output/metrics/fasttext/per_dopo/full/fasttext_ex2____avg_metrics.csv",
    "data/output/metrics/fasttext/pre_lemma_per_dopo/full/fasttext_ex2____avg_metrics.csv"
]

labels = [
    "UNK per dopo",
    "PREP per dopo",
    "CLS per dopo",
    "BASELINE ITWAC NOUN per dopo",
    "BASELINE ITWAC PRE LEMMA per dopo",
    "BASELINE FASTTEXT NOUN per dopo",
    "BASELINE FASTTEXT PRE LEMMA per dopo"
     
]

plot_multiple_metric_files(
    csv_files=csvs,
    labels=labels,
    output_path="data/prediction/simple_dataset/plot_all_metrics_simple.png",
    title="Comparison of Evaluation Metrics Across Layers"
)