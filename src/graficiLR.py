import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_multiple_metric_files(csv_files, labels, output_path, title=None):

    plt.figure(figsize=(10, 6))

    for csv_file, label in zip(csv_files, labels):
        df = pd.read_csv(csv_file)


        plt.errorbar(
            df["layer"],
            df["mean_accuracy"],
            yerr=df["std_accuracy"],
            fmt='-o',
            capsize=4,
            label=f"{label}",
            alpha=0.9
        )

    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.ylim(0.4, 1.0)
    plt.grid(True)

    if title:
        plt.title(title)

    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Grafico salvato in: {output_path}")

    plt.close()
    

csvs = [
    "data/output/metrics/other/sampled/BERT_ex1_UNK_other_240_avg_metrics.csv",
    "data/output/metrics/other/sampled/BERT_ex1_UNK_other_120_avg_metrics.csv",
    "data/output/metrics/other/sampled/BERT_ex1_UNK_other_60_avg_metrics.csv"
]

labels = [
    "240",
    "120",
    "60"
]

plot_multiple_metric_files(
    csv_files=csvs,
    labels=labels,
    output_path="data/prediction/simple_dataset/plot_all_metrics_simple.png",
    title="Confronto metriche per layer"
)
