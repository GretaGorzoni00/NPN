import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LinearSegmentedColormap

output_path = "data/output/predictions"
model = "BERT"
experiment = "ex1"
key = "UNK"
split_name = "other"

n_splits = 5
n_layers = 12
labels = ["no", "yes"]

# cmap personalizzato: rosso = errore, bianco = neutro, verde = corretto
colors = [(1,0,0), (1,1,1), (0,1,0)]  # rosso, bianco, verde
cmap = LinearSegmentedColormap.from_list("red_white_green", colors)

fig, axes = plt.subplots(n_splits, n_layers, figsize=(3*n_layers, 3*n_splits))

for split_idx in range(n_splits):
	csv_file = f"{output_path}/{model}_{experiment}_{key}_{split_name}_split{split_idx}_predictions.csv"
	df = pd.read_csv(csv_file)

	y_true = df["gold"].map({"no":0,"yes":1}).tolist()

	for layer_idx in range(1, n_layers+1):
		y_pred = df[f"layer_{layer_idx}"].map({"no":0,"yes":1}).tolist()

		cm = confusion_matrix(y_true, y_pred, labels=[1,0])  # invertiamo ordine per TP in alto a sx
		cm_norm = cm.astype("float") / cm.sum()  # percentuali

		# array colori: diagonale positiva (verde), fuori diagonale negativa (rosso)
		color_matrix = np.zeros_like(cm_norm)
		for i in range(2):
			for j in range(2):
				color_matrix[i,j] = cm_norm[i,j] if i==j else -cm_norm[i,j]

		ax = axes[split_idx, layer_idx-1] if n_splits>1 else axes[layer_idx-1]

		im = ax.imshow(color_matrix, cmap=cmap, vmin=-1, vmax=1)

		# annotazioni con percentuali
		for i in range(2):
			for j in range(2):
				ax.text(j, i, f"{cm_norm[i,j]*100:.1f}%", ha="center", va="center", color="black")

		if split_idx == 0:
			ax.set_title(f"Layer {layer_idx}", fontsize=10)
		if layer_idx == 1:
			ax.set_ylabel(f"Split {split_idx}", fontsize=10)

		ax.set_xticks([0,1])
		ax.set_xticklabels(["yes", "no"])  # pred
		ax.set_yticks([0,1])
		ax.set_yticklabels(["yes", "no"])  # true

plt.tight_layout()
plt.savefig(f"{output_path}/{model}_{experiment}_{key}_{split_name}_confusion_matrix", dpi=300)
plt.show()