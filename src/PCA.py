import numpy as np
import pandas as pd	
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import plotly.express as px


emb = pd.read_pickle("data/embeddings/bert/semantic/BERT_embedding_UNK_ex2_simple_train_2.pkl")
df = pd.read_csv("data/data_set/ex_2/simple/full/ex2_simple_train_2.csv", sep=";")

lemma_list = []
construction_list = []
semantic_list = []
type_list = []
sentence_list = []
syntax_list = []

for _, line in df.iterrows():


	lemma = line["noun"].strip()
	construction = line["construction"].strip()

	meaning = line["meaning"].strip()
	type = line["Type"].strip()
	print(line)
	sentence = line["context_pre"].strip() + " " + line["costr"].strip() + " " + line["context_post"].strip()
	syntax = line["syntactic_function"].strip()
 
	sentence_list.append(sentence)	
	type_list.append(type)
	construction_list.append(construction)
	semantic_list.append(meaning)
	lemma_list.append(lemma)
	syntax_list.append(syntax)

#X = np.array([np.array(e[f"UNK_layer_12"]) for e in emb])

dfs = []

layers = list(range(1, 13))

for layer in layers:
	X_layer = np.array([e[f"UNK_layer_{layer}"] for e in emb])

	pca = PCA(n_components=3)
	Y = pca.fit_transform(X_layer)

	df_layer = pd.DataFrame({
		"PCA1": Y[:, 0],
		"PCA2": Y[:, 1],
		"PCA3": Y[:, 2],
		"layer": layer,
		"construction": construction_list,
		"type": type_list,
		"lemma": lemma_list,
		"semantic": semantic_list,
		"syntax": syntax_list,
		"sentence": sentence_list,
	})

	dfs.append(df_layer)

df_plot = pd.concat(dfs, ignore_index=True)


# pca = PCA(n_components=3)
# y = pca.fit_transform(X)




print(pca.explained_variance_ratio_)

print(pca.singular_values_)


# unique_lemmas = sorted(set(lemma_list))
# lemma2id = {lemma: i for i, lemma in enumerate(unique_lemmas)}

# colors = [lemma2id[l] for l in lemma_list]




# plt.figure(figsize=(8, 6))
# scatter = plt.scatter(
#     y[:, 0],
#     y[:, 1],
#     c=colors,
#     cmap="tab20",
#     alpha=0.7
# )

# plt.xlabel("PCA 1")
# plt.ylabel("PCA 2")
# plt.title("PCA of BERT embeddings (colored by lemma)")
# plt.colorbar(scatter, label="Lemma ID")

# plt.show()

# plt.savefig("pca_bert_ex2_simple_lemmas.png")

# print(lemma2id.items())


# plt.figure(figsize=(8, 6))

# for lemma in unique_lemmas:
#     idx = [i for i, l in enumerate(lemma_list) if l == lemma]
#     points = y[idx]

#     plt.scatter(points[:, 0], points[:, 1], alpha=0.4)

#     centroid = points.mean(axis=0)
#     plt.scatter(
#         centroid[0], centroid[1],
#         marker="X", s=200, edgecolors="black"
#     )
#     plt.text(centroid[0], centroid[1], lemma, fontsize=10)

# plt.xlabel("PCA 1")
# plt.ylabel("PCA 2")
# plt.title("PCA centroids of lemmas")
# plt.show()

# plt.savefig("pca_bert_ex2_simple_lemmas.png")


# fig = plt.figure(figsize=(9, 7))
# ax = fig.add_subplot(111, projection="3d")

# for lemma in unique_lemmas:
#     idx = [i for i, l in enumerate(lemma_list) if l == lemma]
#     points = y[idx]

#     # punti
#     ax.scatter(
#         points[:, 0],
#         points[:, 1],
#         points[:, 2],
#         alpha=0.4
#     )

#     # centroide
#     centroid = points.mean(axis=0)
#     ax.scatter(
#         centroid[0],
#         centroid[1],
#         centroid[2],
#         marker="X",
#         s=200,
#         edgecolors="black"
#     )

#     ax.text(
#         centroid[0],
#         centroid[1],
#         centroid[2],
#         lemma,
#         fontsize=9
#     )

# ax.set_xlabel("PCA 1")
# ax.set_ylabel("PCA 2")
# ax.set_zlabel("PCA 3")
# ax.set_title("3D PCA of BERT embeddings with lemma centroids")

# plt.show()
# plt.savefig("pca3d_bert_ex2_simple_lemmas.png")



# df_plot = pd.DataFrame({
# 	"PCA1": y[:, 0],
# 	"PCA2": y[:, 1],
# 	"PCA3": y[:, 2],
# 	"type": type_list,
# 	"sentence": sentence_list,
# 	"semantic": semantic_list,
# 	"lemma": lemma_list,
# 	"syntax": syntax_list,
# })

# fig = px.scatter_3d(
# 	df_plot,
# 	x="PCA1",
# 	y="PCA2",
# 	z="PCA3",
# 	color="semantic",
# 	symbol="syntax",
# 	opacity=0.6,
# 	hover_data={
# 	"sentence": True,
# 	"PCA1": False,  # puoi anche nascondere coordinate se vuoi
# 	"PCA2": False,
# 	"PCA3": False
# },

# 	title="3D PCA of BERT embeddings by lemma"
# )

# fig.show()






# fig = px.scatter_3d(
# 	df_plot,
# 	x="PCA1",
# 	y="PCA2",
# 	z="PCA3",
# 	color="semantic",
# 	symbol="syntax",
# 	animation_frame="layer",
# 	opacity=0.6,
# 	hover_data={
# 		"sentence": True,
# 		"lemma": True,
# 		"layer": True,
# 		"PCA1": False,
# 		"PCA2": False,
# 		"PCA3": False
# 	},
# 	title="Evolution of BERT representations across layers"



# fig.update_layout(
#     transition=dict(duration=0),
# )

# fig.update_layout(
#     scene=dict(
#         xaxis=dict(range=[x_min, x_max], title="PCA 1"),
#         yaxis=dict(range=[y_min, y_max], title="PCA 2"),
#         zaxis=dict(range=[z_min, z_max], title="PCA 3"),
#     )
# )


x_min, x_max = df_plot["PCA1"].min(), df_plot["PCA1"].max()
y_min, y_max = df_plot["PCA2"].min(), df_plot["PCA2"].max()
z_min, z_max = df_plot["PCA3"].min(), df_plot["PCA3"].max()

fig = px.scatter_3d(
	df_plot,
	x="PCA1",
	y="PCA2",
	z="PCA3",
	color="semantic",
	animation_frame="layer",
	opacity=0.6,
	hover_data={
		"sentence": True,
		"lemma": False,
		"layer": False,
		"PCA1": False,
		"PCA2": False,
		"PCA3": False,
	},
	title="PCA indipendente per layer BERT"
)


fig.update_layout(
	transition={"duration": 0},
	scene=dict(
		xaxis=dict(range=[x_min, x_max], title="PCA 1"),
		yaxis=dict(range=[y_min, y_max], title="PCA 2"),
		zaxis=dict(range=[z_min, z_max], title="PCA 3"),
	),
	sliders=[{
		"currentvalue": {"prefix": "Layer BERT: "}
	}]
)

fig.show()


