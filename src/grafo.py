# ======================================
# Grafo preposizioni–lemmi (NPN)
# Lemmi colorati in base al significato (colori chiari)
# ======================================

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
import matplotlib.colors as mc

# --------------------------------------
# FUNZIONE PER SCHIARIRE I COLORI
# --------------------------------------
def lighten_color(color, amount=0.6):
    """
    Schiarisce un colore mescolandolo con il bianco.
    amount ∈ [0,1]: più alto = più chiaro
    """
    c = np.array(mc.to_rgb(color))
    return tuple(c + (1.0 - c) * amount)

# --------------------------------------
# 1. CARICAMENTO DATI
# --------------------------------------
df = pd.read_csv("lemmi_NPN.csv", sep=";")

MIN_FREQ = 4       # frequenza minima per visualizzare nodo
LABEL_FREQ = 20    # frequenza minima per mostrare etichetta

df = df[df["token_frequency"] >= MIN_FREQ]

# Significati
unique_meanings = df["meaning"].unique()

# Colori base → versione chiara/pastello
base_colors = plt.cm.tab20.colors
meaning_colors = {
    m: lighten_color(c, amount=0.6)
    for m, c in zip(unique_meanings, base_colors)
}

# --------------------------------------
# 2. COSTRUZIONE DEL GRAFO
# --------------------------------------
G = nx.Graph()

for _, row in df.iterrows():
    prep = row["preposition"]
    lemma = row["reduplicated_noun"]
    freq = int(row["token_frequency"])
    meaning = row["meaning"]

    if not G.has_node(prep):
        G.add_node(
            prep,
            node_type="preposition",
            size=2600,
            freq=freq
        )

    if not G.has_node(lemma):
        node_size = np.log1p(freq) * 120
        G.add_node(
            lemma,
            node_type="lemma",
            size=node_size,
            freq=freq,
            meaning=meaning
        )

    G.add_edge(prep, lemma, weight=freq)

# --------------------------------------
# 3. LAYOUT
# --------------------------------------
pos = nx.spring_layout(
    G,
    k=4.6,
    iterations=300,
    seed=42
)

# --------------------------------------
# 3b. DISTANZIA I LEMMI DALLE PREPOSIZIONI
# --------------------------------------
for u, v in G.edges():
    if G.nodes[u]["node_type"] == "preposition":
        prep, lemma = u, v
    else:
        prep, lemma = v, u

    vec = pos[lemma] - pos[prep]
    pos[lemma] = pos[prep] + 0.5 * vec + 0.05 * np.random.randn(2)

# --------------------------------------
# 4. COLORI E DIMENSIONI NODI
# --------------------------------------
node_colors = []
node_sizes = []

for node, data in G.nodes(data=True):
    if data["node_type"] == "preposition":
        node_colors.append("#7b4fa3")  # viola più scuro
        node_sizes.append(data["size"])
    else:
        node_colors.append(meaning_colors.get(data["meaning"], "#cccccc"))
        node_sizes.append(data["size"])

# --------------------------------------
# 5. PLOT
# --------------------------------------
plt.figure(figsize=(16, 14))
ax = plt.gca()

# EDGES
segments = [[pos[u], pos[v]] for u, v in G.edges()]
lc = LineCollection(
    segments,
    colors="gray",
    linewidths=1.2,
    alpha=0.35
)
ax.add_collection(lc)

# NODES (con bordo per leggibilità)
nx.draw_networkx_nodes(
    G,
    pos,
    node_color=node_colors,
    node_size=node_sizes,
    edgecolors="k",
    linewidths=0.4,
    ax=ax
)

# LABELS PREPOSIZIONI
prep_labels = {
    n: n for n, d in G.nodes(data=True)
    if d["node_type"] == "preposition"
}
nx.draw_networkx_labels(
    G,
    pos,
    labels=prep_labels,
    font_size=16,
    font_weight="bold",
    font_family="serif",
    ax=ax
)

# LABELS LEMMI (solo frequenti)
lemma_labels = {
    n: n for n, d in G.nodes(data=True)
    if d["node_type"] == "lemma" and d["freq"] >= LABEL_FREQ
}
nx.draw_networkx_labels(
    G,
    pos,
    labels=lemma_labels,
    font_size=8,
    font_family="serif",
    ax=ax
)

# --------------------------------------
# 6. LEGENDA
# --------------------------------------
legend_elements = [
    Patch(facecolor=col, edgecolor="k", label=meaning)
    for meaning, col in meaning_colors.items()
]

plt.legend(
    handles=legend_elements,
    title="Meaning",
    loc="upper right",
    fontsize=10,
    title_fontsize=12,
    frameon=True
)

ax.set_axis_off()
plt.tight_layout()

# --------------------------------------
# 7. SALVATAGGIO
# --------------------------------------
plt.savefig(
    "npn_preposition_lemma_graph_meaning_legend.pdf",
    bbox_inches="tight"
)
plt.savefig(
    "npn_preposition_lemma_graph_meaning_legend.jpg",
    dpi=900,
    bbox_inches="tight"
)

plt.show()
