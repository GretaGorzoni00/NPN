import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --------------------
# Percorsi dei CSV
# --------------------
csv_a = "/Users/gretagorzoni/Desktop/TESI_code/data/data_set/A_distractor.csv"
csv_b = "/Users/gretagorzoni/Desktop/TESI_code/data/data_set/SU_distractor.csv"

# --------------------
# Funzione per leggere e pulire un CSV
# --------------------
def load_and_clean(csv_path):
    df = pd.read_csv(csv_path, sep=";", header=None, dtype=str)
    df.columns = [
        "construction", "valid", "prep", "lemma", "left_context",
        "target", "right_context", "number", "distractor pattern", "function"
    ]
    df = df.fillna("").applymap(lambda x: x.strip() if isinstance(x, str) else x)
    return df

# --------------------
# Carica i due dataset
# --------------------
df_a = load_and_clean(csv_a)
df_b = load_and_clean(csv_b)

# --------------------
# Scegli la colonna da confrontare
# --------------------
column = "distractor pattern"  # puoi cambiare in "number" o "function"

# --------------------
# Calcola le frequenze per ciascun file
# --------------------
counts_a = df_a[column].value_counts().rename("File A")
counts_b = df_b[column].value_counts().rename("File B")

# --------------------
# Unisci i conteggi
# --------------------
combined = pd.concat([counts_a, counts_b], axis=1).fillna(0).astype(int)

# Ordina per somma totale (frequenze più alte in cima)
combined = combined.loc[combined.sum(axis=1).sort_values(ascending=False).index]

# --------------------
# Crea grafico comparativo
# --------------------
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.35
x = np.arange(len(combined.index))

bars_a = ax.bar(x - bar_width/2, combined["File A"], bar_width, label="NaN", color='cornflowerblue', edgecolor='black')
bars_b = ax.bar(x + bar_width/2, combined["File B"], bar_width, label="NsuN", color='lightcoral', edgecolor='black')

ax.set_title(f" {column.upper()} (semi-logaritmic scale)", fontsize=20, weight='bold')
ax.set_xlabel("", fontsize=16)
ax.set_ylabel("Frequency", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(combined.index, rotation=0, ha='center', fontsize=12)
ax.legend(fontsize=12)
ax.set_yscale('log')
ax.grid(which='major', linestyle='-', linewidth=0.8, alpha=0.7)
ax.grid(which='minor', linestyle='--', linewidth=0.5, alpha=0.4)

# Controlla la densità dei tick minori (opzionale)
ax.yaxis.set_minor_locator(plt.LogLocator(subs='auto'))

ax.grid(True)

# Aggiungi i valori sopra le barre
for bars in [bars_a, bars_b]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{int(height)}",
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()
