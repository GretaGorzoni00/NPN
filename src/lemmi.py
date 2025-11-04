import pandas as pd
import math

# Percorso CSV
csv_path = "/Users/gretagorzoni/Desktop/TESI_code/data/data_set/SU.csv"

# Leggi CSV e pulizia
df = pd.read_csv(csv_path, sep=";", header=None, dtype=str)
df.columns = [
    "construction", "valid", "prep", "lemma", "left_context",
    "target", "right_context", "number", "meaning", "function"
]
df = df.fillna("").applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Frequenze dei lemmi
lemma_counts = df["lemma"].value_counts().reset_index()
lemma_counts.columns = ["Lemma", "Frequenza"]

# Funzione per escare caratteri speciali
def escape_latex(s):
    if isinstance(s, str):
        s = s.replace("&", "\\&").replace("%", "\\%").replace("#", "\\#")
        s = s.replace("_", "\\_").replace("{", "\\{").replace("}", "\\}")
        s = s.replace("$", "\\$").replace("^", "\\^{}").replace("~", "\\~{}")
        s = s.replace("\\", "\\textbackslash{}")
    return s

# Numero di colonne nella tabella
num_cols = 3

# Calcola quante righe servono
num_rows = math.ceil(len(lemma_counts) / num_cols)

# Riordina lemmi per righe e colonne
table_matrix = []
for i in range(num_rows):
    row = []
    for j in range(num_cols):
        idx = i + j*num_rows
        if idx < len(lemma_counts):
            lemma = escape_latex(lemma_counts.loc[idx, "Lemma"])
            freq = lemma_counts.loc[idx, "Frequenza"]
            row.append(f"{lemma} ({freq})")
        else:
            row.append("")  # cella vuota
    table_matrix.append(row)

# Genera tabella LaTeX
output_path = "/Users/gretagorzoni/Desktop/TESI/lemmi_multicol.tex"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("\\begin{longtable}{" + "l "*num_cols + "}\n")
    f.write("\\caption{Frequenza di tutti i lemmi (più colonne per pagina)} \\\\\n")
    f.write("\\hline\n")
    for row in table_matrix:
        f.write(" & ".join(row) + " \\\\\n")
    f.write("\\hline\n")
    f.write("\\end{longtable}\n")

print(f"Tabella multicolonna generata: {output_path}")
