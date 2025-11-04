import csv
import os
import random
from collections import defaultdict

input_files = ["data/source/SU_distractor.csv",
            "data/source/A_distractor.csv",
            "data/source/SU_construction.csv",
            "data/source/A_construction.csv"]

for input_file in input_files:
    # === Parametri ===
    min_len = 5
    max_len = 30
    max_per_lemma = 30

    # === Costruzione nome file output ===
    base, ext = os.path.splitext(input_file)
    output_file = f"{base}_filtrato{ext}"

    # === Dizionario per salvare righe per lemma ===
    lemma_rows = defaultdict(list)

    # === Lettura e filtraggio preliminare ===
    with open(input_file, "r", encoding="utf-8") as infile:
        reader = csv.reader(infile, delimiter=";")
        for row in reader:
            if len(row) < 9:
                continue

            lemma = row[0].strip()
            contesto = row[4].strip()

            # Conta token
            num_tokens = len(contesto.split())

            # Filtra righe per lunghezza del contesto
            if num_tokens < min_len or num_tokens > max_len:
                continue

            # Salva riga nel dizionario per lemma
            lemma_rows[lemma].append(row)

    # === Campionamento casuale per ogni lemma ===
    filtered_rows = []
    for lemma, rows in lemma_rows.items():
        if len(rows) > max_per_lemma:
            sampled = random.sample(rows, max_per_lemma)
            filtered_rows.extend(sampled)
        else:
            filtered_rows.extend(rows)

    # === Scrittura del file filtrato ===
    with open(output_file, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.writer(outfile, delimiter=";")
        writer.writerows(filtered_rows)

    print(f"Campionamento completato: {output_file}")
    print(f"Totale occorrenze risultanti dal filtraggio: {len(filtered_rows)}")
