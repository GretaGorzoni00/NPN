import csv
import os
import random
from collections import defaultdict

input_files = ["data/dataset_construction/SU_distractor.csv",
            "data/dataset_construction/A_distractor.csv",
            "data/dataset_construction/SU_construction.csv",
            "data/dataset_construction/A_construction.csv"]

for input_file in input_files:
    # === Parametri ===
    min_len = 5
    max_len = 30
    max_per_lemma = 30

    # === Costruzione nome file output ===
    basename = os.path.basename(input_file)
    base, ext = os.path.splitext(basename)
    output_file = f"data/source/{base}_max{max_per_lemma}{ext}"

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
