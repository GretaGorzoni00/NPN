import csv
import os
import random
from collections import defaultdict

# input_files = ["data/dataset_construction/SU_distractor.csv",
#             "data/dataset_construction/A_distractor.csv",
#             "data/dataset_construction/SU_construction.csv",
#             "data/dataset_construction/A_construction.csv"]

# input_files = ["data/data_set/scivetti/cxns_normalized.csv",
#             "data/data_set/scivetti/distr_normalized.csv"]


def main(input_files, min_len, max_len, max_per_lemma, output_folder):
    
    for input_file in input_files:

        # === Costruzione nome file output ===
        basename = os.path.basename(input_file)
        base, ext = os.path.splitext(basename)
        output_file = f"{output_folder}/{base}_max{max_per_lemma}{ext}"

        # === Dizionario per salvare righe per lemma ===
        lemma_rows = defaultdict(list)

        # === Lettura e filtraggio preliminare ===
        with open(input_file, "r", encoding="utf-8") as infile:
            reader = csv.reader(infile, delimiter=";")
            fieldnames = reader.fieldnames
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
            writer = csv.writer(outfile, fieldnames=fieldnames, delimiter=";")
            writer.writeheader()
            writer.writerows(filtered_rows)

        print(f"Campionamento completato: {output_file}")
        print(f"Totale occorrenze risultanti dal filtraggio: {len(filtered_rows)}")


if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Campionamento righe per lemma")
    parser.add_argument("--input_files", nargs="+", required=True, help="Lista di file di input da processare")
    parser.add_argument("--min_len", type=int, default=5, help="Lunghezza minima del contesto (in token)")
    parser.add_argument("--max_len", type=int, default=30, help="Lunghezza massima del contesto (in token)")
    parser.add_argument("--max_per_lemma", type=int, default=30, help="Numero massimo di righe per lemma")
    parser.add_argument("--output_folder", type=str, required=True, help="Cartella di output per i file filtrati")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder, exist_ok=True)
    
    main(args.input_files, args.min_len, args.max_len, args.max_per_lemma, args.output_folder)
    