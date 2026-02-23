# import csv
# import os
# import random
# from collections import defaultdict

# # input_files = ["data/dataset_construction/SU_distractor.csv",
# #             "data/dataset_construction/A_distractor.csv",
# #             "data/dataset_construction/SU_construction.csv",
# #             "data/dataset_construction/A_construction.csv"]

# # input_files = ["data/data_set/scivetti/cxns_normalized.csv",
# #             "data/data_set/scivetti/distr_normalized.csv"]


# def main(input_files, min_len, max_len, max_per_lemma, output_folder):
	
# 	for input_file in input_files:

# 		# === Costruzione nome file output ===
# 		basename = os.path.basename(input_file)
# 		base, ext = os.path.splitext(basename)
# 		output_file = f"{output_folder}/{base}_max{max_per_lemma}{ext}"

# 		# === Dizionario per salvare righe per lemma ===
# 		lemma_rows = defaultdict(list)

# 		# === Lettura e filtraggio preliminare ===
# 		with open(input_file, "r", encoding="utf-8") as infile:
# 			reader = csv.reader(infile, delimiter=";")
# 			fieldnames = next(reader)
# 			for row in reader:
# 				if len(row) < 9:
# 					continue  # Salta righe con meno di 9 campi

# 				lemma = row[0].strip()
# 				contesto = row[4].strip()

# 				# Conta token
# 				num_tokens = len(contesto.split())

# 				# Filtra righe per lunghezza del contesto
# 				if num_tokens < min_len or num_tokens > max_len:
# 					continue

# 				# Salva riga nel dizionario per lemma
# 				lemma_rows[lemma].append(row)

# 		# === Campionamento casuale per ogni lemma ===
# 		filtered_rows = []
# 		for lemma, rows in lemma_rows.items():
# 			if len(rows) > max_per_lemma:
# 				sampled = random.sample(rows, max_per_lemma)
# 				filtered_rows.extend(sampled)
# 			else:
# 				filtered_rows.extend(rows)

# 		# === Scrittura del file filtrato ===
# 		with open(output_file, "w", encoding="utf-8", newline="") as outfile:
# 			writer = csv.writer(outfile, delimiter=";")
# 			# writer.writeheader()
# 			writer.writerows(filtered_rows)

# 		print(f"Campionamento completato: {output_file}")
# 		print(f"Totale occorrenze risultanti dal filtraggio: {len(filtered_rows)}")


# if __name__ == "__main__":
	
# 	import argparse
	
# 	parser = argparse.ArgumentParser(description="Campionamento righe per lemma")
# 	parser.add_argument("--input_files", nargs="+", required=True, help="Lista di file di input da processare")
# 	parser.add_argument("--min_len", type=int, default=5, help="Lunghezza minima del contesto (in token)")
# 	parser.add_argument("--max_len", type=int, default=30, help="Lunghezza massima del contesto (in token)")
# 	parser.add_argument("--max_per_lemma", type=int, default=30, help="Numero massimo di righe per lemma")
# 	parser.add_argument("--output_folder", type=str, required=True, help="Cartella di output per i file filtrati")
	
# 	args = parser.parse_args()
	
# 	if not os.path.exists(args.output_folder):
# 		os.makedirs(args.output_folder, exist_ok=True)
	
# 	main(args.input_files, args.min_len, args.max_len, args.max_per_lemma, args.output_folder)


import csv
import os
import random
from collections import defaultdict

random.seed(42)

def main(input_files, min_len, max_len, max_per_lemma, output_folder, nome_file):
	os.makedirs(output_folder, exist_ok=True)

	all_filtered_rows = []
	merged_header = None

	for input_file in input_files:
		basename = os.path.basename(input_file)
		base, ext = os.path.splitext(basename)

		# output "per-file" opzionale (se lo vuoi tenere)
		per_file_output = os.path.join(output_folder, f"{base}_max{max_per_lemma}{ext}")

		lemma_rows = defaultdict(list)

		with open(input_file, "r", encoding="utf-8", newline="") as infile:
			reader = csv.reader(infile, delimiter=";")
			header = next(reader)

			# salva header del primo file e controlla coerenza
			if merged_header is None:
				merged_header = header
			elif header != merged_header:
				raise ValueError(
					f"Header diverso in {input_file}.\n"
					f"Header atteso: {merged_header}\n"
					f"Header trovato: {header}"
				)

			for row in reader:
				if len(row) < 9:
					continue

				lemma = row[1].strip()

				contesto = row[5].strip() + " " + row[6].strip() + " " + row[7].strip() + " " + row[8].strip()


				num_tokens = len(contesto.split())
				# if num_tokens < min_len or num_tokens > max_len:
				if num_tokens < min_len:
					continue

				lemma_rows[lemma].append(row)

		# campionamento per lemma
		filtered_rows = []
		for lemma, rows in lemma_rows.items():
			if len(rows) > max_per_lemma:
				filtered_rows.extend(random.sample(rows, max_per_lemma))
			else:
				filtered_rows.extend(rows)

		# (OPZIONALE) salva anche il file filtrato per singolo input
		with open(per_file_output, "w", encoding="utf-8", newline="") as outfile:
			writer = csv.writer(outfile, delimiter=";")
			writer.writerow(header)         
			writer.writerows(filtered_rows)

		print(f"Campionamento completato (per-file): {per_file_output} | righe: {len(filtered_rows)}")

		# accumula per merge finale
		all_filtered_rows.extend(filtered_rows)

	# ===== Scrittura file unico =====
	merged_name = f"{nome_file}.csv"

	merged_path = os.path.join(output_folder, merged_name)

	with open(merged_path, "w", encoding="utf-8", newline="") as out:
		writer = csv.writer(out, delimiter=";")
		writer.writerow(merged_header)      
		writer.writerows(all_filtered_rows)

	print(f"\nFile unico creato: {merged_path}")
	print(f"Totale righe nel merge: {len(all_filtered_rows)}")


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Campionamento righe per lemma + merge finale")
	parser.add_argument("--input_files", nargs="+", required=True)
	parser.add_argument("--min_len", type=int, default=5)
	parser.add_argument("--max_len", type=int, default=30)
	parser.add_argument("--max_per_lemma", type=int, default=30)
	parser.add_argument("--output_folder", type=str, required=True)
	parser.add_argument("--nome_file", type=str, default="full_dataset_asu", help="Nome del file unico (opzionale)")
	args = parser.parse_args()

	main(args.input_files, args.min_len, args.max_len, args.max_per_lemma, args.output_folder, args.nome_file)