import sys
import csv
import os
import fasttext


# /Users/ludovica/Documents/projects/fastText/cc.it.300.bin
# fasttext_model = sys.argv[1]
# output_folder = sys.argv[2]
# input_files = sys.argv[3:]

# _FIELD = "noun"

def get_noun(row):
    return row["noun"].strip()

def get_pre_lemma(row):
	return row["pre_lemma"].strip()

def get_costrN(row):
	return row["costr"].split()[0].strip()


def main(fasttext_model, output_folder, input_files, get_token_fun):

	model = fasttext.load_model(fasttext_model)

	for filename in input_files:
		basename = os.path.basename(filename)
		base, ext = os.path.splitext(basename)

		with open(filename) as fin, open(os.path.join(output_folder, f"{base}.tsv"), "w") as fout:
			csvfile = csv.DictReader(fin, delimiter=";")

			for row in csvfile:
				noun = get_token_fun(row)
				fout.write(f"{noun}\t{' '.join([str(x) for x in model[noun]])}\n")
   
if __name__ == "__main__":
	import argparse
	import os
 
	parser = argparse.ArgumentParser(description="Estrazione embedding fastText")
	parser.add_argument("--fasttext_model", type=str, required=True, help="Path al modello fastText")
	parser.add_argument("--output_folder", type=str, required=True, help="Cartella di output per i file con gli embedding")
	parser.add_argument("--input_files", nargs="+", required=True, help="Lista di file di input da processare")
	parser.add_argument("--FIELD", choices=['noun', 'pre_lemma', 'costrN'], default="noun", help="Nome della colonna da cui estrarre i token per gli embedding")
 
	args = parser.parse_args()
	
	if args.FIELD == "noun":
		get_token_fun = get_noun
	elif args.FIELD == "pre_lemma":
		get_token_fun = get_pre_lemma
	elif args.FIELD == "costrN":
		get_token_fun = get_costrN
  
	if not os.path.exists(args.output_folder):
		os.makedirs(args.output_folder, exist_ok=True)

	main(args.fasttext_model, args.output_folder, args.input_files, get_token_fun)