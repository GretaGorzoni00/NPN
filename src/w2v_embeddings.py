import sys
import random
import os
import csv

# w2v_model = sys.argv[1]
# output_folder = sys.argv[2]
# input_files = sys.argv[3:]

# random.seed(999)
# _LEN = 0


# _FIELD = 'noun'


def get_noun(row):
	return row["noun"].strip()

def get_pre_lemma(row):
	return row["pre_lemma"].strip()

def get_costrN(row):
	return row["costr"].split()[0].strip()


def main(w2v_model, output_folder, input_files, get_token_fun, _LEN, seed):
	
	random.seed(seed)

	to_load = set()
	for filename in input_files:
		with open(filename) as fin:
			csvfile = csv.DictReader(fin, delimiter=";")

			for row in csvfile:
				noun = get_token_fun(row).strip()
				to_load.add(noun)


	model = open(w2v_model).readlines()

	model_dict = {}

	for line in model:
		line = line.strip().split()
		if len(line) > 0 and line[0] in to_load:
			_LEN = len(line) - 1
			model_dict[line[0]] = line[1:]

	print("vectors loaded")

	for filename in input_files:
		basename = os.path.basename(filename)
		base, ext = os.path.splitext(basename)

		with open(filename) as fin, open(os.path.join(output_folder, f"{base}.tsv"), "w") as fout:
			csvfile = csv.DictReader(fin, delimiter=";")

			for row in csvfile:
				noun = get_token_fun(row).strip()
				if noun in model_dict:
					fout.write(f"{noun}\t{' '.join(model_dict[noun])}\n")
				else:
					print(noun, "not found")
					random_vec = [random.random() for _ in range(_LEN)]
					fout.write(f"{noun}\t{' '.join([str(x) for x in random_vec])}\n")
	 
	 

if __name__ == "__main__":
	import argparse
	import os
 
	parser = argparse.ArgumentParser(description="Estrazione embedding w2vec")
	parser.add_argument("--w2vec_model", type=str, required=True, help="Path al modello w2vec")
	parser.add_argument("--output_folder", type=str, required=True, help="Cartella di output per i file con gli embedding")
	parser.add_argument("--input_files", nargs="+", required=True, help="Lista di file di input da processare")
	parser.add_argument("--FIELD", choices=['noun', 'pre_lemma', 'costrN'], default="noun", help="Nome della colonna da cui estrarre i token per gli embedding")
	parser.add_argument("--seed", type=int, default=999, help="Seed per il generatore casuale")
	parser.add_argument("--len", type=int, default=0, help="Lunghezza degli embedding (necessaria se il modello w2vec non contiene tutti i token)")
 
	args = parser.parse_args()
	
	if args.FIELD == "noun":
		get_token_fun = get_noun
	elif args.FIELD == "pre_lemma":
		get_token_fun = get_pre_lemma
	elif args.FIELD == "costrN":
		get_token_fun = get_costrN
  
	if not os.path.exists(args.output_folder):
		os.makedirs(args.output_folder, exist_ok=True)

	main(args.w2vec_model, args.output_folder, args.input_files, get_token_fun, args.len, args.seed)