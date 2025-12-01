import sys
import random
import os
import csv

w2v_model = sys.argv[1]
output_folder = sys.argv[2]
input_files = sys.argv[3:]

random.seed(999)
_LEN = 0


to_load = set()
for filename in input_files:
	with open(filename) as fin:
		csvfile = csv.DictReader(fin, delimiter=";")

		for row in csvfile:
			noun = row['noun'].strip()
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
			noun = row['noun'].strip()
			if noun in model_dict:
				fout.write(f"{noun}\t{' '.join(model_dict[noun])}\n")
			else:
				print(noun, "not found")
				random_vec = [random.random() for _ in range(_LEN)]
				fout.write(f"{noun}\t{' '.join([str(x) for x in random_vec])}\n")