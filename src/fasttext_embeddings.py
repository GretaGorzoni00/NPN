import sys
import csv
import os
import fasttext


# /Users/ludovica/Documents/projects/fastText/cc.it.300.bin
fasttext_model = sys.argv[1]
output_folder = sys.argv[2]
input_files = sys.argv[3:]

model = fasttext.load_model(fasttext_model)


for filename in input_files:
	basename = os.path.basename(filename)
	base, ext = os.path.splitext(basename)

	with open(filename) as fin, open(os.path.join(output_folder, f"{base}.tsv"), "w") as fout:
		csvfile = csv.DictReader(fin, delimiter=";")

		for row in csvfile:
			noun = row['noun'].strip()
			fout.write(f"{noun}\t{' '.join([str(x) for x in model['casa']])}\n")
# print(model['casa'])