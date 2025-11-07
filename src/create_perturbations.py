import csv
import sys
import os

perturbations = ["PNN", "PN", "NNP", "NP"]

for filename in sys.argv[1:]:
	basename = os.path.basename(filename)

	with open(filename, encoding="utf-8") as fin:
		csvfile = csv.DictReader(fin, delimiter=";")
		writers = {x:csv.DictWriter(open(f"data/data_set/perturbed/{basename[:-4]}_{x}.csv", "w", encoding="utf-8"),
                    fieldnames=csvfile.fieldnames,
                    delimiter=";")
    				for x in perturbations}
		for row in csvfile:
			noun1, prep, noun2 = row["costr"].strip().split(" ")
			for x in writers:
				if x == "PNN":
					row["costr"] = f"{prep} {noun1} {noun2}"
				if x == "PN":
					row["costr"] = f"{prep} {noun1}"
				if x == "NNP":
					row["costr"] = f"{noun1} {noun2} {prep}"
				if x == "NP":
					row["costr"] = f"{noun1} {prep}"
				writers[x].writerow(row)
