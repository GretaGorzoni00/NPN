import csv
import sys
import os

perturbations = ["PNN", "PN", "NNP", "NP"]

for filename in sys.argv[1:]:
	basename = os.path.basename(filename)

	with open(filename, encoding="utf-8") as fin:
		csvfile = csv.DictReader(fin, delimiter=";")
		writers = {
    		x : csv.DictWriter(open(f"data/data_set/ex_1/perturbed/full/{basename[:-4]}_{x}.csv", "w", encoding="utf-8"),
        			fieldnames=csvfile.fieldnames,
                    delimiter=";"
                    )
    		for x in perturbations
        }
  
  
		for w in writers.values():
			w.writeheader()
   
		for row in csvfile:
			noun1, prep, noun2 = row["costr"].strip().split(" ")
			for x in writers:
				new_row = row.copy()  # <--- QUESTA È LA CHIAVE

				if x == "PNN":
					new_costr = f"{prep} {noun1} {noun2}"
				if x == "PN":
					new_costr = f"{prep} {noun1}"
				if x == "NNP":
					new_costr = f"{noun1} {noun2} {prep}"
				if x == "NP":
					new_costr = f"{noun1} {prep}"

				if new_row["construction"] == "yes":
					new_row["costr"] = new_costr

				new_row["construction"] = "no"
				writers[x].writerow(new_row)