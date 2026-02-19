# import numpy as np
# import pandas as pd

import csv
import random

random.seed(2542)

_FIELD = "noun"

_LABEL_FIELD = "meaning"

MEANINGS = [
	"succession/iteration/distributivity",
	"greater_plurality/accumulation",
	"juxtaposition/contact",
]


files_train = ["data/data_set/ex_2/simple/full/ex2_simple_train_0.csv", "data/data_set/ex_2/simple/full/ex2_simple_train_2.csv", "data/data_set/ex_2/simple/full/ex2_simple_train_2.csv", "data/data_set/ex_2/simple/full/ex2_simple_train_3.csv", "data/data_set/ex_2/simple/full/ex2_simple_train_4.csv"]
files_test = ["data/data_set/ex_2/simple/full/ex2_simple_test_0.csv", "data/data_set/ex_2/simple/full/ex2_simple_test_2.csv", "data/data_set/ex_2/simple/full/ex2_simple_test_2.csv", "data/data_set/ex_2/simple/full/ex2_simple_test_3.csv", "data/data_set/ex_2/simple/full/ex2_simple_test_4.csv"]
control_train = [f"data/data_set/ex_2/simple/control/ex2_simple_train_0.csv", f"data/data_set/ex_2/simple/control/ex2_simple_train_2.csv", f"data/data_set/ex_2/simple/control/ex2_simple_train_2.csv", f"data/data_set/ex_2/simple/control/ex2_simple_train_3.csv", f"data/data_set/ex_2/simple/control/ex2_simple_train_4.csv"]
control_test = [f"data/data_set/ex_2/simple/control/ex2_simple_test_0.csv", f"data/data_set/ex_2/simple/control/ex2_simple_test_2.csv", f"data/data_set/ex_2/simple/control/ex2_simple_test_2.csv", f"data/data_set/ex_2/simple/control/ex2_simple_test_3.csv", f"data/data_set/ex_2/simple/control/ex1_simple_test_4.csv"]



for train, test, c_train, c_test in zip(files_train, files_test, control_train, control_test):
	assigned = {}
	tot = {m: 0 for m in MEANINGS}
	with open(train) as fin_train, open(test) as fin_test:
		csvtrain = csv.DictReader(fin_train, delimiter=";")
		csvtest = csv.DictReader(fin_test, delimiter=";")

		for csvfile in csvtrain, csvtest:
			for row in csvfile:
				tot[row[_LABEL_FIELD]] += 1
	print(tot)
 
 
	with open(train) as fin_train, open(test) as fin_test:
		csvtrain = csv.DictReader(fin_train, delimiter=";")
		csvtest = csv.DictReader(fin_test, delimiter=";")

		trainheader = csvtrain.fieldnames
		testheader = csvtest.fieldnames

		csvtrain, csvtest = list(csvtrain), list(csvtest)

		for csvfile in csvtrain, csvtest:
			for row in csvfile:
				noun = row[_FIELD].strip()
				if noun not in assigned:
					p = random.random()
	 
					total_left = sum(tot.values())

					# fallback if quotas are exhausted for any reason
					if total_left <= 0:
						assigned[noun] = random.choice(MEANINGS)
					else:
						# build cumulative thresholds like yes/no, but for 3 labels
						cum = 0.0
						label = MEANINGS[-1]  # default
						for m in MEANINGS:
							cum += tot[m] / total_left
							if p < cum:
								label = m
								break
						assigned[noun] = label

				row[_LABEL_FIELD] = assigned[noun]
				tot[assigned[noun]] -= 1
				
	new_tot = {m: 0 for m in MEANINGS}          
	with open(c_train, "w") as fout_train, open(c_test, "w") as fout_test:
		csvtrain_out = csv.DictWriter(fout_train, delimiter=";", fieldnames=trainheader)
		csvtest_out = csv.DictWriter(fout_test, delimiter=";", fieldnames=testheader)
		csvtrain_out.writeheader()
		csvtest_out.writeheader()

		for row in csvtrain:
			new_tot[row[_LABEL_FIELD]] += 1
			csvtrain_out.writerow(row)

		for row in csvtest:
			new_tot[row[_LABEL_FIELD]] += 1
			csvtest_out.writerow(row)

	print("NEW", new_tot)
