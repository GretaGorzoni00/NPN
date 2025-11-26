# import numpy as np
# import pandas as pd

import csv
import random

random.seed(2542)

# TODO: AGGIUNGERE CONSTRAINT SU YES/NO


files_train = ["data/data_set/ex_1/simple/full/ex1_simple_train_0.csv", "data/data_set/ex_1/simple/full/ex1_simple_train_1.csv", "data/data_set/ex_1/simple/full/ex1_simple_train_2.csv", "data/data_set/ex_1/simple/full/ex1_simple_train_3.csv", "data/data_set/ex_1/simple/full/ex1_simple_train_4.csv"]
files_test = ["data/data_set/ex_1/simple/full/ex1_simple_test_0.csv", "data/data_set/ex_1/simple/full/ex1_simple_test_1.csv", "data/data_set/ex_1/simple/full/ex1_simple_test_2.csv", "data/data_set/ex_1/simple/full/ex1_simple_test_3.csv", "data/data_set/ex_1/simple/full/ex1_simple_test_4.csv"]
control_train = [f"data/data_set/ex_1/simple/control_simple/ex1_simple_train_0.csv", f"data/data_set/ex_1/simple/control_simple/ex1_simple_train_1.csv", f"data/data_set/ex_1/simple/control_simple/ex1_simple_train_2.csv", f"data/data_set/ex_1/simple/control_simple/ex1_simple_train_3.csv", f"data/data_set/ex_1/simple/control_simple/ex1_simple_train_4.csv"]
control_test = [f"data/data_set/ex_1/simple/control_simple/ex1_simple_test_0.csv", f"data/data_set/ex_1/simple/control_simple/ex1_simple_test_1.csv", f"data/data_set/ex_1/simple/control_simple/ex1_simple_test_2.csv", f"data/data_set/ex_1/simple/control_simple/ex1_simple_test_3.csv", f"data/data_set/ex_1/simple/control_simple/ex1_simple_test_4.csv"]



for train, test, c_train, c_test in zip(files_train, files_test, control_train, control_test):
	assigned = {}
	tot = {"yes": 0, "no": 0}
	# TOT = 0
	with open(train) as fin_train, open(test) as fin_test:
		csvtrain = csv.DictReader(fin_train, delimiter=";")
		csvtest = csv.DictReader(fin_test, delimiter=";")

		for csvfile in csvtrain, csvtest:
			for row in csvfile:
				tot[row["construction"]] += 1
	print(tot)
	with open(train) as fin_train, open(test) as fin_test:
		csvtrain = csv.DictReader(fin_train, delimiter=";")
		csvtest = csv.DictReader(fin_test, delimiter=";")

		trainheader = csvtrain.fieldnames
		testheader = csvtest.fieldnames

		csvtrain, csvtest = list(csvtrain), list(csvtest)

		for csvfile in csvtrain, csvtest:
			for row in csvfile:
				noun = row["noun"].strip()
				if noun not in assigned:
					p = random.random()
					yes_threshold = tot["yes"]/(tot["yes"]+tot["no"])
					label = "no"
					if p < yes_threshold:
						label = "yes"
					assigned[noun] = label
				row["construction"] = assigned[noun]
				tot[assigned[noun]] -=1

	new_tot = {"yes":0, "no":0}
	with open(c_train, "w") as fout_train, open(c_test, "w") as fout_test:
		csvtrain_out = csv.DictWriter(fout_train, delimiter=";", fieldnames=trainheader)
		csvtest_out = csv.DictWriter(fout_test, delimiter=";", fieldnames=testheader)
		csvtrain_out.writeheader()
		csvtest_out.writeheader()

		for row in csvtrain:
			new_tot[row["construction"]] += 1
			csvtrain_out.writerow(row)

		for row in csvtest:
			new_tot[row["construction"]] += 1
			csvtest_out.writerow(row)

	print("NEW", new_tot)
