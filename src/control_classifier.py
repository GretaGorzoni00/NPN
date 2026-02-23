# import numpy as np
# import pandas as pd

import csv
import random
import argparse
import os

# random.seed(2542)

# _FIELD = "noun"

def main(random_seed, files_train, files_test, control_dir, _FIELD, _LABEL_FIELD):
	
	random.seed(random_seed)

	if _LABEL_FIELD == "meaning":

		VALUES = [
			"succession/iteration/distributivity",
			"greater_plurality/accumulation",
			"juxtaposition/contact",
		]
  
	else:
		VALUES = [
			"yes",
			"no"
		]


	# files_train = ["data/data_set/ex_2/simple/full/ex2_simple_train_0.csv", "data/data_set/ex_2/simple/full/ex2_simple_train_2.csv", "data/data_set/ex_2/simple/full/ex2_simple_train_2.csv", "data/data_set/ex_2/simple/full/ex2_simple_train_3.csv", "data/data_set/ex_2/simple/full/ex2_simple_train_4.csv"]
	# files_test = ["data/data_set/ex_2/simple/full/ex2_simple_test_0.csv", "data/data_set/ex_2/simple/full/ex2_simple_test_2.csv", "data/data_set/ex_2/simple/full/ex2_simple_test_2.csv", "data/data_set/ex_2/simple/full/ex2_simple_test_3.csv", "data/data_set/ex_2/simple/full/ex2_simple_test_4.csv"]
	# control_train = [f"data/data_set/ex_2/simple/control/ex2_simple_train_0.csv", f"data/data_set/ex_2/simple/control/ex2_simple_train_2.csv", f"data/data_set/ex_2/simple/control/ex2_simple_train_2.csv", f"data/data_set/ex_2/simple/control/ex2_simple_train_3.csv", f"data/data_set/ex_2/simple/control/ex2_simple_train_4.csv"]
	# control_test = [f"data/data_set/ex_2/simple/control/ex2_simple_test_0.csv", f"data/data_set/ex_2/simple/control/ex2_simple_test_2.csv", f"data/data_set/ex_2/simple/control/ex2_simple_test_2.csv", f"data/data_set/ex_2/simple/control/ex2_simple_test_3.csv", f"data/data_set/ex_2/simple/control/ex1_simple_test_4.csv"]

	os.makedirs(control_dir, exist_ok=True)

	for train, test in zip(files_train, files_test):
	 
		# genera nomi output automaticamente
		c_train = os.path.join(control_dir, os.path.basename(train))
		c_test = os.path.join(control_dir, os.path.basename(test))

		assigned = {}
		tot = {m: 0 for m in VALUES}
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
							assigned[noun] = random.choice(VALUES)
						else:
							# build cumulative thresholds like yes/no, but for 3 labels
							cum = 0.0
							label = VALUES[-1]  # default
							for m in VALUES:
								cum += tot[m] / total_left
								if p < cum:
									label = m
									break
							assigned[noun] = label

					row[_LABEL_FIELD] = assigned[noun]
					tot[assigned[noun]] -= 1
					
		new_tot = {m: 0 for m in VALUES}          
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



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	
	parser.add_argument("--random_seed", default=2542, type=int, help="Random seed for reproducibility")
	parser.add_argument("--files_train", required=True, nargs="+", help="List of training CSV files")
	parser.add_argument("--files_test", required=True, nargs="+", help="List of test CSV files")
	parser.add_argument("--control_dir", required=True, help="Directory where control CSV files are stored")
	parser.add_argument("--_FIELD", default="noun", help="Field to use for classification ")
	parser.add_argument("--_LABEL_FIELD", default="meaning", help="Field to use for labels")	
 
	args = parser.parse_args()
	
	main(args.random_seed, args.files_train, args.files_test, args.control_dir, args._FIELD, args._LABEL_FIELD)