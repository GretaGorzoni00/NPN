# import numpy as np
# import pandas as pd

import csv
import fasttext
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
import random

random.seed(2542)

words = [x.strip() for x in open("/Users/ludovica/Documents/projects/NPN/data/embeddings/fasttext/simple/lemmas_tot.txt").readlines()]

word_to_id = {word:i for i, word in enumerate(words)}
id_to_word = {y:x for x, y in word_to_id.items()}

model = fasttext.load_model("/Users/ludovica/Documents/projects/fastText/cc.it.300.bin")

A = np.array([model[word] for word in words])

dist_out = 1-pairwise_distances(A, metric="cosine")

# for i, word in enumerate(words):
#     for j, word2 in enumerate(words):
#         if word2>=word:
#             print(f"{word},{word2},{dist_out[i][j]}")


files_train = ["data/data_set/ex1_simple_train_0.csv", "data/data_set/ex1_simple_train_1.csv", "data/data_set/ex1_simple_train_2.csv", "data/data_set/ex1_simple_train_3.csv", "data/data_set/ex1_simple_train_4.csv"]
files_test = ["data/data_set/ex1_simple_test_0.csv", "data/data_set/ex1_simple_test_1.csv", "data/data_set/ex1_simple_test_2.csv", "data/data_set/ex1_simple_test_3.csv", "data/data_set/ex1_simple_test_4.csv"]
control_train = ["data/data_set/control/ex1_simple_train_0.csv", "data/data_set/control/ex1_simple_train_1.csv", "data/data_set/control/ex1_simple_train_2.csv", "data/data_set/control/ex1_simple_train_3.csv", "data/data_set/control/ex1_simple_train_4.csv"]
control_test = ["data/data_set/control/ex1_simple_test_0.csv", "data/data_set/control/ex1_simple_test_1.csv", "data/data_set/control/ex1_simple_test_2.csv", "data/data_set/control/ex1_simple_test_3.csv", "data/data_set/control/ex1_simple_test_4.csv"]

for train, test, c_train, c_test in zip(files_train, files_test, control_train, control_test):
	assigned = {}
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
					label = "no"
					if p > 0.5:
						label = "yes"
					noun_id = word_to_id[noun]
					similar_items = [id_to_word[i] for i, el in enumerate(dist_out[noun_id]) if el > 0.5]
					for element in similar_items:
						assigned[element] = label

				row["construction"] = assigned[noun]

	with open(c_train, "w") as fout_train, open(c_test, "w") as fout_test:
		csvtrain_out = csv.DictWriter(fout_train, delimiter=";", fieldnames=trainheader)
		csvtest_out = csv.DictWriter(fout_test, delimiter=";", fieldnames=testheader)
		csvtrain_out.writeheader()
		csvtest_out.writeheader()

		for row in csvtrain:
			csvtrain_out.writerow(row)

		for row in csvtest:
			csvtest_out.writerow(row)





# emb_train = pd.read_csv(X_train_path, header=None, sep=" ")

