import csv
import random
import collections
import logging

# === Lista di righe, evita righe vuote ===
def read_csv(file_path, delimiter=";"):
	with open(file_path, "r", encoding="utf-8-sig") as f:
		reader = csv.DictReader(f, delimiter=delimiter)
		return [row for row in reader if len(row) > 0]

def sample(orig_lemmas, counts_dict, lemmas_dict, category, preposition, max_n):
	i = 0
	label=category+"-"+preposition
	while i<len(orig_lemmas[category]) and counts_dict[label] <= max_n:
		lemma, count_a, count_su = orig_lemmas[category][i]
		# print(lemma, count_a, count_su)
		if preposition == "a":
			counts_dict[label] += count_a
			if count_a > 0:
				lemmas_dict[label].append(lemma)
		else:
			counts_dict[label] += count_su
			if count_su > 0:
				lemmas_dict[label].append(lemma)
		i += 1

# === Scrivere CSV ===
def write_csv(file_path, rows, delimiter=";"):
	with open(file_path, "w", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(f, delimiter=delimiter, fieldnames=["ID", "NPN", "construction", 'preposition',
																	'noun', 'pre_lemma', 'context_pre', "costr",
																	'context_post','number_of_noun',
																	'Type', "other_cxn",
																	'meaning',"syntactic_function"],
								restval= '_')
		writer.writeheader()
		writer.writerows(rows)

def count_lemmas_simple(costruzioni_A, costruzioni_SU, distrattori_A, distrattori_SU):
	lemmi = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(int)))
	lemmi_tot = collections.defaultdict(lambda:collections.defaultdict(int))

	for el in costruzioni_A:
		lemmi[el['noun']][el['meaning']]["A"] += 1
		lemmi_tot[el['noun']][el['meaning']] +=1
	for el in costruzioni_SU:
		lemmi[el['noun']][el['meaning']]["SU"] += 1
		lemmi_tot[el['noun']][el['meaning']] +=1

	lemmi_comuni = [lemma for lemma, lemma_data in lemmi_tot.items() if lemma_data["juxtaposition/contact"] > 0 and lemma_data["succession/iteration/distributivity"] > 0 and lemma_data["greater_plurality/accumulation"] > 0]
	lemmi_comuni_juxt_acc = [lemma for lemma, lemma_data in lemmi_tot.items() if lemma_data["juxtaposition/contact"] > 0 and lemma_data["succession/iteration/distributivity"] == 0 and lemma_data["greater_plurality/accumulation"] > 0]
	lemmi_comuni_juxt_succ = [lemma for lemma, lemma_data in lemmi_tot.items() if lemma_data["juxtaposition/contact"] > 0 and lemma_data["succession/iteration/distributivity"] > 0 and lemma_data["greater_plurality/accumulation"] == 0]
	lemmi_comuni_acc_succ = [lemma for lemma, lemma_data in lemmi_tot.items() if lemma_data["juxtaposition/contact"] == 0 and lemma_data["succession/iteration/distributivity"] > 0 and lemma_data["greater_plurality/accumulation"] > 0]


	lemmi_acc = [lemma for lemma, lemma_data in lemmi_tot.items() if lemma_data["juxtaposition/contact"] == 0 and lemma_data["succession/iteration/distributivity"] == 0 and lemma_data["greater_plurality/accumulation"] > 0]
	lemmi_juxt = [lemma for lemma, lemma_data in lemmi_tot.items() if lemma_data["juxtaposition/contact"] > 0 and lemma_data["succession/iteration/distributivity"] == 0 and lemma_data["greater_plurality/accumulation"] == 0]
	lemmi_succ = [lemma for lemma, lemma_data in lemmi_tot.items() if lemma_data["juxtaposition/contact"] == 0 and lemma_data["succession/iteration/distributivity"] > 0 and lemma_data["greater_plurality/accumulation"] == 0]

	return list(lemmi.items()), lemmi_tot, \
		lemmi_comuni_juxt_acc, lemmi_comuni_juxt_succ, lemmi_comuni_acc_succ, \
		lemmi_acc, lemmi_juxt, lemmi_succ

def populate_split(orig_file, train_curr_lemmi, train_file,
							test_curr_lemmi, test_file,
							add_filter_train=lambda x: True,
							add_filter_test=lambda x: True):

	for element in orig_file:
		if element["noun"] in train_curr_lemmi and add_filter_train(element):
			train_file.append(element)

		if element["noun"] in test_curr_lemmi and add_filter_test(element):
			test_file.append(element)

###########################

if __name__ == "__main__":

	logging.basicConfig(
		filename='logs/split.log',          # Log file name
		level=logging.INFO,          # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
		format='%(asctime)s - %(levelname)s - %(message)s'
	)

	_NFOLDS = 5
	_SEED = 232

	costruzioni_A_file = "data/source/A_construction_max30copy.csv"
	costruzioni_SU_file = "data/source/SU_construction_max30.csv"

	distrattori_A_file = "data/source/A_distractor_max30.csv"
	distrattori_SU_file = "data/source/SU_distractor_max30.csv"

	logging.info(f"sampling {_NFOLDS} folds with seed {_SEED}")

	random.seed(_SEED)

	# === Lettura file filtrati ===
	costruzioni_A = read_csv(costruzioni_A_file)
	costruzioni_SU = read_csv(costruzioni_SU_file)
	distrattori_A = read_csv(distrattori_A_file)
	distrattori_SU = read_csv(distrattori_SU_file)

	# === SIMPLE SETTING ===
	logging.info("Beginning sampling for setting SIMPLE")

	lemmi, lemmi_tot, \
		lemmi_comuni_juxt_acc, lemmi_comuni_juxt_succ, lemmi_comuni_acc_succ, \
		lemmi_acc, lemmi_juxt, lemmi_succ = count_lemmas_simple(costruzioni_A, costruzioni_SU, distrattori_A, distrattori_SU)

	dict_juxt = {lemma:lemma_data for lemma, lemma_data in lemmi if lemma in lemmi_comuni_juxt_acc or lemma in lemmi_comuni_juxt_succ or lemma in lemmi_juxt}
	dict_acc_succ = {lemma:lemma_data for lemma, lemma_data in lemmi if lemma in lemmi_comuni_acc_succ}

	for it in range(_NFOLDS):
		logging.info(f"Iteration n. {it}")

		random.shuffle(lemmi)

		train_lemmi = {"succ": [], "acc": [], "juxt": []}
		test_lemmi = {"succ": [], "acc": [], "juxt": []}

		for lemma, lemma_data in dict_juxt.items():
			p = random.random()
			if p <= 0.85 and sum([el[1] for el in train_lemmi["juxt"]])<=110:
				train_lemmi["juxt"].append((lemma,
								lemma_data["juxtaposition/contact"]["A"],
								lemma_data["juxtaposition/contact"]["SU"]))
				if lemma in lemmi_comuni_juxt_acc:
					test_lemmi["acc"].append((lemma,
								lemma_data["greater_plurality/accumulation"]["A"],
								lemma_data["greater_plurality/accumulation"]["SU"]))
				if lemma in lemmi_comuni_juxt_succ:
	   				test_lemmi["succ"].append((lemma,
								lemma_data["succession/iteration/distributivity"]["A"],
								lemma_data["succession/iteration/distributivity"]["SU"]))
			else:
				test_lemmi["juxt"].append((lemma,
								lemma_data["juxtaposition/contact"]["A"],
								lemma_data["juxtaposition/contact"]["SU"]))
				if lemma in lemmi_comuni_juxt_acc:
					train_lemmi["acc"].append((lemma,
								lemma_data["greater_plurality/accumulation"]["A"],
								lemma_data["greater_plurality/accumulation"]["SU"]))
				if lemma in lemmi_comuni_juxt_succ:
	   				train_lemmi["succ"].append((lemma,
								lemma_data["succession/iteration/distributivity"]["A"],
								lemma_data["succession/iteration/distributivity"]["SU"]))

		# for lemma, lemma_data in dict_acc_succ.items():
		# 	p = random.random()
		# 	if p <= 0.8:
		# 		train_lemmi["succ"].append((lemma,
		# 						lemma_data["succession/iteration/distributivity"]["A"],
		# 						lemma_data["succession/iteration/distributivity"]["SU"]))

		# 		test_lemmi["acc"].append((lemma,
		# 					lemma_data["greater_plurality/accumulation"]["A"],
		# 					lemma_data["greater_plurality/accumulation"]["SU"]))

		# 	else:
		# 		test_lemmi["succ"].append((lemma,
		# 						lemma_data["succession/iteration/distributivity"]["A"],
		# 						lemma_data["succession/iteration/distributivity"]["SU"]))
		# 		train_lemmi["acc"].append((lemma,
		# 						lemma_data["greater_plurality/accumulation"]["A"],
		# 						lemma_data["greater_plurality/accumulation"]["SU"]))


		lemmi = [(lemma, lemma_data) for lemma, lemma_data in lemmi if lemma not in dict_juxt]
		for lemma, lemma_data in lemmi:
			p = random.random()

			if lemma in lemmi_comuni_acc_succ:
				if p <= 0.8 and sum([el[1] for el in train_lemmi["succ"]])<=75: ## COSTRUZIONI IN TRAIN e DISTRATTORI IN TEST
					train_lemmi["succ"].append((lemma,
								lemma_data["succession/iteration/distributivity"]["A"],
								lemma_data["succession/iteration/distributivity"]["SU"]))
					test_lemmi["acc"].append((lemma,
							lemma_data["greater_plurality/accumulation"]["A"],
							lemma_data["greater_plurality/accumulation"]["SU"]))
				else:   ## COSTRUZIONI IN TEST e DISTRATTORI IN TRAIN
					train_lemmi["acc"].append((lemma,
								lemma_data["greater_plurality/accumulation"]["A"],
								lemma_data["greater_plurality/accumulation"]["SU"]))
					test_lemmi["succ"].append((lemma,
								lemma_data["succession/iteration/distributivity"]["A"],
								lemma_data["succession/iteration/distributivity"]["SU"]))

			else:
				if p <= 0.7: ## TRAIN
					train_lemmi["juxt"].append((lemma,
								lemma_data["juxtaposition/contact"]["A"],
								lemma_data["juxtaposition/contact"]["SU"]))
					train_lemmi["acc"].append((lemma,
								lemma_data["greater_plurality/accumulation"]["A"],
								lemma_data["greater_plurality/accumulation"]["SU"]))
					train_lemmi["succ"].append((lemma,
								lemma_data["succession/iteration/distributivity"]["A"],
								lemma_data["succession/iteration/distributivity"]["SU"]))
				else:
					test_lemmi["juxt"].append((lemma,
								lemma_data["juxtaposition/contact"]["A"],
								lemma_data["juxtaposition/contact"]["SU"]))
					test_lemmi["acc"].append((lemma,
								lemma_data["greater_plurality/accumulation"]["A"],
								lemma_data["greater_plurality/accumulation"]["SU"]))
					test_lemmi["succ"].append((lemma,
								lemma_data["succession/iteration/distributivity"]["A"],
								lemma_data["succession/iteration/distributivity"]["SU"]))

		train_curr = {"juxt-a": sum(el[1] for el in train_lemmi["juxt"]),
				"succ-a": 0,
				"succ-su":0,
				"acc-su": 0}
		test_curr = {"juxt-a": 0,
				"succ-a": 0,
				"succ-su":0,
				"acc-su": 0}
		train_curr_lemmi = {"juxt-a": [el[0] for el in train_lemmi["juxt"] if el[1]>0],
					"succ-a": [],
					"succ-su":[],
					"acc-su": []}
		test_curr_lemmi = {"juxt-a": [],
					"succ-a": [],
					"succ-su":[],
					"acc-su": []}
		sample(test_lemmi, test_curr, test_curr_lemmi, "juxt", "a", 30)
		sample(train_lemmi, train_curr, train_curr_lemmi, "succ", "su", 60)
		sample(test_lemmi, test_curr, test_curr_lemmi, "succ", "su", 15)
		# sample(train_lemmi, train_curr, train_curr_lemmi, "juxt", "a", 60-train_curr["succ-su"])
		sample(train_lemmi, train_curr, train_curr_lemmi, "succ", "a", 60)
		sample(test_lemmi, test_curr, test_curr_lemmi, "succ", "a", 15)

		sample(train_lemmi, train_curr, train_curr_lemmi, "acc", "su", 120)
		sample(test_lemmi, test_curr, test_curr_lemmi, "acc", "su", 30)

		logging.info(f"Setting 'SIMPLE', iter {it}, train: {train_curr}, TOT={sum(train_curr.values())}")

		# logging.info(f"TRAIN: Selected lemmas for A-juxt: {', '.join(train_curr_lemmi['juxt-a'])}")
		# logging.info(f"TRAIN: Selected lemmas for A-succ: {', '.join(train_curr_lemmi['succ-a'])}")
		# logging.info(f"TRAIN: Selected lemmas for SU-succ: {', '.join(train_curr_lemmi['succ-su'])}")
		# logging.info(f"TRAIN: Selected lemmas for SU-acc: {', '.join(train_curr_lemmi['acc-su'])}")

		logging.info(f"Setting 'SIMPLE', iter {it}, test: {test_curr}, TOT={sum(test_curr.values())}")

		# logging.info(f"TEST: Selected lemmas for A-juxt: {', '.join(test_curr_lemmi['juxt-a'])}")
		# logging.info(f"TEST: Selected lemmas for A-succ: {', '.join(test_curr_lemmi['succ-a'])}")
		# logging.info(f"TEST: Selected lemmas for SU-succ: {', '.join(test_curr_lemmi['succ-su'])}")
		# logging.info(f"TEST: Selected lemmas for SU-acc: {', '.join(test_curr_lemmi['acc-su'])}")

		train_file = []
		test_file = []

		populate_split(costruzioni_A, train_curr_lemmi["juxt-a"], train_file,
									test_curr_lemmi["juxt-a"], test_file)
		populate_split(costruzioni_A, train_curr_lemmi["succ-a"], train_file,
									test_curr_lemmi["succ-a"], test_file)
		populate_split(costruzioni_SU, train_curr_lemmi["succ-su"], train_file,
									test_curr_lemmi["succ-su"], test_file)
		populate_split(costruzioni_SU, train_curr_lemmi["acc-su"], train_file,
									test_curr_lemmi["acc-su"], test_file)

		random.shuffle(train_file)
		random.shuffle(test_file)
		write_csv(f"data/data_set/ex_2/simple/full/ex2_simple_train_{it}.csv", train_file)
		write_csv(f"data/data_set/ex_2/simple/full/ex2_simple_test_{it}.csv", test_file)