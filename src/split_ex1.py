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
	label=category+preposition
	while i<len(orig_lemmas[category]) and counts_dict[label] <= max_n:
		lemma, count_a, count_su = orig_lemmas[category][i]
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
		writer = csv.DictWriter(f, delimiter=delimiter, fieldnames=["NPN", "construction", 'preposition',
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
		lemmi[el['noun']][el['construction']]["A"] += 1
		lemmi_tot[el['noun']][el['construction']] +=1
	for el in costruzioni_SU:
		lemmi[el['noun']][el['construction']]["SU"] += 1
		lemmi_tot[el['noun']][el['construction']] +=1

	for el in distrattori_A:
		lemmi[el['noun']][el['construction']]["A"] += 1
		lemmi_tot[el['noun']][el['construction']] +=1

	for el in distrattori_SU:
		lemmi[el['noun']][el['construction']]["SU"] += 1
		lemmi_tot[el['noun']][el['construction']] +=1

	lemmi_comuni = [lemma for lemma, lemma_data in lemmi_tot.items() if lemma_data["yes"] > 0 and lemma_data["no"] > 0]
	lemmi_distrattori = [lemma for lemma, lemma_data in lemmi_tot.items() if lemma_data["yes"] == 0 and lemma_data["no"] > 0]
	lemmi_costruzioni = [lemma for lemma, lemma_data in lemmi_tot.items() if lemma_data["yes"] > 0 and lemma_data["no"] == 0]

	return list(lemmi.items()), lemmi_tot, lemmi_comuni, lemmi_costruzioni, lemmi_distrattori

def count_pnpn(distrattori_A):
	lemmi = set()

	for el in distrattori_A:
		if el["Type"] == "PNPN":
			lemmi.add(el['noun'])

	return lemmi

def count_lemmas_distractor(costruzioni_A, costruzioni_SU,
							distrattori_A, distrattori_SU):
	lemmi = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(int)))
	lemmi_tot = collections.defaultdict(lambda:collections.defaultdict(int))
	lemmi_other = collections.defaultdict(lambda:collections.defaultdict(int))

	for el in costruzioni_A:
		lemmi[el['noun']][el['construction']]["A"] += 1
		lemmi_tot[el['noun']][el['construction']] +=1
	for el in costruzioni_SU:
		lemmi[el['noun']][el['construction']]["SU"] += 1
		lemmi_tot[el['noun']][el['construction']] +=1

	for el in distrattori_A:
		lemmi[el['noun']][el['construction']]["A"] += 1
		lemmi_other[el['noun']][el['other_cxn']] +=1
		lemmi_tot[el['noun']][el['construction']] +=1

	for el in distrattori_SU:
		lemmi[el['noun']][el['construction']]["SU"] += 1
		lemmi_other[el['noun']][el['other_cxn']] +=1
		lemmi_tot[el['noun']][el['construction']] +=1


	lemmi_comuni = [lemma for lemma, lemma_data in lemmi_tot.items() if lemma_data["yes"] > 0 and lemma_data["no"] > 0]
	lemmi_distrattori_other = [lemma for lemma, lemma_data in lemmi_other.items() if lemma_data["yes"] > 0 and lemma_data["no"] == 0]
	lemmi_distrattori_pseudo = [lemma for lemma, lemma_data in lemmi_other.items() if lemma_data["yes"] == 0 and lemma_data["no"] > 0]+[lemma for lemma, lemma_data in lemmi_other.items() if lemma_data["yes"] > 0 and lemma_data["no"] > 0]
	lemmi_costruzioni = [lemma for lemma, lemma_data in lemmi_tot.items() if lemma_data["yes"] > 0 and lemma_data["no"] == 0]

	return list(lemmi.items()), lemmi_tot, lemmi_comuni, lemmi_costruzioni, lemmi_distrattori_other, lemmi_distrattori_pseudo


def count_lemmas_distractor2(costruzioni_A, costruzioni_SU,
							distrattori_A, distrattori_SU):
	lemmi = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(int)))
	lemmi_tot = collections.defaultdict(lambda:collections.defaultdict(int))
	lemmi_other = collections.defaultdict(lambda:collections.defaultdict(int))

	for el in costruzioni_A:
		lemmi[el['noun']][el['construction']]["A"] += 1
		lemmi_tot[el['noun']][el['construction']] +=1
	for el in costruzioni_SU:
		lemmi[el['noun']][el['construction']]["SU"] += 1
		lemmi_tot[el['noun']][el['construction']] +=1

	for el in distrattori_A:
		lemmi[el['noun']][el['construction']]["A"] += 1
		lemmi_other[el['noun']][el['other_cxn']] +=1
		lemmi_tot[el['noun']][el['construction']] +=1

	for el in distrattori_SU:
		lemmi[el['noun']][el['construction']]["SU"] += 1
		lemmi_other[el['noun']][el['other_cxn']] +=1
		lemmi_tot[el['noun']][el['construction']] +=1


	lemmi_comuni = [lemma for lemma, lemma_data in lemmi_tot.items() if lemma_data["yes"] > 0 and lemma_data["no"] > 0]
	lemmi_distrattori_other = [lemma for lemma, lemma_data in lemmi_other.items() if lemma_data["yes"] > 0 and lemma_data["no"] == 0]+[lemma for lemma, lemma_data in lemmi_other.items() if lemma_data["yes"] > 0 and lemma_data["no"] > 0]
	lemmi_distrattori_pseudo = [lemma for lemma, lemma_data in lemmi_other.items() if lemma_data["yes"] == 0 and lemma_data["no"] > 0]
	lemmi_costruzioni = [lemma for lemma, lemma_data in lemmi_tot.items() if lemma_data["yes"] > 0 and lemma_data["no"] == 0]

	return list(lemmi.items()), lemmi_tot, lemmi_comuni, lemmi_costruzioni, lemmi_distrattori_other, lemmi_distrattori_pseudo

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
	_SEED = 1362

	costruzioni_A_file = "data/source/A_construction_max30.csv"
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

	lemmi, lemmi_tot,\
		lemmi_comuni, \
		lemmi_costruzioni, \
		lemmi_distrattori = count_lemmas_simple(costruzioni_A, costruzioni_SU,
										distrattori_A, distrattori_SU)
	distrattori_SU_dict = {lemma:lemma_data for lemma, lemma_data in lemmi if lemma_data["no"]["SU"]>0}
	lemmi_distrattori_SU = list(distrattori_SU_dict.keys())
	distrattori_SU_dict = list(distrattori_SU_dict.items())

	for it in range(_NFOLDS):
		logging.info(f"Iteration n. {it}")

		random.shuffle(lemmi)

		train_lemmi = {"C": [], "D": []}
		test_lemmi = {"C": [], "D": []}

		for lemma, lemma_data in distrattori_SU_dict:
			p = random.random()

			if lemma in lemmi_comuni:
				if p <= 0.2: ## DISTRATTORI IN TEST
					test_lemmi["D"].append((lemma, lemma_data["no"]["A"], lemma_data["no"]["SU"]))
				else:   ## DISTRATTORI IN TRAIN
					train_lemmi["D"].append((lemma, lemma_data["no"]["A"], lemma_data["no"]["SU"]))
			else:
				if p <= 0.8: ## TRAIN
					train_lemmi["D"].append((lemma, lemma_data["no"]["A"], lemma_data["no"]["SU"]))
				else:  ## TEST
					test_lemmi["D"].append((lemma, lemma_data["no"]["A"], lemma_data["no"]["SU"]))

		lemmi = [(lemma, lemma_data) for lemma, lemma_data in lemmi if lemma not in lemmi_distrattori_SU]

		for lemma, lemma_data in lemmi:
			p = random.random()

			if lemma in lemmi_comuni:
				if p <= 0.8: ## COSTRUZIONI IN TRAIN e DISTRATTORI IN TEST
					train_lemmi["C"].append((lemma, lemma_data["yes"]["A"], lemma_data["yes"]["SU"]))
					test_lemmi["D"].append((lemma, lemma_data["no"]["A"], lemma_data["no"]["SU"]))
				else:   ## COSTRUZIONI IN TEST e DISTRATTORI IN TRAIN
					train_lemmi["D"].append((lemma, lemma_data["no"]["A"], lemma_data["no"]["SU"]))
					test_lemmi["C"].append((lemma, lemma_data["yes"]["A"], lemma_data["yes"]["SU"]))
			else:
				if p <= 0.8: ## TRAIN
					if lemma in lemmi_costruzioni:  ## COSTRUZIONE
						train_lemmi["C"].append((lemma, lemma_data["yes"]["A"], lemma_data["yes"]["SU"]))
					else:                           ## DISTRATTORE
						train_lemmi["D"].append((lemma, lemma_data["no"]["A"], lemma_data["no"]["SU"]))
				else:  ## TEST
					if lemma in lemmi_costruzioni:   ## COSTRUZIONE
						test_lemmi["C"].append((lemma, lemma_data["yes"]["A"], lemma_data["yes"]["SU"]))
					else:                            ## DISTRATTORE
						test_lemmi["D"].append((lemma, lemma_data["no"]["A"], lemma_data["no"]["SU"]))
		# input()
		train_curr = {"Ca": 0, "Csu": 0, "Da":0, "Dsu": sum(el[2] for el in train_lemmi["D"])}
		test_curr = {"Ca": 0, "Da": 0, "Csu": 0, "Dsu": sum(el[2] for el in test_lemmi["D"])}
		train_curr_lemmi = {"Ca": [], "Csu": [], "Da":[],  "Dsu": [el[0] for el in train_lemmi["D"]]}
		test_curr_lemmi = {"Ca": [], "Da": [], "Csu": [], "Dsu": [el[0] for el in test_lemmi["D"]]}

		sample(train_lemmi, train_curr, train_curr_lemmi, "C", "a", 120)
		sample(train_lemmi, train_curr, train_curr_lemmi, "C", "su", 120)
		sample(train_lemmi, train_curr, train_curr_lemmi, "D", "a", 120)

		sample(test_lemmi, test_curr, test_curr_lemmi, "C", "a", 30)
		sample(test_lemmi, test_curr, test_curr_lemmi, "C", "su", 30)
		sample(test_lemmi, test_curr, test_curr_lemmi, "D", "a", 30)

		logging.info(f"Setting 'SIMPLE', iter {it}, train: {train_curr}, TOT={sum(train_curr.values())}")

		logging.info(f"TRAIN: Selected lemmas for A-constructions: {', '.join(train_curr_lemmi['Ca'])}")
		logging.info(f"TRAIN: Selected lemmas for SU-constructions: {', '.join(train_curr_lemmi['Csu'])}")
		logging.info(f"TRAIN: Selected lemmas for A-distractors: {', '.join(train_curr_lemmi['Da'])}")
		logging.info(f"TRAIN: Selected lemmas for SU-distractors: {', '.join(train_curr_lemmi['Dsu'])}")

		logging.info(f"Setting 'SIMPLE', iter {it}, test: {test_curr}, TOT={sum(test_curr.values())}")

		logging.info(f"TEST: Selected lemmas for A-constructions: {', '.join(test_curr_lemmi['Ca'])}")
		logging.info(f"TEST: Selected lemmas for SU-constructions: {', '.join(test_curr_lemmi['Csu'])}")
		logging.info(f"TEST: Selected lemmas for A-distractors: {', '.join(test_curr_lemmi['Da'])}")
		logging.info(f"TEST: Selected lemmas for SU-distractors: {', '.join(test_curr_lemmi['Dsu'])}")

		train_file = []
		test_file = []

		populate_split(costruzioni_A, train_curr_lemmi["Ca"], train_file,
									test_curr_lemmi["Ca"], test_file)
		populate_split(costruzioni_SU, train_curr_lemmi["Csu"], train_file,
									test_curr_lemmi["Csu"], test_file)
		populate_split(distrattori_A, train_curr_lemmi["Da"], train_file,
									test_curr_lemmi["Da"], test_file)
		populate_split(distrattori_SU, train_curr_lemmi["Dsu"], train_file,
							test_curr_lemmi["Dsu"], test_file)

		random.shuffle(train_file)
		random.shuffle(test_file)
		write_csv(f"data/data_set/ex_1/simple/full/ex1_simple_train_{it}.csv", train_file)
		write_csv(f"data/data_set/ex_1/simple/full/ex1_simple_test_{it}.csv", test_file)

	# === DISTRACTOR SETTING ===
	lemmi, lemmi_tot,\
		lemmi_comuni, \
		lemmi_costruzioni, \
		lemmi_distrattori_other, \
		lemmi_distrattori_pseudo = count_lemmas_distractor(costruzioni_A, costruzioni_SU,
														distrattori_A, distrattori_SU)

	lemmi_pnpn = count_pnpn(distrattori_A)
	lemmi_pnpn_dict = {lemma:lemma_data for lemma, lemma_data in lemmi if lemma in lemmi_pnpn}

	distrattori_SU_dict = {lemma:lemma_data for lemma, lemma_data in lemmi if lemma_data["no"]["SU"]>0}
	lemmi_distrattori_SU = list(distrattori_SU_dict.keys())
	distrattori_SU_dict = list(distrattori_SU_dict.items())

	### OTHER
	logging.info("Beginning sampling for setting SIMPLE")

	for it in range(_NFOLDS):
		logging.info(f"Iteration n. {it}")
		random.shuffle(lemmi)

		train_lemmi = {"C": [], "D": []}
		test_lemmi = {"C": [], "D": []}
		selected_cxn_a = ()

		for lemma, lemma_data in distrattori_SU_dict:
			if lemma in lemmi_distrattori_other:
				train_lemmi["D"].append((lemma, lemma_data["no"]["A"], lemma_data["no"]["SU"]))
			elif lemma in lemmi_distrattori_pseudo:
				test_lemmi["D"].append((lemma, lemma_data["no"]["A"], lemma_data["no"]["SU"]))

		lemmi = [(lemma, lemma_data) for lemma, lemma_data in lemmi if lemma not in lemmi_distrattori_SU]
		for lemma, lemma_data in lemmi:
			p = random.random()

			if lemma in lemmi_comuni:
				if lemma in lemmi_pnpn:
					if p <= 0.8: ## TRAIN
						train_lemmi["C"].append((lemma, lemma_data["yes"]["A"], lemma_data["yes"]["SU"]))
						if lemma in lemmi_distrattori_pseudo:
							test_lemmi["D"].append((lemma, lemma_data["no"]["A"], lemma_data["no"]["SU"]))
					else:
						test_lemmi["C"].append((lemma, lemma_data["yes"]["A"], lemma_data["yes"]["SU"]))
						if lemma in lemmi_distrattori_other:
							train_lemmi["D"].append((lemma, lemma_data["no"]["A"], lemma_data["no"]["SU"]))
				elif lemma in lemmi_distrattori_other:
					train_lemmi["D"].append((lemma, lemma_data["no"]["A"], lemma_data["no"]["SU"]))
					test_lemmi["C"].append((lemma, lemma_data["yes"]["A"], lemma_data["yes"]["SU"]))
				elif lemma in lemmi_distrattori_pseudo:
					test_lemmi["D"].append((lemma, lemma_data["no"]["A"], lemma_data["no"]["SU"]))
					train_lemmi["C"].append((lemma, lemma_data["yes"]["A"], lemma_data["yes"]["SU"]))

			else:
				if lemma in lemmi_distrattori_other and not lemma in lemmi_pnpn:
					train_lemmi["D"].append((lemma, lemma_data["no"]["A"], lemma_data["no"]["SU"]))
				elif lemma in lemmi_distrattori_pseudo:
					test_lemmi["D"].append((lemma, lemma_data["no"]["A"], lemma_data["no"]["SU"]))
				else:
					if p <= 0.8: ## TRAIN
						train_lemmi["C"].append((lemma, lemma_data["yes"]["A"], lemma_data["yes"]["SU"]))
					else:  ## TEST
						test_lemmi["C"].append((lemma, lemma_data["yes"]["A"], lemma_data["yes"]["SU"]))

		train_curr = {"Ca": 0,
				"Csu": 0,
				"Da":sum(el[1] for el in train_lemmi["D"]),
				"Dsu": sum(el[2] for el in train_lemmi["D"])}
		test_curr = {"Ca": 0,
			"Da": 0,
			"Csu": 0,
			"Dsu": 0}
		train_curr_lemmi = {"Ca": [],
						"Csu": [],
						"Da":[el[0] for el in train_lemmi["D"] if el[1]>0],
					 	"Dsu": [el[0] for el in train_lemmi["D"] if el[2]>0]}
		test_curr_lemmi = {"Ca": [],
						"Da": [],
					 	"Csu": [],
					  	"Dsu": []}

		pnpn_lemma_to_sample = [(lemma, lemma_data["no"]["A"], lemma_data["no"]["SU"]) for lemma, lemma_data in lemmi_pnpn_dict.items()]

		sample(train_lemmi, train_curr, train_curr_lemmi, "C", "a", 120)
		sample(train_lemmi, train_curr, train_curr_lemmi, "C", "su", 120)
		sample({"D":pnpn_lemma_to_sample}, train_curr, train_curr_lemmi, "D", "a", 240 - train_curr["Dsu"])

		sample(test_lemmi, test_curr, test_curr_lemmi, "C", "a", 30)
		sample(test_lemmi, test_curr, test_curr_lemmi, "C", "su", 30)
		sample(test_lemmi, test_curr, test_curr_lemmi, "D", "a", 30)
		sample(test_lemmi, test_curr, test_curr_lemmi, "D", "su", 30)

		logging.info(f"Setting 'OTHER', iter {it}, train: {train_curr}, TOT={sum(train_curr.values())}")

		logging.info(f"TRAIN: Selected lemmas for A-constructions: {', '.join(train_curr_lemmi['Ca'])}")
		logging.info(f"TRAIN: Selected lemmas for SU-constructions: {', '.join(train_curr_lemmi['Csu'])}")
		logging.info(f"TRAIN: Selected lemmas for A-distractors: {', '.join(train_curr_lemmi['Da'])}")
		logging.info(f"TRAIN: Selected lemmas for SU-distractors: {', '.join(train_curr_lemmi['Dsu'])}")

		logging.info(f"Setting 'OTHER', iter {it}, test: {test_curr}, TOT={sum(test_curr.values())}")

		logging.info(f"TEST: Selected lemmas for A-constructions: {', '.join(test_curr_lemmi['Ca'])}")
		logging.info(f"TEST: Selected lemmas for SU-constructions: {', '.join(test_curr_lemmi['Csu'])}")
		logging.info(f"TEST: Selected lemmas for A-distractors: {', '.join(test_curr_lemmi['Da'])}")
		logging.info(f"TEST: Selected lemmas for SU-distractors: {', '.join(test_curr_lemmi['Dsu'])}")

		train_file = []
		test_file = []

		populate_split(costruzioni_A, train_curr_lemmi["Ca"], train_file,
									test_curr_lemmi["Ca"], test_file)
		populate_split(costruzioni_SU, train_curr_lemmi["Csu"], train_file,
									test_curr_lemmi["Csu"], test_file)
		populate_split(distrattori_A, train_curr_lemmi["Da"], train_file,
									test_curr_lemmi["Da"], test_file)
		populate_split(distrattori_SU, train_curr_lemmi["Dsu"], train_file,
									test_curr_lemmi["Dsu"], test_file)

		random.shuffle(train_file)
		random.shuffle(test_file)

		write_csv(f"data/data_set/ex_1/other/full/ex1_other_train_{it}.csv", train_file)
		write_csv(f"data/data_set/ex_1/other/full/ex1_other_test_{it}.csv", test_file)

	lemmi, lemmi_tot,\
		lemmi_comuni, \
		lemmi_costruzioni, \
		lemmi_distrattori_other, \
		lemmi_distrattori_pseudo = count_lemmas_distractor2(costruzioni_A, costruzioni_SU,
														distrattori_A, distrattori_SU)

	lemmi_pnpn = count_pnpn(distrattori_A)
	lemmi_pnpn_dict = {lemma:lemma_data for lemma, lemma_data in lemmi if lemma in lemmi_pnpn}

	distrattori_SU_dict = {lemma:lemma_data for lemma, lemma_data in lemmi if lemma_data["no"]["SU"]>0}
	lemmi_distrattori_SU = list(distrattori_SU_dict.keys())
	distrattori_SU_dict = list(distrattori_SU_dict.items())

	### PSEUDO
	logging.info("Beginning sampling for setting PSEUDO")

	for it in range(_NFOLDS):
		logging.info(f"Iteration n. {it}")
		random.shuffle(lemmi)

		train_lemmi = {"C": [], "D": []}
		test_lemmi = {"C": [], "D": []}
		selected_cxn_a = ()

		for lemma, lemma_data in distrattori_SU_dict:
			if lemma in lemmi_distrattori_other:
				test_lemmi["D"].append((lemma, lemma_data["no"]["A"], lemma_data["no"]["SU"]))
			elif lemma in lemmi_distrattori_pseudo:
				train_lemmi["D"].append((lemma, lemma_data["no"]["A"], lemma_data["no"]["SU"]))

		lemmi = [(lemma, lemma_data) for lemma, lemma_data in lemmi if lemma not in lemmi_distrattori_SU]
		for lemma, lemma_data in lemmi:
			p = random.random()
			if lemma in lemmi_comuni:
				if p <= 0.8: ## TRAIN
					train_lemmi["C"].append((lemma, lemma_data["yes"]["A"], lemma_data["yes"]["SU"]))
					if lemma in lemmi_distrattori_other:
						test_lemmi["D"].append((lemma, lemma_data["no"]["A"], lemma_data["no"]["SU"]))
				else:
					test_lemmi["C"].append((lemma, lemma_data["yes"]["A"], lemma_data["yes"]["SU"]))
					if lemma in lemmi_distrattori_pseudo:
						train_lemmi["D"].append((lemma, lemma_data["no"]["A"], lemma_data["no"]["SU"]))
			else:
				if lemma in lemmi_distrattori_other:
					test_lemmi["D"].append((lemma, lemma_data["no"]["A"], lemma_data["no"]["SU"]))
				elif lemma in lemmi_distrattori_pseudo:
					train_lemmi["D"].append((lemma, lemma_data["no"]["A"], lemma_data["no"]["SU"]))
				else:
					if p <= 0.8: ## TRAIN
						train_lemmi["C"].append((lemma, lemma_data["yes"]["A"], lemma_data["yes"]["SU"]))
					else:  ## TEST
						test_lemmi["C"].append((lemma, lemma_data["yes"]["A"], lemma_data["yes"]["SU"]))

		train_curr = {"Ca": 0,
				"Csu": 0,
				"Da":sum(el[1] for el in train_lemmi["D"]),
				"Dsu": sum(el[2] for el in train_lemmi["D"])}
		test_curr = {"Ca": 0,
			"Da": 0,
			"Csu": 0,
			"Dsu": sum(el[2] for el in test_lemmi["D"])}
		train_curr_lemmi = {"Ca": [],
						"Csu": [],
						"Da":[el[0] for el in train_lemmi["D"] if el[1]>0],
					 	"Dsu": [el[0] for el in train_lemmi["D"] if el[2]>0]}

		test_curr_lemmi = {"Ca": [],
						"Da": [],
					 	"Csu": [],
					  	"Dsu": [el[0] for el in test_lemmi["D"] if el[2]>0]}


		pnpn_lemma_to_sample = [(lemma, lemma_data["no"]["A"], lemma_data["no"]["SU"]) for lemma, lemma_data in lemmi_pnpn_dict.items()]

		sample(train_lemmi, train_curr, train_curr_lemmi, "C", "a", 120)
		sample(train_lemmi, train_curr, train_curr_lemmi, "C", "su", 120)

		sample({"D":pnpn_lemma_to_sample}, test_curr, test_curr_lemmi, "D", "a", 60 - test_curr["Dsu"])


		sample(test_lemmi, test_curr, test_curr_lemmi, "C", "a", 30)
		sample(test_lemmi, test_curr, test_curr_lemmi, "C", "su", 30)
		sample(test_lemmi, test_curr, test_curr_lemmi, "D", "a", 30)

		logging.info(f"Setting 'PSEUDO', iter {it}, train: {train_curr}, TOT={sum(train_curr.values())}")

		logging.info(f"TRAIN: Selected lemmas for A-constructions: {', '.join(train_curr_lemmi['Ca'])}")
		logging.info(f"TRAIN: Selected lemmas for SU-constructions: {', '.join(train_curr_lemmi['Csu'])}")
		logging.info(f"TRAIN: Selected lemmas for A-distractors: {', '.join(train_curr_lemmi['Da'])}")
		logging.info(f"TRAIN: Selected lemmas for SU-distractors: {', '.join(train_curr_lemmi['Dsu'])}")

		logging.info(f"Setting 'PSEUDO', iter {it}, test: {test_curr}, TOT={sum(test_curr.values())}")

		logging.info(f"TEST: Selected lemmas for A-constructions: {', '.join(test_curr_lemmi['Ca'])}")
		logging.info(f"TEST: Selected lemmas for SU-constructions: {', '.join(test_curr_lemmi['Csu'])}")
		logging.info(f"TEST: Selected lemmas for A-distractors: {', '.join(test_curr_lemmi['Da'])}")
		logging.info(f"TEST: Selected lemmas for SU-distractors: {', '.join(test_curr_lemmi['Dsu'])}")
		train_file = []
		test_file = []

		populate_split(costruzioni_A, train_curr_lemmi["Ca"], train_file,
									test_curr_lemmi["Ca"], test_file)
		populate_split(costruzioni_SU, train_curr_lemmi["Csu"], train_file,
									test_curr_lemmi["Csu"], test_file)
		populate_split(distrattori_A, train_curr_lemmi["Da"], train_file,
									test_curr_lemmi["Da"], test_file)
		populate_split(distrattori_SU, train_curr_lemmi["Dsu"], train_file,
									test_curr_lemmi["Dsu"], test_file)

		random.shuffle(train_file)
		random.shuffle(test_file)

		write_csv(f"data/data_set/ex_1/pseudo/full/ex1_pseudo_train_{it}.csv", train_file)
		write_csv(f"data/data_set/ex_1/pseudo/full/ex1_pseudo_test_{it}.csv", test_file)