
import csv
import random
import collections


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

# === Campionamento casuale ===
def sample_rows(rows, n=150, num_seeds=5):
	"""Esegue più campionamenti casuali con diversi seed"""
	samples = []
	for i in range(num_seeds):
		seed = math.floor(random.random() * 100)
		random.seed(seed)
		if len(rows) <= n:
			samples.append(rows)
		else:
			samples.append(random.sample(rows, n))
	return samples


# === Divisione trainig e test ===
# Così si evita che lo stesso lemma compaia solo nel train o solo nel test, mantenendo un equilibrio interno a ciascun lemma
def split_rows_lemma_per_lemma(rows, train_ratio, num_seeds=5):
	"""Split per lemma 80/20, eseguito con diversi seed"""

	sample_train_test =[]
	for i in range (num_seeds):
		seed = math.floor(random.random()*100)
		random.seed(seed)

		lemma_dict = defaultdict(list)
		for r in rows:
			lemma_dict[r["noun"].strip()].append(r)

		train_rows = []
		test_rows = []

		for lemma, lemma_rows in lemma_dict.items():
			n_train = int(len(lemma_rows) * train_ratio)
			sampled_train = random.sample(lemma_rows, n_train)
			sampled_test = [r for r in lemma_rows if r not in sampled_train]

			train_rows.extend(sampled_train)
			test_rows.extend(sampled_test)

		sample_train_test.append((train_rows, test_rows))

		return sample_train_test
# Separare righe per lemma

def split_by_lemma(rows):
	lemma_dict = defaultdict(list)
	for r in rows:
		lemma_dict[r["noun"].strip()].append(r)
	return lemma_dict


def split_rows_type_per_type(rows, train_ratio):
	"""Split per Type 80/20"""
	type_dict = defaultdict(list)
	for r in rows:
		type_dict[r["Type"].strip()].append(r)

	train_rows = []
	test_rows = []

	for t, t_rows in type_dict.items():
		if len(t_rows) > 1:
			n_train = int(len(t_rows) * train_ratio)
			sampled_train = random.sample(t_rows, n_train)
			sampled_test = [r for r in t_rows if r not in sampled_train]
		else:
			# Se c'è una sola istanza, la metto tutta nel training
			sampled_train = t_rows
			sampled_test = []
		train_rows.extend(sampled_train)
		test_rows.extend(sampled_test)

	return train_rows, test_rows

if __name__ == "__main__":

	_NFOLDS = 5
	random.seed(1362)
	costruzioni_A_file = "data/source/A_construction_max30.csv"
	costruzioni_SU_file = "data/source/SU_construction_max30.csv"

	distrattori_A_file = "data/source/A_distractor_max30.csv"
	distrattori_SU_file = "data/source/SU_distractor_max30.csv"

	# === Lettura file filtrati ===
	costruzioni_A = read_csv(costruzioni_A_file)
	costruzioni_SU = read_csv(costruzioni_SU_file)
	distrattori_A = read_csv(distrattori_A_file)
	distrattori_SU = read_csv(distrattori_SU_file)

	# === SIMPLE SETTING ===
	lemmi, lemmi_tot,\
		lemmi_comuni, \
		lemmi_costruzioni, \
		lemmi_distrattori = count_lemmas_simple(costruzioni_A, costruzioni_SU,
										distrattori_A, distrattori_SU)
	distrattori_SU_dict = {lemma:lemma_data for lemma, lemma_data in lemmi if lemma_data["no"]["SU"]>0}
	lemmi_distrattori_SU = list(distrattori_SU_dict.keys())
	distrattori_SU_dict = list(distrattori_SU_dict.items())

	for it in range(_NFOLDS):

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

		print(f"Setting 'SIMPLE', iter {it}, train: {train_curr}, TOT={sum(train_curr.values())}")
		print(f"Setting 'SIMPLE', iter {it}, test: {test_curr}, TOT={sum(test_curr.values())}")

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

	for it in range(_NFOLDS):
		random.shuffle(lemmi)

		train_lemmi = {"C": [], "D": []}
		test_lemmi = {"C": [], "D": []}
		selected_cxn_a = ()

		for lemma, lemma_data in distrattori_SU_dict:
			# p = random.random()
			if lemma in lemmi_distrattori_other:
				train_lemmi["D"].append((lemma, lemma_data["no"]["A"], lemma_data["no"]["SU"]))
			elif lemma in lemmi_distrattori_pseudo:
				test_lemmi["D"].append((lemma, lemma_data["no"]["A"], lemma_data["no"]["SU"]))

		lemmi = [(lemma, lemma_data) for lemma, lemma_data in lemmi if lemma not in lemmi_distrattori_SU]
		# lemmi_distrattori_other = [lemma for lemma in lemmi_distrattori_other if not lemma in lemmi_pnpn]
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
		# input()
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

		# pnpn_to_sample = 240 - train_curr["Da"]+train_curr["Dsu"]
		pnpn_lemma_to_sample = [(lemma, lemma_data["no"]["A"], lemma_data["no"]["SU"]) for lemma, lemma_data in lemmi_pnpn_dict.items()]

		# sample({"D":pnpn_lemma_to_sample}, train_curr, train_curr_lemmi, "D", "a", 240)

		sample(train_lemmi, train_curr, train_curr_lemmi, "C", "a", 120)
		sample(train_lemmi, train_curr, train_curr_lemmi, "C", "su", 120)
		sample({"D":pnpn_lemma_to_sample}, train_curr, train_curr_lemmi, "D", "a", 240 - train_curr["Dsu"])

		sample(test_lemmi, test_curr, test_curr_lemmi, "C", "a", 30)
		sample(test_lemmi, test_curr, test_curr_lemmi, "C", "su", 30)
		sample(test_lemmi, test_curr, test_curr_lemmi, "D", "a", 30)
		sample(test_lemmi, test_curr, test_curr_lemmi, "D", "su", 30)

		print(f"Setting 'OTHER', iter {it}, train: {train_curr}, TOT={sum(train_curr.values())}")
		print(f"Setting 'OTHER', iter {it}, test: {test_curr}, TOT={sum(test_curr.values())}")

		train_file = []
		test_file = []

		populate_split(costruzioni_A, train_curr_lemmi["Ca"], train_file,
									test_curr_lemmi["Ca"], test_file)
		populate_split(costruzioni_SU, train_curr_lemmi["Csu"], train_file,
									test_curr_lemmi["Csu"], test_file)
		# curr_len_train = len(train_file)
		populate_split(distrattori_A, train_curr_lemmi["Da"], train_file,
									test_curr_lemmi["Da"], test_file)
		# n_distrattori_A = len(train_file) - curr_len_train
		# curr_len_train = len(train_file)
		populate_split(distrattori_SU, train_curr_lemmi["Dsu"], train_file,
									test_curr_lemmi["Dsu"], test_file)
		# n_distrattori_SU = len(train_file) - curr_len_train


		# train_file = []
		# test_file = []

		# populate_split(costruzioni_A, train_curr_lemmi["Ca"], train_file,
		#                             test_curr_lemmi["Ca"], test_file)
		# populate_split(costruzioni_SU, train_curr_lemmi["Csu"], train_file,
		#                             test_curr_lemmi["Csu"], test_file)
		# curr_len_train = len(train_file)
		# populate_split(distrattori_A, train_curr_lemmi["Da"], train_file,
		#                             test_curr_lemmi["Da"], test_file,
		#                             add_filter_train=lambda x: x["Type"] in ["verbal", "NsuNgiù"])
		# n_distrattori_A = len(train_file) - curr_len_train
		# curr_len_train = len(train_file)
		# populate_split(distrattori_SU, train_curr_lemmi["Dsu"], train_file,
		#                             test_curr_lemmi["Dsu"], test_file,
		#                             add_filter_train=lambda x: x["Type"] in ["verbal", "NsuNgiù"])
		# n_distrattori_SU = len(train_file) - curr_len_train

	#     for el in distrattori_A:
	#         if el['noun'] in train_curr_lemmi["Da"] and el["Type"] in ["PNPN"] and n_distrattori_A < 240:
	#             train_file.append(el)
	#             n_distrattori_A += 1

		random.shuffle(train_file)
		random.shuffle(test_file)

		write_csv(f"data/data_set/ex_1/other/full/ex1_other_train_{it}.csv", train_file)
		write_csv(f"data/data_set/ex_1/other/full/ex1_other_test_{it}.csv", test_file)

	# for it in range(_NFOLDS):
	#     random.shuffle(lemmi)

	#     train_lemmi = {"C": [], "D": []}
	#     test_lemmi = {"C": [], "D": []}

	#     for lemma, lemma_data in lemmi:
	#         p = random.random()
	#         if lemma in lemmi_comuni:
	#             category = list(lemma_data["no"].keys())[0]
	#             if category == "yes":
	#                 test_lemmi["D"].append((lemma, lemma_data["no"]["no"]["A"], lemma_data["no"]["no"]["SU"]))
	#                 train_lemmi["C"].append((lemma, lemma_data["yes"]["A"], lemma_data["yes"]["SU"]))
	#             else:
	#                 train_lemmi["D"].append((lemma, lemma_data["no"]["yes"]["A"], lemma_data["no"]["yes"]["SU"]))
	#                 test_lemmi["C"].append((lemma, lemma_data["yes"]["A"], lemma_data["yes"]["SU"]))

	#         else:
	#             if lemma in lemmi_distrattori:
	#                 sorted_other = sorted(lemma_data["no"].items(), key= lambda y: y[1]["A"]+y[1]["SU"])
	#                 category = list(sorted_other)[0][0]

	#                 if category == "yes":
	#                     test_lemmi["D"].append((lemma, lemma_data["no"]["no"]["A"], lemma_data["no"]["no"]["SU"]))
	#                 else:
	#                     train_lemmi["D"].append((lemma, lemma_data["no"]["yes"]["A"], lemma_data["no"]["yes"]["SU"]))

	#             else:
	#                 if p <= 0.5: ## TRAIN
	#                     train_lemmi["C"].append((lemma, lemma_data["yes"]["A"], lemma_data["yes"]["SU"]))
	#                 else:  ## TEST
	#                     test_lemmi["C"].append((lemma, lemma_data["yes"]["A"], lemma_data["yes"]["SU"]))

	#     train_curr = {"Ca": 0, "Csu": 0, "Da":0, "Dsu": 0}
	#     test_curr = {"Ca": 0, "Da": 0, "Csu": 0, "Dsu": 0}
	#     train_curr_lemmi = {"Ca": [], "Csu": [], "Da":[],  "Dsu": []}
	#     test_curr_lemmi = {"Ca": [], "Da": [], "Csu": [], "Dsu": []}

	#     for orig_lemmas, counts, lemmas, label, max_n in [(train_lemmi, train_curr, train_curr_lemmi, "Ca", 120),
	#                                                     (train_lemmi, train_curr, train_curr_lemmi, "Csu", 120),
	#                                                     (train_lemmi, train_curr, train_curr_lemmi, "Da", 120),
	#                                                     (train_lemmi, train_curr, train_curr_lemmi, "Dsu", 120),
	#                                                     (test_lemmi, test_curr, test_curr_lemmi, "Ca", 30),
	#                                                     (test_lemmi, test_curr, test_curr_lemmi, "Csu", 30),
	#                                                     (test_lemmi, test_curr, test_curr_lemmi, "Da", 30),
	#                                                     (test_lemmi, test_curr, test_curr_lemmi, "Dsu", 30)]:
	#         i = 0
	#         category, preposition = label[0], label[1:]
	#         while i<len(orig_lemmas[category]) and counts[label] <= max_n:
	#             lemma, count_a, count_su = orig_lemmas[category][i]
	#             if preposition == "a":
	#                 counts[label] += count_a
	#                 if count_a > 0:
	#                     lemmas[label].append(lemma)
	#             else:
	#                 counts[label] += count_su
	#                 if count_su > 0:
	#                     lemmas[label].append(lemma)
	#             i += 1

	#     train_file = []
	#     test_file = []

	#     populate_split(costruzioni_A, train_curr_lemmi["Ca"], train_file,
	#                                 test_curr_lemmi["Ca"], test_file)
	#     populate_split(costruzioni_SU, train_curr_lemmi["Csu"], train_file,
	#                                 test_curr_lemmi["Csu"], test_file)
	#     curr_len_train = len(train_file)
	#     populate_split(distrattori_A, train_curr_lemmi["Da"], train_file,
	#                                 test_curr_lemmi["Da"], test_file)
	#                                 # add_filter_train=lambda x: x["Type"] in ["verbal", "NsuNgiù"])
	#     n_distrattori_A = len(train_file) - curr_len_train
	#     curr_len_train = len(train_file)
	#     populate_split(distrattori_SU, train_curr_lemmi["Dsu"], train_file,
	#                                 test_curr_lemmi["Dsu"], test_file)
	#                                 # add_filter_train=lambda x: x["Type"] in ["verbal", "NsuNgiù"])
	#     n_distrattori_SU = len(train_file) - curr_len_train

	#     # for el in distrattori_A:
	#     #     if el['noun'] in train_curr_lemmi["Da"] and el["Type"] in ["PNPN"] and n_distrattori_A < 240:
	#     #         train_file.append(el)
	#     #         n_distrattori_A += 1

	#     random.shuffle(train_file)
	#     random.shuffle(test_file)

	#     write_csv(f"data/data_set/ex1_pseudo_train_{it}.csv", train_file)
	#     write_csv(f"data/data_set/ex1_pseudo_test_{it}.csv", test_file)