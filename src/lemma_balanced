
import csv
import os
import random
from collections import defaultdict

costruzioni_A_file = "/Users/gretagorzoni/Desktop/TESI_CODE/data/data_set/A_filtrato.csv"
costruzioni_SU_file = "/Users/gretagorzoni/Desktop/TESI_CODE/data/data_set/SU_filtrato.csv"

distrattori_A_file = "/Users/gretagorzoni/Desktop/TESI_CODE/data/data_set/A_distractor_filtrato.csv"
distrattori_SU_file = "/Users/gretagorzoni/Desktop/TESI_CODE/data/data_set/SU_distractor_filtrato.csv"



train_ratio = 0.8

# === Lista di righe, evita righe vuote ===
def read_csv(file_path, delimiter=";"):
    with open(file_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        return [row for row in reader if len(row) > 0]



# === Scrivere CSV ===
def write_csv(file_path, rows, delimiter=";"):
    with open(file_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter=delimiter, fieldnames= ["NPN",	"construction",	'preposition',	'noun',	'context_pre',	"costr",	'context_post','number_of_noun',	'Type',	'meaning',"syntactic_function"], restval= '_')
        writer.writeheader()
        writer.writerows(rows)
        
# === Divisione trainig e test ===
def split_rows_lemma_per_lemma(rows, train_ratio):
    """Split per lemma 80/20"""
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

    return train_rows, test_rows

# === Lettura file ===
costruzioni = read_csv(costruzioni_A_file) + read_csv(costruzioni_SU_file)
distrattori = read_csv(distrattori_A_file) + read_csv(distrattori_SU_file)

# === Distinzione lemmi condivisi ===
lemmi_costruzioni = set([row["noun"].strip() for row in costruzioni])
lemmi_distrattori = set([row["noun"].strip() for row in distrattori])
# intersezione tra i due set
lemmi_comuni = lemmi_costruzioni & lemmi_distrattori

# Separare righe per lemma
def split_by_lemma(rows):
    lemma_dict = defaultdict(list)
    for r in rows:
        lemma_dict[r["noun"].strip()].append(r)
    return lemma_dict

costruzioni_dict = split_by_lemma(costruzioni)
distrattori_dict = split_by_lemma(distrattori)

# === SCENARIO 1: Nessun vincolo sui lemmi ===
# verifica su ogni lemma!!!!!
train_c1, test_c1 = split_rows_lemma_per_lemma(costruzioni, train_ratio)
train_d1, test_d1 = split_rows_lemma_per_lemma(distrattori, train_ratio)
train1 = train_c1 + train_d1
test1 = test_c1 + test_d1
write_csv("/Users/gretagorzoni/Desktop/TESI_CODE/data/data_set/train_test_lemma_setting/lemma_balanced_train.csv", train1)
write_csv("/Users/gretagorzoni/Desktop/TESI_CODE/data/data_set/train_test_lemma_setting/lemma_balanced_test.csv", test1)

# === Scenario 2: training solo lemmi distrattori ===
# Training: costruzioni + distrattori che non sono comuni

train_c2, test_c2= split_rows_lemma_per_lemma(costruzioni, train_ratio)
# verifica su ogni lemma!!!!!
distrattori_unbalanced =[]
for rows in distrattori:
    if rows["noun"] not in lemmi_comuni:
        distrattori_unbalanced.append(rows)
train_d2, test_d_unbalabced_2 = split_rows_lemma_per_lemma(distrattori_unbalanced, train_ratio)

test_d_lemmi_comuni = []
for rows in distrattori:
    if rows["noun"] in lemmi_comuni:
        test_d_lemmi_comuni.append(rows)
        
test_d2 = test_d_unbalabced_2 + test_d_lemmi_comuni

train2 = train_c2 + train_d2
test2 = test_c2 + test_d2
write_csv("/Users/gretagorzoni/Desktop/TESI_CODE/data/data_set/train_test_lemma_setting/distr_unbalanced_train.csv", train2)
write_csv("/Users/gretagorzoni/Desktop/TESI_CODE/data/data_set/train_test_lemma_setting/distr_unbalanced_test.csv", test2)

# === Scenario 3: training solo lemmi distrattori ===
# Training: distrattori + costruzioni che non sono comuni

train_d3, test_d3= split_rows_lemma_per_lemma(distrattori, train_ratio)
# verifica su ogni lemma!!!!!
costruzioni_unbalanced =[]
for rows in costruzioni:
    if rows["noun"] not in lemmi_comuni:
        costruzioni_unbalanced.append(rows)
train_c3, test_c_unbalabced_3 = split_rows_lemma_per_lemma(distrattori_unbalanced, train_ratio)

test_c_lemmi_comuni = []
for rows in costruzioni:
    if rows["noun"] in lemmi_comuni:
        test_c_lemmi_comuni.append(rows)
        
test_c3 = test_c_unbalabced_3 + test_c_lemmi_comuni

train3 = train_c3 + train_d3
test3 = test_c3 + test_d3
write_csv("/Users/gretagorzoni/Desktop/TESI_CODE/data/data_set/train_test_lemma_setting/constr_unbalanced_train.csv", train3)
write_csv("/Users/gretagorzoni/Desktop/TESI_CODE/data/data_set/train_test_lemma_setting/constr_unbalanced_test.csv", test3)

print(lemmi_comuni)


