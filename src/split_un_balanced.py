
import csv
import os
import random
from collections import defaultdict




costruzioni_A_file = "data/data_set/A_filtrato.csv"
costruzioni_SU_file = "data/data_set/SU_filtrato.csv"

distrattori_A_file = "data/data_set/A_distractor_filtrato.csv"
distrattori_SU_file = "data/data_set/SU_distractor_filtrato.csv"



train_ratio = 0.8

# === Lista di righe, evita righe vuote ===
def read_csv(file_path, delimiter=";"):
    with open(file_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        return [row for row in reader if len(row) > 0]
    
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



# === Scrivere CSV ===
def write_csv(file_path, rows, delimiter=";"):
    with open(file_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter=delimiter, fieldnames= ["NPN",	"construction",	'preposition',	'noun',	'context_pre',	"costr",	'context_post','number_of_noun',	'Type',	'meaning',"syntactic_function"], restval= '_')
        writer.writeheader()
        writer.writerows(rows)
        
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

# === Lettura e campionamento (150 per ciascun file) ===
costruzioni_A = sample_rows(read_csv(costruzioni_A_file), 150)
costruzioni_SU = sample_rows(read_csv(costruzioni_SU_file), 150)
distrattori_A = sample_rows(read_csv(distrattori_A_file), 150)
distrattori_SU = sample_rows(read_csv(distrattori_SU_file), 150)

# Uniamo i campioni
costruzioni = costruzioni_A + costruzioni_SU
distrattori = distrattori_A + distrattori_SU

print(f"Campioni selezionati: costruzioni={len(costruzioni)}, distrattori={len(distrattori)}")

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
write_csv("data/data_set/train_test_lemma_setting/lemma_balanced_train.csv", train1)
write_csv("data/data_set/train_test_lemma_setting/lemma_balanced_test.csv", test1)

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
write_csv("data/data_set/train_test_lemma_setting/distr_unbalanced_train.csv", train2)
write_csv("data/data_set/train_test_lemma_setting/distr_unbalanced_test.csv", test2)

# === Scenario LEMMI===
# Training costruzioni + distrattori lemmi comuni - test solo lemmi non associati a stessa label del training
    
    train_c, test_c = split_rows_lemma_per_lemma(costruzioni, train_ratio)
    train_d, test_d = split_rows_lemma_per_lemma(distrattori, train_ratio)

    # insiemi dei lemmi visti nel train
    lemmi_train_costruzioni = set([r["noun"].strip() for r in train_c])
    lemmi_train_distrattori = set([r["noun"].strip() for r in train_d])

    # Test filtrato:
    #    - se un lemma è nel train come costruzione → nel test può comparire solo come distrattore
    #    - se un lemma è nel train come distrattore → nel test può comparire solo come costruzione
    filtered_test_c = [r for r in test_c if r["noun"].strip() not in lemmi_train_costruzioni]
    filtered_test_d = [r for r in test_d if r["noun"].strip() not in lemmi_train_distrattori]

    # appaiono in entrambe le categorie ma con label diversa)
    lemmi_comuni = lemmi_train_costruzioni.intersection(lemmi_train_distrattori)

    # lemmi comuni ma etichettati con label diversa
    extra_test_c = [r for r in test_c if r["noun"].strip() in lemmi_train_distrattori]
    extra_test_d = [r for r in test_d if r["noun"].strip() in lemmi_train_costruzioni]

    test_c_final = filtered_test_c + extra_test_c
    test_d_final = filtered_test_d + extra_test_d


    train_final = train_c + train_d
    test_final = test_c_final + test_d_final

    print(f"Costruzioni train/test: {len(train_c)} / {len(test_c_final)}")
    print(f"Distrattori train/test: {len(train_d)} / {len(test_d_final)}")

    return train_final, test_final

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
write_csv("data/data_set/train_test_lemma_setting/constr_unbalanced_train.csv", train3)
write_csv("data/data_set/train_test_lemma_setting/constr_unbalanced_test.csv", test3)

print(lemmi_comuni)

# === Scenario 4: Type-balanced ===
# Ogni tipo di distrattore ('Type') compare sia in train che in test

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


output_dir = "data/data_set/train_test_distractor_setting"
os.makedirs(output_dir, exist_ok=True)

train_d4, test_d4 = split_rows_type_per_type(distrattori, train_ratio)
train_c4, test_c4 = split_rows_lemma_per_lemma(costruzioni, train_ratio)

train4 = train_c4 + train_d4
test4 = test_c4 + test_d4

write_csv("data/data_set/train_test_distractor_setting/type_balanced_train.csv", train4)
write_csv("data/data_set/train_test_distractor_setting/type_balanced_test.csv", test4)


# === Scenario 5: Selected types only in training ===
# Alcuni 'Type' solo nel training, altri solo nel test

train_only_types = {"NUMsuNUM", "NUMaNUM", "N_extended", "proper_name_inglobation", "thematic_target"}

distr_train5 = [r for r in distrattori if r["Type"].strip() in train_only_types]
distr_test5 = [r for r in distrattori if r["Type"].strip() not in train_only_types]

# Manteniamo la stessa divisione lemma-based per le costruzioni (bilanciate, senza particolari spartizioni)
train_c5, test_c5 = split_rows_lemma_per_lemma(costruzioni, train_ratio)

train5 = train_c5 + distr_train5
test5 = test_c5 + distr_test5

write_csv("data/data_set/train_test_distractor_setting/type_exclusive_train.csv", train5)
write_csv("data/data_set/train_test_distractor_setting/type_exclusive_test.csv", test5)

print("Scenario 4 e 5 completati ")



