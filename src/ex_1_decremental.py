import glob
import decremental_training

train_sizes = [240, 120, 60]


# # 1. CONTROL on SIMPLE - UNK 0.5

# dataset_folder =  "data/data_set/control/"
# embeddings_folder = "data/output/embeddings/simple/"
# output_folder = "data/data_set/control/sampled/"
# output_emb_folder = "data/output/embeddings/simple/sampled/control/"
# csv_pattern="*simple_train*.csv"
# emb_pattern="*UNK*train*.pkl"

# # 1.q CONTROL on SIMPLE - UNK 0.7

# dataset_folder =  "data/data_set/control/0.7/"
# embeddings_folder = "data/output/embeddings/simple/"
# output_folder = "data/data_set/control/0.7/sampled"
# output_emb_folder = "data/output/embeddings/simple/sampled/control/0.7/"
# csv_pattern="*simple_train*.csv"
# emb_pattern="*UNK*train*.pkl"

# # 2. PROBE on SIMPLE 

# dataset_folder =  "data/data_set/"
# embeddings_folder = "data/output/embeddings/simple/"
# output_folder = "data/data_set/sampled/"
# output_emb_folder = "data/output/embeddings/simple/sampled/"
# csv_pattern="*simple_train*.csv"
# emb_pattern="*train*.pkl"

# # 3. PROBE on OTHER

# dataset_folder =  "data/data_set/"
# embeddings_folder = "data/output/embeddings/other/"
# output_folder = "data/data_set/sampled/"
# output_emb_folder = "data/output/embeddings/other/sampled/"
# csv_pattern="*other_train*.csv"
# emb_pattern="*train*.pkl"



# # 4. PROBE on PSEUDO

# dataset_folder =  "data/data_set/"
# embeddings_folder = "data/output/embeddings/pseudo/"
# output_folder = "data/data_set/sampled/"
# output_emb_folder = "data/output/embeddings/pseudo/sampled/"
# csv_pattern="*pseudo_train*.csv"
# emb_pattern="*train*.pkl"

# # 5. BASELINE on SIMPLE - fastText

# dataset_folder =  "data/data_set/"
# embeddings_folder = "data/embeddings/fasttext/simple/"
# output_folder = "data/data_set/sampled/"
# output_emb_folder = "data/embeddings/fasttext/simple/sampled"
# csv_pattern="*simple_train*.csv"
# emb_pattern="*train*"


# # 6. BASELINE on OTHER - fastText

# dataset_folder =  "data/data_set/"
# embeddings_folder = "data/embeddings/fasttext/other/"
# output_folder = "data/data_set/sampled/"
# output_emb_folder = "data/embeddings/fasttext/other/sampled"
# csv_pattern="*other_train*.csv"
# emb_pattern="*train*"

# # 7. BASELINE on PSEUDO - fastText

# dataset_folder =  "data/data_set/"
# embeddings_folder = "data/embeddings/fasttext/pseudo/"
# output_folder = "data/data_set/sampled/"
# output_emb_folder = "data/embeddings/fasttext/pseudo/sampled"
# csv_pattern="*pseudo_train*.csv"
# emb_pattern="*train*"

# #26/11 PROBE SIMPLE 

# dataset_folder =  "data/data_set/ex_1/simple/full/"
# embeddings_folder = "data/embeddings/bert/simple/full/"
# output_folder = "data/data_set/ex_1/simple/sampled/"
# output_emb_folder = "data/embeddings/bert/simple/sampled/"

# key_values = ["UNK", "CLS", "PREP"]

# for k in key_values:

# 	csv_pattern="*train*.csv"
# 	emb_pattern="*" + k + "*train*.pkl"

# 	decremental_training.main(dataset_folder, embeddings_folder, output_folder, output_emb_folder, train_sizes, csv_pattern, emb_pattern, k)

 
#26/11 PROBE OTHEr
#non riesco a runnare perché ValueError: Non ci sono abbastanza esempi per no/su: 9 < 60

dataset_folder =  "data/data_set/ex_1/other/full/"
embeddings_folder = "data/embeddings/bert/other/full/"
output_folder = "data/data_set/ex_1/other/sampled/"
output_emb_folder = "data/embeddings/bert/other/sampled/"

key_values = ["UNK", "CLS", "PREP"]

for k in key_values:

	csv_pattern="*train*.csv"
	emb_pattern="*" + k + "*train*.pkl"

	decremental_training.main(dataset_folder, embeddings_folder, output_folder, output_emb_folder, train_sizes, csv_pattern, emb_pattern, k)

