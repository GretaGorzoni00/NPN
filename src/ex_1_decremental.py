import glob
import decremental_training

train_sizes = [240, 120, 60]


# # 1. CONTROL on SIMPLE - UNK

# dataset_folder =  "data/data_set/control"
# embeddings_folder = "data/output/embeddings/simple"
# output_folder = "data/data_set/control/sampled"
# output_emb_folder "data/output/embeddings/simple/sampled/control"
# csv_pattern="*simple_train*.csv"
# emb_pattern="*UNK*train*.pkl"

# # 1. PROBE on SIMPLE 

# dataset_folder =  "data/data_set"
# embeddings_folder = "data/output/embeddings/simple"
# output_folder = "data/data_set/control/sampled"
# output_emb_folder "data/output/embeddings/simple/sampled"
# csv_pattern="*simple_train*.csv"
# emb_pattern="*train*.pkl"

# 2. PROBE on OTHER

dataset_folder =  "data/data_set"
embeddings_folder = "data/output/embeddings/other"
output_folder = "data/data_set/control/sampled"
output_emb_folder = "data/output/embeddings/other/sampled"
csv_pattern="*other_train*.csv"
emb_pattern="*train*.pkl"



# # 2. PROBE on PSEUDO

# dataset_folder =  "data/data_set"
# embeddings_folder = "data/output/embeddings/pseudo"
# output_folder = "data/data_set/control/sampled"
# output_emb_folder = "data/output/embeddings/pseudo/sampled"
# csv_pattern="*pseudo_train*.csv"
# emb_pattern="*train*.pkl"

# decremental_training.main(dataset_folder,
#         embeddings_folder, output_folder, output_emb_folder, train_sizes, csv_pattern, emb_pattern)
