import glob
import sampling



_SEED = 3569
_N_SAMPLES = 5

# df_minima_path = "data/ex1_other.csv"
# df_dataset_path = "data/source/full_dataset.csv"
# bucket_str_fun = lambda x: f"{x.split}|{x.construction}|{x.preposition}|{x.Type}|{x.other_cxn}"

# output_folder = "data/data_set/ex_1"
# simple ="other"
# ex = "1"

# dataset_dict = sampling.main(df_minima_path, df_dataset_path, _N_SAMPLES, _SEED, bucket_str_fun)

# for it in dataset_dict:
# 	sampling.write_csv(f"{output_folder}/{simple}/full/ex{ex}_{simple}_train_{it}.csv", dataset_dict[it]["train"])
# 	sampling.write_csv(f"{output_folder}/{simple}/full/ex{ex}_{simple}_test_{it}.csv", dataset_dict[it]["test"])

# ===========================================================================================================

# df_minima_path = "data/ex1_pseudo.csv"
# df_dataset_path = "data/source/full_dataset.csv"
# bucket_str_fun = lambda x: f"{x.split}|{x.construction}|{x.preposition}|{x.Type}|{x.other_cxn}"

# output_folder = "data/data_set/ex_1"
# simple ="pseudo"
# ex = "1"

# dataset_dict = sampling.main(df_minima_path, df_dataset_path, _N_SAMPLES, _SEED, bucket_str_fun)

# for it in dataset_dict:
# 	sampling.write_csv(f"{output_folder}/{simple}/full/ex{ex}_{simple}_train_{it}.csv", dataset_dict[it]["train"])
# 	sampling.write_csv(f"{output_folder}/{simple}/full/ex{ex}_{simple}_test_{it}.csv", dataset_dict[it]["test"])

# ===========================================================================================================

# df_minima_path = "data/ex1_simple.csv"
# df_dataset_path = "data/source/full_dataset.csv"
# bucket_str_fun = lambda x: f"{x.split}|{x.construction}|{x.preposition}|{x.Type}"

# output_folder = "data/data_set/ex_1"
# simple ="simple"
# ex = "1"

# dataset_dict = sampling.main(df_minima_path, df_dataset_path, _N_SAMPLES, _SEED, bucket_str_fun)

# for it in dataset_dict:
# 	sampling.write_csv(f"{output_folder}/{simple}/full/ex{ex}_{simple}_train_{it}.csv", dataset_dict[it]["train"])
# 	sampling.write_csv(f"{output_folder}/{simple}/full/ex{ex}_{simple}_test_{it}.csv", dataset_dict[it]["test"])

# ===========================================================================================================

df_minima_path = "data/ex2_simple.csv"
df_dataset_path = "data/source/full_dataset.csv"
bucket_str_fun = lambda x: f"{x.split}|{x.construction}|{x.preposition}|{x.meaning}"

output_folder = "data/data_set/ex_2"
simple ="simple"
ex = "2"

dataset_dict = sampling.main(df_minima_path, df_dataset_path, _N_SAMPLES, _SEED, bucket_str_fun)

for it in dataset_dict:
	sampling.write_csv(f"{output_folder}/{simple}/full/ex{ex}_{simple}_train_{it}.csv", dataset_dict[it]["train"])
	sampling.write_csv(f"{output_folder}/{simple}/full/ex{ex}_{simple}_test_{it}.csv", dataset_dict[it]["test"])