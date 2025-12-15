import random
import pandas as pd
import sample_dataset
import csv
import time
import pathlib

def write_csv(file_path, rows, delimiter=";"):

    rows.to_csv(file_path, sep=delimiter, index=False)
	# with open(file_path, "w", encoding="utf-8", newline="") as f:
	# 	writer = csv.DictWriter(f, delimiter=delimiter,
    #                     fieldnames=["ID", "NPN", "construction", 'preposition',
	# 								'noun', 'pre_lemma', 'context_pre', "costr",
	# 								'context_post','number_of_noun',
	# 								'Type', "other_cxn",
	# 								'meaning',"syntactic_function"],
	# 							restval= '_', extrasaction='ignore')
	# 	writer.writeheader()
	# 	for el in rows.itertuples():
	# 		writer.writerows(dict(el))

def main(df_minima_path, df_dataset_path,
         n_samples, random_seed,
         bucket_str_fun,
         constraint_on_rows = ["noun", "construction"],
         constrain_on_buckets = ["split"],
         sep = ";",
         output_folder = "data/sampling/"):

	timestr = time.strftime("%Y%m%d-%H%M%S")
	df_minima_basename = pathlib.Path(df_minima_path).stem

	df_minima = pd.read_csv(df_minima_path, sep=sep)
	df_dataset = pd.read_csv(df_dataset_path, sep=sep)

	# Run the CP model multiple times, get assignment_0 .. assignment_4
	df_buckets = sample_dataset.assign_buckets_multiple(
		df=df_dataset,
		df_minima=df_minima,
		n_samples=n_samples,
		constraint_on_rows=constraint_on_rows,
		constraint_on_buckets=constrain_on_buckets,
		base_seed=random_seed,
	)

	df_buckets.to_csv(f"{output_folder}/{timestr}_{df_minima_basename}.csv",
                	sep=sep, index=False)

	# For reproducible sampling per run
	rng = random.Random(random_seed)
	ret = {}
	for k in range(n_samples):
		ret[k] = {}
		assignment_col = f"assignment_{k}"

		print(f"\n=== Building train/test for {assignment_col} ===")

		sampled_rows_all = []

		# For each bucket definition in df_minima, sample rows assigned to it
		for row in df_minima.itertuples():
			# Build the bucket label exactly as sample_dataset did
			bucket = bucket_str_fun(row)
			to_sample = row.min_required
			if to_sample == 0:
				to_sample = 1

			print("Looking at bucket", bucket)
			print("sampling", to_sample, "items")

			# Rows assigned to this bucket in this assignment run
			rows_from_dataset = df_buckets[df_buckets[assignment_col] == bucket]
			n_available = rows_from_dataset.shape[0]
			print("Rows in df_buckets:", n_available)

			if n_available == 0:
				continue

			# Avoid sampling more than available
			n = min(to_sample, n_available)

			# Use a run-dependent seed to keep sampling reproducible but different per run
			sampled = rows_from_dataset.sample(
				n=n,
				random_state=random_seed + k
			)
			print(sampled.shape)
			sampled_rows_all.append(sampled)

		if not sampled_rows_all:
			print(f"No rows sampled for {assignment_col}, skipping.")
			continue

		# Concatenate all sampled rows for this assignment
		sampled_df = pd.concat(sampled_rows_all, ignore_index=True)

		# Derive split ("train"/"test") from assignment bucket string
		# bucket format: split|construction|preposition|Type
		sampled_df["split"] = sampled_df[assignment_col].str.split("|").str[0]

		train_df = sampled_df[sampled_df["split"] == "train"].copy()
		test_df  = sampled_df[sampled_df["split"] == "test"].copy()

		ret[k]["train"] = train_df
		ret[k]["test"] = test_df

		print(f"{assignment_col}: {len(train_df)} train rows, {len(test_df)} test rows")

		# # Save to CSV
		# train_out_path = f"{output_folder}/train_run_{k}.csv"
		# test_out_path  = f"test_run_{k}.csv"

		# train_df.to_csv(train_out_path, index=False)
		# test_df.to_csv(test_out_path, index=False)

		# print(f"Saved {train_out_path} and {test_out_path}")

	return ret

if __name__ == "__main__":

	_SEED = 3569
	_N_SAMPLES = 5

	df_minima_path = "/Users/ludovica/Documents/projects/NPN/data/ex1_other.csv"
	df_dataset_path = "/Users/ludovica/Documents/projects/NPN/data/source/full_dataset.csv"
	bucket_str_fun = lambda x: f"{x.split}|{x.construction}|{x.preposition}|{x.Type}|{x.other_cxn}"

	output_folder = "/Users/ludovica/Documents/projects/NPN/data/data_set/ex_1"
	simple ="other"
	ex = "1"

	dataset_dict = main(df_minima_path, df_dataset_path, _N_SAMPLES, _SEED, bucket_str_fun)

	for it in dataset_dict:
		write_csv(f"{output_folder}/{simple}/full/ex{ex}_{simple}_train_{it}.csv", dataset_dict[it]["train"])
		write_csv(f"{output_folder}/{simple}/full/ex{ex}_{simple}_test_{it}.csv", dataset_dict[it]["test"])
