import pandas as pd
import glob

input_files = sorted(glob.glob("data/data_set/ex_1/pseudo/full/ex1_pseudo_test_2.csv"))
pred_files  = sorted(glob.glob("data/output/predictions/pseudo/full/BERT_ex1_UNK_split2___predictions.csv"))

print(input_files)

print(input_files)
pred_dict = {}

df_all = pd.DataFrame()

for in_file, pred_file in zip(input_files, pred_files):

	df_in = pd.read_csv(in_file, sep = ";")
	df_pred = pd.read_csv(pred_file, sep = ",")



	layer = df_pred["layer_12"]
	print(layer)	

	df_in["pred"] = layer
	
	identity = df_in["pred"] == df_in["construction"]
	df_in["match"] = identity
	# print()
 
	# for l  in df_in:
	
	# 	df_in["match"] = df_in["pred"] == df_in["construction"]

	df_all = pd.concat([df_all, df_in], ignore_index=True)


df_final = df_all[df_all["match"] == False]

# if df_all["match"] == False:
#     df_final.append(df_all)



print(df_final)
df_final.to_csv("wrong_predictions.csv", sep=";", index=False)

print(df_final["construction"].value_counts())
print(df_final["preposition"].value_counts())
