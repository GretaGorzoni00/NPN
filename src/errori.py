import pandas as pd
import glob

input_files = sorted(glob.glob("data/data_set/ex_2/simple/full/ex1_simple_test_1.csv"))
pred_files  = sorted(glob.glob("data/output/predictions/semantic/full/BERT_ex1_UNK_split1___predictions.csv"))

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
	
	identity = df_in["pred"] == df_in["meaning"]
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

print(df_final["meaning"].value_counts())
print(df_final["meaning"].value_counts())


import pandas as pd

# df_all contiene tutte le istanze (gold + pred), df_final contiene solo gli errori
# aggiungiamo una colonna gold + preposition per suddividere alcune classi
df_final["gold_subcat"] = df_final.apply(
    lambda row: f"{row['meaning']}_a" if (row['meaning'] == "succession/iteration/distributivity" and row['preposition'] == "a")
    else (f"{row['meaning']}_su" if (row['meaning'] == "succession/iteration/distributivity" and row['preposition'] == "su")
    else row['meaning']),
    axis=1
)

# calcola numero totale di istanze per ciascuna sottoclasse (gold_subcat)
# prendiamo df_all per includere tutte le istanze, non solo gli errori
df_all["gold_subcat"] = df_all.apply(
    lambda row: f"{row['meaning']}_a" if (row['meaning'] == "succession/iteration/distributivity" and row['preposition'] == "a")
    else (f"{row['meaning']}_su" if (row['meaning'] == "succession/iteration/distributivity" and row['preposition'] == "su")
    else row['meaning']),
    axis=1
)
total_per_subcat = df_all["gold_subcat"].value_counts()

# conta quante di quelle istanze sono state predette come ciascuna classe (solo errori)
error_counts = pd.crosstab(df_final["gold_subcat"], df_final["pred"])

# percentuale rispetto al numero totale di istanze della sottoclasse
error_percent = (error_counts.T / total_per_subcat).T * 100
error_percent = error_percent.round(2)

# opzionale: aggiungi colonna totale errori per la sottoclasse
error_percent["tot_error_percent"] = error_percent.sum(axis=1)

print(error_percent)


import pandas as pd

# Leggi il CSV
df = pd.read_csv("data/data_set/ex_2/simple/full/ex1_simple_train_4.csv", sep=";")

# Crea una colonna "category" con le sottocategorie per succession
def categorize(row):
    if row["meaning"] == "succession/iteration/distributivity":
        if row["preposition"] == "a":
            return "succ_a"
        elif row["preposition"] == "su":
            return "succ_su"
        else:
            return "succ_other"
    else:
        # le altre categorie rimangono uguali
        return row["meaning"]

df["category"] = df.apply(categorize, axis=1)


counts = df["category"].value_counts()
print(counts)

