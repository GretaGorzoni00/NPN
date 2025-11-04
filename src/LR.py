import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import sys
import random
import math
import argparse
import os
import matplotlib.pyplot as plt
from collections import Counter

def main(seed, y_train_path, y_test_path, X_train_path, X_test_path, output_path, key, model, split):

	random.seed(seed)


	df_train = pd.read_csv(y_train_path, sep=";")
	df_test = pd.read_csv(y_test_path, sep=";")

	y_train = np.array([1 if label == "yes" else 0 for label in df_train["construction"]])
	y_test = np.array([1 if label == "yes" else 0 for label in df_test["construction"]])

	emb_train = pd.read_pickle(X_train_path)
	emb_test = pd.read_pickle(X_test_path)

	
	# inizializzazione strutture per i risultati
	metrics = ["accuracy", "precision", "recall", "f1"]
	num_seeds = 5
	all_layers = []
	pred_df = pd.DataFrame()
	pred_df["id"] = df_test.index

	for n in range(1, 13):
	 
		# print([emb.keys() for emb in emb_test])
		X_train = np.array([np.array(emb[f"{key}_layer_{n}"]) for emb in emb_train])
		X_test = np.array([np.array(emb[f"{key}_layer_{n}"]) for emb in emb_test])
		
		layer_results = {"layer": n}
		acc_list, prec_list, rec_list, f1_list = [], [], [], []
  
		all_seed_preds = []
		for i in range (num_seeds):
			current_state = math.floor(random.random()*100)
			clf = LogisticRegression(random_state=current_state, max_iter=10000, solver='saga')
			
			clf.fit(X_train, y_train)
		
			preds = clf.predict(X_test)
			all_seed_preds.append(preds)
			#preds_prob = clf.predict_proba(X_test)
			#print(preds_prob[:10])
   
			report_dict = classification_report(y_test, preds, digits=4, output_dict=True)
			acc = report_dict["accuracy"]
			prec = report_dict["weighted avg"]["precision"]
			rec = report_dict["weighted avg"]["recall"]
			f1 = report_dict["weighted avg"]["f1-score"]
			
			acc_list.append(acc)
			prec_list.append(prec)
			rec_list.append(rec)
			f1_list.append(f1)
			

			# salvataggio in dizionario strutturato per seed
			layer_results[f"seed_{i+1}_accuracy"] = acc
			layer_results[f"seed_{i+1}_precision"] = prec
			layer_results[f"seed_{i+1}_recall"] = rec
			layer_results[f"seed_{i+1}_f1"] = f1

		print(acc_list)
		layer_results["mean_accuracy"] = np.mean(acc_list)
		layer_results["std_accuracy"] = np.std(np.array(acc_list))
		layer_results["mean_precision"] = np.mean(prec_list)
		layer_results["std_precision"] = np.std(prec_list)
		layer_results["mean_recall"] = np.mean(rec_list)
		layer_results["std_recall"] = np.std(rec_list)
		layer_results["mean_f1"] = np.mean(f1_list)
		layer_results["std_f1"] = np.std(f1_list)


		all_layers.append(layer_results)
  
		# calcola predizione finale come valore più frequente tra i seed
		final_preds = []
		for idx in range(len(df_test)):
			votes = [p[idx] for p in all_seed_preds]  # raccoglie le predizioni di tutti i seed per questo sample
			most_common = Counter(votes).most_common(1)[0][0]  # trova il valore più frequente
			final_preds.append("yes" if most_common == 1 else "no")  # converte 1/0 in yes/no

		# aggiungi colonna con nome modello_key_layer
		col_name = f"{model}_{key}_layer_{n}"
		pred_df[col_name] = final_preds

	# crea unico DataFrame con layer nelle righe e metriche × seed nelle colonne
	df = pd.DataFrame(all_layers)

	# salvataggio CSV unico
	csv_name = f"{model}_{key}_{split}_all_metrics.csv"
	csv_path = os.path.join(output_path, csv_name)
	df.to_csv(csv_path, index=False)
	print(f"\n Risultati salvati in: {csv_path}")
 
 
	pred_file = os.path.join(output_path, f"{model}_{key}_{split}_layer_predictions.csv")
	pred_df.to_csv(pred_file, index=False)
	print(f"Predizioni finali salvate in: {pred_file}")
	
	df = pd.read_csv(f"data/output/predictions/{model}_{key}_{split}_all_metrics.csv")

	# Disegna il grafico della media della accuracy per layer
	plt.figure(figsize=(10, 6))
	plt.plot(df["layer"], df["mean_accuracy"], marker="o", linestyle="-", linewidth=2, label="Mean Accuracy")

	# Aggiungi etichette e titolo
	plt.xlabel("Layer")
	plt.ylabel("Accuracy")
	plt.title(f"Accuracy {model}_{key}")
	plt.grid(True)
	plt.ylim(0.6, 1.0)
  
	for i, acc in enumerate(df["mean_accuracy"]):
		plt.text(df["layer"][i], acc + 0.005, f"{acc:.4f}", ha='center', va='bottom', fontsize=9)
	
	plt.legend()
	plt.tight_layout()

	img_name = f"{model}_{key}_{split}_mean_accuracy.png"
	img_path = os.path.join(output_path, img_name)
	plt.savefig(img_path, dpi=300)
	print(f"Grafico salvato in: {img_path}\n")
			
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	
	parser.add_argument("--seed", default = 42)
	parser.add_argument("--X_train", default ="data/output/embeddings/type_balanced/BERT_embedding_UNK_train_type_balanced.pkl")
	parser.add_argument("--y_train", default = "data/output/embeddings/type_balanced_train_pred.csv")
	parser.add_argument("--X_test", default = "data/output/embeddings/type_balanced/BERT_embedding_UNK_test_type_balanced.pkl")
	parser.add_argument("--y_test", default = "data/output/embeddings/type_balanced_test_pred.csv")
	parser.add_argument("-o", "--output_path", default = "data/output/predictions")
	parser.add_argument("-k", "--key", default = "UNK")
	parser.add_argument("-m", "--model", default = "BERT")
	parser.add_argument("-s", "--split", default = "type_balanced")
	args = parser.parse_args()
	main(args.seed, args.y_train, args.y_test, args.X_train, args.X_test, args.output_path, args.key, args.model, args.split)