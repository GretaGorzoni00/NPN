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


def main(seed, X_train_files, y_train_files, X_test_files, y_test_files, output_path, key, model, split, experiment):

	#random.seed(seed)
	all_layers = []
 
	for n in range(1, 13):
		acc_list, prec_list, rec_list, f1_list = [], [], [], []
  
  		# ciclo su tutte le 5 coppie di file
		for i, (X_train_path, y_train_path, X_test_path, y_test_path) in enumerate(
			zip(X_train_files, y_train_files, X_test_files, y_test_files)
		):
			print(f"  → Split {i+1}")

			df_train = pd.read_csv(y_train_path, sep=";")
			df_test = pd.read_csv(y_test_path, sep=";")

			y_train = np.array([1 if label == "yes" else 0 for label in df_train["construction"]])
			y_test = np.array([1 if label == "yes" else 0 for label in df_test["construction"]])

			emb_train = pd.read_pickle(X_train_path)
			emb_test = pd.read_pickle(X_test_path)
	
			X_train = np.array([np.array(emb[f"{key}_layer_{n}"]) for emb in emb_train])
			X_test = np.array([np.array(emb[f"{key}_layer_{n}"]) for emb in emb_test])

	
	# inizializzazione strutture per i risultati
	#metrics = ["accuracy", "precision", "recall", "f1"]
	# num_seeds = 5

	#pred_df = pd.DataFrame()
	#pred_df["id"] = df_test.index


	 
		# print([emb.keys() for emb in emb_test])


		
		#layer_results = {"layer": n}
		#acc_list, prec_list, rec_list, f1_list = [], [], [], []
  
		#all_seed_preds = []
		#for i in range (num_seeds):
		#current_state = math.floor(random.random()*100)
			clf = LogisticRegression(random_state=seed, max_iter=10000, solver='liblinear')
			
			clf.fit(X_train, y_train)
		
			preds = clf.predict(X_test)
		#all_seed_preds.append(preds)
		#preds_prob = clf.predict_proba(X_test)
		#print(preds_prob[:10])

			report = classification_report(y_test, preds, digits=4, output_dict=True)
   
			acc = report["accuracy"]
			prec = report["weighted avg"]["precision"]
			rec = report["weighted avg"]["recall"]
			f1 = report["weighted avg"]["f1-score"]
	
			acc_list.append(acc)
			prec_list.append(prec)
			rec_list.append(rec)
			f1_list.append(f1)
		

			print(acc_list)


		layer_results = {
			"layer": n,
			"mean_accuracy": np.mean(acc_list),
			"std_accuracy": np.std(acc_list),
			"mean_precision": np.mean(prec_list),
			"std_precision": np.std(prec_list),
			"mean_recall": np.mean(rec_list),
			"std_recall": np.std(rec_list),
			"mean_f1": np.mean(f1_list),
			"std_f1": np.std(f1_list)
		}
   
		print(f"  Layer {n}: ACC={layer_results['mean_accuracy']:.4f} ± {layer_results['std_accuracy']:.4f}, F1={layer_results['mean_f1']:.4f} ± {layer_results['std_f1']:.4f}")
		all_layers.append(layer_results)
		
	# salva risultati in CSV
	df = pd.DataFrame(all_layers)
	csv_name = f"{model}_{experiment}_{key}_{split}_avg_metrics.csv"
	csv_path = os.path.join(output_path, csv_name)
	df.to_csv(csv_path, index=False)
	print(f"\nRisultati medi salvati in: {csv_path}")

	# deselezionare blocco per plottare solo accuracy
	# plt.figure(figsize=(10, 6))
	# plt.errorbar(
	# 	df["layer"],
	# 	df["mean_accuracy"],
	# 	yerr=df["std_accuracy"],
	# 	fmt='-o',
	# 	capsize=5,
	# 	label="Mean Accuracy ± SD"
	# )

	# plt.xlabel("Layer")
	# plt.ylabel("Accuracy")
	# plt.title(f"Mean Accuracy {model}_{key}_{experiment}_{split}")
	# plt.grid(True)
	# plt.ylim(0.4, 1.0)

	# for i, acc in enumerate(df["mean_accuracy"]):
	# 	plt.text(df["layer"][i], acc + 0.005, f"{acc:.4f}", ha='center', va='bottom', fontsize=9)

	# plt.legend()
	# plt.tight_layout()

	# img_name = f"{model}_{experiment}_{key}_{split}_mean_accuracy.png"
	# img_path = os.path.join(output_path, img_name)
	# plt.savefig(img_path, dpi=300)
	# print(f"Grafico salvato in: {img_path}\n")
 
	#blocco seguente plotta accuracy, precision, recall, f1
	plt.figure(figsize=(10, 6))

	# Plot delle 4 metriche con le rispettive barre di errore
	plt.errorbar(
		df["layer"], df["mean_accuracy"], yerr=df["std_accuracy"],
		fmt='-o', capsize=4, label="Accuracy"
	)
	plt.errorbar(
		df["layer"], df["mean_precision"], yerr=df["std_precision"],
		fmt='-s', capsize=4, label="Precision"
	)
	plt.errorbar(
		df["layer"], df["mean_recall"], yerr=df["std_recall"],
		fmt='-^', capsize=4, label="Recall"
	)
	plt.errorbar(
		df["layer"], df["mean_f1"], yerr=df["std_f1"],
		fmt='-d', capsize=4, label="F1-score"
	)

	plt.xlabel("Layer")
	plt.ylabel("Score")
	plt.title(f"Performance metrics per layer – {model}_{key}_{experiment}_{split}")
	plt.grid(True)
	plt.ylim(0.4, 1.0)
	plt.legend()
	plt.tight_layout()


	# Salva grafico
	img_name = f"{model}_{experiment}_{key}_{split}_metrics.png"
	img_path = os.path.join(output_path, img_name)
	plt.savefig(img_path, dpi=300)
	print(f"Grafico salvato in: {img_path}\n")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--seed", default=42, type=int)
	parser.add_argument("--X_train", nargs='+', default=["data/output/embeddings/pseudo/BERT_embedding_PREP_ex1_pseudo_train_0.pkl", "data/output/embeddings/pseudo/BERT_embedding_PREP_ex1_pseudo_train_1.pkl", "data/output/embeddings/pseudo/BERT_embedding_PREP_ex1_pseudo_train_2.pkl", "data/output/embeddings/pseudo/BERT_embedding_PREP_ex1_pseudo_train_3.pkl", "data/output/embeddings/pseudo/BERT_embedding_PREP_ex1_pseudo_train_4.pkl"])
	parser.add_argument("--y_train", nargs='+', default = ["data/data_set/ex1_pseudo_train_0.csv", "data/data_set/ex1_pseudo_train_1.csv", "data/data_set/ex1_pseudo_train_2.csv", "data/data_set/ex1_pseudo_train_3.csv", "data/data_set/ex1_pseudo_train_4.csv"])
	parser.add_argument("--X_test", nargs='+', default=["data/output/embeddings/pseudo/BERT_embedding_PREP_ex1_pseudo_test_0.pkl", "data/output/embeddings/pseudo/BERT_embedding_PREP_ex1_pseudo_test_1.pkl", "data/output/embeddings/pseudo/BERT_embedding_PREP_ex1_pseudo_test_2.pkl", "data/output/embeddings/pseudo/BERT_embedding_PREP_ex1_pseudo_test_3.pkl", "data/output/embeddings/pseudo/BERT_embedding_PREP_ex1_pseudo_test_4.pkl"])
	parser.add_argument("--y_test", nargs='+', default = ["data/data_set/ex1_pseudo_test_0.csv", "data/data_set/ex1_pseudo_test_1.csv", "data/data_set/ex1_pseudo_test_2.csv", "data/data_set/ex1_pseudo_test_3.csv", "data/data_set/ex1_pseudo_test_4.csv"])
	parser.add_argument("-o", "--output_path", default="data/output/predictions")
	parser.add_argument("-k", "--key", default="PREP")
	parser.add_argument("-m", "--model", default="BERT")
	parser.add_argument("-s", "--split", default="pseudo")
	parser. add_argument("-e", "--experiment", default="ex1")
	args = parser.parse_args()


	main(
		args.seed,
		args.X_train,
		args.y_train,
		args.X_test,
		args.y_test,
		args.output_path,
		args.key,
		args.model,
		args.split,
		args.experiment
	)