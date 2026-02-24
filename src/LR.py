import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import sys
import random
import math
import argparse
import os
import copy
import matplotlib.pyplot as plt
from collections import Counter

MEANINGS = [
	"juxtaposition/contact",
	"succession/iteration/distributivity",
	"greater_plurality/accumulation",
]


def plot_metrics(df_metrics,
                model, experiment, key, decremental,
				perturbed, output_path, split, sampled, clf_name):


	#blocco seguente plotta accuracy, precision, recall, f1
	plt.figure(figsize=(10, 6))

	# Plot delle 4 metriche con le rispettive barre di errore
	if "mean_accuracy" in df_metrics:
		plt.errorbar(
			df_metrics["layer"], df_metrics["mean_accuracy"], yerr=df_metrics["std_accuracy"],
			fmt='-o', capsize=4, label="Accuracy"
		)
	plt.errorbar(
		df_metrics["layer"], df_metrics["mean_precision"], yerr=df_metrics["std_precision"],
		fmt='-s', capsize=4, label="Precision"
	)
	plt.errorbar(
		df_metrics["layer"], df_metrics["mean_recall"], yerr=df_metrics["std_recall"],
		fmt='-^', capsize=4, label="Recall"
	)
	plt.errorbar(
		df_metrics["layer"], df_metrics["mean_f1"], yerr=df_metrics["std_f1"],
		fmt='-d', capsize=4, label="F1-score"
	)

	plt.xlabel("Layer")
	plt.ylabel("Score")
	plt.title(f"Performance metrics per layer – {model}_{key}_{experiment}_{split}")
	plt.grid(True)
	plt.ylim(0, 1.0)
	plt.legend()
	plt.tight_layout()

	# Salva grafico
	img_name = f"{model}_{experiment}_{key}_{clf_name}_{decremental}_pmetrics.png"
	output_path_graph = output_path + f"graphs/{split}/{sampled}"
	img_path = os.path.join(output_path_graph, img_name)
	plt.savefig(img_path, dpi=300)
	print(f"Grafico salvato in: {img_path}\n")


def save_metrics_to_csv(metrics_all_layers, metrics_by_label,
						model, experiment, key, decremental,
						perturbed, output_path, split, sampled, clf_name,
	   					label):

	if len(metrics_all_layers) == 1:
		print("Warning: Only one layer's metrics collected.")
		for i in range(2, 13):
			new_result = copy.deepcopy(metrics_all_layers[0])
			new_result["layer"] = i  # increment layer number for clarity
			metrics_all_layers.append(new_result)
			if label == "meaning":
				new_result_lbl = copy.deepcopy(metrics_by_label[0])
				new_result_lbl["layer"] = i
				metrics_by_label.append(new_result_lbl)

	df = pd.DataFrame(metrics_all_layers)
	csv_name = f"{model}_{experiment}_{key}_{clf_name}_{decremental}_pavg_metrics.csv"
	output_path_metrics = output_path + f"metrics/{split}/{sampled}/"
	csv_path = os.path.join(output_path_metrics, csv_name)
	df.to_csv(csv_path, index=False)
	print(f"\nRisultati medi salvati in: {csv_path}")


	if label == "meaning":
		df_lbl = pd.DataFrame(metrics_by_label)
		csv_name_lbl = f"{model}_{experiment}_{key}_{clf_name}_{decremental}_pavg_metrics_by_label.csv"
		csv_path_lbl = os.path.join(output_path_metrics, csv_name_lbl)
		df_lbl.to_csv(csv_path_lbl, index=False)
		print(f"Per-label results saved in: {csv_path_lbl}")
  
		# plot_metrics(df_lbl, model, experiment, key, decremental,
		# 			perturbed, output_path, split, sampled, clf_name)

	plot_metrics(df, model, experiment, key, decremental,
				perturbed, output_path, split, sampled, clf_name)


def initialize_metrics(label, layer_range):
	metrics = {}
	for n in layer_range:
		metrics[n] = {
			"accuracy": [],
			"precision": [],
			"recall": [],
			"f1": []
		}

		if label == "meaning":
			metrics[n]["per_label"] = {
				m: {"precision": [],
					"recall": [],
					"f1-score": [],
					"support": []
				} for m in MEANINGS
			}
	return metrics

def compute_overall_metrics(metrics, label):

	metrics_all_layers = []
	metrics_by_label = []

	for n in metrics:
		if len(metrics[n]["accuracy"]) == 0:
			print(f"Warning: No metrics collected for layer {n}. Skipping.")
			continue
		layer_results = {
			"layer": n,
			"mean_accuracy": np.mean(metrics[n]["accuracy"]),
			"std_accuracy": np.std(metrics[n]["accuracy"]),
			"mean_precision": np.mean(metrics[n]["precision"]),
			"std_precision": np.std(metrics[n]["precision"]),
			"mean_recall": np.mean(metrics[n]["recall"]),
			"std_recall": np.std(metrics[n]["recall"]),
			"mean_f1": np.mean(metrics[n]["f1"]),
			"std_f1": np.std(metrics[n]["f1"])
		}

		print(f"  Layer {n}: ")
		print(f"    ACC={layer_results['mean_accuracy']:.4f} ± {layer_results['std_accuracy']:.4f}, ")
		print(f"    F1={layer_results['mean_f1']:.4f} ± {layer_results['std_f1']:.4f}")

		if label == "meaning":
			for m in MEANINGS:
				row = {
					"layer": n,
					"label": m,
					"mean_precision": float(np.mean(metrics[n]["per_label"][m]["precision"])),
					"std_precision": float(np.std(metrics[n]["per_label"][m]["precision"])),
					"mean_recall": float(np.mean(metrics[n]["per_label"][m]["recall"])),
					"std_recall": float(np.std(metrics[n]["per_label"][m]["recall"])),
					"mean_f1": float(np.mean(metrics[n]["per_label"][m]["f1-score"])),
					"std_f1": float(np.std(metrics[n]["per_label"][m]["f1-score"])),
					# support is deterministic given the dataset, but we keep it for completeness
					"mean_support": float(np.mean(metrics[n]["per_label"][m]["support"])),
					"std_support": float(np.std(metrics[n]["per_label"][m]["support"])),
				}
				metrics_by_label.append(row)

		metrics_all_layers.append(layer_results)

	return metrics_all_layers, metrics_by_label

def update_metrics(metrics, n, y_test, preds, label):
	if label == "meaning":
		report = classification_report(y_test, preds, digits=4,
								output_dict=True, labels=MEANINGS, target_names=MEANINGS)
	else:
		report = classification_report(y_test, preds, digits=4, output_dict=True)

	acc = accuracy_score(y_test, preds)
	prec = report["weighted avg"]["precision"]
	rec = report["weighted avg"]["recall"]
	f1 = report["weighted avg"]["f1-score"]

	metrics[n]["accuracy"].append(acc)
	metrics[n]["precision"].append(prec)
	metrics[n]["recall"].append(rec)
	metrics[n]["f1"].append(f1)

	if label == "meaning":
		for m in MEANINGS:
			metrics[n]["per_label"][m]["precision"].append(report[m]["precision"])
			metrics[n]["per_label"][m]["recall"].append(report[m]["recall"])
			metrics[n]["per_label"][m]["f1-score"].append(report[m]["f1-score"])
			metrics[n]["per_label"][m]["support"].append(report[m]["support"])


def create_data(X_train_path, y_train_path,
				X_test_path, y_test_path,
				key, layer_range, is_contextual, label_type):

	df_train = pd.read_csv(y_train_path, sep=";")
	df_test = pd.read_csv(y_test_path, sep=";")
	if label_type == "construction":
		y_train = np.array([1 if label == "yes" else 0 for label in df_train[label_type]])
		y_test = np.array([1 if label == "yes" else 0 for label in df_test[label_type]])
	else:
		y_train = df_train[label_type].values
		y_test = df_test[label_type].values

	if X_test_path.endswith(".pkl"):
		emb_train = pd.read_pickle(X_train_path)
		emb_test = pd.read_pickle(X_test_path)
	else:
		emb_train = []
		emb_test = []
		for line in open(X_train_path, "r"):
			line = line.strip().split()
			line[1:] = (float(x) for x in line[1:])
			emb_train.append(line)
		for line in open(X_test_path, "r"):
			line = line.strip().split()
			line[1:] = (float(x) for x in line[1:])
			emb_test.append(line)

		
		# emb_test = pd.DataFrame(emb_test)

	if is_contextual:
		X_train = []
		X_test = []
		for emb in emb_train:
			layers_emb = []
			for n in range(1, layer_range+1):
				layer_emb = np.array(emb[f"{key}_layer_{n}"])
				layers_emb.append(layer_emb)
			X_train.append(layers_emb)
		for emb in emb_test:
			layers_emb = []
			for n in range(1, layer_range+1):
				layer_emb = np.array(emb[f"{key}_layer_{n}"])
				layers_emb.append(layer_emb)
			X_test.append(layers_emb)

		X_train = np.array(X_train)
		X_test = np.array(X_test)
	else:
		X_train = []
		X_test = []

		for emb in emb_train:
			X_train.append(np.array([emb[1:]]))  # assuming the first column is an ID or similar
		for emb in emb_test:
			X_test.append(np.array([emb[1:]]))  # assuming the first column is an ID or similar

		X_train = np.array([x for x in X_train])
		X_test = np.array([x for x in X_test])

	targets = df_train[label_type].values
	if label_type == "construction":
		targets = np.array([1 if label == "yes" else 0 for label in df_train[label_type]])

	return X_train, X_test, y_train, y_test, targets

def main(seed, X_train_files, y_train_files, X_test_files, y_test_files,
		output_path, key, model, split, experiment, decremental, perturbed,
		is_contextual, clf_name, solver = "liblinear", label = "construction",
		n_layers = 12):

	sampled = "full"
	if decremental:
		sampled = "sampled"

	# Numero di layer da esplorare solo se il modello è BERT-like
	layer_range = range(1, n_layers + 1)

	splits_data = []
	metrics = initialize_metrics(label, layer_range)
 

	for i, (X_train_path, y_train_path, X_test_path, y_test_path) in \
	  	enumerate(
			zip(X_train_files, y_train_files, X_test_files, y_test_files)
		):
			print(f"  → Split {i+1}")
			X_train, X_test, \
			y_train, y_test, \
   			targets = create_data(X_train_path, y_train_path,
									X_test_path, y_test_path,
		  							key, n_layers, is_contextual, label)
			splits_data.append((X_train, y_train, X_test, y_test, targets))

	for i, (X_train, y_train, X_test, y_test, _) in enumerate(splits_data):
		split_idx = i + 1
		layer_pred_dict = {"gold": y_test.tolist()}
		for n in range(X_train.shape[1]):
			if clf_name == "logistic_regression":
				clf = LogisticRegression(random_state=seed, max_iter=10000, solver=solver)
			elif clf_name == "SVM":
				clf = LinearSVC(random_state=seed, max_iter=10000)
   
			clf.fit(X_train[:, n, :], y_train)
			preds = clf.predict(X_test[:, n, :])

			layer_pred_dict[f"layer_{n+1}"] = preds.tolist()

			update_metrics(metrics, n+1, y_test, preds, label)

		df_preds = pd.DataFrame(layer_pred_dict)
		output_path_pred = output_path + f"predictions/{split}/{sampled}/"
		csv_path = os.path.join(output_path_pred, f"{model}_{experiment}_{key}_split{split_idx}_{clf_name}_{decremental}_predictions.csv")
		df_preds.to_csv(csv_path, index=False)
		print(f"    Predizioni salvate in: {csv_path}")

	metrics_all_layers, metrics_by_label = compute_overall_metrics(metrics, label)

	# salva risultati in CSV
	save_metrics_to_csv(metrics_all_layers, metrics_by_label,
							model, experiment, key, decremental,
							perturbed, output_path, split, sampled,
	   						clf_name, label)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--seed", default=42, type=int)
	parser.add_argument("--X_train", nargs='+', required=True)
	parser.add_argument("--y_train", nargs='+', required=True)
	parser.add_argument("--X_test", nargs='+', required=True)
	parser.add_argument("--y_test", nargs='+', required=True)
	parser.add_argument("-o", "--output_path", required=True)
	parser.add_argument("-k", "--key", default="UNK")
	parser.add_argument("-m", "--model", default="BERT")
	parser.add_argument("-s", "--split", default="simple")
	parser.add_argument("-e", "--experiment", default="ex1")
	parser.add_argument("-d", "--decremental", action="store_true")
	parser.add_argument("-p", "--perturbed", default="")
	parser.add_argument("--contextual", action="store_true",
                    help="Whether the embeddings are contextual (e.g., BERT) or static (e.g., FastText)")
	parser.add_argument("--clf_name", default="logistic_regression",
                    choices=["logistic_regression", "SVM"],
                    help="Classifier to use (default: logistic_regression)")
	parser.add_argument("--solver", default="liblinear",
                    choices=["liblinear", "lbfgs"], help="Solver for Logistic Regression")
	parser.add_argument("--label", default="construction",
                    choices=["construction", "meaning"],
                    help="Label to predict: 'construction' or 'meaning'")
	args = parser.parse_args()


	if not os.path.exists(args.output_path):
		os.makedirs(args.output_path, exist_ok=True)

	if args.decremental:
		subpath = f"{args.split}/sampled/"
	else:
		subpath = f"{args.split}/full/"

	for part in ["metrics/", "predictions/", "graphs/"]:
		if not os.path.exists(args.output_path+f"{part}{subpath}"):
			os.makedirs(args.output_path+f"{part}{subpath}", exist_ok=True)

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
		args.experiment,
		args.decremental,
		args.perturbed,
		args.contextual,
		args.clf_name,
		solver=args.solver,
		label=args.label
	)