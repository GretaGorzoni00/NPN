import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import sys
import random
import math
import argparse

def main(seed, y_train, y_test, X_train, X_test, output_path, key):

	random.seed(seed)


	df_train = pd.read_csv(y_train, sep=";")
	df_test = pd.read_csv(y_test, sep=";")

	y_train = np.array([1 if label == "yes" else 0 for label in df_train["construction"]])
	y_test = np.array([1 if label == "yes" else 0 for label in df_test["construction"]])

	emb_train = pd.read_pickle(X_train)
	emb_test = pd.read_pickle(X_test)
 
	for n in range(1,13):
     
		# print([emb.keys() for emb in emb_test])
		X_train = np.array([np.array(emb[f"{key}_layer_{n}"]) for emb in emb_train])
		X_test = np.array([np.array(emb[f"{key}_layer_{n}"]) for emb in emb_test])
		print("Layer"+ str(n))	

		for i in range (3):
			current_state = math.floor(random.random()*100)
			clf = LogisticRegression(random_state=current_state, max_iter=10000)
			
			clf.fit(X_train, y_train)
		
			preds = clf.predict(X_test)
			with open (f"{output_path}/report_layer{n}_{current_state}.txt", "w") as file_output:
				print(classification_report(y_test, preds, digits=4), file=file_output)
			
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	
	parser.add_argument("--seed", default = 42)
	parser.add_argument("--X_train", default ="data/output/embeddings/BERT_embedding_UNK_train.pkl")
	parser.add_argument("--y_train", default = "data/output/embeddings/lemma_balanced_train_pred.csv")
	parser.add_argument("--X_test", default = "data/output/embeddings/BERT_embedding_UNK_test.pkl")
	parser.add_argument("--y_test", default = "data/output/embeddings/lemma_balanced_test_pred.csv")
	parser.add_argument("-o", "--output_path", default = "data/output/predictions")
	parser.add_argument("-k", "--key", default = "UNK")
	args = parser.parse_args()
	main(args.seed, args.y_train, args.y_test, args.X_train, args.X_test, args.output_path, args.key)