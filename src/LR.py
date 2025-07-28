import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import sys

clf = LogisticRegression(random_state=0, max_iter=10000)

df_train = pd.read_csv("code/data.tsv", sep="\t")
df_test = pd.read_csv("code/data.tsv", sep="\t")

y_train = np.array([1 if label == "yes" else 0 for label in df_train["construction"]])
y_test = np.array([1 if label == "yes" else 0 for label in df_test["construction"]])

emb_train = pd.read_pickle("embedding_layers_UNK.pkl")["embeddings"].tolist()
emb_test = pd.read_pickle("embedding_layers_UNK.pkl")["embeddings"].tolist()

for n in range(1,23):
    X_train = np.array([emb[n - 1] for emb in emb_train])
    X_test = np.array([emb[n - 1] for emb in emb_test])
    
    print("Layer"+ str(n))

    clf.fit(X_train, y_train)
 
    preds = clf.predict(X_test)
    print(classification_report(y_test, preds, digits=4), file=sys.stderr)