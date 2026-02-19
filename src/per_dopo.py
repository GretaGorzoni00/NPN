import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report



#y_train contiene stringhe 
#LogisticRegression in scikit-learn accetta label nominali come stringhe
#non serve LabelEncoder sono internamente modificate 

y_train_file = "data/data_set/ex_2/simple/full/ex2_simple_train_0.csv"
y_test_file = "data/data_set/ex_2/simple/per_dopo.csv"
X_train_file = "data/embeddings/bert/semantic/BERT_embedding_PREP_ex2_simple_train_0.pkl"
X_test_file = "data/embeddings/bert/semantic/per_dopo/BERT_embedding_PREP_per_dopo.pkl"
label = "meaning"
key = "PREP"



df_train = pd.read_csv(y_train_file, sep=";")

df_test = pd.read_csv(y_test_file, sep=";")

y_train = df_train[label].values

y_test  = df_test[label].values

emb_train = pd.read_pickle(X_train_file)

emb_test = pd.read_pickle(X_test_file)


acc_list = []
prec_list = []
rec_list = []
f1_list = []   

#cross entropy loss
#integer value for each integer encoded class label


layer_range = range(1, 13)
for n in layer_range:
    

    model = LogisticRegression(multi_class='multinomial', solver='lbfgs')

    X_train = np.array([np.array(emb[f"{key}_layer_{n}"]) for emb in emb_train])
    X_test = np.array([np.array(emb[f"{key}_layer_{n}"]) for emb in emb_test])
    

    model.fit(X_train, y_train) 
    preds = model.predict(X_test)



    
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
print(prec_list)
print(rec_list)
print(f1_list)      