1. campionamento
```
python3 src/campionamento.py 
		--output_folder data/data_set/scivetti/ 
		--input_files data/data_set/scivetti/cxns_normalized.csv data/data_set/scivetti/distr_normalized.csv

python3 src/campionamento.py 
		--output_folder data/source/ 
		--input_files data/dataset_construction/SU_distractor.csv data/dataset_construction/A_distractor.csv data/dataset_construction/SU_construction.csv data/dataset_construction/A_construction.csv
```

2. merge degli output prodotti da campionamento
	TODO: forse campionamento.py potrebbe già produrre un unico output?
	DONE: campionamento.py dovrebbe ristampare l'header

3. sampling
```
python3 src/sampling.py 
 		--df_minima_path data/ex1_scivetti.csv
  		--df_dataset_path data/data_set/scivetti/cxns_normalized_max30.csv
  		--output_folder data/data_set/
 		--experiment_type ex1_scivetti
  		--configuration simple

python3 src/sampling.py 
 		--df_minima_path data/ex1_simple.csv
  		--df_dataset_path data/source/full_dataset.csv
  		--output_folder data/data_set/
 		--experiment_type ex1
  		--configuration simple

python3 src/sampling.py 
 		--df_minima_path data/ex1_other.csv
  		--df_dataset_path data/source/full_dataset.csv
  		--output_folder data/data_set/
 		--experiment_type ex1
  		--configuration other

python3 src/sampling.py 
 		--df_minima_path data/ex1_pseudo.csv
  		--df_dataset_path data/source/full_dataset.csv
  		--output_folder data/data_set/
 		--experiment_type ex1
  		--configuration pseudo
```

4. estrazione embeddings -- fasttext

```
python3 src/fasttext_embeddings.py --fasttext_model cc.it.300.bin
								--output_folder data/embeddings/fasttext/ex1/simple/noun/
								--input_files data/data_set/ex1/simple/*.csv
								--FIELD noun

python3 src/fasttext_embeddings.py --fasttext_model cc.it.300.bin
								--output_folder data/embeddings/fasttext/ex1/simple/prelemma/
								--input_files data/data_set/ex1/simple/*.csv
								--FIELD pre_lemma

python3 src/fasttext_embeddings.py --fasttext_model cc.it.300.bin
								--output_folder data/embeddings/fasttext/ex1/simple/costrN/
								--input_files data/data_set/ex1/simple/*.csv
								--FIELD costrN

python3 src/fasttext_embeddings.py --fasttext_model cc.en.300.bin
								--output_folder data/embeddings/fasttext/ex1_scivetti/simple/
								--input_files data/data_set/ex1_scivetti/simple/*.csv
								--FIELD noun
```

```
python3 src/fasttext_embeddings.py --fasttext_model cc.it.300.bin
								--output_folder data/embeddings/fasttext/ex1/other/noun/
								--input_files data/data_set/ex1/other/*.csv
								--FIELD noun

python3 src/fasttext_embeddings.py --fasttext_model cc.it.300.bin
								--output_folder data/embeddings/fasttext/ex1/other/prelemma/
								--input_files data/data_set/ex1/other/*.csv
								--FIELD pre_lemma

python3 src/fasttext_embeddings.py --fasttext_model cc.it.300.bin
								--output_folder data/embeddings/fasttext/ex1/other/costrN/
								--input_files data/data_set/ex1/other/*.csv
								--FIELD costrN
```

```
python3 src/fasttext_embeddings.py --fasttext_model cc.it.300.bin
								--output_folder data/embeddings/fasttext/ex1/pseudo/noun/
								--input_files data/data_set/ex1/pseudo/*.csv
								--FIELD noun

python3 src/fasttext_embeddings.py --fasttext_model cc.it.300.bin
								--output_folder data/embeddings/fasttext/ex1/pseudo/prelemma/
								--input_files data/data_set/ex1/pseudo/*.csv
								--FIELD pre_lemma

python3 src/fasttext_embeddings.py --fasttext_model cc.it.300.bin
								--output_folder data/embeddings/fasttext/ex1/pseudo/costrN/
								--input_files data/data_set/ex1/pseudo/*.csv
								--FIELD costrN
```

5. estrazione embeddings -- word2vec

```
python3 src/w2v_embeddings.py --w2v_model vettori_itwac.txt
								--output_folder data/embeddings/itwac/ex1/simple/noun/
								--input_files data/data_set/ex1/simple/*.csv
								--FIELD noun

python3 src/w2v_embeddings.py --w2v_model vettori_itwac.txt
								--output_folder data/embeddings/itwac/ex1/simple/prelemma/
								--input_files data/data_set/ex1/simple/*.csv
								--FIELD pre_lemma

python3 src/w2v_embeddings.py --w2v_model vettori_itwac.txt
								--output_folder data/embeddings/itwac/ex1/simple/costrN/
								--input_files data/data_set/ex1/simple/*.csv
								--FIELD costrN

```

```
python3 src/w2v_embeddings.py --w2v_model vettori_itwac.txt
								--output_folder data/embeddings/itwac/ex1/other/noun/
								--input_files data/data_set/ex1/other/*.csv
								--FIELD noun

python3 src/w2v_embeddings.py --w2v_model vettori_itwac.txt
								--output_folder data/embeddings/itwac/ex1/other/prelemma/
								--input_files data/data_set/ex1/other/*.csv
								--FIELD pre_lemma

python3 src/w2v_embeddings.py --w2v_model vettori_itwac.txt
								--output_folder data/embeddings/itwac/ex1/other/costrN/
								--input_files data/data_set/ex1/other/*.csv
								--FIELD costrN

```

```
python3 src/w2v_embeddings.py --w2v_model vettori_itwac.txt
								--output_folder data/embeddings/itwac/ex1/pseudo/noun/
								--input_files data/data_set/ex1/pseudo/*.csv
								--FIELD noun

python3 src/w2v_embeddings.py --w2v_model vettori_itwac.txt
								--output_folder data/embeddings/itwac/ex1/pseudo/prelemma/
								--input_files data/data_set/ex1/pseudo/*.csv
								--FIELD pre_lemma

python3 src/w2v_embeddings.py --w2v_model vettori_itwac.txt
								--output_folder data/embeddings/itwac/ex1/pseudo/costrN/
								--input_files data/data_set/ex1/pseudo/*.csv
								--FIELD costrN

```
6. estrazione embeddings -- glove

python3 src/w2v_embeddings.py --w2v_mode glove.6B.300d.txt
								--output_folder data/embeddings/glove/ex1_scivetti/simple/
								--input_files data/data_set/ex1_scivetti/simple/*.csv
								--FIELD noun


7. estrazione embeddings encoder - bert base cased
> Scivetti e Schneider

8. estrazione embeddings encoder - bert base cased ita
> NOI

9. estrazione embeddings encoder - mBERT e xmlr
> NOI e loro

10. embeddings encoder - UmBERTo
> NOI

11. classificatore
	TODO: parametrizzare anche SVM

12. matrici di confusione

13. grafici
