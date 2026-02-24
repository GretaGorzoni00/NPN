1. campionamento
```
python3 src/campionamento.py \
		--output_folder data/source/ \
		--input_files data/data_set/scivetti/cxns_normalized.csv data/data_set/scivetti/distr_normalized.csv \
    --nome_file full_dataset_scivetti

python3 src/campionamento.py \
		--output_folder data/source/ \
		--input_files data/dataset_construction/SU_distractor.csv data/dataset_construction/A_distractor.csv data/dataset_construction/SU_construction.csv data/dataset_construction/A_construction.csv \
    --nome_file full_dataset_itanpn
```

<!-- 2. merge degli output prodotti da campionamento
	TODO: forse campionamento.py potrebbe già produrre un unico output?
	TODO: campionamento.py dovrebbe ristampare l'header BUGGAVA ANCORA DA IMPLEMENTARE -->

2. sampling
```
python3 src/sampling.py \
 		--df_minima_path data/ex1_scivetti.csv \
  		--df_dataset_path data/source/full_dataset_scivetti.csv \
  		--output_folder data/data_set/ \
 		--experiment_type ex1_scivetti \
  		--configuration simple

python3 src/sampling.py \
 		--df_minima_path data/ex1_simple.csv \
  		--df_dataset_path data/source/full_dataset_itanpn.csv \
  		--output_folder data/data_set/ \
 		--experiment_type ex1 \
  		--configuration simple

python3 src/sampling.py \
 		--df_minima_path data/ex1_other.csv \
  		--df_dataset_path data/source/full_dataset_itanpn.csv \
  		--output_folder data/data_set/ \
 		--experiment_type ex1 \
  		--configuration other

python3 src/sampling.py \
 		--df_minima_path data/ex1_pseudo.csv \
  		--df_dataset_path data/source/full_dataset_itanpn.csv \
  		--output_folder data/data_set/ \
 		--experiment_type ex1 \
  		--configuration pseudo

```

3. control classifier 

python3 src/control_classifier.py --files_train data/data_set/ex1/simple/full/*train*.csv \
                --files_test data/data_set/ex1/simple/full/*test*.csv \
                --control_dir data/data_set/ex1/simple/control/ \
                --_FIELD noun \
                --_LABEL_FIELD construction

python3 src/control_classifier.py --files_train data/data_set/ex1/other/full/*train*.csv \
                --files_test data/data_set/ex1/other/full/*test*.csv \
                --control_dir data/data_set/ex1/other/control/ \
                --_FIELD noun \
                --_LABEL_FIELD construction


python3 src/control_classifier.py --files_train data/data_set/ex1/pseudo/full/*train*.csv \
                --files_test data/data_set/ex1/pseudo/full/*test*.csv \
                --control_dir data/data_set/ex1/pseudo/control/ \
                --_FIELD noun \
                --_LABEL_FIELD construction





4. estrazione embeddings -- fasttext

```
python3 src/fasttext_embeddings.py --fasttext_model static_model/cc.it.300.bin \
								--output_folder data/embedding/fasttext/ex1/simple/noun/ \
								--input_files data/data_set/ex1/simple/full/*.csv \
								--FIELD noun

python3 src/fasttext_embeddings.py --fasttext_model static_model/cc.it.300.bin \
								--output_folder data/embedding/fasttext/ex1/simple/pre_lemma/ \
								--input_files data/data_set/ex1/simple/full/*.csv \
								--FIELD pre_lemma

python3 src/fasttext_embeddings.py --fasttext_model static_model/cc.it.300.bin \
								--output_folder data/embedding/fasttext/ex1/simple/costrN/ \
								--input_files data/data_set/ex1/simple/full/*.csv \
								--FIELD costrN 

python3 src/fasttext_embeddings.py --fasttext_model static_model/cc.en.300.bin \
								--output_folder data/embedding/fasttext/ex1_scivetti/simple/ \
								--input_files data/data_set/ex1_scivetti/simple/full/*.csv \
								--FIELD noun
```

```
python3 src/fasttext_embeddings.py --fasttext_model static_model/cc.it.300.bin \
								--output_folder data/embedding/fasttext/ex1/other/noun/ \
								--input_files data/data_set/ex1/other/full/*.csv \
								--FIELD noun

python3 src/fasttext_embeddings.py --fasttext_model static_model/cc.it.300.bin
								--output_folder data/embedding/fasttext/ex1/other/prelemma/
								--input_files data/data_set/ex1/other/full/*.csv
								--FIELD pre_lemma

python3 src/fasttext_embeddings.py --fasttext_model static_model/cc.it.300.bin \
								--output_folder data/embedding/fasttext/ex1/other/costrN/ \
								--input_files data/data_set/ex1/other/full/*.csv \
								--FIELD costrN
```

```
python3 src/fasttext_embeddings.py --fasttext_model static_model/cc.it.300.bin \
								--output_folder data/embedding/fasttext/ex1/pseudo/noun/ \
								--input_files data/data_set/ex1/pseudo/full/*.csv \
								--FIELD noun

python3 src/fasttext_embeddings.py --fasttext_model static_model/cc.it.300.bin
								--output_folder data/embedding/fasttext/ex1/pseudo/prelemma/
								--input_files data/data_set/ex1/pseudo/full/*.csv
								--FIELD pre_lemma

python3 src/fasttext_embeddings.py --fasttext_model static_model/cc.it.300.bin \
								--output_folder data/embedding/fasttext/ex1/pseudo/costrN/ \
								--input_files data/data_set/ex1/pseudo/full/*.csv \
								--FIELD costrN
```

5. estrazione embeddings -- word2vec

```
python3 src/w2v_embeddings.py --w2vec_model static_model/vettori_itwac.txt \
								--output_folder data/embedding/itwac/ex1/simple/noun/ \
								--input_files data/data_set/ex1/simple/full/*.csv \
								--FIELD noun

python3 src/w2v_embeddings.py --w2vec_model static_model/vettori_itwac.txt
								--output_folder data/embedding/itwac/ex1/simple/prelemma/
								--input_files data/data_set/ex1/simple/full/*.csv
								--FIELD pre_lemma

python3 src/w2v_embeddings.py --w2vec_model static_model/vettori_itwac.txt \
								--output_folder data/embedding/itwac/ex1/simple/costrN/ \
								--input_files data/data_set/ex1/simple/full/*.csv \
								--FIELD costrN

```

```
python3 src/w2v_embeddings.py --w2vec_model static_model/vettori_itwac.txt \
								--output_folder data/embedding/itwac/ex1/other/noun/ \
								--input_files data/data_set/ex1/other/full/*.csv \
								--FIELD noun

python3 src/w2v_embeddings.py --w2vec_model static_model/vettori_itwac.txt
								--output_folder data/embedding/itwac/ex1/other/prelemma/
								--input_files data/data_set/ex1/other/full/*.csv
								--FIELD pre_lemma

python3 src/w2v_embeddings.py --w2vec_model static_model/vettori_itwac.txt \
								--output_folder data/embedding/itwac/ex1/other/costrN/ \
								--input_files data/data_set/ex1/other/full/*.csv \
								--FIELD costrN

```

```
python3 src/w2v_embeddings.py --w2vec_model static_model/vettori_itwac.txt \
								--output_folder data/embedding/itwac/ex1/pseudo/noun/ \
								--input_files data/data_set/ex1/pseudo/full/*.csv \
								--FIELD noun

python3 src/w2v_embeddings.py --w2vec_model static_model/vettori_itwac.txt \
								--output_folder data/embedding/itwac/ex1/pseudo/prelemma/ \
								--input_files data/data_set/ex1/pseudo/full/*.csv \
								--FIELD pre_lemma

python3 src/w2v_embeddings.py --w2vec_model static_model/vettori_itwac.txt \
								--output_folder data/embedding/itwac/ex1/pseudo/costrN/ \
								--input_files data/data_set/ex1/pseudo/full/*.csv \
								--FIELD costrN

```
6. estrazione embeddings -- glove

python3 src/w2v_embeddings.py --w2vec_model static_model/glove.6B.300d.txt \
								--output_folder data/embedding/glove/ex1_scivetti/simple/ \
								--input_files data/data_set/ex1_scivetti/simple/full/*.csv \
								--FIELD noun


7. estrazione embeddings encoder - bert base cased

DONE: creazione cartelle non esistenti

python3 src/embedding.py \
  --model bert-base-cased \
  --prefix BERT \
  --tokenizer_path data/tokenizer \
  --output_path data/embedding/bert_en/ex1_scivetti/simple \
  --split scivetti_simple \
  --perturbed no \
  --train data/data_set/ex1_scivetti/simple/full/*train*.csv \
  --test  data/data_set/ex1_scivetti/simple/full/*test*.csv \
  --UNK "[UNK]" \
  --MASK "[MASK]" \
  --UNK_ID 100 \
  --MASK_ID 103


8. estrazione embeddings encoder - bert base cased ita

python3 src/embedding.py \
  --model dbmdz/bert-base-italian-cased \
  --prefix BERT \
  --tokenizer_path data/tokenizer \
  --output_path data/embedding/bert_ita/ex1/simple/full \
  --split simple \
  --perturbed no \
  --train data/data_set/ex1/simple/full/*train*.csv \
  --test  data/data_set/ex1/simple/full/*test*.csv \
  --UNK "[UNK]" \
  --MASK "[MASK]" \
  --UNK_ID 101 \
  --MASK_ID 104


python3 src/embedding.py \
  --model dbmdz/bert-base-italian-cased \
  --prefix BERT \
  --tokenizer_path data/tokenizer \
  --output_path data/embedding/bert_ita/ex1/other/full \
  --split other \
  --perturbed no \
  --train data/data_set/ex1/other/full/*train*.csv \
  --test  data/data_set/ex1/other/full/*test*.csv \
  --UNK "[UNK]" \
  --MASK "[MASK]" \
  --UNK_ID 101 \
  --MASK_ID 104


python3 src/embedding.py \
  --model dbmdz/bert-base-italian-cased \
  --prefix BERT \
  --tokenizer_path data/tokenizer \
  --output_path data/embedding/bert_ita/ex1/pseudo \
  --split pseudo \
  --perturbed no \
  --train data/data_set/ex1/pseudo/full/*train*.csv \
  --test  data/data_set/ex1/pseudo/full/*test*.csv \
  --UNK "[UNK]" \
  --MASK "[MASK]" \
  --UNK_ID 101 \
  --MASK_ID 104

9. estrazione embeddings encoder - mBERT e xmlr
python3 src/embedding.py \
  --model bert-base-multilingual-cased \
  --prefix mBERT \
  --tokenizer_path data/tokenizer \
  --output_path data/embedding/bert_multilingual/ex1_scivetti/simple \
  --split scivetti_simple \
  --perturbed no \
  --train data/data_set/ex1_scivetti/simple/full/*train*.csv \
  --test  data/data_set/ex1_scivetti/simple/full/*test*.csv \
  --UNK "[UNK]" \
  --MASK "[MASK]" \
  --UNK_ID 100 \
  --MASK_ID 103

  python3 src/embedding.py \
  --model bert-base-multilingual-cased \
  --prefix mBERT \
  --tokenizer_path data/tokenizer \
  --output_path data/embedding/bert_multilingual/ex1/simple/full \
  --split simple \
  --perturbed no \
  --train data/data_set/ex1/simple/full/*train*.csv \
  --test  data/data_set/ex1/simple/full/*test*.csv \
  --UNK "[UNK]" \
  --MASK "[MASK]" \
  --UNK_ID 100 \
  --MASK_ID 103


python3 src/embedding.py \
  --model bert-base-multilingual-cased \
  --prefix mBERT \
  --tokenizer_path data/tokenizer \
  --output_path data/embedding/bert_multilingual/ex1/other/full \
  --split other \
  --perturbed no \
  --train data/data_set/ex1/other/full/*train*.csv \
  --test  data/data_set/ex1/other/full/*test*.csv \
  --UNK "[UNK]" \
  --MASK "[MASK]" \
  --UNK_ID 100 \
  --MASK_ID 103


python3 src/embedding.py \
  --model bert-base-multilingual-cased \
  --prefix mBERT \
  --tokenizer_path data/tokenizer \
  --output_path data/embedding/bert_multilingual/ex1/pseudo/full \
  --split pseudo \
  --perturbed no \
  --train data/data_set/ex1/pseudo/full/*train*.csv \
  --test  data/data_set/ex1/pseudo/full/*test*.csv \
  --UNK "[UNK]" \
  --MASK "[MASK]" \
  --UNK_ID 100 \
  --MASK_ID 103




python3 src/embedding.py \
  --model FacebookAI/xlm-roberta-base \
  --prefix xlm_roberta \
  --tokenizer_path data/tokenizer \
  --output_path data/embedding/xlm_roberta/ex1_scivetti/simple \
  --split scivetti_simple \
  --perturbed no \
  --train data/data_set/ex1_scivetti/simple/full/*train*.csv \
  --test  data/data_set/ex1_scivetti/simple/full/*test*.csv \
  --UNK "<unk>" \
  --MASK "<mask>" \
  --UNK_ID 3 \
  --MASK_ID 250001


python3 src/embedding.py \
  --model FacebookAI/xlm-roberta-base \
  --prefix xlm_robert \
  --tokenizer_path data/tokenizer \
  --output_path data/embedding/xlm_roberta/ex1/simple/full \
  --split simple \
  --perturbed no \
  --train data/data_set/ex1/simple/full/*train*.csv \
  --test  data/data_set/ex1/simple/full/*test*.csv \
  --UNK "<unk>" \
  --MASK "<mask>" \
  --UNK_ID 3 \
  --MASK_ID 250001


python3 src/embedding.py \
  --model FacebookAI/xlm-roberta-base \
  --prefix xlm_robert \
  --tokenizer_path data/tokenizer \
  --output_path data/embedding/xlm_roberta/ex1/other/full \
  --split other \
  --perturbed no \
  --train data/data_set/ex1/other/full/*train*.csv \
  --test  data/data_set/ex1/other/full/*test*.csv \
  --UNK "<unk>" \
  --MASK "<mask>" \
  --UNK_ID 3 \
  --MASK_ID 250001


python3 src/embedding.py \
  --model FacebookAI/xlm-roberta-base \
  --prefix xlm_robert \
  --tokenizer_path data/tokenizer \
  --output_path data/embedding/xlm_roberta/ex1/pseudo/full \
  --split pseudo \
  --perturbed no \
  --train data/data_set/ex1/pseudo/full/*train*.csv \
  --test  data/data_set/ex1/pseudo/full/*test*.csv  \
  --UNK "<unk>" \
  --MASK "<mask>" \
  --UNK_ID 3 \
  --MASK_ID 250001


10. embeddings encoder - UmBERTo

python3 src/embedding.py \
  --model Musixmatch/umberto-commoncrawl-cased-v1 \
  --prefix UMBERTO \
  --tokenizer_path data/tokenizer \
  --output_path data/embedding/umberto/ex1/simple/full \
  --split umberto_simple \
  --perturbed no \
  --train data/data_set/ex1/simple/full/*train*.csv \
  --test  data/data_set/ex1/simple/full/*test*.csv \
  --UNK "<unk>" \
  --MASK "<mask>" \
  --UNK_ID 3 \
  --MASK_ID 32004

python3 src/embedding.py \
  --model Musixmatch/umberto-commoncrawl-cased-v1 \
  --prefix UMBERTO \
  --tokenizer_path data/tokenizer \
  --output_path data/embedding/umberto/ex1/other/full \
  --split umberto_other \
  --perturbed no \
  --train data/data_set/ex1/other/full/*train*.csv \
  --test  data/data_set/ex1/other/full/*test*.csv \
  --UNK "<unk>" \
  --MASK "<mask>" \
  --UNK_ID 3 \
  --MASK_ID 32004 

python3 src/embedding.py \
  --model Musixmatch/umberto-commoncrawl-cased-v1 \
  --prefix UMBERTO \
  --tokenizer_path data/tokenizer \
  --output_path data/embedding/umberto/ex1/pseudo/full \
  --split umberto_pseudo \
  --perturbed no \
  --train data/data_set/ex1/pseudo/full/*train*.csv \
  --test  data/data_set/ex1/pseudo/full/*test*.csv \
  --UNK "<unk>" \
  --MASK "<mask>" \
  --UNK_ID 3 \
  --MASK_ID 32004 


5. decremental training

python3 src/decremental_training.py \
                --dataset_folder data/data_set/ex1/simple/full/ \
                --embeddings_folder data/embedding/bert_ita/ex1/simple/full/ \
                --output_folder data/data_set/ex1/simple/sampled/ \
                --output_emb_folder data/embedding/bert_ita/ex1/simple/sampled/ \
                --csv_pattern "*train*.csv" \
                --emb_pattern "*UNK*train*.pkl" \
                --key UNK


python3 src/decremental_training.py \
                --dataset_folder data/data_set/ex1/simple/full/ \
                --embeddings_folder data/embedding/bert_ita/ex1/simple/full/ \
                --output_folder data/data_set/ex1/simple/sampled/ \
                --output_emb_folder data/embedding/bert_ita/ex1/simple/sampled/ \
                --csv_pattern "*train*.csv" \
                --emb_pattern "*PREP*train*.pkl" \
                --key PREP

python3 src/decremental_training.py \
                --dataset_folder data/data_set/ex1/simple/full/ \
                --embeddings_folder data/embedding/bert_ita/ex1/simple/full/ \
                --output_folder data/data_set/ex1/simple/sampled/ \
                --output_emb_folder data/embedding/bert_ita/ex1/simple/sampled/ \
                --csv_pattern "*train*.csv" \
                --emb_pattern "*CLS*train*.pkl" \
                --key CLS

11. classificatore
DONE: parametrizzare anche SVM

BERT ITA 
SIMPLE

for k in UNK CLS PREP
do
  python3 src/LR.py \
    --seed 42 \
    --X_train data/embedding/bert_ita/ex1/simple/full/*${k}*train*.pkl \
    --X_test  data/embedding/bert_ita/ex1/simple/full/*${k}*test*.pkl \
    --y_train data/data_set/ex1/simple/full/*train*.csv \
    --y_test  data/data_set/ex1/simple/full/*test*.csv \
    -o data/outputs/ \
    -k $k \
    -m bert_ita \
    -s bert_ita/ex1/simple \
    -e ex1 \
    --contextual \
    --clf_name logistic_regression \
    --solver liblinear \
    --label construction
  done

for k in UNK CLS PREP
do
  python3 src/LR.py \
    --seed 42 \
    --X_train data/embedding/bert_ita/ex1/simple/full/*${k}*train*.pkl \
    --X_test  data/embedding/bert_ita/ex1/simple/full/*${k}*test*.pkl \
    --y_train data/data_set/ex1/simple/full/*train*.csv \
    --y_test  data/data_set/ex1/simple/full/*test*.csv \
    -o data/outputs/ \
    -k $k \
    -m bert_ita \
    -s bert_ita/ex1/simple \
    -e ex1 \
    --contextual \
    --clf_name SVM \
    --solver liblinear \
    --label construction
  done


BERT ITA 
OTHER

for k in UNK CLS PREP
do
  python3 src/LR.py \
    --seed 42 \
    --X_train data/embedding/bert_ita/ex1/other/full/*${k}*train*.pkl \
    --X_test  data/embedding/bert_ita/ex1/other/full/*${k}*test*.pkl \
    --y_train data/data_set/ex1/other/full/*train*.csv \
    --y_test  data/data_set/ex1/other/full/*test*.csv \
    -o data/outputs/ \
    -k $k \
    -m bert_ita \
    -s bert_ita/ex1/other \
    -e ex1 \
    --contextual \
    --clf_name logistic_regression \
    --solver liblinear \
    --label construction
  done

for k in UNK CLS PREP
do
  python3 src/LR.py \
    --seed 42 \
    --X_train data/embedding/bert_ita/ex1/other/full/*${k}*train*.pkl \
    --X_test  data/embedding/bert_ita/ex1/other/full/*${k}*test*.pkl \
    --y_train data/data_set/ex1/other/full/*train*.csv \
    --y_test  data/data_set/ex1/other/full/*test*.csv \
    -o data/outputs/ \
    -k $k \
    -m bert_ita \
    -s bert_ita/ex1/other \
    -e ex1 \
    --contextual \
    --clf_name SVM \
    --solver liblinear \
    --label construction
  done


BERT ITA 
PSEUDO

for k in UNK CLS PREP
do
  python3 src/LR.py \
    --seed 42 \
    --X_train data/embedding/bert_ita/ex1/pseudo/full/*${k}*train*.pkl \
    --X_test  data/embedding/bert_ita/ex1/pseudo/full/*${k}*test*.pkl \
    --y_train data/data_set/ex1/pseudo/full/*train*.csv \
    --y_test  data/data_set/ex1/pseudo/full/*test*.csv \
    -o data/outputs/ \
    -k $k \
    -m bert_ita \
    -s bert_ita/ex1/pseudo \
    -e ex1 \
    --contextual \
    --clf_name logistic_regression \
    --solver liblinear \
    --label construction
  done

for k in UNK CLS PREP
do
  python3 src/LR.py \
    --seed 42 \
    --X_train data/embedding/bert_ita/ex1/pseudo/full/*${k}*train*.pkl \
    --X_test  data/embedding/bert_ita/ex1/pseudo/full/*${k}*test*.pkl \
    --y_train data/data_set/ex1/pseudo/full/*train*.csv \
    --y_test  data/data_set/ex1/pseudo/full/*test*.csv \
    -o data/outputs/ \
    -k $k \
    -m bert_ita \
    -s bert_ita/ex1/pseudo \
    -e ex1 \
    --contextual \
    --clf_name SVM \
    --solver liblinear \
    --label construction
  done


12. classifictaore 

BERT ENG 


for k in UNK CLS PREP
do
  python3 src/LR.py \
    --seed 42 \
    --X_train data/embedding/bert_en/ex1_scivetti/simple/*${k}*train*.pkl \
    --X_test  data/embedding/bert_en/ex1_scivetti/simple/*${k}*test*.pkl \
    --y_train data/data_set/ex1_scivetti/simple/full/*train*.csv \
    --y_test  data/data_set/ex1_scivetti/simple/full/*test*.csv \
    -o data/outputs/ \
    -k $k \
    -m bert_en \
    -s bert_en/ex1_scivetti/simple \
    -e ex1_scivetti \
    --contextual \
    --clf_name SVM \
    --solver liblinear \
    --label construction
  done


for k in UNK CLS PREP
do
  python3 src/LR.py \
    --seed 42 \
    --X_train data/embedding/bert_en/ex1_scivetti/simple/*${k}*train*.pkl \
    --X_test  data/embedding/bert_en/ex1_scivetti/simple/*${k}*test*.pkl \
    --y_train data/data_set/ex1_scivetti/simple/full/*train*.csv \
    --y_test  data/data_set/ex1_scivetti/simple/full/*test*.csv \
    -o data/outputs/ \
    -k $k \
    -m bert_en \
    -s bert_en/ex1_scivetti/simple \
    -e ex1_scivetti \
    --contextual \
    --clf_name logistic_regression \
    --solver liblinear \
    --label construction
  done


13. classificatore m-BERT

m-BERT 
ita simple

for k in UNK CLS PREP
do
  python3 src/LR.py \
    --seed 42 \
    --X_train data/embedding/bert_multilingual/ex1/simple/full/*${k}*train*.pkl \
    --X_test  data/embedding/bert_multilingual/ex1/simple/full/*${k}*test*.pkl \
    --y_train data/data_set/ex1/simple/full/*train*.csv \
    --y_test  data/data_set/ex1/simple/full/*test*.csv \
    -o data/outputs/ \
    -k $k \
    -m bert_multilingual \
    -s bert_multilingual/ex1/simple \
    -e ex1 \
    --contextual \
    --clf_name logistic_regression \
    --solver liblinear \
    --label construction
  done


for k in UNK CLS PREP
do
  python3 src/LR.py \
    --seed 42 \
    --X_train data/embedding/bert_multilingual/ex1/simple/full/*${k}*train*.pkl \
    --X_test  data/embedding/bert_multilingual/ex1/simple/full/*${k}*test*.pkl \
    --y_train data/data_set/ex1/simple/full/*train*.csv \
    --y_test  data/data_set/ex1/simple/full/*test*.csv \
    -o data/outputs/ \
    -k $k \
    -m bert_multilingual \
    -s bert_multilingual/ex1/simple \
    -e ex1 \
    --contextual \
    --clf_name SVM \
    --solver liblinear \
    --label construction
  done

OTHER


for k in UNK CLS PREP
do
  python3 src/LR.py \
    --seed 42 \
    --X_train data/embedding/bert_multilingual/ex1/other/full/*${k}*train*.pkl \
    --X_test  data/embedding/bert_multilingual/ex1/other/full/*${k}*test*.pkl \
    --y_train data/data_set/ex1/other/full/*train*.csv \
    --y_test  data/data_set/ex1/other/full/*test*.csv \
    -o data/outputs/ \
    -k $k \
    -m bert_multilingual \
    -s bert_multilingual/ex1/other \
    -e ex1 \
    --contextual \
    --clf_name logistic_regression \
    --solver liblinear \
    --label construction
  done


for k in UNK CLS PREP
do
  python3 src/LR.py \
    --seed 42 \
    --X_train data/embedding/bert_multilingual/ex1/other/full/*${k}*train*.pkl \
    --X_test  data/embedding/bert_multilingual/ex1/other/full/*${k}*test*.pkl \
    --y_train data/data_set/ex1/other/full/*train*.csv \
    --y_test  data/data_set/ex1/other/full/*test*.csv \
    -o data/outputs/ \
    -k $k \
    -m bert_multilingual \
    -s bert_multilingual/ex1/other \
    -e ex1 \
    --contextual \
    --clf_name SVM \
    --solver liblinear \
    --label construction
  done

PSEUDO

for k in UNK CLS PREP
do
  python3 src/LR.py \
    --seed 42 \
    --X_train data/embedding/bert_multilingual/ex1/pseudo/full/*${k}*train*.pkl \
    --X_test  data/embedding/bert_multilingual/ex1/pseudo/full/*${k}*test*.pkl \
    --y_train data/data_set/ex1/pseudo/full/*train*.csv \
    --y_test  data/data_set/ex1/pseudo/full/*test*.csv \
    -o data/outputs/ \
    -k $k \
    -m bert_multilingual \
    -s bert_multilingual/ex1/pseudo \
    -e ex1 \
    --contextual \
    --clf_name logistic_regression \
    --solver liblinear \
    --label construction
  done


for k in UNK CLS PREP
do
  python3 src/LR.py \
    --seed 42 \
    --X_train data/embedding/bert_multilingual/ex1/pseudo/full/*${k}*train*.pkl \
    --X_test  data/embedding/bert_multilingual/ex1/pseudo/full/*${k}*test*.pkl \
    --y_train data/data_set/ex1/pseudo/full/*train*.csv \
    --y_test  data/data_set/ex1/pseudo/full/*test*.csv \
    -o data/outputs/ \
    -k $k \
    -m bert_multilingual \
    -s bert_multilingual/ex1/pseudo \
    -e ex1 \
    --contextual \
    --clf_name SVM \
    --solver liblinear \
    --label construction
  done

eng

for k in UNK CLS PREP
do
  python3 src/LR.py \
    --seed 42 \
    --X_train data/embedding/bert_multilingual/ex1_scivetti/simple/*${k}*train*.pkl \
    --X_test  data/embedding/bert_multilingual/ex1_scivetti/simple/*${k}*test*.pkl \
    --y_train data/data_set/ex1_scivetti/simple/full/*train*.csv \
    --y_test  data/data_set/ex1_scivetti/simple/full/*test*.csv \
    -o data/outputs/ \
    -k $k \
    -m bert_multilingual \
    -s bert_multilingual/ex1_scivetti/simple \
    -e ex1_scivetti \
    --contextual \
    --clf_name logistic_regression \
    --solver liblinear \
    --label construction
  done

for k in UNK CLS PREP
do
  python3 src/LR.py \
    --seed 42 \
    --X_train data/embedding/bert_multilingual/ex1_scivetti/simple/*${k}*train*.pkl \
    --X_test  data/embedding/bert_multilingual/ex1_scivetti/simple/*${k}*test*.pkl \
    --y_train data/data_set/ex1_scivetti/simple/full/*train*.csv \
    --y_test  data/data_set/ex1_scivetti/simple/full/*test*.csv \
    -o data/outputs/ \
    -k $k \
    -m bert_multilingual \
    -s bert_multilingual/ex1_scivetti/simple \
    -e ex1_scivetti \
    --contextual \
    --clf_name SVM \
    --solver liblinear \
    --label construction
  done

14. umberto


SIMPLE

for k in UNK CLS PREP
do
  python3 src/LR.py \
    --seed 42 \
    --X_train data/embedding/umberto/ex1/simple/full/*${k}*train*.pkl \
    --X_test  data/embedding/umberto/ex1/simple/full/*${k}*test*.pkl \
    --y_train data/data_set/ex1/simple/full/*train*.csv \
    --y_test  data/data_set/ex1/simple/full/*test*.csv \
    -o data/outputs/ \
    -k $k \
    -m umberto \
    -s umberto/ex1/simple \
    -e ex1 \
    --contextual \
    --clf_name logistic_regression \
    --solver liblinear \
    --label construction
  done


for k in UNK CLS PREP
do
  python3 src/LR.py \
    --seed 42 \
    --X_train data/embedding/umberto/ex1/simple/full/*${k}*train*.pkl \
    --X_test  data/embedding/umberto/ex1/simple/full/*${k}*test*.pkl \
    --y_train data/data_set/ex1/simple/full/*train*.csv \
    --y_test  data/data_set/ex1/simple/full/*test*.csv \
    -o data/outputs/ \
    -k $k \
    -m umberto \
    -s umberto/ex1/simple \
    -e ex1 \
    --contextual \
    --clf_name SVM \
    --solver liblinear \
    --label construction
  done

OTHER

for k in UNK CLS PREP
do
  python3 src/LR.py \
    --seed 42 \
    --X_train data/embedding/umberto/ex1/other/full/*${k}*train*.pkl \
    --X_test  data/embedding/umberto/ex1/other/full/*${k}*test*.pkl \
    --y_train data/data_set/ex1/other/full/*train*.csv \
    --y_test  data/data_set/ex1/other/full/*test*.csv \
    -o data/outputs/ \
    -k $k \
    -m umberto \
    -s umberto/ex1/other \
    -e ex1 \
    --contextual \
    --clf_name logistic_regression \
    --solver liblinear \
    --label construction
  done


for k in UNK CLS PREP
do
  python3 src/LR.py \
    --seed 42 \
    --X_train data/embedding/umberto/ex1/other/full/*${k}*train*.pkl \
    --X_test  data/embedding/umberto/ex1/other/full/*${k}*test*.pkl \
    --y_train data/data_set/ex1/other/full/*train*.csv \
    --y_test  data/data_set/ex1/other/full/*test*.csv \
    -o data/outputs/ \
    -k $k \
    -m umberto \
    -s umberto/ex1/other \
    -e ex1 \
    --contextual \
    --clf_name SVM \
    --solver liblinear \
    --label construction
  done

PSEUDO

for k in UNK CLS PREP
do
  python3 src/LR.py \
    --seed 42 \
    --X_train data/embedding/umberto/ex1/pseudo/full/*${k}*train*.pkl \
    --X_test  data/embedding/umberto/ex1/pseudo/full/*${k}*test*.pkl \
    --y_train data/data_set/ex1/pseudo/full/*train*.csv \
    --y_test  data/data_set/ex1/pseudo/full/*test*.csv \
    -o data/outputs/ \
    -k $k \
    -m umberto \
    -s umberto/ex1/pseudo \
    -e ex1 \
    --contextual \
    --clf_name logistic_regression \
    --solver liblinear \
    --label construction
  done


for k in UNK CLS PREP
do
  python3 src/LR.py \
    --seed 42 \
    --X_train data/embedding/umberto/ex1/pseudo/full/*${k}*train*.pkl \
    --X_test  data/embedding/umberto/ex1/pseudo/full/*${k}*test*.pkl \
    --y_train data/data_set/ex1/pseudo/full/*train*.csv \
    --y_test  data/data_set/ex1/pseudo/full/*test*.csv \
    -o data/outputs/ \
    -k $k \
    -m umberto \
    -s umberto/ex1/pseudo \
    -e ex1 \
    --contextual \
    --clf_name SVM \
    --solver liblinear \
    --label construction
  done


15. xml_roberta

simple ita


for k in UNK CLS PREP
do
  python3 src/LR.py \
    --seed 42 \
    --X_train data/embedding/xlm_roberta/ex1/simple/full/*${k}*train*.pkl \
    --X_test  data/embedding/xlm_roberta/ex1/simple/full/*${k}*test*.pkl \
    --y_train data/data_set/ex1/simple/full/*train*.csv \
    --y_test  data/data_set/ex1/simple/full/*test*.csv \
    -o data/outputs/ \
    -k $k \
    -m xlm_roberta \
    -s xlm_roberta/ex1/simple \
    -e ex1 \
    --contextual \
    --clf_name logistic_regression \
    --solver liblinear \
    --label construction
  done


for k in UNK CLS PREP
do
  python3 src/LR.py \
    --seed 42 \
    --X_train data/embedding/xlm_roberta/ex1/simple/full/*${k}*train*.pkl \
    --X_test  data/embedding/xlm_roberta/ex1/simple/full/*${k}*test*.pkl \
    --y_train data/data_set/ex1/simple/full/*train*.csv \
    --y_test  data/data_set/ex1/simple/full/*test*.csv \
    -o data/outputs/ \
    -k $k \
    -m xlm_roberta \
    -s xlm_roberta/ex1/simple \
    -e ex1 \
    --contextual \
    --clf_name SVM \
    --solver liblinear \
    --label construction
  done

OTHER


for k in UNK CLS PREP
do
  python3 src/LR.py \
    --seed 42 \
    --X_train data/embedding/xlm_roberta/ex1/other/full/*${k}*train*.pkl \
    --X_test  data/embedding/xlm_roberta/ex1/other/full/*${k}*test*.pkl \
    --y_train data/data_set/ex1/other/full/*train*.csv \
    --y_test  data/data_set/ex1/other/full/*test*.csv \
    -o data/outputs/ \
    -k $k \
    -m xlm_roberta \
    -s xlm_roberta/ex1/other \
    -e ex1 \
    --contextual \
    --clf_name logistic_regression \
    --solver liblinear \
    --label construction
  done


for k in UNK CLS PREP
do
  python3 src/LR.py \
    --seed 42 \
    --X_train data/embedding/xlm_roberta/ex1/other/full/*${k}*train*.pkl \
    --X_test  data/embedding/xlm_roberta/ex1/other/full/*${k}*test*.pkl \
    --y_train data/data_set/ex1/other/full/*train*.csv \
    --y_test  data/data_set/ex1/other/full/*test*.csv \
    -o data/outputs/ \
    -k $k \
    -m xlm_roberta \
    -s xlm_roberta/ex1/other \
    -e ex1 \
    --contextual \
    --clf_name SVM \
    --solver liblinear \
    --label construction
  done

PSEUDO

for k in UNK CLS PREP
do
  python3 src/LR.py \
    --seed 42 \
    --X_train data/embedding/xlm_roberta/ex1/pseudo/full/*${k}*train*.pkl \
    --X_test  data/embedding/xlm_roberta/ex1/pseudo/full/*${k}*test*.pkl \
    --y_train data/data_set/ex1/pseudo/full/*train*.csv \
    --y_test  data/data_set/ex1/pseudo/full/*test*.csv \
    -o data/outputs/ \
    -k $k \
    -m xlm_roberta \
    -s xlm_roberta/ex1/pseudo \
    -e ex1 \
    --contextual \
    --clf_name logistic_regression \
    --solver liblinear \
    --label construction
  done


for k in UNK CLS PREP
do
  python3 src/LR.py \
    --seed 42 \
    --X_train data/embedding/xlm_roberta/ex1/pseudo/full/*${k}*train*.pkl \
    --X_test  data/embedding/xlm_roberta/ex1/pseudo/full/*${k}*test*.pkl \
    --y_train data/data_set/ex1/pseudo/full/*train*.csv \
    --y_test  data/data_set/ex1/pseudo/full/*test*.csv \
    -o data/outputs/ \
    -k $k \
    -m xlm_roberta \
    -s xlm_roberta/ex1/pseudo \
    -e ex1 \
    --contextual \
    --clf_name SVM \
    --solver liblinear \
    --label construction
  done

eng

for k in UNK CLS PREP
do
  python3 src/LR.py \
    --seed 42 \
    --X_train data/embedding/xlm_roberta/ex1_scivetti/simple/*${k}*train*.pkl \
    --X_test  data/embedding/xlm_roberta/ex1_scivetti/simple/*${k}*test*.pkl \
    --y_train data/data_set/ex1_scivetti/simple/full/*train*.csv \
    --y_test  data/data_set/ex1_scivetti/simple/full/*test*.csv \
    -o data/outputs/ \
    -k $k \
    -m xlm_roberta \
    -s xlm_roberta/ex1_scivetti/simple \
    -e ex1_scivetti \
    --contextual \
    --clf_name logistic_regression \
    --solver liblinear \
    --label construction
  done

for k in UNK CLS PREP
do
  python3 src/LR.py \
    --seed 42 \
    --X_train data/embedding/xlm_roberta/ex1_scivetti/simple/*${k}*train*.pkl \
    --X_test  data/embedding/xlm_roberta/ex1_scivetti/simple/*${k}*test*.pkl \
    --y_train data/data_set/ex1_scivetti/simple/full/*train*.csv \
    --y_test  data/data_set/ex1_scivetti/simple/full/*test*.csv \
    -o data/outputs/ \
    -k $k \
    -m xlm_roberta \
    -s xlm_roberta/ex1_scivetti/simple \
    -e ex1_scivetti \
    --contextual \
    --clf_name SVM \
    --solver liblinear \
    --label construction
  done


16. baseline fasttext costrN


ITA NPN SIMPLE


python3 src/LR.py \
  --seed 42 \
  --X_train data/embedding/fasttext/ex1/simple/costrN/*train*.tsv \
  --X_test  data/embedding/fasttext/ex1/simple/costrN/*test*.tsv \
  --y_train data/data_set/ex1/simple/full/*train*.csv \
  --y_test data/data_set/ex1/simple/full/*test*.csv \
  -o data/outputs/ \
  -m fasttext \
  -s fasttext/ex1/simple/costrN \
  -e ex1 \
  --clf_name SVM \
  --solver liblinear \
  --label construction

python3 src/LR.py \
  --seed 42 \
  --X_train data/embedding/fasttext/ex1/simple/costrN/*train*.tsv \
  --X_test  data/embedding/fasttext/ex1/simple/costrN/*test*.tsv \
  --y_train data/data_set/ex1/simple/full/*train*.csv \
  --y_test data/data_set/ex1/simple/full/*test*.csv \
  -o data/outputs/ \
  -m fasttext \
  -s fasttext/ex1/simple/costrN \
  -e ex1 \
  --clf_name logistic_regression \
  --solver liblinear \
  --label construction


OTHER


python3 src/LR.py \
  --seed 42 \
  --X_train data/embedding/fasttext/ex1/other/costrN/*train*.tsv \
  --X_test  data/embedding/fasttext/ex1/other/costrN/*test*.tsv \
  --y_train data/data_set/ex1/other/full/*train*.csv \
  --y_test data/data_set/ex1/other/full/*test*.csv \
  -o data/outputs/ \
  -m fasttext \
  -s fasttext/ex1/other/costrN \
  -e ex1 \
  --clf_name SVM \
  --solver liblinear \
  --label construction

python3 src/LR.py \
  --seed 42 \
  --X_train data/embedding/fasttext/ex1/other/costrN/*train*.tsv \
  --X_test  data/embedding/fasttext/ex1/other/costrN/*test*.tsv \
  --y_train data/data_set/ex1/other/full/*train*.csv \
  --y_test data/data_set/ex1/other/full/*test*.csv \
  -o data/outputs/ \
  -m fasttext \
  -s fasttext/ex1/other/costrN \
  -e ex1 \
  --clf_name logistic_regression \
  --solver liblinear \
  --label construction


PSEUDO

python3 src/LR.py \
  --seed 42 \
  --X_train data/embedding/fasttext/ex1/pseudo/costrN/*train*.tsv \
  --X_test  data/embedding/fasttext/ex1/pseudo/costrN/*test*.tsv \
  --y_train data/data_set/ex1/pseudo/full/*train*.csv \
  --y_test data/data_set/ex1/pseudo/full/*test*.csv \
  -o data/outputs/ \
  -m fasttext \
  -s fasttext/ex1/pseudo/costrN \
  -e ex1 \
  --clf_name SVM \
  --solver liblinear \
  --label construction

python3 src/LR.py \
  --seed 42 \
  --X_train data/embedding/fasttext/ex1/pseudo/costrN/*train*.tsv \
  --X_test  data/embedding/fasttext/ex1/pseudo/costrN/*test*.tsv \
  --y_train data/data_set/ex1/pseudo/full/*train*.csv \
  --y_test data/data_set/ex1/pseudo/full/*test*.csv \
  -o data/outputs/ \
  -m fasttext \
  -s fasttext/ex1/pseudo/costrN \
  -e ex1 \
  --clf_name logistic_regression \
  --solver liblinear \
  --label construction



17. fasttext ita noun


python3 src/LR.py \
  --seed 42 \
  --X_train data/embedding/fasttext/ex1/simple/noun/*train*.tsv \
  --X_test  data/embedding/fasttext/ex1/simple/noun/*test*.tsv \
  --y_train data/data_set/ex1/simple/full/*train*.csv \
  --y_test data/data_set/ex1/simple/full/*test*.csv \
  -o data/outputs/ \
  -m fasttext \
  -s fasttext/ex1/simple/noun \
  -e ex1 \
  --clf_name SVM \
  --solver liblinear \
  --label construction

python3 src/LR.py \
  --seed 42 \
  --X_train data/embedding/fasttext/ex1/simple/noun/*train*.tsv \
  --X_test  data/embedding/fasttext/ex1/simple/noun/*test*.tsv \
  --y_train data/data_set/ex1/simple/full/*train*.csv \
  --y_test data/data_set/ex1/simple/full/*test*.csv \
  -o data/outputs/ \
  -m fasttext \
  -s fasttext/ex1/simple/noun \
  -e ex1 \
  --clf_name logistic_regression \
  --solver liblinear \
  --label construction


OTHER


python3 src/LR.py \
  --seed 42 \
  --X_train data/embedding/fasttext/ex1/other/noun/*train*.tsv \
  --X_test  data/embedding/fasttext/ex1/other/noun/*test*.tsv \
  --y_train data/data_set/ex1/other/full/*train*.csv \
  --y_test data/data_set/ex1/other/full/*test*.csv \
  -o data/outputs/ \
  -m fasttext \
  -s fasttext/ex1/other/noun \
  -e ex1 \
  --clf_name SVM \
  --solver liblinear \
  --label construction

python3 src/LR.py \
  --seed 42 \
  --X_train data/embedding/fasttext/ex1/other/noun/*train*.tsv \
  --X_test  data/embedding/fasttext/ex1/other/noun/*test*.tsv \
  --y_train data/data_set/ex1/other/full/*train*.csv \
  --y_test data/data_set/ex1/other/full/*test*.csv \
  -o data/outputs/ \
  -m fasttext \
  -s fasttext/ex1/other/noun \
  -e ex1 \
  --clf_name logistic_regression \
  --solver liblinear \
  --label construction


PSEUDO

python3 src/LR.py \
  --seed 42 \
  --X_train data/embedding/fasttext/ex1/pseudo/noun/*train*.tsv \
  --X_test  data/embedding/fasttext/ex1/pseudo/noun/*test*.tsv \
  --y_train data/data_set/ex1/pseudo/full/*train*.csv \
  --y_test data/data_set/ex1/pseudo/full/*test*.csv \
  -o data/outputs/ \
  -m fasttext \
  -s fasttext/ex1/pseudo/noun \
  -e ex1 \
  --clf_name SVM \
  --solver liblinear \
  --label construction

python3 src/LR.py \
  --seed 42 \
  --X_train data/embedding/fasttext/ex1/pseudo/noun/*train*.tsv \
  --X_test  data/embedding/fasttext/ex1/pseudo/noun/*test*.tsv \
  --y_train data/data_set/ex1/pseudo/full/*train*.csv \
  --y_test data/data_set/ex1/pseudo/full/*test*.csv \
  -o data/outputs/ \
  -m fasttext \
  -s fasttext/ex1/pseudo/noun \
  -e ex1 \
  --clf_name logistic_regression \
  --solver liblinear \
  --label construction

18. itwac ita npn costrN

python3 src/LR.py \
  --seed 42 \
  --X_train data/embedding/itwac/ex1/simple/costrN/*train*.tsv \
  --X_test  data/embedding/itwac/ex1/simple/costrN/*test*.tsv \
  --y_train data/data_set/ex1/simple/full/*train*.csv \
  --y_test data/data_set/ex1/simple/full/*test*.csv \
  -o data/outputs/ \
  -m itwac \
  -s itwac/ex1/simple/costrN \
  -e ex1 \
  --clf_name SVM \
  --solver liblinear \
  --label construction

python3 src/LR.py \
  --seed 42 \
  --X_train data/embedding/itwac/ex1/simple/costrN/*train*.tsv \
  --X_test  data/embedding/itwac/ex1/simple/costrN/*test*.tsv \
  --y_train data/data_set/ex1/simple/full/*train*.csv \
  --y_test data/data_set/ex1/simple/full/*test*.csv \
  -o data/outputs/ \
  -m itwac \
  -s itwac/ex1/simple/costrN \
  -e ex1 \
  --clf_name logistic_regression \
  --solver liblinear \
  --label construction


OTHER


python3 src/LR.py \
  --seed 42 \
  --X_train data/embedding/itwac/ex1/other/costrN/*train*.tsv \
  --X_test  data/embedding/itwac/ex1/other/costrN/*test*.tsv \
  --y_train data/data_set/ex1/other/full/*train*.csv \
  --y_test data/data_set/ex1/other/full/*test*.csv \
  -o data/outputs/ \
  -m itwac \
  -s itwac/ex1/other/costrN \
  -e ex1 \
  --clf_name SVM \
  --solver liblinear \
  --label construction

python3 src/LR.py \
  --seed 42 \
  --X_train data/embedding/itwac/ex1/other/costrN/*train*.tsv \
  --X_test  data/embedding/itwac/ex1/other/costrN/*test*.tsv \
  --y_train data/data_set/ex1/other/full/*train*.csv \
  --y_test data/data_set/ex1/other/full/*test*.csv \
  -o data/outputs/ \
  -m itwac \
  -s itwac/ex1/other/costrN \
  -e ex1 \
  --clf_name logistic_regression \
  --solver liblinear \
  --label construction


PSEUDO

python3 src/LR.py \
  --seed 42 \
  --X_train data/embedding/itwac/ex1/pseudo/costrN/*train*.tsv \
  --X_test  data/embedding/itwac/ex1/pseudo/costrN/*test*.tsv \
  --y_train data/data_set/ex1/pseudo/full/*train*.csv \
  --y_test data/data_set/ex1/pseudo/full/*test*.csv \
  -o data/outputs/ \
  -m itwac \
  -s itwac/ex1/pseudo/costrN \
  -e ex1 \
  --clf_name SVM \
  --solver liblinear \
  --label construction

python3 src/LR.py \
  --seed 42 \
  --X_train data/embedding/itwac/ex1/pseudo/costrN/*train*.tsv \
  --X_test  data/embedding/itwac/ex1/pseudo/costrN/*test*.tsv \
  --y_train data/data_set/ex1/pseudo/full/*train*.csv \
  --y_test data/data_set/ex1/pseudo/full/*test*.csv \
  -o data/outputs/ \
  -m itwac \
  -s itwac/ex1/pseudo/costrN \
  -e ex1 \
  --clf_name logistic_regression \
  --solver liblinear \
  --label construction

19. baseline itwac noun

python3 src/LR.py \
  --seed 42 \
  --X_train data/embedding/itwac/ex1/simple/noun/*train*.tsv \
  --X_test  data/embedding/itwac/ex1/simple/noun/*test*.tsv \
  --y_train data/data_set/ex1/simple/full/*train*.csv \
  --y_test data/data_set/ex1/simple/full/*test*.csv \
  -o data/outputs/ \
  -m itwac \
  -s itwac/ex1/simple/noun \
  -e ex1 \
  --clf_name SVM \
  --solver liblinear \
  --label construction

python3 src/LR.py \
  --seed 42 \
  --X_train data/embedding/itwac/ex1/simple/noun/*train*.tsv \
  --X_test  data/embedding/itwac/ex1/simple/noun/*test*.tsv \
  --y_train data/data_set/ex1/simple/full/*train*.csv \
  --y_test data/data_set/ex1/simple/full/*test*.csv \
  -o data/outputs/ \
  -m itwac \
  -s itwac/ex1/simple/noun \
  -e ex1 \
  --clf_name logistic_regression \
  --solver liblinear \
  --label construction


OTHER


python3 src/LR.py \
  --seed 42 \
  --X_train data/embedding/itwac/ex1/other/noun/*train*.tsv \
  --X_test  data/embedding/itwac/ex1/other/noun/*test*.tsv \
  --y_train data/data_set/ex1/other/full/*train*.csv \
  --y_test data/data_set/ex1/other/full/*test*.csv \
  -o data/outputs/ \
  -m itwac \
  -s itwac/ex1/other/noun \
  -e ex1 \
  --clf_name SVM \
  --solver liblinear \
  --label construction

python3 src/LR.py \
  --seed 42 \
  --X_train data/embedding/itwac/ex1/other/noun/*train*.tsv \
  --X_test  data/embedding/itwac/ex1/other/noun/*test*.tsv \
  --y_train data/data_set/ex1/other/full/*train*.csv \
  --y_test data/data_set/ex1/other/full/*test*.csv \
  -o data/outputs/ \
  -m itwac \
  -s itwac/ex1/other/noun \
  -e ex1 \
  --clf_name logistic_regression \
  --solver liblinear \
  --label construction


PSEUDO

python3 src/LR.py \
  --seed 42 \
  --X_train data/embedding/itwac/ex1/pseudo/noun/*train*.tsv \
  --X_test  data/embedding/itwac/ex1/pseudo/noun/*test*.tsv \
  --y_train data/data_set/ex1/pseudo/full/*train*.csv \
  --y_test data/data_set/ex1/pseudo/full/*test*.csv \
  -o data/outputs/ \
  -m itwac \
  -s itwac/ex1/pseudo/noun \
  -e ex1 \
  --clf_name SVM \
  --solver liblinear \
  --label construction

python3 src/LR.py \
  --seed 42 \
  --X_train data/embedding/itwac/ex1/pseudo/noun/*train*.tsv \
  --X_test  data/embedding/itwac/ex1/pseudo/noun/*test*.tsv \
  --y_train data/data_set/ex1/pseudo/full/*train*.csv \
  --y_test data/data_set/ex1/pseudo/full/*test*.csv \
  -o data/outputs/ \
  -m itwac \
  -s itwac/ex1/pseudo/noun \
  -e ex1 \
  --clf_name logistic_regression \
  --solver liblinear \
  --label construction


20. baseline glove eng 

python3 src/LR.py \
  --seed 42 \
  --X_train data/embedding/glove/ex1_scivetti/simple/*train*.tsv \
  --X_test  data/embedding/glove/ex1_scivetti/simple/*test*.tsv \
  --y_train data/data_set/ex1_scivetti/simple/full/*train*.csv \
  --y_test data/data_set/ex1_scivetti/simple/full/*test*.csv \
  -o data/outputs/ \
  -m glove \
  -s glove/ex1_scivetti/simple \
  -e ex1_scivetti \
  --clf_name logistic_regression \
  --solver liblinear \
  --label construction


python3 src/LR.py \
  --seed 42 \
  --X_train data/embedding/glove/ex1_scivetti/simple/*train*.tsv \
  --X_test  data/embedding/glove/ex1_scivetti/simple/*test*.tsv \
  --y_train data/data_set/ex1_scivetti/simple/full/*train*.csv \
  --y_test data/data_set/ex1_scivetti/simple/full/*test*.csv \
  -o data/outputs/ \
  -m glove \
  -s glove/ex1_scivetti/simple \
  -e ex1_scivetti \
  --clf_name SVM \
  --solver liblinear \
  --label construction



21. fasttext eng


python3 src/LR.py \
  --seed 42 \
  --X_train data/embedding/fasttext/ex1_scivetti/simple/*train*.tsv \
  --X_test  data/embedding/fasttext/ex1_scivetti/simple/*test*.tsv \
  --y_train data/data_set/ex1_scivetti/simple/full/*train*.csv \
  --y_test data/data_set/ex1_scivetti/simple/full/*test*.csv \
  -o data/outputs/ \
  -m fasttext \
  -s fasttext/ex1_scivetti/simple \
  -e ex1_scivetti \
  --clf_name logistic_regression \
  --solver liblinear \
  --label construction


python3 src/LR.py \
  --seed 42 \
  --X_train data/embedding/fasttext/ex1_scivetti/simple/*train*.tsv \
  --X_test  data/embedding/fasttext/ex1_scivetti/simple/*test*.tsv \
  --y_train data/data_set/ex1_scivetti/simple/full/*train*.csv \
  --y_test data/data_set/ex1_scivetti/simple/full/*test*.csv \
  -o data/outputs/ \
  -m fasttext \
  -s fasttext/ex1_scivetti/simple \
  -e ex1_scivetti \
  --clf_name SVM \
  --solver liblinear \
  --label construction


22. decremental

BERT ITA 
SIMPLE

for d in 240 120 60
do

  for k in UNK CLS PREP
  do
    python3 src/LR.py \
      --seed 42 \
      --X_train data/embedding/bert_ita/ex1/simple/sampled/*train*${d}_${k}.pkl \
      --X_test  data/embedding/bert_ita/ex1/simple/full/*${k}*test*.pkl \
      --y_train data/data_set/ex1/simple/sampled/*train*${d}.csv \
      --y_test  data/data_set/ex1/simple/full/*test*.csv \
      -o data/outputs/ \
      -k $k \
      -m bert_ita \
      -s bert_ita/ex1/simple \
      -e ex1_${d} \
      --decremental \
      --contextual \
      --clf_name logistic_regression \
      --solver liblinear \
      --label construction
    done
done

23. control classifier

for k in UNK CLS PREP
do
  python3 src/LR.py \
    --seed 42 \
    --X_train data/embedding/bert_ita/ex1/simple/full/*${k}*train*.pkl \
    --X_test  data/embedding/bert_ita/ex1/simple/full/*${k}*test*.pkl \
    --y_train data/data_set/ex1/simple/control/*train*.csv \
    --y_test  data/data_set/ex1/simple/control/*test*.csv \
    -o data/outputs/ \
    -k $k \
    -m bert_ita \
    -s bert_ita/ex1/simple/control \
    -e ex1 \
    --contextual \
    --clf_name logistic_regression \
    --solver liblinear \
    --label construction
  done

20. matrici di confusione

21. grafici
