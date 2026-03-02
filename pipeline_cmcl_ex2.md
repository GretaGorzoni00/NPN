1. campionamento.py has already been carried out for Experiment 1.

2. sampling.py

```

python3 src/sampling.py \
 		--df_minima_path data/ex2_scivetti.csv \
  		--df_dataset_path data/source/full_dataset_scivetti.csv \
  		--output_folder data/data_set/ \
 		--experiment_type ex2_scivetti \
  		--configuration simple

python3 src/sampling.py \
 		--df_minima_path data/ex2_simple.csv \
  		--df_dataset_path data/source/full_dataset_itanpn.csv \
  		--output_folder data/data_set/ \
 		--experiment_type ex2 \
  		--configuration simple

python3 src/sampling.py \
 		--df_minima_path data/ex2_distractors.csv \
  		--df_dataset_path data/source/full_dataset.csv \
  		--output_folder data/data_set/ \
 		--experiment_type ex2 \
  		--configuration per_dopo
```



3. Data for Control Classifier
```

python3 src/control_classifier.py --files_train data/data_set/ex2/simple/full/*train*.csv \
                --files_test data/data_set/ex2/simple/full/*test*.csv \
                --control_dir data/data_set/ex2/simple/control/ \
                --_FIELD noun \
                --_LABEL_FIELD meaning
```


4. Embeddings extraction  -- fasttext

```

python3 src/fasttext_embeddings.py --fasttext_model static_model/cc.it.300.bin \
								--output_folder data/embedding/fasttext/ex2/simple/noun/ \
								--input_files data/data_set/ex2/simple/full/*.csv \
								--FIELD noun

python3 src/fasttext_embeddings.py --fasttext_model static_model/cc.it.300.bin \
								--output_folder data/embedding/fasttext/ex2/simple/costrN/ \
								--input_files data/data_set/ex2/simple/full/*.csv \
								--FIELD costrN 


python3 src/fasttext_embeddings.py --fasttext_model static_model/cc.en.300.bin \
								--output_folder data/embedding/fasttext/ex2_scivetti/simple/ \
								--input_files data/data_set/ex2_scivetti/simple/full/*.csv \
								--FIELD noun

python3 src/fasttext_embeddings.py --fasttext_model static_model/cc.it.300.bin \
								--output_folder data/embedding/fasttext/ex2/per_dopo/noun/ \
								--input_files data/data_set/ex2/per_dopo/full/*.csv \
								--FIELD noun

python3 src/fasttext_embeddings.py --fasttext_model static_model/cc.it.300.bin \
								--output_folder data/embedding/fasttext/ex2/per_dopo/costrN/ \
								--input_files data/data_set/ex2/per_dopo/full/*.csv \
								--FIELD costrN

```

5. Embeddings extraction  -- word2vec

```

python3 src/w2v_embeddings.py --w2vec_model static_model/vettori_itwac.txt \
								--output_folder data/embedding/itwac/ex2/simple/noun/ \
								--input_files data/data_set/ex2/simple/full/*.csv \
								--FIELD noun



python3 src/w2v_embeddings.py --w2vec_model static_model/vettori_itwac.txt \
								--output_folder data/embedding/itwac/ex2/simple/costrN/ \
								--input_files data/data_set/ex2/simple/full/*.csv \
								--FIELD costrN


python3 src/w2v_embeddings.py --w2vec_model static_model/vettori_itwac.txt \
								--output_folder data/embedding/itwac/ex2/per_dopo/noun/ \
								--input_files data/data_set/ex2/per_dopo/full/*.csv \
								--FIELD noun



python3 src/w2v_embeddings.py --w2vec_model static_model/vettori_itwac.txt \
								--output_folder data/embedding/itwac/ex2/per_dopo/costrN/ \
								--input_files data/data_set/ex2/per_dopo/full/*.csv \
								--FIELD costrN

```

6. Embeddings extraction  -- Glove

```

python3 src/w2v_embeddings.py --w2vec_model static_model/glove.6B.300d.txt \
								--output_folder data/embedding/glove/ex2_scivetti/simple/ \
								--input_files data/data_set/ex2_scivetti/simple/full/*.csv \
								--FIELD noun


```

7. Embeddings extraction -- bert base cased eng

```

python3 src/embedding.py \
  --model bert-base-cased \
  --prefix BERT \
  --tokenizer_path data/tokenizer \
  --output_path data/embedding/bert_en/ex2_scivetti/simple \
  --split scivetti_simple \
  --perturbed no \
  --train data/data_set/ex2_scivetti/simple/full/*train*.csv \
  --test  data/data_set/ex2_scivetti/simple/full/*test*.csv \
  --UNK "[UNK]" \
  --MASK "[MASK]" \
  --UNK_ID 100 \
  --MASK_ID 103

```

8. Embeddings extraction - bert base cased ita

```
python3 src/embedding.py \
  --model dbmdz/bert-base-italian-cased \
  --prefix BERT \
  --tokenizer_path data/tokenizer \
  --output_path data/embedding/bert_ita/ex2/simple/full \
  --split simple \
  --perturbed no \
  --train data/data_set/ex2/simple/full/*train*.csv \
  --test  data/data_set/ex2/simple/full/*test*.csv \
  --UNK "[UNK]" \
  --MASK "[MASK]" \
  --UNK_ID 101 \
  --MASK_ID 104


python3 src/embedding.py \
  --model dbmdz/bert-base-italian-cased \
  --prefix BERT \
  --tokenizer_path data/tokenizer \
  --output_path data/embedding/bert_ita/ex2/per_dopo/full \
  --split per_dopo \
  --perturbed no \
  --train data/data_set/ex2/per_dopo/full/*train*.csv \
  --test  data/data_set/ex2/per_dopo/full/*test*.csv \
  --UNK "[UNK]" \
  --MASK "[MASK]" \
  --UNK_ID 101 \
  --MASK_ID 104

```


9. Embeddings extraction multilingual - mBERT e xmlr

```

python3 src/embedding.py \
  --model bert-base-multilingual-cased \
  --prefix mBERT \
  --tokenizer_path data/tokenizer \
  --output_path data/embedding/bert_multilingual/ex2_scivetti/simple \
  --split scivetti_simple \
  --perturbed no \
  --train data/data_set/ex2_scivetti/simple/full/*train*.csv \
  --test  data/data_set/ex2_scivetti/simple/full/*test*.csv \
  --UNK "[UNK]" \
  --MASK "[MASK]" \
  --UNK_ID 100 \
  --MASK_ID 103



  python3 src/embedding.py \
  --model bert-base-multilingual-cased \
  --prefix mBERT \
  --tokenizer_path data/tokenizer \
  --output_path data/embedding/bert_multilingual/ex2/simple/full \
  --split simple \
  --perturbed no \
  --train data/data_set/ex2/simple/full/*train*.csv \
  --test  data/data_set/ex2/simple/full/*test*.csv \
  --UNK "[UNK]" \
  --MASK "[MASK]" \
  --UNK_ID 100 \
  --MASK_ID 103


  python3 src/embedding.py \
  --model bert-base-multilingual-cased \
  --prefix mBERT \
  --tokenizer_path data/tokenizer \
  --output_path data/embedding/bert_multilingual/ex2/per_dopo/full \
  --split per_dopo \
  --perturbed no \
  --train data/data_set/ex2/per_dopo/full/*train*.csv \
  --test  data/data_set/ex2/per_dopo/full/*test*.csv \
  --UNK "[UNK]" \
  --MASK "[MASK]" \
  --UNK_ID 100 \
  --MASK_ID 103

```

```

python3 src/embedding.py \
  --model FacebookAI/xlm-roberta-base \
  --prefix xlm_roberta \
  --tokenizer_path data/tokenizer \
  --output_path data/embedding/xlm_roberta/ex2_scivetti/simple \
  --split scivetti_simple \
  --perturbed no \
  --train data/data_set/ex2_scivetti/simple/full/*train*.csv \
  --test  data/data_set/ex2_scivetti/simple/full/*test*.csv \
  --UNK "<unk>" \
  --MASK "<mask>" \
  --UNK_ID 3 \
  --MASK_ID 250001


python3 src/embedding.py \
  --model FacebookAI/xlm-roberta-base \
  --prefix xlm_robert \
  --tokenizer_path data/tokenizer \
  --output_path data/embedding/xlm_roberta/ex2/simple/full \
  --split simple \
  --perturbed no \
  --train data/data_set/ex2/simple/full/*train*.csv \
  --test  data/data_set/ex2/simple/full/*test*.csv \
  --UNK "<unk>" \
  --MASK "<mask>" \
  --UNK_ID 3 \
  --MASK_ID 250001


python3 src/embedding.py \
  --model FacebookAI/xlm-roberta-base \
  --prefix xlm_robert \
  --tokenizer_path data/tokenizer \
  --output_path data/embedding/xlm_roberta/ex2/per_dopo/full \
  --split per_dopo \
  --perturbed no \
  --train data/data_set/ex2/per_dopo/full/*train*.csv \
  --test  data/data_set/ex2/per_dopo/full/*test*.csv \
  --UNK "<unk>" \
  --MASK "<mask>" \
  --UNK_ID 3 \
  --MASK_ID 250001

```

10. Embeddings extraction - UmBERTo


```

python3 src/embedding.py \
  --model Musixmatch/umberto-commoncrawl-cased-v1 \
  --prefix UMBERTO \
  --tokenizer_path data/tokenizer \
  --output_path data/embedding/umberto/ex2/simple/full \
  --split umberto_simple \
  --perturbed no \
  --train data/data_set/ex2/simple/full/*train*.csv \
  --test  data/data_set/ex2/simple/full/*test*.csv \
  --UNK "<unk>" \
  --MASK "<mask>" \
  --UNK_ID 3 \
  --MASK_ID 32004


python3 src/embedding.py \
  --model Musixmatch/umberto-commoncrawl-cased-v1 \
  --prefix UMBERTO \
  --tokenizer_path data/tokenizer \
  --output_path data/embedding/umberto/ex2/per_dopo/full \
  --split umberto_per_dopo \
  --perturbed no \
  --train data/data_set/ex2/per_dopo/full/*train*.csv \
  --test  data/data_set/ex2/per_dopo/full/*test*.csv \
  --UNK "<unk>" \
  --MASK "<mask>" \
  --UNK_ID 3 \
  --MASK_ID 32004


```

5. Decremental Training

```

python3 src/decremental_training.py \
                --dataset_folder data/data_set/ex2/simple/full/ \
                --embeddings_folder data/embedding/bert_ita/ex2/simple/full/ \
                --output_folder data/data_set/ex2/simple/sampled/ \
                --output_emb_folder data/embedding/bert_ita/ex2/simple/sampled/ \
                --csv_pattern "*train*.csv" \
                --emb_pattern "*UNK*train*.pkl" \
                --key UNK


python3 src/decremental_training.py \
                --dataset_folder data/data_set/ex2/simple/full/ \
                --embeddings_folder data/embedding/bert_ita/ex2/simple/full/ \
                --output_folder data/data_set/ex2/simple/sampled/ \
                --output_emb_folder data/embedding/bert_ita/ex2/simple/sampled/ \
                --csv_pattern "*train*.csv" \
                --emb_pattern "*PREP*train*.pkl" \
                --key PREP

python3 src/decremental_training.py \
                --dataset_folder data/data_set/ex2/simple/full/ \
                --embeddings_folder data/embedding/bert_ita/ex2/simple/full/ \
                --output_folder data/data_set/ex2/simple/sampled/ \
                --output_emb_folder data/embedding/bert_ita/ex2/simple/sampled/ \
                --csv_pattern "*train*.csv" \
                --emb_pattern "*CLS*train*.pkl" \
                --key CLS

```


11. Probing Classifier -- BERT ITA 

```

for k in UNK CLS PREP
do
  python3 src/LR.py \
    --seed 42 \
    --X_train data/embedding/bert_ita/ex2/simple/full/*${k}*train*.pkl \
    --X_test  data/embedding/bert_ita/ex2/simple/full/*${k}*test*.pkl \
    --y_train data/data_set/ex2/simple/full/*train*.csv \
    --y_test  data/data_set/ex2/simple/full/*test*.csv \
    -o data/outputs/ \
    -k $k \
    -m bert_ita \
    -s bert_ita/ex2/simple \
    -e ex2 \
    --contextual \
    --clf_name logistic_regression \
    --solver lbfgs \
    --label meaning
  done



for k in UNK CLS PREP
do
  python3 src/LR.py \
    --seed 42 \
    --X_train data/embedding/bert_ita/ex2/per_dopo/full/*${k}*train*.pkl \
    --X_test  data/embedding/bert_ita/ex2/per_dopo/full/*${k}*test*.pkl \
    --y_train data/data_set/ex2/per_dopo/full/*train*.csv \
    --y_test  data/data_set/ex2/per_dopo/full/*test*.csv \
    -o data/outputs/ \
    -k $k \
    -m bert_ita \
    -s bert_ita/ex2/per_dopo \
    -e ex2 \
    --contextual \
    --clf_name logistic_regression \
    --solver lbfgs \
    --label meaning
  done

```

12. Probing Classifier -- BERT ENG 

```

for k in UNK CLS PREP
do
  python3 src/LR.py \
    --seed 42 \
    --X_train data/embedding/bert_en/ex2_scivetti/simple/*${k}*train*.pkl \
    --X_test  data/embedding/bert_en/ex2_scivetti/simple/*${k}*test*.pkl \
    --y_train data/data_set/ex2_scivetti/simple/full/*train*.csv \
    --y_test  data/data_set/ex2_scivetti/simple/full/*test*.csv \
    -o data/outputs/ \
    -k $k \
    -m bert_en \
    -s bert_en/ex2_scivetti/simple \
    -e ex2_scivetti \
    --contextual \
    --clf_name logistic_regression \
    --solver lbfgs \
    --label meaning
  done

```


13. Probing Classifier -- m-BERT ITA

```

for k in UNK CLS PREP
do
  python3 src/LR.py \
    --seed 42 \
    --X_train data/embedding/bert_multilingual/ex2/simple/full/*${k}*train*.pkl \
    --X_test  data/embedding/bert_multilingual/ex2/simple/full/*${k}*test*.pkl \
    --y_train data/data_set/ex2/simple/full/*train*.csv \
    --y_test  data/data_set/ex2/simple/full/*test*.csv \
    -o data/outputs/ \
    -k $k \
    -m bert_multilingual \
    -s bert_multilingual/ex2/simple \
    -e ex2 \
    --contextual \
    --clf_name logistic_regression \
    --solver lbfgs \
    --label meaning
  done


for k in UNK PREP CLS
do
  python3 src/LR.py \
    --seed 42 \
    --X_train data/embedding/bert_multilingual/ex2/per_dopo/full/*${k}*train*.pkl \
    --X_test  data/embedding/bert_multilingual/ex2/per_dopo/full/*${k}*test*.pkl \
    --y_train data/data_set/ex2/per_dopo/full/*train*.csv \
    --y_test  data/data_set/ex2/per_dopo/full/*test*.csv \
    -o data/outputs/ \
    -k $k \
    -m bert_multilingual \
    -s bert_multilingual/ex2/per_dopo \
    -e ex2 \
    --contextual \
    --clf_name logistic_regression \
    --solver lbfgs \
    --label meaning
  done


```

14. Probing Classifier -- m-BERT ENG

```

for k in UNK CLS PREP
do
  python3 src/LR.py \
    --seed 42 \
    --X_train data/embedding/bert_multilingual/ex2_scivetti/simple/*${k}*train*.pkl \
    --X_test  data/embedding/bert_multilingual/ex2_scivetti/simple/*${k}*test*.pkl \
    --y_train data/data_set/ex2_scivetti/simple/full/*train*.csv \
    --y_test  data/data_set/ex2_scivetti/simple/full/*test*.csv \
    -o data/outputs/ \
    -k $k \
    -m bert_multilingual \
    -s bert_multilingual/ex2_scivetti/simple \
    -e ex2_scivetti \
    --contextual \
    --clf_name logistic_regression \
    --solver lbfgs \
    --label meaning
  done

 ``` 

15. Probing Classifier -- UmBERTo

```



for k in UNK CLS PREP
do
  python3 src/LR.py \
    --seed 42 \
    --X_train data/embedding/umberto/ex2/simple/full/*${k}*train*.pkl \
    --X_test  data/embedding/umberto/ex2/simple/full/*${k}*test*.pkl \
    --y_train data/data_set/ex2/simple/full/*train*.csv \
    --y_test  data/data_set/ex2/simple/full/*test*.csv \
    -o data/outputs/ \
    -k $k \
    -m umberto \
    -s umberto/ex2/simple \
    -e ex2 \
    --contextual \
    --clf_name logistic_regression \
    --solver lbfgs \
    --label meaning
  done


for k in UNK CLS PREP
do
  python3 src/LR.py \
    --seed 42 \
    --X_train data/embedding/umberto/ex2/per_dopo/full/*${k}*train*.pkl \
    --X_test  data/embedding/umberto/ex2/per_dopo/full/*${k}*test*.pkl \
    --y_train data/data_set/ex2/per_dopo/full/*train*.csv \
    --y_test  data/data_set/ex2/per_dopo/full/*test*.csv \
    -o data/outputs/ \
    -k $k \
    -m umberto \
    -s umberto/ex2/per_dopo \
    -e ex2 \
    --contextual \
    --clf_name logistic_regression \
    --solver lbfgs \
    --label meaning
  done

```



16. Probing Classifier -- RoBERTa ITA


```


for k in UNK CLS PREP
do
  python3 src/LR.py \
    --seed 42 \
    --X_train data/embedding/xlm_roberta/ex2/simple/full/*${k}*train*.pkl \
    --X_test  data/embedding/xlm_roberta/ex2/simple/full/*${k}*test*.pkl \
    --y_train data/data_set/ex2/simple/full/*train*.csv \
    --y_test  data/data_set/ex2/simple/full/*test*.csv \
    -o data/outputs/ \
    -k $k \
    -m xlm_roberta \
    -s xlm_roberta/ex2/simple \
    -e ex2 \
    --contextual \
    --clf_name logistic_regression \
    --solver lbfgs \
    --label meaning
  done



for k in UNK CLS PREP
do
  python3 src/LR.py \
    --seed 42 \
    --X_train data/embedding/xlm_roberta/ex2/per_dopo/full/*${k}*train*.pkl \
    --X_test  data/embedding/xlm_roberta/ex2/per_dopo/full/*${k}*test*.pkl \
    --y_train data/data_set/ex2/per_dopo/full/*train*.csv \
    --y_test  data/data_set/ex2/per_dopo/full/*test*.csv \
    -o data/outputs/ \
    -k $k \
    -m xlm_roberta \
    -s xlm_roberta/ex2/per_dopo \
    -e ex2 \
    --contextual \
    --clf_name logistic_regression \
    --solver lbfgs \
    --label meaning
  done

```



17. Probing Classifier -- RoBERTa ENG


```


for k in UNK CLS PREP
do
  python3 src/LR.py \
    --seed 42 \
    --X_train data/embedding/xlm_roberta/ex2_scivetti/simple/*${k}*train*.pkl \
    --X_test  data/embedding/xlm_roberta/ex2_scivetti/simple/*${k}*test*.pkl \
    --y_train data/data_set/ex2_scivetti/simple/full/*train*.csv \
    --y_test  data/data_set/ex2_scivetti/simple/full/*test*.csv \
    -o data/outputs/ \
    -k $k \
    -m xlm_roberta \
    -s xlm_roberta/ex2_scivetti/simple \
    -e ex2_scivetti \
    --contextual \
    --clf_name logistic_regression \
    --solver lbfgs \
    --label meaning
  done

```


18. Probing Classifier -- baseline fasttext ITA

```

python3 src/LR.py \
  --seed 42 \
  --X_train data/embedding/fasttext/ex2/simple/costrN/*train*.tsv \
  --X_test  data/embedding/fasttext/ex2/simple/costrN/*test*.tsv \
  --y_train data/data_set/ex2/simple/full/*train*.csv \
  --y_test data/data_set/ex2/simple/full/*test*.csv \
  -o data/outputs/ \
  -m fasttext \
  -s fasttext/ex2/simple/costrN \
  -e ex2 \
  --clf_name logistic_regression \
  --solver lbfgs \
  --label meaning


python3 src/LR.py \
  --seed 42 \
  --X_train data/embedding/fasttext/ex2/simple/noun/*train*.tsv \
  --X_test  data/embedding/fasttext/ex2/simple/noun/*test*.tsv \
  --y_train data/data_set/ex2/simple/full/*train*.csv \
  --y_test data/data_set/ex2/simple/full/*test*.csv \
  -o data/outputs/ \
  -m fasttext \
  -s fasttext/ex2/simple/noun \
  -e ex2 \
  --clf_name logistic_regression \
  --solver lbfgs \
  --label meaning


python3 src/LR.py \
  --seed 42 \
  --X_train data/embedding/fasttext/ex2/per_dopo/costrN/*train*.tsv \
  --X_test  data/embedding/fasttext/ex2/per_dopo/costrN/*test*.tsv \
  --y_train data/data_set/ex2/per_dopo/full/*train*.csv \
  --y_test data/data_set/ex2/per_dopo/full/*test*.csv \
  -o data/outputs/ \
  -m fasttext \
  -s fasttext/ex2/per_dopo/costrN \
  -e ex2 \
  --clf_name logistic_regression \
  --solver lbfgs \
  --label meaning


python3 src/LR.py \
  --seed 42 \
  --X_train data/embedding/fasttext/ex2/per_dopo/noun/*train*.tsv \
  --X_test  data/embedding/fasttext/ex2/per_dopo/noun/*test*.tsv \
  --y_train data/data_set/ex2/per_dopo/full/*train*.csv \
  --y_test data/data_set/ex2/per_dopo/full/*test*.csv \
  -o data/outputs/ \
  -m fasttext \
  -s fasttext/ex2/per_dopo/noun \
  -e ex2 \
  --clf_name logistic_regression \
  --solver lbfgs \
  --label meaning

```

19. Probing Classifier -- word2vec ITA

```

python3 src/LR.py \
  --seed 42 \
  --X_train data/embedding/itwac/ex2/simple/costrN/*train*.tsv \
  --X_test  data/embedding/itwac/ex2/simple/costrN/*test*.tsv \
  --y_train data/data_set/ex2/simple/full/*train*.csv \
  --y_test data/data_set/ex2/simple/full/*test*.csv \
  -o data/outputs/ \
  -m itwac \
  -s itwac/ex2/simple/costrN \
  -e ex2 \
  --clf_name logistic_regression \
  --solver lbfgs \
  --label meaning


python3 src/LR.py \
  --seed 42 \
  --X_train data/embedding/itwac/ex2/simple/noun/*train*.tsv \
  --X_test  data/embedding/itwac/ex2/simple/noun/*test*.tsv \
  --y_train data/data_set/ex2/simple/full/*train*.csv \
  --y_test data/data_set/ex2/simple/full/*test*.csv \
  -o data/outputs/ \
  -m itwac \
  -s itwac/ex2/simple/noun \
  -e ex2 \
  --clf_name logistic_regression \
  --solver lbfgs \
  --label meaning


python3 src/LR.py \
  --seed 42 \
  --X_train data/embedding/itwac/ex2/per_dopo/costrN/*train*.tsv \
  --X_test  data/embedding/itwac/ex2/per_dopo/costrN/*test*.tsv \
  --y_train data/data_set/ex2/per_dopo/full/*train*.csv \
  --y_test data/data_set/ex2/per_dopo/full/*test*.csv \
  -o data/outputs/ \
  -m itwac \
  -s itwac/ex2/per_dopo/costrN \
  -e ex2 \
  --clf_name logistic_regression \
  --solver lbfgs \
  --label meaning


python3 src/LR.py \
  --seed 42 \
  --X_train data/embedding/itwac/ex2/per_dopo/noun/*train*.tsv \
  --X_test  data/embedding/itwac/ex2/per_dopo/noun/*test*.tsv \
  --y_train data/data_set/ex2/per_dopo/full/*train*.csv \
  --y_test data/data_set/ex2/per_dopo/full/*test*.csv \
  -o data/outputs/ \
  -m itwac \
  -s itwac/ex2/per_dopo/noun \
  -e ex2 \
  --clf_name logistic_regression \
  --solver lbfgs \
  --label meaning

```

20. Probing Classifier -- Glove ENG

```

python3 src/LR.py \
  --seed 42 \
  --X_train data/embedding/glove/ex2_scivetti/simple/*train*.tsv \
  --X_test  data/embedding/glove/ex2_scivetti/simple/*test*.tsv \
  --y_train data/data_set/ex2_scivetti/simple/full/*train*.csv \
  --y_test data/data_set/ex2_scivetti/simple/full/*test*.csv \
  -o data/outputs/ \
  -m glove \
  -s glove/ex2_scivetti/simple \
  -e ex2_scivetti \
  --clf_name logistic_regression \
  --solver lbfgs \
  --label meaning

```

21. Probing Classifier -- fasttext ENG

```

python3 src/LR.py \
  --seed 42 \
  --X_train data/embedding/fasttext/ex2_scivetti/simple/*train*.tsv \
  --X_test  data/embedding/fasttext/ex2_scivetti/simple/*test*.tsv \
  --y_train data/data_set/ex2_scivetti/simple/full/*train*.csv \
  --y_test data/data_set/ex2_scivetti/simple/full/*test*.csv \
  -o data/outputs/ \
  -m fasttext \
  -s fasttext/ex2_scivetti/simple \
  -e ex2_scivetti \
  --clf_name logistic_regression \
  --solver lbfgs \
  --label meaning

```


22. Probing Classifier -- decremental

```

for d in 240 120 60
do

  for k in UNK CLS PREP
  do
    python3 src/LR.py \
      --seed 42 \
      --X_train data/embedding/bert_ita/ex2/simple/sampled/*train*${d}_${k}.pkl \
      --X_test  data/embedding/bert_ita/ex2/simple/full/*${k}*test*.pkl \
      --y_train data/data_set/ex2/simple/sampled/*train*${d}.csv \
      --y_test  data/data_set/ex2/simple/full/*test*.csv \
      -o data/outputs/ \
      -k $k \
      -m bert_ita \
      -s bert_ita/ex2/simple \
      -e ex1_${d} \
      --decremental \
      --contextual \
      --clf_name logistic_regression \
	  --solver lbfgs \
	  --label meaning
    done
done

```

23. Probing Classifier -- control

```

for k in UNK CLS PREP
do
  python3 src/LR.py \
    --seed 42 \
    --X_train data/embedding/bert_ita/ex2/simple/full/*${k}*train*.pkl \
    --X_test  data/embedding/bert_ita/ex2/simple/full/*${k}*test*.pkl \
    --y_train data/data_set/ex2/simple/control/*train*.csv \
    --y_test  data/data_set/ex2/simple/control/*test*.csv \
    -o data/outputs/ \
    -k $k \
    -m bert_ita \
    -s bert_ita/ex2/simple/control \
    -e ex2 \
    --contextual \
    --clf_name logistic_regression \
	--solver lbfgs \
	--label meaning
  done
