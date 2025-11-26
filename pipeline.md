1. la cartella `./data/dataset_construction` contiene la versione completa del dataset annotato, composto di :

| n. instances | file                |
| ------------ | ------------------- |
| 714          | A_construction.csv  |
| 802          | A_distractor.csv    |
| 567          | SU_construction.csv |
| 185          | SU_distractor.csv   |

TODO: descrivere formato file e aggiungere header

2. AGREEMENT

TODO: controllare `dataset_construction` e depositare su osf, zenodo, qualcosa

3. script `campionamento.py` su file in `./data/dataset_construction` che produce versione filtrata in `./data/source`

How to run:
`python src/campionamento.py`

TODO: aggiungere header all'output


4. aggiunta a versione filtrata in `source` colonna `pre_lemma` e `other_cxn` nei file `*distractor`

5. Creazione train/test split (cartella `data_set`):
	`python src/split_ex1.py `
   - condizione "simple" per esperimento 1 >
   - condizione "other" per esperimento 1
   - condizione "pseudo" per esperimento 1



6. perturbazioni: DA FARE TODO ANCORA
   `python create_perturbations.py [filename_1]... [filename_n]`

7. TODO FASTTEXT

8. TODO CONTROL CLASSIFIER

9. TODO SIMILARITÀ SEMANTICA NECESSARI FASTTEXT

10. decremento training TODO RIMUOVERE CREAZIONE DATA SET UGAULE PER OGNI KEY SOVRASCRITTA NEL CICLO FOR. CICLO FOR SOLO PER TRAINING EMBEDDING IL DATA SET TRAINIGN SAMPLED PUÒ RESTARE UNICO
   `python src/decremental_training.py`
   `python src/ex_1_decremental.py`

11. estrazione embedding
   `python src/embedding.py`
   `python src/ex_1_embedding.py`

12. addestramento e valutazione classificatore lineare
   `python src/LR.py`
   `python src/experiment_1.py`

