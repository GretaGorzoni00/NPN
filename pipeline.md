1. script `campionamento.py` su file in `dataset_construction` che produce versione filtrata in `source`
2. aggiunta a versione filtrata in `source` colonna `pre_lemma` e `other_cxn` nei file `*distractor`
3. TODO: controllare `dataset_construction` e depositare su osf, zenodo, qualcosa
4. Creazione train/test split (cartella `data_set`):
   - condizione "simple" per esperimento 1 >
   - condizione "other" per esperimento 1 >
   - condizione "pseudo" per esperimento 1

5. perturbazioni: DA FARE TODO ANCORA
   `python create_perturbations.py [filename_1]... [filename_n]`

6. TODO FASTTEXT

7. TODO CONTROL CLASSIFIER

8. TODO SIMILARITÀ SEMANTICA

9. decremento training TODO RIMUOVERE CREAZIONE DATA SET UGAULE PER OGNI KEY SOVRASCRITTA NEL CICLO FOR. CICLO FOR SOLO PER TRAINING EMBEDDING IL DATA SET TRAINIGN SAMPLED PUÒ RESTARE UNICO
   `python src/decremental_training.py`
   `python src/ex_1_decremental.py`

10. estrazione embedding 
   `python src/embedding.py`
   `python src/ex_1_embedding.py`

11. addestramento e valutazione classificatore lineare
   `python src/LR.py`
   `python src/experiment_1.py`

