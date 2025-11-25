1. script `campionamento.py` su file in `dataset_construction` che produce versione filtrata in `source`
2. aggiunta a versione filtrata in `source` colonna `pre_lemma` e `other_cxn` nei file `*distractor`
3. TODO: controllare `dataset_construction` e depositare su osf, zenodo, qualcosa
4. Creazione train/test split (cartella `data_set`):
   - condizione "simple" per esperimento 1 >
   - condizione "other" per esperimento 1
   - condizione "pseudo" per esperimento 1

5. perturbazioni:
   `python create_perturbations.py [filename_1]... [filename_n]`