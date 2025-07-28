creazione venv

`python3 -m venv .venv`

`source .venv/bin/activate`

dataset

estrazione CORIS
file input generale `data/data_set/CORIS_NPN_17170.txt`

eseguire script `src/coris.py`che suddivide le costruzioni NPN in due file: `data/data_set/CORIS_nuovo.csv`, `data/data_set/distrattori.csv`.

Il file `data/data_set/CORIS_annotato.xlsx` viene annotato a partire da `data/data_set/CORIS_nuovo.csv`.

!!!! filtraggio dati 
!!!! creazione data set nel formato input script
!!!! divisione training test 

Nello script `embedding.py`
Caricare il modello (ModernBERT)

carica il modello e il tokenizer da HuggingFace
salva vocabolario e tokenizer in `data/tokenizer`
legge data set in input 
!!! al momento file artificiale `data/data.tsv`
estrae embedding
salva file pkl in `data/embeddings/embedding_layers_UNK.pkl`

!!! aggiungere qui file .csv - ogni livello una colonna, le  dimensioni (768) comma separeated





