creazione venv

`python3 -m venv .venv`

`source .venv/bin/activate`

dataset

estrazione CORIS
file input generale `data/data_set/CORIS_NPN_17170.txt`

eseguire script `src/coris.py`che suddivide le costruzioni NPN in due file: `data/data_set/CORIS_nuovo.csv`, `data/data_set/distrattori.csv`.

Il file `data/data_set/CORIS_annotato.xlsx` viene annotato a partire da `data/data_set/CORIS_nuovo.csv`.

