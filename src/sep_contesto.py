#script con regex per separare contesto prec e successivo
#regex virgolette
#regex elimina spazi prima di punteggiatura (escluse virgolette)

import re

def processa_testo(testo):
    # 1) Rimuovere gli underscore
    testo = testo.replace('_', '')

    # 2) Eliminare solo spazi (non tab) prima della punteggiatura
    testo = re.sub(r' +([,.;:!?])', r'\1', testo)

    # 3) Eliminare solo spazi (non tab) prima e dopo apostrofo
    # a) Rimuovi spazio prima e dopo apostrofo
    testo = re.sub(r" *' *", "'", testo)

    # b) Mantieni l'apostrofo dopo "po" (es. "po' ")
    testo = re.sub(r"(\bpo)'", r"\1 '", testo)

    # 4) Gestione speciale per parentesi e virgolette
    # Rimuove spazio dopo la parentesi di apertura e prima della parentesi di chiusura (non tab)
    testo = re.sub(r'\( +', '(', testo)  # dopo "("
    testo = re.sub(r' +\)', ')', testo)  # prima di ")"

    # Gestione virgolette doppie "" (non tab)
    testo = re.sub(r'" +', '"', testo)  # dopo virgolette aperte
    testo = re.sub(r' +"', '"', testo)  # prima virgolette chiuse

    # Gestione virgolette angolari «» (non tab)
    testo = re.sub(r'« +', '«', testo)  # dopo «
    testo = re.sub(r' +»', '»', testo)  # prima »

    return testo


def processa_file(input_file):
    # Legge il file di input
    with open(input_file, "r", encoding="utf-8") as f:
        testo = f.read()

    # Applica la trasformazione
    testo_processato = processa_testo(testo)

    # Sovrascrive il file senza toccare i tab
    with open(input_file, "w", encoding="utf-8") as f:
        f.write(testo_processato)


# Percorso del file TSV
input_file = "data/data_set/concordance_01.tsv"

# Esegui il processamento
processa_file(input_file)

print(f"File '{input_file}' modificato con successo.")



