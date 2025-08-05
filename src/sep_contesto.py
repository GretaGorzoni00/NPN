#script con regex per separare contesto prec e successivo
#regex virgolette
#regex elimina spazi prima di punteggiatura (escluse virgolette)

import re

def processa_testo(testo):
    
    testo = testo.replace('_', '')

    
    testo = re.sub(r' +([,.;:!?])', r'\1', testo)

   
    testo = re.sub(r" *' *", "'", testo)

   
    testo = re.sub(r"(\bpo)'", r"\1 '", testo)


    testo = re.sub(r'\( +', '(', testo)  # dopo "("
    testo = re.sub(r' +\)', ')', testo)  # prima di ")"

    testo = re.sub(r'" +', '"', testo)  # dopo virgolette aperte
    testo = re.sub(r' +"', '"', testo)  # prima virgolette chiuse

    testo = re.sub(r'« +', '«', testo)  # dopo «
    testo = re.sub(r' +»', '»', testo)  # prima »

    return testo


def processa_file(input_file):

    with open(input_file, "r", encoding="utf-8") as f:
        testo = f.read()


    testo_processato = processa_testo(testo)

    with open(input_file, "w", encoding="utf-8") as f:
        f.write(testo_processato)


input_file = "data/estrazione_coris/concordance_03.tsv"


processa_file(input_file)

print(f"File '{input_file}' modificato con successo.")



