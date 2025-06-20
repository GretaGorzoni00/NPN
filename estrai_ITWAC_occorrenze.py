import tqdm
import pandas as pd
import os

def estrai_token_e_tag_gen(path):

    with open(path, encoding='ISO-8859-15') as f:
        for riga in tqdm.tqdm(f):
            riga = riga.strip()
            if riga.startswith("<"):
                if riga == "<s>":
                    frase_corrente = []
                if riga == "</s>":
                    yield frase_corrente
            else:
                colonne = riga.split('\t')
                if len(colonne) >= 2:
                    token = colonne[0]
                    pos = colonne[1]
                    frase_corrente.append((token,pos))


def estrai_token_e_tag(path):
    frasi = []
    with open(path, encoding='ISO-8859-15') as f:
        for riga in f:
            riga = riga.strip()
            if riga.startswith("<"):
                if riga == "<s>":
                    frase_corrente = []
                if riga == "</s>":
                    frasi.append(frase_corrente)
            else:
                colonne = riga.split('\t')
                if len(colonne) >= 2:
                    token = colonne[0]
                    pos = colonne[1]
                    frase_corrente.append((token,pos))
    return frasi


if __name__ == "__main__":
    import sys

    # percorso = "ITWAC_5000.xml"
    percorsi = sys.argv[1:]

    for percorso in percorsi: #perchè il corpus è diviso in più file?

        frasi = estrai_token_e_tag_gen(percorso)

        preposizioni ={}
        count = 0
        chiave = 'NOUNPRENOUN'
        for frase in tqdm.tqdm(frasi):
            for i in range(len(frase)-2):
                costruzione = frase[i][1] + frase[i+1][1] + frase[i+2][1]
                if costruzione == chiave:
                    if frase[i][0] == frase[i+2][0]:
                        # print(frase[i][0] + " " + frase[i+1][0] +  " " + frase[i+2][0])
                        # print(frase)
                        preposizione = frase[i+1][0]
                        nome = frase[i][0]
                        
                            # Se la preposizione non è ancora nel dizionario
                        if preposizione not in preposizioni:
                            preposizioni[preposizione] = {'occorrenze': 0}
                        
                                            # Incrementa occorrenze totali
                        preposizioni[preposizione]['occorrenze'] += 1

                        # Incrementa contatore del nome
                        if nome in preposizioni[preposizione]:
                            preposizioni[preposizione][nome] += 1
                        else:
                            preposizioni[preposizione][nome] = 1
                        
                        count += 1
        print(percorso, count)
        
        tutti_nomi = set()
        for info in preposizioni.values():
            for k in info:
                if k != 'occorrenze':
                    tutti_nomi.add(k)
        tutti_nomi = sorted(tutti_nomi)

        data = []
        for preposizione, counts in preposizioni.items():
            row = {
                'Preposizione': preposizione,
                'Occorrenze SES': counts['occorrenze'],
            }
            for nome in tutti_nomi:
                row[nome] = counts.get(nome, 0)
            data.append(row)

        df = pd.DataFrame(data)
        nome_file = os.path.basename(percorso)
        output_name = 'occorrenze_' + nome_file + '.csv'
        df.to_csv(output_name, index=False)

        print("File CSV creato con successo!")    
