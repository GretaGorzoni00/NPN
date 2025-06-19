import pandas as pd

def estrai_token_e_tag(path):
    frasi = []
    with open(path, encoding='utf-8') as f:
        for riga in f:
            riga = riga.strip()
            if riga.startswith("<"):
                if riga == "<s>":
                    frase_corrente = []
                if riga == "</s>":
                    frasi.append(frase_corrente)
            else:
                colonne = riga.split('\t')
                if len(colonne) >= 4:
                    token = colonne[1]
                    pos = colonne[3]
                    pos_spec = colonne[4]
                    frase_corrente.append((token,pos,pos_spec))
    return frasi

if __name__ == "__main__":
    import sys

    percorso = sys.argv[1]
    frasi = estrai_token_e_tag(percorso)

    preposizioni ={}
    count = 0
    chiave = 'SES'
    for frase in frasi:
        for i in range(len(frase)-2):
            costruzione = frase[i][1] + frase[i+1][1] + frase[i+2][1]
            if costruzione == chiave:
                if frase[i][0] == frase[i+2][0] and frase[i+1][2] == 'E':
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

                    # Debug opzionale
                    print(frase[i][0] + " " + frase[i+1][0] + " " + frase[i+2][0])
                    print(frase)
                    count += 1

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
    df.to_csv('occorrenze_preposizioni.csv', index=False)

    print("File CSV creato con successo!")