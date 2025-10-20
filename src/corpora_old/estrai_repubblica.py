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

percorso = "repubblica.xml"
frasi = estrai_token_e_tag(percorso)

count = 0
chiave = 'SES'
for frase in frasi:
    for i in range(len(frase)-2):
        costruzione = frase[i][1] + frase[i+1][1] + frase[i+2][1]
        if costruzione == chiave:
            if frase[i][0] == frase[i+2][0] and frase[i+1][2] == 'E':
                print(frase[i][0] + " " + frase[i+1][0] +  " " + frase[i+2][0])
                print(frase)
                count += 1