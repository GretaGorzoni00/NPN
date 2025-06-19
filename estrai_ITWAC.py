import tqdm

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

    for percorso in percorsi:

        frasi = estrai_token_e_tag_gen(percorso)

        count = 0
        chiave = 'NOUNPRENOUN'
        for frase in tqdm.tqdm(frasi):
            for i in range(len(frase)-2):
                costruzione = frase[i][1] + frase[i+1][1] + frase[i+2][1]
                if costruzione == chiave:
                    if frase[i][0] == frase[i+2][0]:
                        # print(frase[i][0] + " " + frase[i+1][0] +  " " + frase[i+2][0])
                        # print(frase)
                        count += 1
        print(percorso, count)
