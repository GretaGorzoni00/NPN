import re
import pandas as pd

input_path = "data/data_set/CORIS_NPN_17170.txt"
output_path = "data/data_set/CORIS_sentence.csv"

pattern_npn = re.compile(r"<(\w+)\s+(\w+)\s+(\w+)\s>")
risultati = []

with open(input_path, encoding='utf-8') as f:
    for linea in f:
        match = pattern_npn.search(linea)
        if match:
            n1, prep, n2 = match.groups()
            costruzione = f"{n1} {prep} {n2}"
            parola_base = n1.lower() if n1.lower() == n2.lower() else f"{n1}/{n2}"

            # Rimuove < > e spazi inutili tra le parole
            linea_pulita = re.sub(r"<\s*(\w+)\s+(\w+)\s+(\w+)\s*>", r"\1 \2 \3", linea)

            # Trova il contesto centrato attorno alla costruzione
            idx = linea_pulita.find(costruzione)
            if idx != -1:
                inizio = max(0, idx - 80)
                fine = idx + len(costruzione) + 80
                contesto = linea_pulita[inizio:fine].strip()
            else:
                contesto = linea.strip()
                
            risultati.append({
                    "costruzione": costruzione,
                    "preposizione": prep,
                    "parola": parola_base,
                    "contesto": contesto
                })
            
df = pd.DataFrame(risultati)
df.sort_values(by=["preposizione", "parola"], inplace=True)
df.to_csv(output_path, index=False)