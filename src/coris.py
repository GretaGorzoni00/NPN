import re
import pandas as pd

# === CONFIGURA QUI ===
input_path = "CORIS_NPN_17170.txt"   # path del file .txt da elaborare
output_path = "CORIS_nuovo.csv"  # output ordinato
distrattori_path = "distrattori.csv"  # output per distrattori

# === PREPOSIZIONI DA ESCLUDERE ===
preposizioni_escludere = {"di", "a", "da", "in", "con", "su", "per", "tra", "fra"}

# === ESTRAZIONE ===
pattern_npn = re.compile(r"<(\w+)\s+(\w+)\s+(\w+)\s>")
risultati = []
distrattori = []  # Lista per i distrattori

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

            # Verifica se la costruzione è preceduta da una preposizione da escludere
            precedente_costruzione = linea_pulita[:idx].strip()  # Parte prima della costruzione
            parole_precedenti = precedente_costruzione.split()
            if parole_precedenti and parole_precedenti[-1].lower() in preposizioni_escludere:
                distrattori.append({
                    "costruzione": costruzione,
                    "preposizione": prep,
                    "parola": parola_base,
                    "contesto": contesto
                })
            else:
                risultati.append({
                    "costruzione": costruzione,
                    "preposizione": prep,
                    "parola": parola_base,
                    "contesto": contesto
                })

        else:
            # Aggiungi la riga ignorata
            distrattori.append({
                "costruzione": "N/A",  # Costruzione non presente
                "preposizione": "N/A",  # Preposizione non presente
                "parola": "N/A",  # Parola non presente
                "contesto": linea.strip()  # Intera riga come contesto
            })

# === ORDINAMENTO ===
df = pd.DataFrame(risultati)
df.sort_values(by=["preposizione", "parola"], inplace=True)

# === ORDINAMENTO DISTRATTORI ===
df_distrattori = pd.DataFrame(distrattori)
df_distrattori.sort_values(by=["preposizione", "parola"], inplace=True)

# === SALVATAGGIO ===
df.to_csv(output_path, index=False)
df_distrattori.to_csv(distrattori_path, index=False)

# === STAMPA RIGHE IGNORATE ===
print(f"{len(risultati)} costruzioni NPN salvate in '{output_path}'")
print(f"{len(distrattori)} righe salvate come distrattori in '{distrattori_path}'")
