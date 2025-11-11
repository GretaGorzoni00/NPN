import pandas as pd
from collections import Counter

# Percorso al file filtrato che contiene la colonna pred_BERT
input_csv = "data/output/MASK/A_construction_filtrato_pred.csv"  # modifica con il tuo file

# Carica il CSV
df = pd.read_csv(input_csv, sep=";")

# Filtro per meaning rilevanti
meaning_values = ["juxtaposition/contact", "succession/iteration/distributivity"]

total_correct = 0
total_rows = 0

for meaning in meaning_values:
    subset = df[df["meaning"] == meaning]
    total = len(subset)
    if total == 0:
        print(f"Nessuna riga per meaning {meaning}")
        continue
    
    # Controlla identità tra preposition e pred_BERT
    correct = (subset["preposition"] == subset["pred_BERT"]).sum()
    percentage = correct / total * 100

    print(f"Meaning: {meaning}")
    print(f"Totale occorrenze: {total}")
    print(f"Predizioni corrette: {correct}")
    print(f"Percentuale di correttezza: {percentage:.2f}%\n")
    
    # Aggiorna i totali
    total_correct += correct
    total_rows += total

    # Mostra le prime tre predizioni più frequenti tra quelle errate
    wrong_preds = subset[subset["preposition"] != subset["pred_BERT"]]["pred_BERT"]
    if len(wrong_preds) > 0:
        freq_counter = Counter(wrong_preds)
        top3 = freq_counter.most_common(3)
        print(f"Top 3 predizioni errate per '{meaning}':")
        for pred, freq in top3:
            print(f"  {pred}: {freq} occorrenze")
        print("\n")

# Percentuale totale su tutti i subset considerati
if total_rows > 0:
    total_percentage = total_correct / total_rows * 100
    print(f"Percentuale totale di correttezza: {total_percentage:.2f}%")
else:
    print("Nessuna riga valida per il calcolo totale.")
