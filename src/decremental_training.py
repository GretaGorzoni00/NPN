import pandas as pd
import os
import glob
import pickle
import argparse
import random
import numpy as np 

random.seed(42)
np.random.seed(42)

# ===== FUNZIONE STRATIFICATA =====
def stratified_indices(df, n):
    """
    Ritorna gli indici stratificati per construction e preposition
    4 combinazioni: yes-su, yes-a, no-su, no-a
    """
    n_per_comb = n // 4
    indices = []
    for construction in ["yes", "no"]:
        for prep in ["su", "a"]:
            subset = df[(df["construction"] == construction) & (df["preposition"] == prep)]
            n_to_sample = min(len(subset), n_per_comb)
            if n_to_sample == 0:
                # se non ci sono righe, salta questa combinazione
                continue
            sampled_idx = subset.sample(n=n_to_sample, random_state=42).index.tolist()
            indices.extend(sampled_idx)
    return indices

# ===== FUNZIONE PRINCIPALE =====
def main(dataset_folder, embeddings_folder, output_folder, output_emb_folder, train_sizes, csv_pattern, emb_pattern, key):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_emb_folder, exist_ok=True)

    csv_files = sorted(glob.glob(dataset_folder+csv_pattern))
    emb_files_all = sorted(glob.glob(embeddings_folder+emb_pattern))
    print(emb_files_all)
    print(csv_files)

    for csv_file in csv_files:
        df = pd.read_csv(csv_file, sep=";")
        base_name = os.path.splitext(os.path.basename(csv_file))[0]

        # Trova embedding corrispondente
        

        emb_file = [f for f in emb_files_all if base_name in f]
        if not emb_file:
            print(f"Embedding file non trovato per {csv_file}, salto...")
            continue
        
        
        emb_file = emb_file[0]
        
        
        if emb_file.endswith(".pkl"):
            # Carica embedding (lista di dizionari)
            with open(emb_file, "rb") as f:
                emb_list = pickle.load(f)

        elif emb_file.endswith(".csv") or emb_file.endswith(".txt"):
            # Carica embedding da CSV o txt (righe con vettori separati da spazio o tab)
            df_emb = pd.read_csv(emb_file, header=None, sep=r"\s+")
            emb_list = df_emb.values.tolist()  # converte ogni riga in lista

        else:
            raise ValueError(f"Formato file embedding non supportato: {emb_file}")

        # Controllo numero righe
        if len(df) != len(emb_list):
            raise ValueError(f"Numero di righe mismatch tra CSV ({len(df)}) e embedding ({len(emb_list)}) in {base_name}")

        # Campionamento per le diverse dimensioni
        for n in train_sizes:
            indices = stratified_indices(df, n)

            # CSV campionato
            sampled_df = df.loc[indices].reset_index(drop=True)
            out_csv = os.path.join(output_folder, f"{base_name}_{n}.csv")
            sampled_df.to_csv(out_csv, index=False, sep=";")

            # Embedding campionato
            sampled_emb = [emb_list[i] for i in indices]
            out_emb = os.path.join(output_emb_folder, f"{base_name}_{n}_{key}.pkl")
            with open(out_emb, "wb") as f:
                pickle.dump(sampled_emb, f)

            print(f"Salvati: {out_csv} e {out_emb}")


      


# ===== RIGA DI COMANDO =====
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Campionamento stratificato CSV e embeddings")
    parser.add_argument("--dataset_folder", type=str, required=True, help="Cartella dei CSV")
    parser.add_argument("--embeddings_folder", type=str, required=True, help="Cartella degli embeddings")
    parser.add_argument("--output_folder", type=str, required=True, help="Cartella di output CSV")
    parser.add_argument("--output_emb_folder", type=str, required=True, help="Cartella di output embeddings")
    parser.add_argument("--train_sizes", type=int, nargs="+", default=[240, 120, 60], help="Dimensioni campionate")
    parser.add_argument("--csv_pattern", type=str)
    parser.add_argument("--emb_pattern", type=str)
    parser.add_argument("--key", type=str)
    args = parser.parse_args()

    main(args.dataset_folder, args.embeddings_folder, args.output_folder, args.output_emb_folder, args.train_sizes, args.csv_pattern, args.emb_pattern, args.key)
