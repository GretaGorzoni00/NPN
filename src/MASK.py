from transformers import AutoTokenizer, AutoModelForMaskedLM
import pandas as pd
import torch
import os
import pickle

def process_file(model_id, input_csv, output_path, prefix="BERT"):
    # Carica il modello e tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForMaskedLM.from_pretrained(model_id, output_hidden_states=True)
    model.eval()

    df = pd.read_csv(input_csv, sep=";")
    predicted_tokens = []
    mask_embeddings = []

    for _, row in df.iterrows():
        lemma1, prep, lemma2 = row["costr"].split(" ")
        # Ricrea la frase con [MASK] al posto della preposizione
        sentence = f"{row['context_pre']} {lemma1} [MASK] {lemma2} {row['context_post']}"
        inputs = tokenizer(sentence, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            mask_index = (inputs["input_ids"][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0].item()
            predicted_id = outputs.logits[0, mask_index].argmax().item()
            predicted_token = tokenizer.decode(predicted_id).strip()
            predicted_tokens.append(predicted_token)

            # Salva embedding della maschera per tutti i layer
            layer_embeddings = [outputs.hidden_states[layer][0, mask_index, :].numpy() for layer in range(1, 13)]
            mask_embeddings.append(layer_embeddings)

    # Crea una copia del dataframe con la nuova colonna
    df_copy = df.copy()
    df_copy["pred_BERT"] = predicted_tokens

    # Salvataggio CSV duplicato
    os.makedirs(output_path, exist_ok=True)
    base_name = os.path.basename(input_csv).replace(".csv", "_pred.csv")
    csv_path = os.path.join(output_path, base_name)
    df_copy.to_csv(csv_path, sep=";", index=False)
    print(f"File duplicato salvato con colonna pred_BERT: {csv_path}")

    # Salvataggio embeddings in PKL (facoltativo)
    pkl_path = os.path.join(output_path, os.path.basename(input_csv).replace(".csv", "_mask_embeddings.pkl"))
    with open(pkl_path, "wb") as f:
        pickle.dump(mask_embeddings, f)
    print(f"Embeddings salvati: {pkl_path}")


if __name__ == "__main__":
    model_id = "dbmdz/bert-base-italian-cased"
    input_csv = "data/source/SU_distractor_filtrato.csv"
    output_path = "data/output/MASK"
    process_file(model_id, input_csv, output_path)
