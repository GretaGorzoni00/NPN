 from transformers import AutoTokenizer, AutoModelForMaskedLM
import pandas as pd
import torch
import numpy as np

model_id = "DeepMount00/ModernBERT-base-ita"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(model_id, output_hidden_states = True)
#AutoModelForMaskedLM carica il modello per il task di Masked Language Modeling
#output_hidden_states=True: dice al modello di restituire anche le rappresentazioni interne (embedding di ogni layer).

tokenizer.save_pretrained("script/")
#salva in locale il tokenizer

# text = "La capitale dell'Italia è [MASK]."
# inputs = tokenizer(text, return_tensors="pt")
# outputs = model(**inputs)

# # To get predictions for the mask:
# masked_index = inputs["input_ids"][0].tolist().index(tokenizer.mask_token_id)
# predicted_token_id = outputs.logits[0, masked_index].argmax(axis=-1)
# predicted_token = tokenizer.decode(predicted_token_id)
# print("Predicted token:", predicted_token)
# Predicted token:  Roma

text = pd.read_csv("code/data.tsv", sep = "\t")

results = []

for _, line in text.iterrows():
	lemma1, prep, lemma2 = line["costr"].strip().split(" ")
	vec_constr = lemma1 + " [UNK] " + lemma2
	sentence = line["contesto_pre"] + " " + vec_constr + " " + line["contesto_post"]
 #itera su ogni riga del data set ricomponendo la frase con la costruzione modificata UNK
 
	inputs = tokenizer(sentence, return_tensors="pt")
 #Converte la frase in ID tokenizzati, creando tensori PyTorch
 
 
 
	embeddings_list = []
	with torch.no_grad():
	#disattiva il tracciamento dei gradienti (risparmia memoria, utile in inference)
		outputs = model(**inputs)
		#print(len(outputs.hidden_states))
	#outputs: contiene logits e tutti gli hidden_states (cioè i vettori di ogni token in ogni layer)
#		inputs["input_ids"][0].tolist().index(tokenizer.mask_token_id)
		target_id = inputs["input_ids"][0].tolist().index(50280)
	#Converte la sequenza input_ids in una lista e cerca l’indice in cui compare il token [UNK] (assunto avere ID 50280)
		print(sentence)
		print(target_id)

		for layer in range(1, 23):
			#ho stampato print(len(outputs.hidden_states)) = 23
			#primo layer 
	 		embeddings = outputs.hidden_states[layer]
	 		target_embedding = embeddings[0, target_id, :].numpy()
	 		embeddings_list.append(target_embedding)
   
			#print(embeddings_list)

		results.append({
		"costruzione": line["costr"],
		"embeddings": embeddings_list  # lista di 22 array di dimensione 768
	})

# === SALVA COME .pkl ===
df = pd.DataFrame(results)
df.to_pickle("embedding_layers_UNK.pkl")

print (results)