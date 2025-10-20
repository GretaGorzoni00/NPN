from transformers import AutoTokenizer, AutoModelForMaskedLM
import pandas as pd
import torch
import numpy as np
import argparse
import sys

def main(model_id, prefix, tokenizer_path, train_dataset, test_dataset, output_path):

	tokenizer = AutoTokenizer.from_pretrained(model_id)
	model = AutoModelForMaskedLM.from_pretrained(model_id, output_hidden_states = True)
	#AutoModelForMaskedLM carica il modello per il task di Masked Language Modeling
	#output_hidden_states=True: dice al modello di restituire anche le rappresentazioni interne (embedding di ogni layer).

	tokenizer.save_pretrained(tokenizer_path)
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


	text_train = pd.read_csv(train_dataset, sep = ";")
	text_test = pd.read_csv(test_dataset, sep = ";")

	results = []
	
	
	for _, line in text_train.iterrows():

		lemma1, prep, lemma2 = line["costr"].strip().split(" ")
		vec_constr = lemma1 + " [UNK] " + lemma2
		sentence = line["context_pre"] + " " + vec_constr + " " + line["context_post"]
		sentence_orig = line["context_pre"] + " " + line["costr"] + " " + line["context_post"]

		posizione_preposizione = len(lemma1) + len([x for x in line["context_pre"] if not x == " "])
		# sentence_orig_nospace = [x for x in  line["context_pre"] if not x == " "] + [x for x in  line["costr"] if not x == " "]
		# print("\n\n")
		# print(''.join(sentence_orig_nospace))
		# print(sentence_orig_nospace[posizione_preposizione])
		# input()

		#itera su ogni riga del data set ricomponendo la frase con la costruzione modificata UNK
		inputs = tokenizer(sentence, return_tensors="pt")
		inputs_orig = tokenizer(sentence_orig, return_tensors="pt")
		print(sentence_orig)
		tokens = tokenizer.tokenize(sentence_orig)

		tot_caratteri = 0
		i = 0
		while i<len(tokens) and tot_caratteri<posizione_preposizione:
			print(tokens[i])
			curr_chars = len([x for x in tokens[i] if not x == "#"])
			tot_caratteri += curr_chars
			i+=1

		print("Trovata preposizione", tokens[i], "in posizione", i, "con id", inputs_orig["input_ids"][0].tolist()[i+1])
		input()
		#Converte la frase in ID tokenizzati, creando tensori PyTorch

		embeddings_list = []
		with torch.no_grad():
		#disattiva il tracciamento dei gradienti (risparmia memoria, utile in inference)
			outputs = model(**inputs)
			output_orig = model(**inputs_orig)
			#print(len(outputs.hidden_states))
		#outputs: contiene logits e tutti gli hidden_states (cioè i vettori di ogni token in ogni layer)
	#		inputs["input_ids"][0].tolist().index(tokenizer.mask_token_id)
			target_id = inputs["input_ids"][0].tolist().index(101)
		#Converte la sequenza input_ids in una lista e cerca l’indice in cui compare il token [UNK] (assunto avere ID 50280) per ModernBERT
		# MASK 104 per BERT
			#print(sentence)
			#print(target_id)
			predicted_token_id = outputs.logits[0, target_id].argmax(axis=-1)
			predicted_token = tokenizer.decode(predicted_token_id)
			print("Predicted token:", predicted_token, sentence)

			for layer in range(1, 13):
				#ho stampato print(len(outputs.hidden_states)) = 23
				#primo layer
				embeddings = outputs.hidden_states[layer]
				target_embedding = embeddings[0, target_id, :].numpy()
				target_embedding_prep = embeddings[0, i+1, :].numpy()
				embeddings_list.append(target_embedding)

				#print(embeddings_list)

			results.append({
			"costruzione": line["costr"],
			"embeddings": embeddings_list  # lista di 22 array di dimensione 768
		})

		# === SALVA COME .pkl ===
		df = pd.DataFrame(results)
		df.to_pickle(f"{output_path}/{prefix}_embedding_layers_train.pkl")
		print("file pkl salvato")


		# === SALVA COME .csv (ogni layer in colonne separate) ===
		rows = []
		for row in results:
			row_data = {"costruzione": row["costruzione"]}
			for layer_idx, layer_emb in enumerate(row["embeddings"], start=1):
				for dim_idx, value in enumerate(layer_emb):
					col_name = f"layer_{layer_idx}_dim_{dim_idx}"
					row_data[col_name] = value
			rows.append(row_data) 

		df_csv = pd.DataFrame(rows)
		df_csv.to_csv(f"{output_path}/{prefix}_embedding_layers_train.csv", index=False)
		print("file csv salvato")

		for _, line in text_test.iterrows():

			lemma1, prep, lemma2 = line["costr"].strip().split(" ")
			vec_constr = lemma1 + " [UNK] " + lemma2
			sentence = line["context_pre"] + " " + vec_constr + " " + line["context_post"]
			sentence_orig = line["context_pre"] + " " + line["costr"] + " " + line["context_post"]

			posizione_preposizione = len(lemma1) + len([x for x in line["context_pre"] if not x == " "])
			# sentence_orig_nospace = [x for x in  line["context_pre"] if not x == " "] + [x for x in  line["costr"] if not x == " "]
			# print("\n\n")
			# print(''.join(sentence_orig_nospace))
			# print(sentence_orig_nospace[posizione_preposizione])
			# input()

			#itera su ogni riga del data set ricomponendo la frase con la costruzione modificata UNK
			inputs = tokenizer(sentence, return_tensors="pt")
			inputs_orig = tokenizer(sentence_orig, return_tensors="pt")
			print(sentence_orig)
			tokens = tokenizer.tokenize(sentence_orig)

			tot_caratteri = 0
			i = 0
			while i<len(tokens) and tot_caratteri<posizione_preposizione:
				print(tokens[i])
				curr_chars = len([x for x in tokens[i] if not x == "#"])
				tot_caratteri += curr_chars
				i+=1

			print("Trovata preposizione", tokens[i], "in posizione", i, "con id", inputs_orig["input_ids"][0].tolist()[i+1])
			input()
			#Converte la frase in ID tokenizzati, creando tensori PyTorch

			embeddings_list = []
			with torch.no_grad():
			#disattiva il tracciamento dei gradienti (risparmia memoria, utile in inference)
				outputs = model(**inputs)
				output_orig = model(**inputs_orig)
				#print(len(outputs.hidden_states))
			#outputs: contiene logits e tutti gli hidden_states (cioè i vettori di ogni token in ogni layer)
		#		inputs["input_ids"][0].tolist().index(tokenizer.mask_token_id)
				target_id = inputs["input_ids"][0].tolist().index(101)
			#Converte la sequenza input_ids in una lista e cerca l’indice in cui compare il token [UNK] (assunto avere ID 50280) per ModernBERT
			# MASK 104 per BERT
				#print(sentence)
				#print(target_id)
				predicted_token_id = outputs.logits[0, target_id].argmax(axis=-1)
				predicted_token = tokenizer.decode(predicted_token_id)
				print("Predicted token:", predicted_token, sentence)

				for layer in range(1, 13):
					#ho stampato print(len(outputs.hidden_states)) = 23
					#primo layer
					embeddings = outputs.hidden_states[layer]
					target_embedding = embeddings[0, target_id, :].numpy()
					target_embedding_prep = embeddings[0, i+1, :].numpy()
					embeddings_list.append(target_embedding)

					#print(embeddings_list)

				results.append({
				"costruzione": line["costr"],
				"embeddings": embeddings_list  # lista di 22 array di dimensione 768
			})

		# === SALVA COME .pkl ===
		df = pd.DataFrame(results)
		df.to_pickle(f"{output_path}/{prefix}_embedding_layers_test.pkl")
		print("file pkl salvato")


		# === SALVA COME .csv (ogni layer in colonne separate) ===
		rows = []
		for row in results:
			row_data = {"costruzione": row["costruzione"]}
			for layer_idx, layer_emb in enumerate(row["embeddings"], start=1):
				for dim_idx, value in enumerate(layer_emb):
					col_name = f"layer_{layer_idx}_dim_{dim_idx}"
					row_data[col_name] = value
			rows.append(row_data) 

		df_csv = pd.DataFrame(rows)
		df_csv.to_csv(f"{output_path}/{prefix}_embedding_layers_te.csv", index=False)
		print("file csv salvato")



if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("-m", "--model", default = "DeepMount00/ModernBERT-base-ita")
	parser.add_argument("--prefix", default ="MB")
	parser.add_argument("-t", "--tokenizer_path", default = "data/tokenizer")
	parser.add_argument("--train", default = "data/data.tsv")
	parser.add_argument("--test", default = "data/data.tsv")
	parser.add_argument("-o", "--output_path", default = "data/output/embeddings")
	args = parser.parse_args()
	main(args.model, args.prefix, args.tokenizer_path, args.train, args.test, args.output_path)