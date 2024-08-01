# from transformers import AutoModelForCausalLM, AutoTokenizer
# from tqdm import tqdm

# #https://huggingface.co/mistralai/Mistral-7B-v0.3/tree/main

# # Chemin vers le tokenizer et le modèle
# tokenizer_path = "mistral/"
# model_path = "mistral/"

# # Charger le tokenizer et le modèle
# tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
# model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, device_map='auto', offload_folder="offload", low_cpu_mem_usage=True)

# # Préparer les inputs
# prompt = (
#     "You are a helpful assistant. Please generate a Python code "
#     "for a recursive function to compute Fibonacci numbers. "
#     "The function should take an integer n and return the nth Fibonacci number. "
#     "Here is the code:"
# )

# inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

# # Générer la sortie avec des paramètres ajustés
# outputs = model.generate(
#     **inputs,
#     max_new_tokens=50,  # Augmenter max_new_tokens
#     num_return_sequences=1,
#     top_k=50,  # Ajuster top_k
# )

# # Décoder et afficher le texte généré
# generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print("***** Generated text: *****\n", generated_text)

# -----------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------

#https://huggingface.co/mistralai/Mistral-7B-v0.3/tree/main

# from transformers import AutoModelForCausalLM, AutoTokenizer
# from tqdm import tqdm
# import torch

# # Chemin vers le tokenizer et le modèle
# tokenizer_path = "mistral/"
# model_path = "mistral/"

# # Charger le tokenizer et le modèle
# tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
# model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, device_map='auto', offload_folder="offload", low_cpu_mem_usage=True)

# # Préparer les inputs
# prompt = (
#     "You are a helpful assistant. Please generate a Python code "
#     "for a recursive function to compute Fibonacci numbers. "
#     "The function should take an integer n and return the nth Fibonacci number. "
#     "Here is the code:"
# )

# inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

# # Initialiser la barre de progression
# max_new_tokens = 50
# progress_bar = tqdm(total=max_new_tokens, desc="Generating Text", unit="token")

# # Initialiser la génération
# input_ids = inputs['input_ids']
# attention_mask = inputs['attention_mask']

# # Créer une liste pour stocker les nouveaux tokens générés
# generated_tokens = input_ids

# # Boucle de génération token par token
# for _ in range(max_new_tokens):
#     outputs = model(input_ids=generated_tokens, attention_mask=attention_mask)
#     next_token_logits = outputs.logits[:, -1, :]
#     next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
    
#     # Ajouter le nouveau token généré à la séquence
#     generated_tokens = torch.cat((generated_tokens, next_token), dim=1)
    
#     # Mettre à jour l'attention_mask
#     attention_mask = torch.cat((attention_mask, torch.ones((attention_mask.shape[0], 1), device=attention_mask.device)), dim=1)
    
#     # Mettre à jour la barre de progression
#     progress_bar.update(1)

# progress_bar.close()

# # Décoder et afficher le texte généré
# generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
# print("***** Generated text: *****\n", generated_text)

# -----------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------- Modèle en float32, load_in_4bit -------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
from time import perf_counter
import torch

start = perf_counter()

# Chemin vers le tokenizer et le modèle
tokenizer_path = "codestral/"
model_path = "codestral/"

# Charger le tokenizer et le modèle
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, padding_side="left", truncation=True)
tokenizer.pad_token = tokenizer.eos_token  # Most LLMs don't have a pad token by default

# Configuration pour la quantification
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_threshold=0.0,
)

model = AutoModelForCausalLM.from_pretrained(
  model_path,
  local_files_only=True, 
  device_map='auto', 
  offload_folder="offload", 
  low_cpu_mem_usage=True, 
  torch_dtype=torch.float32, 
  quantization_config=quantization_config
)

# Bonne taille de token
tokenizer.model_max_length = model.config.max_position_embeddings

# Préparer les inputs
prompt = (
"""
You are a professsional programmer that can write code in any language.
You must answer to any request concerning programming to help the user.
Please write some Java code to create a small tic tac toe game.
Do not forget to comment your code.
Also, notify the user when you are done with the code.
"""

)

inputs = tokenizer(prompt, return_tensors="pt", padding=True).to("cuda:0")

# Initialiser la barre de progression
max_new_tokens = 5000
progress_bar = tqdm(total=max_new_tokens, desc="Generating Text", unit="token")

# Générer la sortie avec des paramètres ajustés
batch_size = 100  # Ajuster le batch_size pour générer plusieurs tokens à la fois

generated_tokens = inputs['input_ids']
attention_mask = inputs['attention_mask']

for step in range(0, max_new_tokens, batch_size):
    try:
      outputs = model.generate(
          input_ids=generated_tokens,
          attention_mask=attention_mask,
          max_length=generated_tokens.shape[1] + batch_size,
          num_return_sequences=1,
          pad_token_id=tokenizer.eos_token_id,
          do_sample=True,
          use_cache=True,
          top_k=50,
          top_p=0.95
      )
      
      # Ajouter les nouveaux tokens générés à la séquence
      new_tokens = outputs[:, generated_tokens.shape[-1]:]
      generated_tokens = torch.cat((generated_tokens, new_tokens), dim=1)
      
      # Mettre à jour l'attention_mask
      attention_mask = torch.cat((attention_mask, torch.ones((attention_mask.shape[0], new_tokens.shape[1]), device=attention_mask.device)), dim=1)
      
      # Mettre à jour la barre de progression
      progress_bar.update(new_tokens.shape[1])

      if step%1000 == 0:
        # Libérer la mémoire GPU
        torch.cuda.empty_cache()

    except RuntimeError as e:
       print(f"Error during generation: {e}")

progress_bar.close()

# Décoder et afficher le texte généré
generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

stop = perf_counter()
print("\n\n***** Generated text: *****\n\n", generated_text)
print(f"\nTime taken = {stop-start}")

# Libérer la mémoire GPU
torch.cuda.empty_cache()

with open('generated-text.txt', "w") as f:
   f.write(generated_text)