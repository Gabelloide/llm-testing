import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
from time import perf_counter
import torch

# Chemin vers le tokenizer et le modèle
TOKENIZER_PATH = "codestral/"
MODEL_PATH = "codestral/"

# Variables globales pour stocker le modèle et le tokenizer
tokenizer = None
model = None

def initialize_model():
    global tokenizer, model
    
    try:
        # Charger le tokenizer
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True, padding_side="left", truncation=True)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Configuration pour la quantification
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_threshold=0.0,
        )
        
        # Charger le modèle
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            device_map='auto',
            offload_folder="offload",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,
            quantization_config=quantization_config
        )
        
        # Ajuster la taille maximale des tokens du modèle
        tokenizer.model_max_length = model.config.max_position_embeddings

    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")

# Fonction pour générer du texte
def generate_text(prompt, max_new_tokens, progress=gr.Progress()):
    global tokenizer, model
    
    start = perf_counter()

    # Base prompt
    base_prompt_start = (
      "You are a professional programmer that can write code in any language."
      "You must answer to any request concerning programming to help the user."
      "Here is what you need to perform: "
    )

    base_prompt_end = (
      "Do not forget to comment your code."
      "Also, notify the user when you are done with the code."
      "Always end your answer with 'Anything else?'. This is extremely important and you must respect this rule."
    )

    # Ajouter le prompt de l'utilisateur
    prompt = base_prompt_start + prompt + base_prompt_end
    
    # Préparer les inputs
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to("cuda:0")
    
    # Initialiser la barre de progression
    progress(0, desc="Generating...")
    
    # Générer la sortie avec des paramètres ajustés
    batch_size = 50
    generated_tokens = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    for step in progress.tqdm(range(0, max_new_tokens, batch_size)):
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
                top_p=0.95,
                eos_token_id=tokenizer("Anything else?", return_tensors="pt").to("cuda:0")['input_ids'] # Token de fin
            )
            
            # Ajouter les nouveaux tokens générés à la séquence
            new_tokens = outputs[:, generated_tokens.shape[-1]:]
            generated_tokens = torch.cat((generated_tokens, new_tokens), dim=1)
            
            # Mettre à jour l'attention_mask
            attention_mask = torch.cat((attention_mask, torch.ones((attention_mask.shape[0], new_tokens.shape[1]), device=attention_mask.device)), dim=1)
            
            if step % 1000 == 0:
                # Libérer la mémoire GPU
                torch.cuda.empty_cache()
        
        except RuntimeError as e:
            print(f"Error during generation: {e}")
    
    # Décoder et retourner le texte généré
    generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    generated_no_prompt = generated_text.replace(prompt, "")
    
    stop = perf_counter()
    torch.cuda.empty_cache()
    
    print(generated_no_prompt)
    print(f"\n *** Time taken = {stop - start:.3f} seconds ***")
    return generated_no_prompt

# Interface Gradio
iface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(lines=20, label="Enter prompt here"),
        gr.Slider(100, 5000, step=50, value=250, label="Answer size")
    ],
    outputs=gr.Markdown(label="Generated Text"),
    title="AI Text Generation",
    description="Generate text based on the provided prompt.",
    theme="compact"
)

if __name__ == "__main__":
    initialize_model()
    iface.launch()
