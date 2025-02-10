from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer

# Define the smallest possible LLaMA model
config = LlamaConfig(
    vocab_size=32000,  # Standard LLaMA vocabulary size
    hidden_size=32,    # Very small embedding size
    intermediate_size=64,  # Feed-forward layer size
    num_hidden_layers=1,  # Only one transformer block
    num_attention_heads=1,  # Only one attention head
    max_position_embeddings=32,  # Small context window
)

# Initialize the tiny LLaMA model
model = LlamaForCausalLM(config)

# Initialize the tokenizer (LLaMA tokenizer is compatible)
tokenizer = AutoTokenizer.from_pretrained("aboonaji/llama2finetune-v3")

# Test if the model loads correctly
input_text = "Hello"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
output = model(input_ids)

print("Tiny LLaMA model loaded and executed correctly!")



# Define paths
model_path = "./tiny-llama2"

# Save the model and tokenizer
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

# Zip the model for download
import shutil
shutil.make_archive(model_path, 'zip', model_path)

# Provide the download link
model_zip_path = "./tiny-llama2.zip"


# Load the saved model and tokenizer
model = LlamaForCausalLM.from_pretrained("./tiny-llama2")
tokenizer = AutoTokenizer.from_pretrained("./tiny-llama2")

# Push to Hugging Face Hub (Replace "your-username" with your actual username)
model.push_to_hub("Jackingchen09/tiny-llama2")
tokenizer.push_to_hub("Jackingchen09/tiny-llama2")

print("Model successfully uploaded!")