from transformers import AutoTokenizer, AutoModelForCausalLM

# Model ID (Replace with your own if needed)
model_id = "Jackingchen09/tiny-llama2"

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
# Load Model in 8-bit mode to save memory
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",  # Auto-distributes across available GPUs
    torch_dtype="auto",
)

from datasets import load_dataset

# Load a dataset (Databricks Dolly 15k)
dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

def format_sample(sample):
    prompt = f"### Instruction:\n{sample['instruction']}\n\n### Response:\n{sample['response']}"
    return {"text": prompt}

# Apply formatting
dataset = dataset.map(format_sample, remove_columns=dataset.column_names)

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

dataset = dataset.map(tokenize_function, batched=True)

print("Dataset processed successfully!")

from peft import get_peft_model, LoraConfig, TaskType

# LoRA Configuration
lora_config = LoraConfig(
    r=8,  # Low-rank matrix size
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Apply LoRA to attention layers
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./fine-tuned-llama",
    per_device_train_batch_size=2,  # Adjust batch size based on GPU memory
    gradient_accumulation_steps=4,  # Accumulate gradients over multiple steps
    num_train_epochs=2,  # Fine-tune for 2 epochs
    save_strategy="epoch",
    logging_steps=10,
    learning_rate=2e-5,
    fp16=True,  # Enable mixed precision for better performance
    push_to_hub=False,  # Set to True if you want to upload to Hugging Face
)

from transformers import Trainer, DataCollatorForLanguageModeling

# Data collator for padding
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

# Start training
trainer.train()
