🧠 tutorial_interview_LLM

This repository is a hands-on tutorial and interview-style project focused on fine-tuning and deploying a lightweight LLM (Tiny-LLaMA2) for instruction-following and debiasing tasks. It includes code for model loading, inference, and evaluation.


📁 Project Structure
```
tutorial_interview_LLM/
├── data/                      # Dataset for fine-tuning
├── fine-tuned-llama/         # Output directory for fine-tuned models
├── tiny-llama2/              # Pretrained Tiny-LLaMA2 model files
├── main.py                   # Main script for loading and running the model
├── test_2.py                 # Additional test script
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```
🚀 Getting Started
1. Clone the Repository
```
git clone https://github.com/JackingChen/tutorial_interview_LLM.git
cd tutorial_interview_LLM
```
2. Install Dependencies
```
pip install -r requirements.txt
```
3. Load the Tiny-LLaMA2 Model

This script uses the transformers library to load the model and tokenizer from the local path:

```
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "./tiny-llama2"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

```

4. Run Inference
Once the model is loaded, you can input a prompt and generate a response:

```
prompt = "Explain the concept of overfitting in machine learning."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```


#### The interview process and feedback checklist
| question                                                                                                                                                | feedback                                                                       |
|---------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| Research                                                                                                                                                |                                                                                |
| Talk about your current research                                                                                                                        |                                                                                |
| Whats the thesis statement of this work                                                                                                                 | can he form research statement                                                 |
| How do you measure the performance                                                                                                                      | can he design methods to quantify performance and justify his thesis statement |
| what's the novelty of this paper                                                                                                                        | does he know the novelty                                                       |
| Code                                                                                                                                                    |                                                                                |
| You are asked to finetune tiny llama to follow instructions or chat with user, you can use google, please start                                         | Can he train a LLM                                                             |
| Now I need to finetune LLM so that it can help me rephrase some biased sentences to debiased one. You are given this dataset, please use it to finetune | Can he do the preprocessing                                                    |
| Deploying a ML service                                                                                                                                  |                                                                                |
| Now you need to deliver the ML model to your customer, what will you do                                                                                 | he has to provide actionable items for product delivering                      |



