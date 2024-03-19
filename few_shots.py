from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", device = "cuda")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1").to("cuda")

# Ask a question
shots = "(1) What is the capital of France? \n The capital of France is Paris. [/end] (2) What is the biggest ocean on earth? \n The biggest ocean is the Pacific ocean [/end]"
question = "(3) Which is the tallest mountain on earth?"

# Tokenize the input
inputs = tokenizer.encode(shots + question, add_special_tokens=False, return_tensors="pt")
inputs = torch.tensor(inputs).to("cuda")
prompt_len = len(inputs[0])

# Perform the question answering
output_tokens = model.generate( 
    inputs,
    max_new_tokens=100,
    do_sample=True,
    top_k=10,
    temperature=1.,
)

# Get the most likely answer
answer = tokenizer.decode(output_tokens.tolist()[0][prompt_len:])
answer_trimmed = answer.split("[/end] ")[0]

print("Question:", question)
print("Answer:", answer_trimmed)