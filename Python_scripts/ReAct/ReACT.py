from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", device = "cuda")
#model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1").to("cuda")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

# TEST
search_explanation = "Everytime you need to perform an internet search to get some information you can just ask for it and we'll give it to you, \
you just have to say \"i would search on the internet for that thing\" and we'll give it to you, here's an example, just imagine that after asking for each \
search you can just stop and wait for our answer" 

# Load react prompt from txt file
with open("Prompts/ReAct_prompts.txt", "r") as file:
    react_shots = file.read()

# question to be answered
question = "What is the result of the age of Brad Pitt times the age of Angelina Jolie?"

# Tokenize the input
#inputs = tokenizer.encode(react_shots + question, add_special_tokens=False, return_tensors="pt")
inputs = tokenizer.encode(search_explanation + react_shots + question, add_special_tokens=False, return_tensors="pt")
#inputs = torch.tensor(inputs).to("cuda")
prompt_len = len(inputs[0])

# Perform the question answering
# if we did things right, we would be generating token by token, stopping generation when the end token is generated.
output_tokens = model.generate( 
    inputs,
    max_new_tokens=200,
    do_sample=True,
    top_k=10,
    temperature=1.,
)

# Get the most likely answer
answer = tokenizer.decode(output_tokens.tolist()[0][prompt_len:])
answer_trimmed = answer.split("[/end] ")[0]

print("Question:", question)
print("Answer:", answer_trimmed)