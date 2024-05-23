from transformers import pipeline
from wiki_api_utils import search_wikipedia
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

text = search_wikipedia("significance of photosynthesis in ecosystems")

text = text[:3000]

# Load the tokenizer and the model
model_name = "microsoft/Phi-3-mini-128k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to("cuda")

prompt = "<|user|>Summarize this text :" + text + "<|end|> \n<|assistant|> Summary: "

# Tokenize the input
inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
prompt_len = len(inputs[0])
model.eval()
outputs = model.generate(inputs,
    max_new_tokens=200,
    do_sample=True,
    top_k=10,
    temperature=1.,
)

outputs = tokenizer.decode(outputs.tolist()[0][prompt_len+2:-1])

with open("../page_summary.txt", "w", encoding="utf-8")  as file:
    file.write(outputs)