from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from wiki_api_utils import search_wikipedia_short, search_wikipedia

def summarization_query(search_q, text, model, tokenizer):
    input = "<|user|> Summarize this text, puttinh emphasis on " + search_q + ", do not summarize too much otherwise a lot of info will be lost. Text : " + text + "<|end|> \n<|assistant|> Summary: "
    inputs = tokenizer.encode(input, add_special_tokens=False, return_tensors="pt").to("cuda")
    prompt_len = len(inputs[0])
    model.eval()
    outputs = model.generate(inputs,
        max_new_tokens=200,
        do_sample=True,
        top_k=10,
        temperature=1.,
    )
    del input, inputs
    return outputs[:, prompt_len:]

# Load the tokenizer and the model
model_name = "microsoft/Phi-3-mini-128k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to("cuda")

# Load react prompt from txt file
with open("ReAct_prompts_phi.txt", "r") as file:
    react_shots = file.read()

# question to be answered
text = "<|user|> Question: Which is the strongest animal? <|end|>"

# Tokenize the input
tokens = tokenizer.encode(react_shots + text, add_special_tokens=False, return_tensors="pt").to("cuda")
# inputs = tokenizer.encode(search_explanation + react_shots + question, add_special_tokens=False, return_tensors="pt")
#inputs = torch.tensor(inputs).to("cuda")
prompt_len = len(tokens[0])

# Perform the question answering
# if we did things right, we would be generating token by token, stopping generation when the end token is generated.
for i in range(3):

    pre = "<|assistant|> Thought "+str(i+1)+": "
    pre_tok = tokenizer.encode(pre, add_special_tokens=False, return_tensors="pt").to("cuda")

    tokens = model.generate( 
        torch.concat((tokens, pre_tok), dim = 1),
        max_new_tokens=200,
        do_sample=False,
        # top_k=10,
        temperature=1.,
    )

    generated_text = tokenizer.decode(tokens.tolist()[0][prompt_len:])
    print(generated_text)
    prompt_len = len(tokens[0])

    pre = "<|assistant|> Action "+str(i+1)+": Search["  
    pre_tok = tokenizer.encode(pre, add_special_tokens=False, return_tensors="pt").to("cuda")

    tokens = model.generate( 
        torch.concat((tokens, pre_tok), dim = 1),
        max_new_tokens=200,
        do_sample=False,
        # top_k=10,
        temperature=1.,
    )

    generated_text = tokenizer.decode(tokens.tolist()[0][prompt_len:])
    print(generated_text)

    search = generated_text.split("Search[")[-1].split("]")[0]

    prompt_len = len(tokens[0])

    search_info = search_wikipedia(search)

    summ_search = summarization_query(search, search_info, model, tokenizer)
    print(tokenizer.decode(summ_search.tolist()[0]))

    # tokens = torch.concat((tokens,tokenizer.encode("<|user|> Search " + str(i+1) + ": " + search_info + "<|end|>", add_special_tokens=False, return_tensors="pt").to("cuda")), dim = 1)
    tokens = torch.concat((tokens,
                           tokenizer.encode("<|user|> Search " + str(i+1) + ": ", add_special_tokens=False, return_tensors="pt").to("cuda"),
                           summ_search,
                           tokenizer.encode("<|end|>", add_special_tokens=False, return_tensors="pt").to("cuda")),
                           dim = 1)
    


pre = "<|assistant|> Final Answer: "  
pre_tok = tokenizer.encode(pre, add_special_tokens=False, return_tensors="pt").to("cuda")

tokens = model.generate( 
    torch.concat((tokens, pre_tok), dim = 1),
    max_new_tokens=200,
    do_sample=False,
    # top_k=10,
    temperature=1.,
)

generated_text = tokenizer.decode(tokens.tolist()[0][prompt_len:])
print(generated_text)
