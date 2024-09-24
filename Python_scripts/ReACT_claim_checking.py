from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from wiki_api_utils import search_wikipedia_short, search_wikipedia
from huggingface_hub import InferenceClient
import requests

import pandas as pd
import random

from Search_scripts.sel_search import google_search

def search_custom(search):
    search_info = google_search(search)
    return search_info  

def load_random_claim(path):
    # load a random claim from the csv, print it and return it
    df = pd.read_csv(path)
    claim = df.iloc[random.randint(0, len(df)-1)]
    result = claim["reviewRating.alternateName"]
    # result = claim["converted_label"]
    claim = "\"" + str(claim["claimReviewed"]) + "\"" + " Made by: " + str(claim["itemReviewed.author.name"]) + ". Date " + str(claim["itemReviewed.datePublished"])

    print(claim)
    print(result)

    return claim, result


# Load the tokenizer and the model
# model_name = "meta-llama/Meta-Llama-3.1-8B"
# model_name = "mistralai/Mistral-7B-Instruct-v0.3"
model_name = "mistralai/Mistral-7B-v0.1"
# for llama 8b
# {
#     "inputs": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>How are you?<|eot_id|><|start_header_id|>assistant<|end_header_id|>",

# }
client = InferenceClient(token="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", model = model_name)


def query_hf(payload, model_id, api_token, temperature=1.0, top_k = None, max_tokens=100):
    parameters = {"temperature": temperature, "max_new_tokens": max_tokens, "top_k": top_k}
    options = {"wait_for_model" : True}
    json_payload = {"inputs": payload, "parameters": parameters, "options" : options}
    headers = {"Authorization": f"Bearer {api_token}"}
    API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
    response = requests.post(API_URL, headers=headers, json=json_payload)
    return response.json()

api_token = "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" # get yours at hf.co/settings/tokens
# data = query_hf("The goal of life is ", model_name, api_token)

temp = 1.5

# Load react prompt from txt file
n_shots = 3
react_shots = ""
for n in range(n_shots):
    with open(f"Prompts/fact_checking_react_{n+1}.txt", "r", encoding='utf-8') as file:
        react_shots += file.read()

# question to be answered
# text = react_shots + "<|user|> Question: Which is the strongest animal? <|end|>"
claim, result = load_random_claim("Datasets/english_150_before_2024_sample.csv")

# Tokenize the input
# tokens = tokenizer.encode(react_shots + text, add_special_tokens=False, return_tensors="pt").to("cuda")
# prompt_len = len(tokens[0])

max_iters = 3

text = react_shots + "Claim 1: " + claim + ". You can choose between the options false, mostly false, mixture, mostly true and true. \n\n"

# Perform the question answering
# if we did things right, we would be generating token by token, stopping generation when the end token is generated.
for i in range(n_shots):
    # pre = "<|assistant|> Thought " + str(i + 1) + ": "
    pre = "Thought " + str(i + 1) + ": "
    # pre_tok = tokenizer.encode(pre, add_special_tokens=False, return_tensors="pt").to("cuda")

    # thinking
    generated_text = query_hf(
                            text + pre,
                            model_id=model_name,
                            api_token=api_token,
                            max_tokens=100,
                            top_k=10,
                            temperature=temp,
                         )

    # generated_text = tokenizer.decode(tokens.tolist()[0][prompt_len:])
    # print(generated_text)
    # prompt_len = len(tokens[0])

    # pre = "<|assistant|> Action " + str(i + 1) + ": Search["  
    pre = "Action " + str(i + 1) + ": Search["
    # pre_tok = tokenizer.encode(pre, add_special_tokens=False, return_tensors="pt").to("cuda")

    generated_text = ("Action "+str(i+1)).join(generated_text[0]["generated_text"].split("Action "+str(i+1))[:max_iters +1])

    generated_text = query_hf(
        generated_text + pre,
        model_id=model_name,
        api_token=api_token,
        max_tokens=15,
        top_k=10,
        temperature=temp,
    )

    generated_text = generated_text[0]["generated_text"]
    # generated_text = tokenizer.decode(tokens.tolist()[0][prompt_len:])
    # print(generated_text)

    # cropped_text = generated_text.split("Search[")[max_iters * n_shots + 1 + i]

    search = generated_text.split("Search[")[max_iters * n_shots + 1 + i].split("]")[0]

    # prompt_len = len(tokens[0])

    search_info = search_custom(search)

    # cure search info

    text =  "Search[".join(generated_text.split("Search[")[:max_iters * n_shots + 1 + i]) + "Search[" + search + "] \n\n Result " + str(i+1) + ":" + search_info + "\n"

    print("Question: " + text.split("Question:")[-1])


pre = " Final Answer: "  
# pre_tok = tokenizer.encode(pre, add_special_tokens=False, return_tensors="pt").to("cuda")

final_text = query_hf(
    text + pre,
    model_id=model_name,
    api_token=api_token,
    max_tokens=10,
    top_k=10,
    temperature=temp,
)[0]["generated_text"]

final_answer = final_text[len(text):]

if "Question: " in final_answer:
    final_answer = final_answer.split("Question: ")[0]

final_text = text + final_answer

# generated_text = tokenizer.decode(tokens.tolist()[0][prompt_len:])
print("Question: " + final_text.split("Question: ")[-1])

# # save final text to file
# file_path = "Results/ReAct_results_faster_stronger_pokemon.txt"
# with open(file_path, "w", encoding="utf-8") as file:
#     file.write("Question: " + final_text.split("Question: ")[-1])
