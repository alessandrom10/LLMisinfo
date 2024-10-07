from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from wiki_api_utils import search_wikipedia_short, search_wikipedia
from huggingface_hub import InferenceClient
import requests

import pandas as pd
import random
import pickle

from Search_scripts.sel_search import google_search
import time

def search_custom(search):
    search_info = google_search(search)
    return search_info  

def load_claims(path):
    # load a random claim from the csv, print it and return it
    df = pd.read_csv(path)
    # claim = df.iloc[random.randint(0, len(df)-1)]
    # result = claim["reviewRating.alternateName"]
    # result = claim["converted_label"]´
    claims = []
    results = []
    for i, row in df.iterrows():
        auth_name = row["itemReviewed.author.name"] if row["itemReviewed.author.name"] == "nan" else "Unknown"
        date = row["itemReviewed.datePublished"] if row["itemReviewed.datePublished"] == "nan" else "Unknown"
        claims.append("\"" + str(row["claimReviewed"]) + "\"" + " Made by: " + str(auth_name) + ". Date " + str(date))
        results.append(row["reviewRating.alternateName"])

    # print(claims)
    # print(results)

    return claims, results


# Load the tokenizer and the model
# model_name = "meta-llama/Meta-Llama-3.1-8B"
# model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
# model_name = "mistralai/Mistral-7B-Instruct-v0.3"
# model_name = "mistralai/Mistral-7B-v0.1"
# for llama 8b
# {
#     "inputs": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>How are you?<|eot_id|><|start_header_id|>assistant<|end_header_id|>",

# }
#get api token from token.txt
with open("token.txt", "r") as file:
    api_token = file.read()

client = InferenceClient(token=api_token, model = model_name)

# load text file to check how many claims we are fact checked
with open("Results/claims_it_answers_70b.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()

checked = len(lines)
print("Starting in claim", checked)

def query_hf(payload, model_id, api_token, temperature=1.0, top_k = None, max_tokens=100):
    parameters = {"temperature": temperature, "max_new_tokens": max_tokens, "top_k": top_k}
    options = {"wait_for_model" : True}
    json_payload = {"inputs": payload, "parameters": parameters, "options" : options}
    headers = {"Authorization": f"Bearer {api_token}"}
    API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
    response = requests.post(API_URL, headers=headers, json=json_payload)
    return response.json()

# data = query_hf("The goal of life is ", model_name, api_token)

temp = 1.0

# Load react prompt from txt file
n_shots = 3
react_shots = ""
for n in range(n_shots):
    with open(f"Prompts/fact_checking_react_{n+1}_it.txt", "r", encoding='utf-8') as file:
        react_shots += file.read()

# question to be answered
# text = react_shots + "<|user|> Question: Which is the strongest animal? <|end|>"
claims, results = load_claims("Datasets/italian_150.csv")

# Tokenize the input
# tokens = tokenizer.encode(react_shots + text, add_special_tokens=False, return_tensors="pt").to("cuda")
# prompt_len = len(tokens[0])

claims = claims[checked:]
results = results[checked:]


max_iters = 3


all_answers = []
for m in range(len(claims)):
    print("Claim: " + claims[m])
    print("Result: " + results[m])

    claim = claims[m]
        
    text = react_shots + "Claim 1: " + claim + ". You can choose between the options false, mostly false, mixture, mostly true and true. \n\n"

    # Perform the question answering
    # if we did things right, we would be generating token by token, stopping generation when the end token is generated.
    for i in range(n_shots):
        # pre = "<|assistant|> Thought " + str(i + 1) + ": "
        pre = "Pensiero " + str(i + 1) + ": "
        # pre_tok = tokenizer.encode(pre, add_special_tokens=False, return_tensors="pt").to("cuda")

        # thinking
        generated_text = {}

        if len(text) + len(pre) > 25000:
            text = text[len(text) + len(pre) - 25000:]

        # do this while generated_text is a dict
        while type(generated_text) is type(dict()):
            generated_text = query_hf(
                                text + pre,
                                model_id=model_name,
                                api_token=api_token,
                                max_tokens=100,
                                top_k=1,
                                temperature=temp,
                            )
            if type(generated_text) is type(dict()):
                # sleep for 10 minutes
                print(generated_text)
                print("Sleeping thought")
                time.sleep(600)

        # generated_text = tokenizer.decode(tokens.tolist()[0][prompt_len:])
        # print(generated_text)
        # prompt_len = len(tokens[0])

        # pre = "<|assistant|> Action " + str(i + 1) + ": Search["  
        pre = "Azione " + str(i + 1) + ": Cerca["
        # pre_tok = tokenizer.encode(pre, add_special_tokens=False, return_tensors="pt").to("cuda")

        generated_text = ("Azione "+str(i+1)).join(generated_text[0]["generated_text"].split("Azione "+str(i+1))[:max_iters +1])

        len_pre_s = len(generated_text + pre)

        generated_text = query_hf(
            generated_text + pre,
            model_id=model_name,
            api_token=api_token,
            max_tokens=15,
            top_k=1,
            temperature=temp,
        )

        generated_text = generated_text[0]["generated_text"]
        # generated_text = tokenizer.decode(tokens.tolist()[0][prompt_len:])
        # print(generated_text)

        # cropped_text = generated_text.split("Search[")[max_iters * n_shots + 1 + i]

        # if no "]" in the last generated search, add it
        new_generated = generated_text[len_pre_s:]
        if not "]" in new_generated:
            generated_text = generated_text + "]"

        # search = generated_text.split("Search[")[max_iters * n_shots + i].split("]")[0]
        search = generated_text.split("Cerca[")[-1].split("]")[0]


        # prompt_len = len(tokens[0])

        search_info = search_custom(search)

        # cure search info

        text =  "Cerca[".join(generated_text.split("Cerca[")[:max_iters * n_shots + 1 + i]) + "Cerca[" + search + "] \n\n Risultato " + str(i+1) + ":" + search_info + "\n"

        # print("Question: " + text.split("Question:")[-1])


    pre = " Risposta finale, scegli tra le opzioni false, mostly false, mixture, mostly true e true: "  
    # pre_tok = tokenizer.encode(pre, add_special_tokens=False, return_tensors="pt").to("cuda")


    if len(text) + len(pre) > 25000:
        text = text[len(text) + len(pre) - 25000:]

    final_text = {}
    while type(final_text) is type(dict()):
        final_text = query_hf(
            text + pre,
            model_id=model_name,
            api_token=api_token,
            max_tokens=10,
            top_k=10,
            temperature=temp,
        )
    if type(generated_text) is type(dict()):
        # sleep for 10 minutes
        print(generated_text)
        print("Sleeping final answer")
        time.sleep(600)
        
    final_text = final_text[0]["generated_text"]

    final_answer = final_text[len(text):]

    if "Claim: " in final_answer:
        final_answer = final_answer.split("Claim: ")[0]

    final_text = text + final_answer

    # generated_text = tokenizer.decode(tokens.tolist()[0][prompt_len:])
    # print("Question: " + final_text.split("Question: ")[-1])

    all_answers.append(final_answer)
    print(final_answer)

    # add answer to file
    with open("Results/claims_it_answers_70b.txt", "a", encoding="utf-8") as file:
        file.write(final_answer.replace("\n","") + "\n")

    # open the pickle file "results/okey.pkl" and append the final_text to the dict with the key the number
    with open("Results/PROMPTS_claims_it_answers_70b.pkl", "rb") as file:
        okey = pickle.load(file)
        okey[checked + m] = final_text
    with open("Results/PROMPTS_claims_it_answers_70b.pkl", "wb") as file:
        pickle.dump(okey, file)


# # save final text to file
# file_path = "Results/ReAct_results_faster_stronger_pokemon.txt"
# with open(file_path, "w", encoding="utf-8") as file:
#     file.write("Question: " + final_text.split("Question: ")[-1])
