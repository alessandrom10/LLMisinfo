# This script performs the mapping of the raw truth labels from the input dataset into the standard ones
# This script fully works

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import os
import pandas as pd
import torch

dataset_input_path = "claim_review_english.csv"
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
hf_key = os.getenv("HF_KEY")
dataset_output_path = "dataset_english_llm_mapping.csv"

df = pd.read_csv(dataset_input_path, encoding="utf-16", sep="\t", dtype={24: str})

bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_use_double_quant = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype = "auto",
    trust_remote_code = True,
    token = hf_key,
    quantization_config = bnb_config
)

tokenizer = AutoTokenizer.from_pretrained(model_name, device = "cuda", token = hf_key)
tokenizer.pad_token_id = tokenizer.eos_token_id

pipe = pipeline(
    "text-generation",
    model = model,
    tokenizer = tokenizer,
)

generation_args = {
    "max_new_tokens": 50,
    "return_full_text": False,
    "do_sample": False 
}

first_index = 0
last_index = len(df) - 1
standard_labels = [ "False", "Mostly False", "Mixture", "Mostly True", "True" ]
current_label_column_name = "reviewRating.alternateName"

system_prompt = """You are an assistant that helps converting some veracity labels to a standard set of labels. Given a label telling the veracity of a certain claim, you need to normalize such label to a standart format, so to one of these labels: True, Mostly True, Mixture, Mostly False or False. Answer only with the mapped label that you have chosen. 
For example:
Label: Missing Context
Your answer should be just the word: Mixture
Label: True
Your answer should be just the word: True
Label: Staged
Your answer should be just the word: False
Label: Edited, Missing Context
Your answer should be just the word: False
"""

inputs = []
mapped_labels = 0
for i in range(first_index, last_index):
    current_label = df[current_label_column_name].iloc[i]
    if(current_label not in standard_labels):
        mapped_labels += 1
        label_query = f"Label: {current_label}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": label_query},
        ]

        prompt = pipe.tokenizer.apply_chat_template(
            messages, 
            tokenize = False, 
            add_generation_prompt = True
        )

        inputs.append(prompt)

outputs = pipe(inputs, **generation_args, batch_size = 4)

mapped_labels_index = 0
for i in range(first_index, last_index):
    current_label = df[current_label_column_name].iloc[i]
    if current_label not in standard_labels:
        input_string = "assistant"
        actual_output = outputs[mapped_labels_index][0]['generated_text']
        if actual_output.startswith(input_string):
            clean_output = actual_output.split("assistant", 1)[1].lstrip()
        else:
            clean_output = actual_output

        df.loc[i, current_label_column_name] = clean_output
        mapped_labels_index += 1

df.to_csv(dataset_output_path, index = False, encoding="utf-16", sep="\t")