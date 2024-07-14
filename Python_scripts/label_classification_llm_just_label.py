from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import pandas as pd
import torch

df = pd.read_csv("Datasets/media_review_english_only.csv", encoding="utf-16", sep="\t", dtype={24: str})

#model_name = "microsoft/Phi-3-mini-4k-instruct"
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

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
    token = "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",
    quantization_config = bnb_config
)

tokenizer = AutoTokenizer.from_pretrained(model_name, device = "cuda", token = "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")
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

a = 0
b = 794
standard_labels = [ "False", "Mostly False", "Mixture", "Mostly True", "True" ]

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
for i in range(a, b):
    #claim = df["claimReviewed"].iloc[i]
    label = df["mediaAuthenticityCategory"].iloc[i]
    if(label not in standard_labels):
        mapped_labels += 1
        #claim_review = f"Claim: {claim}\nLabel: {label}"
        claim_review = f"Label: {label}"

        messages = [
            {"role": "System", "content": system_prompt},
            {"role": "user", "content": claim_review},
        ]

        prompt = pipe.tokenizer.apply_chat_template(
            messages, 
            tokenize = False, 
            add_generation_prompt = True
        )

        inputs.append(prompt)

outputs = pipe(inputs, **generation_args, batch_size=4)

mapped_labels = 0
for i in range(a, b):
    label = df["mediaAuthenticityCategory"].iloc[i]
    #print(f"Old label is: {label}")
    if label not in standard_labels:
        input_string = "assistant"
        actual_output = outputs[mapped_labels][0]['generated_text']
        if actual_output.startswith(input_string):
            clean_output = actual_output.split("assistant", 1)[1].lstrip()
        else:
            clean_output = actual_output

        if clean_output != "Mixture":
            print(f"Old label is: {label}")
            print(f"New label is: {clean_output}")

        df.loc[i, "mediaAuthenticityCategory"] = clean_output
        mapped_labels += 1

df.to_csv("Datasets/media_review_english_mapped.csv", index = False, encoding="utf-16", sep="\t")