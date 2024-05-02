import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pandas as pd

torch.random.manual_seed(0)

#df = pd.read_csv("Datasets/new_dataset.csv", encoding="utf-16", sep="\t")
df = pd.read_csv("Datasets/claim_review.csv")

# print pd column names
print(df.columns)

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    torch_dtype="auto", 
    trust_remote_code=True, 
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

system_prompt = "You are an assistant that helps fact-checking claims. Given a claim and the label telling its veracity, you need to normalize the label to standart format, between this choices: True, Mostly False, False, Mixture. Answer only with the label."

#a = 100
a = 98
claim = df["claimReviewed"].iloc[a]
#label = df["reviewRating.alternateName"].iloc[a]
label = "This statement is totally false"

claim_review = f"Claim: {claim}\nLabel: {label}"

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": claim_review}
]

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 3,
    "return_full_text": False,
    "do_sample": False,
}

print(claim_review)
#output = pipe(messages, **generation_args)
#print(output[0]['generated_text'])
