import pandas as pd
from langdetect import detect, LangDetectException

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

#This function returns the language that is idenitfied for the given sentence
def detect_language(sentence):
    if isinstance(sentence, str):
        try:
            return detect(sentence)
        except LangDetectException:
            print(f"Unknown: {sentence}")
            return "unknown"
    else:
        print(f"Numbers: {sentence}")
        return "unknown"

#This function returns a new data frame with only the requested language (usually English)
def keep_only_language(df, language):  
    df["language"] = df["claimReviewed"].apply(detect_language)
    new_df = df[df["language"] == language]
    new_df = new_df.drop(columns="language")
    return new_df

#Mapping from the review's rating names in the dataset and the ones we have chosen to standardize
word_mapping = {
    "FALSE": "False",
    "Misleading": "Mostly False",
    "Pants on Fire": "False",
    "MISLEADING": "Mostly False",
    "Unproven": "Mixture",
    "Miscaptioned": "Mostly False",
    "Half True": "Mixture",
    "Labeled Satire": "False",
    "False.": "Mixture",
    "Fake": "False",
    "Correct Attribution": "True",
    "Missing Context": "Mixture",
    "Satire": "False",
    "Scam": "False",
    "Missing context": "Mixture",
    "Legend": "False",
    "Half true": "Mixture",
    "Four Pinocchios" : "False"
}

#df = pd.read_csv("Datasets/claim_review.csv")

#df = keep_only_language(df, "en")
#df["reviewRating.alternateName"] = df["reviewRating.alternateName"].replace(word_mapping)
#df = df[df["reviewRating.alternateName"].isin(word_mapping.values())]

#Final dataset with only the claims in english and the review's rating name standardized
#df.to_csv("Datasets/new_dataset.csv", index = False, encoding="utf-16", sep="\t")

def standardize_labels(labels_list):

  unique_labels = list(set(labels_list))

  # Load the tokenizer and the model
  tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", token="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", device = "cuda")
  model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", token="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").to("cuda")

  for label in unique_labels:

    input_prompt = f"If we'd need to map the label {label} depending on its veracity as False, Mostly False, Mixture, Mostly True or True we'd map it as"

    #input_prompt = f"Scam is false, misleading is mostly false, half true is mixture, Correct is true, {label} is"

    #input_prompt = f"What is the veracity of the label {label}?"

    # Tokenize the input
    inputs = tokenizer.encode(input_prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
    prompt_len = len(inputs[0])

    # Perform the question answering
    # if we did things right, we would be generating token by token, stopping generation when the end token is generated.
    output_tokens = model.generate(
        inputs,
        max_new_tokens=200,
        do_sample=True,
        top_k=10,
        temperature=1,
    )

    answer = tokenizer.decode(output_tokens.tolist()[0][prompt_len:])
    print("The result of the model is " + answer)
    print("END ANSWER")

standardize_labels(["Totally False", "Somewhat true", "So and so" ])